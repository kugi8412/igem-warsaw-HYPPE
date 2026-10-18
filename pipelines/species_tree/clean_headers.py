#! /usr/bin/env python3

"""Replace protein accessions in alignment headers with their proteome name.

    python clean_headers.py

AMAS and ASTRAL decide which sequences belong to the same taxon by comparing
header text, so until this runs each protein counts as its own species.
Anything unmapped stops the run rather than passing through, because an
untranslated header becomes an extra tip in the finished tree.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_config import (  # noqa: E402
    ALN_DIRECTORY,
    CLEAN_OUTPUT_DIRECTORY,
    accessions,
    local_proteomes,
    proteome_files,
    taxon_label,
)


def build_map(faa_files):
    """{protein header id: taxon label} over every proteome."""
    out = {}
    for faa in faa_files:
        if not os.path.isfile(faa):
            raise SystemExit("missing proteome %r" % faa)
        label = taxon_label(faa)
        with open(faa, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(">"):
                    out[line.split()[0]] = label
    return out


def check_coverage(faa_files):
    """Every requested genome arrived, and nothing arrived uninvited."""
    expected = set(accessions())
    expected |= {taxon_label(f) for f in local_proteomes()}
    present = {os.path.basename(f) for f in faa_files}

    missing = sorted(e for e in expected
                     if not any(p.startswith(e) for p in present))
    if missing:
        raise SystemExit(
            "Requested but not present: %s. The download did not finish, or "
            "NCBI returned a different assembly version, or an in-house "
            "proteome was removed after the run started." % ", ".join(missing))

    extra = sorted(p for p in present
                   if not any(p.startswith(e) for e in expected))
    if extra:
        raise SystemExit(
            "Present but never requested: %s. Each would become a tip in the "
            "tree." % ", ".join(extra))


def main() -> int:
    faa_files = proteome_files()
    check_coverage(faa_files)
    n_local = len(local_proteomes())
    print("proteomes: %d (%d public, %d in-house)"
          % (len(faa_files), len(faa_files) - n_local, n_local))

    protein_to_taxon = build_map(faa_files)
    print("mapped %d headers to %d taxa"
          % (len(protein_to_taxon), len(set(protein_to_taxon.values()))))

    os.makedirs(CLEAN_OUTPUT_DIRECTORY, exist_ok=True)
    aln_files = sorted(glob.glob(os.path.join(ALN_DIRECTORY, "*.trim.aln")))
    if not aln_files:
        raise SystemExit("no .trim.aln files in %r" % ALN_DIRECTORY)

    unknown, written = [], 0
    for aln_path in aln_files:
        name = os.path.basename(aln_path).replace(".trim.aln", ".clean.aln")
        with open(aln_path, encoding="utf-8") as src, \
                open(os.path.join(CLEAN_OUTPUT_DIRECTORY, name), "w",
                     encoding="utf-8", newline="\n") as dst:
            for line in src:
                if not line.startswith(">"):
                    dst.write(line)
                    continue
                label = protein_to_taxon.get(line.split()[0])
                if label is None:
                    unknown.append((os.path.basename(aln_path),
                                    line.split()[0]))
                    dst.write(line)
                else:
                    dst.write(">%s\n" % label)
        written += 1

    if unknown:
        raise SystemExit(
            "%d Headers could not be mapped, the first being %s in %s"
            % (len(unknown), unknown[0][1], unknown[0][0]))

    print("Wrote %d cleaned alignments to %s"
          % (written, CLEAN_OUTPUT_DIRECTORY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
