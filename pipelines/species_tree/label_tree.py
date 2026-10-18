#! /usr/bin/env python3
"""Rename tree tips from proteome labels to strain names.

The tree leaves IQ-TREE or ASTRAL carrying labels like
`GCF_000167675.1_protein`, which nothing downstream can read. Names come from
`strain_names.tsv` first, which maps an accession or a proteome file name to a
strain in the model's registry and is the only way to name an in-house genome,
then from NCBI's assembly report for public accessions the table does not
cover. A tip that neither source names is an error: it would enter the tree as
a species of its own.

    python label_tree.py run1/species_tree/trichoderma.treefile \\
        --out ../../data/trees/supermatrix.treefile
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_config import ASSEMBLY_REPORT, STRAIN_NAMES, strain_names  # noqa

#: Fallback for public genomes when strain_names.tsv does not list them.
SPECIES_ALIASES = {
    "Trichoderma reesei": "reesei",
    "Trichoderma virens": "virens",
    "Trichoderma atroviride": "IMI206040",
    "Trichoderma harzianum": "harzianum",
    "Trichoderma asperellum": "asperellum",
    "Trichoderma asperelloides": "asperelloides",
    "Trichoderma guizhouense": "guizhouense",
    "Trichoderma longibrachiatum": "longibrachiatum",
    "Trichoderma citrinoviride": "citrinoviride",
    "Trichoderma velutinum": "velutinum",
    "Trichoderma viride": "viride",
}

TIP = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z0-9][A-Za-z0-9_.]*?)"
                 r"(?=\s*[:,)])")


def read_report(path):
    """{accession: (organism name, strain)} from NCBI's report."""
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            acc = rec.get("accession") or rec.get("currentAccession")
            org = rec.get("organism") or {}
            if acc:
                out[acc] = (org.get("organismName", ""),
                            (org.get("infraspecificNames") or {}).get(
                                "strain", ""))
    return out


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")


def from_report(name, strain):
    """A readable name for a public genome the table does not cover."""
    binomial = " ".join(name.split()[:2]) if name else ""
    if binomial in SPECIES_ALIASES:
        return SPECIES_ALIASES[binomial]
    label = slug(name) or "unknown"
    s = slug(strain)
    # NCBI usually repeats the strain inside the organism name.
    if s and s.lower() not in label.lower().split("_"):
        label += "_" + s
    return label


def resolve(tip, table, report):
    """Strain name for one tip, or None when no source knows it."""
    if tip in table:
        return table[tip]
    base = tip[:-len("_protein")] if tip.endswith("_protein") else tip
    if base in table:
        return table[base]
    if base in report:
        return from_report(*report[base])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tree")
    ap.add_argument("--report", default=ASSEMBLY_REPORT)
    ap.add_argument("--names", default=STRAIN_NAMES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    table = strain_names(args.names)
    report = read_report(args.report)
    text = open(args.tree, encoding="utf-8").read()

    tips = sorted(set(TIP.findall(text)))
    tips = [t for t in tips if not re.fullmatch(r"[\d.eE+-]+", t)]
    if not tips:
        raise SystemExit("no tip labels found in %r" % args.tree)

    mapping, unknown = {}, []
    for tip in tips:
        name = resolve(tip, table, report)
        if name is None:
            unknown.append(tip)
        else:
            mapping[tip] = name
    if unknown:
        raise SystemExit(
            "No name for: %s\nAdd them to %s, two tab separated columns: the "
            "accession or proteome file name, then the strain."
            % (", ".join(unknown), args.names))

    clash = sorted({v for v in mapping.values()
                    if list(mapping.values()).count(v) > 1})
    if clash:
        raise SystemExit("two tips map to the same name (%s), which would "
                         "merge them" % ", ".join(clash))

    out = text
    for tip in sorted(mapping, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9_.])%s(?=\s*[:,)])" % re.escape(tip),
                     mapping[tip], out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8", newline="\n").write(out)

    print("%-34s %s" % ("tip", "strain"))
    print("-" * 56)
    for tip in tips:
        print("%-34s %s" % (tip, mapping[tip]))
    print("\nWrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
