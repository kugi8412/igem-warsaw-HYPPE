"""Build a train / val / spy / test split from a corpus of promoters.

The split is grouped by homology cluster, never by row, because orthologues
of one gene share most of their promoter and a row-wise split reports a number
that will not survive on new loci.

Input is either a FASTA file or a TSV with an identifier column and a sequence
column. An optional orthogroup column makes the orthogroup the atom: members
of one group land in the same split whatever their sequences say.

    python scripts/build_splits.py \
        --sequences data/final_datasets/sequences/sequences.tsv.gz \
        --id-column ID --sequence-column SEQUENCE --group-column OG_ID \
        --filter-column GROUP --filter-value Promotor \
        --output data/splits/splits.tsv --split-column H_kmer90

Exit codes: 0 the split was written and every blocking falsifier passed,
2 bad input, 4 a falsifier failed (the table is still written, so that the
failure can be inspected rather than guessed at).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from hyppe.data.make_splits import (  # noqa: E402
    SPLIT_NAMES,
    SplitSpec,
    build_splits,
    falsifiers,
    write_split_table,
)


def _open(path: str):
    """Transparently read .gz, because the delivered corpus ships gzipped."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def read_fasta(path: str):
    """Yield (identifier, sequence). The identifier is the first token."""
    name, chunks = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def read_table(path: str, id_column: str, sequence_column: str,
               group_column=None, filter_column=None, filter_value=None):
    """Read a delimited table; returns (records, groups).

    The filter exists because the delivered corpus holds four rows per gene --
    the promoter, the gene body and the two flanks -- in one file, and only
    one of them is the model's input.
    """
    records, groups = [], {}
    with _open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for col in (id_column, sequence_column):
            if reader.fieldnames and col not in reader.fieldnames:
                raise SystemExit(
                    "column %r is not in %s; the columns are %s"
                    % (col, path, reader.fieldnames))
        for row in reader:
            if filter_column and row.get(filter_column) != filter_value:
                continue
            rid, seq = row[id_column], row[sequence_column]
            if not rid or not seq:
                continue
            records.append((rid, seq))
            if group_column:
                g = row.get(group_column)
                if g:
                    groups[rid] = g
    return records, groups


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a grouped train/val/spy/test split.")
    ap.add_argument("--sequences", required=True,
                    help="FASTA, or a TSV (optionally gzipped)")
    ap.add_argument("--format", choices=("auto", "fasta", "tsv"),
                    default="auto", help="input format; auto reads the suffix")
    ap.add_argument("--id-column", default="ID",
                    help="TSV only: the identifier column")
    ap.add_argument("--sequence-column", default="SEQUENCE",
                    help="TSV only: the sequence column")
    ap.add_argument("--group-column", default=None,
                    help="TSV only: an orthogroup column. Supply it when you "
                         "have one: k-mer clustering catches near-duplicates, "
                         "not diverged orthologues")
    ap.add_argument("--filter-column", default=None,
                    help="TSV only: keep rows whose column equals "
                         "--filter-value (e.g. GROUP == Promotor)")
    ap.add_argument("--filter-value", default=None)
    ap.add_argument("--output", required=True, help="the split table to write")
    ap.add_argument("--group-output-column", default="og",
                    help="name of the group column in the output")
    ap.add_argument("--split-column", default="split",
                    help="name of the split column in the output; the "
                         "training loop is told this name on its own command "
                         "line and records it in the checkpoint")
    ap.add_argument("--report", default=None,
                    help="write the build report and the falsifiers here")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--spy", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--kmer", type=int, default=25,
                    help="k for the similarity graph")
    ap.add_argument("--scale", type=int, default=8,
                    help="FracMinHash denominator; 1 disables sketching")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="containment above which two records are homologous")
    ap.add_argument("--min-shared", type=int, default=2,
                    help="sketch k-mers two records must share before the "
                         "containment is computed at all")
    ap.add_argument("--seed", type=int, default=7898,
                    help="salts the assignment hash; recorded in the report")
    args = ap.parse_args()

    fmt = args.format
    if fmt == "auto":
        low = args.sequences.lower()
        stem = low[:-3] if low.endswith(".gz") else low
        fmt = "fasta" if stem.endswith((".fa", ".fasta", ".fna")) else "tsv"

    try:
        if fmt == "fasta":
            records, groups = list(read_fasta(args.sequences)), {}
            if args.group_column:
                print("note: --group-column is ignored for FASTA input")
        else:
            records, groups = read_table(
                args.sequences, args.id_column, args.sequence_column,
                args.group_column, args.filter_column, args.filter_value)
    except OSError as exc:
        # Exit 2, not a traceback: a queue reads these codes, and "the input
        # was unreadable" is a different outcome from "a falsifier failed".
        print("REFUSED: cannot read %s: %s" % (args.sequences, exc))
        return 2

    if not records:
        print("REFUSED: no records read from %s" % args.sequences)
        return 2

    spec = SplitSpec(
        fractions={"train": args.train, "val": args.val,
                   "spy": args.spy, "test": args.test},
        k=args.kmer, scale=args.scale, threshold=args.threshold,
        min_shared=args.min_shared, seed=args.seed)
    try:
        spec.validate()
    except ValueError as exc:
        print("REFUSED: %s" % exc)
        return 2

    print("read %d records%s" % (len(records),
                                 ", %d grouped" % len(groups) if groups else ""))
    build = build_splits(records, groups or None, spec)
    verdict = falsifiers(build)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    write_split_table(build, args.output,
                      group_column=args.group_output_column,
                      split_column=args.split_column)

    report = {"build": build.report(), "falsifiers": verdict,
              "output": args.output, "n_records": len(records)}
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    print("%d groups in %d clusters, largest %d"
          % (build.n_groups, build.n_clusters, build.largest_cluster))
    for s in SPLIT_NAMES:
        obs = build.report()["fractions_observed"][s]
        print("  %-6s %6d groups (%.3f, asked %.3f)"
              % (s, build.counts[s], obs, spec.fractions[s]))
    print("written %s" % args.output)

    if not verdict["ok"]:
        print("FALSIFIER FAILED: %s"
              % json.dumps({k: v for k, v in verdict.items() if k != "ok"}))
        return 4
    print("falsifiers: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
