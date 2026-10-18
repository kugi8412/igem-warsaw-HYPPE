"""The ceiling on any model that predicts one number per category.

    python scripts/probe_per_experiment_ceiling.py \
        --matrix "$HYPPE_DATA_ROOT/data/final_datasets/og_scope/anchor3_log2.tsv.gz" \
        --samples "$HYPPE_DATA_ROOT/data/samples_metadata.tsv" \
        --out results/per_experiment_ceiling.json

The baselines in `docs/ablations.md` emit one output per condition category
and are therefore constant within a category: two experiments of the same kind
on the same strain receive the same prediction. The full model is not constant
there, since it reads each experiment's own written description. A comparison
on the cell target -- the mean over the samples of an (orthogroup, genome,
category) cell -- averages away exactly the capability that distinguishes
them.

This puts a number on the difference without running a model. Each measured
sample is predicted by its own cell mean, which is the best a per-category
model can do because the cell mean is the least-squares constant within the
cell, and that is correlated against the per-sample truth. No per-category
model can exceed the resulting ceiling.

Reads the expression matrix and the sample metadata directly, so it depends on
neither the model, the loader, nor any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))


def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def pearson(a, b):
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(a, b):
    def rank(v):
        o = v.argsort()
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(v.size, dtype=np.float64)
        return r
    return pearson(rank(a), rank(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--matrix", required=True,
                    help="og_scope expression matrix, og_id + one column per sample")
    ap.add_argument("--samples", required=True, help="samples_metadata.tsv")
    ap.add_argument("--genome-column", default="genome")
    ap.add_argument("--category-column", default="condition_category_olek")
    ap.add_argument("--min-cell", type=int, default=2,
                    help="a cell needs this many samples to contribute; with "
                         "one sample the cell mean IS the sample and the "
                         "correlation would be inflated by construction")
    ap.add_argument("--out", default="results/per_experiment_ceiling.json")
    args = ap.parse_args()

    # Sample -> (genome, category)
    meta = {}
    with _open(args.samples) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cat_col = args.category_column
        if rd.fieldnames and cat_col not in rd.fieldnames:
            alt = [c for c in rd.fieldnames if "condition_category" in c]
            if not alt:
                print("REFUSED: no category column in %s" % args.samples)
                return 2
            cat_col = alt[0]
            print("category column %r not found; using %r" %
                  (args.category_column, cat_col))
        for row in rd:
            sid = row.get("sample") or row.get("sample_id")
            if not sid:
                continue
            cat = (row.get(cat_col) or "").split(",")[0].strip() or "other"
            meta[sid] = (row.get(args.genome_column, ""), cat)
    print("samples with metadata: %d" % len(meta))

    # Matrix
    per_sample, cell_mean_of = [], []
    n_cells = n_rows = 0
    with _open(args.matrix) as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        cols = header[1:]
        groups = {}
        for j, s in enumerate(cols):
            if s in meta:
                groups.setdefault(meta[s], []).append(j)
        print("expression columns: %d, of which %d have metadata, in %d "
              "(genome, category) cells" %
              (len(cols), sum(len(v) for v in groups.values()), len(groups)))
        for row in rd:
            vals = np.array([float(v) if v not in ("", "NA", "NaN") else np.nan
                             for v in row[1:]], dtype=np.float64)
            n_rows += 1
            for members in groups.values():
                v = vals[members]
                ok = np.isfinite(v)
                if int(ok.sum()) < args.min_cell:
                    continue
                vv = v[ok]
                m = float(vv.mean())
                per_sample.append(vv)
                cell_mean_of.append(np.full(vv.shape, m))
                n_cells += 1

    if not per_sample:
        print("REFUSED: no cell had at least %d measured samples"
              % args.min_cell)
        return 2
    y = np.concatenate(per_sample)
    yhat = np.concatenate(cell_mean_of)

    pcc = pearson(y, yhat)
    scc = spearman(y, yhat)
    within = float(np.mean((y - yhat) ** 2))
    total = float(np.var(y))
    print()
    print("orthogroup rows read          %d" % n_rows)
    print("cells contributing            %d" % n_cells)
    print("per-sample measurements       %d" % y.size)
    print()
    print("CEILING for a per-category model")
    print("  pearson                     %.5f" % pcc)
    print("  spearman                    %.5f" % scc)
    print("  variance within cells       %.5f of %.5f total (%.1f %%)"
          % (within, total, 100.0 * within / total if total else float("nan")))
    print()
    print("Read: a model that emits one number per (orthogroup, genome,")
    print("category) cell cannot exceed %.5f on the per-sample target, because" % pcc)
    print("%.1f %% of the variance lies BETWEEN experiments of one cell."
          % (100.0 * within / total if total else float("nan")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"pcc_ceiling": pcc, "scc_ceiling": scc,
                   "n_measurements": int(y.size), "n_cells": int(n_cells),
                   "within_cell_variance": within, "total_variance": total,
                   "within_cell_fraction": (within / total) if total else None,
                   "min_cell": args.min_cell,
                   "matrix": args.matrix, "samples": args.samples},
                  fh, indent=2)
    print("written %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
