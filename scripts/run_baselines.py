"""Run the baseline ablation grid and write the table that justifies the model.

Every baseline reads the same corpus, the same orthogroup-grouped split and
the same per-(orthogroup, genome, category) target as the full architecture,
and is scored by the same pooled Pearson and Spearman over valid cells. A
baseline trained on a different subset, or scored per category and averaged,
answers a different question.

    export HYPPE_DATA_ROOT=/path/to/hyppe-data
    python scripts/run_baselines.py --config config/default.yaml \
        --splits "$HYPPE_DATA_ROOT/data/splits/SPLIT_V6.tsv" \
        --split-column H_kmer90 --max-orthogroups 4000 \
        --device cuda --out runs/baselines.json

`--eval-split val` (the default) is the honest setting while iterating: the
ridge penalty and the CNN epoch are both selected on `val`, so reporting on
`val` is optimistic by exactly that selection. Pass `--eval-split test` once,
at the end, and never again.

Exit codes: 0 the grid ran, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from hyppe.baselines import run_cnn_onehot, run_kmer_ridge  # noqa: E402
from hyppe.config import load_config  # noqa: E402
from hyppe.training.batches import Corpus  # noqa: E402


def build_corpus(cfg, args):
    """The same call the training loop makes, with the same resolved paths."""
    strain = tuple(getattr(cfg.conditioning, "species", ()) or ())
    return Corpus.from_paths(
        sequences_path=cfg.data.sequences_path,
        combat_path=cfg.data.combat_path,
        samples_path=cfg.data.samples_path,
        splits_path=args.splits,
        split_column=args.split_column,
        strain_names=strain,
        seq_len=int(cfg.data.sequence_length),
        max_orthogroups=int(args.max_orthogroups),
        min_obs_for_sigma=int(cfg.data.min_obs_for_sigma),
        # The baselines read sequence and strain only, so every optional
        # cache is left out: including them would turn a "sequence plus
        # strain" lower bound into something else.
        include_marginal=not bool(getattr(cfg.data, "target_per_category",
                                          False)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-column", default="H_kmer90")
    ap.add_argument("--max-orthogroups", type=int, default=0,
                    help="0 = the whole corpus")
    ap.add_argument("--eval-split", default="val", choices=("val", "test"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kmers", default="3,4,5",
                    help="comma-separated k for the ridge baseline")
    ap.add_argument("--cnn-epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/baselines.json")
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("config  %s" % args.config)
    print("splits  %s [%s]" % (args.splits, args.split_column))

    t0 = time.time()
    corpus = build_corpus(cfg, args)
    for line in corpus.report.lines() if hasattr(corpus.report, "lines") \
            else []:
        print(line)
    n_cat = int(corpus.mu.shape[1])
    print("corpus  %d rows, %d target columns, loaded in %.1f s"
          % (len(corpus), n_cat, time.time() - t0))
    for split in ("train", "val", "spy", "test"):
        try:
            print("  %-6s %d rows" % (split, corpus.indices(split).size))
        except KeyError:
            print("  %-6s absent" % split)

    results = []

    # Baseline 1: composition only, with and without the strain.
    for k in [int(v) for v in args.kmers.split(",") if v.strip()]:
        for with_species in (False, True):
            r = run_kmer_ridge(corpus, k=k, with_species=with_species,
                               eval_split=args.eval_split)
            print(r.row())
            results.append(r)

    # Baseline 2: a plain CNN, no reconstruction, no quantiser.
    if not args.skip_cnn:
        for with_species in (False, True):
            r = run_cnn_onehot(corpus, epochs=args.cnn_epochs,
                               with_species=with_species, seed=args.seed,
                               device=args.device,
                               eval_split=args.eval_split)
            print(r.row())
            results.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    payload = {
        "config": args.config,
        "splits": args.splits,
        "split_column": args.split_column,
        "eval_split": args.eval_split,
        "max_orthogroups": args.max_orthogroups,
        "n_rows": len(corpus),
        "n_target_columns": n_cat,
        "seed": args.seed,
        "results": [{"name": r.name, "n_features": r.n_features,
                     "n_params": r.n_params, "metrics": r.metrics,
                     "chosen": r.chosen, "seconds": r.seconds}
                    for r in results],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print()
    print("%-22s %10s %12s  %s" % ("baseline", "features", "params", "score"))
    for r in sorted(results, key=lambda r: -(r.metrics.get("pcc") or -9)):
        print(r.row())
    print("written %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
