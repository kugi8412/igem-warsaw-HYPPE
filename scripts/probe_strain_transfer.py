"""Does the one-hot strain channel transfer, or only separate?

    python scripts/probe_strain_transfer.py --config config/default.yaml \
        --splits "$HYPPE_DATA_ROOT/data/splits/SPLIT_V6.tsv" \
        --split-column H_kmer90 --out results/strain_transfer.json

On the strain axis a flat 19-way one-hot ridge beats the full model's
phylogenetic embedding, 0.426 against 0.289. Either the one-hot is better at
this and the tree prior costs us, or it is the best possible discriminator
between strains it has seen and knows nothing of the relations between them,
while the embedding spends capacity on exactly those relations. The readings
differ on a strain the model never saw: a one-hot has no column for it and
falls back to the intercept plus sequence, while an embedding placed by
patristic distance has a position borrowed from its neighbours.

Only the baseline side is measured here, by holding out one strain's rows from
training and comparing against the same fit with those rows in. That does not
decide between the readings -- the full model's half needs a retrain per
strain -- but it says how much of the one-hot's advantage is transfer and how
much is memorised separation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from hyppe.baselines import (  # noqa: E402
    DEFAULT_LAMBDAS,
    evaluate_cells,
    kmer_matrix,
    ridge_normals,
    solve_ridge,
    species_onehot,
    targets_of,
)
from hyppe.config import load_config  # noqa: E402
from hyppe.training.batches import Corpus  # noqa: E402


def _apply(x, w):
    return x.astype(np.float64) @ w[:-1] + w[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-column", default="H_kmer90")
    ap.add_argument("--kmer", type=int, default=4)
    ap.add_argument("--max-orthogroups", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=200,
                    help="skip strains with fewer val rows than this; below "
                         "it the held-out correlation is noise")
    ap.add_argument("--out", default="results/strain_transfer.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus = Corpus.from_paths(
        sequences_path=cfg.data.sequences_path,
        combat_path=cfg.data.combat_path,
        samples_path=cfg.data.samples_path,
        splits_path=args.splits, split_column=args.split_column,
        strain_names=tuple(getattr(cfg.conditioning, "species", ()) or ()),
        seq_len=int(cfg.data.sequence_length),
        max_orthogroups=int(args.max_orthogroups),
        min_obs_for_sigma=int(cfg.data.min_obs_for_sigma),
        include_marginal=not bool(getattr(cfg.data, "target_per_category",
                                          False)))

    mu, valid = targets_of(corpus)
    km = kmer_matrix(corpus.codes, args.kmer)
    sp = species_onehot(corpus.species_id)
    x_all = np.concatenate([km, sp], axis=1)
    x_seq = km
    tr, va = corpus.indices("train"), corpus.indices("val")
    sid = np.asarray(corpus.species_id)
    print("corpus %d rows | train %d | val %d | %d strains"
          % (len(corpus), tr.size, va.size, sp.shape[1]))

    def fit_eval(xf, rows_tr, rows_ev):
        normals = ridge_normals(xf[rows_tr], np.nan_to_num(mu[rows_tr]),
                                valid[rows_tr])
        best = (float("-inf"), None)
        for lam in DEFAULT_LAMBDAS:
            w = solve_ridge(normals, xf.shape[1], float(lam))
            m = evaluate_cells(_apply(xf[rows_tr], w), mu[rows_tr],
                               valid[rows_tr])
            s = m["pcc"] if m["pcc"] == m["pcc"] else float("-inf")
            if s > best[0]:
                best = (s, w)
        return evaluate_cells(_apply(xf[rows_ev], best[1]), mu[rows_ev],
                              valid[rows_ev])

    rows = []
    for s in range(sp.shape[1]):
        ev = va[sid[va] == s]
        if ev.size < args.min_rows:
            continue
        seen = fit_eval(x_all, tr, ev)                       # strain in train
        held = fit_eval(x_all, tr[sid[tr] != s], ev)         # strain held out
        seqonly = fit_eval(x_seq, tr, ev)                    # no strain at all
        rows.append({
            "strain_index": int(s),
            "val_rows": int(ev.size),
            "train_rows": int((sid[tr] == s).sum()),
            "pcc_seen": seen["pcc"],
            "pcc_held_out": held["pcc"],
            "pcc_sequence_only": seqonly["pcc"],
            "transfer_cost": seen["pcc"] - held["pcc"],
            "onehot_value_when_seen": seen["pcc"] - seqonly["pcc"],
            "onehot_value_when_unseen": held["pcc"] - seqonly["pcc"],
        })
        print("strain %2d | val %5d rows | seen %.5f | held out %.5f | "
              "sequence only %.5f"
              % (s, ev.size, seen["pcc"], held["pcc"], seqonly["pcc"]))

    if not rows:
        print("no strain had at least %d val rows" % args.min_rows)
        return 2

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print()
    print("strains measured          %d" % len(rows))
    print("mean pcc, strain seen     %.5f" % mean("pcc_seen"))
    print("mean pcc, strain held out %.5f" % mean("pcc_held_out"))
    print("mean pcc, sequence only   %.5f" % mean("pcc_sequence_only"))
    print("the one-hot is worth      %+.5f when the strain was in training"
          % mean("onehot_value_when_seen"))
    print("                          %+.5f when it was not"
          % mean("onehot_value_when_unseen"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"kmer": args.kmer, "split": args.splits,
                   "split_column": args.split_column,
                   "per_strain": rows,
                   "mean_pcc_seen": mean("pcc_seen"),
                   "mean_pcc_held_out": mean("pcc_held_out"),
                   "mean_pcc_sequence_only": mean("pcc_sequence_only"),
                   "mean_onehot_value_seen": mean("onehot_value_when_seen"),
                   "mean_onehot_value_unseen":
                       mean("onehot_value_when_unseen")},
                  fh, indent=2)
    print("written %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
