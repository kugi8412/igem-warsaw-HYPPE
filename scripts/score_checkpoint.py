"""Score a checkpoint through the same loader the baselines use.

A model's own evaluation pass and the baseline harness do not see
byte-identical rows: the training loop also draws synthetic negatives, and it
decomposes its axes on the per-sample target while the baselines predict
cells, so quoting the two side by side is indicative rather than exact. This
script builds the corpus exactly as `run_baselines.py` does, runs the model
over the same rows, and applies the same three measurements.

    export HYPPE_DATA_ROOT=/path/to/hyppe-data
    python scripts/score_checkpoint.py --checkpoint runs/model.pt \
        --splits "$HYPPE_DATA_ROOT/data/splits/SPLIT_V6.tsv" \
        --split-column H_kmer90 --out results/model_scored.json

Exit codes: 0 scored, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "src"),
           os.path.join(_HERE, "..", "src", "panel")):
    sys.path.insert(0, os.path.abspath(_p))

import model_io  # noqa: E402
import torch  # noqa: E402

from hyppe.baselines import axis_report, evaluate_cells, targets_of, within_row_contrast  # noqa: E402
from hyppe.config import from_mapping  # noqa: E402
from hyppe.training.batches import Corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-column", default="H_kmer90")
    ap.add_argument("--eval-split", default="val", choices=("val", "spy", "test"))
    ap.add_argument("--max-orthogroups", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--with-conditioning", action="store_true",
                    help="NOT IMPLEMENTED, and it refuses rather than "
                         "pretending. See the note in the source")
    ap.add_argument("--out", default="results/model_scored.json")
    args = ap.parse_args()

    model, prov = model_io.load_model(args.checkpoint, device=args.device)
    model.eval()
    print(prov.one_line())

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = from_mapping(blob["config"])
    root = os.environ.get("HYPPE_DATA_ROOT", "")

    def rooted(p):
        return p if (not p or os.path.isabs(p)) else os.path.join(root, p)

    if args.with_conditioning:
        # Loading the caches into the corpus without passing their vectors to
        # `forward` returns the unconditioned numbers under a conditioned
        # name. The vectors are assembled inside `BatchMaker._build`, not
        # stored on the corpus, so doing this properly means scoring through
        # BatchMaker with negatives disabled; until then, refuse.
        raise SystemExit(
            "REFUSED: --with-conditioning is not implemented. The caches "
            "would be loaded and then ignored, giving the unconditioned "
            "number under a conditioned name. Note that the baselines in "
            "docs/ablations.md also see sequence and strain only, so the "
            "DEFAULT run here is the information-matched comparison.")
    kw = {}

    corpus = Corpus.from_paths(
        sequences_path=rooted(cfg.data.sequences_path),
        combat_path=rooted(cfg.data.combat_path),
        samples_path=rooted(cfg.data.samples_path),
        splits_path=args.splits, split_column=args.split_column,
        strain_names=tuple(getattr(cfg.conditioning, "species", ()) or ()),
        seq_len=int(cfg.data.sequence_length),
        max_orthogroups=int(args.max_orthogroups),
        min_obs_for_sigma=int(cfg.data.min_obs_for_sigma),
        include_marginal=not bool(getattr(cfg.data, "target_per_category",
                                          False)),
        **kw)

    mu, valid = targets_of(corpus)
    ev = corpus.indices(args.eval_split)
    print("corpus %d rows | %s %d rows | %d target columns"
          % (len(corpus), args.eval_split, ev.size, mu.shape[1]))

    codes = torch.from_numpy(np.asarray(corpus.codes, dtype=np.int64))
    sp_all = torch.from_numpy(np.asarray(corpus.species_id, dtype=np.int64))
    dev = torch.device(args.device)

    preds = np.zeros((ev.size, mu.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, ev.size, args.batch_size):
            idx = ev[s:s + args.batch_size]
            c = codes[idx].to(dev)
            x = torch.zeros(c.shape[0], 4, c.shape[1], device=dev)
            # Code 4 is N and stays an all-zero column, the same "no
            # evidence" the loader gives it.
            x.scatter_(1, c.clamp(max=3).unsqueeze(1),
                       (c < 4).unsqueeze(1).float())
            out = model(x, species_id=sp_all[idx].to(dev), reconstruct=False)
            m = out["absolute"].mu
            if m.dim() == 1:
                m = m.unsqueeze(-1)
            preds[s:s + idx.size] = m[:, :mu.shape[1]].float().cpu().numpy()
            if s % (args.batch_size * 20) == 0:
                print("  scored %d / %d" % (s, ev.size))

    metrics = evaluate_cells(preds, mu[ev], valid[ev])
    metrics.update(within_row_contrast(preds, mu[ev], valid[ev]))
    metrics.update(axis_report(preds, mu[ev], valid[ev],
                               corpus.og, corpus.genome, ev))

    print()
    for k in ("pcc", "scc", "n_cells", "within_row_pcc", "within_row_cells",
              "within_row_pred_sd", "within_fraction", "os_ortogrupa_pcc",
              "os_genomu_pcc", "os_warunku_pcc"):
        if k in metrics:
            print("  %-22s %12.5f" % (k, metrics[k]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint,
                   "sha256": model_io.sha256(args.checkpoint),
                   "eval_split": args.eval_split,
                   "split": args.splits, "split_column": args.split_column,
                   "with_conditioning": bool(args.with_conditioning),
                   "n_rows": int(ev.size),
                   "metrics": {k: float(v) for k, v in metrics.items()}},
                  fh, indent=2)
    print("written %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
