"""Export a training run to CSV, so plots can be made without loading torch.

Two sources, because neither has everything: the per-epoch checkpoints carry
the structured bulk of the validation metrics, and the run log carries the
train-loss decomposition and the spy lines, which are printed and never
stored in a checkpoint.

Two files are written from the same parse, so they cannot disagree. The long
one is tidy -- one row per (run, epoch, split, metric, value) -- and is what a
plotting script wants; the wide one is one row per epoch with the metrics as
columns, which is what a spreadsheet wants.

    python scripts/training_report_csv.py \
        --checkpoints _work/full --log _work/full.log \
        --out-dir results/training_report

Exit codes: 0 something was written, 2 nothing was found to read.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

#: `epoch 7 (626.9 s) | train loss 2.554 = task_absolute 1.183 + ...`
_EPOCH = re.compile(
    r"^\s*epoch\s+(\d+)\s+\(([\d.]+)\s*s\)\s*\|\s*train loss\s+([\d.eE+-]+)\s*=\s*(.*)$")
_TERM = re.compile(r"([A-Za-z_][\w]*)\s+([-\d.eE+]+)")
#: `spy: ADMITTED score 0.48610 | motif_auc_small 0.4189->0.000 rec_full ...`
_SPY = re.compile(r"^\s*spy:\s*(\w+)[^|]*\|?\s*(.*)$")
_SPY_KV = re.compile(r"([A-Za-z_][\w]*)\s+([-\d.eE+]+)(?:->([-\d.eE+]+))?")


def from_checkpoints(pattern: str):
    """(epoch -> {metric: value}) read from the per-epoch archives."""
    import torch
    out = {}
    paths = sorted(glob.glob(pattern))
    for p in paths:
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:                        # noqa: BLE001
            print("skipped %s: %s" % (p, exc))
            continue
        ep = int(blob.get("epoch", 0) or 0)
        row = {k: float(v) for k, v in (blob.get("metrics") or {}).items()
               if isinstance(v, (int, float))}
        row["step"] = float(blob.get("step", 0) or 0)
        for key in ("constraint", "prefix_constraint", "occ_gate_constraint"):
            sub = blob.get(key)
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if isinstance(v, (int, float)):
                        row["%s.%s" % (key, k)] = float(v)
        tw = blob.get("task_weights")
        if isinstance(tw, dict):
            for k, v in tw.items():
                if isinstance(v, (int, float)):
                    row["task_weight.%s" % k] = float(v)
        out[ep] = row
        print("read epoch %2d from %s (%d metrics)" % (ep, os.path.basename(p),
                                                       len(row)))
    return out


def from_log(path: str):
    """(train rows, spy rows) parsed from the printed log."""
    train, spy = {}, {}
    if not path or not os.path.exists(path):
        return train, spy
    seen_epoch = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _EPOCH.match(line)
            if m:
                ep = int(m.group(1))
                seen_epoch = ep
                row = {"epoch_seconds": float(m.group(2)),
                       "train_loss": float(m.group(3))}
                for name, val in _TERM.findall(m.group(4)):
                    try:
                        row["train_" + name] = float(val)
                    except ValueError:
                        continue
                train[ep] = row
                continue
            s = _SPY.match(line)
            if s and seen_epoch:
                row = {"spy_verdict_admitted":
                       1.0 if s.group(1).upper().startswith("ADMIT") else 0.0}
                for name, a, b in _SPY_KV.findall(s.group(2)):
                    # `x 0.41->0.00` is before->after; keep both, named.
                    try:
                        row["spy_" + name] = float(b if b else a)
                        if b:
                            row["spy_%s_before" % name] = float(a)
                    except ValueError:
                        continue
                spy[seen_epoch] = row
    return train, spy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoints", default="",
                    help="directory holding <name>_ep###.pt archives")
    ap.add_argument("--log", default="", help="the run's stdout log")
    ap.add_argument("--run", default="", help="run name for the CSV column")
    ap.add_argument("--out-dir", default="results/training_report")
    args = ap.parse_args()

    val = {}
    if args.checkpoints:
        val = from_checkpoints(os.path.join(args.checkpoints, "*_ep*.pt"))
    train, spy = from_log(args.log)
    if not val and not train:
        print("REFUSED: neither checkpoints nor a parsable log were found")
        return 2

    run = args.run or os.path.basename(
        os.path.normpath(args.checkpoints or args.log) or "run")
    epochs = sorted(set(val) | set(train) | set(spy))
    os.makedirs(args.out_dir, exist_ok=True)

    long_path = os.path.join(args.out_dir, "training_metrics_long.csv")
    with open(long_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run", "epoch", "split", "metric", "value"])
        for ep in epochs:
            for split, rows in (("train", train), ("val", val), ("spy", spy)):
                for k, v in sorted((rows.get(ep) or {}).items()):
                    w.writerow([run, ep, split, k, "%.10g" % v])

    cols = {}
    for split, rows in (("train", train), ("val", val), ("spy", spy)):
        for ep in rows:
            for k in rows[ep]:
                cols["%s.%s" % (split, k)] = True
    header = ["run", "epoch"] + sorted(cols)
    wide_path = os.path.join(args.out_dir, "training_metrics_wide.csv")
    with open(wide_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for ep in epochs:
            row = [run, ep]
            for c in header[2:]:
                split, _, key = c.partition(".")
                src = {"train": train, "val": val, "spy": spy}[split]
                v = (src.get(ep) or {}).get(key)
                row.append("" if v is None else "%.10g" % v)
            w.writerow(row)

    print()
    print("epochs      %s" % (epochs,))
    print("val metrics %d per epoch" % (len(next(iter(val.values()))) if val
                                        else 0))
    print("train rows  %d | spy rows %d" % (len(train), len(spy)))
    print("written %s" % long_path)
    print("written %s" % wide_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
