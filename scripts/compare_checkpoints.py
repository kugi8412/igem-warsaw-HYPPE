"""Which tensors differ between two checkpoints, grouped by module.

The ranking head is fine-tuned with the rest of the model frozen, and freezing
by `requires_grad = False` is not sufficient: the coarse codebook updates by
moving average and the normalisation layers keep running statistics, so both
keep moving on any forward pass in train mode. This compares every
floating-point entry and says what actually moved.

    python scripts/compare_checkpoints.py parent.pt finetuned.pt [--expect pairwise]

Exit codes: 0 the comparison ran (and, with --expect, only the named prefixes
moved), 2 a file could not be read or the two carry different tensor sets,
4 something outside the expected prefixes moved.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import torch  # noqa: E402


def _state(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise SystemExit("REFUSED: %s is not a dict" % path)
    for key in ("model", "state_dict", "model_state_dict", "weights"):
        if isinstance(blob.get(key), dict):
            return blob[key]
    if all(torch.is_tensor(v) for v in blob.values()):
        return blob
    raise SystemExit("REFUSED: no weights found in %s" % path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parent")
    ap.add_argument("child")
    ap.add_argument("--expect", action="append", default=[],
                    help="module prefix allowed to differ; repeatable. "
                         "Anything else that moved makes this exit 4")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="absolute tolerance below which a tensor counts as "
                         "unchanged (default 0: bit-identical)")
    args = ap.parse_args()

    a, b = _state(args.parent), _state(args.child)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        print("REFUSED: different tensor sets")
        print("  only in parent (%d): %s" % (len(only_a), only_a[:6]))
        print("  only in child  (%d): %s" % (len(only_b), only_b[:6]))
        return 2

    moved, same, by_module = [], 0, {}
    for k in sorted(a):
        ta, tb = a[k], b[k]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            continue
        if ta.shape != tb.shape:
            moved.append((k, float("inf")))
            continue
        d = float((ta.double() - tb.double()).abs().max())
        if d > args.tol:
            moved.append((k, d))
            top = k.split(".")[0]
            by_module[top] = by_module.get(top, 0) + 1
        else:
            same += 1

    print("tensors: %d total, %d identical, %d moved"
          % (len(a), same, len(moved)))
    if by_module:
        print("moved, by top-level module:")
        for mod, n in sorted(by_module.items(), key=lambda kv: -kv[1]):
            print("  %-24s %d" % (mod, n))
    for k, d in sorted(moved, key=lambda kv: -kv[1])[:10]:
        print("  %-54s max|delta| %.4e" % (k, d))

    if args.expect:
        stray = [k for k, _ in moved
                 if not any(k.startswith(p) for p in args.expect)]
        if stray:
            print("FAILED: %d tensors moved outside %s"
                  % (len(stray), args.expect))
            for k in stray[:20]:
                print("  %s" % k)
            return 4
        print("PASSED: only %s moved" % (args.expect,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
