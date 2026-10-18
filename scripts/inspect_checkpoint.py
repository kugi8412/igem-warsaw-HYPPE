"""Report what a checkpoint is, without building the model.

A checkpoint is only interpretable next to the data and the configuration it
was produced from, so those are recorded inside it. This prints them, plus
the file's own sha256, which is what a result should be quoted next to.

    python scripts/inspect_checkpoint.py runs/model.pt [more.pt ...]

Exit codes: 0 every file was read, 2 a file could not be read or carried no
recognisable payload.
"""

from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import torch  # noqa: E402

_IDENTITY = ("name", "source", "epoch", "seed", "split_path", "splits_path",
             "split_column", "parent_sha256", "method", "select_on")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _walk(obj, prefix="", depth=0):
    """Yield (key, value) for scalar entries, a couple of levels deep."""
    if depth > 2 or not isinstance(obj, dict):
        return
    for k, v in obj.items():
        key = "%s%s" % (prefix, k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            yield key, v
        elif isinstance(v, dict):
            yield from _walk(v, key + ".", depth + 1)


def describe(path: str) -> int:
    if not os.path.exists(path):
        print("MISSING  %s" % path)
        return 2
    size_mb = os.path.getsize(path) / (1 << 20)
    print("=" * 72)
    print("%s" % path)
    print("  size      %.1f MiB" % size_mb)
    print("  sha256    %s" % sha256(path))
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                      # noqa: BLE001
        print("  UNREADABLE: %s" % exc)
        return 2

    if not isinstance(blob, dict):
        print("  payload   %s (not a dict)" % type(blob).__name__)
        return 2

    print("  top-level %s" % ", ".join(sorted(blob)[:12]))

    state = None
    for key in ("model", "state_dict", "model_state_dict", "weights"):
        if isinstance(blob.get(key), dict):
            state, state_key = blob[key], key
            break
    if state is None and all(torch.is_tensor(v) for v in blob.values()):
        state, state_key = blob, "<root>"

    if state is not None:
        n_tensors = sum(1 for v in state.values() if torch.is_tensor(v))
        n_params = sum(int(v.numel()) for v in state.values()
                       if torch.is_tensor(v))
        print("  weights   %d tensors, %d parameters, under %r"
              % (n_tensors, n_params, state_key))
        # Which parts of the architecture are present distinguishes a
        # foundation model from a fine-tuned head at a glance.
        groups = {}
        for k in state:
            groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + 1
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:8]
        print("  modules   %s" % ", ".join("%s(%d)" % kv for kv in top))
        # Codebook points must satisfy <x,x>_L = -1. Weight averaging over
        # ambient coordinates silently breaks this, and the model still runs.
        for k, v in state.items():
            if torch.is_tensor(v) and "codebook" in k and v.dim() >= 2 \
                    and v.shape[-1] > 1:
                x = v.double()
                inner = (x[..., 1:] ** 2).sum(-1) - x[..., 0] ** 2
                print("  manifold  %s: max|<x,x>_L + 1| = %.3e"
                      % (k, float((inner + 1).abs().max())))
    else:
        print("  weights   none found")

    recorded = {k: v for k, v in _walk(blob) if k.split(".")[-1] in _IDENTITY}
    if recorded:
        print("  identity")
        for k in sorted(recorded):
            print("    %-28s %s" % (k, recorded[k]))

    metrics = {k: v for k, v in _walk(blob)
               if isinstance(v, float)
               and any(t in k.lower() for t in ("pcc", "scc", "auc", "acc",
                                                "rec_", "loss"))}
    if metrics:
        print("  metrics (%d recorded, best-named first)" % len(metrics))
        for k in sorted(metrics)[:12]:
            print("    %-36s %.5f" % (k, metrics[k]))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    worst = 0
    for path in sys.argv[1:]:
        worst = max(worst, describe(path))
    return worst


if __name__ == "__main__":
    sys.exit(main())
