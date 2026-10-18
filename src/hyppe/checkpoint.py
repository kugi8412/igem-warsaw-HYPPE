"""The checkpoint contract: loading, run identity, mismatch diagnosis.

A checkpoint is not self-sufficient. `torch.load` hands back a dictionary,
but building a model out of it also needs the inverse of `to_mapping`
(`hyppe.config.from_mapping`), three data files the checkpoint neither names
nor digests, and code that builds the same shapes:

    data/motifs.meme                       -> shape of `occupancy.frozen.*`
    data/trees/supermatrix.treefile        -> `genome.points`
    data/conditions/condition_taxonomy.tsv -> number of category logits

The last requirement is not theoretical. A checkpoint written one afternoon
stopped loading three hours later because the absolute head went from 8 free
columns to 7 free plus a derived marginal and a new buffer appeared. Nothing
in the checkpoint detects that, because `heads.absolute.n_outputs` is not in
the config: it is computed at build time from `SamplePolicy().categories`.

So `load_checkpoint` raises `CheckpointMismatch` with the differences and
the identity of both sides, rather than a bare `RuntimeError` from
`load_state_dict`.
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = ["CheckpointMismatch", "load_checkpoint", "identity", "describe",
           "data_file_digests", "IDENTITY_KEYS", "DATA_FILES"]

#: Run-identity keys. Checkpoints written before they existed lack them,
#: which is an absence rather than an error: it has to be reported, not
#: filled in with a default.
IDENTITY_KEYS: Tuple[str, ...] = ("name", "splits_path", "split_column",
                                  "source_mtimes")

#: Data files that co-determine the SHAPE of the model. They are not in the
#: checkpoint.
DATA_FILES: Tuple[str, ...] = (
    "data/motifs.meme",
    "data/trees/supermatrix.treefile",
    "data/conditions/condition_taxonomy.tsv",
)


class CheckpointMismatch(RuntimeError):
    """The checkpoint does not match the model THIS code builds.

    Carries `missing`, `unexpected`, `shape` (a list of `(key, ckpt_shape,
    model_shape)`) and `ident`, so that the caller can decide for itself
    instead of parsing a string.
    """

    def __init__(self, message: str, missing: List[str], unexpected: List[str],
                 shape: List[Tuple[str, tuple, tuple]], ident: Dict[str, Any]):
        super().__init__(message)
        self.missing = list(missing)
        self.unexpected = list(unexpected)
        self.shape = list(shape)
        self.ident = dict(ident)


def identity(ck: Dict[str, Any]) -> Dict[str, Any]:
    """Run identity taken from the checkpoint. Missing keys come back `None`.

    `None` means the checkpoint does not carry this, not that the value is
    empty. Without `splits_path` and `split_column` there is no way to say
    which rows the model has already not seen, so fine-tuning on one's own
    data can silently train on what will later serve as validation.
    """
    out = {k: ck.get(k, None) for k in IDENTITY_KEYS}
    cfg = ck.get("config") or {}
    out["config_source"] = cfg.get("source")
    out["config_name"] = cfg.get("name")
    out["epoch"] = ck.get("epoch")
    out["step"] = ck.get("step")
    out["method"] = ck.get("method")
    out["n_tensors"] = len(ck.get("model") or {})
    out["brakujace_klucze_tozsamosci"] = [k for k in IDENTITY_KEYS
                                          if ck.get(k) is None]
    return out


def data_file_digests(root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """sha256 (16 chars) of the data files that co-determine the model shape.

    The checkpoint does not store them, so all that can be done is compute
    them now and give the reader a number to compare against the published
    one.
    """
    root = root or os.environ.get("HYPPE_DATA_ROOT") or "."
    out: Dict[str, Optional[str]] = {}
    for rel in DATA_FILES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            out[rel] = hashlib.sha256(
                io.open(p, "rb").read()).hexdigest()[:16]
        else:
            out[rel] = None
    return out


def load_checkpoint(path: str, device: str = "cpu", strict: bool = True):
    """(model in `eval`, checkpoint dict). Raises `CheckpointMismatch`.

    `strict=False` prints the same report and returns a partially loaded
    model: tensors whose shape agrees go in, the rest stays at
    initialisation. That is occasionally what is wanted (old trunk, new
    head), but a partially loaded model reports the metric of a partially
    random one, so it must be a conscious choice rather than the default.
    """
    from .config import from_mapping
    from .model.assemble import HyppeModel

    ck = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ck or "model" not in ck:
        raise CheckpointMismatch(
            "%s does not look like a HYPPE checkpoint: key %r is missing"
            % (path, "config" if "config" not in ck else "model"),
            [], [], [], {"plik": path})
    cfg = from_mapping(ck["config"])
    model = HyppeModel.build(cfg)

    have = model.state_dict()
    want = ck["model"]
    shape = [(k, tuple(want[k].shape), tuple(have[k].shape))
             for k in want
             if k in have and hasattr(want[k], "shape")
             and tuple(want[k].shape) != tuple(have[k].shape)]
    missing = [k for k in have if k not in want]
    unexpected = [k for k in want if k not in have]

    if shape or missing or unexpected:
        ident = identity(ck)
        msg = [
            "checkpoint %s does not match the model THIS code builds." % path,
            "  missing from checkpoint: %d  %s" % (len(missing), missing[:4]),
            "  unexpected             : %d  %s" % (len(unexpected),
                                                   unexpected[:4]),
            "  shape mismatches       : %d" % len(shape),
        ]
        for k, a, b in shape[:6]:
            msg.append("      %-34s ckpt %s  code %s" % (k, a, b))
        msg += [
            "  checkpoint identity    : config.source=%s, epoch=%s, step=%s"
            % (ident.get("config_source"), ident.get("epoch"),
               ident.get("step")),
            "  identity keys the checkpoint does NOT carry: %s"
            % (ident["brakujace_klucze_tozsamosci"] or "none"),
            "",
            "The usual cause is not a corrupt file but code newer than the "
            "checkpoint: head and motif-bank shapes are computed at build "
            "time from the code and the data files, not read from the "
            "checkpoint. Check `data_file_digests()` and `source_mtimes` (if "
            "present), then either return to the code as of the save date or "
            "retrain the heads.",
        ]
        if strict:
            raise CheckpointMismatch("\n".join(msg), missing, unexpected,
                                     shape, ident)
        print("\n".join(msg))
        model.load_state_dict(
            {k: v for k, v in want.items()
             if k in have and tuple(getattr(v, "shape", ())) ==
             tuple(have[k].shape)}, strict=False)
    else:
        model.load_state_dict(want, strict=True)
    model.eval().to(device)
    return model, ck


def describe(path: str, device: str = "cpu") -> str:
    """A report on the checkpoint WITHOUT building the model. Never raises."""
    ck = torch.load(path, map_location=device, weights_only=False)
    ident = identity(ck)
    met = ck.get("metrics") or {}
    lines = ["file              %s (%d B)" % (path, os.path.getsize(path))]
    for k in ("config_source", "config_name", "epoch", "step", "method",
              "n_tensors"):
        lines.append("%-17s %s" % (k, ident.get(k)))
    for k in IDENTITY_KEYS:
        lines.append("%-17s %s" % (k, ident.get(k)
                                   if ident.get(k) is not None
                                   else "<NOT IN CHECKPOINT>"))
    for k in ("pcc", "scc", "pair_acc", "disc_auc", "rec_full", "n_rows"):
        if k in met:
            lines.append("%-17s %s" % ("metrics." + k, met[k]))
    lines.append("data files as they stand now:")
    for k, v in data_file_digests().items():
        lines.append("    %-42s %s" % (k, v or "<FILE MISSING>"))
    return "\n".join(lines)
