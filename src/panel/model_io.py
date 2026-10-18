"""Load a checkpoint into a built HyppeModel, or refuse and say why."""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ROOT = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "src")


def _ensure_src_first() -> None:
    """Put `<ROOT>/src` at the front of sys.path, before anything imports."""
    real = os.path.realpath(SRC)
    for p in list(sys.path):
        try:
            if os.path.realpath(p) == real:
                sys.path.remove(p)
        except OSError:                      # unreadable entry; leave it alone
            pass
    sys.path.insert(0, real)


_ensure_src_first()

import torch  # noqa: E402

import hyppe  # noqa: E402
from hyppe.config import from_mapping, load_config  # noqa: E402
from hyppe.model.assemble import HyppeModel  # noqa: E402


def _check_package_identity() -> str:
    """Return the tree `hyppe` really came from, or raise naming both."""
    got = os.path.realpath(os.path.dirname(hyppe.__file__))
    want = os.path.realpath(SRC)
    if os.path.dirname(got) != want:
        raise SystemExit(
            "REFUSED: `hyppe` was imported from %r, not from %r.\n"
            "TWO packages named hyppe exist on this host; %s is an OLD copy "
            "that shadows src/hyppe whenever the working directory is %s. "
            "The import would succeed, every name would resolve, and the "
            "model built would be a different architecture from the one the "
            "checkpoint was trained with. Run with cwd=%s and absolute "
            "paths." % (got, want, os.path.join(ROOT, "hyppe"), ROOT, SRC))
    return got


PACKAGE_DIR = _check_package_identity()


def sha256(path: str) -> str:
    """sha256 of a file, streamed. Empty string for a path that is absent."""
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Provenance:
    """Everything needed to say which object produced a number.

    A dataclass rather than a free dict, so that a missing field is an
    AttributeError at the call site instead of a silent `None` in a report.
    """
    checkpoint: str
    checkpoint_sha256: str
    config_source: str               # "checkpoint" or a filesystem path
    config_sha256: str               # "" when the config came from the blob
    epoch: Optional[int]
    step: Optional[int]
    run_name: str
    method: str
    splits_path: str
    split_column: str
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    header: str = ""
    n_parameters: int = 0
    device: str = "cpu"
    package_dir: str = ""
    key_mismatch_allowed: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_source": self.config_source,
            "config_sha256": self.config_sha256,
            "epoch": self.epoch,
            "step": self.step,
            "run_name": self.run_name,
            "method": self.method,
            "splits_path": self.splits_path,
            "split_column": self.split_column,
            "n_missing_keys": len(self.missing_keys),
            "n_unexpected_keys": len(self.unexpected_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "header": self.header,
            "n_parameters": self.n_parameters,
            "device": self.device,
            "package_dir": self.package_dir,
            "key_mismatch_allowed": self.key_mismatch_allowed,
        }

    def one_line(self) -> str:
        # The key counts are read, not hard-coded: a model loaded with
        # `allow_key_mismatch=True` is exactly the case these fields exist to
        # make visible, and printing a literal 0/0 hid it.
        return ("%s | epoch %s step %s | cfg %s | %d params | keys %d/%d "
                "missing/unexpected | %s"
                % (os.path.basename(self.checkpoint), self.epoch, self.step,
                   self.config_source, self.n_parameters,
                   len(self.missing_keys), len(self.unexpected_keys),
                   self.checkpoint_sha256[:12]))

    def early_epoch_warning(self, final_epochs: int = 30) -> str:
        """The sentence a notebook must print when the model is not final.

        The live run is 30 epochs. A number produced against epoch 3 shows
        that the wiring works and is not a design result; the panel that goes
        to synthesis has to come from the finished model.
        """
        ep = self.epoch
        if ep is None:
            return ("WIRING DEMONSTRATION: this checkpoint carries no epoch "
                    "field, so it cannot be shown to be the final model.")
        if int(ep) >= int(final_epochs):
            return ("checkpoint epoch %d of %d: this is the finished run."
                    % (int(ep), int(final_epochs)))
        return ("WIRING DEMONSTRATION, NOT A DESIGN RESULT: checkpoint epoch "
                "%d of %d. Every number below shows that the path from codes "
                "to DNA to prediction is connected; none of them is a "
                "property of the competition model, which does not exist "
                "yet." % (int(ep), int(final_epochs)))


def load_model(checkpoint: str,
               config: str = "",
               device: str = "cpu",
               allow_key_mismatch: bool = False
               ) -> Tuple[HyppeModel, Provenance]:
    """Build the model, load the weights, refuse on any key mismatch.

    `checkpoint` is a path to a `runs/*.pt` blob, required and never
    defaulted: a hard-coded default is how an earlier generation of these
    scripts ended up pointing at a checkpoint that no longer loads. An empty
    `config` means the config mapping stored inside the checkpoint, which is
    the one the weights were produced under. `allow_key_mismatch` is an escape
    hatch for diagnosis, recorded in the provenance so that a report built
    from such a model says so.
    """
    if not checkpoint:
        raise ValueError(
            "load_model(checkpoint=...) has no default on purpose. Pass the "
            "path; a default silently becomes the competition model.")
    if not os.path.exists(checkpoint):
        raise SystemExit("REFUSED: checkpoint %r does not exist" % (checkpoint,))

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise SystemExit(
            "REFUSED: %s is not a HYPPE checkpoint (no 'model' key; found "
            "%s)." % (checkpoint, sorted(blob)[:8] if isinstance(blob, dict)
                      else type(blob).__name__))

    if config:
        if not os.path.exists(config):
            raise SystemExit("REFUSED: config %r does not exist" % (config,))
        cfg = load_config(config)
        cfg_src, cfg_sha = config, sha256(config)
    else:
        if "config" not in blob:
            raise SystemExit(
                "REFUSED: %s carries no embedded config and none was passed. "
                "Guessing an architecture for a set of weights is exactly "
                "the failure this module exists to prevent." % (checkpoint,))
        cfg = from_mapping(blob["config"], strict=False)
        cfg_src, cfg_sha = "checkpoint", ""

    model = HyppeModel.build(cfg)
    res = model.load_state_dict(blob["model"], strict=False)
    missing = [str(k) for k in res.missing_keys]
    unexpected = [str(k) for k in res.unexpected_keys]

    prov = Provenance(
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        config_source=cfg_src,
        config_sha256=cfg_sha,
        epoch=blob.get("epoch"),
        step=blob.get("step"),
        run_name=str(blob.get("name", "")),
        method=str(blob.get("method", "")),
        splits_path=str(blob.get("splits_path", "")),
        split_column=str(blob.get("split_column", "")),
        missing_keys=missing,
        unexpected_keys=unexpected,
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        device=str(device),
        package_dir=PACKAGE_DIR,
        key_mismatch_allowed=bool(allow_key_mismatch),
    )

    if (missing or unexpected) and not allow_key_mismatch:
        raise SystemExit(
            "REFUSED to return a model: %d missing and %d unexpected keys.\n"
            "  missing   : %s\n"
            "  unexpected: %s\n"
            "A partially loaded model does not crash -- it runs with randomly "
            "initialised sub-modules and reports numbers about an object "
            "nobody built. Fix the config/checkpoint pairing, or pass "
            "allow_key_mismatch=True and accept that every number is stamped "
            "as coming from a model that was not fully restored."
            % (len(missing), len(unexpected),
               ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""),
               ", ".join(unexpected[:12])
               + (" ..." if len(unexpected) > 12 else "")))

    model.eval()
    model.to(torch.device(device))
    prov.header = model.header()
    return model, prov


def load_from_run(run: str = "HYPPE", runs_dir: str = "",
                  device: str = "cpu") -> Tuple[HyppeModel, Provenance]:
    """Newest `<run>_ep*.pt` that actually loads on the current tree.

    For development only, and it prints which file it picked. Older runs do
    not load here (cond_d_lm went 384 -> 768 with the SapBERT prototypes), so
    "newest that loads" is a real search and not a sort. A notebook must still
    take its checkpoint as a parameter.
    """
    import glob
    rd = runs_dir or os.path.join(ROOT, "runs")
    cands = sorted(glob.glob(os.path.join(rd, "%s_ep*.pt" % run)))
    if not cands:
        raise SystemExit("REFUSED: no %s_ep*.pt under %s" % (run, rd))
    errors = []
    for path in reversed(cands):
        try:
            return load_model(path, device=device)
        except SystemExit as exc:
            errors.append("%s: %s" % (os.path.basename(path),
                                      str(exc).splitlines()[0]))
    raise SystemExit("REFUSED: none of %d checkpoints loaded:\n  %s"
                     % (len(cands), "\n  ".join(errors)))


__all__ = ["ROOT", "SRC", "PACKAGE_DIR", "Provenance", "load_model",
           "load_from_run", "sha256"]
