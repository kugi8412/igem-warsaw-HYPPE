"""Provenance stamps, and the refusal to assemble a panel from two models."""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model_io import ROOT, Provenance, sha256  # noqa: E402

DESIGN_FLANKS = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "design_flanks.npz")
TEXT_ARTEFACT = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "experiment_text_emb.npz")
DEFAULT_SPLITS = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "splits", "SPLIT_V6.tsv")
IDENTITY_FIELDS = ("checkpoint_sha256", "config_sha256", "config_source",
                   "splits_sha256", "split_column", "design_flanks_sha256",
                   "text_artefact_sha256")


@dataclass
class Stamp:
    """What produced a number. Compared by hash, printed by path."""
    checkpoint: str
    checkpoint_sha256: str
    config_source: str
    config_sha256: str
    epoch: Optional[int]
    step: Optional[int]
    run_name: str
    splits_path: str
    splits_sha256: str
    split_column: str
    design_flanks: str
    design_flanks_sha256: str
    text_artefact: str
    text_artefact_sha256: str
    experiment_text: str
    text_is_unknown: bool
    constructs: List[str] = field(default_factory=list)
    species: str = "P1"
    created_utc: str = ""
    host: str = ""
    user: str = ""
    python: str = ""
    strategy: str = ""
    note: str = ""

    @classmethod
    def build(cls, provenance: Provenance,
              experiment_text: str,
              text_is_unknown: bool,
              constructs: Sequence[str] = (),
              species: str = "P1",
              strategy: str = "",
              splits_path: str = "",
              design_flanks: str = DESIGN_FLANKS,
              text_artefact: str = TEXT_ARTEFACT,
              note: str = "") -> "Stamp":
        sp = splits_path or provenance.splits_path or DEFAULT_SPLITS
        return cls(
            checkpoint=provenance.checkpoint,
            checkpoint_sha256=provenance.checkpoint_sha256,
            config_source=provenance.config_source,
            config_sha256=provenance.config_sha256,
            epoch=provenance.epoch, step=provenance.step,
            run_name=provenance.run_name,
            splits_path=sp, splits_sha256=sha256(sp),
            split_column=provenance.split_column,
            design_flanks=design_flanks,
            design_flanks_sha256=sha256(design_flanks),
            text_artefact=text_artefact,
            text_artefact_sha256=sha256(text_artefact),
            experiment_text=str(experiment_text or ""),
            text_is_unknown=bool(text_is_unknown),
            constructs=list(constructs),
            species=species,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            host=socket.gethostname(), user=_user(),
            python=platform.python_version(),
            strategy=strategy, note=note)

    @classmethod
    def from_scorer(cls, scorer, strategy: str = "", note: str = "") -> "Stamp":
        """Build straight off a `scoring.Scorer`, so nothing is retyped."""
        return cls.build(
            scorer.prov,
            experiment_text=getattr(scorer, "experiment_text", "") or "",
            text_is_unknown=bool(scorer.text_is_unknown),
            constructs=sorted(scorer.contexts),
            species=scorer.species,
            design_flanks=getattr(scorer, "design_flanks", DESIGN_FLANKS),
            strategy=strategy, note=note)

    def identity(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in IDENTITY_FIELDS}

    def key(self) -> str:
        return json.dumps(self.identity(), sort_keys=True)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def one_line(self) -> str:
        return ("%s ep%s [%s] | cfg %s | splits %s:%s [%s] | flanks [%s] | "
                "text [%s]%s"
                % (os.path.basename(self.checkpoint), self.epoch,
                   self.checkpoint_sha256[:10], self.config_source,
                   os.path.basename(self.splits_path), self.split_column,
                   self.splits_sha256[:10], self.design_flanks_sha256[:10],
                   self.text_artefact_sha256[:10],
                   "  TEXT=UNKNOWN" if self.text_is_unknown else ""))

    def describe(self) -> str:
        lines = ["STAMP  strategy %r" % (self.strategy or "unnamed",),
                 "  checkpoint   %s" % (self.checkpoint,),
                 "               sha256 %s  epoch %s  step %s  run %r"
                 % (self.checkpoint_sha256, self.epoch, self.step,
                    self.run_name),
                 "  config       %s  sha256 %s"
                 % (self.config_source, self.config_sha256 or "(in checkpoint)"),
                 "  splits       %s  column %r  sha256 %s"
                 % (self.splits_path, self.split_column, self.splits_sha256),
                 "  design flanks %s  sha256 %s"
                 % (self.design_flanks, self.design_flanks_sha256),
                 "  text artefact %s  sha256 %s"
                 % (self.text_artefact, self.text_artefact_sha256),
                 "  constructs   %s   species %s"
                 % (", ".join(self.constructs) or "(none)", self.species),
                 "  created      %s on %s by %s (python %s)"
                 % (self.created_utc, self.host, self.user, self.python)]
        if self.text_is_unknown:
            lines.append("  experiment text: UNKNOWN TOKEN -- these numbers "
                         "are NOT conditioned on the WetLab experiment")
        else:
            t = self.experiment_text
            lines.append("  experiment text: %r"
                         % (t if len(t) <= 120 else t[:117] + "..."))
        if self.note:
            lines.append("  note: %s" % (self.note,))
        return "\n".join(lines)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Stamp":
        with open(path) as fh:
            return cls(**json.load(fh))


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:                      # no passwd entry in a container
        return os.environ.get("USER", "?")


def same_model(stamps: Sequence[Stamp]) -> bool:
    return len({s.key() for s in stamps}) <= 1


def differences(stamps: Sequence[Stamp]) -> Dict[str, List[object]]:
    """Which identity fields disagree, and what values were seen."""
    out: Dict[str, List[object]] = {}
    for f in IDENTITY_FIELDS:
        vals = []
        for s in stamps:
            v = getattr(s, f)
            if v not in vals:
                vals.append(v)
        if len(vals) > 1:
            out[f] = vals
    return out


def assert_one_model(stamps: Sequence[Stamp], what: str = "panel") -> Stamp:
    """Refuse to assemble `what` from candidates of different provenance.

    Returns the one common stamp when they agree, so the caller has a single
    object to write into the deliverable.
    """
    stamps = list(stamps)
    if not stamps:
        raise SystemExit("REFUSED: %s has no candidates, so it has no "
                         "provenance either" % (what,))
    if same_model(stamps):
        return stamps[0]
    diff = differences(stamps)
    lines = ["REFUSED to assemble the %s: its candidates were produced by "
             "more than one model/definition." % (what,)]
    for f, vals in diff.items():
        lines.append("  %s differs across %d values:" % (f, len(vals)))
        for v in vals:
            lines.append("      %s" % (v,))
    lines.append("  A pooled ranking across two checkpoints compares numbers "
                 "produced by two different functions, and nothing about the "
                 "ranking looks wrong. Re-score every candidate through ONE "
                 "scorer before ranking them.")
    raise SystemExit("\n".join(lines))


def epoch_warning(stamps: Sequence[Stamp], final_epochs: int = 30) -> str:
    """One sentence about whether this panel came from the finished model."""
    eps = sorted({s.epoch for s in stamps})
    if any(e is None or int(e) < int(final_epochs) for e in eps):
        return ("WIRING DEMONSTRATION, NOT A DESIGN RESULT: epochs %s of %s. "
                "The competition model does not exist yet."
                % (eps, final_epochs))
    return "all candidates come from a finished %d-epoch run" % (final_epochs,)


__all__ = ["Stamp", "IDENTITY_FIELDS", "same_model", "differences",
           "assert_one_model", "epoch_warning", "DESIGN_FLANKS",
           "TEXT_ARTEFACT", "DEFAULT_SPLITS"]
