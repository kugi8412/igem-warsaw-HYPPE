"""Reconstruction as a constraint, not as a player and not as a fixed weight.

Both alternatives are measured on this objective. As a Nash player it took
2.31 of a 3.0 bargain at recon 0.9998, pushing prediction down to 0.39 and
0.31, a bargain handing the weakest gradient the largest share; as a fixed
weight at effective coefficient 0.15 x 0.40 = 0.060 it collapsed to 0.5549 in
one epoch. The coefficient that holds reconstruction up is not a constant.

A constraint says instead that accuracy must not fall below `target` and the
rest goes to prediction. Ascent on lambda (GECO, Rezende & Viola 2018) is
multiplicative, `lambda <- clamp(lambda * exp(lr * C))` with
`C = target - accuracy_ema`, so the controller cannot change sign and one step
of violation multiplies rather than adds. A controller pinned to a bound is a
constant weight wearing a controller's name: hence `saturated_fraction`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["ConstraintController", "ConstraintState"]


@dataclass
class ConstraintState:
    """One snapshot, for the checkpoint. `lam` is the state that matters.

    `ema` is carried too: restoring `lam` without the accuracy average
    restarts the controller's memory, so it overshoots once per resume.
    """

    lam: float
    ema: Optional[float] = None
    steps: int = 0
    history: List[float] = field(default_factory=list)


class ConstraintController(object):
    """Lagrange multiplier for one inequality constraint `value >= target`.

    For reconstruction `target` is an accuracy, read against
    `RECONSTRUCTION_FLOOR` (0.2626 on this corpus, always writing the most
    frequent base) and never against 0.25. `lam_min` must not be zero, since
    `0 * exp(x) = 0` would kill the controller the first time the constraint
    is comfortably satisfied. `lam_lr` is per step in units of the constraint:
    0.1 of violation at lr 1.0 multiplies lambda by exp(0.1) = 1.105. The
    constraint is evaluated on the `ema` average, a per-batch controller
    chasing sampling noise instead. `warmup_steps` freezes lambda while that
    average fills, the first batches sitting at chance; the observable still
    accumulates.
    """

    def __init__(self, target: float, lam_init: float = 1.0,
                 lam_min: float = 1e-3, lam_max: float = 100.0,
                 lam_lr: float = 1.0, ema: float = 0.98,
                 saturation_limit: float = 0.5, warmup_steps: int = 0,
                 name: str = "reconstruction",
                 multipliers: Optional[Dict[str, float]] = None) -> None:
        if lam_min <= 0.0:
            raise ValueError(
                "lam_min = %r. A multiplicative controller cannot leave zero: "
                "0 * exp(x) = 0 for every x, so lam_min = 0 turns the first "
                "comfortably-satisfied stretch into a permanent switch-off "
                "that looks exactly like a controller doing its job."
                % (lam_min,))
        if not lam_min <= lam_init <= lam_max:
            raise ValueError("lam_init %.4g outside the box [%.4g, %.4g]"
                             % (lam_init, lam_min, lam_max))
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema must be in [0, 1), got %r" % (ema,))
        if lam_lr <= 0.0:
            raise ValueError("lam_lr must be positive; a zero rate is a fixed "
                             "weight and should be configured as one")
        self.target = float(target)
        self.lam = float(lam_init)
        self.lam_init = float(lam_init)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.lam_lr = float(lam_lr)
        self.ema = float(ema)
        self.saturation_limit = float(saturation_limit)
        self.warmup_steps = int(warmup_steps)
        self.name = str(name)
        self.multipliers = dict(multipliers or {})
        self._avg: Optional[float] = None
        self._steps = 0
        self._epoch_steps = 0
        self._n_at_max = 0
        self._n_at_min = 0
        self._lam_hist: List[float] = []
        self._val_hist: List[float] = []
        self._last_c = float("nan")

    # Epochs
    def new_epoch(self) -> None:
        """Reset the per-epoch counters; `lam` and the average survive.

        Zeroing the multiplier at an epoch boundary would restart the search
        and make epoch 1 of every run the same experiment.
        """
        self._epoch_steps = 0
        self._n_at_max = 0
        self._n_at_min = 0
        self._lam_hist = []
        self._val_hist = []

    # Step
    def update(self, value: float) -> float:
        """Feed the observable, return the multiplier for this step.

        It is the multiplier after the update, so a violated constraint is
        answered in the step it was observed, not the next one.
        """
        v = float(value)
        if not math.isfinite(v):
            # A NaN observable must not become a NaN multiplier: the run would
            # die several steps later in the optimiser, pointing nowhere.
            raise ValueError("%s: non-finite observable %r" % (self.name, v))
        self._avg = v if self._avg is None else \
            self.ema * self._avg + (1.0 - self.ema) * v
        self._steps += 1
        self._epoch_steps += 1
        c = self.target - self._avg          # > 0 means VIOLATED
        self._last_c = c
        if self._steps > self.warmup_steps:
            self.lam = self.lam * math.exp(self.lam_lr * c)
            if self.lam >= self.lam_max:
                self.lam = self.lam_max
                self._n_at_max += 1
            elif self.lam <= self.lam_min:
                self.lam = self.lam_min
                self._n_at_min += 1
        self._lam_hist.append(self.lam)
        self._val_hist.append(v)
        return self.lam

    # Contract
    def effective(self) -> Dict[str, float]:
        """lambda times every declared factor on the way to the loss."""
        if not self.multipliers:
            return {"": self.lam}
        return {k: self.lam * float(v)
                for k, v in sorted(self.multipliers.items())}

    def describe(self) -> str:
        eff = " ".join("%s=%.4f" % (k or "flat", v)
                       for k, v in sorted(self.effective().items()))
        return ("%s: CONSTRAINT value >= %.4f | lambda %.4f in [%.4g, %.4g] "
                "| lr %.3g | ema %.3f | effective %s"
                % (self.name, self.target, self.lam, self.lam_min,
                   self.lam_max, self.lam_lr, self.ema, eff))

    # Falsifiers
    def saturated_fraction(self) -> float:
        if not self._epoch_steps:
            return float("nan")
        return (self._n_at_max + self._n_at_min) / float(self._epoch_steps)

    def report(self) -> Dict[str, object]:
        """Per-epoch observables, with the thresholds evaluated.

        `sat` is the fraction of steps with lambda on a bound; `sat_hi` splits
        out `lam_max`, where the constraint was unreachable at any admissible
        price, from `sat_lo` at `lam_min`, where it was slack -- a result, not
        a violation. `c` is `target - average`, positive when violated. A
        `lam_span` of exactly 1.0 means the multiplier never moved, which at
        `sat = 0` means warm-up outlasted the epoch and the update never ran.
        """
        # The floor is reported always, binding or not: a floor doing nothing
        # and a floor holding the whole run look identical in every other key.
        # `effective_floor` goes through the same multiplier chain, a bare
        # weight having once been compared against an effective one.
        eff_floor = ({k: round(self.lam_min * float(v), 6)
                      for k, v in sorted(self.multipliers.items())}
                     if self.multipliers else {"": round(self.lam_min, 6)})
        out: Dict[str, object] = {
            "name": self.name,
            "target": round(self.target, 4),
            "lam": round(self.lam, 6),
            "steps": self._epoch_steps,
            "effective": {k: round(v, 6) for k, v in self.effective().items()},
            "lam_min": self.lam_min,
            "lam_max": self.lam_max,
            # Binding now, on the current multiplier; `sat_lo` is the fraction
            # of the epoch spent on the floor. Neither implies the other.
            "floor_binding": bool(self.lam <= self.lam_min * (1.0 + 1e-12)),
            "effective_floor": eff_floor,
        }
        if not self._epoch_steps:
            out.update({"ok": False, "violations": ["no steps recorded"]})
            return out
        sat_hi = self._n_at_max / float(self._epoch_steps)
        sat_lo = self._n_at_min / float(self._epoch_steps)
        lo, hi = min(self._lam_hist), max(self._lam_hist)
        out.update({
            "value_mean": round(sum(self._val_hist) / len(self._val_hist), 4),
            "value_min": round(min(self._val_hist), 4),
            "value_last": round(self._val_hist[-1], 4),
            "value_ema": round(self._avg if self._avg is not None
                               else float("nan"), 4),
            "c": round(self._last_c, 6),
            "sat": round(sat_hi + sat_lo, 4),
            "sat_hi": round(sat_hi, 4),
            "sat_lo": round(sat_lo, 4),
            "lam_span": round(hi / lo if lo > 0 else float("inf"), 4),
        })
        bad: List[str] = []
        if sat_hi + sat_lo > self.saturation_limit:
            where = "lam_max" if sat_hi >= sat_lo else "lam_min"
            bad.append(
                "sat %.3f > %.2f (mostly at %s): the multiplier sat on a "
                "bound, so this run has a CONSTANT weight of %.4g and "
                "measures the box, not the constraint"
                % (sat_hi + sat_lo, self.saturation_limit, where,
                   self.lam_max if where == "lam_max" else self.lam_min))
        if sat_hi > self.saturation_limit:
            bad.append(
                "sat_hi %.3f: the constraint value >= %.4f was NOT reachable "
                "at any lambda in the box (mean observed %.4f). Either the "
                "target is wrong or the model cannot satisfy it -- raising "
                "lam_max only buys a larger constant."
                % (sat_hi, self.target,
                   sum(self._val_hist) / len(self._val_hist)))
        if abs(hi / lo - 1.0) < 1e-9 and self._steps <= self.warmup_steps:
            bad.append(
                "lambda never moved and the run is still inside warm-up "
                "(%d steps of %d): the controller has not started, and a "
                "frozen multiplier looks identical to a perfectly tracking "
                "one in every other number here"
                % (self._steps, self.warmup_steps))
        out["violations"] = bad
        out["ok"] = not bad
        return out

    def format_report(self) -> str:
        r = self.report()
        if not r.get("steps"):
            return "%s: no steps" % (self.name,)
        # "slack" carries the headroom as a ratio, so the distance to the
        # floor reads without dividing; the binding branch carries the
        # fraction of the epoch spent there.
        if r["floor_binding"]:
            floor_state = "BINDING now, %.1f%% of this epoch's steps" % (
                100.0 * float(r["sat_lo"]),)
        else:
            floor_state = "slack (lambda = %.3gx floor)" % (
                float(r["lam"]) / self.lam_min,)
        return ("%s[%d] value %.4f (ema %.4f, min %.4f) vs target %.4f "
                "| c %+.5f | lambda %.4f (span %.2fx, sat %.3f = hi %.3f / "
                "lo %.3f) | floor %.4g %s | effective %s%s | %s"
                % (self.name, r["steps"], r["value_mean"], r["value_ema"],
                   r["value_min"], r["target"], r["c"], r["lam"],
                   r["lam_span"], r["sat"], r["sat_hi"], r["sat_lo"],
                   self.lam_min, floor_state,
                   " ".join("%s=%.4f" % (k or "flat", v)
                            for k, v in sorted(r["effective"].items())),
                   getattr(self, "_load_clamp_note", ""),
                   "OK" if r["ok"] else "VIOLATIONS: " + " ; ".join(
                       r["violations"])))

    # State
    def state(self) -> ConstraintState:
        return ConstraintState(lam=self.lam, ema=self._avg, steps=self._steps,
                               history=list(self._lam_hist))

    def load_state(self, st: ConstraintState) -> None:
        # The box belongs to the config, not to the checkpoint: a multiplier
        # restored from a run with a different floor would leave the
        # configured one read by nothing for the first steps after a resume.
        # The clamp is announced, being invisible in every other number here.
        lam_in = float(st.lam)
        self.lam = min(max(lam_in, self.lam_min), self.lam_max)
        if self.lam != lam_in:
            self._load_clamp_note = (
                " | RESUME CLAMPED lambda %.6g -> %.6g into the configured "
                "box [%.4g, %.4g]" % (lam_in, self.lam, self.lam_min,
                                      self.lam_max))
        else:
            self._load_clamp_note = ""
        self._avg = None if st.ema is None else float(st.ema)
        self._steps = int(st.steps)
        # `state()` writes the trajectory and nothing read it back, so the
        # first `report()` after a resume measured `lam_span` over the steps
        # since the resume rather than over the epoch.
        self._lam_hist = [float(x) for x in st.history]
