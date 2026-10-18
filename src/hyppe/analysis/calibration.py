"""Is the predicted uncertainty honest, and is the predicted scale honest?

Two calibrations, and the project needs both. Scale: `mu` is trained with
beta-NLL, whose gradient for the mean is a re-weighted MSE, and that shrinks
-- here slope 0.84-0.89 and sd(pred)/sd(target) 0.87. PCC is invariant to an
affine rescale, so a squeezed predictor is invisible in the headline metric
and then breaks the design loop, which compares a candidate against an
absolute threshold. Spread: the design objective `mu - lambda * sigma^2`
reads sigma directly, so an uncalibrated sigma does not degrade that slot but
inverts it.

Affine is the default: two numbers, PCC bit-identical. Isotonic is offered
because a monotone-but-nonlinear miscalibration is real, but it can memorise
a small calibration set, so `fit_isotonic` refuses below `min_points`. Either
is fitted on the validation budget (val, or the spy panel) and applied
unchanged to test, since fitting on test and reporting test is circular.

None of it transfers to the wet readout: GFP protein from a plasmid construct
against labels that are mRNA rank at the native locus, three domain shifts at
once. Order transfers, absolute scale does not, which is what
`Calibration.transferable` is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "Calibration",
    "fit_affine",
    "fit_isotonic",
    "reliability",
    "variance_calibration",
    "interval_coverage",
    "calibration_report",
]

_EPS = 1e-12


# The correction
@dataclass
class Calibration:
    """A fitted correction plus the provenance that makes it quotable."""

    kind: str                                 # "affine" | "isotonic"
    provenance: str                           # which split it was fitted on
    n: int
    slope: float = 1.0
    intercept: float = 0.0
    #: isotonic only: the fitted step function, as (x, y) knots.
    knots_x: Optional[np.ndarray] = None
    knots_y: Optional[np.ndarray] = None
    #: Whether the correction survives the domain shift to the wet readout.
    #: False for every absolute scale here: mRNA rank -> GFP protein crosses
    #: three shifts at once.
    transferable: bool = False
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def apply(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).ravel()
        if self.kind == "affine":
            return self.slope * v + self.intercept
        if self.knots_x is None or self.knots_y is None:
            raise RuntimeError("isotonic calibration has no knots")
        return np.interp(v, self.knots_x, self.knots_y,
                         left=float(self.knots_y[0]),
                         right=float(self.knots_y[-1]))

    def report(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "kind": self.kind,
            "fitted_on": self.provenance,
            "n": int(self.n),
            "transferable_to_wet": bool(self.transferable),
            "note": ("this corrects the SCALE of an mRNA-rank prediction. The "
                     "wet readout is GFP protein from a plasmid construct, "
                     "which differs in quantity (mRNA -> protein), context "
                     "(native locus -> reporter) and task (rank within a "
                     "library -> one absolute value), so it does NOT convert "
                     "a prediction into a fold change. Only the ORDER "
                     "survives."),
        }
        if self.kind == "affine":
            out["slope"] = float(self.slope)
            out["intercept"] = float(self.intercept)
        out.update({k: float(v) for k, v in self.diagnostics.items()})
        return out


def fit_affine(pred: np.ndarray, target: np.ndarray,
               provenance: str = "val") -> Calibration:
    """Least squares `a * pred + b ~ target`. Two numbers, PCC unchanged.

    The diagnostics are the point: `slope` below 1 is a squeezed predictor,
    `sd_ratio` says the same without a fit, and `mse_floor_gap` near zero
    means the corrected error has reached the (1-r^2)*var(y) floor, so no
    monotone rescaling can do better and a fancier calibrator is a waste.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size < 3 or p.std() < _EPS:
        return Calibration(kind="affine", provenance=provenance, n=int(p.size),
                           slope=1.0, intercept=0.0,
                           diagnostics={"degenerate": 1.0})
    a = float(np.cov(p, t, ddof=1)[0, 1] / max(p.var(ddof=1), _EPS))
    b = float(t.mean() - a * p.mean())
    r = float(np.corrcoef(p, t)[0, 1])
    corrected = a * p + b
    mse = float(np.mean((corrected - t) ** 2))
    floor = float((1.0 - r * r) * t.var())
    return Calibration(
        kind="affine", provenance=provenance, n=int(p.size),
        slope=a, intercept=b, transferable=False,
        diagnostics={
            "pcc": r,
            "sd_ratio": float(p.std() / max(t.std(), _EPS)),
            "mse_raw": float(np.mean((p - t) ** 2)),
            "mse_corrected": mse,
            "mse_floor": floor,
            # ~0 means an affine fix is all there is to gain.
            "mse_floor_gap": mse - floor,
        })


def fit_isotonic(pred: np.ndarray, target: np.ndarray,
                 provenance: str = "val",
                 min_points: int = 60) -> Calibration:
    """Pool-adjacent-violators monotone fit; refuses on a small set.

    An isotonic fit on the spy panel's 47 natives has up to 47 blocks and
    reproduces them exactly, so below `min_points` this raises rather than
    return an object that flatters the calibration set.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size < int(min_points):
        raise ValueError(
            "isotonic calibration on %d points is a memorisation of the "
            "calibration set: the fit has up to one block per point. Use "
            "fit_affine (two parameters) or supply at least %d points."
            % (p.size, int(min_points)))
    order = np.argsort(p, kind="mergesort")
    xs, ys = p[order], t[order]
    # PAVA, weights all 1.
    vals: List[float] = []
    wts: List[float] = []
    for y in ys:
        vals.append(float(y))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-1] + wts[-2]
            v = (vals[-1] * wts[-1] + vals[-2] * wts[-2]) / w
            vals.pop(); wts.pop()
            vals[-1] = v
            wts[-1] = w
    fitted = np.concatenate([np.full(int(w), v)
                             for v, w in zip(vals, wts)])
    resid = ys - fitted
    return Calibration(
        kind="isotonic", provenance=provenance, n=int(p.size),
        knots_x=xs, knots_y=fitted, transferable=False,
        diagnostics={
            "n_blocks": float(len(vals)),
            # Close to n means the fit memorised: one block per point.
            "blocks_per_point": float(len(vals)) / float(p.size),
            "mse_corrected": float(np.mean(resid ** 2)),
        })


# Measurement
def reliability(pred: np.ndarray, target: np.ndarray, n_bins: int = 10
                ) -> Dict[str, object]:
    """Binned mean prediction against binned mean target, plus ECE.

    Bins are quantiles of the prediction, not equal-width: on an equal-width
    grid most bins of a squeezed predictor are empty, and the ECE becomes an
    average over whichever bins happened to catch a point.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size < n_bins:
        return {"ece": float("nan"), "bins": [], "n": int(p.size)}
    edges = np.quantile(p, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bins: List[Dict[str, float]] = []
    ece = 0.0
    scale = float(t.std()) or 1.0
    for i in range(int(n_bins)):
        sel = (p >= edges[i]) & (p < edges[i + 1])
        if not sel.any():
            continue
        mp, mt = float(p[sel].mean()), float(t[sel].mean())
        bins.append({"n": float(sel.sum()), "mean_pred": mp,
                     "mean_target": mt, "gap": mp - mt})
        ece += (sel.sum() / float(p.size)) * abs(mp - mt)
    return {"ece": float(ece), "ece_normalised": float(ece / scale),
            "bins": bins, "n": int(p.size)}


def variance_calibration(mu: np.ndarray, log_var: np.ndarray,
                         target: np.ndarray, seed: int = 0
                         ) -> Dict[str, float]:
    """Does the predicted variance track the actual squared error?

    `lv_sd` is the spread of the predicted log-variance, and zero means the
    head learned a constant. `lv_r` is the Spearman between predicted
    variance and squared error; near zero is a constant head with a
    non-constant output. `lv_r_ctrl` repeats it on permuted predictions, and
    `lv_r` equal to it is an artefact of the estimator rather than a signal,
    since squared errors are heavy tailed. `z_var` is the variance of
    (y - mu) / sigma: 1.0 is calibrated, below 1 over-cautious, above 1
    over-confident, the dangerous direction for a loop that trusts sigma.
    """
    m = np.asarray(mu, dtype=np.float64).ravel()
    lv = np.asarray(log_var, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    ok = np.isfinite(m) & np.isfinite(lv) & np.isfinite(t)
    m, lv, t = m[ok], lv[ok], t[ok]
    if m.size < 8:
        return {"n": float(m.size), "lv_sd": float("nan"),
                "lv_r": float("nan"), "z_var": float("nan")}
    err2 = (t - m) ** 2
    var = np.exp(lv)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(var.size)
    z = (t - m) / np.sqrt(np.maximum(var, _EPS))
    return {
        "n": float(m.size),
        "lv_sd": float(lv.std()),
        "lv_r": _spearman(var, err2),
        "lv_r_ctrl": _spearman(var[perm], err2),
        "z_var": float(z.var()),
        "z_mean": float(z.mean()),
        "sigma_mean": float(np.sqrt(np.maximum(var, _EPS)).mean()),
        "rmse": float(np.sqrt(err2.mean())),
    }


def interval_coverage(mu: np.ndarray, log_var: np.ndarray,
                      target: np.ndarray,
                      levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)
                      ) -> Dict[str, float]:
    """Fraction of targets inside the predicted Gaussian interval.

    At level 0.9 the measured coverage should be 0.9; below it the model is
    over-confident. No assumption beyond the one the head already makes by
    emitting a log-variance.
    """
    m = np.asarray(mu, dtype=np.float64).ravel()
    s = np.sqrt(np.exp(np.asarray(log_var, dtype=np.float64).ravel()))
    t = np.asarray(target, dtype=np.float64).ravel()
    ok = np.isfinite(m) & np.isfinite(s) & np.isfinite(t)
    m, s, t = m[ok], s[ok], t[ok]
    out: Dict[str, float] = {"n": float(m.size)}
    if m.size < 8:
        return out
    for lv in levels:
        # Two-sided Gaussian quantile without scipy; the standard levels are
        # tabulated.
        z = {0.5: 0.67449, 0.8: 1.28155, 0.9: 1.64485,
             0.95: 1.95996, 0.99: 2.57583}.get(round(float(lv), 2))
        if z is None:
            continue
        inside = np.abs(t - m) <= z * s
        out["coverage_%d" % int(round(100 * lv))] = float(inside.mean())
        out["coverage_gap_%d" % int(round(100 * lv))] = float(
            inside.mean() - lv)
    return out


def calibration_report(mu: np.ndarray, target: np.ndarray,
                       log_var: Optional[np.ndarray] = None,
                       provenance: str = "val",
                       n_bins: int = 10) -> Dict[str, object]:
    """Everything above in one dict, plus the fitted affine correction."""
    cal = fit_affine(mu, target, provenance=provenance)
    out: Dict[str, object] = {
        "affine": cal.report(),
        "reliability_raw": reliability(mu, target, n_bins),
        "reliability_corrected": reliability(cal.apply(mu), target, n_bins),
    }
    if log_var is not None:
        out["variance"] = variance_calibration(mu, log_var, target)
        out["intervals"] = interval_coverage(mu, log_var, target)
        v = out["variance"]
        # Stated as a verdict: the two ways a variance head fails look
        # different in these numbers and identical in a headline metric.
        if float(v.get("lv_sd", 0.0)) < 1e-4:
            out["variance_verdict"] = (
                "CONSTANT: the variance head learned one number, so "
                "mu - lambda*sigma^2 ranks identically to mu and the "
                "'stable promoter' design slot is not implemented")
        elif abs(float(v.get("lv_r", 0.0))
                 - float(v.get("lv_r_ctrl", 0.0))) < 0.05:
            out["variance_verdict"] = (
                "INDISTINGUISHABLE FROM ITS PERMUTATION CONTROL: the "
                "predicted variance carries no information about the actual "
                "error, whatever its spread")
        else:
            out["variance_verdict"] = "informative (lv_r %.4f vs control %.4f)" \
                % (float(v.get("lv_r", float("nan"))),
                   float(v.get("lv_r_ctrl", float("nan"))))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() < _EPS or rb.std() < _EPS:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(x.size, dtype=np.float64)
    s = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and s[j + 1] == s[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out
