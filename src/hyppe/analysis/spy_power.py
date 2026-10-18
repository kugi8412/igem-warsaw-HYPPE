"""How many choices the spy set will carry before it starts lying.

The spy set exists only to choose between models. Every opening of it is a
maximisation over a noisy number, and the maximum of m draws is biased
upwards even when all the models are identical. This module computes that
bias and turns it into a rule for how many spy comparisons are affordable.

Everything is a Monte Carlo simulation. Closed forms -- E[max] for m = 2, 3;
the sqrt(2 ln m) asymptotic; 1/sqrt(n) -- appear only as sanity controls and
are labelled as such in the output, because a ceiling formula applied to the
wrong quantity once circulated through the documentation for weeks.

The input numbers, each from a measurement elsewhere: `s_eval` = 0.0064 is
the sd of the evaluation alone with frozen weights over 8 loader passes, i.e.
the sampling error of the evaluation set and not a difference between models.
`s_seed` = 0.0045 is the sd between training seeds, a different component
that combines with `s_eval` in quadrature, because two models with different
seeds differ in weights and in the evaluation draw at once. The differences
worth detecting are 0.0038, the spread of 11 deep runs of one family, and
0.0104, the largest difference this track has ever reported as a result.

A saved checkpoint once landed +1.62 sd above the mean of those 8 passes, and
that serves as a control here: the simulated winner bias must be of that
order at an m equal to the number of epochs `select_on` chose from.

What this does not do is measure the training cost of a larger spy, so the
proportion recommendation in part 3 is conditional and says so.

Run with:
  python -m hyppe.analysis.spy_power

  --fast   fewer replicates, for a code-path test, not for reporting
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from . import matrices as data  # noqa: E402
from . import paths

# Input constants; sources are in the module docstring.
SEED = 20260911

S_EVAL = 0.0064
S_SEED = 0.0045
S_COMBINED = float(math.sqrt(S_EVAL ** 2 + S_SEED ** 2))
NOISES = (("s_seed", S_SEED), ("s_eval", S_EVAL), ("combined", S_COMBINED))

MEASURED_SELECTION_SD = 1.62          # szum_oceny.py, CONCLUSION block

DIFFERENCES = (("spread of 11 deep runs", 0.0038),
               ("MambaNet - LegNetV2 (HYPPE)", 0.0104))

M_GRID = (2, 3, 5, 10, 20, 50)

# Tolerances for the rule "spy carries up to N comparisons".
TOLERANCES = ((" 1/2 sd of eval", 0.0032),
              ("   sd of eval  ", 0.0064),
              ("HYPPE difference", 0.0104))

# Part 3 works empirically on og_scope/combat.
SCOPE = "og_scope"
MATRIX = "combat"                       # `rank` is DESCENDING; combat is expression
MIN_IN_COLUMN = 20                      # as in baselines.column_spearman
COL_STEP = 2                            # every second column (control: part 3c)
SIZES = (40, 80, 160, 320, 640, 1159, 2320, 4640)
REPLICATES = (300, 300, 250, 200, 160, 120, 60, 30)
N_SPY_HYPPE = 1159                      # split 9167/1159/1179 (CLAUDE.md)
N_POOL_HYPPE = 11505                    # ML_Dataset, genes (CLAUDE.md)
QUALITY_TARGETS = (0.30, 0.50, 0.70)


def _secs(t0: float) -> str:
    return "%.1fs" % (time.time() - t0)


# Part 1: winner bias
def max_of_m(m: int, s: float, R: int, rng) -> np.ndarray:
    """R realisations of the maximum of m independent N(0, s): models equal."""
    out = np.empty(R, dtype=np.float64)
    chunk = max(1, int(4_000_000 // max(m, 1)))
    i = 0
    while i < R:
        k = min(chunk, R - i)
        out[i:i + k] = rng.normal(0.0, s, size=(k, m)).max(axis=1)
        i += k
    return out


def part1(R: int, rng) -> dict:
    print("\n" + "=" * 78)
    print("PART 1 -- WINNER BIAS  (m models with IDENTICAL truth)")
    print("=" * 78)
    print("  E[max] = by how much the winner's spy score exceeds the TRUTH when")
    print("  there is no difference between the models at all.  %d realisations "
          "per cell." % R)
    print()
    print("  %-4s | %-22s | %-22s | %-22s" % ("m", "s_seed = 0.0045",
                                              "s_eval = 0.0064",
                                              "combined = %.5f" % S_COMBINED))
    print("  %-4s | %-22s | %-22s | %-22s"
          % ("", "E[max]    sd", "E[max]    sd", "E[max]    sd"))
    print("  " + "-" * 74)
    tab: dict = {}
    for m in M_GRID:
        row = []
        for name, s in NOISES:
            mx = max_of_m(m, s, R, rng)
            tab.setdefault(name, {})[m] = {"E_max": float(mx.mean()),
                                           "sd": float(mx.std(ddof=1))}
            row.append("%8.5f %8.5f" % (mx.mean(), mx.std(ddof=1)))
        print("  %-4d | %-22s | %-22s | %-22s" % (m, row[0], row[1], row[2]))

    # Control: closed forms, not the source of any number above.
    print()
    print("  CONTROL (closed form, not used for any conclusion):")
    print("    m=2  E[max]/s: formula 1/sqrt(pi) = %.5f | simulation = %.5f"
          % (1.0 / math.sqrt(math.pi), tab["s_eval"][2]["E_max"] / S_EVAL))
    print("    m=3  E[max]/s: formula 3/(2 sqrt(pi)) = %.5f | simulation = %.5f"
          % (3.0 / (2.0 * math.sqrt(math.pi)),
             tab["s_eval"][3]["E_max"] / S_EVAL))
    print("    m=50 asymptotic sqrt(2 ln m) = %.5f | simulation = %.5f  "
          "(the asymptotic OVERSTATES, which is why it is not used)"
          % (math.sqrt(2 * math.log(50)), tab["s_eval"][50]["E_max"] / S_EVAL))

    # Control: the measured +1.62 sd of the saved checkpoint.
    print()
    print("  CONTROL against the in-situ measurement: `select_on` chose from "
          "~30 epochs,")
    print("  and the saved checkpoint landed +%.2f sd above the mean "
          "(szum_oceny.py)." % MEASURED_SELECTION_SD)
    wsd = {}
    for m in (10, 20, 30, 50):
        mx = max_of_m(m, 1.0, R, rng)
        wsd[m] = (float(mx.mean()), float(mx.std(ddof=1)))
        print("    m=%-3d simulated E[max] = %+.2f sd (sd of the maximum %.2f)"
              % (m, wsd[m][0], wsd[m][1]))
    compatible = [m for m in sorted(wsd)
                  if abs(MEASURED_SELECTION_SD - wsd[m][0]) <= wsd[m][1]]
    print("  -> the measured +%.2f sd is a SINGLE realisation, so it is "
          "compatible with any" % MEASURED_SELECTION_SD)
    print("     m for which |1.62 - E[max]| <= sd of the maximum: m in %s"
          % (compatible if compatible
             else "NONE -- an inconsistency to be explained"))
    if not compatible:
        print("     WARNING: no compatible m means the noise model does not "
              "describe")
        print("     this measurement; part 1 should then be read as a lower "
              "bound.")

    # At what m the bias exceeds the difference we want to detect.
    print()
    print("  AT WHAT m THE BIAS EXCEEDS THE DIFFERENCE WE WANT TO DETECT")
    print("  (smallest m for which E[max] > difference; searched m = 2..300)")
    print()
    print("  %-32s | %-13s | %-13s | %-13s"
          % ("difference", "s_seed", "s_eval", "combined"))
    print("  " + "-" * 78)
    threshold: dict = {}
    Rs = max(40_000, R // 5)
    for label, d in DIFFERENCES:
        cells = []
        for name, s in NOISES:
            found = None
            for m in range(2, 301):
                if max_of_m(m, s, Rs, rng).mean() > d:
                    found = m
                    break
            threshold.setdefault(label, {})[name] = found
            cells.append("m = %s" % (found if found else "> 300"))
        print("  %-32s | %-13s | %-13s | %-13s"
              % ("%s = %.4f" % (label, d), cells[0], cells[1], cells[2]))
    print()
    print("  READING: under the combined noise, the SECOND comparison ALREADY")
    print("  produces a bias larger than the spread of the whole family of 11 "
          "deep runs.")
    return {"E_max": tab, "threshold_m": threshold}


# Part 2: how many times spy may be opened
def selection(m: int, s_spy: float, s_test: float, tau: float, R: int,
              rng) -> dict:
    """m models with true quality N(0, tau); chosen on spy, read on test."""
    theta = rng.normal(0.0, tau, size=(R, m)) if tau > 0 else np.zeros((R, m))
    spy = theta + rng.normal(0.0, s_spy, size=(R, m))
    test = theta + rng.normal(0.0, s_test, size=(R, m))
    w = spy.argmax(axis=1)
    ar = np.arange(R)
    spy_w, th_w, te_w = spy[ar, w], theta[ar, w], test[ar, w]
    th_max = theta.max(axis=1)
    drop = spy_w - te_w
    return {"m": m, "tau": tau,
            "spy_of_winner": float(spy_w.mean()),
            "truth_of_winner": float(th_w.mean()),
            "test_of_winner": float(te_w.mean()),
            "spy_bias": float((spy_w - th_w).mean()),
            "drop_spy_to_test": float(drop.mean()),
            "drop_sd": float(drop.std(ddof=1)),
            "p_picked_best": float((w == theta.argmax(axis=1)).mean()),
            "regret": float((th_max - th_w).mean())}


def part2(R: int, rng) -> dict:
    print("\n" + "=" * 78)
    print("PART 2 -- HOW MANY TIMES SPY MAY BE OPENED")
    print("=" * 78)
    print("  The model chosen on spy is then read on TEST.  The question is:")
    print("  how much of the winner's edge is real and how much evaporates on "
          "test.")
    print("  s_spy = s_test = %.5f (combined).  %d realisations."
          % (S_COMBINED, R))

    for tau, label in (
            (0.0, "tau = 0 -- models TRULY identical "
                  "(the case supported by measurement: spread 0.0038 < seed sd "
                  "0.0045)"),
            (0.0045, "tau = 0.0045 -- real differences of the order of the seed sd"),
            (0.0100, "tau = 0.0100 -- real differences of the order of the "
                     "MambaNet-LegNetV2 gap")):
        print()
        print("  " + label)
        print("  %-4s | %9s | %9s | %9s | %9s | %7s | %8s"
              % ("m", "spy win.", "TRUTH", "test win.", "drop", "P(best)",
                 "regret"))
        print("  " + "-" * 74)
        for m in M_GRID:
            d = selection(m, S_COMBINED, S_COMBINED, tau, R, rng)
            print("  %-4d | %+9.5f | %+9.5f | %+9.5f | %+9.5f | %7.3f | %8.5f"
                  % (m, d["spy_of_winner"], d["truth_of_winner"],
                     d["test_of_winner"], d["drop_spy_to_test"],
                     d["p_picked_best"], d["regret"]))
        if tau == 0.0:
            print("  -> at tau = 0 the TRUTH column is zero by construction;")
            print("     the winner's ENTIRE edge evaporates, and drop = E[max].")

    # The rule: how many comparisons at what tolerance.
    print()
    print("  RULE: the largest m at which the EXPECTED spy->test drop")
    print("  does not exceed the tolerance (tau = 0, i.e. the worst case")
    print("  and the one supported by measurement).  Searched m = 1..300.")
    print()
    print("  %-18s | %-12s | %-12s | %-12s"
          % ("tolerance", "s_seed", "s_eval", "combined"))
    print("  " + "-" * 62)
    Rs = max(40_000, R // 5)
    rule: dict = {}
    for label, tol in TOLERANCES:
        cells = []
        for name, s in NOISES:
            last = 1
            for m in range(2, 301):
                if max_of_m(m, s, Rs, rng).mean() <= tol:
                    last = m
                else:
                    break
            rule.setdefault(label.strip(), {})[name] = last
            cells.append("N = %d" % last)
        print("  %-18s | %-12s | %-12s | %-12s"
              % ("%s = %.4f" % (label, tol), cells[0], cells[1], cells[2]))

    # How many comparisons if the winner must really be best.
    print()
    print("  THE OTHER SIDE, AS A CONTROL: P(the truly best was picked) at")
    print("  tau = 0.0045.  A larger m gives more CHOICE but worse ACCURACY.")
    acc = {}
    for m in M_GRID:
        d = selection(m, S_COMBINED, S_COMBINED, 0.0045, R, rng)
        acc[m] = d
        print("    m=%-3d P(best) = %.3f | regret = %.5f | real edge = %+.5f"
              % (m, d["p_picked_best"], d["regret"], d["truth_of_winner"]))
    return {"rule_N": rule,
            "accuracy_tau0045": {str(k): v for k, v in acc.items()}}


# Part 3: spy size, measured on og_scope/combat
def masked_ranks(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean ranks within a column, over masked entries only.

    Two things must be exact, or this is a different metric from
    `baselines.column_spearman`.  Entries outside the mask get +BIG and sort
    to the end, so masked entries rank as if computed on the subset alone.
    Ties get the mean rank, as in `scipy.stats.rankdata`: 12.4 % of
    `og_scope/combat` entries sit in ties (mostly exact zeros), and ordinal
    ranks deviate by 3.6e-04 against a gate threshold of 1e-04.
    """
    BIG = np.float32(1e30)
    n = A.shape[0]
    X = np.where(mask, A, BIG).astype(np.float32, copy=False)
    o = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, o, axis=0)
    ar = np.arange(n, dtype=np.int32)[:, None]
    idxs = np.broadcast_to(ar, X.shape)
    equal = Xs[1:] == Xs[:-1]
    starts = np.empty(X.shape, dtype=bool)
    starts[0] = True
    starts[1:] = ~equal
    ends = np.empty(X.shape, dtype=bool)
    ends[-1] = True
    ends[:-1] = ~equal
    lo = np.maximum.accumulate(np.where(starts, idxs, np.int32(-1)), axis=0)
    hi = np.flip(np.minimum.accumulate(
        np.flip(np.where(ends, idxs, np.int32(n)), axis=0), axis=0), axis=0)
    mean_rank = (lo + hi).astype(np.float32) * np.float32(0.5)
    r = np.empty(X.shape, dtype=np.float32)
    np.put_along_axis(r, o, mean_rank, axis=0)
    return r


def column_spearman(Y: np.ndarray, P: np.ndarray,
                    min_in_column: int = MIN_IN_COLUMN,
                    ry: np.ndarray | None = None) -> tuple[float, int]:
    """Mean per-column Spearman, the same definition as
    `baselines.column_spearman` but vectorised; checked in `metric_gate`."""
    m = np.isfinite(Y) & np.isfinite(P)
    if ry is None:
        ry = masked_ranks(Y, m)
    rp = masked_ranks(P, m)
    mf = m.astype(np.float32)
    k = mf.sum(axis=0)
    ry = ry * mf
    rp = rp * mf
    sy, sp = ry.sum(axis=0), rp.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (ry * rp).sum(axis=0) - sy * sp / k
        vy = (ry * ry).sum(axis=0) - sy * sy / k
        vp = (rp * rp).sum(axis=0) - sp * sp / k
        r = cov / np.sqrt(vy * vp)
    ok = (k >= min_in_column) & np.isfinite(r)
    if not ok.any():
        return float("nan"), 0
    return float(r[ok].mean()), int(ok.sum())


def metric_gate(M: np.ndarray, P: np.ndarray, rng) -> None:
    """Falsifier: the vectorised metric must agree with scipy."""
    from scipy.stats import spearmanr
    sub = rng.choice(M.shape[0], 500, replace=False)
    Y, Pp = M[sub], P[sub]
    v, _ = column_spearman(Y, Pp)
    out = []
    for j in range(Y.shape[1]):
        mm = np.isfinite(Y[:, j]) & np.isfinite(Pp[:, j])
        if mm.sum() < MIN_IN_COLUMN or np.std(Pp[mm, j]) < 1e-12:
            continue
        out.append(spearmanr(Y[mm, j], Pp[mm, j]).statistic)
    s = float(np.nanmean(out))
    print("  METRIC GATE: vectorised %.6f | scipy %.6f | difference %.2e"
          % (v, s, abs(v - s)))
    if not abs(v - s) < 1e-4:
        raise SystemExit("GATE FAILED: vectorised metric != scipy.spearmanr")


def build_predictor(M: np.ndarray, z: np.ndarray, sigma: float,
                    noise: np.ndarray) -> np.ndarray:
    """Predictor = row effect (per promoter) + noise, quality set by sigma.

    A column effect is invisible to the metric -- a within-column Spearman is
    invariant to adding a constant to the whole column -- so the `col - g`
    term of `baselines.fit_predict` changes no number in part 3.
    """
    return (z + sigma * noise)[:, None].astype(np.float32)


def calibrate(M: np.ndarray, z: np.ndarray, noise: np.ndarray, target: float,
              sub: np.ndarray) -> tuple[float, float]:
    """Bisection over log sigma: find the sigma giving a column Spearman ~= target."""
    lo, hi = math.log(1e-3), math.log(1e3)
    Y = M[sub]
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        P = build_predictor(M, z, math.exp(mid), noise)
        v, _ = column_spearman(Y, np.broadcast_to(P[sub], Y.shape))
        if v > target:
            lo = mid
        else:
            hi = mid
    sig = math.exp(0.5 * (lo + hi))
    P = build_predictor(M, z, sig, noise)
    v, _ = column_spearman(Y, np.broadcast_to(P[sub], Y.shape))
    return sig, v


def part3(rng, fast: bool) -> dict:
    print("\n" + "=" * 78)
    print("PART 3 -- SPY SIZE  (empirical: %s/%s)" % (SCOPE, MATRIX))
    print("=" * 78)
    t0 = time.time()
    idx, cols, M = data.load_orthogroup_matrix(SCOPE, MATRIX)
    M = M[:, ::COL_STEP]
    N, C = M.shape
    print("  matrix %d orthogroups x %d samples (every %dth column), NaN "
          "%.2f %%, %s"
          % (N, C, COL_STEP, 100 * (~np.isfinite(M)).mean(), _secs(t0)))

    g = float(np.nanmean(M))
    row = np.nanmean(M, axis=1)
    row = np.where(np.isfinite(row), row, g)
    z = (row - row.mean()) / row.std()
    noise = rng.standard_normal(N)
    sub_cal = rng.choice(N, 2000, replace=False)

    predictors = []
    print()
    print("  PREDICTORS (a stand-in for the model; quality set by calibration):")
    for target in QUALITY_TARGETS:
        sig, _reached = calibrate(M, z, noise, target, sub_cal)
        P = build_predictor(M, z, sig, noise)
        full, nk = column_spearman(M, np.broadcast_to(P, M.shape))
        predictors.append((target, sig, P, full))
        print("    target %.2f -> sigma %.4f | on the FULL pool %.4f "
              "(columns %d)" % (target, sig, full, nk))
    P0 = build_predictor(M, z, 0.0, noise)
    full0, _ = column_spearman(M, np.broadcast_to(P0, M.shape))
    predictors.append(("sigma=0", 0.0, P0, full0))
    print("    sigma 0 (the bare row effect) -> %.4f  -- the upper end, a "
          "predictor computed from the same cells" % full0)

    metric_gate(M, np.broadcast_to(predictors[0][2], M.shape).copy(), rng)

    sizes = SIZES if not fast else (40, 160, 640, 1159)
    reps = REPLICATES if not fast else (40, 30, 20, 15)

    print()
    print("  sd OF THE METRIC against SPY SIZE.  Subsamples WITHOUT "
          "REPLACEMENT from a pool")
    print("  of %d rows; `sd_inf` is the sd after removing the finite-pool "
          "correction" % N)
    print("  (FPC: Var_without_repl = Var_inf * (N-n)/(N-1)), i.e. the sd a "
          "spy of that")
    print("  size would have if drawn from the population of promoters.")
    results: dict = {}
    for target, sig, P, full in predictors:
        key = ("quality %.2f" % full) if not isinstance(target, str) \
            else "sigma=0"
        print()
        print("  --- predictor %s (column Spearman on the pool %.4f) ---"
              % (key, full))
        print("  %-6s | %-4s | %9s | %9s | %9s | %7s"
              % ("n", "R", "mean", "sd", "sd_inf", "columns"))
        print("  " + "-" * 58)
        series = {}
        for n, R in zip(sizes, reps):
            t1 = time.time()
            v = np.empty(R)
            nk = 0
            for r in range(R):
                sub = rng.choice(N, n, replace=False)
                Y = M[sub]
                v[r], nk = column_spearman(Y, np.broadcast_to(P[sub], Y.shape))
            sd = float(np.nanstd(v, ddof=1))
            fpc = math.sqrt((N - n) / (N - 1.0))
            series[n] = {"mean": float(np.nanmean(v)), "sd": sd,
                         "sd_inf": sd / fpc, "R": R, "columns": nk,
                         "time_s": round(time.time() - t1, 1)}
            print("  %-6d | %-4d | %9.5f | %9.5f | %9.5f | %7d"
                  % (n, R, series[n]["mean"], sd, series[n]["sd_inf"], nk))
        # Exponent: is it really 1/sqrt(n)?
        ln = np.log(np.array(list(series.keys()), dtype=float))
        ls = np.log(np.array([series[k]["sd_inf"] for k in series]))
        beta, a = np.polyfit(ln, ls, 1)
        print("  fit log sd_inf = a + beta * log n:  beta = %+.4f"
              "   (CONTROL: pure sampling error would give -0.5)" % beta)
        results[key] = {"sigma": sig, "quality_pool": full,
                        "series": {str(k): v for k, v in series.items()},
                        "beta": float(beta), "a": float(a)}

    # Control A: the bootstrap must agree with sd_inf.
    print()
    print("  CONTROL A -- bootstrap WITH REPLACEMENT (needs no FPC) against")
    print("  sd_inf from subsamples without replacement.  Agreement = the FPC "
          "is right.")
    target, sig, P, full = predictors[1]
    for n in (320, 1159):
        if str(n) not in results["quality %.2f" % full]["series"]:
            continue
        R = 120 if not fast else 25
        v = np.empty(R)
        for r in range(R):
            sub = rng.integers(0, N, size=n)
            Y = M[sub]
            v[r], _ = column_spearman(Y, np.broadcast_to(P[sub], Y.shape))
        boot = float(np.nanstd(v, ddof=1))
        ref = results["quality %.2f" % full]["series"][str(n)]["sd_inf"]
        print("    n=%-5d bootstrap sd %.5f | sd_inf from subsamples %.5f | "
              "ratio %.3f" % (n, boot, ref, boot / ref))
    print("    (the bootstrap duplicates rows and therefore creates rank ties, "
          "so a ratio")
    print("     within ~0.9-1.15 is taken as agreement)")

    # Control B: does the number of columns change sd?
    print()
    print("  CONTROL B -- does sd depend on the number of COLUMNS (samples), "
          "or only")
    print("  on the number of ROWS (promoters).  n = %d, the same predictor,"
          % N_SPY_HYPPE)
    print("  columns thinned 1x / 2x / 4x.")
    for step in (1, 2, 4):
        Mk = M[:, ::step] if step > 1 else M
        Pk = P
        R = 60 if not fast else 15
        v = np.empty(R)
        for r in range(R):
            sub = rng.choice(N, N_SPY_HYPPE, replace=False)
            Y = Mk[sub]
            v[r], nk = column_spearman(Y, np.broadcast_to(Pk[sub], Y.shape))
        print("    columns %5d -> sd %.5f  (mean %.5f)"
              % (Mk.shape[1], float(np.nanstd(v, ddof=1)),
                 float(np.nanmean(v))))
    print("    -> if sd barely rises with 4x FEWER columns, the evaluation "
          "noise is")
    print("       ROW-WISE (per promoter) and the number of samples does not "
          "cut it.")
    return results


def n_comparisons(s: float, tol: float, R: int, rng, maximum: int = 5000) -> int:
    """Largest m at which E[max of m] <= tol, found by bisection: at a small
    s the answer runs into the hundreds and a loop over m is too slow."""
    if max_of_m(2, s, 20_000, rng).mean() > tol:
        return 1
    lo, hi = 2, maximum
    if max_of_m(maximum, s, 20_000, rng).mean() <= tol:
        return maximum
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if max_of_m(mid, s, R, rng).mean() <= tol:
            lo = mid
        else:
            hi = mid
    return lo


def part3_conclusions(w3: dict, rng) -> dict:
    """Turn the measured shape of sd(n) into a spy proportion.

    Two frames of reference meet here and must not be mixed.  Scale A is the
    HYPPE model metric, the source of 0.0064 and 0.0045; scale B is
    og_scope/combat, measured in part 3.  Only the dimensionless exponent
    beta crosses from B to A; the level comes from the in-situ measurement on
    scale A.

    That carry-over assumes one thing, named here: both metrics belong to the
    same family -- a mean rank correlation computed within a context, over
    genomes in one case and over samples in the other -- and
    `GenomeBatchSampler` draws rows, so 0.0064 is the sampling error of rows
    at n = 1159, the same kind of number whose shape part 3 measures.
    """
    print("\n" + "-" * 78)
    print("  WHAT THIS MEANS FOR THE 0.10 PROPORTION")
    print("-" * 78)
    betas = {k: w3[k]["beta"] for k in w3}
    print("  1. SHAPE.  The exponent beta measured at four qualities:")
    for k in w3:
        print("       %-16s (quality %.4f) beta = %+.4f"
              % (k, w3[k]["quality_pool"], w3[k]["beta"]))
    bt = float(np.mean(list(betas.values())))
    print("     mean %.4f, spread %.4f.  The falsifier was set BEFORE the"
          % (bt, max(betas.values()) - min(betas.values())))
    print("     measurement: pure sampling error gives -0.5.  Passed.")
    print("     -> doubling spy cuts sd by %.3f; quadrupling it by %.3f."
          % (2.0 ** bt, 4.0 ** bt))

    print()
    print("  2. The LEVEL depends on model QUALITY, not only on n.  sd_inf at")
    print("     n = %d (10 %% of the pool), same set, different predictors:"
          % N_SPY_HYPPE)
    for k in w3:
        ser = w3[k]["series"].get(str(N_SPY_HYPPE))
        if ser:
            print("       quality %.4f -> sd_inf %.5f"
                  % (w3[k]["quality_pool"], ser["sd_inf"]))
    print("     -> a better model is CHEAPER to evaluate.  These numbers are on")
    print("        scale B (combat) and MUST NOT be quoted as HYPPE metric "
          "noise.")

    print()
    print("  3. The noise is ROW-WISE.  Control B: 4x fewer columns does not "
          "raise sd")
    print("     distinguishably from the noise of sd itself.  Enlarging the "
          "sample")
    print("     panel is no substitute for enlarging the promoter set.")

    # Anchor: level from scale A, shape from scale B.
    print()
    print("  4. THE PROPORTION, COMPUTED.  Anchored curve:")
    print("       s_spy(p) = %.5f * (0.10 / p)^%.3f" % (S_EVAL, -bt))
    print("     (level: the in-situ measurement at p = 0.10; shape: part 3)")
    print("     Comparing TWO models with different seeds adds the seed noise")
    print("     %.5f, which does NOT depend on spy size -- that is the FLOOR."
          % S_SEED)
    print()
    print("     N = how many spy comparisons are affordable (E[max] <= "
          "tolerance), tau = 0:")
    print("     %-8s | %-7s | %-9s | %-9s | %-9s | %-9s"
          % ("p", "n", "s_spy", "combined", "N(0.0064)", "N(0.0104)"))
    print("     " + "-" * 66)
    prop: dict = {}
    for p_ in (0.05, 0.10, 0.20, 0.40, 0.80):
        s_spy = S_EVAL * (0.10 / p_) ** (-bt)
        comp = math.sqrt(s_spy ** 2 + S_SEED ** 2)
        n1 = n_comparisons(comp, 0.0064, 60_000, rng)
        n2 = n_comparisons(comp, 0.0104, 60_000, rng)
        prop["%.2f" % p_] = {"n": int(round(p_ * N_POOL_HYPPE)),
                             "s_spy": s_spy, "combined": comp,
                             "N_tol0064": n1, "N_tol0104": n2}
        print("     %-8.2f | %-7d | %9.5f | %9.5f | %9d | %9d"
              % (p_, round(p_ * N_POOL_HYPPE), s_spy, comp, n1, n2))
    n1f = n_comparisons(S_SEED, 0.0064, 60_000, rng)
    n2f = n_comparisons(S_SEED, 0.0104, 60_000, rng)
    prop["floor"] = {"s_spy": 0.0, "combined": S_SEED,
                     "N_tol0064": n1f, "N_tol0104": n2f}
    print("     %-8s | %-7s | %9.5f | %9.5f | %9d | %9d"
          % ("p -> 1", "-", 0.0, S_SEED, n1f, n2f))
    print()
    print("     READING: the gain from enlarging spy SATURATES at the seed "
          "floor.")
    print("     Going 0.10 -> 0.20 costs %.1f %% of the training rows"
          % (100 * 0.10 * N_POOL_HYPPE / 9167.0))
    print("     (9167 -> %d), and going 0.20 -> 0.40 adds almost nothing"
          % (9167 - round(0.10 * N_POOL_HYPPE)))
    print("     above the floor.  This module does NOT measure the training "
          "cost.")

    # The other lever: averaging over seeds.
    print()
    print("  5. A LEVER SPY SIZE DOES NOT HAVE: averaging over k seeds cuts")
    print("     BOTH components by sqrt(k).  At p = 0.10:")
    for k in (1, 2, 3, 5):
        comp = math.sqrt(S_EVAL ** 2 + S_SEED ** 2) / math.sqrt(k)
        print("       k = %d -> combined %.5f | N(0.0104) = %d"
              % (k, comp, n_comparisons(comp, 0.0104, 60_000, rng)))
    print("     Two seeds per model buy more than doubling spy, and take not a")
    print("     single row away from training.")
    return {"beta_mean": bt, "betas": betas, "proportions": prop}


# Part 4: the decision rule
def part4(R: int, rng) -> dict:
    print("\n" + "=" * 78)
    print("PART 4 -- THE DECISION RULE")
    print("=" * 78)
    print("  NULL distribution of the difference of two models on spy: both "
          "have the")
    print("  same true quality, each measurement carries combined noise %.5f."
          % S_COMBINED)
    print("  %d realisations." % R)
    print()
    print("  %-16s | %9s | %9s | %9s"
          % ("noise per model", "sd of diff", "95 % |d|", "99 % |d|"))
    print("  " + "-" * 54)
    threshold: dict = {}
    for name, s in NOISES:
        d = rng.normal(0.0, s, R) - rng.normal(0.0, s, R)
        q95, q99 = np.quantile(np.abs(d), [0.95, 0.99])
        threshold[name] = {"sd": float(d.std(ddof=1)), "q95": float(q95),
                           "q99": float(q99)}
        print("  %-16s | %9.5f | %9.5f | %9.5f"
              % (name, d.std(ddof=1), q95, q99))
    print("  CONTROL (formula): sd of the difference = sqrt(2) * s = %.5f "
          "against the simulated %.5f"
          % (math.sqrt(2) * S_COMBINED, threshold["combined"]["sd"]))

    print()
    print("  AND IF SPY HAS ALREADY BEEN OPENED m TIMES: the threshold must "
          "apply to the")
    print("  LARGEST of m differences from the mean, not to one picked after "
          "the fact.")
    print("  95 % quantile of (max_i obs_i - mean obs), combined noise:")
    multi = {}
    for m in M_GRID:
        x = rng.normal(0.0, S_COMBINED, size=(min(R, 200_000), m))
        stat = x.max(axis=1) - x.mean(axis=1)
        q = float(np.quantile(stat, 0.95))
        multi[m] = q
        print("    m = %-3d -> %.5f" % (m, q))

    sentence = (
        "A difference between two models on the spy set is real only above "
        "%.3f of a metric point (95 %%, two models with different seeds, ONE "
        "comparison declared before the measurement); if spy has already been "
        "opened m times, what counts is the winner's excess over the MEAN of "
        "all m results, with a threshold of %.3f (m=3), %.3f (m=10), %.3f "
        "(m=50), because the maximum of m noisy numbers alone exceeds the "
        "truth by %.3f at m=10 even when the true difference is zero."
        % (threshold["combined"]["q95"], multi[3], multi[10], multi[50],
           max_of_m(10, S_COMBINED, 200_000, rng).mean()))
    print()
    print("  SENTENCE FOR THE DOCUMENTATION (source: "
          "hyppe/analysis/spy_power.py,")
    print("  part 4, %d realisations, combined noise %.5f, seed %d):"
          % (R, S_COMBINED, SEED))
    print()
    for line in textwrap.wrap(sentence, width=72):
        print("    " + line)
    return {"pair": threshold, "multiple_q95": multi, "sentence": sentence}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spy_power",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true",
                    help="fewer replicates; for a code-path test, NOT for "
                         "reporting")
    ap.add_argument("--load", default=None,
                    help="load PART 3 from an existing JSON instead of "
                         "recomputing it (the empirical part takes ~19 min; "
                         "parts 1/2/4 are always computed)")
    ap.add_argument("-o", "--out",
                    default=os.path.join(paths.LOGS, "spy_power.json"))
    a = ap.parse_args(argv)

    R = 40_000 if a.fast else 300_000
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("spy_power -- how many choices the SPY set will carry")
    print("=" * 78)
    print("  seed %d | R = %d | mode %s"
          % (SEED, R, "FAST (do not report)" if a.fast else "full"))
    print("  s_eval   = %.5f  (hyppe/szum_oceny.py, 8 passes, frozen weights)"
          % S_EVAL)
    print("  s_seed   = %.5f  (CLAUDE.md, DeepSTARR track)" % S_SEED)
    print("  combined = %.5f  = sqrt(s_eval^2 + s_seed^2)" % S_COMBINED)

    t0 = time.time()
    w1 = part1(R, rng)
    w2 = part2(R, rng)
    if a.load:
        with open(a.load, encoding="utf-8") as f:
            w3 = json.load(f)["part3"]
        print("\n[spy_power] PART 3 loaded from %s (the empirical part was NOT "
              "recomputed)" % a.load)
    else:
        w3 = part3(rng, a.fast)
    w3b = part3_conclusions(w3, rng)
    w4 = part4(R, rng)

    print("\n" + "=" * 78)
    print("SUMMARY  (total %s)" % _secs(t0))
    print("=" * 78)
    print("  1. Winner bias under combined noise: m=2 %.5f | m=5 %.5f"
          " | m=10 %.5f | m=20 %.5f"
          % (w1["E_max"]["combined"][2]["E_max"],
             w1["E_max"]["combined"][5]["E_max"],
             w1["E_max"]["combined"][10]["E_max"],
             w1["E_max"]["combined"][20]["E_max"]))
    print("  2. Spy carries N comparisons at tolerance: "
          + " | ".join("%s: N=%d" % (k.strip(), v["combined"])
                       for k, v in w2["rule_N"].items()))
    print("  3. The metric sd falls like n^%+.3f (4 independent measurements); "
          "at p = 0.20 spy carries N = %d comparisons against %d at 0.10"
          % (w3b["beta_mean"], w3b["proportions"]["0.20"]["N_tol0104"],
             w3b["proportions"]["0.10"]["N_tol0104"]))
    print("  4. Reality threshold for a difference of two models (95 %%, "
          "combined noise): %.4f" % w4["pair"]["combined"]["q95"])

    paths.ensure_dir(os.path.dirname(a.out) or ".")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "R": R, "fast": bool(a.fast),
                   "s_eval": S_EVAL, "s_seed": S_SEED,
                   "s_combined": S_COMBINED,
                   "part1": w1, "part2": w2, "part3": w3,
                   "part3_conclusions": w3b, "part4": w4},
                  f, ensure_ascii=False, indent=1)
    print("\n[spy_power] written to %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
