"""What a panel of n = 10 synthetic promoters can detect.

Everything here is a Monte Carlo simulation; no result comes from a
closed-form expression. The formulas (Fisher z, Spearman-Brown) appear only
as sanity controls and are marked as such in the output.

The context is fixed and not measured here: a GFP readout from a
promoter::GFP construct in Trichoderma atroviride P1, zero-shot inference
with no design-build-test rounds, and a budget of exactly 10 promoters.

Three input numbers are assumed rather than measured. `rho_model` = 0.50 is a
defensible prediction quality on held-out data, and is a correlation with
mRNA rank, not with GFP. The mRNA->protein ceiling times measurement noise is
a parameter with scenarios 0.90 / 0.75 / 0.60, computed separately in
`noise_ceiling.py`. The reference points are tef1 at y = 0.9983 and native
pks1 at y = 0.9437, the lowest of three recorded values.

`y` is a quantile of the mRNA expression rank, so converting it to a z scale
through Phi^-1 assumes the shape of the distribution. That conversion is
needed only in part 4, the question about strength, and is flagged there;
parts 1, 2, 3 and 5 test Spearman and are invariant to a monotone rescaling.

Run with:
  python -m hyppe.analysis.panel_power [--extra | --extra2]
"""

from __future__ import annotations

import sys
import time
from math import asin, pi, sqrt

import numpy as np
from scipy import stats as st

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RNG = np.random.default_rng(20260911)

ALPHA = 0.05
POWER_TARGET = 0.80

# Ceiling scenarios: a parameter, not a measurement.
CEILINGS = {"0.90": 0.90, "0.75": 0.75, "0.60": 0.60}
RHO_MODEL = 0.50          # brief: ~0.5 rank correlation on held-out data

# Reference points on the rank scale.
Y_TEF1 = 0.9983
Y_PKS1 = 0.9437

Z_TEF1 = float(st.norm.ppf(Y_TEF1))     # 2.929
Z_PKS1 = float(st.norm.ppf(Y_PKS1))     # 1.586

N_POOL = 5000            # how many candidates the design pipeline generates
                         # (an assumption)


# Vectorised Pearson and Spearman over the rows of an (R x n) matrix.

def pearson_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Xc * Yc).sum(axis=1)
    den = np.sqrt((Xc * Xc).sum(axis=1) * (Yc * Yc).sum(axis=1))
    return num / np.where(den == 0.0, np.nan, den)


def row_ranks(X: np.ndarray) -> np.ndarray:
    """Ranks within a row.  Continuous data -> no ties, two argsorts suffice."""
    R, n = X.shape
    order = np.argsort(X, axis=1, kind="stable")
    out = np.empty((R, n), dtype=np.float64)
    out[np.arange(R)[:, None], order] = np.arange(n, dtype=np.float64)[None, :]
    return out


def spearman_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return pearson_rows(row_ranks(X), row_ranks(Y))


# Critical values from a simulated null distribution, not from tables.

_CRIT: dict[int, tuple] = {}


def critical_values(n: int, R: int = 400_000, alpha: float = ALPHA) -> tuple:
    """Return (c_pearson, alpha_pearson, c_spearman, alpha_spearman) for n.

    At small n the Spearman statistic has a DISCRETE distribution, so the
    0.95 quantile does not give an alpha of exactly 0.05.  We take the
    smallest ATTAINABLE threshold at which P(|rs| >= threshold) <= alpha, and
    report the alpha actually achieved.  A falsifier that cannot be satisfied
    by construction is worse than none -- so the threshold must be attainable
    and its alpha must be known.
    """
    if n in _CRIT:
        return _CRIT[n]
    x = RNG.standard_normal((R, n))
    y = RNG.standard_normal((R, n))
    ap = np.abs(pearson_rows(x, y))
    asr = np.abs(spearman_rows(x, y))

    cp = float(np.quantile(ap, 1.0 - alpha))
    a_p = float((ap >= cp).mean())

    srt = np.sort(asr)
    uq = np.unique(np.round(asr, 12))
    sf = 1.0 - np.searchsorted(srt, uq, side="left") / srt.size
    admissible = uq[sf <= alpha]
    cs = float(admissible.min())
    a_s = float((asr >= cs).mean())

    _CRIT[n] = (cp, a_p, cs, a_s)
    return _CRIT[n]


# Power under a bivariate normal with correlation rho.

def power(rho: float, n: int, R: int = 40_000) -> tuple:
    cp, _, cs, _ = critical_values(n)
    z1 = RNG.standard_normal((R, n))
    z2 = RNG.standard_normal((R, n))
    x = z1
    y = rho * z1 + sqrt(max(0.0, 1.0 - rho * rho)) * z2
    mp = float((np.abs(pearson_rows(x, y)) >= cp).mean())
    ms = float((np.abs(spearman_rows(x, y)) >= cs).mean())
    return mp, ms


def rho_for_power(n: int, target: float = POWER_TARGET, R: int = 40_000,
                  step: float = 0.01) -> tuple:
    """Smallest population rho (Pearson) giving power >= target; separately for
    the Pearson and the Spearman test.  Linear interpolation between grid
    points."""
    grid = np.arange(0.0, 0.995, step)
    mp = np.empty(grid.size)
    ms = np.empty(grid.size)
    for i, r in enumerate(grid):
        mp[i], ms[i] = power(float(r), n, R=R)

    def cross(curve):
        idx = np.argmax(curve >= target)
        if not (curve >= target).any():
            return float("nan")
        if idx == 0:
            return float(grid[0])
        x0, x1 = grid[idx - 1], grid[idx]
        y0, y1 = curve[idx - 1], curve[idx]
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    return cross(mp), cross(ms), grid, mp, ms


def population_spearman_rho(pearson_rho: float) -> float:
    """For a bivariate normal: rho_S = (6/pi) * arcsin(rho/2).
    A theoretical control; the measured simulation sits below it."""
    return 6.0 / pi * asin(pearson_rho / 2.0)


# Part 1

def part1() -> dict:
    print("=" * 78)
    print("PART 1.  THE CORRELATION DETECTABLE AT n = 10")
    print("=" * 78)

    n = 10
    cp, a_p, cs, a_s = critical_values(n)
    print("\nCritical values from a simulated null distribution (R = 400 000):")
    print("  |r| Pearson    >= %.4f   actual alpha %.4f   "
          "(control: the t formula gives %.4f)"
          % (cp, a_p, float(st.t.ppf(1 - ALPHA / 2, n - 2)
                            / sqrt(st.t.ppf(1 - ALPHA / 2, n - 2) ** 2 + n - 2))))
    print("  |rs| Spearman  >= %.4f   actual alpha %.4f   "
          "(DISCRETE distribution, alpha < 0.05 by construction)" % (cs, a_s))

    rp, rs_, grid, mp, ms = rho_for_power(n)
    print("\nPopulation correlation needed for power 0.80 at alpha 0.05 "
          "two-sided, n = 10:")
    print("  Pearson test  : rho = %.3f" % rp)
    print("  Spearman test : rho = %.3f  (population rho_S = %.3f)"
          % (rs_, population_spearman_rho(rs_)))
    print("\n  power curve (Spearman test, n = 10):")
    for r in (0.2, 0.3, 0.375, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        m1, m2 = power(r, n, R=40_000)
        print("    rho = %.3f   power Pearson %.3f   power Spearman %.3f"
              % (r, m1, m2))

    # Confidence interval for an observed r = 0.6 at n = 10.
    print("\n95%% confidence interval for an OBSERVED coefficient of 0.60, n = 10")
    print("  (test inversion: find the rho at which 0.60 sits on the edge of "
          "the 2.5%% tail; simulation, not a Fisher transform)")

    def ci_inversion(obs: float, stat: str, R: int = 60_000) -> tuple:
        grid = np.arange(-0.60, 0.996, 0.01)
        p_ge = np.empty(grid.size)   # P(statistic >= obs)
        p_le = np.empty(grid.size)   # P(statistic <= obs)
        for i, r in enumerate(grid):
            z1 = RNG.standard_normal((R, 10))
            z2 = RNG.standard_normal((R, 10))
            x = z1
            y = r * z1 + sqrt(max(0.0, 1 - r * r)) * z2
            v = pearson_rows(x, y) if stat == "pearson" else spearman_rows(x, y)
            p_ge[i] = (v >= obs).mean()
            p_le[i] = (v <= obs).mean()

        def cross(curve, threshold, increasing):
            k = curve if increasing else -curve
            p = threshold if increasing else -threshold
            idx = np.argmax(k >= p)
            if not (k >= p).any():
                return float("nan")
            if idx == 0:
                return float(grid[0])
            x0, x1 = grid[idx - 1], grid[idx]
            y0, y1 = k[idx - 1], k[idx]
            return float(x0 + (p - y0) * (x1 - x0) / (y1 - y0))

        low = cross(p_ge, 0.025, increasing=True)    # P(r>=obs) rises with rho
        high = cross(p_le, 0.025, increasing=False)  # P(r<=obs) falls with rho
        return low, high

    d_p, g_p = ci_inversion(0.60, "pearson")
    d_s, g_s = ci_inversion(0.60, "spearman")
    z = np.arctanh(0.60)
    fis = np.tanh(np.array([z - 1.959964 / sqrt(7), z + 1.959964 / sqrt(7)]))
    print("  Pearson  r = 0.60 -> CI [%.3f ; %.3f], width %.3f"
          % (d_p, g_p, g_p - d_p))
    print("           control via the Fisher formula: [%.3f ; %.3f], width %.3f"
          % (fis[0], fis[1], fis[1] - fis[0]))
    print("  Spearman rs = 0.60 -> CI on the Pearson rho [%.3f ; %.3f], "
          "width %.3f" % (d_s, g_s, g_s - d_s))
    print("           in units of rho_S: [%.3f ; %.3f]"
          % (population_spearman_rho(d_s), population_spearman_rho(g_s)))
    print("  Reading: an observed 0.60 at n = 10 rules out NEITHER the absence "
          "of a useful relationship NOR an almost perfect one.")

    return {"cp": cp, "cs": cs, "a_s": a_s, "rho_p": rp, "rho_s": rs_,
            "ci_p": (d_p, g_p), "ci_s": (d_s, g_s)}


# Part 2: the ceiling

def n_for_power(rho: float, target: float = POWER_TARGET, R: int = 15_000,
                n_max: int = 400) -> int:
    """Smallest n giving power >= target for the Spearman test.

    A coarse grid first, so the null distribution is not simulated for every
    n separately, then a scan inside the bracketed interval.
    """
    coarse = [5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80,
              100, 120, 150, 200, 250, 300, 400]
    prev = None
    for n in coarse:
        if n > n_max:
            break
        critical_values(n, R=60_000)
        if power(rho, n, R=R)[1] >= target:
            if prev is None:
                return n
            for m in range(prev + 1, n + 1):
                critical_values(m, R=60_000)
                if power(rho, m, R=R)[1] >= target:
                    return m
            return n
        prev = n
    return -1


def part2() -> dict:
    print()
    print("=" * 78)
    print("PART 2.  EFFECT OF THE mRNA -> PROTEIN CEILING AND MEASUREMENT NOISE")
    print("=" * 78)
    print("\nModel: observable correlation = rho_model * ceiling.")
    print("  rho_model = %.2f (brief: ~0.5 rank correlation on held-out data, "
          "one seed)" % RHO_MODEL)
    print("  ceiling = the maximum attainable correlation of an mRNA-rank "
          "prediction")
    print("          with a GFP measurement (second track; a scenario "
          "parameter here)")
    print("\n%-8s %-10s %-12s %-10s %-10s %-12s"
          % ("ceiling", "rho_obs", "power n=10", "power n=20", "power n=30",
             "n for 0.80"))
    print("-" * 78)
    out = {}
    for name, s in CEILINGS.items():
        ro = RHO_MODEL * s
        _, m10 = power(ro, 10, R=60_000)
        critical_values(20, R=200_000)
        critical_values(30, R=200_000)
        _, m20 = power(ro, 20, R=40_000)
        _, m30 = power(ro, 30, R=40_000)
        nn = n_for_power(ro)
        out[name] = {"rho_obs": ro, "power10": m10, "n80": nn}
        print("%-8s %-10.3f %-12.3f %-10.3f %-10.3f %-12s"
              % (name, ro, m10, m20, m30, nn if nn > 0 else ">400"))

    print("\nUpper bound: a PERFECT model (rho_model = 1.00), i.e. the ceiling "
          "alone:")
    print("%-8s %-10s %-12s %-12s"
          % ("ceiling", "rho_obs", "power n=10", "n for 0.80"))
    print("-" * 78)
    for name, s in CEILINGS.items():
        _, m10 = power(s, 10, R=60_000)
        nn = n_for_power(s)
        print("%-8s %-10.3f %-12.3f %-12s"
              % (name, s, m10, nn if nn > 0 else ">400"))

    print("\nPower is computed against the ATTAINABLE threshold of the Spearman "
          "test (actual alpha %.4f)," % critical_values(10)[3])
    print("so it cannot be inflated by picking a convenient critical value.")
    return out


# Part 3: position effect and transformants

def power_position(rho0: float, lam: float, k: int, n: int = 10,
                   R: int = 30_000) -> float:
    """Power of the rank test under positional noise.

    `rho0` is the correlation of the prediction with true promoter strength
    after the ceiling, `lam` the ratio of positional to promoter variance,
    and `k` the number of independent transformants averaged per construct.
    """
    cp, _, cs, _ = critical_values(n)
    g = RNG.standard_normal((R, n))                       # true promoter strength
    p = rho0 * g + sqrt(max(0.0, 1 - rho0 ** 2)) * RNG.standard_normal((R, n))
    y = g + sqrt(lam / k) * RNG.standard_normal((R, n))   # mean of k transformants
    return float((np.abs(spearman_rows(p, y)) >= cs).mean())


def part3() -> dict:
    print()
    print("=" * 78)
    print("PART 3.  POSITION EFFECT: HOW MANY TRANSFORMANTS PER CONSTRUCT")
    print("=" * 78)
    print("\nlam = variance between transformants of the same construct /")
    print("      variance between constructs (the promoter variance).")
    print("The mean of k transformants has noise lam/k -- simulated directly, "
          "not taken from a formula.")

    out = {}
    for rho0_name, rho0 in (("ceiling 0.90 -> 0.450", 0.450),
                            ("ceiling 0.75 -> 0.375", 0.375),
                            ("ceiling 0.60 -> 0.300", 0.300)):
        print("\n--- rho_0 = %.3f  (%s) ---" % (rho0, rho0_name))
        print("%-6s %-9s %-9s %-9s %-9s %-9s %-9s %-9s"
              % ("lam", "k=1", "k=2", "k=3", "k=5", "k=8", "k=12", "k=inf"))
        print("-" * 78)
        for lam in (0.25, 1.0, 4.0):
            row = []
            for k in (1, 2, 3, 5, 8, 12):
                row.append(power_position(rho0, lam, k))
            ceiling_k = power(rho0, 10, R=60_000)[1]
            out[(rho0, lam)] = {"k": row, "inf": ceiling_k}
            print("%-6.2f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f"
                  % (lam, *row, ceiling_k))

        print("  k needed for power 0.80 (at this rho_0):")
        for lam in (0.25, 1.0, 4.0):
            found = -1
            for k in range(1, 61):
                if power_position(rho0, lam, k, R=20_000) >= POWER_TARGET:
                    found = k
                    break
            ceiling_k = power(rho0, 10, R=60_000)[1]
            if found < 0:
                print("    lam = %.2f : UNREACHABLE for any k "
                      "(power at k -> inf is %.3f)" % (lam, ceiling_k))
            else:
                print("    lam = %.2f : k = %d  (%d transformants for 10 "
                      "constructs)" % (lam, found, 10 * found))

    print("\nDiminishing returns -- power gained by adding ONE transformant")
    print("to each of the 10 constructs (i.e. +10 transformants of work),")
    print("at rho_0 = 0.450:")
    print("%-6s %-8s %-8s %-8s %-8s %-8s"
          % ("lam", "1->2", "2->3", "3->4", "4->5", "5->6"))
    print("-" * 78)
    for lam in (0.25, 1.0, 4.0):
        m = [power_position(0.450, lam, k, R=40_000) for k in range(1, 7)]
        print("%-6.2f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f"
              % (lam, m[1] - m[0], m[2] - m[1], m[3] - m[2],
                 m[4] - m[3], m[5] - m[4]))
    return out


# Part 4: order against strength

def top_quantiles(N: int, m: int) -> np.ndarray:
    """Expected order statistics of the top m of N draws from N(0,1), by
    Blom's approximation.  The design is deterministic -- the lab picks one
    particular top-m -- so redrawing the pool per replicate would be wrong."""
    i = np.arange(1, m + 1, dtype=float)
    q = 1.0 - (i - 0.375) / (N + 0.25)
    return st.norm.ppf(q)


def evaluate_design(p_design: np.ndarray, rho: float, lam: float, k: int,
                    threshold: float, R: int = 30_000,
                    alpha: float = ALPHA) -> dict:
    """Three quantities for a vector of predicted strengths `p_design`.

    P(rank test significant), i.e. does the model rank; P(the best construct
    beats the threshold significantly), i.e. can we build a strong one; and
    the distribution of the best construct's true strength.  `p` and the true
    `y` are standardised on the scale of natural promoters, which assumes
    synthetic designs reach the same range.
    """
    n = p_design.size
    _, _, cs, _ = critical_values(n)
    P = np.broadcast_to(p_design[None, :], (R, n))
    ytrue = rho * P + sqrt(max(0.0, 1 - rho ** 2)) * RNG.standard_normal((R, n))

    # k transformants per construct, plus k for the threshold control.
    obs = ytrue[:, :, None] + sqrt(lam) * RNG.standard_normal((R, n, k))
    ctrl = threshold + sqrt(lam) * RNG.standard_normal((R, 1, k))

    mean_obs = obs.mean(axis=2)
    ranking = float((np.abs(spearman_rows(
        np.broadcast_to(p_design[None, :], (R, n)).copy(),
        mean_obs)) >= cs).mean())

    result = {"ranking": ranking,
              "sd_p": float(p_design.std(ddof=1)),
              "y_max_median": float(np.median(ytrue.max(axis=1)))}

    if k >= 2:
        allobs = np.concatenate([obs, ctrl], axis=1)           # (R, n+1, k)
        m = allobs.mean(axis=2)
        ss = ((allobs - m[:, :, None]) ** 2).sum(axis=(1, 2))
        df = (n + 1) * (k - 1)
        s2 = ss / df
        se = np.sqrt(2.0 * s2 / k)[:, None]
        t = (m[:, :n] - m[:, n:n + 1]) / se
        tmax = t.max(axis=1)
        result["strength_bonf"] = float(
            (tmax >= st.t.ppf(1 - alpha / n, df)).mean())
        result["strength_raw"] = float((tmax >= st.t.ppf(1 - alpha, df)).mean())
    else:
        result["strength_bonf"] = float("nan")
        result["strength_raw"] = float("nan")
    return result


def part4() -> dict:
    print()
    print("=" * 78)
    print("PART 4.  ORDER vs STRENGTH -- how to spend a budget of 10")
    print("=" * 78)
    z_top = top_quantiles(N_POOL, 10)
    print("\nCandidate pool: N = %d.  Top of predicted strength: "
          "z from %.2f to %.2f (spread %.2f)."
          % (N_POOL, z_top[-1], z_top[0], z_top[0] - z_top[-1]))
    print("Comparison thresholds (z scale, from rank via Phi^-1 -- an "
          "ASSUMPTION about the shape of the distribution):")
    print("  native pks1 y = %.4f -> z = %.3f" % (Y_PKS1, Z_PKS1))
    print("  tef1          y = %.4f -> z = %.3f" % (Y_TEF1, Z_TEF1))

    designs = {
        "(a) 10 x top": z_top,
        "(b1) 10 evenly, full range": np.linspace(-z_top[0], z_top[0], 10),
        "(b2) 10 evenly, only >= mean": np.linspace(0.0, z_top[0], 10),
        "(c) 6 top + 4 spread out": np.concatenate(
            [top_quantiles(N_POOL, 6), np.linspace(-z_top[0], 2.0, 4)]),
    }

    lam, k = 1.0, 3
    out = {}
    for ceiling_name, ceiling in CEILINGS.items():
        rho = RHO_MODEL * ceiling
        print("\n--- ceiling %s -> rho = %.3f ; lam = %.2f ; k = %d "
              "transformants/construct ---" % (ceiling_name, rho, lam, k))
        print("%-38s %-7s %-10s %-12s %-12s %-10s"
              % ("design", "sd(p)", "RANKING", "STR>pks1", "STR>tef1",
                 "y_max med"))
        print("-" * 78)
        for name, p in designs.items():
            w_pks = evaluate_design(p, rho, lam, k, Z_PKS1)
            w_tef = evaluate_design(p, rho, lam, k, Z_TEF1)
            out[(ceiling_name, name)] = (w_pks, w_tef)
            print("%-38s %-7.2f %-10.3f %-12.3f %-12.3f %-10.2f"
                  % (name, w_pks["sd_p"], w_pks["ranking"],
                     w_pks["strength_bonf"], w_tef["strength_bonf"],
                     w_pks["y_max_median"]))
        print("  (RANKING = P(|rs| >= %.4f at n = 10);  STRENGTH = "
              "P(best construct beats the control," % critical_values(10)[2])
        print("   one-sided t test, Bonferroni over 10 constructs, variance "
              "pooled across the panel))")
        print("  median true strength of the best construct, as a rank "
              "quantile: %.4f (a), %.4f (b1)"
              % (st.norm.cdf(out[(ceiling_name,
                                  "(a) 10 x top")][0]["y_max_median"]),
                 st.norm.cdf(out[(ceiling_name,
                                  "(b1) 10 evenly, full range")][0]["y_max_median"])))

    print("\nNEGATIVE CONTROL: rho = 0 (a model with no value at all), "
          "the same designs:")
    print("%-38s %-10s %-12s" % ("design", "RANKING", "STR>pks1"))
    print("-" * 78)
    for name, p in designs.items():
        w = evaluate_design(p, 0.0, lam, k, Z_PKS1)
        print("%-38s %-10.3f %-12.3f" % (name, w["ranking"],
                                         w["strength_bonf"]))
    print("  RANKING must come out at ~%.3f (the test's alpha), otherwise the "
          "simulation is broken." % critical_values(10)[3])
    return out


# Part 5: controls

def part5() -> dict:
    print()
    print("=" * 78)
    print("PART 5.  CONTROLS -- how many slots they eat and what that costs")
    print("=" * 78)

    print("\n5a. Power of the rank test against the number of synthetic points:")
    print("%-6s %-10s %-10s %-10s %-10s %-14s"
          % ("n", "thr |rs|", "alpha", "rho=0.450", "rho=0.375", "rho=0.300"))
    print("-" * 78)
    results_n = {}
    for n in (10, 9, 8, 7, 6, 5):
        cp, a_p, cs, a_s = critical_values(n, R=400_000)
        m = [power(r, n, R=60_000)[1] for r in (0.450, 0.375, 0.300)]
        results_n[n] = (cs, a_s, m)
        print("%-6d %-10.4f %-10.4f %-10.3f %-10.3f %-14.3f"
              % (n, cs, a_s, m[0], m[1], m[2]))

    print("\n5b. Correlation detectable with power 0.80:")
    for n in (10, 7):
        rp, rs_, *_ = rho_for_power(n, R=40_000)
        print("  n = %2d : the Spearman test needs rho >= %.3f "
              "(Pearson %.3f)" % (n, rs_, rp))

    print("\n5c. Mixed panel: 7 synthetic + 3 controls as correlation points.")
    print("Controls: no promoter (z = -3.50, truth known), native pks1 "
          "(z = %.2f), tef1 (z = %.2f)." % (Z_PKS1, Z_TEF1))
    print("The model is assumed to predict the controls with the same "
          "correlation rho as the synthetics (an OPTIMISTIC ASSUMPTION:")
    print("natural promoters are inside the training distribution, synthetic "
          "ones are not).")

    z_top = top_quantiles(N_POOL, 7)
    variants = {
        "7 synthetic = top": z_top,
        "7 synthetic spread out": np.linspace(-z_top[0], z_top[0], 7),
    }
    controls_p = np.array([-3.50, Z_PKS1, Z_TEF1])
    controls_y = np.array([-3.50, Z_PKS1, Z_TEF1])

    def mixed_panel(p_syn, rho, R=40_000, lam=1.0, k=3,
                    controls=True, no_promoter=True):
        n_syn = p_syn.size
        kp = controls_p if no_promoter else controls_p[1:]
        ky = controls_y if no_promoter else controls_y[1:]
        n = n_syn + (kp.size if controls else 0)
        _, _, cs, _ = critical_values(n, R=400_000)
        P = np.broadcast_to(p_syn[None, :], (R, n_syn))
        ytrue = (rho * P + sqrt(max(0.0, 1 - rho ** 2))
                 * RNG.standard_normal((R, n_syn)))
        y = ytrue + sqrt(lam / k) * RNG.standard_normal((R, n_syn))
        if controls:
            yk = ky[None, :] + sqrt(lam / k) * RNG.standard_normal((R, kp.size))
            Pfull = np.concatenate(
                [P, np.broadcast_to(kp[None, :], (R, kp.size))], axis=1).copy()
            yfull = np.concatenate([y, yk], axis=1)
        else:
            Pfull, yfull = P.copy(), y
        return float((np.abs(spearman_rows(Pfull, yfull)) >= cs).mean())

    print("\n%-32s %-10s %-12s %-12s %-12s"
          % ("variant", "n", "rho=0.450", "rho=0.300", "rho=0 (FP)"))
    print("-" * 78)
    fp = {}
    for name, p in variants.items():
        for controls, label, nn in ((False, "7 synthetic alone", 7),
                                    (True, "+3 controls", 10)):
            m45 = mixed_panel(p, 0.450, controls=controls)
            m30 = mixed_panel(p, 0.300, controls=controls)
            m00 = mixed_panel(p, 0.000, controls=controls)
            fp[(name, label)] = m00
            print("%-32s %-10s %-12.3f %-12.3f %-12.3f"
                  % (name + " / " + label, nn, m45, m30, m00))

    print("\n  The 'rho = 0 (FP)' column is the fraction of FALSELY "
          "significant results when the model has")
    print("  no predictive value for the synthetics, while the controls still "
          "sit where they are supposed to.")
    print("  That number MUST be compared against alpha = %.4f."
          % critical_values(10)[3])

    print("\n5d. The same without the 'no promoter' construct (2 controls, "
          "n = 9):")
    print("%-32s %-10s %-12s %-12s"
          % ("variant", "n", "rho=0.450", "rho=0 (FP)"))
    print("-" * 78)
    for name, p in variants.items():
        m45 = mixed_panel(p, 0.450, controls=True, no_promoter=False)
        m00 = mixed_panel(p, 0.000, controls=True, no_promoter=False)
        print("%-32s %-10s %-12.3f %-12.3f" % (name, 9, m45, m00))
    return {"n": results_n, "fp": fp}



def part3b() -> None:
    """Power 0.80 at n = 10 is unreachable for any k, so the question becomes
    how much of the attainable power (the k -> infinity limit) a given k
    recovers, and whether the same transformant budget buys more when spent
    on constructs instead."""
    print()
    print("=" * 78)
    print("PART 3b.  HOW MUCH ATTAINABLE POWER k RECOVERS (0.80 is impossible)")
    print("=" * 78)
    print("\nfraction = power(k) / power(k -> inf) at n = 10; rho_0 = 0.450")
    print("%-6s %-8s %-8s %-8s %-8s %-8s %-8s  %-10s"
          % ("lam", "k=1", "k=2", "k=3", "k=4", "k=6", "k=10", "power(inf)"))
    print("-" * 78)
    for lam in (0.25, 1.0, 4.0):
        inf = power(0.450, 10, R=120_000)[1]
        fr = [power_position(0.450, lam, k, R=60_000) / inf
              for k in (1, 2, 3, 4, 6, 10)]
        print("%-6.2f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f  %-10.3f"
              % (lam, *fr, inf))
    print("\n  k needed to recover 90%% and 95%% of the attainable power "
          "(rho_0 = 0.450):")
    for lam in (0.25, 1.0, 4.0):
        inf = power(0.450, 10, R=120_000)[1]
        k90 = k95 = -1
        for k in range(1, 41):
            f = power_position(0.450, lam, k, R=60_000) / inf
            if k90 < 0 and f >= 0.90:
                k90 = k
            if k95 < 0 and f >= 0.95:
                k95 = k
                break
        print("    lam = %.2f : 90%% at k = %s ; 95%% at k = %s "
              "(i.e. %s and %s transformants for 10 constructs)"
              % (lam, k90 if k90 > 0 else ">40", k95 if k95 > 0 else ">40",
                 10 * k90 if k90 > 0 else "-", 10 * k95 if k95 > 0 else "-"))

    print("\nTRADE-OFF: the same TRANSFORMANT budget (T = n * k) spent "
          "differently.")
    print("rho_0 = 0.450, lam = 1.00.  T = number of measurements to perform.")
    print("%-6s %-6s %-6s %-10s" % ("n", "k", "T", "power"))
    print("-" * 78)
    for n, k in ((10, 1), (10, 2), (10, 3), (10, 6), (10, 10),
                 (15, 2), (20, 1), (20, 3), (30, 1), (30, 2), (41, 1), (60, 1)):
        critical_values(n, R=200_000)
        print("%-6d %-6d %-6d %-10.3f"
              % (n, k, n * k, power_position(0.450, 1.0, k, n=n)))


def part4b() -> None:
    """The strength question has no ceiling in n and rises with k without
    bound.  That asymmetry decides how to spend the budget."""
    print()
    print("=" * 78)
    print("PART 4b.  HOW MANY TRANSFORMANTS FOR THE STRENGTH QUESTION "
          "(design (a))")
    print("=" * 78)
    z_top = top_quantiles(N_POOL, 10)
    print("\nP(the best of 10 top constructs beats the control significantly,")
    print("one-sided t test, Bonferroni over 10):")
    for threshold, name in ((Z_PKS1, "native pks1 (z = 1.587)"),
                            (Z_TEF1, "tef1 (z = 2.929)")):
        print("\n  threshold = %s" % name)
        print("  %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s"
              % ("lam\\k", "2", "3", "5", "8", "12", "20", "30"))
        print("  " + "-" * 74)
        for lam in (0.25, 1.0, 4.0):
            w = [evaluate_design(z_top, 0.450, lam, k, threshold,
                                 R=20_000)["strength_bonf"]
                 for k in (2, 3, 5, 8, 12, 20, 30)]
            print("  %-6.2f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f"
                  % (lam, *w))
        print("  k for power 0.80 (rho = 0.450):")
        for lam in (0.25, 1.0, 4.0):
            found = -1
            for k in range(2, 81):
                if evaluate_design(z_top, 0.450, lam, k, threshold,
                                   R=12_000)["strength_bonf"] >= POWER_TARGET:
                    found = k
                    break
            print("    lam = %.2f : k = %s  (%s transformants for 10 "
                  "constructs)"
                  % (lam, found if found > 0 else ">80",
                     10 * found if found > 0 else "-"))


def part4c() -> None:
    """Is the tef1 threshold a power problem or an effect-size problem?

    Computed on true strengths with no measurement at all, so the answer is
    a ceiling no amount of transformant effort can cross.
    """
    print()
    print("=" * 78)
    print("PART 4c.  DESIGN CEILING: does the best of 10 beat the threshold "
          "AT ALL")
    print("=" * 78)
    print("\nNo measurement, no positional noise: P(max true strength of the "
          "10 top designs > threshold).")
    print("Regression to the mean: a design with predicted z has a true "
          "y ~ N(rho*z, 1-rho^2).")
    R = 200_000
    print("\n%-10s %-8s %-12s %-14s %-14s %-14s"
          % ("pool N", "rho", "z_top", "E[y_max]", "P(y_max>pks1)",
             "P(y_max>tef1)"))
    print("-" * 78)
    for N in (5_000, 100_000, 1_000_000):
        zt = top_quantiles(N, 10)
        for rho in (0.300, 0.375, 0.450, 0.600, 0.800, 1.000):
            P = np.broadcast_to(zt[None, :], (R, 10))
            y = (rho * P + sqrt(max(0.0, 1 - rho ** 2))
                 * RNG.standard_normal((R, 10)))
            ym = y.max(axis=1)
            print("%-10d %-8.3f %-12.2f %-14.2f %-14.3f %-14.3f"
                  % (N, rho, zt[0], ym.mean(),
                     (ym > Z_PKS1).mean(), (ym > Z_TEF1).mean()))
        print("-" * 78)


def part5b() -> None:
    """Why the '7 top + 3 controls' panel has power below alpha."""
    print()
    print("=" * 78)
    print("PART 5b.  DIAGNOSIS: the distribution of rs in the mixed panel")
    print("=" * 78)
    R, lam, k = 60_000, 1.0, 3
    z7 = top_quantiles(N_POOL, 7)
    _, _, cs10, a10 = critical_values(10, R=400_000)
    for kp, label, nk in ((np.array([-3.50, Z_PKS1, Z_TEF1]), "3 controls", 3),
                          (np.array([Z_PKS1, Z_TEF1]),
                           "2 controls (no empty)", 2)):
        _, _, cs, a = critical_values(7 + nk, R=400_000)
        print("\n  --- %s, n = %d, threshold |rs| >= %.4f, alpha %.4f ---"
              % (label, 7 + nk, cs, a))
        for name, psyn in (("7 top", z7),
                           ("7 spread out", np.linspace(-z7[0], z7[0], 7))):
            for rho in (0.0, 0.450):
                P = np.broadcast_to(psyn[None, :], (R, 7))
                y = (rho * P + sqrt(max(0.0, 1 - rho ** 2))
                     * RNG.standard_normal((R, 7))
                     + sqrt(lam / k) * RNG.standard_normal((R, 7)))
                yk = kp[None, :] + sqrt(lam / k) * RNG.standard_normal((R, nk))
                Pf = np.concatenate([P, np.broadcast_to(kp[None, :], (R, nk))],
                                    axis=1).copy()
                yf = np.concatenate([y, yk], axis=1)
                rs = spearman_rows(Pf, yf)
                print("  %-16s rho=%.3f : rs mean %+.3f  sd %.3f  "
                      "P(rs>=+thr) %.4f  P(rs<=-thr) %.4f"
                      % (name, rho, rs.mean(), rs.std(),
                         (rs >= cs).mean(), (rs <= -cs).mean()))
    print("\n  Two-sided threshold at n = 10: |rs| >= %.4f, alpha %.4f."
          % (cs10, a10))
    print("  If the mean rs sits between -thr and +thr with a small sd, the "
          "test CANNOT fire")
    print("  either under a real model or under a null one -- a falsifier "
          "unsatisfiable by construction.")


def main(argv=None) -> int:
    t0 = time.time()
    argv = sys.argv[1:] if argv is None else argv
    only = argv[0] if argv else ""
    if only == "--extra2":
        part4c()
        part5b()
        print("\nTime: %.1f s" % (time.time() - t0))
        return 0
    print("panel_power.py -- power simulation for a panel of n = 10")
    import scipy
    print("seed %d, numpy %s, scipy %s"
          % (20260911, np.__version__, scipy.__version__))
    if only == "--extra":
        part3b()
        part4b()
        print("\nTime: %.1f s" % (time.time() - t0))
        return 0
    w1 = part1()
    w2 = part2()
    part3()
    part3b()
    part4()
    part4b()
    part5()
    print()
    print("=" * 78)
    print("Time: %.1f s" % (time.time() - t0))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
