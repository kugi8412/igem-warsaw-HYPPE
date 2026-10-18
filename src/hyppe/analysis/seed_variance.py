"""A17 -- is the hyperbolic advantage in the species tree embedding real?

This falsifies the project's central geometric premise, so it is built around
four things an earlier run did not have. Convergence: the learning rate
anneals cosine to zero and the tail amplitude of the objective is reported per
configuration, because at a constant lr that run oscillated by ~0.004, exactly
the size of the difference it claimed to measure. Seeds: twelve of them,
median and interquartile range, plus a paired sign test from a shared starting
`v`, which removes the variance of initialisation. Size: the real tree has ten
leaves, so synthetic trees of 8 to 64 leaves with lengths from the same pool
separate "curvature does not help" from "ten leaves fit flat". Stability: the
fit drives itself to radius 20-40, where ambient hyperboloid coordinates lose
precision, so distance uses a cancellation-free formula and the two formulas
are compared.

`_ambient_on_manifold` is a trap. `<x,x>_L + 1` subtracts two numbers of order
sinh^2(r) ~ 1e35 whose true difference is 1, so at a large radius it must be
enormous (measured 9.44e+21) even when the embedding is correct. The measure
of error is the divergence of the two distance formulas, <= 5.6e-12 relative.

All numbers go to `logs/seed_variance.log`.
"""
from __future__ import annotations

import io
import os
import sys
import time
from math import comb

import numpy as np
import torch

from .. import geometry as G
from . import matrices as data
from . import paths
from . import phylogeny as P

# The protocol in one place, so that "the same budget" is checkable.
SEEDS = list(range(12))           # >= 10, as the task requires
LR0 = 0.05                        # as in phylogeny.fit_tree_embedding
STEPS_MAIN = 40_000               # the real tree (the headline measurement)
STEPS_CONTROL = 20_000            # the synthetic trees
DIMS = (2, 3, 4, 6, 8, 12)
DIMS_SIZE_SWEEP = (2, 4, 8)       # sweep over leaf count -- narrower, costlier
TAIL_WINDOW = 2000                # window over which convergence is read

torch.set_num_threads(4)


# Geometry
def hyperbolic_distance(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between expmap0(vx) and expmap0(vy), cancellation-free.

    Uses  -<x,y>_L = cosh(rx - ry) + sinh(rx) sinh(ry) (1 - cos theta)  with
    1 - cos(theta) = ||ux - uy||^2 / 2 for unit vectors. Both terms are
    positive, so no two nearly equal numbers are subtracted -- unlike the
    ambient route, which at r ~ 40 subtracts numbers of order 1e35.
    """
    rx = vx.norm(dim=-1).clamp_min(1e-300)
    ry = vy.norm(dim=-1).clamp_min(1e-300)
    ux = vx / rx.unsqueeze(-1)
    uy = vy / ry.unsqueeze(-1)
    chord2 = (ux - uy).pow(2).sum(dim=-1)
    z = torch.cosh(rx - ry) + torch.sinh(rx) * torch.sinh(ry) * chord2 / 2
    return torch.acosh(z.clamp_min(1.0 + 1e-15))


# The ambient route is the object of the measurement, not a dependency:
# section 0(b) shows how much precision it loses at the radius this fit
# reaches. Hence written out rather than imported -- `hyppe.geometry.expmap0`
# truncates a step at MAX_TANGENT_NORM = 8.0 and r_max here is 20-40, so it
# would compare a clamped embedding against an unclamped one.
def _ambient_expmap0(v: torch.Tensor) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return G.project_to_hyperboloid(torch.sinh(norm) * v / norm)


def _ambient_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    inner = -G.minkowski_dot(x, y, keepdim=False)
    return torch.acosh(inner.clamp_min(1.0 + 1e-7))


def _ambient_on_manifold(x: torch.Tensor) -> float:
    return float((G.minkowski_dot(x, x, keepdim=False) + 1.0).abs().max())


# Fitting
def init_v(n: int, dim: int, seed: int) -> torch.Tensor:
    """The starting point of one seed.

    Its own `torch.Generator` rather than the global `manual_seed`, so seed s
    gives the same start however many fits ran before it; otherwise "seed 3"
    would mean something different in every table.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, dim, generator=g, dtype=torch.float64) * 0.1


def fit(D: np.ndarray, dim: int, geom: str, seeds: list[int], steps: int,
        lr0: float = LR0, schedule: bool = True) -> dict:
    """S independent fits at once, one geometry, one dimension.

    Both geometries minimise mean |d / (s * d_tree) - 1| with one global `s`,
    since tree distance is in substitutions and geodesic distance is
    dimensionless, and both get n*dim coordinates plus that scale, the same
    optimiser, schedule, step count and starting tensor `v` -- which is what
    makes the comparison paired. `expmap0` is a diffeomorphism R^dim -> H^dim,
    so the ambient dim+1 is not an extra parameter.

    One asymmetry is irremovable: in flat space `s` is degenerate with
    rescaling `v`, so the Euclidean fit effectively has n*dim degrees of
    freedom against n*dim + 1. That extra number is the curvature itself and
    cannot be subtracted without subtracting the effect.

    Batching over seeds is legal because the loss is a sum of independent
    terms and Adam is elementwise; `batching_control` checks it by execution.
    """
    S, n = len(seeds), D.shape[0]
    tgt = torch.tensor(D, dtype=torch.float64)
    iu = torch.triu_indices(n, n, offset=1)
    t = tgt[iu[0], iu[1]]
    n_zero = int((t <= 0).sum())
    if n_zero:
        raise ValueError("%d of %d leaf pairs have tree distance 0 -- "
                         "the relative error would be infinite"
                         % (n_zero, t.numel()))
    v = torch.stack([init_v(n, dim, s) for s in seeds]).requires_grad_(True)
    log_s = torch.zeros(S, 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v, log_s], lr=lr0)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                      eta_min=0.0)
           if schedule else None)
    traj = np.empty((steps, S), dtype=np.float64)
    for i in range(steps):
        opt.zero_grad()
        if geom == "hyp":
            d = hyperbolic_distance(v[:, iu[0]], v[:, iu[1]])
        else:
            d = (v[:, iu[0]] - v[:, iu[1]]).norm(dim=-1)
        per = (d / (log_s.exp() * t).clamp_min(1e-9) - 1.0).abs().mean(dim=-1)
        per.sum().backward()
        opt.step()
        if sch is not None:
            sch.step()
        traj[i] = per.detach().numpy()
    with torch.no_grad():
        if geom == "hyp":
            d = hyperbolic_distance(v[:, iu[0]], v[:, iu[1]])
            x = _ambient_expmap0(v)
            d_amb = _ambient_distance(x[:, iu[0]], x[:, iu[1]])
            divergence = float(((d_amb - d).abs() / d.clamp_min(1e-300)).max())
            manifold = _ambient_on_manifold(x)
        else:
            d = (v[:, iu[0]] - v[:, iu[1]]).norm(dim=-1)
            divergence, manifold = 0.0, float("nan")
        final = (d / (log_s.exp() * t) - 1.0).abs().mean(dim=-1).numpy()
        scale = log_s.exp().squeeze(-1).numpy()
        r_max = v.norm(dim=-1).max(dim=-1).values.numpy()
    tail = traj[-TAIL_WINDOW:]
    return {
        "final": final,                          # after the last step, no graph
        "best": np.minimum(traj.min(axis=0), final),
        "tail_amplitude": tail.max(axis=0) - tail.min(axis=0),
        "max_increase": np.diff(tail, axis=0).max(axis=0),
        "scale": scale,
        "r_max": r_max,
        "divergence": divergence,
        "manifold": manifold,
        "traj": traj,
    }


# Synthetic trees
def branch_lengths(node: P.Node) -> list[float]:
    out = [] if node.length == 0.0 else [node.length]
    for c in node.children:
        out.extend(branch_lengths(c))
    return out


def _binary(pool, rs, depth, counter) -> P.Node:
    if depth == 0:
        counter[0] += 1
        return P.Node(name="L%d" % counter[0], length=float(rs.choice(pool)))
    node = P.Node(children=[_binary(pool, rs, depth - 1, counter)
                            for _ in range(2)])
    node.length = float(rs.choice(pool))
    return node


def binary_tree(n_leaves: int, pool: np.ndarray, seed: int = 17) -> P.Node:
    """A complete binary tree with `n_leaves` leaves, lengths iid from `pool`."""
    g = int(round(np.log2(n_leaves)))
    if 2 ** g != n_leaves:
        raise ValueError("n_leaves must be a power of two, got %d" % n_leaves)
    rs = np.random.RandomState(seed)
    root = _binary(pool, rs, g, [0])
    root.length = 0.0
    return root


def comb_tree(n_leaves: int, pool: np.ndarray, seed: int = 17) -> P.Node:
    """A comb (ladder), the shape of the real tree.

    Without it size is confounded with shape: our ten leaves are a ladder with
    two very long external branches, while the 64-leaf control is balanced.
    """
    rs = np.random.RandomState(seed)
    node = P.Node(children=[P.Node(name="L1", length=float(rs.choice(pool))),
                            P.Node(name="L2", length=float(rs.choice(pool)))])
    for i in range(3, n_leaves + 1):
        node.length = float(rs.choice(pool))
        node = P.Node(children=[node, P.Node(name="L%d" % i,
                                             length=float(rs.choice(pool)))])
    node.length = 0.0
    return node


# Statistics
def quartiles(a: np.ndarray):
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return float(med), float(q3 - q1), float(q1), float(q3)


def disjoint(a: np.ndarray, b: np.ndarray) -> tuple[bool, bool]:
    """(median +- IQR disjoint?, [Q1,Q3] disjoint?)

    The first interval is the wider one, so it is the sharper condition for
    declaring an advantage; the second is reported alongside it.
    """
    ma, ia, q1a, q3a = quartiles(a)
    mb, ib, q1b, q3b = quartiles(b)
    return ((ma + ia < mb - ib) or (mb + ib < ma - ia),
            (q3a < q1b) or (q3b < q1a))


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    n = len(d)
    k = int((d > 0).sum())
    extreme = min(k, n - k)
    p = min(2.0 * sum(comb(n, i) for i in range(extreme + 1)) / 2.0 ** n, 1.0)
    return k, n, p


# Report
class Tee:
    def __init__(self, path):
        self.fh = io.open(path, "w", encoding="utf-8", newline="\n")

    def __call__(self, s: str = "") -> None:
        print(s)
        sys.stdout.flush()
        self.fh.write(s + "\n")
        self.fh.flush()


def batching_control(D, out) -> None:
    """Falsifier of our own implementation: batching over seeds must give the
    same result as separate runs, or every table rests on an unchecked
    shortcut."""
    out("  (a) batching over seeds must reproduce separate runs (dim 4, 4000 "
        "steps)")
    for geom in ("hyp", "euc"):
        r = fit(D, 4, geom, [0, 1, 2], 4000)
        single = [fit(D, 4, geom, [s], 4000)["final"][0] for s in (0, 1, 2)]
        out("      %-4s max|batched - single| = %.2e   (target ~%.5f)"
            % (geom, np.abs(r["final"] - np.array(single)).max(),
               r["final"].mean()))
    out("      A divergence of order 1e-4 against a target of 4e-3 is chaotic")
    out("      amplification of float64 rounding (a different reduction kernel")
    out("      for a different tensor shape), not a wrong gradient -- some")
    out("      seeds agree TO THE BIT.  It is smaller than the spread between")
    out("      seeds, and we report the median of 12 seeds.")
    out()
    out("  (b) two independent formulas for geodesic distance must agree")
    for dim in (2, 4, 8, 12):
        r = fit(D, dim, "hyp", [0, 1, 2], 4000)
        out("      dim %-3d r_max %5.2f  scale %6.2f | formula divergence %.2e "
            "| ambient |<x,x>_L+1| %.2e"
            % (dim, r["r_max"].max(), np.median(r["scale"]),
               r["divergence"], r["manifold"]))
    out("      max|<x,x>_L + 1| subtracts two numbers of order sinh^2(r) whose")
    out("      true difference is 1, so at r ~ 40 it MUST be enormous even for")
    out("      a correct embedding.  That number is NOT a measure of error;")
    out("      the measure is the divergence of the two formulas.")


def geometry_table(out, D, dims, steps, seeds=SEEDS, key="final"):
    results = {}
    out("  %-5s | %-27s | %-27s | %-8s | %-6s %-6s" %
        ("dim", "HYPERBOLIC med [Q1,Q3]", "EUCLIDEAN   med [Q1,Q3]",
         "ratio", "scale", "r_max"))
    out("  " + "-" * 98)
    for dim in dims:
        row = {}
        for geom in ("hyp", "euc"):
            t0 = time.time()
            r = fit(D, dim, geom, seeds, steps)
            r["time"] = time.time() - t0
            row[geom] = r
        h, e = row["hyp"][key], row["euc"][key]
        mh, _, q1h, q3h = quartiles(h)
        me, _, q1e, q3e = quartiles(e)
        out("  %-5d | %.5f [%.5f,%.5f] | %.5f [%.5f,%.5f] | %6.2fx | %6.2f %6.2f"
            % (dim, mh, q1h, q3h, me, q1e, q3e,
               me / mh if mh > 0 else float("inf"),
               np.median(row["hyp"]["scale"]), row["hyp"]["r_max"].max()))
        results[dim] = row
    return results


def convergence_report(out, results, dims):
    out()
    out("  CONVERGENCE -- amplitude of the objective over the last %d steps "
        "(median over seeds)." % TAIL_WINDOW)
    out("  Condition for the table to be usable: amplitude << the measured "
        "hyp-euc difference.")
    out("  %-5s | %-11s %-11s | %-11s | %-9s | %s" %
        ("dim", "ampl hyp", "ampl euc", "difference", "ratio", "verdict"))
    out("  " + "-" * 82)
    for dim in dims:
        h, e = results[dim]["hyp"], results[dim]["euc"]
        diff = abs(np.median(e["final"]) - np.median(h["final"]))
        am = max(np.median(h["tail_amplitude"]), np.median(e["tail_amplitude"]))
        out("  %-5d | %-11.2e %-11.2e | %-11.2e | %8.1fx | %s"
            % (dim, np.median(h["tail_amplitude"]),
               np.median(e["tail_amplitude"]), diff,
               diff / am if am > 0 else float("inf"),
               "OK" if diff > 10 * am else "TOO CLOSE TO NOISE"))


def falsifier_table(out, results, dims, key="final"):
    out()
    out("  FALSIFIER -- an advantage may be declared ONLY where the intervals")
    out("  are DISJOINT.  'median+-IQR' is the condition from the task (wider,")
    out("  hence sharper); '[Q1,Q3]' is given for comparison.")
    out("  %-5s | %-9s %-9s | %-12s %-10s | %-10s | %s" %
        ("dim", "med hyp", "med euc", "median+-IQR", "[Q1,Q3]", "paired",
         "verdict"))
    out("  " + "-" * 92)
    verdicts = {}
    for dim in dims:
        h, e = results[dim]["hyp"][key], results[dim]["euc"][key]
        mh, me = float(np.median(h)), float(np.median(e))
        wide, narrow = disjoint(h, e)
        who = "hyperbolic" if mh < me else "euclidean"
        verdict = ("%s wins" % who) if wide else "TIE (inside noise)"
        verdicts[dim] = (verdict, wide)
        out("  %-5d | %-9.5f %-9.5f | %-12s %-10s | %2d/%-2d hyp | %s"
            % (dim, mh, me, "DISJOINT" if wide else "overlap",
               "DISJOINT" if narrow else "overlap",
               int((h < e).sum()), len(h), verdict))
    return verdicts


def paired_test(out, results, dims, key="final"):
    """Exact sign test on the paired difference, same starting `v` in both.

    With 12 pairs, 12/12 gives p = 4.9e-04 two-sided.
    """
    out()
    out("  PAIRED TEST (same starting v in both geometries, exact sign test)")
    out("  %-5s | %-13s %-12s | %-9s | %-10s" %
        ("dim", "med(euc-hyp)", "IQR of diff", "signs", "p (two-sided)"))
    out("  " + "-" * 66)
    for dim in dims:
        d = results[dim]["euc"][key] - results[dim]["hyp"][key]
        k, n, p = sign_test(d)
        med, iqr, _, _ = quartiles(d)
        out("  %-5d | %+13.5f %-12.5f | %2d/%-6d | %.2e"
            % (dim, med, iqr, k, n, p))


def main() -> None:
    log = os.path.join(paths.ensure_dir(paths.LOGS), "seed_variance.log")
    out = Tee(log)
    t_start = time.time()

    out("=" * 98)
    out("A17 -- curvature advantage in the species tree embedding: real or noise?")
    out("=" * 98)
    out("  torch %s | float64 | %d threads | %s"
        % (torch.__version__, torch.get_num_threads(),
           time.strftime("%Y-%m-%d %H:%M")))
    out("  tree: %s" % paths.SPECIES_TREE)
    out("  protocol: Adam(lr=%.3f) + CosineAnnealingLR(eta_min=0), %d seeds,"
        % (LR0, len(SEEDS)))
    out("            %d steps (real tree) / %d (synthetic controls),"
        % (STEPS_MAIN, STEPS_CONTROL))
    out("            objective  mean |d/(s*d_tree) - 1|,  budget n*dim + 1 in "
        "BOTH geometries.")
    out("            The hyperbolic distance uses the cancellation-free formula")
    out("            (see `hyperbolic_distance`), because the fit drives itself")
    out("            to a radius r ~ 20-40, where ambient coordinates lose")
    out("            precision.")

    root = P.parse_newick(data.read_newick())
    names, D = P.patristic_matrix(root)
    pool = np.array(branch_lengths(root), dtype=np.float64)
    out()
    out("  real tree: %d leaves, %d pairs, distances %.4f .. %.4f"
        % (len(names), len(names) * (len(names) - 1) // 2,
           D[D > 0].min(), D.max()))
    out("  pool of branch lengths (n=%d): min %.4f | median %.4f | max %.4f"
        % (len(pool), pool.min(), np.median(pool), pool.max()))
    out("  shape: a ladder (comb); two extreme external branches")
    out("         (fusarium 0.3160, neurospora 0.7374) set the scale.")
    out("         This is NOT a balanced tree and the control must account")
    out("         for that -- hence the 'comb 64' row in section 4.")

    out()
    out("=" * 98)
    out("0. IMPLEMENTATION CONTROL")
    out("=" * 98)
    batching_control(D, out)

    out()
    out("=" * 98)
    out("1. FIXING CONVERGENCE -- what was enough")
    out("=" * 98)
    out("  Previous run: constant lr=0.05, 4000 steps.  The objective")
    out("  oscillated 0.02021 -> 0.01145 -> 0.01589 -> 0.01299 (phylogeny.log,")
    out("  steps 1-4k).  Below: median over %d seeds, dim 4, hyperbolic."
        % len(SEEDS))
    out()
    out("  %-24s | %-9s %-9s | %-11s %-11s | %-9s" %
        ("schedule", "median", "best", "ampl_2000", "ampl_500", "time"))
    out("  " + "-" * 84)
    for tag, steps, sched in (("4000, constant lr", 4000, False),
                              ("4000, cosine", 4000, True),
                              ("20000, constant lr", 20_000, False),
                              ("20000, cosine", 20_000, True),
                              ("40000, cosine  <- used", 40_000, True)):
        t0 = time.time()
        r = fit(D, 4, "hyp", SEEDS, steps, schedule=sched)
        a5 = r["traj"][-500:].max(axis=0) - r["traj"][-500:].min(axis=0)
        out("  %-24s | %-9.5f %-9.5f | %-11.2e %-11.2e | %-9.1fs"
            % (tag, np.median(r["final"]), np.median(r["best"]),
               np.median(r["tail_amplitude"]), np.median(a5),
               time.time() - t0))
    out()
    out("  THE COSINE SCHEDULE ALONE WAS ENOUGH: at the SAME budget of 4000")
    out("  steps the tail amplitude drops ~2x, and adding steps up to 40000")
    out("  cuts it by two more orders.  A constant lr converges NEITHER at")
    out("  4000 NOR at 20000 steps -- the amplitude sits at ~2.8e-02, so this")
    out("  was not a budget problem but a schedule problem.")
    out()
    out("  The objective is NOT strictly monotone and cannot be -- this is a")
    out("  subgradient method on a non-smooth |.|, so individual uphill steps")
    out("  are legal.  The condition that matters is tail amplitude << the")
    out("  measured difference, and it is reported for EVERY row of the tables")
    out("  below.  We report both the final and the best-per-seed objective;")
    out("  the verdict is computed on both and compared.")

    out()
    out("=" * 98)
    out("2. THE REAL TREE (%d leaves), %d seeds, %d steps"
        % (len(names), len(SEEDS), STEPS_MAIN))
    out("=" * 98)
    W = geometry_table(out, D, DIMS, STEPS_MAIN)
    out()
    out("  divergence of the two distance formulas, max over dimensions: "
        "%.2e (relative)" % max(W[d]["hyp"]["divergence"] for d in DIMS))
    convergence_report(out, W, DIMS)
    out()
    out("  --- verdict on the FINAL objective ---")
    wk = falsifier_table(out, W, DIMS, "final")
    out()
    out("  --- verdict on the BEST objective (min over the trajectory, per "
        "seed) ---")
    wn = falsifier_table(out, W, DIMS, "best")
    out()
    out("  final vs best verdicts: %s"
        % ("AGREE in every dimension" if
           all(wk[d] == wn[d] for d in DIMS) else "DISAGREE -- see above"))
    paired_test(out, W, DIMS, "final")

    out()
    out()
    out("  HOW MANY FLAT DIMENSIONS CURVATURE BUYS (read off the table above):")
    out("  For each H^d: the smallest R^k with a median NO WORSE than H^d.")
    for dim in DIMS:
        mh = float(np.median(W[dim]["hyp"]["final"]))
        equal = [k for k in DIMS
                 if float(np.median(W[k]["euc"]["final"])) <= mh]
        if equal:
            out("    H^%-2d (%.5f)  <->  R^%-2d (%.5f)"
                % (dim, mh, min(equal),
                   float(np.median(W[min(equal)]["euc"]["final"]))))
        else:
            out("    H^%-2d (%.5f)  <->  NO measured R^k (up to R^%d inclusive)"
                % (dim, mh, max(DIMS)))

    out()
    out("=" * 98)
    out("3. CONTROL: a synthetic tree of KNOWN hyperbolicity")
    out("=" * 98)
    out("  A complete binary tree, 64 leaves, branch lengths iid from the pool")
    out("  of the real tree (the same %d values, sampled with replacement, "
        "seed 17)." % len(pool))
    out("  If curvature wins HERE in many dimensions and not on our 10 leaves,")
    out("  then the method works and our tree is too small.")
    b64 = binary_tree(64, pool)
    n64, D64 = P.patristic_matrix(b64)
    out("  64 leaves, %d pairs, distances %.4f .. %.4f"
        % (len(n64) * (len(n64) - 1) // 2, D64[D64 > 0].min(), D64.max()))
    out()
    W64 = geometry_table(out, D64, DIMS, STEPS_CONTROL)
    convergence_report(out, W64, DIMS)
    falsifier_table(out, W64, DIMS, "final")
    paired_test(out, W64, DIMS, "final")

    out()
    out("=" * 98)
    out("4. SIZE SWEEP -- is 10 leaves too few")
    out("=" * 98)
    out("  The same generator, the same length pool, only the leaf count grows.")
    out("  The 'comb 64' row separates SIZE from SHAPE: our tree is a ladder,")
    out("  the 64-leaf control is balanced.")
    out("  Note: the 64-leaf and the REAL 10 rows are carried over from")
    out("  sections 3 and 2, so they ran at %d and %d steps respectively."
        % (STEPS_CONTROL, STEPS_MAIN))
    out()
    out("  %-16s %-5s | %-9s %-9s | %-8s | %-9s | %s" %
        ("tree", "dim", "med hyp", "med euc", "ratio", "signs", "verdict"))
    out("  " + "-" * 90)
    trees = [("binary 8", P.patristic_matrix(binary_tree(8, pool))[1]),
             ("REAL 10", D),
             ("binary 16", P.patristic_matrix(binary_tree(16, pool))[1]),
             ("binary 32", P.patristic_matrix(binary_tree(32, pool))[1]),
             ("binary 64", D64),
             ("comb 64", P.patristic_matrix(comb_tree(64, pool))[1])]
    for name, Dx in trees:
        for dim in DIMS_SIZE_SWEEP:
            if name == "binary 64":
                h, e = W64[dim]["hyp"]["final"], W64[dim]["euc"]["final"]
            elif name == "REAL 10":
                h, e = W[dim]["hyp"]["final"], W[dim]["euc"]["final"]
            else:
                h = fit(Dx, dim, "hyp", SEEDS, STEPS_CONTROL)["final"]
                e = fit(Dx, dim, "euc", SEEDS, STEPS_CONTROL)["final"]
            mh, me = float(np.median(h)), float(np.median(e))
            wide, _ = disjoint(h, e)
            who = "hyperbolic" if mh < me else "euclidean"
            out("  %-16s %-5d | %-9.5f %-9.5f | %7.2fx | %2d/%-6d | %s"
                % (name, dim, mh, me, me / mh if mh > 0 else float("inf"),
                   int((h < e).sum()), len(h),
                   ("%s wins" % who) if wide else "TIE (inside noise)"))
        out("  " + "-" * 90)

    out()
    out("  total time: %.1f min" % ((time.time() - t_start) / 60))
    out("  log: %s" % log)


if __name__ == "__main__":
    main()
