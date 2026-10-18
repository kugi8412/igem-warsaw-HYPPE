"""Lorentz (hyperboloid) geometry: the arithmetic every other module stands on.

The ladder is a tree by construction: L0 quantises position into 25 windows
of 32 bp with a 128-entry hyperbolic codebook, and four tangent FSQ levels
refine it. Hyperbolic volume grows like ``sinh^(n-1)(r)``, so a parent far
from the origin holds the same number of descendants in a narrower angle, and
that is the only reason curvature is here. It is NOT justified by the input
data: Gromov delta on 296 JASPAR motif profiles came out 0.0754 against
0.0359 for random points in R^16, i.e. motif space is less tree-like than
random Euclidean. What is tree-like is the codebook hierarchy, because we
build it that way, and the species tree -- both of which live in this file's
codomain, not its domain.

The Lorentz model is used rather than the Poincare ball because the ball's
coordinates blow up near its boundary: points at large radius crowd into a
shell where float32 cannot separate them, and gradients explode exactly where
the hierarchy's leaves live. The hyperboloid has no boundary in coordinates.

Six failures of the "looks right and is not" kind are documented at the
functions they belong to, each re-introducible by reading clean code and
simplifying it: operator precedence in the transport alpha
(:func:`parallel_transport`), metric raising in the gradient conversion
(:func:`riemannian_grad`), ``arccosh(1 + eps)`` reporting a nonzero distance
from a point to itself (:func:`geodesic_distance`), ``d**2`` not being
differentiable at zero (:func:`lorentz_sqdist`), arithmetic means of ambient
coordinates leaving the manifold (:func:`frechet_mean`), and Adam's enormous
first step (:class:`RiemannianAdam`).

Two deliberate deviations from the source this was ported from:
``minkowski_dot`` defaults to ``keepdim=False``, and :func:`expmap` clamps
the step LENGTH while normalising by the TRUE tangent norm. Both silently
produce a plausible wrong answer when reversed.

Exported falsifiers: :func:`assert_on_manifold`, :func:`assert_in_tangent`,
:func:`manifold_report`, :func:`embed_tree_distances` and :func:`self_test`,
the last runnable as ``python -m hyppe.geometry.lorentz``.

Weight decay is a known killer: lambda 0.0762 over 34 epochs is a factor
1.8e-5, which drags hyperboloid points to the origin and kills branch gates.
:class:`RiemannianAdam` refuses a group that is both ``manifold=True`` and
``weight_decay > 0``, and :func:`manifold_param_groups` reports the NAMES it
exempted, because the pattern matches a name substring and ``maska_wektor``
once matched where ``wektor_maski`` did not.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "MAX_TANGENT_NORM",
    "minkowski_dot",
    "project_to_hyperboloid",
    "project_to_tangent",
    "riemannian_grad",
    "expmap",
    "logmap",
    "expmap0",
    "logmap0",
    "parallel_transport",
    "parallel_transport0",
    "geodesic_distance",
    "lorentz_distance",
    "lorentz_sqdist",
    "lorentz_sqdist_exact",
    "radius",
    "origin_like",
    "frechet_mean",
    "lorentz_centroid",
    "assert_on_manifold",
    "assert_in_tangent",
    "manifold_report",
    "RiemannianAdam",
    "mark_manifold",
    "DEFAULT_NO_DECAY_PATTERNS",
    "is_manifold_param",
    "manifold_param_groups",
    "TreeEmbedding",
    "embed_tree_distances",
    "balanced_tree_distances",
    "self_test",
]


# Numerical constants

# cosh(8) ~ 1490, so |s|^2 and x0^2 are both ~2.2e6 and their difference must
# come out as exactly 1; float64 leaves ~1e-10 of absolute slack there and
# float32 has already lost the constraint. Longer steps are TRUNCATED
# (direction kept, length clipped), never silently taken.
MAX_TANGENT_NORM = 8.0

# Per-dtype floor, applied to SQUARED quantities as ``eps*eps`` so that the
# resulting NORM floor is ``eps`` and not ``sqrt(eps)``. Clamping the square
# by ``eps`` itself floors the norm at 3.16e-4 for eps=1e-7, enough to make
# the Frechet mean of a point with itself drift.
_EPS_BY_DTYPE = {
    torch.float64: 1e-15,
    torch.float32: 1e-7,
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


def _eps(t: torch.Tensor) -> float:
    """Numerical floor appropriate to ``t``'s dtype."""
    return _EPS_BY_DTYPE.get(t.dtype, 1e-7)


# Lorentz primitives

def minkowski_dot(x: torch.Tensor, y: torch.Tensor,
                  keepdim: bool = False) -> torch.Tensor:
    """``<x, y>_L = -x_0 y_0 + sum_i x_i y_i`` -- negative in the time slot.

    Note the default: ``keepdim=False``. Getting it wrong does not raise, a
    ``[..., 1]`` result broadcasts against a ``[..., D+1]`` point and yields a
    wrong tensor of the right shape, so every call in this file is explicit.
    """
    prod = x * y
    if keepdim:
        return prod[..., 1:].sum(dim=-1, keepdim=True) - prod[..., :1]
    return prod[..., 1:].sum(dim=-1) - prod[..., 0]


def project_to_hyperboloid(x: torch.Tensor, ambient: bool = False) -> torch.Tensor:
    """Coordinates -> a point on H^n, exactly on the manifold by construction.

    With ``ambient=False`` (default) ``x`` is the SPATIAL part ``[..., D]``
    and the time coordinate is solved from the constraint,
    ``x_0 = sqrt(1 + ||x||^2)``; with ``ambient=True`` it is a full
    ``[..., D+1]`` vector whose time slot is DISCARDED and recomputed, which
    is how a drifted point is re-seated after an exp map, an EMA or a cast.

    The two meanings differ by one coordinate and nothing in the shapes
    distinguishes them: passing an ambient point with the default silently
    embeds a D+1 dimensional "spatial" part and every dimension check
    downstream still passes.
    """
    spatial = x[..., 1:] if ambient else x
    time = torch.sqrt(1.0 + spatial.pow(2).sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial], dim=-1)


def project_to_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project an ambient VECTOR onto the tangent space at ``p``.

    Correct because ``<p, P(v)>_L = <p,v>_L + <p,v>_L * <p,p>_L = 0`` when
    ``<p,p>_L = -1``. This takes a vector, not a gradient: a gradient is a
    COVECTOR and needs :func:`riemannian_grad`.
    """
    return v + minkowski_dot(p, v, keepdim=True) * p


def riemannian_grad(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Euclidean gradient -> Riemannian gradient on the hyperboloid.

    Two steps, and skipping the first is a silent bug that looks like a
    working optimiser: a gradient is a covector, so turning it into a vector
    means RAISING ITS INDEX with the metric, and the Minkowski signature
    ``(-,+,...,+)`` makes the TIME COMPONENT FLIP SIGN. Projecting the raw
    Euclidean gradient instead gives a direction that is not descent, and it
    presents as a loss that rises for a few steps and then diverges to NaN.

    Built with ``cat`` rather than an in-place ``g[..., 0] = -g[..., 0]``,
    which raises on a leaf inside an autograd region.
    """
    lifted = torch.cat([-grad[..., :1], grad[..., 1:]], dim=-1)
    return project_to_tangent(p, lifted)


def expmap(p: torch.Tensor, v: torch.Tensor,
           eps: Optional[float] = None,
           max_norm: float = MAX_TANGENT_NORM) -> torch.Tensor:
    """Exponential map: follow the geodesic from ``p`` along tangent ``v``.

    ``exp_p(v) = cosh(t) p + sinh(t) v/|v|`` with ``t = |v|``. The norm floor
    is applied to the SQUARE as ``eps*eps``, so the norm is floored at ``eps``
    and ``cosh(eps) == 1.0`` exactly in the relevant dtypes, which makes
    ``expmap(p, 0) == p`` bit for bit -- the Frechet mean's fixed point
    depends on it. The step is clipped to ``max_norm`` but the DIRECTION still
    divides by the true norm: clipping the length is a truncation, clipping
    the denominator returns a point that is not on the manifold at all.
    """
    eps = _eps(v) if eps is None else eps
    sq = minkowski_dot(v, v, keepdim=True).clamp_min(eps * eps)
    norm = torch.sqrt(sq)                      # true tangent length, floored
    step = norm.clamp(max=float(max_norm))     # length actually travelled
    return torch.cosh(step) * p + torch.sinh(step) * (v / norm)


def logmap(p: torch.Tensor, q: torch.Tensor,
           eps: Optional[float] = None) -> torch.Tensor:
    """Inverse of :func:`expmap` -- the tangent at ``p`` pointing to ``q``.

    ``log_p(q) = d(p,q) * u / |u|`` with ``u = q - alpha p`` and
    ``alpha = -<p,q>_L``. Tangency is exact, ``<p,u>_L = -alpha + alpha = 0``.

    The precedence trap, stated once for the whole file: the parentheses must
    wrap the NEGATION, as they do below. Both of

        -minkowski_dot(p, q).clamp_min(1.0)      # clamps, then negates
        -(minkowski_dot(p, q)).clamp_min(1.0)    # identical -- also wrong

    clamp a dot product of about -1 up to +1 and negate it back, which makes
    the transport denominator ~-1e-7 and amplifies momentum by 1e11 per step.
    Wrapping the call changes nothing, because Python binds unary minus after
    the method call either way.

    The magnitude comes from ``asinh(|u|)`` and not from ``acosh(alpha)``,
    which is exact analytically but has infinite derivative at 1: ``alpha`` is
    a difference of large numbers carrying ~2e-16 of noise even for ``q`` a
    bit-for-bit copy of ``p``, and ``acosh`` turns that into ~2e-8, so the
    tangent from a point to itself comes back with length ~4e-9.
    :func:`geodesic_distance` keeps the ``acosh`` route on purpose, because it
    must stay a pure inner product for assignment to be one GEMM; the logmap
    already materialises ``u``, so the better form is free here.

    The floors sit INSIDE the sqrt, masking the operand rather than the
    result, so the backward pass is finite at coincidence; ``torch.where`` on
    the result would still propagate NaN.
    """
    eps = _eps(p) if eps is None else eps
    alpha = (-minkowski_dot(p, q, keepdim=True)).clamp_min(1.0)
    u = q - alpha * p
    # <u,u>_L = alpha^2 - 1 = sinh^2(d) >= 0 analytically; the clamp only
    # catches the rounding regime where q is numerically p and u comes out
    # timelike by 1e-32.
    u_norm = torch.sqrt(minkowski_dot(u, u, keepdim=True).clamp_min(eps * eps))
    return torch.asinh(u_norm) * u / u_norm


def parallel_transport(p: torch.Tensor, q: torch.Tensor, v: torch.Tensor,
                       eps: Optional[float] = None) -> torch.Tensor:
    """Carry tangent ``v`` from ``T_p`` to ``T_q`` along the connecting geodesic.

    Needed for momentum: a velocity accumulated at the old point is not a
    valid tangent at the new one. ``PT(v) = v + <q,v>_L / (1 - <p,q>_L) *
    (p + q)`` is an isometry and lands exactly in ``T_q``.

    The denominator is structurally safe -- for two points on the sheet
    ``<p,q>_L <= -1``, so ``1 - <p,q>_L >= 2`` -- and the clamp guards inputs
    that are already off the manifold rather than being a numerical fudge.
    This is where the precedence bug lived: the buggy alpha turned a
    denominator of >= 2 into ~-1e-7.
    """
    eps = _eps(p) if eps is None else eps
    denom = (1.0 - minkowski_dot(p, q, keepdim=True)).clamp_min(eps)
    return v + minkowski_dot(q, v, keepdim=True) / denom * (p + q)


def geodesic_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``arccosh(-<p,q>_L)``, evaluated stably near zero. Shape ``[...]``.

    The naive ``acosh(clamp(inner, min=1+eps))`` is wrong in a way that is
    easy to miss: ``arccosh(1+eps) ~ sqrt(2 eps)``, so an eps of 1e-7 reports
    a distance of 4.5e-4 between a point and itself, which corrupts
    nearest-code assignment for exactly the close pairs quantisation cares
    about. Writing ``t = -<p,q>_L - 1`` and using
    ``arccosh(1+t) = log1p(t + sqrt(t^2+2t))`` gives exactly 0 at ``t = 0``
    and stays accurate for large ``t``.

    That removes the artefact but not the CONDITIONING of ``arccosh``, which
    turns the ~2e-16 of noise in ``-<p,q>_L`` into ~2e-8: distances below
    ~1e-7 are noise rather than small numbers, and a test asserting
    ``d(p,p) == 0`` exactly will fail about half the time. Where that matters,
    use the ``asinh`` route of :func:`logmap` instead.

    Do not differentiate this at coincidence: ``sqrt(t^2+2t)`` has infinite
    slope at ``t = 0``, which is where a commitment loss drives. Use
    :func:`lorentz_sqdist` in a differentiated objective.
    """
    inner = -minkowski_dot(p, q, keepdim=False)
    t = (inner - 1.0).clamp_min(0.0)
    return torch.log1p(t + torch.sqrt(t * t + 2.0 * t))


#: Alias kept so ported call sites read the same.
lorentz_distance = geodesic_distance


def lorentz_sqdist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Smooth surrogate for the SQUARED geodesic distance: ``-<p,q>_L - 1``.

    Use this, never ``geodesic_distance(...)**2``, in anything that is
    differentiated: the geodesic distance contains ``sqrt(t^2+2t)``, whose
    slope is infinite at ``t = 0``, so the moment a point converges onto its
    target autograd evaluates ``2 * 0 * inf`` and returns NaN. This quantity
    equals ``d^2 / 2`` to leading order, is monotone in ``d`` and is a
    polynomial in the coordinates, so its gradient is finite everywhere.
    """
    return (-minkowski_dot(p, q, keepdim=False) - 1.0).clamp_min(0.0)


def lorentz_sqdist_exact(p: torch.Tensor, q: torch.Tensor,
                         knee: float = 1e-2) -> torch.Tensor:
    """``d^2 / 2`` computed from the GEODESIC distance, not from the chord.

    :func:`lorentz_sqdist` returns ``cosh(d) - 1``, which agrees with
    ``d^2 / 2`` to leading order and then diverges without limit (ratio 1.086
    at d = 1, 220 at d = 10, 1.5e14 at d = 40), so a weight chosen for
    ``d^2 / 2`` stops being a weight once the encoder leaves the
    neighbourhood of its codes: at radius 41.3 a commitment weight of 0.25
    became 99.999 % of the total loss. Returning ``d^2 / 2`` rather than
    ``d^2`` makes the two agree to fourth order at the origin, so swapping
    them changes behaviour at large distance only.

    ``d = arccosh(1 + t)`` has infinite slope at ``t = 0`` but ``d^2`` does
    not (``d(d^2)/dt -> 2``); the singularity is in the FORM
    ``2 d / sqrt(t^2 + 2t)``, which is 0/0, so below ``knee`` the series
    ``d^2 = 2t - t^2/3 + 4t^3/45 - ...`` is used instead. The ``torch.where``
    is safe because the OPERAND is masked, not the result -- an unmasked
    operand makes the unused branch produce NaN and ``0 * NaN`` poisons the
    gradient. Truncation error at the switch is ~1.3e-5 relative.
    """
    t = (-minkowski_dot(p, q, keepdim=False) - 1.0).clamp_min(0.0)
    t_big = t.clamp_min(float(knee))
    d = torch.log1p(t_big + torch.sqrt(t_big * (t_big + 2.0)))
    far = 0.5 * d * d
    near = t * (1.0 - t * (1.0 / 6.0) + t * t * (2.0 / 45.0))
    return torch.where(t > float(knee), far, near)


def origin_like(x: torch.Tensor) -> torch.Tensor:
    """The hyperboloid origin ``(1, 0, ..., 0)`` broadcast to ``x``'s shape."""
    o = torch.zeros_like(x)
    o[..., 0] = 1.0
    return o


def radius(p: torch.Tensor) -> torch.Tensor:
    """Geodesic distance of each point from the origin -- the "level" coordinate.

    Computed as ``asinh(||spatial||)``, which is what the radius IS
    (``cosh r = x_0``, ``sinh r = ||s||``), rather than ``acosh(x_0)``: at
    large radius ``x_0`` is near ``||s||``, so ``acosh`` works on a difference
    of large numbers while ``||s||`` is accurate directly.

    Radius being a level label is a DESIGN INTENT, not an observation: on a
    sibling model the per-sequence radius tracked GC content (r = +0.84) at a
    spread of 0.5-3.5 %, i.e. the space was working like a sphere.
    """
    spatial = p[..., 1:]
    return torch.asinh(spatial.norm(dim=-1))


def expmap0(v: torch.Tensor, max_norm: float = MAX_TANGENT_NORM) -> torch.Tensor:
    """Tangent vector at the origin (``[..., D]``, spatial only) -> point on H^n.

    The origin's tangent space is the set of vectors with zero time
    component, so only the spatial part carries information. Writing the map
    as ``sinh(|v|) v/|v|`` in the spatial slot and letting
    :func:`project_to_hyperboloid` recompute the time slot keeps the result
    exactly on the manifold, which the closed form does not once ``|v|`` is
    large enough for ``cosh`` and ``sqrt(1+sinh^2)`` to disagree.
    """
    eps = _eps(v)
    norm = v.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    step = norm.clamp(max=float(max_norm))
    return project_to_hyperboloid(torch.sinh(step) * v / norm)


def logmap0(p: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`expmap0` -- a point on H^n as a tangent at the origin.

    Returns ``D`` spatial coordinates, not ``D+1``: the time component of a
    tangent at the origin is identically zero, so carrying it would hand every
    downstream layer a column of zeros. ``asinh`` rather than ``acosh(x_0)``
    for the reason given in :func:`radius`.
    """
    eps = _eps(p)
    spatial = p[..., 1:]
    norm = spatial.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    return torch.asinh(norm) * spatial / norm


def parallel_transport0(q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Transport a SPATIAL tangent ``u`` (``[..., D]``) from the origin to ``T_q``.

    The second half of the ladder's composition rule
    ``p_{k+1} = expmap(p_k, PT(origin -> p_k, v_{k+1}))``, where ``v_{k+1}``
    is the level's refinement vector, produced in the origin's tangent space
    because that is where the FSQ grid lives. Lifting ``u`` to ``(0, u)`` and
    transporting is what makes the refinement mean the same thing at every
    parent, which is the point of composing geodesically rather than adding
    coordinates.
    """
    zeros = torch.zeros_like(u[..., :1])
    lifted = torch.cat([zeros, u], dim=-1)
    return parallel_transport(origin_like(q), q, lifted)


def frechet_mean(points: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 iters: int = 8) -> torch.Tensor:
    """Karcher/Frechet mean by tangent-space averaging, over dim ``-2``.

    The hyperbolic analogue of an arithmetic mean has no closed form, so this
    iterates: log-map every point into the current estimate's tangent space,
    average there, exp-map back.

    This is the function a codebook EMA target must use: averaging ambient
    coordinates and re-projecting is biased, and the bias GROWS WITH RADIUS
    (0.2281 / 1.3309 / 2.2894 at shell radii 1.42 / 2.25 / 2.92), so the
    outermost shell, with the most codes, is damaged most. Re-projected after
    every step, because repeated exp maps accumulate float error and drift off
    the sheet silently corrupts every later distance.
    """
    if weights is None:
        weights = points.new_ones(points.shape[:-1])
    w = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    w = w.unsqueeze(-1)
    mean = project_to_hyperboloid(points[..., 1:].mean(dim=-2))
    for _ in range(int(iters)):
        tangents = logmap(mean.unsqueeze(-2), points)
        step = (w * tangents).sum(dim=-2)
        mean = project_to_hyperboloid(expmap(mean, step), ambient=True)
    return mean


def lorentz_centroid(points: torch.Tensor,
                     weights: Optional[torch.Tensor] = None,
                     dim: int = -2) -> torch.Tensor:
    """Weighted centroid on the hyperboloid, in closed form.

    The Lorentzian centroid minimises squared Lorentzian distance rather than
    squared geodesic distance and so has an exact solution: the weighted
    ambient sum renormalised back onto the sheet. That makes it the standard
    aggregation for hyperbolic attention -- no inner loop to unroll through,
    finite gradient everywhere -- while :func:`frechet_mean` remains the right
    choice for the codebook update, which runs under ``no_grad``.
    """
    if weights is None:
        summed = points.sum(dim=dim)
    else:
        summed = (points * weights.unsqueeze(-1)).sum(dim=dim)
    # |<s,s>_L| is positive for any non-degenerate timelike sum; the clamp only
    # catches the all-weights-zero case, which cannot be normalised at all.
    norm = torch.sqrt((-minkowski_dot(summed, summed, keepdim=True)).clamp_min(1e-9))
    out = summed / norm
    # One re-projection: the division lands on the sheet analytically, but in
    # float32 it drifts by ~1e-6 per layer and that accumulates into acosh
    # domain errors several layers later.
    return project_to_hyperboloid(out, ambient=True)


# Falsifiers

def assert_on_manifold(x: torch.Tensor, tol: float = 1e-6,
                       raise_on_fail: bool = True,
                       name: str = "points") -> float:
    """Return ``max |<x,x>_L + 1|``; raise if it exceeds ``tol``.

    The test is structural, not a dtype check: the failure that made it
    necessary kept the codebook in float64 throughout and still left it off
    the manifold, because the EMA averaged ambient coordinates, and a reported
    reconstruction of 0.9945 was really 0.8012.

    Also checks the SHEET, ``x_0 > 0``. The hyperboloid has two components and
    an average of ambient coordinates can land on the lower one, where every
    distance is meaningless and nothing raises.
    """
    if x.numel() == 0:
        return 0.0
    with torch.no_grad():
        viol = float((minkowski_dot(x, x, keepdim=False) + 1.0).abs().max())
        wrong_sheet = int((x[..., 0] <= 0).sum())
    if raise_on_fail and (viol > tol or wrong_sheet > 0):
        raise ValueError(
            f"{name}: off the hyperboloid -- max|<x,x>_L + 1| = {viol:.3e} "
            f"(tol {tol:.1e}), wrong-sheet entries = {wrong_sheet}. "
            "Usual causes: an arithmetic mean of ambient coordinates (use "
            "frechet_mean), an EMA over raw coordinates, or a missing "
            "re-projection after expmap.")
    return viol


def assert_in_tangent(p: torch.Tensor, v: torch.Tensor, tol: float = 1e-6,
                      raise_on_fail: bool = True,
                      name: str = "tangent") -> float:
    """Return ``max |<p,v>_L|``; raise if it exceeds ``tol``.

    A "tangent" that is not tangent makes :func:`expmap` leave the manifold,
    and the result still looks like a point. Element-wise rescaling -- what
    Adam does to a direction -- does NOT preserve tangency, which is why
    :class:`RiemannianAdam` re-projects after the rescale.
    """
    if v.numel() == 0:
        return 0.0
    with torch.no_grad():
        viol = float(minkowski_dot(p, v, keepdim=False).abs().max())
    if raise_on_fail and viol > tol:
        raise ValueError(f"{name}: not tangent -- max|<p,v>_L| = {viol:.3e} "
                         f"(tol {tol:.1e})")
    return viol


@torch.no_grad()
def manifold_report(x: torch.Tensor, sample: int = 1024) -> dict:
    """Health of a set of manifold points. Every entry is a falsifier.

    ``violation`` is ``max |<x,x>_L + 1|``; above 1e-12 in float64 something
    averaged, cast or stepped without re-projecting. ``wrong_sheet`` counts
    ``x_0 <= 0`` and must be 0. A FLAT ``radius_*`` distribution means the
    hyperbolic structure is not being used at all. ``min_pair_d`` is the
    smallest pairwise geodesic distance on a subsample; near zero means codes
    have merged, and collapse is ABSORBING without a restart (measured
    ``bits_used_frac`` 0.2970 in epoch 1 with the restart disabled).
    """
    flat = x.reshape(-1, x.shape[-1])
    out = {
        "n": int(flat.shape[0]),
        "dim": int(flat.shape[-1]) - 1,
        "dtype": str(flat.dtype),
        "violation": assert_on_manifold(flat, raise_on_fail=False),
        "wrong_sheet": int((flat[..., 0] <= 0).sum()),
    }
    r = radius(flat)
    out["radius_mean"] = float(r.mean())
    out["radius_std"] = float(r.std()) if r.numel() > 1 else 0.0
    out["radius_min"] = float(r.min())
    out["radius_max"] = float(r.max())
    n = int(flat.shape[0])
    if n > 1:
        take = flat[:sample] if n > sample else flat
        m = int(take.shape[0])
        # PAIRWISE DISTANCES AS ONE GEMM, never by broadcasting the points.
        # `geodesic_distance(take[:,None], take[None])` materialises
        # [m, m, D+1] -- at KAAMMB's shapes that was 6.3 GB for one level.
        # The Minkowski product is bilinear, so flipping the time sign turns
        # it into a matrix product at [m, m].
        flipped = torch.cat([-take[..., :1], take[..., 1:]], dim=-1)
        inner = flipped @ take.t()
        t = (-inner - 1.0).clamp_min(0.0)
        d = torch.log1p(t + torch.sqrt(t * t + 2.0 * t))
        off = d + torch.eye(m, device=d.device, dtype=d.dtype) * (d.max() + 1.0)
        out["min_pair_d"] = float(off.min())
        out["mean_pair_d"] = float(d.sum() / max(m * m - m, 1))
        out["pair_sample"] = m
    return out


# Riemannian optimiser

class RiemannianAdam(torch.optim.Optimizer):
    """Adam on the Lorentz manifold, mixed with ordinary AdamW.

    Three differences from Euclidean Adam, each required for correctness: the
    gradient is converted with :func:`riemannian_grad` (metric raising, then
    projection); the step is taken with the exponential map, not by addition;
    and momentum is parallel-transported to the new point, because a velocity
    accumulated at the old one is not a valid tangent at the new one.

    A group carrying ``manifold=True`` is treated as hyperbolic and everything
    else falls through to ordinary AdamW, so one optimiser can drive a mixed
    model; use :func:`mark_manifold` and :func:`manifold_param_groups` to
    build the groups. Weight decay on a manifold group is REFUSED, not
    ignored, because lambda 0.0762 over 34 epochs is a factor 1.8e-5 on the
    point itself.

    The Euclidean branch decays in the DECOUPLED (AdamW) form, which matters
    because a global ``riemannian: true`` routes every parameter through this
    optimiser: the coupled form scales decay by Adam's per-parameter
    denominator, so parameters with small gradients are decayed far harder
    (measured 0.4019 against 0.5048 at epoch 1 on otherwise identical runs).
    """

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 manifold: bool = False, max_step: float = 1.0):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        if not 0.0 < betas[0] < 1.0 or not 0.0 < betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      manifold=manifold, max_step=max_step))

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group.get("manifold") and group.get("weight_decay"):
            raise ValueError(
                "weight_decay > 0 on a manifold param group. Decay pulls "
                "hyperboloid points to the origin (measured: lambda 0.0762 "
                "for 34 epochs = factor 1.8e-5) and the damage is invisible "
                "in the loss. Put manifold parameters in a group with "
                "weight_decay=0.")
        if group.get("manifold"):
            for p in group["params"]:
                if p.dim() < 1 or p.shape[-1] < 2:
                    raise ValueError(
                        "a manifold parameter must have a last dimension of "
                        f"D+1 >= 2, got shape {tuple(p.shape)}")
        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    # ELEMENT-WISE second moment, as in Euclidean Adam. A
                    # single scalar Minkowski norm per point diverges to NaN:
                    # as the loss approaches zero the norm collapses, the
                    # denominator falls to eps, and the step explodes into an
                    # expmap that overflows.
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                if not group["manifold"]:
                    # DECOUPLED weight decay -- see the class docstring.
                    if group["weight_decay"]:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    m.mul_(b1).add_(grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denom = (v / (1 - b2 ** t)).sqrt().add_(group["eps"])
                    p.addcdiv_(m / (1 - b1 ** t), denom, value=-group["lr"])
                    continue

                rgrad = riemannian_grad(p, grad)
                m.mul_(b1).add_(rgrad, alpha=1 - b1)
                v.mul_(b2).addcmul_(rgrad, rgrad, value=1 - b2)
                denom = (v / (1 - b2 ** t)).sqrt().add_(group["eps"])
                # Re-project after the element-wise rescale: dividing
                # component by component does not preserve tangency, and an
                # off-tangent step walks the point off the hyperboloid.
                direction = project_to_tangent(p, (m / (1 - b1 ** t)) / denom)

                # Clip the geodesic step length. Element-wise Adam divides by
                # a second moment near `eps` on the first steps, and one such
                # step throws the point to a radius where
                # `<x,x>_L = ||s||^2 - x0^2` is catastrophic cancellation even
                # in float64, after which every distance is garbage.
                step_norm = torch.sqrt(
                    minkowski_dot(direction, direction, keepdim=True).clamp_min(0.0))
                scale = (group["max_step"] / step_norm.clamp_min(1e-12)).clamp(max=1.0)
                new_p = expmap(p, -group["lr"] * direction * scale)
                # RE-PROJECT: repeated exp maps accumulate float error and the
                # constraint drifts, which silently corrupts every distance.
                new_p = project_to_hyperboloid(new_p, ambient=True)
                m.copy_(parallel_transport(p, new_p, m))
                p.copy_(new_p)
        return loss


# Parameter bookkeeping: the weight-decay contract

def mark_manifold(param: torch.Tensor) -> torch.Tensor:
    """Tag ``param`` as living on the hyperboloid. Returns it, for chaining.

    An explicit flag, not a name pattern: exempting by substring once lost a
    parameter to word order (``maska_wektor`` matched, ``wektor_maski`` did
    not) and the difference was invisible until the branch was dead.
    """
    setattr(param, "is_manifold", True)
    return param


def is_manifold_param(param: torch.Tensor) -> bool:
    """True if ``param`` was tagged by :func:`mark_manifold`."""
    return bool(getattr(param, "is_manifold", False))


#: Substrings that exempt a parameter from weight decay when no explicit flag
#: is present. Gates are listed because a decayed gate dies while the run
#: keeps training as if nothing had happened.
DEFAULT_NO_DECAY_PATTERNS: Tuple[str, ...] = (
    "bias", "gate", "brama", "scale", "skala", "temp", "tau", "log_",
    "norm", "embed_time",
)


def manifold_param_groups(model: nn.Module, weight_decay: float = 0.0,
                          no_decay_patterns: Sequence[str] = DEFAULT_NO_DECAY_PATTERNS,
                          lr: Optional[float] = None) -> Tuple[List[dict], dict]:
    """Split a model into (manifold, no-decay, decay) groups for the optimiser.

    Returns ``(groups, report)``, where the report lists the ACTUAL NAMES in
    each group: the exemption rule matches a name substring, so the only way
    to know it matched what was meant is to read back what it matched.

    Manifold parameters never receive decay -- :class:`RiemannianAdam` refuses
    the combination anyway -- and one-dimensional parameters are exempt by the
    usual convention.
    """
    manifold: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    decay: List[torch.nn.Parameter] = []
    report = {"manifold": [], "no_decay": [], "decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if is_manifold_param(p):
            manifold.append(p)
            report["manifold"].append(name)
        elif p.dim() <= 1 or any(pat in low for pat in no_decay_patterns):
            no_decay.append(p)
            report["no_decay"].append(name)
        else:
            decay.append(p)
            report["decay"].append(name)
    groups: List[dict] = []
    if manifold:
        g = {"params": manifold, "manifold": True, "weight_decay": 0.0}
        groups.append(g)
    if no_decay:
        groups.append({"params": no_decay, "manifold": False, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "manifold": False,
                       "weight_decay": float(weight_decay)})
    if lr is not None:
        for g in groups:
            g["lr"] = float(lr)
    report["n_manifold"] = len(manifold)
    report["n_no_decay"] = len(no_decay)
    report["n_decay"] = len(decay)
    report["manifold_params"] = int(sum(p.numel() for p in manifold))
    return groups, report


# Tree embedding: the species tree of 10 taxa

class TreeEmbedding(NamedTuple):
    """Result of :func:`embed_tree_distances`.

    ``points``      ``[N, dim+1]`` on the hyperboloid.
    ``distortion``  mean relative distortion after scale alignment (0 = exact).
    ``report``      every falsifier; see :func:`embed_tree_distances`.
    """
    points: torch.Tensor
    distortion: float
    report: dict


def _safe_geodesic(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    """``geodesic_distance`` with a gradient-safe floor.

    The public :func:`geodesic_distance` is exact at coincidence and has an
    infinite slope there. The fit below is differentiated and two taxa can be
    driven together, so the sqrt argument is floored -- masking the OPERAND,
    which blocks the gradient through the clamp instead of producing a NaN
    that ``torch.where`` on the result would propagate.
    """
    inner = -minkowski_dot(p, q, keepdim=False)
    t = (inner - 1.0).clamp_min(0.0)
    return torch.log1p(t + torch.sqrt((t * t + 2.0 * t).clamp_min(eps * eps)))


def _distortion_stats(d_hat: torch.Tensor, d_ref: torch.Tensor) -> dict:
    """Scale-free distortion. ``d_hat`` and ``d_ref`` are flat pair vectors.

    The scale is removed by the MEDIAN RATIO rather than a fitted constant:
    tree branch lengths are in arbitrary units, so a distortion that is really
    a unit mismatch would be a number about the caller's file format.
    ``worst`` is the classical ``max ratio / min ratio``, exactly 1 for a
    scaled isometry.
    """
    with torch.no_grad():
        ratio = d_hat / d_ref.clamp_min(1e-12)
        scale = ratio.median().clamp_min(1e-12)
        rel = (ratio / scale - 1.0).abs()
        dh = d_hat - d_hat.mean()
        dr = d_ref - d_ref.mean()
        denom = (dh.pow(2).sum().sqrt() * dr.pow(2).sum().sqrt()).clamp_min(1e-12)
        return {
            "distortion_avg": float(rel.mean()),
            "distortion_max": float(rel.max()),
            "distortion_worst": float(ratio.max() / ratio.min().clamp_min(1e-12)),
            "pearson": float((dh * dr).sum() / denom),
            "median_ratio": float(scale),
        }


def _fit_embedding(d_ref: torch.Tensor, dim: int, steps: int, lr: float,
                   hyperbolic: bool, riemannian: bool, seed: int,
                   init_scale: float) -> Tuple[torch.Tensor, dict]:
    """Shared fitting loop for the hyperbolic model and its Euclidean control."""
    device, dtype = d_ref.device, d_ref.dtype
    n = int(d_ref.shape[0])
    eps = _EPS_BY_DTYPE.get(dtype, 1e-7)
    iu = torch.triu_indices(n, n, offset=1, device=device)
    ref = d_ref[iu[0], iu[1]]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    init = (torch.randn(n, dim, generator=gen, dtype=torch.float64)
            .to(device=device, dtype=dtype) * float(init_scale))

    log_tau = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    if hyperbolic and riemannian:
        # Optimise the AMBIENT point directly with RiemannianAdam, so the
        # optimiser is exercised by an objective with a known answer; the
        # default path below is unconstrained and more robust.
        pts = project_to_hyperboloid(init).clone().requires_grad_(True)
        opt = RiemannianAdam([{"params": [pts], "manifold": True,
                               "weight_decay": 0.0},
                              {"params": [log_tau], "manifold": False}], lr=lr)
        free = pts
    else:
        free = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([free, log_tau], lr=lr)

    def _pairs():
        if hyperbolic:
            pts_now = free if riemannian else project_to_hyperboloid(free)
            return pts_now, _safe_geodesic(pts_now[iu[0]], pts_now[iu[1]], eps)
        diff = free[iu[0]] - free[iu[1]]
        return free, diff.pow(2).sum(dim=-1).clamp_min(eps * eps).sqrt()

    loss_init = None
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        _, d_hat = _pairs()
        target = torch.exp(log_tau) * ref
        # Squared RELATIVE error: an absolute error spends the whole budget
        # on the longest paths, and a tree's information is in the short ones.
        loss = ((d_hat - target) / target.clamp_min(1e-9)).pow(2).mean()
        if loss_init is None:
            loss_init = float(loss.detach())
        loss.backward()
        opt.step()

    with torch.no_grad():
        coords, d_hat = _pairs()
        target = torch.exp(log_tau) * ref
        loss_final = float(((d_hat - target) / target.clamp_min(1e-9)).pow(2).mean())
    stats = _distortion_stats(d_hat.detach(), ref)
    stats["loss_init"] = float(loss_init if loss_init is not None else float("nan"))
    stats["loss_final"] = loss_final
    stats["tau"] = float(torch.exp(log_tau.detach()))
    return coords.detach(), stats


def embed_tree_distances(dist_matrix, dim: int = 8, steps: int = 1500,
                         lr: float = 0.05, seed: int = 0,
                         init_scale: float = 0.1,
                         euclidean_control: bool = True,
                         riemannian: bool = False,
                         dtype: torch.dtype = torch.float64,
                         device=None) -> TreeEmbedding:
    """Embed a distance matrix on H^dim by minimising relative distortion.

    A tree metric embeds into hyperbolic space with distortion that stays
    bounded as depth grows, while in R^n it grows -- the one claim about
    curvature this project can make from the object itself rather than from
    the data. Parameterised by FREE SPATIAL COORDINATES, so the constraint
    holds for any ``s``, plain Adam suffices and no projection step can be
    forgotten; ``riemannian=True`` optimises the ambient point with
    :class:`RiemannianAdam` instead, which is slower and kept because it gives
    that optimiser an objective with a known answer.

    ``dist_matrix`` is ``[N, N]``, symmetric, non-negative, zero diagonal.
    ``euclidean_control`` ALSO fits R^dim at the same dimension, step budget
    and loss, and is the falsifier for the whole geometry: if flat space ties
    on a metric that IS a tree, curvature is decoration here.

    Besides the distortion statistics the report carries ``distortion_init``
    (a final value equal to it means the fit never moved),
    ``distortion_euclidean`` and ``gain_vs_euclidean`` (positive means
    curvature paid), ``manifold_violation``, ``radius_max`` (beyond ~8 the
    Minkowski cancellation eats float64 precision and the distortion is not
    trustworthy) and ``pearson`` (near zero at a distortion near zero would
    mean the scale alignment is hiding a dead fit).
    """
    if isinstance(dist_matrix, torch.Tensor):
        d = dist_matrix.to(dtype=dtype)
    else:
        d = torch.as_tensor(np.asarray(dist_matrix), dtype=dtype)
    if device is not None:
        d = d.to(device)
    if d.dim() != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"dist_matrix must be square, got {tuple(d.shape)}")
    n = int(d.shape[0])
    if n < 3:
        raise ValueError(f"need at least 3 taxa, got {n}")
    asym = float((d - d.t()).abs().max())
    if asym > 1e-8:
        raise ValueError(f"dist_matrix is not symmetric: max|d - d^T| = {asym:.3e}")
    if float(d.diagonal().abs().max()) > 1e-8:
        raise ValueError("dist_matrix must have a zero diagonal")
    if float(d.min()) < 0.0:
        raise ValueError("dist_matrix must be non-negative")
    iu = torch.triu_indices(n, n, offset=1, device=d.device)
    if float(d[iu[0], iu[1]].min()) <= 0.0:
        raise ValueError("two taxa are at distance 0; merge them before "
                         "embedding -- a relative-error loss cannot use them")

    points, stats = _fit_embedding(d, dim, steps, lr, True, riemannian, seed,
                                   init_scale)
    if not torch.is_tensor(points) or points.shape[-1] != dim + 1:
        points = project_to_hyperboloid(points)

    # Control: the random initialisation, i.e. did the fit move at all.
    init_pts, init_stats = _fit_embedding(d, dim, 0, lr, True, False, seed,
                                          init_scale)
    if init_pts.shape[-1] != dim + 1:
        init_pts = project_to_hyperboloid(init_pts)

    report = dict(stats)
    report["n_taxa"] = n
    report["dim"] = int(dim)
    report["steps"] = int(steps)
    report["riemannian"] = bool(riemannian)
    report["distortion_init"] = init_stats["distortion_avg"]
    report["manifold_violation"] = assert_on_manifold(points, raise_on_fail=False)
    with torch.no_grad():
        r = radius(points)
        report["radius_mean"] = float(r.mean())
        report["radius_max"] = float(r.max())
    if report["radius_max"] > MAX_TANGENT_NORM:
        report["warning_radius"] = (
            f"radius {report['radius_max']:.2f} exceeds MAX_TANGENT_NORM "
            f"{MAX_TANGENT_NORM}; ||s||^2 - x0^2 is now catastrophic "
            "cancellation and the distortion below is not trustworthy")
    if euclidean_control:
        _, euc = _fit_embedding(d, dim, steps, lr, False, False, seed, init_scale)
        report["distortion_euclidean"] = euc["distortion_avg"]
        report["distortion_worst_euclidean"] = euc["distortion_worst"]
        report["gain_vs_euclidean"] = euc["distortion_avg"] - stats["distortion_avg"]
    return TreeEmbedding(points=points,
                         distortion=float(stats["distortion_avg"]),
                         report=report)


def balanced_tree_distances(n_leaves: int = 10,
                            dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A deterministic tree metric on ``n_leaves`` leaves, for tests and demos.

    Each leaf gets a binary code of ``ceil(log2(n))`` bits and the distance is
    ``2 * (bits - common_prefix)``, the path metric of a complete balanced
    binary tree with unit edges. It is therefore exactly a tree metric (Gromov
    delta 0) and a fair target: a hyperbolic fit that cannot beat the
    Euclidean control here will not beat it on a real phylogeny.
    """
    if n_leaves < 2:
        raise ValueError("need at least 2 leaves")
    bits = 1
    while (1 << bits) < n_leaves:
        bits += 1
    d = torch.zeros(n_leaves, n_leaves, dtype=dtype)
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            common = 0
            for b in range(bits - 1, -1, -1):
                if ((i >> b) & 1) != ((j >> b) & 1):
                    break
                common += 1
            dist = 2.0 * (bits - common)
            d[i, j] = dist
            d[j, i] = dist
    return d


# Self test

def _random_points(n: int, dim: int, gen: torch.Generator,
                   scale: float = 0.7,
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    spatial = torch.randn(n, dim, generator=gen, dtype=dtype) * scale
    return project_to_hyperboloid(spatial)


def _random_tangents(p: torch.Tensor, gen: torch.Generator,
                     scale: float = 0.3) -> torch.Tensor:
    raw = torch.randn(p.shape, generator=gen, dtype=p.dtype) * scale
    return project_to_tangent(p, raw)


def self_test(verbose: bool = True, seed: int = 0,
              dim: int = 6, n: int = 64) -> int:
    """Check the identities this whole package rests on. Returns #failures.

    Plain arithmetic and explicit tolerances rather than ``torch.testing``, so
    the numbers stay visible. Every check corresponds to a bug that has
    actually been shipped here.
    """
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    dt = torch.float64
    tol = 1e-10

    p = _random_points(n, dim, gen, dtype=dt)
    q = _random_points(n, dim, gen, dtype=dt)
    r = _random_points(n, dim, gen, dtype=dt)
    v = _random_tangents(p, gen)

    # 0. Contract: keepdim defaults to False.
    shape_nk = tuple(minkowski_dot(p, q).shape)
    shape_k = tuple(minkowski_dot(p, q, keepdim=True).shape)
    check("minkowski_dot keepdim=False by default",
          shape_nk == (n,) and shape_k == (n, 1),
          f"no-keepdim {shape_nk}, keepdim {shape_k}")

    # 1. Points are on the manifold.
    viol = assert_on_manifold(p, raise_on_fail=False)
    check("project_to_hyperboloid lands on the sheet", viol < 1e-14,
          f"max|<x,x>_L + 1| = {viol:.3e}")

    viol_amb = assert_on_manifold(project_to_hyperboloid(p * 1.003, ambient=True),
                                  raise_on_fail=False)
    check("project_to_hyperboloid(ambient=True) re-seats a drifted point",
          viol_amb < 1e-14, f"max|<x,x>_L + 1| = {viol_amb:.3e}")

    # 2. Tangency.
    tviol = assert_in_tangent(p, v, raise_on_fail=False)
    check("project_to_tangent gives a tangent", tviol < 1e-12,
          f"max|<p,v>_L| = {tviol:.3e}")

    # 3. logmap(expmap) is the identity.
    back = logmap(p, expmap(p, v))
    err = float((back - v).abs().max())
    check("logmap(p, expmap(p, v)) == v", err < 1e-9, f"max|err| = {err:.3e}")

    fwd = expmap(p, logmap(p, q))
    err2 = float((fwd - q).abs().max())
    check("expmap(p, logmap(p, q)) == q", err2 < 1e-9, f"max|err| = {err2:.3e}")

    # 4. expmap stays on the manifold, including past the clip.
    big = v * (50.0 / max(float(torch.sqrt(minkowski_dot(v, v).clamp_min(1e-30)).max()), 1e-30))
    moved = expmap(p, big)
    viol_big = assert_on_manifold(moved, raise_on_fail=False)
    check("expmap with |v| >> MAX_TANGENT_NORM stays on the manifold",
          viol_big < 1e-9,
          f"max|<x,x>_L + 1| = {viol_big:.3e} (KAAMMB divided by the CLAMPED "
          "norm here and left the sheet)")

    zero_step = expmap(p, torch.zeros_like(p))
    err_zero = float((zero_step - p).abs().max())
    check("expmap(p, 0) == p exactly", err_zero == 0.0,
          f"max|err| = {err_zero:.3e}")

    # 5. Parallel transport is an isometry and lands in T_q.
    u = _random_tangents(p, gen)
    w = _random_tangents(p, gen)
    pu, pw = parallel_transport(p, q, u), parallel_transport(p, q, w)
    iso = float((minkowski_dot(pu, pw) - minkowski_dot(u, w)).abs().max())
    check("parallel transport is an isometry", iso < 1e-9,
          f"max|<PT u, PT w> - <u, w>| = {iso:.3e}")
    land = assert_in_tangent(q, pu, raise_on_fail=False)
    check("parallel transport lands in T_q", land < 1e-9,
          f"max|<q, PT u>_L| = {land:.3e}")

    # The precedence bug's signature: the denominator is >= 2 for points on
    # the sheet and the buggy form turned it into ~-1e-7. Tested on nearly
    # coincident points, where the buggy version amplified by 1e11.
    q_near = project_to_hyperboloid(expmap(p, v * 1e-6), ambient=True)
    pu_n = parallel_transport(p, q_near, u)
    pw_n = parallel_transport(p, q_near, w)
    iso_n = float((minkowski_dot(pu_n, pw_n) - minkowski_dot(u, w)).abs().max())
    denom_min = float((1.0 - minkowski_dot(p, q_near)).min())
    check("transport isometry survives near-coincident points",
          iso_n < 1e-9 and denom_min > 1.999,
          f"iso err {iso_n:.3e}, min denominator {denom_min:.6f} (must be >= 2; "
          "the precedence bug made it ~-1e-7)")

    # 6. Distances. NOT `== 0`: arccosh's conditioning puts a floor of ~2e-8
    # here, so asserting exact zero would fail half the time on a correct
    # implementation. What matters is the comparison with the eps-clamped
    # form, which returned 4.5e-4.
    d_self = float(geodesic_distance(p, p).abs().max())
    check("d(p, p) == 0 to arccosh's conditioning limit", d_self < 1e-7,
          f"max d = {d_self:.3e}; acosh(clamp(.,1+1e-7)) would give 4.5e-4, "
          "and logmap's asinh route gives ~1e-16")

    d_pq = geodesic_distance(p, q)
    d_qr = geodesic_distance(q, r)
    d_pr = geodesic_distance(p, r)
    slack = float((d_pq + d_qr - d_pr).min())
    check("triangle inequality", slack > -1e-12,
          f"min slack = {slack:.3e}")

    sym = float((d_pq - geodesic_distance(q, p)).abs().max())
    check("geodesic distance is symmetric", sym < 1e-12, f"max|err| = {sym:.3e}")

    d_step = float((geodesic_distance(p, expmap(p, v))
                    - torch.sqrt(minkowski_dot(v, v))).abs().max())
    check("d(p, expmap(p, v)) == |v|", d_step < 1e-9, f"max|err| = {d_step:.3e}")

    # 7. Radius agrees with the distance to the origin.
    r_err = float((radius(p) - geodesic_distance(p, origin_like(p))).abs().max())
    check("radius(p) == d(p, origin)", r_err < 1e-10, f"max|err| = {r_err:.3e}")

    # 8. lorentz_sqdist ~ d^2/2 near zero, and monotone.
    near = expmap(p, v * 1e-3)
    d_near = geodesic_distance(p, near)
    sq = lorentz_sqdist(p, near)
    ratio = float((sq / (0.5 * d_near.pow(2)).clamp_min(1e-300) - 1.0).abs().max())
    check("lorentz_sqdist == d^2/2 to leading order", ratio < 1e-4,
          f"max relative deviation = {ratio:.3e}")

    # 9. Gradients are finite where the losses drive.
    s_leaf = (torch.randn(16, dim, generator=gen, dtype=dt)).requires_grad_(True)
    pt = project_to_hyperboloid(s_leaf)
    tgt = pt.detach().clone()                      # exact coincidence
    lorentz_sqdist(pt, tgt).sum().backward()
    g1 = s_leaf.grad
    check("lorentz_sqdist gradient finite at coincidence",
          g1 is not None and bool(torch.isfinite(g1).all()),
          "the geodesic distance squared would be NaN here")

    s_leaf2 = (torch.randn(16, dim, generator=gen, dtype=dt)).requires_grad_(True)
    pt2 = project_to_hyperboloid(s_leaf2)
    logmap(pt2, pt2.detach().clone()).sum().backward()
    g2 = s_leaf2.grad
    check("logmap gradient finite at coincidence",
          g2 is not None and bool(torch.isfinite(g2).all()),
          "0/0 in u/|u| is floored on the OPERAND, not patched on the result")

    lg = float(logmap(p, p.clone()).abs().max())
    check("logmap(p, p) == 0", lg < 1e-10,
          f"max|log_p(p)| = {lg:.3e}; the acosh route would return ~4e-9 here "
          "and the eps-clamped one ~4.5e-4")

    # 10. Frechet mean.
    single = p[0]
    repeated = single.unsqueeze(0).repeat(7, 1)
    fm = frechet_mean(repeated)
    err_fm = float((fm - single).abs().max())
    check("frechet_mean of a point with itself is that point", err_fm < 1e-14,
          f"max|err| = {err_fm:.3e}")

    cloud = _random_points(32, dim, gen, scale=0.4, dtype=dt)
    fm2 = frechet_mean(cloud)
    viol_fm = assert_on_manifold(fm2, raise_on_fail=False)
    # First-order optimality: the tangent vectors to the mean must cancel.
    resid = float(logmap(fm2.unsqueeze(0), cloud).mean(dim=0).abs().max())
    check("frechet_mean is on the manifold and first-order optimal",
          viol_fm < 1e-14 and resid < 1e-6,
          f"violation {viol_fm:.3e}, |mean tangent| {resid:.3e}")

    cen = lorentz_centroid(cloud)
    check("lorentz_centroid is on the manifold",
          assert_on_manifold(cen, raise_on_fail=False) < 1e-14,
          f"violation {assert_on_manifold(cen, raise_on_fail=False):.3e}")

    # 11. Origin maps.
    tang0 = torch.randn(16, dim, generator=gen, dtype=dt) * 0.5
    rt = logmap0(expmap0(tang0))
    err0 = float((rt - tang0).abs().max())
    check("logmap0(expmap0(v)) == v", err0 < 1e-10, f"max|err| = {err0:.3e}")

    # The ladder's composition rule must land on the manifold and must reduce
    # to expmap0 when the parent is the origin.
    origin = origin_like(p[:16])
    composed = expmap(origin, parallel_transport0(origin, tang0))
    err_comp = float((composed - expmap0(tang0)).abs().max())
    check("ladder composition at the origin reduces to expmap0",
          err_comp < 1e-10, f"max|err| = {err_comp:.3e}")

    parent = p[:16]
    child = expmap(parent, parallel_transport0(parent, tang0))
    viol_child = assert_on_manifold(child, raise_on_fail=False)
    step_len = float((geodesic_distance(parent, child)
                      - tang0.norm(dim=-1)).abs().max())
    check("ladder composition keeps the manifold and the step length",
          viol_child < 1e-12 and step_len < 1e-9,
          f"violation {viol_child:.3e}, |d - |v|| {step_len:.3e}")

    # 12. RiemannianAdam.
    target = _random_points(8, dim, gen, scale=0.5, dtype=dt)
    param = torch.nn.Parameter(_random_points(8, dim, gen, scale=0.5, dtype=dt).clone())
    mark_manifold(param)
    opt = RiemannianAdam([{"params": [param], "manifold": True,
                           "weight_decay": 0.0}], lr=0.02, max_step=1.0)
    first = None
    last = None
    # Adam's per-coordinate normalisation means the step does NOT shrink on
    # convergence: the point orbits the target at a radius of about `lr`, so
    # the residual floor is ~lr^2/2 and the threshold must sit above it.
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        loss = lorentz_sqdist(param, target).mean()
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = float(lorentz_sqdist(param, target).mean())
    viol_opt = assert_on_manifold(param.detach(), raise_on_fail=False)
    check("RiemannianAdam descends and stays on the manifold",
          last < 0.1 * max(first, 1e-30) and viol_opt < 1e-12,
          f"loss {first:.4e} -> {last:.4e} (orbit floor ~ lr^2/2 = 2e-4), "
          f"violation {viol_opt:.3e}")

    # A positive control that can actually fire: without the metric raise the
    # direction is not steepest descent. Both are normalised to the SAME
    # Minkowski length first, or the comparison would be about step size
    # rather than direction and could come out either way by accident.
    with torch.no_grad():
        pp = _random_points(8, dim, gen, scale=0.5, dtype=dt)
        tt = _random_points(8, dim, gen, scale=0.5, dtype=dt)
    pp_leaf = pp.clone().requires_grad_(True)
    lorentz_sqdist(pp_leaf, tt).sum().backward()
    with torch.no_grad():
        def _unit(d):
            nrm = torch.sqrt(minkowski_dot(d, d, keepdim=True).clamp_min(1e-30))
            return d / nrm

        good = _unit(riemannian_grad(pp_leaf, pp_leaf.grad))
        bad = _unit(project_to_tangent(pp_leaf, pp_leaf.grad))   # no raise
        differ = float((good - bad).abs().max())
        l0 = float(lorentz_sqdist(pp_leaf, tt).sum())
        step = 1e-4
        l_good = float(lorentz_sqdist(
            project_to_hyperboloid(expmap(pp_leaf, -step * good), ambient=True),
            tt).sum())
        l_bad = float(lorentz_sqdist(
            project_to_hyperboloid(expmap(pp_leaf, -step * bad), ambient=True),
            tt).sum())
    # Steepest descent: among directions of equal length, -grad_R gives the
    # largest first-order decrease. The slack absorbs the O(step^2) term.
    check("metric raise is what makes the direction steepest descent",
          l_good < l0 and l_good <= l_bad + 1e-12 and differ > 1e-6,
          f"loss {l0:.8f} -> raised {l_good:.8f} / unraised {l_bad:.8f}; "
          f"max|raised - unraised| = {differ:.3e}")

    # 13. Weight-decay guard.
    guard_fired = False
    try:
        RiemannianAdam([{"params": [torch.nn.Parameter(torch.zeros(4, dim + 1,
                                                                   dtype=dt))],
                         "manifold": True, "weight_decay": 0.01}], lr=1e-3)
    except ValueError:
        guard_fired = True
    check("weight decay on a manifold group is refused", guard_fired,
          "lambda 0.0762 for 34 epochs = factor 1.8e-5 on the point itself")

    # 14. assert_on_manifold actually catches a violation.
    caught = False
    try:
        assert_on_manifold(p * 1.1, tol=1e-6, name="deliberately broken")
    except ValueError:
        caught = True
    check("assert_on_manifold raises on an off-sheet tensor", caught,
          "positive control -- without it the falsifier cannot fire")

    sheet_caught = False
    try:
        flipped = p.clone()
        flipped[..., 0] = -flipped[..., 0]
        assert_on_manifold(flipped, tol=1e-6, name="lower sheet")
    except ValueError:
        sheet_caught = True
    check("assert_on_manifold catches the lower sheet", sheet_caught,
          "x_0 <= 0 satisfies <x,x>_L = -1 and is still nonsense")

    # 15. Tree embedding, 10 taxa.
    dist = balanced_tree_distances(10, dtype=dt)
    emb = embed_tree_distances(dist, dim=8, steps=400, lr=0.1, seed=seed,
                               euclidean_control=True)
    rep = emb.report
    check("tree embedding: fit moved off its initialisation",
          rep["loss_final"] < 0.5 * max(rep["loss_init"], 1e-30)
          and rep["distortion_avg"] < rep["distortion_init"],
          f"loss {rep['loss_init']:.4f} -> {rep['loss_final']:.4f}, "
          f"distortion {rep['distortion_init']:.4f} -> "
          f"{rep['distortion_avg']:.4f}")
    check("tree embedding: result is on the manifold",
          rep["manifold_violation"] < 1e-12,
          f"violation {rep['manifold_violation']:.3e}, "
          f"radius_max {rep['radius_max']:.3f}")
    check("tree embedding: distances correlate with the tree",
          rep["pearson"] > 0.8, f"pearson {rep['pearson']:.4f}")
    if verbose:
        results.append((
            "tree embedding: hyperbolic vs Euclidean control (REPORT ONLY)",
            True,
            f"distortion H^8 {rep['distortion_avg']:.4f} vs R^8 "
            f"{rep.get('distortion_euclidean', float('nan')):.4f}, "
            f"gain {rep.get('gain_vs_euclidean', float('nan')):+.4f}, "
            f"worst {rep['distortion_worst']:.3f}"))

    # 16. manifold_report.
    mrep = manifold_report(p)
    check("manifold_report reports the falsifiers",
          {"violation", "wrong_sheet", "radius_mean", "min_pair_d"} <= set(mrep),
          ", ".join(f"{k}={mrep[k]}" for k in
                    ("violation", "wrong_sheet", "radius_mean", "min_pair_d")
                    if k in mrep))

    failures = sum(1 for _, ok, _ in results if not ok)
    if verbose:
        width = max(len(nm) for nm, _, _ in results)
        for nm, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {nm.ljust(width)}  {detail}")
        print(f"\n{len(results) - failures}/{len(results)} checks passed "
              f"({failures} failure{'s' if failures != 1 else ''}).")
    return failures


if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    sys.exit(self_test(verbose=True))
