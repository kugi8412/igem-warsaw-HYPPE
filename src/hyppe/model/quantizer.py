"""Quantisers for the HYPPE ladder: one hyperbolic codebook, N scalar grids.

The ladder quantises two kinds of object. Level 0, the coarsest map, takes a
POSITION -- a point on the hyperboloid -- through `HyperbolicCodebook`; levels
>= 1 take a REFINEMENT, a tangent vector at the origin, through `ScalarFSQ`.
The tangent space at the origin is Euclidean by definition, so a scalar grid
there is correct typing and not a compromise: what is hyperbolic is the
composition ``p_{k+1} = expmap(p_k, PT(origin -> p_k, v_{k+1}))``, which
`ladder.py` owns and not the individual step.

Strides [2,2,2,2,2] over 800 bp give maps 400/200/100/50/25 and the budget
this package is sized against, from the coarsest level:

    level  length  quantiser            alphabet  bits/pos  bits/seq
    L0       25    HyperbolicCodebook       128     7.0000     175.0
    L1       50    ScalarFSQ (3,)*6         729     9.5098     475.5
    L2      100    ScalarFSQ (3,)*4          81     6.3399     634.0
    L3      200    ScalarFSQ (3,)*2           9     3.1699     634.0
    L4      400    ScalarFSQ (3,)*2           9     3.1699    1268.0
                                                            --------
                                                              3186.5

against a measured 1598.7 bits of sequence (unigram entropy 1.9983 b/bp over
800 bp), i.e. 1.99x headroom; the reconstruction floor is the unigram argmax
0.2626, not 0.25. With that much headroom a failure to reconstruct is a
failure of OPTIMISATION -- straight-through, codebook collapse, a saturated
grid -- not of a narrow bottleneck, and `report()` tells the two apart.

Every calibrating measurement is taken in the MODE, on the DATA and at the
BATCH SIZE the run will use; `ScalarFSQ.calibrate_gain` enforces that and
stamps the mode into `report()`. The input gain was calibrated wrongly four
times by ignoring it, and a constant is not the fix anyway, since the defect
behind those failures is DRIFT of the encoder scale.

What the consumer (`ladder.py`) must know:

* `HyperbolicCodebook.forward` returns the SPATIAL PART of the code point,
  unscaled, so ``project_to_hyperboloid(q.transpose(1, 2))`` recovers the
  manifold point exactly; a scale on the way out would make that inverse
  depend on a knob. The point also comes back in `stats["points"]`.
* `ScalarFSQ.forward` returns a tangent vector at the origin in ambient
  spatial coordinates ``[B, dim, L]``, with ``||v|| <= aperture`` exactly --
  a theorem about the construction, not the effect of a clamp.
* The zero code means "no refinement". An `expmap` that clamps the tangent
  norm from below, e.g. ``sqrt(<v,v>.clamp_min(1e-7))``, returns
  ``cosh(3.16e-4) * x``: a point off the manifold displaced radially OUTWARD,
  so "no refinement" becomes a systematic outward drift at every level. The
  composition must use the ``sinh(n)/n -> 1`` limit; this module only reports
  `zero_frac`, so the exposure is never a guess.
* `decode_indices` is a pure function of the indices on both classes, and
  `self_test()` requires agreement with the forward pass of exactly
  0.000000e+00. That is the diffusion contract.

Neither class promises better accuracy; their falsifiers are reconstruction
and entropy, and the first threshold is reconstruction from codes alone.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..geometry.lorentz import (
    DEFAULT_NO_DECAY_PATTERNS,
    MAX_TANGENT_NORM,
    assert_on_manifold,
    expmap,
    frechet_mean,
    geodesic_distance,
    is_manifold_param,
    logmap,
    lorentz_sqdist,
    lorentz_sqdist_exact,
    mark_manifold,
    minkowski_dot,
    project_to_hyperboloid,
    radius,
)

_LOG2 = math.log(2.0)

#: Weight-decay exemption patterns, re-exported so the naming contract is
#: checkable rather than remembered. Manifold tensors are protected by the
#: explicit `mark_manifold` flag instead, because substring matching once lost
#: a parameter to word order. `ScalarFSQ` has no manifold tensors, so its
#: operating point relies on this list -- hence the name `scale_in` -- and
#: `ScalarFSQ.self_test` asserts the match rather than trusting it.
NO_DECAY_SUBSTRINGS: Tuple[str, ...] = tuple(DEFAULT_NO_DECAY_PATTERNS)


# Grid selection
def grid_bits(levels: Sequence[int]) -> float:
    """Declared budget of a scalar grid, in bits per position."""
    return sum(math.log(int(l), 2.0) for l in levels)


def grid_size(levels: Sequence[int]) -> int:
    """Number of cells in the product grid."""
    size = 1
    for l in levels:
        size *= int(l)
    return size


def levels_for_bits(
    bits: float,
    bases: Sequence[int] = (3, 5, 7),
    tolerance: float = 0.6,
    max_dims: int = 12,
    pair_halves: bool = True,
) -> Tuple[int, ...]:
    """Pick a scalar grid whose declared budget is about ``bits`` per position.

    Bases must be ODD: an even base gives an asymmetric grid under the
    half-offset convention (for L=4 the values are {-1, -0.5, 0, +0.5}, a
    systematic drift on a level whose job is correction), L=2 makes the
    bounding function evaluate ``atanh(0.5 / 0.4995)`` and return NaN, and
    only an odd base puts a bucket exactly on ZERO. Inside the tolerance more
    DIMENSIONS win over finer resolution at equal budget, since what is scarce
    is directions in the tangent space; outside it the two criteria swap
    order, because with dimension count still first a target of 7 bits
    returned ``(3,)*12 = 19.02`` bits and still looked like a working selector.

    ``pair_halves`` searches over half grids and mirrors them, forcing an even
    dimension count with ``levels[i] == levels[i + d/2]``. The constraint came
    from reverse-complement equivariance, which HYPPE does not have, and is
    the default only because the project's budget arithmetic is stated in the
    grids it produces: 3 bits -> (3, 3), 6 -> (3,)*4, 8 -> (5, 3, 5, 3),
    9 -> (3,)*6. ``pair_halves=False`` runs the unconstrained search and gives
    a different alphabet -- 8 bits becomes ``(3,)*5 = 243`` cells, not 225 --
    so flipping the flag is never a silent change of budget.
    """
    target = float(bits)
    bases = tuple(sorted(int(b) for b in bases))
    if any(b % 2 == 0 or b < 3 for b in bases):
        raise ValueError(
            "grid bases must be ODD and >= 3 (symmetric grid containing the "
            "zero code): %r" % (bases,)
        )
    if max_dims < 1:
        raise ValueError("max_dims must be >= 1")
    limit = max_dims // 2 if pair_halves else max_dims
    if limit < 1:
        raise ValueError("max_dims must be >= 2 when pair_halves is set")
    factor = 2.0 if pair_halves else 1.0

    candidates: List[Tuple[Tuple[int, ...], float]] = []

    def expand(prefix: List[int], start: int) -> None:
        if prefix:
            candidates.append(
                (tuple(prefix), factor * sum(math.log(l, 2.0) for l in prefix))
            )
        if len(prefix) >= limit:
            return
        for i in range(start, len(bases)):
            expand(prefix + [bases[i]], i)

    expand([], 0)
    near = [(p, b) for p, b in candidates if abs(b - target) <= tolerance]
    if near:
        chosen, _ = min(
            near, key=lambda pb: (-len(pb[0]), abs(pb[1] - target), max(pb[0]))
        )
    else:
        chosen, _ = min(
            candidates, key=lambda pb: (abs(pb[1] - target), -len(pb[0]), max(pb[0]))
        )
    ordered = tuple(sorted(chosen, reverse=True))
    return ordered + ordered if pair_halves else ordered


# Shared helpers
def _manifold_error(x: torch.Tensor) -> float:
    """``max |<x,x>_L + 1|`` -- the structural test that a point is on H^n.

    Computed here rather than through `assert_on_manifold` because a report
    must never raise: one exception inside a diagnostic line loses every
    falsifier on that line, not just this one. `assert_on_manifold` is used in
    `self_test`, where raising is the point.
    """
    with torch.no_grad():
        inner = minkowski_dot(x, x, keepdim=False)
        return float((inner + 1.0).abs().max())


def _cell_entropy(indices: torch.Tensor, size: int) -> Dict[str, float]:
    """Empirical code entropy PER CELL, from one batch of ``[B, L]`` indices.

    Per cell, not over the flattened tensor: a latent emitting one fixed code
    per position has flattened entropy log2(L) and looks healthy, while its
    per-cell entropy is exactly 0. The ceiling is the BATCH, not the alphabet,
    so `entropy_ceiling_bits` is returned alongside -- at batch 32 comparing
    against ``log2(128) = 7`` invents a deficit.

    `mode_mass` is the mean probability of the most frequent code per cell,
    and 1.0 means the level has become a learned positional table. A collapse
    alarm reading EMA counts cannot see that state, because a dead-code
    restart writes those counts to 1.0 on revival.
    """
    if indices.dim() != 2:
        return {}
    idx = indices.detach()
    b, length = int(idx.shape[0]), int(idx.shape[1])
    if b < 1 or length < 1:
        return {}
    hist = torch.zeros(length, size, device=idx.device, dtype=torch.float32)
    hist.scatter_add_(
        1,
        idx.t().long(),
        torch.ones(length, b, device=idx.device, dtype=torch.float32),
    )
    p = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    # Mask the OPERAND, never the result: `torch.where` on ``p * log(p)``
    # evaluates the unselected branch anyway, gets NaN at p = 0 and multiplies
    # it by the zero of the mask, and 0 * NaN = NaN.
    ent = -(p * torch.log2(p.clamp_min(1e-12))).sum(dim=1)
    alive = hist.sum(dim=0) > 0
    return {
        "entropy_bits": round(float(ent.mean()), 4),
        "entropy_bits_min": round(float(ent.min()), 4),
        "entropy_ceiling_bits": round(math.log(max(b, 2), 2.0), 4),
        "mode_mass": round(float(p.max(dim=1).values.mean()), 4),
        "codes_in_batch": int(alive.sum()),
    }


def _finding(
    name: str,
    value: Optional[float],
    threshold: float,
    rule: str,
    can_fire: bool,
    why: str,
) -> Dict[str, object]:
    """One falsifier, with an explicit statement of whether it CAN fire.

    A falsifier that cannot fire is worse than no falsifier, because its
    passing value reads as health: a quantity bounded below its own threshold
    by construction reports "ok" for ever. `can_fire` states the
    answerability of the question next to its answer.
    """
    if value is None:
        verdict = "no-measurement"
    elif not can_fire:
        verdict = "unanswerable"
    elif rule == ">=":
        verdict = "ok" if value >= threshold else "ALARM"
    elif rule == "<=":
        verdict = "ok" if value <= threshold else "ALARM"
    elif rule == "==":
        verdict = "ok" if value == threshold else "ALARM"
    else:
        raise ValueError("unknown rule %r" % (rule,))
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "rule": rule,
        "can_fire": bool(can_fire),
        "verdict": verdict,
        "why": why,
    }


class no_codebook_update:
    """Context manager: suspend every quantiser statistic under ``modules``.

    An extra forward pass in training mode -- negatives, an equivariance
    control, a probe -- rewrites the codebook with data that is not the
    reconstruction target; measured cost when this went unnoticed was
    reconstruction 0.79 -> 0.62. The criterion is the `vq_update` FLAG and
    not a class signature: an earlier version matched on method names and
    silently found neither of the two quantisers that then existed.
    """

    def __init__(self, *modules: nn.Module):
        self._modules = modules
        self._saved: List[Tuple[nn.Module, bool]] = []

    def __enter__(self) -> "no_codebook_update":
        for root in self._modules:
            for m in root.modules():
                if hasattr(m, "vq_update"):
                    self._saved.append((m, bool(m.vq_update)))
                    m.vq_update = False
        return self

    def __exit__(self, *exc) -> bool:
        for m, previous in self._saved:
            m.vq_update = previous
        self._saved = []
        return False


def reset_statistics(module: nn.Module) -> int:
    """Clear usage statistics on every quantiser under ``module``.

    Call before an evaluation pass: the histograms are exponential averages
    over about 100 steps, so without a reset an evaluation reports a blend of
    the validation distribution and the tail of training.
    """
    count = 0
    for m in module.modules():
        fn = getattr(m, "reset_statistics", None)
        if callable(fn) and m is not module:
            fn()
            count += 1
    return count


def collect_manifold_state_names(module: nn.Module, prefix: str = "") -> List[str]:
    """Fully qualified names of every tensor that lives ON the hyperboloid.

    Exists so a weight averager can skip them structurally. Averaging the
    ambient coordinates of manifold points took a codebook off the sheet by
    ``max|<x,x>_L + 1|`` up to 2.49e-01 against a tolerance of 1e-16, and a
    reported reconstruction of 0.9945 read back from the checkpoint as 0.8012.
    """
    names: List[str] = []
    for path, m in module.named_modules(prefix=prefix):
        fn = getattr(m, "manifold_state_names", None)
        if callable(fn):
            for local in fn():
                names.append("%s.%s" % (path, local) if path else local)
    return names


# Level 0: the hyperbolic codebook
class HyperbolicCodebook(nn.Module):
    """VQ whose codebook lives on the hyperboloid; geodesic EMA, live restarts.

    Assignment is by geodesic distance, the update is a Frechet-type mean of
    the assigned encoder outputs, and the codebook is either an EMA buffer --
    the reference behaviour -- or an ``nn.Parameter`` optimised by
    `RiemannianAdam`, which is the only way the reconstruction loss can reach
    the code entries at all.

    Plain straight-through returns ``points + (quant - points).detach()``,
    whose gradient goes only to the encoder, so the codebook sees nothing but
    the commitment term pulling it toward the encoder's shell; the measured
    consequence was a spread of code radii collapsing to sd 0.0002. The point
    returned here is ``codebook[idx] + (st - st.detach())``, whose value is
    exactly ``codebook[idx]`` (the second term is a tensor minus itself) and
    whose gradient flows both ways; with a buffer codebook the first term
    carries no gradient, so the formula is uniform across modes.

    ``radius_scale`` shrinks the lift at initialisation, because points far
    from the origin sit where the manifold expands fastest and floating point
    loses precision. dtype is float64 by default for the same reason: a
    projection computed in float32 leaves ``max|<x,x>_L + 1|`` of order 1e-6,
    and a later ``.double()`` does not recover the lost bits.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 128,
        decay: float = 0.99,
        commit_weight: float = 0.25,
        codebook_weight: float = 1.0,
        radius_scale: float = 0.3,
        max_radius: float = MAX_TANGENT_NORM,
        commit_measure: str = "chord",
        lift_norm: str = "none",
        lift_squash: float = 0.0,
        lift_norm_target: float = 1.1752011936438014,
        eps: float = 1e-5,
        dead_threshold: float = 0.5,
        learnable: bool = False,
        geodesic_target: bool = True,
        dead_restart: bool = True,
        diagnostics: bool = True,
        dtype: torch.dtype = torch.float64,
        seed: int = 7898,
    ):
        super().__init__()
        if int(dim) < 1:
            raise ValueError("dim must be >= 1")
        if int(codebook_size) < 2:
            raise ValueError("codebook_size must be >= 2")
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.decay = float(decay)
        self.commit_weight = float(commit_weight)
        self.codebook_weight = float(codebook_weight)
        self.radius_scale = float(radius_scale)
        # The lift is the one place a point reaches the manifold without a
        # length check -- `expmap` truncates at MAX_TANGENT_NORM = 8.0, past
        # which <x,x>_L = -1 stops surviving the subtraction in float64, while
        # `lift` solves the constraint and returns. Without this bound the
        # encoder ran to radius 41.33 with `manifold_error` 9.44e21 against
        # the 1e-16 `assert_on_manifold` accepts, the commitment surrogate
        # taking 99.998 % of the loss. 0.0 restores the unbounded lift.
        self.max_radius = float(max_radius)
        # Which function `commit_weight` multiplies: 'chord' is cosh(d) - 1,
        # 'exact' is d^2 / 2, and returning d^2/2 rather than d^2 keeps the
        # switch from silently doubling the weight. Falsifier of the 'exact'
        # arm: its AMBIENT gradient decays like exp(-d), 9.6e-17 at d = 41.3
        # against a constant 1.0 for 'chord', so if the commitment is what
        # holds the encoder in, 'exact' removes it where it is needed.
        if str(commit_measure) not in ("chord", "exact"):
            raise ValueError("commit_measure must be 'chord' or 'exact',"
                             " got %r" % (commit_measure,))
        self.commit_measure = str(commit_measure)
        # Where the scale of the encoder map is pinned. 'scale' sets the mean
        # spatial norm per SEQUENCE, keeping relative radii within it and
        # removing the global drift that ran to enc_norm_mean 1.846e18; 'rms'
        # normalises per POSITION, putting every point on one shell and so
        # stopping the radius being a coordinate at level 0. Neither reduces
        # across the batch, so train and eval compute the same function.
        if str(lift_norm) not in ("none", "scale", "rms"):
            raise ValueError("lift_norm must be 'none', 'scale' or 'rms',"
                             " got %r" % (lift_norm,))
        self.lift_norm = str(lift_norm)
        # sinh(1.0), so a mean NORM of that is a mean RADIUS of 1.0 -- where
        # the healthy part of the reference run sat, not a round number.
        self.lift_norm_target = float(lift_norm_target)
        # Soft radial squash; 0.0 = off and is the default. A hard clip at the
        # same threshold bound on 2.30 % of positions against a 1.00 %
        # ceiling, and a clamp that binds often removes the radial gradient
        # and blinds every post-clamp observable at once.
        if float(lift_squash) < 0.0:
            raise ValueError("lift_squash must be >= 0 (0 = off)")
        self.lift_squash = float(lift_squash)
        self.eps = float(eps)
        self.dead_threshold = float(dead_threshold)
        self.learnable = bool(learnable)
        self.geodesic_target = bool(geodesic_target)
        self.dead_restart = bool(dead_restart)
        self.diagnostics = bool(diagnostics)

        generator = torch.Generator().manual_seed(int(seed))
        spatial = (
            torch.randn(self.codebook_size, self.dim, generator=generator).to(dtype)
            * self.radius_scale
        )
        codebook = project_to_hyperboloid(spatial)
        if self.learnable:
            # Only as a Parameter does `RiemannianAdam` have anything to
            # optimise; as a buffer the Riemannian flag is a silent no-op.
            # `mark_manifold` is what `manifold_param_groups` sorts by --
            # unflagged, this tensor lands in the DECAYED group, where decay
            # destroys the constraint rather than shrinking the value.
            self.codebook = mark_manifold(nn.Parameter(codebook))
        else:
            self.register_buffer("codebook", codebook)
        self.register_buffer(
            "cluster_size", torch.zeros(self.codebook_size, dtype=dtype)
        )

        #: Respected by `no_codebook_update`.
        self.vq_update = True
        self.loss: Optional[torch.Tensor] = None
        self.indices: Optional[torch.Tensor] = None
        self._restarts_total = 0
        self._restarts_last = 0
        self._steps = 0
        self._last_indices: Optional[torch.Tensor] = None
        self._commit_raw: Optional[float] = None
        self._enc_radius: Optional[Tuple[float, float]] = None
        #: Packed pre/post-clamp encoder scale, filled by `lift` on EVERY
        #: forward (not behind `diagnostics`). See `runaway_report`.
        self._runaway: Optional[torch.Tensor] = None

    # Contracts
    def manifold_parameters(self) -> List[nn.Parameter]:
        """Parameters on the hyperboloid -- for `RiemannianAdam`, and the
        authoritative weight-decay exemption for the manifold half."""
        return [self.codebook] if self.learnable else []

    def euclidean_parameters(self) -> List[nn.Parameter]:
        """Everything else. Empty here; kept so callers can treat all
        quantisers alike instead of special-casing on class."""
        manifold = {id(p) for p in self.manifold_parameters()}
        return [p for p in self.parameters() if id(p) not in manifold]

    def manifold_state_names(self) -> List[str]:
        """State tensors that must be excluded from ambient averaging."""
        return ["codebook"]

    # Pieces
    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean encoder map ``[B, D, L]`` -> points ``[B, L, D+1]``.

        Solving the constraint for the time coordinate lands on the sheet by
        construction, where normalising a free (D+1)-vector leaves points
        drifting off whenever the optimiser takes a large step.
        """
        if x.dim() != 3 or x.shape[1] != self.dim:
            raise ValueError(
                "expected x of shape [B, %d, L], got %s" % (self.dim, tuple(x.shape))
            )
        z = x.transpose(1, 2).to(self.codebook.dtype) * self.radius_scale
        # Taken before the clamp, and unconditionally. `max_radius` bounds the
        # radius by construction, so a gate reading the post-clamp norm would
        # report a healthy 8.0 for ever while the encoder map ran to 1.8e18;
        # the runaway is visible only here. Not behind `self.diagnostics`
        # either, because a guard a flag can switch off will be off in the run
        # that needs it.
        n = z.norm(dim=-1, keepdim=True).detach()
        if self.lift_norm != "none":
            # NOT the detached `n` above: the divisor has to carry a gradient,
            # or the encoder keeps a free scale to inflate and the
            # normalisation is cosmetic.
            div = z.norm(dim=-1, keepdim=True)
            if self.lift_norm == "scale":
                div = div.mean(dim=1, keepdim=True)
            z = z * (self.lift_norm_target / div.clamp_min(1e-12))
        if self.lift_squash > 0.0:
            # r' = c tanh(r/c) in RADIUS, then back to a norm through sinh.
            # asinh/sinh rather than log/exp because |spatial| = sinh(r)
            # exactly and asinh is accurate at both ends.
            c = self.lift_squash
            nn_ = z.norm(dim=-1, keepdim=True)
            r = torch.asinh(nn_)
            r2 = c * torch.tanh(r / c)
            z = z * (torch.sinh(r2) / nn_.clamp_min(1e-30))
        # Recomputed so the clamp reads the tensor it actually clamps.
        # With `lift_norm == 'none'` this is `n` bit for bit.
        n_post = z.norm(dim=-1, keepdim=True).detach()
        if self.max_radius > 0.0:
            # |spatial| = sinh(r), so a radius bound is a bound on the norm.
            # Direction kept, length clipped: below the bound this is the
            # identity, above it the radial gradient goes and the tangential
            # one does not.
            cap = math.sinh(self.max_radius)
            clipped = n_post.clamp(max=cap)
            frac = (n_post > cap).to(z.dtype).mean()
            z = z * (cap / n_post.clamp_min(1e-30)).clamp(max=1.0)
        else:
            clipped = n_post
            frac = torch.zeros((), dtype=z.dtype, device=z.device)
        self._runaway = torch.stack([
            n.mean(), n.max(),                      # pre-clamp, the runaway
            torch.asinh(n).mean(), torch.asinh(n.max()),
            torch.asinh(clipped).mean(),            # post-clamp, the geometry
            torch.asinh(clipped.max()),
            frac.reshape(()),
        ])
        return project_to_hyperboloid(z)

    def runaway_report(self) -> Dict[str, float]:
        """Pre-clamp scale of the encoder map. One device synchronisation.

        Empty before the first forward, never a fabricated zero: a gate
        comparing 0.0 against a ceiling passes, and "no measurement" must not
        read as "measured and fine".

        `enc_radius_raw_*` is PRE-clamp and is the quantity to gate on;
        `enc_radius_*` is POST-clamp and is what the geometry sees. The two
        are equal exactly while `enc_clip_frac` is 0, and their separation is
        the alarm.
        """
        if self._runaway is None:
            return {}
        p = self._runaway.double().tolist()
        return {"enc_norm_mean": p[0], "enc_norm_max": p[1],
                "enc_radius_raw_mean": p[2], "enc_radius_raw_max": p[3],
                "enc_radius_mean": p[4], "enc_radius_max": p[5],
                "enc_clip_frac": p[6]}

    def quantise(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest code by geodesic distance, as a single GEMM.

        The Minkowski product is bilinear, so flipping the sign of the time
        coordinate turns the broadcast ``[N, K, D+1]`` -- 105 MB per level at
        batch 32, which autograd would keep -- into ``flat @ codebook.T``.
        The assignment is unchanged and not merely close, since
        ``d = arccosh(-<x,y>_L)`` is monotone in ``-<x,y>_L``; skipping the
        arccosh also skips its precision loss near zero distance, which is
        where a commitment loss drives every assigned point.
        """
        flat = points.reshape(-1, points.shape[-1])
        flipped = torch.cat([-flat[..., :1], flat[..., 1:]], dim=-1)
        inner = flipped @ self.codebook.t()
        idx = inner.argmax(dim=1)
        return self.codebook[idx], idx

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` indices -> ``[B, D, L]`` spatial part of the code points.

        Pure function of the indices: no encoder tensor, no batch statistic.
        ``project_to_hyperboloid(out.transpose(1, 2))`` returns the manifold
        point bit for bit, because that is how the entry was built.
        """
        codes = self.codebook[indices.long()]
        return codes[..., 1:].transpose(-1, -2)

    def decode_indices_points(self, indices: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` indices -> ``[B, L, D+1]`` points on the hyperboloid."""
        return self.codebook[indices.long()]

    # Forward
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """``x`` ``[B, D, L]`` -> ``(quantised [B, D, L], indices [B, L], stats)``.

        ``quantised`` is the SPATIAL PART of the code point in the codebook
        dtype, unscaled. ``stats`` holds tensors only: each ``.item()`` forces
        a device synchronisation, and diagnostics have measured at 70 % of the
        step time. `report()` converts, once per logging interval.
        """
        batch, _, length = x.shape
        points = self.lift(x)
        quant_flat, idx_flat = self.quantise(points)
        quant = quant_flat.view(batch, length, -1)
        idx = idx_flat.view(batch, length)

        if self.training and self.vq_update:
            self._update(points.detach().reshape(-1, points.shape[-1]), idx_flat)

        # Commitment uses the SMOOTH squared-distance surrogate, never d**2:
        # the geodesic distance has infinite slope at zero, which is exactly
        # where a commitment loss pushes, so d**2 returns NaN on convergence.
        sqd = (lorentz_sqdist_exact if self.commit_measure == "exact"
               else lorentz_sqdist)
        if self.learnable:
            # Two detached terms so each side carries its own weight; a single
            # term would move encoder and codes toward each other under one
            # coefficient.
            commit = sqd(points, quant.detach()).mean()
            codebook_term = sqd(points.detach(), quant).mean()
        else:
            commit = sqd(points, quant).mean()
            codebook_term = torch.zeros((), device=x.device, dtype=commit.dtype)

        straight = points + (quant - points).detach()
        # Value is `codebook[idx]` exactly; gradient reaches BOTH the encoder
        # (through `straight`) and the code entries (through the gather).
        out_points = self.codebook[idx] + (straight - straight.detach())

        loss = self.commit_weight * commit + self.codebook_weight * codebook_term
        self.loss = loss
        self.indices = idx

        if self.diagnostics:
            with torch.no_grad():
                self._last_indices = idx.detach()
                self._commit_raw = commit.detach()
                enc_r = radius(points.detach())
                self._enc_radius = (enc_r.mean(), enc_r.std())

        stats = {
            "loss": loss,
            "commit": commit.detach(),
            "codebook_loss": codebook_term.detach(),
            # The same point the caller would obtain by re-projecting
            # `quantised`; handed back to save a sqrt, not as new information.
            "points": out_points,
            "encoder_points": points.detach(),
        }
        return out_points[..., 1:].transpose(1, 2), idx, stats

    # Update
    @torch.no_grad()
    def _update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        """Geodesic EMA of the codebook plus the dead-code restart.

        The target must be geodesic. Moving toward an ARITHMETIC mean of
        ambient coordinates and projecting is an estimator whose bias grows
        with radius -- 0.2281 / 1.3309 / 2.2894 from the true Frechet mean at
        radii 1.42 / 2.25 / 2.92 -- so the worst-hit level is the outermost
        one, which holds the most codes. Averaging in the tangent space at
        the current code is one Karcher iteration, and since
        ``logmap(e, expmap(e, s)) = s`` a move of ``(1 - decay)`` along the
        mean tangent is the Riemannian EMA. This is `frechet_mean` with
        ``iters=1`` evaluated by scatter, which is O(N) where calling it
        directly would materialise ``[K, N, D]``; `frechet_drift()` measures
        the residual rather than assuming it.
        """
        onehot = torch.zeros(
            flat.shape[0], self.codebook_size, device=flat.device, dtype=flat.dtype
        )
        onehot.scatter_(1, idx.unsqueeze(1), 1.0)
        counts = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)

        used = counts > 0
        if used.any() and not self.learnable:
            if self.geodesic_target:
                tangents = logmap(self.codebook[idx], flat)
                step = (onehot.t() @ tangents)[used] / counts[used].unsqueeze(1)
                moved = expmap(self.codebook[used], (1.0 - self.decay) * step)
            else:
                # Ablation only, kept so the cost of the wrong target can be
                # re-measured rather than cited.
                summed = onehot.t() @ flat
                target = project_to_hyperboloid(
                    summed[used][..., 1:] / counts[used].unsqueeze(1)
                )
                tangent = logmap(self.codebook[used], target)
                moved = expmap(self.codebook[used], (1.0 - self.decay) * tangent)
            # Re-project: the closed forms land on the sheet analytically but
            # drift in floating point, and the drift accumulates across steps.
            self.codebook[used] = project_to_hyperboloid(moved, ambient=True)

        self._restarts_last = 0
        if not self.dead_restart:
            self._steps += 1
            return

        # Dead-code restart. Without it a codebook can only lose capacity: a
        # code that stops winning receives neither gradient nor EMA update, so
        # collapse is ABSORBING rather than merely likely. Measured with the
        # restart disabled: used-bit fraction 0.2970 in epoch 1 and
        # reconstruction 0.8227 against 0.9952. It must therefore also run
        # when the codebook is a Parameter, which needs `.data`.
        dead = self.cluster_size < self.dead_threshold
        if dead.any() and flat.shape[0] > 0:
            n_dead = int(dead.sum())
            pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            revived = project_to_hyperboloid(flat[pick], ambient=True)
            if isinstance(self.codebook, nn.Parameter):
                self.codebook.data[dead] = revived
            else:
                self.codebook[dead] = revived
            self.cluster_size[dead] = 1.0
            self._restarts_total += n_dead
            self._restarts_last = n_dead
        self._steps += 1

    @torch.no_grad()
    def renormalise_radius(self, target: Optional[float] = None) -> float:
        """Rescale code radii back to ``target`` (default: their own mean).

        Separates the DIRECTION on the hyperboloid, which carries the content
        of a code and must be learned, from the RADIUS, which encodes the
        level and should be stable. Left free the radius is a gauge: measured
        elsewhere, radii 1.4235 / 2.2472 / 2.9279 fell to 0.3335 / 0.5542 /
        1.6233 within one epoch and then inverted the order of the hierarchy.

        Optional and not called automatically, because on this ladder the
        radial spread comes from the cone composition in `ladder.py` and
        pinning it here would make `radius_mean` a readout of a knob.
        """
        spatial = self.codebook[..., 1:]
        norm = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        r = torch.asinh(norm)
        mean_r = float(r.mean())
        if mean_r < 1e-9:
            return 0.0
        goal = mean_r if target is None else float(target)
        factor = goal / mean_r
        rescaled = spatial * (torch.sinh(r * factor) / norm)
        new = project_to_hyperboloid(rescaled)
        if isinstance(self.codebook, nn.Parameter):
            self.codebook.data.copy_(new)
        else:
            self.codebook.copy_(new)
        return factor

    @torch.no_grad()
    def reset_statistics(self) -> None:
        """Forget the restart counters and the cached batch. Does NOT touch
        `cluster_size`, which is codebook STATE and not a statistic."""
        self._restarts_total = 0
        self._restarts_last = 0
        self._steps = 0
        self._last_indices = None
        self._commit_raw = None
        self._enc_radius = None

    # Falsifiers
    @torch.no_grad()
    def report(self) -> Dict[str, object]:
        """Everything that can show this codebook is dead or broken.

        Never raises: with no forward pass behind it the batch-dependent keys
        are simply absent, because a diagnostic that throws loses the whole
        log line.

        used_frac         codes above the dead threshold. READ IT ONLY NEXT TO
                          `entropy_bits` -- the restart writes
                          ``cluster_size = 1.0`` on revival, so a code revived
                          and never chosen again counts as alive; measured
                          case, ``used_frac ~ 1.0`` with 16 of 128 selected.
        entropy_bits      per-cell entropy of the last batch, the honest
                          budget falsifier; ceiling `log2(batch)`, alongside.
        mode_mass         1.0 = the level became a positional lookup table.
        dead_restarts     cumulative; a high steady rate late means thrashing.
        radius_sd         stratification of the codes; flat means the
                          curvature is unused.
        manifold_error    ``max|<x,x>_L + 1|``, structural, tolerance 1e-16.
        dist_ratio        encoder-to-code distance over median code-to-code
                          distance; >= 1 means assignment no longer follows
                          direction.
        """
        out: Dict[str, object] = {
            "codebook_size": self.codebook_size,
            "dim": self.dim,
            "learnable": self.learnable,
            "geodesic_target": self.geodesic_target,
            "dead_restart": self.dead_restart,
            "dead_restarts": int(self._restarts_total),
            "dead_restarts_last": int(self._restarts_last),
            "steps": int(self._steps),
        }
        codes = self.codebook.detach()
        mass = self.cluster_size
        total = float(mass.sum())
        if total > 0.0:
            p = (mass / mass.sum()).double()
            live = p[p > 0]
            out["perplexity"] = round(float(torch.exp(-(live * live.log()).sum())), 3)
            out["used_codes"] = int((mass >= self.dead_threshold).sum())
            out["used_frac"] = round(out["used_codes"] / self.codebook_size, 4)
        r = radius(codes)
        out["max_radius"] = self.max_radius
        out["commit_measure"] = self.commit_measure
        out["lift_norm"] = self.lift_norm
        out["lift_squash"] = self.lift_squash
        out["radius_mean"] = round(float(r.mean()), 4)
        out["radius_sd"] = round(float(r.std()), 4)
        out["manifold_error"] = float(_manifold_error(codes))
        # The hyperboloid has TWO sheets and an average of ambient coordinates
        # can land on the lower one, where every distance is meaningless and
        # nothing raises. Must stay 0.
        out["wrong_sheet"] = int((codes[..., 0] <= 0).sum())
        # Deterministic subsample. `randperm` would advance the global RNG
        # stream, and a report has no business touching the forward pass.
        stride = max(1, self.codebook_size // 128)
        sample = codes[::stride][:128]
        if sample.shape[0] > 1:
            dmat = geodesic_distance(sample.unsqueeze(1), sample.unsqueeze(0))
            off = ~torch.eye(sample.shape[0], dtype=torch.bool, device=dmat.device)
            code_dist = float(dmat[off].median())
            out["code_dist_median"] = round(code_dist, 4)
            if self._commit_raw is not None:
                enc_dist = math.sqrt(max(2.0 * float(self._commit_raw), 0.0))
                out["enc_to_code_dist"] = round(enc_dist, 4)
                out["dist_ratio"] = round(enc_dist / max(code_dist, 1e-9), 4)
        if self._enc_radius is not None:
            out["enc_radius_mean"] = round(float(self._enc_radius[0]), 4)
            out["enc_radius_sd"] = round(float(self._enc_radius[1]), 4)
        if self._last_indices is not None:
            out.update(_cell_entropy(self._last_indices, self.codebook_size))
        return out

    def falsifiers(self) -> List[Dict[str, object]]:
        """Explicit verdicts, each with whether the question can be answered."""
        r = self.report()
        found: List[Dict[str, object]] = []
        entropy = r.get("entropy_bits")
        ceiling = r.get("entropy_ceiling_bits")
        declared = math.log(self.codebook_size, 2.0)
        if entropy is not None and ceiling is not None:
            usable = min(float(ceiling), declared)
            found.append(
                _finding(
                    "entropy_fraction",
                    round(float(entropy) / max(usable, 1e-9), 4),
                    0.50,
                    ">=",
                    # Unanswerable when the batch cannot express the alphabet:
                    # at batch 32 the ceiling is 5 bits of the 7 declared.
                    float(ceiling) >= 0.5 * declared,
                    "per-cell code entropy against min(log2(batch), log2(K)); "
                    "low means the level is turning into a positional table",
                )
            )
        mode = r.get("mode_mass")
        if mode is not None:
            found.append(
                _finding(
                    "mode_mass",
                    float(mode),
                    0.90,
                    "<=",
                    True,
                    "mean probability of the most frequent code per cell; "
                    "1.0 is a learned positional lookup table",
                )
            )
        found.append(
            _finding(
                "manifold_error",
                float(r["manifold_error"]),
                1e-9 if self.codebook.dtype == torch.float64 else 1e-5,
                "<=",
                True,
                "max|<x,x>_L + 1|; ambient averaging of manifold points was "
                "measured at 5.05e-02 ... 2.49e-01 while reporting health",
            )
        )
        ratio = r.get("dist_ratio")
        found.append(
            _finding(
                "dist_ratio",
                None if ratio is None else float(ratio),
                1.0,
                "<=",
                True,
                "encoder-to-code distance over median code-to-code distance; "
                ">= 1 means assignment no longer follows direction",
            )
        )
        sd = r.get("radius_sd")
        found.append(
            _finding(
                "radius_sd",
                None if sd is None else float(sd),
                1e-3,
                ">=",
                # If something pins the radius, this question is not askable.
                True,
                "spread of code radii; collapse to ~0 (measured 0.0002 and "
                "0.0023) means the codebook has no route to learn a radius",
            )
        )
        return found

    @torch.no_grad()
    def frechet_drift(
        self, points: torch.Tensor, indices: torch.Tensor, max_codes: int = 32
    ) -> Dict[str, float]:
        """Distance from a code to the TRUE Frechet mean of its assigned points.

        The direct measurement of the EMA-target defect. Read it against
        `code_dist_median`: an absolute drift means nothing without the scale
        of the level. Offline only.
        """
        flat = points.reshape(-1, points.shape[-1])
        idx = indices.reshape(-1).long()
        counts = torch.bincount(idx, minlength=self.codebook_size)
        order = torch.argsort(counts, descending=True)[: int(max_codes)]
        drifts: List[float] = []
        for code_id in order.tolist():
            if int(counts[code_id]) < 2:
                continue
            assigned = flat[idx == code_id]
            mean = frechet_mean(assigned)
            drifts.append(float(geodesic_distance(mean, self.codebook[code_id])))
        if not drifts:
            return {}
        tensor = torch.tensor(drifts)
        return {
            "frechet_drift_mean": round(float(tensor.mean()), 4),
            "frechet_drift_max": round(float(tensor.max()), 4),
            "frechet_drift_codes": len(drifts),
        }

    # Self-test
    @torch.no_grad()
    def self_test(self, batch: int = 2, length: int = 8) -> Dict[str, object]:
        """Structural checks that do not need training, with positive controls.

        `control_*` keys must be NON-zero; if they are zero the test has
        failed, not the model. Without them the checks pass trivially on a
        module that returns a constant.
        """
        device = self.codebook.device
        was_training = self.training
        previous_update = self.vq_update
        self.eval()
        self.vq_update = False
        try:
            x = torch.randn(batch, self.dim, length, device=device)
            quant, idx, stats = self.forward(x)

            decoded = self.decode_indices(idx)
            agreement = float((decoded - quant).abs().max())

            points = project_to_hyperboloid(quant.transpose(1, 2))
            projection_error = float((points - stats["points"]).abs().max())

            # Assignment is idempotent: quantising a code returns that code.
            _, again = self.quantise(self.decode_indices_points(idx))
            idempotent = float((again.view_as(idx) - idx).abs().max())

            # POSITIVE CONTROL: a different index must give a different vector.
            other = (idx + 1) % self.codebook_size
            control = float((self.decode_indices(other) - decoded).abs().max())

            # `raise_on_fail=False`: this returns a verdict, so the caller
            # sees every check and not just the first to blow up. The raising
            # form belongs in the training loop, where stopping is the point.
            manifold = float(
                assert_on_manifold(
                    self.codebook.detach(), raise_on_fail=False, name="codebook"
                )
            )
            wrong_sheet = int((self.codebook.detach()[..., 0] <= 0).sum())
        finally:
            self.vq_update = previous_update
            self.train(was_training)

        tolerance = 1e-12 if self.codebook.dtype == torch.float64 else 1e-5
        # The decay contract, made executable. A learnable codebook that is
        # not flagged goes into the decayed group -- it is 2-dimensional and
        # its name matches no exemption pattern -- and decay on a manifold
        # point destroys the constraint rather than shrinking the value.
        flagged = (not self.learnable) or is_manifold_param(self.codebook)
        checks = {
            "decode_agreement": agreement,
            "projection_error": projection_error,
            "quantise_idempotent": idempotent,
            "control_index_changes_output": control,
            "manifold_error": manifold,
            "wrong_sheet": wrong_sheet,
            "codebook_marked_manifold": flagged,
            "restart_reachable": bool(self.dead_restart and self.dead_threshold > 0.0),
        }
        checks["ok"] = bool(
            agreement == 0.0
            and projection_error == 0.0
            and idempotent == 0.0
            and control > 0.0
            and wrong_sheet == 0
            and flagged
            and manifold <= max(tolerance, 1e-9)
        )
        return checks

    def extra_repr(self) -> str:
        return (
            "dim=%d, codes=%d (%.4f bits), learnable=%s, geodesic_target=%s, "
            "dtype=%s"
            % (
                self.dim,
                self.codebook_size,
                math.log(self.codebook_size, 2.0),
                self.learnable,
                self.geodesic_target,
                self.codebook.dtype,
            )
        )


# Levels >= 1: the scalar grid
class ScalarFSQ(nn.Module):
    """Finite scalar quantisation of a TANGENT vector; no codebook at all.

    Each of ``d`` coordinates is squashed by ``tanh`` and rounded to one of
    ``levels[i]`` buckets, so the implicit alphabet is the product grid, and
    the quantised coordinates are mapped into the ``dim``-dimensional tangent
    space at the origin by a FIXED orthonormal frame.

    Having no codebook removes a whole family of failure modes at once -- EMA
    leaving the manifold, a non-geodesic EMA target, dead codes and their
    restart, an extra forward pass overwriting entries -- because there is
    nothing to update. A grid point exists whether or not anything landed on
    it, so utilisation is a property of the ENCODER and not a failure mode of
    the quantiser, which is why the report measures entropy, not coverage.

    The aperture is exact rather than clamped: grid values lie in
    ``[-1, 1]^d`` and the frame rows are orthonormal, so
    ``||v|| = (aperture / sqrt(d)) * ||q|| <= aperture``. The codebook version
    had to clamp the step length on input and after every EMA update, and a
    forgotten clamp silently removes the containment the cone rests on.

    `scale_in` sets the operating point -- the standard deviation of its
    output is the standard deviation of the input to `tanh`, hence the entropy
    of the code -- so its name is a contract; see `self_test`. It is
    initialised at unit gain, ``N(0, gain / sqrt(dim))``, where the PyTorch
    default would give ``0.577 * std(in)`` and about 78 % of the achievable
    entropy instead of 98.6 %. The curve is flat, so an error in that estimate
    is not fatal; a saturated `tanh` is, since it turns a three-level grid
    into a binary one and kills the gradient with it.
    """

    def __init__(
        self,
        dim: int,
        levels: Sequence[int] = (3, 3, 3, 3, 3, 3),
        aperture: float = 1.0,
        gain: float = 1.0,
        entropy_weight: float = 0.0,
        rate_weight: float = 0.0,
        rate_seq_len: float = 0.0,
        entropy_temp: float = 1.0,
        stat_decay: float = 0.99,
        eps: float = 1e-3,
        diagnostics: bool = True,
        seed: int = 7898,
    ):
        super().__init__()
        levels = tuple(int(l) for l in levels)
        d = len(levels)
        if d < 1:
            raise ValueError("the grid needs at least one coordinate")
        if any(l % 2 == 0 or l < 3 for l in levels):
            # See `levels_for_bits`: an even base gives an asymmetric grid with
            # no bucket on zero, and L=2 makes the bound evaluate atanh of a
            # number above 1.
            raise ValueError("grid levels must be ODD and >= 3: %r" % (levels,))
        if int(dim) < d:
            raise ValueError(
                "dim (%d) must be >= the number of grid coordinates (%d): the "
                "frame rows have to be orthonormal" % (int(dim), d)
            )
        size = grid_size(levels)
        if size > (1 << 20):
            raise ValueError(
                "grid of %d cells is too large to keep a usage histogram for; "
                "the report would cost more than the quantiser" % size
            )

        self.dim = int(dim)
        self.levels = levels
        self.d = d
        self.aperture = float(aperture)
        self.gain = float(gain)
        self.entropy_weight = float(entropy_weight)
        #: Weight of the RATE term, which penalises H(marginal), the real code
        #: length. `entropy_weight` is not a rate term -- its expression is
        #: H(p) - H(mean p) <= 0, pressure for a confident assignment under a
        #: uniform marginal, which pushes bits UP. 0.0 = unchanged behaviour.
        self.rate_weight = float(rate_weight)
        #: Sequence length in bp. Above 0 the rate term counts this level's
        #: TOTAL bits divided by it, i.e. its share of `r`; at 0 it counts
        #: bits PER POSITION, which was measured to erase the SHORT levels
        #: (L1 and L2 fell to 4 and 8 bits in total while L3 and L4 grew).
        self.rate_seq_len = float(rate_seq_len)
        self._rate_value = 0.0
        self.entropy_temp = float(entropy_temp)
        self.stat_decay = float(stat_decay)
        self.eps = float(eps)
        self.diagnostics = bool(diagnostics)
        self.step = self.aperture / math.sqrt(float(d))

        #: Respected by `no_codebook_update`. There is no codebook to
        #: overwrite, so the flag suspends the one thing an extra forward pass
        #: could still corrupt: the usage statistics, which would otherwise
        #: count negatives and control passes as if they were training data.
        self.vq_update = True
        self.loss: Optional[torch.Tensor] = None
        self.indices: Optional[torch.Tensor] = None

        level_t = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels", level_t, persistent=False)
        # Half-width of the bounded range and of the integer grid. For odd
        # levels the offset/shift correction of the even case is identically
        # zero, so `bound(z) = tanh(z) * half` with no branch.
        self.register_buffer(
            "_half", (level_t - 1.0) * (1.0 - self.eps) / 2.0, persistent=False
        )
        self.register_buffer("_half_width", (level_t // 2).clamp_min(1.0),
                             persistent=False)
        basis = torch.cumprod(
            torch.tensor((1,) + levels[:-1], dtype=torch.long), dim=0
        )
        self.register_buffer("_basis", basis, persistent=False)

        # `cyfry_czynne` ("live digits"): a smaller alphabet without a change
        # of shape. `d` is the shape of `scale_in` and of the persistent
        # `frame`, so changing it invalidates every checkpoint; freezing the
        # TRAILING digits at their central value drops the alphabet while
        # shapes, parameter count and `zero_index()` stay bit for bit. The
        # mask is non-persistent of necessity, or existing checkpoints would
        # load with a missing key; the value travels in the config instead.
        self.cyfry_czynne = int(d)
        self.register_buffer("_maska_cyfr", torch.ones(d),
                             persistent=False)

        # Learned reduction dim -> d; the NAME selects the decay group.
        self.scale_in = nn.Conv1d(self.dim, d, 1)
        with torch.no_grad():
            self.scale_in.weight.normal_(
                0.0, float(gain) / math.sqrt(float(self.dim))
            )
            if self.scale_in.bias is not None:
                self.scale_in.bias.zero_()

        # Fixed orthonormal frame: d directions in R^dim. Not learning it
        # costs nothing, since every d-dimensional subspace of the tangent
        # space at the origin is equivalent to every other up to a rotation
        # and `scale_in` ahead of the grid is learned; the gain is that
        # `decode_indices` maps index to vector with no learned quantity on
        # the way, making the diffusion contract structural.
        generator = torch.Generator().manual_seed(int(seed))
        q, _ = torch.linalg.qr(torch.randn(self.dim, d, generator=generator))
        # Persistent deliberately: QR sign conventions differ between LAPACK
        # builds, so regenerating the frame from a seed at load time could
        # silently permute the meaning of every index.
        self.register_buffer("frame", q.t().contiguous().float())

        # Bucket centres, only for the optional entropy term.
        max_levels = max(levels)
        centres = torch.zeros(d, max_levels)
        mask = torch.zeros(d, max_levels, dtype=torch.bool)
        for i, l in enumerate(levels):
            for j in range(l):
                centres[i, j] = float(j - l // 2)
                mask[i, j] = True
        self.register_buffer("_centres", centres, persistent=False)
        self.register_buffer("_centre_mask", mask, persistent=False)

        # Statistics: persistent=False, they describe a PASS.
        self.register_buffer("_hist", torch.zeros(size), persistent=False)
        self.register_buffer("_seen", torch.zeros(size, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("_digit_hist", torch.zeros(d, max_levels),
                             persistent=False)
        self.register_buffer("_sat", torch.zeros(()), persistent=False)
        self.register_buffer("_std_in", torch.zeros(()), persistent=False)
        self.register_buffer("_aperture_use", torch.zeros(()), persistent=False)
        self.register_buffer("_quant_error", torch.zeros(()), persistent=False)
        self.register_buffer("_clamped", torch.zeros(()), persistent=False)
        self._stat_steps = 0
        self._entropy_value: Optional[torch.Tensor] = None
        self._calibration: Optional[Dict[str, object]] = None
        self._last_indices: Optional[torch.Tensor] = None

    # Contracts
    @property
    def codebook_size(self) -> int:
        """Size of the GRID. Not to be confused with the number of cells
        actually used -- that is `used_cells` in the report, and its entropy
        is `bits`."""
        return grid_size(self.levels)

    def bits_declared(self) -> float:
        """Budget actually REACHABLE, i.e. after any digits are frozen.

        Without this, `bit_fraction` (threshold 0.85) would divide the
        measured entropy by a budget the quantiser cannot attain and fire
        arithmetically on every freeze. The grid's own size is `grid_cells`.
        """
        return grid_bits(self.levels[:self.cyfry_czynne])

    def manifold_parameters(self) -> List[nn.Parameter]:
        """Empty: nothing here lives on the manifold, so this level is also
        free of the radius-dependent step scaling a Riemannian optimiser
        imposes (measured ratio to the Euclidean step 0.2435 at r=0.25 against
        9.3035 at r=4.0)."""
        return []

    def euclidean_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    def manifold_state_names(self) -> List[str]:
        return []

    def zero_index(self) -> int:
        """Index of the code meaning "no refinement" -- all digits central."""
        return int(
            sum((l // 2) * int(b) for l, b in zip(self.levels, self._basis.tolist()))
        )

    # Live digits
    def alfabet_czynny(self) -> int:
        """Number of cells `forward` can actually PRODUCE.

        The grid does not shrink as an object, only its image does, and
        indices stay in the numbering of the full grid -- so `decode_indices`
        remains the same pure function and the diffusion contract holds.
        """
        return grid_size(self.levels[:self.cyfry_czynne])

    def ustaw_cyfry_czynne(self, ile: Optional[int]) -> None:
        """Keep the first ``ile`` digits live, pin the rest to the centre.

        ``None`` or ``d`` disables the mechanism, and `forward` then skips the
        multiplication entirely rather than multiplying by ones, so the
        default model is bit for bit the same.

        The TRAILING digits are frozen because the frame is a fixed
        orthonormal basis, so no direction is privileged in advance and
        `scale_in` can place the content in whichever rows it likes. "The
        first `a`" is a convention, and the only one needing no permutation
        in the checkpoint.
        """
        ile = self.d if ile is None else int(ile)
        if not 1 <= ile <= self.d:
            raise ValueError(
                "cyfry_czynne = %r outside [1, %d]. Zero live digits is a\n"
                "level that emits ONE code -- a collapsed codebook\n"
                "posing as a configuration; remove a level through\n"
                "model.strides / model.channels, not here."
                % (ile, self.d))
        self.cyfry_czynne = int(ile)
        with torch.no_grad():
            self._maska_cyfr.zero_()
            self._maska_cyfr[:ile] = 1.0

    # Pieces
    def indices_to_values(self, indices: torch.Tensor) -> torch.Tensor:
        """``[...]`` indices -> ``[..., d]`` grid values in ``[-1, 1]``.

        Mixed-radix decoding: digit ``i`` is ``(idx // basis_i) % levels_i``,
        its value ``(digit - levels_i // 2) / (levels_i // 2)``.
        """
        levels = self._levels.long()
        digits = (indices.long().unsqueeze(-1) // self._basis) % levels
        centred = digits.to(self._levels.dtype) - (levels // 2).to(self._levels.dtype)
        return centred / self._half_width

    def values_to_indices(self, values: torch.Tensor) -> torch.Tensor:
        """Inverse of `indices_to_values`, for values already on the grid."""
        digits = (values * self._half_width.to(values.dtype)).round().long()
        digits = digits + (self._levels.long() // 2)
        digits = digits.clamp(
            torch.zeros_like(self._levels.long()), self._levels.long() - 1
        )
        return (digits * self._basis).sum(dim=-1)

    def decode_indices(
        self, indices: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """``[B, L]`` indices -> tangent vectors ``[B, dim, L]`` at the origin.

        A pure function of the indices: no batch statistic, no encoder tensor,
        no learned weight. ``dtype`` MUST match the dtype of the forward pass,
        or the agreement is only of order 1e-7 -- multiplying in float32 and
        casting is not the same as multiplying after the cast, and a decoder
        that drifts from the forward pass produces plausible sequences that do
        not match the reconstruction, with no metric showing it.
        """
        values = self.indices_to_values(indices)
        dt = self.frame.dtype if dtype is None else dtype
        vectors = self.step * (values.to(dt) @ self.frame.to(dt))
        return vectors.transpose(-1, -2)

    # Forward
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """``x`` ``[B, dim, L]`` -> ``(tangent [B, dim, L], indices [B, L], stats)``.

        The returned tangent vector is in the ambient SPATIAL coordinates of
        the tangent space at the origin; the ladder lifts it to ``(0, v)`` and
        parallel-transports it. ``||v|| <= aperture`` exactly.

        Straight-through is composed on the VECTOR, not the scalar. The usual
        ``x + (round(x) - x).detach()`` does not reproduce the rounded value
        bit for bit -- the difference is itself rounded -- and an error of one
        ULP passes through `expmap` to make `decode_indices` disagree with the
        forward pass by about 1e-7. Here the value comes from the grid by
        exactly the route `decode_indices` uses and the gradient from the
        continuous branch, with ``v_soft - v_soft.detach()`` exactly zero.

        No ``.item()`` is called here; see `HyperbolicCodebook.forward`.
        """
        if x.dim() != 3 or x.shape[1] != self.dim:
            raise ValueError(
                "expected x of shape [B, %d, L], got %s" % (self.dim, tuple(x.shape))
            )
        z = self.scale_in(x).transpose(1, 2)  # [B, L, d]
        squashed = torch.tanh(z)
        y = squashed * self._half.to(z.dtype)  # bounded, bucket centres integral
        if self.cyfry_czynne < self.d:
            # Zero BEFORE rounding, on the OPERAND. A frozen digit then takes
            # its central value, its frame row drops out of ``values @ frame``
            # bit for bit, and `soft` and `hard` vanish together, so no
            # gradient reaches the dead row of `scale_in`. Masking the RESULT
            # instead would leave the gradient in place.
            y = y * self._maska_cyfr.to(y.dtype)
        digits = torch.round(y).long() + (self._levels.long() // 2)
        # The clamp is arithmetically unreachable: |y| <= half = hw * (1 - eps)
        # so round(|y|) <= hw for every odd level. It is kept as a TRIPWIRE and
        # counted in the report, where it must stay at exactly 0 -- if it ever
        # fires, the bounding function has been changed underneath the grid.
        low = torch.zeros_like(self._levels.long())
        clamped = digits.clamp(low, self._levels.long() - 1)
        idx = (clamped * self._basis).sum(dim=-1)

        dt = x.dtype
        frame = self.frame.to(dt)
        hard = self.step * (self.indices_to_values(idx).to(dt) @ frame)
        soft = self.step * ((y / self._half_width.to(y.dtype)).to(dt) @ frame)
        v = hard + (soft - soft.detach())

        self.indices = idx
        if self.entropy_weight > 0.0:
            entropy_term = self._entropy_term(y)
            loss = self.entropy_weight * entropy_term
        else:
            entropy_term = torch.zeros((), device=x.device, dtype=torch.float32)
            # Zero, but a TENSOR: the ladder sums the losses of its levels.
            loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.rate_weight > 0.0:
            rate_bits = self._rate_term(y)
            loss = loss + self.rate_weight * rate_bits
            self._rate_value = rate_bits.detach()
        self.loss = loss
        self._entropy_value = entropy_term.detach()

        if self.diagnostics:
            with torch.no_grad():
                self._observe(idx, squashed.detach(), z.detach(), v.detach(),
                              soft.detach(), hard.detach(), digits, clamped)

        stats = {
            "loss": loss,
            "entropy_term": entropy_term.detach(),
            # Position-major [B, L, dim], NOT the [B, dim, L] of the primary
            # return: this is the layout the ladder needs, since it adds the
            # vector to base points of shape [B, L, D+1].
            "tangent": v,
        }
        return v.transpose(1, 2), idx, stats

    @torch.no_grad()
    def _observe(self, idx, squashed, z, v, soft, hard, digits, clamped) -> None:
        """Update the usage statistics. Tensors only -- no device sync."""
        decay = self.stat_decay
        if self.vq_update:
            flat = idx.reshape(-1)
            counts = torch.bincount(flat, minlength=self.codebook_size)
            self._hist.mul_(decay).add_(counts.to(self._hist.dtype),
                                        alpha=1.0 - decay)
            self._seen |= counts > 0
            digit_flat = clamped.reshape(-1, self.d)
            for i in range(self.d):
                c = torch.bincount(
                    digit_flat[:, i], minlength=self._digit_hist.shape[1]
                )
                self._digit_hist[i].mul_(decay).add_(
                    c.to(self._digit_hist.dtype), alpha=1.0 - decay
                )
            self._stat_steps += 1
        self._last_indices = idx
        if self.cyfry_czynne < self.d:
            # A dead coordinate has unbounded `z`, so its `tanh` is often
            # saturated; counted with the live ones, `saturation` and `std_in`
            # would describe an object the quantiser does not use.
            squashed = squashed[..., :self.cyfry_czynne]
            z = z[..., :self.cyfry_czynne]
        # `saturation` is the fraction of |tanh| beyond 0.99 -- the quantity
        # `used_cells` cannot see. Measured failure: saturation 0.9998 with 16
        # effective codes of 81 and exactly 4.000 bits, while cell coverage
        # read 81 of 81.
        self._sat.copy_((squashed.abs() > 0.99).to(self._sat.dtype).mean())
        self._std_in.copy_(z.to(self._std_in.dtype).std())
        self._aperture_use.copy_(
            v.norm(dim=-1).to(self._aperture_use.dtype).mean()
            / max(self.aperture, 1e-12)
        )
        # Relative quantisation error with the ENCODER side in the denominator.
        # With the quantised norm there instead, a collapse of the input to the
        # zero code printed 5.0e+08 rather than a readable 1.0000, because it
        # divided by zero.
        self._quant_error.copy_(
            (soft - hard).norm(dim=-1).mean().to(self._quant_error.dtype)
            / soft.norm(dim=-1).mean().clamp_min(1e-12).to(self._quant_error.dtype)
        )
        self._clamped.add_((digits != clamped).to(self._clamped.dtype).sum())

    # Entropy control
    def _entropy_term(self, y: torch.Tensor) -> torch.Tensor:
        """Soft assignment entropy: ``H(p) per sample MINUS H(mean p)``.

        The sign is the trap. By Jensen's inequality this quantity is always
        <= 0, so a liveness test written as ``any(value > 0)`` declares
        failure exactly when the term is working. The honest test is that the
        value is negative, that its MAGNITUDE moves, and that `bit_fraction`
        moves with it.

        Batch dependence is allowed here and nowhere else: the second term is
        a mean over the batch, so the LOSS depends on batch composition, but
        the assignment stays a function of the single sequence, which leaves
        `decode_indices` and diffusion untouched.
        """
        centres = self._centres.to(y.dtype)
        mask = self._centre_mask
        logits = -(
            y.reshape(-1, self.d).unsqueeze(-1) - centres.unsqueeze(0)
        ).pow(2) / self.entropy_temp
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        p = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        # `masked_fill(-inf)` gives p = 0 and logp = -inf, so p * logp is NaN.
        # Masking the PRODUCT fixes the value and not the gradient: backward
        # evaluates it for every element, gets NaN on the masked ones and
        # multiplies by the zero of the mask, and 0 * NaN = NaN reaches the
        # encoder. The unselected branch is made finite before the multiply.
        logp = torch.where(mask.unsqueeze(0), logp, torch.zeros_like(logp))
        per_sample = -(p * logp).sum(-1).mean()
        marginal = p.mean(0)
        marginal_entropy = -torch.where(
            mask, marginal * marginal.clamp_min(1e-12).log(),
            torch.zeros_like(marginal),
        ).sum(-1).mean()
        return per_sample - marginal_entropy

    def _rate_term(self, y: torch.Tensor) -> torch.Tensor:
        """Differentiable code length, in BITS per position.

        ``d * H(digit marginal) / ln 2``: the upper bound on cell entropy
        under independent digits, the convention `report()` uses for
        `marginal_bits`. The term lowers `r` trivially on its own, which is a
        Goodhart pattern and must be reported as one -- the number that means
        something is `r` at the SAME `rate_weight` with and without the
        residual predictor, ``rec_full`` held at its constraint in both arms.
        """
        centres = self._centres.to(y.dtype)
        mask = self._centre_mask
        logits = -(
            y.reshape(-1, self.d).unsqueeze(-1) - centres.unsqueeze(0)
        ).pow(2) / self.entropy_temp
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        p = torch.softmax(logits, dim=-1)
        marginal = p.mean(0)
        # Operand, not result: a `where` applied to a product that already
        # contains log(0) leaves NaN in the backward pass.
        ent = -torch.where(
            mask, marginal * marginal.clamp_min(1e-12).log(),
            torch.zeros_like(marginal),
        ).sum(-1).sum()
        bity_na_pozycje = ent / float(math.log(2.0))
        if self.rate_seq_len > 0.0:
            # `y` is [B, L, d], so L is dimension 1. This level's TOTAL bits
            # are bits-per-position times L; divided by the sequence length
            # they give the level's share of `r`.
            return bity_na_pozycje * (float(y.shape[1])
                                      / self.rate_seq_len)
        return bity_na_pozycje

    # Calibration
    def expected_zero_fraction(self, std_in: float) -> float:
        """Probability of the zero code at ``std_in``, under independence.

        A fixed alarm on the bare zero fraction is grid-dependent and nearly a
        tautology on small grids: at ``std(z) = 1`` the zero code appears at
        0.53 % on a 6-dimensional grid, 3.0 % on a 4-dimensional one and
        17.4 % on the 2-dimensional grids this ladder uses at 200 and 400 bp,
        where a flat threshold of 0.20 says nothing. `report()` therefore
        gives the ratio to this expectation. Independence is an assumption,
        and the gap between `bits` and `marginal_bits` measures how wrong.
        """
        s = max(float(std_in), 1e-6)
        p = 1.0
        # Live digits only: a frozen one is central with probability 1 and so
        # does not lower the chance of the zero code. Counted with the others,
        # the expectation would be too small and `zero_frac_ratio` would raise
        # a false alarm by orders of magnitude.
        for i, l in enumerate(self.levels[:self.cyfry_czynne]):
            half = (l - 1) * (1.0 - self.eps) / 2.0
            boundary = math.atanh(min(0.5 / half, 1.0 - 1e-12))
            p *= math.erf(boundary / (s * math.sqrt(2.0)))
        return p

    @torch.no_grad()
    def calibrate_gain(
        self,
        feature_map: torch.Tensor,
        candidates: Sequence[float] = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0),
        apply: bool = True,
        allow_eval: bool = False,
    ) -> Dict[str, object]:
        """Choose the input gain BY MEASUREMENT, in the regime the run will use.

        Pass the real feature map at the real batch size. Training mode is
        required unless ``allow_eval`` is set explicitly, because a
        `BatchNorm` on the path makes train and eval two different models.

        Selection maximises `bit_fraction` among candidates with saturation
        below 0.5; ties within 0.01 go to the lower saturation. The result is
        a starting point, not a fix -- the defect is drift of the encoder
        scale during training, which no constant can track.
        """
        if not self.training and not allow_eval:
            raise RuntimeError(
                "calibrate_gain refuses to run under eval(): a normalisation "
                "upstream uses running statistics there and, on an untrained "
                "model, does not normalise at all -- this is the exact mistake "
                "that produced a gain of 15 where the measured optimum was 2.0 "
                "(bit fraction 0.670 against 0.867). Call model.train() first, "
                "or pass allow_eval=True and say so in the report."
            )
        if feature_map.dim() != 3 or feature_map.shape[1] != self.dim:
            raise ValueError(
                "feature_map must be [B, %d, L], got %s"
                % (self.dim, tuple(feature_map.shape))
            )
        bias = self.scale_in.bias
        raw = self.scale_in(feature_map)
        if bias is not None:
            raw = raw - bias.view(1, -1, 1)
        current = max(self.gain, 1e-12)

        trials: List[Dict[str, float]] = []
        for cand in candidates:
            z = raw * (float(cand) / current)
            if bias is not None:
                z = z + bias.view(1, -1, 1)
            z = z.transpose(1, 2)
            squashed = torch.tanh(z)
            y = squashed * self._half.to(z.dtype)
            digits = torch.round(y).long() + (self._levels.long() // 2)
            digits = digits.clamp(
                torch.zeros_like(self._levels.long()), self._levels.long() - 1
            )
            idx = (digits * self._basis).sum(dim=-1).reshape(-1)
            counts = torch.bincount(idx, minlength=self.codebook_size).double()
            p = counts / counts.sum().clamp_min(1.0)
            live = p[p > 0]
            bits = float(-(live * live.log()).sum() / _LOG2)
            std_in = float(z.std())
            trials.append(
                {
                    "gain": float(cand),
                    "std_in": round(std_in, 4),
                    "bits": round(bits, 4),
                    "bit_fraction": round(bits / max(self.bits_declared(), 1e-9), 4),
                    "saturation": round(float((squashed.abs() > 0.99).double().mean()), 4),
                    "zero_frac": round(float(p[self.zero_index()]), 5),
                    "expected_zero_frac": round(self.expected_zero_fraction(std_in), 5),
                    "used_cells": int((counts > 0).sum()),
                }
            )

        usable = [t for t in trials if t["saturation"] < 0.5] or trials
        best_fraction = max(t["bit_fraction"] for t in usable)
        finalists = [t for t in usable if t["bit_fraction"] >= best_fraction - 0.01]
        best = min(finalists, key=lambda t: (t["saturation"], abs(t["gain"] - 1.0)))

        result: Dict[str, object] = {
            "chosen_gain": best["gain"],
            "previous_gain": self.gain,
            "trials": trials,
            "measured_in_training_mode": bool(self.training),
            "batch": int(feature_map.shape[0]),
            "length": int(feature_map.shape[2]),
            "applied": bool(apply),
        }
        if apply:
            self.scale_in.weight.mul_(best["gain"] / current)
            self.gain = float(best["gain"])
        self._calibration = result
        return result

    # Falsifiers
    @torch.no_grad()
    def reset_statistics(self) -> None:
        """Make the statistics describe THIS pass, not the tail of training."""
        self._hist.zero_()
        self._seen.zero_()
        self._digit_hist.zero_()
        self._sat.zero_()
        self._std_in.zero_()
        self._aperture_use.zero_()
        self._quant_error.zero_()
        self._clamped.zero_()
        self._stat_steps = 0
        self._last_indices = None

    @torch.no_grad()
    def report(self) -> Dict[str, object]:
        """Entropy first, coverage last -- and never raises.

        bits            joint entropy of the code distribution (EMA), the
                        honest budget falsifier. A grid was once measured
                        carrying 29.3 bits of 48 declared while touching 7-8
                        of 8 buckets per channel, the distribution being
                        peaked (70.8 % of the mass in one bucket).
        marginal_bits   sum of per-coordinate entropies; ``marginal_bits -
                        bits`` is the redundancy between coordinates.
        bit_fraction    bits / target_bits. Threshold 0.85.
        saturation      fraction of |tanh| above 0.99, invisible to
                        `used_cells`: measured 0.9998 with 16 effective codes
                        of 81 and exactly 4.000 bits -- a ternary grid used as
                        binary -- while coverage read 81 of 81. A saturated
                        tanh also kills the gradient through `scale_in`.
        effective_cells 2 ** bits. Compare with `used_cells`: one checkpoint
                        used 74 and 97 of 729 cells on two levels, i.e. the
                        grids were 7-10x oversized, while the log printed 729.
        used_cells      cells touched EVER, with EMA history. Reported only
                        next to the two above, for exactly that reason.
        zero_frac       share of the "no refinement" code, with its expected
                        value at the measured input scale and their ratio.
        std_in          the operating point `scale_in` sets, and the first
                        thing weight decay destroys.
        clamped         must be exactly 0; see the tripwire in `forward`.
        """
        out: Dict[str, object] = {
            "levels": list(self.levels),
            "grid_cells": self.codebook_size,
            "target_bits": round(self.bits_declared(), 4),
            "aperture": self.aperture,
            "gain": self.gain,
            "entropy_weight": self.entropy_weight,
            "rate_weight": self.rate_weight,
            # Bits measured by the rate term. The key is on the whitelist in
            # `ConeLadder._merge_sub_reports`; without that the term would be
            # computed and invisible.
            "rate_bits": float(self._rate_value),
            "dim": self.dim,
            "grid_dims": self.d,
            "cyfry_czynne": int(self.cyfry_czynne),
            "alfabet_czynny": int(self.alfabet_czynny()),
            "stat_steps": int(self._stat_steps),
            "frame_orth_error": round(
                float(
                    (
                        self.frame @ self.frame.t()
                        - torch.eye(self.d, device=self.frame.device)
                    )
                    .abs()
                    .max()
                ),
                9,
            ),
            "clamped": int(self._clamped),
            "saturation": round(float(self._sat), 4),
            "std_in": round(float(self._std_in), 4),
            "aperture_use": round(float(self._aperture_use), 4),
            "quant_error": round(float(self._quant_error), 4),
        }
        if self._entropy_value is not None:
            out["entropy_term"] = round(float(self._entropy_value), 5)
        total = float(self._hist.sum())
        if total > 0.0:
            p = (self._hist / self._hist.sum()).double()
            live = p[p > 0]
            bits = float(-(live * live.log()).sum() / _LOG2)
            marginal = 0.0
            for i, l in enumerate(self.levels):
                hi = self._digit_hist[i, :l].double()
                hi = hi / hi.sum().clamp_min(1e-30)
                hi = hi[hi > 0]
                marginal += float(-(hi * hi.log()).sum() / _LOG2)
            declared = max(self.bits_declared(), 1e-9)
            zero_frac = float(p[self.zero_index()])
            expected = self.expected_zero_fraction(float(self._std_in))
            out.update(
                {
                    "bits": round(bits, 4),
                    "marginal_bits": round(marginal, 4),
                    "bit_fraction": round(bits / declared, 4),
                    "redundancy_bits": round(marginal - bits, 4),
                    "perplexity": round(float(torch.exp(-(live * live.log()).sum())), 2),
                    "effective_cells": round(2.0 ** bits, 1),
                    "used_cells": int(self._seen.sum()),
                    "zero_frac": round(zero_frac, 5),
                    "expected_zero_frac": round(expected, 5),
                    "zero_frac_ratio": round(zero_frac / max(expected, 1e-9), 3),
                }
            )
        if self._last_indices is not None:
            out.update(_cell_entropy(self._last_indices, self.codebook_size))
        if self._calibration is not None:
            out["calibrated_gain"] = self._calibration["chosen_gain"]
            out["calibrated_in_training_mode"] = self._calibration[
                "measured_in_training_mode"
            ]
        return out

    def falsifiers(self) -> List[Dict[str, object]]:
        """Verdicts, each stating whether the question can be answered."""
        r = self.report()
        found: List[Dict[str, object]] = []
        fraction = r.get("bit_fraction")
        found.append(
            _finding(
                "bit_fraction",
                None if fraction is None else float(fraction),
                0.85,
                ">=",
                True,
                "joint code entropy over the declared budget; 0.61 was "
                "measured while coverage looked healthy",
            )
        )
        found.append(
            _finding(
                "saturation",
                float(r["saturation"]),
                0.50,
                "<=",
                True,
                "fraction of |tanh| > 0.99; at 0.9998 a ternary grid was used "
                "as binary (16 effective codes of 81, exactly 4.000 bits) and "
                "the gradient through scale_in is dead",
            )
        )
        ratio = r.get("zero_frac_ratio")
        found.append(
            _finding(
                "zero_frac_ratio",
                None if ratio is None else float(ratio),
                3.0,
                "<=",
                True,
                "zero code against its expectation at the measured input "
                "scale; a flat 0.20 threshold is 17.4 % at initialisation on "
                "a 2-dimensional grid and says nothing there",
            )
        )
        found.append(
            _finding(
                "std_in",
                float(r["std_in"]),
                0.2,
                ">=",
                True,
                "operating point of the quantiser; a collapse to 0.065 made "
                "every position round to the zero code (1 cell of 729)",
            )
        )
        found.append(
            _finding(
                "clamped",
                float(r["clamped"]),
                0.0,
                "==",
                # This one is arithmetically unreachable and says so: |y| is
                # bounded below the outermost bucket edge by construction. It
                # is a tripwire on the bounding function, not a measurement of
                # the data.
                False,
                "digits outside the grid; unreachable unless bound() changed",
            )
        )
        found.append(
            _finding(
                "frame_orth_error",
                float(r["frame_orth_error"]),
                1e-5,
                "<=",
                True,
                "orthonormality of the frame rows; the exact aperture bound "
                "||v|| <= aperture depends on it",
            )
        )
        used = r.get("used_cells")
        effective = r.get("effective_cells")
        if used is not None and effective is not None:
            found.append(
                _finding(
                    "grid_oversize",
                    round(float(effective) / max(float(self.codebook_size), 1.0), 4),
                    0.25,
                    ">=",
                    True,
                    "effective cells over grid cells; measured 74/729 and "
                    "97/729 on a checkpoint whose log printed 729 of 729, so "
                    "shrinking the grid there was free",
                )
            )
        return found

    # Self-test
    @torch.no_grad()
    def self_test(self, batch: int = 2, length: int = 8) -> Dict[str, object]:
        """Structural checks with positive controls; no training required.

        What is not checked here and cannot be: whether the consumer's
        `expmap` handles the zero code correctly. That failure lives in
        `ladder.py`; `zero_exact` below at least guarantees the zero code
        really is the zero VECTOR, so the responsibility is unambiguous.
        """
        device = self.frame.device
        was_training = self.training
        previous_update = self.vq_update
        self.eval()
        self.vq_update = False
        try:
            x = torch.randn(batch, self.dim, length, device=device)
            tangent, idx, _ = self.forward(x)

            decoded = self.decode_indices(idx, dtype=x.dtype)
            agreement = float((decoded - tangent).abs().max())

            values = self.indices_to_values(idx)
            roundtrip = float((self.values_to_indices(values) - idx).abs().max())

            norms = tangent.transpose(1, 2).norm(dim=-1)
            aperture_excess = float((norms - self.aperture).max())

            zero = torch.full_like(idx, self.zero_index())
            zero_exact = float(self.decode_indices(zero, dtype=x.dtype).abs().max())

            # POSITIVE CONTROL: a different index must give a different
            # vector. Without it the agreement check passes trivially on a
            # module that returns a constant, which is what an untrained model
            # in eval often does.
            other = (idx + 1) % self.codebook_size
            control = float(
                (self.decode_indices(other, dtype=x.dtype) - decoded).abs().max()
            )

            frame_error = float(
                (self.frame @ self.frame.t() - torch.eye(self.d, device=device))
                .abs()
                .max()
            )

            # Control on the freezing mechanism. With `cyfry_czynne == d` the
            # slice is empty and the value is 0.0, so the test says nothing
            # and does not pretend otherwise. Under freezing every dead digit
            # must come out central; a non-zero value means the mask is not
            # working and the alphabet is not the declared one.
            martwe = ((idx.unsqueeze(-1) // self._basis)
                      % self._levels.long())[..., self.cyfry_czynne:]
            srodek = (self._levels.long() // 2)[self.cyfry_czynne:]
            frozen = (float((martwe - srodek).abs().max())
                      if martwe.numel() else 0.0)
        finally:
            self.vq_update = previous_update
            self.train(was_training)

        # The name is a contract, so check it rather than remember it: the
        # operating point of this quantiser is the norm of `scale_in`, and
        # under decay 0.06 at lr 1e-3 over 35 epochs the multiplier is
        # 1.5e-04, so the scale falls 6800x and every code becomes the zero
        # code. A rename outside the exemption patterns does that silently,
        # so the rename has to fail here.
        names = [n for n, _ in self.named_parameters()]
        exempt = all(
            any(pat in name.lower() for pat in NO_DECAY_SUBSTRINGS)
            for name in names
        )

        checks = {
            "decode_agreement": agreement,
            "index_roundtrip": roundtrip,
            "aperture_excess": aperture_excess,
            "zero_code_exact": zero_exact,
            "control_index_changes_output": control,
            "frame_orth_error": frame_error,
            "frozen_digits_central": frozen,
            "cyfry_czynne": int(self.cyfry_czynne),
            "alfabet_czynny": int(self.alfabet_czynny()),
            "parameter_names": names,
            "names_exempt_from_decay": exempt,
        }
        checks["ok"] = bool(
            agreement == 0.0
            and roundtrip == 0.0
            and aperture_excess <= 1e-6
            and zero_exact == 0.0
            and control > 0.0
            and frame_error <= 1e-5
            and frozen == 0.0
            and exempt
        )
        return checks

    def extra_repr(self) -> str:
        czynne = ("" if self.cyfry_czynne >= self.d else
                  ", czynne=%d/%d (alfabet %d)"
                  % (self.cyfry_czynne, self.d, self.alfabet_czynny()))
        return (
            "dim=%d, grid=%r (%d cells, %.4f bits), aperture=%.3f, gain=%.3f%s"
            % (
                self.dim,
                self.levels,
                self.codebook_size,
                self.bits_declared(),
                self.aperture,
                self.gain,
                czynne,
            )
        )


# Cross-level audit
@torch.no_grad()
def code_audit(
    indices: Sequence[torch.Tensor],
    sizes: Sequence[int],
    names: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """How many codes are alive per level, and how many of them repeat the parent.

    ``indices`` are ``[B, L_k]`` tensors in LADDER ORDER, i.e. from the
    coarsest level, matching `ConeLadder.forward`.

    Mutual information is the point: a child that repeats its parent carries
    no new bits however healthy its perplexity looks. A residual ladder
    elsewhere had ``corr(residual, -parent code)`` of 0.9997, i.e. it was
    quantising itself.

    Read ``I_excess``, not ``I``. The plug-in estimate of mutual information
    is biased UPWARD, and for the sparse tables this ladder produces the bias
    is the whole number: on INDEPENDENT random indices over a 128 x 729 joint
    table, I was 5.95 bits at batch 8 and still 2.09 at batch 512, while the
    Miller-Madow correction reported 0.54 and 0.63. A threshold of "I < 1.0
    bit" therefore fails for pure noise at every batch this model will train
    on. The fix is a control, not a correction: ``I_shuffled`` is the same
    statistic with the parent decoupled by a fixed roll, which preserves both
    marginals and destroys only the pairing, so it is the value ``I`` would
    take under independence AT THIS SAMPLE SIZE AND SPARSITY. Working
    threshold ``I_excess < 1.0 bit``; the roll is deterministic because
    `randperm` would advance the global RNG stream.

    `H_mm` is given separately and never substituted silently, because for
    sparse tables the plug-in ENTROPY estimate is biased downward and that is
    easy to mistake for missing entropy. Known limitation: the parent of a
    child position is taken as the nearest one, while the ladder lifts the
    parent by linear interpolation, so the audit understates the dependence.
    """
    results: List[Dict[str, object]] = []
    for k, idx in enumerate(indices):
        cells = int(sizes[k])
        flat = idx.reshape(-1).long()
        n = int(flat.numel())
        counts = torch.bincount(flat, minlength=cells).double()
        p = counts / counts.sum().clamp_min(1.0)
        live = p[p > 0]
        entropy = float(-(live * live.log()).sum() / _LOG2)
        support = int((counts > 0).sum())
        row: Dict[str, object] = {
            "level": names[k] if names else "L%d" % k,
            "n": n,
            "cells": cells,
            "used": support,
            "H": round(entropy, 4),
            "H_mm": round(entropy + (support - 1) / (2.0 * max(n, 1) * _LOG2), 4),
            "fraction": round(entropy / math.log(cells, 2.0), 4),
        }
        if k > 0:
            parent = indices[k - 1].reshape(idx.shape[0], -1)
            child_len, parent_len = int(idx.shape[-1]), int(parent.shape[-1])
            pos = (
                (
                    torch.arange(child_len, device=idx.device, dtype=torch.float64)
                    + 0.5
                )
                * parent_len
                / child_len
            ).floor().long().clamp(0, parent_len - 1)
            parent_flat = parent[:, pos].reshape(-1).long()
            parent_cells = int(sizes[k - 1])

            def _mutual(par: torch.Tensor) -> Tuple[float, int, int]:
                joint = torch.bincount(
                    par * cells + flat, minlength=parent_cells * cells
                ).double()
                pj = joint / joint.sum().clamp_min(1.0)
                pjl = pj[pj > 0]
                joint_entropy = float(-(pjl * pjl.log()).sum() / _LOG2)
                pc = torch.bincount(par, minlength=parent_cells).double()
                pcp = pc / pc.sum().clamp_min(1.0)
                pcl = pcp[pcp > 0]
                parent_entropy = float(-(pcl * pcl.log()).sum() / _LOG2)
                return (
                    joint_entropy - parent_entropy,
                    int((joint > 0).sum()),
                    int((pc > 0).sum()),
                )

            conditional, joint_support, parent_support = _mutual(parent_flat)
            # The control: same marginals, no pairing. A roll rather than a
            # permutation, so the report leaves the RNG stream untouched.
            rolled, _, _ = _mutual(torch.roll(parent_flat, shifts=n // 2 + 1))
            row["H_cond"] = round(conditional, 4)
            row["I"] = round(entropy - conditional, 4)
            row["I_shuffled"] = round(entropy - rolled, 4)
            row["I_excess"] = round(rolled - conditional, 4)
            row["I_bias"] = round(
                (joint_support - parent_support) / (2.0 * max(n, 1) * _LOG2), 4
            )
        results.append(row)
    return results


__all__ = [
    "NO_DECAY_SUBSTRINGS",
    "HyperbolicCodebook",
    "ScalarFSQ",
    "levels_for_bits",
    "grid_bits",
    "grid_size",
    "code_audit",
    "no_codebook_update",
    "reset_statistics",
    "collect_manifold_state_names",
]
