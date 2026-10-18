"""Cone ladder -- the one route to hyperbolicity this project has not closed.

Entailment cones need a containment hierarchy and a residual ladder has none:
cos(child, parent) measured 0.0981 against a random-pairing control of the same
0.0981. This ladder instead quantises a POSITION at level 0 and a tangent
REFINEMENT at level k, composing them geodesically:

    p_0     = codebook entry (already a point)
    p_{k+1} = expmap(p_k, PT(origin -> p_k, v_{k+1}) + s_k * radial(p_k))

Containment is then structural, the child lying at geodesic distance ||step||
from its parent, and reconstruction reaches the codebook because every deeper
point is a function of the level-0 entry through `expmap` rather than a
straight-through detach. Quantising the tangent vector is itself Euclidean;
what is hyperbolic is the composition. The aperture and radius plan are
buffers, because this model collapses every learnable stratifying mechanism it
is given, and containment is read as geodesic distance rather than as a cosine,
which mixes distance with radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.lorentz import (
    assert_on_manifold,
    expmap,
    frechet_mean,
    geodesic_distance,
    logmap,
    minkowski_dot,
    parallel_transport,
    project_to_hyperboloid,
    radius,
)

# Measured constants of the Trichoderma promoter corpus, kept here because the
# prefix falsifier is read against them.
UNIGRAM_BITS_PER_BP = 1.9983     # measured unigram entropy of the corpus
RECONSTRUCTION_FLOOR = 0.2626    # argmax of the unigram; not 0.25, not 0.2934

#: English report key -> the Polish key of the DeepSTARR `band_report` logs,
#: so old and new runs can be read against each other.
LEGACY_KEYS = {
    "radius": "promien",
    "radius_plan": "promien_plan",
    "radius_gap": "promien_luka",
    "radius_step_measured": "krok_prom",
    "radius_step_nominal": "krok_prom_nom",
    "d_parent": "d_rodzic",
    "d_random": "d_losowy",
    "d_ratio": "d_iloraz",
    "cos_parent": "cos_rodzic",
    "cos_random": "cos_losowy",
    "cos_parent_centred": "cos_rodzic_c",
    "cos_random_centred": "cos_losowy_c",
    "angle": "kat",
    "angle_rel": "kat_wzg",
    "angle_max": "kat_maks",
    "angle_ceiling": "kat_sufit",
    "outside_cone": "poza_stozkiem",
    "cone_can_fire": "stozek_moze",
    "aperture": "aperture",
    "cone_weight": "waga_stozka",
    "step_norm_in": "dl_we_sr",
    "step_clamped_frac": "zaciete",
    "aperture_use": "wyk_apert",
    "perplexity": "perplexity",
    "used_frac": "frakcja_kodow",
}

_EPS = 1e-12


@dataclass
class LadderOutput:
    """What a ladder pass hands to the decoder, the heads and the logger.

    `z` is per level [B, C, L_k] and the decoder's only input; `indices` are
    the [B, L_k] integer codes making up the diffusion state; `points` are
    [B, L_k, D+1] on the hyperboloid. `total_bits` is ACHIEVED entropy,
    sum_k L_k * bits_k, not the declared budget.
    """

    z: List[torch.Tensor]
    indices: List[torch.Tensor]
    points: List[torch.Tensor]
    stats: dict
    total_bits: float
    loss: Optional[torch.Tensor] = None
    cone_loss: Optional[torch.Tensor] = None


class PrzewidywaczReszty(nn.Module):
    """Predict a child quantiser's input from a WINDOW of parent points.

    The convolutional pyramid is a strict cascade, so cross-level redundancy is
    structural rather than learned -- 512.8 bits shared out of 2653.2 marginal
    -- and subtracting a prediction leaves the code carrying only the residual.
    A window, not a point: ridge from a single parent point removes only
    R2 = 0.2043 / 0.0385 / 0.0424 / 0.0658, so a 1x1 predictor would live,
    carry a gradient, remove 4 % instead of 44 % and look wired up in the log.
    The target is detached, so that the predictor learns to predict rather than
    the encoder to be predictable.
    """

    def __init__(self, dim: int, out_dim: int, hidden: int, window: int,
                 gain: float = 0.1):
        super().__init__()
        window = int(window)
        if window < 1 or window % 2 == 0:
            raise ValueError(
                "pred_okno must be ODD and >= 1, it is %r.  An even window "
                "shifts the output by half a position, i.e. the child would "
                "be predicted from a parent at a DIFFERENT position."
                % (window,))
        self.window = window
        self.body = nn.Sequential(
            nn.Conv1d(int(dim), int(hidden), window, padding=window // 2),
            nn.SiLU(),
            nn.Conv1d(int(hidden), int(out_dim), 1, bias=False))
        nn.init.xavier_uniform_(self.body[0].weight, gain=1.0)
        nn.init.zeros_(self.body[0].bias)
        nn.init.xavier_uniform_(self.body[2].weight, gain=float(gain))

    def forward(self, parent_feature: torch.Tensor) -> torch.Tensor:
        """[B, dim, L_child] -> [B, out_dim, L_child].

        No BatchNorm and no dropout: the module sits on the `decode_indices`
        path, so any batch statistic would split the diffusion path from the
        forward pass, and silently.
        """
        return self.body(parent_feature)


class ConeLadder(nn.Module):
    """Geodesic composition of one position code and K-1 refinement codes.

    `position_quantizer` serves level 0 and is expected to be a
    `HyperbolicCodebook`, whose entries are points of the hyperboloid and the
    only tensors here `RiemannianAdam` can optimise; `tangent_quantizers` holds
    one Euclidean `ScalarFSQ` per refinement level. `in_channels` is the
    encoder width per level, coarsest first; `tangent_dims` is each tangent
    quantiser's input width, read from `quantizer.dim` when omitted.

    `step_length` is the cone aperture per refinement level, a hard cap on the
    tangent step norm, broadcast if scalar. Measured [0.25, 0.5] over two
    levels: the cap has to grow with depth because sinh(r) does, so one scalar
    across many levels shows as `step_clamped_frac` near 1. `radius_step` is
    the nominal radial descent per level; given 0.5 the model produced 0.06 and
    0.31. `seq_len`, in bp, is not inferred: the finest map is a strided view
    whose stride is not knowable here.
    """

    def __init__(self,
                 position_quantizer: nn.Module,
                 tangent_quantizers: Sequence[nn.Module],
                 dim: int,
                 in_channels: Sequence[int],
                 tangent_dims: Optional[Sequence[int]] = None,
                 step_length=0.25,
                 radius_step=0.15,
                 cone_weight: float = 0.0,
                 aperture: float = 1.0,
                 z_mode: str = "spatial",
                 seq_len: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 out_dtype: torch.dtype = torch.float32,
                 audit_every: int = 1,
                 seed: int = 7898,
                 cyfry_czynne: Optional[Sequence[Optional[int]]] = None,
                 pred_reszty: bool = False,
                 pred_okno: int = 5,
                 pred_ukryte: int = 32,
                 pred_waga: float = 0.0):
        super().__init__()
        self.position_quantizer = position_quantizer
        self.tangent_quantizers = nn.ModuleList(list(tangent_quantizers))
        self.n_levels = 1 + len(self.tangent_quantizers)
        self.dim = int(dim)
        self.cone_weight = float(cone_weight)
        self.aperture = float(aperture)
        self.seq_len = None if seq_len is None else int(seq_len)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.audit_every = max(1, int(audit_every))
        if z_mode not in ("spatial", "tangent0"):
            raise ValueError("z_mode must be 'spatial' or 'tangent0', got %r"
                             % (z_mode,))
        self.z_mode = z_mode

        in_channels = [int(c) for c in in_channels]
        if len(in_channels) != self.n_levels:
            raise ValueError("in_channels has %d entries, ladder has %d levels"
                             % (len(in_channels), self.n_levels))
        self.in_channels = in_channels

        if tangent_dims is None:
            tangent_dims = []
            for k, q in enumerate(self.tangent_quantizers):
                w = getattr(q, "dim", None)
                if w is None:
                    raise ValueError(
                        "tangent_dims[%d] is unresolvable: quantiser %s has no "
                        "`dim` attribute.  Pass tangent_dims explicitly -- a "
                        "guessed width would silently mis-size the grid."
                        % (k, type(q).__name__))
                tangent_dims.append(int(w))
        tangent_dims = [int(w) for w in tangent_dims]
        if len(tangent_dims) != self.n_levels - 1:
            raise ValueError("tangent_dims has %d entries, expected %d"
                             % (len(tangent_dims), self.n_levels - 1))
        self.tangent_dims = tangent_dims

        # 1x1 only: a wider kernel would make `decode_indices` a function of
        # neighbouring codes, and the diffusion contract would stop holding.
        pos_in = getattr(position_quantizer, "dim", self.dim)
        self.to_code = (nn.Identity() if in_channels[0] == int(pos_in)
                        else nn.Conv1d(in_channels[0], int(pos_in), 1))
        self.to_tangent = nn.ModuleList([
            nn.Conv1d(in_channels[k + 1], tangent_dims[k], 1)
            for k in range(self.n_levels - 1)])
        # No bias, so the all-zero cell maps to the exactly zero step: that is
        # the prefix property. Gain 0.1 is small but non-zero, a zero
        # initialisation orphaning its own source.
        self.from_tangent = nn.ModuleList([
            nn.Conv1d(tangent_dims[k], self.dim, 1, bias=False)
            for k in range(self.n_levels - 1)])
        for m in self.from_tangent:
            nn.init.xavier_uniform_(m.weight, gain=0.1)

        # Buffers: invisible to weight decay, which is measured to destroy
        # exactly this kind of small scalar, and present in the checkpoint, so
        # a constant aperture stays auditable after the fact.
        self.register_buffer("step_length",
                             self._broadcast(step_length, self.n_levels - 1,
                                             "step_length"))
        self.register_buffer("radius_step",
                             self._broadcast(radius_step, self.n_levels - 1,
                                             "radius_step"))

        # Private RNG for the random-pairing control: a global `randperm` would
        # make a run depend on how often the report was computed.
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))

        # Alphabets before `_resolve_null_indices`. The order is fixed here so
        # it cannot depend on which branch of the resolver happened to run.
        self._rozwiaz_cyfry_czynne(cyfry_czynne)

        self.lift_convention = self._probe_lift_convention(seed)
        self._null_index: List[Optional[int]] = []
        self._null_norm: List[float] = []
        self._resolve_null_indices()

        self._stats: dict = {}
        self._calls = 0
        self._pred_diag_stats: dict = {}

        # A new module goes LAST in `__init__`: the RNG stream is a contract,
        # and anything above would draw different numbers than a run without.
        # `pred_reszty` with a zero weight would build a predictor whose loss
        # is never added and whose reconstruction gradient cancels through
        # the straight-through estimator. That combination is refused in
        # config validation, which is the only path a run takes; it is not
        # re-checked here, so a direct construction is the caller's own.
        self.pred_weight = float(pred_waga)
        self.residual_pred = None
        if bool(pred_reszty):
            self.residual_pred = nn.ModuleList([
                PrzewidywaczReszty(dim=self.dim,
                                   out_dim=self.tangent_dims[k],
                                   hidden=int(pred_ukryte),
                                   window=int(pred_okno))
                for k in range(self.n_levels - 1)])

    # Live FSQ digits
    def _rozwiaz_cyfry_czynne(self, podane) -> None:
        """How many FSQ digits stay live per level -- from argument or config.

        `None` reads `model.fsq_cyfry_czynne` from the ACTIVE config, which
        `load_config` sets but a `copy.deepcopy(cfg)` with a field swapped does
        not. The digit count that config implies must therefore match the
        quantisers this ladder received, or the key is refused.
        """
        d_drabiny = [int(getattr(q, "d", 0)) for q in self.tangent_quantizers]
        zrodlo = "argument"
        if podane is None:
            zrodlo = "domyslne"
            cfg = None
            try:
                from ..config import active_config, fsq_grid
                cfg = active_config()
            except Exception:
                cfg = None
            mc = None if cfg is None else getattr(cfg, "model", None)
            deklaracja = tuple(getattr(mc, "fsq_cyfry_czynne", ()) or ())
            if deklaracja:
                d_cfg = [fsq_grid(int(b))[0] for b in mc.fsq_bits]
                if d_cfg != d_drabiny:
                    raise ValueError(
                        "model.fsq_cyfry_czynne = %r is set in the ACTIVE "
                        "config, but that config describes grids of %r "
                        "digits, while this ConeLadder was given quantisers "
                        "of %r.  Applying the key to a different model is "
                        "worse than ignoring it, so it is refused: pass "
                        "`cyfry_czynne=` explicitly in the call."
                        % (list(deklaracja), d_cfg, d_drabiny))
                podane = deklaracja
                zrodlo = "config"
        self._cyfry_zrodlo = zrodlo
        if not podane:
            self.cyfry_czynne = list(d_drabiny)
            return
        podane = [int(x) for x in podane]
        if len(podane) != len(d_drabiny):
            raise ValueError(
                "cyfry_czynne has %d entries, the ladder has %d refinement "
                "levels.  Silently stretching the list is how an "
                "equal-budget experiment stops being one."
                % (len(podane), len(d_drabiny)))
        for k, ile in enumerate(podane):
            self.tangent_quantizers[k].ustaw_cyfry_czynne(ile)
        self.cyfry_czynne = [int(getattr(q, "cyfry_czynne", 0))
                             for q in self.tangent_quantizers]
        if any(a < d for a, d in zip(self.cyfry_czynne, d_drabiny)):
            print("  ConeLadder: live digits %r of %r (source %s) -> "
                  "alphabets %r instead of %r"
                  % (self.cyfry_czynne, d_drabiny, zrodlo,
                     [int(q.alfabet_czynny())
                      for q in self.tangent_quantizers],
                     [int(q.codebook_size)
                      for q in self.tangent_quantizers]))

    @staticmethod
    def _broadcast(value, n: int, name: str) -> torch.Tensor:
        if isinstance(value, (int, float)):
            out = [float(value)] * n
        else:
            out = [float(v) for v in value]
            if len(out) != n:
                raise ValueError(
                    "%s has %d entries, expected %d (one per refinement "
                    "level).  A silently recycled list is how an aperture "
                    "sweep stops being an equal-budget experiment."
                    % (name, len(out), n))
        return torch.tensor(out, dtype=torch.float64)

    def _probe_lift_convention(self, seed: int) -> str:
        """Measure, once, whether `project_to_hyperboloid` takes D or D+1."""
        g = torch.Generator()
        g.manual_seed(int(seed))
        s = torch.randn(4, self.dim, generator=g, dtype=torch.float64)
        for name, arg in (("spatial", s),
                          ("ambient", torch.cat([torch.zeros(4, 1,
                                                             dtype=torch.float64),
                                                 s], dim=-1))):
            try:
                out = project_to_hyperboloid(arg)
            except Exception:
                continue
            if out.shape[-1] != self.dim + 1:
                continue
            try:
                residual = float(assert_on_manifold(out, tol=1e-6))
            except Exception:
                continue
            if residual < 1e-6:
                return name
        raise RuntimeError(
            "project_to_hyperboloid accepted neither a [N, %d] spatial part "
            "nor a [N, %d] ambient point with an on-manifold result.  The "
            "ladder refuses to guess the lift convention." % (self.dim, self.dim + 1))

    def _lift(self, spatial: torch.Tensor) -> torch.Tensor:
        """[..., D] spatial part -> [..., D+1] point, via the shared arithmetic."""
        if self.lift_convention == "spatial":
            return project_to_hyperboloid(spatial)
        pad = torch.zeros_like(spatial[..., :1])
        return project_to_hyperboloid(torch.cat([pad, spatial], dim=-1))

    def _origin_like(self, point: torch.Tensor) -> torch.Tensor:
        o = torch.zeros_like(point)
        o[..., 0] = 1.0
        return o

    def _to_tangent_space(self, point: torch.Tensor,
                          ambient: torch.Tensor) -> torch.Tensor:
        """Project an ambient vector onto T_point, using <x,x>_L = -1."""
        return ambient + minkowski_dot(point, ambient, keepdim=True) * point

    def _radial(self, point: torch.Tensor) -> torch.Tensor:
        """Unit tangent vector at `point` pointing away from the origin.

        Without a radial component the radius performs a random walk: measured
        1.3297 -> 1.2180 -> 1.2364, levels at no distinct depth at all. At the
        origin the outward direction is undefined, so the OPERAND is masked --
        `torch.where` over a quotient holding 0/0 keeps NaN in the backward.
        """
        sp = point[..., 1:]
        n_sp = sp.norm(dim=-1, keepdim=True)
        live = n_sp > _EPS
        sp_safe = torch.where(live, sp, torch.ones_like(sp))
        u = sp_safe / sp_safe.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        amb = torch.cat([torch.zeros_like(point[..., :1]), u], dim=-1)
        t = self._to_tangent_space(point, amb)
        norm = torch.sqrt(minkowski_dot(t, t, keepdim=True).clamp_min(_EPS))
        return (t / norm) * live.to(t.dtype)

    def _transport_from_origin(self, point: torch.Tensor,
                               spatial: torch.Tensor) -> torch.Tensor:
        """PT(origin -> point, v) -- the origin's tangent vectors are not
        `point`'s, so the transport is mandatory.
        """
        pad = torch.zeros_like(spatial[..., :1])
        v0 = torch.cat([pad, spatial], dim=-1)
        return parallel_transport(self._origin_like(point), point, v0)

    def _clamp_step(self, v: torch.Tensor, level: int):
        """Cap the step norm to the aperture by SCALING, never by truncation.

        Truncation zeroes the gradient for exactly the vectors that need to be
        pulled back in. Below the aperture the factor is an explicit 1, not
        ``n / n``, which is ``0 / eps`` at ``n == 0`` and makes the legal zero
        step -- the state every position starts in -- absorbing.
        """
        limit = float(self.step_length[level])
        n = v.norm(dim=-1, keepdim=True)
        # Guard the operand, not the result; see `_radial`.
        factor = torch.where(n > limit, limit / n.clamp_min(_EPS),
                             torch.ones_like(n))
        scaled = v * factor
        with torch.no_grad():
            stats = {
                "step_norm_in": float(n.mean()),
                "step_norm_out": float(scaled.norm(dim=-1).mean()),
                "step_clamped_frac": float((n > limit).to(torch.float64).mean()),
                "aperture_use": float(scaled.norm(dim=-1).mean()) / max(limit, _EPS),
                "step_limit": limit,
            }
        return scaled, stats

    def _step_from_code_map(self, parent: torch.Tensor,
                            code_map: torch.Tensor, level: int):
        """ONE geodesic step, shared by `forward` and `decode_indices`.

        Written twice the two could drift, and silently: `decode_indices` would
        return a plausible sequence corresponding to nothing trained on.
        """
        v = self.from_tangent[level](code_map)          # [B, D, L]
        v = v.transpose(1, 2).to(parent.dtype)          # [B, L, D]
        v, stats = self._clamp_step(v, level)
        step = self._transport_from_origin(parent, v)
        step = step + self.radius_step[level].to(step.dtype) * self._radial(parent)
        return expmap(parent, step), stats

    def _upsample(self, parent: torch.Tensor, length: int, level: int):
        """Carry a parent point to the child's resolution.

        Nearest, not linear: containment is a statement about THE parent
        window, and linear blending produces a point that is not any code,
        inflating `d_parent`. Repeating a point also keeps it exactly on the
        manifold.
        """
        cur = parent.shape[1]
        if cur == length:
            self._up_mode[level] = "identity"
            return parent
        if length < cur:
            raise ValueError(
                "level %d is COARSER than its parent (%d < %d); maps must be "
                "ordered coarsest first" % (level + 1, length, cur))
        if length % cur == 0:
            self._up_mode[level] = "repeat"
            return parent.repeat_interleave(length // cur, dim=1)
        self._up_mode[level] = "nearest"
        out = F.interpolate(parent.transpose(1, 2), size=length, mode="nearest")
        return out.transpose(1, 2)

    def _points_from_map(self, code_map: torch.Tensor) -> torch.Tensor:
        """[B, C, L] from the position quantiser -> [B, L, D+1] points.

        Its D channels may be the spatial part or the whole ambient point;
        both are accepted and reported, and any other width raises.
        """
        t = code_map.transpose(1, 2)
        if t.shape[-1] == self.dim:
            self._point_layout = "spatial"
            return self._lift(t.to(self.dtype))
        if t.shape[-1] == self.dim + 1:
            self._point_layout = "ambient"
            return t.to(self.dtype)
        raise ValueError(
            "position quantiser returned %d channels; expected %d (spatial) "
            "or %d (ambient)" % (t.shape[-1], self.dim, self.dim + 1))

    def _z_from_points(self, points: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """The decoder's input, derived from points ONLY.

        No continuous bypass: with one open the decoder holds a lossless copy
        of the encoder and the codes become optional. Measured in DeepSTARR, a
        reported recon of 0.9984 with the bypass was 0.29 from the codes alone.
        Both `z_mode` settings are bijective with the point.
        """
        out = []
        for p in points:
            if self.z_mode == "spatial":
                s = p[..., 1:]
            else:
                s = logmap(self._origin_like(p), p)[..., 1:]
            out.append(s.transpose(1, 2).to(self.out_dtype).contiguous())
        return out

    def _resolve_null_indices(self) -> None:
        """Find, per level, the grid cell that decodes to the ZERO step.

        The search uses `decode_indices` only, so it has no side effects: a
        probe through `forward` would touch the history `used_cells` counts.
        """
        self._null_index, self._null_norm, self._null_source = [], [], []
        for q in self.tangent_quantizers:
            idx = None
            for attr in ("null_index", "zero_index"):
                val = getattr(q, attr, None)
                if isinstance(val, int):
                    idx = int(val)
                    self._null_source.append("attr:" + attr)
                    break
            if idx is None:
                size = None
                for attr in ("n_codes", "codebook_size", "num_codes",
                             "n_cells", "size"):
                    val = getattr(q, attr, None)
                    if isinstance(val, int) and val > 0:
                        size = int(val)
                        break
                if size is not None and size <= 4096:
                    try:
                        with torch.no_grad():
                            probe = torch.arange(size).view(1, size)
                            dec = q.decode_indices(probe)
                            norms = dec.squeeze(0).norm(dim=0)
                            idx = int(norms.argmin())
                        self._null_source.append("probe")
                    except Exception:
                        idx = None
            if idx is None:
                self._null_source.append("unresolved")
                self._null_index.append(None)
                self._null_norm.append(float("nan"))
                continue
            self._null_index.append(idx)
            try:
                with torch.no_grad():
                    dec = q.decode_indices(torch.tensor([[idx]]))
                    self._null_norm.append(float(dec.norm()))
            except Exception:
                self._null_norm.append(float("nan"))

    def null_indices(self) -> List[Optional[int]]:
        """Per refinement level, the index meaning 'no refinement'."""
        return list(self._null_index)

    def prefix_indices(self, indices: Sequence[torch.Tensor],
                       keep: int) -> List[torch.Tensor]:
        """Replace levels deeper than `keep` with the NULL code."""
        if not 0 <= keep < self.n_levels:
            raise ValueError("keep must be in [0, %d), got %d"
                             % (self.n_levels, keep))
        out = [indices[0]]
        for k in range(self.n_levels - 1):
            if k + 1 <= keep:
                out.append(indices[k + 1])
                continue
            null = self._null_index[k]
            if null is None:
                raise RuntimeError(
                    "level %d has no resolved null code, so a prefix there "
                    "would not be a legal code sequence.  Report key "
                    "`prefix_ready` says which levels are armed." % (k + 1,))
            out.append(torch.full_like(indices[k + 1], int(null)))
        return out

    def decode_prefix(self, indices: Sequence[torch.Tensor], keep: int):
        """(z, points) for the prefix keeping levels 0..keep inclusive."""
        return self.decode_indices(self.prefix_indices(indices, keep))

    def _parent_feature(self, parent: torch.Tensor) -> torch.Tensor:
        """Parent point -> [B, dim, L] in the TANGENT space at the origin.

        `logmap(origin, p)` has norm r rather than sinh(r): the same
        information on a scale that does not explode with the radius.
        """
        t = logmap(self._origin_like(parent), parent)[..., 1:]
        return t.transpose(1, 2).to(self.out_dtype).contiguous()

    @torch.no_grad()
    def _observe_pred(self, k: int, feat: torch.Tensor,
                      pred: torch.Tensor) -> None:
        """Predictor falsifiers, computed without a second forward pass.

        `pred_r2` must rise above the linear 0.2043 / 0.0385 / 0.0424 / 0.0658;
        `pred_share` drifting to 1.0 while `used_cells` falls is the opposite
        collapse, onto the feature itself.
        """
        f = feat.detach().double()
        p = pred.detach().double()
        var = f.var(unbiased=False)
        res = (f - p).var(unbiased=False)
        self._pred_diag_stats["pred_r2_l%d" % (k + 1)] = float(
            1.0 - (res / var.clamp_min(1e-30)))
        self._pred_diag_stats["pred_share_l%d" % (k + 1)] = float(
            p.pow(2).mean().sqrt() / f.pow(2).mean().sqrt().clamp_min(1e-30))
        self._pred_diag_stats["pred_res_rms_l%d" % (k + 1)] = float(
            (f - p).pow(2).mean().sqrt())
        self._pred_diag_stats["pred_feat_rms_l%d" % (k + 1)] = float(
            f.pow(2).mean().sqrt())

    def _check_maps(self, maps: Sequence[torch.Tensor]) -> None:
        if len(maps) != self.n_levels:
            raise ValueError("expected %d maps, got %d"
                             % (self.n_levels, len(maps)))
        last = 0
        for k, m in enumerate(maps):
            if m.dim() != 3:
                raise ValueError("map %d must be [B, C, L], got %r"
                                 % (k, tuple(m.shape)))
            if m.shape[1] != self.in_channels[k]:
                raise ValueError("map %d has %d channels, config says %d"
                                 % (k, m.shape[1], self.in_channels[k]))
            if m.shape[-1] < last:
                raise ValueError(
                    "maps must be ordered COARSEST FIRST; map %d has length %d "
                    "after %d" % (k, m.shape[-1], last))
            last = m.shape[-1]

    def forward(self, maps: Sequence[torch.Tensor]) -> LadderOutput:
        """`maps`: per level, [B, C_k, L_k], COARSEST FIRST."""
        self._check_maps(maps)
        self._up_mode = ["identity"] * (self.n_levels - 1)
        self._calls += 1

        sub_stats: List[dict] = []
        step_stats: List[dict] = []
        pred_losses: List[torch.Tensor] = []
        self._pred_diag_stats = {}
        losses: List[torch.Tensor] = []
        loss_sources: List[str] = []

        # Level 0: POSITION
        q_out = self.position_quantizer(self.to_code(maps[0]))
        q0, idx0, st0 = self._unpack(q_out, 0)
        sub_stats.append(st0)
        self._collect_loss(st0, self.position_quantizer, losses, loss_sources)

        # Two gradient routes to one value. `quantized` carries the
        # straight-through gradient to the encoder and, by construction, none
        # to the codebook; `decode_indices(idx)` has the same value and carries
        # the gradient to the codebook, their difference term being zero.
        p_code = self._points_from_map(self.position_quantizer.decode_indices(idx0))
        p_st = self._points_from_map(q0)
        point = p_code + (p_st - p_st.detach())

        points = [point]
        indices = [idx0]
        cone_loss = point.new_zeros(())

        # Levels 1..K-1: REFINEMENT
        for k, q in enumerate(self.tangent_quantizers):
            child_len = maps[k + 1].shape[-1]
            parent = self._upsample(points[-1], child_len, k)
            feat = self.to_tangent[k](maps[k + 1])
            pred = None
            if self.residual_pred is not None:
                pred = self.residual_pred[k](self._parent_feature(parent))
                # `1 - R2`, not MSE: MSE scales with the encoder's own feature
                # scale, so one weight gave pred_r2 0.026 at std_in 0.14 and
                # 0.195 at std_in 5.1.
                _cel = feat.detach()
                _war = _cel.var(unbiased=False).clamp_min(1e-12)
                pred_losses.append(F.mse_loss(pred, _cel) / _war)
                self._observe_pred(k, feat, pred)
                feat = feat - pred
            qv, idx, st = self._unpack(q(feat), k + 1)
            sub_stats.append(st)
            self._collect_loss(st, q, losses, loss_sources)
            # Add the prediction back on the synthesis side: without it the
            # subtraction only reparametrises the encoder, removing no bits.
            code_map = qv if pred is None else qv + pred.to(qv.dtype)
            child, s_stats = self._step_from_code_map(parent, code_map, k)
            step_stats.append(s_stats)
            points.append(child)
            indices.append(idx)
            if self.cone_weight > 0.0:
                cone_loss = cone_loss + self._cone_penalty(parent, child)

        loss = None
        if losses:
            loss = losses[0]
            for extra in losses[1:]:
                loss = loss + extra
        if self.cone_weight > 0.0:
            cone = self.cone_weight * cone_loss
            loss = cone if loss is None else loss + cone
        if pred_losses and self.pred_weight > 0.0:
            # Mean over levels, not sum, or the weight would mean something
            # different at every ladder depth.
            pl = pred_losses[0]
            for extra in pred_losses[1:]:
                pl = pl + extra
            pl = (self.pred_weight / float(len(pred_losses))) * pl
            loss = pl if loss is None else loss + pl

        z = self._z_from_points(points)
        self._stats = self.report(points=points, indices=indices,
                                  sub_stats=sub_stats, step_stats=step_stats,
                                  loss_sources=loss_sources)
        return LadderOutput(z=z, indices=indices, points=points,
                            stats=dict(self._stats),
                            total_bits=float(self._stats.get("bits_total",
                                                             float("nan"))),
                            loss=loss,
                            cone_loss=(self.cone_weight * cone_loss
                                       if self.cone_weight > 0.0 else cone_loss))

    @staticmethod
    def _unpack(out, level: int):
        """(quantized, indices, stats) with a named error, not an unpack crash."""
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise TypeError(
                "quantiser at level %d returned %r; the contract is "
                "(quantized, indices, stats)" % (level, type(out).__name__))
        q, idx, stats = out
        return q, idx, (stats if isinstance(stats, dict) else {})

    @staticmethod
    def _collect_loss(stats: dict, module: nn.Module,
                      losses: List[torch.Tensor], sources: List[str]) -> None:
        """Pick up a commitment loss, and RECORD when there is none.

        A missing commitment term neither raises nor shows in the metrics.
        """
        val = stats.get("loss", None)
        if val is None:
            val = getattr(module, "loss", None)
            src = "attr" if isinstance(val, torch.Tensor) else "none"
        else:
            src = "stats"
        if isinstance(val, torch.Tensor):
            losses.append(val)
        else:
            src = "none"
        sources.append(src)

    def _cone_penalty(self, parent: torch.Tensor,
                      child: torch.Tensor) -> torch.Tensor:
        """relu(angle - aperture)^2 on the ANGULAR component of the step.

        The radial component is unconstrained on purpose: it carries a level
        onto a deeper shell, and penalising it would undo the stratification.
        """
        angle = self._angle(parent, child)
        return F.relu(angle - self.aperture).pow(2).mean()

    def _angle(self, parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        v = logmap(parent, child)
        rad = self._radial(parent)
        tangential = v - minkowski_dot(v, rad, keepdim=True) * rad
        n_t = torch.sqrt(minkowski_dot(tangential, tangential,
                                       keepdim=True).clamp_min(1e-30))
        sinh_r = torch.sinh(radius(parent).unsqueeze(-1)).clamp_min(1e-6)
        return (n_t / sinh_r).squeeze(-1)

    def decode_indices(self, indices: Sequence[torch.Tensor]):
        """(z list, points list) from INDICES ALONE -- the diffusion contract.

        No batch statistics, no encoder quantity, no continuous slot. Checked
        by `self_test`, because a broken decode looks exactly like a working
        one until the reconstruction is compared bit for bit.
        """
        if len(indices) != self.n_levels:
            raise ValueError("expected %d index tensors, got %d"
                             % (self.n_levels, len(indices)))
        self._up_mode = ["identity"] * (self.n_levels - 1)
        point = self._points_from_map(
            self.position_quantizer.decode_indices(indices[0]))
        points = [point]
        for k, q in enumerate(self.tangent_quantizers):
            child_len = indices[k + 1].shape[-1]
            parent = self._upsample(points[-1], child_len, k)
            qv = q.decode_indices(indices[k + 1])
            if self.residual_pred is not None:
                qv = qv + self.residual_pred[k](
                    self._parent_feature(parent)).to(qv.dtype)
            child, _ = self._step_from_code_map(parent, qv, k)
            points.append(child)
        return self._z_from_points(points), points

    def manifold_parameters(self) -> List[nn.Parameter]:
        """Parameters living ON the hyperboloid -- for `RiemannianAdam`.

        Exposed explicitly, not matched by name: the exemption pattern matches
        a name SUBSTRING and has already lost a tensor to that here.
        """
        fn = getattr(self.position_quantizer, "manifold_parameters", None)
        if callable(fn):
            return list(fn())
        return [p for p in self.position_quantizer.parameters()
                if getattr(p, "is_manifold", False)]

    def runaway_report(self) -> dict:
        """Level-0 lift observables, bypassing `report`'s `wanted` whitelist.

        That whitelist once kept `term_commitment` out of 40 epochs of log.
        """
        fn = getattr(self.position_quantizer, "runaway_report", None)
        return dict(fn()) if callable(fn) else {}

    def no_decay_parameters(self) -> List[nn.Parameter]:
        """Exempt from weight decay -- it destroys hyperboloid points."""
        return self.manifold_parameters()

    def report(self, points=None, indices=None, sub_stats=None,
               step_stats=None, loss_sources=None) -> dict:
        """Falsifiers. NEVER raises -- a caught exception loses the whole line.

        `manifold_err` is max |<x,x>_L + 1| per level, tolerance around 1e-16;
        EMA over ambient coordinates once left it at 2.49e-01 while a reported
        reconstruction of 0.9945 was really 0.8012. `d_ratio` is containment,
        d(child, a random other parent) over d(child, its own parent), where
        1.000 means none; `crossover` is its sharp form; and `step_spread` is
        d_parent over the spread of the parent cloud, above 1.0 of which a
        child can reach any parent at all. `bits` is achieved entropy against
        the declared 3187-bit budget, over 1598.7 measured bits of sequence, so
        a latent that reconstructs badly with 1.99x of headroom has an
        optimisation rather than a capacity problem. `cone_can_fire` is 1 when
        `outside_cone` is answerable: under a step clamp tighter than
        aperture*sinh(r) it is zero arithmetically, and a 0.0000 then means the
        question cannot be asked.
        """
        points = points if points is not None else []
        indices = indices if indices is not None else []
        sub_stats = sub_stats or []
        step_stats = step_stats or []
        r: dict = {
            "levels": self.n_levels,
            "lengths": [int(p.shape[1]) for p in points],
            "aperture": round(self.aperture, 4),
            "cone_weight": round(self.cone_weight, 4),
            "step_limit": [round(float(v), 4) for v in self.step_length],
            "radius_step_nominal": [round(float(v), 4) for v in self.radius_step],
            "lift_convention": self.lift_convention,
            "point_layout": getattr(self, "_point_layout", "unknown"),
            "upsample_mode": list(getattr(self, "_up_mode", [])),
            "loss_sources": list(loss_sources or []),
            "z_mode": self.z_mode,
            # The alphabet a level can really use: without it a run with half
            # its digits frozen looks identical to one with none frozen.
            "cyfry_czynne": list(getattr(self, "cyfry_czynne", [])),
            "alfabet_czynny": [int(q.alfabet_czynny())
                               if hasattr(q, "alfabet_czynny")
                               else int(getattr(q, "codebook_size", 0))
                               for q in self.tangent_quantizers],
            "cyfry_zrodlo": getattr(self, "_cyfry_zrodlo", "domyslne"),
        }
        if not points:
            return r

        audit = (self._calls % self.audit_every) == 0

        # Structural: on the manifold, per level
        errs = []
        for p in points:
            try:
                # tol=1.0 on purpose: the report READS the residual and must
                # not raise on it. The raising check lives in `self_test`.
                errs.append(float(assert_on_manifold(p.detach(), tol=1.0)))
            except Exception as exc:            # never lose the line
                errs.append(float("nan"))
                r.setdefault("report_errors", []).append(
                    "manifold:" + type(exc).__name__)
        r["manifold_err"] = [float("%.3e" % e) for e in errs]

        # Radius: plan vs measurement
        try:
            meas = [float(radius(p.detach()).mean()) for p in points]
            r["radius"] = [round(v, 4) for v in meas]
            plan = [meas[0]]
            for k in range(self.n_levels - 1):
                plan.append(plan[-1] + float(self.radius_step[k]))
            r["radius_plan"] = [round(v, 4) for v in plan]
            r["radius_gap"] = [round(plan[i] - meas[i], 4)
                               for i in range(len(meas))]
            r["radius_step_measured"] = [round(meas[i + 1] - meas[i], 4)
                                         for i in range(len(meas) - 1)]
        except Exception as exc:
            r.setdefault("report_errors", []).append(
                "radius:" + type(exc).__name__)

        # Per-step aperture usage
        for key in ("step_norm_in", "step_norm_out", "step_clamped_frac",
                    "aperture_use"):
            vals = [s.get(key, float("nan")) for s in step_stats]
            if vals:
                r[key] = [round(float(v), 4) for v in vals]

        # Quantiser sub-reports
        self._merge_sub_reports(r, sub_stats, points)

        # Containment, cones, cosines: the expensive block
        if audit:
            try:
                self._containment(r, points)
            except Exception as exc:
                r.setdefault("report_errors", []).append(
                    "containment:" + type(exc).__name__)

        # Prefix machinery
        try:
            r["null_index"] = [(-1 if i is None else int(i))
                               for i in self._null_index]
            r["null_step_norm"] = [float("%.3e" % v) for v in self._null_norm]
            r["null_source"] = list(getattr(self, "_null_source", []))
            r["prefix_ready"] = [
                int(self._null_index[k] is not None
                    and not math.isnan(self._null_norm[k])
                    and self._null_norm[k] < 1e-6)
                for k in range(self.n_levels - 1)]
            r["reconstruction_floor"] = RECONSTRUCTION_FLOOR
            if indices:
                r["null_frac"] = [
                    (round(float((indices[k + 1] == self._null_index[k])
                                 .to(torch.float64).mean()), 4)
                     if self._null_index[k] is not None else float("nan"))
                    for k in range(self.n_levels - 1)]
        except Exception as exc:
            r.setdefault("report_errors", []).append(
                "prefix:" + type(exc).__name__)

        # Level 0: is there a gradient route to the codebook?
        try:
            emb = getattr(self.position_quantizer, "embed", None)
            r["l0_codebook_is_param"] = int(isinstance(emb, nn.Parameter))
            r["l0_codebook_grad"] = int(bool(
                isinstance(emb, torch.Tensor) and emb.requires_grad))
            r["l0_manifold_params"] = len(self.manifold_parameters())
        except Exception as exc:
            r.setdefault("report_errors", []).append(
                "l0:" + type(exc).__name__)

        # Scalars. `assemble` passes only int/float entries into `out["stats"]`,
        # so without this block the per-level lists never reach the epoch line
        # and a run produces no number at all about coding overhead.
        try:
            lens = r.get("lengths", [])
            # L0 has no `bits` key, the hyperbolic codebook not computing one,
            # so `bits_total` would otherwise skip the level the argument is
            # about. log2(perplexity) is the FSQ `bits` convention.
            _b = list(r.get("bits", []))
            _px = list(r.get("perplexity", []))
            r["bits_l0_z_perplexity"] = 0
            if (_b and math.isnan(_b[0]) and _px
                    and not math.isnan(_px[0]) and _px[0] > 0.0):
                _b[0] = round(math.log2(float(_px[0])), 4)
                r["bits"] = _b
                r["bits_l0_z_perplexity"] = 1
                if lens and len(_b) == len(lens):
                    r["bits_total"] = round(
                        float(sum(l * b for l, b in zip(lens, _b)
                                  if not math.isnan(b))), 1)
                    if r.get("sequence_bits"):
                        r["bits_headroom"] = round(
                            r["bits_total"] / float(r["sequence_bits"]), 3)
            for key in ("bits", "rate_bits", "used_frac", "perplexity",
                        "saturation", "zero_frac"):
                for k, v in enumerate(r.get(key, [])):
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        r["%s_l%d" % (key, k)] = round(float(v), 4)
            for k, v in enumerate(r.get("bits", [])):
                if (k < len(lens) and isinstance(v, (int, float))
                        and not math.isnan(v)):
                    r["bits_total_l%d" % k] = round(float(v) * lens[k], 1)
            if "bits_headroom" in r:
                # `bits_headroom` again, under the name the falsifier was
                # written against; today 1.59-1.64 and expected to fall.
                r["r_code"] = r["bits_headroom"]
            r["pred_on"] = int(self.residual_pred is not None)
            r["pred_weight"] = float(self.pred_weight)
            r.update(self._pred_diag_stats)
        except Exception as exc:
            r.setdefault("report_errors", []).append(
                "scalars:" + type(exc).__name__)
        return r

    def _merge_sub_reports(self, r: dict, sub_stats: List[dict],
                           points: List[torch.Tensor]) -> None:
        """Pull `bits`, `perplexity`, `saturation`, ... out of the quantisers.

        Every absence is NAMED in `missing_report_keys`: a renamed key in a
        sibling would otherwise remove a falsifier and nothing else.
        """
        wanted = ("bits", "target_bits", "perplexity", "saturation",
                  "zero_frac", "used_cells", "used_frac", "dead_restarts",
                  "rate_bits")
        missing: List[str] = []
        reports: List[dict] = []
        for k, stats in enumerate(sub_stats):
            module = (self.position_quantizer if k == 0
                      else self.tangent_quantizers[k - 1])
            rep = dict(stats)
            fn = getattr(module, "report", None)
            if callable(fn):
                try:
                    got = fn()
                    if isinstance(got, dict):
                        rep.update(got)
                except Exception as exc:
                    r.setdefault("report_errors", []).append(
                        "sub%d:%s" % (k, type(exc).__name__))
            reports.append(rep)
            for key in wanted:
                if key not in rep:
                    missing.append("L%d.%s" % (k, key))
        for key in wanted:
            vals = [rep.get(key, None) for rep in reports]
            if any(v is not None for v in vals):
                r[key] = [(round(float(v), 4) if isinstance(v, (int, float))
                           else float("nan")) for v in vals]
        r["missing_report_keys"] = missing

        # Both achieved and declared bits, because meeting the declared budget
        # while achieving half of it is the normal failure here.
        try:
            lens = [int(p.shape[1]) for p in points]
            bits = r.get("bits", [])
            tgt = r.get("target_bits", [])
            if bits and len(bits) == len(lens):
                r["bits_total"] = round(
                    float(sum(l * b for l, b in zip(lens, bits)
                              if not math.isnan(b))), 1)
            if tgt and len(tgt) == len(lens):
                r["bits_target_total"] = round(
                    float(sum(l * b for l, b in zip(lens, tgt)
                              if not math.isnan(b))), 1)
            if self.seq_len is not None:
                seq_bits = UNIGRAM_BITS_PER_BP * float(self.seq_len)
                r["sequence_bits"] = round(seq_bits, 1)
                if "bits_total" in r:
                    r["bits_headroom"] = round(r["bits_total"] / seq_bits, 3)
        except Exception as exc:
            r.setdefault("report_errors", []).append(
                "bits:" + type(exc).__name__)

    @torch.no_grad()
    def _containment(self, r: dict, points: List[torch.Tensor]) -> None:
        """Containment, step/spread, crossover, cone angles, cosines.

        The cosines are kept because they are in the run history, but they are
        read last and are unreliable; see the module docstring.
        """
        keys = ("d_parent", "d_random", "d_ratio", "spread", "spread_pair",
                "step_spread", "crossover", "angle", "angle_rel", "angle_max",
                "angle_ceiling", "outside_cone", "cone_can_fire",
                "cos_parent", "cos_random", "cos_parent_centred",
                "cos_random_centred")
        for key in keys:
            r[key] = []

        for k in range(len(points) - 1):
            child = points[k + 1].detach()
            parent = self._upsample(points[k].detach(), child.shape[1], k)
            length = parent.shape[1]

            # Random pairing from a private stream: the report must not advance
            # the global RNG the run's reproducibility depends on.
            perm = torch.randperm(length, generator=self._rng).to(parent.device)
            fixed = perm == torch.arange(length, device=parent.device)
            keep = ~fixed if bool((~fixed).any()) else torch.ones_like(fixed)
            other = parent[:, perm]

            d_par = geodesic_distance(parent, child)
            d_rnd = geodesic_distance(other, child)
            d_par_m = float(d_par.mean())
            d_rnd_m = float(d_rnd[:, keep].mean())
            r["d_parent"].append(round(d_par_m, 4))
            r["d_random"].append(round(d_rnd_m, 4))
            r["d_ratio"].append(round(d_rnd_m / max(d_par_m, 1e-9), 3))
            # The sharp form: can a child reach a different parent?
            r["crossover"].append(round(
                float((d_rnd[:, keep] < d_par[:, keep]).to(torch.float64).mean()), 4))

            # spread = radius of the parent cloud, mean geodesic distance to
            # their Frechet mean; the pairwise version is reported beside it.
            spread = float("nan")
            try:
                flat = parent.reshape(-1, parent.shape[-1])
                centre = frechet_mean(flat)
                if centre.numel() == parent.shape[-1]:
                    centre = centre.reshape(*([1] * (parent.dim() - 1)),
                                            parent.shape[-1])
                    spread = float(geodesic_distance(
                        parent, centre.expand_as(parent)).mean())
            except Exception as exc:
                r.setdefault("report_errors", []).append(
                    "spread:" + type(exc).__name__)
            pair = float(geodesic_distance(parent, other)[:, keep].mean())
            r["spread"].append(round(spread, 4))
            r["spread_pair"].append(round(pair, 4))
            r["step_spread"].append(
                round(d_par_m / spread, 3) if spread == spread and spread > 1e-9
                else float("nan"))

            # cone angle, and whether the question is answerable at all
            angle = self._angle(parent, child)
            ap = max(self.aperture, 1e-12)
            r["angle"].append(round(float(angle.mean()), 4))
            r["angle_rel"].append(round(float(angle.mean()) / ap, 4))
            r["angle_max"].append(round(float(angle.max()) / ap, 4))
            r["outside_cone"].append(round(
                float((angle > self.aperture).to(torch.float64).mean()), 4))
            sinh_min = float(torch.sinh(radius(parent)).min().clamp_min(1e-6))
            ceiling = float(self.step_length[k]) / sinh_min
            r["angle_ceiling"].append(round(ceiling, 4))
            r["cone_can_fire"].append(
                int(ceiling > self.aperture and self.cone_weight > 0.0))

            # cosines: addendum only
            ka = parent[..., 1:]
            ka = ka / ka.norm(dim=-1, keepdim=True).clamp_min(_EPS)
            kb = child[..., 1:]
            kb = kb / kb.norm(dim=-1, keepdim=True).clamp_min(_EPS)
            r["cos_parent"].append(round(float((ka * kb).sum(-1).mean()), 4))
            r["cos_random"].append(
                round(float((ka[:, perm] * kb).sum(-1).mean()), 4))
            cac = self._centre(ka)
            kbc = self._centre(kb)
            r["cos_parent_centred"].append(
                round(float((cac * kbc).sum(-1).mean()), 4))
            r["cos_random_centred"].append(
                round(float((cac[:, perm] * kbc).sum(-1).mean()), 4))

    @staticmethod
    def _centre(t: torch.Tensor) -> torch.Tensor:
        """Remove the component common to all positions, then renormalise.

        Encoder maps share a strong component across positions, so a randomly
        paired 'parent' points almost the same way and the raw control
        saturates at 0.9597 / 0.9019.
        """
        u = t - t.mean(dim=(0, 1), keepdim=True)
        return u / u.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    @torch.no_grad()
    def self_test(self, maps: Sequence[torch.Tensor], tol: float = 1e-9) -> dict:
        """Structural gate. Run it BEFORE a run, on the training-mode model.

        `decode_match` requires the decode path to equal the forward points bit
        for bit, and `positive_control` perturbs one index and requires the
        points to move -- without it the first check passes trivially on an
        untrained model, which here returns a constant.
        """
        out: dict = {}
        # `forward` increments `_calls` (the `audit_every` counter that gates
        # the containment block) and overwrites `_stats`. A gate run before
        # training would therefore shift which epochs get a containment
        # report, and the reads below would see this call's statistics rather
        # than the run's. Both are saved and put back.
        _calls_before, _stats_before = self._calls, self._stats
        try:
            fwd = self.forward(maps)
            _, dec_points = self.decode_indices(fwd.indices)
        finally:
            self._calls, self._stats = _calls_before, _stats_before
        deltas = [float((a.double() - b.double()).abs().max())
                  for a, b in zip(fwd.points, dec_points)]
        out["decode_match"] = [float("%.3e" % d) for d in deltas]
        out["decode_ok"] = int(max(deltas) <= tol)

        # Positive control: move ONE index on the deepest level.
        probe = [i.clone() for i in fwd.indices]
        deep = probe[-1]
        if deep.numel() > 0:
            size = None
            q = self.tangent_quantizers[-1] if self.n_levels > 1 else \
                self.position_quantizer
            for attr in ("n_codes", "codebook_size", "num_codes", "n_cells"):
                val = getattr(q, attr, None)
                if isinstance(val, int) and val > 1:
                    size = val
                    break
            modulus = size if size is not None else int(deep.max()) + 2
            deep[0, 0] = (int(deep[0, 0]) + 1) % modulus
            _, moved = self.decode_indices(probe)
            delta = float((moved[-1].double() - dec_points[-1].double())
                          .abs().max())
            out["positive_control"] = float("%.3e" % delta)
            out["positive_control_ok"] = int(delta > tol)

        out["manifold_err"] = [float("%.3e" % float(assert_on_manifold(p, tol=1.0)))
                               for p in fwd.points]
        out["null_step_norm"] = [float("%.3e" % v) for v in self._null_norm]
        # From this call's own output, not from `self._stats`: the counters
        # are restored above, so the attribute no longer holds them.
        out["prefix_ready"] = (fwd.stats or {}).get("prefix_ready", [])
        try:
            if all(i is not None for i in self._null_index):
                _, pre = self.decode_prefix(fwd.indices, 0)
                out["prefix_manifold_err"] = [
                    float("%.3e" % float(assert_on_manifold(p, tol=1.0)))
                    for p in pre]
                out["prefix_moves"] = float("%.3e" % float(
                    (pre[-1].double() - dec_points[-1].double()).abs().max()))
        except Exception as exc:
            out["prefix_error"] = type(exc).__name__
        return out

    def extra_repr(self) -> str:
        return ("levels=%d, dim=%d, step_length=%s, radius_step=%s, "
                "aperture=%.3f, cone_weight=%.3f, cyfry_czynne=%s"
                % (self.n_levels, self.dim,
                   [round(float(v), 3) for v in self.step_length],
                   [round(float(v), 3) for v in self.radius_step],
                   self.aperture, self.cone_weight,
                   list(getattr(self, "cyfry_czynne", []))))
