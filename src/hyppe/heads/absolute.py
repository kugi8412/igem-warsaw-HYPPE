"""Absolute promoter-strength head: mu, predictive log-variance, measured sigma.

A plain heteroscedastic Gaussian NLL diverged here in three epochs (loss
1.3e6 -> 9.5e7 -> 6.5e8, PCC -0.007), and not through a bug: every residual
enters as (y-mu)^2 / sigma^2, so lowering sigma on the easy samples raises the
relative weight of the hard ones, whose sigma then has to grow. beta-NLL
(Seitzer et al., ICLR 2022) multiplies the per-sample NLL by a DETACHED power
of the predicted variance, `stopgrad(var ** beta) * ((y - mu)^2 / (2 var) +
0.5 * log var)`, with beta = 0 the plain NLL that blew up, beta = 1 an MSE
gradient on mu, and beta = 0.5 the default here. Sigma still scales the mu
gradient, var ** (beta - 1) * (mu - y), so a sample the head is already
confident about pulls harder than one it is not -- a route to chasing the
training set that val SCC, which sees only the ordering of mu, cannot report.
`beta_mu` is a second exponent on the mu path alone. Detaching `var` inside
`resid.pow(2)/var` would not do the same job: that term is the only thing
pulling log_var towards the squared error, so the residual is the operand to
detach, not the variance.

`log_var` and `log_sigma_obs` get separate output layers over a shared trunk
because they are different objects: the spread of the MODEL's own error, and
a measured label, the standard deviation of `combat` across the kept
conditions of one (orthogroup, genome) pair. A promoter can be measured
reliably and still be strongly condition-dependent, and tying the two would
make the beta-NLL weight a function of a label. `sigma` is NaN wherever
n_obs < 2 (`build_targets(min_obs=2)`), so the second term carries its own
mask.

Masks multiply a finite operand, never a result: `torch.where(mask, f(y), 0)`
still evaluates f'(y) at the masked-out entries, and NaN * 0 = NaN, so the NaN
reaches the shared trunk while the forward loss prints a number. Targets go
through `sanitise` first and the mask becomes a weight in `masked_mean`.

The primitives the other heads share -- pooling, conditioning, reservoir
sampling, streaming moments -- live here because the heads layer is three
files by contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

__all__ = [
    "PoolingSlice",
    "LatentPooling",
    "ConditionedTrunk",
    "Reservoir",
    "StreamingMoments",
    "masked_mean",
    "sanitise",
    "average_ranks",
    "pearson",
    "spearman",
    "as_features",
    "AbsoluteOutput",
    "AbsoluteStrengthHead",
]


# Masks
def sanitise(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf in a TARGET before it is used in any arithmetic.

    The operand-side half of masking: without it a NaN label produces a NaN
    gradient even when its loss term is multiplied by a zero mask.
    """
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of `values` over `mask`, optionally sample-weighted.

    The denominator is the mask (times the weight), so switching examples off
    does not quietly rescale the loss: with a constant denominator a batch
    with few valid targets would produce a smaller gradient than a full one,
    which is a learning-rate schedule nobody wrote down.
    """
    if mask is None:
        m = torch.ones_like(values)
    else:
        m = mask.to(values.dtype)
    if weight is not None:
        m = m * weight.to(values.dtype)
    denom = m.sum().clamp_min(1e-8)
    return (values * m).sum() / denom


# Pooling
@dataclass(frozen=True)
class PoolingSlice:
    """Which columns of the pooled vector came from which level and statistic."""

    level: int
    stat: str
    start: int
    stop: int


class LatentPooling(nn.Module):
    """`list of [B, C_k, L_k]` -> `[B, D]`, with the statistics DECLARED.

    `sd` carries real signal next to `mean` (a linear probe gained +0.0672 /
    +0.0285 on the two coarsest trunk blocks at a NEGATIVE capacity control).
    `max` is available but not default, having measured worse than the mean on
    every block; the 81x advantage of max quoted elsewhere in this project was
    measured on PWM occupancy, not on a dense feature map. `anchor` is the
    mean over the last `anchor_frac`, where position L is the base
    immediately before the ATG and so is the core promoter.

    The maps are expected COARSEST FIRST, matching `ConeLadder`, and with the
    position index increasing towards the ATG, matching `sequences.py`.
    """

    _STATS = ("mean", "sd", "max", "anchor")

    def __init__(self, channels: Sequence[int],
                 stats: Sequence[str] = ("mean", "sd", "anchor"),
                 anchor_frac: float = 0.125,
                 eps: float = 1e-12) -> None:
        super().__init__()
        if not channels:
            raise ValueError("LatentPooling needs at least one level")
        bad = [s for s in stats if s not in self._STATS]
        if bad:
            raise ValueError("unknown pooling statistic(s) %r; known: %r"
                             % (bad, self._STATS))
        if not stats:
            raise ValueError("empty `stats` would pool to a zero-width vector")
        if not 0.0 < anchor_frac <= 1.0:
            raise ValueError("anchor_frac must be in (0, 1], got %r"
                             % (anchor_frac,))
        self.channels = tuple(int(c) for c in channels)
        self.stats = tuple(stats)
        self.anchor_frac = float(anchor_frac)
        self.eps = float(eps)

    @property
    def out_dim(self) -> int:
        return sum(self.channels) * len(self.stats)

    def slices(self) -> List[PoolingSlice]:
        """Column ranges per (level, statistic). Used by `attribution`."""
        out: List[PoolingSlice] = []
        at = 0
        for k, c in enumerate(self.channels):
            for s in self.stats:
                out.append(PoolingSlice(k, s, at, at + c))
                at += c
        return out

    def forward(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(levels):
            levels = [levels]
        levels = list(levels)
        if len(levels) != len(self.channels):
            raise ValueError("expected %d levels, got %d"
                             % (len(self.channels), len(levels)))
        parts: List[torch.Tensor] = []
        for k, x in enumerate(levels):
            if x.dim() != 3:
                raise ValueError("level %d must be [B, C, L], got %r"
                                 % (k, tuple(x.shape)))
            if x.shape[1] != self.channels[k]:
                raise ValueError("level %d has %d channels, config says %d"
                                 % (k, x.shape[1], self.channels[k]))
            mean = x.mean(dim=-1)
            for s in self.stats:
                if s == "mean":
                    parts.append(mean)
                elif s == "sd":
                    dev = x - mean.unsqueeze(-1)
                    # clamp_min before sqrt: the derivative of sqrt at exactly
                    # 0 is infinite, and a constant channel (a dead code, say)
                    # lands exactly on 0 here rather than near it.
                    parts.append(dev.pow(2).mean(dim=-1)
                                 .clamp_min(self.eps).sqrt())
                elif s == "max":
                    parts.append(x.amax(dim=-1))
                else:  # anchor
                    w = max(1, int(round(self.anchor_frac * x.shape[-1])))
                    parts.append(x[..., -w:].mean(dim=-1))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def attribution(self, first_layer: nn.Linear) -> Dict[str, float]:
        """Mean column norm of the first linear layer, per (level, statistic).

        A block whose norm collapses is a part of the latent the head stopped
        reading. Xavier gives every block a comparable non-zero norm at
        initialisation, so a near-zero entry is starvation during training and
        not an artefact of a zero-initialised branch.
        """
        w = first_layer.weight.detach()
        out: Dict[str, float] = {}
        for sl in self.slices():
            block = w[:, sl.start:sl.stop]
            out["L%d/%s" % (sl.level, sl.stat)] = float(
                block.norm(dim=0).mean())
        return out


def as_features(latent, pooling: Optional[LatentPooling] = None
                ) -> torch.Tensor:
    """Accept either a pooled `[B, D]` vector or a list of level maps."""
    if torch.is_tensor(latent):
        if latent.dim() == 2:
            return latent
        if latent.dim() == 3:
            if pooling is None:
                raise ValueError("a [B, C, L] map needs a LatentPooling")
            return pooling([latent])
        raise ValueError("latent must be [B, D] or [B, C, L], got %r"
                         % (tuple(latent.shape),))
    if pooling is None:
        raise ValueError("a list of level maps needs a LatentPooling")
    return pooling(list(latent))


# Conditioning
class ConditionedTrunk(nn.Module):
    """Shared MLP trunk with an explicit conditioning mode.

    `mode="concat"` is the default because a multiplicative gate orphans its
    own source: a near-zero gate passes no gradient upward and starves the
    features that would open it. When FiLM is chosen anyway the projection is
    initialised small but NON-ZERO (xavier gain 0.1, gamma bias 1.0), since a
    zero initialisation makes d(output)/d(context) identically zero and a
    positive control on the channel then passes while it is dead.

    `center` applies `BatchNorm1d(affine=False)` to the context, worth 12x the
    per-sequence content for zero parameters: a LayerNorm normalises across
    features and leaves ~91 % of the projected norm in the direction shared
    between samples. In eval the running statistics are used, so the
    prediction stays a function of the single sequence.
    """

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, context_dim: int = 0,
                 mode: str = "concat", center: bool = True) -> None:
        super().__init__()
        if mode not in ("concat", "film", "none"):
            raise ValueError("mode must be concat|film|none, got %r" % (mode,))
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.context_dim = int(context_dim)
        self.mode = "none" if self.context_dim == 0 else mode
        self.hidden = int(hidden)
        self.center = (nn.BatchNorm1d(self.context_dim, affine=False)
                       if (center and self.context_dim) else None)

        d_in = int(in_dim) + (self.context_dim if self.mode == "concat" else 0)
        layers: List[nn.Module] = []
        for i in range(depth):
            lin = nn.Linear(d_in if i == 0 else hidden, hidden)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.film: Optional[nn.Linear] = None
        if self.mode == "film":
            self.film = nn.Linear(self.context_dim, 2 * hidden)
            nn.init.xavier_uniform_(self.film.weight, gain=0.1)
            with torch.no_grad():
                self.film.bias[:hidden].fill_(1.0)    # gamma starts at identity
                self.film.bias[hidden:].zero_()       # beta starts at zero
        self.in_dim = d_in

    def first_layer(self) -> nn.Linear:
        return self.layers[0]

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode != "none":
            if context is None:
                # Silence is not a disabled channel: a configured conditioning
                # input arriving as None would be a declared-but-inert key, so
                # it raises rather than degrading quietly.
                raise ValueError(
                    "conditioning mode %r was configured but context is None"
                    % (self.mode,))
            if context.shape[-1] != self.context_dim:
                raise ValueError("context has %d dims, configured %d"
                                 % (context.shape[-1], self.context_dim))
            if self.center is not None:
                context = self.center(context)
        h = x
        for i, lin in enumerate(self.layers):
            if i == 0 and self.mode == "concat":
                h = torch.cat([h, context], dim=-1)
            h = lin(h)
            if i == 0 and self.mode == "film" and self.film is not None:
                gb = self.film(context)
                gamma, beta = gb[:, :self.hidden], gb[:, self.hidden:]
                h = h * gamma + beta
            h = self.drop(self.act(h))
        return h


# Numerics
def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties AVERAGED.

    An untrained model in eval returns a constant (measured sd 3.19e-08). With
    ties broken arbitrarily such a scorer gets AUC 1.0 or 0.0 depending on the
    sort order and a rank correlation of +-1; with averaged ranks it gets
    exactly 0.5 and NaN, which is what a falsifier is for.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or a.size != b.size:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da <= 0.0 or db <= 0.0:
        # A constant input has no correlation, not a correlation of zero.
        return float("nan")
    return float((a * b).sum() / (da * db))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or a.size != b.size:
        return float("nan")
    return pearson(average_ranks(a), average_ranks(b))


class Reservoir:
    """Uniform sample of an epoch's values, for the statistics that need them.

    Keeping the first n values is not a sample -- the first rows of this
    project's validation file were 71 % positive peaks with no negatives --
    so this is Algorithm R: every value seen has the same probability of
    being in the buffer, at O(capacity) memory and no sort.
    """

    def __init__(self, capacity: int = 20000, width: int = 1,
                 seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.width = int(width)
        self._buf = np.zeros((self.capacity, self.width), dtype=np.float64)
        self._filled = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._filled = 0
        self._seen = 0

    def update(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError("expected rows of width %d, got %r"
                             % (self.width, tuple(rows.shape)))
        m = rows.shape[0]
        if m == 0:
            return
        i = 0
        if self._filled < self.capacity:
            take = min(self.capacity - self._filled, m)
            self._buf[self._filled:self._filled + take] = rows[:take]
            self._filled += take
            self._seen += take
            i = take
        if i < m:
            rest = rows[i:]
            # Item (self._seen + t) replaces a uniformly chosen slot with
            # probability capacity/(seen+t+1). Collisions inside one batch
            # resolve last-write-wins, as the sequential algorithm does.
            prior = self._seen + np.arange(rest.shape[0])
            j = self._rng.integers(0, prior + 1)
            sel = j < self.capacity
            if bool(sel.any()):
                self._buf[j[sel]] = rest[sel]
            self._seen += rest.shape[0]

    @property
    def n_seen(self) -> int:
        return self._seen

    def values(self) -> np.ndarray:
        return self._buf[:self._filled]


class StreamingMoments:
    """Exact mean/variance/covariance of (prediction, target) without storage.

    Chan-Golub-LeVeque parallel merge, so the numbers do not depend on the
    batch size and no sum of squares is ever formed in float32.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mp = 0.0
        self.my = 0.0
        self.m2p = 0.0
        self.m2y = 0.0
        self.cpy = 0.0
        self.sse = 0.0          # sum of (pred - target)^2, for the raw MSE

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        p = np.asarray(pred, dtype=np.float64).ravel()
        y = np.asarray(target, dtype=np.float64).ravel()
        if p.size == 0:
            return
        nb = p.size
        mpb, myb = float(p.mean()), float(y.mean())
        dp, dy = p - mpb, y - myb
        m2pb, m2yb = float((dp * dp).sum()), float((dy * dy).sum())
        cpyb = float((dp * dy).sum())
        self.sse += float(((p - y) ** 2).sum())
        na = self.n
        if na == 0:
            self.n, self.mp, self.my = nb, mpb, myb
            self.m2p, self.m2y, self.cpy = m2pb, m2yb, cpyb
            return
        n = na + nb
        ddp, ddy = mpb - self.mp, myb - self.my
        self.m2p += m2pb + ddp * ddp * na * nb / n
        self.m2y += m2yb + ddy * ddy * na * nb / n
        self.cpy += cpyb + ddp * ddy * na * nb / n
        self.mp += ddp * nb / n
        self.my += ddy * nb / n
        self.n = n

    def report(self) -> Dict[str, float]:
        """PCC, the affine calibration, and where MSE sits against its floor."""
        nan = float("nan")
        if self.n < 3:
            return {"n": float(self.n), "pcc": nan, "calib_slope": nan,
                    "calib_intercept": nan, "sd_ratio": nan, "mse": nan,
                    "mse_calibrated": nan, "mse_floor": nan,
                    "pred_sd": nan, "target_sd": nan}
        vp, vy = self.m2p / self.n, self.m2y / self.n
        cov = self.cpy / self.n
        sp, sy = math.sqrt(max(vp, 0.0)), math.sqrt(max(vy, 0.0))
        r = cov / (sp * sy) if sp > 0 and sy > 0 else nan
        # Slope of the TARGET on the PREDICTION: the coefficient of the
        # post-hoc affine correction, measured at 0.84-0.89 on this family.
        # Regressing the other way answers a different question.
        slope = cov / vp if vp > 0 else nan
        intercept = self.my - slope * self.mp if vp > 0 else nan
        mse = self.sse / self.n
        mse_cal = (vy - (cov * cov / vp)) if vp > 0 else nan
        floor = vy * (1.0 - r * r) if r == r else nan
        return {"n": float(self.n), "pcc": float(r),
                "calib_slope": float(slope), "calib_intercept": float(intercept),
                "sd_ratio": float(sp / sy) if sy > 0 else nan,
                "mse": float(mse), "mse_calibrated": float(mse_cal),
                "mse_floor": float(floor),
                "pred_sd": float(sp), "target_sd": float(sy)}


# Output
@dataclass
class AbsoluteOutput:
    """What the head returns. `lv_raw` is kept for the wall falsifier."""

    mu: torch.Tensor                       # [B]
    log_var: torch.Tensor                  # [B], after the clamp
    lv_raw: torch.Tensor                   # [B], before the clamp
    log_sigma_obs: Optional[torch.Tensor]  # [B] or None
    #: [B, hidden] -- trunk output, BEFORE the columns, so that a categoriser
    #: can read the regression module rather than the description vector.
    trunk: Optional[torch.Tensor] = None
    #: [B] -- genome index of THIS forward pass, copied through from
    #: `forward`. `loss` and `observe` receive only `out` and the target, so
    #: without this field the head cannot compute a single number about the
    #: genome axis -- an axis measured at R^2 -0.0045 with slope 0.495, i.e.
    #: worse than a constant, with no trace of it in the epoch line.
    species_id: Optional[torch.Tensor] = None

    def var(self) -> torch.Tensor:
        return self.log_var.exp()

    def sigma(self) -> torch.Tensor:
        return (0.5 * self.log_var).exp()


class KalibracjaOsi(object):
    """Per-axis calibration of predictions; the measured fix for the genome axis.

    On the "genome within orthogroup" axis the model has a NEGATIVE R^2 at a
    positive correlation, which is purely amplitude: with `ap`, `ay` the
    prediction and target residuals after removing the orthogroup mean,
    R^2 = |ap|^2 (2s - 1)/|ay|^2 for s = <ap,ay>/<ap,ap>, so R^2 < 0 exactly
    when s < 0.5 however good the correlation is. Measured s = 0.495 at PCC
    +0.3325 against a shuffle control of +0.0033 +- 0.0159. This is not
    `sd(pred)/sd(target)`, which is 0.672 on the same axis: the prediction is
    narrower than the target and still too strong.

    The decomposition is `p = p_og + r_b + r_w` -- orthogroup mean, per-genome
    mean of the remainder, interaction residual. One coefficient is not
    enough, because the last two have opposite faults (slope 1.241 on 20.3 %
    of the variance, 0.236 on 79.7 %). `<r_b, r_w>` and the cross terms with
    the target are exactly zero, so those two may be fitted INDEPENDENTLY --
    hence `DEFAULTS`. `p_og` is not orthogonal to them, orthogroup and genome
    being crossed rather than nested (overlap max|<x,y>|/n = 2.1e-02 against
    1e-10 for an orthogonal pair), so `osie=("og", "gen", "inter")` depends on
    the peeling order and solves no least-squares problem; it is for
    measurement only.

    This is not a rescaling and therefore does move PCC, which is invariant
    only to `p -> a*p + b` with ONE `a`: +0.01225 for one genome coefficient
    and +0.02340 for two, against 1.1e-16 for the affine control, fitted on
    `spy` and applied to validation. The genome axis goes from R^2 -0.0045 to
    +0.1988 at slope 1.052.

    `p_og` is an orthogroup mean, so this operates on a SET of rows and not on
    one forward pass; it cannot be applied inside `forward` and is not wired
    in. Design work evaluates candidates in batches within an orthogroup, so
    that is no obstacle there.
    """

    OSIE = ("og", "gen", "inter")
    DEFAULTS = ("gen", "inter")

    def __init__(self, wsp=None) -> None:
        self.wsp: Dict[str, float] = {k: float(v)
                                      for k, v in dict(wsp or {}).items()}

    @staticmethod
    def _srednia(kod: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
        s = np.bincount(kod, weights=v, minlength=n)
        c = np.bincount(kod, minlength=n).astype(np.float64)
        return np.where(c > 0, s / np.maximum(c, 1.0), 0.0)

    @classmethod
    def rozloz(cls, v, og_ix, gen_ix):
        """`v` -> (orthogroup component, genome main effect, interaction).

        The indices may be any labels, int or string: `np.unique` renumbers
        them, so the caller's numbering fixes nothing.
        """
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        _, o = np.unique(np.asarray(og_ix).reshape(-1), return_inverse=True)
        _, g = np.unique(np.asarray(gen_ix).reshape(-1), return_inverse=True)
        if v.size != o.size or v.size != g.size:
            raise ValueError(
                "rozloz: %d values, %d orthogroups, %d genomes -- every "
                "axis must carry a label FOR EVERY ROW, otherwise the "
                "components describe different subsets"
                % (v.size, o.size, g.size))
        v_og = cls._srednia(o, v, int(o.max()) + 1)[o]
        r = v - v_og
        r_b = cls._srednia(g, r, int(g.max()) + 1)[g]
        return v_og, r_b, r - r_b

    def dopasuj(self, pred, target, og_ix, gen_ix, osie=None):
        """Least squares without an intercept, separately on each axis."""
        osie = tuple(self.DEFAULTS if osie is None else osie)
        zle = [o for o in osie if o not in self.OSIE]
        if zle:
            raise ValueError("nieznane osie %r; dozwolone %r"
                             % (zle, self.OSIE))
        p = self.rozloz(pred, og_ix, gen_ix)
        a = self.rozloz(target, og_ix, gen_ix)
        for i, name in enumerate(self.OSIE):
            if name not in osie:
                continue
            d = float(np.dot(p[i], p[i]))
            self.wsp[name] = (float(np.dot(p[i], a[i]) / d) if d > 1e-12
                               else 1.0)
        return dict(self.wsp)

    def zastosuj(self, pred, og_ix, gen_ix):
        """Calibrated prediction. Without a fit it returns a copy of the input."""
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if not self.wsp:
            return pred.copy()
        p = self.rozloz(pred, og_ix, gen_ix)
        out = pred.copy()
        for i, name in enumerate(self.OSIE):
            if name in self.wsp:
                out = out + (self.wsp[name] - 1.0) * p[i]
        return out

    def raport(self, pred, target, og_ix, gen_ix):
        """The genome axis before and after calibration."""
        def _os(pr):
            _, r_b, r_w = self.rozloz(pr, og_ix, gen_ix)
            ap = r_b + r_w
            _, y_b, y_w = self.rozloz(target, og_ix, gen_ix)
            ay = y_b + y_w
            spp = float(np.dot(ap, ap))
            spy = float(np.dot(ap, ay))
            syy = float(np.dot(ay, ay))
            return {
                "r2": 1.0 - float(((ay - ap) ** 2).sum()) / max(syy, 1e-30),
                "nachylenie": spy / max(spp, 1e-30),
                "pcc": spy / max(math.sqrt(spp * syy), 1e-30)}
        return {"przed": _os(pred),
                "po": _os(self.zastosuj(pred, og_ix, gen_ix)),
                "wspolczynniki": dict(self.wsp)}

    @staticmethod
    def kara_amplitudy(pred: torch.Tensor, target: torch.Tensor,
                       grupa: torch.Tensor) -> torch.Tensor:
        """One-sided scalar penalty for EXCESS amplitude within a group.

        The calibration above as a loss term, for the orthogroup blocks
        `BatchMaker.epoch` already reserves (`ortholog_fraction`), where the
        rows of one `grupa` are one orthogroup across several genomes.
        Calibration wants `s = <ap,ay>/<ap,ap> >= 1`, so the penalty is
        `relu(<ap,ap> - <ap,ay>)`: exactly zero when the axis is calibrated or
        UNDER-scaled, because the genome main effect has slope 1.241 and a
        two-sided penalty would shrink it along with the interaction. The
        gradient flows only through `ap`, so the term cannot improve by moving
        the target.

        Nothing calls this -- `loss` is not given an orthogroup identifier --
        and it is a `staticmethod` with an explicit argument rather than a
        config flag, so that it cannot sit declared and inert.
        """
        p = pred.reshape(-1).float()
        y = target.reshape(-1).float().detach()
        g = grupa.reshape(-1).long()
        if p.numel() != y.numel() or p.numel() != g.numel():
            raise ValueError(
                "kara_amplitudy: %d predictions, %d targets, %d group labels"
                % (p.numel(), y.numel(), g.numel()))
        uniq, inv = torch.unique(g, return_inverse=True)
        k = int(uniq.numel())
        licz = torch.zeros(k, device=p.device).index_add_(
            0, inv, torch.ones_like(p))
        # Singleton groups give a zero residual and contribute nothing; they
        # stay, because dropping them would make the mask batch-dependent.
        sp = (torch.zeros(k, device=p.device).index_add_(0, inv, p)
              / licz.clamp_min(1.0))
        sy = (torch.zeros(k, device=p.device).index_add_(0, inv, y)
              / licz.clamp_min(1.0))
        ap = p - sp[inv]
        ay = y - sy[inv]
        return (torch.relu((ap * ap).sum() - (ap * ay).sum())
                / p.numel())


# Head
class AbsoluteStrengthHead(nn.Module):
    """Latent -> (mu, log_var, log_sigma_obs) with a beta-NLL objective.

    `in_dim` is the width of the pooled latent, or of `pooling.out_dim` when
    `pooling` is given. `beta_mu = None` takes the original code branch, so
    the neutral setting is bit-identical to the head as it was.
    `lv_min`/`lv_max` clamp the log-variance and both walls are reported,
    because a hard clamp has gradient exactly 0 outside the range and a sample
    at the wall is frozen rather than bounded; `clamp_mode="soft"` replaces
    the wall by a sigmoid that never reaches the bound but saturates, and
    saturation is how an 81-cell FSQ grid measured exactly 4.000 bits while
    `used_cells` reported everything as used. `sigma_weight` weights the
    regression on the MEASURED sigma, which carries its own mask because sigma
    exists only where n_obs >= 2; `tie_sigma_to_var` applies that term to
    `log_var` directly, which makes the beta-NLL weight a function of a label
    and is off unless the comparison is being measured.
    """

    # Thresholds are attributes, not literals inside `report`, so that a run
    # which changes one has to say so.
    LV_SD_MIN = 1e-3
    LV_R_MIN = 0.10
    LV_WALL_MAX = 0.05
    MU_SD_MIN = 1e-3
    SLOPE_TOL = 0.10
    # Spread, in nats, of log(var ** (beta_mu - 1)), the per-sample
    # coefficient of the mu gradient; measured at 0.182-0.194 (a 2.1x span
    # across +-2 sd) at beta = 0.5. 0.35 is a 4.06x span, above which the mu
    # gradient concentrates on the samples the head already fits.
    MU_W_LOGSD_MAX = 0.35

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, context_dim: int = 0,
                 context_mode: str = "concat", context_center: bool = True,
                 beta: float = 0.5, beta_mu: Optional[float] = None,
                 lv_min: float = -8.0, lv_max: float = 4.0,
                 clamp_mode: str = "hard", predict_sigma: bool = True,
                 sigma_weight: float = 0.2, sigma_floor: float = 1e-3,
                 tie_sigma_to_var: bool = False,
                 pooling: Optional[LatentPooling] = None,
                 reservoir_capacity: int = 20000, seed: int = 0,
                 n_outputs: int = 1,
                 derive_marginal: bool = False,
                 cell_weights=None,
                 per_row_only: bool = False,
                 text_emb=None) -> None:
        super().__init__()
        if clamp_mode not in ("hard", "soft"):
            raise ValueError("clamp_mode must be hard|soft, got %r"
                             % (clamp_mode,))
        if lv_max <= lv_min:
            raise ValueError("lv_max must exceed lv_min")
        if beta < 0.0 or beta > 1.0:
            raise ValueError("beta-NLL exponent lives in [0, 1], got %r"
                             % (beta,))
        if beta_mu is not None and (beta_mu < 0.0 or beta_mu > 1.0):
            raise ValueError("beta_mu lives in [0, 1] like beta, got %r. It is "
                             "an exponent on the DETACHED variance, not a loss "
                             "weight, so a value outside the range does not "
                             "mean 'more of the same'." % (beta_mu,))
        self.pooling = pooling
        self.trunk = ConditionedTrunk(in_dim, hidden, depth, dropout,
                                      context_dim, context_mode,
                                      context_center)
        # K category outputs. With `derive_marginal` the last of the K + 1
        # columns is the genome marginal COMPUTED as a weighted mean of the
        # categories rather than predicted: there is no `mean` experiment, and
        # a free column would let the head contradict itself. The weighted
        # mean reproduces the measured marginal at R^2 0.999849.
        self.n_outputs = max(1, int(n_outputs))
        self.derive_marginal = bool(derive_marginal) and self.n_outputs > 1
        self.width = self.n_outputs + (1 if self.derive_marginal else 0)
        # `mu_layer` is the fallback for the single-output path; with
        # `text_emb` the columns come from `columns` and it is unreachable.
        # It is still BUILT and discarded rather than skipped, because
        # `nn.Linear` draws from the RNG and omitting it would shift the
        # stream for every module constructed afterwards.
        _mu_layer = nn.Linear(hidden, self.n_outputs)
        nn.init.xavier_uniform_(_mu_layer.weight)
        nn.init.zeros_(_mu_layer.bias)
        _tekst_bedzie = text_emb is not None and self.n_outputs > 1
        self.mu_layer: Optional[nn.Linear] = (
            None if _tekst_bedzie else _mu_layer)
        self.text_conditioned = False
        if text_emb is not None and self.n_outputs > 1:
            import torch as _t
            _T = _t.as_tensor(text_emb, dtype=_t.float32)
            if _T.dim() != 2 or int(_T.shape[0]) != self.n_outputs:
                raise ValueError(
                    "text_emb must be [n_outputs, dim]; got %s for "
                    "%d outputs. A description per category is the "
                    "contract -- a mismatch here would silently "
                    "describe the wrong experiment."
                    % (tuple(_T.shape), self.n_outputs))
            self.register_buffer("text_emb", _T)
            self.text_proj = nn.Linear(int(_T.shape[1]), hidden)
            self.text_bias = nn.Linear(int(_T.shape[1]), 1)
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
            nn.init.zeros_(self.text_bias.weight)
            nn.init.zeros_(self.text_bias.bias)
            self.text_conditioned = True
        # Per-genome category proportions, row-normalised. As good as the
        # exact per-pair n_obs (R^2 0.999849 either way), and unlike n_obs
        # they exist at design time, where there is no measured pair to count.
        if self.derive_marginal:
            import torch as _t
            if cell_weights is None:
                _w = _t.ones(1, self.n_outputs)
            else:
                _w = _t.as_tensor(cell_weights, dtype=_t.float32)
                if _w.dim() == 1:
                    _w = _w.unsqueeze(0)
            _w = _w.clamp_min(0.0)
            _bad = _w.sum(dim=-1, keepdim=True) <= 0
            if bool(_bad.any()):
                # A genome with no samples at all (the registry has 20 names,
                # the corpus 19) would divide by zero; those rows fall back to
                # the corpus-wide profile.
                _fb = _w.sum(dim=0, keepdim=True)
                if float(_fb.sum()) <= 0:
                    _fb = _t.ones(1, self.n_outputs)
                _w = _t.where(_bad, _fb.expand_as(_w), _w)
            _w = _w / _w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("cell_weights", _w)
        # Scalar variance heads for the per-row (per-sample) regime. Built
        # unconditionally so the state_dict has one shape whichever regime a
        # run uses: a checkpoint whose keys change with a config flag is how
        # `strict=False` starts dropping tensors silently. 257 params each.
        self.lv_row = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.lv_row.weight, gain=0.1)
        nn.init.zeros_(self.lv_row.bias)
        self.ls_row = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.ls_row.weight, gain=0.1)
        nn.init.zeros_(self.ls_row.bias)
        # In the per-sample regime nothing trains a PER-CATEGORY variance:
        # every training row carries its own description, so `row_text` is
        # never None and the `lv_layer`/`ls_layer` branch is unreachable by
        # configuration, measured at exactly 0 gradient while |x| was still
        # at its initialisation value.
        self.per_row_only = bool(per_row_only)
        # Built unconditionally and assigned conditionally, for the same RNG
        # reason as `mu_layer` above.
        _lv_layer = nn.Linear(hidden, self.width)
        # Small but NON-ZERO. A zero-initialised variance layer gives
        # d(log_var)/d(feature) = 0 for every feature, so `lv_sd = 0` at the
        # start would be indistinguishable from a head that never learns, and
        # any positive control on this branch would pass trivially.
        nn.init.xavier_uniform_(_lv_layer.weight, gain=0.1)
        nn.init.zeros_(_lv_layer.bias)
        self.lv_layer: Optional[nn.Linear] = (
            None if self.per_row_only else _lv_layer)
        self.predict_sigma = bool(predict_sigma)
        self.tie_sigma_to_var = bool(tie_sigma_to_var)
        self.ls_layer: Optional[nn.Linear] = None
        if self.predict_sigma and not self.tie_sigma_to_var:
            _ls_layer = nn.Linear(hidden, self.width)
            nn.init.xavier_uniform_(_ls_layer.weight, gain=0.1)
            nn.init.zeros_(_ls_layer.bias)
            if not self.per_row_only:
                self.ls_layer = _ls_layer
        self.beta = float(beta)
        # None is not stored as a float copy of `beta`: `loss` branches on it,
        # and the None branch IS the pre-beta_mu arithmetic. Collapsing it to
        # a number would replace the control with a re-derivation of it.
        self.beta_mu = None if beta_mu is None else float(beta_mu)
        self.lv_min = float(lv_min)
        self.lv_max = float(lv_max)
        self.clamp_mode = clamp_mode
        self.sigma_weight = float(sigma_weight)
        self.sigma_floor = float(sigma_floor)

        self._moments = StreamingMoments()
        self._sigma_moments = StreamingMoments()
        # columns: log_var, squared error, |residual|
        self._res = Reservoir(reservoir_capacity, width=3, seed=seed)
        self._n = 0
        self._n_wall_lo = 0
        self._n_wall_hi = 0
        self._n_sigma_floor = 0
        self._n_sigma = 0
        self._lv_sum = 0.0
        self._lv_sq = 0.0
        #: How often `forward` returned the ROW variance in place of the
        #: per-category one. Observable, because substituting one quantity
        #: for another silently is its own class of failure.
        self._n_lv_wiersz_zamiast_kolumn = 0
        #: {genome index: [n, sum of mu, sum of target]}. See `observe`.
        self._gen: Dict[int, List[float]] = {}

    #: Layers whose EXISTENCE depends on the regime. A checkpoint from the
    #: other regime has a different key set and must raise rather than load
    #: quietly: the training loop calls `load_state_dict(..., strict=False)`
    #: for resume and SWA, so `strict` alone does not cover this.
    WARSTWY_REZIMOWE = ("mu_layer", "lv_layer", "ls_layer")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        _zle = []
        for _n in self.WARSTWY_REZIMOWE:
            _ck = any(k.startswith(prefix + _n + ".")
                      for k in state_dict)
            _kod = getattr(self, _n, None) is not None
            if _ck != _kod:
                _zle.append((_n, _ck, _kod))
        if _zle:
            # Raises rather than appending to `error_msgs`, because an
            # appended message is ignored under `strict=False` -- exactly
            # where this guard is needed.
            raise RuntimeError(
                "AbsoluteStrengthHead: the checkpoint comes from a "
                "DIFFERENT REGIME than the model this config builds. "
                "per_row_only=%r, text_conditioned=%r. Divergences "
                "(layer, in checkpoint, in code): %r. `mu_layer` "
                "exists only without `text_emb`, and "
                "`lv_layer`/`ls_layer` only outside the per-sample "
                "regime (`data.target_per_category`). Loading this "
                "silently would give a model with one set of weights "
                "reporting the metric of another.\n"
                "WAY OUT for a checkpoint from BEFORE this change "
                "(it has those layers because they were built "
                "unconditionally): they are dead -- gradient "
                "0.000e+00 from all four objectives -- so they may "
                "simply be dropped:\n"
                "    st = torch.load(p)['model']\n"
                "    st = {k: v for k, v in st.items() if not "
                "any(k.startswith('absolute.' + n + '.') for n "
                "in ('mu_layer', 'lv_layer', 'ls_layer'))}\n"
                "Dropping anything ELSE is not a regime conversion but a "
                "partial load."
                % (getattr(self, "per_row_only", None),
                   getattr(self, "text_conditioned", None), _zle))
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def forward(self, latent, context: Optional[torch.Tensor] = None,
                species_id: Optional[torch.Tensor] = None,
                row_text: Optional[torch.Tensor] = None
                ) -> AbsoluteOutput:
        """`row_text` [B, d_lm] switches the head to ONE output per row.

        That is the per-sample regime: the target is the combat value of one
        (orthogroup, sample) pair and the column is generated from that
        sample's experiment description. It is the same projection that makes
        the per-category read-out, so nothing forks -- the category read-out
        is `columns` called with the category prototypes instead.
        """
        feats = as_features(latent, self.pooling)
        h = self.trunk(feats, context)
        if row_text is not None:
            if not self.text_conditioned:
                raise ValueError(
                    "row_text given but this head has no text projection "
                    "(built without `text_emb`), so a per-row description "
                    "cannot generate a column. Build it with the category "
                    "descriptions or stop passing row_text.")
            mu = self.columns(h, text_emb=row_text, per_row=True)
        else:
            # [B, K]; squeezed only at K = 1, so every existing consumer of a
            # [B] tensor keeps working and nothing silently changes rank.
            if self.text_conditioned:
                mu = self.columns(h)
            elif self.mu_layer is not None:
                mu = self.mu_layer(h)
            else:
                # Unreachable by construction (`mu_layer` is absent exactly
                # when `text_conditioned`), so this guards a future change
                # rather than a working branch.
                raise RuntimeError(
                    "AbsoluteStrengthHead: no `mu_layer` and no text "
                    "projection -- the head has nothing to compute "
                    "mu from. Build it with `text_emb`, or with a "
                    "single output.")
        if self.derive_marginal:
            w = self.cell_weights
            if species_id is not None and w.shape[0] > 1:
                idx = species_id.reshape(-1).long().clamp(0, w.shape[0] - 1)
                w = w.index_select(0, idx)
            else:
                w = w.mean(dim=0, keepdim=True)
            w = w.to(mu.dtype)
            if w.shape[0] != mu.shape[0]:
                w = w.expand(mu.shape[0], -1)
            marg = ((mu * w).sum(dim=-1, keepdim=True)
                    / w.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            mu = torch.cat([mu, marg], dim=-1)
        if row_text is None and mu.shape[-1] == 1:
            # Only the per-CATEGORY path may squeeze: there the last axis is
            # K, and K = 1 is a single column every consumer expects as [B].
            # In the per-row path the last axis is already the batch, so a
            # batch of ONE would squeeze to a 0-d tensor -- measured with 5
            # sequences at batch_size 4, on the final batch.
            mu = mu.squeeze(-1)
        if row_text is not None:
            # ONE variance per row, from its own layer. A column of the
            # K-wide `lv_layer` would be a category's variance for a
            # prediction that is not about that category, and their mean is
            # the mean of seven things none of which was asked for.
            lv_raw = self.lv_row(h).squeeze(-1)
        elif self.lv_layer is not None:
            lv_raw = self.lv_layer(h).squeeze(-1)
        else:
            # Per-category read-out inside the per-sample regime, where
            # `lv_layer` does not exist because nothing would train it. The
            # only variance learned is `lv_row`, that ROW's, broadcast across
            # the columns to keep the contract that `log_var` has the shape
            # of `mu`. Raising would be worse: the design code calls in here
            # without `row_text` and never reads the variance.
            self._n_lv_wiersz_zamiast_kolumn += 1
            lv_raw = self.lv_row(h).squeeze(-1)
            if mu.dim() == 2:
                lv_raw = lv_raw.unsqueeze(-1).expand_as(mu)
        if self.clamp_mode == "hard":
            log_var = lv_raw.clamp(self.lv_min, self.lv_max)
        else:
            span = self.lv_max - self.lv_min
            log_var = self.lv_min + span * torch.sigmoid(lv_raw)
        ls = None
        if self.predict_sigma:
            if self.tie_sigma_to_var:
                ls = 0.5 * log_var
            elif row_text is not None:
                ls = self.ls_row(h).squeeze(-1)
            elif self.ls_layer is not None:
                ls = self.ls_layer(h).squeeze(-1)
            else:
                # As with `log_var` above: `ls_row` is the only sigma that
                # gets trained in this regime.
                ls = self.ls_row(h).squeeze(-1)
                if mu.dim() == 2:
                    ls = ls.unsqueeze(-1).expand_as(mu)
        return AbsoluteOutput(mu=mu, log_var=log_var, lv_raw=lv_raw,
                              log_sigma_obs=ls, trunk=h,
                              species_id=(None if species_id is None
                                          else species_id.reshape(-1)))

    def columns(self, h, text_emb=None, per_row: bool = False):
        """Outputs whose weights are GENERATED from a description.

        There are no free per-category columns here: one shared projection
        turns any description into a column. `per_row=False` with
        T = [K, d_lm] gives [B, K], K columns shared by the batch -- the
        category read-out with the category prototypes, the subcategory
        read-out with subcategory descriptions, at zero extra parameters.
        `per_row=True` with T = [B, d_lm] gives [B], one output per row
        conditioned on THAT ROW's description; this is the per-sample training
        target and the only regime in which the condition channel receives
        gradient.

        `per_row` is explicit rather than inferred from the shapes, because at
        B == K the two cases are indistinguishable. The rows must be
        L2-normalised the way the training descriptions were.
        """
        T = self.text_emb if text_emb is None else text_emb
        T = T.to(h.dtype).to(h.device)
        W = self.text_proj(T)
        b = self.text_bias(T).squeeze(-1)
        if not per_row:
            return h @ W.t() + b              # [B, hidden] x [hidden, K]
        if W.shape[0] != h.shape[0]:
            raise ValueError(
                "per_row=True needs one description per row: got %d "
                "descriptions for %d rows. A broadcast here would pair every "
                "promoter with another row's experiment."
                % (int(W.shape[0]), int(h.shape[0])))
        # Row-wise inner product, NOT a matmul: each row uses its own column.
        return (h * W).sum(dim=-1) + b

    # Loss
    def _waga_kary_amplitudy(self) -> float:
        """Weight of the genome-axis term: instance attribute, else config.

        The `kara_amplitudy_waga` attribute wins, so that a test can set the
        weight without touching the global config it shares with a run.
        """
        w = getattr(self, "kara_amplitudy_waga", None)
        if w is not None:
            return float(w)
        try:
            from ..config import active_config
            cfg = active_config()
        except Exception:                       # pragma: no cover
            return 0.0
        if cfg is None:
            return 0.0
        return float(getattr(cfg.heads.absolute, "kara_amplitudy", 0.0) or 0.0)

    def loss(self, out: AbsoluteOutput, target: torch.Tensor,
             valid: Optional[torch.Tensor] = None,
             sigma: Optional[torch.Tensor] = None,
             sigma_valid: Optional[torch.Tensor] = None,
             weight: Optional[torch.Tensor] = None,
             grupa: Optional[torch.Tensor] = None,
             track: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """beta-NLL on mu, plus the measured-sigma regression.

        `valid` masks elements with no absolute label at all: the pairwise
        head feeds non-promoters into the batch as guaranteed losers, and
        those carry a ranking label and no strength label, so using them here
        would train mu towards a number nobody measured.
        `PairwiseRankingHead.absolute_valid` produces exactly this mask.

        `weight` is the per-sample reliability weight. `Targets.weight()`
        returns sqrt(n_obs) rather than n_obs, because reliability is already
        0.994-0.999 at the low end and linear weighting would hand `reesei`
        (386 samples) 16x the pull of `P1` (24) for no measured gain.
        """
        y = sanitise(target).to(out.mu.dtype)
        mask = (torch.ones_like(out.mu, dtype=torch.bool) if valid is None
                else valid.to(torch.bool))
        var = out.log_var.exp()
        resid = y - out.mu
        w = var.detach().pow(self.beta)
        if self.beta_mu is None:
            # Control branch: literally the two lines that ran before
            # `beta_mu` existed, so "the neutral setting is bit-identical" is
            # a property of the code rather than an argument about floats.
            nll = 0.5 * (resid.pow(2) / var + out.log_var)
            term = w * nll
        else:
            # Detaching the residual inside `nll` keeps the value and the
            # log_var gradient of the control branch and drops only its mu
            # gradient. The correction term is exactly 0.0 in the forward
            # pass and supplies the mu gradient alone, var^(beta_mu-1)*(mu-y).
            # var stays live in `nll`, which is where log_var gets its pull
            # towards the squared error.
            rd = resid.detach()
            nll = 0.5 * (rd.pow(2) / var + out.log_var)
            term = w * nll + 0.5 * var.detach().pow(
                self.beta_mu - 1.0) * (resid.pow(2) - rd.pow(2))
        # The reliability weight is per ROW; with a multi-cell head the mask
        # is per CELL, so it broadcasts here rather than in every caller.
        rw = weight
        if rw is not None and rw.dim() == 1 and out.mu.dim() == 2:
            rw = rw.unsqueeze(-1).expand_as(out.mu)
        main = masked_mean(term, mask, rw)

        parts: Dict[str, float] = {"beta_nll": float(main.detach())}
        total = main

        if (self.predict_sigma and sigma is not None
                and out.log_sigma_obs is not None):
            s_mask = mask.clone()
            if sigma_valid is not None:
                s_mask = s_mask & sigma_valid.to(torch.bool)
            s_mask = s_mask & torch.isfinite(sigma)
            s = sanitise(sigma).to(out.mu.dtype).clamp_min(self.sigma_floor)
            at_floor = s_mask & (sanitise(sigma) <= self.sigma_floor)
            sw = weight
            if sw is not None and sw.dim() == 1 and out.mu.dim() == 2:
                sw = sw.unsqueeze(-1).expand_as(out.mu)
            sig_term = masked_mean((out.log_sigma_obs - s.log()).pow(2),
                                   s_mask, sw)
            total = total + self.sigma_weight * sig_term
            parts["sigma_mse"] = float(sig_term.detach())
            parts["sigma_effective_weight"] = self.sigma_weight
            if track:
                self._n_sigma += int(s_mask.sum())
                self._n_sigma_floor += int(at_floor.sum())
                sel = s_mask.detach().cpu().numpy().astype(bool)
                if sel.any():
                    self._sigma_moments.update(
                        out.log_sigma_obs.detach().cpu().numpy()[sel],
                        s.log().detach().cpu().numpy()[sel])

        # Genome axis, amplitude. `KalibracjaOsi.kara_amplitudy` runs on the
        # VALID rows only: unlabelled rows (negatives, empty `combat` cells)
        # have `mu_valid = False` and would enter the group mean as a number
        # nobody measured, shifting the axis instead of calibrating it.
        _w_kara = self._waga_kary_amplitudy()
        if _w_kara > 0.0:
            if grupa is None:
                raise ValueError(
                    "heads.absolute.kara_amplitudy = %.4g, but `loss` was "
                    "given `grupa=None`. The term acts on the GENOME "
                    "axis within an orthogroup, so it needs a per-row "
                    "orthogroup identifier. The missing line is on "
                    "the loop side (`loop.compute_losses`, just "
                    "before `model.absolute.loss(...)`):\n"
                    "    _ko = self._kod_ortogrup()\n"
                    "    _ri = self._dopelnij(batch.row_index, len(batch))\n"
                    "    _g = np.where(_ri >= 0,\n"
                    "                  _ko[np.clip(_ri, 0, _ko.size - 1)], -1)\n"
                    "    ... loss(..., grupa=torch.as_tensor(\n"
                    "        _g, device=out['absolute'].mu.device))\n"
                    "`_dopelnij` is necessary: `row_index` has the length "
                    "of the POSITIVE rows, while `mu` covers the "
                    "whole batch including negatives. Group -1 (a "
                    "negative) is dropped by the `mu_valid` mask "
                    "anyway, so it cannot move the axis.\n"
                    "Computing the loss without the term would be a "
                    "run that declares a mechanism it does not have."
                    % (_w_kara,))
            if out.mu.dim() != 1:
                raise ValueError(
                    "kara_amplitudy works in the PER-SAMPLE REGIME (mu [B]); "
                    "the head returned %s. In the cell regime one row "
                    "has K columns, and one group label per row would "
                    "describe K different axes at once."
                    % (tuple(out.mu.shape),))
            _g = grupa.reshape(-1).to(out.mu.device)
            if _g.numel() != out.mu.numel():
                raise ValueError(
                    "kara_amplitudy: %d group labels for %d predictions"
                    % (int(_g.numel()), int(out.mu.numel())))
            _sel = mask.reshape(-1)
            if bool(_sel.any()):
                _kara = KalibracjaOsi.kara_amplitudy(
                    out.mu.reshape(-1)[_sel], y.reshape(-1)[_sel], _g[_sel])
            else:
                _kara = out.mu.sum() * 0.0
            total = total + _w_kara * _kara
            parts["kara_amplitudy"] = float(_kara.detach())
            parts["kara_amplitudy_waga"] = float(_w_kara)
            parts["kara_amplitudy_n"] = float(int(_sel.sum()))
            # The term is arithmetically zero when every row sits in its own
            # group, so `relu(...) == 0` means "axis already calibrated" and
            # "orthogroup blocks are not arriving" equally well; only the
            # count of rows in groups of size >= 2 separates them.
            # `kara_amplitudy_bez_og` counts rows whose group is -1: if the
            # mask leaves the negatives in, they form one artificial
            # orthogroup and the term calibrates a phantom axis.
            with torch.no_grad():
                _gv = _g[_sel]
                if _gv.numel() > 0:
                    _, _cnt = torch.unique(_gv, return_counts=True)
                    parts["kara_amplitudy_par"] = float(
                        int(_cnt[_cnt > 1].sum()))
                    parts["kara_amplitudy_grup"] = float(
                        int((_cnt > 1).sum()))
                    parts["kara_amplitudy_bez_og"] = float(
                        int((_gv < 0).sum()))
                else:
                    parts["kara_amplitudy_par"] = 0.0
                    parts["kara_amplitudy_grup"] = 0.0
                    parts["kara_amplitudy_bez_og"] = 0.0
        parts["total"] = float(total.detach())
        if track:
            self.observe(out, y, mask)
        return total, parts

    # Falsifiers
    @torch.no_grad()
    def observe(self, out: AbsoluteOutput, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> None:
        """Accumulate everything `report` needs. Cheap, and no gradient."""
        sel = (torch.ones_like(out.mu, dtype=torch.bool) if mask is None
               else mask.to(torch.bool))
        if not bool(sel.any()):
            return
        mu = out.mu[sel].detach().float().cpu().numpy()
        y = sanitise(target)[sel].detach().float().cpu().numpy()
        lv = out.log_var[sel].detach().float().cpu().numpy()
        raw = out.lv_raw[sel].detach().float().cpu().numpy()
        err = (mu - y) ** 2
        self._moments.update(mu, y)
        self._res.update(np.stack([lv, err, np.abs(mu - y)], axis=1))
        self._n += int(mu.size)
        self._lv_sum += float(lv.sum())
        self._lv_sq += float((lv.astype(np.float64) ** 2).sum())
        # Genome main effect: n, sum of predictions and sum of targets per
        # genome are enough for `report` to publish the slope and R^2 of that
        # axis, over 19 unique genomes. Skipped in the cell regime
        # (`sel.dim() == 2`), where a row has K cells whose sum is not the
        # genome mean.
        _gid = getattr(out, "species_id", None)
        if _gid is not None and sel.dim() == 1:
            _g = _gid.detach().reshape(-1).cpu().numpy()
            _m = sel.detach().cpu().numpy().astype(bool)
            if _g.size == _m.size:
                _g = _g[_m]
                if _g.size == mu.size:
                    _yd = y.astype(np.float64)
                    _pd = mu.astype(np.float64)
                    for _k in np.unique(_g):
                        _w = _g == _k
                        _a = self._gen.setdefault(int(_k), [0.0, 0.0, 0.0])
                        _a[0] += float(_w.sum())
                        _a[1] += float(_pd[_w].sum())
                        _a[2] += float(_yd[_w].sum())
        if self.clamp_mode == "hard":
            tol = 1e-6 * max(1.0, abs(self.lv_max - self.lv_min))
            self._n_wall_lo += int((raw <= self.lv_min + tol).sum())
            self._n_wall_hi += int((raw >= self.lv_max - tol).sum())
        else:
            # Soft mode never touches the bound, so the wall is defined as
            # saturation of the sigmoid: within 1 % of either end of the range.
            span = self.lv_max - self.lv_min
            self._n_wall_lo += int((lv <= self.lv_min + 0.01 * span).sum())
            self._n_wall_hi += int((lv >= self.lv_max - 0.01 * span).sum())

    def reset(self) -> None:
        """Start a new epoch. `report` describes an epoch, not a batch."""
        self._moments = StreamingMoments()
        self._sigma_moments = StreamingMoments()
        self._res.reset()
        self._n = 0
        self._n_wall_lo = 0
        self._n_wall_hi = 0
        self._n_sigma = 0
        self._n_sigma_floor = 0
        self._lv_sum = 0.0
        self._lv_sq = 0.0
        self._gen = {}

    def report(self) -> Dict[str, object]:
        """Falsifiers for this head. `ok` is False when any threshold fires.

        `lv_sd` near zero means the variance head learned a constant, and
        `lv_r` near its permutation control `lv_r_ctrl` means the same with a
        non-constant output. `lv_wall` is the fraction of samples at the clamp
        boundary, where the gradient on log_var is exactly zero. `mu_sd` near
        zero means the prediction is a constant: an untrained model in eval
        measured sd 3.19e-08 and still scored a defensible-looking loss.
        `calib_slope` and `sd_ratio` catch a predictor squeezed towards the
        mean, which PCC cannot see.
        """
        nan = float("nan")
        out: Dict[str, object] = {"n": self._n}
        if self._n < 3:
            out.update({"ok": False,
                        "violations": ["fewer than 3 observed samples"]})
            return out
        mean_lv = self._lv_sum / self._n
        var_lv = max(self._lv_sq / self._n - mean_lv * mean_lv, 0.0)
        rows = self._res.values()
        lv_s, err_s = rows[:, 0], rows[:, 1]
        lv_r = spearman(np.exp(lv_s), err_s)
        # Control for lv_r: permuting the predictions destroys the pairing and
        # keeps both marginals, so this is the value `lv_r` must beat. Equal
        # values mean the correlation is an artefact of the estimator.
        rng = np.random.default_rng(12345)
        lv_ctrl = spearman(np.exp(lv_s[rng.permutation(lv_s.size)]), err_s)

        mom = self._moments.report()
        out.update(mom)
        out.update(self._raport_osi_genomu(mom))
        out["lv_wiersz_zamiast_kolumn"] = float(
            self._n_lv_wiersz_zamiast_kolumn)
        out["lv_mean"] = mean_lv
        out["lv_sd"] = math.sqrt(var_lv)
        # Spread of log(var^(beta_mu-1)), the per-sample coefficient of the mu
        # gradient. 0.0 means mu is trained with a plain MSE gradient and the
        # predicted variance cannot reweight the mu path at all.
        _bmu = self.beta if self.beta_mu is None else self.beta_mu
        out["beta_mu_effective"] = float(_bmu)
        out["mu_grad_w_logsd"] = abs(_bmu - 1.0) * out["lv_sd"]
        # Predicted variance against the ACTUAL squared error, both averaged
        # over the reservoir; 1.0 is a calibrated variance head. It separates
        # "the variance shrank because the error shrank" from "the variance
        # shrank on its own", and held at 0.96 +- 0.002 while both fell 44 %.
        _err_mean = float(np.mean(err_s))
        out["lv_scale"] = (float(np.mean(np.exp(lv_s))) / _err_mean
                           if _err_mean > 0.0 else nan)
        out["lv_r"] = lv_r
        out["lv_r_ctrl"] = lv_ctrl
        out["lv_wall"] = (self._n_wall_lo + self._n_wall_hi) / float(self._n)
        out["lv_wall_lo"] = self._n_wall_lo / float(self._n)
        out["lv_wall_hi"] = self._n_wall_hi / float(self._n)
        out["mu_sd"] = mom.get("pred_sd", nan)
        out["n_sigma"] = self._n_sigma
        out["sigma_floor_frac"] = (self._n_sigma_floor / float(self._n_sigma)
                                   if self._n_sigma else nan)
        sig = self._sigma_moments.report()
        out["sigma_pcc"] = sig.get("pcc", nan)

        bad: List[str] = []
        if out["lv_sd"] < self.LV_SD_MIN:
            bad.append("lv_sd %.2e < %.0e: the variance head learned a CONSTANT"
                       % (out["lv_sd"], self.LV_SD_MIN))
        if not (lv_r == lv_r) or lv_r < self.LV_R_MIN:
            bad.append("lv_r %.3f < %.2f: predicted variance does not track "
                       "the actual error (control %.3f)"
                       % (lv_r, self.LV_R_MIN, lv_ctrl))
        if lv_r == lv_r and lv_ctrl == lv_ctrl and lv_r <= lv_ctrl + 0.02:
            bad.append("lv_r %.3f does not beat its permutation control %.3f"
                       % (lv_r, lv_ctrl))
        if out["mu_grad_w_logsd"] > self.MU_W_LOGSD_MAX:
            bad.append("mu_grad_w_logsd %.3f > %.2f: the mu gradient is "
                       "weighted by var^%.2f, so a %.2fx span separates the "
                       "samples the head already fits from the rest, and val "
                       "SCC cannot see it. Raise heads.absolute.beta_mu "
                       "towards 1.0"
                       % (out["mu_grad_w_logsd"], self.MU_W_LOGSD_MAX,
                          _bmu - 1.0,
                          math.exp(4.0 * out["mu_grad_w_logsd"])))
        if out["lv_wall"] > self.LV_WALL_MAX:
            bad.append("lv_wall %.3f > %.2f: those samples have exactly zero "
                       "gradient on log_var (lo %.3f / hi %.3f)"
                       % (out["lv_wall"], self.LV_WALL_MAX,
                          out["lv_wall_lo"], out["lv_wall_hi"]))
        if mom.get("pred_sd", nan) == mom.get("pred_sd", nan) and \
                mom["pred_sd"] < self.MU_SD_MIN:
            bad.append("mu_sd %.2e: the prediction is a CONSTANT (an untrained "
                       "model in eval measured sd 3.19e-08)" % (mom["pred_sd"],))
        slope = mom.get("calib_slope", nan)
        if slope == slope and abs(slope - 1.0) > self.SLOPE_TOL:
            bad.append("calib_slope %.4f: predictions are squeezed towards the "
                       "mean; PCC cannot see this and the affine fix is free, "
                       "but it must be applied before the predictor is used as "
                       "a design target" % (slope,))
        out["violations"] = bad
        out["ok"] = not bad
        return out

    def _raport_osi_genomu(self, mom: Dict[str, float]) -> Dict[str, float]:
        """The genome MAIN EFFECT, plus the drift of the global calibration.

        `gen_*` is computed on per-genome means without centring by
        orthogroup, the head having no orthogroup identifier. It is therefore
        the genome main effect, the part of the axis the species channel owns,
        and NOT the "genome within orthogroup" axis of `KalibracjaOsi`: that
        part works (R^2 +0.8485) and is mildly under-scaled, while the damage
        sits in the interaction, which is invisible from here.

        `amp_ratio` is `pred_sd / (pcc * target_sd)`, 1.0 being exactly the
        amplitude the model's own correlation allows; it drifts upwards over a
        run (0.976 / 1.027 / 1.187 at epochs 1, 4, 14). It needs its own key
        because `sd_ratio` on the same log line means `pred_sd / target_sd`.
        """
        nan = float("nan")
        out: Dict[str, float] = {"gen_n": float(len(self._gen))}
        _pc = float(mom.get("pcc", nan))
        _ps = float(mom.get("pred_sd", nan))
        _ts = float(mom.get("target_sd", nan))
        if _pc == _pc and _ps == _ps and _ts == _ts and _pc * _ts > 0.0:
            out["amp_ratio"] = _ps / (_pc * _ts)
        if len(self._gen) < 3:
            return out
        n = np.asarray([v[0] for v in self._gen.values()], dtype=np.float64)
        p = np.asarray([v[1] for v in self._gen.values()]) / np.maximum(n, 1.0)
        y = np.asarray([v[2] for v in self._gen.values()]) / np.maximum(n, 1.0)
        p = p - float(np.average(p, weights=n))
        y = y - float(np.average(y, weights=n))
        spp = float((n * p * p).sum())
        spy = float((n * p * y).sum())
        syy = float((n * y * y).sum())
        if spp <= 0.0 or syy <= 0.0:
            return out
        out["gen_slope"] = spy / spp
        out["gen_pcc"] = spy / math.sqrt(spp * syy)
        out["gen_r2"] = 1.0 - float((n * (y - p) ** 2).sum()) / syy
        out["gen_sd_ratio"] = math.sqrt(spp / syy)
        return out

    def calibrate(self) -> Tuple[float, float]:
        """(slope, intercept) such that `slope * mu + intercept` is unbiased.

        PCC is bit-identical before and after. MSE after this correction sat
        on the (1-r^2)*var(y) floor to within -0.0000 in four out of four
        measured cases, so there is no separate MSE lever: improving MSE
        beyond this point requires improving the ordering.
        """
        mom = self._moments.report()
        return float(mom["calib_slope"]), float(mom["calib_intercept"])
