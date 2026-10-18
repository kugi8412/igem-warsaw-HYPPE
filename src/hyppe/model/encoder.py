"""Promoter encoder: the only path from bases to the quantiser.

The decoder sees only codes -- diffusion generates by decoding -- so everything
the latent can ever say about a promoter has to be put here. Starving this path
is not recoverable downstream: at 6.6 % of the model it cost -0.036 PCC.

Six residual blocks with kernel 7 sit between the stem and the first stride, for
phase rather than depth: after stride 2 a 6-15 bp motif loses its phase, while
kernel 7 gives 13 bp of view after two layers. The convolution alone never
reaches the whole promoter, so the two global paths -- the billboard of
component 6 and the damped context of component 5 -- are a requirement, and both
touch only the coarsest map. Promoters are not reverse-complement invariant, so
there is no strand canonisation here.

Channel layout of the stem input is a contract; the weight-norm falsifiers slice
`stem_conv.weight` by it:

    [ 0 : 4 )            one-hot ACGT, N = 0.25 uniform
    [ 4 : 4+M )          motif occupancy, M channels, both strands
    [ 4+M : 4+M+A )      positional anchor, A channels (A = 1 by default)

`context_*` is the global depthwise module of component 5; `cond_*` is the
conditioning cross-attention of component 7, whose keys arrive as the `context`
argument of `forward`. Parameter names are a contract too, since the trainer
exempts from weight decay by substring, so `no_decay_parameter_names()`
publishes the real names.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PromoterEncoder",
    "PositionalAnchor",
    "DilatedResidualBlock",
    "BillboardAttention",
    "ConditionCrossAttention",
    "DampedGlobalContext",
    "convolutional_receptive_field",
    "FalsifierRow",
]

_EPS = 1e-6

# Squeeze statistics and how many copies of `inner` each one produces.
_SQUEEZE_STATS = {"mean": 1, "mean_sd": 2, "mean_max_sd": 3}


def _rms_per_sample(t: torch.Tensor) -> torch.Tensor:
    """RMS over (channel, position) for each sample: [B, 1, 1].

    Per sample, never per batch: a batch statistic makes the codes a function
    of who else was in the batch, voiding the diffusion API.
    """
    return t.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(_EPS)


def _make_norm(kind: str, dim: int, groups: int = 8) -> nn.Module:
    """Normalisation factory.

    `batch` reproduces the measured family bit for bit, at the price of making
    the map depend on batch composition in train mode; `group` removes even
    that. A flag, not a silent switch: with BatchNorm on a path, train and
    eval are two different models.
    """
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "group":
        g = math.gcd(int(groups), int(dim))
        return nn.GroupNorm(max(g, 1), dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError("norm must be one of {'batch','group','none'}, got %r"
                     % (kind,))


def _squeeze_features(h: torch.Tensor, stat: str) -> torch.Tensor:
    """Build the squeeze vector [B, inner * k] for the requested statistic.

    `mean_sd` and `mean_max_sd` are the measured-positive combinations; max
    alone is worse on every block, the 81x that made it look obvious having
    been measured on sparse PWM occupancy, not on a dense trunk map.
    """
    m = h.mean(dim=-1)
    if stat == "mean":
        return m
    if stat == "mean_sd":
        return torch.cat([m, h.std(dim=-1)], dim=1)
    if stat == "mean_max_sd":
        return torch.cat([m, h.amax(dim=-1), h.std(dim=-1)], dim=1)
    raise ValueError("unknown squeeze statistic %r" % (stat,))


@dataclass
class FalsifierRow:
    """One observable that can show the mechanism is dead or broken.

    `ok` is None until the quantity is measured, so "not measured" can never
    be read as "passed".
    """

    name: str
    value: Optional[float]
    rule: str
    bound: Tuple[float, ...]
    note: str

    @property
    def ok(self) -> Optional[bool]:
        if self.value is None:
            return None
        v = float(self.value)
        if self.rule == "<":
            return v < self.bound[0]
        if self.rule == ">":
            return v > self.bound[0]
        if self.rule == "range":
            return self.bound[0] < v < self.bound[1]
        if self.rule == "info":
            return None
        raise ValueError("unknown rule %r" % (self.rule,))

    def format(self) -> str:
        v = "n/a" if self.value is None else "%.4g" % float(self.value)
        ok = {None: "?", True: "ok", False: "FAIL"}[self.ok]
        return "%-24s %-10s %-6s %s  | %s" % (self.name, v, ok, self.rule
                                              + str(self.bound), self.note)


# Component 8: positional anchor
class PositionalAnchor(nn.Module):
    """Position measured from the start codon, as input channels.

    1-based promoter position p runs 1..L with p = L immediately before ATG,
    so the distance upstream of the start codon is d = L - p + 1. Channels, in
    order: `linear` = 2*(p-1)/(L-1) - 1; `log` = 2*log(d)/log(L) - 1, which
    the linear ramp cannot substitute for near ATG (10 bp against 30 bp is
    2.5 % of its range against 34 % of the log channel's, and that is where
    core-promoter elements sit); and `fourier` = sin/cos(2*pi*d/P) for P
    log-spaced between `min_period` and L, two channels per period. No
    parameters and no training state, so this cannot become a per-run knob.
    The default of one channel is the `in_ch = 4 + M + 1` of the architecture
    contract; the others change `in_ch`, which `n_channels` reports.
    """

    def __init__(self, linear: bool = True, log: bool = False,
                 fourier: int = 0, min_period: float = 10.0):
        super().__init__()
        self.linear = bool(linear)
        self.log = bool(log)
        self.fourier = int(fourier)
        self.min_period = float(min_period)
        if self.fourier < 0:
            raise ValueError("fourier must be >= 0")
        if self.n_channels == 0:
            raise ValueError(
                "PositionalAnchor with zero channels: pass anchor=False to "
                "drop the anchor instead of building an empty module -- an "
                "empty module reads as 'present' in every repr and report.")
        self._cache: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}

    @property
    def n_channels(self) -> int:
        return int(self.linear) + int(self.log) + 2 * self.fourier

    def channel_names(self) -> List[str]:
        names: List[str] = []
        if self.linear:
            names.append("anchor_linear")
        if self.log:
            names.append("anchor_log")
        for k in range(self.fourier):
            names += ["anchor_sin%d" % k, "anchor_cos%d" % k]
        return names

    def periods(self, length: int) -> List[float]:
        """Log-spaced periods in bp, from `min_period` up to `length`."""
        if self.fourier == 0:
            return []
        if self.fourier == 1:
            return [float(length)]
        lo, hi = math.log(self.min_period), math.log(float(length))
        return [math.exp(lo + (hi - lo) * k / (self.fourier - 1))
                for k in range(self.fourier)]

    def forward(self, length: int, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
        """-> [1, A, L]; broadcast over the batch by the caller."""
        key = (int(length), str(device), dtype)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        L = int(length)
        if L < 2:
            raise ValueError("positional anchor needs length >= 2, got %d" % L)
        p = torch.arange(1, L + 1, device=device, dtype=torch.float64)
        d = float(L) - p + 1.0                      # bp upstream of ATG
        rows: List[torch.Tensor] = []
        if self.linear:
            rows.append(2.0 * (p - 1.0) / (L - 1.0) - 1.0)
        if self.log:
            rows.append(2.0 * torch.log(d) / math.log(float(L)) - 1.0)
        for period in self.periods(L):
            w = 2.0 * math.pi / float(period)
            rows.append(torch.sin(w * d))
            rows.append(torch.cos(w * d))
        out = torch.stack(rows, dim=0).unsqueeze(0).to(dtype)
        self._cache[key] = out
        return out

    def extra_repr(self) -> str:
        return "channels=%d (%s)" % (self.n_channels,
                                     ", ".join(self.channel_names()))


# Components 2 and 4: residual blocks
class DilatedResidualBlock(nn.Module):
    """Inverted residual with parallel dilated branches, optional SE.

    Kernel 7 at dilations (1, 2, 4) spans 7, 13 and 25 bases, i.e. a fungal PWM
    plus roughly two helical turns of spacing, at full resolution. `se_stat`
    defaults to None because SE reads a scalar per channel over all positions,
    so one SE block before the quantiser makes every level's input a function
    of the whole sequence.
    """

    def __init__(self, dim: int, kernel_size: int = 7,
                 dilations: Sequence[int] = (1, 2, 4), expand: int = 2,
                 dropout_kind: str = "element",
                 se_stat: Optional[str] = None, se_reduction: int = 4,
                 dropout: float = 0.1, norm: str = "batch",
                 norm_groups: int = 8):
        super().__init__()
        dilations = tuple(int(d) for d in dilations)
        if not dilations:
            raise ValueError("dilations must not be empty")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep length, got %d"
                             % kernel_size)
        inner = int(dim) * int(expand)
        per = max(inner // len(dilations), 1)
        inner = per * len(dilations)              # keep the split exact
        self.dim, self.inner, self.per = int(dim), inner, per
        self.kernel_size = int(kernel_size)
        self.dilations = dilations
        self.norm = _make_norm(norm, int(dim), norm_groups)
        self.act = nn.SiLU()
        self.expand = nn.Conv1d(int(dim), inner, 1)
        self.branches = nn.ModuleList(
            nn.Conv1d(per, per, self.kernel_size, groups=per,
                      dilation=d, padding=d * (self.kernel_size // 2))
            for d in dilations)
        self.se_stat = se_stat
        if se_stat is None:
            self.se = None
        else:
            if se_stat not in _SQUEEZE_STATS:
                raise ValueError("se_stat must be one of %s or None, got %r"
                                 % (sorted(_SQUEEZE_STATS), se_stat))
            squeezed = max(inner // int(se_reduction), 8)
            self.se = nn.Sequential(
                nn.Linear(inner * _SQUEEZE_STATS[se_stat], squeezed),
                nn.SiLU(), nn.Linear(squeezed, inner), nn.Sigmoid())
        self.project = nn.Conv1d(inner, int(dim), 1)
        # Neighbouring positions share 91.7 % of their receptive field, so
        # zeroing single activations is cosmetic; `spatial` zeroes channels.
        if str(dropout_kind) == 'spatial':
            self.drop = nn.Dropout1d(float(dropout))
        elif str(dropout_kind) == 'element':
            self.drop = nn.Dropout(float(dropout))
        else:
            raise ValueError(
                'dropout_kind = %r; dozwolone sa element i spatial'
                % (dropout_kind,))

    @property
    def max_dilation(self) -> int:
        return max(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = self.expand(h)
        parts = [b(h[:, i * self.per:(i + 1) * self.per])
                 for i, b in enumerate(self.branches)]
        h = self.act(torch.cat(parts, dim=1))
        if self.se is not None:
            gate = self.se(_squeeze_features(h, self.se_stat))
            h = h * gate.unsqueeze(-1)
        h = self.drop(self.project(h))
        return x + h

    def extra_repr(self) -> str:
        return "dim=%d, k=%d, dilations=%s, se=%s" % (
            self.dim, self.kernel_size, self.dilations, self.se_stat)


# Component 6: billboard attention (motifs -> coarsest map)
class BillboardAttention(nn.Module):
    """Queries from coarse windows, keys and values from the whole occupancy.

    Each window of the coarsest map asks the full 800 bp occupancy a question,
    so the level-0 code stops saying "what sits locally in this window" and
    starts saying "which combination of factors occurs in this promoter". The
    licence is measured: motif pairs without distance beat pairs binned into
    distance shells by +0.01932 at 3.4x fewer features.

    Merged structurally rather than through a gate -- concatenate, one 1x1
    conv back to `dim`, no path onward that avoids it -- because a scalar gate
    is free to reach zero and zero is absorbing. The null key is a learned
    extra key whose value starts at zero, so the softmax need not spread mass
    over the occupancy when the window has nothing to ask; zero init is safe
    here because the value enters additively. Key pooling uses max, and only
    for occupancy, which is sparse and extremal (0.329 against 0.004 for mean
    as a linear probe); on a dense trunk map max is worse.
    """

    def __init__(self, dim: int, occ_channels: int, heads: int = 4,
                 dim_head: int = 32, dropout: float = 0.0,
                 null_key: bool = True, key_pool: int = 1):
        super().__init__()
        self.h, self.dk = int(heads), int(dim_head)
        w = self.h * self.dk
        self.dim = int(dim)
        self.key_pool = max(int(key_pool), 1)
        self.q = nn.Conv1d(int(dim), w, 1, bias=False)
        self.k = nn.Conv1d(int(occ_channels), w, 1, bias=False)
        self.v = nn.Conv1d(int(occ_channels), w, 1, bias=False)
        self.o = nn.Conv1d(w, int(dim), 1, bias=False)
        self.drop = nn.Dropout(float(dropout))
        if null_key:
            self.null_key = nn.Parameter(torch.randn(1, self.h, 1, self.dk)
                                         * 0.02)
            self.null_value = nn.Parameter(torch.zeros(1, self.h, 1, self.dk))
        else:
            self.register_parameter("null_key", None)
            self.register_parameter("null_value", None)
        self.merge = nn.Conv1d(2 * int(dim), int(dim), 1)
        self._diag: Dict[str, torch.Tensor] = {}

    def forward(self, h: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        """`h`: [B, C, W] coarsest map.  `occ`: [B, M, L] full occupancy."""
        B, _, W = h.shape
        if self.key_pool > 1:
            occ = F.max_pool1d(occ, self.key_pool, ceil_mode=True)
        q = self.q(h).view(B, self.h, self.dk, W).transpose(-1, -2)
        Lk = occ.shape[-1]
        k = self.k(occ).view(B, self.h, self.dk, Lk).transpose(-1, -2)
        v = self.v(occ).view(B, self.h, self.dk, Lk).transpose(-1, -2)
        if self.null_key is not None:
            k = torch.cat([k, self.null_key.expand(B, -1, -1, -1)], dim=-2)
            v = torch.cat([v, self.null_value.expand(B, -1, -1, -1)], dim=-2)
        att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dk), -1)
        y = (self.drop(att) @ v).transpose(-1, -2).reshape(B, -1, W)
        y = self.o(y)
        # RMS match per sample: `cat([h, y])` assumes the two are comparable,
        # which a normalisation layer inside the branch makes false.
        rh = _rms_per_sample(h)
        y = y * (rh / _rms_per_sample(y))
        out = self.merge(torch.cat([h, y], dim=1))
        with torch.no_grad():
            ent = -(att * (att + 1e-12).log2()).sum(-1).mean()
            wm = self.merge.weight[:, :self.dim].norm()
            wa = self.merge.weight[:, self.dim:].norm()
            self._diag = {
                "billboard_H": ent.detach(),
                "billboard_H_max": torch.tensor(
                    math.log2(att.shape[-1]), device=h.device),
                "billboard_share": (wa / wm.clamp_min(1e-9)).detach(),
                "billboard_dom": (y.pow(2).mean().sqrt()
                                  / rh.mean().clamp_min(1e-9)).detach(),
                "billboard_null_mass": (
                    att[..., -1].mean().detach() if self.null_key is not None
                    else torch.zeros((), device=h.device)),
            }
        return out

    def report(self) -> Dict[str, torch.Tensor]:
        return dict(self._diag)

    def extra_repr(self) -> str:
        return "heads=%d, dim_head=%d, null_key=%s, key_pool=%d" % (
            self.h, self.dk, self.null_key is not None, self.key_pool)


# Component 7: conditioning cross-attention
class ConditionCrossAttention(nn.Module):
    """Queries from the DNA map, keys and values from the conditioning tokens.

    The conditioning is species + free text + two flank vectors, i.e. a handful
    of tokens rather than channels: broadcasting a constant vector across 800
    positions broadcasts the same number, while a token has its own place in
    the softmax. It can sit at full resolution because it is cheap -- 1.6 MiB
    of scores, against 1.71 GiB for banded self-attention at the same length.

    `context=None` resolves to a learned null embedding rather than skipping
    the module, so the branch has one behaviour and classifier-free guidance
    has a defined unconditional pass. `drop_prob` substitutes that embedding
    per sample during training; left at 0.0, guidance at sampling time
    extrapolates from a condition the model has never been without. And this
    sits before the downs, so the codes are a function of (sequence,
    condition) and the prior must be conditioned on the same vector --
    `encoder_report()` exposes `conditioning_touches_latent` to assert it.
    """

    def __init__(self, dim: int, cond_dim: int, tokens: int = 1,
                 heads: int = 4, dim_head: int = 32, dropout: float = 0.0,
                 drop_prob: float = 0.0):
        super().__init__()
        self.h, self.dk = int(heads), int(dim_head)
        w = self.h * self.dk
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.tokens = int(tokens)
        self.drop_prob = float(drop_prob)
        self.q = nn.Conv1d(int(dim), w, 1, bias=False)
        self.k = nn.Linear(int(cond_dim), w, bias=False)
        self.v = nn.Linear(int(cond_dim), w, bias=False)
        self.o = nn.Conv1d(w, int(dim), 1, bias=False)
        self.drop = nn.Dropout(float(dropout))
        # Non-zero: a zero embedding makes every null token identical.
        self.null_context = nn.Parameter(
            torch.randn(1, self.tokens, int(cond_dim)) * 0.02)
        self.merge = nn.Conv1d(2 * int(dim), int(dim), 1)
        self._diag: Dict[str, torch.Tensor] = {}

    def prepare(self, context: Optional[torch.Tensor], batch: int,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalise the conditioning to [B, T, Dc], filling in the null."""
        null = self.null_context.to(dtype).expand(batch, -1, -1)
        if context is None:
            return null
        c = context
        if c.dim() == 2:
            # A multi-token condition arrives flattened; this check is the
            # only thing between a transposed condition and a reinterpreted one.
            if c.shape[1] != self.tokens * self.cond_dim:
                raise ValueError(
                    "context has %d features; with tokens=%d and cond_dim=%d "
                    "it must have %d. Pass [B, T, Dc] if the layout differs."
                    % (c.shape[1], self.tokens, self.cond_dim,
                       self.tokens * self.cond_dim))
            c = c.view(c.shape[0], self.tokens, self.cond_dim)
        elif c.dim() == 3:
            if c.shape[1] != self.tokens or c.shape[2] != self.cond_dim:
                raise ValueError(
                    "context is %s; expected [B, %d, %d]"
                    % (tuple(c.shape), self.tokens, self.cond_dim))
        else:
            raise ValueError("context must be [B, Dc] or [B, T, Dc], got %s"
                             % (tuple(c.shape),))
        c = c.to(dtype)
        if self.training and self.drop_prob > 0.0:
            keep = (torch.rand(c.shape[0], 1, 1, device=c.device)
                    >= self.drop_prob).to(c.dtype)
            # Mask the operands, not the result: `torch.where` on the output
            # would still backpropagate through the discarded branch.
            c = keep * c + (1.0 - keep) * null
        return c

    def forward(self, h: torch.Tensor,
                context: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, L = h.shape
        c = self.prepare(context, B, h.device, h.dtype)
        q = self.q(h).view(B, self.h, self.dk, L).transpose(-1, -2)
        k = self.k(c).view(B, self.tokens, self.h, self.dk).transpose(1, 2)
        v = self.v(c).view(B, self.tokens, self.h, self.dk).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dk), -1)
        y = (self.drop(att) @ v).transpose(-1, -2).reshape(B, -1, L)
        y = self.o(y)
        rh = _rms_per_sample(h)
        y = y * (rh / _rms_per_sample(y))
        out = self.merge(torch.cat([h, y], dim=1))
        with torch.no_grad():
            ent = -(att * (att + 1e-12).log2()).sum(-1).mean()
            wm = self.merge.weight[:, :self.dim].norm()
            wa = self.merge.weight[:, self.dim:].norm()
            self._diag = {
                "cond_H": ent.detach(),
                "cond_H_max": torch.tensor(math.log2(self.tokens),
                                           device=h.device),
                "cond_share": (wa / wm.clamp_min(1e-9)).detach(),
                "cond_dom": (y.pow(2).mean().sqrt()
                             / rh.mean().clamp_min(1e-9)).detach(),
            }
        return out

    def report(self) -> Dict[str, torch.Tensor]:
        return dict(self._diag)

    def extra_repr(self) -> str:
        return "tokens=%d, cond_dim=%d, heads=%d, dim_head=%d, drop=%.3f" % (
            self.tokens, self.cond_dim, self.h, self.dk, self.drop_prob)


# Component 5: damped global context
class DampedGlobalContext(nn.Module):
    """Depthwise k=15 + pointwise, added to the map under a hard damper.

    The damper is the point: without it the same module reached dominance
    22.64 and took a run from 0.7377 to 0.7146, because the normalisation
    layer inside reset the output scale to RMS 11.68 against an input of RMS
    0.92. The repaired form -- scale the contribution to the map's RMS per
    sample, then multiply by `share * tanh(gate)` per channel -- ran 31 epochs
    at dominance 0.26-0.29. `context_dom_effective` is bounded by `share`, so
    alone it reads back the knob; `context_dom_raw`, taken before
    normalisation, is the one that can still surprise.
    """

    def __init__(self, dim: int, kernel_size: int = 15, share: float = 0.3,
                 gate0: float = 0.5, norm: str = "batch",
                 norm_groups: int = 8):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("context kernel must be odd, got %d" % kernel_size)
        self.kernel_size = int(kernel_size)
        self.share = float(share)
        self.depthwise = nn.Conv1d(int(dim), int(dim), self.kernel_size,
                                   padding=self.kernel_size // 2, groups=int(dim))
        self.norm = _make_norm(norm, int(dim), norm_groups)
        self.act = nn.SiLU()
        self.pointwise = nn.Conv1d(int(dim), int(dim), 1)
        # The name contains "gate", so the no-decay list catches it.
        self.context_gate = nn.Parameter(torch.full((1, int(dim), 1),
                                                    float(gate0)))
        self._diag: Dict[str, torch.Tensor] = {}

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        f = self.pointwise(self.act(self.norm(self.depthwise(h))))
        rh, rf = _rms_per_sample(h), _rms_per_sample(f)
        f = f * (rh / rf)
        w = self.share * torch.tanh(self.context_gate)
        contribution = w * f
        out = h + contribution
        with torch.no_grad():
            self._diag = {
                # Branch scale against the map before damping; 12.7 broke a run.
                "context_dom_raw": (rf / rh).mean().detach(),
                # What actually reaches the map. Must stay below 0.5.
                "context_dom_effective": (
                    _rms_per_sample(contribution) / rh).mean().detach(),
                "context_gate": self.context_gate.mean().detach(),
                "context_gate_max": self.context_gate.abs().max().detach(),
            }
        return out

    def report(self) -> Dict[str, torch.Tensor]:
        return dict(self._diag)

    def extra_repr(self) -> str:
        return "k=%d, share=%.3f" % (self.kernel_size, self.share)


# Receptive field: pure integer arithmetic, checkable without torch
def convolutional_receptive_field(
        stem_kernel: int,
        pre_blocks: Sequence[Tuple[int, int]],
        downs: Sequence[Tuple[int, int]],
        level_blocks: Sequence[Sequence[Tuple[int, int]]],
        context_kernel: int = 0,
        seq_len: Optional[int] = None) -> Dict[str, object]:
    """Receptive field of the convolutional path, in base pairs.

    `pre_blocks` and each entry of `level_blocks` are lists of
    (kernel, max_dilation); `downs` is a list of (kernel, stride). Per-level
    fields come back coarsest first, matching the encoder's output order.

    Checked against the architecture contract: stem k=5, six blocks k=7 at
    dilation 1 and five downs (k=4, s=2) gives 134 bp of 800; at the default
    dilations 242 bp, and with the depthwise k=15 690 bp -- still short, which
    is why the billboard makes level 0 global. That kernel covers a whole map
    only if the map has 15 positions: at 25 windows full coverage needs k = 49.
    """
    rf, jump = 1, 1
    rf += (int(stem_kernel) - 1) * jump
    for k, d in pre_blocks:
        rf += (int(k) - 1) * int(d) * jump
    per_level: List[Dict[str, float]] = []
    for i, (k, s) in enumerate(downs):
        rf += (int(k) - 1) * jump
        jump *= int(s)
        for k2, d2 in (level_blocks[i] if i < len(level_blocks) else ()):
            rf += (int(k2) - 1) * int(d2) * jump
        per_level.append({"rf_bp": float(rf), "jump_bp": float(jump)})
    coarse = dict(per_level[-1]) if per_level else {"rf_bp": float(rf),
                                                    "jump_bp": float(jump)}
    rf_with_context = coarse["rf_bp"]
    if context_kernel and context_kernel > 1:
        rf_with_context += (int(context_kernel) - 1) * coarse["jump_bp"]
    out: Dict[str, object] = {
        # Coarsest first, like the encoder's return value.
        "rf_bp_by_level": [d["rf_bp"] for d in reversed(per_level)],
        "jump_bp_by_level": [d["jump_bp"] for d in reversed(per_level)],
        "rf_bp_coarsest": coarse["rf_bp"],
        "rf_bp_coarsest_with_context": float(rf_with_context),
    }
    # Span across levels, the number the hierarchy falsifier stands on. At
    # fields [898, 338, 314, 302, 296] and jumps [32, 16, 8, 4, 2] the jump
    # changes 16x and L1..L4 only 1.14x: sampling density, not range.
    _lv = [float(v) for v in out["rf_bp_by_level"]]
    out["rf_span_ratio"] = (max(_lv) / min(_lv)) if _lv and min(_lv) > 0 else 0.0
    out["rf_bp_finest"] = min(_lv) if _lv else 0.0
    # Overlap of neighbouring windows, 1 - jump/field. `rf_span_ratio` alone
    # can rise through L0 while the finest level still carries 400 shifted
    # copies of one view, measured 1 - 2/296 = 0.9932.
    _jp = [float(v) for v in out["jump_bp_by_level"]]
    out["rf_overlap_by_level"] = [
        (round(1.0 - (j / r), 4) if r > 0 else 0.0)
        for r, j in zip(_lv, _jp)]
    # The list is coarsest first, so the finest level is [-1].
    out["rf_overlap_finest"] = (out["rf_overlap_by_level"][-1]
                                if out["rf_overlap_by_level"] else 0.0)
    if seq_len:
        out["rf_coverage_coarsest"] = coarse["rf_bp"] / float(seq_len)
        out["rf_coverage_coarsest_with_context"] = (float(rf_with_context)
                                                    / float(seq_len))
        out["rf_coverage_by_level"] = [round(v / float(seq_len), 4)
                                       for v in _lv]
    return out


class PromoterEncoder(nn.Module):
    """Bases (+ occupancy, + anchor, + condition) -> one map per ladder level.

    Contract:
        forward(x: [B, 4, L], occupancy: [B, M, L] | None,
                context: [B, Dc] | None) -> list[Tensor], coarsest first.

    Order of operations:

        stem (k=5)                      one conv over [ACGT | occupancy | anchor]
        6 x residual block (k=7)        0.48545 -> 0.70010 in epoch 1; phase
        conditioning cross-attention    after the blocks, before the tower
        5 x down (k=2s, stride s)       800 -> 400 -> 200 -> 100 -> 50 -> 25
          [+ optional per-level blocks, SE forbidden]
        billboard attention on L0       queries from 25 windows, keys from 800
        damped global context on L0     depthwise k=15, RMS-damped

    The billboard runs before the global context so the smoothing module sees
    a map already enriched with factor combinations, and both touch only the
    coarsest map: making levels 1..4 global would destroy the factorisation
    the diffusion prior is built on. Nothing disables silently -- every
    configuration that turns a mechanism into a no-op is recorded by
    `inert_keys()`.
    """

    def __init__(self,
                 dim: int = 128,
                 strides: Sequence[int] = (2, 2, 2, 2, 2),
                 occupancy_channels: int = 0,
                 occupancy_bottleneck: int = 0,
                 # Occupancy channel-selection gate
                 occupancy_gate: bool = False,
                 occupancy_gate_init: float = 4.0,
                 # Component 8
                 anchor: bool = True,
                 anchor_linear: bool = True,
                 anchor_log: bool = False,
                 anchor_fourier: int = 0,
                 anchor_min_period: float = 10.0,
                 # Components 1, 2, 4
                 stem_kernel: int = 5,
                 pre_blocks: int = 6,
                 pre_kernel: int = 7,
                 pre_dilations: Sequence[int] = (1, 2, 4),
                 pre_se_stat: Optional[str] = None,
                 pre_se_blocks: int = 0,
                 level_blocks: int = 0,
                 level_kernel: int = 3,
                 # A flat tuple is the same list at every level; a list of
                 # lists gives each level its own, finest first.
                 level_dilations: Sequence = (1,),
                 level_gate0: object = 0.1,
                 # Gate on the 1x1 bypass; 1.0 builds no parameter at all.
                 skip_gate0: object = 1.0,
                 expand: int = 2,
                 dropout: float = 0.1,
                 dropout_kind: str = "element",
                 norm: str = "batch",
                 norm_groups: int = 8,
                 skip: bool = True,
                 # Component 5
                 context_kernel: int = 15,
                 context_share: float = 0.3,
                 context_gate0: float = 0.5,
                 # Component 6
                 billboard_layers: int = 1,
                 billboard_heads: int = 4,
                 billboard_dim_head: int = 32,
                 billboard_null_key: bool = True,
                 billboard_key_pool: int = 1,
                 # Component 7
                 cond_dim: int = 0,
                 cond_tokens: int = 1,
                 cond_heads: int = 4,
                 cond_dim_head: int = 32,
                 cond_drop_prob: float = 0.0,
                 attn_dropout: float = 0.0,
                 diag_every: int = 50):
        super().__init__()
        strides = tuple(int(s) for s in strides)
        if not strides:
            raise ValueError("strides must not be empty")
        if any(s < 2 for s in strides):
            # A "down" with stride 1 still shortens the map by one position,
            # an off-by-one that surfaces as a shape error five levels later.
            raise ValueError("strides must be >= 2, got %s" % (strides,))
        self.dim = int(dim)
        self.strides = strides
        self.n_levels = len(strides)
        self.occupancy_channels = int(occupancy_channels)
        self.diag_every = max(int(diag_every), 1)
        self._inert: List[Tuple[str, str]] = []
        self._diag: Dict[str, torch.Tensor] = {}
        self._diag_step = 0
        self._last_mode = "never"
        self.level_lengths: Tuple[int, ...] = ()

        # Component 8: positional anchor
        if anchor:
            self.anchor = PositionalAnchor(linear=anchor_linear,
                                           log=anchor_log,
                                           fourier=int(anchor_fourier),
                                           min_period=float(anchor_min_period))
            anchor_ch = self.anchor.n_channels
        else:
            self.anchor = None
            anchor_ch = 0
            # A legitimate ablation, and it must be visible as one.
            self._inert.append(("anchor", "anchor=False: the measured "
                                          "ATG-relative channel is absent"))
        self.anchor_channels = anchor_ch

        # Occupancy bottleneck: one projection shared by the stem and the
        # billboard. The signal is effectively ~11-dimensional (participation
        # ratio 10.8 of 284), so full width pays for repetition. k >=
        # occupancy_channels or k <= 0 means off.
        _k = int(occupancy_bottleneck)
        if 0 < _k < self.occupancy_channels:
            self.occ_proj = nn.Conv1d(self.occupancy_channels, _k, 1)
            self.occupancy_effective = _k
        else:
            self.occ_proj = None
            self.occupancy_effective = self.occupancy_channels
            if _k > 0:
                self._inert.append(
                    ("occupancy_bottleneck",
                     "%d >= occupancy_channels %d: the projection would be a "
                     "widening, so there is none"
                     % (_k, self.occupancy_channels)))

        self.in_channels = 4 + self.occupancy_effective + anchor_ch
        # After the bottleneck the occupancy slice is occupancy_effective wide,
        # not occupancy_channels: the falsifiers must cut what the stem sees.
        self.slice_nucleotide = (0, 4)
        self.slice_occupancy = (4, 4 + self.occupancy_effective)
        self.slice_anchor = (4 + self.occupancy_effective, self.in_channels)

        # Component 1: stem
        if stem_kernel % 2 == 0:
            raise ValueError("stem_kernel must be odd, got %d" % stem_kernel)
        # Not zero-initialised, which is load-bearing: a zero input branch
        # gives d(output)/d(that input) = 0, so the anchor falsifier would
        # pass trivially. Reached through the `stem_conv` property rather than
        # stored twice: one module under two names writes both to state_dict.
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.dim, int(stem_kernel),
                      padding=int(stem_kernel) // 2),
            _make_norm(norm, self.dim, norm_groups), nn.SiLU())

        # Component 2: six residual blocks at full resolution
        if pre_se_stat is not None and int(pre_se_blocks) <= 0:
            raise ValueError(
                "pre_se_stat=%r with pre_se_blocks=0 would build no SE at all "
                "-- this is the exact shape of the bug that made `se_stat` "
                "dead for a whole family (gated by a count defaulting to 0). "
                "Set pre_se_blocks > 0 or leave pre_se_stat=None."
                % (pre_se_stat,))
        self.pre = nn.ModuleList(
            DilatedResidualBlock(
                self.dim, kernel_size=int(pre_kernel),
                dilations=pre_dilations, expand=int(expand),
                dropout_kind=str(dropout_kind),
                se_stat=(pre_se_stat if i < int(pre_se_blocks) else None),
                dropout=float(dropout), norm=norm, norm_groups=norm_groups)
            for i in range(int(pre_blocks)))
        if int(pre_blocks) == 0:
            self._inert.append(("pre_blocks", "0 blocks: the measured "
                                              "0.48545 -> 0.70010 repair is off"))
        self.pre_se_blocks = int(pre_se_blocks) if pre_se_stat else 0

        # Component 7: conditioning, after the blocks and before the tower
        if int(cond_dim) > 0:
            self.cond = ConditionCrossAttention(
                self.dim, int(cond_dim), tokens=int(cond_tokens),
                heads=int(cond_heads), dim_head=int(cond_dim_head),
                dropout=float(attn_dropout), drop_prob=float(cond_drop_prob))
            if float(cond_drop_prob) <= 0.0:
                self._inert.append(
                    ("cond_drop_prob", "0.0: no unconditional pass is ever "
                                       "trained, so classifier-free guidance "
                                       "at sampling time is uncalibrated"))
            if int(cond_tokens) == 1:
                # A softmax over one key is 1, so the module reduces to a
                # broadcast bias. That may be wanted; it is not attention.
                self._inert.append(
                    ("cond_tokens", "1: softmax over a single key is "
                                    "identically 1, so the conditioning is a "
                                    "broadcast bias and `cond_H` is 0 by "
                                    "construction; pass the species / text / "
                                    "left-flank / right-flank vectors as 4 "
                                    "tokens for the attention to mean "
                                    "anything"))
        else:
            self.cond = None
            self._inert.append(("cond_dim", "0: conditioning cross-attention "
                                            "is not built"))

        # Component 3: the downsampling tower
        downs, skips = [], []
        for s in strides:
            k = 2 * s
            downs.append(nn.Sequential(
                nn.Conv1d(self.dim, self.dim, k, stride=s, padding=s // 2),
                _make_norm(norm, self.dim, norm_groups), nn.SiLU()))
            # Local residual around each down: without it the path from the
            # stem to the coarsest map crosses five nonlinearities, and the
            # stem alone sees the anchor and the occupancy. A 1x1 conv at the
            # same stride globalises nothing.
            skips.append(nn.Conv1d(self.dim, self.dim, 1, stride=s))
        self.downs = nn.ModuleList(downs)
        self.skips = nn.ModuleList(skips) if skip else None
        # Bypass gate per level. At 1.0 everywhere no parameter is built, so
        # an old checkpoint gains no key -- 1.0 is the identity, not the zero.
        if isinstance(skip_gate0, (list, tuple)):
            if len(skip_gate0) != len(strides):
                raise ValueError("skip_gate0 as a list must have %d entries, "
                                 "it has %d" % (len(strides), len(skip_gate0)))
            _sg = [float(v) for v in skip_gate0]
        else:
            _sg = [float(skip_gate0)] * len(strides)
        if any(v <= 0.0 for v in _sg):
            raise ValueError("skip_gate0 must be > 0 (zero closes the "
                             "gradient path to the stem, and the stem is "
                             "the only place that sees the occupancy and "
                             "the anchors), got %s" % (_sg,))
        if self.skips is not None and any(v != 1.0 for v in _sg):
            self.skip_gate_log = nn.Parameter(
                torch.tensor([math.log(v) for v in _sg],
                             dtype=torch.float32))
        else:
            self.register_parameter("skip_gate_log", None)
        if not skip:
            self._inert.append(("skip", "False: gradient to the stem crosses "
                                        "%d nonlinearities" % len(strides)))

        # Component 4: optional per-level blocks. A list names the levels --
        # one block on each of five costs +18.4 % parameters -- and is
        # ordered finest first, so [0,0,0,1,1] means the two coarsest.
        if isinstance(level_blocks, (list, tuple)):
            self.level_blocks_per = [int(v) for v in level_blocks]
            if len(self.level_blocks_per) != len(strides):
                raise ValueError(
                    'level_blocks as a list must have length %d '
                    '(one per level), but has %d'
                    % (len(strides), len(self.level_blocks_per)))
        else:
            self.level_blocks_per = [int(level_blocks)] * len(strides)
        self.level_blocks_n = max(self.level_blocks_per)
        # Dilations per level, finest first. A level without a block gets (1,)
        # and never uses it; an empty list would be a construction error.
        _ld = (list(level_dilations)
               if isinstance(level_dilations, (list, tuple))
               else [int(level_dilations)])
        if _ld and all(isinstance(v, (list, tuple)) for v in _ld):
            if len(_ld) != len(strides):
                raise ValueError(
                    "level_dilations as a list of lists must have %d entries "
                    "(one per level), but has %d"
                    % (len(strides), len(_ld)))
            self.level_dilations_per = [
                tuple(int(d) for d in v) if len(v) else (1,) for v in _ld]
        else:
            self.level_dilations_per = [
                tuple(int(d) for d in _ld)] * len(strides)
        if isinstance(level_gate0, (list, tuple)):
            if len(level_gate0) != len(strides):
                raise ValueError(
                    "level_gate0 as a list must have %d entries, it has %d"
                    % (len(strides), len(level_gate0)))
            _g0 = [float(v) for v in level_gate0]
        else:
            _g0 = [float(level_gate0)] * len(strides)
        if any(v <= 0.0 for v in _g0):
            # A gate at 0 starves both sides of the gradient.
            raise ValueError("level_gate0 must be > 0 at every level, "
                             "got %s" % (_g0,))
        if self.level_blocks_n > 0:
            self.level = nn.ModuleList(nn.ModuleList(
                # Hard-coded None, not a default: SE squeezes over all of the
                # level, destroying the factorisation the prior needs.
                DilatedResidualBlock(self.dim, kernel_size=int(level_kernel),
                                     dilations=self.level_dilations_per[_lv],
                                     dropout_kind=str(dropout_kind),
                                     expand=int(expand), se_stat=None,
                                     dropout=float(dropout), norm=norm,
                                     norm_groups=norm_groups)
                for _ in range(self.level_blocks_per[_lv]))
                for _lv in range(len(strides)))
            # Residual scalar per level, small but not zero: zero starves the
            # gradient, large injects noise. "gate" is for the no-decay list.
            self.level_gate_log = nn.Parameter(
                torch.tensor([math.log(v) for v in _g0],
                             dtype=torch.float32))
        else:
            self.level = None
            self.register_parameter("level_gate_log", None)

        # Component 6: billboard attention on the coarsest map
        n_bb = int(billboard_layers)
        if n_bb > 0 and self.occupancy_channels > 0:
            self.billboard = nn.ModuleList(
                BillboardAttention(self.dim, self.occupancy_effective,
                                   heads=int(billboard_heads),
                                   dim_head=int(billboard_dim_head),
                                   dropout=float(attn_dropout),
                                   null_key=bool(billboard_null_key),
                                   key_pool=int(billboard_key_pool))
                for _ in range(n_bb))
        else:
            self.billboard = None
            if n_bb > 0:
                self._inert.append(
                    ("billboard_layers", "%d layers requested but "
                                         "occupancy_channels=0, so there are "
                                         "no keys to attend to" % n_bb))
            else:
                self._inert.append(("billboard_layers", "0: the only global "
                                                        "path into the "
                                                        "quantised map is off"))

        # Component 5: damped global context on the coarsest map
        if float(context_share) > 0.0 and int(context_kernel) > 1:
            self.context = DampedGlobalContext(
                self.dim, kernel_size=int(context_kernel),
                share=float(context_share), gate0=float(context_gate0),
                norm=norm, norm_groups=norm_groups)
        else:
            self.context = None
            self._inert.append(("context_share", "%.3g: the global depthwise "
                                                 "module is off"
                                % float(context_share)))

        # Built last so the RNG stream is untouched when the gate is on
        # (`torch.full` does not draw). The name carries `gate`, which is a
        # contract: `no_decay_parameter_names` matches on the leaf name.
        if bool(occupancy_gate) and self.occupancy_channels > 0:
            # +4.0 is sigmoid 0.982, i.e. today's model up to a constant that
            # `occ_proj` absorbs. Not zero, which is dead from step one.
            self.occ_gate_logit = nn.Parameter(
                torch.full((self.occupancy_channels,),
                           float(occupancy_gate_init), dtype=torch.float32))
        else:
            self.register_parameter("occ_gate_logit", None)
            if bool(occupancy_gate):
                self._inert.append(
                    ("occupancy_gate",
                     "occupancy_channels=0: there is nothing to gate"))

    @torch.no_grad()
    def occ_gate_report(self) -> Dict[str, float]:
        """Read out the occupancy gate; empty dict when there is no gate.

        `occ_gate_ef` = (sum g)^2 / sum g^2 is the effective number of channels
        let through. Without the Lagrangian controller the gate feels no
        sparsity pressure, so a value near the channel count means "no reason
        to drop any", not "the model needs them all". `occ_gate_masa44` is the
        share held by the 44 largest gates, 44 being the fewest raw channels
        reconstructing R^2 >= 0.90 of all 252; its uniform floor is 0.1549.
        """
        g = getattr(self, "occ_gate_logit", None)
        if g is None:
            return {}
        g = torch.sigmoid(g.detach().float())
        s = g.sum().clamp_min(1e-12)
        _ef_w = float("nan")
        _wp = getattr(self, "occ_proj", None)
        if _wp is not None and getattr(_wp, "weight", None) is not None:
            _kol = _wp.weight.detach().float().reshape(
                _wp.weight.shape[0], _wp.weight.shape[1], -1)
            _kol = _kol.pow(2).sum(dim=(0, 2)).sqrt()
            if int(_kol.numel()) == int(g.numel()):
                _v = g * _kol
                _ef_w = float(_v.sum().pow(2)
                              / _v.pow(2).sum().clamp_min(1e-12))
        srt = torch.sort(g, descending=True).values
        k = min(44, int(g.numel()))
        return {
            "occ_gate_ef": float((s * s) / g.pow(2).sum().clamp_min(1e-12)),
            # What the controller acts on: `occ_gate_ef` normalised to [0, 1),
            # because a target in channels depends on how many there are.
            "occ_gate_rzadkosc": float(
                1.0 - ((s * s) / g.pow(2).sum().clamp_min(1e-12))
                / float(g.numel())),
            # Falsifier for a gate that plays with scale: a gate can fall on
            # channel i while `occ_proj` raises column W[:, i], so `ef` drops
            # while the bottleneck receives the same thing. `ef_w` is the same
            # statistic on g_i * ||W[:, i]||; selection needs both to fall.
            "occ_gate_ef_w": _ef_w,
            "occ_gate_srednia": float(g.mean()),
            "occ_gate_min": float(g.min()),
            "occ_gate_max": float(g.max()),
            "occ_gate_ponizej_01": float((g < 0.1).float().mean()),
            "occ_gate_masa44": float(srt[:k].sum() / s),
            "occ_gate_kanaly": float(g.numel()),
        }

    def occ_gate_sparsity(self) -> Optional[torch.Tensor]:
        """Gate sparsity as a tensor that carries gradient.

        `1 - ef / C` over [0, 1 - 1/C]: 0 under a uniform distribution,
        maximal when one channel is left. Separate from `occ_gate_report`
        although the arithmetic is the same, because that method is
        `@torch.no_grad` and returns floats: a loss term built from it would
        have gradient exactly zero while every number in the log looked
        identical to a working controller.
        """
        g = getattr(self, "occ_gate_logit", None)
        if g is None:
            return None
        s = torch.sigmoid(g.float())
        ef = s.sum().pow(2) / s.pow(2).sum().clamp_min(1e-12)
        return 1.0 - ef / float(s.numel())

    def occ_gate_kara(self) -> Optional[torch.Tensor]:
        """L1 loss term on the gate: `mean(sigmoid(logit))`.

        Not the sparsity itself, although sparsity is the goal, because
        `occ_gate_sparsity()` has its maximum at the uniform point and so
        gradient exactly zero there (60 training steps under it left `ef` at
        284.000 of 284). No scale-invariant symmetric measure fixes that: all
        channels are equivalent under it, so the uniform point is stationary
        by symmetry. L1 has non-zero gradient everywhere, but what it does not
        guarantee -- and why `ef_w` ranks equally with `ef` -- is lowering
        `ef` at all, since a uniform decline leaves `ef` invariant.
        """
        g = getattr(self, "occ_gate_logit", None)
        if g is None:
            return None
        return torch.sigmoid(g.float()).mean()

    @property
    def stem_conv(self) -> nn.Conv1d:
        """The stem convolution, whose input channels the falsifiers slice."""
        return self.stem[0]

    # Construction helpers
    @classmethod
    def from_mapping(cls, mapping: Dict[str, object], strict: bool = True
                     ) -> "PromoterEncoder":
        """Build from a plain dict, reporting keys this module does not read.

        `strict=True` raises on an unknown key, `strict=False` records it in
        `inert_keys()`; both are loud, because a silently ignored key is how a
        family of runs ran without a mechanism its config claimed.
        """
        known = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = sorted(k for k in mapping if k not in known)
        if unknown and strict:
            raise ValueError("keys not read by PromoterEncoder: %s"
                             % ", ".join(unknown))
        obj = cls(**{k: v for k, v in mapping.items() if k in known})
        for k in unknown:
            obj._inert.append((k, "passed to the encoder but not an encoder key"))
        return obj

    # Parameter-group contracts
    def manifold_parameters(self) -> List[nn.Parameter]:
        """Empty by construction: the encoder is Euclidean throughout.

        Declared rather than discovered -- an undeclared empty list is how
        `riemannian: true` printed "AdamW" unnoticed for months.
        """
        return []

    def no_decay_parameter_names(self, prefix: str = "") -> List[str]:
        """Fully-qualified names that must be exempt from weight decay.

        Decay at lambda 0.0762 over 34 epochs is a factor 1.8e-5, death for a
        gate, and the trainer's exemption matches a substring of the name, so
        returning the real names removes the guess.
        """
        out = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            leaf = name.split(".")[-1]
            if ("gate" in leaf or "null_" in leaf or leaf.endswith("_log")):
                out.append(prefix + name)
        return sorted(out)

    def _build_input(self, x: torch.Tensor, occupancy: Optional[torch.Tensor],
                     ablate: Sequence[str] = ()) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError("x must be [B, 4, L], got %s" % (tuple(x.shape),))
        B, _, L = x.shape
        parts = [torch.zeros_like(x) if "nucleotide" in ablate else x]
        if self.occupancy_channels:
            if occupancy is None:
                raise ValueError(
                    "encoder built with occupancy_channels=%d but occupancy is "
                    "None -- a zero-filled fallback would look like a working "
                    "run with a dead motif channel"
                    % self.occupancy_channels)
            if occupancy.shape[1] != self.occupancy_effective:
                raise ValueError(
                    "occupancy has %d channels, expected %d (occupancy_channels"
                    " %d, bottleneck %s) -- `_forward` applies the "
                    "projection, so by this point the tensor is ALREADY "
                    "narrowed"
                    % (occupancy.shape[1], self.occupancy_effective,
                       self.occupancy_channels,
                       "absent" if self.occ_proj is None else "present"))
            if occupancy.shape[-1] != L:
                raise ValueError(
                    "occupancy length %d != sequence length %d; resampling it "
                    "would move motifs relative to the anchor"
                    % (occupancy.shape[-1], L))
            occ = occupancy.to(x.dtype)
            parts.append(torch.zeros_like(occ) if "occupancy" in ablate else occ)
        elif occupancy is not None:
            raise ValueError("occupancy passed, but the encoder was built with "
                             "occupancy_channels=0")
        if self.anchor is not None:
            a = self.anchor(L, x.device, x.dtype).expand(B, -1, -1)
            # Mask the operand, not the result.
            parts.append(torch.zeros_like(a) if "anchor" in ablate else a)
        return torch.cat(parts, dim=1)

    def output_lengths(self, seq_len: int) -> List[int]:
        """Map lengths per level, coarsest first, from the actual conv params.

        Computed rather than assumed, because the length is a config value:
        nothing here may hard-code 800 or five levels.
        """
        L = int(seq_len)
        out: List[int] = []
        for down in self.downs:
            conv = down[0]
            k, s = conv.kernel_size[0], conv.stride[0]
            p, d = conv.padding[0], conv.dilation[0]
            L = (L + 2 * p - d * (k - 1) - 1) // s + 1
            out.append(L)
        return out[::-1]

    def forward(self, x: torch.Tensor,
                occupancy: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        return self._forward(x, occupancy, context)

    def _forward(self, x: torch.Tensor,
                 occupancy: Optional[torch.Tensor] = None,
                 context: Optional[torch.Tensor] = None,
                 ablate: Sequence[str] = ()) -> List[torch.Tensor]:
        # Gate, then projection, both before the paths diverge: the stem gets
        # the result through `_build_input` and the billboard reads the same
        # tensor below. Narrowing in one would give two occupancies.
        if getattr(self, "occ_gate_logit", None) is not None \
                and occupancy is not None:
            occupancy = (occupancy.to(x.dtype)
                         * torch.sigmoid(self.occ_gate_logit)
                         .to(x.dtype).view(1, -1, 1))
        if self.occ_proj is not None and occupancy is not None:
            occupancy = self.occ_proj(occupancy.to(x.dtype))
        we = self._build_input(x, occupancy, ablate)
        L = we.shape[-1]
        lens = self.output_lengths(L)
        if min(lens) < 1:
            raise ValueError(
                "sequence length %d collapses under strides %s: level lengths "
                "would be %s. The ladder needs at least one window per level."
                % (L, self.strides, lens[::-1]))
        heavy = (self._diag_step % self.diag_every) == 0
        self._diag_step += 1
        self._last_mode = "train" if self.training else "eval"

        h = self.stem(we)
        if heavy:
            self._stem_group_diagnostics(we)

        for blk in self.pre:
            h = blk(h)

        if self.cond is not None:
            h = self.cond(h, context)
        elif context is not None:
            raise ValueError("context passed, but the encoder was built with "
                             "cond_dim=0 -- a silently ignored condition is a "
                             "conditional model that is not conditioned")

        feats: List[torch.Tensor] = []
        for i, down in enumerate(self.downs):
            h_in = h
            h = down(h)
            if self.skips is not None:
                sk = self.skips[i](h_in)
                if sk.shape[-1] != h.shape[-1]:
                    sk = self._match_len(sk, h.shape[-1])
                if self.skip_gate_log is not None:
                    sk = torch.exp(self.skip_gate_log[i]).to(sk.dtype) * sk
                h = h + sk
            if self.level is not None:
                y = h
                for blk in self.level[i]:
                    y = blk(y)
                # At gate 0 this is the baseline bit for bit.
                h = h + torch.exp(self.level_gate_log[i]).to(h.dtype) * (y - h)
            feats.append(h)

        coarse = feats[-1]
        if self.billboard is not None:
            # `occupancy` cannot be None here and is deliberately not guarded
            # on: a guard would turn a missing input into a skipped mechanism.
            # Keys come from the full-length occupancy so a 32 bp window can
            # ask about a motif 700 bp away.
            occ = occupancy.to(coarse.dtype)
            if "occupancy" in ablate:
                occ = torch.zeros_like(occ)
            for layer in self.billboard:
                coarse = layer(coarse, occ)
        if self.context is not None:
            coarse = self.context(coarse)
        feats[-1] = coarse

        # Cross-checked against the lengths predicted from the conv parameters
        # so that `output_lengths` cannot drift from what the tower builds.
        self.level_lengths = tuple(int(f.shape[-1]) for f in feats[::-1])
        if self.level_lengths != tuple(lens):
            raise RuntimeError(
                "output_lengths predicted %s but the tower produced %s"
                % (lens, list(self.level_lengths)))
        with torch.no_grad():
            self._diag["seq_len"] = torch.tensor(float(L))
            self._diag["coarse_map_rms"] = _rms_per_sample(coarse).mean().detach()
        # Coarsest first, matching `ConeLadder.forward(maps)`.
        return feats[::-1]

    @staticmethod
    def _match_len(t: torch.Tensor, size: int) -> torch.Tensor:
        if t.shape[-1] == size:
            return t
        return F.interpolate(t, size=size, mode="nearest")

    @torch.no_grad()
    def _stem_group_diagnostics(self, we: torch.Tensor) -> None:
        """RMS of each input group's contribution to the stem pre-activation.

        Costs extra convolutions, hence `diag_every`. It answers what the
        weight norms cannot: a group can hold large weights and contribute
        nothing if its channels are near-constant.
        """
        w = self.stem_conv.weight
        groups = {"nucleotide": self.slice_nucleotide,
                  "occupancy": self.slice_occupancy,
                  "anchor": self.slice_anchor}
        for name, (a, b) in groups.items():
            if b <= a:
                continue
            part = F.conv1d(we[:, a:b], w[:, a:b],
                            padding=self.stem_conv.padding[0])
            self._diag["stem_rms_" + name] = _rms_per_sample(part).mean()
            self._diag["stem_wnorm_" + name] = (w[:, a:b].norm()
                                                / math.sqrt(float(b - a)))

    def receptive_field(self) -> Dict[str, object]:
        """Receptive field read from the built modules, not from the config.

        Reading the config would let this describe a model it does not build.
        """
        pre = [(b.kernel_size, b.max_dilation) for b in self.pre]
        downs = [(d[0].kernel_size[0], d[0].stride[0]) for d in self.downs]
        levels: List[List[Tuple[int, int]]] = []
        for i in range(len(self.downs)):
            if self.level is None:
                levels.append([])
            else:
                levels.append([(b.kernel_size, b.max_dilation)
                               for b in self.level[i]])
        ctx_k = self.context.kernel_size if self.context is not None else 0
        seq_len = int(self._diag["seq_len"].item()) if "seq_len" in self._diag \
            else None
        out = convolutional_receptive_field(
            self.stem_conv.kernel_size[0], pre, downs, levels, ctx_k, seq_len)
        if self.level_lengths:
            w = self.level_lengths[0]          # coarsest map, 25 windows at 800
            out["context_span_windows"] = float(ctx_k)
            out["context_full_coverage_kernel"] = float(2 * w - 1)
            out["context_covers_map"] = bool(ctx_k >= 2 * w - 1)
        out["global_paths"] = [
            p for p, on in (
                ("billboard_attention(all %d bp -> coarse windows)"
                 % (seq_len or 0), self.billboard is not None),
                ("depthwise_context(k=%d on the coarse map only)" % ctx_k,
                 self.context is not None),
                ("conditioning_cross_attention(tokens, not positions)",
                 self.cond is not None))
            if on]
        return out

    def encoder_report(self) -> Dict[str, object]:
        """Everything a log line needs, floats converted exactly once."""
        rep: Dict[str, object] = {
            "in_channels": self.in_channels,
            "channels_nucleotide": 4,
            "channels_occupancy": self.occupancy_channels,
            # The stem and all three slices count `occupancy_effective`, so
            # without this key the log reads 4 + occupancy + anchor !=
            # in_channels whenever the bottleneck is on. Added beside the raw
            # count rather than replacing it, that name being grepped.
            "channels_occupancy_effective": self.occupancy_effective,
            "channels_anchor": self.anchor_channels,
            "anchor_names": (self.anchor.channel_names()
                             if self.anchor is not None else []),
            "levels": self.n_levels,
            "strides": list(self.strides),
            "level_lengths": list(self.level_lengths),
            "pre_blocks": len(self.pre),
            # The dilations, not just the block count: a pre-block reaches
            # (k-1)*max(d), so six blocks give 288 bp at (1,2,4,8) and 36 bp
            # at (1,) for the same log line and parameter count.
            "pre_dilations": next(
                (list(b.dilations) for b in self.pre), []),
            "pre_kernel": next(
                (int(b.kernel_size) for b in self.pre), 0),
            "level_dilations": next(
                (list(b.dilations) for lv in (self.level or []) for b in lv),
                []),
            "pre_se_blocks": self.pre_se_blocks,
            "level_blocks": self.level_blocks_n,
            "billboard_layers": (0 if self.billboard is None
                                 else len(self.billboard)),
            "conditioning_touches_latent": self.cond is not None,
            "parameters": sum(p.numel() for p in self.parameters()),
            "diag_mode": self._last_mode,
            "diag_step": self._diag_step,
        }
        rep.update(self.receptive_field())
        # Structural anchor observable, ~1 at init by construction and so
        # readable only after training: a decay toward 0 says it is unused.
        w = self.stem_conv.weight.detach()
        na, nb = self.slice_nucleotide
        aa, ab = self.slice_anchor
        nuc = float(w[:, na:nb].norm()) / math.sqrt(float(nb - na))
        rep["anchor_weight_rms"] = (
            float(w[:, aa:ab].norm()) / math.sqrt(float(ab - aa))
            if ab > aa else 0.0)
        rep["nucleotide_weight_rms"] = nuc
        rep["anchor_weight_ratio"] = (rep["anchor_weight_rms"] / nuc
                                      if nuc > 0 else 0.0)
        if self.occupancy_channels:
            oa, ob = self.slice_occupancy
            rep["occupancy_weight_rms"] = (float(w[:, oa:ob].norm())
                                           / math.sqrt(float(ob - oa)))
        sources: List[Dict[str, torch.Tensor]] = [self._diag]
        if self.context is not None:
            sources.append(self.context.report())
        if self.cond is not None:
            sources.append(self.cond.report())
        if self.billboard is not None:
            for i, layer in enumerate(self.billboard):
                sources.append({("%s_l%d" % (k, i) if len(self.billboard) > 1
                                 else k): v
                                for k, v in layer.report().items()})
        if self.level is not None and self.level_gate_log is not None:
            g = torch.exp(self.level_gate_log.detach())
            sources.append({"level_gate_mean": g.mean(),
                            "level_gate_min": g.min()})
            # `level_gate_log` has length len(strides) whatever levels carry a
            # block, so the mean and minimum mix in gates the forward pass
            # never uses, and blocks on intermediate levels get no own number.
            rep["level_gate_by_level"] = [round(float(v), 4) for v in g]
            rep["level_blocks_per"] = list(self.level_blocks_per)
            # Without this, two models of different dilations print alike.
            rep["level_dilations_by_level"] = [
                list(v) for v in self.level_dilations_per]
        if getattr(self, "skip_gate_log", None) is not None:
            rep["skip_gate_by_level"] = [
                round(float(v), 4)
                for v in torch.exp(self.skip_gate_log.detach())]
        for src in sources:
            for k, v in src.items():
                rep[k] = float(v) if isinstance(v, torch.Tensor) else v
        rep["inert_keys"] = list(self._inert)
        return rep

    def inert_keys(self) -> List[Tuple[str, str]]:
        """Settings that make a mechanism a no-op, with the reason."""
        return list(self._inert)

    def falsifiers(self) -> List[FalsifierRow]:
        """Quantities that can show a mechanism here is dead or broken.

        An unmeasured row reports "?", never a pass: silence read as a passed
        gate is how three models ran past a gate that never fired.
        """
        rep = self.encoder_report()

        def get(key: str) -> Optional[float]:
            v = rep.get(key)
            return float(v) if isinstance(v, (int, float)) else None

        rows: List[FalsifierRow] = [
            FalsifierRow(
                "context_dom_effective", get("context_dom_effective"), "<",
                (0.5,),
                "22.64 broke a run (0.7377 -> 0.7146, gap +0.166); repaired "
                "runs sat at 0.26-0.29"),
            FalsifierRow(
                "context_dom_raw", get("context_dom_raw"), "<", (5.0,),
                "the branch's own scale before damping; 12.7x was the "
                "runaway. Reported because the EFFECTIVE value is bounded by "
                "`share` and would be a tautological falsifier alone"),
            FalsifierRow(
                "context_covers_map", 1.0 if rep.get("context_covers_map")
                else 0.0, "info", (),
                "k=15 covered the reference's 15-window map exactly; the "
                "coarse map here has %s windows, so every position sees every "
                "other only at k=%s"
                % (self.level_lengths[0] if self.level_lengths else "?",
                   rep.get("context_full_coverage_kernel", "?"))),
            FalsifierRow(
                "rf_span_ratio", get("rf_span_ratio"), ">", (8.0,),
                "max/min receptive field across levels. Measured on the run "
                "this axis is meant to fix: 898/296 = 3.03 at strides "
                "32/2 = 16, i.e. the levels sample more sparsely but see "
                "the same thing. The 8.0 threshold is HALF the stride "
                "ratio: below it the tower is still copying the "
                "pre-blocks' reach instead of stratifying it. Readable "
                "BEFORE epoch 1 -- it is integer arithmetic over the "
                "built modules, not a statistic"),
            FalsifierRow(
                "rf_bp_finest", get("rf_bp_finest"), "<", (60.0,),
                "receptive field of the FINEST level. It was 296 bp, i.e. "
                "37 % of the sequence: the level called the motif level "
                "saw a third of the promoter. The 60 bp threshold is the "
                "upper bound of the motif-plus-spacer scale (fungal PWMs "
                "are 6-15 bp, two helical turns about 21 bp)"),
            FalsifierRow(
                "rf_overlap_finest", get("rf_overlap_finest"), "<", (0.95,),
                "1 - stride/receptive field at the FINEST level, i.e. how "
                "much ground two neighbouring windows share. Measured on "
                "the run this axis is meant to fix: 1 - 2/296 = 0.9932, "
                "so 400 L4 positions carried 400 shifted copies of the "
                "same view and the fine levels yielded 2 bits out of "
                "3.17. The 0.95 threshold is twenty distinct windows per "
                "receptive field; at stride 2 only a field below 40 bp "
                "meets it, which is the motif scale. It is NOT satisfiable "
                "by construction: the baseline FAILS it (0.9932), and it "
                "measures something other than `rf_span_ratio`, which can "
                "be raised by inflating L0 alone while the finest level "
                "stays as it was"),
            FalsifierRow(
                "rf_coverage_coarsest", get("rf_coverage_coarsest"), "info",
                (),
                "convolution alone reaches 134 bp of 800 (16.8 %) at dilation "
                "1; below 1.0 is EXPECTED and is why global paths exist"),
        ]
        n_bb = int(rep.get("billboard_layers", 0) or 0)
        if n_bb:
            suffix = "" if n_bb == 1 else "_l0"
            hmax = get("billboard_H_max" + suffix)
            rows += [
                FalsifierRow(
                    "billboard_H" + suffix, get("billboard_H" + suffix),
                    "range",
                    (0.05 * (hmax or 1.0), 0.95 * (hmax or 1.0)),
                    "upper end = uniform attention, i.e. the mechanism equals "
                    "a plain projection of the occupancy mean; lower end = "
                    "one key, the degeneracy measured as krzyz_H 0.063 of "
                    "7.966. READ AFTER TRAINING: on a fresh model random Q/K "
                    "give a near-uniform softmax, measured here as 9.646 of "
                    "9.646, so this row is AT the upper bound by construction "
                    "at step 0 and says nothing about the mechanism yet"),
                FalsifierRow(
                    "billboard_share" + suffix,
                    get("billboard_share" + suffix), ">", (0.05,),
                    "||W_attention|| / ||W_map|| in the merge conv; 0 means "
                    "the model rejected the channel although it had no free "
                    "gate to do it with"),
                FalsifierRow(
                    "billboard_null_mass" + suffix,
                    get("billboard_null_mass" + suffix), "<", (0.95,),
                    "attention mass on the learned null key; ~1 means every "
                    "window abstains and the billboard is off by the model's "
                    "own choice"),
            ]
        if self.cond is not None:
            rows.append(FalsifierRow(
                "cond_share", get("cond_share"), ">", (0.05,),
                "conditioning merged structurally; 0 = the condition does "
                "not reach the map"))
            if self.cond.tokens > 1:
                # With one token the entropy is 0 by arithmetic, and a row
                # that can only fail is noise.
                rows.append(FalsifierRow(
                    "cond_H", get("cond_H"), "<",
                    (max(get("cond_H_max") or 0.0, 1e-9),),
                    "at exactly H_max every position takes the same mixture "
                    "of tokens, i.e. conditioning degenerates to a constant "
                    "bias. Same caveat as billboard_H: measured 1.998 of "
                    "2.000 on a fresh model, so read it after training"))
        if self.level is not None:
            rows.append(FalsifierRow(
                "level_gate_min", get("level_gate_min"), ">", (1e-3,),
                "per-level residual scalars; gates died to weight decay four "
                "times in this project, so they are on "
                "`no_decay_parameter_names()`"))
        if self.anchor is not None:
            rows.append(FalsifierRow(
                "anchor_weight_ratio", get("anchor_weight_ratio"), ">",
                (0.05,),
                "structural only: ~1 at init by construction, so it is "
                "readable AFTER training; run `anchor_falsifier()` for the "
                "functional test"))
        return rows

    def format_falsifiers(self) -> str:
        return "\n".join(r.format() for r in self.falsifiers())

    @torch.no_grad()
    def anchor_falsifier(self, x: torch.Tensor,
                         occupancy: Optional[torch.Tensor] = None,
                         context: Optional[torch.Tensor] = None
                         ) -> Dict[str, object]:
        """Zeroing the positional anchor must change the output.

        The data licence -- the promoter abuts ATG -- makes the channel
        meaningful, not used; this is the functional test. The positive
        control is what can refute the test itself: zeroing the four
        nucleotide channels must change the output much more, and if both
        deltas are ~0 the measurement is broken rather than the anchor. Run in
        eval, or the extra passes update the normalisation statistics.
        """
        if self.anchor is None:
            return {"available": False,
                    "reason": "encoder built with anchor=False"}
        was_training = self.training
        self.eval()
        try:
            base = self._forward(x, occupancy, context)
            no_anchor = self._forward(x, occupancy, context, ablate=("anchor",))
            no_bases = self._forward(x, occupancy, context,
                                     ablate=("nucleotide",))
        finally:
            self.train(was_training)

        def rel(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[float]:
            return [float((p - q).norm() / q.norm().clamp_min(1e-12))
                    for p, q in zip(a, b)]

        d_anchor = rel(no_anchor, base)
        d_bases = rel(no_bases, base)
        worked = max(d_bases) > 1e-4
        return {
            "available": True,
            # Coarsest first, like the encoder output.
            "delta_anchor_by_level": d_anchor,
            "delta_nucleotide_by_level": d_bases,
            "delta_anchor_max": max(d_anchor),
            "delta_nucleotide_max": max(d_bases),
            "ratio_anchor_over_nucleotide": (max(d_anchor) / max(d_bases)
                                             if worked else float("nan")),
            "measurement_valid": bool(worked),
            "anchor_alive": bool(worked and max(d_anchor) > 1e-3),
            "threshold": 1e-3,
            "note": ("measurement_valid=False means the positive control did "
                     "not fire: the model output does not depend on the BASES "
                     "either, so nothing can be concluded about the anchor"),
        }

    def extra_repr(self) -> str:
        # The EFFECTIVE occupancy width, so the sum shown adds up to
        # `in_ch`: with a bottleneck active the raw count does not, and the
        # repr read as `4 + 284 + 1 = 69`.
        return ("dim=%d, in_ch=%d (4 + %d occupancy + %d anchor), "
                "strides=%s, pre_blocks=%d, billboard=%d"
                % (self.dim, self.in_channels, self.occupancy_effective,
                   self.anchor_channels, self.strides, len(self.pre),
                   0 if self.billboard is None else len(self.billboard)))
