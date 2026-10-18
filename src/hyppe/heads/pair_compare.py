"""Compare two promoters BEFORE pooling, position by position in the code.

Sensitivity to a point difference has to come from the way the comparison is
made, because the pair sampler cannot supply it. Measured on validation, 4000
pairs per kind over a 775-position code: a random pair within one genome
differs at 85.7 % of code positions, the nearest pair inside a cell at 81.7 %,
and orthologues at 67.1 %. The distribution of code distance within a cell is
a narrow peak -- 99.9 % of 793 624 pairs lie in [630, 700] -- so the class
"codewise close pair in one genome" does not exist in the supply rather than
being rare; and rho(dcode, |dy|) = +0.0254, so code distance carries no
information about the difference in strength.

`PairwiseRankingHead` receives `[B, in_dim]` AFTER pooling and computes
`s(A) - s(B)`, so which positions differ reaches it only through that narrow
channel. Here the comparison happens on `[B, C, L]`, before the averaging.

Antisymmetry is structural rather than learned: `delta(A, B) = -delta(B, A)`
bit for bit, because the attention weight depends on `|h_a - h_b|`, which is
symmetric under the swap, while the value depends on `h_a - h_b`, which is
antisymmetric. That is the property `P(A>B) + P(B>A) = 1` stands on.

`falsify()` computes three numbers on real tensors. F1: an identically coded
pair must give delta exactly 0, and the class exists in every run because
4.0-4.5 % of draws are orthologues with a bit-identical 800 bp sequence. F2:
max |delta(A,B) + delta(B,A)| at float precision. F3 is the positive control
-- changing ONE position at one level must move delta -- without which F1
passes trivially on a module that returns a constant zero.

At `weight = 0.0` (the default) `forward` returns exactly zero and the module
changes nothing, but it is still BUILT so that `state_dict` has one shape
whichever regime a run uses. That constant is the only off switch: the `gate`
parameters start at 1.0 and must not be used as one, for the reason recorded
at their initialisation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn

__all__ = ["PairComparison"]


class PairComparison(nn.Module):
    """`[B, C_k, L_k]` per level x 2 elements -> `[B]`, added to the delta.

    `channels` is the channel count per level, COARSEST FIRST, in the order
    `encode` returns `z`. `hidden` is the width of the shared per-level
    projection, and `weight` multiplies the whole term, 0.0 being bit for bit
    the run without this module.

    `levels` selects which levels take part; empty means all. Level 0 has 25
    positions over 800 bp and level 4 has 400, so choosing levels is choosing
    the RESOLUTION of the comparison.
    """

    def __init__(self, channels: Sequence[int], hidden: int = 64,
                 weight: float = 0.0, dropout: float = 0.0,
                 levels: Sequence[int] = ()) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if not self.channels:
            raise ValueError("PairComparison needs at least one level")
        self.hidden = int(hidden)
        self.weight = float(weight)
        use = (tuple(range(len(self.channels))) if not levels
               else tuple(sorted({int(i) for i in levels})))
        for i in use:
            if not 0 <= i < len(self.channels):
                raise ValueError(
                    "level %d out of range for %d levels"
                    % (i, len(self.channels)))
        self.levels = use

        self.proj = nn.ModuleDict()
        self.query = nn.ParameterDict()
        self.value = nn.ParameterDict()
        self.gate = nn.ParameterDict()
        for i in self.levels:
            k = str(i)
            self.proj[k] = nn.Conv1d(self.channels[i], self.hidden, 1)
            nn.init.xavier_uniform_(self.proj[k].weight)
            nn.init.zeros_(self.proj[k].bias)
            sd = 1.0 / float(self.hidden) ** 0.5
            self.query[k] = nn.Parameter(torch.randn(self.hidden) * sd)
            self.value[k] = nn.Parameter(torch.randn(self.hidden) * sd)
            # The gate must not start at zero together with its source. The
            # first version had `gate = 0` AND `value = 0`, a product of two
            # zeros: d/d(gate) = (attn*val).sum = 0 because val = 0, and
            # d/d(value) = attn*tanh(0) = 0 because gate = 0. Measured on a
            # real batch, all 15 tensors of the module had gradient 0.0000e+00
            # -- dead from the first step. Only F3, the positive control,
            # catches that; F1 and F2 pass on a module returning a constant
            # zero. The off switch is `weight`, which is a constant rather
            # than a parameter and needs no gradient at all.
            self.gate[k] = nn.Parameter(torch.ones(1))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(dropout)) if dropout > 0.0 else None
        self._diag: Dict[str, float] = {}

    def forward(self, z_a: Sequence[torch.Tensor],
                z_b: Sequence[torch.Tensor]) -> torch.Tensor:
        """Antisymmetric addition to the pairwise head's `delta`, `[B]`."""
        if len(z_a) != len(z_b):
            raise ValueError("z_a has %d levels, z_b has %d"
                             % (len(z_a), len(z_b)))
        out: Optional[torch.Tensor] = None
        diag: Dict[str, float] = {}
        for i in self.levels:
            k = str(i)
            a, b = z_a[i], z_b[i]
            if a.shape != b.shape:
                raise ValueError(
                    "level %d: %s vs %s -- the two elements of a pair must "
                    "have the same code shape" % (i, tuple(a.shape),
                                                  tuple(b.shape)))
            ha = self.act(self.proj[k](a))               # [B, H, L]
            hb = self.act(self.proj[k](b))
            d = ha - hb                                   # antisymmetric
            if self.drop is not None:
                d = self.drop(d)
            # The attention weight comes from the MAGNITUDE of the
            # difference, which is symmetric under A<->B and so leaves the
            # product below antisymmetric.
            logit = torch.einsum("h,bhl->bl", self.query[k], d.abs())
            attn = torch.softmax(logit, dim=-1)           # [B, L]
            val = torch.einsum("h,bhl->bl", self.value[k], d)
            part = (attn * val).sum(dim=-1) * self.gate[k].tanh()
            out = part if out is None else out + part
            with torch.no_grad():
                # Attention concentration: 1.0 is all the mass on one
                # position, 0.0 uniform. Without it there is no way to see
                # whether the module reads POSITIONS or only their mean.
                L = attn.shape[-1]
                ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1).mean()
                import math as _m
                diag["pc_skup_l%d" % i] = float(
                    1.0 - ent.item() / _m.log(max(L, 2)))
                diag["pc_brama_l%d" % i] = float(self.gate[k].tanh().item())
        if out is None:                                   # no levels in use
            out = torch.zeros(z_a[0].shape[0], device=z_a[0].device,
                              dtype=z_a[0].dtype)
        self._diag = diag
        return out * self.weight

    def report(self) -> Dict[str, float]:
        return dict(self._diag)

    # Falsifiers
    @torch.no_grad()
    def falsify(self, z_a: Sequence[torch.Tensor],
                z_b: Sequence[torch.Tensor]) -> Dict[str, float]:
        """F1/F2/F3 on the GIVEN tensors. Returns numbers, not a verdict."""
        was = self.training
        self.eval()
        w = self.weight
        try:
            # The falsifiers are about the SHAPE of the mapping and not its
            # scale, and at weight = 0 all three would pass trivially.
            self.weight = 1.0 if w == 0.0 else w
            d_ab = self.forward(z_a, z_b)
            d_ba = self.forward(z_b, z_a)
            d_aa = self.forward(z_a, z_a)
            zc = [t.clone() for t in z_a]
            lvl = self.levels[-1] if self.levels else 0
            zc[lvl][:, :, 0] = zc[lvl][:, :, 0] + 1.0
            d_mut = self.forward(zc, z_a)
            return {
                "f1_identyczne": float(d_aa.abs().max()),
                "f2_antysymetria": float((d_ab + d_ba).abs().max()),
                "f3_jedna_pozycja": float(d_mut.abs().max()),
            }
        finally:
            self.weight = w
            self.train(was)
