"""Fine levels sharpen the picture built on the coarsest one.

The heads see the latent only through `LatentPooling`, i.e. through three
statistics taken over the WHOLE length of a level. Measured mutual
information with strength, as a percentage of ceiling, is 4.2-6.8 % at L0
against 0.36 % at L1 and less above, and the twelve most informative
positions are L0 in every epoch measured -- the heads see exactly what the
pooling passes through and not a bit more.

This module does not add another mean over a whole level; that is the thing
washing position out. It computes statistics of the fine levels INSIDE a
level-0 window, which at the shipped strides is exactly 2 / 4 / 8 / 16 cells
of L1 / L2 / L3 / L4. A mean and a deviation taken in that window are
positional with respect to the L0 grid, where a global mean is their average
and cannot tell one window from another. A non-linearity (Conv1d 1x1 + SiLU)
sits between the two poolings; without it the whole construction collapses
back to a mean over the level.

The per-window pooling uses the same three statistics as `LatentPooling`.
`mean` and `sd` are measured together: a linear probe gains +0.0672 / +0.0285
on the two coarsest trunk blocks with a NEGATIVE capacity control. `max` is
absent because it measured worse than the mean on 5 blocks of 5 -- the 81x
advantage of the maximum belonged to PWM occupancy, a sparse non-negative
signal and a different object. `anchor` is the mean over the last
`anchor_frac` windows, which lands where L0's mutual information peaks.

There is no softmax over distances here, so the temperature law does not
apply: the squared-distance range on a shell of radius r is `[0, 2 sinh^2
r]`, so one shared temperature forces a uniform assignment on inner shells
arithmetically. Anyone adding geodesic attention here must come back with
`tau_k ~ sinh^2(r_k)`.

Only the final linear layer is zeroed, so the model starts bit for bit as it
would without the module while `d(loss)/d(W_out)` stays non-zero, its input
being non-zero. There is deliberately no multiplicative gate: a zero gate on
a zero value gives a zero derivative in both arguments and the branch is dead
from the first step.

The module reads only `z`, the latent after the quantiser, and is called in
`_heads`, through which both `forward` and `predict_from_codes` pass, so
prediction stays a function of (codes, condition) by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LevelSharpening"]


class LevelSharpening(nn.Module):
    """A list of maps `[B, C_k, L_k]`, coarsest first -> correction to `pooled`."""

    _STAT = ("mean", "sd", "anchor")

    def __init__(self,
                 channels: Sequence[int],
                 output: int,
                 levels: Sequence[int] = (1, 2, 3, 4),
                 dim: int = 24,
                 stats: Sequence[str] = ("mean", "sd", "anchor"),
                 anchor_frac: float = 0.125,
                 bands: int = 4,
                 eps: float = 1e-12) -> None:
        super().__init__()
        channels = [int(c) for c in channels]
        levels = [int(k) for k in levels]
        if not levels:
            raise ValueError("LevelSharpening with no levels has nothing to "
                             "sharpen -- disable the module with a config "
                             "key, not with an empty list")
        if 0 in levels:
            # L0 is the reference axis (the windows), and its own positional
            # readout is a separate module. Reading it here would give two
            # modules one role and make their contributions inseparable.
            raise ValueError("level 0 is the window grid, not a source of "
                             "refinement")
        if max(levels) >= len(channels):
            raise ValueError("level %d is outside a tower of %d levels"
                             % (max(levels), len(channels)))
        bad = [s for s in stats if s not in self._STAT]
        if bad:
            raise ValueError("unknown statistics %r; known %r"
                             % (bad, self._STAT))
        if not 0.0 < float(anchor_frac) <= 1.0:
            raise ValueError("anchor_frac must lie in (0, 1], got %r"
                             % (anchor_frac,))
        self.channels = tuple(channels)
        self.levels = tuple(levels)
        self.stats = tuple(stats)
        self.anchor_frac = float(anchor_frac)
        self.eps = float(eps)
        self.dim = int(dim)
        self.bands = int(bands)
        if self.bands < 0:
            raise ValueError("bands must be >= 0")
        we = 2 * sum(channels[k] for k in levels)
        self.point = nn.Conv1d(we, self.dim, 1)
        self.out_head = nn.Linear(self.dim * (len(self.stats) + self.bands),
                             int(output))
        # This layer alone is zeroed; see the module docstring.
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self._diag: Dict[str, float] = {}

    def extra_repr(self) -> str:
        return ("levels=%s channels=%s dim=%d stats=%s bands=%d "
                "anchor_frac=%.3f"
                % (list(self.levels), list(self.channels), self.dim,
                   list(self.stats), self.bands, self.anchor_frac))

    # Forward
    def forward(self, z: Sequence[torch.Tensor],
                reference: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        z = list(z)
        if len(z) != len(self.channels):
            raise ValueError("expected %d levels, got %d"
                             % (len(self.channels), len(z)))
        okna = int(z[0].shape[-1])
        czesci: List[torch.Tensor] = []
        sd_w: List[float] = []
        sd_m: List[float] = []
        for k in self.levels:
            zk = z[k]
            if zk.dim() != 3:
                raise ValueError("level %d must be [B, C, L], got %r"
                                 % (k, tuple(zk.shape)))
            L = int(zk.shape[-1])
            if okna <= 0 or L % okna != 0:
                raise ValueError(
                    "level %d has length %d, which is not divisible by the %d "
                    "level-0 windows -- an L0 window has stopped being a "
                    "whole number of this level's cells, so a statistic "
                    "within a window would no longer be positional with "
                    "respect to the L0 grid"
                    % (k, L, okna))
            n = L // okna
            v = zk.reshape(zk.shape[0], zk.shape[1], okna, n)
            sr = v.mean(dim=-1)                                  # [B, C, W]
            od = (v - sr.unsqueeze(-1)).pow(2).mean(dim=-1)
            od = od.clamp_min(self.eps).sqrt()                   # [B, C, W]
            czesci.append(torch.cat([sr, od], dim=1))
            with torch.no_grad():
                # An observable, not a loss: how much detail there is INSIDE
                # a window against the variation BETWEEN windows. Zero in the
                # first column means the level has nothing to sharpen, which
                # is a different failure from a sleeping branch.
                sd_w.append(float(od.mean()))
                sd_m.append(float(sr.std(dim=-1).mean()))
        cechy = torch.cat(czesci, dim=1)                         # [B, 2CK, W]
        u = F.silu(self.point(cechy))                            # [B, dim, W]
        parts: List[torch.Tensor] = []
        sr_u = u.mean(dim=-1)
        for s in self.stats:
            if s == "mean":
                parts.append(sr_u)
            elif s == "sd":
                dev = u - sr_u.unsqueeze(-1)
                parts.append(dev.pow(2).mean(dim=-1)
                             .clamp_min(self.eps).sqrt())
            else:
                w = max(1, int(round(self.anchor_frac * u.shape[-1])))
                parts.append(u[..., -w:].mean(dim=-1))
        # Positional bands. Without them the whole readout is invariant to a
        # permutation of windows -- mean, deviation and anchor cannot say
        # which window the detail sits in, and swapping the contents of two
        # windows moved the output by 5.96e-08, i.e. by float32 noise. L0's
        # mutual information rises monotonically with window position, so
        # the window axis is exactly the one that must not be lost. The
        # edges are fractional, so the module never sees a sequence length.
        W = int(u.shape[-1])
        if self.bands > 0:
            if W < self.bands:
                raise ValueError(
                    "%d windows but %d bands -- an empty band has no mean"
                    % (W, self.bands))
            for i in range(self.bands):
                a = (i * W) // self.bands
                b = ((i + 1) * W) // self.bands
                parts.append(u[..., a:max(b, a + 1)].mean(dim=-1))
        delta = self.out_head(torch.cat(parts, dim=1))                # [B, D]
        with torch.no_grad():
            r_d = float(delta.pow(2).mean().clamp_min(0.0).sqrt())
            self._diag["ostr_rms"] = r_d
            self._diag["ostr_w_norma"] = float(self.out_head.weight.norm())
            self._diag["ostr_u_rms"] = float(u.pow(2).mean().sqrt())
            for i, k in enumerate(self.levels):
                self._diag["ostr_sdw_L%d" % k] = sd_w[i]
                self._diag["ostr_sdm_L%d" % k] = sd_m[i]
            if reference is not None:
                r_p = float(reference.pow(2).mean().clamp_min(0.0).sqrt())
                self._diag["ostr_udzial"] = (r_d / r_p) if r_p > 0.0 else 0.0
        return delta

    # Reporting
    def report(self) -> Dict[str, float]:
        return dict(self._diag)

    def inert_keys(self) -> List[Tuple[str, str]]:
        return []
