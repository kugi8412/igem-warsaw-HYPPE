"""Content-weighted readout of the latent, for the single-sequence path.

The absolute head sees the latent only through `pooled`, whose statistics are
fixed functionals of position: `mean` and `sd` are permutation-invariant by
definition and `anchor` separates exactly two regions. This module weighs
positions by content instead, using each sequence's own mean as the second
argument, so the prediction stays a function of (codes, condition) and no
batch statistic enters.

Position accounts for about 8 % of what the latent contributes, so the whole
pool this module plays for is small: it can show that it harms, and whether
the mechanism is alive (`odcz_rms`), but not that it helps.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


class AttentionReadout(nn.Module):
    """Maps [B, C_k, L_k], COARSEST FIRST -> a correction to `pooled`.

    Only `out_head` is zeroed: a gate initialised at zero has zero derivative
    in BOTH arguments, and the branch dies at the first step.
    """

    def __init__(self, channels: Sequence[int], output: int,
                 levels: Sequence[int] = (0, 1, 2), dim: int = 64,
                 zapytania: int = 8) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.levels = tuple(int(k) for k in levels)
        if not self.levels:
            raise ValueError(
                "an empty level list is NOT the module's off switch -- the "
                "module is disabled with the key `model.attention_readout: "
                "false`. An empty list would build a branch without a single "
                "input, which costs `out_head` parameters and can contribute "
                "nothing.")
        if max(self.levels) >= len(self.channels):
            raise ValueError(
                "level %d is outside a tower of %d levels"
                % (max(self.levels), len(self.channels)))
        self.dim, self.q = int(dim), int(zapytania)

        self.proj = nn.ModuleDict()
        self.query_vec = nn.ParameterDict()
        self.value_vec = nn.ParameterDict()
        for k in self.levels:
            s = str(k)
            self.proj[s] = nn.Conv1d(self.channels[k], self.dim, 1)
            nn.init.xavier_uniform_(self.proj[s].weight)
            nn.init.zeros_(self.proj[s].bias)
            sd = 1.0 / float(self.dim) ** 0.5
            self.query_vec[s] = nn.Parameter(torch.randn(self.q, self.dim) * sd)
            self.value_vec[s] = nn.Parameter(torch.randn(self.q, self.dim) * sd)

        self.out_head = nn.Linear(len(self.levels) * self.q, int(output))
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self._diag: Dict[str, float] = {}

    def forward(self, z: Sequence[torch.Tensor],
                reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = list(z)
        if len(z) != len(self.channels):
            raise ValueError("expected %d levels, got %d"
                             % (len(self.channels), len(z)))
        czesci = []
        diag: Dict[str, float] = {}
        skala = 1.0 / float(self.dim) ** 0.5
        for k in self.levels:
            s = str(k)
            zk = z[k]
            if zk.dim() != 3 or int(zk.shape[1]) != self.channels[k]:
                raise ValueError(
                    "level %d has shape %r, but the config says %d channels"
                    % (k, tuple(zk.shape), self.channels[k]))
            h = self.proj[s](zk)                            # [B, H, L]
            # Self-contrast against the sequence's own mean over positions.
            h = h - h.mean(dim=-1, keepdim=True)
            # 1/sqrt(H), or the softmax temperature drifts with the encoder's
            # scale. `PairComparison` omits it; that is its defect.
            logit = torch.einsum("qh,bhl->bql",
                                 self.query_vec[s], h.abs()) * skala
            attn = torch.softmax(logit, dim=-1)             # [B, Q, L]
            val = torch.einsum("qh,bhl->bql", self.value_vec[s], h)
            czesci.append((attn * val).sum(dim=-1))         # [B, Q]
            with torch.no_grad():
                L = int(attn.shape[-1])
                ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1).mean()
                # 0 = uniform attention, i.e. the mean the head already has
                # through `pooled`; 1 = all the mass on one position.
                diag["odcz_skup_l%d" % k] = float(
                    1.0 - ent.item() / math.log(max(L, 2)))
                diag["odcz_logit_sd_l%d" % k] = float(logit.std())
        delta = self.out_head(torch.cat(czesci, dim=1))          # [B, D]
        with torch.no_grad():
            r_d = float(delta.pow(2).mean().clamp_min(0.0).sqrt())
            diag["odcz_rms"] = r_d
            diag["odcz_w_norma"] = float(self.out_head.weight.norm())
            if reference is not None:
                r_p = float(reference.pow(2).mean().clamp_min(0.0).sqrt())
                diag["odcz_udzial"] = (r_d / r_p) if r_p > 0.0 else 0.0
        self._diag = diag
        return delta

    def report(self) -> Dict[str, float]:
        return dict(self._diag)

    @torch.no_grad()
    def kontrola(self, z: Sequence[torch.Tensor]) -> Dict[str, float]:
        """Two controls on the GIVEN tensors. Returns numbers, not a verdict.

        K1: a map constant along positions must give exactly 0, and uniform
        attention gives zero too, so `odcz_rms > 0` in a run is evidence of
        non-uniform weighting rather than merely of a live branch. K2 moves
        ONE position and requires the output to move, without which K1 passes
        trivially on a module returning a constant zero. `out_head` is
        zero-initialised, so it is randomised here and restored in `finally`.
        """
        w = self.out_head.weight.detach().clone()
        b = self.out_head.bias.detach().clone()
        try:
            if float(w.norm()) == 0.0:
                nn.init.xavier_uniform_(self.out_head.weight)
            zs = [t.mean(dim=-1, keepdim=True).expand_as(t).contiguous()
                  for t in z]
            k1 = float(self.forward(zs).abs().max())
            baza = self.forward(list(z))
            zc = [t.clone() for t in z]
            kk = self.levels[-1]
            zc[kk][:, :, 0] = zc[kk][:, :, 0] + 1.0
            k2 = float((self.forward(zc) - baza).abs().max())
            return {"k1_stala_mapa": k1, "k2_jedna_pozycja": k2}
        finally:
            self.out_head.weight.data.copy_(w)
            self.out_head.bias.data.copy_(b)
