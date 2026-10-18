"""Positional readout of level 0: P equal bands of windows, zeroed correction.

The heads see the latent only through `LatentPooling`, whose statistics are
fixed functionals of position: `mean` and `sd` are permutation-invariant by
definition, and `anchor` separates exactly two regions, the tail and the rest.
`LevelSharpening` gives levels 1 and up their own bands but explicitly skips
level 0, which is a grid of windows; this module fills that gap.

Position is a small pool to play for: permuting latent positions within a
level moves PCC by -0.042 against -0.522 for a global shuffle, so position is
about 8 % of what the latent contributes, two thirds of which `anchor`
already carries. Only level 0 is read, because mutual information with
strength is 4.2-6.8 % of ceiling there against at most 0.36 % above it.

The argument for this module is circular and must be reported that way: the
permutation effect was measured on a model that never had a positional
readout finer than two regions, so it could not have learned to use position.
If that effect grows and selectivity does not, position was not needed and
the loss was free -- a negative result, not a metric failing to see a working
mechanism.

There is no non-linearity here, for a rank reason rather than a thrift one. A
head's first layer composed with this projection realises any linear
functional of the bands, so a zeroed projection into `pooled` is exactly as
expressive as widening every head's input by C*P columns, with one shared
matrix and without invalidating the `pooled` width of family checkpoints.
Only the output layer is zeroed, so the model starts bit for bit as it would
without the module while `d(loss)/d(W_out)` stays non-zero, its input being
non-zero; there is deliberately no multiplicative gate, since a zero gate on
a zero value is dead in both arguments from the first step.

The positive control is an arithmetic identity with no degrees of freedom:
band means weighted by their own widths equal `mean(z0)`, the `L0/mean` slice
of `pooled`. A mismatch means the band grid has drifted from the level-0 map,
and then every positional statement from that run is void. The consequence is
that `L0/mean` is now an exact linear combination of the bands, so weight
decay moves mass between that slice and the band columns -- a falling
`L0/mean` norm in `LatentPooling.attribution()` no longer means the head
stopped reading L0 and must be read together with the bands.

The module reads only `z`, the latent after the quantiser, and is called in
`_heads`, through which both `forward` and `predict_from_codes` pass, so
prediction stays a function of (codes, condition) by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["BandsL0"]


class BandsL0(nn.Module):
    """A list of maps `[B, C_k, L_k]`, coarsest first -> correction to `pooled`."""

    def __init__(self,
                 channels: Sequence[int],
                 output: int,
                 bands: int = 8,
                 anchor_frac: float = 0.125,
                 eps: float = 1e-12) -> None:
        super().__init__()
        channels = [int(c) for c in channels]
        if not channels:
            raise ValueError("BandsL0 with no levels has nothing to read")
        if int(bands) < 2:
            # P = 1 is the mean over the whole level, i.e. exactly the
            # `L0/mean` slice the head already has: a reparameterisation of an
            # existing column carrying no bit of position -- configurable and
            # inert.
            raise ValueError("bands must be >= 2, got %r -- at 1 the module "
                             "is a copy of the `L0/mean` slice" % (bands,))
        self.channels = tuple(channels)
        self.bands = int(bands)
        self.anchor_frac = float(anchor_frac)
        self.eps = float(eps)
        self.out_head = nn.Linear(self.channels[0] * self.bands, int(output))
        # This layer alone is zeroed; see the module docstring.
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self._diag: Dict[str, float] = {}

    def extra_repr(self) -> str:
        return ("bands=%d channels_L0=%d output=%d anchor_frac=%.3f"
                % (self.bands, self.channels[0], self.out_head.out_features,
                   self.anchor_frac))

    # Geometry
    def granice(self, W: int) -> List[Tuple[int, int]]:
        """Band edges as fractions, so the module never sees a sequence length.

        The same formula as `LevelSharpening.forward`: two formulas would have
        two modules writing into one space on two different grids, and their
        contributions would not be comparable.
        """
        W = int(W)
        out: List[Tuple[int, int]] = []
        for i in range(self.bands):
            a = (i * W) // self.bands
            b = ((i + 1) * W) // self.bands
            out.append((a, max(b, a + 1)))
        return out

    # Forward
    def forward(self, z: Sequence[torch.Tensor],
                reference: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        z = list(z)
        if len(z) != len(self.channels):
            raise ValueError("expected %d levels, got %d"
                             % (len(self.channels), len(z)))
        z0 = z[0]
        if z0.dim() != 3:
            raise ValueError("level 0 must be [B, C, L], got %r"
                             % (tuple(z0.shape),))
        if int(z0.shape[1]) != self.channels[0]:
            raise ValueError("level 0 has %d channels, the config says %d"
                             % (int(z0.shape[1]), self.channels[0]))
        W = int(z0.shape[-1])
        if W < self.bands:
            raise ValueError("%d windows but %d bands -- an empty band has "
                             "no mean" % (W, self.bands))
        parts: List[torch.Tensor] = []
        for a, b in self.granice(W):
            parts.append(z0[..., a:b].mean(dim=-1))              # [B, C]
        cechy = torch.stack(parts, dim=1)                        # [B, P, C]
        delta = self.out_head(cechy.reshape(cechy.shape[0], -1))      # [B, D]
        with torch.no_grad():
            r_d = float(delta.pow(2).mean().clamp_min(0.0).sqrt())
            self._diag["pasma_rms"] = r_d
            self._diag["pasma_w_norma"] = float(self.out_head.weight.norm())
            # SD of the band means across bands, per sequence and channel.
            # Zero means every band is equal, so the module carries what the
            # `L0/mean` slice already carries and there is no position -- a
            # different failure from a sleeping branch (`pasma_udzial` = 0).
            self._diag["pasma_sd_miedzy"] = float(
                cechy.std(dim=1, unbiased=False).mean())
            C = self.channels[0]
            w = self.out_head.weight.detach().abs()                   # [D, P*C]
            for i in range(self.bands):
                # The column layout is a contract: `cechy` is [B, P, C]
                # flattened over C, so band i owns columns [i*C:(i+1)*C].
                # Mass concentrated on the last band means the module has
                # rediscovered `anchor` and nothing more.
                self._diag["pasma_prof_%d" % i] = float(
                    w[:, i * C:(i + 1) * C].mean())
            if reference is not None:
                r_p = float(reference.pow(2).mean().clamp_min(0.0).sqrt())
                self._diag["pasma_udzial"] = (r_d / r_p) if r_p > 0.0 else 0.0
        return delta

    # Positive control
    @torch.no_grad()
    def kontrola_siatki(self, z0: torch.Tensor) -> Dict[str, float]:
        """Identity: band means weighted by width == `mean` of level 0.

        No degrees of freedom, so it beats any threshold. Also returns the
        MEASURED difference between the last band and the `anchor` slice --
        they are not equal at the default `anchor_frac`, and assuming they
        were is the kind of number taken from a different object.
        """
        W = int(z0.shape[-1])
        gr = self.granice(W)
        acc = torch.zeros(z0.shape[0], z0.shape[1], dtype=z0.dtype,
                          device=z0.device)
        for a, b in gr:
            acc = acc + z0[..., a:b].mean(dim=-1) * float(b - a)
        acc = acc / float(W)
        mean = z0.mean(dim=-1)
        wa = max(1, int(round(self.anchor_frac * W)))
        anchor = z0[..., -wa:].mean(dim=-1)
        a_ost, b_ost = gr[-1]
        ost = z0[..., a_ost:b_ost].mean(dim=-1)
        return {
            "okna": float(W),
            "szerokosci": [int(b - a) for a, b in gr],
            "blad_tozsamosci": float((acc - mean).abs().max()),
            "anchor_komorek": float(wa),
            "pasmo_ost_komorek": float(b_ost - a_ost),
            "blad_pasmo_vs_anchor": float((ost - anchor).abs().max()),
        }

    # Reporting
    def report(self) -> Dict[str, float]:
        return dict(self._diag)

    def inert_keys(self) -> List[Tuple[str, str]]:
        return []
