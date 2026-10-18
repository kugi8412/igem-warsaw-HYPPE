"""Condition tokens as a geodesic cone over the species point.

The model's only Lorentz point, `genome.points` [20, 4], leaves the manifold
through `logmap0` and two linear layers, so curvature takes part in no
operation. Here each condition token becomes a geodesic refinement of the
species point instead, hanging off a learnt channel root K_c:

    anchor  p_g = expmap(K_c, PT(o -> K_c, pad(t_species)))   [B, d_m+1]
    point   p_c = expmap(p_g, PT(o -> p_g, v_c)),  v_c = aperture*tanh(W_c t)
    readout token_c <- token_c + U_c logmap0(p_c)

The resulting hierarchy is a property of that composition, not of the data,
which are less tree-like than noise (Gromov delta 0.1455 on the text against
0.0631 for Gaussian noise in R^768). All of it runs after `ConditionEncoder`,
so the latent and `encode_codes` / `decode_codes` / `predict_from_codes` stay
functions of the sequence alone.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..geometry.lorentz import (
    assert_on_manifold,
    expmap,
    expmap0,
    geodesic_distance,
    logmap0,
    mark_manifold,
    minkowski_dot,
    parallel_transport0,
    radius,
)

__all__ = ["ConditionCone"]


class ConditionCone(nn.Module):
    """Geodesic refinement of the species point by the condition tokens.

    `d_m` must be at least `d_species` (= `GenomeEmbedding.dim`): the anchor
    embeds H^d_species into H^d_m by zero padding, so a smaller `d_m` would
    delete part of the tree rather than reduce the dimension. `aperture` is a
    constant bound on step length, not a weight, so a run with and without it
    measures the geometry rather than the size.
    """

    def __init__(self, d_ctx: int, d_m: int = 8,
                 channels: Sequence[str] = ("text", "gene"),
                 d_species: int = 3, aperture: float = 0.25,
                 zero_init: bool = True, korzenie: bool = True,
                 diag_co: int = 50) -> None:
        super().__init__()
        if int(d_m) < int(d_species):
            raise ValueError(
                "d_m %d is smaller than the species manifold dimension %d: "
                "the anchor is an EMBEDDING of H^%d into H^%d by "
                "zero-padding, so truncating would delete part of the "
                "tree rather than reduce the dimension"
                % (int(d_m), int(d_species), int(d_species),
                   int(d_m)))
        if not 0.0 < float(aperture) <= 2.0:
            raise ValueError("aperture must lie in (0, 2], got %r"
                             % (aperture,))
        if not channels:
            raise ValueError(
                "an empty channel list: the module would build, do nothing "
                "and read in the log exactly like a working one")
        if len(set(channels)) != len(tuple(channels)):
            raise ValueError("channels has a duplicate: %r" % (tuple(channels),))
        self.d_ctx = int(d_ctx)
        self.d_m = int(d_m)
        self.d_species = int(d_species)
        self.aperture = float(aperture)
        self.diag_co = max(1, int(diag_co))
        self.channels: Tuple[str, ...] = tuple(str(k) for k in channels)
        self.step = nn.ModuleDict(
            {k: nn.Linear(self.d_ctx, self.d_m, bias=False)
             for k in self.channels})
        self.readout = nn.ModuleDict(
            {k: nn.Linear(self.d_m, self.d_ctx, bias=False)
             for k in self.channels})
        if zero_init:
            # Identity at start, which also hides broken wiring: gradient to
            # `step`, `roots_manifold` and `genome.points` flows only through
            # `readout`, so a fresh model measures nothing. Test after steps.
            for k in self.channels:
                nn.init.zeros_(self.readout[k].weight)
        if korzenie:
            # Root exactly at the origin, where `expmap(o, PT(o -> o, t))` is
            # `expmap0(t)` bit for bit, so enabling roots changes nothing at
            # init; `ktx_st_korzen_r` leaving zero is the falsifier.
            pkt = torch.zeros(len(self.channels), self.d_m + 1,
                              dtype=torch.float64)
            pkt[:, 0] = 1.0
            self.roots_manifold = nn.Parameter(pkt)
            mark_manifold(self.roots_manifold)
        else:
            self.register_parameter("roots_manifold", None)
        self._diag: Dict[str, float] = {}
        self._n_przebiegow = 0
        self.zbieraj = True

    def manifold_parameters(self) -> List[nn.Parameter]:
        """Hyperboloid tensors, which `param_groups` must keep out of decay.

        Weight decay 0.05 over 34 epochs is a factor 1.8e-5 on a point, so a
        root left in the decayed group is squeezed to the origin silently.
        """
        p = getattr(self, "roots_manifold", None)
        return [p] if (p is not None and p.requires_grad) else []

    def manifold_state_names(self) -> List[str]:
        """Names SWA must not average: the ambient mean leaves the sheet."""
        return [] if getattr(self, "roots_manifold", None) is None \
            else ["roots_manifold"]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _styczna(self, styczna_gatunku: Optional[torch.Tensor],
                 B: int, device) -> torch.Tensor:
        """[B, d_m] tangent at the origin, zero padded. No species -> 0."""
        if styczna_gatunku is None:
            return torch.zeros(B, self.d_m, device=device,
                               dtype=torch.float64)
        t = styczna_gatunku.to(device=device, dtype=torch.float64)
        if t.dim() != 2 or int(t.shape[-1]) != self.d_species:
            raise ValueError(
                "the species tangent has shape %r, the module was built "
                "for [B, %d]. This is the tangent FROM "
                "`GenomeEmbedding.tangent()` (logmap0 of the point, dim "
                "3, float64), NOT the output of `genome.forward`, which "
                "passes through `proj` and is already a Euclidean "
                "feature of width out_dim = 32."
                % (tuple(t.shape), self.d_species))
        if int(t.shape[0]) != int(B):
            raise ValueError(
                "the species tangent has %d rows at batch %d"
                % (int(t.shape[0]), int(B)))
        if self.d_m == self.d_species:
            return t
        pad = torch.zeros(t.shape[0], self.d_m - self.d_species,
                          device=device, dtype=torch.float64)
        return torch.cat([t, pad], dim=-1)

    def anchor(self, j: int, t_g: torch.Tensor) -> torch.Tensor:
        """[B, d_m+1]: the species composed geodesically on root `j`.

        With the root at the origin this is exactly `expmap0(t_g)`.
        """
        k = getattr(self, "roots_manifold", None)
        if k is None:
            return expmap0(t_g)
        root = k[j].unsqueeze(0).expand(t_g.shape[0], -1)
        return expmap(root, parallel_transport0(root, t_g))

    def forward(self, tokens: torch.Tensor, nazwy: Sequence[str],
                styczna_gatunku: Optional[torch.Tensor] = None,
                kotwicz: bool = True) -> torch.Tensor:
        """`tokens` [B, T, d_ctx] -> [B, T, d_ctx] with the readout added.

        `kotwicz=False` anchors at the channel root instead of the species,
        making the gradient to `genome.points` exactly zero: the negative
        control for the panel gates.
        """
        B = int(tokens.shape[0])
        dev, dt = tokens.device, tokens.dtype
        nazwy = list(nazwy)
        # float64 throughout: the hyperboloid condition is a difference of
        # large numbers, and float32 has already lost it (2.861e-06 against a
        # 1e-9 threshold, versus 3.553e-15).
        t_g = self._styczna(styczna_gatunku if kotwicz else None, B, dev)
        out_head = tokens.clone()
        punkty: Dict[str, torch.Tensor] = {}
        anchors: Dict[str, torch.Tensor] = {}
        nasyc: Dict[str, torch.Tensor] = {}
        dlug: Dict[str, torch.Tensor] = {}
        for j, k in enumerate(self.channels):
            if k not in nazwy:
                raise ValueError(
                    "channel %r is not one of the tokens %r; a channel skipped "
                    "silently would do nothing and would read in the log "
                    "exactly like a working one" % (k, tuple(nazwy)))
            i = nazwy.index(k)
            p0 = self.anchor(j, t_g)
            s = torch.tanh(self.step[k](tokens[:, i]))
            v = (self.aperture * s).to(torch.float64)
            p = expmap(p0, parallel_transport0(p0, v))
            punkty[k] = p
            anchors[k] = p0
            nasyc[k] = (s.detach().abs() > 0.99).to(torch.float32).mean()
            dlug[k] = v.detach().norm(dim=-1).mean()
            out_head[:, i] = out_head[:, i] + self.readout[k](logmap0(p).to(dt))
        self._n_przebiegow += 1
        if self.zbieraj and (self._n_przebiegow == 1
                             or self._n_przebiegow % self.diag_co == 0):
            # Sampled: every `float()` on a CUDA tensor is a sync. The first
            # pass always counts, so a missing number in the log never reads
            # the same as "not this epoch yet".
            with torch.no_grad():
                self._diag = self._numbers(anchors, punkty, nasyc, dlug)
        return out_head

    def _numbers(self, anchors: Dict[str, torch.Tensor],
                punkty: Dict[str, torch.Tensor],
                nasyc: Dict[str, torch.Tensor],
                dlug: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Containment falsifiers, computed together with a random control.

        `d_rodzic` is the child's distance to its own anchor, `d_losowy` to
        another sample's. A direction cosine is deliberately not used: on a
        shell of radius r a fixed geodesic distance subtends ~ d/sinh(r), so
        a cosine mixes distance with radius. `krok_na_rozrzut` matters more
        than the ratio, since containment constrains position only while the
        step is shorter than the spacing between anchors. HYPPE batches are
        single-genome, so pairs sharing an anchor would collapse the control
        and are dropped; at `kontrola_frakcja` zero the dependent numbers
        become -1.0, which no threshold passes.
        """
        d: Dict[str, float] = {}
        k = getattr(self, "roots_manifold", None)
        if k is not None:
            kd = k.detach()
            d["ktx_st_korzen_r"] = float(radius(kd).mean())
            d["ktx_st_korzen_r_max"] = float(radius(kd).max())
            d["ktx_st_korzen_bl"] = float(
                (minkowski_dot(kd, kd, keepdim=False) + 1.0).abs().max())
        for name, p in punkty.items():
            p0 = anchors[name]
            B = int(p0.shape[0])
            perm = torch.roll(torch.arange(B, device=p0.device), 1)
            dr = geodesic_distance(p, p0)
            dl = geodesic_distance(p, p0[perm])
            roz = geodesic_distance(p0, p0[perm])
            # `geodesic_distance` is an arccosh of a difference of large
            # numbers, so below ~1e-7 it is noise: treat as the same anchor.
            rozne = roz > 1e-6
            n_roz = int(rozne.sum())
            d["ktx_st_%s_d_rodzic" % name] = float(dr.mean())
            d["ktx_st_%s_kontrola_frakcja" % name] = float(n_roz) / max(B, 1)
            if n_roz == 0:
                # No control on this batch: -1.0, not a plausible number.
                d["ktx_st_%s_d_losowy" % name] = -1.0
                d["ktx_st_%s_iloraz" % name] = -1.0
                d["ktx_st_%s_krok_na_rozrzut" % name] = -1.0
            else:
                d["ktx_st_%s_d_losowy" % name] = float(dl[rozne].mean())
                d["ktx_st_%s_iloraz" % name] = float(
                    dl[rozne].mean() / dr[rozne].mean().clamp_min(1e-12))
                d["ktx_st_%s_krok_na_rozrzut" % name] = float(
                    dr[rozne].mean() / roz[rozne].mean().clamp_min(1e-12))
            d["ktx_st_%s_promien" % name] = float(radius(p).mean())
            d["ktx_st_%s_pr_kotwicy" % name] = float(radius(p0).mean())
            d["ktx_st_%s_bl" % name] = float(
                (minkowski_dot(p, p, keepdim=False) + 1.0).abs().max())
            d["ktx_st_%s_nasyc" % name] = float(nasyc[name])
            # Ceiling is aperture*sqrt(d_m); usage near 1.0 together with
            # `nasyc` near 1.0 means the module is pressed against the
            # aperture, which then constrains rather than safeguards.
            d["ktx_st_%s_apert_uzycie" % name] = float(
                dlug[name] / (self.aperture * (self.d_m ** 0.5)))
            # Zero readout is a dead module; zero-init looks the same.
            d["ktx_st_%s_norma_odczytu" % name] = float(
                self.readout[name].weight.detach().norm())
        return d

    def report(self) -> Dict[str, float]:
        return dict(self._diag)

    @torch.no_grad()
    def check_manifold(self, tol: float = 1e-12) -> float:
        """max|<x,x>_L + 1| over the roots. Raises when `tol` is exceeded."""
        k = getattr(self, "roots_manifold", None)
        if k is None:
            return 0.0
        return assert_on_manifold(k.detach(), tol=tol,
                                  name="cone.roots_manifold")

    def extra_repr(self) -> str:
        return ("d_ctx=%d d_m=%d channels=%r aperture=%.3f korzenie=%r"
                % (self.d_ctx, self.d_m, self.channels, self.aperture,
                   getattr(self, "roots_manifold", None) is not None))
