"""Conditioning enters through one narrow bottleneck and modulates the latent.

    per channel   u_t = E_t(C_t)            [B, d_channel]   one encoder each
    bottleneck    c   = W [u_1 .. u_T]      [B, d_bottleneck]  narrow
    modulation    gamma, beta = FiLM(c)
    output        h   = gamma * h0 + beta   [B, D]      the only head input

The measurement this answers to: 50.50 % of labelled rows share their 800 bp
exactly with another row, and within those groups the measured strength differs
by 0.3588 in the median and 13.84 at the extreme. On half the corpus the
promoter does not identify the row, so a prediction that is a function of the
promoter alone scores exactly zero correlation there -- not small, zero. Hence
FiLM rather than a residual: `beta` alone can place two identical latents
anywhere relative to each other, which the previous 31:1 residual could not.
Prediction still comes only from the latent, which is all the heads receive.

The two halves are not priced alike, because the damper after `gamma*lat +
beta` is scale-free and what reaches the head is the ratio `beta / rms(lat)`.
Shifting is therefore the expensive path and the optimiser does what it is
charged for (||(gamma-1)*lat|| 298.35 against ||beta|| 141.47); `fuse_beta_rms`
is not comparable across epochs, hence `fuse_lat_rms` beside it. Neither half
is free to delete: `gamma := 1` costs -0.01457, `beta := 0` costs -0.01100.

A field here is not reachable from YAML until `FusionHeadConfig` carries it,
since `assemble.py` builds `FusionConfig` field by field; likewise a `_diag`
key reaches the log only once it is in `FUSION_STAT_KEYS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "FusionConfig",
    "FusionOutput",
    "ConditionBottleneck",
    "LatentConditioner",
    "channel_floor_loss",
    "IDENTICAL_SEQUENCE_STATS",
]

_EPS = 1e-6

#: The identical-sequence statistics that justify this module, kept checkable.
IDENTICAL_SEQUENCE_STATS: Dict[str, float] = {
    "share_of_rows": 0.5050,
    "groups": 24217.0,
    "within_group_range_median": 0.3588,
    "within_group_range_max": 13.8382,
    "groups_above_noise_floor": 0.8506,
    "noise_floor": 0.049,
    "fraction_unreachable_from_sequence": 0.0542,
}


@dataclass
class FusionConfig:
    """Widths, and the numbers that decide what "controlled" and "active"
    mean."""

    #: Width of each per-channel encoder before the bottleneck.
    d_channel: int = 64
    #: The bottleneck: everything the condition can say passes through this
    #: many numbers, against a 807-wide concatenated context. See `validate`
    #: for what the narrowness buys.
    d_bottleneck: int = 32
    hidden: int = 128
    dropout: float = 0.0
    #: gamma lives in [1-r, 1+r], strictly below 1 (see `validate`). It is
    #: also the exact scalar on the multiplicative term, so lowering it
    #: re-prices modulation.
    film_range: float = 0.9
    #: Express the additive term in the latent's own units by dividing the
    #: FiLM input by its per-sample RMS, equivalent to `beta*rms(lat)` since
    #: the damper downstream is scale-free. No parameters.
    beta_w_skali_latentu: bool = False
    #: Scale of the FiLM projection at initialisation. Non-zero, or
    #: d(output)/d(context) is identically zero and the path never starts.
    init_scale: float = 0.1
    #: Lower bound on every per-channel gate, for the same orphaning reason.
    gate_floor: float = 0.05
    #: Share of the total channel effect below which a channel is starved;
    #: 0.0 resolves to 1/(4T), a quarter of an equal share.
    active_floor: float = 0.0
    floor_weight: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.film_range < 1.0:
            raise ValueError(
                "fusion.film_range must be in (0, 1): at 1.0 gamma can reach "
                "exactly 0, and a multiplicative gate that closes orphans its "
                "source -- no gradient reaches the features that would reopen "
                "it. Recorded four times in this project.")
        if self.d_bottleneck < 2 or self.d_channel < 4:
            raise ValueError("fusion widths are implausibly small")
        if self.d_bottleneck > self.d_channel * 8:
            raise ValueError(
                "fusion.d_bottleneck = %d is not a bottleneck relative to "
                "d_channel = %d. The narrowness IS the control: a wide "
                "conditioning path can represent the per-condition lookup "
                "table that measured 0.43448 under a random split and 0.08615 "
                "under the study split."
                % (self.d_bottleneck, self.d_channel))
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("fusion.gate_floor must be in [0, 1)")
        if self.floor_weight < 0.0:
            raise ValueError("fusion.floor_weight must be >= 0")
        if not isinstance(self.beta_w_skali_latentu, bool):
            raise ValueError(
                "fusion.beta_w_skali_latentu is a FLAG, not an amount. The "
                "amount is film_range, which scales the OTHER half; a float "
                "here would read as 'scale beta by this', which is not what "
                "it does.")
        if self.active_floor and not 0.0 < self.active_floor < 1.0:
            raise ValueError("fusion.active_floor is a SHARE in (0, 1)")

    def resolved_active_floor(self, n_tokens: int) -> float:
        if self.active_floor:
            return float(self.active_floor)
        return 1.0 / (4.0 * max(int(n_tokens), 1))


@dataclass
class FusionOutput:
    """The conditioned latent and everything needed to audit the route."""

    latent: torch.Tensor                 # [B, D] -- the only head input
    code: torch.Tensor                   # [B, d_bottleneck] the whole channel
    gamma: torch.Tensor                  # [B, D]
    beta: torch.Tensor                   # [B, D]
    channel: torch.Tensor                # [B, T, d_channel] per-channel codes
    gates: torch.Tensor                  # [T]
    diagnostics: Dict[str, float]


class ConditionBottleneck(nn.Module):
    """`[B, T, d_ctx] -> [B, d_bottleneck]`, through one encoder per channel.

    Separate encoders rather than one shared projection over a concatenation,
    for measurement and not capacity: a channel can then be replaced by its
    own null without disturbing the others, which makes `channel_effect` exact.
    """

    def __init__(self, ctx_dim: int, token_names: Sequence[str],
                 config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        cfg = config or FusionConfig()
        cfg.validate()
        self.cfg = cfg
        self.names = tuple(str(n) for n in token_names)
        self.n_tokens = len(self.names)
        if self.n_tokens < 1:
            raise ValueError("a bottleneck needs at least one channel")
        self.ctx_dim = int(ctx_dim)
        d_ch = int(cfg.d_channel)

        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(self.ctx_dim, int(cfg.hidden)),
                          nn.GELU(),
                          nn.Linear(int(cfg.hidden), d_ch))
            for _ in range(self.n_tokens)])
        # One null per channel: a shared null would make "no species" and "no
        # experiment" the same point of the space.
        self.null = nn.Parameter(torch.randn(self.n_tokens, d_ch) * 0.02)
        self.gate_logit = nn.Parameter(torch.zeros(self.n_tokens))
        self.to_code = nn.Linear(self.n_tokens * d_ch, int(cfg.d_bottleneck))
        nn.init.xavier_uniform_(self.to_code.weight)
        nn.init.zeros_(self.to_code.bias)
        # Parameter-free: a learnable affine here would duplicate FiLM's own.
        self.norm = nn.LayerNorm(int(cfg.d_bottleneck),
                                 elementwise_affine=False)
        self.drop = nn.Dropout(float(cfg.dropout))

    def gates(self) -> torch.Tensor:
        f = float(self.cfg.gate_floor)
        return f + (1.0 - f) * torch.sigmoid(self.gate_logit)

    def channel_codes(self, tokens: torch.Tensor,
                      drop: Optional[Sequence[int]] = None,
                      zastap: Optional[Dict[int, torch.Tensor]] = None
                      ) -> torch.Tensor:
        """`[B, T, d_channel]`, with `drop` channels replaced by their null.

        `zastap` substitutes a given vector instead of `null`, and exists
        because `null` is never trained: it enters the graph only through
        `drop`, which only the `@torch.no_grad()` `channel_effect` passes, so
        distance to `null` measures displacement towards a random vector.
        """
        dropped = set(int(i) for i in (drop or ()))
        pod = dict((int(k), v) for k, v in (zastap or {}).items())
        outs = []
        for i, enc in enumerate(self.encoders):
            if i in pod:
                v = pod[i]
                outs.append(v.unsqueeze(0).expand(tokens.shape[0], -1)
                            if v.dim() == 1 else v)
            elif i in dropped:
                outs.append(self.null[i].unsqueeze(0).expand(
                    tokens.shape[0], -1))
            else:
                outs.append(enc(tokens[:, i]))
        return torch.stack(outs, dim=1)

    def forward(self, tokens: torch.Tensor,
                drop: Optional[Sequence[int]] = None,
                zastap: Optional[Dict[int, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3 or tokens.shape[1] != self.n_tokens:
            raise ValueError(
                "tokens must be [B, %d, %d] in the order %r, got %r. The "
                "order is a checkpoint contract: a permutation silently swaps "
                "which encoder and which gate belong to which channel."
                % (self.n_tokens, self.ctx_dim, self.names,
                   tuple(tokens.shape)))
        u = self.channel_codes(tokens, drop, zastap)
        g = self.gates().to(u.dtype).view(1, -1, 1)
        u = u * g
        c = self.norm(self.to_code(self.drop(u.reshape(u.shape[0], -1))))
        return c, u


class LatentConditioner(nn.Module):
    """`(pooled latent, context tokens) -> one conditioned latent`."""

    def __init__(self, latent_dim: int, ctx_dim: int,
                 token_names: Sequence[str],
                 config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        cfg = config or FusionConfig()
        cfg.validate()
        self.cfg = cfg
        self.names = tuple(str(n) for n in token_names)
        self.n_tokens = len(self.names)
        self.latent_dim = int(latent_dim)
        self.ctx_dim = int(ctx_dim)
        self.active_floor = cfg.resolved_active_floor(self.n_tokens)

        self.bottleneck = ConditionBottleneck(ctx_dim, self.names, cfg)
        self.film = nn.Linear(int(cfg.d_bottleneck), 2 * self.latent_dim)
        nn.init.xavier_uniform_(self.film.weight, gain=float(cfg.init_scale))
        # Identity at step 0, so a rising metric is the conditioning being
        # learned rather than a restart.
        nn.init.zeros_(self.film.bias)
        self._diag: Dict[str, float] = {}
        #: Multiply/shift split from the last pass of the FULL branch. With a
        #: non-empty `head_withhold` assemble calls this module twice and the
        #: trimmed second call overwrites `_diag`, so without carrying these
        #: over every `fuse_*` number would describe the narrow branch.
        self._pelne: Dict[str, float] = {}
        #: Branch marker, used by assemble to withhold statistics on the
        #: trimmed pass. Defined here rather than only in the centred
        #: subclass, because `film_center: none` needs it too.
        self.aktualizuj_statystyki = True

    def _wejscie_film(self, latent: torch.Tensor) -> torch.Tensor:
        """The latent in units where `beta` means the same as `gamma`.

        No `detach`: a cut denominator would leave the encoder free to drift
        in its own scale, the very non-stationarity this divisor closes.
        """
        if not bool(getattr(self.cfg, "beta_w_skali_latentu", False)):
            return latent
        s = latent.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
        return latent / s

    @torch.no_grad()
    def _rozklad(self, lat: torch.Tensor, gamma: torch.Tensor,
                 beta: torch.Tensor) -> Dict[str, float]:
        """How much of the condition is multiplication and how much shift.

        Two numbers for two questions: `fuse_udzial_przes` is energy before
        the damper and `fuse_udzial_przes_h` movement after it, and since the
        damper removes a common scale they need not agree. Counterfactual,
        because after normalisation the sum of terms is not the sum of effects.
        """
        d = gamma.detach().double() - 1.0
        la = lat.detach().double()
        be = beta.detach().double()
        n_mul = float((d * la).norm())
        n_bet = float(be.norm())

        def _t(x: torch.Tensor) -> torch.Tensor:
            return x / x.pow(2).mean(dim=-1, keepdim=True).sqrt(
            ).clamp_min(_EPS)

        ref = _t(la)
        d_sh = float((_t(la + be) - ref).norm())
        d_mu = float((_t((1.0 + d) * la) - ref).norm())
        return {
            "fuse_lat_rms": float(la.pow(2).mean().sqrt()),
            "fuse_mult_rms": float((d * la).pow(2).mean().sqrt()),
            "fuse_udzial_przes": n_bet / max(n_bet + n_mul, _EPS),
            "fuse_udzial_przes_h": d_sh / max(d_sh + d_mu, _EPS),
            "fuse_beta_skala": float(
                bool(getattr(self.cfg, "beta_w_skali_latentu", False))),
        }

    def modulate(self, latent: torch.Tensor, code: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gb = self.film(code)
        raw_g, beta = gb[:, :self.latent_dim], gb[:, self.latent_dim:]
        gamma = 1.0 + float(self.cfg.film_range) * torch.tanh(raw_g)
        h = gamma * latent + beta
        # Per sample, never per batch: a batch statistic would make one
        # promoter's prediction a function of who else was in the batch.
        h = h / h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
        return h, gamma, beta

    def forward(self, latent: torch.Tensor,
                tokens: torch.Tensor) -> FusionOutput:
        if latent.dim() != 2:
            raise ValueError("latent must be [B, D], got %r"
                             % (tuple(latent.shape),))
        code, chan = self.bottleneck(tokens)
        lat_we = self._wejscie_film(latent)
        h, gamma, beta = self.modulate(lat_we, code)

        with torch.no_grad():
            g = self.bottleneck.gates()
            ref = latent / latent.pow(2).mean(dim=-1, keepdim=True).sqrt(
            ).clamp_min(_EPS)
            self._diag = {
                "fuse_gamma_sd": float(gamma.std()),
                "fuse_gamma_mean": float(gamma.mean()),
                "fuse_beta_rms": float(beta.pow(2).mean().sqrt()),
                # How far the condition moves the latent, relative to its own
                # scale. Not required to be small: half the corpus needs it not.
                "fuse_shift_ratio": float(
                    (h - ref).norm(dim=-1).mean()
                    / max(float(h.norm(dim=-1).mean()), _EPS)),
                "fuse_code_rms": float(code.pow(2).mean().sqrt()),
                "fuse_code_pr": _participation_ratio(code),
                "fuse_bottleneck_dim": float(self.cfg.d_bottleneck),
            }
            for i, name in enumerate(self.names):
                self._diag["fuse_gate_%s" % name] = float(g[i])
                self._diag["fuse_code_rms_%s" % name] = float(
                    chan[:, i].pow(2).mean().sqrt())
            # The full branch computes the split, the narrow one carries it
            # forward; `_diag` is reassigned here, so without the carry these
            # keys vanish on every non-empty `head_withhold`.
            if bool(getattr(self, "aktualizuj_statystyki", True)):
                self._pelne = self._rozklad(lat_we, gamma, beta)
            self._diag.update(self._pelne)
        return FusionOutput(latent=h, code=code, gamma=gamma, beta=beta,
                            channel=chan, gates=g.detach(),
                            diagnostics=dict(self._diag))

    @torch.no_grad()
    def channel_effect(self, latent: torch.Tensor, tokens: torch.Tensor
                       ) -> Dict[str, object]:
        """Leave-one-channel-out: how far `h` moves when each is nulled.

        Exact, not indicative: under a bottleneck the channels are mixed
        before the output, so there is no additive per-channel term to take a
        norm of. It says whether a channel is wired, not whether it carries
        anything -- for that, see `decomposition.ablation_gain`.
        """
        was = self.training
        self.eval()
        # The same scale as in `forward`, or this measures the module on a
        # different input from the one producing the metric.
        latent = self._wejscie_film(latent)
        base_code, _ = self.bottleneck(tokens)
        base, _g, _b = self.modulate(latent, base_code)
        scale = max(float(base.norm(dim=-1).mean()), _EPS)
        eff: Dict[str, float] = {}
        for i, name in enumerate(self.names):
            code_i, _ = self.bottleneck(tokens, drop=(i,))
            alt, _g2, _b2 = self.modulate(latent, code_i)
            eff[name] = float((alt - base).norm(dim=-1).mean() / scale)
        # A fair reference at zero parameters: the channel replaced by its own
        # batch mean. `null` is untrained (see `channel_codes`), so `eff` above
        # measures distance to an initialisation constant; the batch mean keeps
        # the common component, i.e. "this channel distinguishes nothing".
        eff_sr: Dict[str, float] = {}
        try:
            u0 = self.bottleneck.channel_codes(tokens)
            if int(u0.shape[0]) > 1:
                for i, name in enumerate(self.names):
                    code_j, _ = self.bottleneck(
                        tokens, zastap={i: u0[:, i].mean(dim=0)})
                    alt_j, _g3, _b3 = self.modulate(latent, code_j)
                    eff_sr[name] = float(
                        (alt_j - base).norm(dim=-1).mean() / scale)
        except Exception:
            eff_sr = {}
        total_sr = sum(eff_sr.values()) or 1.0
        gates = self.bottleneck.gates()
        self.train(was)
        total = sum(eff.values()) or 1.0
        share = {k: v / total for k, v in eff.items()}
        starved = [k for k, v in share.items() if v < self.active_floor]
        return {
            "effect": eff,
            "share": share,
            "effect_mean": eff_sr,
            "share_mean": {k: v / total_sr for k, v in eff_sr.items()},
            "gate": {n: float(gates[i]) for i, n in enumerate(self.names)},
            "active_floor": self.active_floor,
            "starved": starved,
            "all_active": not starved,
            "caveat": ("effect measures whether a channel REACHES the latent, "
                       "not whether it carries information: a floored gate "
                       "can push noise through. Quote only beside a "
                       "substitution measurement -- "
                       "heads.decomposition.ablation_gain or "
                       "channel_attribution."),
        }

    @torch.no_grad()
    def falsifiers(self, batch: int = 64, seed: int = 0,
                   noise_floor: float = 0.049) -> Dict[str, object]:
        """The two properties this block exists to have. Both are required.

        `identical_sequence_separates`: with the latent held fixed and only
        the context changed, the conditioned latent must move by more than the
        noise floor. `sequence_still_matters` catches the opposite failure, a
        block that overwrites the latent and predicts from the condition alone.
        """
        gen = torch.Generator().manual_seed(int(seed))
        dev = self.film.weight.device
        was = self.training
        self.eval()
        h0 = torch.randn(batch, self.latent_dim, generator=gen).to(dev)
        h1 = torch.randn(batch, self.latent_dim, generator=gen).to(dev)
        c1 = torch.randn(batch, self.n_tokens, self.ctx_dim,
                         generator=gen).to(dev)
        c2 = torch.randn(batch, self.n_tokens, self.ctx_dim,
                         generator=gen).to(dev)
        a = self(h0, c1).latent
        b = self(h0, c2).latent          # same sequence, other condition
        d = self(h1, c1).latent          # other sequence, same condition
        ctx_effect = float((a - b).norm(dim=-1).mean())
        seq_effect = float((a - d).norm(dim=-1).mean())
        scale = max(float(a.norm(dim=-1).mean()), _EPS)
        self.train(was)
        out: Dict[str, object] = {
            "context_effect": ctx_effect,
            "context_effect_relative": ctx_effect / scale,
            "sequence_effect": seq_effect,
            "sequence_effect_relative": seq_effect / scale,
            "noise_floor": float(noise_floor),
            "identical_sequence_separates": bool(
                ctx_effect / scale > float(noise_floor)),
            "sequence_still_matters": bool(seq_effect / scale > 1e-3),
            "measured_justification": dict(IDENTICAL_SEQUENCE_STATS),
        }
        out["ok"] = bool(out["identical_sequence_separates"]
                         and out["sequence_still_matters"])
        return out

    def no_decay_names(self) -> List[str]:
        """Gates, nulls and the FiLM bias must not be decayed.

        Decay on `gate_logit` drives every gate to a uniform prior nobody
        chose; decay on `film.bias` drives gamma to 1 and beta to 0, the
        identity, so it would slowly switch the conditioning path off.
        """
        return [n for n, _ in self.named_parameters()
                if "gate_logit" in n or "null" in n or n.endswith("film.bias")]

    def extra_repr(self) -> str:
        return ("tokens=%r, latent=%d, ctx=%d, d_channel=%d, bottleneck=%d, "
                "film_range=%.2f, gate_floor=%.3f"
                % (self.names, self.latent_dim, self.ctx_dim,
                   self.cfg.d_channel, self.cfg.d_bottleneck,
                   self.cfg.film_range, self.cfg.gate_floor))


def channel_floor_loss(effect_share: Dict[str, float], floor: float,
                       weight: float = 1.0) -> float:
    """Penalty for a channel whose leave-one-out share sits below `floor`.

    A last resort, off by default: this can make a channel loud, not useful.
    Forcing a channel that carries nothing to occupy 1/(4T) of the bottleneck
    spends a narrow resource on noise and lowers every metric that matters
    while the activity report turns green. It exists for the one case that
    does occur, an informative channel shut out early by the winner-take-all
    dynamic of a shared bottleneck: small weight, first epochs only, and check
    `ablation_gain` afterwards.
    """
    if weight <= 0.0:
        return 0.0
    return float(weight) * sum(max(0.0, float(floor) - v) ** 2
                               for v in effect_share.values())


def _participation_ratio(x: torch.Tensor) -> float:
    """Effective rank of the bottleneck code across the batch.

    A code whose participation ratio collapses towards 1 has reduced the whole
    conditioning to a single direction, so every condition differs only in how
    much of one thing it has and cannot express "stronger in reesei, weaker in
    P1".
    """
    a = x.detach().to(torch.float64)
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(dim=0, keepdim=True)
    cov = (a.T @ a) / float(a.shape[0] - 1)
    # Without spread there is no effective rank. The panel scores a batch of
    # candidates in one construct, so the condition tokens are identical and
    # all spread comes from the latent; when that is negligible the matrix is
    # numerically zero and `eigvalsh` does not converge.
    slad = float(torch.diagonal(cov).sum())
    if not (slad > 0.0) or not bool(torch.isfinite(cov).all()):
        return float("nan")
    try:
        ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    except Exception:                                      # noqa: BLE001
        # The same quantity by a stabler route: singular values of the centred
        # matrix give `ev = s^2/(n-1)` without squaring the condition number.
        # An observable may not break the forward pass.
        try:
            s = torch.linalg.svdvals(a)
            ev = (s * s / float(a.shape[0] - 1)).clamp_min(0.0)
        except Exception:                                  # noqa: BLE001
            return float("nan")
    s1 = float(ev.sum())
    s2 = float((ev ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else float("nan")
