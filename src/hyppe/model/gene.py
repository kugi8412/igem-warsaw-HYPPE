"""The gene token: one context vector per orthogroup, zero-shot by default.

`ConditionEncoder` stacks its sources into `tokens [B, T, d_ctx]` whose order
is a checkpoint contract, being the key order of the fusion softmax, so the
gene token is appended at index 4 and never inserted:

    0 species   1 text   2 flank_l   3 flank_r   4 gene

Two of the three sources survive an unseen gene: `numeric`, the phyletic
profile and copy structure of any gene in the orthology table, and `text`, a
description through the same frozen LM as the condition channel. The third,
`identity`, is a hashed row that exists only for genes seen in training, and
`identity_dim = 0` drops it by default -- the split is orthogroup-disjoint, so
the identity row of any evaluated gene received zero gradient and can raise a
training curve but not a validation one.

Source dropout is what makes zero-shot real: `p_drop_numeric` and
`p_drop_text` blank one source at a time, so the head cannot learn only their
sum and then meet a gene with a description but no orthology row. The draws
are shared within a `group`, because a per-row draw hands the two rows of a
siamese pair different contexts 2p(1-p) of the time.

The numeric block mixes binary presence flags with log-counts, and a single
new gene at inference has no batch to standardise against. `BatchNorm1d(
affine=False)` has no learnable parameters but saves running_mean and
running_var, so the statistics come from training and travel with the
checkpoint.

This module must never read an expression matrix.
`data.gene_features.FORBIDDEN_GENE_SOURCES` names them and
`build_gene_features` refuses them by filename; `falsifiers()` re-checks the
spec fingerprint so a hand-built feature block cannot enter unnoticed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.gene_features import (
    FORBIDDEN_GENE_SOURCES,
    GENE_FEATURE_VERSION,
    GeneFeatureSpec,
)

__all__ = [
    "GeneConfig",
    "GeneOutput",
    "GeneEncoder",
]

_EPS = 1e-6


def _rms_normalise(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-sample unit RMS, matching `condition._rms_normalise` exactly.

    The two must apply the same scaling, or the gene token enters the fusion
    softmax on a different scale from the other four and dominates or vanishes
    for a reason no config key records.
    """
    return x / x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


@dataclass
class GeneConfig:
    """Widths and dropout rates of the gene channel."""

    d_ctx: int = 224                  # must equal ConditionConfig.cond_d_ctx
    d_lm: int = 384                   # must equal ConditionConfig.cond_d_lm
    hidden: int = 128
    identity_dim: int = 0             # 0 = content only; see the docstring
    identity_buckets: int = 4096
    use_text: bool = True
    p_drop_numeric: float = 0.15
    p_drop_text: float = 0.15
    #: Applied to the WHOLE token, so "gene unknown" is an event of its own:
    #: classifier-free guidance at design time needs a trained null for the
    #: channel it guides.
    p_drop_all: float = 0.10

    def validate(self) -> None:
        for name in ("p_drop_numeric", "p_drop_text", "p_drop_all"):
            p = float(getattr(self, name))
            if not 0.0 <= p < 1.0:
                raise ValueError("gene.%s must be in [0, 1), got %r"
                                 % (name, p))
        if self.identity_dim and self.identity_buckets < 2:
            raise ValueError("identity_buckets must be >= 2 when the identity "
                             "table is enabled")
        if self.d_ctx < 8 or self.hidden < 4:
            raise ValueError("gene channel widths are implausibly small")


@dataclass
class GeneOutput:
    token: torch.Tensor                      # [B, d_ctx], RMS normalised
    available: torch.Tensor                  # [B] float, 1 where a gene was given
    diagnostics: Dict[str, float]


class GeneEncoder(nn.Module):
    """(numeric, identity, text) -> one context token per row.

    Every input is optional. A row with nothing resolves to the learned
    `unknown_token_gene`, which is the same contract the flank channel uses
    and is what an inference on a gene outside the orthology table hits.
    """

    def __init__(self, config: Optional[GeneConfig] = None,
                 n_numeric: int = 0,
                 spec: Optional[GeneFeatureSpec] = None) -> None:
        super().__init__()
        cfg = config or GeneConfig()
        cfg.validate()
        self.cfg = cfg
        self.spec = spec
        self.spec_fingerprint = spec.fingerprint() if spec is not None else ""
        d = int(cfg.d_ctx)
        self.d_ctx = d
        self.n_numeric = int(n_numeric if n_numeric
                             else (spec.n_numeric() if spec else 0))
        if self.n_numeric <= 0 and not cfg.use_text and not cfg.identity_dim:
            raise ValueError(
                "a gene encoder with no numeric block, no text and no "
                "identity table has nothing to encode; that is the "
                "`enabled with dim=0` failure ChannelConfig.validate refuses")

        # Numeric block
        if self.n_numeric > 0:
            self.numeric_center = nn.BatchNorm1d(self.n_numeric, affine=False)
            self.numeric_proj = nn.Sequential(
                nn.Linear(self.n_numeric, int(cfg.hidden)),
                nn.GELU(),
                nn.Linear(int(cfg.hidden), d))
        else:
            self.numeric_center = None
            self.numeric_proj = None

        # Text block
        self.text_proj = (nn.Linear(int(cfg.d_lm), d) if cfg.use_text
                          else None)
        self.text_center = (nn.BatchNorm1d(d, affine=False) if cfg.use_text
                            else None)

        # Identity block
        self.identity_emb: Optional[nn.Embedding] = None
        if int(cfg.identity_dim) > 0:
            self.identity_emb = nn.Embedding(int(cfg.identity_buckets),
                                             int(cfg.identity_dim))
            nn.init.normal_(self.identity_emb.weight, std=0.02)
            self.identity_proj = nn.Linear(int(cfg.identity_dim), d)
        else:
            self.identity_proj = None

        # Fusion by sum, not concat: the three sources describe the same
        # object, and a concat would make the width depend on which of them a
        # deployment happens to have. A missing source contributes its learned
        # null, which the dropout above trains. Each null exists only where
        # its source does -- `null_identity` under `identity_dim = 0` would be
        # checkpointed, weight-decayed and never reached, coming back from a
        # full forward+backward with `grad is None`.
        self.null_numeric = (nn.Parameter(torch.randn(d) * 0.02)
                             if self.numeric_proj is not None else None)
        self.null_text = (nn.Parameter(torch.randn(d) * 0.02)
                          if self.text_proj is not None else None)
        self.null_identity = (nn.Parameter(torch.randn(d) * 0.02)
                              if self.identity_emb is not None else None)
        self.unknown_token_gene = nn.Parameter(torch.randn(d) * 0.02)
        self.mix = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.mix.weight, gain=0.5)
        nn.init.zeros_(self.mix.bias)
        self._diag: Dict[str, float] = {}

    # Helpers
    def _draw(self, p: float, batch: int, device: torch.device,
              dtype: torch.dtype, group: Optional[torch.Tensor]
              ) -> torch.Tensor:
        """Keep-mask [B, 1], one draw per `group` id, zeros outside training."""
        if not self.training or p <= 0.0:
            return torch.ones((batch, 1), device=device, dtype=dtype)
        if group is None:
            return (torch.rand(batch, 1, device=device) >= p).to(dtype)
        g = group.to(device=device).reshape(-1).long()
        uniq, inv = torch.unique(g, return_inverse=True)
        draw = (torch.rand(uniq.numel(), device=device) >= p)
        return draw[inv].to(dtype).unsqueeze(1)

    def _centre(self, layer: Optional[nn.BatchNorm1d], x: torch.Tensor
                ) -> torch.Tensor:
        if layer is None:
            return x
        if self.training:
            return layer(x)
        m, v = layer.running_mean, layer.running_var
        return (x - m) / torch.sqrt(v + layer.eps)

    # Forward
    def forward(self, numeric: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                text_emb: Optional[torch.Tensor] = None,
                available: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None,
                group: Optional[torch.Tensor] = None) -> GeneOutput:
        """
        numeric    [B, n_numeric]  phyletic block from `GeneFeatures`
        identity   [B]             hash bucket, -1 where unknown
        text_emb   [B, d_lm]       frozen LM vector of the gene description
        available  [B]             0 where this row has NO gene information
        """
        ref = next((t for t in (numeric, text_emb, identity)
                    if t is not None), None)
        if ref is None and batch_size is None:
            raise ValueError("no gene input and no batch_size: nothing says "
                             "how many unknown tokens to build")
        B = int(batch_size if ref is None else ref.shape[0])
        device = self.mix.weight.device
        dtype = self.mix.weight.dtype
        avail = (torch.ones((B, 1), device=device, dtype=dtype)
                 if available is None
                 else available.to(device=device, dtype=dtype).reshape(B, 1))

        raw: Dict[str, float] = {}
        acc = torch.zeros((B, self.d_ctx), device=device, dtype=dtype)

        # Numeric
        if self.numeric_proj is not None:
            if numeric is None:
                acc = acc + self.null_numeric.unsqueeze(0).expand(B, -1)
                keep_n = torch.zeros((B, 1), device=device, dtype=dtype)
            else:
                nv = numeric.to(device=device, dtype=dtype)
                if nv.shape[-1] != self.n_numeric:
                    raise ValueError(
                        "numeric block has %d columns, the encoder was built "
                        "for %d. The column order is the spec fingerprint and "
                        "a silent reshape would regress a different feature."
                        % (nv.shape[-1], self.n_numeric))
                raw["numeric"] = float(nv.pow(2).mean().sqrt().item())
                p = self.numeric_proj(self._centre(self.numeric_center, nv))
                keep_n = self._draw(float(self.cfg.p_drop_numeric), B, device,
                                    dtype, group) * avail
                acc = acc + keep_n * p + (1.0 - keep_n) * self.null_numeric
        else:
            keep_n = torch.zeros((B, 1), device=device, dtype=dtype)

        # Text
        if self.text_proj is not None:
            if text_emb is None:
                acc = acc + self.null_text.unsqueeze(0).expand(B, -1)
                keep_t = torch.zeros((B, 1), device=device, dtype=dtype)
            else:
                tv = text_emb.to(device=device, dtype=dtype)
                raw["text"] = float(tv.pow(2).mean().sqrt().item())
                p = self._centre(self.text_center, self.text_proj(tv))
                keep_t = self._draw(float(self.cfg.p_drop_text), B, device,
                                    dtype, group) * avail
                acc = acc + keep_t * p + (1.0 - keep_t) * self.null_text
        else:
            keep_t = torch.zeros((B, 1), device=device, dtype=dtype)

        # Identity
        if self.identity_emb is not None:
            if identity is None:
                acc = acc + self.null_identity.unsqueeze(0).expand(B, -1)
            else:
                idx = identity.to(device=device).long().reshape(B)
                seen = (idx >= 0).to(dtype).unsqueeze(1) * avail
                e = self.identity_proj(
                    self.identity_emb(idx.clamp_min(0)))
                acc = acc + seen * e + (1.0 - seen) * self.null_identity

        token = self.mix(acc)
        # The WHOLE channel is dropped last, so "gene unknown" is one event
        # rather than the coincidence of two source dropouts.
        keep_all = self._draw(float(self.cfg.p_drop_all), B, device, dtype,
                              group) * avail
        null = self.unknown_token_gene.unsqueeze(0).expand(B, -1)
        token = keep_all * token + (1.0 - keep_all) * null
        token = _rms_normalise(token)

        with torch.no_grad():
            self._diag = {
                "gene_dostepnosc": float(avail.mean().item()),
                "gene_drop_all": float(1.0 - keep_all.mean().item()),
                "gene_drop_numeric": float(1.0 - keep_n.mean().item()),
                "gene_drop_text": float(1.0 - keep_t.mean().item()),
                "gene_token_rms": float(token.pow(2).mean().sqrt().item()),
                "gene_dev_nad_srednia": float(_deviation_over_mean(token)),
            }
            for k, v in raw.items():
                self._diag["gene_rms_we_%s" % k] = v
        return GeneOutput(token=token, available=avail.reshape(B),
                          diagnostics=dict(self._diag))

    # Falsifiers
    @torch.no_grad()
    def falsifiers(self, batch: int = 32, seed: int = 0) -> Dict[str, object]:
        """Checks that fail when the channel is present but inert.

        `constant_token`   two different gene vectors must not give the same
                           token. A channel collapsed to a constant still
                           reports a context share of 1/T by construction,
                           since the tokens are RMS normalised, while
                           carrying nothing.
        `unknown_distinct` the unknown token must differ from a real one, or
                           "gene absent" and "gene present" are one event.
        `spec_version`     the feature spec the weights were built against.
        """
        gen = torch.Generator().manual_seed(int(seed))
        was = self.training
        self.eval()
        out: Dict[str, object] = {
            "spec_version": GENE_FEATURE_VERSION,
            "spec_fingerprint": self.spec_fingerprint,
            "identity_enabled": bool(self.identity_emb is not None),
            "forbidden_sources": list(FORBIDDEN_GENE_SOURCES),
        }
        num = (torch.randn(batch, self.n_numeric, generator=gen)
               if self.n_numeric else None)
        txt = (torch.randn(batch, int(self.cfg.d_lm), generator=gen)
               if self.cfg.use_text else None)
        a = self(numeric=num, text_emb=txt, batch_size=batch).token
        spread = float(a.std(dim=0).mean())
        out["token_spread"] = spread
        out["constant_token"] = bool(spread < 1e-4)
        ref = self(batch_size=batch).token
        out["unknown_distance"] = float((a - ref).abs().mean())
        out["unknown_distinct"] = bool(out["unknown_distance"] > 1e-3)
        self.train(was)
        out["ok"] = bool(not out["constant_token"] and out["unknown_distinct"])
        return out

    def no_decay_names(self) -> List[str]:
        """Null and unknown tokens must not be decayed towards zero.

        Zero is a legal point of the space that the model has already learned
        something about; the recorded precedent is a gate erased from 0.02 to
        1.42e-04 by 34 epochs of weight decay.
        """
        return [n for n, _ in self.named_parameters()
                if "null_" in n or "unknown_token" in n]


def _deviation_over_mean(x: torch.Tensor) -> float:
    """Per-sample content divided by the component shared by all samples.

    Below ~0.1 the token is a constant plus noise: the precedent is a
    LayerNorm projection that left 91 % of its norm in the shared direction,
    at 0.0869, until batch centring took it to 1.0701.
    """
    if x.shape[0] < 2:
        return float("nan")
    m = x.mean(dim=0, keepdim=True)
    dev = float((x - m).pow(2).mean().sqrt())
    base = float(m.pow(2).mean().sqrt())
    return dev / base if base > _EPS else float("inf")
