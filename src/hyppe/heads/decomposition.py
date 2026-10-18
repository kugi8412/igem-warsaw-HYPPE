"""One head per conditioning channel, each with a target only that channel has.

Paired ablation put the genome channel at -0.0020 SCC and the experiment
channel at -0.0029 when zeroed, against -0.0875 for the gene context. The
usual reading is that genome and experiment are weak channels; the correct one
is that they had NOTHING TO PREDICT, because every other head regresses `mu`,
the mean of `combat` over the kept samples of one genome, which is constant in
the condition by construction and whose dependence on the genome is already
carried by the (orthogroup, genome) cell the row IS.

`data/decomposition.py` splits the same matrix into additive terms, and this
module attaches one head to each, giving each head ONLY the context token that
is supposed to explain its term:

    head     target                     context it may read
    gene     gene[o]                    the gene token
    strain   genome[g] + inter[o, g]    the species token
    cond     cond[o, s]                 the text token

The isolation is the mechanism: a `cond` head that could read every token
would predict `cond[o, s]` from the promoter and the strain, and the text
channel would again receive no gradient it could not be denied. Restricting
the input makes the text token the only route to a term worth 14.78 % of the
variance, which `ablation_gain` measures.

Nothing here can move a pooled PCC/SCC by more than seed noise. `cond` has
14.78 % of the variance of which 73.92 % is reproducible between replicates, a
ceiling near 10.92 %; `genome` has 2.31 % and `inter` 0.70 %; the `gene` head
faces the 91.86 % promoter main effect but sees a phyletic profile rather than
the sequence. These heads exist so that "same promoter, other condition" and
"same promoter, other strain" become answerable, which a model trained only on
the mean cannot do.

Every target is NaN somewhere -- `gene` where an orthogroup has no valid cell,
`cond` where the cell has fewer than two samples -- so targets are sanitised
BEFORE use and the mask is a multiplicative weight. `torch.where(mask, f(y),
0)` would not protect the gradient: the backward pass evaluates f' at the
masked entries and NaN * 0 = NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .absolute import (
    ConditionedTrunk,
    LatentPooling,
    as_features,
    masked_mean,
    pearson,
    sanitise,
    spearman,
)

__all__ = [
    "DECOMPOSITION_TARGETS",
    "TokenSlicer",
    "DecompositionOutput",
    "DecompositionHeads",
    "channel_attribution",
]

#: target name -> the context token it is allowed to read. Order is the order
#: the losses are reported in; append-only, because it indexes the report.
DECOMPOSITION_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("gene", "gene"),
    ("strain", "species"),
    ("cond", "text"),
)

_EPS = 1e-8


# Slicing one token out of the flattened context
class TokenSlicer(object):
    """Pull named tokens out of the flattened `[B, T * d_ctx]` context.

    `HyppeModel.context()` flattens `tokens [B, T, d_ctx]` in the order
    declared by `ConditionOutput.TOKEN_NAMES`, which is a checkpoint
    contract. Slicing by NAME rather than by a literal offset at three call
    sites is what keeps a later appended channel from silently shifting which
    token a head reads.
    """

    def __init__(self, token_names: Sequence[str], d_ctx: int) -> None:
        self.names = tuple(str(n) for n in token_names)
        self.d_ctx = int(d_ctx)
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate token name in %r" % (self.names,))

    @property
    def total_dim(self) -> int:
        return len(self.names) * self.d_ctx

    def has(self, name: str) -> bool:
        return name in self.names

    def width(self, names: Sequence[str]) -> int:
        return len(names) * self.d_ctx

    def take(self, context: Optional[torch.Tensor], names: Sequence[str]
             ) -> Optional[torch.Tensor]:
        """`[B, len(names) * d_ctx]`, or None when `context` is None."""
        if context is None:
            return None
        if context.shape[-1] != self.total_dim:
            raise ValueError(
                "context is %d wide, this slicer describes %d tokens of %d "
                "(= %d). A width mismatch here would hand a head the wrong "
                "token with no shape error downstream."
                % (context.shape[-1], len(self.names), self.d_ctx,
                   self.total_dim))
        parts = []
        for n in names:
            if n not in self.names:
                raise KeyError("no context token %r; have %r"
                               % (n, self.names))
            i = self.names.index(n)
            parts.append(context[..., i * self.d_ctx:(i + 1) * self.d_ctx])
        return torch.cat(parts, dim=-1)

    def replace(self, context: torch.Tensor, name: str,
                value: torch.Tensor) -> torch.Tensor:
        """Copy of `context` with one token overwritten -- for ablation."""
        i = self.names.index(name)
        out = context.clone()
        out[..., i * self.d_ctx:(i + 1) * self.d_ctx] = value
        return out


# Output
@dataclass
class DecompositionOutput:
    """One prediction per additive term. Every field is [B]."""

    gene: Optional[torch.Tensor] = None
    strain: Optional[torch.Tensor] = None
    cond: Optional[torch.Tensor] = None

    def get(self, name: str) -> Optional[torch.Tensor]:
        return getattr(self, name, None)

    def sum_terms(self) -> Optional[torch.Tensor]:
        """gene + strain, i.e. the part of `mu` this module can explain.

        `cond` is deliberately NOT added: it is centred inside the same cell
        `mu` describes, so its expectation over conditions is zero and adding
        it to a prediction of `mu` would be adding noise with a name.
        """
        parts = [t for t in (self.gene, self.strain) if t is not None]
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out


# The heads
class DecompositionHeads(nn.Module):
    """Three small conditioned regressors, one per additive term.

    Each is a `ConditionedTrunk` over the pooled latent with a RESTRICTED
    context, plus a linear output. They are small on purpose: `cond` carries
    14.78 % of the variance and `genome` 2.31 %, and 26.08 % of `cond` is
    replicate noise, so a head the size of the absolute head would spend most
    of its capacity fitting that noise.
    """

    def __init__(self, in_dim: int, slicer: TokenSlicer,
                 hidden: int = 128, depth: int = 2, dropout: float = 0.1,
                 context_mode: str = "concat", context_center: bool = True,
                 targets: Sequence[str] = ("gene", "strain", "cond"),
                 weights: Optional[Mapping[str, float]] = None,
                 huber_delta: float = 0.0,
                 pooling: Optional[LatentPooling] = None) -> None:
        super().__init__()
        known = {n for n, _ in DECOMPOSITION_TARGETS}
        bad = [t for t in targets if t not in known]
        if bad:
            raise ValueError("unknown decomposition target(s) %r; known %r"
                             % (bad, sorted(known)))
        self.slicer = slicer
        self.pooling = pooling
        self.huber_delta = float(huber_delta)
        # Default weights follow the MEASURED variance shares, not a sweep:
        # equal weights would hand a 2 % term the pull of an 85 % one.
        self.weights: Dict[str, float] = {"gene": 0.30, "strain": 0.15,
                                          "cond": 0.15}
        self.weights.update({k: float(v) for k, v in (weights or {}).items()})

        self.token_for: Dict[str, str] = dict(DECOMPOSITION_TARGETS)
        self.targets: Tuple[str, ...] = tuple(
            t for t in targets if slicer.has(self.token_for[t]))
        self.skipped: Tuple[str, ...] = tuple(
            t for t in targets if t not in self.targets)
        trunks: Dict[str, nn.Module] = {}
        outs: Dict[str, nn.Module] = {}
        for t in self.targets:
            names = (self.token_for[t],)
            trunks[t] = ConditionedTrunk(
                in_dim, hidden, depth, dropout,
                context_dim=slicer.width(names),
                mode=context_mode, center=context_center)
            lin = nn.Linear(hidden, 1)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            outs[t] = lin
        self.trunks = nn.ModuleDict(trunks)
        self.outs = nn.ModuleDict(outs)
        self._last: Dict[str, float] = {}

    def forward(self, latent, context: Optional[torch.Tensor] = None
                ) -> DecompositionOutput:
        feats = as_features(latent, self.pooling)
        out = DecompositionOutput()
        for t in self.targets:
            ctx = self.slicer.take(context, (self.token_for[t],))
            h = self.trunks[t](feats, ctx)
            setattr(out, t, self.outs[t](h).squeeze(-1))
        return out

    def loss(self, out: DecompositionOutput,
             targets: Mapping[str, torch.Tensor],
             masks: Optional[Mapping[str, torch.Tensor]] = None,
             weight: Optional[torch.Tensor] = None,
             track: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted, masked squared (or Huber) error, summed over the terms.

        Huber is available and off by default. `cond` has heavy tails, being a
        deviation from a cell mean over as few as two samples, and a squared
        loss chases them; `huber_delta = 0` reproduces plain MSE bit for bit.
        """
        masks = dict(masks or {})
        total = None
        parts: Dict[str, float] = {}
        for t in self.targets:
            pred = out.get(t)
            if pred is None:
                continue
            if t not in targets:
                raise KeyError(
                    "head %r produced a prediction but no target %r was "
                    "supplied. A head whose loss is silently skipped trains "
                    "nothing and still reports a number." % (t, t))
            y = sanitise(targets[t].to(pred.dtype))
            m = masks.get(t)
            m = (torch.ones_like(pred) if m is None
                 else m.to(pred.dtype))
            r = pred - y
            if self.huber_delta > 0.0:
                d = self.huber_delta
                per = torch.where(r.abs() <= d, 0.5 * r * r,
                                  d * (r.abs() - 0.5 * d))
            else:
                per = 0.5 * r * r
            term = masked_mean(per, m, weight)
            w = float(self.weights.get(t, 1.0))
            total = term * w if total is None else total + term * w
            if track:
                parts["dec_%s" % t] = float(term.detach())
                parts["dec_%s_n" % t] = float(m.sum().detach())
                parts["dec_%s_pred_sd" % t] = float(pred.detach().std())
        if total is None:
            # An all-masked batch is legitimate (a batch of pure negatives has
            # no gene effect) but must contribute a zero that is attached to
            # the graph, or the optimiser step silently changes shape.
            ref = next(iter(self.parameters()))
            total = ref.sum() * 0.0
        if track:
            self._last = dict(parts)
        return total, parts

    # Falsifiers
    @torch.no_grad()
    def ablation_gain(self, latent, context: torch.Tensor,
                      targets: Mapping[str, np.ndarray],
                      masks: Optional[Mapping[str, np.ndarray]] = None,
                      substitute: str = "roll") -> Dict[str, float]:
        """Metric drop when each head's OWN token is replaced.

        `substitute="roll"` gives every row ANOTHER row's token, which is the
        only ablation that measures anything here: batches are
        genome-homogeneous, so a within-batch shuffle of the species token is
        an empty intervention -- it measured +0.00000 and was read as a dead
        channel. `"zero"` and `"mean"` are also offered because a substitution
        moves the input off-distribution while a mean does not.

        A head whose gain is ~0 is reading the latent only, and its channel is
        decoration however large its projection.
        """
        was = self.training
        self.eval()
        base = self(latent, context)
        out: Dict[str, float] = {}
        masks = dict(masks or {})
        for t in self.targets:
            y = np.asarray(targets[t], dtype=np.float64).ravel()
            m = masks.get(t)
            sel = (np.ones(y.shape, dtype=bool) if m is None
                   else np.asarray(m, dtype=bool).ravel())
            sel = sel & np.isfinite(y)
            if int(sel.sum()) < 8:
                out["abl_%s" % t] = float("nan")
                continue
            p0 = base.get(t).detach().cpu().numpy().ravel()
            tok = self.token_for[t]
            i = self.slicer.names.index(tok)
            d = self.slicer.d_ctx
            block = context[..., i * d:(i + 1) * d]
            if substitute == "roll":
                repl = torch.roll(block, shifts=1, dims=0)
            elif substitute == "mean":
                repl = block.mean(dim=0, keepdim=True).expand_as(block)
            elif substitute == "zero":
                repl = torch.zeros_like(block)
            else:
                raise ValueError("substitute must be roll|mean|zero, got %r"
                                 % (substitute,))
            p1 = self(latent, self.slicer.replace(context, tok, repl)
                      ).get(t).detach().cpu().numpy().ravel()
            out["abl_%s" % t] = float(spearman(p0[sel], y[sel])
                                      - spearman(p1[sel], y[sel]))
            out["abl_%s_pcc" % t] = float(pearson(p0[sel], y[sel])
                                          - pearson(p1[sel], y[sel]))
            out["abl_%s_shift" % t] = float(np.abs(p0[sel] - p1[sel]).mean())
        self.train(was)
        out["substitute"] = substitute
        return out

    def report(self) -> Dict[str, object]:
        return {
            "targets": list(self.targets),
            "skipped_no_token": list(self.skipped),
            "token_for": dict(self.token_for),
            "weights": {k: self.weights.get(k, 1.0) for k in self.targets},
            "huber_delta": self.huber_delta,
            "last_loss": dict(self._last),
            "context_tokens": list(self.slicer.names),
        }

    def no_decay_names(self) -> List[str]:
        return [n for n, _ in self.named_parameters() if n.endswith("bias")]


# A channel-level attribution that does not need this module's targets
@torch.no_grad()
def channel_attribution(forward, context: torch.Tensor, slicer: TokenSlicer,
                        substitute: str = "roll") -> Dict[str, float]:
    """Mean |change in prediction| when each token is substituted, per token.

    Works on any callable `forward(context) -> [B]`, so one implementation
    covers the absolute head, the pairwise head and each decomposition head.
    It is a SENSITIVITY and not a gain: a channel can move a prediction a lot
    and move it the wrong way, which is why
    `DecompositionHeads.ablation_gain` reports a metric drop as well.

    Normalised by the spread of the baseline prediction, because an unnormed
    shift is not comparable between heads whose outputs live on different
    scales, and comparing them is the only use this number has.
    """
    base = forward(context).detach()
    scale = float(base.std()) or 1.0
    out: Dict[str, float] = {}
    d = slicer.d_ctx
    for i, name in enumerate(slicer.names):
        block = context[..., i * d:(i + 1) * d]
        if substitute == "roll":
            repl = torch.roll(block, shifts=1, dims=0)
        elif substitute == "mean":
            repl = block.mean(dim=0, keepdim=True).expand_as(block)
        else:
            repl = torch.zeros_like(block)
        alt = forward(slicer.replace(context, name, repl)).detach()
        out["sens_%s" % name] = float((alt - base).abs().mean() / scale)
    tot = sum(v for v in out.values()) or 1.0
    for name in list(out):
        out[name.replace("sens_", "share_")] = out[name] / tot
    out["baseline_sd"] = scale
    return out
