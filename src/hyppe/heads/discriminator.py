"""Promoter / non-promoter discriminator, read from the latent.

The generative half of this project decodes codes into sequence, and a decoder
is free to produce a string that is nowhere near a promoter, so something has
to say "this is a promoter" as a function of the latent. The way that head can
look good while being useless is COMPOSITION: elsewhere in this project a
trinucleotide-composition feature alone reached AUC 0.8422 on a realness axis,
and a single amino-acid fraction reached AUC 0.84 on a label that was supposed
to be about structure. A discriminator can score 0.9 AUC against random or
genomic negatives having learned only that promoters are AT-richer than coding
sequence.

The falsifier is therefore the DINUCLEOTIDE-SHUFFLED negative: same length,
same mononucleotide and dinucleotide composition, motif order destroyed. AUC
clearly above 0.5 there is evidence of grammar; AUC 0.99 against random
negatives is evidence of nothing. Three controls can refute it. `auc_gc` is
the single-feature GC baseline on the SAME batch, with the sign chosen in the
baseline's favour, and the head has to beat it. `auc_gc[dinucleotide_shuffle]`
must be ~0.5 by construction, because the shuffle preserves composition; if it
is not, the NEGATIVE GENERATOR is broken and the falsifier is void -- that
control tests the data, not the model, and nothing else would notice. And the
absence of shuffled negatives is itself a violation, because an empty bucket
must not read like a passed gate.

Genomic negatives are contaminated and the number is measured: flank-derived
negatives have 36.3 % FlankRight coverage of which 5.1 % are themselves
promoters, so their AUC has a ceiling below 1.0 and a false-positive floor
near 5 %, and a poor AUC there is partly a property of the bucket. Every
number is therefore reported per negative kind, with the overall AUC last.

`BCEWithLogitsLoss` does not weight classes on its own, unlike a hinge, which
does so implicitly through where the margin bites. `pos_weight` must be set
from the sampler's measured counts, and is reported so that a silently
unbalanced sampler is visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..data.sequences import NUCLEOTIDE_ORDER
from .absolute import (
    ConditionedTrunk,
    LatentPooling,
    Reservoir,
    as_features,
    average_ranks,
    masked_mean,
)

__all__ = ["NEGATIVE_KINDS", "DINUCLEOTIDE_SHUFFLE", "roc_auc",
           "gc_fraction", "DiscriminatorOutput", "PromoterDiscriminator"]

#: The shuffle that preserves dinucleotide composition is the one the
#: falsifier is about; the others are reported but prove less.
DINUCLEOTIDE_SHUFFLE = "dinucleotide_shuffle"
NEGATIVE_KINDS: Tuple[str, ...] = (
    DINUCLEOTIDE_SHUFFLE,
    "mononucleotide_shuffle",
    "coding",
    "flank",
    "random",
)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with ties handled by AVERAGED ranks.

    Tie handling decides what an untrained model scores: in eval one of this
    family returns a constant (measured sd 3.19e-08), which scores 1.0 or 0.0
    with ties broken by sort order and exactly 0.5 with averaged ranks.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size != y.size or s.size < 2:
        return float("nan")
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int(s.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(s)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (float(n_pos) * float(n_neg)))


def gc_fraction(x: torch.Tensor) -> torch.Tensor:
    """GC fraction from a one-hot `[B, 4, L]`, using the declared channel order.

    The channel order is read from `data.sequences.NUCLEOTIDE_ORDER` rather
    than hard-coded, because the sibling track uses A,C,T,G and lost
    measurements twice to the assumption that one file's order is universal.
    `N` is encoded as 0.25 everywhere and so contributes a neutral 0.5 to GC.
    """
    gi = NUCLEOTIDE_ORDER.index("G")
    ci = NUCLEOTIDE_ORDER.index("C")
    return (x[:, gi] + x[:, ci]).mean(dim=-1)


@dataclass
class DiscriminatorOutput:
    logit: torch.Tensor        # [B]

    def prob(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class PromoterDiscriminator(nn.Module):
    """Latent -> logit(promoter). Reported per negative kind, never pooled only.

    `pos_weight` is the weight of the positive class inside
    `BCEWithLogitsLoss`; `None` means 1.0 and is reported as such, a declared
    default rather than a safe one. `label_smoothing` smooths BOTH classes and
    is off by default, because with genomic negatives of which 5.1 % are
    promoters, smoothing and contamination are the same operation applied
    twice and the second one is invisible.
    """

    AUC_DINUC_MIN = 0.60
    AUC_OVER_GC_MIN = 0.05
    GC_MATCH_TOL = 0.05

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, context_dim: int = 0,
                 context_mode: str = "concat", context_center: bool = True,
                 pos_weight: Optional[float] = None,
                 label_smoothing: float = 0.0,
                 negative_kinds: Sequence[str] = NEGATIVE_KINDS,
                 pooling: Optional[LatentPooling] = None,
                 reservoir_capacity: int = 20000, seed: int = 0) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        self.pooling = pooling
        self.trunk = ConditionedTrunk(in_dim, hidden, depth, dropout,
                                      context_dim, context_mode,
                                      context_center)
        self.out_layer = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.label_smoothing = float(label_smoothing)
        # "positive" occupies code -1 so every negative kind keeps the index
        # it was declared with, and an unknown kind is an error rather than a
        # silent new bucket.
        self.kinds: Tuple[str, ...] = tuple(negative_kinds)
        self._code = {k: i for i, k in enumerate(self.kinds)}
        # columns: score, label, kind code, GC fraction
        self._res = Reservoir(reservoir_capacity, width=4, seed=seed)
        self._n = 0
        self._n_pos = 0

    def forward(self, latent, context: Optional[torch.Tensor] = None
                ) -> DiscriminatorOutput:
        feats = as_features(latent, self.pooling)
        return DiscriminatorOutput(
            logit=self.out_layer(self.trunk(feats, context)).squeeze(-1))

    def loss(self, out: DiscriminatorOutput, label: torch.Tensor,
             kind: Optional[Sequence[str]] = None,
             one_hot: Optional[torch.Tensor] = None,
             valid: Optional[torch.Tensor] = None,
             weight: Optional[torch.Tensor] = None,
             track: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Binary cross-entropy on the logit.

        `kind` is the negative kind per sample, ignored for positives, and is
        REQUIRED for the falsifier to mean anything; `one_hot` is the input
        sequence and is used only for the GC control. Both are optional so a
        smoke test can run, and `report` treats either one missing as a
        violation rather than a pass.
        """
        y = label.to(out.logit.dtype)
        if self.label_smoothing > 0.0:
            e = self.label_smoothing
            y = y * (1.0 - 2.0 * e) + e
        pw = (None if self.pos_weight is None
              else torch.tensor(self.pos_weight, dtype=out.logit.dtype,
                                device=out.logit.device))
        per = nn.functional.binary_cross_entropy_with_logits(
            out.logit, y, pos_weight=pw, reduction="none")
        mask = (torch.ones_like(out.logit, dtype=torch.bool) if valid is None
                else valid.to(torch.bool))
        total = masked_mean(per, mask, weight)
        parts = {"bce": float(total.detach()),
                 "pos_weight": 1.0 if self.pos_weight is None
                 else self.pos_weight,
                 "total": float(total.detach())}
        if track:
            self.observe(out, label, kind, one_hot, mask)
        return total, parts

    # Falsifiers
    @torch.no_grad()
    def observe(self, out: DiscriminatorOutput, label: torch.Tensor,
                kind: Optional[Sequence[str]] = None,
                one_hot: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> None:
        sel = (torch.ones_like(out.logit, dtype=torch.bool) if mask is None
               else mask.to(torch.bool))
        if not bool(sel.any()):
            return
        keep = sel.detach().cpu().numpy().astype(bool)
        s = out.logit.detach().float().cpu().numpy()[keep]
        y = label.detach().float().cpu().numpy()[keep]
        if kind is None:
            code = np.full(y.shape, -1.0)
        else:
            kinds = list(kind)
            # `kind` describes the WHOLE batch while `y` is already masked,
            # so the two are aligned explicitly: `zip` would truncate a length
            # mismatch silently and shift every label by one.
            if len(kinds) != keep.size:
                raise ValueError("kind has %d entries for a batch of %d"
                                 % (len(kinds), keep.size))
            kept_kinds = [k for k, m in zip(kinds, keep) if m]
            codes = []
            for k, is_pos in zip(kept_kinds, (y > 0.5).tolist()):
                if is_pos:
                    codes.append(-1.0)
                elif k in self._code:
                    codes.append(float(self._code[k]))
                else:
                    raise ValueError(
                        "unknown negative kind %r; declared kinds are %r. A new "
                        "kind must be declared, not absorbed into an existing "
                        "bucket" % (k, self.kinds))
            code = np.asarray(codes, dtype=np.float64)
        if one_hot is None:
            gc = np.full(y.shape, np.nan)
        else:
            gc = gc_fraction(one_hot).detach().float().cpu().numpy()[keep]
        self._res.update(np.stack([s, y, code, gc], axis=1))
        self._n += int(y.size)
        self._n_pos += int((y > 0.5).sum())

    def reset(self) -> None:
        self._res.reset()
        self._n = 0
        self._n_pos = 0

    def report(self) -> Dict[str, object]:
        """AUC overall and per negative kind, each against its GC control."""
        nan = float("nan")
        out: Dict[str, object] = {"n": self._n, "n_pos": self._n_pos,
                                  "pos_frac": (self._n_pos / float(self._n)
                                               if self._n else nan)}
        rows = self._res.values()
        if rows.shape[0] < 8 or self._n_pos == 0 or self._n_pos == self._n:
            out.update({"ok": False, "violations": [
                "not enough observed samples of both classes"]})
            return out
        s, y, code, gc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        pos = y > 0.5
        out["auc"] = roc_auc(s, y)
        gc_ok = np.isfinite(gc)

        def _gc_auc(sub: np.ndarray) -> float:
            m = sub & gc_ok
            if not m.any():
                return nan
            a = roc_auc(gc[m], y[m])
            # The baseline gets the benefit of the sign: beating
            # max(auc, 1-auc) means beating GC used optimally, not GC used in
            # whichever direction happens to be convenient.
            return nan if a != a else max(a, 1.0 - a)

        out["auc_gc"] = _gc_auc(np.ones_like(pos, dtype=bool))
        per_kind: Dict[str, float] = {}
        per_kind_gc: Dict[str, float] = {}
        per_kind_n: Dict[str, int] = {}
        for k, i in self._code.items():
            sub = pos | (code == float(i))
            n_neg = int((code == float(i)).sum())
            per_kind_n[k] = n_neg
            if n_neg < 4:
                per_kind[k] = nan
                per_kind_gc[k] = nan
                continue
            per_kind[k] = roc_auc(s[sub], y[sub])
            per_kind_gc[k] = _gc_auc(sub)
        out["auc_by_kind"] = per_kind
        out["auc_gc_by_kind"] = per_kind_gc
        out["n_by_kind"] = per_kind_n

        bad: List[str] = []
        d = per_kind.get(DINUCLEOTIDE_SHUFFLE, nan)
        d_gc = per_kind_gc.get(DINUCLEOTIDE_SHUFFLE, nan)
        if per_kind_n.get(DINUCLEOTIDE_SHUFFLE, 0) < 4:
            bad.append(
                "no dinucleotide-shuffled negatives were observed, so the "
                "mandated falsifier DID NOT RUN. An empty gate is not a passed "
                "gate")
        else:
            if not (d == d) or d < self.AUC_DINUC_MIN:
                bad.append("auc[%s] %.3f < %.2f: the head separates "
                           "COMPOSITION, not grammar"
                           % (DINUCLEOTIDE_SHUFFLE, d, self.AUC_DINUC_MIN))
            if d == d and d_gc == d_gc and d < d_gc + self.AUC_OVER_GC_MIN:
                bad.append("auc[%s] %.3f does not beat the single-feature GC "
                           "control %.3f by %.2f"
                           % (DINUCLEOTIDE_SHUFFLE, d, d_gc,
                              self.AUC_OVER_GC_MIN))
            if d_gc == d_gc and abs(d_gc - 0.5) > self.GC_MATCH_TOL:
                bad.append(
                    "GC separates the dinucleotide-shuffled negatives at %.3f, "
                    "but the shuffle is supposed to preserve composition: the "
                    "NEGATIVE GENERATOR is broken and this falsifier is void"
                    % (d_gc,))
        if per_kind_n.get("flank", 0) >= 4:
            out["flank_note"] = (
                "flank negatives are contaminated: measured 36.3 % FlankRight "
                "coverage of which 5.1 % are themselves promoters, so this "
                "bucket has an AUC ceiling below 1.0")
        a_gc = out["auc_gc"]
        if out["auc"] == out["auc"] and a_gc == a_gc and \
                out["auc"] < a_gc + self.AUC_OVER_GC_MIN:
            bad.append("auc %.3f does not beat the GC baseline %.3f overall"
                       % (out["auc"], a_gc))
        if not gc_ok.any():
            bad.append("no sequences were passed to `observe`, so the GC "
                       "controls could not be computed")
        out["violations"] = bad
        out["ok"] = not bad
        return out
