"""Pairwise ranking head: P(A > B | C) for two promoters under one condition.

The measured variance decomposition of `combat` puts the promoter main effect
at 79.7-92.7 % and the promoter x condition interaction at 6.8-18.6 %. The
absolute head regresses the main effect; this head takes the interaction,
which is about which of two promoters wins under a given condition -- a
question that survives every monotone rescaling of the label.

Conditioning the SCORER rather than the comparison is what lets the winner
change with the condition; conditioning only the comparison would give one
global ordering plus a per-condition offset that cancels in the difference.
The difference form also makes `P(A>B|C) + P(B>A|C) = 1` hold structurally,
which `antisymmetry_error` checks.

A non-promoter is a legitimate pair element and an illegitimate regression
target: it carries a ranking label ("weaker than any promoter") and no
strength label. `absolute_valid` is the mask that keeps those out of the
absolute head, and `nonprom_frac` and `acc_nonprom` are reported separately so
that a batch made mostly of free wins cannot hide inside the accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .absolute import (
    ConditionedTrunk,
    LatentPooling,
    Reservoir,
    as_features,
    masked_mean,
)

__all__ = ["PairOutput", "PairwiseRankingHead"]


@dataclass
class PairOutput:
    """Scores of both elements plus the derived comparison."""

    score_a: torch.Tensor     # [B]
    score_b: torch.Tensor     # [B]
    delta: torch.Tensor       # [B] = score_a - score_b
    prob: torch.Tensor        # [B] = sigmoid(delta / temperature)


class PairwiseRankingHead(nn.Module):
    """(z_A, z_B, C) -> P(A > B | C), trained with the Bradley-Terry likelihood.

    There is no margin here, and that is a deletion rather than a default.
    A hinge has EXACTLY ZERO gradient once a pair is ordered by more than its
    margin: on a correctly ordered pair dL/ds is -0.000e+00 at a gap of 0.50
    for both `m=0.1` and `m=0.0`, against -3.775e-01 for the logistic form.
    So `margin = 0` is not the margin-free loss, it is the hinge that gives up
    soonest, and a hinge is indifferent to the SPACING of the two scores past
    its threshold -- which is what PCC reads.

    `nonprom_weight` sits below 1.0 because pairs with a non-promoter loser
    are free wins, are plentiful, and at equal weight dominate the gradient of
    a loss already satisfied on them. `temperature` changes the probabilities
    and NOT the ordering, so it is never a quality knob; it exists so the
    probability is calibratable.
    """

    ACC_GAP_MIN = 0.10
    ACC_TOP_MIN = 0.60
    SHUFFLE_TOL = 0.05
    ANTISYM_MAX = 1e-5

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, context_dim: int = 0,
                 context_mode: str = "concat", context_center: bool = True,
                 temperature: float = 1.0, nonprom_weight: float = 0.25,
                 pooling: Optional[LatentPooling] = None,
                 reservoir_capacity: int = 20000, seed: int = 0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.pooling = pooling
        self.trunk = ConditionedTrunk(in_dim, hidden, depth, dropout,
                                      context_dim, context_mode,
                                      context_center)
        self.score_layer = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.score_layer.weight)
        nn.init.zeros_(self.score_layer.bias)
        self.temperature = float(temperature)
        self.nonprom_weight = float(nonprom_weight)

        # columns: |delta y|, correct (0/1), is_nonpromoter (0/1), delta score
        self._res = Reservoir(reservoir_capacity, width=4, seed=seed)
        self._n = 0
        self._n_nonprom = 0
        self._n_correct = 0
        self._n_correct_nonprom = 0
        self._rng = np.random.default_rng(seed + 1)

    def score(self, latent, context: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
        """s(z, C) -> [B]. Shared weights for both elements of a pair."""
        feats = as_features(latent, self.pooling)
        return self.score_layer(self.trunk(feats, context)).squeeze(-1)

    def forward(self, latent_a, latent_b,
                context: Optional[torch.Tensor] = None) -> PairOutput:
        sa = self.score(latent_a, context)
        sb = self.score(latent_b, context)
        delta = sa - sb
        return PairOutput(score_a=sa, score_b=sb, delta=delta,
                          prob=torch.sigmoid(delta / self.temperature))

    def loss(self, out: PairOutput, target: torch.Tensor,
             delta_y: Optional[torch.Tensor] = None,
             nonpromoter: Optional[torch.Tensor] = None,
             valid: Optional[torch.Tensor] = None,
             weight: Optional[torch.Tensor] = None,
             track: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Bradley-Terry likelihood. There is no margin.

        `target` is +1 when A should outrank B and -1 otherwise; it is not a
        probability and not a 0/1 label. `delta_y` is the measured difference
        and is used ONLY for the falsifier binning: a promoter/non-promoter
        pair has no measured difference and passes `delta_y = NaN` with
        `nonpromoter` set, which the reporting keeps in its own bucket.

        The loss depends on the label's sign and not on its magnitude. That is
        deliberate -- the magnitude is what the absolute head regresses, and
        having both heads read it made this one a second, worse copy of the
        regression.
        """
        sign = torch.sign(target.to(out.delta.dtype))
        if bool((sign == 0).any()):
            raise ValueError("a ranking target of exactly 0 has no direction; "
                             "drop the pair or give it a sign")
        # -log sigmoid(sign * delta / T), computed as softplus so it is finite
        # at every delta and has a non-zero gradient everywhere.
        per = torch.nn.functional.softplus(
            -sign * out.delta / self.temperature)

        mask = (torch.ones_like(out.delta, dtype=torch.bool) if valid is None
                else valid.to(torch.bool))
        w = torch.ones_like(out.delta)
        if nonpromoter is not None:
            w = torch.where(nonpromoter.to(torch.bool),
                            torch.full_like(w, self.nonprom_weight), w)
        if weight is not None:
            w = w * weight.to(w.dtype)
        total = masked_mean(per, mask, w)

        # Every value is a float on purpose: `TrainLoop._step` merges this
        # dict into `Dict[str, float]` without a cast and averages it over
        # batches, so a string entry would fail at the aggregation rather than
        # at the line that wrote it.
        parts = {"rank_nll": float(total.detach()),
                 "nonprom_weight": self.nonprom_weight,
                 "total": float(total.detach())}
        if track:
            self.observe(out, sign, delta_y, nonpromoter, mask)
        return total, parts

    @staticmethod
    def absolute_valid(is_promoter: torch.Tensor,
                       has_label: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """Which pair elements may reach the ABSOLUTE head.

        A non-promoter is a legitimate ranking element and an illegitimate
        regression target. The mask lives next to the head that creates the
        problem rather than in the training loop, where it would be one `if`
        away from being forgotten.
        """
        m = is_promoter.to(torch.bool)
        if has_label is not None:
            m = m & has_label.to(torch.bool)
        return m

    # Falsifiers
    @torch.no_grad()
    def observe(self, out: PairOutput, sign: torch.Tensor,
                delta_y: Optional[torch.Tensor],
                nonpromoter: Optional[torch.Tensor],
                mask: Optional[torch.Tensor]) -> None:
        sel = (torch.ones_like(out.delta, dtype=torch.bool) if mask is None
               else mask.to(torch.bool))
        if not bool(sel.any()):
            return
        d = out.delta[sel].detach().float().cpu().numpy()
        s = sign[sel].detach().float().cpu().numpy()
        correct = (np.sign(d) == np.sign(s)).astype(np.float64)
        # A tie in the score is not a win: with `np.sign(0) == 0 != +-1` a
        # constant scorer counts zero correct rather than all of them.
        if delta_y is None:
            dy = np.full(d.shape, np.nan)
        else:
            dy = np.abs(delta_y[sel].detach().float().cpu().numpy())
        if nonpromoter is None:
            npm = np.zeros(d.shape, dtype=np.float64)
        else:
            npm = nonpromoter[sel].detach().float().cpu().numpy()
        self._res.update(np.stack([dy, correct, npm, d], axis=1))
        self._n += int(d.size)
        self._n_nonprom += int(npm.sum())
        self._n_correct += int(correct.sum())
        self._n_correct_nonprom += int((correct * npm).sum())

    def reset(self) -> None:
        self._res.reset()
        self._n = 0
        self._n_nonprom = 0
        self._n_correct = 0
        self._n_correct_nonprom = 0

    def report(self) -> Dict[str, object]:
        """Accuracy by |delta y| quartile, with a permutation control.

        `acc_gap` is the top quartile of |delta y| minus the bottom: a head
        that ranks must find big differences easier than small ones, and a gap
        near 0 means the overall accuracy comes from the pair sampler.
        `acc_gap_ctrl` and `acc_shuffle` are the controls that can refute it.
        """
        nan = float("nan")
        out: Dict[str, object] = {"n": self._n,
                                  "n_nonprom": self._n_nonprom}
        if self._n < 8:
            out.update({"ok": False,
                        "violations": ["fewer than 8 observed pairs"]})
            return out
        rows = self._res.values()
        dy, corr, npm = rows[:, 0], rows[:, 1], rows[:, 2]
        real = (npm < 0.5) & np.isfinite(dy)
        out["acc"] = self._n_correct / float(self._n)
        out["nonprom_frac"] = self._n_nonprom / float(self._n)
        out["acc_nonprom"] = (self._n_correct_nonprom / float(self._n_nonprom)
                              if self._n_nonprom else nan)
        out["acc_real"] = float(corr[real].mean()) if real.any() else nan

        quart: List[float] = [nan, nan, nan, nan]
        gap = nan
        if int(real.sum()) >= 8:
            d = dy[real]
            c = corr[real]
            edges = np.quantile(d, [0.25, 0.5, 0.75])
            idx = np.digitize(d, edges, right=False)
            for q in range(4):
                m = idx == q
                quart[q] = float(c[m].mean()) if m.any() else nan
            if quart[3] == quart[3] and quart[0] == quart[0]:
                gap = quart[3] - quart[0]
            # Permuting the |delta y| column keeps both marginals and
            # destroys the pairing, so a gap that survives it came from the
            # binning and not from the head.
            perm = self._rng.permutation(d.size)
            cq: List[float] = []
            idxp = np.digitize(d[perm], edges, right=False)
            for q in range(4):
                m = idxp == q
                cq.append(float(c[m].mean()) if m.any() else nan)
            out["acc_gap_ctrl"] = (cq[3] - cq[0]
                                   if cq[3] == cq[3] and cq[0] == cq[0]
                                   else nan)
            # A second control, on the labels rather than the bins: the same
            # scores judged against PERMUTED targets, with the target sign
            # recovered from (sign of the score difference, correct) so it
            # costs no storage. Chance here is p^2 + (1-p)^2, not 0.5, when
            # the sampler is unbalanced, which is why the falsifier compares
            # `acc_real` against this number. A sampler that always puts the
            # winner first and a head that always answers "A" both score 1.00
            # here and are caught.
            sgn = np.sign(rows[real, 3])
            tgt = np.where(c > 0.5, sgn, -sgn)
            shuffled = tgt[self._rng.permutation(tgt.size)]
            out["acc_shuffle"] = float((sgn == shuffled).mean())
            out["target_balance"] = float((tgt > 0).mean())
        out["acc_q1"], out["acc_q2"] = quart[0], quart[1]
        out["acc_q3"], out["acc_q4"] = quart[2], quart[3]
        out["acc_gap"] = gap

        bad: List[str] = []
        if gap != gap:
            bad.append("acc_gap could not be computed: no pairs with a "
                       "measured |delta y|. Silence is not a passed gate")
        else:
            if gap < self.ACC_GAP_MIN:
                bad.append("acc_gap %.3f < %.2f: large differences are not "
                           "easier than small ones, so the head is guessing "
                           "(q1 %.3f, q4 %.3f)"
                           % (gap, self.ACC_GAP_MIN, quart[0], quart[3]))
            ctrl = out.get("acc_gap_ctrl", nan)
            if ctrl == ctrl and gap <= ctrl + 0.02:
                bad.append("acc_gap %.3f does not beat its permutation control "
                           "%.3f: the gap is in the binning" % (gap, ctrl))
            if quart[3] == quart[3] and quart[3] < self.ACC_TOP_MIN:
                bad.append("acc_q4 %.3f < %.2f: even the easiest quartile is "
                           "near chance" % (quart[3], self.ACC_TOP_MIN))
        sh = out.get("acc_shuffle", nan)
        ar = out.get("acc_real", nan)
        if sh == sh and ar == ar and ar <= sh + self.SHUFFLE_TOL:
            bad.append("acc_real %.3f does not beat its label-permutation "
                       "control %.3f (target_balance %.3f): the accuracy is "
                       "carried by the sampler, not by the head"
                       % (ar, sh, out.get("target_balance", nan)))
        if self._n_nonprom and out["nonprom_frac"] > 0.5:
            bad.append("nonprom_frac %.3f: most pairs are guaranteed wins, so "
                       "`acc` describes the sampler, not the head"
                       % (out["nonprom_frac"],))
        out["violations"] = bad
        out["ok"] = not bad
        return out

    # Structural checks
    @torch.no_grad()
    def antisymmetry_error(self, latent_a, latent_b,
                           context: Optional[torch.Tensor] = None) -> float:
        """max |P(A>B|C) + P(B>A|C) - 1|. Structural, so ~1e-7 or it is wired
        wrong.

        Run in EVAL: `context_center` is a BatchNorm, so in training mode the
        score of one element depends on the batch it arrived with and the two
        forward passes are not comparable.
        """
        ab = self.forward(latent_a, latent_b, context).prob
        ba = self.forward(latent_b, latent_a, context).prob
        return float((ab + ba - 1.0).abs().max())

    def reference_equivalence(self, n: int = 64, seed: int = 0) -> float:
        """max |our loss - nn.SoftMarginLoss|, which must be 0.

        `SoftMarginLoss` is `log(1 + exp(-y*x))`, which IS
        `softplus(-sign * delta)` at this head's temperature. It runs through
        `self.loss` rather than re-typing the formula, so it tests the path the
        trainer takes: a check that re-derives the arithmetic can agree with
        itself while the trainer does something else.
        """
        g = torch.Generator().manual_seed(seed)
        sa = torch.randn(n, generator=g)
        sb = torch.randn(n, generator=g)
        sign = torch.where(torch.rand(n, generator=g) > 0.5,
                           torch.ones(n), -torch.ones(n))
        delta = sa - sb
        ref = nn.SoftMarginLoss()(delta / self.temperature, sign)
        out = PairOutput(score_a=sa, score_b=sb, delta=delta,
                         prob=torch.sigmoid(delta / self.temperature))
        ours, _ = self.loss(out, sign, delta_y=None, track=False)
        return float((ref - ours).abs().max())
