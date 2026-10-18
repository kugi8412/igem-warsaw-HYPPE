"""The scale an AUC has to be read against: a k-mer probe on the same batch.

`PromoterDiscriminator` gates on `AUC_DINUC_MIN = 0.60` and on a
single-feature GC control, and both sit on the wrong object. 0.60 cannot be
failed informatively: a linear logistic regression on 5-mer frequencies
reaches 0.9820 against an order-1 shuffle and 0.9483 against an order-2
shuffle, so a head passing at 0.61 is 0.34 WORSE than logistic regression. And
GC is the k = 1 case of the same probe, while an Euler shuffle of order `o`
preserves every k <= o+1 exactly, so GC against the shuffle class is 0.5000 by
construction -- a real control, but one that tests the negative GENERATOR
rather than the head.

The replacement is one object used in both roles: a k-mer probe fitted on the
training split and applied to the evaluation batch the head is scored on.
``ceiling(class)`` is the best AUC the probe reaches over k; the gate is
``auc_model >= ceiling + margin(n)``; and the generator check requires the
probe to sit at 0.5 +- `GENERATOR_TOL` for every k the generator claims to
preserve.

The ceiling is PER CLASS and a pool's ceiling is the MAXIMUM over its classes,
never the mean, because a discriminator trained on a union can solve the whole
union through its leakiest member. `report` therefore refuses to emit a single
pooled pass/fail.

`margin(n)` is 3 Hanley-McNeil standard errors at the evaluation size, floored
at `MARGIN_FLOOR`, and has to scale with n: at 2500 + 2500 the SE of an AUC
near 0.85 is ~0.0065 and a 0.02 bar is real, while on a 128 + 128 batch the
same SE is ~0.029 and that bar would fire on noise alone. The gate is meant
for the head's evaluation reservoir, not for one batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CompositionProbe", "MEASURED_CEILING", "margin", "auc_se"]

#: k values the probe sweeps. 6 is the largest whose 4**k features fit a few
#: thousand sequences without measuring the probe's capacity instead of the
#: corpus.
KS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
#: An Euler shuffle of order `o` preserves (o+1)-mers EXACTLY, so the probe
#: must land on chance there: at k = o+1 the two feature sets are identical up
#: to the sampling of the pair, and anything but chance means the generator is
#: broken.
GENERATOR_TOL: float = 0.02
MARGIN_FLOOR: float = 0.02
MARGIN_SIGMA: float = 3.0

#: Out-of-sample ceilings on this corpus, measured on the `C_block5`
#: conservative split as the maximum over k = 1..6 and two ridge strengths.
#: They are a floor and a regression test on the generator rather than a
#: substitute for measuring: a class whose measured ceiling drifts far below
#: the recorded one is a class whose generator changed.
MEASURED_CEILING: Dict[str, float] = {
    "reverse": 0.9999,             # banned: not a biological object at all
    "random_composition": 0.9968,  # banned (Markov-0)
    "column_shuffle": 0.9944,
    "gene": 0.9473,
    "shuffle_o2": 0.9136,          # the class in use today
    "flank_r": 0.8079,
    "revcomp": 0.8045,             # orientation test, not a negative class
    "flank_l": 0.7924,
    "rotate": 0.7225,
    "block_50": 0.7078,
    "shuffle_o4": 0.6305,
    "shuffle_o5": 0.5265,          # chance: max over 12 draws at SE 0.0138
    "chimera_2": 0.5195,           # chance; held out, the label is a hypothesis
}
#: Chance level of a ceiling: it is a maximum over 6 k values and 2 ridge
#: strengths, so under the null it sits ~1.6 SE above 0.5, not at 0.5. Reading
#: 0.5265 as a leak rather than as chance is the mistake this table invites.
SELECTION_BIAS_SIGMA: float = 1.6


def auc_se(n_pos: int, n_neg: int, auc: float) -> float:
    """Hanley-McNeil SE of a single AUC (exponential approximation)."""
    a = min(max(float(auc), 0.5), 0.999999)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / float(max(n_pos * n_neg, 1))
    return float(np.sqrt(max(var, 0.0)))


def margin(n_pos: int, n_neg: int, ceiling: float) -> float:
    """How far above the ceiling counts as evidence, at this evaluation size."""
    return max(MARGIN_FLOOR, MARGIN_SIGMA * auc_se(n_pos, n_neg, ceiling))


def kmer_freq(codes: np.ndarray, k: int) -> np.ndarray:
    """uint8 codes [n, L] -> float32 [n, 4**k]; windows with N are skipped."""
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)
    n, L = codes.shape
    m = L - k + 1
    if m <= 0:
        raise ValueError("k=%d exceeds the window length %d" % (k, L))
    idx = np.zeros((n, m), dtype=np.int64)
    valid = np.ones((n, m), dtype=bool)
    for j in range(k):
        c = codes[:, j:j + m].astype(np.int64)
        ok = c < 4
        idx = idx * 4 + np.where(ok, c, 0)
        valid &= ok
    dim = 4 ** k
    out = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        v = idx[i][valid[i]]
        if v.size:
            out[i] = np.bincount(v, minlength=dim) / float(v.size)
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with averaged ranks. Same convention as the head."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.shape[0], dtype=np.float64)
    sr = s[order]
    i = 0
    while i < sr.shape[0]:
        j = i
        while j + 1 < sr.shape[0] and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


@dataclass
class ClassVerdict:
    name: str
    ceiling: float
    ceiling_k: int
    auc_by_k: Dict[int, float]
    model_auc: float
    margin: float
    n_pos: int
    n_neg: int
    generator_ok: Optional[bool] = None
    generator_detail: str = ""

    @property
    def ok(self) -> bool:
        if self.model_auc != self.model_auc:                 # NaN
            return False
        if self.generator_ok is False:
            return False
        return self.model_auc >= self.ceiling + self.margin

    def line(self) -> str:
        return ("%-20s model %.4f  ceiling %.4f (k=%d)  +%.4f needed  -> %s%s"
                % (self.name, self.model_auc, self.ceiling, self.ceiling_k,
                   self.margin, "PASS" if self.ok else "FAIL",
                   "" if self.generator_ok is not False
                   else "  [GENERATOR BROKEN: %s]" % self.generator_detail))


class CompositionProbe:
    """Logistic probe on k-mer frequencies, fitted once, applied per class.

    Fitted on the TRAINING split, so the number it produces on an evaluation
    batch is a ceiling for that batch and not the probe's memory of it. One
    probe per (class, k): a probe fitted against the union would be weaker
    than the per-class one and would understate the ceiling, which is the one
    direction a ceiling must not err in.
    """

    def __init__(self, ks: Sequence[int] = KS, C: float = 0.3,
                 max_iter: int = 600) -> None:
        self.ks = tuple(int(k) for k in ks)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self._fit: Dict[Tuple[str, int], object] = {}
        self._norm: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def fit_class(self, name: str, pos: np.ndarray, neg: np.ndarray
                  ) -> "CompositionProbe":
        from sklearn.linear_model import LogisticRegression
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        for k in self.ks:
            x = np.concatenate([kmer_freq(pos, k), kmer_freq(neg, k)])
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True)
            sd[sd < 1e-12] = 1.0
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                     solver="lbfgs")
            clf.fit((x - mu) / sd, y)
            self._fit[(name, k)] = clf
            self._norm[(name, k)] = (mu, sd)
        return self

    def score(self, name: str, codes: np.ndarray, k: int) -> np.ndarray:
        clf = self._fit[(name, k)]
        mu, sd = self._norm[(name, k)]
        return clf.decision_function((kmer_freq(codes, k) - mu) / sd)

    def ceiling(self, name: str, pos: np.ndarray, neg: np.ndarray
                ) -> Tuple[float, int, Dict[int, float]]:
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        by_k: Dict[int, float] = {}
        for k in self.ks:
            s = np.concatenate([self.score(name, pos, k),
                                self.score(name, neg, k)])
            by_k[k] = roc_auc(s, y)
        best = max(by_k, key=lambda kk: by_k[kk])
        return by_k[best], best, by_k

    def verdict(self, name: str, pos: np.ndarray, neg: np.ndarray,
                model_auc: float, preserves_upto_k: Optional[int] = None
                ) -> ClassVerdict:
        """One class, one verdict.

        `preserves_upto_k` turns on the generator check: for an Euler shuffle
        of order `o`, pass `o + 1`.
        """
        ceil, ck, by_k = self.ceiling(name, pos, neg)
        v = ClassVerdict(name=name, ceiling=ceil, ceiling_k=ck,
                         auc_by_k={k: round(a, 4) for k, a in by_k.items()},
                         model_auc=float(model_auc),
                         margin=margin(len(pos), len(neg), ceil),
                         n_pos=int(len(pos)), n_neg=int(len(neg)))
        if preserves_upto_k is not None:
            bad = [(k, by_k[k]) for k in self.ks
                   if k <= preserves_upto_k
                   and abs(by_k[k] - 0.5) > GENERATOR_TOL]
            v.generator_ok = not bad
            if bad:
                v.generator_detail = ("the class claims to preserve every "
                                      "k <= %d, but the probe separates at %s"
                                      % (preserves_upto_k,
                                         ["k=%d: %.4f" % b for b in bad]))
        return v


def report(verdicts: Iterable[ClassVerdict]) -> Dict[str, object]:
    """Per class, plus the pool ceiling as the MAXIMUM, never the mean."""
    vs = list(verdicts)
    if not vs:
        return {"ok": False,
                "violations": ["no negative class was scored, so the gate DID "
                               "NOT RUN; an empty gate is not a passed gate"]}
    worst = max(vs, key=lambda v: v.ceiling)
    return {
        "by_class": {v.name: {"model_auc": round(v.model_auc, 4),
                              "ceiling": round(v.ceiling, 4),
                              "ceiling_k": v.ceiling_k,
                              "margin": round(v.margin, 4),
                              "auc_by_k": v.auc_by_k,
                              "generator_ok": v.generator_ok,
                              "ok": v.ok} for v in vs},
        "pool_ceiling": round(worst.ceiling, 4),
        "pool_ceiling_class": worst.name,
        "violations": [v.line() for v in vs if not v.ok],
        "ok": all(v.ok for v in vs),
    }
