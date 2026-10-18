"""Trust region for the generator, built from CLOSE negatives.

The discriminator exists so the generator does not mutate greedily. High AUC
is not the goal; a boundary in the right place is. The discriminator in
`heads/discriminator.py` is trained against `gene`, `flank_r` and `shuffle`,
which separate a promoter from the interior of a gene, while the generator
produces promoters with point mutations -- so the boundary is learned where it
will not be used. A linear 5-mer probe reaches AUC 0.9704 against `gene` and
0.9820 against `shuffle`, so those classes are largely separable by
composition alone, and a compositional boundary can stop compositional drift
and nothing else. This module supplies the close negatives, calling into
`data/mutations.py` rather than restating it.

Substitution count is not a unit -- its meaning depends on the window length.
The unit here is the 4-mer L1 distance between two independent real promoters,
0.7171 +- 0.0792, re-measured per corpus by :meth:`NaturalScale.measure`, so
100 % means "as far from its wild type as two unrelated promoters are from
each other". On that scale 2 substitutions is 2.7 % and 87 is 50 %.

The gate is applied inside the optimisation loop. All 17 construct rules
together are satisfied by 0.63 % of natural P1 promoters, so a gate applied
after a panel is generated returns 0.0063 of the budget and lets the rule set,
rather than the model, choose the panel. :func:`post_hoc_equivalent` measures
that difference on the same budget instead of asserting it.

:func:`falsifiers` collects every check this module can fail, including the
two controls of the gate itself: an empty region reads exactly like a strict
one, and a region admitting a shuffled sequence reads exactly like a passed
gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .mutations import (
    PwmScanner,
    hamming,
    kmer_l1,
    kmer_spectrum,
    substitute,
)

__all__ = [
    "LADDER", "NaturalScale", "GradedMutant", "graded_ladder",
    "motif_retention", "TrustRegion", "RegionVerdict", "constrained_walk",
    "post_hoc_equivalent", "falsifiers",
]

#: Substitution counts on a log scale. The tracked quantity, 4-mer L1, grows
#: like sqrt(k) at small k and saturates at large k, so a linear ladder would
#: spend most of its rungs inside the saturated part.
LADDER: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: p-value at which a hit counts as destroyed. The same value `MutationConfig`
#: uses for `p_strict`; a second threshold here would make the two reports
#: incomparable.
P_STRICT = 1e-4


# Scale
@dataclass(frozen=True)
class NaturalScale:
    """The unit: 4-mer L1 between two INDEPENDENT real promoters.

    `mean` is the denominator of every percentage this module reports. It is a
    property of the corpus, so it is measured and carried in the report rather
    than hard-coded at a call site.
    """

    mean: float
    sd: float
    n_pairs: int
    k: int = 4

    @classmethod
    def measure(cls, codes: np.ndarray, n_pairs: int = 4000,
                seed: int = 7898, k: int = 4) -> "NaturalScale":
        """Mean 4-mer L1 over `n_pairs` DISJOINT pairs of rows.

        A pair (i, i) contributes exactly 0 and pulls the unit down, which
        would make every mutant look further from its wild type than it is.
        """
        codes = np.asarray(codes)
        n = int(codes.shape[0])
        if n < 2:
            raise ValueError("need at least two sequences to measure a scale")
        rng = np.random.default_rng(seed)
        spec = np.stack([kmer_spectrum(codes[i], k) for i in range(n)])
        a = rng.integers(0, n, size=n_pairs)
        b = rng.integers(0, n, size=n_pairs)
        same = a == b
        b[same] = (b[same] + 1) % n
        d = np.abs(spec[a] - spec[b]).sum(axis=1)
        return cls(mean=float(d.mean()), sd=float(d.std()),
                   n_pairs=int(n_pairs), k=int(k))

    def percent(self, distance: float) -> float:
        """A raw 4-mer L1 as a percentage of the natural distance."""
        return 100.0 * float(distance) / self.mean

    def distance(self, percent: float) -> float:
        """The inverse: a percentage back into a raw 4-mer L1."""
        return float(percent) * self.mean / 100.0


# Ladder
@dataclass(frozen=True)
class GradedMutant:
    """One mutant, with everything needed to audit it."""

    row: int
    n_sub: int
    codes: np.ndarray
    positions: np.ndarray

    @property
    def realised(self) -> int:
        return int(self.positions.size)


def graded_ladder(codes: np.ndarray, rows: Sequence[int],
                  ladder: Sequence[int] = LADDER, per_rung: int = 1,
                  seed: int = 7898) -> Dict[int, List[GradedMutant]]:
    """Close negatives: each chosen promoter with k substitutions, per rung.

    Positions are drawn without replacement and `substitute` replaces each by
    a different base, so the realised Hamming distance is exactly k.
    :func:`falsifiers` asserts that equality, because a ladder whose rung
    label is a declaration rather than a measurement calibrates nothing.
    """
    codes = np.asarray(codes)
    L = int(codes.shape[1])
    rng = np.random.default_rng(seed)
    out: Dict[int, List[GradedMutant]] = {}
    for k in ladder:
        if k > L:
            raise ValueError("rung k=%d exceeds the window (%d bp)" % (k, L))
        bucket: List[GradedMutant] = []
        for r in rows:
            for _ in range(per_rung):
                pos = rng.choice(L, size=int(k), replace=False)
                mut = substitute(codes[r], pos, rng)
                bucket.append(GradedMutant(int(r), int(k), mut, np.sort(pos)))
        out[int(k)] = bucket
    return out


# Motifs
def motif_retention(scanner: PwmScanner, enriched_idx: Sequence[int],
                    wild: np.ndarray, mutant: np.ndarray,
                    pvalue: float = P_STRICT) -> Dict[str, float]:
    """Fraction of the wild type's ENRICHED hits that survive the mutation.

    Only the 23 enriched profiles are counted: the full 252-profile library
    has an aggregate enrichment of 0.986 over dinucleotide shuffles of the
    same promoters, so destroying a non-enriched hit measures nothing about
    binding and would dilute the signal elevenfold.
    """
    sub = list(enriched_idx)
    wt = scanner.scan_many(np.asarray(wild), [pvalue], subset=sub)[pvalue]
    mt = scanner.scan_many(np.asarray(mutant), [pvalue], subset=sub)[pvalue]
    wt_keys = {(h.motif_index, h.start, h.strand) for h in wt}
    mt_keys = {(h.motif_index, h.start, h.strand) for h in mt}
    kept = len(wt_keys & mt_keys)
    n = len(wt_keys)
    return {"wt_hits": float(n), "mut_hits": float(len(mt_keys)),
            "kept": float(kept),
            "retention": float(kept) / n if n else float("nan"),
            "created": float(len(mt_keys - wt_keys))}


# Region
@dataclass(frozen=True)
class RegionVerdict:
    """Why a candidate was accepted or rejected; never a bare boolean.

    A boolean cannot tell "rejected because it drifted" from "rejected because
    it broke a site", and the two call for opposite corrections.
    """

    inside: bool
    percent: float
    retention: float
    reasons: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {"inside": self.inside, "percent": round(self.percent, 3),
                "retention": round(self.retention, 4),
                "reasons": list(self.reasons)}


@dataclass
class TrustRegion:
    """The feasible set for a generator, centred on one wild-type promoter.

    Two constraints, independent by construction: drift, the 4-mer L1 to the
    wild type as a percentage of `scale.mean`, which is what a composition
    probe can see; and integrity, the fraction of the wild type's enriched
    motif hits that survive, which it cannot, and which is the whole reason a
    close negative is needed. Both bounds are quantiles of measured
    distributions and are written into the report -- a round number chosen
    because it looked strict is not a calibration.
    """

    scale: NaturalScale
    max_percent: float
    min_retention: float
    scanner: Optional[PwmScanner] = None
    enriched_idx: Tuple[int, ...] = ()
    pvalue: float = P_STRICT
    #: Cache keyed by wild-type bytes, because the enriched hit set of the
    #: centre is re-used by every proposal in the walk.
    _wt_cache: Dict[bytes, Set[Tuple[int, int, int]]] = field(
        default_factory=dict, repr=False)

    def _wt_hits(self, wild: np.ndarray) -> Set[Tuple[int, int, int]]:
        key = np.asarray(wild).tobytes()
        got = self._wt_cache.get(key)
        if got is None:
            hits = self.scanner.scan_many(
                np.asarray(wild), [self.pvalue],
                subset=list(self.enriched_idx))[self.pvalue]
            got = {(h.motif_index, h.start, h.strand) for h in hits}
            self._wt_cache[key] = got
        return got

    def verdict(self, wild: np.ndarray, candidate: np.ndarray
                ) -> RegionVerdict:
        pct = self.scale.percent(kmer_l1(wild, candidate, self.scale.k))
        reasons: List[str] = []
        if pct > self.max_percent:
            reasons.append("drift %.2f%% > %.2f%%" % (pct, self.max_percent))
        ret = float("nan")
        if self.scanner is not None and self.enriched_idx:
            wt = self._wt_hits(wild)
            mt = self.scanner.scan_many(
                np.asarray(candidate), [self.pvalue],
                subset=list(self.enriched_idx))[self.pvalue]
            mt_keys = {(h.motif_index, h.start, h.strand) for h in mt}
            # No enriched site in the wild type means this constraint has
            # nothing to say, which is not the same as satisfied and is
            # reported as NaN rather than passing silently.
            ret = len(wt & mt_keys) / len(wt) if wt else float("nan")
            if wt and ret < self.min_retention:
                reasons.append("retention %.3f < %.3f"
                               % (ret, self.min_retention))
        return RegionVerdict(not reasons, pct, ret, tuple(reasons))

    def accept(self, wild: np.ndarray, candidate: np.ndarray) -> bool:
        return self.verdict(wild, candidate).inside


# In-loop constraint
def constrained_walk(wild: np.ndarray, region: TrustRegion,
                     score: Callable[[np.ndarray], float],
                     n_steps: int = 32, proposals: int = 8,
                     seed: int = 7898,
                     block: int = 1) -> Dict[str, object]:
    """Greedy point-mutation walk in which an infeasible step is never taken.

    At every step the proposals that leave the region are dropped and the best
    of the survivors is taken, so the optimiser spends its whole budget on
    feasible candidates. Returns the trajectory, the rejections per step and
    the final candidate, so that a walk which never had a feasible move is
    visible instead of looking like a walk that converged.
    """
    wild = np.asarray(wild)
    L = int(wild.size)
    rng = np.random.default_rng(seed)
    cur = wild.copy()
    j0 = float(score(cur))
    trace: List[Dict[str, float]] = []
    rejected_total = 0
    stalled = 0
    for step in range(n_steps):
        best, best_j, rejected = None, -math.inf, 0
        for _ in range(proposals):
            cand = cur.copy()
            pos = rng.choice(L, size=int(block), replace=False)
            cand = substitute(cand, pos, rng)
            if not region.accept(wild, cand):
                rejected += 1
                continue
            j = float(score(cand))
            if j > best_j:
                best, best_j = cand, j
        rejected_total += rejected
        if best is None:
            stalled += 1
            trace.append({"step": step + 1, "J": float(score(cur)),
                          "rejected": float(rejected), "moved": 0.0})
            continue
        cur = best
        v = region.verdict(wild, cur)
        trace.append({"step": step + 1, "J": best_j,
                      "rejected": float(rejected), "moved": 1.0,
                      "percent": v.percent, "retention": v.retention})
    v = region.verdict(wild, cur)
    return {"codes": cur, "j0": j0, "j1": float(score(cur)),
            "gain": float(score(cur)) - j0, "n_sub": hamming(wild, cur),
            "rejected": rejected_total, "stalled_steps": stalled,
            "verdict": v.as_dict(), "trace": trace}


def post_hoc_equivalent(wild: np.ndarray, region: TrustRegion,
                        score: Callable[[np.ndarray], float],
                        n_steps: int = 32, proposals: int = 8,
                        seed: int = 7898, block: int = 1) -> Dict[str, object]:
    """The SAME budget, spent without the constraint, filtered at the end.

    Identical to :func:`constrained_walk` except that `region.accept` is not
    consulted until the final candidate exists, so "in-loop is not the same as
    post-hoc" ends up as a measurement in the report rather than a claim.
    """
    wild = np.asarray(wild)
    L = int(wild.size)
    rng = np.random.default_rng(seed)
    cur = wild.copy()
    j0 = float(score(cur))
    for _ in range(n_steps):
        best, best_j = None, -math.inf
        for _ in range(proposals):
            cand = substitute(cur.copy(),
                              rng.choice(L, size=int(block), replace=False),
                              rng)
            j = float(score(cand))
            if j > best_j:
                best, best_j = cand, j
        cur = best
    v = region.verdict(wild, cur)
    return {"codes": cur, "j0": j0, "j1": float(score(cur)),
            "gain": float(score(cur)) - j0, "n_sub": hamming(wild, cur),
            "survives_filter": v.inside, "verdict": v.as_dict()}


# Falsifiers
def falsifiers(codes: np.ndarray, scale: NaturalScale,
               ladder: Dict[int, List[GradedMutant]],
               region: Optional[TrustRegion] = None,
               seed: int = 7898) -> Dict[str, object]:
    """Every check this module can fail, each satisfiable by construction."""
    out: Dict[str, object] = {}
    codes = np.asarray(codes)

    out["scale_measured"] = {
        "mean": round(scale.mean, 4), "sd": round(scale.sd, 4),
        "n_pairs": scale.n_pairs,
        "pass": bool(scale.mean > 0.0 and scale.n_pairs > 0)}

    bad = [(m.n_sub, hamming(codes[m.row], m.codes))
           for b in ladder.values() for m in b
           if hamming(codes[m.row], m.codes) != m.n_sub]
    out["hamming_equals_k"] = {
        "violations": len(bad), "example": bad[:3],
        "pass": not bad,
        "why": "substitute() draws a DIFFERENT base, so k is a measurement"}

    means = {}
    for k in sorted(ladder):
        d = [kmer_l1(codes[m.row], m.codes, scale.k) for m in ladder[k]]
        means[k] = float(np.mean(d)) if d else float("nan")
    ks = sorted(means)
    mono = all(means[ks[i]] <= means[ks[i + 1]] + 1e-12
               for i in range(len(ks) - 1))
    out["monotone_distance"] = {
        "percent_by_k": {k: round(scale.percent(means[k]), 3) for k in ks},
        "pass": bool(mono)}

    if region is not None:
        rng = np.random.default_rng(seed)
        r = int(rng.integers(0, codes.shape[0]))
        wt = codes[r]
        out["region_admits_wild_type"] = {
            **region.verdict(wt, wt.copy()).as_dict(),
            "pass": bool(region.accept(wt, wt.copy())),
            "why": "an empty region reads exactly like a strict one"}
        far = wt.copy()
        rng.shuffle(far)                     # same composition, no order left
        vf = region.verdict(wt, far)
        out["region_rejects_shuffled"] = {
            **vf.as_dict(), "pass": bool(not vf.inside),
            "why": ("a column shuffle preserves the 4-mer spectrum poorly but "
                    "destroys every site; if the region admits it, the "
                    "integrity constraint is not doing anything")}

    out["all_pass"] = all(bool(v.get("pass", True)) for v in out.values()
                          if isinstance(v, dict))
    return out
