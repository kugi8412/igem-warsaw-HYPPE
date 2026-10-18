"""Controlled mutations of a design target, and the gates they feed.

SCC on a fixed split cannot tell whether a model reads motifs or merely reads
composition. This protocol answers that separately: one promoter is perturbed
in two ways matched on everything except whether the perturbed positions sit
inside an enriched motif, and |delta prediction| is compared between the two
classes. A small mutation is 2 substitutions (destroys a hit with p = 0.9373,
moves the 4-mer L1 by 2.7 % of the distance between two real promoters); a
large one is 88 (87 is where that L1 reaches 50 % of the natural distance,
rounded up so it divides by the 4 classes).

The verdict is a PASS/FAIL gate, never a term summed into the score: the sd of
the mutation component is 0.135 against 0.0073 for the sd of an SCC difference
between two models, so a weight small enough not to drown the ranking is one
that decides nothing. `mutation_weight_ceiling()` recomputes that.

"In motif" means a hit of one of the 23 profiles enriched against the
promoters' own dinucleotide shuffles; the library as a whole shows none (ratio
0.986) because 23 enriched and 17 depleted profiles cancel. "Outside" uses the
looser p < 1e-3: at 1e-4 the free pool looks like 600 bp, but 464 of those sit
inside a weak hit and the claim "this hits no motif" would be false.

Two readings are easy to get wrong. AUC is taken on |delta| and never on
signed delta, since destroying an activating and a repressing site move the
prediction in opposite directions. And AUC is clustered by LOCUS: on the
delivered design the true sd is 0.0788 where the independent-sample formula
reports 0.0463, so `locus_clustered_auc()` returns both SEs at the same n.

An all-zero |delta| on the outside class is a broken intervention, not a
specific model. The protocol measures consistency, not correctness -- there
are no wet measurements for pks1 variants -- so it can exclude a model and
never confirm one.

No sequence is reverse-complemented, canonicalised or RC-augmented anywhere
here; promoters are directional. Motif occupancy is scanned on both strands,
which is a symmetry of binding rather than of the sequence.

Every path is an argument, and the constants below are frozen with their
provenance because they are results, not settings: recomputing them needs the
203 MB corpus and 2000 dinucleotide shuffles.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .sequences import NUCLEOTIDE_ORDER

NUCLEOTIDES = NUCLEOTIDE_ORDER           # "ACGT" -- asserted below
_CODE: Dict[str, int] = {c: i for i, c in enumerate(NUCLEOTIDES)}
N_CODE = 4                               # anything not ACGT

if NUCLEOTIDES != "ACGT":
    raise ImportError(
        "data.sequences.NUCLEOTIDE_ORDER is %r; this module's complement "
        "permutation and MEME column order both assume 'ACGT'. The sibling "
        "track used A,C,T,G and silently mapped A->G, C->T."
        % (NUCLEOTIDES,))

# Complement as an index permutation under ACGT: A<->T, C<->G.
_COMPLEMENT_INDEX: Tuple[int, ...] = (3, 2, 1, 0)


# Measurements
#: Measured base composition of the promoter corpus (A, C, G, T), the null for
#: every p-value threshold. The uniform header background changes no
#: conclusion, but the null has to be the distribution actually being scanned
#: or "p < 1e-4" does not mean 1e-4.
MEASURED_BACKGROUND: Tuple[float, float, float, float] = (
    0.2522, 0.2548, 0.2304, 0.2626)

#: The background declared in the MEME header. Kept for the control that the
#: choice of null does not move the ranking (max |dIC| = 0.6275 bits).
UNIFORM_BACKGROUND: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

#: Profiles enriched in 2000 real promoters against their own dinucleotide
#: shuffles, |z| > 3.48 (Bonferroni over 252 tests), 1.25x-2.47x.
ENRICHED_MOTIFS: Tuple[str, ...] = (
    "SFP1", "mat1-Mc", "FGRRES_07914", "fmf-1", "NCU06907", "NCU09556",
    "NDT80", "tah-3", "NCU02666", "ANIA_11222", "ANIA_07921", "ANIA_09047",
    "RO3G_02020", "NCU03699", "ANIA_07189", "NCU00019", "ANIA_10912",
    "RDS1", "NCU06503", "GZF3", "SKN7", "DAL81", "NCU08527",
)

#: Profiles depleted at the same threshold. Not used by the protocol; recorded
#: because their existence is why the aggregate enrichment is flat.
DEPLETED_MOTIFS: Tuple[str, ...] = (
    "GIS1", "ANIA_00835", "ANIA_09516", "ANIA_02122", "ANIA_11818", "ada-3",
    "NCU05294", "msn-1", "vad-2", "NCU05064", "sub-1", "NCU02307",
    "NCU03043", "REB1", "CAT8", "STB4", "SCHCODRAFT_110010",
)

#: |z| for Bonferroni over 252 tests at alpha = 0.05.
ENRICHMENT_Z: float = 3.48

#: Profiles known to behave badly as scanners. They are not removed -- DAL81
#: is enriched and dropping it would silently change the definition of "in
#: motif" -- but a per-profile report has to be able to name them.
SUSPECT_PROFILES: Dict[str, str] = {
    "DAL81": "stored CONSENSUS, not a PWM: 2.000 bits/position, w=19. At "
             "p<1e-4 it admits any 19-mer within ~6.9 mismatches and behaves "
             "as an approximate-match search, not as a PWM.",
    "HAA1": "cannot reach p<1e-4 at all: threshold 9.550 against a maximum "
            "score of 9.548, so 251 profiles serve that threshold, not 252.",
    "xlr-1": "ortholog of XYR1, a key Trichoderma TF, at IC 1.749 over 9 "
             "positions -- useless as a scanner, and worse than absent "
             "because it looks like coverage of that regulator.",
}

#: (mean, sd) L1 distance between the k-mer spectra of two random real
#: promoters -- the unit in which a mutation's compositional size is expressed.
NATURAL_KMER_L1: Dict[int, Tuple[float, float]] = {
    3: (0.4417, 0.1077),
    4: (0.7171, 0.0792),
    6: (1.6667, 0.0443),
}

#: P(a hit is destroyed) by k random substitutions inside its window, on the
#: 39 pks1 hits at p<1e-4, 400 repeats per point. Targeting the highest-IC
#: columns instead gives 1.0000 at k=1 (39/39).
DESTRUCTION_PROBABILITY: Dict[int, float] = {
    1: 0.7992, 2: 0.9373, 3: 0.9806, 5: 0.9958,
}

#: sd of SCC on this split at SCC ~ 0.53, by n. From a model-free surrogate
#: (mean of the 18 other genomes -> P1) degraded to realistic accuracy.
SCC_SD_BY_N: Dict[int, float] = {
    500: 0.0339, 1000: 0.0209, 1179: 0.0187, 2000: 0.0158, 4000: 0.0091,
}

#: sd of the difference of two models' SCC on the same split (n=1179,
#: SCC ~ 0.53), by the correlation of their residuals.
SCC_DIFF_SD_BY_RHO: Dict[float, float] = {
    0.00: 0.0277, 0.50: 0.0213, 0.80: 0.0137, 0.90: 0.0101, 0.94: 0.0073,
    0.97: 0.0054,
}

#: sd of the mutation component (2*AUC-1) simulated with independence at the
#: locus level, keyed by (loci, variants per locus).
MUTATION_SD_BY_DESIGN: Dict[Tuple[int, int], float] = {
    (3, 5): 0.3622, (5, 5): 0.2799, (12, 5): 0.1841, (22, 5): 0.1346,
    (22, 20): 0.1270, (60, 5): 0.0800,
}

#: Gate threshold for B1: the 95th percentile of the null AUC at 22 loci x 5
#: variants; power 0.93 at d=1.0. On the locus structure the design really
#: produces on pks1 that percentile is 0.6311, so 0.64 is conservative here --
#: the safe direction for an exclusion gate.
B1_AUC_THRESHOLD: float = 0.64

#: Seed used by the whole track.
DEFAULT_SEED: int = 7898

# Class names
SMALL_IN_MOTIF = "SMALL_IN_MOTIF"
SMALL_OUTSIDE = "SMALL_OUTSIDE"
LARGE_IN_MOTIF = "LARGE_IN_MOTIF"
LARGE_OUTSIDE = "LARGE_OUTSIDE"
VARIANT_CLASSES: Tuple[str, ...] = (
    SMALL_IN_MOTIF, SMALL_OUTSIDE, LARGE_IN_MOTIF, LARGE_OUTSIDE)

#: The delivered variant FASTA carries the class names in Polish. It is the
#: artefact that goes to the cluster, so both spellings are accepted on read;
#: only the English ones are written.
LEGACY_CLASS_NAMES: Dict[str, str] = {
    "MALA_W_MOTYWIE": SMALL_IN_MOTIF,
    "MALA_POZA": SMALL_OUTSIDE,
    "DUZA_W_MOTYWIE": LARGE_IN_MOTIF,
    "DUZA_POZA": LARGE_OUTSIDE,
}


# Encoding
def encode_sequence(seq: str) -> np.ndarray:
    """ACGT -> 0..3 as uint8; every other character -> 4.

    Code 4 is not "unknown, treat as average": a window containing it scores
    -inf and cannot be a hit. Averaging an N would let a run of Ns invent
    binding sites at the background rate.
    """
    a = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
    out = np.full(a.shape, N_CODE, dtype=np.uint8)
    for c, i in _CODE.items():
        out[a == ord(c)] = i
    return out


def decode_sequence(codes: np.ndarray) -> str:
    """0..3 -> ACGT, 4 -> N."""
    return "".join(NUCLEOTIDES[int(c)] if int(c) < 4 else "N" for c in codes)


def kmer_spectrum(codes: np.ndarray, k: int = 4) -> np.ndarray:
    """Normalised k-mer frequency vector, windows containing N dropped."""
    codes = np.asarray(codes)
    L = int(codes.size)
    if L < k:
        return np.zeros(4 ** k, dtype=np.float64)
    idx = np.zeros(L - k + 1, dtype=np.int64)
    ok = np.ones(L - k + 1, dtype=bool)
    for j in range(k):
        col = codes[j:L - k + 1 + j].astype(np.int64)
        ok &= col < 4
        idx = idx * 4 + np.minimum(col, 3)
    c = np.bincount(idx[ok], minlength=4 ** k).astype(np.float64)
    s = c.sum()
    return c / s if s > 0 else c


def kmer_l1(a: np.ndarray, b: np.ndarray, k: int = 4) -> float:
    """L1 distance between two sequences' k-mer spectra.

    Two random real promoters sit at 0.7171 +- 0.0792 for k=4, which is the
    unit: 2 substitutions move 2.7 % of it, 88 move about 43 %.
    """
    return float(np.abs(kmer_spectrum(a, k) - kmer_spectrum(b, k)).sum())


def gc_fraction(codes: np.ndarray) -> float:
    """GC fraction from codes, using the declared channel order."""
    codes = np.asarray(codes)
    valid = codes < 4
    if not valid.any():
        return float("nan")
    gc = (codes == _CODE["C"]) | (codes == _CODE["G"])
    return float(gc[valid].sum() / valid.sum())


def substitute(codes: np.ndarray, positions: Sequence[int],
               rng: np.random.Generator) -> np.ndarray:
    """Replace each position by a different base, uniformly among the other 3.

    Drawing from all four would leave 25 % of the "mutations" identical to the
    wild type, making `nsub` in the header a declaration rather than a
    measurement; F6 asserts `hamming()` against `nsub` for that reason.
    """
    v = np.asarray(codes).copy()
    for p in positions:
        cur = int(v[p])
        alt = [b for b in range(4) if b != cur]
        v[p] = alt[int(rng.integers(3))]
    return v


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing positions; requires equal length."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("length mismatch: %d vs %d" % (a.size, b.size))
    return int((a != b).sum())


def blocks_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs of True as half-open [start, end) intervals."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    n = int(np.asarray(mask).size)
    for i in range(n + 1):
        v = bool(mask[i]) if i < n else False
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


# Motif model
@dataclass(frozen=True)
class ScanMotif:
    """One PWM in scanning layout: probabilities as (w, 4), rows sum to 1."""

    name: str
    pwm: np.ndarray
    nsites: int = 20

    @property
    def width(self) -> int:
        return int(self.pwm.shape[0])

    def information_bits(self, background: Sequence[float] = MEASURED_BACKGROUND
                         ) -> float:
        """Total relative entropy to the background, in bits."""
        bg = np.asarray(background, dtype=np.float64)[None, :]
        p = np.asarray(self.pwm, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = p * np.log2(p / bg)
        t[~np.isfinite(t)] = 0.0
        return float(t.sum())

    def position_information(self,
                             background: Sequence[float] = MEASURED_BACKGROUND,
                             strand: int = 1) -> np.ndarray:
        """Per-column relative entropy, oriented to `strand`.

        Targeting the two highest-IC columns destroys a hit in 39/39 cases
        against 0.9373 for two random ones, which is what makes the small
        in-motif class a motif experiment rather than a coin flip.
        """
        bg = np.asarray(background, dtype=np.float64)[None, :]
        p = np.asarray(self.pwm, dtype=np.float64)
        if strand < 0:
            p = p[::-1, ::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = p * np.log2(p / bg)
        t[~np.isfinite(t)] = 0.0
        return t.sum(axis=1)


def load_motifs(path: str) -> List[ScanMotif]:
    """Read a MEME v4 file through `model.motifs.read_meme` and transpose.

    The import is lazy because that parser pulls in torch while the rest of
    this module is numpy and runs on a laptop. Reusing it is deliberate: it
    asserts `ALPHABET= ACGT` rather than assuming it.
    """
    from ..model.motifs import read_meme  # noqa: WPS433

    out: List[ScanMotif] = []
    for m in read_meme(path):
        pwm = np.asarray(m.pwm.detach().cpu().numpy(), dtype=np.float64).T
        pwm = pwm / pwm.sum(axis=1, keepdims=True)
        out.append(ScanMotif(name=m.name, pwm=np.ascontiguousarray(pwm),
                             nsites=int(m.nsites) or 20))
    return out


# Scanner
@dataclass(frozen=True, order=True)
class Hit:
    """One motif occurrence: profile, strand, start. Hashable on purpose.

    Destroyed and created hits are set differences between the wild type's
    hits and a variant's, so a hit's identity has to be exactly (profile,
    strand, start) -- one that moved by a base is a different hit.
    """

    motif_index: int
    motif_name: str
    strand: int
    start: int
    width: int


class PwmScanner:
    """Log-odds PWM scan on both strands with exact p-value thresholds.

    The threshold is computed, not approximated: "p < 1e-4" has to mean 1e-4
    under the background actually being scanned, or the in-motif / outside
    split is a split on an unknown quantity. With no scipy available the null
    score distribution is built by dynamic programming over the log-odds
    matrix quantised to `granularity` bits (TFM-PVALUE style), bounding the
    quantisation error by w * granularity = 0.019 bits at w = 9.4.
    `validate_threshold_by_enumeration()` checks that DP against all 4^w
    words.

    A binding site can sit on either strand, so the reverse-complement
    profile is scanned. The sequence itself is never reverse-complemented,
    canonicalised or augmented.
    """

    def __init__(self, motifs: Sequence[ScanMotif],
                 background: Sequence[float] = MEASURED_BACKGROUND,
                 granularity: float = 0.002,
                 pseudocount: float = 0.1,
                 dtype: type = np.float64) -> None:
        if not motifs:
            raise ValueError("no motifs")
        bg = np.asarray(background, dtype=np.float64)
        if bg.shape != (4,) or not np.isfinite(bg).all() or (bg <= 0).any():
            raise ValueError("background must be 4 positive numbers (A,C,G,T)")
        self.motifs = list(motifs)
        self.background = bg / bg.sum()
        self.granularity = float(granularity)
        self.pseudocount = float(pseudocount)
        self.dtype = dtype

        names = [m.name for m in self.motifs]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                "duplicate motif names %s -- names are the contract used by "
                "ENRICHED_MOTIFS, an index would silently follow file order"
                % dup)
        self._index = {n: i for i, n in enumerate(names)}

        self._logodds: List[np.ndarray] = []
        self._fwd: List[np.ndarray] = []
        self._rc: List[np.ndarray] = []
        for m in self.motifs:
            lo = self._log_odds(m)
            self._logodds.append(lo)
            self._fwd.append(self._table(lo))
            self._rc.append(self._table(lo[::-1, ::-1].copy()))
        self._thresholds: Dict[Tuple[int, float], Tuple[float, float]] = {}

    # Internals
    def _log_odds(self, m: ScanMotif) -> np.ndarray:
        """Pseudocount-corrected log2 odds, (w, 4).

        The pseudocount mass is spread by the background, so a zero cell
        becomes background-rare rather than -inf. Without it a single 0.0 in
        the MEME file makes every window containing that base impossible and
        the profile stops being a probabilistic model.
        """
        n = float(m.nsites) if m.nsites else 20.0
        p = (m.pwm * n + self.pseudocount * self.background[None, :])
        p = p / p.sum(axis=1, keepdims=True)
        return np.log2(p / self.background[None, :])

    def _table(self, lo: np.ndarray) -> np.ndarray:
        """(w, 5) lookup with column 4 = -inf, so an N kills the window."""
        t = np.zeros((lo.shape[0], 5), dtype=self.dtype)
        t[:, :4] = lo.astype(self.dtype)
        t[:, 4] = -np.inf
        return t

    def _null_pmf(self, lo: np.ndarray) -> Tuple[int, np.ndarray]:
        """Exact pmf of the quantised score under an i.i.d. background."""
        q = np.round(lo / self.granularity).astype(np.int64)
        pmf = np.zeros(int(q.max(axis=1).sum() - q.min(axis=1).sum()) + 1,
                       dtype=np.float64)
        pmf[0] = 1.0
        cur_min, cur_len = 0, 1
        for pos in range(q.shape[0]):
            pmin, pmax = int(q[pos].min()), int(q[pos].max())
            new_len = cur_len + (pmax - pmin)
            new = np.zeros(new_len, dtype=np.float64)
            for b in range(4):
                sh = int(q[pos, b]) - pmin
                new[sh:sh + cur_len] += pmf[:cur_len] * self.background[b]
            pmf[:new_len] = new
            pmf[new_len:] = 0.0
            cur_min += pmin
            cur_len = new_len
        return cur_min, pmf[:cur_len]

    # Public
    def __len__(self) -> int:
        return len(self.motifs)

    def index(self, name: str) -> int:
        if name not in self._index:
            raise KeyError("no motif named %r in this bank" % name)
        return self._index[name]

    def indices(self, names: Iterable[str]) -> List[int]:
        return [self.index(n) for n in names]

    def known(self, names: Iterable[str]) -> Tuple[List[str], List[str]]:
        """(present, missing) -- so a renamed profile fails loudly, not quietly."""
        present, missing = [], []
        for n in names:
            (present if n in self._index else missing).append(n)
        return present, missing

    def max_score(self, motif_index: int) -> float:
        return float(self._logodds[motif_index].max(axis=1).sum())

    def threshold(self, motif_index: int, pvalue: float) -> Tuple[float, float]:
        """(score in bits, achieved tail probability) for P(S >= t) <= pvalue.

        The achieved p is returned because the score is discrete: 1e-4 on a
        short profile can give 3.1e-5, and an unreachable request gives a
        threshold above the profile's maximum at p 0.0. A caller reporting
        "252 profiles at p<1e-4" has to be able to see that it is 251.
        """
        key = (int(motif_index), float(pvalue))
        if key not in self._thresholds:
            self._fill_thresholds([int(motif_index)], [float(pvalue)])
        return self._thresholds[key]

    def _fill_thresholds(self, indices: Sequence[int],
                         pvalues: Sequence[float]) -> None:
        """Compute the null once per profile and read every threshold off it."""
        for mi in indices:
            wanted = [p for p in pvalues if (mi, p) not in self._thresholds]
            if not wanted:
                continue
            off, pmf = self._null_pmf(self._logodds[mi])
            tail = np.cumsum(pmf[::-1])[::-1]
            for p in wanted:
                ok = np.nonzero(tail <= p)[0]
                if ok.size == 0:
                    # Unreachable: put the bar above the maximum achievable
                    # score so that no window can ever clear it.
                    self._thresholds[(mi, p)] = (
                        float(off + pmf.size) * self.granularity, 0.0)
                else:
                    i = int(ok[0])
                    self._thresholds[(mi, p)] = (
                        float(off + i) * self.granularity, float(tail[i]))

    def unreachable(self, pvalue: float,
                    subset: Optional[Sequence[int]] = None) -> List[str]:
        """Profiles whose maximum score is below their own threshold."""
        idx = list(range(len(self.motifs))) if subset is None else list(subset)
        self._fill_thresholds(idx, [float(pvalue)])
        return [self.motifs[i].name for i in idx
                if self.threshold(i, pvalue)[0] > self.max_score(i)]

    def scan_many(self, codes: np.ndarray, pvalues: Sequence[float],
                  subset: Optional[Sequence[int]] = None
                  ) -> Dict[float, Set[Hit]]:
        """Hits at several thresholds in one pass over the scores.

        The design loop rescans a whole variant after every candidate
        substitution and needs the strict and the loose hit sets together.
        """
        codes = np.asarray(codes, dtype=np.uint8)
        L = int(codes.size)
        pvs = [float(p) for p in pvalues]
        idx = list(range(len(self.motifs))) if subset is None else sorted(
            {int(i) for i in subset})
        self._fill_thresholds(idx, pvs)
        out: Dict[float, Set[Hit]] = {p: set() for p in pvs}
        for mi in idx:
            w = self.motifs[mi].width
            if w > L:
                continue
            M = L - w + 1
            thr = {p: self.threshold(mi, p)[0] for p in pvs}
            name = self.motifs[mi].name
            for strand, tab in ((1, self._fwd[mi]), (-1, self._rc[mi])):
                s = np.zeros(M, dtype=self.dtype)
                for k in range(w):
                    s += tab[k][codes[k:k + M]]
                for p in pvs:
                    for i in np.nonzero(s >= thr[p])[0]:
                        out[p].add(Hit(mi, name, strand, int(i), w))
        return out

    def scan(self, codes: np.ndarray, pvalue: float,
             subset: Optional[Sequence[int]] = None) -> Set[Hit]:
        """Hits at one threshold."""
        return self.scan_many(codes, [pvalue], subset)[float(pvalue)]

    @staticmethod
    def coverage(hits: Iterable[Hit], length: int) -> np.ndarray:
        """Boolean mask of positions inside at least one hit."""
        cov = np.zeros(int(length), dtype=bool)
        for h in hits:
            cov[h.start:h.start + h.width] = True
        return cov

    def validate_threshold_by_enumeration(self, motif_index: int,
                                          pvalue: float,
                                          max_width: int = 8
                                          ) -> Dict[str, object]:
        """Check the DP tail against all 4^w words.

        Without this the phrase "p < 1e-4" is unverifiable on a machine with
        no scipy: the DP could be self-consistently wrong. Enumeration is
        exact but costs 4^w, so it is capped at `max_width` (4^8 = 65 536).
        """
        w = self.motifs[motif_index].width
        if w > max_width:
            raise ValueError(
                "width %d exceeds max_width %d; enumeration is 4^w" %
                (w, max_width))
        lo = self._logodds[motif_index]
        t, achieved = self.threshold(motif_index, pvalue)
        probs = np.ones(1, dtype=np.float64)
        scores = np.zeros(1, dtype=np.float64)
        for pos in range(w):
            probs = (probs[:, None] * self.background[None, :]).ravel()
            scores = (scores[:, None] + lo[pos][None, :]).ravel()
        brute = float(probs[scores >= t].sum())
        return {
            "motif": self.motifs[motif_index].name,
            "width": int(w),
            "words": int(probs.size),
            "pvalue": float(pvalue),
            "threshold_bits": float(t),
            "dp_tail": float(achieved),
            "brute_tail": brute,
            "abs_difference": abs(brute - float(achieved)),
            "max_score": self.max_score(motif_index),
        }

    def precision_check(self, codes: np.ndarray, pvalue: float
                        ) -> Dict[str, object]:
        """How many hits depend on float32 vs float64 accumulation.

        A score can land exactly on the threshold. If the two precisions
        disagree on any window then "this variant destroys exactly k hits" is
        precision-dependent and has to be reported as such.
        """
        other = PwmScanner(self.motifs, self.background, self.granularity,
                           self.pseudocount,
                           np.float32 if self.dtype is np.float64
                           else np.float64)
        a = self.scan(codes, pvalue)
        b = other.scan(codes, pvalue)
        return {
            "pvalue": float(pvalue),
            "hits_here": len(a),
            "hits_other_precision": len(b),
            "symmetric_difference": len(a ^ b),
        }

    def report(self, pvalues: Sequence[float] = (1e-3, 1e-4)) -> Dict[str, object]:
        """Bank-level facts that a run needs in its log."""
        ic = np.array([m.information_bits(self.background) for m in self.motifs])
        widths = np.array([m.width for m in self.motifs])
        out: Dict[str, object] = {
            "n_motifs": len(self.motifs),
            "width_mean": float(widths.mean()),
            "width_median": float(np.median(widths)),
            "ic_mean": float(ic.mean()),
            "ic_median": float(np.median(ic)),
            "ic_below_8_bits": int((ic < 8.0).sum()),
            "background": [float(v) for v in self.background],
            "granularity_bits": self.granularity,
            "suspect_profiles": sorted(
                set(SUSPECT_PROFILES) & set(self._index)),
        }
        for p in pvalues:
            out["unreachable_at_%g" % p] = self.unreachable(p)
        return out


# Variants
@dataclass(frozen=True)
class Variant:
    """One mutated sequence with its full provenance.

    F6 asserts that the Hamming distance to the wild type equals
    `len(positions)` equals `n_substitutions`, which is what catches a
    substitution that redrew the base it replaced.
    """

    name: str
    variant_class: str
    sequence: str
    positions: Tuple[int, ...]
    locus: str
    motif: str

    @property
    def n_substitutions(self) -> int:
        return len(self.positions)

    @property
    def codes(self) -> np.ndarray:
        return encode_sequence(self.sequence)

    def header(self) -> str:
        return "%s|nsub=%d|locus=%s|motif=%s|pos=%s" % (
            self.name, self.n_substitutions, self.locus, self.motif,
            ",".join(str(p) for p in self.positions))


@dataclass
class MutationConfig:
    """Every number here is a measurement, and changing one changes what the
    gate means, so they travel together and are written into the report.
    """

    #: Destroys a hit with p = 0.9373 (one substitution gives 0.7992) while
    #: moving the 4-mer spectrum by 2.7 % of the distance between two real
    #: promoters -- destructive for a motif, invisible to composition.
    small_substitutions: int = 2
    #: 87 is where the 4-mer L1 reaches 50 % of that natural distance;
    #: rounded up so it divides by the 4 classes.
    large_substitutions: int = 88
    #: Going from 5 to 20 per locus cuts the component's sd by 5 % for 4x the
    #: work, because variants on one locus are not independent. More
    #: resolution needs more promoters, not more variants.
    per_locus: int = 5
    #: Loci come from the enriched profiles at the loose threshold; at 1e-4
    #: pks1 yields only 3 loci, too few for any test.
    p_in_motif: float = 1e-3
    #: "Outside" means outside every profile at the loose threshold.
    p_outside: float = 1e-3
    #: The threshold at which "destroyed / created a hit" is counted.
    p_strict: float = 1e-4
    seed: int = DEFAULT_SEED
    kmer_k: int = 4
    enriched: Tuple[str, ...] = ENRICHED_MOTIFS
    #: Cap on rejection sampling; exceeding it means the design is infeasible
    #: on this target, which is a result rather than an error.
    max_tries: int = 500000
    prefix: str = "pks1"
    #: Background for the per-column IC that ranks which columns to hit. Not
    #: the p-value null, which stays measured. Uniform is the default because
    #: the measured background pushes the top-2 columns up to 6 bp apart, and
    #: the free pool (136 bp in blocks of median 5.5) can then rarely match
    #: the span: 71 variants over 15 loci at acceptance 0.0739, against 76
    #: over 16 at 0.6230. `None` falls back to the scanner's background.
    ic_background: Optional[Tuple[float, float, float, float]] = \
        UNIFORM_BACKGROUND


@dataclass
class VariantSet:
    """The wild type plus its variants, with the design's own diagnostics."""

    wild_type: str
    variants: List[Variant]
    config: MutationConfig = field(default_factory=MutationConfig)
    design_notes: Dict[str, object] = field(default_factory=dict)

    @property
    def wild_type_name(self) -> str:
        return "%s_WT" % self.config.prefix

    def __len__(self) -> int:
        return len(self.variants)

    def by_class(self, variant_class: str) -> List[Variant]:
        return [v for v in self.variants if v.variant_class == variant_class]

    def loci(self, variant_class: str) -> List[str]:
        return [v.locus for v in self.by_class(variant_class)]

    def names(self) -> List[str]:
        return [self.wild_type_name] + [v.name for v in self.variants]

    # I/O
    def to_fasta(self, path: str, line_width: int = 70) -> str:
        """Write WT first, then the variants, with provenance in the header."""
        def wrap(s: str) -> str:
            if line_width <= 0:
                return s
            return "\n".join(s[i:i + line_width]
                             for i in range(0, len(s), line_width))

        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(">%s|nsub=0|locus=-|motif=-|pos=\n%s\n"
                     % (self.wild_type_name, wrap(self.wild_type)))
            for v in self.variants:
                fh.write(">%s\n%s\n" % (v.header(), wrap(v.sequence)))
        return path

    @classmethod
    def from_fasta(cls, path: str, config: Optional[MutationConfig] = None
                   ) -> "VariantSet":
        """Read a variant FASTA back, accepting the legacy Polish class names.

        The recorded artefact is what goes to the cluster, and re-deriving the
        set under a different RNG call order would not be the same file, so
        the falsifiers have to be runnable on the file and not only on a fresh
        design.
        """
        cfg = config or MutationConfig()
        wt: Optional[str] = None
        variants: List[Variant] = []
        for header, seq in _read_fasta(path):
            meta = parse_variant_header(header)
            if meta["variant_class"] == "WT":
                wt = seq.upper()
                continue
            variants.append(Variant(
                name=meta["name"], variant_class=str(meta["variant_class"]),
                sequence=seq.upper(),
                positions=tuple(meta["positions"]),           # type: ignore
                locus=str(meta["locus"]), motif=str(meta["motif"])))
        if wt is None:
            raise ValueError("no wild-type record (*_WT) in %s" % path)
        return cls(wild_type=wt, variants=variants, config=cfg,
                   design_notes={"source": path})

    # Measures
    def summary(self, scanner: "PwmScanner",
                enriched_index: Optional[Sequence[int]] = None
                ) -> Dict[str, Dict[str, float]]:
        """Per-class table: destroyed / created hits, k-mer L1, span, GC.

        Every variant is rescanned rather than trusting what the design loop
        recorded: a set written by the same code that filtered it is not
        evidence about that code.
        """
        cfg = self.config
        if enriched_index is None:
            present, _ = scanner.known(cfg.enriched)
            enriched_index = scanner.indices(present)
        wt = encode_sequence(self.wild_type)
        wt_hits = scanner.scan_many(wt, [cfg.p_in_motif, cfg.p_strict])
        wt_enr = scanner.scan(wt, cfg.p_in_motif, subset=enriched_index)
        wt_strict = wt_hits[cfg.p_strict]

        out: Dict[str, Dict[str, float]] = {}
        for cname in VARIANT_CLASSES:
            vs = self.by_class(cname)
            if not vs:
                continue
            d_enr, d_strict, created, l1, spans, gcs, hams = (
                [], [], [], [], [], [], [])
            for v in vs:
                c = v.codes
                h = scanner.scan_many(c, [cfg.p_in_motif, cfg.p_strict])
                he = scanner.scan(c, cfg.p_in_motif, subset=enriched_index)
                d_enr.append(len(wt_enr - he))
                d_strict.append(len(wt_strict - h[cfg.p_strict]))
                created.append(len(h[cfg.p_strict] - wt_strict))
                l1.append(kmer_l1(c, wt, cfg.kmer_k))
                spans.append(max(v.positions) - min(v.positions) + 1
                             if v.positions else 0)
                gcs.append(gc_fraction(c))
                hams.append(hamming(c, wt))
            nat = NATURAL_KMER_L1.get(cfg.kmer_k, (float("nan"),))[0]
            out[cname] = {
                "n": float(len(vs)),
                "loci": float(len(set(v.locus for v in vs))),
                "substitutions": float(np.mean([v.n_substitutions for v in vs])),
                "hamming_mean": float(np.mean(hams)),
                "destroyed_enriched": float(np.mean(d_enr)),
                "destroyed_strict": float(np.mean(d_strict)),
                "created_strict": float(np.mean(created)),
                "kmer_l1": float(np.mean(l1)),
                "percent_natural": float(100.0 * np.mean(l1) / nat),
                "span_mean": float(np.mean(spans)),
                "gc": float(np.mean(gcs)),
                "min_destroyed_enriched": float(np.min(d_enr)),
                "max_destroyed_strict": float(np.max(d_strict)),
            }
        return out


def parse_variant_header(header: str) -> Dict[str, object]:
    """Split `name|nsub=..|locus=..|motif=..|pos=..` into fields.

    The class is recovered from the name, in either the English or the legacy
    Polish spelling; the artefact already on disk uses the latter.
    """
    parts = header.split("|")
    name = parts[0].strip()
    meta: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k.strip()] = v.strip()
    positions = tuple(int(x) for x in meta.get("pos", "").split(",")
                      if x.strip())
    cls = "WT" if name.endswith("_WT") else ""
    if not cls:
        for legacy, english in LEGACY_CLASS_NAMES.items():
            if legacy in name:
                cls = english
                break
    if not cls:
        for english in VARIANT_CLASSES:
            if english in name:
                cls = english
                break
    if not cls:
        raise ValueError(
            "cannot tell the class of %r; known: %s (+ legacy %s)"
            % (name, list(VARIANT_CLASSES), list(LEGACY_CLASS_NAMES)))
    return {
        "name": name,
        "variant_class": cls,
        "n_substitutions": int(meta.get("nsub", len(positions))),
        "locus": meta.get("locus", "-"),
        "motif": meta.get("motif", "-"),
        "positions": positions,
    }


def _read_fasta(path: str) -> List[Tuple[str, str]]:
    """Local FASTA reader: headers keep everything after '>'."""
    out: List[Tuple[str, str]] = []
    name: Optional[str] = None
    buf: List[str] = []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name is not None:
        out.append((name, "".join(buf)))
    return out


# Design
def design_variants(sequence: str, scanner: PwmScanner,
                    config: Optional[MutationConfig] = None) -> VariantSet:
    """Build the four matched classes on one target sequence.

    Small / in motif substitutes the two highest-IC columns of each enriched
    locus's widest hit, `per_locus` times, keeping only draws that destroy an
    enriched hit. Small / outside puts two substitutions in the free pool with
    the span drawn from the class above, rejected unless the strict hit set is
    unchanged and no loose hit is lost. Large / in motif draws 88
    substitutions from a random subset of the enriched-covered positions the
    size of the free pool, so density matches on both sides; large / outside
    draws 88 in the free pool and rejects any loss of a strict hit.

    Matching the span, not only the count, is what makes the small classes
    comparable: substitutions inside a motif are clustered and destroy
    overlapping k-mer windows, so matching counts alone gave a 4-mer L1 ratio
    of 0.691 and any |delta| difference would have been composition rather
    than motif. With span matched it is 0.997.

    `design_notes` carry pool sizes, acceptance rates and the IC-background
    disagreement -- what tells "infeasible on this target" from "wrong code".
    """
    cfg = config or MutationConfig()
    seq = sequence.upper()
    codes = encode_sequence(seq)
    L = int(codes.size)
    if (codes >= 4).any():
        raise ValueError(
            "target contains %d non-ACGT positions; substitution positions "
            "and k-mer spectra are defined on ACGT only"
            % int((codes >= 4).sum()))

    present, missing = scanner.known(cfg.enriched)
    enriched_idx = scanner.indices(present)
    ic_bg = cfg.ic_background or tuple(float(v) for v in scanner.background)
    # The other null for column ranking, so the design can report how much
    # that choice moved it instead of leaving it a hidden knob.
    alt_bg = (tuple(float(v) for v in scanner.background)
              if tuple(round(float(v), 6) for v in ic_bg)
              == tuple(round(float(v), 6) for v in UNIFORM_BACKGROUND)
              else UNIFORM_BACKGROUND)

    hits = scanner.scan_many(codes, [cfg.p_in_motif, cfg.p_strict])
    wt_loose = hits[cfg.p_in_motif]
    wt_strict = hits[cfg.p_strict]
    wt_enriched = {h for h in wt_loose if h.motif_index in set(enriched_idx)}

    cov_enriched = PwmScanner.coverage(wt_enriched, L)
    cov_all_loose = PwmScanner.coverage(wt_loose, L)
    free = np.nonzero(~cov_all_loose)[0]
    covered = np.nonzero(cov_enriched)[0]
    loci = blocks_from_mask(cov_enriched)
    free_blocks = blocks_from_mask(~cov_all_loose)

    if free.size < cfg.large_substitutions:
        raise ValueError(
            "free pool is %d bp but a large mutation needs %d positions; the "
            "target admits no outside class at p<%g"
            % (free.size, cfg.large_substitutions, cfg.p_outside))

    rng = np.random.default_rng(cfg.seed)
    notes: Dict[str, object] = {
        "length": L,
        "enriched_present": len(present),
        "enriched_missing": missing,
        "loci_in_motif": len(loci),
        "bp_in_motif": int(cov_enriched.sum()),
        "free_bp": int(free.size),
        "free_blocks": len(free_blocks),
        "free_block_median": float(np.median([b - a for a, b in free_blocks]))
        if free_blocks else 0.0,
        "wt_hits_loose": len(wt_loose),
        "wt_hits_strict": len(wt_strict),
        "wt_hits_enriched": len(wt_enriched),
        "ic_background": list(ic_bg),
        "ic_background_alternative": list(alt_bg),
    }

    variants: List[Variant] = []

    # Small / in motif
    spans: List[int] = []
    ic_disagree = 0
    for (a, b) in loci:
        cand = [h for h in wt_enriched if a <= h.start < b]
        if not cand:
            continue
        hit = max(cand, key=lambda h: (h.width, h.start, h.strand))
        m = scanner.motifs[hit.motif_index]
        ic = m.position_information(ic_bg, hit.strand)
        ic_alt = m.position_information(alt_bg, hit.strand)
        top = np.sort(np.argsort(-ic)[:cfg.small_substitutions])
        top_alt = np.sort(np.argsort(-ic_alt)[:cfg.small_substitutions])
        if not np.array_equal(top, top_alt):
            ic_disagree += 1
        pos = np.sort(hit.start + top)
        pos = pos[pos < L]
        if pos.size < cfg.small_substitutions:
            continue
        for _ in range(cfg.per_locus):
            v = substitute(codes, pos.tolist(), rng)
            he = scanner.scan(v, cfg.p_in_motif, subset=enriched_idx)
            if len(wt_enriched - he) < 1:
                continue
            variants.append(Variant(
                name="%s_%s_%03d" % (cfg.prefix, SMALL_IN_MOTIF,
                                     len([x for x in variants
                                          if x.variant_class == SMALL_IN_MOTIF])),
                variant_class=SMALL_IN_MOTIF, sequence=decode_sequence(v),
                positions=tuple(int(p) for p in pos),
                locus="locus%d-%d" % (a, b), motif=hit.motif_name))
            spans.append(int(pos[-1] - pos[0] + 1))
    notes["ic_background_disagreement_loci"] = ic_disagree
    target_n = len([v for v in variants if v.variant_class == SMALL_IN_MOTIF])
    if target_n == 0:
        raise ValueError(
            "no small in-motif variant destroyed an enriched hit; with "
            "targeted substitution that is a broken scan, not a hard target")

    # Small / outside
    accepted, tries = 0, 0
    while accepted < target_n and tries < cfg.max_tries:
        tries += 1
        want = spans[accepted % len(spans)]
        anchor = int(free[rng.integers(free.size)])
        window = free[(free >= anchor) & (free <= anchor + want - 1)]
        if window.size < cfg.small_substitutions:
            continue
        pos = np.sort(rng.choice(window, size=cfg.small_substitutions,
                                 replace=False))
        if int(pos[-1] - pos[0] + 1) != want:
            continue
        v = substitute(codes, pos.tolist(), rng)
        h = scanner.scan_many(v, [cfg.p_in_motif, cfg.p_strict])
        if h[cfg.p_strict] != wt_strict:
            continue
        if len(wt_loose - h[cfg.p_in_motif]) != 0:
            continue
        variants.append(Variant(
            name="%s_%s_%03d" % (cfg.prefix, SMALL_OUTSIDE, accepted),
            variant_class=SMALL_OUTSIDE, sequence=decode_sequence(v),
            positions=tuple(int(p) for p in pos),
            locus="free%d" % int(pos[0]), motif="-"))
        accepted += 1
    notes["small_outside_tries"] = tries
    notes["small_outside_acceptance"] = (float(accepted) / tries) if tries else 0.0
    if accepted < target_n:
        notes["small_outside_short_by"] = target_n - accepted

    # Large / in motif
    subset_size = int(min(free.size, covered.size))
    if covered.size < cfg.large_substitutions:
        raise ValueError(
            "enriched coverage is %d bp, below the %d substitutions of a "
            "large mutation" % (covered.size, cfg.large_substitutions))
    for i in range(target_n):
        sub = rng.choice(covered, size=subset_size, replace=False)
        pos = np.sort(rng.choice(sub, size=cfg.large_substitutions,
                                 replace=False))
        v = substitute(codes, pos.tolist(), rng)
        variants.append(Variant(
            name="%s_%s_%03d" % (cfg.prefix, LARGE_IN_MOTIF, i),
            variant_class=LARGE_IN_MOTIF, sequence=decode_sequence(v),
            positions=tuple(int(p) for p in pos),
            locus="enriched_pool", motif="-"))
    notes["large_density_in_motif"] = cfg.large_substitutions / float(subset_size)
    notes["large_density_outside"] = cfg.large_substitutions / float(free.size)

    # Large / outside
    accepted, tries = 0, 0
    while accepted < target_n and tries < cfg.max_tries:
        tries += 1
        pos = np.sort(rng.choice(free, size=cfg.large_substitutions,
                                 replace=False))
        v = substitute(codes, pos.tolist(), rng)
        if len(wt_strict - scanner.scan(v, cfg.p_strict)) != 0:
            continue
        variants.append(Variant(
            name="%s_%s_%03d" % (cfg.prefix, LARGE_OUTSIDE, accepted),
            variant_class=LARGE_OUTSIDE, sequence=decode_sequence(v),
            positions=tuple(int(p) for p in pos),
            locus="free_pool", motif="-"))
        accepted += 1
    notes["large_outside_tries"] = tries
    notes["large_outside_acceptance"] = (float(accepted) / tries) if tries else 0.0
    if accepted < target_n:
        notes["large_outside_short_by"] = target_n - accepted

    return VariantSet(wild_type=seq, variants=variants, config=cfg,
                      design_notes=notes)


# Selection arithmetic
def expected_max_of_m(sd: float, m: int, grid: int = 60001,
                      span: float = 12.0) -> float:
    """E[max of m draws from N(0, sd)] -- the bias model selection adds.

    Picking the best of m runs on one split measures the luckiest draw, not
    the best model. At sd = 0.0073 the expected inflation is 0.0041 at m=2 and
    0.0113 at m=10, so an effect of 0.010 is inside the selection bias by the
    tenth comparison. Numerical integration of m * x * phi(x) * Phi(x)^(m-1).
    """
    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")
    if m == 1:
        return 0.0
    x = np.linspace(-span, span, int(grid))
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf = np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0)))
                    for v in x])
    y = m * x * phi * np.power(cdf, m - 1)
    return float(sd) * float(np.trapezoid(y, x)
                             if hasattr(np, "trapezoid") else np.trapz(y, x))


def selection_budget(sd: float, effect: float, max_m: int = 1000
                     ) -> Dict[str, object]:
    """How many models may be compared before selection eats `effect`.

    Two conventions, reported separately because they differ by one step:
    `largest_m_below` is the last m whose expected bias stays under the
    effect, `first_m_at_or_above` the m at which it reaches it. The recorded
    budget of 10 models at sd = 0.0073 / effect = 0.010 is the second.
    """
    last_below, first_at = 1, None
    for m in range(2, int(max_m) + 1):
        e = expected_max_of_m(sd, m)
        if e < effect:
            last_below = m
        else:
            first_at = m
            break
    return {
        "sd": float(sd),
        "effect": float(effect),
        "largest_m_below": int(last_below),
        "first_m_at_or_above": first_at,
        "bias_at_2": expected_max_of_m(sd, 2),
        "bias_at_10": expected_max_of_m(sd, 10),
    }


def mutation_weight_ceiling(sd_mutation: float = 0.1346,
                            sd_scc_difference: float = 0.0073
                            ) -> Dict[str, float]:
    """Largest weight a mutation term could carry without dominating SCC.

    The arithmetic behind "gate, not a summed term": above 0.054 the metric's
    noise becomes mutational rather than predictive, and below it the term
    cannot change any ranking. At c = 0.10 the metric's sd is 2.1x the SCC's
    own and the comparison budget falls from 10 models to 3.
    """
    c = float(sd_scc_difference) / float(sd_mutation)
    sd_at_c10 = math.sqrt(sd_scc_difference ** 2 + (0.10 * sd_mutation) ** 2)
    return {
        "sd_mutation": float(sd_mutation),
        "sd_scc_difference": float(sd_scc_difference),
        "weight_ceiling": c,
        "sd_metric_at_weight_0.10": sd_at_c10,
        "inflation_at_weight_0.10": sd_at_c10 / float(sd_scc_difference),
    }


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """SE of an AUC under independent samples -- the wrong model here.

    Kept so the cost of ignoring the clustering can be quoted at the same n as
    the clustered estimate instead of being inherited from a table: on the
    delivered design it returns 0.0463 where the true sampling sd is 0.0788.
    Hanley & McNeil, Q1 = A/(2-A), Q2 = 2A^2/(1+A).
    """
    a = float(auc)
    if not (0.0 <= a <= 1.0) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (float(n_pos) * float(n_neg))
    return math.sqrt(max(var, 0.0))


def locus_clustered_auc(values_pos: Sequence[float], loci_pos: Sequence[str],
                        values_neg: Sequence[float], loci_neg: Sequence[str],
                        n_boot: int = 2000, seed: int = DEFAULT_SEED
                        ) -> Dict[str, object]:
    """AUC with an error bar that respects the clustering by locus.

    The unit of independence is the locus, not the variant: five variants on
    one site share whatever that site does. The point estimate uses every
    variant, with ties on averaged ranks so a constant predictor scores
    exactly 0.500; the error bar resamples loci with replacement inside each
    class. Both SEs are returned, and their ratio says how much a naive
    report would have overstated the significance.
    """
    from ..heads.discriminator import roc_auc  # noqa: WPS433

    vp = np.asarray(values_pos, dtype=np.float64).ravel()
    vn = np.asarray(values_neg, dtype=np.float64).ravel()
    lp = [str(x) for x in loci_pos]
    ln = [str(x) for x in loci_neg]
    if vp.size != len(lp) or vn.size != len(ln):
        raise ValueError("values and loci lengths differ")
    if vp.size == 0 or vn.size == 0:
        return {"auc": float("nan"), "n_pos": int(vp.size),
                "n_neg": int(vn.size), "se_cluster": float("nan"),
                "se_independent": float("nan"), "se_ratio": float("nan"),
                "n_loci_pos": 0, "n_loci_neg": 0, "n_boot": 0}

    scores = np.concatenate([vp, vn])
    labels = np.concatenate([np.ones(vp.size), np.zeros(vn.size)])
    auc = float(roc_auc(scores, labels))

    groups_p: Dict[str, List[float]] = {}
    for v, g in zip(vp, lp):
        groups_p.setdefault(g, []).append(float(v))
    groups_n: Dict[str, List[float]] = {}
    for v, g in zip(vn, ln):
        groups_n.setdefault(g, []).append(float(v))
    keys_p, keys_n = list(groups_p), list(groups_n)

    rng = np.random.default_rng(seed)
    boots: List[float] = []
    for _ in range(int(n_boot)):
        sp = [x for k in rng.choice(len(keys_p), len(keys_p))
              for x in groups_p[keys_p[int(k)]]]
        sn = [x for k in rng.choice(len(keys_n), len(keys_n))
              for x in groups_n[keys_n[int(k)]]]
        s = np.asarray(sp + sn, dtype=np.float64)
        y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
        a = roc_auc(s, y)
        if np.isfinite(a):
            boots.append(float(a))
    se_cluster = float(np.std(boots, ddof=1)) if len(boots) > 2 else float("nan")
    se_ind = hanley_mcneil_se(auc, int(vp.size), int(vn.size))
    return {
        "auc": auc,
        "n_pos": int(vp.size), "n_neg": int(vn.size),
        "n_loci_pos": len(keys_p), "n_loci_neg": len(keys_n),
        "se_cluster": se_cluster,
        "se_independent": se_ind,
        "se_ratio": (se_cluster / se_ind) if se_ind > 0 else float("nan"),
        "n_boot": len(boots),
    }


# The gates
def selection_gates(vset: VariantSet, predictions: Mapping[str, float],
                    wild_type_repeats: Sequence[float] = (),
                    auc_threshold: float = B1_AUC_THRESHOLD,
                    n_boot: int = 2000, seed: int = DEFAULT_SEED
                    ) -> Dict[str, object]:
    """B1 / B2 / B3 plus the reading that keeps them interpretable.

    `predictions` maps every variant name, and the wild-type name, to one
    number -- the quantity being ranked, e.g. mu for P1. Deltas are taken
    against the wild type in the same mapping.

    B1 is specificity: AUC(|delta| in-motif against outside, small class) at
    or above 0.64. B2 is dose: mean |delta| larger for the large class in both
    motif splits, which any model reading composition must show, since the
    4-mer L1 differs 24.6x between the doses. B3 is determinism: the wild-type
    prediction identical bit for bit across runs, reported as failed when no
    repeat was given, because a gate that cannot fall is not a gate. G2 goes
    alongside: |delta| non-zero for most outside variants, since all-zero
    means the substitution never reached the input.
    """
    wt_name = vset.wild_type_name
    if wt_name not in predictions:
        raise KeyError("no prediction for the wild type %r" % wt_name)
    wt = float(predictions[wt_name])

    def deltas(cname: str) -> Tuple[List[float], List[str]]:
        vals, loci = [], []
        for v in vset.by_class(cname):
            if v.name in predictions:
                vals.append(abs(float(predictions[v.name]) - wt))
                loci.append(v.locus)
        return vals, loci

    small_in, loci_in = deltas(SMALL_IN_MOTIF)
    small_out, loci_out = deltas(SMALL_OUTSIDE)
    large_in, _ = deltas(LARGE_IN_MOTIF)
    large_out, _ = deltas(LARGE_OUTSIDE)

    b1 = locus_clustered_auc(small_in, loci_in, small_out, loci_out,
                             n_boot=n_boot, seed=seed)
    b1_pass = bool(np.isfinite(b1["auc"])                     # type: ignore
                   and float(b1["auc"]) >= auc_threshold)     # type: ignore

    def mean(xs: Sequence[float]) -> float:
        return float(np.mean(xs)) if len(xs) else float("nan")

    dose_in = mean(large_in) > mean(small_in)
    dose_out = mean(large_out) > mean(small_out)
    b2_pass = bool(dose_in and dose_out)

    wt_reps = [float(x) for x in wild_type_repeats]
    b3_pass = bool(wt_reps) and all(r == wt for r in wt_reps)

    nonzero = float(np.mean([1.0 if d > 0 else 0.0 for d in small_out])) \
        if small_out else float("nan")
    g2_pass = bool(np.isfinite(nonzero) and nonzero > 0.5)

    return {
        "wild_type": wt,
        "B1_motif_specificity": {
            "passed": b1_pass, "threshold": float(auc_threshold), **b1},
        "B2_dose_response": {
            "passed": b2_pass,
            "mean_abs_delta": {
                SMALL_IN_MOTIF: mean(small_in), LARGE_IN_MOTIF: mean(large_in),
                SMALL_OUTSIDE: mean(small_out),
                LARGE_OUTSIDE: mean(large_out)},
            "in_motif_ordered": bool(dose_in),
            "outside_ordered": bool(dose_out)},
        "B3_wild_type_determinism": {
            "passed": b3_pass, "repeats": wt_reps,
            "measured": bool(wt_reps)},
        "G2_intervention_alive": {
            "passed": g2_pass, "nonzero_fraction_outside": nonzero,
            "n": len(small_out)},
        "all_gates_passed": bool(b1_pass and b2_pass and b3_pass and g2_pass),
    }


def gate_falsifiers(gates: Mapping[str, object]
                    ) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail) for the gate readings of one model."""
    b1 = dict(gates["B1_motif_specificity"])                  # type: ignore
    b2 = dict(gates["B2_dose_response"])                      # type: ignore
    b3 = dict(gates["B3_wild_type_determinism"])              # type: ignore
    g2 = dict(gates["G2_intervention_alive"])                 # type: ignore
    return [
        ("B1 |delta| separates in-motif from outside (small)",
         bool(b1["passed"]),
         "AUC %.4f (>= %.2f) on %d vs %d variants, %d vs %d loci; "
         "SE cluster %.4f against independent %.4f (ratio %.2f)"
         % (b1["auc"], b1["threshold"], b1["n_pos"], b1["n_neg"],
            b1["n_loci_pos"], b1["n_loci_neg"], b1["se_cluster"],
            b1["se_independent"], b1["se_ratio"])),
        ("B2 large mutations move the prediction more than small ones",
         bool(b2["passed"]),
         "in motif %s, outside %s; means %s"
         % (b2["in_motif_ordered"], b2["outside_ordered"],
            {k: round(v, 6) for k, v in
             dict(b2["mean_abs_delta"]).items()})),      # type: ignore
        ("B3 wild-type prediction is bit-for-bit reproducible",
         bool(b3["passed"]),
         "measured=%s repeats=%s -- an unmeasured determinism gate is "
         "reported as FAILED, because B1 and B2 are meaningless without it"
         % (b3["measured"], b3["repeats"])),
        ("G2 the intervention reaches the model (outside |delta| > 0)",
         bool(g2["passed"]),
         "non-zero on %.4f of %d outside variants; exactly 0 everywhere "
         "means a broken intervention, not a specific model"
         % (g2["nonzero_fraction_outside"], g2["n"])),
    ]


# Falsifiers
def falsifiers(vset: VariantSet, scanner: PwmScanner,
               summary: Optional[Mapping[str, Mapping[str, float]]] = None
               ) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail) for the variant set itself.

    These are properties of the designed sequences and need no model. F3 is
    the one that forced the span matching, and it is the reason a |delta|
    difference between the small classes can be read as motif specificity at
    all.
    """
    cfg = vset.config
    summ = dict(summary) if summary is not None else vset.summary(scanner)
    wt = encode_sequence(vset.wild_type)
    L = int(wt.size)
    present, missing = scanner.known(cfg.enriched)

    def get(cname: str, key: str, default: float = float("nan")) -> float:
        d = summ.get(cname)
        return float(d[key]) if d and key in d else default

    out_classes = [SMALL_OUTSIDE, LARGE_OUTSIDE]
    f1 = max([get(c, "max_destroyed_strict", 0.0) for c in out_classes
              if c in summ] or [float("nan")])
    in_classes = [SMALL_IN_MOTIF, LARGE_IN_MOTIF]
    f2 = min([get(c, "min_destroyed_enriched", 0.0) for c in in_classes
              if c in summ] or [float("nan")])

    r_small = (get(SMALL_IN_MOTIF, "kmer_l1")
               / get(SMALL_OUTSIDE, "kmer_l1")) \
        if get(SMALL_OUTSIDE, "kmer_l1") else float("nan")
    r_large = (get(LARGE_IN_MOTIF, "kmer_l1")
               / get(LARGE_OUTSIDE, "kmer_l1")) \
        if get(LARGE_OUTSIDE, "kmer_l1") else float("nan")
    sep = (get(LARGE_IN_MOTIF, "kmer_l1") / get(SMALL_IN_MOTIF, "kmer_l1")) \
        if get(SMALL_IN_MOTIF, "kmer_l1") else float("nan")

    bad_len = [v.name for v in vset.variants if len(v.sequence) != L]
    alphabet = set(NUCLEOTIDES)
    bad_alpha = [v.name for v in vset.variants
                 if set(v.sequence) - alphabet]
    bad_ham = [(v.name, hamming(v.codes, wt), v.n_substitutions)
               for v in vset.variants
               if hamming(v.codes, wt) != v.n_substitutions]

    span_small = (get(SMALL_IN_MOTIF, "span_mean")
                  / get(SMALL_OUTSIDE, "span_mean")) \
        if get(SMALL_OUTSIDE, "span_mean") else float("nan")
    span_large = (get(LARGE_IN_MOTIF, "span_mean")
                  / get(LARGE_OUTSIDE, "span_mean")) \
        if get(LARGE_OUTSIDE, "span_mean") else float("nan")

    counts = {c: int(get(c, "n", 0)) for c in VARIANT_CLASSES if c in summ}
    balanced = len(set(counts.values())) <= 1 and bool(counts)

    return [
        ("F1 no OUTSIDE variant destroys a hit at p<%g" % cfg.p_strict,
         bool(f1 == 0),
         "max destroyed over %s = %g (the outside class is the clean arm of "
         "B1; a non-zero here means the definition of 'outside' leaks)"
         % (out_classes, f1)),
        ("F2 every IN-MOTIF variant destroys >= 1 enriched hit at p<%g"
         % cfg.p_in_motif,
         bool(f2 >= 1),
         "min destroyed over %s = %g" % (in_classes, f2)),
        ("F3 4-mer L1 matched between classes (0.80-1.25)",
         bool(0.80 <= r_small <= 1.25 and 0.80 <= r_large <= 1.25),
         "small in/out %.3f, large in/out %.3f -- this failed at 0.691 "
         "before span matching, and any |delta| difference would then have "
         "been composition, not motif" % (r_small, r_large)),
        ("F4 large is separated from small on composition",
         bool(np.isfinite(sep) and sep >= 5.0),
         "L1 ratio large/small = %.1fx (B2 needs the two doses to be "
         "distinguishable at all)" % sep),
        ("F5 every variant is %d bp over the ACGT alphabet" % L,
         bool(not bad_len and not bad_alpha),
         "%d wrong length, %d with a non-ACGT character, %d variants checked"
         % (len(bad_len), len(bad_alpha), len(vset.variants))),
        ("F6 Hamming distance equals the declared substitution count",
         bool(not bad_ham),
         "%d mismatches%s -- a substitution that redrew the same base would "
         "make nsub a declaration instead of a measurement"
         % (len(bad_ham), (" e.g. %s" % bad_ham[:3]) if bad_ham else "")),
        ("F7 every enriched profile name resolves in this bank",
         bool(not missing),
         "%d of %d present; missing %s -- a renamed profile silently shrinks "
         "the definition of 'in motif'"
         % (len(present), len(cfg.enriched), missing)),
        ("F8 the four classes have equal counts",
         balanced, "counts %s" % counts),
        ("F9 span matched between classes (REPORTED, weak on the large ones)",
         bool(0.80 <= span_small <= 1.25 and 0.80 <= span_large <= 1.25),
         "small %.3f, large %.3f -- span is a poor descriptor of 88 scattered "
         "substitutions drawn from pools of different reach; the quantity "
         "that has to match is the 4-mer L1 (F3). The analysis layer recorded "
         "this as failing at 0.762 (608.8 against 798.9), but those spans are "
         "from the 25-per-class iteration `variants_v2.fasta`; measured on "
         "the delivered 76-per-class set the means are 710.0 and 799.1, i.e. "
         "0.888. Reported, not repaired." % (span_small, span_large)),
    ]


def report(vset: VariantSet, scanner: PwmScanner,
           validate_dp_on: Optional[str] = "HAA1") -> Dict[str, object]:
    """Everything a run needs in its log about this variant set.

    Includes the DP control when a narrow enough profile is available;
    without it every "p < 1e-4" in this module would be self-certified.
    """
    cfg = vset.config
    summ = vset.summary(scanner)
    out: Dict[str, object] = {
        "config": {
            "small_substitutions": cfg.small_substitutions,
            "large_substitutions": cfg.large_substitutions,
            "per_locus": cfg.per_locus,
            "p_in_motif": cfg.p_in_motif,
            "p_outside": cfg.p_outside,
            "p_strict": cfg.p_strict,
            "seed": cfg.seed,
            "kmer_k": cfg.kmer_k,
            "n_enriched": len(cfg.enriched),
        },
        "design_notes": dict(vset.design_notes),
        "n_variants": len(vset.variants),
        "summary": summ,
        "bank": scanner.report([cfg.p_in_motif, cfg.p_strict]),
        "falsifiers": [
            {"name": n, "passed": p, "detail": d}
            for n, p, d in falsifiers(vset, scanner, summ)],
        "weight_arithmetic": mutation_weight_ceiling(),
        "selection_budget_rho_0.94": selection_budget(
            SCC_DIFF_SD_BY_RHO[0.94], 0.010),
        "natural_kmer_l1": NATURAL_KMER_L1.get(cfg.kmer_k),
    }
    if validate_dp_on and validate_dp_on in [m.name for m in scanner.motifs]:
        mi = scanner.index(validate_dp_on)
        if scanner.motifs[mi].width <= 8:
            out["dp_control"] = [
                scanner.validate_threshold_by_enumeration(mi, cfg.p_strict),
                scanner.validate_threshold_by_enumeration(mi, cfg.p_in_motif),
            ]
    return out
