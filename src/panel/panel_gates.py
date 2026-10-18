"""Safety and sanity gates. Every threshold is measured on real promoters."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import os as _os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import anchors as K  # noqa: E402

ROOT = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE = os.path.join(_HERE, "cache")
KMER_K = 25
BASES = "ACGT"


def _load_safety():
    """Load the project's uORF scanner by path.

    By path because `<ROOT>/hyppe` is the old package that shadows
    `src/hyppe`: an `import hyppe.safety` would either fail or succeed by
    importing the shadowing copy and dragging the rest of it along. A
    file-path import takes the one module and adds nothing to `sys.path`.
    """
    cand = [os.path.join(ROOT, "src", "panel", "safety.py"),
            os.path.join(ROOT, "hyppe", "safety.py")]
    path = next((c for c in cand if os.path.exists(c)), None)
    if path is None:
        raise SystemExit("REFUSED: no safety.py at any of %s. The uORF "
                         "scanner is not re-implemented here." % (cand,))
    spec = importlib.util.spec_from_file_location("_hyppe_safety", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("find_uorfs", "uorf_risk", "dinucleotide_shuffle"):
        if not hasattr(mod, name):
            raise SystemExit("REFUSED: %s has no %r" % (path, name))
    mod.__hyppe_path__ = path
    return mod


SAFETY = _load_safety()
SAFETY_PATH = SAFETY.__hyppe_path__


def gc(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def longest_homopolymer(seq: str) -> int:
    best = run = 1
    s = seq.upper()
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] else 1
        best = max(best, run)
    return best


def uorf_counts(seq: str, near_tss_bp: int = 100) -> Dict[str, float]:
    """ATGs upstream, how many lie near the 3' end, how many are contained."""
    recs = SAFETY.find_uorfs(seq.upper(), tss_at_end=True)
    n = len(seq)
    near = [r for r in recs if (n - r["pos"]) <= near_tss_bp]
    contained = [r for r in recs if r.get("terminated")]
    return {"atg": float(len(recs)),
            "atg_near_tss": float(len(near)),
            "uorf_contained": float(len(contained)),
            "longest_peptide_aa": float(max([r.get("peptide_len",
                                                   r.get("length", 0))
                                             for r in recs] or [0]))}


def atg_positions(seq: str) -> List[int]:
    """Start positions of every ATG. The cheap half of the uORF scan."""
    s, out, i = seq.upper(), [], 0
    while True:
        i = s.find("ATG", i)
        if i < 0 or i > len(s) - 3:
            break
        out.append(i)
        i += 1
    return out


def atg_counts_fast(seq: str, near_tss_bp: int = 100) -> Tuple[float, float]:
    """`(total ATG, ATG within near_tss_bp of the 3' end)`, without the ORFs."""
    n = len(seq)
    pos = atg_positions(seq)
    return float(len(pos)), float(sum(1 for p in pos
                                      if (n - p) <= near_tss_bp))


def _encode2bit(seq: str) -> Optional[np.ndarray]:
    lut = np.full(256, 255, dtype=np.uint8)
    for i, c in enumerate(BASES):
        lut[ord(c)] = i
    a = lut[np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)]
    if int((a == 255).sum()):
        return None                       # a non-ACGT base: no k-mer from it
    return a


def kmers(seq: str, k: int = KMER_K) -> np.ndarray:
    """Distinct k-mers of one sequence, packed 2 bits per base into uint64."""
    a = _encode2bit(seq)
    if a is None or a.size < k:
        return np.empty(0, dtype=np.uint64)
    v = a.astype(np.uint64)
    out = np.zeros(a.size - k + 1, dtype=np.uint64)
    for i in range(k):
        out = (out << np.uint64(2)) | v[i:i + out.size]
    return np.unique(out)


@dataclass
class KmerIndex:
    """Corpus k-mers with the number of distinct sequences carrying each one.

    The document count is what makes a leave-one-out reference possible: a
    real promoter's own k-mer has count >= 2 exactly when some other corpus
    sequence also has it.
    """
    k: int
    keys: np.ndarray                   # sorted uint64
    doc_count: np.ndarray              # int32, same length
    n_sequences: int
    members: Set[str] = field(default_factory=set)   # sha256 of each record

    @classmethod
    def build(cls, seqs: Sequence[str], k: int = KMER_K) -> "KmerIndex":
        parts = [kmers(s, k) for s in seqs]
        allk = (np.concatenate([p for p in parts if p.size])
                if any(p.size for p in parts)
                else np.empty(0, dtype=np.uint64))
        keys, counts = np.unique(allk, return_counts=True)
        return cls(k=k, keys=keys, doc_count=counts.astype(np.int32),
                   n_sequences=len(seqs),
                   members={hashlib.sha256(s.upper().encode()).hexdigest()
                            for s in seqs})

    def is_member(self, seq: str) -> bool:
        return hashlib.sha256(seq.upper().encode()).hexdigest() in self.members

    def containment(self, seq: str, min_docs: int = 1) -> float:
        """Fraction of the sequence's distinct k-mers present in the corpus.

        `min_docs=2` gives the leave-one-out figure for a sequence that is
        itself in the corpus: its own copy contributes one document, so
        requiring two means "some other sequence has it too".
        """
        q = kmers(seq, self.k)
        if q.size == 0 or self.keys.size == 0:
            return float("nan")
        pos = np.clip(np.searchsorted(self.keys, q), 0, self.keys.size - 1)
        hit = (self.keys[pos] == q) & (self.doc_count[pos] >= int(min_docs))
        return float(hit.mean())

    def containment_auto(self, seq: str) -> Tuple[float, str]:
        """The comparison the sequence actually needs, and which one it was."""
        if self.is_member(seq):
            return self.containment(seq, 2), "leave-one-out (corpus member)"
        return self.containment(seq, 1), "against the whole corpus"


@dataclass
class Reference:
    """Every band in this file, derived from real promoters. No constants."""
    genome: str
    n_sequences: int
    n_distinct_sequences: int
    n_novelty_sample: int
    gc_mean: float
    gc_sd: float
    gc_band: Tuple[float, float]
    gc_reject_frac: float
    homopolymer_median: float
    homopolymer_band_hi: int
    homopolymer_reject_frac: float
    atg_mean: float
    atg_near_tss_mean: float
    atg_near_tss_band_hi: float
    atg_reject_frac: float
    containment_quantiles: Dict[str, float]
    containment_zero_frac: float
    containment_threshold: float
    containment_threshold_quantile: float
    containment_reject_frac: float
    containment_sample: List[float]
    kmer_k: int
    source: str = ""

    @classmethod
    def measure(cls, genome: str = "P1", limit: int = 0,
                novelty_sample: int = 512, seed: int = 7898,
                kmer_k: int = KMER_K,
                containment_quantile: float = 0.95) -> "Reference":
        rows = K.corpus_promoters(genome, limit=limit)
        seqs = [s for _i, s in rows]
        g = np.array([gc(s) for s in seqs])
        h = np.array([longest_homopolymer(s) for s in seqs], dtype=float)
        _ac = [atg_counts_fast(s) for s in seqs]
        a = np.array([t for t, _n in _ac])
        u = np.array([n for _t, n in _ac])

        gc_lo, gc_hi = float(np.quantile(g, 0.01)), float(np.quantile(g, 0.99))
        gc_rej = float(((g < gc_lo) | (g > gc_hi)).mean())
        h_hi = int(np.quantile(h, 0.99))
        h_rej = float((h > h_hi).mean())
        u_hi = float(np.quantile(u, 0.99))
        u_rej = float((u > u_hi).mean())

        rng = np.random.default_rng(seed)
        pick = rng.choice(len(seqs), size=min(novelty_sample, len(seqs)),
                          replace=False)
        index = KmerIndex.build(seqs, k=kmer_k)
        loo = np.array([index.containment(seqs[int(i)], min_docs=2)
                        for i in pick])
        loo = loo[np.isfinite(loo)]
        q = float(containment_quantile)
        thr = float(np.quantile(loo, q)) if loo.size else float("nan")
        c_rej = float((loo > thr).mean()) if loo.size else float("nan")
        qs = ({("q%g" % (100 * p)): float(np.quantile(loo, p))
               for p in (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0)}
              if loo.size else {})

        return cls(genome=genome, n_sequences=len(seqs),
                   n_distinct_sequences=len(set(seqs)),
                   n_novelty_sample=int(loo.size),
                   gc_mean=float(g.mean()), gc_sd=float(g.std()),
                   gc_band=(gc_lo, gc_hi), gc_reject_frac=gc_rej,
                   homopolymer_median=float(np.median(h)),
                   homopolymer_band_hi=h_hi,
                   homopolymer_reject_frac=h_rej,
                   atg_mean=float(a.mean()),
                   atg_near_tss_mean=float(u.mean()),
                   atg_near_tss_band_hi=u_hi,
                   atg_reject_frac=u_rej,
                   containment_quantiles=qs,
                   containment_zero_frac=(float((loo == 0.0).mean())
                                          if loo.size else float("nan")),
                   containment_threshold=thr,
                   containment_threshold_quantile=q,
                   containment_reject_frac=c_rej,
                   containment_sample=[float(v) for v in loo],
                   kmer_k=int(kmer_k),
                   source="corpus %d %s promoters (%d distinct)"
                          % (len(seqs), genome, len(set(seqs))))

    @classmethod
    def load_or_measure(cls, path: str = "", **kw) -> "Reference":
        p = path or os.path.join(CACHE, "reference_%s_k%d.json"
                                 % (kw.get("genome", "P1"),
                                    kw.get("kmer_k", KMER_K)))
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            d["gc_band"] = tuple(d["gc_band"])
            return cls(**d)
        ref = cls.measure(**kw)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            d = ref.__dict__.copy()
            d["gc_band"] = list(d["gc_band"])
            json.dump(d, fh, indent=1)
        return ref

    def containment_percentile(self, value: float) -> float:
        """Where `value` sits in the real leave-one-out distribution, in %.

        The gate is yes/no; this is by how much, which is how an edit-based
        candidate should be read.
        """
        s = np.asarray(self.containment_sample, dtype=np.float64)
        if s.size == 0 or not np.isfinite(value):
            return float("nan")
        return float(100.0 * (s <= float(value)).mean())

    def describe(self) -> str:
        qs = self.containment_quantiles
        return "\n".join([
            "REFERENCE (%s)" % (self.source,),
            "  GC           mean %.4f sd %.4f | band [%.4f, %.4f] | rejects "
            "%.2f %% of real promoters"
            % (self.gc_mean, self.gc_sd, self.gc_band[0], self.gc_band[1],
               100 * self.gc_reject_frac),
            "  homopolymer  median %.0f | band <= %d | rejects %.2f %%"
            % (self.homopolymer_median, self.homopolymer_band_hi,
               100 * self.homopolymer_reject_frac),
            "  ATG          mean %.2f total, %.2f within 100 bp of the TSS | "
            "band <= %.0f near-TSS | rejects %.2f %%"
            % (self.atg_mean, self.atg_near_tss_mean,
               self.atg_near_tss_band_hi, 100 * self.atg_reject_frac),
            "  novelty      %d-mer containment of a real promoter against a "
            "corpus WITHOUT it, n=%d:" % (self.kmer_k, self.n_novelty_sample),
            "               exactly 0.0 for %.1f %% | %s"
            % (100 * self.containment_zero_frac,
               " ".join("%s %.4f" % (k, v) for k, v in sorted(
                   qs.items(), key=lambda kv: float(kv[0][1:])))),
            "               threshold %.4f (the q%g of that distribution) | "
            "rejects %.2f %% of real promoters -- that tail is real: %d of %d "
            "records are exact duplicates of another record"
            % (self.containment_threshold,
               100 * self.containment_threshold_quantile,
               100 * self.containment_reject_frac,
               self.n_sequences - self.n_distinct_sequences,
               self.n_sequences),
        ])


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    reference: str
    comment: str = ""

    def __str__(self) -> str:
        return ("  [%s] %-13s %10.4f   ref: %s%s"
                % ("PASS" if self.passed else "FAIL", self.name, self.value,
                   self.reference, ("  -- " + self.comment) if self.comment
                   else ""))


@dataclass
class SequenceGates:
    """The sequence-only gates, bound to one measured `Reference`."""
    reference: Reference
    index: Optional[KmerIndex] = None
    near_tss_bp: int = 100

    def ensure_index(self, genome: str = "P1", limit: int = 0) -> KmerIndex:
        if self.index is None:
            rows = K.corpus_promoters(genome, limit=limit)
            self.index = KmerIndex.build([s for _i, s in rows],
                                         k=self.reference.kmer_k)
        return self.index

    def __call__(self, seq: str) -> List[GateResult]:
        r = self.reference
        out: List[GateResult] = []
        v = gc(seq)
        out.append(GateResult(
            "GC", r.gc_band[0] <= v <= r.gc_band[1], v,
            "real %s: mean %.4f sd %.4f, band [q01, q99] = [%.4f, %.4f]"
            % (r.genome, r.gc_mean, r.gc_sd, r.gc_band[0], r.gc_band[1]),
            "this band rejects %.2f %% of REAL promoters; an absolute GC rule "
            "would reject far more (recorded: 86 %%)"
            % (100 * r.gc_reject_frac)))
        v = float(longest_homopolymer(seq))
        out.append(GateResult(
            "homopolymer", v <= r.homopolymer_band_hi, v,
            "real %s: median %.0f, q99 %d" % (r.genome, r.homopolymer_median,
                                              r.homopolymer_band_hi),
            "rejects %.2f %% of real promoters"
            % (100 * r.homopolymer_reject_frac)))
        u = uorf_counts(seq, self.near_tss_bp)
        out.append(GateResult(
            "uORF_near_TSS", u["atg_near_tss"] <= r.atg_near_tss_band_hi,
            u["atg_near_tss"],
            "real %s: mean %.2f ATG within %d bp of the TSS, q99 %.0f"
            % (r.genome, r.atg_near_tss_mean, self.near_tss_bp,
               r.atg_near_tss_band_hi),
            "FLAG, NOT A VERDICT: reading an ATG as a uORF requires the TSS "
            "to lie inside these 800 bp, which is UNRESOLVED in this project "
            "(this sequence carries %.0f ATG in total and opens %.0f "
            "contained uORFs, longest peptide %.0f aa)"
            % (u["atg"], u["uorf_contained"], u["longest_peptide_aa"])))
        if self.index is not None:
            v, mode = self.index.containment_auto(seq)
            pct = r.containment_percentile(v)
            out.append(GateResult(
                "novelty", v <= r.containment_threshold, v,
                "%d-mer containment, %s; real leave-one-out q%g = %.4f"
                % (r.kmer_k, mode, 100 * r.containment_threshold_quantile,
                   r.containment_threshold),
                "at the %.1f-th percentile of real promoters; the gate "
                "rejects %.2f %% of them. A candidate produced by EDITING a "
                "native promoter is EXPECTED to score high here -- for those "
                "this measures how much of the native survived and is not "
                "grounds for rejection"
                % (pct, 100 * r.containment_reject_frac)))
        return out


@dataclass
class RealnessReference:
    """The discriminator gate, and the measurement that decides its floor."""
    construct: str
    n_real: int
    real_mean: float
    real_sd: float
    real_min: float
    real_q01: float
    pks1: float
    pks1_shuffled_mean: float
    pks1_minus_shuffle: float
    floor: float
    pks1_would_pass: bool

    def describe(self) -> str:
        return "\n".join([
            "REALNESS (discriminator logit, construct %s)" % (self.construct,),
            "  %d real P1 promoters: mean %+.4f sd %.4f min %+.4f q01 %+.4f"
            % (self.n_real, self.real_mean, self.real_sd, self.real_min,
               self.real_q01),
            "  pks1 (the design target) %+.4f | its dinucleotide shuffles "
            "%+.4f | difference %+.4f"
            % (self.pks1, self.pks1_shuffled_mean, self.pks1_minus_shuffle),
            "  floor %+.4f -> pks1 itself %s"
            % (self.floor, "PASSES" if self.pks1_would_pass
               else "WOULD BE REJECTED"),
            "  NOTE: an earlier measurement had this model rank pks1 1.67 "
            "BELOW its own shuffle. If `pks1_minus_shuffle` is negative here, "
            "the discriminator does not consider the design target a promoter "
            "and a hard floor would reject the very sequence the panel is "
            "replacing -- which is why the floor is set from the real "
            "distribution and reported together with whether pks1 clears it.",
        ])

    def gate(self, logit: float) -> GateResult:
        return GateResult(
            "realness", float(logit) >= self.floor, float(logit),
            "%d real promoters in %s: min %+.4f, floor %+.4f"
            % (self.n_real, self.construct, self.real_min, self.floor),
            ("pks1 itself scores %+.4f and %s this floor; a value far ABOVE "
             "the real mean %+.4f is also worth looking at -- that is usually "
             "a generator optimising its own critic"
             % (self.pks1, "clears" if self.pks1_would_pass else "FAILS",
                self.real_mean)))

_WYKLUCZONE_PLIK = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "cache", "wykluczone_test.txt")
_WYKLUCZONE = None


def wykluczone_test() -> frozenset:
    """Ids the gates must not sample; raises if the list is missing.

    Falling back to the unfiltered corpus would leave the gate looking
    identical while calibrating its threshold on held-out material.
    """
    global _WYKLUCZONE
    if _WYKLUCZONE is None:
        if not _os.path.isfile(_WYKLUCZONE_PLIK):
            raise SystemExit(
                "REFUSED: %s is missing. The gates calibrate their reference "
                "distributions on real promoters and must not draw them from "
                "TEST orthogroups.\n"
                "This file does not ship: it is a list of corpus record ids "
                "(one per line, '#' comments allowed), and the corpus itself "
                "is not part of this repository. Rebuild it where the corpus "
                "is available, by taking every id in the sequences table "
                "whose orthogroup is marked `test` in "
                "data/splits/splits.tsv (column H_kmer90).\n"
                "Refusing is deliberate: falling back to the unfiltered "
                "corpus would leave the gate looking identical while "
                "calibrating its threshold on held-out material."
                % _WYKLUCZONE_PLIK)
        with io.open(_WYKLUCZONE_PLIK, encoding="utf-8") as fh:
            _WYKLUCZONE = frozenset(
                ln.strip() for ln in fh
                if ln.strip() and not ln.startswith("#"))
    return _WYKLUCZONE


def _bez_testu(rows):
    """Drop records whose orthogroup is in `test`, and refuse an empty result."""
    zb = wykluczone_test()
    out = [r for r in rows if r[0] not in zb]
    if not out:
        raise SystemExit("REFUSED: excluding TEST left no promoters to "
                         "calibrate against")
    return out


def realness_reference(scorer, construct: str, n_real: int = 64,
                       n_shuffles: int = 8, seed: int = 7898,
                       genome: str = "P1") -> RealnessReference:
    """Measure the discriminator on real promoters, on pks1, and on shuffles.

    The floor is the minimum over the real sample, not their mean: the task
    reads "indistinguishable from real promoters", not "better than them".
    The shuffles are the negative control and preserve base and dinucleotide
    composition exactly (Euler walk), so only arrangement is destroyed; a
    discriminator that cannot separate pks1 from its own shuffle is reading
    composition.
    """
    rng = np.random.default_rng(seed)
    rows = _bez_testu(K.corpus_promoters(genome, limit=0))
    pick = rng.choice(len(rows), size=min(n_real, len(rows)), replace=False)
    seqs = [rows[int(i)][1] for i in pick]
    real = scorer.raw(seqs, construct)["realness"]

    anc = scorer.anchor_seqs
    shuffles = [SAFETY.dinucleotide_shuffle(anc.pks1, rng=rng)
                for _ in range(int(n_shuffles))]
    shuffles = [s if isinstance(s, str) else "".join(s) for s in shuffles]
    both = scorer.raw([anc.pks1] + shuffles, construct)["realness"]
    pks1, sh = float(both[0]), float(np.mean(both[1:]))

    floor = float(np.min(real))
    return RealnessReference(
        construct=construct, n_real=int(len(seqs)),
        real_mean=float(np.mean(real)), real_sd=float(np.std(real)),
        real_min=float(np.min(real)),
        real_q01=float(np.quantile(real, 0.01)),
        pks1=pks1, pks1_shuffled_mean=sh, pks1_minus_shuffle=pks1 - sh,
        floor=floor, pks1_would_pass=bool(pks1 >= floor))


def diversity(seqs: Sequence[str]) -> Dict[str, float]:
    """Pairwise Hamming distance within a panel, at the panel's own size.

    Compare it with the same statistic over the same number of real
    promoters: comparing the minimum over n(n-1)/2 draws with a quantile of
    the all-pairs distribution is the arithmetic that produced a false failure
    once already.
    """
    n = len(seqs)
    if n < 2:
        return {"n": float(n)}
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = seqs[i], seqs[j]
            m = min(len(a), len(b))
            d.append(1.0 - sum(1 for t in range(m) if a[t] == b[t]) / float(m))
    d = np.asarray(d)
    return {"n": float(n), "min": float(d.min()),
            "median": float(np.median(d)), "max": float(d.max())}


def diversity_reference(n: int, genome: str = "P1", seed: int = 7898,
                        repeats: int = 16) -> Dict[str, float]:
    """The same statistic over `n` random real promoters, `repeats` times."""
    rows = _bez_testu(K.corpus_promoters(genome, limit=0))
    rng = np.random.default_rng(seed)
    mins, meds = [], []
    for _ in range(int(repeats)):
        pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        r = diversity([rows[int(i)][1] for i in pick])
        mins.append(r["min"])
        meds.append(r["median"])
    return {"n": float(n), "repeats": float(repeats),
            "min_mean": float(np.mean(mins)),
            "min_p05": float(np.quantile(mins, 0.05)),
            "median_mean": float(np.mean(meds))}


def report(seqs: Sequence[str], gates: SequenceGates,
           realness: Optional[RealnessReference] = None,
           logits: Optional[Sequence[float]] = None,
           names: Optional[Sequence[str]] = None) -> str:
    lines: List[str] = []
    for i, s in enumerate(seqs):
        label = names[i] if names and i < len(names) else "candidate %d" % (i,)
        lines.append("%s (%d bp)" % (label, len(s)))
        for g in gates(s):
            lines.append(str(g))
        if realness is not None and logits is not None:
            lines.append(str(realness.gate(logits[i])))
    if len(seqs) >= 2:
        d = diversity(seqs)
        ref = diversity_reference(len(seqs))
        lines.append("")
        lines.append("diversity: min %.4f median %.4f | %d REAL promoters "
                     "give min %.4f (p05 %.4f), median %.4f"
                     % (d["min"], d["median"], len(seqs), ref["min_mean"],
                        ref["min_p05"], ref["median_mean"]))
    return "\n".join(lines)


__all__ = ["SAFETY_PATH", "KMER_K", "gc", "longest_homopolymer",
           "uorf_counts", "atg_positions", "atg_counts_fast", "kmers",
           "KmerIndex", "Reference", "GateResult", "SequenceGates",
           "RealnessReference", "realness_reference", "diversity",
           "diversity_reference", "report"]
