"""Synthetic negatives: sequences that are not promoters but look like them.

They must be composition-matched: the discriminator head asks "is this a
promoter", and against negatives of a different composition that collapses
into "is this GC-rich". Each class holds one statistic of the positive fixed
and destroys another -- `dinucleotide_shuffle` keeps 2-mer counts exactly,
`chimera` keeps local sequence and the distance to the start codon, `reverse`
keeps composition, and `donor` is real non-promoter sequence, the only class
that is not a transformation of a positive.

The shuffle is the head's own falsifier, so the configuration refuses to build
the head without it: it preserves composition by construction, so a GC-only
probe must sit near 0.5 AUC. Chimeras were added on a measurement -- a 5-mer
probe separated the pools at AUC 0.7993 with shuffles alone and 0.4133 with
same-genome chimeras added, against a real-versus-real band of 0.4335.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from hyppe.data.negatives import assert_shuffle_preserves, euler_shuffle, kmer_counts

__all__ = [
    "NEGATIVE_KINDS",
    "SyntheticNegative",
    "dinucleotide_shuffle",
    "dinucleotide_identical",
    "chimera",
    "reverse_complement",
    "make_negatives",
    "composition_falsifier",
]

#: The classes this module can build. `coding` and `flank` of the production
#: configuration are the `donor` class here.
NEGATIVE_KINDS: Tuple[str, ...] = ("dinucleotide_shuffle", "chimera",
                                   "reverse", "donor")

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class SyntheticNegative:
    """One negative, with the provenance needed to audit it later."""

    sequence: str
    kind: str
    #: Identifier of the positive this was derived from. For a chimera, the
    #: left parent; `partner_id` carries the right one.
    source_id: str
    partner_id: Optional[str] = None
    #: Breakpoint for a chimera, in bases from the 5' end.
    breakpoint: Optional[int] = None


def dinucleotide_shuffle(sequence: str,
                         rng: Optional[random.Random] = None) -> str:
    """Shuffle preserving dinucleotide counts exactly (Altschul-Erikson).

    Order 2 of `euler_shuffle`. Raises rather than returning the input: a
    negative identical to its source teaches the head to call a promoter a
    non-promoter.
    """
    out = euler_shuffle(sequence, order=1, rng=rng or random.Random(7898))
    assert_shuffle_preserves(sequence, out, order=1)
    return out


def chimera(left: str, right: str, breakpoint: Optional[int] = None,
            rng: Optional[random.Random] = None) -> Tuple[str, int]:
    """Splice two promoters at one breakpoint; return the sequence and it.

    Parents must be the same length and, in training, from one genome: equal
    length keeps every position at its real distance from the start codon,
    one genome keeps composition and motif vocabulary. What is left to detect
    is coherence between distant positions.
    """
    if len(left) != len(right):
        raise ValueError("chimera parents differ in length: %d vs %d"
                         % (len(left), len(right)))
    n = len(left)
    if n < 4:
        raise ValueError("sequences too short to splice")
    rng = rng or random.Random(7898)
    # Away from the ends, or the result is one parent with a cosmetic edit.
    lo, hi = max(1, n // 10), min(n - 1, n - n // 10)
    if breakpoint is None:
        breakpoint = rng.randrange(lo, hi)
    if not 0 < breakpoint < n:
        raise ValueError("breakpoint %d outside (0, %d)" % (breakpoint, n))
    return left[:breakpoint] + right[breakpoint:], int(breakpoint)


def reverse_complement(sequence: str) -> str:
    """The other strand, 5' to 3'.

    A valid negative precisely because promoters are not reverse-complement
    invariant. A model that treats RC as an augmentation destroys this class.
    """
    s = sequence.upper()
    if any(c not in "ACGT" for c in s):
        raise ValueError("non-ACGT base in a reverse-complement source")
    return s.translate(_COMPLEMENT)[::-1]


def make_negatives(
    positives: Sequence[Tuple[str, str]],
    kinds: Sequence[str] = ("dinucleotide_shuffle", "chimera"),
    mix: Optional[Dict[str, float]] = None,
    fraction: float = 0.25,
    group_of: Optional[Dict[str, str]] = None,
    donors: Optional[Sequence[Tuple[str, str]]] = None,
    seed: int = 7898,
) -> List[SyntheticNegative]:
    """Build a pool of negatives from a pool of positives.

    `positives` is `(record_id, sequence)` and `fraction` the share of them to
    make. `mix` weights the classes, equal by default. `group_of` maps a
    record to its genome and chimeras form only within a group. `donors`
    supplies real non-promoter sequence. Deterministic in `seed`.
    """
    if not positives:
        raise ValueError("no positives")
    unknown = [k for k in kinds if k not in NEGATIVE_KINDS]
    if unknown:
        raise ValueError("unknown negative kinds %s; known are %s"
                         % (unknown, list(NEGATIVE_KINDS)))
    if "donor" in kinds and not donors:
        raise ValueError(
            "the 'donor' class needs real non-promoter sequence; it cannot "
            "be synthesised from a promoter")
    if fraction <= 0.0:
        return []

    rng = random.Random(seed)
    weights = {k: float((mix or {}).get(k, 1.0)) for k in kinds}
    total_w = sum(weights.values())
    if total_w <= 0.0:
        raise ValueError("the mix weights sum to zero")
    n_total = max(1, int(round(len(positives) * float(fraction))))

    # Index positives by group so a chimera never crosses one.
    by_group: Dict[str, List[int]] = {}
    for i, (rid, _seq) in enumerate(positives):
        g = (group_of or {}).get(rid, "_all")
        by_group.setdefault(g, []).append(i)

    out: List[SyntheticNegative] = []
    for kind in kinds:
        n_kind = int(round(n_total * weights[kind] / total_w))
        made = 0
        attempts = 0
        # The cap turns an impossible request into a short pool, not a hang;
        # the caller sees the shortfall in len(out).
        while made < n_kind and attempts < 20 * max(1, n_kind):
            attempts += 1
            i = rng.randrange(len(positives))
            rid, seq = positives[i]
            try:
                if kind == "dinucleotide_shuffle":
                    out.append(SyntheticNegative(
                        dinucleotide_shuffle(seq, rng), kind, rid))
                elif kind == "reverse":
                    out.append(SyntheticNegative(
                        reverse_complement(seq), kind, rid))
                elif kind == "chimera":
                    g = (group_of or {}).get(rid, "_all")
                    pool = [j for j in by_group[g]
                            if j != i and len(positives[j][1]) == len(seq)]
                    if not pool:
                        continue
                    j = pool[rng.randrange(len(pool))]
                    pid, pseq = positives[j]
                    s, bp = chimera(seq, pseq, None, rng)
                    out.append(SyntheticNegative(s, kind, rid, pid, bp))
                elif kind == "donor":
                    did, dseq = donors[rng.randrange(len(donors))]
                    if len(dseq) < len(seq):
                        continue
                    start = rng.randrange(0, len(dseq) - len(seq) + 1)
                    out.append(SyntheticNegative(
                        dseq[start:start + len(seq)], kind, did))
            except ValueError:
                # A source with an N, or too short to shuffle. Filtering at
                # input is the contract; raising here would let one bad row
                # abort a whole pool.
                continue
            made += 1
    return out


def composition_falsifier(positives: Sequence[str],
                          negatives: Sequence[str]) -> Dict[str, object]:
    """Is the pool separable on composition alone?

    The AUC of the best mononucleotide frequency and of GC content, by rank.
    Both must sit near 0.5: a pool a GC threshold separates measures the
    head's input, not the head. `ok` is advisory and generous (0.40 to 0.60),
    the real gate being the probe AUC against the shuffles.
    """
    def gc(s: str) -> float:
        return (s.count("G") + s.count("C")) / max(1, len(s))

    def freq(s: str, base: str) -> float:
        return s.count(base) / max(1, len(s))

    def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
        # Rank-based Mann-Whitney U, ties counted as a half.
        if not pos or not neg:
            return float("nan")
        wins = 0.0
        for p in pos:
            for n in neg:
                wins += 1.0 if p > n else (0.5 if p == n else 0.0)
        return wins / (len(pos) * len(neg))

    gc_auc = auc([gc(s) for s in positives], [gc(s) for s in negatives])
    per_base = {b: auc([freq(s, b) for s in positives],
                       [freq(s, b) for s in negatives])
                for b in "ACGT"}
    worst = max(per_base.values(), key=lambda v: abs(v - 0.5))
    out = {
        "gc_auc": gc_auc,
        "base_auc": per_base,
        "most_separating_base_auc": worst,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
    }
    out["ok"] = bool(0.40 <= gc_auc <= 0.60 and 0.40 <= worst <= 0.60)
    return out


def dinucleotide_identical(a: str, b: str) -> bool:
    """True when two sequences have identical dinucleotide counts."""
    return kmer_counts(a, 2) == kmer_counts(b, 2)
