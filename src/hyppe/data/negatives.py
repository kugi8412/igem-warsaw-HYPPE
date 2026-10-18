"""Negatives for the promoter / non-promoter discriminator.

The self-supervised objective has to learn which sequences are promoters. A
non-promoter is known to rank below a promoter, but its activity and variance
are unknown and a head emitting a number for them would be inventing data, so
negatives enter the ranking term only and are masked everywhere else
including every regression metric: pooling them into a Spearman raises it
simply by adding a class with low predicted strength.

A negative class is useless if a linear probe on 5-mers can separate it, and
two candidates were removed for exactly that -- the promoter read backwards
(AUC 0.9998) and a Markov-0 resample (0.9964), which transfer to each other
at 0.9758 and are one class wearing two hats. The shuffle order is 2 rather
than 1, preserving 3-mer counts exactly: a 3-mer probe drops from 0.9145 to
0.5000 while the compositional signal survives. Order 6 and above is closed,
because the negative degenerates into the positive.

The 31-mer filter is mandatory at zero hits. 9.5 % of FlankRight windows are
literally a neighbouring gene's promoter, so a "negative" that is a promoter
teaches the opposite of the truth. The index covers every split of that
genome: rejecting a candidate leaks nothing, while an index built from train
alone would pass flanks overlapping validation promoters. Contaminated flanks
are not waste either -- 84.7 % of those with at least 700 hits have a partner
orthogroup carrying an existing label -- so they become positives.

The class ratio is set by loss weights, not counts: AUC is flat from 1:1 to
1:8, and what imbalance breaks is the threshold (0.9256 -> 0.8924), which
class weighting repairs (0.9337). Sampling 1:1 per step is a budgetary
decision, since every extra negative is another forward pass through the
shared encoder.

Discrimination is not a Nash player. A 5-mer probe already reaches 0.97-0.98,
so the task is saturated from the first epoch, and a bargaining solver hands
the largest share to the player with the smallest gradient -- in the related
track a reconstruction sitting at 0.9998 took 2.31 of a 3.0 bargain and
suppressed the real objectives. It goes into a subordinate term with a fixed
weight.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Classes that are trained on.
NEGATIVE_CLASSES: Tuple[str, ...] = ("gene", "flank_r", "shuffle")
#: Built but never trained on: a held-out class is the only way to tell "the
#: discriminator generalises" from "it memorised the three it saw".
HELD_OUT_CLASSES: Tuple[str, ...] = ("flank_l",)
#: Removed by measurement; see the module docstring for the AUCs.
BANNED_CLASSES: Dict[str, str] = {
    "reverse": "5-mer linear probe AUC 0.9998 -- a free shortcut",
    "random_composition": "5-mer linear probe AUC 0.9964 (Markov-0)",
}
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class NegativeConfig:
    classes: Tuple[str, ...] = NEGATIVE_CLASSES
    held_out: Tuple[str, ...] = HELD_OUT_CLASSES
    shuffle_order: int = 2
    max_hits: int = 0
    kmer_k: int = 31
    window: int = 800
    ratio: float = 1.0
    #: Candidates longer than `window` are cut down to it. Flanks are 1000 bp
    #: while promoters are 800, so a length equality check silently empties
    #: the flank classes on real data.
    window_stride: int = 20
    drop_non_acgt: bool = True
    recover_contaminated: bool = True
    contaminated_min_hits: int = 700
    loss_weight: float = 0.2
    nash_player: bool = False

    def __post_init__(self) -> None:
        for c in self.classes:
            if c in BANNED_CLASSES:
                raise ValueError(
                    "negative class %r is banned: %s" % (c, BANNED_CLASSES[c]))
        if self.nash_player:
            raise ValueError(
                "discrimination must not be a Nash player: a 5-mer probe "
                "reaches AUC 0.97-0.98, so the task is saturated from epoch "
                "1, and a bargaining solver hands the largest share to the "
                "smallest gradient (measured: 2.31 of 3.0 to a task at "
                "0.9998). Use a subordinate term with a fixed weight.")
        if self.shuffle_order >= 6:
            raise ValueError(
                "shuffle order >= 6 degenerates the negative into the "
                "positive (branching fraction 0.2426 at order 6)")


# K-mer index
def _encode_kmers(seq: str, k: int) -> List[int]:
    """2-bit codes of every k-mer; windows containing non-ACGT are skipped."""
    out: List[int] = []
    code = 0
    valid = 0
    mask = (1 << (2 * k)) - 1
    for ch in seq:
        c = _CODE.get(ch)
        if c is None:
            code, valid = 0, 0
            continue
        code = ((code << 2) | c) & mask
        valid += 1
        if valid >= k:
            out.append(code)
    return out


class PromoterKmerIndex:
    """31-mer identity index over the promoter corpus of one genome.

    Built over every split of that genome deliberately: rejecting a candidate
    leaks nothing, while an index from the training split alone would pass
    flank windows overlapping a validation promoter.
    """

    def __init__(self, k: int = 31) -> None:
        self.k = int(k)
        self._codes: Set[int] = set()
        self.n_promoters = 0

    def build(self, promoter_seqs: Iterable[str]) -> "PromoterKmerIndex":
        for s in promoter_seqs:
            self.n_promoters += 1
            self._codes.update(_encode_kmers(s.upper(), self.k))
        return self

    def hit_count(self, seq: str) -> int:
        return sum(1 for c in _encode_kmers(seq.upper(), self.k)
                   if c in self._codes)

    def best_window(self, seq: str, window: int = 800, stride: int = 20
                    ) -> int:
        """Largest hit count over any `window`-long stretch."""
        s = seq.upper()
        if len(s) <= window:
            return self.hit_count(s)
        best = 0
        for i in range(0, len(s) - window + 1, stride):
            best = max(best, self.hit_count(s[i:i + window]))
        return best

    def verify(self, inside: str, outside: str) -> Dict[str, object]:
        """A promoter from the index must hit fully; one outside must not.

        An assertion, not a log line. Without it an index with a broken case
        mapping passes unnoticed: the symptom was 66 k-mers found instead of
        770, scaling every downstream percentage by about 3x while still
        looking like a working filter.
        """
        n_in = self.hit_count(inside)
        n_out = self.hit_count(outside)
        expect = max(len(inside) - self.k + 1, 0)
        ok = (n_in == expect) and (n_out == 0)
        if not ok:
            raise AssertionError(
                "k-mer index verification failed: a promoter from the index "
                "hit %d of %d expected, one outside hit %d of %d (both must "
                "be %d and 0). A case-mapping bug here silently scales every "
                "contamination figure by ~3x."
                % (n_in, expect, n_out,
                   max(len(outside) - self.k + 1, 0), expect))
        return {"inside_hits": n_in, "expected": expect,
                "outside_hits": n_out, "n_promoters": self.n_promoters,
                "n_kmers": len(self._codes)}


# Euler shuffle
def kmer_counts(seq: str, k: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    s = seq.upper()
    for i in range(max(len(s) - k + 1, 0)):
        w = s[i:i + k]
        out[w] = out.get(w, 0) + 1
    return out


def euler_shuffle(seq: str, order: int = 2,
                  rng: Optional[random.Random] = None,
                  max_tries: int = 2000) -> str:
    """Altschul-Erikson shuffle: preserves (order+1)-mer counts exactly.

    Vertices are order-mers and edges are (order+1)-mers. A random
    arborescence toward the final vertex fixes one last-edge per vertex and
    the remaining edges are permuted freely, so walking the resulting
    Eulerian path gives identical (order+1)-mer composition.

    Raises rather than returning the input: a "negative" identical to the
    sample it came from trains the discriminator to call a promoter a
    non-promoter.
    """
    rng = rng or random.Random(7898)
    s = seq.upper()
    k = int(order)
    if k < 1:
        raise ValueError("order must be >= 1")
    if len(s) < k + 2:
        raise ValueError("sequence shorter than order+2")
    if any(ch not in _CODE for ch in s):
        raise ValueError(
            "non-ACGT in a shuffle source. Filter these at input rather than "
            "raising max_tries: 0.298 %% of promoters contain N, and at "
            "max_tries=100 73.24 %% of them throw, while max_tries=2000 still "
            "leaves 16.90 %% -- raising the limit moves the problem, it does "
            "not solve it.")

    verts = [s[i:i + k] for i in range(len(s) - k + 1)]
    last_v = verts[-1]
    edges: Dict[str, List[str]] = {}
    for i in range(len(verts) - 1):
        edges.setdefault(verts[i], []).append(verts[i + 1])

    for _ in range(max_tries):
        # Pick one last-edge per vertex and test that following them reaches
        # `last_v` from everywhere -- that is exactly the arborescence test.
        last_edge: Dict[str, str] = {}
        for v, outs in edges.items():
            if v == last_v:
                continue
            last_edge[v] = outs[rng.randrange(len(outs))]
        ok = True
        for v in last_edge:
            seen, cur = set(), v
            while cur != last_v:
                if cur in seen or cur not in last_edge:
                    ok = False
                    break
                seen.add(cur)
                cur = last_edge[cur]
            if not ok:
                break
        if not ok:
            continue

        order_out: Dict[str, List[str]] = {}
        for v, outs in edges.items():
            rest = list(outs)
            if v in last_edge:
                rest.remove(last_edge[v])
                rng.shuffle(rest)
                rest.append(last_edge[v])
            else:
                rng.shuffle(rest)
            order_out[v] = rest

        out = [verts[0]]
        cur = verts[0]
        cursor: Dict[str, int] = {v: 0 for v in order_out}
        for _step in range(len(verts) - 1):
            i = cursor.get(cur, 0)
            outs = order_out.get(cur, [])
            if i >= len(outs):
                out = []
                break
            nxt = outs[i]
            cursor[cur] = i + 1
            out.append(nxt[-1])
            cur = nxt
        if not out:
            continue
        result = "".join(out)
        if len(result) != len(s):
            continue
        return result
    raise RuntimeError(
        "euler_shuffle failed to find an arborescence in %d tries" % max_tries)


def assert_shuffle_preserves(inp: str, out: str, order: int = 2) -> None:
    """(order+1)-mer counts must match exactly, and the string must differ.

    Both halves matter: without the count check the negative control could
    silently be mononucleotide and would overstate every motif enrichment,
    and without the difference check an identity shuffle passes.
    """
    a, b = kmer_counts(inp, order + 1), kmer_counts(out, order + 1)
    if a != b:
        diff = max(abs(a.get(kk, 0) - b.get(kk, 0))
                   for kk in set(a) | set(b))
        raise AssertionError(
            "shuffle changed %d-mer counts, max|delta| = %d" % (order + 1,
                                                                diff))
    if out == inp:
        raise AssertionError("shuffle returned the input unchanged")


# Pool
@dataclass
class NegativePool:
    """Candidate negatives with their provenance and rejection accounting."""

    sequences: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    considered: Dict[str, int] = field(default_factory=dict)
    recovered_positives: List[Tuple[str, str]] = field(default_factory=list)
    #: Why a candidate was rejected, per class; `rejected` stays the total.
    #: Without the split the headline rate is uninterpretable: gene bodies
    #: have a median length of 1 308 bp but a minimum of 102, so many are
    #: rejected for being shorter than the window and not for carrying
    #: promoter sequence.
    rejected_short: Dict[str, int] = field(default_factory=dict)
    rejected_non_acgt: Dict[str, int] = field(default_factory=dict)
    rejected_contaminated: Dict[str, int] = field(default_factory=dict)
    rejected_shuffle_failed: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequences)

    def add(self, seq: str, cls: str, source: str) -> None:
        self.sequences.append(seq)
        self.classes.append(cls)
        self.sources.append(source)

    def sample(self, n: int, rng: random.Random
               ) -> Tuple[List[str], List[str], List[str]]:
        """Uniform over classes, then uniform within a class.

        Sampling uniformly over the pool would weight a class by how many
        candidates it happens to yield, which is an artefact of the genome's
        gene density rather than a decision.
        """
        by_class: Dict[str, List[int]] = {}
        for i, c in enumerate(self.classes):
            by_class.setdefault(c, []).append(i)
        names = sorted(by_class)
        if not names:
            return [], [], []
        picks = [by_class[names[rng.randrange(len(names))]] for _ in range(n)]
        idx = [p[rng.randrange(len(p))] for p in picks]
        return ([self.sequences[i] for i in idx],
                [self.classes[i] for i in idx],
                [self.sources[i] for i in idx])

    def class_weights(self) -> Dict[str, float]:
        """Inverse frequency, normalised to mean 1, applied in the loss.

        Imbalance does not move AUC (flat 1:1 to 1:8, +0.0023 at 2.1 SE); it
        moves the threshold (0.9256 -> 0.8924), and this repairs it (0.9337).
        """
        counts: Dict[str, int] = {}
        for c in self.classes:
            counts[c] = counts.get(c, 0) + 1
        if not counts:
            return {}
        inv = {k: 1.0 / v for k, v in counts.items()}
        mean = sum(inv.values()) / len(inv)
        return {k: v / mean for k, v in inv.items()}

    def report(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for c in self.classes:
            counts[c] = counts.get(c, 0) + 1
        rates = {k: round(self.rejected.get(k, 0) / v, 4)
                 for k, v in self.considered.items() if v}
        return {
            "n_candidates": len(self.sequences),
            "by_class": counts,
            "considered": dict(self.considered),
            "rejected": dict(self.rejected),
            "rejection_rate": rates,
            "rejected_because": {
                "too_short": dict(self.rejected_short),
                "non_acgt": dict(self.rejected_non_acgt),
                "promoter_contaminated": dict(self.rejected_contaminated),
                "shuffle_failed": dict(self.rejected_shuffle_failed),
            },
            "contamination_rate_of_long_enough": {
                k: round(self.rejected_contaminated.get(k, 0)
                         / max(v - self.rejected_short.get(k, 0)
                               - self.rejected_non_acgt.get(k, 0), 1), 4)
                for k, v in self.considered.items() if v},
            "class_weights": {k: round(v, 3)
                              for k, v in self.class_weights().items()},
            "n_recovered_positives": len(self.recovered_positives),
        }


def clean_windows(seq: str, index: "PromoterKmerIndex",
                  window: int = 800, stride: int = 20,
                  max_hits: int = 0) -> List[str]:
    """Every `window`-long stretch whose promoter identity is within budget.

    All clean windows are returned and the caller draws uniformly. Picking
    the one with the fewest hits would bias the pool toward the least
    promoter-like stretch of each flank, which is a shortcut of exactly the
    kind that disqualified the banned classes.
    """
    s = seq.upper()
    if len(s) < window:
        return []
    # An index that can answer the whole scan at once does so: the windows
    # overlap, so the loop below recomputes the same k-mers up to
    # `window/stride` times, while `clean_window_starts` takes window counts
    # from one cumulative sum. The two must agree exactly.
    fast = getattr(index, "clean_window_starts", None)
    if fast is not None:
        return [s[int(i):int(i) + window]
                for i in fast(s, window, stride, max_hits)]
    out: List[str] = []
    for i in range(0, len(s) - window + 1, max(stride, 1)):
        w = s[i:i + window]
        if index.hit_count(w) <= max_hits:
            out.append(w)
    return out


def build_pool(candidates: Iterable[Tuple[str, str, str]],
               index: PromoterKmerIndex,
               cfg: Optional[NegativeConfig] = None,
               rng: Optional[random.Random] = None,
               labelled_partners: Optional[Set[str]] = None) -> NegativePool:
    """Filter candidates into a pool, recovering the contaminated ones.

    `candidates` is [(sequence, class, source_id)]; `labelled_partners` is
    the set of source ids whose neighbouring orthogroup carries a label, used
    to turn a heavily contaminated flank into a positive instead of dropping
    it.
    """
    cfg = cfg or NegativeConfig()
    rng = rng or random.Random(7898)
    pool = NegativePool()
    for seq, cls, source in candidates:
        pool.considered[cls] = pool.considered.get(cls, 0) + 1
        s = seq.upper()
        if cfg.drop_non_acgt and any(ch not in _CODE for ch in s):
            pool.rejected[cls] = pool.rejected.get(cls, 0) + 1
            pool.rejected_non_acgt[cls] = (
                pool.rejected_non_acgt.get(cls, 0) + 1)
            continue
        if len(s) < cfg.window:
            pool.rejected[cls] = pool.rejected.get(cls, 0) + 1
            pool.rejected_short[cls] = pool.rejected_short.get(cls, 0) + 1
            continue
        if cls == "shuffle":
            # Shuffle sources are promoters, which are already `window` long;
            # a longer source is cut to the gene-proximal end, because that
            # is the stretch the positives actually contain.
            s = s[-cfg.window:]
            try:
                out = euler_shuffle(s, cfg.shuffle_order, rng)
                assert_shuffle_preserves(s, out, cfg.shuffle_order)
            except (ValueError, RuntimeError, AssertionError):
                pool.rejected[cls] = pool.rejected.get(cls, 0) + 1
                pool.rejected_shuffle_failed[cls] = (
                    pool.rejected_shuffle_failed.get(cls, 0) + 1)
                continue
            pool.add(out, cls, source)
            continue
        windows = clean_windows(s, index, cfg.window, cfg.window_stride,
                                cfg.max_hits)
        if not windows:
            hits = index.best_window(s, cfg.window, cfg.window_stride)
            pool.rejected[cls] = pool.rejected.get(cls, 0) + 1
            pool.rejected_contaminated[cls] = (
                pool.rejected_contaminated.get(cls, 0) + 1)
            if (cfg.recover_contaminated
                    and hits >= cfg.contaminated_min_hits
                    and labelled_partners is not None
                    and source in labelled_partners):
                pool.recovered_positives.append((s, source))
            continue
        pool.add(windows[rng.randrange(len(windows))], cls, source)
    return pool


def falsifiers(pool: NegativePool, cfg: Optional[NegativeConfig] = None
               ) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail)."""
    cfg = cfg or NegativeConfig()
    r = pool.report()
    by = r["by_class"]
    out = [
        ("no banned class is in the pool",
         not (set(by) & set(BANNED_CLASSES)),
         "pool classes %s; banned %s"
         % (sorted(by), sorted(BANNED_CLASSES))),
        ("every trained class is represented",
         all(by.get(c, 0) > 0 for c in cfg.classes),
         "counts %s" % by),
        ("the held-out class is NOT in the pool",
         not (set(by) & set(cfg.held_out)),
         "held out %s -- it is the only way to tell generalisation from "
         "memorising the three trained classes" % list(cfg.held_out)),
        ("discrimination is not a Nash player",
         not cfg.nash_player,
         "saturated task: 5-mer probe AUC 0.9704 (gene) / 0.9820 (shuffle)"),
        # This exists because a length-equality check once silently emptied
        # the flank classes: flanks are 1000 bp, promoters 800.
        ("no class was emptied by filtering",
         all(r["rejection_rate"].get(c, 0.0) < 0.99 for c in cfg.classes),
         "rejection rates %s" % r["rejection_rate"]),
    ]
    return out


# Close negatives
#
# Everything above separates a promoter from the interior of a gene, but the
# generator produces promoters with point mutations, not gene interiors: the
# composition ceiling of a mutant against its own wild type is 0.5032 at one
# substitution and stays inside the real-vs-real null band up to 8, so a
# discriminator trained on the three classes above has never seen the
# boundary it will be asked for.
#
# The classes are named by their rung (`close_k8`) because the rung is the
# whole content of the class: at k = 1 the negative is almost the positive
# and at k = 256 almost a random sequence, so pooling them would report one
# number for a difficulty set by the mixing weights. They are not added to
# `NEGATIVE_CLASSES`, since that default is what every existing run trained
# on and changing it would make two runs of the same config incomparable.

def close_negative_class(k: int) -> str:
    """Canonical class name for the rung with `k` substitutions."""
    return "close_k%d" % int(k)


def add_close_negatives(pool: "NegativePool", codes, rows,
                        ladder: Sequence[int] = (2, 8, 32),
                        per_rung: int = 1, seed: int = 7898,
                        source: str = "mutations.substitute") -> Dict[str, int]:
    """Graded mutants of real promoters, added to the pool as close negatives.

    `codes` is the uint8 [n, L] promoter matrix in the canonical alphabet
    (`sequences.NUCLEOTIDE_ORDER`), `rows` the promoters to mutate.

    Each default rung is a measurement rather than a spacing: 2 substitutions
    destroy a motif hit with p = 0.9373 while moving the 4-mer spectrum by
    2.7 % of the natural distance, 8 is the last rung whose composition
    ceiling stays inside the real-vs-real null band, and 32 is where about a
    quarter of the enriched hits are gone and composition has started, but
    only started, to separate.

    Returns the number added per class, so a rung that produced nothing is
    visible instead of reading as a rung that was easy.
    """
    from .sequences import NUCLEOTIDE_ORDER
    from .trust_region import graded_ladder  # local: avoids a cycle

    # This module carries its own `_CODE` and the mutation machinery carries
    # the canonical order. If they disagree the mutants are decoded under one
    # alphabet and scanned under another. Checked, not assumed.
    if tuple(_CODE[c] for c in NUCLEOTIDE_ORDER) != (0, 1, 2, 3):
        raise ImportError(
            "negatives._CODE and sequences.NUCLEOTIDE_ORDER (%r) disagree"
            % (NUCLEOTIDE_ORDER,))
    letters = NUCLEOTIDE_ORDER
    lad = graded_ladder(codes, rows, ladder, per_rung=per_rung, seed=seed)
    added: Dict[str, int] = {}
    for k in sorted(lad):
        cname = close_negative_class(k)
        for m in lad[k]:
            pool.add("".join(letters[c] for c in m.codes), cname, source)
            added[cname] = added.get(cname, 0) + 1
        pool.considered[cname] = pool.considered.get(cname, 0) + len(lad[k])
    return added


def close_falsifiers(pool: "NegativePool", ladder: Sequence[int] = (2, 8, 32)
                     ) -> List[Tuple[str, bool, str]]:
    """Controls for the close class, each satisfiable by construction."""
    by = pool.report()["by_class"]
    names = [close_negative_class(k) for k in ladder]
    lengths = {len(s) for s, c in zip(pool.sequences, pool.classes)
               if c in names}
    return [
        ("every requested rung is present",
         all(by.get(nm, 0) > 0 for nm in names),
         "counts %s" % {nm: by.get(nm, 0) for nm in names}),
        ("close negatives keep the window length",
         len(lengths) <= 1,
         "lengths seen: %s -- a close negative that changed length would be "
         "separable by length alone" % sorted(lengths)),
        ("the rungs are kept apart, not pooled",
         len(names) == len({nm for nm in names}) and len(names) > 1,
         "a single `close` class would report a difficulty set by its own "
         "mixing weights"),
    ]
