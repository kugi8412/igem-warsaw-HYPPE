"""Build a train / val / spy / test split from sequences alone.

The delivered split came from a precomputed homology table that is not in this
repository, so a clone could load `data/splits/splits.tsv` but could not make
one for its own corpus. This builds one from sequences and nothing else.

    train  fits the weights
    val    chooses an epoch WITHIN one run
    spy    chooses BETWEEN runs; opened once per comparison
    test   opened ONCE, at the end

The atom is the cluster and not the row: on this corpus a split by orthogroup
still left 31.87 % of homologous pairs sharing at least 169 bp of exact
sequence across the boundary. `val` and `spy` must also be free of shared
homologues, or the spy veto repeats val instead of controlling it -- the
evaluation statistic's own sd is 0.0064, enough for selection on one set to
inflate a result by 1.62 sd.

The graph catches near-duplicates, not orthology: at k = 25 five scattered
SNPs in 800 bp already drop containment below the 0.90 threshold, and real
Trichoderma orthologues sit near 51-55 % identity, where 25-mer containment is
essentially zero. Orthology is an input -- pass `groups` and the group becomes
the atom. Without it, a corpus of diverged orthologues gives a leaky split.

Assignment uses a blake2b hash of the cluster key rather than a shuffle, so
the same corpus gives the same split on any machine and any Python version.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SplitSpec",
    "SplitBuild",
    "canonical_kmers",
    "sketch",
    "cluster_by_similarity",
    "assign_clusters",
    "build_splits",
    "write_split_table",
    "falsifiers",
]

#: The four splits, in the order they are filled. Fixed: a reordering would
#: change which cluster lands where for an unchanged corpus.
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "spy", "test")

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class SplitSpec:
    """Everything that decides a split, in one object that is written out.

    `fractions` are of the number of clusters, not of rows. Clusters differ in
    size, so row counts come out near the fractions but not on them; the
    report states both.
    """

    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.80, "val": 0.10,
                                 "spy": 0.05, "test": 0.05})
    k: int = 25
    #: FracMinHash denominator. 8 leaves ~97 sketch elements on an 800 bp
    #: promoter: enough for a stable containment estimate, 8x cheaper.
    scale: int = 8
    #: Containment above which two records are called homologous; the
    #: threshold the delivered split column was built at.
    threshold: float = 0.90
    #: Minimum shared sketch k-mers before containment is computed at all.
    #: One shared 25-mer between unrelated promoters is common and the ratio
    #: is unstable there.
    min_shared: int = 2
    seed: int = 7898

    def validate(self) -> None:
        missing = [s for s in SPLIT_NAMES if s not in self.fractions]
        if missing:
            raise ValueError("fractions is missing %s" % (missing,))
        total = sum(float(self.fractions[s]) for s in SPLIT_NAMES)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("fractions sum to %.6f, not 1.0" % total)
        if any(float(self.fractions[s]) < 0.0 for s in SPLIT_NAMES):
            raise ValueError("a fraction is negative")
        if self.k < 4:
            raise ValueError("k must be >= 4")
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")


@dataclass
class SplitBuild:
    """The result: one split per group, plus everything needed to defend it."""

    #: group key -> split name
    assignment: Dict[str, str]
    #: group key -> cluster id
    cluster_of: Dict[str, int]
    #: cluster id -> split name
    cluster_split: Dict[int, str]
    spec: SplitSpec
    n_groups: int
    n_clusters: int
    largest_cluster: int
    counts: Dict[str, int]

    def report(self) -> Dict[str, object]:
        return {
            "n_groups": self.n_groups,
            "n_clusters": self.n_clusters,
            "largest_cluster": self.largest_cluster,
            "counts": dict(self.counts),
            "fractions_observed": {
                s: (self.counts.get(s, 0) / self.n_groups) if self.n_groups
                else 0.0 for s in SPLIT_NAMES},
            "spec": {"k": self.spec.k, "scale": self.spec.scale,
                     "threshold": self.spec.threshold,
                     "min_shared": self.spec.min_shared,
                     "seed": self.spec.seed,
                     "fractions": dict(self.spec.fractions)},
        }


# Sequences
def canonical_kmers(sequence: str, k: int = 25) -> set:
    """Canonical k-mers: each k-mer and its reverse complement, min of the two.

    Canonicalisation belongs to the audit, never to the model: two promoters
    are homologous whichever strand the assembly stores them on, but a
    promoter read on the other strand is not the same promoter.
    """
    s = sequence.upper()
    if len(s) < k:
        return set()
    out = set()
    for i in range(len(s) - k + 1):
        fwd = s[i:i + k]
        if any(c not in "ACGT" for c in fwd):
            continue
        rev = fwd.translate(_COMPLEMENT)[::-1]
        out.add(fwd if fwd <= rev else rev)
    return out


def _hash64(text: str, salt: int) -> int:
    h = hashlib.blake2b(("%d:%s" % (salt, text)).encode("utf-8"),
                        digest_size=8)
    return int.from_bytes(h.digest(), "big")


def sketch(sequence: str, k: int = 25, scale: int = 8,
           seed: int = 7898) -> frozenset:
    """FracMinHash sketch: the canonical k-mers whose hash falls in 1/scale.

    A hash-defined subset rather than a random sample, so a k-mer is kept or
    dropped consistently in every record. That is what makes the intersection
    of two sketches an unbiased estimate of the intersection of the full sets.
    """
    if scale <= 1:
        return frozenset(canonical_kmers(sequence, k))
    cut = (1 << 64) // int(scale)
    return frozenset(km for km in canonical_kmers(sequence, k)
                     if _hash64(km, seed) < cut)


def _blake_fraction(text: str, salt: int) -> float:
    return _hash64(text, salt) / float(1 << 64)


# Clusters
class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {it: it for it in items}

    def find(self, a: str) -> str:
        root = a
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[a] != root:          # path compression
            self._parent[a], a = root, self._parent[a]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def cluster_by_similarity(
    sketches: Mapping[str, frozenset],
    spec: Optional[SplitSpec] = None,
) -> Dict[str, int]:
    """Connected components of the graph "containment above the threshold".

    Containment, `|A and B| / min(|A|, |B|)`, rather than Jaccard: a short
    promoter fully contained in a long one is a leak, and Jaccard would score
    that pair low merely because the sets differ in size. Candidate pairs come
    from an inverted index, so cost follows shared k-mers rather than n^2.
    """
    spec = spec or SplitSpec()
    spec.validate()
    keys = list(sketches)
    uf = _UnionFind(keys)

    index: Dict[str, List[str]] = {}
    for key in keys:
        for km in sketches[key]:
            index.setdefault(km, []).append(key)

    shared: Dict[Tuple[str, str], int] = {}
    for members in index.values():
        if len(members) < 2:
            continue
        # A k-mer in very many records is repeat-like: no information about
        # any particular pair, and quadratic in that bucket.
        if len(members) > 1000:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a < b else (b, a)
                shared[pair] = shared.get(pair, 0) + 1

    for (a, b), n in shared.items():
        if n < spec.min_shared:
            continue
        smaller = min(len(sketches[a]), len(sketches[b]))
        if smaller == 0:
            continue
        if n / smaller >= spec.threshold:
            uf.union(a, b)

    roots: Dict[str, int] = {}
    out: Dict[str, int] = {}
    for key in keys:
        r = uf.find(key)
        if r not in roots:
            roots[r] = len(roots)
        out[key] = roots[r]
    return out


def assign_clusters(cluster_of: Mapping[str, int],
                    spec: Optional[SplitSpec] = None) -> Dict[int, str]:
    """Assign whole clusters to splits, deterministically and greedily.

    Clusters are visited in the order of a salted hash of their key, then each
    goes to whichever split is furthest below its target share: hash order so
    the result does not depend on the random implementation, greedy filling so
    that one very large cluster cannot exhaust a small split.
    """
    spec = spec or SplitSpec()
    spec.validate()

    size: Dict[int, int] = {}
    members: Dict[int, List[str]] = {}
    for key, cid in cluster_of.items():
        size[cid] = size.get(cid, 0) + 1
        members.setdefault(cid, []).append(key)

    total = sum(size.values())
    order = sorted(size, key=lambda c: _blake_fraction(min(members[c]),
                                                       spec.seed))
    want = {s: float(spec.fractions[s]) * total for s in SPLIT_NAMES}
    have = {s: 0.0 for s in SPLIT_NAMES}
    out: Dict[int, str] = {}
    for cid in order:
        # Deficit, not share: a split whose fraction is zero is never chosen.
        pick = max(SPLIT_NAMES, key=lambda s: (want[s] - have[s], s))
        out[cid] = pick
        have[pick] += size[cid]
    return out


def build_splits(
    records: Sequence[Tuple[str, str]],
    groups: Optional[Mapping[str, str]] = None,
    spec: Optional[SplitSpec] = None,
) -> SplitBuild:
    """Cluster `records` and assign whole clusters to the four splits.

    `records` is a sequence of `(record_id, sequence)`. `groups` maps a record
    id to a pre-existing key -- an orthogroup, say -- which then becomes the
    atom, so two records of one group can never be separated whatever their
    sequences say; omit it and each record is its own group. The returned
    assignment is keyed by group, which is how a training loop looks a row up.
    """
    spec = spec or SplitSpec()
    spec.validate()
    if not records:
        raise ValueError("no records")

    # One sketch per group, the union over its records, so a group clusters
    # as the single object it is meant to be.
    by_group: Dict[str, set] = {}
    for rid, seq in records:
        g = str(groups[rid]) if groups is not None and rid in groups else rid
        by_group.setdefault(g, set()).update(
            sketch(seq, spec.k, spec.scale, spec.seed))
    sketches = {g: frozenset(s) for g, s in by_group.items()}

    empty = [g for g, s in sketches.items() if not s]
    if empty and len(empty) == len(sketches):
        raise ValueError(
            "every sketch is empty: k=%d is longer than the sequences, or "
            "scale=%d is too coarse for them" % (spec.k, spec.scale))

    cluster_of = cluster_by_similarity(sketches, spec)
    cluster_split = assign_clusters(cluster_of, spec)
    assignment = {g: cluster_split[c] for g, c in cluster_of.items()}

    counts: Dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    for s in assignment.values():
        counts[s] += 1
    sizes: Dict[int, int] = {}
    for c in cluster_of.values():
        sizes[c] = sizes.get(c, 0) + 1

    return SplitBuild(
        assignment=assignment,
        cluster_of=dict(cluster_of),
        cluster_split=dict(cluster_split),
        spec=spec,
        n_groups=len(assignment),
        n_clusters=len(set(cluster_of.values())),
        largest_cluster=max(sizes.values()) if sizes else 0,
        counts=counts,
    )


def write_split_table(build: SplitBuild, path: str,
                      group_column: str = "og",
                      split_column: str = "split") -> None:
    """Write the two-column table the training loop reads.

    Column names are arguments because the delivered table calls them `og`
    and `H_kmer90`; the loop is told which to use and records the choice.
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow([group_column, split_column])
        for g in sorted(build.assignment):
            w.writerow([g, build.assignment[g]])


# Falsifiers
def falsifiers(build: SplitBuild) -> Dict[str, object]:
    """Checks that must hold for the split to mean what it claims.

    `ok` is the conjunction of the blocking ones. A split that fails any of
    them still looks like a split and still trains, hence a verdict rather
    than a warning.
    """
    straddling = 0
    seen: Dict[int, str] = {}
    for g, cid in build.cluster_of.items():
        s = build.assignment[g]
        if cid in seen and seen[cid] != s:
            straddling += 1
        seen.setdefault(cid, s)

    non_empty = all(build.counts.get(s, 0) > 0
                    for s in SPLIT_NAMES
                    if float(build.spec.fractions[s]) > 0.0)
    val_spy_disjoint = not (set(
        g for g, s in build.assignment.items() if s == "val") & set(
        g for g, s in build.assignment.items() if s == "spy"))
    # One cluster swallowing most of the corpus means the threshold is too
    # loose, and then no held-out set is independent of training.
    not_degenerate = build.largest_cluster <= max(
        1, int(0.5 * build.n_groups))

    out = {
        "clusters_straddling_splits": straddling,
        "every_requested_split_non_empty": non_empty,
        "val_spy_disjoint": val_spy_disjoint,
        "largest_cluster_below_half": not_degenerate,
        "largest_cluster": build.largest_cluster,
        "n_clusters": build.n_clusters,
        "n_groups": build.n_groups,
    }
    out["ok"] = bool(straddling == 0 and non_empty and val_spy_disjoint
                     and not_degenerate)
    return out
