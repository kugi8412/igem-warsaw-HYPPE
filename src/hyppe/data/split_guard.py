"""The split the pipeline actually uses, and the guard that refuses a leak.

`SPLIT_FINAL.tsv` offers three assignments of the same 26 327 orthogroups,
differing in how many shared canonical 25-mers join two orthogroups into a
leakage block, and so in how large the pks1 holdout ("spy") comes out:

    column       train   val    test   spy    spy promoters
    A_og         18702   3811   3813     1               21
    B_block10    18749   3725   3772    81              896
    C_block5     18590   3838   3728   171             1978

`B_block10` is the choice. `A_og` is refused: its spy is the single
orthogroup OG0000996, and holding out only P1_G009919 leaves six
byte-identical copies of that promoter in training, so a "zero-shot pks1"
number read off it is a memory read. `C_block5` is not wrong but answers a
different question -- it more than doubles the holdout for no measured gain,
since `B_block10` already measured 0.0000 cross-boundary leakage on the audit
that defined both -- and stays available through `column=`.

The guard runs against an independent clustering, and it fires.
`SPLIT_FINAL.tsv` cannot fail a disjointness test on its own unit: its blocks
are unions of orthogroups, so "no orthogroup in two parts" holds by
construction, and a guard that cannot fail is not a guard. The independent
unit is `data/splits/homology_clusters/rows.tsv`, built by a different
procedure (FracMinHash, k = 11, Jaccard >= 0.05). Each of the 26 327
orthogroups lies in exactly one of its 22 159 clusters, so the cluster is a
strict coarsening of the orthogroup and the two artefacts are comparable.
Against that unit every column fails -- 1555, 594 and 367 spanning clusters
for A_og, B_block10 and C_block5 -- which is two similarity relations
disagreeing rather than a bug in either file, the k = 11 one being the more
sensitive. The pks1 holdout is the sharpest case: cluster 902 holds 5
orthogroups, of which `B_block10` calls 1 spy, 3 train and 1 val.

So the guard raises and :func:`resolve_clusters` repairs the assignment,
moving every orthogroup of a spanning cluster into one part. "conservative"
(the default) ranks spy > test > val > train, so evaluation keeps every
sequence homologous to its own part and training gives up the orthogroups
tying the two together; it is the only order that keeps the pks1 cluster
whole. "majority" gives the cluster to the part holding most of it, ties
broken by the same priority: leak-free and much cheaper (776 orthogroups
moved against 1 206), but on `B_block10` it moves pks1 into train, and
:func:`build_split` raises rather than let the cheaper resolution delete the
holdout it was chosen for. The default costs 1 206 moves, taking train from
18 749 to 17 755; no evaluation part shrinks. Both are deterministic, and
`SplitAssignment.report()` prints how many orthogroups moved and in which
direction, because an uncounted repair is indistinguishable from one that did
not happen.

Not repaired here: `viride` (10 255 promoters) has no expression label at
all, being in the orthology's 20 genomes and not the metadata's 19. It stays
in the split and is excluded by the label mask, so reconstruction and the
discriminator can use it while no PCC/SCC denominator counts it.
"""
from __future__ import annotations

import collections
import io
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .splits import (
    DEFAULT_COLUMN,
    PKS1_ORTHOGROUP,
    SPLIT_NAMES,
    SplitTable,
    load_splits,
)

#: Default location of the independent clustering, relative to the repo root.
CLUSTER_ROWS = os.path.join("data", "splits", "homology_clusters", "rows.tsv")

#: Part priority for the conservative resolution. Earlier wins.
CONSERVATIVE_PRIORITY: Tuple[str, ...] = ("spy", "test", "val", "train")

RESOLUTIONS: Tuple[str, ...] = ("conservative", "majority")

#: Measured rejection of `A_og`, kept next to the code that refuses it.
A_OG_REFUSAL = (
    "column 'A_og' holds out the single orthogroup %s, leaving six "
    "byte-identical copies of the pks1 promoter in training (B10, F7, "
    "I1237, P3041, P3080, P3116). A zero-shot number from that split is a "
    "memory read. Pass allow_single_orthogroup_spy=True only for a run whose "
    "point is to measure that memory." % (PKS1_ORTHOGROUP,)
)


class SplitLeak(AssertionError):
    """A unit of homology appears in more than one part of the split."""


@dataclass
class ClusterMap:
    """promoter id -> cluster, orthogroup -> cluster, and the provenance."""

    cluster_of_orthogroup: Dict[str, str]
    orthogroups_of_cluster: Dict[str, List[str]]
    rows_per_cluster: Dict[str, int]
    source: str = ""
    n_rows: int = 0
    n_duplicate_ids: int = 0

    def __len__(self) -> int:
        return len(self.orthogroups_of_cluster)

    def report(self) -> Dict[str, object]:
        sizes = sorted((len(v) for v in self.orthogroups_of_cluster.values()),
                       reverse=True)
        return {
            "source": os.path.basename(self.source),
            "n_clusters": len(self.orthogroups_of_cluster),
            "n_orthogroups": len(self.cluster_of_orthogroup),
            "n_rows": self.n_rows,
            "duplicate_promoter_ids": self.n_duplicate_ids,
            "largest_cluster_orthogroups": sizes[0] if sizes else 0,
            "singleton_clusters": sum(1 for s in sizes if s == 1),
        }


def load_cluster_map(path: str) -> ClusterMap:
    """Read `rows.tsv` of the homology clustering.

    The file's own `split` column is deliberately not read: it is a 10 %
    partition of clusters whose "spy" is a random holdout rather than the pks1
    block (OG0000996 sits entirely in its `train`), so mixing the two would
    silently swap which question the spy part answers.
    """
    cl_of_og: Dict[str, str] = {}
    ogs_of_cl: Dict[str, List[str]] = collections.defaultdict(list)
    rows_per_cl: Dict[str, int] = collections.Counter()
    seen_ids: Dict[str, int] = collections.Counter()
    n_rows = 0
    with io.open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        need = ("ID", "OG_ID", "cluster")
        missing = [c for c in need if c not in header]
        if missing:
            raise ValueError("%s lacks columns %r (has %r)"
                             % (path, missing, header))
        ix = {c: header.index(c) for c in need}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            n_rows += 1
            og, cl = p[ix["OG_ID"]], p[ix["cluster"]]
            seen_ids[p[ix["ID"]]] += 1
            rows_per_cl[cl] += 1
            prev = cl_of_og.get(og)
            if prev is None:
                cl_of_og[og] = cl
                ogs_of_cl[cl].append(og)
            elif prev != cl:
                # Never observed (0 of 26 327), but if the artefact is rebuilt
                # with a procedure that splits orthogroups, every guarantee
                # below changes meaning.
                raise ValueError(
                    "orthogroup %s is in clusters %s and %s; the guard below "
                    "assumes the cluster is a COARSENING of the orthogroup "
                    "(measured: 0 of 26 327 orthogroups span two clusters)"
                    % (og, prev, cl))
    dup = sum(1 for v in seen_ids.values() if v > 1)
    return ClusterMap(cluster_of_orthogroup=cl_of_og,
                      orthogroups_of_cluster=dict(ogs_of_cl),
                      rows_per_cluster=dict(rows_per_cl),
                      source=path, n_rows=n_rows, n_duplicate_ids=dup)


@dataclass
class SplitAssignment:
    """A repaired, guarded orthogroup -> part map."""

    part_of: Dict[str, str]
    column: str
    resolution: str
    moved: Dict[Tuple[str, str], int] = field(default_factory=dict)
    n_clusters_repaired: int = 0
    n_clusters_checked: int = 0
    rows_repaired: int = 0
    source: str = ""
    cluster_source: str = ""
    notes: List[str] = field(default_factory=list)

    def of(self, orthogroup: str, default: Optional[str] = None
           ) -> Optional[str]:
        return self.part_of.get(orthogroup, default)

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in SPLIT_NAMES}
        for v in self.part_of.values():
            out[v] = out.get(v, 0) + 1
        return out

    def report(self) -> Dict[str, object]:
        c = self.counts()
        total = max(sum(c.values()), 1)
        return {
            "column": self.column,
            "resolution": self.resolution,
            "source": os.path.basename(self.source),
            "cluster_source": os.path.basename(self.cluster_source),
            "counts": c,
            "fractions": {k: round(v / total, 4) for k, v in c.items()},
            "clusters_checked": self.n_clusters_checked,
            "clusters_repaired": self.n_clusters_repaired,
            "promoter_rows_in_repaired_clusters": self.rows_repaired,
            "orthogroups_moved": sum(self.moved.values()),
            "moves": {"%s->%s" % k: v
                      for k, v in sorted(self.moved.items())},
            "pks1_part": self.part_of.get(PKS1_ORTHOGROUP),
            "notes": list(self.notes),
        }


def spanning_clusters(part_of: Dict[str, str], clusters: ClusterMap
                      ) -> Dict[str, Dict[str, int]]:
    """cluster -> {part: how many of its orthogroups} for clusters that span.

    Orthogroups absent from `part_of` are ignored rather than counted as a
    part of their own, so that a future subset cannot turn "not assigned" into
    a fourth part and report a leak that is really a join failure.
    """
    out: Dict[str, Dict[str, int]] = {}
    for cl, ogs in clusters.orthogroups_of_cluster.items():
        parts: Dict[str, int] = collections.Counter()
        for og in ogs:
            p = part_of.get(og)
            if p is not None:
                parts[p] += 1
        if len(parts) > 1:
            out[cl] = dict(parts)
    return out


def assert_disjoint(part_of: Dict[str, str],
                    clusters: Optional[ClusterMap] = None,
                    max_examples: int = 5) -> Dict[str, object]:
    """Raise unless every homology unit lies in exactly one part. (Gate D2)

    First the orthogroup, trivially true for a dict and checked anyway because
    a caller may pass pairs built from a join that duplicated rows; then the
    cluster, which is the unit that can actually fail, and does -- 594 of
    22 159 span two parts under raw `B_block10`. Raises `SplitLeak`, or
    returns the audit dict when it passes.
    """
    dupes = [og for og, p in _pairs(part_of)
             if not isinstance(p, str) or p not in SPLIT_NAMES]
    if dupes:
        raise SplitLeak("orthogroups with a non-part value: %r"
                        % (dupes[:max_examples],))
    audit: Dict[str, object] = {
        "n_orthogroups": len(part_of),
        "unit_orthogroup_ok": True,
        "cluster_check": "skipped (no cluster map given)",
    }
    if clusters is None:
        return audit
    bad = spanning_clusters(part_of, clusters)
    audit["n_clusters"] = len(clusters)
    audit["n_clusters_spanning"] = len(bad)
    if bad:
        rows = sum(clusters.rows_per_cluster.get(c, 0) for c in bad)
        examples = sorted(bad.items(),
                          key=lambda kv: -sum(kv[1].values()))[:max_examples]
        raise SplitLeak(
            "%d of %d homology clusters lie in more than one part of the "
            "split (%d promoter rows). A cluster is a set of orthogroups "
            "sharing >= 5 %% of their 11-mer sketch, so this is the same "
            "sequence on both sides of the boundary. Examples "
            "(cluster: {part: orthogroups}): %s. Repair the assignment with "
            "`resolve_clusters` -- the guard is not the place to lower the "
            "bar." % (len(bad), len(clusters), rows, examples))
    audit["cluster_check"] = "passed"
    return audit


def _pairs(part_of: Dict[str, str]) -> Iterable[Tuple[str, str]]:
    return part_of.items()


def resolve_clusters(part_of: Dict[str, str], clusters: ClusterMap,
                     resolution: str = "conservative"
                     ) -> Tuple[Dict[str, str], Dict[Tuple[str, str], int],
                                int, int]:
    """Move every split-spanning cluster into one part. Deterministic.

    Returns (new map, {(from, to): n}, clusters repaired, rows repaired).
    """
    if resolution not in RESOLUTIONS:
        raise ValueError("resolution must be one of %r, got %r"
                         % (RESOLUTIONS, resolution))
    out = dict(part_of)
    moved: Dict[Tuple[str, str], int] = collections.Counter()
    bad = spanning_clusters(part_of, clusters)
    rows = 0
    for cl, parts in sorted(bad.items()):
        if resolution == "conservative":
            winner = next(p for p in CONSERVATIVE_PRIORITY if p in parts)
        else:
            best = max(parts.values())
            winner = next(p for p in CONSERVATIVE_PRIORITY
                          if parts.get(p, -1) == best)
        rows += clusters.rows_per_cluster.get(cl, 0)
        for og in clusters.orthogroups_of_cluster[cl]:
            old = out.get(og)
            if old is not None and old != winner:
                out[og] = winner
                moved[(old, winner)] += 1
    return out, dict(moved), len(bad), rows


def build_split(split_path: str,
                cluster_path: Optional[str] = None,
                column: str = DEFAULT_COLUMN,
                resolution: str = "conservative",
                allow_single_orthogroup_spy: bool = False
                ) -> SplitAssignment:
    """Load, repair and guard the split in one call. The entry point.

    `cluster_path=None` is allowed and reported rather than silent: without
    the independent clustering the guard degenerates to the tautology
    described above, and the returned `notes` say so.
    """
    table: SplitTable = load_splits(split_path, column=column)
    counts = table.counts()
    notes: List[str] = []
    if counts.get("spy", 0) <= 1 and not allow_single_orthogroup_spy:
        raise ValueError(A_OG_REFUSAL + " (column %r has spy=%d)"
                         % (column, counts.get("spy", 0)))

    part_of = dict(table.assignment)
    moved: Dict[Tuple[str, str], int] = {}
    n_bad = n_rows = 0
    clusters: Optional[ClusterMap] = None
    if cluster_path:
        clusters = load_cluster_map(cluster_path)
        part_of, moved, n_bad, n_rows = resolve_clusters(
            part_of, clusters, resolution=resolution)
    else:
        notes.append(
            "no cluster map: the disjointness guard runs on the orthogroup, "
            "which SPLIT_FINAL satisfies by construction, so it measures "
            "nothing. Pass cluster_path to make it falsifiable.")

    assert_disjoint(part_of, clusters)          # gate D2, after the repair

    asg = SplitAssignment(
        part_of=part_of, column=column, resolution=resolution, moved=moved,
        n_clusters_repaired=n_bad,
        n_clusters_checked=len(clusters) if clusters else 0,
        rows_repaired=n_rows, source=split_path,
        cluster_source=clusters.source if clusters else "", notes=notes)
    if asg.part_of.get(PKS1_ORTHOGROUP) != "spy":
        raise ValueError(
            "the pks1 orthogroup %s landed in %r, not 'spy'. The design "
            "target must not be in any part the model trains or selects on."
            % (PKS1_ORTHOGROUP, asg.part_of.get(PKS1_ORTHOGROUP)))
    return asg


def compare_columns(split_path: str, cluster_path: str,
                    columns: Sequence[str] = ("A_og", "B_block10", "C_block5")
                    ) -> List[Dict[str, object]]:
    """The tables above, recomputed on the files. Use it in a report."""
    clusters = load_cluster_map(cluster_path)
    out: List[Dict[str, object]] = []
    for col in columns:
        t = load_splits(split_path, column=col)
        bad = spanning_clusters(t.assignment, clusters)
        rows = sum(clusters.rows_per_cluster.get(c, 0) for c in bad)
        row: Dict[str, object] = {"column": col, "counts": t.counts(),
                                  "clusters_spanning": len(bad),
                                  "clusters_total": len(clusters),
                                  "rows_in_spanning": rows,
                                  "rows_total": clusters.n_rows}
        for res in RESOLUTIONS:
            fixed, moved, n_bad, _ = resolve_clusters(t.assignment, clusters,
                                                      res)
            c = collections.Counter(fixed.values())
            row["after_" + res] = {"counts": dict(c),
                                   "orthogroups_moved": sum(moved.values())}
        out.append(row)
    return out
