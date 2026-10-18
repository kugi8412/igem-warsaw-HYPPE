"""The composition layer: the one module that turns files into model batches.

`sequences`, `labels`, `splits`, `split_guard`, `conditioning`,
`condition_taxonomy`, `pairs` and `negatives` each solve one problem and none
of them calls another, so there was no object a training loop could ask for a
batch. This is that object. It adds no new policy: every threshold comes from
one of those modules, and each choice between their options is recorded in
`SplitDecision` / `Pipeline.report` with the number that settled it.

Four decisions are made here.

The split column is `C_block5` at `conservative` resolution. Of the three
columns, C_block5 has the fewest homology clusters spanning parts (367 of
22 159, 10 598 promoter rows, against 594/14 941 and 1 555/32 020) and needs
the fewest orthogroups moved by the repair, leaving the largest share of the
decision with the published artefact. A_og is refused outright, because one
held-out orthogroup leaves six byte-identical copies of the pks1 promoter in
training and its `majority` repair empties spy altogether. `conservative` is
chosen because its priority spy > test > val > train cannot move a cluster
INTO training, which `majority` can; the price is 6 583 promoters of training
material.

The condition vector describes the cell the label describes. `mu` is a mean
over the kept samples of one (orthogroup, genome), so the taxonomy channel
carries that cell's empirical composition -- the mean of the per-sample
taxonomy vectors over exactly the samples that entered `mu`. Handing such a
row one randomly drawn sample's vector would train the model to read a
variable the label does not depend on, and it would spend the unknown token
on "averaged" instead of on "no samples". The consequence is reported rather
than hidden: a one-hot condition at inference is off the training marginal,
and `CellConditioning.concentration()` says how far.

Pairs are drawn inside a genome cell, not inside a sample column, and
`pairs.PairConfig` raises on `genome_sample`. The missingness mask is a
property of (orthogroup, genome) -- uniform across sample columns in 1.0000 of
sampled blocks -- so a per-column cell yields no new pair at all, while the
median single-column sd is 0.6104 against 0.0992 for the se of the genome
mean, a 6.16x ratio that the rejection threshold scales with. The cell does
narrow to (genome, category) with probability `p_category`.

Negatives are filtered against a k-mer index built from every part of the
split. Rejecting a candidate removes information and never adds any, so there
is nothing to leak, whereas an index built from training alone would pass
flank windows containing a validation promoter verbatim. Both candidate
classes need the filter: 2.51 % of long-enough gene bodies contain a promoter
31-mer, 12x below flank_r but not zero.

Two things this module refuses to do. It does not invent a value for a
channel whose cache is absent: with no frozen-LM and no DNA-LM cache in this
repo, `text` and `gene_context` resolve to `available=False` and the model
answers with its learned unknown token, where a hashed stand-in would report
a share of the context while carrying no information. And it does not pretend
the taxonomy channel reaches the model: `ConditionEncoder` emits a fixed
four-token context and that order is a checkpoint contract, so
`Pipeline.channel_consumers` checks the real forward signature at runtime and
says so.
"""
from __future__ import annotations

import collections
import gzip
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .condition_taxonomy import TAXONOMY_PATH, load_taxonomy
from .conditioning import (
    CHANNEL_ORDER,
    ChannelSet,
    ChannelValues,
    EmbeddingCache,
    TaxonomyEncoder,
    default_channels,
)
from .kmer_index import N_CODE, GenomeKmerIndex, encode_codes
from .labels import (
    SamplePolicy,
    SampleTable,
    Targets,
    build_targets,
    load_samples,
    read_matrix,
)
from .negatives import NegativeConfig, build_pool
from .pairs import (
    CategoryStats,
    PairConfig,
    PairIndex,
    PairSampler,
    build_pair_index,
    category_stats_from_matrix,
    samples_per_genome,
)
from .sequences import NUCLEOTIDE_ORDER
from .split_guard import (
    CLUSTER_ROWS,
    CONSERVATIVE_PRIORITY,
    SplitAssignment,
    build_split,
)
from .splits import SPLIT_NAMES

#: Channel order is a contract; this is the order the pipeline resolves in.
assert CHANNEL_ORDER == ("species", "text", "taxonomy", "gene_context")
#: The promoter contract. 238 460 of 238 831 rows are exactly this long.
PROMOTER_LENGTH = 800
#: Chosen split; the measurement is in the module docstring.
SPLIT_COLUMN = "C_block5"
SPLIT_RESOLUTION = "conservative"
#: Sequence groups, as spelled in sequences.tsv.gz (note: `Promotor`).
GROUP_PROMOTER = "Promotor"
GROUP_GENE = "Gene"
GROUP_FLANK_R = "FlankRight"
GROUP_FLANK_L = "FlankLeft"

_PART_INDEX = {name: i for i, name in enumerate(SPLIT_NAMES)}


# Paths
@dataclass
class CorpusPaths:
    """Every file the pipeline reads, resolved once and reported.

    `default()` prefers the unpacked tree the environment names and falls back
    to the in-repo mirror. The fallback is reported in `root_used`, never
    silent: the two trees hold byte-identical files today and a divergence
    would otherwise be invisible.
    """

    sequences: str
    combat: str
    samples: str
    splits: str
    clusters: str
    taxonomy: str = TAXONOMY_PATH
    text_cache: str = ""
    flank_cache: str = ""
    root_used: str = ""

    @staticmethod
    def repo_root() -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "..", ".."))

    @classmethod
    def default(cls, data_root: Optional[str] = None) -> "CorpusPaths":
        repo = cls.repo_root()
        candidates = [data_root] if data_root else []
        candidates += [
            os.path.abspath(os.path.join(
                repo, "..", "HYperbolicPromoterProgrammingEngine",
                "final_datasets")),
            os.path.join(repo, "data", "final_datasets"),
        ]
        root = None
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "sequences")):
                root = c
                break
        if root is None:
            raise FileNotFoundError(
                "no final_datasets tree with a sequences/ directory among %r"
                % (candidates,))
        # The one canonical sample table (see hyppe/data/metadata.py). It is
        # bit-identical to the copy this used to read in every column
        # `labels.load_samples` touches, so pointing here changes no label and
        # no number; it only stops the dataset depending on which of the four
        # copies in circulation happened to be present.
        samples = os.path.join(repo, "data", "samples_metadata.tsv")
        if not os.path.exists(samples):
            raise FileNotFoundError(
                "canonical sample metadata not found at %s -- build it with "
                "`python src/scripts/build_samples_metadata.py`. (Refusing to "
                "fall back to a superseded copy: the four copies in circulation "
                "disagree on `condition` in 2300 of 2332 rows.)" % samples)
        return cls(
            sequences=os.path.join(root, "sequences", "sequences.tsv.gz"),
            combat=os.path.join(root, "og_scope", "combat.tsv.gz"),
            samples=samples,
            splits=os.path.join(repo, "data", "splits", "SPLIT_FINAL.tsv"),
            clusters=os.path.join(repo, CLUSTER_ROWS),
            taxonomy=TAXONOMY_PATH,
            root_used=root)

    def missing(self) -> List[str]:
        out = []
        for name in ("sequences", "combat", "samples", "splits", "clusters",
                     "taxonomy"):
            p = getattr(self, name)
            if not p or not os.path.exists(p):
                out.append("%s=%s" % (name, p))
        return out

    def report(self) -> Dict[str, object]:
        return {"root_used": self.root_used,
                "samples": os.path.relpath(self.samples, self.repo_root()),
                "missing": self.missing()}


# One-hot
def one_hot_from_codes(codes: np.ndarray, n_value: float = 0.25,
                       dtype=np.float32) -> np.ndarray:
    """uint8 codes [B, L] -> [B, 4, L] in ACGT order. Code 4 becomes uniform.

    N is not an error and must not be folded into A: a uniform column is the
    only encoding that adds no information. The channel order is
    `sequences.NUCLEOTIDE_ORDER` = "ACGT", not the A,C,T,G of the sibling
    track.
    """
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)
    b, L = codes.shape
    out = np.zeros((b, 4, L), dtype=dtype)
    idx = codes.astype(np.int64)
    known = idx < 4
    rows, cols = np.nonzero(known)
    out[rows, idx[known], cols] = 1.0
    if (~known).any():
        r2, c2 = np.nonzero(~known)
        out[r2, :, c2] = n_value
    return out


# Corpus
@dataclass
class GroupStore:
    """One sequence group as a code matrix plus its keys."""

    group: str
    ids: List[str]
    genome: np.ndarray            # int16, index into `genomes`
    og: np.ndarray                # int32, index into `ogs`
    gene_id: List[str]
    codes: np.ndarray             # uint8 [n, width], padded with N_CODE
    length: np.ndarray            # int32, true length before padding
    genomes: List[str]
    ogs: List[str]
    n_rows_seen: int = 0
    n_dropped_length: int = 0
    n_duplicate_ids: int = 0

    def __len__(self) -> int:
        return len(self.ids)

    def seq(self, i: int) -> np.ndarray:
        return self.codes[i, :int(self.length[i])]

    def report(self) -> Dict[str, object]:
        return {"group": self.group, "rows_in_file": self.n_rows_seen,
                "kept": len(self.ids), "dropped_by_length":
                self.n_dropped_length, "duplicate_ids": self.n_duplicate_ids,
                "width": int(self.codes.shape[1]) if len(self.ids) else 0,
                "megabytes": round(self.codes.nbytes / 1e6, 1)}


def load_groups(path: str,
                groups: Mapping[str, Optional[int]],
                genomes: Optional[Set[str]] = None,
                exact_length: Optional[Mapping[str, int]] = None,
                max_rows: Optional[int] = None,
                per_genome_limit: Optional[int] = None,
                seed: int = 7898) -> Dict[str, GroupStore]:
    """One streaming pass over sequences.tsv.gz. Codes, not strings.

    `groups` maps a GROUP value to the width to store (None = the group's own
    maximum). `exact_length` names groups where a row of a different length is
    dropped and counted rather than padded: padding a 493 bp promoter to 800
    would invent 307 bases the model cannot tell from real ones.

    `per_genome_limit` caps rows per (group, genome) at read time, not after
    -- the two candidate groups at 2 000 bp across twenty genomes are 956 MB,
    against 15 MB at a limit of 200, and capping afterwards would allocate the
    956 MB first. The cap is a seeded reservoir, not a prefix: the file is
    sorted by genome coordinate, so the first N genes of a strain are one end
    of one contig rather than a sample.
    """
    want = dict(groups)
    exact = dict(exact_length or {})
    buf: Dict[str, List[Tuple[str, str, str, str, np.ndarray]]] = {
        g: [] for g in want}
    seen_rows = collections.Counter()
    dropped = collections.Counter()
    seen_cell = collections.Counter()
    # Row indices of each (group, genome) inside `buf[group]`, so a reservoir
    # replacement is O(1) instead of a scan of the whole buffer.
    cell_rows: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
    rng = np.random.default_rng(seed)
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        need = ("ID", "GENOME", "SEQUENCE", "GROUP", "OG_ID", "GENE_ID")
        missing = [c for c in need if c not in header]
        if missing:
            raise ValueError("%s lacks columns %r (has %r)"
                             % (path, missing, header))
        ix = {c: header.index(c) for c in need}
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            g = p[ix["GROUP"]]
            if g not in want:
                continue
            gen = p[ix["GENOME"]]
            if genomes is not None and gen not in genomes:
                continue
            seen_rows[g] += 1
            slot = -1
            if per_genome_limit is not None:
                n_seen = seen_cell[(g, gen)]
                seen_cell[(g, gen)] = n_seen + 1
                if n_seen < per_genome_limit:
                    slot = -1                      # append
                else:
                    j = int(rng.integers(0, n_seen + 1))
                    if j >= per_genome_limit:
                        continue                   # reservoir rejects it
                    slot = j
            raw = p[ix["SEQUENCE"]]
            want_len = exact.get(g)
            if want_len is not None and len(raw) != want_len:
                dropped[g] += 1
                continue
            # `.upper()` is mandatory: 7.26 % of promoter bases are soft
            # masked and a case-sensitive reader loses every one of them.
            c = encode_codes(raw.upper())
            cap = want[g]
            if cap is not None and c.shape[0] > cap:
                c = c[:cap]
            item = (p[ix["ID"]], gen, p[ix["OG_ID"]], p[ix["GENE_ID"]], c)
            if slot < 0:
                cell_rows[(g, gen)].append(len(buf[g]))
                buf[g].append(item)
            else:
                buf[g][cell_rows[(g, gen)][slot]] = item
            n += 1
            if max_rows is not None and n >= max_rows:
                break

    genome_names: List[str] = []
    og_names: List[str] = []
    gmap: Dict[str, int] = {}
    omap: Dict[str, int] = {}
    out: Dict[str, GroupStore] = {}
    for g, rows in buf.items():
        width = max((r[4].shape[0] for r in rows), default=0)
        codes = np.full((len(rows), width), N_CODE, dtype=np.uint8)
        ids: List[str] = []
        gene_ids: List[str] = []
        gen_ix = np.zeros(len(rows), dtype=np.int16)
        og_ix = np.zeros(len(rows), dtype=np.int32)
        length = np.zeros(len(rows), dtype=np.int32)
        seen_id = collections.Counter()
        for i, (sid, gen, og, gid, c) in enumerate(rows):
            ids.append(sid)
            gene_ids.append(gid)
            seen_id[sid] += 1
            if gen not in gmap:
                gmap[gen] = len(genome_names)
                genome_names.append(gen)
            if og not in omap:
                omap[og] = len(og_names)
                og_names.append(og)
            gen_ix[i] = gmap[gen]
            og_ix[i] = omap[og]
            length[i] = c.shape[0]
            codes[i, :c.shape[0]] = c
        out[g] = GroupStore(
            group=g, ids=ids, genome=gen_ix, og=og_ix, gene_id=gene_ids,
            codes=codes, length=length, genomes=genome_names, ogs=og_names,
            n_rows_seen=int(seen_rows[g]),
            n_dropped_length=int(dropped[g]),
            n_duplicate_ids=sum(1 for v in seen_id.values() if v > 1))
    return out


class PromoterIdLeak(AssertionError):
    """One promoter id carries two different parts of the split."""


def id_part_clashes(ids: Sequence[str], part: np.ndarray
                    ) -> Dict[str, List[int]]:
    """promoter id -> the distinct parts it appears under, when > 1.

    The split is defined on orthogroups, so the orthogroup-level guard is
    satisfied by construction and cannot see this. On the real corpus 950 of
    238 460 promoter ids are duplicated, 5 of them under two different
    orthogroups, and 1 of those lands in two parts with different sequences:
    one row, invisible to every orthogroup-level check and to every metric.
    """
    first: Dict[str, int] = {}
    out: Dict[str, List[int]] = {}
    for i, sid in enumerate(ids):
        p = int(part[i])
        if sid not in first:
            first[sid] = p
        elif first[sid] != p:
            cur = out.setdefault(sid, [first[sid]])
            if p not in cur:
                cur.append(p)
    return out


def assert_ids_disjoint(ids: Sequence[str], part: np.ndarray
                        ) -> Dict[str, object]:
    """Raise unless every promoter id lies in exactly one part."""
    bad = id_part_clashes(ids, part)
    if bad:
        ex = sorted(bad.items())[:5]
        raise PromoterIdLeak(
            "%d promoter ids carry more than one part of the split: %s. The "
            "split is defined on orthogroups, so no orthogroup-level guard "
            "can see this; repair it with the same conservative priority "
            "used for clusters (spy > test > val > train), which cannot move "
            "a row INTO training."
            % (len(bad), [(k, [SPLIT_NAMES[q] for q in v]) for k, v in ex]))
    return {"n_ids": len(set(ids)), "n_clashing": 0}


@dataclass
class PromoterCorpus:
    """Promoters, their split part, and the (orthogroup, genome) index."""

    store: GroupStore
    part: np.ndarray                       # int8, index into SPLIT_NAMES
    split: SplitAssignment
    example_row: Dict[Tuple[str, str], int] = field(default_factory=dict)
    n_unassigned: int = 0
    n_multicopy_cells: int = 0
    id_clashes_repaired: Dict[str, List[str]] = field(default_factory=dict)
    n_ids_in_two_orthogroups: int = 0

    @property
    def n(self) -> int:
        return len(self.store)

    def rows_of_part(self, part: str) -> np.ndarray:
        return np.flatnonzero(self.part == _PART_INDEX[part])

    def report(self) -> Dict[str, object]:
        per = collections.Counter()
        for p in self.part:
            per[SPLIT_NAMES[p] if p >= 0 else "unassigned"] += 1
        tot = max(sum(per.values()), 1)
        return {
            "sequences": self.store.report(),
            "promoters_per_part": dict(per),
            "fractions_per_part": {k: round(v / tot, 4)
                                   for k, v in per.items()},
            "orthogroups_without_split": self.n_unassigned,
            "cells_with_more_than_one_promoter": self.n_multicopy_cells,
            "example_cells": len(self.example_row),
            "promoter_ids_in_two_orthogroups": self.n_ids_in_two_orthogroups,
            "promoter_id_clashes_repaired": self.id_clashes_repaired,
        }


def promoter_cache_path(paths: CorpusPaths) -> str:
    return os.path.join(CorpusPaths.repo_root(), "cache", "promoters_800.npz")


def _cache_key(path: str) -> str:
    """Size and mtime of the source. Not a hash: hashing 281 MB on every run
    costs more than the read it is meant to save."""
    s = os.stat(path)
    return "%d:%d" % (s.st_size, int(s.st_mtime))


def save_promoter_cache(store: GroupStore, source: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, codes=store.codes, length=store.length, genome=store.genome,
        og=store.og, ids=np.asarray(store.ids), gene_id=np.asarray(
            store.gene_id), genomes=np.asarray(store.genomes),
        ogs=np.asarray(store.ogs),
        meta=np.asarray([_cache_key(source), str(store.n_rows_seen),
                         str(store.n_dropped_length),
                         str(store.n_duplicate_ids)]))
    return path


def load_promoter_cache(source: str, path: str) -> Optional[GroupStore]:
    """Return the cached store, or None if it is absent or stale. Never a
    partial read: a cache that half-matches is worse than no cache."""
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = [str(x) for x in z["meta"]]
        if not meta or meta[0] != _cache_key(source):
            return None
        return GroupStore(
            group=GROUP_PROMOTER, ids=[str(x) for x in z["ids"]],
            genome=z["genome"], og=z["og"],
            gene_id=[str(x) for x in z["gene_id"]], codes=z["codes"],
            length=z["length"], genomes=[str(x) for x in z["genomes"]],
            ogs=[str(x) for x in z["ogs"]], n_rows_seen=int(meta[1]),
            n_dropped_length=int(meta[2]), n_duplicate_ids=int(meta[3]))


def build_corpus(paths: CorpusPaths, split: SplitAssignment,
                 max_rows: Optional[int] = None,
                 cache: Optional[str] = None) -> PromoterCorpus:
    """Load promoters at exactly 800 bp and attach the split part.

    `cache` names an .npz holding the code matrix, keyed on the size and mtime
    of sequences.tsv.gz and ignored rather than repaired when those differ: a
    cache silently serving an older corpus would make every number below
    unattributable.
    """
    if cache and max_rows is None:
        st = load_promoter_cache(paths.sequences, cache)
        if st is not None:
            return _attach_split(st, split)
    stores = load_groups(paths.sequences, {GROUP_PROMOTER: PROMOTER_LENGTH},
                         exact_length={GROUP_PROMOTER: PROMOTER_LENGTH},
                         max_rows=max_rows)
    st = stores[GROUP_PROMOTER]
    if cache and max_rows is None:
        save_promoter_cache(st, paths.sequences, cache)
    return _attach_split(st, split)


def _attach_split(st: GroupStore, split: SplitAssignment) -> PromoterCorpus:
    """Split part per promoter row, plus the two repairs and their guards."""
    part = np.full(len(st), -1, dtype=np.int8)
    unassigned = 0
    example: Dict[Tuple[str, str], int] = {}
    multi = 0
    for i in range(len(st)):
        og = st.ogs[int(st.og[i])]
        p = split.of(og)
        if p is None:
            unassigned += 1
        else:
            part[i] = _PART_INDEX[p]
        key = (og, st.genomes[int(st.genome[i])])
        if key in example:
            multi += 1
        else:
            example[key] = i
    # A promoter id under two orthogroups can land in two parts. Repaired with
    # the same conservative priority as the cluster repair -- one mechanism,
    # and one that cannot move a row into training -- then guarded. One id on
    # this corpus.
    by_id: Dict[str, List[int]] = collections.defaultdict(list)
    for i in range(len(st)):
        by_id[st.ids[i]].append(i)
    two_og = sum(1 for rows in by_id.values()
                 if len({int(st.og[r]) for r in rows}) > 1)
    clashes = id_part_clashes(st.ids, part)
    repaired: Dict[str, List[str]] = {}
    for sid, parts_seen in clashes.items():
        winner = next(_PART_INDEX[q] for q in CONSERVATIVE_PRIORITY
                      if _PART_INDEX[q] in parts_seen)
        repaired[sid] = ([SPLIT_NAMES[q] for q in parts_seen]
                         + ["->", SPLIT_NAMES[winner]])
        for r in by_id[sid]:
            part[r] = winner
    assert_ids_disjoint(st.ids, part)          # after the repair, not before

    return PromoterCorpus(store=st, part=part, split=split,
                          example_row=example, n_unassigned=unassigned,
                          n_multicopy_cells=multi,
                          id_clashes_repaired=repaired,
                          n_ids_in_two_orthogroups=two_og)


# Conditioning
@dataclass
class CellConditioning:
    """Per-(genome, category) condition vectors, aggregated like the label.

    `taxonomy[g, k]` is the mean of the per-sample taxonomy vectors over the
    kept samples of genome `g` in category `k`; `taxonomy_marginal[g]` is the
    same mean over all kept samples of the genome. Category -1 means "the
    label is the genome mean", and the marginal is then the right vector: the
    target averages over conditions, so the condition input must too.
    """

    genomes: List[str]
    categories: Tuple[str, ...]
    taxonomy: np.ndarray                # float32 [n_gen, n_cat, dim]
    taxonomy_present: np.ndarray        # bool [n_gen, n_cat]
    taxonomy_marginal: np.ndarray       # float32 [n_gen, dim]
    marginal_present: np.ndarray        # bool [n_gen]
    n_samples: np.ndarray               # int32 [n_gen, n_cat]
    encoder: TaxonomyEncoder = None

    def vector(self, genome_ix: np.ndarray, category_ix: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(value [B, dim], present [B]) for a batch of cells."""
        g = np.asarray(genome_ix, dtype=np.int64)
        k = np.asarray(category_ix, dtype=np.int64)
        out = np.where((k < 0)[:, None],
                       self.taxonomy_marginal[g],
                       self.taxonomy[g, np.maximum(k, 0)])
        pres = np.where(k < 0, self.marginal_present[g],
                        self.taxonomy_present[g, np.maximum(k, 0)])
        return out.astype(np.float32), pres.astype(bool)

    def concentration(self) -> Dict[str, float]:
        """How far a one-hot condition at inference is off the training
        marginal. Reported because the design target will be given as one."""
        m = self.taxonomy_marginal[self.marginal_present]
        if m.size == 0:
            return {}
        par = m[:, :len(self.categories)]
        return {
            "max_parent_component_mean": float(par.max(axis=1).mean()),
            "max_parent_component_min": float(par.max(axis=1).min()),
            "max_parent_component_max": float(par.max(axis=1).max()),
            "parent_entropy_bits_mean": float(np.mean([
                -np.sum(r[r > 0] * np.log2(r[r > 0])) for r in par])),
        }


def build_cell_conditioning(samples: SampleTable, taxonomy: TaxonomyEncoder,
                            genomes: Sequence[str],
                            categories: Tuple[str, ...]) -> CellConditioning:
    """Aggregate per-sample taxonomy vectors to the cells the labels use."""
    tax_table = load_taxonomy(taxonomy.path)
    mat, avail = taxonomy.encode(samples.sample_ids, tax_table)
    gi = {g: i for i, g in enumerate(genomes)}
    n_gen, n_cat, dim = len(genomes), len(categories), taxonomy.dim
    acc = np.zeros((n_gen, n_cat, dim), dtype=np.float64)
    cnt = np.zeros((n_gen, n_cat), dtype=np.int32)
    acc_m = np.zeros((n_gen, dim), dtype=np.float64)
    cnt_m = np.zeros(n_gen, dtype=np.int32)
    for i, sid in enumerate(samples.sample_ids):
        if not samples.keep[i] or not avail[i]:
            continue
        j = gi.get(samples.genome[i])
        if j is None:
            continue
        acc_m[j] += mat[i]
        cnt_m[j] += 1
        for k in range(n_cat):
            if samples.category[i, k] > 0.0:
                acc[j, k] += mat[i]
                cnt[j, k] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        tax = (acc / np.maximum(cnt, 1)[:, :, None]).astype(np.float32)
        marg = (acc_m / np.maximum(cnt_m, 1)[:, None]).astype(np.float32)
    tax[cnt == 0] = 0.0
    marg[cnt_m == 0] = 0.0
    return CellConditioning(
        genomes=list(genomes), categories=tuple(categories), taxonomy=tax,
        taxonomy_present=cnt > 0, taxonomy_marginal=marg,
        marginal_present=cnt_m > 0, n_samples=cnt, encoder=taxonomy)


@dataclass
class ConditionResolver:
    """One mechanism wired to the four sources, with no per-channel code.

    Every channel goes through `ChannelSet.draw`, which owns the dropout and
    the availability bookkeeping. All that differs between channels is where
    the value comes from, and that is a lookup here rather than a branch
    inside the mechanism.
    """

    channels: ChannelSet
    cells: CellConditioning
    genome_index: Dict[str, int]
    registry_index: Dict[str, int]
    text_cache: Optional[EmbeddingCache] = None
    flank_cache: Optional[EmbeddingCache] = None
    seed: int = 7898

    def draw(self, genome_ix: np.ndarray, category_ix: np.ndarray,
             promoter_ids: Sequence[str], group: Optional[np.ndarray] = None,
             training: bool = True,
             rng: Optional[np.random.Generator] = None
             ) -> Dict[str, ChannelValues]:
        """Resolve all four channels for one batch, sharing one dropout draw.

        `group` is optional only in form: the siamese head scores a pair under
        one condition, and a per-row draw breaks the pair with probability
        2p(1-p) = 18 % at p = 0.1. The pair collate passes it; a plain
        promoter batch may leave it None, since each row is its own group.
        """
        n = len(promoter_ids)
        rng = rng or np.random.default_rng(self.seed)
        g = np.asarray(genome_ix, dtype=np.int64)
        spec = self.channels.by_name()
        out: Dict[str, ChannelValues] = {}

        # species: one-hot over the registry; unknown is absent, never index 0
        sp_dim = spec["species"].dim
        sp = np.zeros((n, sp_dim), dtype=np.float32)
        sp_ok = np.zeros(n, dtype=bool)
        for i in range(n):
            name = self.cells.genomes[int(g[i])]
            j = self.registry_index.get(name)
            if j is not None and j < sp_dim:
                sp[i, j] = 1.0
                sp_ok[i] = True
        out["species"] = self.channels.draw("species", sp, sp_ok, group,
                                            training, rng)

        # text: frozen LM, cache or nothing, no stand-in. The key is the cell
        # rather than the genome or the sample, because the label is a cell
        # aggregate; `cell_key` is the contract the precompute script writes.
        t_dim = spec["text"].dim
        if self.text_cache is not None:
            keys = [self.cell_key(int(g[i]), int(category_ix[i]))
                    for i in range(n)]
            tv, tok = self.text_cache.lookup(keys)
        else:
            tv, tok = np.zeros((n, t_dim), np.float32), np.zeros(n, bool)
        out["text"] = self.channels.draw("text", tv, tok, group, training, rng)

        # taxonomy: the cell's empirical composition
        tx, tx_ok = self.cells.vector(g, category_ix)
        out["taxonomy"] = self.channels.draw("taxonomy", tx, tx_ok, group,
                                             training, rng)

        # gene context: frozen DNA-LM over both flanks, cache or nothing. The
        # channel is 2 * d_dna wide and holds [left | right] in that order:
        # `ConditionEncoder` wants two tokens, and one vector duplicated into
        # both would be two tokens carrying one number.
        d_dim = spec["gene_context"].dim
        if d_dim % 2:
            raise ValueError(
                "gene_context is %d wide; it must be 2 * d_dna so that "
                "[left | right] splits exactly in half" % d_dim)
        if self.flank_cache is not None:
            fv, fok = self.flank_cache.lookup(list(promoter_ids))
        else:
            fv, fok = np.zeros((n, d_dim), np.float32), np.zeros(n, bool)
        out["gene_context"] = self.channels.draw("gene_context", fv, fok,
                                                 group, training, rng)
        return out

    def cell_key(self, genome_ix: int, category_ix: int) -> str:
        """Cache key of a conditioning cell. `*` is the genome marginal."""
        g = self.cells.genomes[genome_ix]
        if category_ix < 0:
            return "%s|*" % g
        return "%s|%s" % (g, self.cells.categories[category_ix])


# Dataset
@dataclass
class Batch:
    """What a training step receives. Numpy; `to_torch` makes it tensors."""

    x: np.ndarray                    # float32 [B, 4, 800]
    row: np.ndarray                  # int64 [B] row in the promoter store
    species_id: np.ndarray           # int64 [B]
    genome_ix: np.ndarray            # int64 [B]
    og_ix: np.ndarray                # int64 [B]
    category_ix: np.ndarray          # int64 [B], -1 = genome marginal
    group: np.ndarray                # int64 [B] shared-condition group
    mu: np.ndarray                   # float32 [B], NaN where unlabelled
    sigma: np.ndarray                # float32 [B], NaN where n_obs < 2
    weight: np.ndarray               # float32 [B] sqrt(n_obs)
    regression_mask: np.ndarray      # bool [B]
    sigma_mask: np.ndarray           # bool [B]
    is_negative: np.ndarray          # bool [B]
    channels: Dict[str, ChannelValues] = field(default_factory=dict)
    cat_targets: Optional[np.ndarray] = None   # float32 [B, n_cat]
    pair_a: Optional[np.ndarray] = None
    pair_b: Optional[np.ndarray] = None
    pair_y: Optional[np.ndarray] = None
    pair_margin: Optional[np.ndarray] = None
    pair_delta: Optional[np.ndarray] = None
    pair_is_negative: Optional[np.ndarray] = None

    def size(self) -> int:
        return int(self.x.shape[0])

    def shapes(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = (tuple(v.shape), str(v.dtype))
        for k, v in self.channels.items():
            out["channel:" + k] = (tuple(v.value.shape), str(v.value.dtype),
                                   round(v.fraction_present(), 4))
        return out

    def to_torch(self, device: str = "cpu") -> Dict[str, object]:
        """Keyword arguments for `HyppeModel.forward`, plus the loss targets.

        `text_emb` / `flank_l` / `flank_r` are omitted when their channel has
        no value for any row, because `ConditionEncoder` reads a missing key
        as "use the learned unknown token". An all-zero tensor would be
        something else: zero is a point in the embedding space, the unknown
        token is a trained one.
        """
        import torch

        def t(a, dtype=None):
            x = torch.from_numpy(np.ascontiguousarray(a))
            return (x if dtype is None else x.to(dtype)).to(device)

        out: Dict[str, object] = {
            "x": t(self.x, torch.float32),
            "species_id": t(self.species_id, torch.long),
            "pair_group": t(self.group, torch.long),
            "mu": t(self.mu, torch.float32),
            "sigma": t(self.sigma, torch.float32),
            "weight": t(self.weight, torch.float32),
            "regression_mask": t(self.regression_mask, torch.bool),
            "sigma_mask": t(self.sigma_mask, torch.bool),
            "is_negative": t(self.is_negative, torch.bool),
        }
        text = self.channels.get("text")
        if text is not None and bool(text.present.any()):
            out["text_emb"] = t(text.value, torch.float32)
        flank = self.channels.get("gene_context")
        if flank is not None and bool(flank.present.any()):
            half = flank.value.shape[1] // 2
            out["flank_l"] = t(flank.value[:, :half], torch.float32)
            out["flank_r"] = t(flank.value[:, half:], torch.float32)
        tax = self.channels.get("taxonomy")
        if tax is not None:
            out["taxonomy"] = t(tax.value, torch.float32)
            out["taxonomy_present"] = t(tax.present, torch.bool)
        if self.cat_targets is not None:
            out["cat_targets"] = t(self.cat_targets, torch.float32)
        for name in ("pair_a", "pair_b", "pair_y", "pair_margin",
                     "pair_delta", "pair_is_negative"):
            v = getattr(self, name)
            if v is not None:
                out[name] = t(v)
        return out

    @staticmethod
    def forward_kwargs(tensors: Mapping[str, object], model
                       ) -> Dict[str, object]:
        """The subset of `to_torch()` that `model.forward` actually accepts.

        `to_torch()` returns the forward inputs and the loss targets in one
        dict, because they are produced together and splitting them at the
        source would make two objects that can disagree. Filtering by the real
        signature drops an unknown key here, visibly, rather than letting a
        `**kwargs` somewhere swallow it.
        """
        import inspect
        params = inspect.signature(model.forward).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values()):
            raise TypeError(
                "model.forward takes **kwargs, so filtering by signature "
                "cannot tell an accepted key from an ignored one. Name the "
                "inputs explicitly in forward().")
        return {k: v for k, v in tensors.items() if k in params}


class PromoterDataset:
    """Rows of one split part, with labels joined and channels resolved."""

    def __init__(self, corpus: PromoterCorpus, targets: Targets,
                 resolver: ConditionResolver, part: str,
                 labelled_only: bool = True, seed: int = 7898) -> None:
        if part not in SPLIT_NAMES:
            raise ValueError("part must be one of %r, got %r"
                             % (SPLIT_NAMES, part))
        self.corpus = corpus
        self.targets = targets
        self.resolver = resolver
        self.part = part
        self.labelled_only = bool(labelled_only)
        self.rng = np.random.default_rng(seed)
        st = corpus.store
        rows = corpus.rows_of_part(part)
        mu = np.full(len(st), np.nan, dtype=np.float32)
        sd = np.full(len(st), np.nan, dtype=np.float32)
        nob = np.zeros(len(st), dtype=np.int32)
        ok = np.zeros(len(st), dtype=bool)
        for r in rows:
            key = (st.ogs[int(st.og[r])], st.genomes[int(st.genome[r])])
            j = targets.residual_index.get(key)
            if j is None:
                continue
            mu[r] = targets.mu[j]
            sd[r] = targets.sigma[j]
            nob[r] = targets.n_obs[j]
            ok[r] = bool(targets.valid[j])
        self.mu, self.sigma, self.n_obs, self.labelled = mu, sd, nob, ok
        self.rows = rows[ok[rows]] if labelled_only else rows
        # Genome index of the promoter store -> index of the conditioning
        # tables, which are ordered by the label genomes. A genome present in
        # the orthology but absent from the expression metadata maps to -1 and
        # is reported rather than silently dropped.
        self._gmap = np.array(
            [resolver.genome_index.get(g, -1) for g in st.genomes],
            dtype=np.int64)
        self.n_unmapped_genome = int((self._gmap[st.genome[self.rows]] < 0
                                      ).sum()) if self.rows.size else 0

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def batch(self, rows: Sequence[int], training: bool = True,
              category_ix: Optional[np.ndarray] = None,
              group: Optional[np.ndarray] = None) -> Batch:
        st = self.corpus.store
        r = np.asarray(rows, dtype=np.int64)
        n = int(r.shape[0])
        gx = self._gmap[st.genome[r]]
        if (gx < 0).any():
            bad = sorted({st.genomes[int(st.genome[i])]
                          for i in r[gx < 0]})
            raise KeyError(
                "genomes %r have promoters but no expression metadata, so "
                "they have no conditioning cell. They belong to the "
                "self-supervised track; filter them out before asking for a "
                "labelled batch." % (bad,))
        cat = (np.full(n, -1, dtype=np.int64) if category_ix is None
               else np.asarray(category_ix, dtype=np.int64))
        grp = np.arange(n, dtype=np.int64) if group is None else np.asarray(
            group, dtype=np.int64)
        ids = [st.ids[int(i)] for i in r]
        ch = self.resolver.draw(gx, cat, ids, grp, training, self.rng)
        n_cat = len(self.resolver.cells.categories)
        tax = ch["taxonomy"].value
        return Batch(
            x=one_hot_from_codes(st.codes[r]),
            row=r, species_id=gx, genome_ix=gx,
            og_ix=st.og[r].astype(np.int64), category_ix=cat, group=grp,
            mu=self.mu[r], sigma=self.sigma[r],
            weight=np.sqrt(np.maximum(self.n_obs[r], 0)).astype(np.float32),
            regression_mask=self.labelled[r],
            sigma_mask=self.labelled[r] & (self.n_obs[r] >= 2),
            is_negative=np.zeros(n, dtype=bool),
            channels=ch,
            cat_targets=tax[:, :n_cat].copy())

    def sample_batch(self, size: int, training: bool = True) -> Batch:
        idx = self.rng.choice(self.rows, size=size, replace=False)
        return self.batch(idx, training=training)

    def report(self) -> Dict[str, object]:
        return {
            "part": self.part,
            "promoters_in_part": int((self.corpus.part
                                      == _PART_INDEX[self.part]).sum()),
            "labelled": int(self.labelled[
                self.corpus.rows_of_part(self.part)].sum()),
            "usable_rows": len(self),
            "labelled_only": self.labelled_only,
            "genomes_without_expression_metadata": self.n_unmapped_genome,
        }


# Pairs
def build_pair_sampler(corpus: PromoterCorpus, targets: Targets,
                       samples: SampleTable, combat_path: str,
                       part: str = "train",
                       cfg: Optional[PairConfig] = None,
                       negatives_by_genome: Optional[
                           Mapping[str, Sequence[int]]] = None
                       ) -> Tuple[PairSampler, PairIndex]:
    """A sampler restricted to one part of the split.

    The restriction is to hand `build_pair_index` only the example rows of
    that part: `valid` means labelled and having a promoter row, so a cell
    whose promoter sits elsewhere cannot be drawn. One mechanism, not a second
    filter that could disagree with the first.
    """
    cfg = cfg or PairConfig()
    st = corpus.store
    keep = _PART_INDEX[part]
    ex = {k: v for k, v in corpus.example_row.items()
          if corpus.part[v] == keep}
    cats: Optional[CategoryStats] = None
    if cfg.cell_mode == "genome_category":
        og_ids, cols, mat = read_matrix(combat_path)
        gens = sorted(set(targets.genomes))
        cats = category_stats_from_matrix(og_ids, cols, mat, samples, gens)
    idx = build_pair_index(targets, ex, samples_per_genome(samples),
                           negatives_by_genome=negatives_by_genome,
                           categories=cats)
    return PairSampler(idx, cfg), idx


def pair_batch_to_batch(pb, corpus: PromoterCorpus,
                        resolver: ConditionResolver,
                        rng: Optional[np.random.Generator] = None,
                        negative_codes: Optional[np.ndarray] = None,
                        training: bool = True) -> Batch:
    """Turn a `pairs.PairBatch` into a model batch.

    The conditioning group is the pair, not the row: both halves must see the
    same condition or the siamese comparison is between two different
    questions. `pairs` guarantees they share a cell; this is where that
    guarantee becomes one dropout draw.
    """
    st = corpus.store
    n = pb.n_elements()
    grp = np.full(n, -1, dtype=np.int64)
    for i in range(pb.n_pairs()):
        grp[pb.pair_a[i]] = i
        grp[pb.pair_b[i]] = i
    loose = grp < 0
    if loose.any():                       # elements in no pair get own group
        grp[loose] = np.arange(pb.n_pairs(), pb.n_pairs() + int(loose.sum()))

    rows = pb.example_row.astype(np.int64)
    codes = np.empty((n, PROMOTER_LENGTH), dtype=np.uint8)
    neg = pb.is_negative
    if (~neg).any():
        codes[~neg] = st.codes[rows[~neg]]
    if neg.any():
        if negative_codes is None:
            raise ValueError(
                "the batch holds %d negative elements but no negative code "
                "matrix was given. Reading them from the promoter store "
                "would put a PROMOTER at the bottom of the scale, which is "
                "the one thing the negatives exist to prevent."
                % int(neg.sum()))
        codes[neg] = negative_codes[rows[neg]]

    gx = pb.genome.astype(np.int64)
    cat = pb.category.astype(np.int64)
    ids = [st.ids[int(rows[i])] if not neg[i] else "" for i in range(n)]
    ch = resolver.draw(gx, cat, ids, grp, training,
                       rng or np.random.default_rng(7898))
    n_cat = len(resolver.cells.categories)
    return Batch(
        x=one_hot_from_codes(codes), row=rows, species_id=gx, genome_ix=gx,
        og_ix=pb.og.astype(np.int64), category_ix=cat, group=grp,
        mu=pb.mu, sigma=np.full(n, np.nan, dtype=np.float32),
        weight=np.ones(n, dtype=np.float32),
        regression_mask=pb.regression_mask, sigma_mask=pb.sigma_mask,
        is_negative=neg, channels=ch,
        cat_targets=ch["taxonomy"].value[:, :n_cat].copy(),
        pair_a=pb.pair_a, pair_b=pb.pair_b, pair_y=pb.pair_y,
        pair_margin=pb.pair_margin, pair_delta=pb.pair_delta,
        pair_is_negative=pb.pair_is_negative)


# Negatives
@dataclass
class NegativeBank:
    """Filtered negative windows as codes, plus the filter's own numbers."""

    codes: np.ndarray                  # uint8 [n, 800]
    cls: List[str]
    source: List[str]
    genome: List[str]
    rows_by_genome: Dict[str, np.ndarray]
    pool_report: Dict[str, object]
    index_report: Dict[str, object]
    verification: Dict[str, object]
    positive_control: Dict[str, object]

    def __len__(self) -> int:
        return int(self.codes.shape[0])

    def report(self) -> Dict[str, object]:
        return {"n": len(self), "by_genome":
                {g: int(v.size) for g, v in self.rows_by_genome.items()},
                "pool": self.pool_report, "index": self.index_report,
                "verification": self.verification,
                "positive_control": self.positive_control}


def build_negative_bank(paths: CorpusPaths, corpus: PromoterCorpus,
                        genomes: Sequence[str],
                        cfg: Optional[NegativeConfig] = None,
                        per_genome: int = 400,
                        candidate_cap: int = 2000,
                        seed: int = 7898) -> NegativeBank:
    """Gene bodies (and the flank/shuffle classes) filtered by k-mer identity.

    The promoter index of a genome is built from every part of the split. The
    positive control is run and reported: a promoter of that genome, offered
    as a candidate, must be rejected. A filter with a rejection rate of 0 is
    an absent filter rather than a conservative one, and nothing downstream
    can tell the two apart.
    """
    import random

    cfg = cfg or NegativeConfig()
    rng = random.Random(seed)
    st = corpus.store
    want = set(genomes)
    groups = {GROUP_GENE: candidate_cap, GROUP_FLANK_R: candidate_cap}
    stores = load_groups(paths.sequences, groups, genomes=want,
                         per_genome_limit=per_genome)

    out_codes: List[np.ndarray] = []
    out_cls: List[str] = []
    out_src: List[str] = []
    out_gen: List[str] = []
    pool_rep: Dict[str, object] = {}
    idx_rep: Dict[str, object] = {}
    ver: Dict[str, object] = {}
    pos_ctrl: Dict[str, object] = {}

    prom_genome = np.asarray(st.genomes, dtype=object)[st.genome]
    cand_genome = {k: np.asarray(v.genomes, dtype=object)[v.genome]
                   for k, v in stores.items()}
    nrng = np.random.default_rng(seed)

    for g in genomes:
        rows = np.flatnonzero(prom_genome == g)
        if rows.size < 2:
            continue
        index = GenomeKmerIndex(cfg.kmer_k)
        index.add_codes(st.codes[rows]).finish()
        idx_rep[g] = index.report()
        # An index with a broken case map finds 66 k-mers instead of 770 and
        # still looks like a working filter; `verify` raises instead.
        outside = nrng.integers(0, 4, PROMOTER_LENGTH).astype(np.uint8)
        ver[g] = index.verify(st.codes[rows[0]], outside)

        cand: List[Tuple[str, str, str]] = []
        for grp_name, cls in ((GROUP_GENE, "gene"),
                              (GROUP_FLANK_R, "flank_r")):
            s = stores.get(grp_name)
            if s is None:
                continue
            take = np.flatnonzero(cand_genome[grp_name] == g)
            for i in take:
                cand.append((_codes_to_str(s.seq(int(i))), cls, s.ids[int(i)]))
        for i in rows[:max(per_genome // 4, 1)]:
            cand.append((_codes_to_str(st.codes[i]), "shuffle", st.ids[int(i)]))

        pool = build_pool(cand, index, cfg, rng)
        pool_rep[g] = pool.report()

        # Positive control: a promoter of this genome must be rejected.
        probe = _codes_to_str(st.codes[rows[1]])
        probe_pool = build_pool([(probe, "gene", "POSITIVE_CONTROL")], index,
                                cfg, rng)
        pos_ctrl[g] = {
            "promoter_offered_as_candidate": st.ids[int(rows[1])],
            "accepted": len(probe_pool) > 0,
            "hit_count": index.hit_count(probe),
            "expected_kmers": PROMOTER_LENGTH - cfg.kmer_k + 1,
        }

        for seq, cls, src in zip(pool.sequences, pool.classes, pool.sources):
            c = encode_codes(seq)
            if c.shape[0] < PROMOTER_LENGTH:
                continue
            out_codes.append(c[:PROMOTER_LENGTH])
            out_cls.append(cls)
            out_src.append(src)
            out_gen.append(g)

    codes = (np.stack(out_codes) if out_codes
             else np.zeros((0, PROMOTER_LENGTH), dtype=np.uint8))
    by_gen: Dict[str, np.ndarray] = {}
    arr = np.array(out_gen)
    for g in genomes:
        by_gen[g] = np.flatnonzero(arr == g).astype(np.int64) if arr.size \
            else np.zeros(0, dtype=np.int64)
    return NegativeBank(codes=codes, cls=out_cls, source=out_src,
                        genome=out_gen, rows_by_genome=by_gen,
                        pool_report=pool_rep, index_report=idx_rep,
                        verification=ver, positive_control=pos_ctrl)


def _codes_to_str(codes: np.ndarray) -> str:
    letters = np.array(list(NUCLEOTIDE_ORDER + "N"))
    return "".join(letters[np.minimum(codes, N_CODE)])


# Pipeline
@dataclass
class SplitDecision:
    """The split choice, with the table that decided it."""

    column: str
    resolution: str
    alternatives: List[Dict[str, object]]
    assignment_report: Dict[str, object]
    promoter_fractions: Dict[str, float]
    rationale: str


@dataclass
class Pipeline:
    """Everything a training loop needs, built once."""

    paths: CorpusPaths
    samples: SampleTable
    targets: Targets
    corpus: PromoterCorpus
    resolver: ConditionResolver
    decision: SplitDecision
    datasets: Dict[str, PromoterDataset]
    taxonomy: TaxonomyEncoder
    notes: List[str] = field(default_factory=list)

    def report(self) -> Dict[str, object]:
        ch = self.resolver.channels
        return {
            "paths": self.paths.report(),
            "split": {
                "column": self.decision.column,
                "resolution": self.decision.resolution,
                "rationale": self.decision.rationale,
                "alternatives": self.decision.alternatives,
                "assignment": self.decision.assignment_report,
                "promoter_fractions": self.decision.promoter_fractions,
            },
            "corpus": self.corpus.report(),
            "samples": {"n": len(self.samples.sample_ids),
                        "kept": self.samples.n_kept()},
            "taxonomy": self.taxonomy.report(),
            "conditioning": {
                "context_dim": ch.context_dim(),
                "channels": {s.name: {"enabled": s.enabled, "dim": s.dim,
                                      "dropout": s.dropout,
                                      "source": s.source}
                             for s in ch.specs},
                "cell_concentration": self.resolver.cells.concentration(),
            },
            "datasets": {k: v.report() for k, v in self.datasets.items()},
            "notes": list(self.notes),
        }

    def channel_consumers(self, model=None) -> Dict[str, str]:
        """Which channels the model actually reads. Checked, not assumed.

        `ConditionEncoder` emits a fixed four-token context and that order is
        a checkpoint contract, so the taxonomy vector has no slot today. An
        enabled channel nobody reads looks exactly like a working one, and the
        only defence is to inspect the signature rather than the config.
        """
        import inspect
        out = {"species": "species_id", "text": "text_emb",
               "gene_context": "flank_l + flank_r", "taxonomy": ""}
        if model is None:
            out["taxonomy"] = ("NOT CONSUMED: no model given to check "
                               "against")
            return out
        params = set(inspect.signature(model.forward).parameters)
        for name, arg in (("species", "species_id"), ("text", "text_emb"),
                          ("gene_context", "flank_l")):
            if arg not in params:
                out[name] = "NOT CONSUMED: forward has no %r" % arg
        out["taxonomy"] = (
            "consumed as `taxonomy`" if "taxonomy" in params else
            "NOT CONSUMED by forward(): ConditionOutput.TOKEN_NAMES is a "
            "fixed 4-token contract (species, text, flank_l, flank_r). The "
            "vector is built and carried in the batch; wiring it needs a "
            "fifth token or a taxonomy slot in ConditionEncoder, which is a "
            "model change, not a data change. It DOES reach the auxiliary "
            "category head as `cat_targets`.")
        return out


def build_pipeline(paths: Optional[CorpusPaths] = None,
                   column: str = SPLIT_COLUMN,
                   resolution: str = SPLIT_RESOLUTION,
                   channels: Optional[ChannelSet] = None,
                   parts: Sequence[str] = ("train", "val", "test"),
                   max_rows: Optional[int] = None,
                   cache: Optional[str] = None,
                   seed: int = 7898) -> Pipeline:
    """Files in, batches out, with every choice reported.

    `cache` defaults to `cache/promoters_800.npz`; pass "" to disable it. Its
    key is the size and mtime of sequences.tsv.gz, so a rebuilt corpus
    invalidates it instead of being served stale.
    """
    from ..model.species import StrainRegistry
    from .split_guard import compare_columns

    paths = paths or CorpusPaths.default()
    missing = paths.missing()
    if missing:
        raise FileNotFoundError("missing inputs: %r" % (missing,))

    notes: List[str] = []
    alternatives = compare_columns(paths.splits, paths.clusters)
    split = build_split(paths.splits, paths.clusters, column=column,
                        resolution=resolution)

    samples = load_samples(paths.samples)
    notes.extend(samples.warnings[:4])
    targets = build_targets(paths.combat, samples)
    corpus = build_corpus(paths, split, max_rows=max_rows,
                          cache=cache if cache is not None
                          else promoter_cache_path(paths))

    per_part = collections.Counter()
    for p in corpus.part:
        per_part[SPLIT_NAMES[p] if p >= 0 else "unassigned"] += 1
    tot = max(sum(per_part.values()), 1)
    decision = SplitDecision(
        column=column, resolution=resolution, alternatives=alternatives,
        assignment_report=split.report(),
        promoter_fractions={k: round(v / tot, 4)
                            for k, v in sorted(per_part.items())},
        rationale=(
            "C_block5 has the fewest split-spanning homology clusters (367 "
            "of 22 159, 10 598 promoter rows) against 594/14 941 for "
            "B_block10 and 1 555/32 020 for A_og, and needs the fewest "
            "orthogroups moved by the repair (848 vs 1 206). A_og is refused "
            "outright: spy = 1 orthogroup leaves six byte-identical copies "
            "of the pks1 promoter in training, and its `majority` repair "
            "leaves no spy part at all. `conservative` over `majority` "
            "because its priority spy > test > val > train cannot move a "
            "cluster INTO training; the price is 6 583 promoters of training "
            "material."))

    genomes = sorted(set(targets.genomes))
    taxonomy = TaxonomyEncoder.build(paths.taxonomy)
    cells = build_cell_conditioning(samples, taxonomy, genomes,
                                    tuple(SamplePolicy().categories))
    registry = StrainRegistry.default()
    text_cache = flank_cache = None
    if paths.text_cache:
        text_cache = EmbeddingCache.load(paths.text_cache)
    if paths.flank_cache:
        flank_cache = EmbeddingCache.load(paths.flank_cache)
    if text_cache is None:
        notes.append("text channel: no frozen-LM cache, so `available=False` "
                     "for every row and the model answers with its unknown "
                     "token. A hashed stand-in is refused on purpose.")
    if flank_cache is None:
        notes.append("gene_context channel: no DNA-LM cache; same treatment. "
                     "Build it with src/scripts/precompute_dna_lm.py.")

    d_dna = 768
    chan = channels or default_channels(
        taxonomy_dim=taxonomy.dim, species_dim=max(len(registry), 32),
        dna_dim=2 * d_dna,
        text_available=text_cache is not None,
        dna_available=flank_cache is not None)
    resolver = ConditionResolver(
        channels=chan, cells=cells,
        genome_index={g: i for i, g in enumerate(genomes)},
        registry_index={n: registry.index(n) for n in registry.names},
        text_cache=text_cache, flank_cache=flank_cache, seed=seed)

    datasets = {p: PromoterDataset(corpus, targets, resolver, p, seed=seed)
                for p in parts}
    return Pipeline(paths=paths, samples=samples, targets=targets,
                    corpus=corpus, resolver=resolver, decision=decision,
                    datasets=datasets, taxonomy=taxonomy, notes=notes)
