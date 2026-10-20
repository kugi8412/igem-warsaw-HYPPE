"""Conditioning channels: one mechanism, four channels, no per-channel code.

Every conditioning channel needs the same three things -- a value, a way of
saying "this row has no value", and a dropout that trains the model to work
without it. Written per channel those three drift, so the channel is data
(`ChannelSpec`) and the resolution is one function (`ChannelSet.draw`) used
identically by all of them. Adding a channel is a `ChannelSpec` in a list; it
cannot come with its own quietly different rule.

`species` is a genome index into `StrainRegistry` and is always available;
`text` and `gene_context` are frozen embeddings and are absent without their
cache file; `taxonomy` is the two-level multi-label encoding of
`condition_taxonomy.py` rather than a flat one-hot, because 371 of 2332
samples carry two or three subcategories and no flat class exists for them.
Its `axis_stated` bits separate "no regime stated on this axis" from "not
that kind of experiment"; without them a zero offset means both, which is the
common case.

Dropout is drawn per group, never per row. The siamese head scores a pair
under one condition, and a per-row draw hands the two halves different
conditions with probability 2p(1-p) = 18 % at p = 0.1 -- 69 broken forwards
in 400, which a single-trial smoke test passes 81 % of the time.

"Unknown" is a learned `nn.Parameter` of `ConditionEncoder`; what this layer
owns is the boolean that fires it. Rows with `present=False` carry an exactly
zero value, so a consumer that forgets the mask degrades to "no conditioning"
rather than to someone else's condition. `apply_unknown` masks the operand
and not the result, because `torch.where` on the result still backpropagates
through the discarded branch.

Ablation is "substitute", never "shuffle". Batches here are
genome-homogeneous by construction, so permuting a channel inside a batch
changes nothing and yields +0.00000 -- once read as a dead channel when it
meant no intervention happened. `substitute` hands a row another group's
value, which is the only variant that moves anything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .condition_taxonomy import (
    AXES,
    PARENT_CATEGORIES,
    TAXONOMY_PATH,
    load_taxonomy,
    support,
)

ABLATIONS: Tuple[str, ...] = ("none", "zero", "substitute")

#: Channel names are a checkpoint contract: the model keeps one unknown token
#: per name. Append, never reorder, never rename.
CHANNEL_ORDER: Tuple[str, ...] = ("species", "text", "taxonomy",
                                  "gene_context")


class MissingChannelSource(FileNotFoundError):
    """A channel is enabled but its cached values are not on disk."""


@dataclass
class ChannelSpec:
    """One conditioning channel. Everything about it is here, and only here."""

    name: str
    dim: int
    dropout: float = 0.0
    enabled: bool = True
    ablate: str = "none"
    #: Where the values come from, for the report only. Dispatching on this
    #: string is how a channel acquires its own private rule.
    source: str = ""

    def validate(self) -> None:
        if self.ablate not in ABLATIONS:
            raise ValueError(
                "channel %r: ablate=%r. 'shuffle' is forbidden -- batches are "
                "genome-homogeneous, so shuffling a channel inside a batch "
                "changes nothing and measures +0.00000. Use 'substitute'."
                % (self.name, self.ablate))
        if self.enabled and self.dim <= 0:
            raise ValueError(
                "channel %r is enabled with dim=%d: an enabled channel of "
                "width zero is the `jaspar: {}` failure -- set and inert."
                % (self.name, self.dim))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("channel %r: dropout must be in [0, 1), got %r"
                             % (self.name, self.dropout))


@dataclass
class ChannelValues:
    """Values plus availability for one channel of one batch."""

    name: str
    value: np.ndarray            # float32 [B, dim]; exactly 0 where not present
    present: np.ndarray          # bool [B]
    dropped: np.ndarray          # bool [B]: present in the data, hidden by us
    missing: np.ndarray          # bool [B]: not in the data at all

    def fraction_present(self) -> float:
        return float(self.present.mean()) if self.present.size else 0.0


@dataclass
class ChannelSet:
    """The shared mechanism. Channels differ only by their `ChannelSpec`."""

    specs: List[ChannelSpec] = field(default_factory=list)
    seed: int = 7898

    def __post_init__(self) -> None:
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel names %r" % (names,))
        for s in self.specs:
            s.validate()

    def by_name(self) -> Dict[str, ChannelSpec]:
        return {s.name: s for s in self.specs}

    def context_dim(self) -> int:
        return sum(s.dim for s in self.specs if s.enabled)

    def draw(self, name: str, value: np.ndarray, available: np.ndarray,
             group: Optional[np.ndarray] = None, training: bool = True,
             rng: Optional[np.random.Generator] = None) -> ChannelValues:
        """Resolve one channel for one batch. The only path, for every channel.

        `available` says whether the data has a value for that row; dropout
        then hides some of the available ones. The two stay apart in the
        output because they mean different things: `missing` is a property of
        the corpus, `dropped` of this draw.
        """
        spec = self.by_name().get(name)
        if spec is None:
            raise KeyError("no channel %r; known: %r"
                           % (name, [s.name for s in self.specs]))
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != spec.dim:
            raise ValueError(
                "channel %r expects [B, %d], got %r. A width mismatch here is "
                "the failure `ConditioningConfig` raises on: the vector and "
                "the vocabulary have to be widened together."
                % (name, spec.dim, tuple(value.shape)))
        n = value.shape[0]
        available = np.asarray(available, dtype=bool).reshape(n)
        rng = rng or np.random.default_rng(self.seed)

        if not spec.enabled:
            return ChannelValues(name, np.zeros_like(value),
                                 np.zeros(n, bool), np.zeros(n, bool),
                                 np.ones(n, bool))

        keep = np.ones(n, dtype=bool)
        if training and spec.dropout > 0.0:
            g = self._group_ids(group, n)
            uniq, inv = np.unique(g, return_inverse=True)
            keep = (rng.random(uniq.size) >= spec.dropout)[inv]

        value = self._ablate(spec, value, group, rng)

        present = available & keep
        out = value.copy()
        out[~present] = 0.0            # operand masked, not the result
        return ChannelValues(name, out, present, available & ~keep, ~available)

    # Internals
    @staticmethod
    def _group_ids(group: Optional[np.ndarray], n: int) -> np.ndarray:
        if group is None:
            # Every row its own group: legal for a batch of independent
            # promoters, not for pair batches. The caller that forgets is the
            # failure this argument exists for.
            return np.arange(n)
        g = np.asarray(group).reshape(-1)
        if g.shape[0] != n:
            raise ValueError("group has %d entries for %d rows"
                             % (g.shape[0], n))
        return g

    @staticmethod
    def _ablate(spec: ChannelSpec, value: np.ndarray,
                group: Optional[np.ndarray],
                rng: np.random.Generator) -> np.ndarray:
        if spec.ablate == "none":
            return value
        if spec.ablate == "zero":
            return np.zeros_like(value)
        # substitute: give each group another group's value.
        g = ChannelSet._group_ids(group, value.shape[0])
        uniq, inv = np.unique(g, return_inverse=True)
        if uniq.size < 2:
            raise ValueError(
                "channel %r: ablate='substitute' needs at least two groups in "
                "the batch, got %d. With one group the intervention is empty "
                "-- the same +0.00000 that 'shuffle' produces."
                % (spec.name, uniq.size))
        first = np.zeros(uniq.size, dtype=np.int64)
        for i in range(value.shape[0] - 1, -1, -1):
            first[inv[i]] = i
        perm = rng.permutation(uniq.size)
        clash = perm == np.arange(uniq.size)
        if clash.any():                      # derangement, so nothing is a no-op
            perm[clash] = perm[(np.nonzero(clash)[0] + 1) % uniq.size]
        return value[first[perm][inv]]

    def report(self, drawn: Mapping[str, ChannelValues]) -> Dict[str, object]:
        out: Dict[str, object] = {"context_dim": self.context_dim()}
        for s in self.specs:
            d = drawn.get(s.name)
            out[s.name] = {
                "enabled": s.enabled, "dim": s.dim, "dropout": s.dropout,
                "ablate": s.ablate, "source": s.source,
                "present": round(d.fraction_present(), 4) if d else None,
                "missing_in_data": round(float(d.missing.mean()), 4)
                if d else None,
                "dropped_by_us": round(float(d.dropped.mean()), 4)
                if d else None,
            }
        return out


def apply_unknown(value, present, unknown):
    """value where present, the learned unknown token where not. Torch.

    The mask multiplies the operands. `torch.where(present, value, unknown)`
    gives the same numbers and still backpropagates through the discarded
    branch.
    """

    if value.shape[0] != present.shape[0]:
        raise ValueError("value has %d rows, present %d"
                         % (value.shape[0], present.shape[0]))
    m = present.to(dtype=value.dtype).reshape(-1, 1)
    null = unknown.to(dtype=value.dtype).reshape(1, -1).expand_as(value)
    return m * value + (1.0 - m) * null


# Taxonomy
@dataclass
class TaxonomyEncoder:
    """Two-level multi-label encoding of the condition.

    Layout, fixed and reported so a checkpoint can be read back:
    `[0 : P)` parent one-hot with P = 7, `[P : P + S)` separating
    subcategories as multi-hot, `[P + S : P + S + A)` axis_stated bits with
    A = 7.

    `separating_only=True` keeps the subcategories that passed the
    permutation falsifier. One confined to a single study is
    indistinguishable from that study's batch effect, so it stays in the
    vocabulary -- dropping it would renumber the vector -- and
    `frozen_columns` names it so the model can hold its offset at zero.
    """

    subcategories: Tuple[str, ...]
    parents: Tuple[str, ...] = PARENT_CATEGORIES
    axes: Tuple[str, ...] = tuple(AXES)
    frozen_columns: Tuple[int, ...] = ()
    path: str = TAXONOMY_PATH
    n_unlabelled: int = 0

    @classmethod
    def build(cls, path: str = TAXONOMY_PATH, separating_only: bool = True
              ) -> "TaxonomyEncoder":
        tax = load_taxonomy(path)
        sup = support(path)
        seen: List[str] = []
        for lab in tax.values():
            for c in (lab.separating if separating_only else lab.subcategories):
                if c not in seen:
                    seen.append(c)
        subs = tuple(sorted(seen))
        frozen = tuple(i for i, s in enumerate(subs)
                       if sup[s].offset_policy == "freeze")
        unl = sum(1 for lab in tax.values() if not lab.subcategories)
        return cls(subcategories=subs, frozen_columns=frozen, path=path,
                   n_unlabelled=unl)

    @property
    def dim(self) -> int:
        return len(self.parents) + len(self.subcategories) + len(self.axes)

    def encode(self, samples: Sequence[str],
               tax: Optional[Mapping[str, object]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(matrix [n, dim], available [n]).

        A sample absent from the table is not an error: its condition is
        genuinely unknown and the channel's unknown token is the right
        answer. It is counted in `available`, never imputed.
        """
        tax = tax if tax is not None else load_taxonomy(self.path)
        p_ix = {c: i for i, c in enumerate(self.parents)}
        s_ix = {c: i for i, c in enumerate(self.subcategories)}
        a_ix = {a: i for i, a in enumerate(self.axes)}
        off_s = len(self.parents)
        off_a = off_s + len(self.subcategories)
        out = np.zeros((len(samples), self.dim), dtype=np.float32)
        avail = np.zeros(len(samples), dtype=bool)
        for i, sid in enumerate(samples):
            lab = tax.get(sid)
            if lab is None:
                continue
            avail[i] = True
            j = p_ix.get(lab.parent_category)
            if j is not None:
                out[i, j] = 1.0
            for c in lab.separating:
                k = s_ix.get(c)
                if k is not None:
                    out[i, off_s + k] = 1.0
            got = set(lab.subcategories)
            for ax, members in AXES.items():
                if got & set(members):
                    out[i, off_a + a_ix[ax]] = 1.0
        return out, avail

    def report(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "n_parents": len(self.parents),
            "n_subcategories": len(self.subcategories),
            "n_axes": len(self.axes),
            "frozen_subcategories": [self.subcategories[i]
                                     for i in self.frozen_columns],
            "samples_without_any_subcategory": self.n_unlabelled,
            "source": os.path.basename(self.path),
        }


# Species
def species_matrix(genomes: Sequence[str], registry) -> Tuple[np.ndarray,
                                                              np.ndarray]:
    """One-hot over the registry. (index, available); index is -1 if absent.

    A genome's index is its embedding row, so an unknown genome must not be
    folded into row 0: it gets -1 and `available=False`, and the model
    answers with its unknown token.
    """
    idx = np.full(len(genomes), -1, dtype=np.int64)
    avail = np.zeros(len(genomes), dtype=bool)
    for i, g in enumerate(genomes):
        if g in registry:
            idx[i] = registry.index(g)
            avail[i] = True
    return idx, avail


# Cached embeddings
@dataclass
class EmbeddingCache:
    """Frozen per-key vectors from an .npz. No fallback, by design.

    The text and gene-context channels are frozen encoders whose weights are
    not in this repo. Hash-embedding them "just to have a vector" would give
    a channel that trains, reports a share of the context and carries no
    information, so a missing cache either raises or switches the channel off
    with `available=False` for every row, and the report says which.
    """

    keys: Dict[str, int]
    matrix: np.ndarray
    source: str = ""

    @classmethod
    def load(cls, path: str, expect_dim: Optional[int] = None
             ) -> "EmbeddingCache":
        if not os.path.exists(path):
            raise MissingChannelSource(
                "no embedding cache at %s. Build it with the frozen encoder, "
                "or disable the channel: a hashed stand-in would report a "
                "context share while carrying no information." % (path,))
        with np.load(path, allow_pickle=False) as z:
            keys = [str(k) for k in z["keys"]]
            mat = np.asarray(z["vectors"], dtype=np.float32)
        if mat.shape[0] != len(keys):
            raise ValueError("%s: %d keys against %d vectors"
                             % (path, len(keys), mat.shape[0]))
        if expect_dim is not None and mat.shape[1] != expect_dim:
            raise ValueError("%s: vectors are %d wide, channel expects %d"
                             % (path, mat.shape[1], expect_dim))
        return cls({k: i for i, k in enumerate(keys)}, mat, path)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def lookup(self, keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((len(keys), self.dim), dtype=np.float32)
        avail = np.zeros(len(keys), dtype=bool)
        for i, k in enumerate(keys):
            j = self.keys.get(k)
            if j is not None:
                out[i] = self.matrix[j]
                avail[i] = True
        return out, avail


# Paired flank embeddings
#: Group names written by `src/scripts/precompute_dna_lm.py --groups`.
FLANK_LEFT_GROUP = "FlankLeft"
FLANK_RIGHT_GROUP = "FlankRight"


@dataclass
class FlankCacheReport:
    """What the pairing found. Printed, never summarised to a bool."""

    npy: str = ""
    index: str = ""
    n_rows: int = 0
    hidden: int = 0
    dtype: str = ""
    n_ids: int = 0
    n_both: int = 0
    n_left_only: int = 0
    n_right_only: int = 0
    n_dup_keys: int = 0
    n_dup_identical: int = 0
    n_ambiguous_ids: int = 0

    def format(self) -> str:
        return ("flank cache %s\n"
                "  index      %s\n"
                "  matrix     %d rows x %d, %s\n"
                "  ids        %d seen, %d with BOTH flanks, "
                "%d left-only, %d right-only (all dropped)\n"
                "  duplicates %d (id, side) keys listed more than once, "
                "%d of them bit-identical and kept;\n"
                "             %d ids dropped as AMBIGUOUS (the rows disagree "
                "and nothing here can choose)"
                % (self.npy, self.index, self.n_rows, self.hidden, self.dtype,
                   self.n_ids, self.n_both, self.n_left_only,
                   self.n_right_only, self.n_dup_keys, self.n_dup_identical,
                   self.n_ambiguous_ids))


@dataclass
class FlankEmbeddingCache:
    """Frozen DNA-LM flank vectors, paired per promoter id.

    Not an `EmbeddingCache`, which serves one vector per key. The DNA-LM
    precompute writes `embeddings.npy` ([N, hidden], row order = index order)
    beside `index.tsv` (ID, GENOME, OG_ID, GROUP, row), and N counts flank
    windows rather than promoters: each id contributes a `FlankLeft` and a
    `FlankRight` row at two different row numbers, so one vector per key
    would hand both flank tokens the same number.

    The pairing reads the `row` column and assumes neither that the sides
    alternate nor that an id's two rows are adjacent. A mis-join is silent --
    the model trains happily on another gene's neighbourhood -- so an id
    whose two sides cannot both be found is dropped and counted rather than
    half-filled, and a single-sided id is never duplicated.
    """

    keys: Dict[str, int]          # promoter id -> row of `pairs`
    pairs: np.ndarray             # [n_ids, 2] int64 rows into `matrix`
    matrix: np.ndarray            # [n_rows, hidden]
    report: FlankCacheReport = field(default_factory=FlankCacheReport)

    @property
    def dim(self) -> int:
        """Width of one side. The channel is 2 * dim wide."""
        return int(self.matrix.shape[1])

    @classmethod
    def default_index_path(cls, npy_path: str) -> str:
        return os.path.join(os.path.dirname(npy_path), "index.tsv")

    @classmethod
    def load(cls, npy_path: str, index_path: str = "",
             left_group: str = FLANK_LEFT_GROUP,
             right_group: str = FLANK_RIGHT_GROUP,
             mmap: bool = True) -> "FlankEmbeddingCache":
        if not os.path.exists(npy_path):
            raise MissingChannelSource(
                "no DNA-LM cache at %s. Build it with "
                "src/scripts/precompute_dna_lm.py, or leave the channel off: "
                "a stand-in would report a context share while carrying no "
                "information." % (npy_path,))
        index_path = index_path or cls.default_index_path(npy_path)
        if not os.path.exists(index_path):
            raise MissingChannelSource(
                "no row index at %s. `embeddings.npy` alone cannot be joined "
                "to promoters: its rows are flank WINDOWS and nothing in the "
                "matrix says which id or which side a row belongs to. "
                "Re-run the precompute, which writes both files together."
                % (index_path,))

        mat = np.load(npy_path, mmap_mode=("r" if mmap else None))
        if mat.ndim != 2:
            raise ValueError("%s: expected [N, hidden], got %s"
                             % (npy_path, (mat.shape,)))
        n_rows, hidden = int(mat.shape[0]), int(mat.shape[1])

        rep = FlankCacheReport(npy=npy_path, index=index_path, n_rows=n_rows,
                               hidden=hidden, dtype=str(mat.dtype))

        left: Dict[str, List[int]] = {}
        right: Dict[str, List[int]] = {}
        with open(index_path, "rt", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            need = ("ID", "GROUP", "row")
            missing = [c for c in need if c not in header]
            if missing:
                raise ValueError("%s lacks columns %r (has %r)"
                                 % (index_path, missing, header))
            ix = {c: header.index(c) for c in need}
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < len(header):
                    continue
                grp = p[ix["GROUP"]]
                side = (left if grp == left_group else
                        right if grp == right_group else None)
                if side is None:
                    continue
                key = p[ix["ID"]]
                row = int(p[ix["row"]])
                if not 0 <= row < n_rows:
                    raise ValueError(
                        "%s: row %d for %s/%s is outside the %d rows of %s. "
                        "The index and the matrix are from different runs."
                        % (index_path, row, key, grp, n_rows, npy_path))
                side.setdefault(key, []).append(row)

        def resolve(rows: List[int]) -> Optional[int]:
            """One row for a side, or None when the listed rows disagree.

            An id listed twice is only a problem if the two listings mean
            different things, which is a question about the vectors and is
            answered by reading them rather than by a rule.
            """
            if len(rows) == 1:
                return rows[0]
            rep.n_dup_keys += 1
            block = np.asarray(matrix[sorted(rows)], dtype=np.float32)
            if float(np.abs(block - block[0]).max()) == 0.0:
                rep.n_dup_identical += 1
                return rows[0]
            return None

        matrix = mat
        ids = sorted(set(left) | set(right))
        rep.n_ids = len(ids)
        keys: Dict[str, int] = {}
        pair_rows: List[Tuple[int, int]] = []
        for key in ids:
            lr, rr = left.get(key), right.get(key)
            if lr is None:
                rep.n_right_only += 1
                continue
            if rr is None:
                rep.n_left_only += 1
                continue
            a, b = resolve(lr), resolve(rr)
            if a is None or b is None:
                # The id names more than one neighbourhood and the rows
                # disagree on which. Serving either would be a coin flip made
                # inside a loader, and a wrong flank is invisible downstream.
                rep.n_ambiguous_ids += 1
                continue
            if a == b:
                raise ValueError(
                    "%s: id %r has %s and %s on the SAME row %d. That would "
                    "make flank_l == flank_r -- one number reported as two."
                    % (index_path, key, left_group, right_group, a))
            keys[key] = len(pair_rows)
            pair_rows.append((a, b))
        rep.n_both = len(pair_rows)
        if not pair_rows:
            raise ValueError(
                "%s: no id has both %s and %s. The cache cannot fill a "
                "two-token flank channel." % (index_path, left_group,
                                              right_group))
        pairs = np.asarray(pair_rows, dtype=np.int64)
        return cls(keys=keys, pairs=pairs, matrix=mat, report=rep)

    def lookup_pair(self, keys: Sequence[str]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(left [n, dim], right [n, dim], available [n]).

        Rows with no entry are left at zero and marked unavailable, so the
        encoder answers with its unknown token. They are filled with neither
        a neighbour's vector nor the mean: both would report a context share
        while carrying no information.
        """
        n, d = len(keys), self.dim
        lf = np.zeros((n, d), dtype=np.float32)
        rt = np.zeros((n, d), dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        for i, k in enumerate(keys):
            j = self.keys.get(k)
            if j is None:
                continue
            a, b = self.pairs[j]
            lf[i] = np.asarray(self.matrix[a], dtype=np.float32)
            rt[i] = np.asarray(self.matrix[b], dtype=np.float32)
            ok[i] = True
        return lf, rt, ok

    def rows_for(self, keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Matrix rows for each key, -1 where the key is absent.

        The corpus resolves ids once at build time and then gathers by row,
        so a training step never touches the python dict.
        """
        lf = np.full(len(keys), -1, dtype=np.int64)
        rt = np.full(len(keys), -1, dtype=np.int64)
        for i, k in enumerate(keys):
            j = self.keys.get(k)
            if j is not None:
                lf[i], rt[i] = self.pairs[j]
        return lf, rt

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """[n, dim] float32 for matrix rows; a -1 row comes back as zeros."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros((rows.size, self.dim), dtype=np.float32)
        hit = rows >= 0
        if hit.any():
            out[hit] = np.asarray(self.matrix[rows[hit]], dtype=np.float32)
        return out


def default_channels(taxonomy_dim: int, species_dim: int = 32,
                     text_dim: int = 768, dna_dim: int = 768,
                     text_available: bool = False,
                     dna_available: bool = False) -> ChannelSet:
    """The four channels with the dropouts this project measured.

    `text` at 0.10 is the rate `model/condition.py` already trains its
    unknown token at, and `taxonomy` matches it because the design target is
    a condition the corpus does not contain (G418 selection, 0 of 2332
    samples), so inference will run with that channel unknown and the model
    has to have seen the state. `species` is 0.0: every promoter has a
    genome, and a model trained to work without it has learned to ignore a
    channel that is always there. `gene_context` is 0.05, because it is the
    strongest channel measured -- substituting it cost -0.1819 SCC against a
    0.74885 baseline -- and so the most able to become a crutch.
    """
    return ChannelSet([
        ChannelSpec("species", species_dim, 0.0, True, "none", "registry"),
        ChannelSpec("text", text_dim, 0.10, text_available, "none",
                    "frozen_text_encoder"),
        ChannelSpec("taxonomy", taxonomy_dim, 0.10, True, "none",
                    "condition_taxonomy.tsv"),
        ChannelSpec("gene_context", dna_dim, 0.05, dna_available, "none",
                    "dna_lm"),
    ])
