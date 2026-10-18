"""Frozen sentence-LM vectors of the EXPERIMENT DESCRIPTION, per condition cell.

`ConditionEncoder` takes four tokens (species, text, flank_l, flank_r), and
the `text` one has never been fed: the training loop supplied no `text_emb`,
so the encoder answered with `unknown_token_text` on every row of every run.
This module is the join between an artefact of frozen description embeddings
and the corpus rows, built as the analogue of `FlankEmbeddingCache`.

The granularity is a measurement rather than a preference. The label is a
per-cell mean, and `labels.build_targets` backs the cell (orthogroup, genome,
category) with a sample list that does not depend on the orthogroup. A per-row
aggregate is therefore a function of the genome alone -- at most 19 distinct
vectors, `H(text | species_id) = 0`, a channel carrying zero bits beyond
species. A per-cell aggregate is a function of (genome, category): at most
19 x 7 keys, 65 of them non-empty here. Neither carries within-cell variation,
because averaging removes it.

Measured with SapBERT over the 1487 kept samples with a non-empty `condition`
string, eta^2 of the 768-dim embedding is 0.1301 on the 7-level category,
0.1348 on the 19-level genome factor, 0.3317 on the 66-level (genome,
category) factor and 0.6031 on the 361-level study factor. So the raw
per-sample text is not a category lookup -- 87 % of its variance is
within-category -- but any per-cell aggregate is one by construction, and
keying on the cell retains 2.5x the variance of keying on the genome. The
0.67 below the cell is unreachable without changing the target granularity.

The key is `<genome>|<category>`, with `<genome>|*` for the genome marginal,
the spelling `dataset.ConditionResolver.cell_key` already uses; a second
spelling would split one cache into two half-empty ones. There is no fallback from a
missing key to the marginal and no mean- or neighbour-filling: each would
report a context share for a quantity other than the one the key names, so a
row whose cell has no description reaches `unknown_token_text` and is counted.
The artefact layout is never guessed either -- a cache that cannot be joined
raises rather than resolving to an empty table.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .conditioning import MissingChannelSource

__all__ = ["ExperimentTextCache", "TextCacheReport", "cell_key",
           "resolve_cells", "MARGINAL_CATEGORY", "SIBLING_MODULES"]

#: The category name that means "the genome marginal", identical to
#: `dataset.ConditionResolver.cell_key`.
MARGINAL_CATEGORY = "*"

#: Import paths tried for the sibling loader, in order.
SIBLING_MODULES: Tuple[str, ...] = ("hyppe.data_text", "hyppe.data.data_text")

#: Field names accepted by the generic .npz reader, in order of preference.
#: The winner is reported: a reader that accepts several spellings without
#: saying which it used makes two different artefacts look like one.
_KEY_FIELDS: Tuple[str, ...] = ("keys", "cell_keys", "ids", "names",
                                "text_keys")
_VEC_FIELDS: Tuple[str, ...] = ("vectors", "embeddings", "matrix", "emb",
                                "text_emb", "V")


def cell_key(genome: str, category: Optional[str] = None) -> str:
    """`<genome>|<category>`; `<genome>|*` for the genome marginal."""
    cat = category if category else MARGINAL_CATEGORY
    return "%s|%s" % (genome, cat)


@dataclass
class TextCacheReport:
    """Where the vectors came from and what the join found. Never a bool."""

    path: str = ""
    backend: str = ""            # "hyppe.data_text:Cls.load" | "npz:<fields>"
    n_rows: int = 0
    dim: int = 0
    n_keys: int = 0
    n_dup_keys: int = 0
    n_dup_identical: int = 0
    n_ambiguous: int = 0
    key_field: str = ""
    vec_field: str = ""
    sample_keys: Tuple[str, ...] = ()

    def format(self) -> str:
        return ("experiment-text cache %s\n"
                "  backend    %s (key field %r, vector field %r)\n"
                "  matrix     %d rows x %d\n"
                "  keys       %d usable, %d listed more than once "
                "(%d bit-identical and kept), %d dropped as AMBIGUOUS\n"
                "  examples   %s"
                % (self.path, self.backend, self.key_field, self.vec_field,
                   self.n_rows, self.dim, self.n_keys, self.n_dup_keys,
                   self.n_dup_identical, self.n_ambiguous,
                   ", ".join(self.sample_keys) or "(none)"))


# Discovery
def _sibling_loader() -> Optional[Tuple[str, str, object]]:
    """(module name, entry point name, callable) or None if no module exists.

    Raises if a sibling module is importable but exposes nothing that can load
    a cache: falling back to the generic reader would hide that mismatch.
    """
    for name in SIBLING_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for attr in sorted(dir(mod)):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            if (isinstance(obj, type) and attr.lower().endswith("cache")
                    and callable(getattr(obj, "load", None))):
                return name, "%s.load" % (attr,), getattr(obj, "load")
        for attr in ("load", "load_cache", "open_cache", "load_text_emb",
                     "load_experiment_text"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return name, attr, fn
        raise MissingChannelSource(
            "%s is importable but exposes no loader: no `<Name>Cache.load` "
            "classmethod and no module-level load/load_cache/open_cache/"
            "load_text_emb/load_experiment_text. Public members: %s. This "
            "wiring reads whatever that module exposes and refuses to guess a "
            "third thing." % (name, ", ".join(
                a for a in sorted(dir(mod)) if not a.startswith("_"))[:400]))
    return None


def _as_key_list(obj) -> Optional[List[str]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        # key -> row. Invert into a positional list.
        n = max(int(v) for v in obj.values()) + 1 if obj else 0
        out = [""] * n
        for k, v in obj.items():
            out[int(v)] = str(k)
        return out
    arr = np.asarray(obj)
    if arr.ndim != 1:
        return None
    return [str(k) for k in arr.tolist()]


def _extract(obj) -> Tuple[Optional[List[str]], Optional[np.ndarray], str,
                           str]:
    """(keys, matrix, key field name, vector field name) from a loaded object.

    Accepts, in this order: a (keys, vectors) pair; a mapping key -> vector; a
    key collection plus a matrix attribute; a key collection plus the
    `rows_for`/`gather` pair of `FlankEmbeddingCache`.
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        keys = _as_key_list(obj[0])
        mat = np.asarray(obj[1])
        if keys is not None and mat.ndim == 2:
            return keys, mat, "tuple[0]", "tuple[1]"
    if isinstance(obj, dict) and obj and not isinstance(
            next(iter(obj.values())), (int, np.integer)):
        keys = [str(k) for k in obj.keys()]
        mat = np.stack([np.asarray(obj[k], dtype=np.float32) for k in keys])
        return keys, mat, "dict keys", "dict values"

    keys = None
    key_field = ""
    for name in ("keys",) + _KEY_FIELDS:
        got = _as_key_list(getattr(obj, name, None))
        if got:
            keys, key_field = got, name
            break
    mat = None
    vec_field = ""
    for name in _VEC_FIELDS:
        cand = getattr(obj, name, None)
        if cand is None:
            continue
        arr = np.asarray(cand)
        if arr.ndim == 2:
            mat, vec_field = arr, name
            break
    if keys is not None and mat is None:
        rows_for = getattr(obj, "rows_for", None)
        gather = getattr(obj, "gather", None)
        if callable(rows_for) and callable(gather):
            rows = np.asarray(rows_for(keys), dtype=np.int64).reshape(-1)
            mat = np.asarray(gather(rows), dtype=np.float32)
            vec_field = "rows_for+gather"
    return keys, mat, key_field, vec_field


# The cache
@dataclass
class ExperimentTextCache(object):
    """Frozen description vectors keyed by conditioning cell.

    `keys` maps `<genome>|<category>` to a row of `matrix`. An absent key
    resolves to -1, which the corpus turns into `text_ok = 0` and
    `ConditionEncoder._resolve` masks on the operand, so the row reaches
    `unknown_token_text` and never `text_proj(zeros)` -- the latter being a
    learned constant that would be reported as a context share.
    """

    keys: Dict[str, int]
    matrix: np.ndarray
    report: TextCacheReport = field(default_factory=TextCacheReport)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    # Load
    @classmethod
    def load(cls, path: str, index_path: str = "",
             expect_dim: Optional[int] = None,
             prefer_sibling: bool = True) -> "ExperimentTextCache":
        """Load through the sibling loader if it exists, else read the .npz."""
        rep = TextCacheReport(path=str(path))
        obj = None
        if prefer_sibling:
            found = _sibling_loader()
            if found is not None:
                mod_name, entry, fn = found
                try:
                    obj = fn(path, index_path) if index_path else fn(path)
                except TypeError:
                    obj = fn(path)
                rep.backend = "%s:%s" % (mod_name, entry)
        if obj is None:
            if not os.path.exists(path):
                raise MissingChannelSource(
                    "no experiment-text cache at %s and no sibling loader "
                    "module (%s). Build the artefact, or leave "
                    "conditioning.experiment_text.enabled at false: a "
                    "stand-in would report a context share while carrying no "
                    "information." % (path, " / ".join(SIBLING_MODULES)))
            z = np.load(path, allow_pickle=False)
            files = list(getattr(z, "files", []))
            kf = next((n for n in _KEY_FIELDS if n in files), "")
            vf = next((n for n in _VEC_FIELDS if n in files), "")
            if not kf or not vf:
                raise MissingChannelSource(
                    "%s carries %s. A key array and a vector array are both "
                    "required: a matrix alone cannot be joined to a corpus "
                    "row, which is the fault that left the old C2 channel "
                    "running on its unknown token for 167 195 rows. Accepted "
                    "key fields %s, accepted vector fields %s."
                    % (path, files, list(_KEY_FIELDS), list(_VEC_FIELDS)))
            keys_raw = _as_key_list(z[kf])
            mat_raw = np.asarray(z[vf])
            rep.backend = "npz"
            rep.key_field, rep.vec_field = kf, vf
        else:
            keys_raw, mat_raw, kf, vf = _extract(obj)
            rep.key_field, rep.vec_field = kf, vf
            if keys_raw is None or mat_raw is None:
                raise MissingChannelSource(
                    "%s returned %r, from which no (keys, vectors) pair could "
                    "be taken. Expose either a key collection plus a 2-D "
                    "matrix, or a key collection plus `rows_for`/`gather` as "
                    "`FlankEmbeddingCache` does."
                    % (rep.backend, type(obj).__name__))

        mat_raw = np.asarray(mat_raw)
        if mat_raw.ndim != 2:
            raise ValueError("%s: vectors must be 2-D, got %s"
                             % (path, (mat_raw.shape,)))
        if len(keys_raw) != int(mat_raw.shape[0]):
            raise ValueError(
                "%s: %d keys against %d vector rows. The join is positional "
                "and a length mismatch means the two arrays describe "
                "different objects." % (path, len(keys_raw),
                                        int(mat_raw.shape[0])))
        mat = np.ascontiguousarray(mat_raw, dtype=np.float32)
        if not bool(np.isfinite(mat).all()):
            raise ValueError(
                "%s: %d non-finite entries in the vectors. A NaN in a frozen "
                "embedding poisons every downstream statistic silently."
                % (path, int((~np.isfinite(mat)).sum())))

        # Duplicates: identical rows are kept, disagreeing rows drop the key,
        # as in `FlankEmbeddingCache`. Nothing here can choose between two
        # vectors for one cell.
        seen: Dict[str, int] = {}
        dup, dup_same, ambiguous = 0, 0, 0
        drop: set = set()
        for i, k in enumerate(keys_raw):
            j = seen.get(k)
            if j is None:
                seen[k] = i
                continue
            dup += 1
            if bool(np.array_equal(mat[i], mat[j])):
                dup_same += 1
            else:
                drop.add(k)
        for k in drop:
            seen.pop(k, None)
            ambiguous += 1
        rep.n_rows = int(mat.shape[0])
        rep.dim = int(mat.shape[1])
        rep.n_keys = len(seen)
        rep.n_dup_keys = dup
        rep.n_dup_identical = dup_same
        rep.n_ambiguous = ambiguous
        rep.sample_keys = tuple(sorted(seen)[:3])
        if not seen:
            raise ValueError(
                "%s: no usable key survived. The channel cannot be filled."
                % (path,))
        if expect_dim is not None and int(expect_dim) != rep.dim:
            raise ValueError(
                "experiment-text cache %s is %d dims wide, but "
                "conditioning.experiment_text.dim declares %d. The condition "
                "encoder builds `text_proj = Linear(cond_d_lm, cond_d_ctx)` "
                "from the SAME number, so a mismatch is a shape error at the "
                "first forward -- raised here, at construction, instead."
                % (path, rep.dim, int(expect_dim)))
        return cls(keys=dict(seen), matrix=mat, report=rep)

    # Lookup
    def rows_for(self, keys: Sequence[str]) -> np.ndarray:
        """Matrix row per key, -1 where absent. Resolved ONCE, at build time."""
        out = np.full(len(keys), -1, dtype=np.int64)
        for i, k in enumerate(keys):
            j = self.keys.get(k)
            if j is not None:
                out[i] = j
        return out

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """[n, dim] float32; a -1 row comes back as ZEROS and must be masked.

        Zeros are not a neutral input: `text_proj` has a bias, so a zero input
        is a learned constant. The caller pairs this with `text_ok`.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.zeros((rows.size, self.dim), dtype=np.float32)
        hit = rows >= 0
        if bool(hit.any()):
            out[hit] = self.matrix[rows[hit]]
        return out


# Corpus join
def resolve_cells(cache: ExperimentTextCache, genomes: Sequence[str],
                  categories: Sequence[str]
                  ) -> Tuple[np.ndarray, Dict[str, object]]:
    """[N, C+1] matrix rows for every corpus row and every cell column.

    Column `c < C` is `<genome>|<categories[c]>`; the last column is the
    genome marginal `<genome>|*`, used for rows that draw no category. There
    are C+1 columns whatever the label's column count is, so the text key does
    not silently follow `include_marginal`.

    `distinct` and `degenerate_frac` are the point of the return value: if
    every category column of a row resolves to one matrix row, the artefact is
    keyed per genome and the channel is the 19-valued variable the species
    token already carries in full.
    """
    cats = tuple(str(c) for c in categories)
    C = len(cats)
    n = len(genomes)
    rows = np.full((n, C + 1), -1, dtype=np.int64)
    uniq = sorted(set(str(g) for g in genomes))
    per_genome = {g: cache.rows_for([cell_key(g, c) for c in cats]
                                    + [cell_key(g, None)]) for g in uniq}
    for i, g in enumerate(genomes):
        rows[i] = per_genome[str(g)]

    covered_cat = (rows[:, :C] >= 0)
    covered_marg = (rows[:, C] >= 0)

    # Degeneracy is measured on the VECTORS, not on the row indices. An
    # artefact writing one genome vector under seven category keys has seven
    # distinct rows and the information of one, and counting row indices would
    # call that non-degenerate.
    _, canon = np.unique(np.asarray(cache.matrix), axis=0,
                         return_inverse=True)
    canon = np.asarray(canon, dtype=np.int64).reshape(-1)
    degenerate = 0
    for i in range(n):
        got = rows[i, :C][covered_cat[i]]
        if got.size and int(np.unique(canon[got]).size) == 1:
            degenerate += 1
    hit = rows[rows >= 0]
    stats: Dict[str, object] = {
        "n_rows": int(n),
        "n_categories": int(C),
        "per_column": [int(covered_cat[:, c].sum()) for c in range(C)],
        "rows_with_any_category": int(covered_cat.any(axis=1).sum()),
        "rows_with_marginal": int(covered_marg.sum()),
        "distinct": int(np.unique(canon[hit]).size) if hit.size else 0,
        "distinct_rows": int(np.unique(hit).size) if hit.size else 0,
        "degenerate_frac": (float(degenerate) / float(n) if n else 0.0),
        "category_names": cats,
    }
    return rows, stats
