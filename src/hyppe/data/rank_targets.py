"""The rank scale: values from `rank.tsv.gz`, validity from `combat.tsv.gz`.

The two matrices answer different questions and each is wrong for the other's
job. `combat.tsv.gz` is the batch-corrected continuous value the heads
regress; its units are arbitrary, and an absent cell is NaN, which makes it
the only honest source of "does this strain have this gene at all" -- 28.6 %
of cells are empty. `rank.tsv.gz` is the within-library rank, and the
project's operational scale is

    y = 1 - mean(rank within library),   y in [0, 1], higher = stronger

with `tef1` at 0.9983, native `pks1` at 0.9437 / 0.9134 / 0.9546 across three
libraries, and a block of silent genes at exactly 0.0000. Every design target
in the brief is quoted on this scale, so a model whose output cannot be placed
on it cannot be checked against the brief.

The trap is that 1.0 means two things: `rank` imputes missing cells with
exactly 1.0 (5 279 648 cells checked, no counterexamples), and 1.0 is also the
legitimate value of a gene at the bottom of its library, so `y = 0` would
conflate "this strain has no such gene" with "this gene is switched off" on
28.6 % of cells. This module therefore takes the values from rank and the
validity from combat, and `report()["imputed_kept"]` counts the cells that
survived the mask at exactly 1.0, which is what says whether the join worked.

It does not replace the ComBat target. A rank is bounded, not comparable
across libraries of different size, and destroys the distance between two
strong promoters far apart in expression, which ComBat keeps. So `combat`
trains the heads, `y` places a prediction on the scale the brief uses, and
:func:`agreement` measures how far the two orderings differ rather than
letting a reader assume they are interchangeable.

The shape, column layout and row index are taken from `labels.Targets` rather
than re-derived: two tables that must align and are built by two functions is
how they come to disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .labels import SamplePolicy, SampleTable, Targets, read_matrix

__all__ = [
    "RANK_IMPUTED_VALUE",
    "RankTargets",
    "build_rank_targets",
    "agreement",
]

#: The value `rank.tsv.gz` writes for a cell that does not exist. Named
#: because a bare `1.0` in a comparison carries a whole failure mode.
RANK_IMPUTED_VALUE = 1.0

_EPS = 1e-12


@dataclass
class RankTargets:
    """`y = 1 - rank`, on the same grid as `labels.Targets`."""

    og_ids: List[str]
    genomes: List[str]
    y: np.ndarray                    # [n_pairs, C+1] mean of (1 - rank)
    y_sigma: np.ndarray              # [n_pairs, C+1] sd over those samples
    n_obs: np.ndarray                # [n_pairs, C+1]
    valid: np.ndarray                # [n_pairs, C+1] -- FROM COMBAT
    residual_index: Dict[Tuple[str, str], int]
    categories: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)
    _imputed_kept: int = 0
    _imputed_masked: int = 0

    @property
    def marginal_column(self) -> int:
        return int(self.y.shape[1]) - 1

    def column_of(self, category_ix: int) -> int:
        return self.marginal_column if category_ix < 0 else int(category_ix)

    def of(self, og: str, genome: str, category_ix: int = -1) -> float:
        """`y` for one cell, NaN when the cell is not valid."""
        i = self.residual_index.get((str(og), str(genome)))
        if i is None:
            return float("nan")
        c = self.column_of(category_ix)
        return float(self.y[i, c]) if bool(self.valid[i, c]) else float("nan")

    def report(self) -> Dict[str, object]:
        v = self.valid
        vals = self.y[v]
        vals = vals[np.isfinite(vals)]
        return {
            "n_pairs": int(self.y.shape[0]),
            "n_columns": int(self.y.shape[1]),
            "categories": list(self.categories),
            "cells_valid": int(v.sum()),
            "y_min": float(vals.min()) if vals.size else float("nan"),
            "y_max": float(vals.max()) if vals.size else float("nan"),
            "y_mean": float(vals.mean()) if vals.size else float("nan"),
            # Cells that survived the combat mask at exactly the imputed
            # value. A large number means the mask did not do its job and
            # `y = 0` is being taught as "silent" for absent genes.
            "imputed_kept": int(self._imputed_kept),
            "imputed_masked": int(self._imputed_masked),
            "imputed_kept_fraction": (float(self._imputed_kept)
                                      / max(int(v.sum()), 1)),
            "notes": list(self.notes),
        }

    def format(self) -> str:
        r = self.report()
        return ("rank targets: %d cells valid, y in [%.4f, %.4f] mean %.4f | "
                "%d cells at the imputed 1.0 survived the combat mask (%.5f "
                "of valid), %d were masked by it"
                % (r["cells_valid"], r["y_min"], r["y_max"], r["y_mean"],
                   r["imputed_kept"], r["imputed_kept_fraction"],
                   r["imputed_masked"]))


def build_rank_targets(rank_path: str, samples: SampleTable,
                       combat: Targets, min_obs: int = 2) -> RankTargets:
    """`y = 1 - rank` per (orthogroup, genome, category), masked by `combat`.

    The row order, column layout and category tuple are taken from `combat`,
    not re-derived: the failure when two implementations of one grid drift is
    silent, every array keeping its shape while every row means something
    else.
    """
    og_ids, cols, mat = read_matrix(rank_path)
    pos = {s: j for j, s in enumerate(cols)}
    notes: List[str] = []

    cats = tuple(combat.categories) or tuple(SamplePolicy().categories)
    C = len(cats)
    n_col = int(combat.mu.shape[1])
    has_marginal = n_col == C + 1

    # Which kept sample columns back each (genome, category) cell. Identical
    # rule to `build_targets`: a sample in two categories appears in both.
    by_genome: Dict[str, List[int]] = {}
    by_cell: Dict[Tuple[str, int], List[int]] = {}
    for sid, gen, keep, catrow in zip(samples.sample_ids, samples.genome,
                                      samples.keep, samples.category):
        if not keep:
            continue
        j = pos.get(sid)
        if j is None:
            continue
        by_genome.setdefault(gen, []).append(j)
        row = np.asarray(catrow)
        for c in range(min(C, int(row.shape[0]))):
            if float(row[c]) > 0.0:
                by_cell.setdefault((gen, c), []).append(j)
    if not by_genome:
        raise ValueError(
            "no kept sample of the metadata appears as a column of %r. The "
            "join key is the `sample` column against the matrix header; a "
            "silent empty intersection would produce an all-NaN rank table "
            "with the right shape." % (rank_path,))

    og_row = {o: i for i, o in enumerate(og_ids)}
    n_pairs = len(combat.og_ids)
    y = np.full((n_pairs, n_col), np.nan, dtype=np.float64)
    ys = np.full((n_pairs, n_col), np.nan, dtype=np.float64)
    nob = np.zeros((n_pairs, n_col), dtype=np.int32)
    missing_og = 0

    for i, (og, gen) in enumerate(zip(combat.og_ids, combat.genomes)):
        gi = og_row.get(og)
        if gi is None:
            missing_og += 1
            continue
        row = mat[gi]
        for col in range(n_col):
            if has_marginal and col == C:
                cidx = by_genome.get(gen, ())
            else:
                cidx = by_cell.get((gen, col), ())
            if not len(cidx):
                continue
            sub = row[list(cidx)]
            ok = np.isfinite(sub)
            n = int(ok.sum())
            nob[i, col] = n
            if n == 0:
                continue
            # Taken per sample and then averaged, not 1 - mean(rank): the two
            # agree while the map is affine, and this form survives a scale
            # that stops being affine.
            vals = 1.0 - sub[ok].astype(np.float64)
            y[i, col] = float(vals.mean())
            if n >= int(min_obs):
                ys[i, col] = float(vals.std(ddof=1))

    # Validity comes from combat, always: rank cannot distinguish "absent"
    # from "silent".
    valid = np.asarray(combat.valid, dtype=bool) & np.isfinite(y)
    at_imputed = np.isclose(y, 1.0 - RANK_IMPUTED_VALUE, atol=1e-9)
    imputed_kept = int((at_imputed & valid).sum())
    imputed_masked = int((at_imputed & ~valid).sum())

    if missing_og:
        notes.append("%d of %d (orthogroup, genome) rows have no orthogroup "
                     "row in %s and are masked"
                     % (missing_og, n_pairs, rank_path.split("/")[-1]))
    notes.append("validity taken from the ComBat mask, never from rank: rank "
                 "writes exactly %.1f for an absent cell and %.1f is also a "
                 "legitimate bottom-of-library value, so a mask from rank "
                 "conflates 'no such gene' with 'switched off' on 28.6 %% of "
                 "cells" % (RANK_IMPUTED_VALUE, RANK_IMPUTED_VALUE))
    if imputed_kept:
        notes.append(
            "%d valid cells sit at exactly y = 0.0 (rank 1.0). Some of those "
            "are genuinely silent genes -- the measured silent block is "
            "11.93 %% of raw data -- but a large share would mean the combat "
            "mask did not remove the absent ones" % (imputed_kept,))

    return RankTargets(
        og_ids=list(combat.og_ids), genomes=list(combat.genomes),
        y=y.astype(np.float32), y_sigma=ys.astype(np.float32),
        n_obs=nob, valid=valid,
        residual_index=dict(combat.residual_index), categories=cats,
        notes=notes, _imputed_kept=imputed_kept,
        _imputed_masked=imputed_masked)


def agreement(combat: Targets, rank: RankTargets,
              column: Optional[int] = None,
              max_cells: int = 200000, seed: int = 7898
              ) -> Dict[str, float]:
    """How far the two scales order the same cells differently.

    They should agree closely, rank being a monotone summary of the same
    measurements ComBat corrects, so a low correlation means one matrix is not
    what it claims -- a wrong join, a transposed file, a different sample set.
    A high correlation is not a licence to use them interchangeably: Spearman
    near 1 with Pearson well below it is what a monotone but strongly
    non-linear map looks like, and is why a threshold set on one scale cannot
    be moved to the other.
    """
    col = (rank.marginal_column if column is None else int(column))
    v = np.asarray(combat.valid, dtype=bool)[:, col] & \
        np.asarray(rank.valid, dtype=bool)[:, col]
    a = np.asarray(combat.mu, dtype=np.float64)[:, col][v]
    b = np.asarray(rank.y, dtype=np.float64)[:, col][v]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size > int(max_cells):
        pick = np.random.default_rng(int(seed)).choice(
            a.size, size=int(max_cells), replace=False)
        a, b = a[pick], b[pick]
    if a.size < 8 or a.std() < _EPS or b.std() < _EPS:
        return {"n": float(a.size), "pearson": float("nan"),
                "spearman": float("nan")}
    p = float(np.corrcoef(a, b)[0, 1])
    s = float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])
    return {
        "n": float(a.size),
        "column": float(col),
        "pearson": p,
        "spearman": s,
        # Spearman >> Pearson: the signature that forbids moving a threshold
        # between the two scales.
        "monotone_but_nonlinear": bool(s - p > 0.05),
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(x.size, dtype=np.float64)
    s = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and s[j + 1] == s[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out
