"""Siamese pair sampling: which pairs exist, which are thrown away, and why.

Every constant here is a measurement on data/final_datasets under the sample
policy of `labels.py` (2332 -> 1498 kept samples, 19 genomes, 7 categories).

The conditioning cell is the genome, optionally (genome, category). A
per-sample cell is refused rather than discouraged: the missingness mask is a
property of (orthogroup, genome), uniform across every genome's own samples,
so a per-sample cell buys no new sequence pair, while its pooled sd of 0.6740
against 0.1116 would make the filter below reject 44.4 % of draws instead of
9.29 %. Category cells exist for the 52 with at least four samples and
reorder pairs on 1.32 % of draws against a floor of 0.0001.

Pairs live inside one cell. The combat scales are aligned across genomes, so
the temptation to pair across them is real and the refusal explicit: the sign
of such a pair contradicts the within-genome order in 7.47 % of cases.
Orthologue order genuinely differs between genomes (reversals 0.0576 against
a within-genome floor of 0.0001, and 0.0025 between studies of one genome),
so it is biology rather than batch and no single conditioning can resolve it.

The margin is the rejection threshold, per pair: m_ij = 1.96 * sqrt(se_A^2 +
se_B^2). One number used twice, so every surviving pair satisfies
|mu_A - mu_B| >= m_ij by construction and none is pushed further apart than
its own measured difference. It is a calibration knob and not a ranking one:
rho moves 0.004 over the whole range while the calibration slope moves 0.8591
(no ranking term) -> 1.0032 (at the 0.33 global equivalent) -> 0.8115 (at
2.157), so config validation refuses a margin at or above 1.0. The filter
itself costs 9.29 % of draws and takes the expected fraction of pairs with
the wrong label sign from 1.85 % to 0.05 %.

Negatives are gene bodies only, at 0.05 negative pairs per positive pair.
Flanks are refused: 22.2 % of FlankRight and 11.5 % of FlankLeft windows
contain another gene's promoter verbatim, so every fifth "guaranteed weaker"
flank is a promoter. The ratio is steered by pair count and never by loss
share, because that share is sublinear -- 8.55 % at f = 0.25 and 10.17 % at
f = 1.00, where rho has already fallen 0.068 -- so gating on it would admit
four times too many negatives without moving the number being watched.
Without any, the ranking head places non-promoters in the middle of the
distribution (AUC 0.5015) and the bottom of the scale does not exist. They
carry no combat label, so they are masked out of (mu, sigma).

This module samples pairs and hands out index arrays plus the per-pair
margin. It does not encode sequences, build the conditioning vector or own
the loss; `reference_hinge` is a numpy reference for the tests and the
falsifiers, not the training path.

Two limits travel with the numbers. Every end-to-end figure above comes from
a linear 4-mer surrogate with a ceiling of rho ~0.39 on P1, so signs and
orders of magnitude transfer but digits do not; the data-side numbers are
free of that caveat. And gene-body negatives also teach codon usage -- a gene
body predicts mu at rho 0.5141, better than its own promoter -- for which a
dinucleotide shuffle is the obvious alternative, as yet unmeasured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .negatives import PromoterKmerIndex

# Constants
#: Rejection threshold and margin, in units of the pooled label error. One
#: number by construction; `PairConfig` refuses to let the two drift apart.
FILTER_Z: float = 1.96
#: Global equivalent of the per-pair margin: the median over genomes of
#: 1.96 * sqrt(se_A^2 + se_B^2). Calibration slope 1.0032 at this value.
GLOBAL_MARGIN: float = 0.33
#: Margin for a (promoter, gene body) pair. A negative has no `se`, so the
#: per-pair rule has nothing to compute; the global equivalent is used.
NEG_MARGIN: float = 0.33
#: Negative pairs per positive pair. 0.05 -> AUC_neg 0.6948 with rho intact;
#: 0.10 already costs -0.021 rho.
F_NEG: float = 0.05
#: Fraction of draws the 1.96-sigma filter is expected to reject, weighted
#: by example count (per genome 0.0393 reesei ... 0.2239 citrinoviride).
EXPECTED_REJECT_FRACTION: float = 0.0929
#: Expected fraction of pairs whose sign is wrong, before and after the
#: filter. Phi(-|d| / se_pair), weighted by example count.
EXPECTED_SIGN_FLIP_RAW: float = 0.0185
EXPECTED_SIGN_FLIP_FILTERED: float = 0.0005
#: Pair accuracy of the trivial model that knows one global promoter order
#: and ignores the genome. A conditional head below this has not learned even
#: the consensus.
UNCONDITIONAL_PAIR_ACCURACY: float = 0.9493
UNCONDITIONAL_PAIR_ACCURACY_P1: float = 0.9227
#: Ceiling of the conditional branch, in percentage points of pair accuracy:
#: how often a single global order contradicts the truth per genome, after
#: the noise filter.
CONDITIONAL_CEILING_PP: float = 5.07
CONDITIONAL_CEILING_PP_P1: float = 7.73

#: Negative classes refused as ranking negatives, with the measurement.
#: Names match `negatives.NEGATIVE_CLASSES` so the two modules cannot
#: disagree about what a class is called.
BANNED_NEGATIVE_CLASSES: Dict[str, str] = {
    "flank_r": "22.2 % of FlankRight windows contain another gene's "
               "promoter verbatim -- every fifth negative is a positive",
    "flank_l": "11.5 % of FlankLeft windows contain another gene's "
               "promoter verbatim",
}
#: The only class measured clean (3 hits in 13 126 gene bodies = 0.02 %).
ALLOWED_NEGATIVE_CLASSES: Tuple[str, ...] = ("gene",)
CELL_MODES: Tuple[str, ...] = ("genome", "genome_category")
MARGIN_MODES: Tuple[str, ...] = ("per_pair", "constant")


def _phi_neg(x: np.ndarray) -> np.ndarray:
    """P(Z < -x) for a standard normal, vectorised, no scipy."""
    return 0.5 * np.array([math.erfc(float(v) / math.sqrt(2.0))
                           for v in np.atleast_1d(x)], dtype=np.float64)


# Config
@dataclass
class PairConfig:
    """Every knob, with the measurement that fixes its default.

    Validation raises rather than warns wherever a setting would silently
    invalidate the labels. A fallback in this layer is invisible in every
    training metric: the loss goes down normally while 7.47 % of the labels
    contradict themselves.
    """

    cell_mode: str = "genome_category"
    #: Rejection threshold in units of the pooled label error.
    filter_z: float = FILTER_Z
    #: Margin in the same units. Must equal `filter_z` in per-pair mode.
    margin_z: float = FILTER_Z
    margin_mode: str = "per_pair"
    #: Only for `margin_mode="constant"`. Values >= 1.0 are refused: slope
    #: 0.938 at 1.00 and 0.8115 at 2.16, against 0.8591 for no ranking term.
    margin_constant: float = GLOBAL_MARGIN
    #: Probability of narrowing a draw from the genome to one of its
    #: eligible (genome, category) cells.
    p_category: float = 0.3
    #: A category cell needs this many samples to be a cell at all; 52 of
    #: the 66 non-empty cells clear it.
    min_category_samples: int = 4
    f_neg: float = F_NEG
    neg_margin: float = NEG_MARGIN
    negative_class: str = "gene"
    #: Genome draw probability ~ n_samples ** this. Square root rather than
    #: linear, so reesei (418 samples) gets 4.2x the pull of P1 (24) instead
    #: of 17.4x, matching the weighting the mu target already uses.
    genome_weight_power: float = 0.5
    #: Candidates examined per cell before the cell is redrawn, and cells
    #: tried before a pair is given up on. Both are reported; an exhausted
    #: draw is a measurement, not a shrug.
    max_draw_tries: int = 64
    max_cell_tries: int = 8
    #: Reuse a promoter already in the batch as the positive half of a
    #: negative pair. It saves one encoder forward and the promoter's only
    #: role there is "top of the scale"; set False to decorrelate.
    reuse_positive_in_negative_pair: bool = True
    allow_cross_cell: bool = False
    seed: int = 7898

    def __post_init__(self) -> None:
        if self.cell_mode == "genome_sample":
            raise ValueError(
                "C = (genome, sample) is refused: the missingness mask is a "
                "property of (orthogroup, genome) -- 0 of 19 genomes breaks "
                "that -- so it yields no new sequence pair, while the "
                "single-cell sd is 0.6740 against 0.1116, which makes the "
                "1.96-sigma filter reject 44.4 % of draws instead of 9.3 %")
        if self.cell_mode not in CELL_MODES:
            raise ValueError("cell_mode must be one of %r, got %r"
                             % (CELL_MODES, self.cell_mode))
        if self.margin_mode not in MARGIN_MODES:
            raise ValueError("margin_mode must be one of %r, got %r"
                             % (MARGIN_MODES, self.margin_mode))
        if self.allow_cross_cell:
            raise ValueError(
                "cross-cell pairs are refused: the sign of a pair drawn "
                "from two genomes contradicts the within-genome order in "
                "7.47 % of cases, and the model has no way to resolve it "
                "because C would have to be two things at once")
        if self.margin_mode == "per_pair" and self.margin_z != self.filter_z:
            raise ValueError(
                "margin_z (%r) must equal filter_z (%r): the whole point of "
                "decision 3 is that the rejection threshold and the margin "
                "are ONE number, so that every surviving pair satisfies "
                "|mu_A - mu_B| >= m_ij by construction. Use "
                "margin_mode='constant' to sweep them apart deliberately."
                % (self.margin_z, self.filter_z))
        if self.margin_mode == "constant" and self.margin_constant >= 1.0:
            raise ValueError(
                "a constant margin >= 1.0 combat units is refused: measured "
                "calibration slope 0.9380 at 1.00 and 0.8115 at 2.16, worse "
                "than the 0.8591 of no ranking term at all (got %r)"
                % (self.margin_constant,))
        if self.filter_z < 0.0 or self.margin_z < 0.0:
            raise ValueError("filter_z and margin_z must be >= 0")
        if self.negative_class in BANNED_NEGATIVE_CLASSES:
            raise ValueError("negative class %r is banned: %s"
                             % (self.negative_class,
                                BANNED_NEGATIVE_CLASSES[self.negative_class]))
        if self.negative_class not in ALLOWED_NEGATIVE_CLASSES:
            raise ValueError(
                "negative class %r is not measured for this head; only %r "
                "is. A dinucleotide shuffle is the obvious candidate and is "
                "recorded as UNMEASURED (N2): the measurement that would "
                "authorise it is AUC(promoter vs shuffle) with a 4-mer "
                "probe, expected 0.5-0.6 against 0.8427 for gene bodies"
                % (self.negative_class, ALLOWED_NEGATIVE_CLASSES))
        if not 0.0 <= self.f_neg <= 0.10:
            raise ValueError(
                "f_neg must be in [0, 0.10]: at 0.25 the measured AUC_neg "
                "is 0.9239 with rho 0.3632 against 0.3866 -- the negatives "
                "eat the ranking task (got %r)" % (self.f_neg,))
        if not 0.0 <= self.p_category <= 1.0:
            raise ValueError("p_category must be in [0, 1]")


# Index
@dataclass
class CategoryStats:
    """mu / se / n per (orthogroup, genome, category), for the 52 cells.

    Kept apart from `labels.Targets` because it is a pairing concern: the
    regression target is the genome mean, while the category cell exists only
    to reorder pairs. Samples carrying several categories count in each of
    them, and that multi-hot semantics rests on 66 of 2332 samples, which is
    the honest limit of any claim about combinations.
    """

    genomes: List[str]
    n_categories: int
    mu: np.ndarray            # float32 [n_og, n_gen, n_cat]
    se: np.ndarray            # float32, NaN where n_obs < 2
    n_obs: np.ndarray         # int32
    valid: np.ndarray         # bool, n_obs >= 1
    n_samples: np.ndarray     # int32 [n_gen, n_cat], kept samples per cell


@dataclass
class PairIndex:
    """Everything the sampler needs, in RAM, with no pair materialised.

    750 135 940 pairs exist at cell = genome, which no realistic budget can
    exhaust, so materialising them would be pointless as well as impossible.
    The matrices here are 0.86 MB and 6.02 MB.
    """

    og_ids: List[str]
    genomes: List[str]
    mu: np.ndarray            # float32 [n_og, n_gen]
    se: np.ndarray            # float32, NaN filled with the genome median
    n_obs: np.ndarray         # int32
    valid: np.ndarray         # bool: labelled AND single-copy 800 bp
    example_row: np.ndarray   # int64 [n_og, n_gen], -1 where no sequence
    n_samples: np.ndarray     # int32 [n_gen]
    neg_rows: Dict[str, np.ndarray] = field(default_factory=dict)
    categories: Optional[CategoryStats] = None
    cat_se_filled: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    def genome_index(self) -> Dict[str, int]:
        """Genome name -> column of the mu / se / valid matrices."""
        return {g: i for i, g in enumerate(self.genomes)}

    def n_valid(self) -> int:
        """Examples a pair can be drawn from: labelled and single-copy."""
        return int(self.valid.sum())


def samples_per_genome(samples) -> Dict[str, int]:
    """Kept samples per genome, the weight base for the genome draw.

    Takes anything with `.genome` and `.keep`, i.e. `labels.SampleTable`;
    duck-typed so this module imports with no dependency on the label loader.
    """
    out: Dict[str, int] = {}
    for gen, keep in zip(samples.genome, samples.keep):
        if keep:
            out[gen] = out.get(gen, 0) + 1
    return out


def category_stats_from_matrix(og_ids: Sequence[str],
                               sample_cols: Sequence[str],
                               matrix: np.ndarray,
                               samples,
                               genomes: Optional[Sequence[str]] = None
                               ) -> CategoryStats:
    """Aggregate the combat matrix per (orthogroup, genome, category).

    A different aggregation from `labels.build_targets`, not a
    re-implementation: that one pools all kept samples of a genome, this one
    splits them by category so a pair can be drawn inside a condition. All
    cells are computed and only the eligible ones used, because the
    eligibility cut belongs to the sampler, where it is logged.
    """
    pos = {s: j for j, s in enumerate(sample_cols)}
    gens = list(genomes) if genomes is not None else sorted(
        set(g for g, k in zip(samples.genome, samples.keep) if k))
    gidx = {g: i for i, g in enumerate(gens)}
    n_cat = int(samples.category.shape[1])
    n_og = len(og_ids)
    n_gen = len(gens)

    cols: Dict[Tuple[int, int], List[int]] = {}
    for sid, gen, keep, vec in zip(samples.sample_ids, samples.genome,
                                   samples.keep, samples.category):
        if not keep or gen not in gidx:
            continue
        j = pos.get(sid)
        if j is None:
            continue
        for k in range(n_cat):
            if vec[k] > 0:
                cols.setdefault((gidx[gen], k), []).append(j)

    mu = np.full((n_og, n_gen, n_cat), np.nan, dtype=np.float32)
    se = np.full((n_og, n_gen, n_cat), np.nan, dtype=np.float32)
    nob = np.zeros((n_og, n_gen, n_cat), dtype=np.int32)
    nsamp = np.zeros((n_gen, n_cat), dtype=np.int32)
    for (gi, k), js in cols.items():
        nsamp[gi, k] = len(js)
        sub = matrix[:, js]
        ok = ~np.isnan(sub)
        cnt = ok.sum(axis=1)
        safe = np.where(ok, sub, 0.0).astype(np.float64)
        tot = safe.sum(axis=1)
        m = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        d2 = np.where(ok, (safe - m[:, None]) ** 2, 0.0).sum(axis=1)
        var = np.where(cnt > 1, d2 / np.maximum(cnt - 1, 1), np.nan)
        mu[:, gi, k] = m.astype(np.float32)
        nob[:, gi, k] = cnt.astype(np.int32)
        with np.errstate(invalid="ignore"):
            se[:, gi, k] = (np.sqrt(var) /
                            np.sqrt(np.maximum(cnt, 1))).astype(np.float32)
    return CategoryStats(gens, n_cat, mu, se, nob, nob >= 1, nsamp)


def _fill_se(se: np.ndarray, axis_median: np.ndarray) -> np.ndarray:
    out = se.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.take(axis_median, np.nonzero(bad)[-1])
    return out


def build_pair_index(targets,
                     example_index: Mapping[Tuple[str, str], int],
                     n_samples: Mapping[str, int],
                     negatives_by_genome: Optional[
                         Mapping[str, Sequence[int]]] = None,
                     categories: Optional[CategoryStats] = None,
                     ) -> PairIndex:
    """Pivot `labels.Targets` into matrices and attach the sequence rows.

    `valid` is the intersection of a combat label and a single-copy 800 bp
    promoter. The promoters without a label are multi-copy orthogroups
    belonging to the self-supervised track; their strength is unknown, so
    calling them weak would invent a label.

    `se` is sigma/sqrt(n_obs), with the n_obs < 2 cells filled by the median
    of their genome: a standard deviation from one observation is not a
    measurement, and dropping those cells would remove the quietest genomes
    rather than the noisiest cells.
    """
    og_order: Dict[str, int] = {}
    for og in targets.og_ids:
        if og not in og_order:
            og_order[og] = len(og_order)
    gens = sorted(set(targets.genomes))
    gidx = {g: i for i, g in enumerate(gens)}
    n_og, n_gen = len(og_order), len(gens)

    mu = np.full((n_og, n_gen), np.nan, dtype=np.float32)
    sd = np.full((n_og, n_gen), np.nan, dtype=np.float32)
    nob = np.zeros((n_og, n_gen), dtype=np.int32)
    val = np.zeros((n_og, n_gen), dtype=bool)
    row = np.full((n_og, n_gen), -1, dtype=np.int64)

    for i, (og, gen) in enumerate(zip(targets.og_ids, targets.genomes)):
        a, b = og_order[og], gidx[gen]
        mu[a, b] = targets.mu[i]
        sd[a, b] = targets.sigma[i]
        nob[a, b] = targets.n_obs[i]
        val[a, b] = bool(targets.valid[i])
    for (og, gen), r in example_index.items():
        a, b = og_order.get(og), gidx.get(gen)
        if a is not None and b is not None:
            row[a, b] = int(r)

    notes: List[str] = []
    labelled = int(val.sum())
    val = val & np.isfinite(mu) & (row >= 0)
    notes.append("valid = labelled AND single-copy promoter: %d of %d "
                 "labelled cells kept" % (int(val.sum()), labelled))

    with np.errstate(invalid="ignore", divide="ignore"):
        se = (sd / np.sqrt(np.maximum(nob, 1))).astype(np.float32)
    se = np.where(val, se, np.nan).astype(np.float32)
    med = np.full(n_gen, np.nan, dtype=np.float32)
    for j in range(n_gen):
        col = se[:, j]
        fin = col[np.isfinite(col)]
        med[j] = float(np.median(fin)) if fin.size else np.nan
    n_filled = int((val & ~np.isfinite(se)).sum())
    se_filled = _fill_se(se, med)
    se_filled = np.where(val, se_filled, np.nan).astype(np.float32)
    notes.append("se: %d of %d valid cells had n_obs < 2 and took the "
                 "genome median" % (n_filled, int(val.sum())))

    cat_se_filled = None
    if categories is not None:
        if list(categories.genomes) != gens:
            raise ValueError(
                "category stats cover genomes %r but the targets cover %r; "
                "a mismatched order would silently pair the wrong cells"
                % (list(categories.genomes)[:4], gens[:4]))
        cs = categories.se.copy()
        cs = np.where(categories.valid & val[:, :, None], cs, np.nan)
        for j in range(n_gen):
            for k in range(categories.n_categories):
                col = cs[:, j, k]
                fin = col[np.isfinite(col)]
                fill = float(np.median(fin)) if fin.size else float(med[j])
                bad = (~np.isfinite(col)) & categories.valid[:, j, k] \
                    & val[:, j]
                col[bad] = fill
                cs[:, j, k] = col
        cat_se_filled = cs.astype(np.float32)

    negs: Dict[str, np.ndarray] = {}
    for g, rows in (negatives_by_genome or {}).items():
        negs[g] = np.asarray(list(rows), dtype=np.int64)

    ns = np.array([int(n_samples.get(g, 0)) for g in gens], dtype=np.int32)
    return PairIndex(list(og_order), gens, mu, se_filled, nob, val, row, ns,
                     negs, categories, cat_se_filled, notes)


# Batch
@dataclass
class PairBatch:
    """One batch: pair elements in flat arrays, pairs as index arrays.

    Elements are not de-duplicated. The same promoter can appear twice, and a
    caller wanting to save an encoder forward should use `unique_examples()`;
    hiding that behind a flag would make the element arrays mean two
    different things. `regression_mask` keeps negatives out of the mu and
    sigma losses; `sigma_mask` is narrower still, since a sigma from one
    observation is not a measurement.
    """

    example_row: np.ndarray      # int64 [n_elem] row in the sequence table
    og: np.ndarray               # int32 [n_elem] orthogroup index, -1 for neg
    genome: np.ndarray           # int32 [n_elem] genome index (C)
    category: np.ndarray         # int8  [n_elem] category index or -1
    is_negative: np.ndarray      # bool  [n_elem]
    regression_mask: np.ndarray  # bool  [n_elem] enters L_mu
    sigma_mask: np.ndarray       # bool  [n_elem] enters L_sigma
    mu: np.ndarray               # float32 [n_elem], NaN for negatives
    se: np.ndarray               # float32 [n_elem], NaN for negatives
    pair_a: np.ndarray           # int32 [n_pairs] element position
    pair_b: np.ndarray           # int32 [n_pairs]
    pair_y: np.ndarray           # float32 [n_pairs] +1 if A ranks above B
    pair_margin: np.ndarray      # float32 [n_pairs] m_ij
    pair_delta: np.ndarray       # float32 [n_pairs] mu_A - mu_B, NaN for neg
    pair_is_negative: np.ndarray  # bool [n_pairs]
    draws: int = 0
    rejected: int = 0
    exhausted: int = 0

    def n_elements(self) -> int:
        """Pair elements in this batch; repeats are not de-duplicated."""
        return int(self.example_row.shape[0])

    def n_pairs(self) -> int:
        """Positive and negative pairs together."""
        return int(self.pair_a.shape[0])

    def n_positive_pairs(self) -> int:
        """Promoter-vs-promoter pairs inside one cell."""
        return int((~self.pair_is_negative).sum())

    def n_negative_pairs(self) -> int:
        """Promoter-vs-gene-body pairs; the bottom of the scale."""
        return int(self.pair_is_negative.sum())

    def f_neg_effective(self) -> float:
        """Realised negative:positive pair ratio -- the quantity to log.

        The configured f_neg is a target the accumulator reaches over many
        batches, so the loss weight must use this number rather than the
        config, or the two drift apart exactly where nobody looks.
        """
        pos = self.n_positive_pairs()
        return float(self.n_negative_pairs()) / pos if pos else 0.0

    def unique_examples(self) -> Tuple[np.ndarray, np.ndarray]:
        """(unique sequence rows, inverse index into them)."""
        uniq, inv = np.unique(self.example_row, return_inverse=True)
        return uniq, inv.astype(np.int64)

    def audit(self) -> Dict[str, object]:
        """Structural measurements of one batch, as numbers not verdicts.

        `falsifiers()` turns them into pass/fail, so the thresholds live in
        exactly one place.
        """
        a, b = self.pair_a, self.pair_b
        same_genome = self.genome[a] == self.genome[b]
        same_cat = self.category[a] == self.category[b]
        pos = ~self.pair_is_negative
        d = np.abs(self.pair_delta[pos])
        m = self.pair_margin[pos]
        viol = int((d < m - 1e-6).sum()) if d.size else 0
        neg_elem = np.nonzero(self.is_negative)[0]
        neg_in_reg = int(self.regression_mask[neg_elem].sum()) \
            if neg_elem.size else 0
        neg_with_mu = int(np.isfinite(self.mu[neg_elem]).sum()) \
            if neg_elem.size else 0
        # A negative may only ever be the B side of a negative pair.
        neg_set = set(neg_elem.tolist())
        misplaced = 0
        for i in range(self.n_pairs()):
            if not self.pair_is_negative[i]:
                if int(a[i]) in neg_set or int(b[i]) in neg_set:
                    misplaced += 1
            else:
                if int(a[i]) in neg_set or int(b[i]) not in neg_set:
                    misplaced += 1
        return {
            "n_pairs": self.n_pairs(),
            "n_positive_pairs": int(pos.sum()),
            "n_negative_pairs": self.n_negative_pairs(),
            "cross_genome_pairs": int((~same_genome).sum()),
            "cross_category_pairs": int((~same_cat).sum()),
            "margin_exceeds_delta": viol,
            "negatives_in_regression": neg_in_reg,
            "negatives_with_mu": neg_with_mu,
            "misplaced_negatives": misplaced,
            "f_neg_effective": round(self.f_neg_effective(), 6),
            "min_abs_delta": float(d.min()) if d.size else float("nan"),
            "median_abs_delta": float(np.median(d)) if d.size
            else float("nan"),
            "mean_margin": float(m.mean()) if m.size else float("nan"),
        }


# Sampler
class PairSampler:
    """Draws pairs inside one cell, rejects the indistinguishable ones.

    Draw a cell (genome with probability ~ sqrt(n_samples), then with
    probability `p_category` narrow to one of its eligible category cells);
    draw A and B from its valid examples; take d = mu_A - mu_B and
    s = sqrt(se_A^2 + se_B^2); reject and redraw while |d| < filter_z * s;
    emit y = sign(d) and m_ij = margin_z * s.

    Every rejected candidate is counted. A rejection fraction of exactly 0.00
    means the filter is not wired at all, and one above 0.30 means `se` is
    being read off single cells, where 0.444 was measured.
    """

    def __init__(self, index: PairIndex, cfg: Optional[PairConfig] = None
                 ) -> None:
        self.index = index
        self.cfg = cfg or PairConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n_draws = 0
        self.n_rejected = 0
        self.n_self_draws = 0
        self.n_exhausted = 0
        self.n_pairs_out = 0
        self.n_neg_pairs_out = 0
        self.n_category_cells_used = 0
        self._flip_sum_examined = 0.0
        self._flip_n_examined = 0
        self._flip_sum_first = 0.0
        self._flip_n_first = 0
        self._flip_sum_kept = 0.0
        self._neg_debt = 0.0

        idx = index
        members: List[np.ndarray] = []
        for j in range(len(idx.genomes)):
            members.append(np.flatnonzero(idx.valid[:, j]).astype(np.int64))
        self._members = members
        usable = [j for j, m in enumerate(members) if m.size >= 2]
        if not usable:
            raise ValueError("no genome has two valid examples; a pair "
                             "sampler over one example is not a sampler")
        self._gidx = np.array(usable, dtype=np.int64)
        w = np.maximum(idx.n_samples[self._gidx].astype(np.float64), 1.0)
        w = w ** float(self.cfg.genome_weight_power)
        self._gp = w / w.sum()

        self._cat: Dict[int, Tuple[np.ndarray, np.ndarray,
                                   List[np.ndarray]]] = {}
        if self.cfg.cell_mode == "genome_category":
            cs = idx.categories
            if cs is None or idx.cat_se_filled is None:
                raise ValueError(
                    "cell_mode='genome_category' without category stats: "
                    "falling back to the genome silently is exactly the "
                    "failure this repo keeps re-finding (four flags that "
                    "killed a loss and left the mechanism running). Pass "
                    "categories=... to build_pair_index or set "
                    "cell_mode='genome'.")
            for j in usable:
                ks, mem = [], []
                for k in range(cs.n_categories):
                    if cs.n_samples[j, k] < self.cfg.min_category_samples:
                        continue
                    mm = np.flatnonzero(cs.valid[:, j, k] & idx.valid[:, j])
                    if mm.size >= 2:
                        ks.append(k)
                        mem.append(mm.astype(np.int64))
                if ks:
                    kw = np.array([cs.n_samples[j, k] for k in ks],
                                  dtype=np.float64)
                    kw = kw ** float(self.cfg.genome_weight_power)
                    self._cat[j] = (np.array(ks, dtype=np.int64),
                                    kw / kw.sum(), mem)
        elif self.cfg.p_category > 0.0:
            raise ValueError("p_category > 0 requires "
                             "cell_mode='genome_category'")

        if self.cfg.f_neg > 0.0:
            missing = [idx.genomes[j] for j in usable
                       if idx.neg_rows.get(idx.genomes[j],
                                           np.zeros(0)).size == 0]
            if missing:
                raise ValueError(
                    "f_neg > 0 but these genomes have no gene-body "
                    "negatives: %r. Drawing the negative from another "
                    "genome would make the pair cross-cell, which is the "
                    "7.47 %% error this module refuses elsewhere"
                    % (missing[:6],))

    # Drawing
    def _cell(self) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """(genome, category or -1, members, mu vector, se vector)."""
        j = int(self.rng.choice(self._gidx, p=self._gp))
        if (self.cfg.p_category > 0.0 and j in self._cat
                and self.rng.random() < self.cfg.p_category):
            ks, kp, mem = self._cat[j]
            t = int(self.rng.choice(len(ks), p=kp))
            k = int(ks[t])
            self.n_category_cells_used += 1
            return (j, k, mem[t], self.index.categories.mu[:, j, k],
                    self.index.cat_se_filled[:, j, k])
        return (j, -1, self._members[j], self.index.mu[:, j],
                self.index.se[:, j])

    def _try_cell(self, members: np.ndarray, mu_v: np.ndarray,
                  se_v: np.ndarray
                  ) -> Optional[Tuple[int, int, float, float]]:
        """Sequential rejection sampling, evaluated in one vector.

        Only the examined prefix is counted, so the accounting matches a
        one-at-a-time loop: a candidate after the accepted one was never
        drawn and must not appear in the rejection fraction.
        """
        t = int(self.cfg.max_draw_tries)
        ia = members[self.rng.integers(0, members.size, size=t)]
        ib = members[self.rng.integers(0, members.size, size=t)]
        same = ia == ib
        d = (mu_v[ia] - mu_v[ib]).astype(np.float64)
        s = np.sqrt(se_v[ia].astype(np.float64) ** 2
                    + se_v[ib].astype(np.float64) ** 2)
        ok = (~same) & (np.abs(d) >= self.cfg.filter_z * s)
        hit = np.flatnonzero(ok)
        end = int(hit[0]) + 1 if hit.size else t
        pref = slice(0, end)
        nsame = int(same[pref].sum())
        self.n_self_draws += nsame
        self.n_draws += end - nsame
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s[pref] > 0, np.abs(d[pref]) / np.maximum(s[pref],
                                                                   1e-12),
                         np.inf)
        flips = _phi_neg(q)
        keep = ~same[pref]
        self._flip_sum_examined += float(flips[keep].sum())
        self._flip_n_examined += int(keep.sum())
        # The first candidate of a draw is the only unbiased sample of the
        # unfiltered pair distribution: every later one exists because its
        # predecessor was rejected, so the examined multiset over-represents
        # near-ties. Against a filter-off run, where the rate is exact: true
        # 0.02404, first-candidate 0.02408, examined-prefix 0.02531. The bias
        # grows with the rejection rate.
        if not bool(same[0]):
            self._flip_sum_first += float(flips[0])
            self._flip_n_first += 1
        if not hit.size:
            self.n_rejected += end - nsame
            self.n_exhausted += 1
            return None
        self.n_rejected += end - 1 - nsame
        self._flip_sum_kept += float(flips[end - 1])
        return int(ia[hit[0]]), int(ib[hit[0]]), float(d[hit[0]]), \
            float(s[hit[0]])

    def _margin(self, s: float) -> float:
        if self.cfg.margin_mode == "constant":
            return float(self.cfg.margin_constant)
        return float(self.cfg.margin_z * s)

    def sample(self, n_pairs: int) -> PairBatch:
        """Draw `n_pairs` positive pairs plus the negatives they earn.

        The negative count is carried in a fractional accumulator across
        batches, so f_neg = 0.05 with batch 64 gives 3 or 4 per batch and
        exactly 1 per 20 pairs in the long run.
        """
        idx = self.index
        ex_row: List[int] = []
        og_l: List[int] = []
        gen_l: List[int] = []
        cat_l: List[int] = []
        neg_l: List[bool] = []
        mu_l: List[float] = []
        se_l: List[float] = []
        nob_l: List[int] = []
        pa: List[int] = []
        pb: List[int] = []
        py: List[float] = []
        pm: List[float] = []
        pd: List[float] = []
        pneg: List[bool] = []
        d0, r0, e0 = self.n_draws, self.n_rejected, self.n_exhausted

        def push(og: int, j: int, k: int, negative: bool) -> int:
            """Append one pair element; `og` is a sequence row when
            negative and an orthogroup index otherwise. A gene body
            belongs to no orthogroup and must not be given one: that
            is what keeps mu NaN and the regression mask False."""
            if negative:
                ex_row.append(int(og))     # already a sequence row
                og_l.append(-1)
                mu_l.append(np.nan)
                se_l.append(np.nan)
                nob_l.append(0)
            else:
                ex_row.append(int(idx.example_row[og, j]))
                og_l.append(int(og))
                if k >= 0:
                    mu_l.append(float(idx.categories.mu[og, j, k]))
                    se_l.append(float(idx.cat_se_filled[og, j, k]))
                    nob_l.append(int(idx.categories.n_obs[og, j, k]))
                else:
                    mu_l.append(float(idx.mu[og, j]))
                    se_l.append(float(idx.se[og, j]))
                    nob_l.append(int(idx.n_obs[og, j]))
            gen_l.append(int(j))
            cat_l.append(int(k))
            neg_l.append(bool(negative))
            return len(ex_row) - 1

        for _ in range(int(n_pairs)):
            got = None
            j = k = -1
            for _c in range(int(self.cfg.max_cell_tries)):
                j, k, members, mu_v, se_v = self._cell()
                got = self._try_cell(members, mu_v, se_v)
                if got is not None:
                    break
            if got is None:
                continue
            ia, ib, d, s = got
            if d < 0:               # A is always the higher-ranked side
                ia, ib, d = ib, ia, -d
            p_a = push(ia, j, k, False)
            p_b = push(ib, j, k, False)
            pa.append(p_a)
            pb.append(p_b)
            py.append(1.0)
            pm.append(self._margin(s))
            pd.append(d)
            pneg.append(False)

        n_pos = len(pa)
        self._neg_debt += n_pos * float(self.cfg.f_neg)
        n_neg = int(self._neg_debt)
        self._neg_debt -= n_neg
        for _ in range(n_neg):
            if self.cfg.reuse_positive_in_negative_pair and pa:
                p_pos = int(pa[int(self.rng.integers(0, len(pa)))])
                j = int(gen_l[p_pos])
            else:
                j, k, members, _mu, _se = self._cell()
                og = int(members[self.rng.integers(0, members.size)])
                p_pos = push(og, j, k, False)
            rows = idx.neg_rows[idx.genomes[j]]
            r = int(rows[self.rng.integers(0, rows.size)])
            p_neg = push(r, j, int(cat_l[p_pos]), True)
            pa.append(p_pos)
            pb.append(p_neg)
            py.append(1.0)
            pm.append(float(self.cfg.neg_margin))
            pd.append(np.nan)
            pneg.append(True)

        self.n_pairs_out += n_pos
        self.n_neg_pairs_out += n_neg
        neg_arr = np.array(neg_l, dtype=bool)
        nob_arr = np.array(nob_l, dtype=np.int32)
        return PairBatch(
            example_row=np.array(ex_row, dtype=np.int64),
            og=np.array(og_l, dtype=np.int32),
            genome=np.array(gen_l, dtype=np.int32),
            category=np.array(cat_l, dtype=np.int8),
            is_negative=neg_arr,
            regression_mask=~neg_arr,
            sigma_mask=(~neg_arr) & (nob_arr >= 2),
            mu=np.array(mu_l, dtype=np.float32),
            se=np.array(se_l, dtype=np.float32),
            pair_a=np.array(pa, dtype=np.int32),
            pair_b=np.array(pb, dtype=np.int32),
            pair_y=np.array(py, dtype=np.float32),
            pair_margin=np.array(pm, dtype=np.float32),
            pair_delta=np.array(pd, dtype=np.float32),
            pair_is_negative=np.array(pneg, dtype=bool),
            draws=self.n_draws - d0,
            rejected=self.n_rejected - r0,
            exhausted=self.n_exhausted - e0,
        )

    # Report
    def rejected_fraction(self) -> float:
        """Share of examined candidates the noise filter threw away.

        0.0929 conditioning on the genome; exactly 0.00 means the filter is
        not wired at all.
        """
        return (self.n_rejected / self.n_draws) if self.n_draws else 0.0

    def sign_flip_unfiltered(self) -> float:
        """Expected wrong-sign fraction of the unfiltered pair population.

        From the first candidate of every draw, the only one sampled without
        conditioning on rejection. This is the figure comparable with the
        measured 1.85 %; `sign_flip_examined` is not, and mistaking the two
        reads as a noisier corpus.
        """
        return (self._flip_sum_first / self._flip_n_first) \
            if self._flip_n_first else float("nan")

    def sign_flip_examined(self) -> float:
        """Same quantity over every candidate the sampler looked at.

        Biased upwards by the rejection loop; reported because it is what a
        naive counter in the training loop would print, and the gap between
        the two is the size of that trap.
        """
        return (self._flip_sum_examined / self._flip_n_examined) \
            if self._flip_n_examined else float("nan")

    def sign_flip_kept(self) -> float:
        """Expected wrong-sign fraction over the accepted pairs.

        The number the filter exists for: 1.85 % before and 0.05 % after, a
        factor of 37 for 9.29 % of the draws.
        """
        return (self._flip_sum_kept / self.n_pairs_out) if self.n_pairs_out \
            else float("nan")

    def report(self) -> Dict[str, object]:
        """Delegates to the module-level `report`, which is the one impl."""
        return report(self)


def report(sampler: PairSampler,
           batches: Optional[Sequence[PairBatch]] = None) -> Dict[str, object]:
    """Everything the log must carry for the gates to be readable.

    `negative_loss_share` is present and explicitly not a gate: at f = 1.00
    it reads 10.17 %, which looks innocent while rho has fallen by 0.068.
    The gates are the pair ratio and AUC(promoter vs gene body).
    """
    idx = sampler.index
    out: Dict[str, object] = {
        "cell_mode": sampler.cfg.cell_mode,
        "filter_z": sampler.cfg.filter_z,
        "margin_mode": sampler.cfg.margin_mode,
        "n_genomes_usable": int(sampler._gidx.size),
        "n_category_cells": int(sum(len(v[0]) for v in sampler._cat.values())),
        "n_valid_examples": idx.n_valid(),
        "draws": sampler.n_draws,
        "rejected": sampler.n_rejected,
        "rejected_fraction": round(sampler.rejected_fraction(), 6),
        "expected_rejected_fraction": EXPECTED_REJECT_FRACTION,
        "self_draws": sampler.n_self_draws,
        "exhausted_cells": sampler.n_exhausted,
        "positive_pairs": sampler.n_pairs_out,
        "negative_pairs": sampler.n_neg_pairs_out,
        "f_neg_configured": sampler.cfg.f_neg,
        "f_neg_realised": round(
            sampler.n_neg_pairs_out / sampler.n_pairs_out, 6)
        if sampler.n_pairs_out else 0.0,
        "sign_flip_unfiltered": round(sampler.sign_flip_unfiltered(), 6),
        "sign_flip_examined": round(sampler.sign_flip_examined(), 6),
        "sign_flip_kept": round(sampler.sign_flip_kept(), 6),
        "expected_sign_flip_raw": EXPECTED_SIGN_FLIP_RAW,
        "expected_sign_flip_filtered": EXPECTED_SIGN_FLIP_FILTERED,
        "category_draws": sampler.n_category_cells_used,
        "zero_se_valid_cells": int(
            (np.isfinite(idx.se) & (idx.se == 0)).sum()),
        "notes": list(idx.notes),
        "negative_loss_share": "diagnostic only -- NOT a gate (10.17 % at "
                               "f_neg=1.00 while rho fell 0.068)",
    }
    if batches:
        aud = [b.audit() for b in batches]
        out["batch_audit"] = {
            "cross_genome_pairs": sum(a["cross_genome_pairs"] for a in aud),
            "cross_category_pairs": sum(a["cross_category_pairs"]
                                        for a in aud),
            "margin_exceeds_delta": sum(a["margin_exceeds_delta"]
                                        for a in aud),
            "negatives_in_regression": sum(a["negatives_in_regression"]
                                           for a in aud),
            "misplaced_negatives": sum(a["misplaced_negatives"] for a in aud),
            "median_abs_delta": float(np.median(
                [a["median_abs_delta"] for a in aud])),
        }
    return out


# Loss and diagnostics
def reference_hinge(scores: np.ndarray, batch: PairBatch,
                    f_neg_effective: Optional[float] = None
                    ) -> Dict[str, float]:
    """Reference numpy implementation of the two ranking terms.

    L_rank = mean relu(m_ij - y (s_A - s_B)) over positive pairs;
    L_neg  = mean relu(m_neg - (s_P - s_N)) over negative pairs;
    L      = L_rank + f_neg_effective * L_neg, where f_neg_effective is the
    realised pair ratio of the batch, so a batch with one negative too many
    weighs it correctly instead of by the config constant. The training loop
    reimplements the same two lines in torch.
    """
    s = np.asarray(scores, dtype=np.float64)
    a, b = batch.pair_a, batch.pair_b
    d = batch.pair_y.astype(np.float64) * (s[a] - s[b])
    h = np.maximum(batch.pair_margin.astype(np.float64) - d, 0.0)
    pos, neg = ~batch.pair_is_negative, batch.pair_is_negative
    l_rank = float(h[pos].mean()) if pos.any() else 0.0
    l_neg = float(h[neg].mean()) if neg.any() else 0.0
    f = batch.f_neg_effective() if f_neg_effective is None \
        else float(f_neg_effective)
    tot = l_rank + f * l_neg
    share = (f * l_neg / tot) if tot > 0 else 0.0
    return {"l_rank": l_rank, "l_neg": l_neg, "f_neg_effective": f,
            "loss": tot, "negative_loss_share": share}


def pair_accuracy(scores: np.ndarray, batch: PairBatch) -> float:
    """Fraction of positive pairs ordered correctly. Floor is 0.9493.

    That floor is not chance but the accuracy of a trivial model that knows
    one global promoter order and ignores the genome (0.9227 for P1). A
    conditional head below it has not learned the consensus.
    """
    s = np.asarray(scores, dtype=np.float64)
    m = ~batch.pair_is_negative
    if not m.any():
        return float("nan")
    d = batch.pair_y[m] * (s[batch.pair_a[m]] - s[batch.pair_b[m]])
    return float((d > 0).mean())


def negative_score_pairs(scores: np.ndarray, batch: PairBatch
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """(promoter scores, gene-body scores) from this batch's negative pairs.

    Exposed because f_neg = 0.05 puts three negative pairs in a batch of 64,
    and an AUC over three against three is 1.0 or 0.0 far too often to gate
    on. The [0.65, 0.90] window is a statement about an epoch, so the caller
    pools these across batches and calls `auc_from_scores` once.
    """
    s = np.asarray(scores, dtype=np.float64)
    m = batch.pair_is_negative
    return s[batch.pair_a[m]], s[batch.pair_b[m]]


def auc_from_scores(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction, no scipy."""
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_p, n_n = p.size, n.size
    r_p = ranks[:n_p].sum()
    return float((r_p - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def negative_auc(scores: np.ndarray, batch: PairBatch) -> float:
    """AUC(promoter vs gene body) for one batch. Window [0.65, 0.90].

    Below 0.65 the bottom of the scale never formed (0.5015 at f_neg = 0);
    above 0.90 the negatives have eaten the ranking task (0.9239 at 0.25,
    with rho 0.3632 against 0.3866). One batch holds about three negative
    pairs, so gate on the pooled value over an epoch rather than this.
    """
    return auc_from_scores(*negative_score_pairs(scores, batch))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v: np.ndarray) -> np.ndarray:
        o = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[o] = np.arange(1, v.size + 1, dtype=np.float64)
        s = v[o]
        i = 0
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def calibration_report(mu: np.ndarray, mu_hat: np.ndarray
                       ) -> Dict[str, float]:
    """Slope of mu on mu_hat, plus the sd ratio.

    The slope is the only quantity that reacts to the margin, rho moving
    0.004 over the whole range. Below 0.95 the margin is too large, above
    1.05 too small.
    """
    y = np.asarray(mu, dtype=np.float64)
    p = np.asarray(mu_hat, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size < 3:
        return {"n": int(y.size), "slope": float("nan"),
                "sd_ratio": float("nan"), "r": float("nan"),
                "rho": float("nan")}
    vy, vp = y - y.mean(), p - p.mean()
    var_p = float((vp ** 2).mean())
    slope = float((vy * vp).mean() / var_p) if var_p > 0 else float("nan")
    sd_y, sd_p = float(y.std()), float(p.std())
    r = float((vy * vp).mean() / (sd_y * sd_p)) if sd_y > 0 and sd_p > 0 \
        else float("nan")
    return {"n": int(y.size), "slope": slope,
            "sd_ratio": (sd_p / sd_y) if sd_y > 0 else float("nan"),
            "r": r, "rho": _spearman(y, p)}


def conditional_gain_pp(scores_with_c: np.ndarray,
                        scores_without_c: np.ndarray,
                        batch: PairBatch) -> float:
    """Pair accuracy with the condition minus without, in percentage points.

    Must reach 1.0 pp by epoch 5 against a ceiling of 5.07 pp (median
    genome) and 7.73 pp for P1. Below that the conditional branch is a
    parameterised constant.
    """
    return 100.0 * (pair_accuracy(scores_with_c, batch)
                    - pair_accuracy(scores_without_c, batch))


def scan_negative_contamination(negative_seqs: Sequence[str],
                                promoter_seqs: Sequence[str],
                                tail: int = 60) -> Dict[str, object]:
    """Do the chosen negatives contain a promoter 3' end verbatim?

    The key is the exact last `tail` bp of every promoter, the stretch
    adjacent to the start codon and therefore the part a flank window would
    swallow. The scan carries its own positive control, because an index
    with a broken case mapping reports clean pools forever.
    """
    tails = [s.upper()[-tail:] for s in promoter_seqs
             if len(s) >= tail]
    if not tails:
        raise ValueError("no promoter is at least %d bp long" % tail)
    index = PromoterKmerIndex(k=tail).build(tails)
    ctrl = [index.hit_count(t) for t in tails[:min(32, len(tails))]]
    if any(c < 1 for c in ctrl):
        raise AssertionError(
            "positive control failed: a promoter tail is absent from the "
            "index built out of it (%r). Every contamination figure from "
            "this scan would read 0.000 for the wrong reason" % (ctrl[:8],))
    hits = [index.hit_count(s.upper()) for s in negative_seqs]
    n_hit = sum(1 for h in hits if h > 0)
    n = max(len(hits), 1)
    return {"n_negatives": len(hits), "n_contaminated": n_hit,
            "fraction": n_hit / n, "tail": tail,
            "n_promoter_keys": len(tails),
            "control_min_hits": int(min(ctrl))}


# Falsifiers
def falsifiers(sampler: PairSampler,
               batches: Optional[Sequence[PairBatch]] = None,
               contamination: Optional[Mapping[str, object]] = None,
               calibration: Optional[Mapping[str, float]] = None,
               auc_negative: Optional[float] = None,
               conditional_pp: Optional[float] = None,
               pair_acc: Optional[float] = None,
               ) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail). A check that cannot fail is not a check.

    The data-side checks run on a sampler and a batch and need no model. The
    model-side gates appear only when their measurement is handed in, so a
    missing number is visibly missing rather than silently passing.
    """
    cfg = sampler.cfg
    r = report(sampler, batches)
    out: List[Tuple[str, bool, str]] = []

    rej = float(r["rejected_fraction"])
    out.append((
        "F4 noise filter is wired and reads on the right object",
        0.01 <= rej <= 0.30 if cfg.filter_z > 0 else rej == 0.0,
        "rejected %.4f of %d draws (expected %.4f +- 0.02 conditioning on "
        "the genome, 0.112 with category cells mixed in; 0.00 would mean "
        "the filter is not wired, > 0.30 that se is read off single cells, "
        "where 0.444 was measured)"
        % (rej, r["draws"], EXPECTED_REJECT_FRACTION)))

    flip = sampler.sign_flip_kept()
    unf = sampler.sign_flip_unfiltered()
    out.append((
        "sign-flip fraction after the filter is below 0.002",
        bool(np.isfinite(flip) and flip <= 0.002) if cfg.filter_z > 0
        else bool(np.isfinite(flip)),
        "kept %.5f vs unfiltered %.5f (measured 0.0005 after / 0.0185 "
        "before; the filter is worth a factor of 37)" % (flip, unf)))

    out.append((
        "the filter removes noise rather than signal",
        bool(flip <= unf + 1e-12),
        "kept %.5f <= unfiltered %.5f (the unfiltered estimate uses the "
        "FIRST candidate of each draw; the examined-prefix figure %.5f is "
        "biased upwards by the rejection loop itself)"
        % (flip, unf, sampler.sign_flip_examined())))

    f_real = float(r["f_neg_realised"])
    out.append((
        "f_neg is realised by PAIR COUNT, not by loss share",
        abs(f_real - cfg.f_neg) <= 0.01,
        "realised %.4f vs configured %.4f over %d positive pairs; the loss "
        "share is sublinear (8.55 %% at f=0.25, 10.17 %% at f=1.00) and "
        "would hide a 4x overdose"
        % (f_real, cfg.f_neg, r["positive_pairs"])))

    if batches:
        aud = r["batch_audit"]
        out.append((
            "F7 every pair lives inside one cell",
            aud["cross_genome_pairs"] == 0 and aud["cross_category_pairs"]
            == 0,
            "cross-genome %d, cross-category %d (a cross-genome pair "
            "carries a wrong order in 7.47 %% of cases and no training "
            "metric shows it)" % (aud["cross_genome_pairs"],
                                  aud["cross_category_pairs"])))
        out.append((
            "F5 the margin never exceeds |mu_A - mu_B|",
            aud["margin_exceeds_delta"] == 0,
            "%d violations; in margin_mode=%r. This is an assertion, not a "
            "metric: if it breaks, the threshold and the margin have become "
            "two numbers and decision 3 no longer holds"
            % (aud["margin_exceeds_delta"], cfg.margin_mode)))
        out.append((
            "negatives are absent from the regression head",
            aud["negatives_in_regression"] == 0
            and aud["misplaced_negatives"] == 0,
            "%d negatives inside the mu mask, %d pairs with a negative in "
            "the wrong slot; a negative has no combat label at all"
            % (aud["negatives_in_regression"],
               aud["misplaced_negatives"])))
        out.append((
            "negative pairs exist at all when f_neg > 0",
            (sum(b.n_negative_pairs() for b in batches) > 0)
            if cfg.f_neg > 0 else
            (sum(b.n_negative_pairs() for b in batches) == 0),
            "%d negative pairs over %d batches at f_neg=%.3f; without them "
            "AUC(promoter vs gene body) measured 0.5015 -- the bottom of "
            "the scale does not exist"
            % (sum(b.n_negative_pairs() for b in batches), len(batches),
               cfg.f_neg)))

    out.append((
        "no draw was abandoned for lack of a distinguishable partner",
        r["exhausted_cells"] == 0,
        "%d exhausted cells over %d draws; a non-zero count means a cell "
        "whose examples are all within 1.96 sigma of each other"
        % (r["exhausted_cells"], r["draws"])))

    if contamination is not None:
        frac = float(contamination.get("fraction", 1.0))
        out.append((
            "F9 negatives contain no promoter 3' end",
            frac == 0.0,
            "%d of %d negatives hit a %s bp promoter tail (%.4f); measured "
            "0.0002 for gene bodies, 0.222 for FlankRight -- this scan is "
            "the only place that would catch flanks being added"
            % (contamination.get("n_contaminated", -1),
               contamination.get("n_negatives", -1),
               contamination.get("tail", "?"), frac)))

    if calibration is not None:
        sl = float(calibration.get("slope", float("nan")))
        sr = float(calibration.get("sd_ratio", float("nan")))
        out.append((
            "F1 calibration slope in [0.95, 1.05] and sd ratio in "
            "[0.90, 1.10]",
            bool(0.95 <= sl <= 1.05 and 0.90 <= sr <= 1.10),
            "slope %.4f, sd_ratio %.4f (0.8591 without a ranking term, "
            "1.0032 at margin 0.33, 0.8115 at 2.16; below 0.95 the margin "
            "is too large, above 1.05 too small)" % (sl, sr)))

    if auc_negative is not None:
        out.append((
            "F3 negatives neither dead nor dominant",
            bool(0.65 <= float(auc_negative) <= 0.90),
            "AUC(promoter vs gene body) %.4f; 0.5015 measured at f_neg=0, "
            "0.9239 at f_neg=0.25 where rho fell to 0.3632"
            % float(auc_negative)))

    if conditional_pp is not None:
        out.append((
            "F2 the conditional branch is alive (>= 1.0 pp)",
            float(conditional_pp) >= 1.0,
            "%.2f pp against a measured ceiling of %.2f pp (median genome) "
            "and %.2f pp for P1" % (float(conditional_pp),
                                    CONDITIONAL_CEILING_PP,
                                    CONDITIONAL_CEILING_PP_P1)))

    if pair_acc is not None:
        out.append((
            "F8 the head beats the unconditional order",
            float(pair_acc) > UNCONDITIONAL_PAIR_ACCURACY,
            "pair accuracy %.4f against the trivial global-order model at "
            "%.4f (%.4f for P1)" % (float(pair_acc),
                                    UNCONDITIONAL_PAIR_ACCURACY,
                                    UNCONDITIONAL_PAIR_ACCURACY_P1)))
    return out
