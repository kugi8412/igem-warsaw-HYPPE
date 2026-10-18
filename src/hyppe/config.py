"""Configuration: the sections, the arithmetic they imply, and the dead keys.

The failure this module is built against is not a wrong value but a key that is
SET and has no receiver, which reads exactly like a key that is set and works.
`jaspar: {}` is falsy, and eleven deep runs fell through it to 20 hand-written
motifs while every header said 296. `se_stat` was gated by a second key that
defaulted to 0. `yaml.safe_load` keeps the LAST of two duplicated keys without a
warning, so every grep on the first one reports the opposite of the effective
value. So beyond the dataclasses this module offers a loader that REFUSES a
silent duplicate, `report_inert_keys`, which lists the keys a config sets that
cannot have an effect, and `parameter_groups`, whose weight-decay exemption
matches by NAME SUBSTRING and is therefore a contract with the model's parameter
names -- one already broken once by a rename, after which the hyperboloid points
decayed off the manifold.

What the defaults encode. 800 bp, alphabet ACGT with N as a uniform 0.25;
strides (2,2,2,2,2) give maps 400/200/100/50/25 with level 0 the COARSEST (25
windows of 32 bp). Level 0 is a 128-code hyperbolic codebook, 7.00 bits per
position; the tangent levels carry FSQ budgets (9, 6, 3, 3), i.e. ternary grids
of 729/81/9/9 and 9.51/6.34/3.17/3.17 real bits. That totals 3187 bits against a
MEASURED 1598.7 bits of sequence, a reserve of 1.99x, and the reconstruction
floor is 0.2626, the unigram argmax -- not 0.25. Measured cost puts the
triangular path on the coarse map only and the band path on the full 800 bp at
batch 32. The convolutional receptive field reaches 134 bp of 800, 16.8 %, which
is why the encoder needs global paths at all.

Orthologues correlate 0.845-0.980 between genomes and an orthogroup spans 14.81
genomes on average, so effective n is ~12 400 rather than 167 724 and anything
other than `split_by: orthogroup` leaks near-duplicates across the split.
`sigma` needs n_obs >= 2, and `rank` imputes a missing cell with exactly 1.0,
which means "silent", so the validity mask must come from `combat`. Promoters
are NOT reverse-complement invariant: no strand canonicalisation, no RC
augmentation, no RC test-time averaging. Motif scanning still uses both strands,
which is occupancy symmetry and not sequence symmetry.

Neither torch nor yaml is imported at module level: a config has to be readable,
diffable and checkable on a machine that has neither.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import re
import typing
from dataclasses import dataclass, field
from typing import (
    Any,  # noqa: F401
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from .data.labels import DEFAULT_CATEGORIES

__all__ = [
    "ChannelConfig", "DataConfig", "ModelConfig", "PoolingConfig",
    "AbsoluteHeadConfig", "PairwiseHeadConfig", "DiscriminatorHeadConfig",
    "HeadsConfig", "NashConfig", "NormPriorConfig", "ConstraintConfig",
    "TrainingConfig",
    "ConditioningConfig",
    "HyppeConfig", "DuplicateKeyError",
    "load_config", "load_mapping", "scan_duplicate_keys", "to_mapping",
    "from_mapping", "active_config",
    "report_inert_keys", "parameter_groups", "fsq_grid",
    "UNIGRAM_BITS_PER_BP", "RECONSTRUCTION_FLOOR",
    "FORBIDDEN_PLAYER_FRAGMENTS", "FIXED_WEIGHT_COLLAPSE",
]

# Measured on this project's sequences, not assumed.
UNIGRAM_BITS_PER_BP = 1.9983
RECONSTRUCTION_FLOOR = 0.2626
LOG2_3 = math.log2(3.0)

#: Name fragments that may not appear among the Nash players. ONE source of
#: truth, imported by `hyppe.training.nash`: this file once matched the English
#: `"recon"` only, and a player named `rekonstrukcja` walked straight past it. A
#: blacklist is defeated by a name nobody thought of, so the load-bearing
#: contract downstream is the ALLOWLIST (`NashMTL.allowed_players`) and this is
#: only the fallback.
FORBIDDEN_PLAYER_FRAGMENTS: Tuple[str, ...] = (
    "recon", "reconstruction",                          # English
    "rekon", "rekonstrukcj", "odtworzen", "odbudow",    # Polish
)

#: The effective coefficient at which reconstruction collapsed 0.9998 -> 0.5549
#: in ONE epoch on the sibling track: a declared 0.15 times the `full` view's
#: 0.40. It is this project's default arithmetic too, which is why
#: `TrainingConfig.validate` refuses to reproduce it silently.
FIXED_WEIGHT_COLLAPSE = 0.060


def fsq_grid(bits: int) -> Tuple[int, int, float]:
    """(dimensions, cells, real bits) of the smallest ternary grid >= 2**bits.

    A finite scalar quantiser with m ternary dimensions has 3**m cells and
    carries m*log2(3) bits, so a declared budget of 9 bits becomes 6 dimensions,
    729 cells and 9.51 real bits. Both numbers are reported because the bit
    budget is computed from the real one, and because a declared budget is not
    what a run uses: a 729-cell grid was measured using 74 cells while
    `used_cells` reported all of them, having counted every cell touched EVER
    through the EMA history.
    """
    if bits <= 0:
        raise ValueError("fsq bit budget must be positive, got %r" % (bits,))
    dims = int(math.ceil(bits / LOG2_3))
    return dims, 3 ** dims, dims * LOG2_3


# Primitives
@dataclass
class ChannelConfig:
    """One conditioning channel, switchable on its own.

    `ablate` admits "none", "zero" and "substitute" only. Shuffling a
    conditioning channel inside a batch is an EMPTY intervention here, because
    batches are genome-homogeneous and permuting the genome changes nothing by
    construction; the measured +0.00000 that produced was read as a dead channel
    when it meant that no intervention had happened. Substitution -- give this
    sequence another example's channel -- is the only ablation of a conditioning
    channel that measures anything, and on the gene context it costs -0.1819 of
    SCC against a 0.74885 baseline.
    """

    enabled: bool = False
    dim: int = 0
    source: str = ""
    dropout: float = 0.0
    ablate: str = "none"

    def validate(self, path: str) -> None:
        if self.ablate not in ("none", "zero", "substitute"):
            raise ValueError(
                "%s.ablate = %r. 'shuffle' is forbidden: batches are "
                "genome-homogeneous, so shuffling a conditioning channel "
                "inside a batch changes nothing and the measured +0.00000 is "
                "an empty intervention, not a dead channel. Use 'substitute'."
                % (path, self.ablate))
        if self.enabled and self.dim <= 0:
            raise ValueError(
                "%s is enabled with dim=%d. An enabled channel of width zero "
                "is the `jaspar: {}` failure: set and inert."
                % (path, self.dim))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("%s.dropout must be in [0, 1)" % (path,))


# Data
@dataclass
class DataConfig:
    """Where the data is, what the target is, and how the split is drawn."""

    sequences_path: str = ""
    combat_path: str = ""
    samples_path: str = ""
    rank_path: str = ""
    flank_embedding_path: str = ""
    #: Sentence-LM cache of the experiment descriptions, keyed
    #: <genome>|<category>. Read by loop.py::build_corpus, and only when
    #: conditioning.experiment_text.enabled is true.
    text_embedding_path: str = ""

    #: Label the CELL (orthogroup, genome, category) instead of the genome
    #: marginal, and give the absolute head one output per cell. Off by
    #: default, so an unchanged config reproduces the previous run bit for bit.
    target_per_category: bool = False
    sequence_length: int = 800

    #: Description embeddings keyed by SAMPLE (`data/sample_text_emb.npz`).
    #: Empty = the cell regime, bit for bit as before. Setting it changes the
    #: TARGET: from the mean over a cell's samples to the ComBat vector of ONE
    #: drawn sample, read out through that sample's own description.
    #:
    #: The per-sample target is noisier -- two independent draws of the same
    #: cell correlate at 0.8466 -- so validation PCC FALLS even for a better
    #: model. A per-sample run may only be compared with a cell-regime one
    #: through the cell readout that `evaluate` reports alongside.
    sample_text_path: str = ""

    #: RANK matrix, read ONLY to take the pairwise label from it. Empty = the
    #: pairwise label comes from the same matrix as the value, as before. The
    #: sources are separable because the continuous head learns VALUES -- only
    #: ComBat relates a prediction to a measurement -- while the pairwise head
    #: learns ORDER, and the matrices differ in exactly what they do to order
    #: (within-sample Spearman against log2(TPM+1): ranks 0.9956, ComBat
    #: 0.8670). What it does not settle: ranks compress the top of the
    #: distribution where both anchors sit, so tef1 against pks1 in P1
    #: separates by 0.31 sd on ranks against 1.96 sd on ComBat.
    rank_path: str = ""
    #: Pair noise threshold IN RANK UNITS. It has to be given explicitly
    #: alongside `rank_path`: `min_gap` is in value units and carrying it over
    #: rejects a different share of pairs (98.50 % against 89.09 % above 0.05).
    min_gap_rank: float = 0.0
    alphabet: str = "ACGT"
    n_value: float = 0.25

    #: Largest admissible FRACTION of N bases in a corpus row. 0.0 = as before,
    #: dropping every row with anything outside ACGT. An admitted N enters as
    #: code 4 and becomes a uniform 0.25 column, the only encoding that adds no
    #: information. Measured over the 238 460 promoter rows: 767 carry a
    #: non-ACGT character and it is EVERY TIME an N; 109 rows are below 5 %,
    #: 658 above. The price this does NOT remove: the reconstruction
    #: cross-entropy has four classes, so an N is given target 'A' and counts
    #: in `rec_*` like any other -- at a threshold of 0.05 that is 1.2e-05 of
    #: positions, below the last digit `rec_full` is printed to.
    max_n_frac: float = 0.0

    # "mu" regresses the mean over conditions; "mu_sigma" additionally
    # regresses the MEASURED spread, which exists only where n_obs >= 2.
    target: str = "mu"
    min_obs_for_sigma: int = 2
    use_rank: bool = False
    mask_from_combat: bool = True
    #: How a row's weight is formed. "sqrt_n_obs" is the behaviour as it stood:
    #: the row weight is sqrt(n_obs) of its cell and an epoch is one pass over
    #: the split. "sqrt_n_obs_per_og" additionally balances the epoch by
    #: ORTHOGROUP, each drawn equally often. The correction lives in the SAMPLER
    #: and not in the loss, which is measured: the absolute head is given
    #: `weight=None` whenever `batch.mu_row` is set, which is every run in the
    #: per-sample regime, so a loss weight would be a key with no receiver for
    #: the second time -- and applying both would square the correction. The
    #: 1/k exponent is measured against the effective count k_eff =
    #: k/(1+(k-1)r) at ICC 0.8121: per-row RMS log error 0.0175 for 1/k, 0.1518
    #: for 1/sqrt(k), 0.3191 for a flat weight.
    sample_weight: str = "sqrt_n_obs"

    split_by: str = "orthogroup"
    allow_leaky_split: bool = False
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    split_seed: int = 7898

    batch_size: int = 32
    num_workers: int = 4
    # Promoters are directional. These three exist so the prohibition is
    # visible in the config rather than only in a comment.
    random_rc: bool = False
    rc_tta: bool = False
    canonicalise_strand: bool = False

    TARGETS = ("mu", "mu_sigma")
    #: Admissible `sample_weight` values. Listed here and not only in
    #: `batches.py`, because without validation a typo in this key would measure
    #: the default and say nothing about it.
    SAMPLE_WEIGHTS = ("sqrt_n_obs", "sqrt_n_obs_per_og")

    def validate(self) -> None:
        if self.target not in self.TARGETS:
            raise ValueError("data.target must be one of %r, got %r"
                             % (self.TARGETS, self.target))
        if self.sample_weight not in self.SAMPLE_WEIGHTS:
            raise ValueError(
                "data.sample_weight must be one of %r, got %r. Until "
                "21.09.2026 this key was read by no line of src/hyppe, so "
                "every value 'worked' -- refusing now is cheaper than a run "
                "that measures the default."
                % (self.SAMPLE_WEIGHTS, self.sample_weight))
        if self.sequence_length < 1:
            raise ValueError("data.sequence_length must be positive")
        if not 0.0 <= self.max_n_frac <= 1.0:
            raise ValueError(
                "data.max_n_frac is a FRACTION of the row, not a count of "
                "bases; got %r" % (self.max_n_frac,))
        if self.max_n_frac > 0.25:
            raise ValueError(
                "data.max_n_frac=%.4f admits rows that are more than a "
                "quarter unknown. Their one-hot is a uniform 0.25 column, "
                "which the reconstruction head cannot do better than chance "
                "on, and the row still carries a full-weight strength label. "
                "The corpus has 658 rows above 5 %% and they are the tail this "
                "key exists to keep OUT." % (self.max_n_frac,))
        if self.min_obs_for_sigma < 2:
            raise ValueError(
                "a standard deviation from fewer than 2 observations is not a "
                "measurement; build_targets writes NaN below this and the "
                "sigma term masks on it")
        if self.use_rank and not self.mask_from_combat:
            raise ValueError(
                "`rank` imputes missing cells with exactly 1.0 (5 279 648 "
                "cells checked, zero counterexamples) and 1.0 means 'silent', "
                "so a mask taken from `rank` conflates 'no such gene in this "
                "strain' with 'gene switched off' on 28.6 % of cells. The mask "
                "must come from `combat`.")
        if self.split_by != "orthogroup" and not self.allow_leaky_split:
            raise ValueError(
                "split_by=%r leaks near-duplicates across the split: "
                "orthologue correlation between genomes is 0.845-0.980 and an "
                "orthogroup spans 14.81 genomes on average, so effective n is "
                "~12 400 rather than 167 724. Set allow_leaky_split=True if "
                "that is the point of the run." % (self.split_by,))
        if self.random_rc or self.rc_tta or self.canonicalise_strand:
            raise ValueError(
                "promoters are not reverse-complement invariant: a promoter "
                "and its reverse complement are different objects, so RC "
                "augmentation, RC test-time averaging and strand "
                "canonicalisation are all forbidden here. Motif scanning still "
                "uses both strands; that is occupancy symmetry.")
        if not 0.0 <= self.val_fraction < 1.0 or \
                not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("split fractions must be in [0, 1)")


# Model
@dataclass
class MotifsConfig:
    """The bank of frozen motifs; until recently not settable from a config.

    Every default reproduces the previous behaviour bit for bit: an empty path
    takes `data/motifs.meme`, empty lists remove nothing, zero thresholds sift
    nothing.
    """

    #: Empty = data/motifs.meme. A Trichoderma-specific bank goes here: the
    #: present one was measured to hold 74 Neurospora profiles, 67 A. nidulans,
    #: and NO hit on CRE1, XYR1, BLR, VEL1 or ACE1.
    bank_path: str = ""
    #: Profile names to REMOVE. Names in the bank are unique (252 of 252), so
    #: the criterion is well defined.
    drop_names: Tuple[str, ...] = ()
    #: A non-empty list keeps ONLY the profiles named.
    keep_names: Tuple[str, ...] = ()
    #: Information-content threshold in bits; 0 = no threshold. The present
    #: bank has 93 profiles below 6 bits and 156 below 8.
    min_bits: float = 0.0
    #: Width threshold; 0 = no threshold. Widths in the bank run 6-22.
    max_width: int = 0
    #: A drop/keep name that is NOT in the bank RAISES. Without it a typo would
    #: run the FULL bank under the name of a run that declared an exclusion.
    strict_names: bool = True

    def validate(self, path: str = "model.motifs") -> None:
        if self.min_bits < 0.0:
            raise ValueError("%s.min_bits must be >= 0" % path)
        if int(self.max_width) < 0:
            raise ValueError("%s.max_width must be >= 0" % path)
        if self.drop_names and self.keep_names:
            raise ValueError(
                "%s: drop_names and keep_names were given AT THE SAME TIME. "
                "Composing two opposite criteria is ambiguous -- choose "
                "one." % path)
        if self.bank_path and not os.path.exists(self.bank_path):
            raise ValueError(
                "%s.bank_path = %r does not exist. Falling back silently to "
                "the default bank would run with a DIFFERENT bank from the "
                "one declared."
                % (path, self.bank_path))


@dataclass
class ModelConfig:
    """The ladder, its alphabets, and the global paths the receptive field
    forces."""

    strides: Tuple[int, ...] = (2, 2, 2, 2, 2)
    # Channels per level, COARSEST FIRST, matching `ConeLadder`.
    channels: Tuple[int, ...] = (192, 160, 128, 96, 64)
    stem_kernel: int = 5
    #: STAYS 6 on purpose, even though the stratified run sets 3. No YAML under
    #: configs/generated carries `n_res_blocks`, so every config takes this
    #: number from here and moving the default would silently change each of
    #: them: with `pre_dilations: [1, 2, 4, 8]` left in place the level-0 reach
    #: falls from 898 to 754 bp, i.e. coverage 0.9425, without changing a single
    #: shape. The 3 belongs in the run's own YAML together with
    #: `pre_dilations: [1]` and `level_blocks: [0, 1, 1, 1, 1]`; those three
    #: keys steer one reach from three places and are set TOGETHER or not at
    #: all.
    n_res_blocks: int = 6
    res_kernel: int = 7

    # Level 0 (the coarsest map) is quantised on the hyperboloid.
    l0_codes: int = 128
    l0_dim: int = 16                 # points are stored as [..., l0_dim + 1]
    #: Radius bound on the level-0 LIFT, the one route onto the hyperboloid
    #: with no length check of its own. Past cosh(8) the constraint
    #: <x,x>_L = -1 is no longer representable in float64. 0.0 = unbounded,
    #: which is what put the encoder at radius 41.33 and the commitment term at
    #: 2.66e16.
    l0_max_radius: float = 8.0
    #: Which function the commitment weight multiplies.
    #:   'chord' -- ``-<p,q>_L - 1 = cosh(d) - 1``, the behaviour of every
    #:              run so far, bit for bit.
    #:   'exact' -- ``d^2 / 2``. Equal to 'chord' to fourth order at the
    #:              origin, so the weight means the same thing at the
    #:              operating point; 1.47e14 times smaller at d = 40, so
    #:              the weight stays a weight when the encoder leaves.
    l0_commit_measure: str = "chord"
    #: Where the scale of the level-0 encoder map is pinned.
    #:   'none'  -- as written.
    #:   'scale' -- mean spatial norm per SEQUENCE set to sinh(1); the
    #:              relative radii inside a sequence survive.
    #:   'rms'   -- per POSITION, i.e. one shell for every point, which
    #:              costs the level its radial coordinate.
    #: Neither variant uses a batch statistic, so train and eval compute
    #: the same function.
    l0_lift_norm: str = "none"
    #: Soft radius saturation on the level-0 lift: r -> c*tanh(r/c). 0.0 = off.
    #: Identity to second order at the origin, bounded by c, and unlike the hard
    #: clamp at `l0_max_radius` it keeps a nonzero radial gradient at every
    #: finite radius -- which is why the hard clamp is only usable where it is
    #: rare. Measured: a hard clamp at 3.0 held 2.30 % of positions at step 35.
    l0_lift_squash: float = 0.0
    vq_ema_decay: float = 0.99
    vq_frechet: bool = True          # geodesic EMA target, not a coordinate mean
    vq_dead_restart: bool = True     # without it collapse is ABSORBING
    vq_learnable: bool = False

    # Tangent levels, finest last. One entry per level after level 0.
    fsq_bits: Tuple[int, ...] = (9, 6, 3, 3)
    fsq_gain: float = 0.7
    fsq_entropy_weight: float = 0.01
    #: RATE term on each FSQ grid. `fsq_entropy_weight` is NOT a rate: its
    #: expression H(p) - H(mean p) <= 0 pushes towards confident assignment
    #: under a uniform marginal, which drives bits UP. This key penalises the
    #: entropy of the marginal, i.e. the real code length. 0.0 = bit for bit as
    #: before. On its own it lowers `r` trivially (Goodhart), so it is
    #: meaningful only as an instrument: `r` at one value with and without
    #: `drab_pred`.
    fsq_rate_weight: float = 0.0

    #: Predict the RESIDUAL between ladder levels. False = bit for bit as
    #: before, with no tensors and no observables -- which the parameter ledger
    #: checks, because `occupancy_gate: true` without the code once produced
    #: zero tensors for a whole run and nobody saw it. At True the quantiser of
    #: level k+1 receives the FEATURE MINUS A PREDICTION from a window of parent
    #: points, and the prediction is added back on synthesis; without that
    #: return the code would still have to specify the whole step and the
    #: subtraction would remove no bits at all.
    drab_pred: bool = False
    #: Ancestor window width, ODD. A single parent point gives linear-regression
    #: R2 of 0.2043 / 0.0385 / 0.0424 / 0.0658; the 512.8 bits of redundancy
    #: only became locatable with a window of three.
    drab_pred_okno: int = 5
    drab_pred_ukryte: int = 32
    #: Weight of MSE(prediction, feature.detach()). Without this term the
    #: predictor has almost no gradient: the subtraction in analysis and the
    #: addition in synthesis cancel to first order through straight-through.
    drab_pred_waga: float = 0.1

    ladder_step_radius: float = 0.15
    #: CONE APERTURE per refinement level: the hard cap on the tangent step's
    #: NORM, i.e. on a geodesic LENGTH. A step of length d on a shell of radius
    #: r subtends about d / sinh(r), so a length must scale as sinh(r) to buy a
    #: CONSTANT angle; these are theta*sinh(r_k) at theta = 0.09 for the
    #: measured parent radii [2.496, 2.699, 2.963, 3.225]. The previous
    #: (0.25, 0.5) spread the angle 1.70x with the DEEPEST level worst served,
    #: and once the clamp is active ||scaled|| equals the limit exactly, so the
    #: FSQ cell's magnitude degree of freedom was erased for 84-99.7 % of
    #: positions. The aperture is a buffer, never a weight. Falsifiers on the
    #: `ladder` line: `step_clamped_frac` below 0.5 everywhere, `d_ratio` at or
    #: above 2.0 on every transition.
    ladder_aperture: Tuple[float, ...] = (0.542, 0.666, 0.869, 1.130)

    band_attention: bool = True
    band_width: int = 128
    triangular_levels: Tuple[int, ...] = (0,)   # coarse map only, by cost
    decoder_sees_codes_only: bool = True
    allow_continuous_decoder: bool = False
    position_channel: bool = True
    #: Kernel of the depthwise context convolution on the COARSEST map
    #: (`DampedGlobalContext`). ODD, and the default is the value every run so
    #: far built. At 800 bp that map has 25 windows, so a depthwise kernel sees
    #: every input window from every output position only at k = 49; at 15 it
    #: reaches 15 of 25, `receptive_field()` reports coverage 0.8625, and
    #: nothing in the convolutional path sees the whole promoter. Widening it
    #: touches the kernel only, not the damper or the merge.
    context_kernel: int = 15

    #: Dilations of the residual blocks between the stem and the first stride.
    #: `PromoterEncoder` has had this parameter since it was written and NO
    #: config passed it, so the encoder default (1, 2, 4) was the only value any
    #: run ever used. The blocks sit BEFORE the first stride on purpose: after
    #: a stride a 6-15 bp motif has lost its phase, so a wider view there is a
    #: view of a different signal. (1,) narrows them to a motif's reach, which
    #: is the point: at [1, 2, 4, 8] the pre-blocks alone covered 97.3 % of the
    #: finest level's receptive field while the whole tower added 93 bp, so the
    #: levels differed in SAMPLING DENSITY rather than in RANGE. Narrowing them
    #: alone BREAKS `rf_coverage_coarsest >= 1.0` (L0 falls to 646 bp of 800);
    #: the reach has to be handed back through `level_blocks`.
    pre_dilations: Tuple[int, ...] = (1,)
    occupancy_channels: int = 0
    #: Project the occupancy channels onto k before the stem and the billboard.
    #: 0 = off, bit for bit as before the module existed. The occupancy signal
    #: is effectively ~11-dimensional (participation ratio 10.8 of 284), so the
    #: full width pays parameters for twins (median max|r| 0.796).
    occupancy_bottleneck: int = 0
    #: Dropout kind inside the encoder blocks: "element" (single activations, as
    #: before) or "spatial" (WHOLE CHANNELS per sample, nn.Dropout1d).
    #: Neighbouring positions of a [B, dim, L] map share 91.7 % of their
    #: receptive field, so zeroing single activations is largely apparent --
    #: the model recovers them from the neighbour.
    dropout_kind: str = "element"
    #: Dilated residual blocks PER LEVEL of the tower, listed from the FINEST
    #: level. `PromoterEncoder` had them, no config set them and `assemble` did
    #: not pass them on, so the mechanism was unreachable. The tower alone adds
    #: only 93 bp of reach in total, so the stride grows 32x and the reach 1.6x
    #: and the levels differ in RESOLUTION, not RANGE -- measurably, since
    #: Var(between orthogroups)/Var(within) comes out identical on all five. A
    #: NUMBER = the same count on every level; a LIST of len(strides) = one per
    #: level, which is what a block on each of the five costing +18.4 % of
    #: parameters argues for. 0 = not built, but then `pre_dilations: (1,)`
    #: leaves L0 without coverage.
    level_blocks: Any = (0, 1, 1, 1, 1)
    #: Kernel of the per-level blocks.
    level_kernel: int = 3
    #: Dilations of the per-level blocks. A block spans (k-1)*max(d)+1 and adds
    #: (span-1)*stride, i.e. eight times the level's stride PER BLOCK at (1,2,4)
    #: and k = 3. A flat tuple = the same list on every level, as before; a list
    #: of lists = one list per level, ordered as `level_blocks`, i.e. from the
    #: FINEST.
    level_dilations: Any = (1,)
    #: Initial residual gate of the level blocks: a number, or a list over the
    #: levels ordered as `level_blocks`. Measured at 0.1 everywhere, a level
    #: block contributes 10 % of the amplitude, so a level's effective reach
    #: stays at the COMMON FLOOR (erf90 10.3 bp against L4's own 10.0) and the
    #: levels are copies in use. The gates do learn, but over 14 epochs they
    #: moved by +-13 %.
    level_gate0: Any = 0.1
    #: Gate on the tower's 1x1 BYPASS: a number or a list over the levels
    #: (ordered as `level_blocks`). 1.0 = identity, no parameter is created, bit
    #: for bit as before. Measured: the bypass carries a full-amplitude copy of
    #: the finer level's map, and removing it raises erf90 at L1 from 30.7 to
    #: 42.0 and at L3 from 12.0 to 14.0 with no change in parameter count.
    skip_gate0: Any = 1.0
    #: Correction to `pooled` computed from the statistics of the FINE levels
    #: inside each window of the COARSEST one. False = the module is not built
    #: and the model is bit for bit as before.
    sharpening: bool = False
    #: Width of the per-L0-window representation.
    sharpen_dim: int = 24
    #: How many positional BANDS over the L0 windows enter the readout; the
    #: boundaries are fractional, so the module never sees a sequence length.
    #: 0 = a readout invariant to permuting the windows, i.e. blind to WHERE a
    #: detail sits -- measured: without bands, swapping the contents of windows
    #: 5 and 10 moved the output by 5.96e-08, which is float32 noise.
    sharpen_bands: int = 4
    #: Which levels sharpen. 0 is forbidden: it is the window grid itself, and
    #: its own positional readout is a separate module.
    sharpen_levels: Any = (1, 2, 3, 4)
    #: POSITIONAL readout of level 0 through P equal bands of windows,
    #: projected into `pooled` by a ZERO-INITIALISED linear layer. False = the
    #: module is not built. It exists because `mean` and `sd` are invariant to
    #: permuting positions BY DEFINITION and `anchor` separates exactly two
    #: regions, while `sharpening` refuses to read level 0 -- so L0, the only
    #: level with measurable MI (4.24-6.83 % of its ceiling against 0.05-0.36 %
    #: on L1..L4), had no positional readout finer than the anchor. The
    #: argument is circular, though, since the dPCC under permutation was
    #: measured on a model that could not use position: a rise in dPCC WITHOUT
    #: a rise in `sel` is a NEGATIVE result.
    bands_l0: bool = False
    #: Number of bands. 8 is DERIVED from 1/`anchor_frac` = 1/0.125 rather than
    #: chosen. The announced identity "the last band is the anchor" is FALSE and
    #: can be checked on paper: at W = 25 windows the anchor is
    #: round(0.125*25) = 3 cells and band 8 is 4, and equality needs
    #: anchor_frac = 0.16. The grid's positive control is a different statement
    #: and is a genuine identity: the width-weighted mean of the bands equals
    #: the `L0/mean` slice.
    bands_l0_p: int = 8

    #: CONTENT-WEIGHTED readout for the single-sequence path. `pooling` gives
    #: the heads fixed functionals of position only, and the sole module here
    #: that weighs positions by CONTENT is `heads/pair_compare.py`, which takes
    #: TWO arguments; this is its one-argument form, with `h - mean_l(h)` in
    #: place of `h_a - h_b`. The ceiling of the axis is small and measured:
    #: permuting positions costs -0.04205 +- 0.00790 of dPCC against -0.52165
    #: for a global shuffle, i.e. 8.1 % of what the latent contributes, against
    #: an A/B resolution on one seed of about 0.016.
    attention_readout: bool = False
    #: Levels the module reads. [0, 1, 2] is not a choice: they are the levels
    #: `heads.pairwise.pair_term_levels` already uses, so what changes is the
    #: WAY of reading and not its reach.
    readout_levels: Tuple[int, ...] = (0, 1, 2)
    #: Inner width of the projection. The softmax temperature is scaled by
    #: 1/sqrt(dim), because without that it drifts with the encoder's scale --
    #: `PairComparison` lacks the scaling, and that is its defect rather than a
    #: pattern to copy.
    readout_dim: int = 64
    #: Queries per level. Cost: 12 336 parameters in total.
    readout_queries: int = 8
    #: PER-CHANNEL sigmoid gate on the occupancy channels, placed BEFORE the
    #: `occ_proj` bottleneck -- the point the stem and the billboard share, so a
    #: gate in only one of the two paths would give one model two different
    #: occupancies. False = no parameter is created. It is a READOUT, not a
    #: repair: the motif cascade measured that the bottleneck destroys nothing
    #: (the codes carry 98.7 % of motif identity), so the expected change in
    #: PCC is ZERO and what the gate shows is which of the 252 frozen MEME
    #: profiles the model suppresses. Without the Lagrangian controller it
    #: carries no sparsity PRESSURE and `occ_gate_ef` will stand at 284.
    occupancy_gate: bool = False
    #: Initial logit. +4.0 gives g = 0.982, i.e. today's model up to a constant
    #: that `occ_proj` absorbs. NOT zero: a gate of 0 together with a value of 0
    #: has zero derivative in both arguments, and the branch is then dead from
    #: the first step.
    occupancy_gate_init: float = 4.0
    #: The motif bank. Until recently there was NO key about the bank at all:
    #: no path, no thresholds, no exclusion list.
    motifs: MotifsConfig = field(default_factory=MotifsConfig)

    #: How many FSQ grid DIGITS stay live per level, coarsest refinement level
    #: first. `()` = the grids exactly as `fsq_bits` implies them. A separate
    #: key because `fsq_bits` changes SHAPES and so invalidates the checkpoint,
    #: whereas pinning the trailing digits to the middle "no correction" value
    #: drops a level's alphabet at unchanged shapes and parameter count. The
    #: headroom is measured: at rec_full 0.9993, L1 carries 3.64 of its
    #: declared 9.51 bits and L3/L4 carry 2.01 of 3.17, a ternary grid used
    #: BINARY. The ladder reads the key from the ACTIVE config and REFUSES when
    #: that config describes other grids, so a deepcopy needs `cyfry_czynne=`.
    fsq_cyfry_czynne: Tuple[int, ...] = ()

    def n_levels(self) -> int:
        return len(self.strides)

    def map_lengths(self, sequence_length: int) -> List[int]:
        """Map lengths COARSEST FIRST, for any sequence length and depth."""
        L = int(sequence_length)
        out = []
        for s in self.strides:
            L = int(math.ceil(L / float(s)))
            out.append(L)
        return list(reversed(out))

    def bit_budget(self, sequence_length: int) -> Dict[str, object]:
        """Per-level and total code budget against the MEASURED entropy.

        This is a falsifier at config level: changing the depth, the strides or
        a budget changes the reserve, and a run whose reserve fell below ~1x
        cannot be described as "reconstruction is achievable and the failure
        would be an optimisation failure".
        """
        lens = self.map_lengths(sequence_length)
        per: List[Dict[str, object]] = [{
            "level": 0, "length": lens[0], "alphabet": self.l0_codes,
            "bits_per_position": math.log2(self.l0_codes),
            "bits": lens[0] * math.log2(self.l0_codes),
            "kind": "hyperbolic codebook"}]
        # This used to count DECLARATIONS only, so `reserve` ignored
        # `fsq_cyfry_czynne` -- the one key that really changes it -- and the
        # falsifier was blind to the intervention it exists to measure. The
        # declarations stay alongside: hiding them would be the same mistake
        # the other way round.
        _czynne = list(getattr(self, "fsq_cyfry_czynne", ()) or ())
        total_decl = float(per[0]["bits"])
        for j, b in enumerate(self.fsq_bits):
            dims, cells, real = fsq_grid(b)
            total_decl += lens[j + 1] * real
            _akt = dims
            if j < len(_czynne) and _czynne[j]:
                _akt = max(1, min(int(_czynne[j]), dims))
            _real_akt = real * (float(_akt) / float(dims)) if dims else real
            _cells_akt = int(round(3.0 ** _akt))
            per.append({"level": j + 1, "length": lens[j + 1],
                        "alphabet": _cells_akt,
                        "alphabet_declared": cells,
                        "dims": _akt, "dims_declared": dims,
                        "declared_bits_per_position": float(b),
                        "bits_per_position": _real_akt,
                        "bits": lens[j + 1] * _real_akt,
                        "kind": "tangent FSQ"})
        total = sum(float(p["bits"]) for p in per)
        seq_bits = sequence_length * UNIGRAM_BITS_PER_BP
        # The FINE levels' bits are kept apart because the hierarchy condition
        # ("the fine levels ALONE must not suffice") rests on them, while the
        # sum including L0 answers a different question.
        fine_bits = total - float(per[0]["bits"])
        return {"levels": per, "total_bits": total,
                "total_bits_declared": total_decl,
                "fine_bits": fine_bits,
                "fine_sufficient": bool(fine_bits >= seq_bits),
                "sequence_bits": seq_bits,
                "reserve": total / seq_bits if seq_bits else float("nan"),
                "reserve_declared": (total_decl / seq_bits if seq_bits
                                     else float("nan")),
                "reconstruction_floor": RECONSTRUCTION_FLOOR}

    def validate(self) -> None:
        # Without this line `MotifsConfig.validate` exists and is never called.
        # A validator nobody calls is worse than none: a reader sees the rule
        # in the class and assumes it is enforced.
        _m = getattr(self, "motifs", None)
        if _m is not None:
            _m.validate("model.motifs")
        if not self.strides:
            raise ValueError("model.strides is empty: no levels")
        if isinstance(self.level_blocks, (list, tuple)) and \
                len(self.level_blocks) != len(self.strides):
            # `PromoterEncoder.__init__` raises on the same thing, but only at
            # build time and naming a MODULE. Here the error names the KEY,
            # before anything allocates memory. The list runs from the finest
            # level, so an off-by-one moves EVERY block to a neighbouring level
            # -- a factor of two in reach per level, with no shape error.
            raise ValueError(
                "model.level_blocks has %d entries for %d levels "
                "(model.strides). As a list it is one entry per level, "
                "FINEST FIRST; as a number -- the same count on every level."
                % (len(self.level_blocks), len(self.strides)))
        if int(self.context_kernel) > 1 and int(self.context_kernel) % 2 == 0:
            # DampedGlobalContext raises on an even kernel; failing at
            # config load names the key, failing at build names a module.
            raise ValueError(
                "model.context_kernel must be odd (a centred depthwise "
                "kernel has no even form), got %r" % (self.context_kernel,))
        if len(self.channels) != self.n_levels():
            raise ValueError(
                "model.channels has %d entries for %d levels; they are listed "
                "COARSEST FIRST and one per level"
                % (len(self.channels), self.n_levels()))
        if len(self.fsq_bits) != self.n_levels() - 1:
            raise ValueError(
                "model.fsq_bits has %d entries for %d levels: level 0 carries "
                "the hyperbolic codebook and every other level carries one FSQ "
                "budget" % (len(self.fsq_bits), self.n_levels()))
        if self.fsq_rate_weight < 0.0:
            raise ValueError("model.fsq_rate_weight must be >= 0, got %r"
                             % (self.fsq_rate_weight,))
        if self.drab_pred:
            if self.drab_pred_okno < 1 or self.drab_pred_okno % 2 == 0:
                raise ValueError(
                    "model.drab_pred_okno must be ODD and >= 1, got %r: an "
                    "even window shifts the output by half a "
                    "position" % (self.drab_pred_okno,))
            if self.drab_pred_ukryte < 1:
                raise ValueError("model.drab_pred_ukryte must be >= 1")
            if self.drab_pred_waga <= 0.0:
                raise ValueError(
                    "model.drab_pred = true with drab_pred_waga = %r: the "
                    "reconstruction gradient to the predictor cancels "
                    "through the straight-through estimator, so without "
                    "this term the branch is dead from the first step -- "
                    "while the log makes it look wired in"
                    % (self.drab_pred_waga,))
        if self.l0_codes < 2 or (self.l0_codes & (self.l0_codes - 1)):
            raise ValueError("model.l0_codes should be a power of two >= 2, "
                             "got %r" % (self.l0_codes,))
        if not self.vq_dead_restart:
            raise ValueError(
                "dead-code restart is not optional: without it codebook "
                "collapse is ABSORBING rather than probable (measured "
                "bits_used_frac 0.2970 in epoch 1). Removing it is a run about "
                "the restart, and then it belongs in the run's name.")
        if self.decoder_sees_codes_only is False and \
                not self.allow_continuous_decoder:
            raise ValueError(
                "the decoder may see the codes ONLY. With a continuous bypass "
                "the decoder gets a lossless copy of the encoder -- measured "
                "identity to 2.6e-07, reconstruction from codes alone 0.29 "
                "against a 0.2626 floor -- and diffusion, which generates by "
                "decoding, has no such channel at sampling time. This is a "
                "blocking condition, not a preference.")
        if any(t >= self.n_levels() for t in self.triangular_levels):
            raise ValueError("model.triangular_levels indexes a level that "
                             "does not exist")
        for b in self.fsq_bits:
            fsq_grid(b)
        if self.fsq_cyfry_czynne:
            if len(self.fsq_cyfry_czynne) != len(self.fsq_bits):
                raise ValueError(
                    "model.fsq_cyfry_czynne has %d entries and model.fsq_bits "
                    "%d. One number per refinement level; a list stretched "
                    "silently is how an equal-budget experiment stops being "
                    "one."
                    % (len(self.fsq_cyfry_czynne), len(self.fsq_bits)))
            for k, (a, b) in enumerate(zip(self.fsq_cyfry_czynne,
                                           self.fsq_bits)):
                d = fsq_grid(int(b))[0]
                if not 1 <= int(a) <= d:
                    raise ValueError(
                        "model.fsq_cyfry_czynne[%d] = %r; this level's grid "
                        "has %d digits (fsq_bits[%d] = %r), so the allowed "
                        "range is [1, %d]. The value 0 is a level with ONE "
                        "code -- a collapsed quantiser posing as a "
                        "configuration; a level is removed through "
                        "model.strides and model.channels."
                        % (k, a, d, k, b, d))


# Conditioning
@dataclass
class ConditioningConfig:
    """Every channel switchable on its own; species and categories extensible.

    Ranking of the channels, measured by paired ablation (80 fixed batches,
    5120 promoters, baseline SCC 0.74885), zeroed / shuffled / substituted:

        gene context   -0.0875  -0.1692  -0.1819
        latent         -0.0109  -0.0730    --
        genome         -0.0020  +0.0000  -0.0024
        experiment      -0.0029  +0.0000  -0.0024

    The two +0.0000 entries are the empty intervention described in
    `ChannelConfig`, which is why `ablate` cannot be "shuffle" here.
    """

    # `ablate` stays "none" in every default: it names the ONLY admissible
    # ablation protocol for a conditioning channel, it does not switch one on.
    # A default of "substitute" would make every run an ablation run, which
    # `report_inert_keys` duly reported while this file was being written.
    gene_context: ChannelConfig = field(default_factory=lambda: ChannelConfig(
        enabled=True, dim=768, source="dna_lm"))
    genome: ChannelConfig = field(default_factory=lambda: ChannelConfig(
        enabled=True, dim=32, source="embedding"))
    categories: ChannelConfig = field(default_factory=lambda: ChannelConfig(
        enabled=True, dim=len(DEFAULT_CATEGORIES), source="multi_hot"))
    # `dim` IS read: build_corpus hands it to Corpus.from_paths as
    # `text_expect_dim`, which ExperimentTextCache.load compares against the
    # artefact and RAISES on a mismatch, and assemble.py compares it against
    # the category prototype width. 384 was a latent shape bug -- the condition
    # encoder takes 768 since the prototypes became real SapBERT vectors -- and
    # it was inert only while nothing supplied a text vector.
    experiment_text: ChannelConfig = field(default_factory=lambda: ChannelConfig(
        enabled=False, dim=768, source="text_encoder"))

    # Adding a species = appending here. Empty means "collect from the data and
    # report", which is fine for training and NOT fine for a checkpoint that
    # has to be reloaded: the embedding row of a species is its index.
    species: Tuple[str, ...] = ()
    categories_list: Tuple[str, ...] = DEFAULT_CATEGORIES

    #: The GENE the promoter drives. `gene_context` above is named for the gene
    #: but its source is `dna_lm` over the FLANKS, so nothing in the model saw
    #: an orthogroup until this channel existed and "make this strong FOR pks1"
    #: had no key to bind to. Disabled by default on a measurement rather than
    #: out of caution: a cheap gene channel (length, GC, GC3, 61 codons) came
    #: out NET NEGATIVE at -0.1279 of out-of-sample R^2 on the model residual.
    #: This channel is a different signal -- the phyletic profile, not codon
    #: usage -- but that is a hypothesis against a measurement, so switch it on
    #: only once `GeneFeatures.audit_against()` and the decomposition head's
    #: `ablation_gain` beat its own shuffled control.
    gene: ChannelConfig = field(default_factory=lambda: ChannelConfig(
        enabled=False, dim=44, source="orthology", dropout=0.10))

    #: Dropout of the SPECIES TOKEN, drawn per pair group. -1.0 keeps the
    #: `ConditionConfig` default of 0.15, so existing configs are unchanged.
    #: The key is explicit because `ConditionConfig` is built in assemble.py
    #: OUTSIDE the HyppeConfig tree and has no YAML path, while changing the
    #: default in condition.py would silently move every config in the
    #: repository. Measured reason for raising it: substituting the species
    #: channel gives dPCC -0.0026 at epoch 6 and +0.0046 at epoch 30, so after
    #: the peak the channel HARMS on new orthogroups.
    species_dropout: float = -1.0

    #: Denominator floor of the tree pull-back, in patristic units. The prior is
    #: a RELATIVE error, so stiffness goes as 1/d^2 and the genomes it holds
    #: hardest are the closest ones: median stiffness 6.04 over 182 pairs, but
    #: the seven strains sharing pks1's 800 bp byte for byte carry a median of
    #: 4.283e+04 although their measured control spans 0.039..7.624 -- the
    #: channel is pinned hardest on exactly the genomes it exists to separate.
    #: 0.005 is where the marker matrix's own relative precision reaches 20 %
    #: (binomial SE 9e-4 / 0.20) and softens 16 of 182 pairs. The DEFAULT equals
    #: `species.PRIOR_MIN_DIST`, under which no pair is softened.
    genome_prior_denom_floor: float = 1e-3

    #: Coefficient of the tree prior, previously hard-coded in `species.py`
    #: (`from_tree(tree_prior=0.1)`) and not exposed. 0.1 reproduces it.
    genome_tree_prior: float = 0.1

    #: Epochs over which `genome_tree_prior` is annealed linearly to zero.
    #: 0 = no annealing, i.e. a constant term as before.
    #:
    #: The `genome` module's gradient has an inter-batch SNR of 33 212, i.e. it
    #: is identical in every batch: that is a prior, not data. With three
    #: tangent dimensions for twenty genomes the channel encodes a position on
    #: the tree rather than promoter grammar, and its measured contribution to
    #: prediction is +0.00003 PCC. The prior is meant to PLANT the points and
    #: then hand them back to the data.
    genome_tree_prior_epochs: int = 0
    #: Whether to lift the tree prior off strains the tree cannot tell apart
    #: (classes at patristic distance 0), keeping one representative per class.
    #: False = as before. See the assembly note in `assemble.py`.
    genome_prior_free_collapsed: bool = False

    # Context cone
    #: Context as a GEODESIC CONE over the species point: the condition tokens
    #: become its geodesic refinements, a three-storey hierarchy (channel root
    #: -> species -> refinement) imposed BY CONSTRUCTION. Gromov's delta says
    #: the data are not tree-like (text 0.1455 against 0.0631 for Gaussian
    #: noise in R^768), but delta measures whether a hierarchy is ALREADY
    #: there, not whether a model can learn one. Off by default, in which case
    #: the module is not built. Observables sit on the `cond_ktx` line under
    #: `ktx_st_`; a zero `<channel>_norma_odczytu` means the module is dead
    #: despite being on.
    cone: bool = False
    #: Dimension of the cone's manifold. Must be at least the species
    #: manifold's (`GenomeEmbedding.dim`, 3 here): the anchor is an EMBEDDING of
    #: H^3 into H^d by zero-padding, so truncating would delete part of the tree
    #: instead of lowering the dimension. The surplus is where a token is meant
    #: to part company with its species.
    cone_dim: int = 8
    #: A CONSTANT step aperture, not a weight. Three times in the sibling track
    #: a learned radius scalar moved in the direction that cancels the
    #: mechanism. A constant also means that a run with and without the aperture
    #: measures geometry rather than size.
    cone_aperture: float = 0.25
    #: Which condition tokens are refinements. `species` is the ANCHOR, so it is
    #: not a channel. The default leaves the pairwise branch one route --
    #: `gene`, because `species` and `text` are withheld from it by
    #: `heads.head_withhold`; closing that one too means setting [text].
    cone_channels: Tuple[str, ...] = ("text", "gene")
    #: Whether a channel's root is a LEARNED point on the hyperboloid. It is the
    #: only new manifold tensor in the whole model and the only reason the
    #: `optimiser ... manifold group` line would read above today's "1 tensor /
    #: 80 params". It starts EXACTLY at the origin, so at initialisation the
    #: composition equals `expmap0(tangent)` bit for bit.
    cone_roots: bool = True
    #: Zero-initialised readout, so the forward pass starts as the identity.
    #: This HIDES a failure until the third optimiser step -- the recorded
    #: `occ_center` precedent read 1.0000 on a fresh model and 4.73e-03 after
    #: three steps -- so every measurement of this module is taken AFTER some
    #: steps.
    cone_zero_init: bool = True
    #: Forward passes between diagnostics. Every `float()` on a CUDA tensor is a
    #: synchronisation, and diagnostics on the hot path ate 151 of 216 minutes
    #: per epoch in the sibling track. The first pass always counts, so the
    #: `cond_ktx` line is never empty.
    cone_diag_every: int = 50

    #: Whether `GenomeEmbedding.reproject_()` is called: after every
    #: `load_state_dict` of this module and on entry to `forward`, and only
    #: when `max|<x,x>_L + 1|` exceeds `REPROJECT_TOL` (1e-12). Not caution:
    #: per-epoch checkpoints lie on the hyperboloid to ~1e-15, while the SWA
    #: mean of those same epochs -- the file the evaluations run on -- was
    #: 1.621e-03 off, because averaging sums AMBIENT coordinates and the
    #: arithmetic mean of hyperboloid points is not on the hyperboloid.
    #: Repaired rather than excluded, because averaging the SPATIAL part is
    #: what carries the information. Effect on prediction: ZERO, measured;
    #: the key protects `tree_prior_loss`, `radius`, `report` and `prior_fit`,
    #: which read `minkowski_dot` directly. False = bit for bit as before.
    genome_reproject: bool = False

    #: Path to `og_table_combined.tsv`. Empty = the repo default location.
    #: An EXPRESSION matrix here is refused by `build_gene_features`: the
    #: gene head's target is derived from exactly those files.
    orthology_path: str = ""
    #: Width of the learned per-orthogroup table. 0 = content only, and it is
    #: the only setting under which a number may be called zero-shot: the
    #: split is orthogroup disjoint, so every val/test/spy gene -- pks1
    #: included -- would carry a row that received no gradient.
    gene_identity_dim: int = 0
    gene_identity_buckets: int = 4096
    gene_use_text: bool = True
    #: Per-source dropout. Without it the numeric and text sources are always
    #: present together, the trunk learns their sum, and the first inference
    #: on a gene that has a description but no orthology row lands off the
    #: training manifold.
    gene_drop_numeric: float = 0.15
    gene_drop_text: float = 0.15

    def vector_channels(self) -> Dict[str, ChannelConfig]:
        return {"gene_context": self.gene_context, "genome": self.genome,
                "categories": self.categories,
                "experiment_text": self.experiment_text,
                "gene": self.gene}

    def context_dim(self) -> int:
        return sum(c.dim for c in self.vector_channels().values() if c.enabled)

    def validate(self) -> None:
        for name, ch in self.vector_channels().items():
            ch.validate("conditioning." + name)
        if self.categories.enabled and \
                self.categories.dim != len(self.categories_list):
            raise ValueError(
                "conditioning.categories.dim = %d but %d categories are "
                "declared. Adding a category means appending to "
                "`categories_list` AND widening the vector; changing only one "
                "of the two is the class of failure this module exists to "
                "catch. Measured atomic categories: %d."
                % (self.categories.dim, len(self.categories_list),
                   len(DEFAULT_CATEGORIES)))
        if len(set(self.categories_list)) != len(self.categories_list):
            raise ValueError("conditioning.categories_list has duplicates; the "
                             "order is a checkpoint contract and must be "
                             "append-only")
        if len(set(self.species)) != len(self.species):
            raise ValueError("conditioning.species has duplicates; the index "
                             "of a species is its embedding row")
        if self.cone:
            if not self.genome.enabled:
                raise ValueError(
                    "conditioning.cone is on while conditioning.genome is "
                    "off. The cone's ANCHOR is the species point; without "
                    "the Lorentz registry every anchor is the origin, the "
                    "hierarchy has two floors instead of three and the "
                    "module degenerates into a residual MLP on the "
                    "conditioning tokens -- which would read in the log "
                    "exactly like a working cone.")
            if self.cone_dim < 1:
                raise ValueError("conditioning.cone_dim must be >= 1, is %d"
                                 % (self.cone_dim,))
            if not 0.0 < self.cone_aperture <= 2.0:
                raise ValueError(
                    "conditioning.cone_aperture must be in (0, 2], is %r"
                    % (self.cone_aperture,))
            if not self.cone_channels:
                raise ValueError(
                    "conditioning.cone_channels is empty: the module would "
                    "build, do nothing, and read in the log like a working "
                    "one")
            if len(set(self.cone_channels)) != len(self.cone_channels):
                raise ValueError("conditioning.cone_channels has duplicates: "
                                 "%r" % (tuple(self.cone_channels),))
            if "species" in self.cone_channels:
                raise ValueError(
                    "conditioning.cone_channels names 'species', which is "
                    "the ANCHOR of the cone, not one of its refinements. "
                    "Refining the species token by a step anchored at the "
                    "species point is a self-reference, not a hierarchy.")
            if self.cone_diag_every < 1:
                raise ValueError(
                    "conditioning.cone_diag_every must be >= 1, is %d"
                    % (self.cone_diag_every,))


# Heads
@dataclass
class PoolingConfig:
    """How the level maps become one vector for the heads."""

    stats: Tuple[str, ...] = ("mean", "sd", "anchor")
    anchor_frac: float = 0.125

    def validate(self) -> None:
        known = ("mean", "sd", "max", "anchor")
        bad = [s for s in self.stats if s not in known]
        if bad:
            raise ValueError("heads.pooling.stats %r unknown; known %r"
                             % (bad, known))
        if not self.stats:
            raise ValueError("heads.pooling.stats is empty: the pooled vector "
                             "would have width zero")
        if not 0.0 < self.anchor_frac <= 1.0:
            raise ValueError("heads.pooling.anchor_frac must be in (0, 1]")


@dataclass
class AbsoluteHeadConfig:
    enabled: bool = True
    weight: float = 1.0
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    context_mode: str = "concat"
    context_center: bool = True
    beta: float = 0.5              # beta-NLL; 0.0 is the NLL that diverged
    #: SECOND beta-NLL exponent, mu path only. None = use `beta`, which takes
    #: the code branch that ran before this key existed, so the loss is
    #: bit-identical. 1.0 removes sigma from the mu gradient entirely.
    #: READER: model/assemble.py passes it into AbsoluteStrengthHead(...);
    #: heads/absolute.py AbsoluteStrengthHead.loss branches on it.
    beta_mu: Optional[float] = None
    lv_min: float = -8.0
    lv_max: float = 4.0
    clamp_mode: str = "hard"
    predict_sigma: bool = True
    sigma_weight: float = 0.2
    sigma_floor: float = 1e-3
    tie_sigma_to_var: bool = False
    #: Weight of a ONE-SIDED penalty on EXCESS amplitude within a group
    #: (`KalibracjaOsi.kara_amplitudy`). 0.0 = off, `loss` bit for bit as
    #: before. On the "genome within orthogroup" axis the model has a NEGATIVE
    #: R^2 (-0.0045) at a POSITIVE PCC (+0.3325), because the fitted scale is
    #: 0.495 < 0.5. The penalty `relu(<ap,ap> - <ap,ay>)` is exactly zero where
    #: the axis is calibrated or UNDER-scaled, and the one-sidedness is
    #: necessary: the main genome effect has slope 1.241 while the interaction
    #: has 0.236. It is the only one of the three genome-axis components that
    #: touches TRAINING, and it REQUIRES `grupa` in the call to
    #: `AbsoluteStrengthHead.loss` -- at a positive weight with `grupa=None` the
    #: head RAISES rather than quietly dropping the term.
    kara_amplitudy: float = 0.0


@dataclass
class CategoryHeadConfig:
    """Categoriser reading the REGRESSION module, not the description vector.

    It replaces `ConditionEncoder._category_logits(text_vector)`, which was
    dead for two independent reasons at once: it predicted the category from a
    description that encodes that category (an UNTRAINED model scored 0.9295
    AUC against 0.8032 trained, so training made it worse), and
    `category_loss` was called from no line of `training/` or `scripts/`.

    `enabled: False` = today's behaviour bit for bit.
    """

    enabled: bool = False
    weight: float = 0.1
    hidden: int = 128
    dropout: float = 0.1
    #: Whether to add the regression head's CENTRED columns to the input.
    use_columns: bool = True
    #: Train ONLY on rows whose description token was dropped. At False the task
    #: is a tautology -- the category is on the input -- and the "untrained is
    #: worse than trained" falsifier cannot fire, so True is the only value
    #: under which this head means anything.
    only_dropped_text: bool = True


@dataclass
class PairwiseHeadConfig:
    enabled: bool = True
    weight: float = 1.0
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    context_mode: str = "concat"
    context_center: bool = True

    #: THERE IS NO MARGIN. The head trains on the Bradley-Terry likelihood
    #: `softplus(-sign * delta / T)`, and the hinge has been DELETED rather than
    #: defaulted away, so a config cannot set one and believe it did something.
    #: A hinge has EXACTLY ZERO gradient once a pair is ordered by more than its
    #: margin, so it is indifferent to how far apart the scores end up -- and
    #: PCC reads spacing, not order. Measured on a correctly ordered pair at gap
    #: 0.50: the hinge gradient is -0.000e+00 at margin 0.1 AND at margin 0.0,
    #: while the likelihood keeps -3.775e-01.
    temperature: float = 1.0
    nonprom_weight: float = 0.25
    nonpromoter_pairs: bool = True

    #: COMPARISON BEFORE POOLING: weight of the `heads/pair_compare.py` term,
    #: which compares two promoters AT CODE POSITIONS before pooling averages
    #: them. 0.0 = as without the module, which is still built so that the
    #: state_dict has one shape. The head otherwise receives `[B, in_dim]`
    #: AFTER pooling, so WHICH positions differ reaches it through that one
    #: narrow channel -- and pair selection cannot fix it, because two
    #: promoters from one genome differ at 664 of 775 code positions and a
    #: nearest-neighbour search within a cell only reaches 633, so the class
    #: "pair close in code within one genome" does not exist in supply.
    pair_term_weight: float = 0.0
    pair_term_hidden: int = 64
    pair_term_dropout: float = 0.0
    #: Which code levels to compare; empty = all of them. Level 0 has 25
    #: positions over 800 bp and level 4 has 400, so this chooses the
    #: RESOLUTION of the comparison and not only its cost.
    pair_term_levels: Tuple[int, ...] = ()

    #: Share of each batch reserved for ORTHOGROUP BLOCKS -- rows of one gene
    #: taken from several genomes -- so the head can be given a pair that varies
    #: the GENOME and holds the gene and the category fixed. Within one genome
    #: the species channel is constant inside every pair and so receives no
    #: gradient from this head at all; the measured consequence is a Spearman
    #: of 0.107 between strains on byte-identical sequences. 0.0 = as before.
    #: It needs a sampler rather than just a pairing rule, because a RANDOM
    #: batch of 96 holds 0.60 such pairs against 48 within-genome ones; at 0.25
    #: the count rises to 5.1 per batch.
    ortholog_fraction: float = 0.0
    #: Rows taken from one orthogroup per block. Below 2 no pair can be
    #: formed from a block at all.
    ortholog_block: int = 4

    def validate(self, path: str = "heads.pairwise") -> None:
        if not 0.0 <= self.ortholog_fraction < 0.9:
            raise ValueError(
                "%s.ortholog_fraction=%r must lie in [0, 0.9). Reserving "
                "almost the whole batch for orthogroup blocks would leave no "
                "rows for the within-genome pairs, which are still the "
                "majority of the ranking signal." % (path,
                                                     self.ortholog_fraction))
        if int(self.ortholog_block) < 2:
            raise ValueError(
                "%s.ortholog_block=%r: a block of fewer than two rows cannot "
                "produce a pair, so the reserved slice would be spent on "
                "nothing." % (path, self.ortholog_block))


@dataclass
class DiscriminatorHeadConfig:
    enabled: bool = True
    weight: float = 0.5
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.1
    context_mode: str = "none"
    context_center: bool = True
    pos_weight: Optional[float] = None
    label_smoothing: float = 0.0
    #: MUST list every kind `BatchMaker` actually produces. It emits
    #: `dinucleotide_shuffle` and -- at the default `chimera_fraction = 0.5` --
    #: `chimera`, and `PromoterDiscriminator.observe` raises on a kind it was
    #: not told about. The previous default named `flank` and `coding`, which no
    #: builder emits, and omitted `chimera`, which every batch contains, so a
    #: default config could not complete one step.
    negative_kinds: Tuple[str, ...] = ("dinucleotide_shuffle",
                                       "chimera")
    #: Weights for drawing a slot's negative class. An empty dict = no mix and
    #: the behaviour as it stood. The keys MUST be contained in
    #: `negative_kinds`, because the head refuses an undeclared name and the run
    #: would otherwise fall over at the first draw.
    negative_mix: Dict[str, float] = field(default_factory=dict)


@dataclass
class FusionHeadConfig:
    """Fuse the context INTO the latent, so every prediction comes from it.

    Before this the heads read `concat(pool(z), context)` and the context was a
    BYPASS: a route from the condition to the prediction that never passed
    through the promoter representation. A model wired that way can answer "how
    strong is this condition on average" without looking at the sequence, and
    the design loop -- which edits codes and re-scores -- would be steering a
    term the edit cannot reach. With `enabled`,
    `model.fusion.LatentConditioner` produces ONE conditioned latent and the
    three prediction heads receive it with `context=None`.

    The conditioning must be able to move the prediction on its own, which is
    measured rather than assumed: 84 700 of 167 724 labelled rows (50.50 %)
    share their 800 bp EXACTLY with another row, those rows differ in mu by
    0.3588 in the median and 13.84 at the extreme, and 85.06 % of the groups
    exceed the 0.049 noise floor. So the modulation is FiLM, multiplicative and
    additive, rather than a small additive residual that could not separate two
    identical sequences by 13.84.

    `gate_floor` keeps every channel's contribution strictly positive, so a
    channel cannot be starved of gradient; `floor_weight` additionally pushes a
    channel back above `active_floor` and is OFF by default, because forcing a
    channel to contribute can only make it loud, never useful.
    """

    enabled: bool = True
    #: Width of each per-channel encoder before the bottleneck.
    d_channel: int = 64
    #: THE BOTTLENECK. Everything the condition can say passes through this
    #: many numbers, against a 807-wide concatenated context. The cap is what
    #: "controlled" means: a per-condition LOOKUP TABLE measured 0.43448
    #: out-of-sample under a random split and 0.08615 under the STUDY split,
    #: so memorising the condition is available and does not transfer. A wide
    #: path can represent that table; a narrow one cannot.
    d_bottleneck: int = 32
    hidden: int = 128
    dropout: float = 0.0
    #: gamma lives in [1-r, 1+r]. Strictly below 1 so gamma never reaches 0,
    #: which would orphan the channel behind it -- a closed multiplicative
    #: gate passes no gradient to the features that would reopen it.
    film_range: float = 0.9
    init_scale: float = 0.1
    gate_floor: float = 0.05
    #: 0.0 = 1/(4T), i.e. a quarter of an equal share.
    active_floor: float = 0.0
    floor_weight: float = 0.0
    #: Centring of the modulation ACROSS THE BATCH. "none" (default) = the model
    #: bit for bit as before this key existed. "batch" has BatchNorm semantics
    #: (batch mean in training, buffers in evaluation); "ema" uses a running
    #: mean in both, so a row's output does NOT depend on who else was in the
    #: batch -- which matters, because the pairwise head's pairs are formed
    #: inside the batch. No learned parameters either way; the implementation
    #: and its measurement are in `model.condition.CentredLatentConditioner`.
    film_center: str = "none"
    #: Express BETA IN LATENT UNITS, with no new parameters. False = as before.
    #: `h = gamma*lat + beta` passes through a per-sample RMS damper, which is
    #: SCALE-FREE, so what decides is the ratio `beta/rms(lat)`: a unit of
    #: weight on `gamma` buys `film_range*rms(lat)` of output and the same unit
    #: on `beta` buys 1, so weight decay prices the additive term rms(lat)
    #: times too high rather than making it weak. The flag divides the FiLM
    #: input by its own per-sample RMS; the denominator is deliberately not
    #: detached, since that would restore the encoder's freedom to drift in
    #: scale. The falsifier of WIRING, not of effect, is `fuse_beta_skala`,
    #: which must read 1.
    beta_w_skali_latentu: bool = False
    #: REVERSED condition attention: queries from EVERY POSITION of the map,
    #: keys and values from the metadata tokens. 0 = the module is not built.
    #: The other direction (`xattn_glowy`) had the condition query the map and
    #: receive ONE summary vector per sample, i.e. act as a global multiplier,
    #: and that was measured to be too little: `ktx_xa_poz_cv` rose to 0.86
    #: while `ktx_xa_udzial` FELL 0.62 -> 0.51 at a rising output norm, so the
    #: branch was learning a contribution COMMON to the positions, which is
    #: what pooling carries anyway. Locality is a requirement of the design
    #: task: what matters is which BASE in which PLACE counts in a given
    #: experiment, not whether the whole promoter is strong.
    xattn_lokalny: int = 0
    #: THE CONDITION MEETS THE SEQUENCE BEFORE POOLING. Number of cross-
    #: attention heads whose QUERIES come from the condition tokens and whose
    #: KEYS and VALUES come from the COARSEST ladder map. The result is ADDED to
    #: the pooled latent just before FiLM, which is otherwise unchanged. 0 = the
    #: module is not built. A cascade of probes cleaned the quantiser (+0.0002),
    #: the pooling (0.0000) and the heads (-0.0009) without finding anything;
    #: the one thing none of them measures is HOW the condition meets the
    #: sequence, which today is FiLM on a vector in which position is already
    #: gone, so the condition can shift the level but cannot choose WHICH window
    #: of the promoter counts. See `model.condition.CoarseConditionAttention`.
    xattn_glowy: int = 0

    def validate(self) -> None:
        if self.film_center not in ("none", "batch", "ema"):
            raise ValueError(
                "heads.fusion.film_center = %r; dozwolone none|batch|ema"
                % (self.film_center,))
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("heads.fusion.gate_floor must be in [0, 1)")
        if not 0.0 < self.film_range < 1.0:
            raise ValueError(
                "heads.fusion.film_range must be in (0, 1): at 1.0 gamma can "
                "reach exactly 0 and the channel behind it is orphaned.")
        if self.d_bottleneck < 2 or self.d_channel < 4:
            raise ValueError("heads.fusion widths are implausibly small")
        if self.d_bottleneck > self.d_channel * 8:
            raise ValueError(
                "heads.fusion.d_bottleneck = %d is not a bottleneck relative "
                "to d_channel = %d; the narrowness IS the control"
                % (self.d_bottleneck, self.d_channel))
        if self.floor_weight < 0.0:
            raise ValueError("heads.fusion.floor_weight must be >= 0")
        if self.active_floor and not 0.0 < self.active_floor < 1.0:
            raise ValueError("heads.fusion.active_floor is a SHARE in (0, 1)")
        if self.xattn_glowy < 0:
            raise ValueError(
                "heads.fusion.xattn_glowy = %r; this module is turned off by "
                "NOT BUILDING it (value 0), and a negative count has no "
                "meaning that could be accepted silently"
                % (self.xattn_glowy,))
        if self.xattn_glowy and not self.enabled:
            raise ValueError(
                "heads.fusion.xattn_glowy = %d with heads.fusion."
                "enabled = false. The attention output is added to the "
                "latent ENTERING FiLM, so without the fusion block it has "
                "nowhere to land; accepting this silently would give "
                "parameters without a mechanism." % (self.xattn_glowy,))


@dataclass
class DecompositionHeadConfig:
    """One head per conditioning channel, each with a target only it reaches.

    `mu` is the mean over conditions, so it is constant in the condition by
    construction and its dependence on the genome is already carried by the
    cell the row IS. Every head in the repo regressed that one number, which
    is why the species and experiment channels measured -0.0020 and -0.0029
    of SCC under zeroing: not weak channels, channels with no gradient.

    Weights are the MEASURED variance shares (gene 0.80, genome 0.013,
    interaction 0.078, condition 0.137 on this corpus), not a sweep: equal
    weights would give a 1.3 % term the pull of an 80 % one.
    """

    enabled: bool = True
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.1
    context_mode: str = "concat"
    context_center: bool = True
    targets: Tuple[str, ...] = ("gene", "strain", "cond")
    weight_gene: float = 0.30
    weight_strain: float = 0.15
    weight_cond: float = 0.15
    huber_delta: float = 0.0
    #: "shuffle" is absent for the reason `ChannelConfig.ablate` records:
    #: batches are genome-homogeneous, so a within-batch permutation of the
    #: species token is an empty intervention.
    ablate: str = "roll"

    def weights(self) -> Dict[str, float]:
        return {"gene": float(self.weight_gene),
                "strain": float(self.weight_strain),
                "cond": float(self.weight_cond)}

    def validate(self) -> None:
        known = ("gene", "strain", "cond")
        bad = [t for t in self.targets if t not in known]
        if bad:
            raise ValueError("heads.decomposition.targets %r unknown; known %r"
                             % (bad, known))
        if self.enabled and not self.targets:
            raise ValueError("heads.decomposition is enabled with no targets")
        if self.ablate not in ("roll", "mean", "zero"):
            raise ValueError("heads.decomposition.ablate must be "
                             "roll|mean|zero, got %r" % (self.ablate,))
        if self.huber_delta < 0.0:
            raise ValueError("heads.decomposition.huber_delta must be >= 0")


@dataclass
class HeadsConfig:
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    absolute: AbsoluteHeadConfig = field(default_factory=AbsoluteHeadConfig)
    pairwise: PairwiseHeadConfig = field(default_factory=PairwiseHeadConfig)
    category: CategoryHeadConfig = field(
        default_factory=CategoryHeadConfig)

    #: Context channels the pairwise branch and the discriminator do NOT see.
    #: Empty = every head sees the same thing. A pair is drawn WITHIN one
    #: (genome, category) cell, so species and description are constant inside
    #: it and cannot change the sign of the difference, while for the optimiser
    #: they remain a route to fitting the cell -- measured in use, substituting
    #: the species channel gives dPCC -0.0026 at the peak epoch and +0.0046 at
    #: epoch 30. A channel can only be withheld if `ConditionEncoder` has an
    #: UNKNOWN token for it; withholding `gene` is REFUSED, because zeros
    #: through a biased `gene_proj` are a learned CONSTANT, not an absence.
    head_withhold: Tuple[str, ...] = ()
    discriminator: DiscriminatorHeadConfig = field(
        default_factory=DiscriminatorHeadConfig)
    fusion: FusionHeadConfig = field(default_factory=FusionHeadConfig)
    decomposition: DecompositionHeadConfig = field(
        default_factory=DecompositionHeadConfig)

    def enabled_heads(self) -> List[str]:
        out = []
        for name in ("absolute", "pairwise", "discriminator",
                     "decomposition"):
            if getattr(self, name).enabled:
                out.append(name)
        return out

    def validate(self) -> None:
        self.pooling.validate()
        self.fusion.validate()
        self.decomposition.validate()
        if self.fusion.enabled and self.decomposition.enabled and \
                self.decomposition.context_mode == "none":
            raise ValueError(
                "heads.decomposition.context_mode='none' removes the one "
                "input that distinguishes these heads from the absolute head: "
                "each exists to give ONE conditioning channel a target only "
                "that channel can reach.")
        if not self.enabled_heads():
            raise ValueError("no head is enabled: there is nothing to train")
        if self.absolute.beta < 0.0 or self.absolute.beta > 1.0:
            raise ValueError("heads.absolute.beta lives in [0, 1]")
        if self.absolute.beta_mu is not None and \
                not 0.0 <= self.absolute.beta_mu <= 1.0:
            raise ValueError(
                "heads.absolute.beta_mu lives in [0, 1] like beta. It is an "
                "exponent on the DETACHED variance, not a loss weight, so a "
                "value outside that range does not mean 'more of the same'.")
        if self.absolute.enabled and self.absolute.beta == 0.0:
            raise ValueError(
                "beta = 0 is the plain Gaussian NLL, which destroyed training "
                "in three epochs here (loss 1.3e6 -> 9.5e7 -> 6.5e8, PCC "
                "-0.007). If the point of the run is to reproduce that, say so "
                "by raising beta to a small positive number and naming the run "
                "after it.")
        _mix = dict(getattr(self.discriminator, "negative_mix", {}) or {})
        if _mix:
            _zle = sorted(set(_mix) - set(self.discriminator.negative_kinds))
            if _zle:
                raise ValueError(
                    "heads.discriminator.negative_mix names classes %r that "
                    "are absent from negative_kinds %r. The head refuses an "
                    "undeclared name, so the run would only fall over the "
                    "first time that class was drawn."
                    % (_zle, tuple(self.discriminator.negative_kinds)))
            if sum(float(v) for v in _mix.values()) <= 0:
                raise ValueError(
                    "heads.discriminator.negative_mix sums to zero")
        if self.discriminator.enabled and \
                "dinucleotide_shuffle" not in self.discriminator.negative_kinds:
            raise ValueError(
                "the discriminator's falsifier IS the dinucleotide-shuffled "
                "negative (same composition, destroyed order). Without it the "
                "head can score 0.99 AUC against random negatives and have "
                "learned composition only -- a trinucleotide feature alone "
                "reached AUC 0.8422 on a comparable axis.")
        for nm in ("absolute", "pairwise", "discriminator"):
            h = getattr(self, nm)
            if h.context_mode not in ("concat", "film", "none"):
                raise ValueError("heads.%s.context_mode must be "
                                 "concat|film|none" % (nm,))


# Training
@dataclass
class NashConfig:
    """Who bargains, inside which box, and which thresholds are checked."""

    enabled: bool = True
    players: Tuple[str, ...] = ("absolute", "pairwise", "discriminator")
    n_iter: int = 20
    min_weight: float = 0.125
    max_weight: float = 8.0
    alpha_ema: float = 0.9
    recompute_every: int = 1
    bypass: bool = False
    clip_threshold: float = 0.2
    nzb_threshold: float = 0.05
    allow_saturated_players: bool = False
    shared_scope: str = "encoder"

    def validate(self) -> None:
        if not self.players:
            raise ValueError("training.nash.players is empty")
        if not (self.min_weight <= 1.0 <= self.max_weight):
            raise ValueError(
                "the box [%.3f, %.3f] excludes 1.0, and the alphas sum to the "
                "number of players over that many coordinates, so their mean "
                "is exactly 1: no feasible alpha exists"
                % (self.min_weight, self.max_weight))
        if self.recompute_every < 1:
            raise ValueError("training.nash.recompute_every must be >= 1")
        # The fragment list is SHARED with `hyppe.training.nash`. It used to be
        # the literal `"recon"` here, i.e. the English-only version of a guard
        # a player named `rekonstrukcja` has already walked past once.
        forbidden = [p for p in self.players
                     if any(f in p.lower() for f in FORBIDDEN_PLAYER_FRAGMENTS)]
        if forbidden and not self.allow_saturated_players:
            raise ValueError(
                "%r cannot be a player: a saturated task takes the LARGEST "
                "share of the bargain because its gradient is the smallest "
                "(measured 2.31 of 3.0 at recon 0.9998, prediction pushed to "
                "0.39 / 0.31). It belongs in `training.subordinate_weights` "
                "with a declared weight." % (forbidden,))


@dataclass
class NormPriorConfig:
    """Explicit importance priors, instead of shares derived from the norms.

    The rule is alpha_i = K * (w_i / ||g_i||) / sum_j (w_j / ||g_j||), so the
    update receives `w_i` parts of task i's UNIT gradient. `w` is the thing a
    bargain does not give you: Nash computes the shares from the gradient norms
    alone and therefore rewards whichever task is currently easiest.

    Measured on the nash run of this very config (real corpus, TRAIN mode,
    epochs 1-7): alphas 0.43 / 1.82 / 0.74 for absolute / discriminator /
    pairwise at gradient norms 2.79 / 1.20 / 0.78 over `shared_scope: encoder`,
    with the degenerate closed form alpha_i = K / (||g_i|| sum_j 1/||g_j||)
    predicting 0.435 for absolute against 0.43 observed, and a Gram offdiagonal
    of 0.177 and falling -- below the 0.3 at which a gradient method has
    anything to resolve. The bargain was a normalisation costing four backward
    passes, and the priors it implied were nobody's decision.

    `priors` MUST NAME EVERY PLAYER. A prior that falls back to 1.0 because the
    key was misspelt is this project's inert-key failure in its purest form: the
    run would report `normprior` and execute uniform shares. A prior of 0.0 is
    refused as well, because the task would remain a player -- the alphas still
    sum to the player count -- while contributing nothing. Dropping a task is
    `training.task_weights[name] = 0.0`, the one operation all three methods
    agree on.
    """

    enabled: bool = False
    players: Tuple[str, ...] = ("absolute", "pairwise", "discriminator")
    #: Resolved by `TrainLoop._shared_parameters`, the SAME function that
    #: resolves `nash.shared_scope` -- a second mapping from scope name to
    #: module list is a second thing to forget to update.
    shared_scope: str = "encoder"
    #: Read by `NormPriorWeighting.prior`, via `TrainLoop._normprior_kwargs`.
    priors: Dict[str, float] = field(default_factory=lambda: {
        "absolute": 1.0, "pairwise": 1.0, "discriminator": 1.0})
    #: Steps between recomputations of the gradient norms, read by
    #: `NormPriorWeighting.apply`. 1 = recompute every step = no behaviour
    #: change, so raising it is an explicit decision. The case for raising it:
    #: across seven epochs the measured alpha of `absolute` moved 0.42, 0.40,
    #: 0.41, 0.42, 0.43, 0.43, 0.43 -- an excursion of +-0.015 -- while each
    #: recomputation costs K extra backward passes.
    refresh_every: int = 1
    #: Floor on ||g_i|| in the division AND the zero-gradient threshold, the
    #: same value `NashMTL.zero_threshold` uses. Below it the method raises
    #: rather than dividing: a dead task does not take the largest share here
    #: (its contribution is K w_i / S, free of its own norm), it inflates S
    #: and shrinks every other task's step while sum(alpha) still reads k.
    eps: float = 1e-12

    def validate(self) -> None:
        if not self.players:
            raise ValueError("training.normprior.players is empty")
        missing = [p for p in self.players if p not in self.priors]
        if missing:
            raise ValueError(
                "training.normprior.priors does not name %r. The prior is the "
                "whole point of this method; a silent fallback to 1.0 would "
                "run uniform shares under a name that claims otherwise, which "
                "is the inert-key failure this project keeps repeating."
                % (missing,))
        extra = [k for k in sorted(self.priors) if k not in self.players]
        if extra:
            raise ValueError(
                "training.normprior.priors names %r, which are not players "
                "%r: a prior with no task is a key with no receiver."
                % (extra, list(self.players)))
        bad = [k for k in sorted(self.priors) if not float(self.priors[k]) > 0]
        if bad:
            raise ValueError(
                "training.normprior.priors[%r] <= 0. That leaves the task a "
                "PLAYER -- the alphas still sum to the player count -- while "
                "it contributes nothing, i.e. two meanings for 'dropped'. Use "
                "training.task_weights[name] = 0.0, which is what all three "
                "methods treat as dropping a task and what gate F3 perturbs."
                % (bad,))
        if self.refresh_every < 1:
            raise ValueError("training.normprior.refresh_every must be >= 1")
        if not self.eps > 0.0:
            raise ValueError(
                "training.normprior.eps must be > 0: it is the division floor "
                "and the zero-gradient threshold at once, and at 0 a dead "
                "task divides by zero instead of raising")
        # Same fragment list and the same measured reason as NashConfig, but a
        # different damage: reconstruction at 0.9998 has a tiny gradient, so it
        # inflates sum_j w_j/||g_j|| and shrinks every prediction task's step.
        forbidden = [p for p in self.players
                     if any(f in p.lower() for f in FORBIDDEN_PLAYER_FRAGMENTS)]
        if forbidden:
            raise ValueError(
                "%r cannot be a player: a saturated task has the smallest "
                "gradient, and under 1/||g|| that inflates the normalising "
                "sum and shrinks EVERY task's contribution (under the bargain "
                "the same fact gave it alpha 2.31 of 3.0 at recon 0.9998). It "
                "belongs in `training.subordinate_weights` with a declared "
                "weight, or under the reconstruction constraint."
                % (forbidden,))


@dataclass
class ConstraintConfig:
    """Reconstruction as an INEQUALITY, not as a player and not as a constant.

    Both alternatives are measured failures in the sibling family on the same
    objective: as a Nash player it took alpha 2.31 of a 3.0 bargain while
    sitting at 0.9998 (a bargain equalises utilities, so the weakest gradient
    takes the largest share), and as a fixed weight at the effective
    coefficient 0.15 x 0.40 = 0.060 it fell to 0.5549 in one epoch. The
    coefficient that holds reconstruction up is not a constant of the model, so
    it is not configured as one: `lambda` is driven by the violation of
    `observable >= target` and everything else is spent on prediction.

    `target` is a CHOICE rather than a measurement of this model, and is stated
    as one: its provenance is the sibling track's codes-only reconstruction
    (0.9945-0.9962 at a constant weight, a DIFFERENT model on DIFFERENT data),
    so it is set below that. Its falsifier is `sat_hi`: a controller pinned at
    `lam_max` says the target was unreachable at any admissible price, and the
    answer is then a lower target, not a larger box.

    `lam_init` deliberately equals the family's declared `recon_weight`, so a
    run starts exactly where the fixed-weight family starts and every later
    value is something the controller did. `observable` names a key of the
    reconstruction report, and `rec_full` is the unmasked view, the number the
    collapse was measured on. Read every value against RECONSTRUCTION_FLOOR =
    0.2626, the base rate of always writing the commonest base, never against
    0.25.
    """

    enabled: bool = True
    observable: str = "rec_full"
    target: float = 0.95
    lam_init: float = 0.15
    lam_min: float = 1e-3
    #: Lowered from a ceiling of 100.0, which did not exist in practice: one
    #: run reached lambda 3.3820 with reconstruction taking 35.2 % of the
    #: epoch-1 loss. 2.0 is ABOVE the peak of every run that held
    #: reconstruction up (1.0120 / 1.2280 / 1.8310 at rec_full 0.9802 / 0.9797
    #: / 0.9908), so it clips none of them: it bounds the integrator's
    #: OVERSHOOT in epoch 1 rather than removing weight. Replaying the
    #: controller over 26 367 steps of a real `rec_full` trace puts 966 of them
    #: on a ceiling of 2.0, i.e. `sat_hi` 0.0366 -- non-zero, so the change is
    #: not dead, and far from `saturation_limit` 0.50.
    lam_max: float = 2.0
    #: PER STEP, in units of the constraint. Measured, not chosen: at
    #: lam_lr = 1.0 an early violation of c ~ 0.68 multiplies lambda by
    #: exp(0.68) = 1.97 EVERY STEP, and the first smoke run drove lambda from
    #: 0.15 to a ceiling of 100 in ten steps -- straight into the bound the
    #: controller exists to avoid, so `sat_hi` fired and the run became a test
    #: of the box. At 0.01 the same violation moves lambda by 0.7 % per step, a
    #: doubling in ~100 steps, which is slower than one epoch of this corpus and
    #: faster than the reconstruction can collapse.
    lam_lr: float = 0.01
    ema: float = 0.98
    warmup_steps: int = 20
    saturation_limit: float = 0.5
    #: Set True to run the FIXED weight deliberately, including at the
    #: coefficient that collapsed. Nothing else lifts that refusal.
    acknowledge_fixed_weight: bool = False

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.lam_min <= 0.0:
            raise ValueError(
                "recon_constraint.lam_min = %r: a multiplicative controller "
                "never leaves zero (0 * exp(x) = 0), so a floor at zero "
                "turns the first stretch of a satisfied constraint into a "
                "PERMANENT shutdown that reads in the log exactly like a "
                "controller at work." % (self.lam_min,))
        if not self.lam_min <= self.lam_init <= self.lam_max:
            raise ValueError("recon_constraint.lam_init %.4g outside [%.4g, "
                             "%.4g]" % (self.lam_init, self.lam_min,
                                        self.lam_max))
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("recon_constraint.ema must be in [0, 1)")
        if self.lam_lr <= 0.0:
            raise ValueError(
                "recon_constraint.lam_lr must be positive; a zero rate IS a "
                "fixed weight and has to be configured as one "
                "(enabled: false), so that the log says which was run")
        if not RECONSTRUCTION_FLOOR < self.target <= 1.0:
            raise ValueError(
                "recon_constraint.target %.4f must lie in (%.4f, 1.0]: below "
                "the unigram floor the constraint is satisfied by a model that "
                "has learnt nothing" % (self.target, RECONSTRUCTION_FLOOR))


@dataclass
class PrefixConstraintConfig(ConstraintConfig):
    """Second controller, computed on `rec_prefix_0`. A separate TYPE.

    A type and not merely another `default_factory`, because `from_mapping`
    builds a nested block from the FIELD'S TYPE and the factory applies only
    when the YAML omits that block. At `prefix_constraint: {enabled: true}` a
    bare `ConstraintConfig()` therefore appeared -- `observable: rec_full`,
    `target: 0.95` -- i.e. a second controller on the FIRST one's observable.
    The guard in `TrainingConfig.validate` catches that, but the cause is here.

    `target` is DERIVED, not chosen, and the value in the field is the
    derivation at the DEFAULT `recon_constraint.target` of 0.95:

        ceiling = ceil_l0 + (1 - ceil_l0) * floor = 0.1095 + 0.8905 * 0.2626
        sharpness = (target_rec_full - floor) / (1 - floor)
        target = floor + sharpness * (ceiling - floor)

    which gives 0.3379; at the 0.98 of HYPPE6/HYPPE7 it gives 0.3412. The loop
    recomputes it on the live module against the run's own target and prints
    both numbers side by side, so the field serves as DOCUMENTATION and as a
    guard ("a target above the information ceiling is refused") rather than as
    the setting -- a transcribed constant would drift from
    `recon_constraint.target` the first time that changed.
    """

    enabled: bool = False
    observable: str = "rec_prefix_0"
    target: float = 0.3379
    lam_init: float = 0.15
    lam_lr: float = 0.01
    lam_max: float = 2.0
    warmup_steps: int = 20


@dataclass
class OccupancyGateConstraintConfig(ConstraintConfig):
    """Third controller, on the SPARSITY of the occupancy gate. A separate TYPE.

    A type for the reason `PrefixConstraintConfig` records: a nested block is
    built from the field's TYPE, so at `occ_gate_constraint: {enabled: true}` a
    bare `ConstraintConfig()` would appear and become a THIRD controller on the
    FIRST one's observable. That is not theory; it has happened once already.

    The gate by itself is a READOUT, not a SELECTION: `occ_gate_ef` stood at
    280.6 of 284 -- practically uniform, with the top 20 holding 9.52 % of the
    norm mass against a floor of 7.04 % -- because NOTHING paid for switching a
    channel off. Selection needs sparsity PRESSURE, and the pressure is lambda
    under the constraint `sparsity >= target`. Without it an `ef` near 284 means
    "there was no reason to" rather than "the model needs every channel", and
    no other number in the log separates those two sentences.

    What it does not promise, and this must travel with every number it
    produces: the motif cascade measured that the `occ_proj` bottleneck destroys
    NOTHING -- reconstruction from codes alone 0.9973 against a 0.2661 floor,
    and the occupancy of the decoded codes against the original 0.9873 locally
    against a 0.0041 floor, so the codes carry 98.7 % of motif identity. The
    falls along the cascade (0.9266 -> 0.5497 -> 0.1564) are a property of the
    LINEAR readout, which understates by ~3.3x, not losses in the bottleneck. So
    the gate is not a repair of motif sensitivity and must not be reported as
    one; the expected change in PCC is ZERO. What it buys is selection and
    interpretability: which of the 252 frozen MEME profiles the model
    suppresses, and which of the 32 de novo ones it uses.

    `cel_kanalow` IS the setting and `target` is DOCUMENTATION. A target in
    sparsity units depends on the channel count, which the config does not know
    (284 today = 252 MEME + 32 de novo, and `motifs` can change the bank), so
    the loop computes `target = 1 - cel_kanalow/C` on the LIVE module and prints
    both numbers side by side.
    """

    enabled: bool = False
    #: NOT `rec_*`. The guard in `TrainingConfig.validate` refuses two
    #: controllers on one observable, and this name is its argument.
    observable: str = "occ_gate_rzadkosc"
    #: 64 = the width of the `occupancy_bottleneck`. Derived, not chosen:
    #: `occ_proj` already projects 284 channels onto 64, so above 64 EFFECTIVE
    #: input channels the gate decides nothing the bottleneck has not decided,
    #: and below it the gate begins to CHOOSE. At 284 channels that is a
    #: sparsity target of 1 - 64/284 = 0.7746 against today's 0.0120.
    cel_kanalow: float = 64.0
    #: DOCUMENTATION of the value the loop computes at C = 284.
    target: float = 0.7746
    #: A NEUTRAL point, not zero. The term is `lambda * mean(sigmoid(logit))`,
    #: so the sparsity pressure on ONE channel is `lambda * sigma' / C` --
    #: DIVIDED by the channel count -- and the nominal lambda and the real
    #: pressure differ by 284x. Measured per channel with the controller off,
    #: |dL_pred/dlogit| has median 1.54e-03 (p10 9.32e-04, p90 3.77e-03), while
    #: the sparsity pressure at lambda = 1 is 3.70e-04, so a lambda matching the
    #: median is about 4. 1.0 sits deliberately BELOW the operating point, so
    #: that the climb is visible in the log and is the CONTROLLER's doing rather
    #: than the config's.
    lam_init: float = 1.0
    lam_min: float = 1e-3
    #: The CEILING, derived rather than chosen. A channel survives when its own
    #: gradient is more negative than the penalty, so the surviving fraction is
    #: Phi(-lambda*sigma'/(C*sd)); for `ef` = 64 of 284 that puts the operating
    #: point at lambda* = 4.56, and 8.0 is 1.75x it. Deliberately NOT the
    #: operating point, where `sat_hi` would always fire and stop being a
    #: falsifier. An earlier 10.2 was wrong because it came from the p90 of the
    #: MEAN |grad|, which mixes signal with noise: the SIGNED preference is
    #: 2.233e-03 at SE 3.686e-04, so 10.2 applied 1.7x the whole per-channel
    #: preference uniformly and the gates went down in LOCKSTEP. The answer to
    #: `sat_hi` firing is a lower target (a larger `cel_kanalow`), never a
    #: higher ceiling, which buys nothing but switching channels off.
    lam_max: float = 8.0
    lam_lr: float = 0.01
    warmup_steps: int = 20

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.lam_min <= 0.0:
            raise ValueError(
                "occ_gate_constraint.lam_min = %r: a multiplicative "
                "controller never leaves zero (0 * exp(x) = 0), so a floor "
                "at zero turns the first stretch of a satisfied constraint "
                "into a PERMANENT shutdown that reads in the log exactly "
                "like a controller at work." % (self.lam_min,))
        if not self.lam_min <= self.lam_init <= self.lam_max:
            raise ValueError(
                "occ_gate_constraint.lam_init %.4g is outside the box "
                "[%.4g, %.4g]" % (self.lam_init, self.lam_min, self.lam_max))
        if self.lam_lr <= 0.0:
            raise ValueError(
                "occ_gate_constraint.lam_lr must be positive; a zero rate IS "
                "a fixed weight and has to be configured as a fixed weight "
                "(enabled: false), so that the log says which was run")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("occ_gate_constraint.ema must lie in [0, 1)")
        if self.cel_kanalow <= 0.0:
            raise ValueError(
                "occ_gate_constraint.cel_kanalow = %r. A target of zero "
                "channels means sparsity 1.0, i.e. ONE occupancy channel for "
                "the whole task -- unreachable, so the controller would ride "
                "up to lam_max and the run would measure the box."
                % (self.cel_kanalow,))
        if self.observable.startswith("rec_"):
            raise ValueError(
                "occ_gate_constraint.observable = %r starts with `rec_`, so "
                "it names an observable of the RECONSTRUCTION report, while "
                "this controller reads the GATE report. A key absent from "
                "its own report would give NaN in `update()` -- and that "
                "only raises in the optimiser, several steps later."
                % (self.observable,))


@dataclass
class SpySelectionConfig:
    """Pick the SHIPPING checkpoint on the spy panel, during training.

    The foundation model and the design model are different questions and the
    run answers both; this block is the second one. Among epochs that pass the
    hard gates it keeps the one that lands closest to the tef1/pks1 anchors,
    separates the most negatives, reconstructs best and is most confident.

    "Most confident" needs defining, because the obvious reading is gameable: a
    head can always shrink sigma, and a smaller sigma is not more trustworthy.
    `spy_confidence` is the Gaussian log-likelihood of the scored natives under
    the model's own calibrated mean and predicted variance, measured AGAINST a
    constant-variance baseline fitted to the same residuals, so a sigma
    carrying no per-sample information scores exactly 0. Mean sigma is reported
    next to it and is NOT scored.

    The anchor term is not a raw difference either. Predictions in this family
    are compressed towards the mean (measured calibration slope 0.8153-0.8414)
    and the intercept was never measured, so `|pred(tef1) - 12.3542|` would
    mostly measure the compression. `spy_anchor_fit` is the anchor error after
    an affine calibration fitted on the natives ONLY -- never on the anchors,
    or the error would be fitted away -- in units of the panel's own sd.

    Selecting on spy SPENDS spy: it is a validation budget like any other, and
    the only number still untouched at the end is `test`, read once.
    """

    #: OFF by default: an unchanged config reproduces earlier runs bit for bit.
    enabled: bool = False
    #: Empty = `data/splits/SPY_SET_V6.tsv`. The loop RAISES when the panel is
    #: missing rather than ending with no `_pks` file and no reason.
    panel_path: str = ""
    #: The design strain, BY NAME. An integer here would silently re-point at
    #: a different genome the day the registry gains one -- the registry is
    #: append-only and the index is the embedding row.
    strain: str = "P1"
    #: Orthogroup of the target gene, so the gene token is the one the design
    #: asks about instead of the unknown token. pks1 is OG0000996.
    orthogroup: str = "OG0000996"
    #: Whether `_pks` may only be taken from an epoch that calls BOTH anchors
    #: (pks1 and tef1) promoters. Judged against the median of `losowa_euler4`
    #: -- 30 order-4 Euler shuffles, composition preserved to 5-mers -- and NOT
    #: against an absolute threshold on the logit: an AUC of 0.916 does not
    #: carry over to a single cut point, because the discriminator's
    #: calibration moves between epochs. The gate exists on a measurement: the
    #: model can score pks1 as LESS real than its own shuffled version (by
    #: 1.67), and all ten designed promoters are to sit in the pks1 locus.
    require_anchor_promoters: bool = True
    #: Hard gates, not score terms: a score can be bought back somewhere else,
    #: a floor cannot. `require_edit_resolution` defaults to FALSE on a
    #: measurement. Its statistic is `shift(88 substitutions) /
    #: shift(2 substitutions)` over the panel, whose EFFECTIVE sample sizes are
    #: 1.53 and 33.64 -- the 88 positions come from a pool of 136 bp, so
    #: sibling variants share 64.6 % of their substitutions and the large arm is
    #: nearly ONE observation. The implied SE of 0.242 leaves an indecision
    #: window of AUC in [0.10, 0.90], and a hard gate on a quantity that cannot
    #: resolve rejects checkpoints on noise. It is still computed and reported
    #: every epoch; raising its power needs more PARENTS in the panel, which is
    #: a panel rebuild and not a config key.
    require_edit_resolution: bool = False
    require_anchor_order: bool = True

    def validate(self, path: str = "training.spy_selection") -> None:
        if self.enabled and not self.strain:
            raise ValueError("%s.strain is empty; it names the registry row "
                             "the panel is scored under" % (path,))


@dataclass
class TrainingConfig:
    epochs: int = 40
    lr: float = 3e-4
    #: Learning-rate schedule: "none" or "cosine". Until recently there was
    #: none at all -- `run()` never touched `param_groups`, so the LR was
    #: constant for a whole run. The symptom that explains: after the epoch-2
    #: peak the score OSCILLATES (0.6426 / 0.6530 / 0.6384 / 0.6524 / 0.6518 /
    #: 0.6611) instead of degrading monotonically, because the model has no way
    #: to settle. "none" is the default and is bit for bit the previous state.
    lr_schedule: str = "none"
    #: Fraction of the WHOLE run spent on linear warm-up before the cosine.
    #: 0.0 = no warm-up. Ignored at lr_schedule = "none".
    lr_warmup_frac: float = 0.0
    #: WEIGHT AVERAGING OVER EPOCHS (SWA): the first epoch whose weights enter
    #: the mean. 0 = off, and then not one new line executes in the run path.
    #:
    #: Measured on one run's checkpoints (full validation, one process, the same
    #: `evaluate` that produces the log's numbers): a 4:4 window, i.e. the
    #: plumbing control, reproduces ep004 to five places at 0.68061; 4:6 gives
    #: 0.68746, 4:8 0.68572, 4:11 0.68164, 4:14 0.67822. The optimum is interior
    #: and wide, and the mean beats EVERY one of its components on EVERY metric
    #: -- the signature of variance reduction rather than a lucky draw.
    swa_start_epoch: int = 0
    #: How many of the LAST epochs enter the running mean. 0 = off. 3 is
    #: measured, not chosen: windows of 3, 5, 8 and 11 give 0.68746 / 0.68572 /
    #: 0.68164 / 0.67822, a monotone fall from 3 upwards, so late epochs
    #: contribute BIAS -- drift out of the good region -- rather than variance.
    swa_window: int = 0
    weight_decay: float = 0.01
    riemannian: bool = True
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 7898
    select_on: str = "val_pairwise"

    #: THE SECOND SELECTION TRACK. One run, two artefacts: `_fundational.pt`
    #: chosen on VAL, the foundation model, and `_pks.pt` chosen on SPY, the
    #: model that ships for the pks1 design task. They are not the same epoch:
    #: `val` is 3 725 orthogroups of general promoters, while the deliverable
    #: is ten sequences for ONE gene in ONE strain, and pairwise accuracy on
    #: the corpus cannot see the two-base edit that each design is. Off by
    #: default; with it on, `SpyScorer` and `SelectionPolicy` finally run,
    #: where before NOTHING under `src/hyppe` imported them. TEST is read by
    #: neither track.
    spy_selection: "SpySelectionConfig" = field(
        default_factory=lambda: SpySelectionConfig())

    # Weight decay must not touch points on the manifold or branch gates.
    # The match is by NAME SUBSTRING, so these strings are a contract with the
    # model's parameter names: renaming `codebook_points` to `points_codebook`
    # silently re-enables decay on the hyperboloid, and lambda 0.0762 over 34
    # epochs already measured as 1.8e-5 of drift off the manifold.
    no_decay_patterns: Tuple[str, ...] = ("codebook", "manifold", "hyperbolic",
                                          "proto", "gate", "bias", "norm")
    manifold_patterns: Tuple[str, ...] = ("codebook", "manifold", "hyperbolic",
                                          "proto")

    #: HOW the task losses become one update. Scalarisation is the BASELINE and
    #: the default. A balancing method changes the effective task weights, and
    #: changing the effective weights alone moves the metric, so "nash gained
    #: +x" is indistinguishable from "the weights nash happened to pick gained
    #: +x" unless the baseline at those weights was also run. The price is
    #: measured too: a gradient method needs K+1 backward passes.
    method: str = "scalarise"
    METHODS = ("scalarise", "nash", "normprior")
    #: Coefficient of each player BEFORE the method sees its loss. A weight of
    #: 0.0 DROPS the task (for both methods, so that F3 stays satisfiable).
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "absolute": 1.0, "pairwise": 1.0, "discriminator": 1.0})

    nash: NashConfig = field(default_factory=NashConfig)
    normprior: NormPriorConfig = field(default_factory=NormPriorConfig)
    recon_constraint: ConstraintConfig = field(
        default_factory=ConstraintConfig)

    #: SECOND CONTROLLER TARGET, computed on `rec_prefix_0`. Off by default.
    #: `recon_constraint` watches `rec_full`, and that rests on L4 alone --
    #: destroying L0 costs 0.0000 of reconstruction while destroying L4 costs
    #: 0.3508 -- so the controller rode its 0.02 FLOOR for 5 428 of 35 588 steps
    #: and the `prefix_0` view's coefficient ended at 0.00144: the one pressure
    #: that could make L0 load-bearing was switched off by a target computed on
    #: a quantity L0 does not affect. `target` is DERIVED, not chosen (see
    #: `PrefixConstraintConfig`). The ceiling is modest: L0 can control AT MOST
    #: 10.95 %, ~88 of 800 bases, and today's prefix_0 of 0.3201 against the
    #: 0.2626 floor is already 71.2 % of it, so the whole headroom is
    #: +0.02..+0.03 of reconstruction. That is NOT a breakthrough.
    prefix_constraint: PrefixConstraintConfig = field(
        default_factory=PrefixConstraintConfig)

    #: SPARSITY PRESSURE on the occupancy gate. Without it
    #: `model.occupancy_gate: true` is a READOUT and not a SELECTION -- see
    #: `OccupancyGateConstraintConfig`. A guard in `HyppeConfig` refuses this
    #: block without `model.occupancy_gate`.
    occ_gate_constraint: OccupancyGateConstraintConfig = field(
        default_factory=OccupancyGateConstraintConfig)

    #: How many EVALUATION SEEDS the checkpoint-selection metric averages.
    #: 1 = the previous state. Measured on five seeds of ONE checkpoint's
    #: validation epoch: mean 0.69468 at sd 0.00596, so an argmax over the ~28
    #: draws of a run takes the LUCKIEST and the recorded peak is biased upward
    #: by about +0.01, i.e. 2 sd. `rec_*` is robust (sd 0.00001), so the
    #: averaging concerns the HEAD's metrics; only the `maker.epoch` seed
    #: changes. Cost: N times the evaluation time per epoch.
    eval_seeds: int = 1

    # Subordinate terms: NOT players. `recon_weight` is the declared weight and
    # `view_weights` are the factors between it and the loss, so the EFFECTIVE
    # coefficient can be printed instead of multiplied by hand.
    recon_weight: float = 0.15
    mask_views: Tuple[str, ...] = ("full", "column", "degree_0", "prefix_k")
    view_weights: Dict[str, float] = field(default_factory=lambda: {
        "full": 0.40, "column": 0.25, "degree_0": 0.20, "prefix_k": 0.15})
    commitment_weight: float = 0.25
    entropy_weight: float = 0.0

    # Guards. Every number below is DERIVED from the run that made them
    # necessary: it reached epoch 40 and exited 0 with a loss of 2.662e16, of
    # which 99.999 % was `term_commitment`, and printed none of it. These are
    # the three gates that would have stopped it, in the order they would have
    # fired.

    #: Relative tolerance on `sum(components) == loss`, the closure assertion.
    #: RELATIVE, not absolute: accumulating K float32 addends carries a relative
    #: error of about K*eps = 3.6e-07, which at the healthy epoch-1 loss of 7.49
    #: is 2.7e-06 ABSOLUTE, so a fixed 1e-6 absolute tolerance fires on a
    #: perfectly healthy step. It is scaled by sum|component| rather than by
    #: |loss|, because what sets the error is cancellation between large terms
    #: of opposite sign and not the total.
    closure_tol: float = 1e-6

    #: Stop the run when the step loss exceeds this. 133x the healthy epoch-1
    #: loss of 7.49 and 2.7e13x below where the runaway ended, so everything in
    #: between is caught; the tasks themselves ran in 0.20-1.83, so a single
    #: hard batch cannot approach it. With `l0_max_radius = 8.0` the commitment
    #: term is bounded near 32-4300, which puts this gate ABOVE the clamp's own
    #: worst case on purpose: it is the backstop for terms OTHER than
    #: commitment, while `enc_radius_max` fires first for commitment. 0.0
    #: disables it, and disabling it is a decision the header prints.
    loss_max: float = 1.0e3

    #: Stop the run when the PRE-CLAMP level-0 encoder radius exceeds this.
    #: 2.45x the healthy latent radius of 2.4447 and 75 % of the hard clamp at
    #: `model.l0_max_radius = 8.0`, so it fires while the clamp is still
    #: INACTIVE. That ordering is the point: once the clamp bites the radial
    #: gradient is gone and the geometry the run reports is no longer the
    #: geometry it is training, while every post-clamp observable still reads
    #: 8.0 and looks stable. At radius 6 the commitment term is 4.1 for the
    #: nearest code and 581 at worst; at the runaway's 41.33 it is 9.6e15.
    #: 0.0 disables it.
    enc_radius_max: float = 6.0

    #: Stop the run when this fraction of level-0 positions is being clipped
    #: by `l0_max_radius`. Not 0.0: the clamp is allowed to catch an outlier
    #: position. 1 % of positions held at the cap is not an outlier, it is the
    #: encoder living on the boundary, and from there the pre-clamp radius is
    #: unobservable in every other statistic.
    enc_clip_frac_max: float = 0.01

    #: HARD ADMISSION for a checkpoint: no file is written below it, whatever
    #: the selection score says. Derived -- the reconstruction floor is 0.2626
    #: and epoch 1 reached 0.9486 rising to 0.9974, so 0.90 is BELOW what the
    #: model demonstrably does and far above the floor. On the scale
    #: (v - floor)/(1 - floor) it is 0.864, against 0.930 for epoch 1 and 0.070
    #: for the 0.3139 that `select_on: val_pairwise` once saved: that checkpoint
    #: had 7 % of the achievable reconstruction and was still written.
    recon_admit: float = 0.90

    #: Weight of reconstruction in the composite selection score, as the
    #: exponent of a GEOMETRIC mean. A weighted SUM lets a collapsing factor
    #: be bought back by the other one, which is precisely how a 0.3139
    #: reconstruction was selected; a geometric mean goes to zero with either
    #: factor and cannot be traded that way. 0.0 reproduces the old behaviour
    #: exactly (score = the `select_on` metric), and the admission floor above
    #: still applies -- the two are independent on purpose.
    select_recon_weight: float = 0.5

    def effective_recon_coefficients(self) -> Dict[str, float]:
        """Declared weight times every factor on the way to the loss."""
        return {v: self.recon_weight * float(self.view_weights.get(v, 0.0))
                for v in self.mask_views}

    def effective_recon_coefficient(self, view: str = "full") -> float:
        """The number to compare against an alpha, for ONE view."""
        return float(self.recon_weight) * float(self.view_weights.get(view, 0.0))

    def shared_scope_for(self, method: str) -> str:
        """Which scope the ACTIVE method competes over.

        The NAME only; `TrainLoop._shared_parameters` turns it into modules
        and is the single place that knows the mapping. `--method` can
        override `self.method`, so the method is an argument rather than read
        off this object: a resolver that read `self.method` would return the
        config's scope for a run that is not using the config's method.
        """
        if str(method) == "normprior":
            return self.normprior.shared_scope
        return self.nash.shared_scope

    def refresh_cadence(self, method: str) -> int:
        """Steps between recomputations of the per-task gradients, AS SEEN BY
        THE CALLER (the `recompute` flag of `TaskWeighting.apply`).

        For `nash` this is `nash.recompute_every`, because `NashMTL` has no
        counter of its own and reuses `last` only when the caller says so.
        For `normprior` it is 1 ON PURPOSE: the cadence lives inside the
        method (`NormPriorWeighting.refresh_every`, fed from
        `normprior.refresh_every`), so the caller always PERMITS a refresh and
        exactly one knob decides. Two knobs driving one quantity is how
        `se_stat` came to be configured with its own gate at zero.
        """
        if str(method) == "normprior":
            return 1
        return max(1, int(self.nash.recompute_every))

    def validate(self) -> None:
        if str(self.lr_schedule) not in ("none", "cosine"):
            raise ValueError(
                "training.lr_schedule = %r; the allowed values are 'none' "
                "and 'cosine'. Accepting a typo silently would run a fixed "
                "LR under the name of a scheduled run." % (self.lr_schedule,))
        if not (0.0 <= float(self.lr_warmup_frac) < 0.5):
            raise ValueError(
                "training.lr_warmup_frac = %r; must lie in [0, 0.5). Beyond "
                "half the run the warm-up eats the schedule."
                % (self.lr_warmup_frac,))
        self.nash.validate()
        self.normprior.validate()
        self.recon_constraint.validate()
        self.prefix_constraint.validate()
        if int(self.eval_seeds) < 1:
            raise ValueError(
                "training.eval_seeds = %r. Zero seeds means no evaluation at "
                "all, and a selection metric without evaluation picks the "
                "last epoch silently." % (self.eval_seeds,))
        self.occ_gate_constraint.validate()
        # Two controllers on one observable are ONE controller with double
        # lam_lr, and nothing in the log says so. The check loops over ALL
        # enabled controllers rather than naming a pair, because a condition
        # that enumerates cases always falls behind the next case, and silence
        # is then indistinguishable from a passed gate.
        _kontrolery = (("recon_constraint", self.recon_constraint),
                       ("prefix_constraint", self.prefix_constraint),
                       ("occ_gate_constraint", self.occ_gate_constraint))
        _widziane = {}
        for _name, _blok in _kontrolery:
            if not bool(getattr(_blok, "enabled", False)):
                continue
            _obs = str(getattr(_blok, "observable", ""))
            if _obs in _widziane:
                raise ValueError(
                    "%s.observable = %r, which is the same as %s. Two "
                    "controllers on one observable are one controller with "
                    "lam_lr doubled, and nothing in the log says so."
                    % (_name, _obs, _widziane[_obs]))
            _widziane[_obs] = _name
        self.spy_selection.validate()
        if self.method not in self.METHODS:
            raise ValueError(
                "training.method must be one of %r, got %r. An unknown name "
                "must not fall back to the baseline: the run would look like "
                "the method it named, produce the baseline's numbers, and "
                "nothing in the log would say which one it was."
                % (self.METHODS, self.method))
        if self.epochs < 1:
            raise ValueError("training.epochs must be >= 1")
        # BOTH OR NEITHER. `swa_start_epoch` alone would collect weights and
        # never average them; `swa_window` alone would never fire. Either way
        # it is cost with no effect, and nothing in the log would say so.
        if (self.swa_start_epoch > 0) != (self.swa_window > 0):
            raise ValueError(
                "training.swa_start_epoch = %r and training.swa_window = %r: "
                "either BOTH are positive or both are zero. One without the "
                "other is a mechanism that never fires, and such a key reads "
                "in the config like an enabled one."
                % (self.swa_start_epoch, self.swa_window))
        if self.swa_window == 1:
            raise ValueError(
                "training.swa_window = 1 is not an average -- it is a copy "
                "of the live weights, costing one extra evaluation per epoch "
                "and carrying zero information. A window of length 1 makes "
                "sense ONLY as a plumbing control for a probe, not in a run.")
        if self.swa_window > 0 and self.swa_window > self.epochs:
            raise ValueError(
                "training.swa_window = %d with training.epochs = %d: the "
                "window never fills, so the average is never formed at all."
                % (self.swa_window, self.epochs))
        if self.closure_tol <= 0.0:
            raise ValueError(
                "training.closure_tol must be > 0. A non-positive tolerance "
                "makes the closure assertion fire on every step, and an "
                "assertion that always fires gets switched off.")
        if not RECONSTRUCTION_FLOOR < self.recon_admit <= 1.0:
            raise ValueError(
                "training.recon_admit=%.4f must lie in (%.4f, 1.0]. At or "
                "below the floor the admission gate admits a model that has "
                "learnt nothing about the sequence, which is the state it "
                "exists to reject."
                % (self.recon_admit, RECONSTRUCTION_FLOOR))
        if not 0.0 <= self.select_recon_weight <= 1.0:
            raise ValueError(
                "training.select_recon_weight=%.4f must lie in [0, 1]; it is "
                "the exponent of a geometric mean, not a multiplier."
                % (self.select_recon_weight,))
        if self.loss_max < 0.0 or self.enc_radius_max < 0.0:
            raise ValueError(
                "training.loss_max and training.enc_radius_max must be >= 0 "
                "(0 = the gate is off, and the header says so).")
        # Reconstruction is a CONSTRAINT. With the controller switched off the
        # run falls back to exactly the arithmetic that collapsed
        # 0.9998 -> 0.5549 in one epoch, through two keys that each look
        # innocent -- which is why the PRODUCT is checked and not either
        # factor.
        if not self.recon_constraint.enabled:
            eff = self.effective_recon_coefficient("full")
            if (abs(eff - FIXED_WEIGHT_COLLAPSE) < 1e-9
                    and not self.recon_constraint.acknowledge_fixed_weight):
                raise ValueError(
                    "recon_constraint is off and recon_weight %.4f x "
                    "view_weights['full'] %.4f = %.4f -- bit for bit the "
                    "effective coefficient at which reconstruction fell from "
                    "0.9998 to 0.5549 in ONE epoch on the sibling track. "
                    "Either leave the controller on (the default), or set "
                    "training.recon_constraint.acknowledge_fixed_weight: true "
                    "to state that reproducing it is the point of the run."
                    % (self.recon_weight, self.view_weights.get("full", 0.0),
                       eff))
        missing = [v for v in self.mask_views if v not in self.view_weights]
        if missing:
            raise ValueError(
                "no view weight for %r: the effective coefficient of the "
                "reconstruction term would be unknown, and an unknown "
                "multiplier chain is how 0.15 x 0.40 = 0.060 was reported as "
                "0.15 against an alpha of 0.924 -- a factor of 15.4."
                % (missing,))
        if not self.no_decay_patterns:
            raise ValueError(
                "an empty no_decay list decays the hyperboloid points and the "
                "branch gates; both have killed mechanisms in this project")


# Config
@dataclass
class HyppeConfig:
    """The whole configuration, plus what the loader found wrong with it."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conditioning: ConditioningConfig = field(
        default_factory=ConditioningConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    name: str = "hyppe"
    source: str = ""
    # Filled by the loader; kept ON the config so that `report_inert_keys` can
    # print them next to the keys that are inert for structural reasons.
    unknown_keys: Tuple[str, ...] = ()
    duplicate_keys: Tuple[str, ...] = ()

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.conditioning.validate()
        self.heads.validate()
        self.training.validate()
        # A controller with no gate has nothing to squeeze: `occ_gate_sparsity`
        # would return None and the loop would raise, but only after the model
        # is built and the corpus loaded -- minutes later, and somewhere that
        # does not point at the config. The reverse is NOT an error: a gate
        # without a controller is a READOUT, a deliberate documented mode.
        if self.training.occ_gate_constraint.enabled and \
                not bool(self.model.occupancy_gate):
            raise ValueError(
                "training.occ_gate_constraint.enabled = true, but "
                "model.occupancy_gate = false: the `occ_gate_logit` "
                "parameter is never created at all, so the controller would "
                "have nothing to squeeze, while the config block would read "
                "like a decision taken for this run. Either the gate "
                "together with the controller, or neither.")
        if self.training.method == "nash" and not self.training.nash.enabled:
            raise ValueError(
                "training.method is 'nash' while training.nash.enabled is "
                "false. One of the two would have to win silently, and a run "
                "that reports a method it did not use is the failure this "
                "whole section exists to prevent.")
        # The same rule in both directions: a `training.normprior` block under
        # `method: nash` would configure priors that nothing reads.
        if self.training.method == "normprior" and \
                not self.training.normprior.enabled:
            raise ValueError(
                "training.method is 'normprior' while "
                "training.normprior.enabled is false: one of the two would "
                "have to win silently.")
        if self.training.method != "normprior" and \
                self.training.normprior.enabled:
            raise ValueError(
                "training.normprior.enabled is true while training.method is "
                "%r. Every normprior key would be inert -- the priors, the "
                "refresh cadence and the shared scope -- while the block "
                "reads like a decision taken for this run."
                % (self.training.method,))
        weighted = set(self.training.task_weights)
        enabled = set(self.heads.enabled_heads())
        if not weighted.issubset(enabled):
            raise ValueError(
                "training.task_weights names %r, which are not enabled heads "
                "%r: a weight with no loss is an inert key, and an inert key "
                "is how this project has repeatedly run the default while "
                "believing otherwise."
                % (sorted(weighted - enabled), sorted(enabled)))
        players = set(self.training.nash.players)
        if self.training.nash.enabled and not players.issubset(enabled):
            raise ValueError(
                "nash players %r are not all enabled heads %r: a player with "
                "no loss would enter the bargain with a zero gradient, and a "
                "zero gradient takes the MAXIMUM weight"
                % (sorted(players - enabled), sorted(enabled)))
        np_players = set(self.training.normprior.players)
        if self.training.normprior.enabled and \
                not np_players.issubset(enabled):
            raise ValueError(
                "normprior players %r are not all enabled heads %r: a player "
                "with no loss has a zero gradient, and under 1/||g|| that "
                "inflates the normalising sum and shrinks EVERY other task's "
                "contribution while sum(alpha) still reads the player count"
                % (sorted(np_players - enabled), sorted(enabled)))

    def latent_dim(self) -> int:
        """Width of the pooled latent the heads receive."""
        return sum(self.model.channels) * len(self.heads.pooling.stats)

    def context_dim(self) -> int:
        return self.conditioning.context_dim()

    def header(self) -> str:
        b = self.model.bit_budget(self.data.sequence_length)
        eff = self.training.effective_recon_coefficients()
        return ("%s | %d bp | %d levels %r | latent %d | context %d | "
                "code budget %.0f bits vs %.1f measured (%.2fx reserve, floor "
                "%.4f) | recon effective %s"
                % (self.name, self.data.sequence_length,
                   self.model.n_levels(),
                   self.model.map_lengths(self.data.sequence_length),
                   self.latent_dim(), self.context_dim(),
                   b["total_bits"], b["sequence_bits"], b["reserve"],
                   RECONSTRUCTION_FLOOR,
                   " ".join("%s=%.4f" % (k, v) for k, v in sorted(eff.items()))))


# Loader
class DuplicateKeyError(ValueError):
    """A key given twice in the same block. `yaml.safe_load` takes the last
    one without a warning; here it is an error unless explicitly tolerated."""


_KEY_RE = re.compile(r"^(\s*)(?:-\s+)?([A-Za-z_][\w.\-]*)\s*:(\s|$)")
_ITEM_RE = re.compile(r"^(\s*)-\s")
_BLOCK_RE = re.compile(r":\s*[|>][-+0-9]*\s*$")


def scan_duplicate_keys(text: str) -> List[str]:
    """Format-independent scan for a key repeated inside one block.

    Deliberately textual: it works without PyYAML, it works on a file that does
    not parse, and it reports the LINE NUMBERS, which a parser hook cannot do
    for every YAML implementation. It is approximate -- flow mappings on one
    line are not inspected -- and it is a belt, not the braces: `load_mapping`
    also detects duplicates through the parser itself.
    """
    stack: List[Tuple[int, str]] = []
    seen: Dict[str, Dict[str, int]] = {}
    out: List[str] = []
    block_indent: Optional[int] = None
    item_counter: Dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if block_indent is not None:
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent > block_indent:
                continue
            block_indent = None
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Strip a trailing comment only when it is clearly outside a string.
        if " #" in line and line.count('"') % 2 == 0 and \
                line.count("'") % 2 == 0:
            line = line.split(" #", 1)[0].rstrip()
        m_item = _ITEM_RE.match(line)
        m = _KEY_RE.match(line)
        if m is None:
            continue
        indent = len(m.group(1))
        key = m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else ""
        if m_item is not None:
            # A key that starts a list item opens its own scope, otherwise the
            # same key in two items of one list would look like a duplicate.
            n = item_counter.get(parent, 0) + 1
            item_counter[parent] = n
            parent = "%s[%d]" % (parent, n)
        bucket = seen.setdefault(parent, {})
        if key in bucket:
            out.append("%s%s%s (lines %d and %d)"
                       % (parent, "." if parent else "", key,
                          bucket[key], lineno))
        else:
            bucket[key] = lineno
        stack.append((indent, "%s%s%s" % (parent, "." if parent else "", key)))
        if _BLOCK_RE.search(line):
            block_indent = indent
    return out


def _json_pairs_hook(duplicates: List[str]):
    def hook(pairs):
        seen: Dict[str, int] = {}
        out: Dict[str, Any] = {}
        for k, v in pairs:
            if k in seen:
                duplicates.append(k)
            seen[k] = 1
            out[k] = v
        return out
    return hook


def load_mapping(path: str, tolerate_duplicates: bool = False
                 ) -> Tuple[Dict[str, Any], List[str]]:
    """Read a JSON or YAML file into a mapping, refusing silent duplicates.

    PyYAML is imported lazily and only for `.yaml`/`.yml`, so this module (and
    a config check) still works in an environment that has neither torch nor
    yaml. JSON needs no dependency at all and is the recommended format for a
    machine-generated config, because generating YAML by text substitution is
    how the duplicated keys got in.
    """
    with io.open(path, encoding="utf-8") as f:
        text = f.read()
    textual = scan_duplicate_keys(text)
    ext = os.path.splitext(path)[1].lower()
    duplicates: List[str] = list(textual)
    if ext in (".json",):
        data = json.loads(text, object_pairs_hook=_json_pairs_hook(duplicates))
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:      # pragma: no cover - environment
            raise ImportError(
                "reading %s needs PyYAML, which is not a dependency of this "
                "package. Either install it or use the JSON form, which the "
                "standard library parses and which cannot hide a duplicated "
                "key." % (path,)) from exc

        class _Loader(yaml.SafeLoader):
            pass

        def _mapping(loader, node, deep=False):
            out: Dict[Any, Any] = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in out:
                    duplicates.append(str(key))
                out[key] = loader.construct_object(value_node, deep=deep)
            return out

        _Loader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
        data = yaml.load(text, Loader=_Loader)
    else:
        raise ValueError("unknown config format %r (expected .json/.yaml/.yml)"
                         % (ext,))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("the top level of %s is %s, expected a mapping"
                         % (path, type(data).__name__))
    if duplicates and not tolerate_duplicates:
        raise DuplicateKeyError(
            "%s defines the same key twice: %s. `yaml.safe_load` keeps the "
            "LAST occurrence without a warning, so every grep on the first one "
            "reports the opposite of the effective value -- measured three "
            "times in one config family. Remove the duplicate, or load with "
            "tolerate_duplicates=True and read `cfg.duplicate_keys`."
            % (path, ", ".join(sorted(set(duplicates)))))
    return data, duplicates


# Construction
_HINT_CACHE: Dict[Any, Dict[str, Any]] = {}


def _hints(cls: Any) -> Dict[str, Any]:
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = typing.get_type_hints(cls)
    return _HINT_CACHE[cls]


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "on", "1"):
            return True
        if s in ("false", "no", "off", "0"):
            return False
        raise ValueError("cannot read %r as a boolean" % (v,))
    return bool(v)


def _coerce(value: Any, tp: Any, path: str, unknown: List[str]) -> Any:
    if tp is Any or tp is None:
        return value
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ValueError("%s expects a mapping, got %r" % (path, value))
        return _from_mapping(tp, value, path, unknown)
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], path, unknown)
    if origin in (tuple,):
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], "%s[%d]" % (path, i), unknown)
                         for i, v in enumerate(value))
        return tuple(_coerce(v, a, "%s[%d]" % (path, i), unknown)
                     for i, (v, a) in enumerate(zip(value, args)))
    if origin in (list,):
        (arg,) = get_args(tp) or (Any,)
        return [_coerce(v, arg, "%s[%d]" % (path, i), unknown)
                for i, v in enumerate(value)]
    if origin in (dict,):
        args = get_args(tp) or (Any, Any)
        return {_coerce(k, args[0], path, unknown):
                _coerce(v, args[1], "%s.%s" % (path, k), unknown)
                for k, v in value.items()}
    if tp is bool:
        return _to_bool(value)
    if tp is int:
        return int(value)
    if tp is float:
        return float(value)
    if tp is str:
        return str(value)
    return value


def _from_mapping(cls: Any, mapping: Mapping[str, Any], prefix: str,
                  unknown: List[str]) -> Any:
    hints = _hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in mapping.items():
        p = "%s.%s" % (prefix, k) if prefix else str(k)
        if k not in names:
            # An unknown key is NOT an error by default: it is the exact thing
            # `report_inert_keys` has to be able to list. Raising here would
            # hide the rest of them behind the first one.
            unknown.append(p)
            continue
        kwargs[k] = _coerce(v, hints[k], p, unknown)
    return cls(**kwargs)


#: The config most recently loaded in this process, or None. `Corpus.from_paths`
#: needs `data.max_n_frac` and its only training call site does not pass it,
#: which would make the key one with NO RECEIVER. This registry is a BRIDGE:
#: `from_paths` takes the value from its argument when given one and from the
#: last loaded config otherwise, and always prints which source it used, in
#: `CorpusReport`. One line in `build_corpus` would make it unnecessary; until
#: then, silence must not be equivalent to the default.
_ACTIVE_CONFIG: Optional["HyppeConfig"] = None


def active_config() -> Optional["HyppeConfig"]:
    """The config most recently loaded by `load_config` / `from_mapping`."""
    return _ACTIVE_CONFIG


def _set_active(cfg: "HyppeConfig") -> "HyppeConfig":
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = cfg
    return cfg


#: Config keys holding a path to data. Resolved against HYPPE_DATA_ROOT when
#: they are relative, because that is the contract the README states and the
#: rest of the package already honours. Without this a relative path resolved
#: against the CURRENT DIRECTORY, so the same config trained from the repo
#: root and from the data root were two different runs -- and the failure was
#: a missing file only when the file happened to be missing.
_DATA_PATH_KEYS: Tuple[Tuple[str, str], ...] = (
    ("data", "sequences_path"),
    ("data", "combat_path"),
    ("data", "rank_path"),
    ("data", "samples_path"),
    ("data", "text_embedding_path"),
    ("data", "sample_text_path"),
    ("data", "flank_embedding_path"),
    ("conditioning", "orthology_path"),
)


def _reroot(path: str, root: str) -> Optional[str]:
    """Re-root an absolute path from another machine, or return None.

    A checkpoint records the ABSOLUTE paths of the machine it was trained
    on, so loading one anywhere else fails on a directory that cannot
    exist. Rather than guess the old root, this strips leading components
    one at a time and returns the first suffix that actually exists under
    `root`. An existing path is never touched, and a path with no existing
    suffix is left alone so the caller still gets the original error naming
    the file it really wanted.
    """
    if os.path.exists(path):
        return None
    parts = os.path.normpath(path).replace("\\", "/").strip("/").split("/")
    for i in range(len(parts)):
        candidate = os.path.join(root, *parts[i:])
        if os.path.exists(candidate):
            return candidate
    return None


#: The repository itself, used when no data root is set. Several inputs
#: ship inside it, so a clone can resolve them without being configured.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def resolve_data_paths(cfg: "HyppeConfig",
                       root: Optional[str] = None) -> List[str]:
    """Make data paths absolute against the data root, in place.

    Two roots are tried: `HYPPE_DATA_ROOT` if it is set, then the repository.
    The second matters because a checkpoint stores the absolute paths of the
    machine that trained it, and without a fallback a clone with no data root
    configured kept those paths verbatim and failed on directories that exist
    on no other machine. The files that ship (the orthology table, the splits,
    the panel) are found by the repository root alone.

    Returns the keys that were rewritten, which the caller may report; a
    silent rewrite of a path is its own kind of trap.
    """
    roots = [r for r in (root, os.environ.get("HYPPE_DATA_ROOT"),
                         _REPO_ROOT) if r]
    if not roots:
        return []
    root = roots[0]
    changed: List[str] = []
    def resolved(value: str) -> Optional[str]:
        """Where this path actually is, or None to leave it as it stands.

        A relative path always joins to the first root, whether or not the
        file is there yet: that is the contract HYPPE_DATA_ROOT states, and
        probing for an existing file instead would silently prefer a
        different root to the one the user set.

        Only an absolute path that does not exist gets searched for, because
        it came from another machine and there is nothing else to do with it.
        """
        if not os.path.isabs(value):
            return os.path.join(root, value)
        if os.path.exists(value):
            return None
        for r in roots:
            moved = _reroot(value, r)
            if moved is not None:
                return moved
        return None

    for section, key in _DATA_PATH_KEYS:
        block = getattr(cfg, section, None)
        value = getattr(block, key, "") if block is not None else ""
        if not isinstance(value, str) or not value:
            continue
        moved = resolved(value)
        if moved is not None and moved != value:
            setattr(block, key, moved)
            changed.append("%s.%s" % (section, key))
    spy = getattr(cfg.training, "spy_selection", None)
    panel = getattr(spy, "panel_path", "") if spy is not None else ""
    if isinstance(panel, str) and panel:
        moved = resolved(panel)
        if moved is not None and moved != panel:
            spy.panel_path = moved
            changed.append("training.spy_selection.panel_path")
    return changed


def load_config(path: str, strict: bool = True,
                tolerate_duplicates: bool = False) -> HyppeConfig:
    """Read a config file, validate it, and keep what was wrong with it.

    `strict` decides what happens to keys the file sets and no dataclass
    declares. They are ALWAYS recorded on `cfg.unknown_keys`; `strict` only
    decides whether that is fatal. A typo in a key name is the cheapest
    possible way to run an experiment that measures the default.
    """
    mapping, duplicates = load_mapping(path, tolerate_duplicates)
    unknown: List[str] = []
    cfg = _from_mapping(HyppeConfig, mapping, "", unknown)
    cfg.source = path
    resolve_data_paths(cfg)
    cfg.unknown_keys = tuple(unknown)
    cfg.duplicate_keys = tuple(sorted(set(duplicates)))
    if unknown and strict:
        raise ValueError(
            "%s sets keys that no section declares: %s. They would have no "
            "effect and the run would silently measure the defaults."
            % (path, ", ".join(unknown)))
    cfg.validate()
    _set_active(cfg)
    return cfg


def from_mapping(mapping: Mapping[str, Any], strict: bool = False
                 ) -> HyppeConfig:
    """Inverse of :func:`to_mapping`: the config a CHECKPOINT carries.

    Until this existed, `to_mapping` had no public inverse, so reloading a
    checkpoint outside this repo meant calling the private `_from_mapping`
    with four positional arguments -- which is what every example ended up
    copying. A private call in the first line of a "minimal example" is a
    contract nobody promised to keep.

    `strict` behaves as in :func:`load_config`: keys that no dataclass
    declares are always recorded on `cfg.unknown_keys`, and `strict` only
    decides whether they are fatal. Default False here and True there on
    purpose -- a config FILE with a stray key is a typo the author can still
    fix, while a checkpoint written by a NEWER version of this code is a fact,
    and refusing to load it would turn a forward-compatible read into a crash.
    """
    unknown: List[str] = []
    cfg = _from_mapping(HyppeConfig, mapping, "", unknown)
    if unknown:
        # The mapping's own `unknown_keys` (if it had one) described the FILE
        # that produced it; these describe THIS read. Both are kept, because
        # dropping either loses a different fact.
        cfg.unknown_keys = tuple(sorted(set(tuple(cfg.unknown_keys or ()))
                                        | set(unknown)))
        if strict:
            raise ValueError(
                "the mapping sets keys that no section declares: %s. Pass "
                "strict=False to load anyway; they are on cfg.unknown_keys "
                "either way." % (", ".join(unknown)))
    # A checkpoint carries the absolute paths of the machine it was trained
    # on. Re-rooting them here is what lets a delivered model be opened
    # anywhere; a path that already exists is never touched.
    resolve_data_paths(cfg)
    cfg.validate()
    _set_active(cfg)
    return cfg


def to_mapping(cfg: HyppeConfig) -> Dict[str, Any]:
    """The config as plain data, for writing next to a checkpoint.

    The contract is "the config file IS what ran", so this is the object that
    should be saved -- not the file that was read, which may have been edited
    since.
    """
    return dataclasses.asdict(cfg)


# Inert keys
# Tokens expected in the model's module or parameter names when a channel is
# actually consumed. These are a NAMING CONTRACT, the same one the weight-decay
# patterns are: a mismatch here means either a dead channel or a rename, and
# both need to be looked at.
_CONSUMERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("conditioning.gene_context", ("gene_context", "context", "ctx")),
    ("conditioning.genome", ("genome", "species")),
    ("conditioning.categories", ("categor", "condition")),
    ("conditioning.experiment_text", ("experiment", "text")),
    ("model.position_channel", ("position", "anchor")),
    ("model.band_attention", ("band", "attn")),
    ("heads.absolute", ("absolute", "mu_layer")),
    ("heads.pairwise", ("pairwise", "score_layer")),
    ("heads.discriminator", ("discrimin", "out_layer")),
)


def _model_names(model: Any) -> List[str]:
    names: List[str] = []
    for attr in ("named_parameters", "named_modules", "named_buffers"):
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        try:
            names.extend(n for n, _ in fn())
        except TypeError:               # pragma: no cover - odd duck
            continue
    return names


def _manifold_tensor_count(model: Any) -> Optional[int]:
    """How many tensors the BUILT model puts in the `manifold` group.

    ``None`` means there is no model to ask -- an unanswerable question must
    not be answered, which is the whole reason this helper exists.

    Two modules supply manifold tensors and `model.vq_learnable` governs only
    ONE of them: `Ladder.manifold_parameters` (model/ladder.py) delegates to
    the level-0 codebook, which is an EMA buffer unless `vq_learnable`, and
    `SpeciesGeometry.manifold_parameters` (model/species.py) returns
    ``[self.points] if self.points.requires_grad else []`` no matter what
    the quantiser does. Deciding the question from the flag therefore gets
    the answer wrong exactly when the genome channel is on -- which is the
    configuration this project's competition model ships.
    """
    if model is None:
        return None
    fn = getattr(model, "param_groups", None)
    if callable(fn):
        try:
            groups = fn()
        except Exception:                  # a report must not raise
            groups = None
        if groups:
            for g in groups:
                try:
                    if g.get("manifold") or g.get("name") == "manifold":
                        return len(list(g["params"]))
                except Exception:
                    continue
            return 0
    total = None
    for holder in ("ladder", "genome"):
        sub = getattr(model, holder, None)
        mp = getattr(sub, "manifold_parameters", None)
        if callable(mp):
            try:
                total = (total or 0) + len(list(mp()))
            except Exception:              # a report must not raise
                continue
    return total


def report_inert_keys(cfg: HyppeConfig, model: Any = None
                      ) -> List[Tuple[str, str]]:
    """Keys this config SETS that cannot have an effect. The point of the module.

    Four kinds of entry:
      * keys the file declares and no section reads (typos, stale names);
      * keys duplicated in the file, where the last one silently won;
      * keys switched off by another key -- the `se_stat` / `se_stat_bloki`
        pattern, where the feature is configured and its gate is zero;
      * keys with no receiver in the MODEL, found by name. That last check is a
        report and not a proof: a miss means either a dead channel or a rename,
        and the naming contract is exactly what broke when `maska_wektor`
        became `wektor_maski` and the hyperboloid points started decaying.
    """
    out: List[Tuple[str, str]] = []
    for key in cfg.unknown_keys:
        out.append((key,
                    "declared in %s and read by no section" % (cfg.source,)))
    for key in cfg.duplicate_keys:
        out.append((key, "given twice; the parser keeps the LAST occurrence "
                         "and every grep on the first one is wrong"))

    # -- structural: a knob whose mechanism is off ---------------------------
    t, h, m, c = cfg.training, cfg.heads, cfg.model, cfg.conditioning
    if t.method != "nash" and t.nash.enabled:
        out.append(("training.nash.*",
                    "training.method is %r, so no alpha is computed and every "
                    "nash key is inert in this run" % (t.method,)))
    if t.method == "normprior":
        # `nash.recompute_every` is the caller-side cadence and normprior
        # ignores it: `TrainingConfig.refresh_cadence` returns 1 for this
        # method so that exactly one knob decides.
        if t.nash.recompute_every != 1:
            out.append(("training.nash.recompute_every",
                        "method is normprior: the refresh cadence comes from "
                        "training.normprior.refresh_every (%d), and this key "
                        "is not consulted"
                        % (t.normprior.refresh_every,)))
        for nm in ("absolute", "pairwise", "discriminator"):
            hd = getattr(h, nm)
            if hd.enabled and hd.weight != 1.0:
                out.append(("heads.%s.weight" % nm,
                            "normprior gives this task normprior.priors[%s] "
                            "parts of its UNIT gradient, so the declared "
                            "%.3f only decides whether it plays at all"
                            % (nm, hd.weight)))
    if t.recon_constraint.enabled:
        out.append(("training.recon_weight",
                    "the reconstruction CONSTRAINT is on, so lambda replaces "
                    "this declared weight (it only supplies lam_init = %.4f). "
                    "The effective coefficient per view is lambda x "
                    "view_weights[view] and is printed every epoch."
                    % (t.recon_constraint.lam_init,)))
    else:
        out.append(("training.recon_constraint.*",
                    "disabled: reconstruction runs at the FIXED effective "
                    "coefficient %.4f for the `full` view, and every other "
                    "controller key is inert"
                    % (t.effective_recon_coefficient("full"),)))
    if t.nash.enabled and not t.nash.bypass:
        for nm in ("absolute", "pairwise", "discriminator"):
            hd = getattr(h, nm)
            if hd.enabled and hd.weight != 1.0:
                out.append(("heads.%s.weight" % nm,
                            "the bargain computes this task's weight, so the "
                            "declared %.3f is ignored while nash.enabled is "
                            "true" % (hd.weight,)))
    if t.nash.bypass:
        out.append(("training.nash.*",
                    "bypass=True: the losses are summed and no alpha is "
                    "computed"))
    if t.nash.enabled:
        # The ceiling is k - (k-1)*min_weight, NOT k: the other coordinates
        # have a floor too. This line used to print `min(max_weight, k)` =
        # 3.000 while `nash.alpha_ceiling` printed 2.750 for the same config.
        # Kept in sync by formula, because importing the function here would
        # pull torch into a module meant to be readable without it.
        n_play = len(t.nash.players)
        ceil = float(n_play) - float(n_play - 1) * float(t.nash.min_weight)
        if t.nash.max_weight >= ceil:
            out.append(("training.nash.max_weight",
                        "unreachable: the alphas sum to %d and the other %d "
                        "coordinates each sit at or above min_weight=%.3f, so "
                        "alpha_i <= %.3f. Effective ceiling %.3f"
                        % (n_play, n_play - 1, t.nash.min_weight, ceil,
                           min(t.nash.max_weight, ceil))))
    if t.nash.recompute_every > 1 and t.nash.alpha_ema <= 0.0:
        out.append(("training.nash.recompute_every",
                    "with no EMA the reused alphas are one batch old rather "
                    "than a smoothed estimate"))
    if m.fsq_cyfry_czynne and list(m.fsq_cyfry_czynne) == [
            fsq_grid(int(b))[0] for b in m.fsq_bits]:
        out.append(("model.fsq_cyfry_czynne",
                    "equal to the full digit count of every grid, so no "
                    "digit is frozen: this run measures the default "
                    "configuration under a new name"))
    if not h.absolute.enabled:
        out.append(("heads.absolute.*", "the absolute head is disabled"))
    elif not h.absolute.predict_sigma:
        out.append(("heads.absolute.sigma_weight",
                    "predict_sigma is false, so the measured-sigma term is "
                    "never built"))
    elif cfg.data.target != "mu_sigma":
        out.append(("heads.absolute.sigma_weight",
                    "data.target is %r, so no sigma column is loaded and the "
                    "second term has no target" % (cfg.data.target,)))
    if h.absolute.enabled and h.absolute.sigma_weight == 0.0 and \
            h.absolute.predict_sigma:
        out.append(("heads.absolute.predict_sigma",
                    "the sigma output is built and weighted 0.0: parameters "
                    "and a forward pass with no effect on the loss"))
    if h.absolute.enabled and h.absolute.tie_sigma_to_var:
        out.append(("heads.absolute.tie_sigma_to_var",
                    "tied: the separate sigma layer is not built, so its "
                    "initialisation and width are inert"))
    if h.absolute.enabled and h.absolute.beta_mu is not None and \
            h.absolute.beta_mu == h.absolute.beta:
        out.append(("heads.absolute.beta_mu",
                    "equals beta, so the mu path keeps the coefficient it "
                    "already had: this run measures the default under a new "
                    "name"))
    if h.absolute.clamp_mode != "hard":
        out.append(("heads.absolute.clamp_mode",
                    "soft mode never reaches the bound, so `lv_wall` measures "
                    "saturation rather than a frozen gradient -- read the "
                    "falsifier accordingly"))
    for nm in ("absolute", "pairwise", "discriminator"):
        hd = getattr(h, nm)
        if hd.enabled and hd.context_mode == "none" and cfg.context_dim():
            out.append(("heads.%s.context_center" % nm,
                        "context_mode is 'none', so the whole conditioning "
                        "vector (%d dims) does not reach this head"
                        % (cfg.context_dim(),)))
    if not h.pairwise.enabled or not h.pairwise.nonpromoter_pairs:
        out.append(("heads.pairwise.nonprom_weight",
                    "no non-promoter pairs are sampled, so their weight is "
                    "never used"))
    if m.fsq_entropy_weight == 0.0:
        out.append(("model.fsq_gain",
                    "the entropy term is off (weight 0.0), so the grid's "
                    "occupancy rests on this constant alone -- and calibrating "
                    "that constant failed four times, always by measuring it "
                    "in a different regime than the run"))
    if not m.band_attention:
        out.append(("model.band_width", "band attention is disabled"))
    # `training.riemannian` is inert only when NOTHING lives on the manifold,
    # and that is a property of the BUILT MODEL rather than of
    # `model.vq_learnable`. The flag-aware version of this entry asserted that
    # nothing did, while three probes found `genome.points` in the optimiser's
    # `manifold` group under a real `RiemannianAdam`. The entry now asks the
    # model, and with no model to ask it says nothing: "I cannot tell" and "it
    # is dead" are different claims.
    _n_manifold = _manifold_tensor_count(model)
    if _n_manifold == 0:
        out.append(("training.riemannian",
                    "the BUILT model puts NO tensor in the `manifold` "
                    "optimiser group (ladder.manifold_parameters() and "
                    "genome.manifold_parameters() are both empty), so a "
                    "Riemannian optimiser has nothing to optimise"))
    for name, ch in c.vector_channels().items():
        if not ch.enabled:
            out.append(("conditioning.%s.*" % name, "the channel is disabled"))
        elif ch.ablate != "none":
            out.append(("conditioning.%s.ablate" % name,
                        "set to %r: this run measures the ablation, not the "
                        "model" % (ch.ablate,)))
    if not c.species and c.genome.enabled:
        out.append(("conditioning.genome.dim",
                    "no species list is declared, so the embedding rows are "
                    "assigned from the data and a checkpoint cannot be "
                    "reloaded against a different data order"))

    # -- model-aware: a key with no receiver ---------------------------------
    if model is not None:
        names = _model_names(model)
        joined = " ".join(names).lower()
        for key, tokens in _CONSUMERS:
            enabled = _is_enabled(cfg, key)
            if not enabled:
                continue
            if not any(tok in joined for tok in tokens):
                out.append((key,
                            "no module or parameter name contains any of %r, "
                            "so nothing in the built model consumes it"
                            % (tokens,)))
        for pat in t.no_decay_patterns:
            if not any(pat in n for n in names):
                out.append(("training.no_decay_patterns:%s" % pat,
                            "matches no parameter name. The exemption is by "
                            "SUBSTRING, so this is either a stale pattern or a "
                            "renamed tensor now decaying -- and decay on the "
                            "hyperboloid points measured 1.8e-5 of drift over "
                            "34 epochs"))
        for pat in t.manifold_patterns:
            if not any(pat in n for n in names):
                out.append(("training.manifold_patterns:%s" % pat,
                            "matches no parameter name, so no tensor is routed "
                            "to the Riemannian optimiser by it"))
    return out


def _is_enabled(cfg: HyppeConfig, key: str) -> bool:
    """Resolve a dotted key to a truth value, for the consumer check."""
    node: Any = cfg
    for part in key.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    if isinstance(node, ChannelConfig):
        return bool(node.enabled)
    if dataclasses.is_dataclass(node):
        return bool(getattr(node, "enabled", True))
    return bool(node)


# Parameter groups
def parameter_groups(model: Any, cfg: HyppeConfig) -> List[Dict[str, Any]]:
    """Split parameters into decay / no-decay / manifold groups.

    Three measured rules are encoded here:
      * weight decay must not touch points on the hyperboloid. Measured drift:
        lambda 0.0762 over 34 epochs is 1.8e-5, which is four orders above the
        1e-16 that `assert_on_manifold` accepts.
      * weight decay must not touch branch gates: a decayed gate closes, and a
        closed multiplicative gate passes no gradient upward, so it also
        starves the branch that would reopen it. That killed four branches.
      * the match is by SUBSTRING of the parameter NAME, which makes the name a
        contract. `maska_wektor` was exempt and `wektor_maski` was not.

    If the model exposes `manifold_parameters()` that is used instead of the
    name patterns, and the returned groups say which path was taken -- a
    pattern-based split that silently disagrees with the model's own opinion is
    worse than either.
    """
    named = list(model.named_parameters())
    manifold_ids = set()
    source = "patterns"
    fn = getattr(model, "manifold_parameters", None)
    if callable(fn):
        manifold_ids = {id(p) for p in fn()}
        source = "model.manifold_parameters()"
    else:
        for n, p in named:
            if any(pat in n for pat in cfg.training.manifold_patterns):
                manifold_ids.add(id(p))

    decay, no_decay, manifold = [], [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        if id(p) in manifold_ids:
            manifold.append(p)
        elif any(pat in n for pat in cfg.training.no_decay_patterns):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"name": "decay", "params": decay,
         "weight_decay": cfg.training.weight_decay},
        {"name": "no_decay", "params": no_decay, "weight_decay": 0.0},
        {"name": "manifold", "params": manifold, "weight_decay": 0.0,
         "manifold": True, "source": source},
    ]
    return [g for g in groups if g["params"]]
