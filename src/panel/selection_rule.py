"""The epoch selection rule, written before it is computed.

A rule chosen after seeing which epoch it favours is not a rule, so this file
is committed first and `epoch_selection` applies it afterwards, unchanged.

Not `val_scc` alone, which is what the run's own best.pt uses: the clustered
SE of val SCC, resampling the 299 val orthogroups rather than rows, is 0.024
against 0.0050 per row, so per-row error bars understate by 4.8x and make
every epoch difference look significant. Worse, val and the independent spy
split disagree about epoch order -- rho 0.587, a reversal at ep10/ep11, and on
an earlier run val put ep4 first where spy put it fifth of seven (best epoch
in 50.3 % of val resamples and 0.1 % of spy resamples).

Not spy alone either: spy is 300 orthogroups of contrasts built for this task,
precise about the target and narrow about everything else, and selecting on it
consumes its independence -- after this choice only `test` is untouched. So
val gives breadth, spy gives precision on the target, and the heads are
weighted by what the pipeline uses rather than by taste. `pks1_sep` is the one
term aimed straight at the goal: separation of the design target from its own
dinucleotide shuffles is weak (AUC 0.648-0.980, median 0.824, against tef1's
0.914-1.000) and improves with epoch, not significantly until ep017, and a
model that cannot tell pks1 from noise cannot gate candidates resembling it.

Every metric is converted to its own noise scale before weighting: raw ranges
differ (scc spans ~0.05 across epochs, rec_full ~0.02), so weighting raw
values would hand the decision to whichever metric has the widest range. Each
is z-scored by its clustered standard error over orthogroups, so a weight of
0.25 means a quarter of the decision, in units of what can be resolved.
"""
from __future__ import annotations

WEIGHTS = {
    "spy_scc": 0.25,      # precision on the contrasts built for this task
    "val_scc": 0.15,      # breadth, so we do not tune to 300 orthogroups
    "pair_acc": 0.20,     # the MID window is an ordering statement
    "disc_auc": 0.20,     # the realness gate runs on all ten
    "rec_full": 0.10,     # every candidate is decoded before scoring
    "pks1_sep": 0.10,     # can the model recognise the design target
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12

ZRODLO = {
    "spy_scc": "spy", "val_scc": "val", "pair_acc": "spy",
    "disc_auc": "spy", "rec_full": "spy", "pks1_sep": "both constructs, mean",
}

PASMO_SE = 1.0
#: The tie-break chain, in the order `epoch_selection.stage2` applies it:
#: longest plateau run, then distance to the run's centre, then `pks1_sep`,
#: then the later epoch. It is recorded into the result JSON as `tie_break`
#: and printed at stage 4, so it has to name the steps that actually run --
#: an earlier version listed a `pks1_umiejscowienie` step that nothing
#: implements.
ROZSTRZYGNIECIE = ("plateau_center", "pks1_sep", "later_epoch")
DOPUSZCZENIE = {
    "rec_full_min": 0.93,
    "tef1_above_pks1": True,
}
OGRANICZENIA = (
    "resolution ~1 SE (~0.02 scc)",
    "consumes spy independence",
    "pks1_sep must be measured per epoch, never interpolated",
)
