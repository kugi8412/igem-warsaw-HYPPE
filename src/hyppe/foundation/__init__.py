"""HYPPE as a foundation model: adapt it, and measure what adapting cost.

    zero_shot   score a sequence for a species, an experiment or a gene the
                model has never seen. This is the delivered mode: the ten
                designs go to the wet lab with no P1 GFP measurement behind
                them, so nothing may be fitted on the target.
    adapt       fine-tune on a subcategory, a strain or a DESeq2 contrast
                without destroying the pretrained representation.

Adaptation needs its own module rather than a flag because its failure is
silent. Fine-tuning a 2.3 M-parameter model on a few hundred rows of one
condition scores beautifully on that subcategory and loses the ranking the
model was selected for. `AdaptationReport` therefore always carries the
metric on the original selection set beside the metric on the new one, and
`ForgettingGuard` stops a run whose original metric has fallen further than
a declared budget.
"""
from __future__ import annotations

__all__ = ["adapt", "zero_shot"]
