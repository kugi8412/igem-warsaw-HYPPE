"""Prediction heads: absolute strength, pairwise ranking, promoter realness.

These three are the players of the multi-task bargain in `training/nash.py`.
Reconstruction is deliberately not among them: a saturated task takes the
largest share of a Nash bargain precisely because its gradient is small, and
it was measured taking 2.31 of 3.0 while sitting at 0.9998. `NashMTL`
enforces that rule, and the heads here are the set it is enforced against.

Each head exposes the same three-part interface, so a training loop never has
to special-case one of them:

    forward(latent, context) -> typed output
    loss(output, targets, ...) -> (tensor, dict of scalars)
    report() -> dict with `ok` and `violations`, accumulated over an epoch
    reset() -> start the next epoch

`report` is the falsifier surface. Each head's thresholds are class
attributes, so a run that relaxes one has to say so in code rather than in a
log message.

The shared primitives (`LatentPooling`, `ConditionedTrunk`, `Reservoir`,
`StreamingMoments`, the rank statistics) live in `absolute.py`: this layer is
three modules by contract, and a fourth utility module would not be part of
the agreed interface.
"""
from .absolute import (
                       AbsoluteOutput,
                       AbsoluteStrengthHead,
                       ConditionedTrunk,
                       LatentPooling,
                       PoolingSlice,
                       Reservoir,
                       StreamingMoments,
                       as_features,
                       average_ranks,
                       masked_mean,
                       pearson,
                       sanitise,
                       spearman,
)
from .category import CategoryHead
from .discriminator import (
                       DINUCLEOTIDE_SHUFFLE,
                       NEGATIVE_KINDS,
                       DiscriminatorOutput,
                       PromoterDiscriminator,
                       gc_fraction,
                       roc_auc,
)
from .pair_compare import PairComparison
from .pairwise import PairOutput, PairwiseRankingHead

__all__ = [
    "AbsoluteOutput", "AbsoluteStrengthHead",
    "CategoryHead", "PairComparison", "PairOutput",
    "PairwiseRankingHead",
    "DiscriminatorOutput", "PromoterDiscriminator",
    "LatentPooling", "PoolingSlice", "ConditionedTrunk",
    "Reservoir", "StreamingMoments",
    "as_features", "masked_mean", "sanitise",
    "average_ranks", "pearson", "spearman", "roc_auc", "gc_fraction",
    "NEGATIVE_KINDS", "DINUCLEOTIDE_SHUFFLE",
]
