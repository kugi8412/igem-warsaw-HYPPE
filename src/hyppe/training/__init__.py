"""Training machinery: the multi-task bargain and what is kept out of it.

The one structural decision this package encodes is which tasks play. The
prediction heads play; reconstruction and every regulariser enter as
subordinate terms with declared weights. A Nash bargain equalises utilities,
so it hands the largest share to the weakest gradient, and a saturated task
therefore wins it: measured at 2.31 of 3.0 while reconstruction sat at 0.9998
and prediction was pushed down to 0.39 / 0.31.

A subordinate term declares its whole multiplier chain, not just its weight.
The recorded misreading was 0.15 times a view weight of 0.40 = 0.060 against
0.924 under the bargain, a factor of 15.4, invisible in the log because each
factor was separately visible and their product was not.

Reconstruction is not a fixed weight either, since 0.060 is exactly where it
collapsed from 0.9998 to 0.5549 in one epoch. It enters as a constraint with
a Lagrange multiplier (`geco`), and the controller reports when it is pinned
to a bound, which is the state in which it has quietly become a constant.

`weighting` holds the ways task losses become one update, with scalarisation
as the baseline every gradient method is read against; `gates` separates a
method that is wired from one that is merely configured; `loop` runs it and
`batches` feeds it.
"""
from .geco import ConstraintController, ConstraintState
from .nash import NashMTL, SubordinateTerm
from .weighting import (
                        NashWeighting,
                        NormPriorWeighting,
                        Scalarisation,
                        TaskWeighting,
                        build_weighting,
                        gram_offdiagonal,
)

__all__ = ["NashMTL", "SubordinateTerm", "ConstraintController",
           "ConstraintState", "TaskWeighting", "Scalarisation",
           "NashWeighting", "NormPriorWeighting", "build_weighting",
           "gram_offdiagonal"]
