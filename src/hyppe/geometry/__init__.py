"""Hyperbolic geometry for HYPPE: the Lorentz model, and nothing else.

Everything here is arithmetic -- no layers, no losses, no quantisers, which
live in ``hyppe/model/``. The split keeps the arithmetic testable in
isolation: ``python -m hyppe.geometry.lorentz`` runs
:func:`~hyppe.geometry.lorentz.self_test`, which checks the identities every
module above depends on.

Curvature is justified by the two objects that are trees by construction, the
cone ladder and the 10-taxon species tree, and not by the sequences: Gromov
delta on 296 JASPAR motif profiles is 0.0754 against 0.0359 for random points
in R^16, so motif space is less tree-like than random Euclidean space.

``lorentz`` carries the detail, including the three traps that bite hardest:
``minkowski_dot`` defaults to ``keepdim=False`` and a wrong assumption
broadcasts into a plausible wrong tensor rather than raising;
``project_to_hyperboloid`` takes the spatial part and needs ``ambient=True``
for a full point; and weight decay on a manifold parameter is fatal and
silent, so :class:`~hyppe.geometry.lorentz.RiemannianAdam` refuses the group.
"""
from __future__ import annotations

from .lorentz import (  # noqa: F401
    DEFAULT_NO_DECAY_PATTERNS,
    MAX_TANGENT_NORM,
    RiemannianAdam,
    TreeEmbedding,
    assert_in_tangent,
    assert_on_manifold,
    balanced_tree_distances,
    embed_tree_distances,
    expmap,
    expmap0,
    frechet_mean,
    geodesic_distance,
    is_manifold_param,
    logmap,
    logmap0,
    lorentz_centroid,
    lorentz_distance,
    lorentz_sqdist,
    manifold_param_groups,
    manifold_report,
    mark_manifold,
    minkowski_dot,
    origin_like,
    parallel_transport,
    parallel_transport0,
    project_to_hyperboloid,
    project_to_tangent,
    radius,
    riemannian_grad,
    self_test,
)

__all__ = [
    "MAX_TANGENT_NORM",
    "DEFAULT_NO_DECAY_PATTERNS",
    "RiemannianAdam",
    "TreeEmbedding",
    "assert_in_tangent",
    "assert_on_manifold",
    "balanced_tree_distances",
    "embed_tree_distances",
    "expmap",
    "expmap0",
    "frechet_mean",
    "geodesic_distance",
    "is_manifold_param",
    "logmap",
    "logmap0",
    "lorentz_centroid",
    "lorentz_distance",
    "lorentz_sqdist",
    "manifold_param_groups",
    "manifold_report",
    "mark_manifold",
    "minkowski_dot",
    "origin_like",
    "parallel_transport",
    "parallel_transport0",
    "project_to_hyperboloid",
    "project_to_tangent",
    "radius",
    "riemannian_grad",
    "self_test",
]
