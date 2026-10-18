"""Invariants of the Lorentz model the ladder composes points in.

Each test states the property rather than a golden number, so it survives a
refactor and fails only when the geometry is actually wrong. The parallel
transport test carries a deliberate counter-example: an operator-precedence
slip that this repository has already paid for once produces code that runs,
looks right, and is not an isometry.
"""

from __future__ import annotations

import torch

from hyppe.geometry import lorentz as L

DIM = 9
N = 64
TOL = 1e-10


def _points(seed: int, scale: float = 0.9) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    space = torch.randn(N, DIM - 1, generator=g, dtype=torch.float64) * scale
    return L.project_to_hyperboloid(space)


def _tangent(p: torch.Tensor, seed: int, scale: float = 0.4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(N, DIM, generator=g, dtype=torch.float64) * scale
    return L.project_to_tangent(p, v)


def test_projection_lands_on_the_sheet():
    p = _points(0)
    assert (L.minkowski_dot(p, p) + 1.0).abs().max() < TOL


def test_tangent_projection_is_orthogonal():
    p = _points(1)
    v = _tangent(p, 2)
    assert L.minkowski_dot(p, v).abs().max() < TOL


def test_expmap_stays_on_the_sheet_and_logmap_inverts_it():
    p = _points(3)
    v = _tangent(p, 4)
    q = L.expmap(p, v)
    assert (L.minkowski_dot(q, q) + 1.0).abs().max() < 1e-9
    assert (L.logmap(p, q) - v).abs().max() < 1e-8


def test_distance_equals_the_length_of_the_step_that_made_it():
    p = _points(5)
    v = _tangent(p, 6)
    moved = L.geodesic_distance(p, L.expmap(p, v)).squeeze(-1)
    length = L.minkowski_dot(v, v).clamp_min(0).sqrt().squeeze(-1)
    assert (moved - length).abs().max() < 1e-8


def test_parallel_transport_is_an_isometry_into_the_target_tangent():
    p, q = _points(7), _points(8)
    v, w = _tangent(p, 9), _tangent(p, 10)
    tv, tw = L.parallel_transport(p, q, v), L.parallel_transport(p, q, w)
    assert L.minkowski_dot(q, tv).abs().max() < 1e-9
    assert (L.minkowski_dot(tv, tw) - L.minkowski_dot(v, w)).abs().max() < 1e-8


def test_the_precedence_slip_would_be_caught():
    """`-dot(p, q).clamp_min(1 + eps)` clamps the NEGATIVE inner product.

    It is below the floor for every distinct pair, so the clamp fires always
    and the transport silently stops being an isometry. Written correctly the
    negation is inside the parenthesis. This test exists so that the isometry
    test above cannot be passing for an accidental reason.
    """
    p, q = _points(11), _points(12)
    v, w = _tangent(p, 13), _tangent(p, 14)
    good = L.minkowski_dot(L.parallel_transport(p, q, v),
                           L.parallel_transport(p, q, w))

    alpha = -L.minkowski_dot(p, q, keepdim=True).clamp_min(1.0 + 1e-7)  # slip
    coef = L.minkowski_dot(q - alpha * p, v, keepdim=True) / (alpha + 1.0)
    bad_v = v + coef * (p + q)
    coef_w = L.minkowski_dot(q - alpha * p, w, keepdim=True) / (alpha + 1.0)
    bad_w = w + coef_w * (p + q)
    bad = L.minkowski_dot(bad_v, bad_w)

    reference = L.minkowski_dot(v, w)
    assert (good - reference).abs().max() < 1e-8
    assert (bad - reference).abs().max() > 1.0


def test_frechet_mean_is_not_the_ambient_arithmetic_mean():
    """The codebook EMA target must be geodesic.

    Averaging ambient coordinates and projecting is cheaper and wrong; on the
    outer shells the two differ by more than two units of distance. The test
    asserts only the direction of the inequality, which is what the update
    relies on.
    """
    base = _points(15)
    g = torch.Generator().manual_seed(16)
    spread = torch.randn(N, 16, DIM, generator=g, dtype=torch.float64) * 0.25
    pts = L.expmap(base.unsqueeze(1).expand(N, 16, DIM),
                   L.project_to_tangent(base.unsqueeze(1).expand(N, 16, DIM),
                                        spread))
    fm = L.frechet_mean(pts)
    naive = L.project_to_hyperboloid(pts.mean(dim=-2), ambient=True)

    def inertia(c):
        d = L.geodesic_distance(c.unsqueeze(1).expand_as(pts), pts)
        return d.squeeze(-1).pow(2).mean()

    assert (L.minkowski_dot(fm, fm) + 1.0).abs().max() < 1e-9
    assert inertia(fm) <= inertia(naive) + 1e-12


def test_riemannian_gradient_raises_the_metric():
    """Signature (-, +, ..., +) means the time component must flip.

    Without it the result is not the gradient of anything, yet it trains
    without error and converges somewhere else, so the only way this is ever
    noticed is a test that compares against the unraised projection.
    """
    p = _points(17)
    g = torch.Generator().manual_seed(18)
    grad = torch.randn(N, DIM, generator=g, dtype=torch.float64)
    rg = L.riemannian_grad(p, grad)
    assert L.minkowski_dot(p, rg).abs().max() < 1e-9
    assert (rg - L.project_to_tangent(p, grad)).abs().max() > 1e-3


def test_radius_is_the_distance_from_the_origin():
    p = _points(19)
    o = L.origin_like(p)
    assert (L.radius(p).squeeze(-1)
            - L.geodesic_distance(o, p).squeeze(-1)).abs().max() < 1e-8
