"""Wiring gates F1 / F2 / F3, run before every training run.

A balancing method that is configured but not wired reports exactly what a
working one reports: the losses are still summed, the metric still moves, the
log still prints alphas. A unit test of the solver does not detect that;
comparing the method's direction against the plain sum does.

    F1  identity. On two tasks whose gradients are equal by construction the
        method must return equal alphas and a direction equal to the sum.
    F2  constructed conflict. On two tasks with an antiparallel component a
        gradient method must return a direction different from the sum, an
        exact zero being the signature of a method that is not wired.
    F3  responsiveness. Zeroing a task's weight must change that task's share
        of the update.

Two ways these gates could be unsatisfiable by construction are closed. F2
against scalarisation would demand it stop being the sum, so there the gate is
inverted rather than skipped, a skipped gate reading like a pass. And with
equal norms the Nash fixed point is symmetric, so the obvious antiparallel
pair fails F2 on a correct method; `_conflict_problem` builds 1 : 2 at
cos = -0.6 instead, whose closed form alpha = (4/3, 2/3) F2b checks. The
problems are synthetic because a gate on the real model would be slow,
batch-dependent, and without a known answer to compare against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import torch

from .weighting import TaskWeighting, flat_grad, gram_offdiagonal, per_task_gradients

__all__ = ["GateResult", "GateReport", "run_wiring_gates",
           "orthogonality_gate", "ORTHOGONALITY_FLOOR"]

#: Below this the task gradients barely overlap, so the fixed point
#: degenerates to `alpha_i = 1/||g_i||` at K+1 backward passes. Not a failure
#: of the run, a statement that there is nothing to negotiate over; real
#: conflict starts above ~0.3.
ORTHOGONALITY_FLOOR = 0.1


@dataclass
class GateResult:
    name: str
    passed: bool
    measured: float
    expected: str
    detail: str = ""

    def format(self) -> str:
        return ("%-3s %-4s %-28s measured %.6e   %s"
                % (self.name, "PASS" if self.passed else "FAIL",
                   self.expected, self.measured, self.detail))


@dataclass
class GateReport:
    method: str
    results: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def exit_code(self) -> int:
        return 0 if self.passed else 3

    def format(self) -> str:
        head = "wiring gates for method %r: %s" % (
            self.method, "ALL PASS" if self.passed else "FAILED")
        return "\n".join([head] + ["  " + r.format() for r in self.results])


# Synthetic problems
class _TwoTaskProblem(object):
    """Two losses over one shared parameter, with prescribed gradients.

    The losses are linear in the parameter, so `d(L_i)/d(p)` is exactly the
    vector handed in and the closed-form alphas are checkable. A quadratic
    surrogate would make the gradients depend on the current point, and the
    gate would test the point instead of the method.
    """

    def __init__(self, g1: torch.Tensor, g2: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(torch.zeros_like(g1))
        self.g1, self.g2 = g1, g2

    def losses(self) -> Dict[str, torch.Tensor]:
        return {"task_a": (self.g1 * self.param).sum(),
                "task_b": (self.g2 * self.param).sum()}

    def params(self) -> List[torch.nn.Parameter]:
        return [self.param]


def _identity_problem(dim: int, seed: int) -> _TwoTaskProblem:
    g = torch.randn(dim, generator=torch.Generator().manual_seed(seed),
                    dtype=torch.float64)
    return _TwoTaskProblem(g, g.clone())


def _conflict_problem(dim: int, seed: int, cos: float = -0.6,
                      ratio: float = 2.0) -> _TwoTaskProblem:
    """||g1|| = 1, ||g2|| = `ratio`, cos(g1, g2) = `cos`.

    The asymmetry is load-bearing: with equal norms the bargain is symmetric
    and returns the sum, so F2 would fail on a correct method.
    """
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(dim, generator=gen, dtype=torch.float64)
    u = u / u.norm()
    w = torch.randn(dim, generator=gen, dtype=torch.float64)
    w = w - (w @ u) * u
    w = w / w.norm()
    v = cos * u + math.sqrt(max(1.0 - cos * cos, 0.0)) * w
    return _TwoTaskProblem(u, ratio * v)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||, with ||b|| = 0 handled as an absolute difference."""
    nb = float(b.norm())
    if nb <= 0.0:
        return float((a - b).norm())
    return float((a - b).norm() / nb)


def _sum_direction(problem: _TwoTaskProblem,
                   weights: Dict[str, float]) -> torch.Tensor:
    losses = problem.losses()
    total = None
    for n, w in weights.items():
        if abs(w) <= 0.0:
            continue
        term = flat_grad(w * losses[n], problem.params())
        total = term if total is None else total + term
    if total is None:
        return torch.zeros(problem.param.numel(), dtype=torch.float64)
    return total


# Gates
def run_wiring_gates(factory: Callable[[Dict[str, float],
                                        Sequence[torch.nn.Parameter]],
                                       TaskWeighting],
                     dim: int = 64, seed: int = 7898,
                     identity_tol: float = 1e-9,
                     conflict_tol: float = 1e-3,
                     share_tol: float = 1e-9) -> GateReport:
    """Build a fresh method per gate and test it on a synthetic problem.

    `factory(task_weights, params) -> TaskWeighting` must build the same class
    with the same settings the run will use; a gate on a differently
    configured object tests a method the run does not have.
    """
    probe = factory({"task_a": 1.0, "task_b": 1.0},
                    _identity_problem(dim, seed).params())
    report = GateReport(method=probe.name)
    grad_based = bool(probe.gradient_based)

    # F1: identity on equal gradients
    p1 = _identity_problem(dim, seed)
    w1 = {"task_a": 1.0, "task_b": 1.0}
    m1 = factory(dict(w1), p1.params())
    a1 = m1.alphas(p1.losses(), p1.params())
    d1 = m1.direction(p1.losses(), p1.params())
    s1 = _sum_direction(p1, w1)
    rel1 = _rel(d1, s1)
    spread = (max(a1.values()) - min(a1.values())) if a1 else float("nan")
    report.results.append(GateResult(
        name="F1", passed=(rel1 <= identity_tol and spread <= identity_tol),
        measured=rel1, expected="||dir - sum||/||sum|| <= %.0e" % identity_tol,
        detail="alphas %s (spread %.3e); equal gradients must give the sum"
               % ({k: round(v, 6) for k, v in a1.items()}, spread)))

    # F2: constructed conflict
    p2 = _conflict_problem(dim, seed + 1)
    w2 = {"task_a": 1.0, "task_b": 1.0}
    m2 = factory(dict(w2), p2.params())
    a2 = m2.alphas(p2.losses(), p2.params())
    d2 = m2.direction(p2.losses(), p2.params())
    s2 = _sum_direction(p2, w2)
    rel2 = _rel(d2, s2)
    if grad_based:
        ok2 = rel2 > conflict_tol
        exp2 = "||dir - sum||/||sum|| > %.0e" % conflict_tol
        det2 = ("alphas %s; closed form for this problem (cos -0.6, norms "
                "1:2) is a = (1.3333, 0.6667). EXACT ZERO here is the "
                "signature of a method that is not wired."
                % {k: round(v, 6) for k, v in a2.items()})
    else:
        # For a non-gradient method the identity with the sum is the
        # definition, so the gate is inverted rather than skipped.
        ok2 = rel2 <= identity_tol
        exp2 = "== sum exactly (scalarisation IS the sum)"
        det2 = ("alphas %s; a non-gradient method that moved away from the "
                "sum would be a plumbing bug, so the gate is inverted, not "
                "skipped" % {k: round(v, 6) for k, v in a2.items()})
    report.results.append(GateResult(name="F2", passed=ok2, measured=rel2,
                                     expected=exp2, detail=det2))

    # F2b: the direction differs from the sum for the right reason. A clipped
    # vector satisfies F2 as easily as a solved one -- on this problem the
    # solver once returned [0.1250, 1.8750], both coordinates on a bound at
    # residual 1.0393, and F2 passed at 1.48 -- so the closed form is checked
    # separately. With two players that fixed point is `a_i ~ 1/||g_i||` at
    # any cosine, so F2b tests that the solver reaches its own solution under
    # conflict, not that it trades one task off against another.
    expected_a = {"task_a": 4.0 / 3.0, "task_b": 2.0 / 3.0}
    dev = max(abs(a2.get(n, float("nan")) - v) for n, v in expected_a.items())
    # The box comes off the method under test: hard-coded bounds stop naming
    # a clipped vector as soon as a config moves them, which is the failure
    # F2b exists to separate from a solve. The literals are only the fallback
    # for a method that declares no box, such as scalarisation.
    _box = getattr(m2, "mtl", m2)
    lo_b = float(getattr(_box, "min_weight", 0.125))
    hi_b = float(getattr(_box, "max_weight", 8.0))
    on_bound = [n for n, v in a2.items()
                if abs(v - lo_b) < 1e-6 or abs(v - hi_b) < 1e-6]
    if grad_based:
        ok2b = dev <= 1e-3
        exp2b = "alpha == closed form (4/3, 2/3)"
        det2b = ("alphas %s; deviation %.4e%s"
                 % ({k: round(v, 6) for k, v in a2.items()}, dev,
                    ("; coordinates on a box bound: %r -- a clipped vector "
                     "passes F2 without a bargain having been struck"
                     % (on_bound,)) if on_bound else ""))
    else:
        ok2b = abs(dev - 1.0 / 3.0) < 1e-6
        exp2b = "alpha == 1.0 per task (the weights are applied, not chosen)"
        det2b = ("alphas %s: scalarisation does not choose weights, so the "
                 "deviation from the bargain's closed form is exactly |1 - "
                 "4/3| = 1/3 by construction"
                 % {k: round(v, 6) for k, v in a2.items()})
    report.results.append(GateResult(name="F2b", passed=ok2b, measured=dev,
                                     expected=exp2b, detail=det2b))

    # F3: zeroing a weight changes that task's share
    p3 = _conflict_problem(dim, seed + 2)
    losses3 = p3.losses()
    g_b = flat_grad(losses3["task_b"], p3.params())
    unit_b = g_b / g_b.norm().clamp_min(1e-30)
    m_full = factory({"task_a": 1.0, "task_b": 1.0}, p3.params())
    d_full = m_full.direction(p3.losses(), p3.params())
    m_zero = factory({"task_a": 1.0, "task_b": 0.0}, p3.params())
    d_zero = m_zero.direction(p3.losses(), p3.params())
    share_full = float(d_full @ unit_b)
    share_zero = float(d_zero @ unit_b)
    delta = abs(share_full - share_zero)
    report.results.append(GateResult(
        name="F3", passed=delta > share_tol, measured=delta,
        expected="|share(w=1) - share(w=0)| > %.0e" % share_tol,
        detail=("projection on task_b: %.6f -> %.6f. `weight 0` means DROP "
                "the task for both methods -- a zero-gradient Nash player "
                "would take the maximum share and trip the zero guard, so "
                "scaling to zero would make this gate unsatisfiable for it."
                % (share_full, share_zero))))
    return report


def orthogonality_gate(losses: Dict[str, torch.Tensor],
                       params: Sequence[torch.nn.Parameter],
                       floor: float = ORTHOGONALITY_FLOOR
                       ) -> Dict[str, object]:
    """Measure the task-gradient overlap on the real model, epoch 1.

    Returns the Gram statistics plus a verdict, and never blocks the run:
    "there is nothing to bargain over" is a result about the tasks, and a gate
    that refused to start would prevent the measurement establishing it.
    """
    names, G = per_task_gradients(losses, params)
    if G.numel() == 0 or len(names) < 2:
        return {"names": names, "verdict": "fewer than two live tasks: the "
                                           "overlap is undefined",
                "offdiag_max": float("nan")}
    out = gram_offdiagonal(G)
    out["names"] = names
    out["norms"] = {n: round(float(G[i].norm()), 6) for i, n in enumerate(names)}
    mx = out["offdiag_max"]
    if mx < floor:
        out["verdict"] = (
            "offdiag %.4f < %.2f: the task gradients are nearly orthogonal, "
            "so G G^T is nearly diagonal and a bargain degenerates to "
            "alpha_i = 1/||g_i|| -- GradNorm at K+1 backward passes. A "
            "gradient method pays here and buys nothing." % (mx, floor))
    elif mx < 0.3:
        out["verdict"] = (
            "offdiag %.4f in [%.2f, 0.30): weak overlap; a gradient method "
            "has little to resolve and the baseline should be expected to "
            "match it." % (mx, floor))
    else:
        out["verdict"] = (
            "offdiag %.4f >= 0.30 (most negative cosine %.4f): real overlap, "
            "and conflict if that cosine is negative -- a gradient method has "
            "something to do here." % (mx, out["cos_min"]))
    return out
