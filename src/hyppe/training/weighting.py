"""How the task losses become one update; scalarisation is the baseline.

`scalarise` is the default and is never switched off as a comparison: a
balancing method changes the effective task weights, and changing those alone
moves the metric, so "Nash gained +x" cannot be told apart from "the weights
Nash picked gained +x". The alphas a bargain chose in the parent project
(2.31 / 0.39 / 0.31) departed from the declared 1 / 1 / 1 by more than any
architectural change moved the metric, and it costs K+1 backward passes,
2.95x the step on this encoder.

All three classes share one surface: `alphas` is pure, so the wiring gates can
call it on synthetic problems; `apply` runs the single backward, which is why
the subordinate term is added inside it; and `task_weights` holds the
coefficients applied before the method sees a loss, where 0.0 drops a task
rather than scaling it, a zero-weight Nash player otherwise tripping the
zero-gradient guard. `normprior` is not a third bargain: with `G G^T` nearly
diagonal the Nash fixed point degenerates to `alpha_i = 1/||g_i||`, measured
here as offdiag 0.177 and falling, the norms 2.79 / 1.20 / 0.78 giving the
closed form 0.435 against the 0.43 the bargain reported.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .nash import NashMTL, SubordinateTerm, alpha_ceiling

__all__ = ["TaskWeighting", "Scalarisation", "NashWeighting",
           "NormPriorWeighting", "build_weighting", "flat_grad",
           "per_task_gradients", "gram_offdiagonal", "measure_step_cost"]


# Utilities
def flat_grad(loss: torch.Tensor, params: Sequence[torch.nn.Parameter],
              retain: bool = True) -> torch.Tensor:
    """d(loss)/d(params) as one float64 vector, unused entries as zeros.

    float64 because the gates difference these against each other, and F1's
    relative difference is ~1e-7 in float32: the noise floor of summation.
    """
    g = torch.autograd.grad(loss, list(params), retain_graph=retain,
                            allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                      for x, p in zip(g, params)]).double()


def per_task_gradients(losses: Mapping[str, torch.Tensor],
                       params: Sequence[torch.nn.Parameter]
                       ) -> Tuple[List[str], torch.Tensor]:
    """(names, G) with G of shape [K, P] in float64."""
    names = [n for n, v in sorted(losses.items())
             if v is not None and torch.is_tensor(v) and v.requires_grad]
    if not names:
        return [], torch.zeros((0, 0), dtype=torch.float64)
    rows = [flat_grad(losses[n], params) for n in names]
    return names, torch.stack(rows)


def gram_offdiagonal(G: torch.Tensor) -> Dict[str, float]:
    """Normalised Gram matrix of the task gradients: how much overlap exists.

    Below |offdiag| ~ 0.1 the gradients are nearly orthogonal and a balancing
    method is buying what is not for sale; above ~0.3 there is real conflict.
    A free function, since the measurement precedes the choice of method.
    Returns max and mean |offdiag|, the most negative entry (conflict lives in
    the negative tail, not the magnitude), and the per-pair cosines.
    """
    k = G.shape[0]
    if k < 2:
        return {"offdiag_max": float("nan"), "offdiag_mean": float("nan"),
                "cos_min": float("nan"), "n_players": k}
    d = G.norm(dim=1).clamp_min(1e-30)
    C = (G @ G.t()) / (d.unsqueeze(0) * d.unsqueeze(1))
    eye = torch.eye(k, dtype=C.dtype, device=C.device)
    off = (C - eye * C.diagonal())
    mask = ~eye.bool()
    vals = off[mask]
    return {"offdiag_max": float(vals.abs().max()),
            "offdiag_mean": float(vals.abs().mean()),
            "cos_min": float(vals.min()),
            "n_players": k,
            "cos": [[round(float(C[i, j]), 4) for j in range(k)]
                    for i in range(k)]}


def measure_step_cost(step_fn: Callable[[], None], n: int = 5,
                      warmup: int = 2) -> float:
    """Seconds per call, after a warm-up. Prices a method, never tunes one:
    the 2.95x above was measured this way on this encoder and has to be
    re-measured whenever the encoder changes."""
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / float(n)


# Methods
class TaskWeighting(object):
    """Common surface. `name` appears in the log and in the checkpoint."""

    name = "base"
    #: True when the method needs per-task gradients, i.e. K+1 backward
    #: passes. The orthogonality gate is only worth reading for these.
    gradient_based = False

    def __init__(self, task_weights: Mapping[str, float]) -> None:
        self.task_weights = {str(k): float(v) for k, v in task_weights.items()}
        if not self.task_weights:
            raise ValueError("no tasks declared")

    # The pure part, which the gates exercise
    def active(self, losses: Mapping[str, torch.Tensor]) -> List[str]:
        """Tasks that are present, differentiable and carry a non-zero weight.

        A zero weight drops the task rather than scaling it -- the same thing
        for scalarisation, not for a bargain, where a zero-gradient player
        takes the maximum share and trips `NashMTL`'s guard.
        """
        return [n for n, v in sorted(losses.items())
                if v is not None and torch.is_tensor(v) and v.requires_grad
                and abs(self.task_weights.get(n, 0.0)) > 0.0]

    def alphas(self, losses: Mapping[str, torch.Tensor],
               params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
        raise NotImplementedError

    def direction(self, losses: Mapping[str, torch.Tensor],
                  params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        """The update direction this method would take, as a flat vector.

        Built from the per-task gradients and the method's own alphas, never
        from `.grad`, which holds whatever the previous step left there.
        """
        names = self.active(losses)
        if not names:
            return torch.zeros(sum(p.numel() for p in params),
                               dtype=torch.float64)
        a = self.alphas(losses, params)
        rows = [flat_grad(self.task_weights[n] * losses[n], params)
                for n in names]
        G = torch.stack(rows)
        w = torch.tensor([a[n] for n in names], dtype=torch.float64,
                         device=G.device)
        return (w.unsqueeze(1) * G).sum(0)

    # The impure part
    def apply(self, losses: Mapping[str, torch.Tensor],
              subordinate: Optional[torch.Tensor] = None,
              recompute: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    # The closure contract
    def components(self) -> Dict[str, float]:
        """Exactly the player contributions the last `apply` summed.

        The products, such that `sum(components()) + subordinate == loss`, an
        identity the loop asserts. The methods scale differently, so a caller
        multiplying alpha by loss itself would double-count for one of them
        and balance the log only on the baseline.
        """
        return dict(getattr(self, "_last_components", {}))

    def subordinate_included(self) -> bool:
        """Whether the last `apply` actually added the subordinate term.

        Both skip it when it does not require grad, and a term computed,
        reported and not optimised is a cost with no effect and no trace.
        """
        return bool(getattr(self, "_last_sub_included", False))

    def new_epoch(self) -> None:
        pass

    def report(self) -> Dict[str, object]:
        return {"method": self.name}

    def header(self, n_players: Optional[int] = None) -> str:
        return "%s: weights %s" % (
            self.name, " ".join("%s=%.3f" % (k, v)
                                for k, v in sorted(self.task_weights.items())))


class Scalarisation(TaskWeighting):
    """total = sum_i w_i * L_i. The baseline, and the definition of "the sum".

    Nothing converges or clips, so the only observables are the weights --
    which is the point. `alphas` returns 1.0 per task, `task_weights` having
    applied the weight already; keeping them apart lets F3 zero one.
    """

    name = "scalarise"
    gradient_based = False

    def alphas(self, losses: Mapping[str, torch.Tensor],
               params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
        return {n: 1.0 for n in self.active(losses)}

    def apply(self, losses: Mapping[str, torch.Tensor],
              subordinate: Optional[torch.Tensor] = None,
              recompute: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        names = self.active(losses)
        total: Optional[torch.Tensor] = None
        parts: List[torch.Tensor] = []
        for n in names:
            term = self.task_weights[n] * losses[n]
            parts.append(term.detach().reshape(()))
            total = term if total is None else total + term
        included = (subordinate is not None and torch.is_tensor(subordinate)
                    and subordinate.requires_grad)
        if included:
            total = subordinate if total is None else total + subordinate
        # One synchronisation for the whole decomposition: a `float()` per
        # player is a device sync each, and diagnostics once ate 70 % of a step.
        vals = (torch.stack(parts).tolist() if parts else [])
        self._last_components = {n: float(v) for n, v in zip(names, vals)}
        self._last_sub_included = bool(included)
        if total is None:
            raise RuntimeError(
                "nothing to optimise: every task weight is zero and there is "
                "no subordinate term. An empty backward would leave the "
                "parameters untouched and the loop would report a step.")
        total.backward()
        self._n_steps = getattr(self, "_n_steps", 0) + 1
        return total.detach(), {n: self.task_weights[n] for n in names}

    def new_epoch(self) -> None:
        self._n_steps = 0

    def report(self) -> Dict[str, object]:
        return {"method": self.name, "n_steps": getattr(self, "_n_steps", 0),
                "alpha": dict(self.task_weights), "ok": True,
                "violations": []}

    def format_report(self) -> str:
        return "scalarise[%d steps] %s" % (
            getattr(self, "_n_steps", 0),
            " ".join("%s=%.3f" % (k, v)
                     for k, v in sorted(self.task_weights.items())))


class NashWeighting(TaskWeighting):
    """Nash bargaining, behind the flag. Wraps `NashMTL` without hiding it.

    The wrapper adds one thing: `alphas` as a pure function, so a gate can ask
    what it would do with these gradients without a backward pass and without
    advancing the EMA, which would change the run the gate is checking.
    """

    name = "nash"
    gradient_based = True

    def __init__(self, task_weights: Mapping[str, float],
                 params: Sequence[torch.nn.Parameter], **kwargs) -> None:
        super().__init__(task_weights)
        kwargs.setdefault("allowed_players", tuple(sorted(self.task_weights)))
        self.mtl = NashMTL(params, **kwargs)
        self.params = self.mtl.params

    def alphas(self, losses: Mapping[str, torch.Tensor],
               params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
        names = self.active(losses)
        if len(names) < 2:
            return {n: 1.0 for n in names}
        rows = [flat_grad(self.task_weights[n] * losses[n], params)
                for n in names]
        G = torch.stack(rows)
        a, _res, _it, _clip, _ok = self.mtl._alpha(G @ G.t())
        return {n: float(a[i]) for i, n in enumerate(names)}

    def apply(self, losses: Mapping[str, torch.Tensor],
              subordinate: Optional[torch.Tensor] = None,
              recompute: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        scaled = {n: self.task_weights[n] * losses[n]
                  for n in self.active(losses)}
        total = self.mtl.step(scaled, subordinate=subordinate,
                              recompute=recompute)
        # `NashMTL.step` sums `alpha_n * scaled_n`, so a contribution is the
        # solved alpha times the already weighted loss. Taken from the solver:
        # a second multiplication here is how a decomposition drifts.
        self._last_components = dict(self.mtl.last_components)
        self._last_sub_included = bool(self.mtl.last_sub_included)
        return total, dict(self.mtl.last)

    def new_epoch(self) -> None:
        self.mtl.new_epoch()

    def report(self) -> Dict[str, object]:
        r = self.mtl.report()
        r["method"] = self.name
        return r

    def format_report(self) -> str:
        return self.mtl.format_report()

    def header(self, n_players: Optional[int] = None) -> str:
        k = n_players or len(self.task_weights)
        return ("%s | declared task weights %s | alpha ceiling %.3f"
                % (self.mtl.header(k),
                   " ".join("%s=%.3f" % (a, b)
                            for a, b in sorted(self.task_weights.items())),
                   alpha_ceiling(k, self.mtl.min_weight)))


class NormPriorWeighting(TaskWeighting):
    """alpha_i = K * (w_i / ||g_i||) / sum_j (w_j / ||g_j||). Explicit priors.

    The alpha cancels the norm it divided by, so the update receives `w_i`
    times the unit gradient of task i: `w` is how many parts of each task go
    into the step. Nash derives the shares from the norms alone and hands the
    largest to whichever task is currently easiest, with no config key saying
    otherwise.

    There is no fixed point, iteration, box or clip, so `clip`, `nzb`, `res`
    and `iter` are absent from the report rather than printed at 0.000;
    `offdiag` is reported but is never a violation, a low one being the reason
    to use this method. Nothing bounds alpha either. What is bounded is the
    contribution, `K w_i / sum_j (w_j/||g_j||)`, which does not contain
    `||g_i||` at all -- so a task whose gradient dies inflates the denominator
    and shrinks every contribution, silently cutting the whole step's
    effective learning rate. `dir_norm` shows that before `_guard_zero` fires.
    """

    name = "normprior"
    gradient_based = True

    def __init__(self, task_weights: Mapping[str, float],
                 params: Sequence[torch.nn.Parameter],
                 priors: Optional[Mapping[str, float]] = None,
                 refresh_every: int = 1,
                 eps: float = 1e-12,
                 subordinate_terms: Sequence[SubordinateTerm] = ()) -> None:
        super().__init__(task_weights)
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("no shared parameters with requires_grad")
        self.priors = {str(k): float(v)
                       for k, v in dict(priors or {}).items()}
        bad = sorted(k for k, v in self.priors.items() if not v > 0.0)
        if bad:
            raise ValueError(
                "prior(s) %r are <= 0. A prior of 0 leaves the task a PLAYER "
                "-- the alphas still sum to the player count -- while it "
                "contributes nothing, i.e. two different meanings for "
                "'dropped'. Dropping a task is `task_weights[name] = 0.0`, "
                "which is the one operation all three methods agree on and "
                "the one gate F3 perturbs." % (bad,))
        self.refresh_every = int(refresh_every)
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be >= 1, got %r"
                             % (refresh_every,))
        self.eps = float(eps)
        if not self.eps > 0.0:
            raise ValueError("eps must be > 0: it is both the division floor "
                             "and the zero-gradient threshold")
        self.subordinate_terms = tuple(subordinate_terms)
        self.n_params = sum(p.numel() for p in self.params)
        self.last: Dict[str, float] = {}
        self.last_norms: Dict[str, float] = {}
        self._cached: Dict[str, float] = {}
        self._cached_names: Tuple[str, ...] = ()
        #: Initialised AT the period so the first `apply` refreshes.
        self._since_refresh = self.refresh_every
        self._reset_accumulator()

    # Priors
    def prior(self, name: str) -> float:
        """The declared prior, defaulting to 1.0 for an undeclared name.

        `run_wiring_gates` builds this class on synthetic tasks, so refusing
        an undeclared name would make it ungateable. That every real player
        has a prior is enforced by `NormPriorConfig.validate`.
        """
        return float(self.priors.get(str(name), 1.0))

    # Accumulator
    def _reset_accumulator(self) -> None:
        self._acc: Dict[str, List[float]] = {}
        self._n_steps = 0
        #: Steps on which the gradients were actually recomputed. Separate
        #: from `_n_steps`: a rate over every optimiser step is diluted by
        #: 1/refresh_every, so the alarm quietens as measurement rarefies.
        self._n_refresh = 0
        self._sum: List[float] = []
        self._offdiag: List[float] = []
        self._dir: List[float] = []

    def new_epoch(self) -> None:
        self._reset_accumulator()
        # Force a refresh on the first step: reusing alphas measured before an
        # evaluation pass carries a number from one regime into another.
        self._since_refresh = self.refresh_every

    @staticmethod
    def _median(v: Sequence[float]) -> float:
        if not v:
            return float("nan")
        s = sorted(v)
        return float(s[len(s) // 2])

    # The rule
    def _guard_zero(self, names: Sequence[str],
                    norms: Sequence[float]) -> None:
        """Same guard as `NashMTL`, same threshold, different reason.

        Under `1/||g||` a dead task does not take the largest share; it
        inflates the denominator, so the whole update collapses while the
        alphas still sum to k and the log looks healthy.
        """
        dead = [n for n, d in zip(names, norms) if float(d) < self.eps]
        if dead:
            raise RuntimeError(
                "NormPrior: task(s) %r have a zero gradient on the shared "
                "parameters (norm < %.1e). Their contribution is w_i times a "
                "UNIT gradient that does not exist; what actually happens is "
                "that sum_j w_j/||g_j|| explodes and every other task's step "
                "shrinks toward zero while sum(alpha) stays exactly at the "
                "player count. Check that the latent is not detached on their "
                "path and that `params` really is the shared set."
                % (dead, self.eps))

    def _shares(self, names: Sequence[str],
                norms: Sequence[float]) -> List[float]:
        """The closed form, as pure float arithmetic so a gate can check it
        against a hand-computed number rather than another implementation of
        the same mistake."""
        k = float(len(names))
        inv = [self.prior(n) / max(float(d), self.eps)
               for n, d in zip(names, norms)]
        s = float(sum(inv))
        if not s > 0.0:
            raise RuntimeError("normprior: sum of w_i/||g_i|| is %r" % (s,))
        return [k * x / s for x in inv]

    # The pure part, which the gates exercise
    def alphas(self, losses: Mapping[str, torch.Tensor],
               params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
        """Pure: no backward, no cache read, no cache write.

        It ignores `refresh_every` deliberately -- returning cached state
        would make every gate a test of history rather than of the rule.
        """
        names = self.active(losses)
        if len(names) < 2:
            return {n: 1.0 for n in names}
        rows = [flat_grad(self.task_weights[n] * losses[n], params)
                for n in names]
        G = torch.stack(rows)
        norms = [float(x) for x in G.norm(dim=1)]
        self._guard_zero(names, norms)
        return dict(zip(names, self._shares(names, norms)))

    # The impure part
    def apply(self, losses: Mapping[str, torch.Tensor],
              subordinate: Optional[torch.Tensor] = None,
              recompute: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One `backward`, on the weighted sum. The per-task gradients go
        through `torch.autograd.grad`, which does not touch `.grad`, so the
        private head parameters keep the gradient of their own task.

        `recompute=False` and `refresh_every > 1` are one permission from two
        sides: the caller says whether a refresh is allowed, this object
        whether one is due. A roster change overrides both, reused alphas
        otherwise handing one task another's share.
        """
        names = self.active(losses)
        roster_changed = tuple(names) != self._cached_names
        due = self._since_refresh >= self.refresh_every
        do_refresh = bool(names) and (roster_changed
                                      or (bool(recompute) and due))

        if do_refresh:
            rows = [flat_grad(self.task_weights[n] * losses[n], self.params)
                    for n in names]
            G = torch.stack(rows)
            norms = [float(x) for x in G.norm(dim=1)]
            self._guard_zero(names, norms)
            shares = self._shares(names, norms)
            w = torch.tensor(shares, dtype=G.dtype, device=G.device)
            self._dir.append(float((w.unsqueeze(1) * G).sum(0).norm()))
            self._offdiag.append(
                float(gram_offdiagonal(G)["offdiag_max"]) if len(names) > 1
                else float("nan"))
            del G, rows
            self.last_norms = {n: float(d) for n, d in zip(names, norms)}
            self._cached = dict(zip(names, shares))
            self._cached_names = tuple(names)
            self._n_refresh += 1
            self._since_refresh = 0

        alpha = [float(self._cached.get(n, 1.0)) for n in names]

        total: Optional[torch.Tensor] = None
        parts: List[torch.Tensor] = []
        for a, n in zip(alpha, names):
            term = a * (self.task_weights[n] * losses[n])
            parts.append(term.detach().reshape(()))
            total = term if total is None else total + term
        included = (subordinate is not None and torch.is_tensor(subordinate)
                    and subordinate.requires_grad)
        if included:
            total = subordinate if total is None else total + subordinate
        # One synchronisation: a `float()` per player is a device sync each.
        vals = (torch.stack(parts).tolist() if parts else [])
        self._last_components = {n: float(v) for n, v in zip(names, vals)}
        self._last_sub_included = bool(included)
        if total is None:
            raise RuntimeError(
                "nothing to optimise: every task weight is zero and there is "
                "no subordinate term. An empty backward would leave the "
                "parameters untouched and the loop would report a step.")
        total.backward()

        self.last = dict(zip(names, alpha))
        self._n_steps += 1
        self._since_refresh += 1
        self._sum.append(float(sum(alpha)))
        for n, a in zip(names, alpha):
            self._acc.setdefault(n, []).append(float(a))
        return total.detach(), dict(self.last)

    # Falsifiers
    def report(self) -> Dict[str, object]:
        """Per-epoch aggregates. The one threshold that can fire is the
        invariant `sum(alpha) == n_players`, without which the effective
        learning rate drifts; `offdiag` is reported and is a violation in
        neither direction, a low one being what motivates this method."""
        out: Dict[str, object] = {"method": self.name,
                                  "n_steps": self._n_steps}
        if not self._n_steps:
            out.update({"ok": False, "violations": ["no steps recorded"]})
            return out
        alphas = {n: self._median(v) for n, v in sorted(self._acc.items())}
        k = float(len(alphas))
        out.update({
            "alpha": alphas,
            "n_players": len(alphas),
            "priors": {n: self.prior(n) for n in alphas},
            "norms": dict(self.last_norms),
            "refresh_every": self.refresh_every,
            "n_refresh": self._n_refresh,
            "refresh_duty": round(self._n_refresh / float(self._n_steps), 4),
            "sum_alpha": self._median(self._sum),
            "offdiag": self._median(self._offdiag),
            "dir_norm": self._median(self._dir),
            "eps": self.eps,
            "subordinate": {t.name: t.effective()
                            for t in self.subordinate_terms},
        })
        bad: List[str] = []
        sa = out["sum_alpha"]
        if sa == sa and abs(float(sa) - k) > 1e-3:
            bad.append("sum_alpha %.4f != %d players: the effective learning "
                       "rate drifts between steps" % (float(sa), int(k)))
        # A prior for a name that never played is an inert key: reported
        # separately, as in `NashMTL`, since it says nothing about this run.
        unused = sorted(set(self.priors) - set(alphas))
        out["inert"] = ["training.normprior.priors.%s" % (n,) for n in unused]
        out["violations"] = bad
        out["ok"] = not bad
        return out

    def format_report(self) -> str:
        r = self.report()
        if not r.get("n_steps"):
            return "normprior: no steps"
        alphas = " ".join("%s=%.2f" % (n, v)
                          for n, v in sorted(r["alpha"].items()))
        norms = " ".join("%s=%.3f" % (n, v)
                         for n, v in sorted(r["norms"].items()))
        priors = " ".join("%s=%.3f" % (n, v)
                          for n, v in sorted(r["priors"].items()))
        return ("normprior[%d steps / %d refreshed every %d] %s | norms %s "
                "| priors %s | sum_alpha %.4f | dir_norm %.4f | offdiag %.3f "
                "| %s"
                % (r["n_steps"], r["n_refresh"], r["refresh_every"], alphas,
                   norms, priors, r["sum_alpha"], r["dir_norm"], r["offdiag"],
                   "OK" if r["ok"] else "VIOLATIONS: " + " ; ".join(
                       r["violations"])))

    def header(self, n_players: Optional[int] = None) -> str:
        k = int(n_players or len(self.task_weights))
        cost = ("%d backward passes on a refresh, 1 in between"
                % (k + 1,))
        sub = ""
        if self.subordinate_terms:
            sub = " | subordinate: " + " ; ".join(
                t.describe() for t in self.subordinate_terms)
        return ("NormPrior: %d shared tensors / %d weights | priors %s | "
                "refresh every %d (%s) | eps %.1e | declared task weights %s"
                "%s"
                % (len(self.params), self.n_params,
                   " ".join("%s=%.3f" % (n, self.prior(n))
                            for n in sorted(self.task_weights)),
                   self.refresh_every, cost, self.eps,
                   " ".join("%s=%.3f" % (a, b)
                            for a, b in sorted(self.task_weights.items())),
                   sub))


# Factory
def build_weighting(method: str, task_weights: Mapping[str, float],
                    params: Sequence[torch.nn.Parameter],
                    nash_kwargs: Optional[Mapping[str, object]] = None,
                    subordinate_terms: Sequence[SubordinateTerm] = (),
                    normprior_kwargs: Optional[Mapping[str, object]] = None
                    ) -> TaskWeighting:
    """`method` in {"scalarise", "nash", "normprior"}. Unknown names raise.

    An unknown method falling back to scalarisation would be the worst version
    of this: the run would carry the method's name, produce the baseline's
    numbers and say nothing. Hence also the refusal of foreign kwargs.
    """
    m = str(method).lower()
    if m in ("scalarise", "scalarize", "sum", "linear"):
        if nash_kwargs or normprior_kwargs:
            raise ValueError(
                "%s keys were given with method=%r. They would have no "
                "receiver, and a configured-but-inert key is how this project "
                "has repeatedly run the default while believing otherwise."
                % ("nash_*" if nash_kwargs else "normprior_*", method))
        return Scalarisation(task_weights)
    if m == "nash":
        if normprior_kwargs:
            raise ValueError(
                "normprior_* keys were given with method=%r: they have no "
                "receiver in the bargain." % (method,))
        kw = dict(nash_kwargs or {})
        kw["subordinate_terms"] = tuple(subordinate_terms)
        return NashWeighting(task_weights, params, **kw)
    if m == "normprior":
        if nash_kwargs:
            raise ValueError(
                "nash_* keys were given with method=%r: `normprior` has no "
                "box, no EMA and no solver, so every one of them would be "
                "inert." % (method,))
        kw = dict(normprior_kwargs or {})
        kw["subordinate_terms"] = tuple(subordinate_terms)
        return NormPriorWeighting(task_weights, params, **kw)
    raise ValueError("unknown weighting method %r; known: scalarise, nash, "
                     "normprior" % (method,))
