"""Nash-MTL: task weights from a Nash bargain over the shared parameters.

Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022: gradients
are summed with weights `alpha` satisfying `(G G^T alpha)_i * alpha_i = 1`,
invariant to the scale of the individual losses -- which is what makes it both
attractive and dangerous.

Only prediction tasks may bargain, because equalising utilities hands the
weakest gradient the largest alpha; `_check_players` carries the measurement.
A task kept out is not thereby free: reconstruction was being held at 0.9998
by that weight, and dropping it to a declared 0.15 collapsed it to 0.5549 in
one epoch. What decides is the effective coefficient -- the declared weight
times every factor down to the loss, 0.924 as a player against 0.060 as a
subordinate -- which is what `SubordinateTerm` declares.

`params` must be what the tasks really compete for: with the latent detached
the shared set can be empty, `G G^T` comes out diagonal and the fixed point
degenerates to GradNorm at K+1 backward passes a step.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

#: Fallback blacklist; it once matched English only, so `"rekonstrukcja"` walked
#: into a bargain. The load-bearing contract is the allowlist. Imported, not
#: redefined: `NashConfig.validate` screens the same names one layer up.
from ..config import FORBIDDEN_PLAYER_FRAGMENTS

__all__ = ["SubordinateTerm", "NashMTL", "FORBIDDEN_PLAYER_FRAGMENTS",
           "alpha_ceiling"]


def alpha_ceiling(n_players: int, min_weight: float) -> float:
    """The largest alpha a single player can reach, given the box floor.

    `k - (k-1) * min_weight`, i.e. 2.750 at k = 3 and min_weight = 0.125 --
    not 3.000, which forgets that the other players have a floor too.
    """
    k = int(n_players)
    if k < 1:
        raise ValueError("n_players must be >= 1")
    return float(k) - float(k - 1) * float(min_weight)


@dataclass
class SubordinateTerm:
    """A loss term that is not a player, with its full multiplier chain.

    `multipliers` maps a branch name to the factor between the declared weight
    and the loss; `effective()` returns the product, the only number comparable
    with an alpha.
    """

    name: str
    weight: float
    multipliers: Dict[str, float] = field(default_factory=dict)

    def effective(self) -> Dict[str, float]:
        if not self.multipliers:
            return {"": float(self.weight)}
        return {k: float(self.weight) * float(v)
                for k, v in self.multipliers.items()}

    def describe(self) -> str:
        eff = self.effective()
        body = " ".join("%s=%.4f" % (k or "flat", v)
                        for k, v in sorted(eff.items()))
        return "%s: declared %.4f -> effective %s" % (self.name, self.weight,
                                                      body)


class NashMTL(object):
    """Nash bargaining weights over a declared set of shared parameters.

    `params` are the shared parameters, typically the encoder up to the
    quantiser; "all parameters" would pull in private head weights and measure
    something else. `alpha_ema` smooths the per-batch ratio of noisy norms,
    unsmoothed flickering between 0.33 and 3.0 under a stable average.
    `bypass` sums the losses instead; verify it on gradients after one step,
    not on a metric after an epoch, since summation order differs by ~1e-7 and
    the quantiser's argmax separates the trajectories chaotically.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter],
                 n_iter: int = 20, min_weight: float = 0.125,
                 max_weight: float = 8.0, alpha_ema: float = 0.9,
                 bypass: bool = False, eps: float = 1e-8,
                 zero_threshold: float = 1e-12,
                 forbid_players: Sequence[str] = FORBIDDEN_PLAYER_FRAGMENTS,
                 allowed_players: Optional[Sequence[str]] = None,
                 allow_saturated_players: bool = False,
                 subordinate_terms: Sequence[SubordinateTerm] = (),
                 clip_threshold: float = 0.2,
                 nzb_threshold: float = 0.05,
                 project_rounds: int = 16) -> None:
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("no shared parameters with requires_grad")
        if not (min_weight <= 1.0 <= max_weight):
            # The mean of k alphas summing to k is exactly 1.
            raise ValueError(
                "the box [%.3f, %.3f] does not contain 1.0, so no alpha can "
                "both stay in it and sum to the number of players"
                % (min_weight, max_weight))
        self.n_iter = int(n_iter)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.alpha_ema = float(alpha_ema)
        self.bypass = bool(bypass)
        self.eps = float(eps)
        self.zero_threshold = float(zero_threshold)
        self.forbid_players = tuple(str(s).lower() for s in forbid_players)
        self.allowed_players = (None if allowed_players is None
                                else tuple(str(s) for s in allowed_players))
        self.allow_saturated_players = bool(allow_saturated_players)
        self.subordinate_terms = tuple(subordinate_terms)
        self.clip_threshold = float(clip_threshold)
        self.nzb_threshold = float(nzb_threshold)
        self.project_rounds = int(project_rounds)
        self.n_params = sum(p.numel() for p in self.params)
        #: Smoothed alphas, keyed by name: the roster varies between steps.
        self._ema: Dict[str, float] = {}
        self.last: Dict[str, float] = {}
        self.last_norms: Dict[str, float] = {}
        #: `alpha_n * loss_n` for the last step -- the addends, not the
        #: factors. `TrainLoop` asserts that these plus the subordinate terms
        #: equal `loss`, so they are recorded where the sum is built.
        self.last_components: Dict[str, float] = {}
        self.last_sub_included: bool = False
        self._warned_max_weight = False
        self._reset_accumulator()

    # Accumulator
    def _reset_accumulator(self) -> None:
        """`last` describes the last batch only; this collects an epoch."""
        self._acc: Dict[str, List[float]] = {}
        self._n_steps = 0
        # `clip` and `nzb` are properties of the solver, so their denominator
        # is the steps it ran: reused-alpha steps can neither clip nor fail, so
        # `_n_steps` would dilute both by 1/r -- 0.87 prints as 0.174 at r = 5.
        self._n_solved = 0
        self._n_clip = 0
        self._n_nzb = 0
        self._n_infeasible = 0
        self._offdiag: List[float] = []
        self._res: List[float] = []
        self._iter: List[int] = []
        self._sum: List[float] = []

    def new_epoch(self) -> None:
        self._reset_accumulator()

    # Solver
    def _project(self, a: torch.Tensor, k: int
                 ) -> Tuple[torch.Tensor, int, bool]:
        """Gauge, then the box, then the sum. Returns (alpha, n_clipped, ok).

        The rescale to `sum = k` is a gauge choice, the fixed point being
        defined up to scale; clipping the raw iterate clips an arbitrary
        quantity, and `clip` then fired in 100 % of steps with no final alpha
        near a bound. Restoring the sum comes last, moving only coordinates off
        a bound -- with the clamp last the sum walked 3.0000 -> 3.0127. Under
        `feasible=False` every alpha sits on one bound and the sum is not faked.
        """
        x = a * (float(k) / a.sum().clamp_min(self.eps))
        clipped = x.clamp(self.min_weight, self.max_weight)
        n_clip = int((clipped != x).sum())
        x = clipped
        for _ in range(self.project_rounds):
            x = x.clamp(self.min_weight, self.max_weight)
            gap = float(k) - float(x.sum())
            if abs(gap) <= 1e-9:
                return x, n_clip, True
            free = (x < self.max_weight - 1e-12) if gap > 0 else \
                   (x > self.min_weight + 1e-12)
            n_free = int(free.sum())
            if n_free == 0:
                return x, n_clip, False
            x = torch.where(free, x + gap / n_free, x)
        x = x.clamp(self.min_weight, self.max_weight)
        return x, n_clip, abs(float(x.sum()) - float(k)) <= 1e-6

    def _alpha(self, M: torch.Tensor
               ) -> Tuple[torch.Tensor, float, int, int, bool]:
        """Solve the fixed point, starting at the diagonal solution, not at 1.

        `a <- sqrt(a / Ma)` is defined only while every component of `Ma` is
        positive; otherwise the clamp to `eps` explodes the root and the
        projection parks that coordinate on a bound for good. At the uniform
        start `(M 1)_i` is the projection of g_i on the sum of all gradients,
        so it is invalid exactly in the strong-conflict regime the bargain is
        for -- for cos <= -0.5, and at cos = -0.6 it leaves residual 1.0393
        after all 50 iterations where `1/||g||`, the exact fixed point for
        diagonal `M`, is already there.
        """
        k = M.shape[0]
        inv = 1.0 / torch.sqrt(torch.diagonal(M).clamp_min(self.eps))
        a = inv * (float(k) / inv.sum().clamp_min(self.eps))
        used = self.n_iter
        n_clip = 0
        feasible = True
        for it in range(self.n_iter):
            Ma = (M @ a).clamp_min(self.eps)
            a_new = torch.sqrt((a / Ma).clamp_min(self.eps))
            a_new, n_clip, feasible = self._project(a_new, k)
            if torch.allclose(a_new, a, rtol=1e-4, atol=1e-6):
                a = a_new
                used = it + 1
                break
            a = a_new
        # Scale-invariant residual: |a_i (Ma)_i - 1| would measure ||G||^2, the
        # sum being pinned to k, so the fixed point gives c^2 rather than 1.
        p = a * (M @ a)
        res = float((p / p.mean().clamp_min(self.eps) - 1.0).abs().max())
        return a, res, used, n_clip, feasible

    # Step
    def _check_players(self, names: Sequence[str]) -> None:
        # The allowlist is checked first and `allow_saturated_players` does not
        # lift it: that flag admits a saturated task, not an undeclared one.
        if self.allowed_players is not None:
            undeclared = [n for n in names if n not in self.allowed_players]
            if undeclared:
                raise ValueError(
                    "%r is not in the declared player list %r. The bargain is "
                    "an ALLOWLIST here: a blacklist only stops the names "
                    "somebody thought of, and this project already put "
                    "'rekonstrukcja' into a bargain guarded by ('recon', "
                    "'reconstruction')." % (undeclared, self.allowed_players))
        # Ahead of the saturated-player exit: a run that admits a saturated
        # task is the likeliest to carry an odd box, and behind the exit the
        # warning never reached exactly that run.
        ceil = alpha_ceiling(len(names), self.min_weight)
        if not self._warned_max_weight and self.max_weight >= ceil:
            self._warned_max_weight = True
            warnings.warn(
                "NashMTL: max_weight=%.3f is UNREACHABLE with %d players -- the "
                "alphas sum to %d and the other %d coordinates are each at "
                "least min_weight=%.3f, so alpha_i <= %.3f. The knob has no "
                "receiver; the effective ceiling is %.3f."
                % (self.max_weight, len(names), len(names), len(names) - 1,
                   self.min_weight, ceil, min(self.max_weight, ceil)),
                RuntimeWarning, stacklevel=2)
        if self.allow_saturated_players:
            return
        bad = [n for n in names
               if any(f in n.lower() for f in self.forbid_players)]
        if bad:
            raise ValueError(
                "%r may not be a player in the bargain. A saturated task takes "
                "the LARGEST share precisely because its gradient is smallest: "
                "measured 2.31 of 3.0 at recon 0.9998, with the prediction "
                "tasks pushed to 0.39 and 0.31. Pass it as a SubordinateTerm "
                "with a declared weight, or set allow_saturated_players=True "
                "if the point of the run is to measure that." % (bad,))

    def step(self, losses: Mapping[str, torch.Tensor],
             subordinate: Optional[torch.Tensor] = None,
             recompute: bool = True) -> torch.Tensor:
        """Compute the alphas and run one `backward` on the weighted sum.

        The per-task gradients exist only to compute `alpha` and go through
        `torch.autograd.grad`, which does not touch `.grad`; writing a combined
        gradient by hand would leave the private head parameters at zero, not
        being in `G`. `subordinate` joins the same backward with weight 1 -- a
        separate one cannot, the graph being freed by the first call.
        """
        names = [n for n, v in losses.items()
                 if v is not None and torch.is_tensor(v) and v.requires_grad]
        if not names:
            return torch.zeros((), device=self.params[0].device)
        self._check_players(names)

        if self.bypass or len(names) == 1:
            total = sum(losses[n] for n in names)
            included = (subordinate is not None
                        and subordinate.requires_grad)
            if included:
                total = total + subordinate
            self._record_components(names, [1.0] * len(names), losses,
                                    included)
            total.backward()
            self.last = {n: 1.0 for n in names}
            self._n_steps += 1
            for n in names:
                self._acc.setdefault(n, []).append(1.0)
            return total.detach()

        if not recompute and self.last:
            # Decay 0.9 is a horizon of ~10 steps while recomputing costs K+1
            # backward passes (0.084 -> 0.219 s at three players), so reuse
            # samples the same bargain more slowly, rather than approximating.
            reused = [float(self.last.get(n, 1.0)) for n in names]
            total = sum(w * losses[n] for w, n in zip(reused, names))
            included = (subordinate is not None
                        and subordinate.requires_grad)
            if included:
                total = total + subordinate
            self._record_components(names, reused, losses, included)
            total.backward()
            self._n_steps += 1
            for n in names:
                self._acc.setdefault(n, []).append(float(self.last.get(n, 1.0)))
            return total.detach()

        rows = []
        for n in names:
            g = torch.autograd.grad(losses[n], self.params,
                                    retain_graph=True, allow_unused=True)
            rows.append(torch.cat(
                [(x if x is not None else torch.zeros_like(p)).reshape(-1)
                 for x, p in zip(g, self.params)]).double())
        G = torch.stack(rows)                       # (K, P) float64
        norms = G.norm(dim=1)
        self.last_norms = {n: float(norms[i]) for i, n in enumerate(names)}
        # `allow_unused` yields a row of zeros -> `Ma` clamps to eps ->
        # sqrt(a/eps), and the run looks healthy throughout.
        if float(norms.min()) < self.zero_threshold:
            dead = [n for i, n in enumerate(names)
                    if float(norms[i]) < self.zero_threshold]
            raise RuntimeError(
                "NashMTL: task(s) %r have a zero gradient on the shared "
                "parameters (norm < %.1e). Without this guard they would "
                "receive the MAXIMUM weight. Check that the latent is not "
                "detached on their path and that `params` really is the shared "
                "set." % (dead, self.zero_threshold))

        M = G @ G.t()
        d = torch.sqrt(torch.diagonal(M).clamp_min(self.eps))
        Mc = M / (d.unsqueeze(0) * d.unsqueeze(1))
        k = M.shape[0]
        offdiag = float((Mc - torch.eye(k, dtype=Mc.dtype, device=Mc.device)
                         ).abs().max())
        a, res, used, n_clip, feasible = self._alpha(M)
        alpha_sum = float(a.sum())
        del G, M, rows

        # A convex combination of vectors summing to k also sums to k, so the
        # EMA preserves the invariant. Keyed by name, not position: a margin
        # loss satisfied on every pair is exactly 0.0 and gets dropped
        # (`pairwise` vanished at step 21 of 40). A positional EMA then raises
        # a shape error, or -- worse, if the length holds -- nothing at all.
        for i, n in enumerate(names):
            prev = self._ema.get(n)
            self._ema[n] = (float(a[i]) if prev is None else
                            self.alpha_ema * prev
                            + (1.0 - self.alpha_ema) * float(a[i]))
        a = torch.tensor([self._ema[n] for n in names],
                         dtype=losses[names[0]].dtype, device=a.device)

        self.last = {n: float(a[i]) for i, n in enumerate(names)}
        self._n_steps += 1
        self._n_solved += 1
        self._n_clip += int(n_clip > 0)
        # Non-convergence is the residual, not the iteration count: counting
        # only exhausted iterations passed an iterate that stopped short of
        # stationarity as converged (res 0.9995 after 8, reported clean).
        self._n_nzb += int(res > 0.1)
        self._n_infeasible += int(not feasible)
        self._offdiag.append(offdiag)
        self._res.append(res)
        self._iter.append(used)
        self._sum.append(alpha_sum)
        for i, n in enumerate(names):
            self._acc.setdefault(n, []).append(float(a[i]))

        weighted = sum(a[i] * losses[n] for i, n in enumerate(names))
        included = subordinate is not None and subordinate.requires_grad
        if included:
            weighted = weighted + subordinate
        self._record_components(names, [float(a[i]) for i in range(len(names))],
                                losses, included)
        weighted.backward()
        return weighted.detach()

    # Components
    def _record_components(self, names: Sequence[str],
                           alphas: Sequence[float],
                           losses: Mapping[str, torch.Tensor],
                           sub_included: bool) -> None:
        """Store `alpha_n * loss_n` -- the addends of the sum just built.

        Recorded at each of the three places `step` builds a total, since they
        use three different alpha vectors and a decomposition from the wrong
        one would still sum to something plausible, just not to `loss`. The
        stack is read out once: a `float()` per player is a device sync each.
        """
        if names:
            vals = torch.stack([losses[n].detach().reshape(())
                                for n in names]).tolist()
            self.last_components = {n: float(a) * float(v)
                                    for n, a, v in zip(names, alphas, vals)}
        else:
            self.last_components = {}
        self.last_sub_included = bool(sub_included)

    # Falsifiers
    @staticmethod
    def _median(v: Sequence[float]) -> float:
        if not v:
            return float("nan")
        s = sorted(v)
        return float(s[len(s) // 2])

    def report(self) -> Dict[str, object]:
        """Per-epoch aggregates, with the thresholds actually evaluated.

        `violations` spells out `clip`, `nzb`, `sum_alpha` and `infeasible`.
        One reading carries no message of its own: `offdiag < 0.1` means the
        tasks are orthogonal and there is nothing to bargain over, while
        losing a conflict starts above 0.3.
        """
        out: Dict[str, object] = {"n_steps": self._n_steps}
        if not self._n_steps:
            out.update({"ok": False, "violations": ["no steps recorded"]})
            return out
        alphas = {n: self._median(v) for n, v in sorted(self._acc.items())}
        solved = float(max(self._n_solved, 1))
        clip = self._n_clip / solved
        nzb = self._n_nzb / solved
        infeasible = self._n_infeasible / solved
        k = float(len(alphas))
        out.update({
            "alpha": alphas,
            "n_players": len(alphas),
            "n_solved": self._n_solved,
            "solver_duty": round(self._n_solved / float(self._n_steps), 4),
            "clip": clip,
            "nzb": nzb,
            "infeasible": infeasible,
            "iter": (sum(self._iter) / float(len(self._iter))
                     if self._iter else float("nan")),
            "offdiag": self._median(self._offdiag),
            "res": self._median(self._res),
            "sum_alpha": self._median(self._sum),
            "norms": dict(self.last_norms),
            "subordinate": {t.name: t.effective()
                            for t in self.subordinate_terms},
        })
        bad: List[str] = []
        if clip > self.clip_threshold:
            bad.append("clip %.3f > %.2f: the box binds, so this run measures "
                       "the limiter rather than the bargain"
                       % (clip, self.clip_threshold))
        if nzb > self.nzb_threshold:
            bad.append("nzb %.3f > %.2f: the solver did not converge in %d "
                       "iterations" % (nzb, self.nzb_threshold, self.n_iter))
        sa = out["sum_alpha"]
        if sa == sa and abs(float(sa) - k) > 1e-3:
            bad.append("sum_alpha %.4f != %d players: the effective learning "
                       "rate drifts between steps" % (float(sa), int(k)))
        if infeasible > 0.0:
            bad.append("infeasible %.3f: alphas on a single bound, the bargain "
                       "negotiated nothing in those steps" % (infeasible,))
        # Separate from the violations: a dead knob says a config key has no
        # receiver, and folding it in would make `ok` false in every run.
        out["inert"] = [k_ for k_, _ in self.inert_keys(int(k))]
        out["violations"] = bad
        out["ok"] = not bad
        return out

    def format_report(self) -> str:
        r = self.report()
        if not r.get("n_steps"):
            return "nash: no steps"
        alphas = " ".join("%s=%.2f" % (n, v)
                          for n, v in sorted(r["alpha"].items()))
        return ("nash[%d steps / %d solved] %s | clip %.3f | nzb %.3f "
                "| iter %.1f | offdiag %.3f | res %.4f | sum_alpha %.4f "
                "| infeasible %.3f | %s"
                % (r["n_steps"], r.get("n_solved", 0),
                   alphas, r["clip"], r["nzb"], r["iter"],
                   r["offdiag"], r["res"], r["sum_alpha"], r["infeasible"],
                   "OK" if r["ok"] else "VIOLATIONS: " + " ; ".join(
                       r["violations"])))

    def assert_healthy(self, raise_on_fail: bool = True) -> List[str]:
        """Evaluate the thresholds. They were a docstring once, which is how
        both stayed exceeded in every epoch of every run."""
        v = list(self.report().get("violations", []))
        if v and raise_on_fail:
            raise RuntimeError("NashMTL thresholds exceeded: " + " ; ".join(v))
        return v

    def inert_keys(self, n_players: int) -> List[Tuple[str, str]]:
        """Config keys that this object will not act on. Used by
        `config.report_inert_keys`, so a dead knob is printed once, at start."""
        out: List[Tuple[str, str]] = []
        ceil = alpha_ceiling(n_players, self.min_weight)
        if self.max_weight >= ceil:
            out.append((
                "training.nash.max_weight",
                "unreachable: the alphas sum to %d and the other %d "
                "coordinates each sit at or above min_weight=%.3f, so "
                "alpha_i <= %.3f; the effective ceiling is %.3f"
                % (n_players, n_players - 1, self.min_weight, ceil,
                   min(self.max_weight, ceil))))
        if self.bypass:
            out.append(("training.nash.*",
                        "bypass=True: the losses are summed and no alpha is "
                        "computed, so every other nash key is inert"))
        return out

    def header(self, n_players: Optional[int] = None) -> str:
        """One-off header. `n_players` comes from the caller because the count
        is only known at the first `step`, and this line prints before it."""
        ceiling = ""
        if n_players:
            ceil = alpha_ceiling(n_players, self.min_weight)
            ceiling = " | alpha ceiling %.3f" % min(self.max_weight, ceil)
            if self.max_weight >= ceil:
                ceiling += (" (max_weight %.3f UNREACHABLE: alphas sum to %d "
                            "and the other %d are >= %.3f, so alpha_i <= %.3f)"
                            % (self.max_weight, n_players, n_players - 1,
                               self.min_weight, ceil))
        sub = ""
        if self.subordinate_terms:
            sub = " | subordinate: " + " ; ".join(
                t.describe() for t in self.subordinate_terms)
        return ("NashMTL: %d shared tensors / %d weights | box [%.3f, %.3f] "
                "| ema %.2f | bypass %s%s%s"
                % (len(self.params), self.n_params, self.min_weight,
                   self.max_weight, self.alpha_ema, self.bypass, ceiling, sub))
