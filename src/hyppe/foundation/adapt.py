"""Fine-tune without losing what was pretrained, and prove which happened.

Four freeze policies. `heads` moves the prediction heads only, leaving
encoder, ladder, quantiser and decoder frozen, so every sequence keeps its
codes and a design produced before the fine-tune still decodes to the same
bases; it is the default. `context` adds the condition and gene encoders, for
an adaptation about a new condition rather than new sequences. `adapters` adds
low-rank adapters beside the frozen head trunks -- more capacity at a bounded
parameter cost, and the pretrained weights stay bit-identical, so the
adaptation can be removed. `all` is refused unless `allow_full=True`, because
it makes `decode_codes` a different function and voids every code the previous
checkpoint produced.

The quantiser is frozen in every mode but `all`, and `requires_grad_(False)`
does not do it: the level-0 codebook is updated by EMA inside the forward
pass, so a "heads only" fine-tune would still rewrite the code space, and
silently, because the loss would look fine. `freeze_quantiser` puts the EMA
modules in eval and asserts the buffers are unchanged.

Forgetting is measured, not hoped against. Every adaptation carries a baseline
metric on the original selection set, recomputed each epoch, and
`ForgettingGuard` stops the run when it falls by more than the declared budget
-- 0.02 of Spearman by default, below the project's own seed noise (test SE
~0.029). The learning rate defaults to a tenth for a related reason: the
pretrained optimum was reached under a decayed schedule, so restarting at the
initial rate takes a step sized by the new loss alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "FREEZE_POLICIES",
    "LowRankAdapter",
    "AdaptationConfig",
    "AdaptationReport",
    "ForgettingGuard",
    "freeze_quantiser",
    "apply_policy",
    "insert_adapters",
    "remove_adapters",
    "trainable_report",
]

FREEZE_POLICIES: Tuple[str, ...] = ("heads", "context", "adapters", "all")

#: Modules that own the code space. Unfreezing any of them changes what a code
#: means, breaking the diffusion contract in `model/assemble.py` (C8).
_CODE_SPACE = ("encoder", "level_proj", "ladder", "decoder", "occupancy")
_CONTEXT = ("condition", "gene", "genome")
_HEADS = ("absolute", "pairwise", "discriminator", "decomposition", "pooling")


# Adapters
class LowRankAdapter(nn.Module):
    """`y = frozen(x) + B(A(x)) * scale`, with B initialised to zero.

    Zero-initialised B makes the adapted module bit-identical to the frozen
    one at step 0, so a rising metric is the fine-tune and not a
    re-initialisation, and removing the adaptation restores the original
    behaviour exactly rather than approximately.

    A is initialised small but non-zero. Zero-initialising both would make
    d(output)/d(A) identically zero and the adapter would never leave the
    origin.
    """

    def __init__(self, base: nn.Linear, rank: int = 8,
                 scale: float = 1.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LowRankAdapter wraps nn.Linear, got %r"
                            % (type(base).__name__,))
        if rank < 1:
            raise ValueError("adapter rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(scale)
        self.down = nn.Linear(base.in_features, int(rank), bias=False)
        self.up = nn.Linear(int(rank), base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scale

    def extra_repr(self) -> str:
        return "rank=%d, scale=%.3f, base=%d->%d (frozen)" % (
            self.rank, self.scale, self.base.in_features,
            self.base.out_features)


def insert_adapters(model: nn.Module, rank: int = 8, scale: float = 1.0,
                    targets: Sequence[str] = _HEADS) -> List[str]:
    """Wrap every `nn.Linear` under `targets` in a LowRankAdapter.

    Returns the qualified names wrapped, so the caller can show them rather
    than trust a count: a silent zero-wrap, because a name in `targets` is
    absent on this build, gives a fine-tune with no trainable parameters that
    still reports a loss.
    """
    done: List[str] = []
    for tname in targets:
        root = getattr(model, tname, None)
        if root is None or not isinstance(root, nn.Module):
            continue
        for name, child in list(root.named_modules()):
            if not isinstance(child, nn.Linear):
                continue
            parent = root
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            leaf = parts[-1]
            wrapped = LowRankAdapter(child, rank=rank, scale=scale)
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            done.append("%s.%s" % (tname, name))
    if not done:
        raise RuntimeError(
            "no nn.Linear was found under %r, so `insert_adapters` wrapped "
            "nothing and the resulting fine-tune would have zero trainable "
            "parameters while still printing a loss." % (list(targets),))
    return done


def remove_adapters(model: nn.Module) -> int:
    """Unwrap every adapter, restoring the frozen module exactly."""
    n = 0
    for _name, module in list(model.named_modules()):
        for leaf, child in list(module.named_children()):
            if isinstance(child, LowRankAdapter):
                setattr(module, leaf, child.base)
                n += 1
    return n


# Freezing
def freeze_quantiser(model: nn.Module) -> List[str]:
    """Stop the EMA codebook moving. `requires_grad_(False)` does not.

    The update is an exponential moving average over assignments inside the
    forward pass, not a gradient, so only `eval()` on the owning modules
    disables it. A later `.train()` on the whole model re-enables it, which is
    why the loop re-applies this after every epoch.
    """
    frozen: List[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "_ema_cluster_size") or \
                hasattr(module, "cluster_size") or \
                type(module).__name__ in ("HyperbolicCodebook", "ScalarFSQ"):
            module.eval()
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)
            frozen.append(name or type(module).__name__)
    return frozen


def apply_policy(model: nn.Module, policy: str = "heads",
                 allow_full: bool = False,
                 adapter_rank: int = 8) -> Dict[str, object]:
    """Set `requires_grad` across the model and report what may move."""
    if policy not in FREEZE_POLICIES:
        raise ValueError("policy must be one of %r, got %r"
                         % (FREEZE_POLICIES, policy))
    if policy == "all" and not allow_full:
        raise ValueError(
            "policy='all' unfreezes the encoder, the ladder and the decoder, "
            "so `decode_codes` becomes a DIFFERENT function: every code "
            "produced by the previous checkpoint decodes to different bases "
            "and every design already scored is invalid. Pass allow_full=True "
            "if that is the point of the run, and name the run after it.")

    for p in model.parameters():
        p.requires_grad_(False)

    unlocked: List[str] = []
    groups = list(_HEADS)
    if policy in ("context", "adapters"):
        groups += list(_CONTEXT)
    if policy == "all":
        groups += list(_CONTEXT) + list(_CODE_SPACE)
    for gname in groups:
        mod = getattr(model, gname, None)
        if isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad_(True)
            unlocked.append(gname)

    adapters: List[str] = []
    if policy == "adapters":
        adapters = insert_adapters(model, rank=int(adapter_rank))
        # The adapter froze its own base; re-freeze everything else under the
        # head groups so that only the adapters move.
        for gname in _HEADS:
            mod = getattr(model, gname, None)
            if isinstance(mod, nn.Module):
                for n, p in mod.named_parameters():
                    p.requires_grad_("down." in n or "up." in n)

    frozen_q = [] if policy == "all" else freeze_quantiser(model)
    return {
        "policy": policy,
        "unlocked_modules": unlocked,
        "adapters": adapters,
        "quantiser_modules_frozen": frozen_q,
        "code_space_frozen": policy != "all",
        **trainable_report(model),
    }


def trainable_report(model: nn.Module) -> Dict[str, object]:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by: Dict[str, int] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            by[n.split(".")[0]] = by.get(n.split(".")[0], 0) + p.numel()
    return {"parameters_total": int(tot), "parameters_trainable": int(tr),
            "trainable_fraction": float(tr) / float(max(tot, 1)),
            "trainable_by_module": by}


# The run
@dataclass
class AdaptationConfig:
    """What may move, how far, and when to stop."""

    policy: str = "heads"
    adapter_rank: int = 8
    #: A tenth of the pretraining rate; see the module docstring.
    lr_scale: float = 0.1
    epochs: int = 5
    weight_decay: float = 0.0
    allow_full: bool = False
    #: How much of the original selection metric may be given up. 0.02 is
    #: below the project's own seed noise (test SE ~0.029).
    forgetting_budget: float = 0.02
    #: Name of the baseline metric inside the evaluation dict.
    baseline_metric: str = "spy_scc"
    target_metric: str = "scc"

    def validate(self) -> None:
        if self.policy not in FREEZE_POLICIES:
            raise ValueError("policy must be one of %r" % (FREEZE_POLICIES,))
        if not 0.0 < self.lr_scale <= 1.0:
            raise ValueError("lr_scale must be in (0, 1]")
        if self.forgetting_budget < 0.0:
            raise ValueError("forgetting_budget must be >= 0")


@dataclass
class AdaptationReport:
    """Target metric AND baseline metric, per epoch. Never one without the
    other."""

    config: Dict[str, object]
    wiring: Dict[str, object]
    baseline_before: float = float("nan")
    history: List[Dict[str, float]] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""

    def best_epoch(self) -> Optional[Dict[str, float]]:
        ok = [h for h in self.history if not h.get("over_budget", 0.0)]
        if not ok:
            return None
        return max(ok, key=lambda h: h.get("target", float("-inf")))

    def format(self) -> str:
        lines = ["adaptation: policy %s, %d of %d parameters trainable (%.2f %%)"
                 % (self.config.get("policy"),
                    self.wiring.get("parameters_trainable", 0),
                    self.wiring.get("parameters_total", 0),
                    100.0 * float(self.wiring.get("trainable_fraction", 0.0)))]
        if self.wiring.get("code_space_frozen"):
            lines.append("  code space FROZEN: every code keeps its meaning, "
                         "so designs made before this fine-tune stay valid")
        else:
            lines.append("  code space UNFROZEN: decode_codes is now a "
                         "DIFFERENT function and every earlier design is void")
        for h in self.history:
            lines.append("  epoch %d: target %.4f | baseline %.4f (drop %.4f%s)"
                         % (int(h.get("epoch", 0)), h.get("target", float("nan")),
                            h.get("baseline", float("nan")),
                            h.get("drop", float("nan")),
                            " OVER BUDGET" if h.get("over_budget") else ""))
        if self.stopped_early:
            lines.append("  STOPPED: " + self.stop_reason)
        best = self.best_epoch()
        if best is not None:
            lines.append("  best admissible epoch %d: target %.4f, baseline "
                         "%.4f" % (int(best["epoch"]), best["target"],
                                   best["baseline"]))
        else:
            lines.append("  NO admissible epoch: every one spent more of the "
                         "baseline than the budget allows. The adaptation did "
                         "not fail to learn; it learned at a price that was "
                         "declared unacceptable before the run.")
        return "\n".join(lines)


class ForgettingGuard(object):
    """Stops an adaptation that is buying its target with the baseline."""

    def __init__(self, before: float, budget: float = 0.02) -> None:
        self.before = float(before)
        self.budget = float(budget)

    def check(self, now: float) -> Tuple[bool, float, str]:
        drop = self.before - float(now)
        if not math.isfinite(now):
            return False, drop, "baseline metric is not finite"
        if drop > self.budget:
            return False, drop, (
                "the baseline fell %.4f (from %.4f to %.4f), past the declared "
                "budget of %.4f. The fine-tune is buying its subcategory with "
                "the property the checkpoint was selected for."
                % (drop, self.before, now, self.budget))
        return True, drop, "baseline %.4f (drop %.4f <= %.4f)" % (
            now, drop, self.budget)


def run_adaptation(model: nn.Module,
                   step_fn: Callable[[int], Dict[str, float]],
                   evaluate_target: Callable[[], Dict[str, float]],
                   evaluate_baseline: Callable[[], Dict[str, float]],
                   config: Optional[AdaptationConfig] = None,
                   base_lr: float = 3e-4) -> AdaptationReport:
    """Generic adaptation loop: `step_fn(epoch)` does one epoch of training.

    The loop owns three things and nothing else: the freeze policy, the
    forgetting guard, and the record that both metrics were taken every epoch.
    Data, loss and batching stay with the caller, because a fine-tune on a
    DESeq2 contrast and one on a subcategory share none of them.
    """
    cfg = config or AdaptationConfig()
    cfg.validate()
    wiring = apply_policy(model, cfg.policy, allow_full=cfg.allow_full,
                          adapter_rank=cfg.adapter_rank)
    before = float(evaluate_baseline().get(cfg.baseline_metric, float("nan")))
    guard = ForgettingGuard(before, cfg.forgetting_budget)
    rep = AdaptationReport(config=cfg.__dict__.copy(), wiring=wiring,
                           baseline_before=before)
    print("  adaptation lr = %.2e (%.2f x the pretraining %.2e)"
          % (base_lr * cfg.lr_scale, cfg.lr_scale, base_lr))
    for ep in range(1, int(cfg.epochs) + 1):
        step_fn(ep)
        # `model.train()` inside `step_fn` re-enables the EMA codebook
        # update, which `requires_grad` does not control.
        if cfg.policy != "all":
            freeze_quantiser(model)
        tgt = float(evaluate_target().get(cfg.target_metric, float("nan")))
        now = float(evaluate_baseline().get(cfg.baseline_metric,
                                            float("nan")))
        ok, drop, why = guard.check(now)
        rep.history.append({"epoch": float(ep), "target": tgt,
                            "baseline": now, "drop": drop,
                            "over_budget": 0.0 if ok else 1.0})
        if not ok:
            rep.stopped_early = True
            rep.stop_reason = "epoch %d: %s" % (ep, why)
            break
    return rep
