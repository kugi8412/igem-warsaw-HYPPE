"""The training loop: the one place where every module of this repo is called.

Modules that are never run next to their siblings are not tested, so the loop
is written to fail loudly at the seams rather than to run at any cost. Four
things here are structural rather than preferences.

Scalarisation is the default and the baseline (``training.method``). A
balancing method changes the effective task weights, and changing the weights
alone moves the metric, so a gradient method without the baseline at its own
weights measures nothing. Its price is K+1 backward passes instead of one,
measured per run and printed.

Reconstruction is a constraint (``training.recon_constraint``), not a player
and not a constant. A Nash bargain equalises utilities and hands the largest
share to the weakest gradient, so a task already at its operating point wins
it; a constant weight small enough to avoid that let the same objective fall
to 0.55 in one epoch. A Lagrange multiplier is driven by the violation
instead, and ``ConstraintController.report()`` calls the run a test of the box
when the multiplier sits on a bound, because a pinned controller is a constant
weight wearing a controller's name.

The wiring gates F1/F2/F3 run before the first step and a failure is an exit
code, because a method that is configured but not wired reports exactly what a
working one reports. Gradient orthogonality is measured in epoch 1 and printed
but never blocks the run: below |offdiag| ~ 0.1 the Gram matrix is nearly
diagonal and the Nash fixed point degenerates to alpha_i = 1/||g_i||, which is
a result about the tasks.

Two silent traps are handled explicitly. ``LatentMask`` draws block positions
from a persistent counter that every call to ``views()`` advances, evaluations
included, so a reloaded model masks other columns and reports a different
reconstruction accuracy for identical weights; ``evaluate`` pins the counter
and restores it. And subordinate terms such as ``commitment`` are already
multiplied by their own weight inside the module that produces them, so each
declares its multiplier chain as a ``SubordinateTerm`` and the loop adds it at
weight 1.0, with the effective coefficient printed rather than inferred.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import (
    RECONSTRUCTION_FLOOR,
    HyppeConfig,
    load_config,
    report_inert_keys,
    to_mapping,
)
from ..geometry.lorentz import RiemannianAdam
from ..heads.absolute import pearson, spearman
from ..heads.discriminator import roc_auc
from ..heads.pairwise import PairOutput
from ..model.assemble import HyppeModel, _repo_root
from ..model.quantizer import reset_statistics
from .batches import BatchMaker, Corpus, TrainBatch
from .gates import ORTHOGONALITY_FLOOR, orthogonality_gate, run_wiring_gates
from .geco import ConstraintController, ConstraintState
from .nash import SubordinateTerm
from .weighting import Scalarisation, TaskWeighting, build_weighting, measure_step_cost

__all__ = ["RunSettings", "TrainLoop", "EpochResult", "main"]

#: Exit codes. Distinct so a queue can tell a failed gate from a crash.
EXIT_OK = 0
EXIT_GATE = 3
EXIT_FALSIFIER = 4
#: The run diverged, or the printed decomposition of `loss` did not add up to
#: `loss`. Distinct from EXIT_FALSIFIER because it says something different:
#: a falsifier is a result, a divergence is a run with no result at all.
EXIT_DIVERGED = 7
#: Every epoch was refused by the reconstruction admission gate, so no
#: checkpoint exists. Distinct from EXIT_OK because "no file" previously read
#: exactly like "the run has not got there yet".
EXIT_NO_CHECKPOINT = 8

#: T5's tolerance for the per-batch loss COMPONENTS only; the evaluation
#: metrics are still held to exactly zero. A float32 reduction over the
#: evaluation batches is not bit-reproducible on this GPU (2.2e-09 between two
#: evaluations with no checkpoint between them), so `== 0.0` tested the
#: device. 1e-6 is 450x above that noise and 2300x below the smallest real
#: leak this gate has caught. Splitting by key prefix keeps that catch:
#: `recon_lambda` is a metric, so it is judged at zero, not here.
T5_COMPONENT_TOL = 1.0e-6


class DivergenceError(RuntimeError):
    """The loss ran away, went non-finite, or stopped equalling its parts.

    Raised rather than printed: a run that spent 99.999 % of its loss in a
    term the log never named still returned 0, so the queue recorded a success
    and started the next arm. An exit code is the only part of that chain a
    queue reads.
    """


class NoAdmissibleCheckpoint(DivergenceError):
    """The run finished but every epoch failed the reconstruction admission.

    A subclass, so a caller that only knows `DivergenceError` still stops, but
    a distinct type because the two get different exit codes. Telling them
    apart by a substring of the message would break the first time somebody
    reworded it.
    """


# Settings
def _source_mtimes() -> Dict[str, float]:
    """Identity of the code that produced a checkpoint.

    The project is not a git repository, so there is no commit to record. The
    mtimes of the files that define the model and the training loop are enough
    to tell two checkpoints apart when one was written by edited code.
    """
    out: Dict[str, float] = {}
    for rel in ("model/assemble.py", "training/loop.py", "config.py"):
        p = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "hyppe", rel)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), rel)
        try:
            out[rel] = float(os.path.getmtime(p))
        except OSError:
            pass
    return out


@dataclass
class RunSettings:
    """Everything that is about this run rather than about the model.

    Paths, device, limits and output. Kept out of `HyppeConfig` because the
    config is saved next to the checkpoint under the contract "the config is
    what ran", and a machine-local path is not part of that.
    """

    name: str = "smoke"
    out_dir: str = "runs"
    device: str = "auto"
    sequences_path: str = ""
    combat_path: str = ""
    samples_path: str = ""
    splits_path: str = ""
    #: DNA-LM flank cache. Empty = the gene_context channel answers with the
    #: encoder's unknown token for every row, and the corpus report says so.
    #: `--flank-cache` sets it, otherwise `cfg.data.flank_embedding_path`.
    flank_cache_path: str = ""
    flank_index_path: str = ""
    #: Sentence-LM cache of the experiment description, keyed
    #: <genome>|<category>. Empty = the text channel answers with the
    #: encoder's unknown token for every row. `--text-cache` sets it,
    #: otherwise `cfg.data.text_embedding_path`; read only when
    #: `conditioning.experiment_text.enabled` is true.
    text_cache_path: str = ""
    text_index_path: str = ""
    split_column: str = "B_block10"
    max_orthogroups: int = 0
    synthetic: int = 0
    max_steps: int = 0
    max_eval_steps: int = 0
    log_every: int = 1
    epochs: int = 0                 # 0 = use the config
    batch_size: int = 0             # 0 = use the config
    negative_fraction: float = 0.25
    method: str = ""                # "" = use the config
    skip_gates: bool = False
    cost_probe: bool = False
    allow_euclidean_manifold: bool = False
    #: Floor for T4b. 0.0 = measure it on the corpus being trained on
    #: (`Corpus.base_floor`). `RECONSTRUCTION_FLOOR` is the same quantity on
    #: the real promoter corpus and is printed beside it; on uniform synthetic
    #: DNA the true floor is 0.25, so judging against 0.2626 would fail a
    #: threshold the data cannot reach.
    recon_min: float = 0.0
    #: Write `<name>_ep###.pt` every epoch, in addition to `_fundational.pt`.
    #: The selection rule is a hypothesis; without an archive it is an
    #: irreversible one. Refused epochs are archived too, because a refused
    #: epoch is the evidence for whether the floor is right.
    save_every_epoch: bool = False

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Breaking the validation metric down by axis
#
# The global metrics average over axes of very different cardinality, so an
# axis on which the model is WORSE THAN A CONSTANT can leave no trace in them:
# global `sd_p/(pcc*sd_t)` read 1.027, which looks calibrated, while the
# genome-within-orthogroup axis sat at R^2 = -0.0289 with slope 0.471 for four
# epochs. The three axes, per sample:
#
#   ortogrupa   which gene; `row_index` through the corpus `og` table
#   genom       which ortholog, AFTER the orthogroup mean is removed, keyed on
#               the genome alone: (og, genome) determines a corpus row
#               uniquely, so grouping by the pair is saturated and would leave
#               exactly zero variance for the condition
#   warunek     `cat_target`: the main effect plus the orthogroup x category
#               interaction
#
# The decomposition is type I, so the ORDER is part of the definition and the
# same order is applied to target and prediction. The takes telescope, so they
# sum to var(y) - var(residual) to machine precision whether or not the axes
# are orthogonal; the gap to the naive version is reported as
# `os_nieortogonalnosc`, a number about the design of the data.


def _srednia_po_grupach(v: np.ndarray, kod: np.ndarray, n: int) -> np.ndarray:
    """Mean of `v` within group `kod`, scattered back onto the rows."""
    s = np.bincount(kod, weights=v, minlength=n)
    c = np.bincount(kod, minlength=n).astype(np.float64)
    np.maximum(c, 1.0, out=c)
    return (s / c)[kod]


def _pair_code(a: np.ndarray, b: np.ndarray):
    """Dense code for the product of two codes, not `a * n_b + b`.

    The arithmetic product would need `minlength = n_a * n_b`, an array
    growing as the product of the group counts. `np.unique` gives a dense
    code, so `bincount` is O(non-empty cells) rather than O(all cells).
    """
    key = (a.astype(np.int64) * (int(b.max()) + 1)) + b.astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).ravel()
    return inv, int(inv.max()) + 1


def _skladowe_sekwencyjne(v: np.ndarray, plan):
    """Type I projections onto successive group spaces, plus residual norms."""
    r = np.asarray(v, dtype=np.float64).ravel().copy()
    r -= r.mean()
    kw = [float(np.dot(r, r))]
    skl = []
    for _, kod, n in plan:
        c = _srednia_po_grupach(r, kod, n)
        skl.append(c)
        r = r - c
        kw.append(float(np.dot(r, r)))
    return skl, r, kw


def _r2_osi(yc: np.ndarray, pc: np.ndarray) -> float:
    """1 - SS(y-p)/SS(y) on one axis. Negative = worse than zero there."""
    ss = float(np.dot(yc, yc))
    if not ss > 0.0:
        return float("nan")
    d = yc - pc
    return 1.0 - float(np.dot(d, d)) / ss


def _nachylenie_osi(yc: np.ndarray, pc: np.ndarray) -> float:
    """OLS of y on p within one axis. < 1 = the model overshoots amplitude."""
    sp = float(np.dot(pc, pc))
    if not sp > 0.0:
        return float("nan")
    return float(np.dot(yc, pc)) / sp


def rozbicie_osi(p: np.ndarray, t: np.ndarray, og: np.ndarray,
                 gen: np.ndarray, cat: np.ndarray):
    """Variance of the target split over three axes, with per-axis quality.

    The diagnostic axes are row-level projections::

        A_og  = mean_og(y) - mean(y)      which gene
        A_gen = y - mean_og(y)            which ortholog of that gene
        A_war = the part of A_gen explained by category (main + og x cat)

    `A_war` is a SUBSET of `A_gen`, not an axis beside it: the condition
    varies within an orthogroup, so measured on the whole target it would
    count the sequence effect along with it. The takes are a different object
    and carry the prefix `os_wziecie_`.

    `os_genomu_nadmiar` is (sd_pred/sd_target) / pcc on the genome axis: 1.0
    is exactly the amplitude that axis's own correlation allows, and 2.1 means
    the model speaks twice as loudly as its coverage warrants, which is how
    R^2 goes negative while pcc stays positive.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    n = int(t.size)
    if n < 8 or p.size != n:
        return {}
    _, og_k = np.unique(np.asarray(og).ravel(), return_inverse=True)
    _, gen_k = np.unique(np.asarray(gen).ravel(), return_inverse=True)
    _, cat_k = np.unique(np.asarray(cat).ravel(), return_inverse=True)
    og_k = np.asarray(og_k, dtype=np.int64).ravel()
    gen_k = np.asarray(gen_k, dtype=np.int64).ravel()
    cat_k = np.asarray(cat_k, dtype=np.int64).ravel()
    n_og = int(og_k.max()) + 1
    n_gen = int(gen_k.max()) + 1
    n_cat = int(cat_k.max()) + 1
    k_int, n_int = _pair_code(og_k, cat_k)

    y0 = t - t.mean()
    p0 = p - p.mean()
    # Diagnostic axes, row level
    ay_og = _srednia_po_grupach(y0, og_k, n_og)
    ap_og = _srednia_po_grupach(p0, og_k, n_og)
    ay_gen = y0 - ay_og
    ap_gen = p0 - ap_og
    wy_g = _srednia_po_grupach(ay_gen, cat_k, n_cat)
    wp_g = _srednia_po_grupach(ap_gen, cat_k, n_cat)
    wy_i = _srednia_po_grupach(ay_gen - wy_g, k_int, n_int)
    wp_i = _srednia_po_grupach(ap_gen - wp_g, k_int, n_int)
    ay_war = wy_g + wy_i
    ap_war = wp_g + wp_i

    # Takes: the type I chain, telescoping
    plan = [("ortogrupa", og_k, n_og),
            ("warunek_glowny", cat_k, n_cat),
            ("warunek_interakcja", k_int, n_int),
            ("genom_netto", gen_k, n_gen)]
    _skl, _rz, kw = _skladowe_sekwencyjne(t, plan)
    wz = [(kw[i] - kw[i + 1]) / n for i in range(len(plan))]
    var_y = kw[0] / n
    var_r = kw[-1] / n

    def _war(v):
        return float(np.dot(v, v)) / n

    def _stat(ay, ap):
        """(r2, slope, pcc, r2 after rescaling, variance of the axis).

        An axis with variance below 1e-12 * var(y) is empty by design (one
        orthogroup in the sample, say) and its R^2 would be computed from the
        rounding of `t - t.mean()`; the smoke corpus produced -0.7315 where
        the axis does not exist. `nan` means "this axis is not here"; a number
        would mean "measured", and would be untrue.
        """
        wy = _war(ay)
        if not wy > 1e-12 * var_y:
            _n = float("nan")
            return _n, _n, _n, _n, wy
        s = _nachylenie_osi(ay, ap)
        return (_r2_osi(ay, ap), s, pearson(ay, ap),
                _r2_osi(ay, ap * s), wy)

    # `*_kal` is a positive control that is computed rather than asserted:
    # the prediction component times that axis's measured slope. It comes out
    # as exactly pcc^2, because R^2(c) is a parabola with its vertex at
    # c = slope, so the checkable thing is the triple (slope, pcc, r2_kal).
    og_r2, og_s, og_r, og_kal, og_w = _stat(ay_og, ap_og)
    gn_r2, gn_s, gn_r, gn_kal, gn_w = _stat(ay_gen, ap_gen)
    wa_r2, wa_s, wa_r, wa_kal, _wa_w = _stat(ay_war, ap_war)
    wg_r2, _a, _b, _c, _d = _stat(wy_g, wp_g)
    wi_r2, _a, _b, _c, _d = _stat(wy_i, wp_i)
    sd_c = float(np.sqrt(gn_w))
    sd_p = float(np.sqrt(_war(ap_gen)))
    met = {
        "os_n": float(n),
        "os_n_ortogrup": float(n_og),
        "os_n_genomow": float(n_gen),
        "os_n_kategorii": float(n_cat),
        "os_ortogrupa_r2": og_r2,
        "os_ortogrupa_nachylenie": og_s,
        "os_ortogrupa_pcc": og_r,
        "os_ortogrupa_r2_kal": og_kal,
        "os_ortogrupa_war": og_w,
        "os_genomu_r2": gn_r2,
        "os_genomu_nachylenie": gn_s,
        "os_genomu_pcc": gn_r,
        "os_genomu_r2_kal": gn_kal,
        "os_genomu_war": gn_w,
        "os_genomu_sd_cel": sd_c,
        "os_genomu_sd_pred": sd_p,
        "os_warunku_r2": wa_r2,
        "os_warunku_nachylenie": wa_s,
        "os_warunku_pcc": wa_r,
        "os_warunku_r2_kal": wa_kal,
        "os_warunku_glowny_r2": wg_r2,
        "os_warunku_interakcja_r2": wi_r2,
        "os_wziecie_ortogrupa": wz[0],
        "os_wziecie_warunek": wz[1] + wz[2],
        "os_wziecie_genom_netto": wz[3],
        "os_reszta_var": var_r,
        "os_var_y": var_y,
    }
    # Excess amplitude on the genome axis: 1.0 is optimal, 2.1 gives a
    # negative R^2 despite a positive pcc.
    met["os_genomu_nadmiar"] = ((sd_p / sd_c) / gn_r
                                if (sd_c > 0.0 and abs(gn_r) > 1e-12)
                                else float("nan"))
    # Closure check 1: the takes over the three axes must sum to the explained
    # variance.
    suma = math.fsum([met["os_wziecie_ortogrupa"], met["os_wziecie_warunek"],
                      met["os_wziecie_genom_netto"]])
    met["os_domkniecie"] = abs(suma - (var_y - var_r))
    # Closure check 2, independent of the first: the diagnostic genome axis
    # must itself split into condition + net genome + residual. If
    # `os_genomu_*` and the takes were measuring different objects, this
    # ledger would not balance, which is the point of keeping both.
    met["os_domkniecie_genom"] = abs(
        met["os_genomu_war"] - math.fsum(
            [met["os_wziecie_warunek"], met["os_wziecie_genom_netto"],
             var_r]))
    # How far the naive version -- summed squared components instead of
    # telescoping differences -- would drift. On an unbalanced design the two
    # need not agree, and this says so about the design, not the model.
    met["os_nieortogonalnosc"] = abs(
        math.fsum([_war(c) for c in _skl]) - (var_y - var_r))
    return met


@dataclass
class EpochResult:
    epoch: int
    train: Dict[str, float] = field(default_factory=dict)
    val: Dict[str, float] = field(default_factory=dict)
    weighting: Dict[str, object] = field(default_factory=dict)
    constraint: Dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0


class TrainLoop(object):
    """Model + data + weighting + constraint, driven step by step."""

    def __init__(self, cfg: HyppeConfig, model: HyppeModel,
                 maker: BatchMaker, settings: RunSettings) -> None:
        self.cfg = cfg
        self.model = model
        self.maker = maker
        self.settings = settings
        self.device = settings.resolved_device()
        self.model.to(self.device)

        tc = cfg.training
        self.method = settings.method or tc.method
        if self.method not in tc.METHODS:
            raise ValueError("unknown method %r; known %r"
                             % (self.method, tc.METHODS))

        # Who plays
        self.task_weights = {k: float(v) for k, v in tc.task_weights.items()
                             if getattr(model, k, None) is not None}
        dropped = sorted(set(tc.task_weights) - set(self.task_weights))
        if dropped:
            raise ValueError(
                "task_weights names %r but the assembled model has no such "
                "head. A weight with no loss is an inert key." % (dropped,))
        if self.maker.negative_fraction <= 0.0 and \
                self.task_weights.get("discriminator", 0.0) > 0.0:
            raise ValueError(
                "the discriminator is a player but negative_fraction is 0: "
                "its only negative class is the dinucleotide shuffle, so it "
                "would be trained on ONE class and its BCE would fall "
                "monotonically while the head learnt to answer 'real' always.")

        # Reconstruction as a constraint. Built BEFORE the weighting: the
        # controller's lambda is what the reconstruction's `SubordinateTerm`
        # declares, and a weighting object built first would print the
        # declared weight instead.
        self.controller: Optional[ConstraintController] = None
        rc = tc.recon_constraint
        views = {v: float(tc.view_weights.get(v, 0.0)) for v in tc.mask_views}
        if rc.enabled:
            self.controller = ConstraintController(
                target=float(rc.target), lam_init=float(rc.lam_init),
                lam_min=float(rc.lam_min), lam_max=float(rc.lam_max),
                lam_lr=float(rc.lam_lr), ema=float(rc.ema),
                saturation_limit=float(rc.saturation_limit),
                warmup_steps=int(rc.warmup_steps),
                name="reconstruction", multipliers=views)
        self.recon_view_weights = views

        # A second target, measured on `prefix_0`. The `reconstruction`
        # controller watches `rec_full`, which rides on L4 alone: destroying
        # L0 costs 0.0000 of it, destroying L4 costs 0.3508. Raising the
        # view's own weight does not help, because the same floored lambda
        # multiplies it; a second controller with its own lambda does.
        self.prefix_controller: Optional[ConstraintController] = None
        pc = getattr(tc, "prefix_constraint", None)
        if pc is not None and bool(getattr(pc, "enabled", False)):
            _ceil = self.model.mask.observable_ceiling()
            _podloga = float(_ceil.get("recon_floor", 0.2626))
            _f0 = float(_ceil.get("ceil_l0", 0.0))
            # Information ceiling of the prefix view: L0 carries `l0_bits` of
            # the sequence's `seq_bits`, so it can control a fraction _f0 of
            # the positions and the rest sits on the composition floor.
            _sufit = _f0 + (1.0 - _f0) * _podloga
            # The same normalised sharpness that `recon_constraint` imposes on
            # `rec_full`. Derived, not chosen.
            _ostrosc = ((float(rc.target) - _podloga)
                        / max(1.0 - _podloga, 1e-9))
            _cel = _podloga + _ostrosc * (_sufit - _podloga)
            self.prefix_controller = ConstraintController(
                target=float(_cel), lam_init=float(pc.lam_init),
                lam_min=float(pc.lam_min), lam_max=float(pc.lam_max),
                lam_lr=float(pc.lam_lr), ema=float(pc.ema),
                saturation_limit=float(pc.saturation_limit),
                warmup_steps=int(pc.warmup_steps),
                name="prefix_0", multipliers={"prefix_0": 1.0})
            print("  prefix constraint: ON | observable %s | ceil_l0 %.4f "
                  "(%.1f bit of %.1f) | composition floor %.4f | CEILING "
                  "%.4f | sharpness %.4f (from rec_full) -> TARGET %.4f | "
                  "config asked for %.4f | today's 0.3201 is %.0f %% of the "
                  "ceiling, headroom +0.02..+0.03 -- NOT a breakthrough"
                  % (pc.observable, _f0, float(_ceil.get("l0_bits", 0.0)),
                     float(_ceil.get("seq_bits", 0.0)), _podloga, _sufit,
                     _ostrosc, _cel, float(pc.target),
                     100.0 * (0.3201 - _podloga)
                     / max(_sufit - _podloga, 1e-9)))
            if float(pc.target) > _sufit:
                raise ValueError(
                    "prefix_constraint.target %.4f lies ABOVE the information "
                    "ceiling %.4f: unreachable at any price, so the "
                    "controller would ride lam_max and the run would measure "
                    "the box rather than the mechanism."
                    % (float(pc.target), _sufit))

        # A third controller: sparsity pressure on the occupancy gate. Left
        # alone, `occ_gate_logit` is a readout rather than a selection, since
        # nothing pays for switching a channel off. The target is computed
        # from the live module rather than copied from the config:
        # `target = 1 - cel_kanalow / C`, with C the actual number of
        # occupancy channels. The config's own `target` goes to the log beside
        # it, so a mismatch is visible rather than guessed at.
        self.occ_gate_controller: Optional[ConstraintController] = None
        ogc = getattr(tc, "occ_gate_constraint", None)
        if ogc is not None and bool(getattr(ogc, "enabled", False)):
            _glog = getattr(self.model.encoder, "occ_gate_logit", None)
            if _glog is None:
                raise ValueError(
                    "training.occ_gate_constraint.enabled = true, but the "
                    "model has no `occ_gate_logit` parameter "
                    "(model.occupancy_gate = false or occupancy_channels "
                    "= 0). The controller has nothing to squeeze and the "
                    "loss term would be a constant.")
            _Cg = int(_glog.numel())
            _celg = 1.0 - float(ogc.cel_kanalow) / float(_Cg)
            if not 0.0 < _celg < 1.0:
                raise ValueError(
                    "occ_gate_constraint.cel_kanalow %.4g over %d channels "
                    "gives a sparsity target of %.4f outside (0, 1): "
                    "unreachable at any price, so the controller would ride "
                    "lam_max and the run would measure the box rather than "
                    "the mechanism."
                    % (float(ogc.cel_kanalow), _Cg, _celg))
            _rz0 = self.model.encoder.occ_gate_sparsity()
            _rz0 = float("nan") if _rz0 is None else float(_rz0.detach())
            # Guard: the gate's starting point must not be one where the term
            # has no usable gradient. L1 always has one, but at an extreme
            # logit sigma' damps it so far that "working weakly" and "not
            # working" stop being distinguishable within an epoch.
            # sigma'(4.0) = 0.0177 against sigma'(2.0) = 0.1050.
            _g0 = torch.sigmoid(_glog.detach().float())
            _sp = float((_g0 * (1.0 - _g0)).mean())
            if _sp < 1e-2:
                raise ValueError(
                    "occupancy_gate_init gives a mean sigma' of %.5f, so "
                    "every gradient reaching the gate is damped %.0fx and "
                    "the gate will not move on the scale of an epoch. "
                    "Measured: at logit 4.0, sigma' = 0.0177 and 60 steps "
                    "did not move `ef` by 0.001 out of 284. Lower "
                    "`model.occupancy_gate_init` (2.0 -> sigma' 0.1050; "
                    "g = 0.881 is an identity up to a constant that "
                    "occ_proj absorbs, exactly as g = 0.982 is)."
                    % (_sp, 1.0 / max(_sp, 1e-9)))
            self.occ_gate_controller = ConstraintController(
                target=float(_celg), lam_init=float(ogc.lam_init),
                lam_min=float(ogc.lam_min), lam_max=float(ogc.lam_max),
                lam_lr=float(ogc.lam_lr), ema=float(ogc.ema),
                saturation_limit=float(ogc.saturation_limit),
                warmup_steps=int(ogc.warmup_steps),
                # The multiplier of 1.0 is a claim, not a filler: the term is
                # `lambda * mean(sigmoid(logit))` and a mean sigmoid lies in
                # (0, 1), so nothing else stands between lambda and the loss.
                # The chain has length one and is printed.
                name="occ_gate", multipliers={"occ_ef": 1.0})
            print("  occupancy gate constraint: ON | observable %s | "
                  "channels %d | cel_kanalow %.1f -> sparsity TARGET %.4f "
                  "(config said %.4f) | sparsity AT START %.4f = ef "
                  "%.1f | sigma_prim %.4f | lambda %.4g in [%.4g, %.4g] "
                  "| lam_lr %.4g"
                  % (str(ogc.observable), _Cg, float(ogc.cel_kanalow),
                     _celg, float(ogc.target), _rz0,
                     (1.0 - _rz0) * float(_Cg), _sp, float(ogc.lam_init),
                     float(ogc.lam_min), float(ogc.lam_max),
                     float(ogc.lam_lr)))
            print("    WHAT THIS TERM DOES NOT PROMISE: the `occ_proj` "
                  "bottleneck destroys nothing (codes carry 98.7 % of motif "
                  "identity, reconstruction from codes alone 0.9973), so the "
                  "gate is NOT a fix for motif sensitivity. The expected PCC "
                  "change is ZERO; this buys SELECTION and INTERPRETABILITY.")
            print("    FALSIFIERS: `ef` must fall below %.0f; `ef_w` (the "
                  "same statistic on g_i*||occ_proj[:, i]||) must fall "
                  "TOGETHER with `ef`, otherwise the gate is playing with "
                  "scale and the selection is only apparent; `sat_hi` > "
                  "%.2f means the target is unreachable, and the answer is "
                  "a LARGER cel_kanalow, not a larger lam_max; `rec_full` "
                  "must not drop below 0.98."
                  % (float(ogc.cel_kanalow),
                     float(ogc.saturation_limit)))

        # Shared parameters the bargain is over
        self.shared = self._shared_parameters(
            tc.shared_scope_for(self.method))
        self.weighting = self._build_weighting()

        self.optimiser = self._build_optimiser()

        self.step_count = 0
        #: How often a player was dropped from a step because its loss was
        #: exactly 0.0. Printed per epoch and in the smoke summary.
        self.zero_gradient_steps: Dict[str, int] = {}
        self.epoch = 0
        self.best: Optional[float] = None

        # The second selection track: spy. One run yields two artefacts,
        # chosen on different splits because they answer different questions.
        # `<name>_fundational.pt` is the best epoch on val, the foundation
        # model meant to be fine-tuned and handed on; `<name>_pks.pt` is the
        # best epoch on spy, the model that ships for the pks1 design task.
        # They need not be the same epoch: val is general promoters, while the
        # deliverable is ten sequences for one gene in one strain, and
        # pairwise accuracy on the corpus cannot see a two-base edit, which
        # every one of the ten designs is. Neither track touches test.
        self.spy_scorer = None
        self.spy_policy = None
        self.spy_best: Optional[float] = None
        #: State of the incumbent, kept so that a challenger can be compared
        #: by paired bootstrap on the same sample, and so a tie is broken on
        #: validation rather than on order of arrival.
        self.spy_best_val: Optional[float] = None
        self.spy_best_raw: Optional[Dict[str, Dict[str, float]]] = None
        self.spy_best_met: Optional[Dict[str, float]] = None
        self.spy_history: List[Dict[str, float]] = []
        #: Why the last epoch was refused a checkpoint, so a run that never
        #: writes one can name the threshold it never met instead of ending
        #: with an absent file and no reason.
        self.last_rejection: str = ""
        self.history: List[EpochResult] = []
        self.orthogonality: Dict[str, object] = {}
        self.step_seconds: Optional[float] = None
        os.makedirs(self.settings.out_dir, exist_ok=True)
        self._build_spy()

    # Spy
    def _build_spy(self) -> None:
        """Attach the spy scorer, or say loudly why there is none.

        A missing `_pks.pt` at the end of a run must be distinguishable from
        "the panel was never loaded"; an absent file that reads like "not
        yet" is the failure mode this guards against.
        """
        sc = getattr(self.cfg.training, "spy_selection", None)
        if sc is None or not bool(getattr(sc, "enabled", False)):
            print("  spy selection: OFF (training.spy_selection.enabled) -- "
                  "this run produces the foundation checkpoint only")
            return
        # A FASTA, not the TSV beside it. `load_spy_panel` parses FASTA
        # headers, so handing it the TSV returns zero records without raising,
        # and an empty panel makes every spy component absent. Absent
        # components are skipped by the geometric mean, so the run would score
        # `rec_full` alone while the config declares six terms.
        path = str(getattr(sc, "panel_path", "") or "")
        if not path:
            path = os.path.join(_repo_root(), "data", "splits",
                                "SPY_SET_V6.fasta")
        if not os.path.isfile(path):
            raise ValueError(
                "training.spy_selection.enabled is true but no panel is at "
                "%r. The loop refuses to run an enabled selection track with "
                "no panel: it would end with no `_pks` checkpoint and no "
                "reason, which is indistinguishable from 'not yet'." % (path,))
        from ..data.spy_panel import load_spy_panel
        from .selection import SelectionPolicy, SpyScorer
        panel = load_spy_panel(path)
        if not panel.records:
            raise ValueError(
                "the spy panel at %r loaded ZERO records. `load_spy_panel` "
                "parses FASTA headers, so a TSV silently yields nothing. An "
                "empty panel does not fail the selection -- it makes every "
                "spy component ABSENT, and absent components are SKIPPED by "
                "the geometric mean, so the run would score `rec_full` alone "
                "and call it the panel score." % (path,))

        # The design strain, looked up by name: `species_id` is the registry
        # row, so an integer in a config would silently re-point at another
        # strain the day the registry gains one.
        want = str(getattr(sc, "strain", "P1") or "P1")
        names = list(self.model.strain_names)
        if want not in names:
            raise ValueError(
                "training.spy_selection.strain=%r is not in the registry %r"
                % (want, names))
        sid = names.index(want)

        # The gene the panel's promoters drive, so the gene token is the one
        # the design asks about rather than the unknown token.
        gnum = None
        og = str(getattr(sc, "orthogroup", "") or "")
        if og and getattr(self.model, "gene_features", None) is not None:
            ix = self.model.gene_features.index()
            if og in ix:
                gnum = self.model.gene_features.numeric[ix[og]]
            else:
                print("  spy selection: orthogroup %r absent from the gene "
                      "table, the panel is scored with the UNKNOWN gene token"
                      % (og,))

        self.spy_scorer = SpyScorer(
            panel, species_id=sid, gene_numeric=gnum,
            seq_len=int(self.cfg.data.sequence_length),
            batch_size=int(self.cfg.data.batch_size),
            # A plasmid is 5.13 sd outside the genomic flank distribution, so
            # the unknown flank token is the honest deployment value.
            use_flanks=False)
        self.spy_policy = SelectionPolicy(
            recon_admit=float(self.cfg.training.recon_admit),
            require_edit_resolution=bool(
                getattr(sc, "require_edit_resolution", True)),
            require_anchor_order=bool(
                getattr(sc, "require_anchor_order", True)),
            require_anchor_promoters=bool(
                getattr(sc, "require_anchor_promoters", True)))
        print("  spy selection: ON | panel %s (%d records) | strain %s (row "
              "%d) | gene %s | components %s"
              % (os.path.basename(path), len(panel.records), want, sid,
                 og or "(unknown token)",
                 ", ".join("%s=%.2f" % (k, v[2])
                           for k, v in self.spy_policy.components.items())))

    def consider_spy_checkpoint(self, val: Dict[str, float]) -> None:
        """Score the panel and keep the best `_pks` checkpoint.

        Takes the whole validation dictionary, not just `rec_full`: a `_pks`
        checkpoint carrying a dozen metric keys instead of all of them left
        the reproducibility check comparing NaN with NaN, so it could neither
        pass nor fail and a missing number looked exactly like agreement.

        `rec_full` still comes from THIS evaluation and is not recomputed: a
        second forward pass disturbs the EMA, and two numbers under one name
        drift apart.
        """
        if self.spy_scorer is None or self.spy_policy is None:
            return
        if not isinstance(val, dict):
            raise TypeError(
                "consider_spy_checkpoint received %r instead of the "
                "validation dictionary. `rec_full` alone was the OLD "
                "signature, and that is why `_pks.pt` had nothing to rebuild "
                "itself from."
                % (type(val).__name__,))
        met, raw = self.spy_scorer.metrics_and_raw(self.model)
        # Order matters: validation goes in first and the panel overwrites.
        # The other way round, a key present in both dictionaries would
        # quietly change what the panel composite scores.
        _val = {k: float(v) for k, v in val.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        _met = dict(_val)
        _met.update(met)
        met = _met
        met["rec_full"] = float(val.get("rec_full", float("nan")))
        met["epoch"] = float(self.epoch)
        # For tie-breaking: without this, the tie branch has nothing to
        # compare two checkpoints by except order of arrival.
        try:
            met["val_composite"] = float(self.selection_score(val))
            met["val_select_metric"] = float(self.selection_metric(val))
        except (KeyError, ValueError) as exc:
            met["val_composite"] = float("nan")
            print("  spy: validation composite not computable (%s) -- a "
                  "tie will have nothing to resolve it" % (exc,))
        self.spy_history.append(dict(met))

        # Which components actually scored. `SelectionPolicy.score` skips an
        # absent or non-finite component and renormalises over what is left,
        # so a missing term does not fail -- it quietly changes what the number
        # means. Reported every epoch.
        present = [k for k in self.spy_policy.components
                   if k in met and np.isfinite(float(met[k]))]
        missing = [k for k in self.spy_policy.components if k not in present]
        if missing:
            print("  spy: components NOT scored this epoch: %s -- the "
                  "geometric mean is over %s only, renormalised"
                  % (", ".join(missing), ", ".join(present) or "NOTHING"))
        verdict = self.spy_policy.judge(met)
        print("  spy: %s" % verdict.format())
        for key in ("spy_scc", "spy_auc", "spy_anchor_fit", "spy_confidence",
                    "spy_anchor_err", "spy_sigma_mean", "spy_calib_slope",
                    "motif_auc_small", "anchor_gap",
                    # Motif contrast raw and after matching on position, with
                    # its own SE and the mean matching distance. Without all
                    # three side by side it is invisible that the raw
                    # contrast mostly measures position, not motif.
                    "motif_contrast_small", "motif_contrast_small_matched",
                    "motif_contrast_small_matched_se",
                    "motif_match_dist_small",
                    "motif_pos_in_small", "motif_pos_out_small"):
            if key in met:
                print("       %-18s %+.5f" % (key, float(met[key])))
        if not verdict.admitted:
            return
        if self.spy_best is not None:
            # A tie is not a win. The panel scores ~45 natives, so the SE of a
            # Spearman on it is ~0.15; an argmax inside that is a coin flip
            # reported as a measurement, and the incumbent keeps the slot.
            n = int(met.get("spy_n", met.get("spy_n_predicted", 0)) or 0)
            # Paired bootstrap over orthogroups, from the raw predictions.
            # `spearman_se(n) = 1/sqrt(n-1)` is the SE of ONE sample under
            # H0, while the question is about the DIFFERENCE of two models on
            # the SAME panel: measured, paired SE 0.0100 against 0.0264.
            _remis = None
            if self.spy_best_raw is not None and self.spy_best_met is not None:
                try:
                    _b = self.spy_policy.bootstrap_tie(
                        self.spy_scorer.panel, raw, self.spy_best_raw,
                        met, self.spy_best_met,
                        n_boot=int(getattr(self.cfg.training,
                                           "spy_tie_boot", 400)),
                        seed=int(self.cfg.training.seed))
                    print("       tie: paired bootstrap over %s (%d units, "
                          "%d draws) | delta_log %+.5f | SE_parowy %.5f | "
                          "SE_nieparowy %.5f (%.2fx) | prog %.5f | "
                          "prog_starym_wzorem %.5f"
                          % (_b["jednostka"], int(_b["n_jednostek"]),
                             int(_b["n_boot"]), _b["delta_log"],
                             _b["se_parowy"], _b["se_nieparowy"],
                             _b["zysk_parowania"], _b["prog"],
                             _b["prog_wzorem"]))
                    _remis = bool(_b["remis"])
                except Exception as exc:                      # noqa: BLE001
                    print("       tie: bootstrap not computed (%s) -- "
                          "falling back to the old formula" % (exc,))
                    _remis = None
            if _remis is None:
                _remis = bool(self.spy_policy.is_tie(
                    verdict.score, self.spy_best, n,
                    q_scc=self.spy_policy.quality(
                        "spy_scc", float(met.get("spy_scc", 0.0)))))
            if _remis:
                # A tie is broken on validation, not on order of arrival. The
                # panel can separate only pairs differing by >= 0.023, and no
                # pair of adjacent epochs (~0.003); an unconditional `return`
                # handed every such pair to the EARLIER epoch, which is not a
                # measurement.
                _teraz = float(met.get("val_composite", float("nan")))
                _inc = (float(self.spy_best_val)
                        if self.spy_best_val is not None else float("nan"))
                if np.isfinite(_teraz) and np.isfinite(_inc) \
                        and _teraz > _inc:
                    print("       TIE on the panel (%.5f against %.5f, "
                          "n=%d) -- VALIDATION DECIDES: composite %.5f > "
                          "%.5f, the challenger wins"
                          % (verdict.score, self.spy_best, n, _teraz, _inc))
                else:
                    print("       TIE with the standing best %.5f (panel "
                          "n=%d cannot resolve it) -- validation cannot "
                          "either (composite %.5f <= %.5f): incumbent kept"
                          % (self.spy_best, n, _teraz, _inc))
                    return
            elif verdict.score <= self.spy_best:
                return
        self.spy_best = verdict.score
        self.spy_best_val = float(met.get("val_composite", float("nan")))
        self.spy_best_raw = raw
        self.spy_best_met = dict(met)
        p = self.save_checkpoint(self.checkpoint_path("pks"), met)
        print("       new best SPY score %.5f -> %s"
              % (verdict.score, os.path.basename(p)))

    # Setup
    def _shared_parameters(self, scope: str) -> List[torch.nn.Parameter]:
        """The parameters the tasks really compete for.

        Passing "all parameters" puts every private head weight into the Gram
        matrix and measures something else; passing too little makes G G^T
        diagonal and degenerates the bargain to GradNorm. The default scope is
        everything in front of the heads -- occupancy, encoder, projections,
        ladder -- which is where this model's capacity is.
        """
        parts = {
            "encoder": [self.model.occupancy, self.model.encoder,
                        self.model.level_proj],
            "latent": [self.model.occupancy, self.model.encoder,
                       self.model.level_proj, self.model.ladder],
        }
        if scope not in parts:
            raise ValueError("training.<method>.shared_scope must be one "
                             "of %r, got %r" % (sorted(parts), scope))
        out: List[torch.nn.Parameter] = []
        seen = set()
        for mod in parts[scope]:
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        if not out:
            raise RuntimeError("shared scope %r selected no trainable "
                               "parameters" % (scope,))
        return out

    def _normprior_kwargs(self) -> Dict[str, object]:
        """Everything `NormPriorWeighting.__init__` reads, from the config.

        The enabled check is here and not only in `HyppeConfig.validate`
        because `--method normprior` overrides the config's method without
        touching the config's blocks, and then the priors of a block that is
        switched off would silently drive the run.
        """
        n = self.cfg.training.normprior
        if not n.enabled:
            raise ValueError(
                "method is 'normprior' (config or --method) while "
                "training.normprior.enabled is false: the priors in a "
                "disabled block would drive the run while reading as inert.")
        return {
            "priors": {str(k): float(v) for k, v in n.priors.items()},
            "refresh_every": int(n.refresh_every),
            "eps": float(n.eps),
        }

    def _nash_kwargs(self) -> Dict[str, object]:
        n = self.cfg.training.nash
        return {
            "n_iter": int(n.n_iter), "min_weight": float(n.min_weight),
            "max_weight": float(n.max_weight),
            "alpha_ema": float(n.alpha_ema), "bypass": bool(n.bypass),
            "clip_threshold": float(n.clip_threshold),
            "nzb_threshold": float(n.nzb_threshold),
            "allow_saturated_players": bool(n.allow_saturated_players),
            "allowed_players": tuple(sorted(self.task_weights)),
        }

    def weighting_factory(self):
        """`(task_weights, params) -> TaskWeighting`, for the wiring gates.

        The gates must build the same class with the same settings the run
        uses: a gate on a differently configured object tests a method the run
        does not have.
        """
        method = self.method
        nash_kwargs = self._nash_kwargs() if method == "nash" else None
        np_kwargs = (self._normprior_kwargs() if method == "normprior"
                     else None)
        terms = self.subordinate_terms()

        def factory(weights, params):
            kw = dict(nash_kwargs) if nash_kwargs else None
            if kw is not None:
                kw["allowed_players"] = tuple(sorted(weights))
            # The gates build this on synthetic task names, and the priors go
            # through unfiltered on purpose: `NormPriorWeighting.prior`
            # defaults an undeclared name to 1.0, so the gates run at uniform
            # shares, which is the configuration whose closed form they hold.
            npk = dict(np_kwargs) if np_kwargs else None
            return build_weighting(method, weights, params,
                                   nash_kwargs=kw, subordinate_terms=terms,
                                   normprior_kwargs=npk)
        return factory

    def _build_weighting(self) -> TaskWeighting:
        return self.weighting_factory()(dict(self.task_weights), self.shared)

    def subordinate_terms(self) -> List[SubordinateTerm]:
        """Every non-player term with its multiplier chain.

        The chain is the point. `commitment` and `tree_prior` are already
        multiplied inside their own modules, so the loop's own coefficient is
        1.0; printing the pair is what stops the next reader from "fixing"
        the loop by multiplying again.
        """
        out = [SubordinateTerm(
            name="reconstruction",
            weight=(self.controller.lam if self.controller is not None
                    else float(self.cfg.training.recon_weight)),
            multipliers=dict(self.recon_view_weights))]
        commit = float(getattr(self.model.ladder.position_quantizer,
                               "commit_weight",
                               self.cfg.training.commitment_weight))
        out.append(SubordinateTerm(name="commitment", weight=1.0,
                                   multipliers={"inside_quantiser": commit}))
        out.append(SubordinateTerm(name="motif", weight=1.0))
        if self.model.genome is not None:
            out.append(SubordinateTerm(
                name="tree_prior", weight=1.0,
                multipliers={"inside_genome":
                             float(self.model.genome.tree_prior)}))
        return out

    def _build_optimiser(self) -> torch.optim.Optimizer:
        tc = self.cfg.training
        groups = [g for g in self.model.param_groups(lr=float(tc.lr),
                                                     weight_decay=float(
                                                         tc.weight_decay))
                  if g["params"]]
        n_manifold = sum(len(g["params"]) for g in groups
                         if g.get("manifold"))
        if tc.riemannian:
            opt = RiemannianAdam(groups, lr=float(tc.lr))
            self._zapamietaj_lr(opt)
            return opt
        if n_manifold and not self.settings.allow_euclidean_manifold:
            raise ValueError(
                "training.riemannian is false while %d tensors live on the "
                "hyperboloid. AdamW would add a Euclidean step to a point of "
                "H^n and the constraint <x,x>_L = -1 drifts silently -- "
                "measured in the sibling track at 5.05e-02 against the 1e-16 "
                "that `assert_on_manifold` accepts, while every headline "
                "number stayed healthy. Pass --allow-euclidean-manifold to "
                "measure that deliberately." % (n_manifold,))
        opt = torch.optim.AdamW(groups, lr=float(tc.lr))
        self._zapamietaj_lr(opt)
        return opt

    # Losses
    def _pair_output(self, scores: torch.Tensor, batch: TrainBatch,
                     z: Optional[Sequence[torch.Tensor]] = None
                     ) -> PairOutput:
        """Build the pair comparison by indexing the per-row scores.

        Not by a second forward: the head is siamese, `pair_score` already is
        `s(z_i, C_i)` for every row, and a second pass would draw a second
        flank dropout for the same condition. `check_pair_glue` measures that
        this construction equals `PairwiseRankingHead.forward` exactly.

        `z`, the per-level codes, turns on the pre-pooling comparison term. It
        is passed here rather than computed in `_heads` because `_heads` does
        not know which rows form a pair: the term is two-argument by
        construction and the pairing is this loop's job. At
        `pair_term_weight: 0.0` it returns zero and `delta` is unchanged bit
        for bit.
        """
        sa = scores[batch.pair_a]
        sb = scores[batch.pair_b]
        delta = sa - sb
        pc = getattr(self.model, "pair_compare", None)
        if z is not None and pc is not None and pc.weight != 0.0:
            za = [t[batch.pair_a] for t in z]
            zb = [t[batch.pair_b] for t in z]
            delta = delta + pc(za, zb)
        temp = float(self.model.pairwise.temperature)
        return PairOutput(score_a=sa, score_b=sb, delta=delta,
                          prob=torch.sigmoid(delta / temp))

    #: Tolerance of the pair-glue check, and not zero for a measured reason:
    #: `score(feats)[rows]` and `score(feats[rows])` are the same linear
    #: algebra over a different batch size, so the matmul reduces in a
    #: different order -- 3.576e-07 in float32 on an untrained model. Exactly
    #: 0.0 would reject a correct implementation, while 1e-5 relative still
    #: rejects a pair built from the wrong rows, which moves it by O(1).
    PAIR_GLUE_TOL = 1e-5

    @torch.no_grad()
    def check_pair_glue(self, out: Dict[str, object], batch: TrainBatch,
                        ctx: Optional[torch.Tensor]) -> float:
        """|indexed score - head.score(that row)| / scale. See PAIR_GLUE_TOL."""
        if self.model.pairwise is None or batch.n_pairs == 0:
            return float("nan")
        # The same features the head used. Under `head_withhold` that is the
        # narrow branch, exported as `fusion_narrow`, not `out["features"]`;
        # comparing the two different tensors made this gate measure something
        # other than what it declares whenever `head_withhold` was non-empty.
        _fw = out.get("fusion_narrow")
        feats = (getattr(_fw, "latent", None)
                 if _fw is not None else None)
        if feats is None:
            feats = out["features"]
        got = out["pair_score"][batch.pair_a]
        ref = self.model.pairwise.score(
            feats[batch.pair_a],
            None if ctx is None else ctx[batch.pair_a])
        scale = float(got.abs().max().clamp_min(1e-12))
        return float((ref - got).abs().max()) / scale

    def compute_losses(self, batch: TrainBatch, track: bool = True
                       ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                  Dict[str, float], Dict[str, object]]:
        """One forward pass -> (players, subordinate, stats, raw output).

        `track=False` also freezes the constraint controller. Feeding it the
        validation reconstruction would fit a Lagrange multiplier -- a trained
        quantity -- on held-out data, and it surfaced as a T5 failure before it
        surfaced as leakage: `recon_lambda` differed by 2.3e-03 between an
        evaluation and its repeat because the first evaluation had moved it.
        """
        model = self.model
        out = model(batch.x, species_id=batch.species_id,
                    flank_l=batch.flank_l, flank_r=batch.flank_r,
                    flank_ok=batch.flank_ok,
                    text_emb=batch.text_emb, text_ok=batch.text_ok,
                    # Only when the model has the channel: the condition
                    # encoder raises on gene inputs it cannot place rather
                    # than dropping them, so a corpus built with the orthology
                    # table and a model built without it fails loudly here
                    # instead of training a silently unconditioned run.
                    gene_numeric=(batch.gene_numeric
                                  if model.gene is not None else None),
                    gene_ok=(batch.gene_ok
                             if model.gene is not None else None),
                    pair_group=batch.pair_group,
                    row_text=batch.row_text, reconstruct=True)
        stats: Dict[str, float] = {
            k: float(v) for k, v in out["stats"].items()
            if isinstance(v, (int, float))}

        players: Dict[str, torch.Tensor] = {}
        if model.absolute is not None and \
                self.task_weights.get("absolute", 0.0):
            # Per-sample regime: target, mask and sigma come from the drawn
            # sample and `out["absolute"].mu` is already [B] because forward
            # was given `row_text`. The row weight goes to None -- `batch.
            # weight` is sqrt(n_obs) of the CELL, the reliability of a mean,
            # and here the target is a single observation.
            _per_row = batch.mu_row is not None
            if _per_row and out["absolute"].mu.dim() != 1:
                raise ValueError(
                    "per-sample regime but the head returned %s: `row_text` "
                    "did not reach it, so the loss would compare a [B] "
                    "target against [B, K] category columns"
                    % (tuple(out["absolute"].mu.shape),))
            # `kara_amplitudy`, the amplitude penalty, lives on the
            # genome-within-orthogroup axis and needs an orthogroup id per
            # batch row; without one the head raises rather than dropping the
            # term it declares. Computed only when the weight is > 0, because
            # `_dopelnij` synchronises with the GPU on every step.
            _w_ka = (model.absolute._waga_kary_amplitudy()
                     if hasattr(model.absolute, "_waga_kary_amplitudy")
                     else 0.0)
            _grupa = None
            if _w_ka > 0.0:
                _ko = self._kod_ortogrup()
                if _ko is None:
                    raise ValueError(
                        "heads.absolute.kara_amplitudy = %.4g, but the "
                        "corpus has no `og` array (`_kod_ortogrup()` = "
                        "None), so the within-orthogroup genome axis cannot "
                        "be assembled. The run would compute a loss without "
                        "the term it declares. Supply a corpus with `og`, or "
                        "set `kara_amplitudy: 0.0`." % (_w_ka,))
                # -1 means "its own group", not a negative index: `ko[-1]`
                # would be the orthogroup of the LAST corpus row, so the
                # negatives would quietly be assigned a real orthogroup. The
                # same arithmetic as `_osie_wiersza`, deliberately, so the
                # loss term and the variance ledger cannot drift apart.
                _ri = self._dopelnij(batch.row_index, len(batch))
                _gnp = np.where(_ri >= 0,
                                _ko[np.clip(_ri, 0, _ko.size - 1)], -1)
                _grupa = torch.as_tensor(
                    _gnp, device=out["absolute"].mu.device)
            loss, parts = model.absolute.loss(
                out["absolute"],
                batch.mu_row if _per_row else batch.mu,
                valid=batch.mu_row_valid if _per_row else batch.mu_valid,
                sigma=batch.sigma_row if _per_row else batch.sigma,
                sigma_valid=(batch.sigma_row_valid if _per_row
                             else batch.sigma_valid),
                weight=None if _per_row else batch.weight,
                grupa=_grupa, track=track)
            players["absolute"] = loss
            stats.update({"abs_" + k: v for k, v in parts.items()})
        if model.pairwise is not None and batch.n_pairs > 0 and \
                self.task_weights.get("pairwise", 0.0):
            po = self._pair_output(out["pair_score"], batch,
                                   z=out.get("z"))
            loss, parts = model.pairwise.loss(
                po, batch.pair_sign, delta_y=batch.pair_delta_y,
                nonpromoter=batch.pair_nonprom, track=track)
            players["pairwise"] = loss
            stats.update({"pair_" + k: v for k, v in parts.items()})
            stats["n_pairs"] = float(batch.n_pairs)
        if model.discriminator is not None and \
                self.task_weights.get("discriminator", 0.0):
            label = batch.is_promoter.to(out["discriminator"].logit.dtype)
            kinds = ["real"] * int(batch.is_promoter.sum().item()) + \
                    list(batch.negative_kind)
            loss, parts = model.discriminator.loss(
                out["discriminator"], label, kind=kinds, one_hot=batch.x,
                track=track)
            players["discriminator"] = loss
            stats.update({"disc_" + k: v for k, v in parts.items()})

        # Subordinate: reconstruction under the constraint, plus the terms
        # whose weight lives inside their own module.
        _widoki_ce: Dict[str, torch.Tensor] = {}
        recon, rec_report = model.reconstruction_loss(
            out, batch.target_idx, per_view=_widoki_ce)
        stats.update({k: float(v) for k, v in rec_report.items()
                      if isinstance(v, (int, float))})
        observable = float(rec_report.get(
            self.cfg.training.recon_constraint.observable, float("nan")))
        if self.controller is not None:
            lam = (self.controller.update(observable) if track
                   else float(self.controller.lam))
        else:
            lam = float(self.cfg.training.recon_weight)
        stats["recon_lambda"] = lam
        stats["recon_effective_full"] = lam * self.recon_view_weights.get(
            "full", 0.0)
        recon_term = lam * recon
        subordinate = recon_term
        # The component list is built FROM the sum, not alongside it: every
        # addend of `subordinate` is appended in the same statement that adds
        # it, so a new term cannot enter the objective without appearing in
        # the log, and if one ever does the closure assertion in `train_step`
        # fails instead of the term hiding.
        comp_names = ["reconstruction"]
        comp_parts = [recon_term.detach().reshape(())]
        # Pressure on the prefix view, with its own lambda and target.
        if self.prefix_controller is not None and "prefix_0" in _widoki_ce:
            _obs0 = float(rec_report.get(
                self.cfg.training.prefix_constraint.observable, float("nan")))
            _lam0 = (self.prefix_controller.update(_obs0) if track
                     else float(self.prefix_controller.lam))
            _term0 = _lam0 * _widoki_ce["prefix_0"]
            subordinate = subordinate + _term0
            comp_names.append("prefix_0")
            comp_parts.append(_term0.detach().reshape(()))
            _w0 = float(self.model.mask.weights.get("prefix_0", 0.0))
            stats["prefix_lambda"] = _lam0
            stats["prefix_obs"] = _obs0
            # The effective coefficient, through every multiplier on the way.
            # The log header used to print an aggregate over three views,
            # which was the coefficient of none of them.
            stats["recon_effective_prefix_0"] = lam * _w0 + _lam0
        # Sparsity pressure on the occupancy gate, with its own lambda and
        # target. No extra forward pass: `occ_gate_sparsity()` reads only the
        # `occ_gate_logit` parameter, so it touches neither the data nor the
        # decoder, the BatchNorms or the EMA.
        if self.occ_gate_controller is not None:
            _rzg = self.model.encoder.occ_gate_sparsity()
            if _rzg is None:
                raise RuntimeError(
                    "the gate controller exists but `occ_gate_sparsity()` "
                    "returned None: the gate parameter vanished between "
                    "building the model and the step. The term would be a "
                    "constant and the log would look identical.")
            _obsg = float(_rzg.detach())
            _lamg = (self.occ_gate_controller.update(_obsg) if track
                     else float(self.occ_gate_controller.lam))
            # The observable and the penalty are deliberately two different
            # quantities: the controller tracks sparsity, but PAYS with the L1
            # term `mean(sigmoid(logit))`, because sparsity has its maximum at
            # the uniform point and a gradient of exactly zero there -- 60
            # steps on `(1 - sparsity)` left `ef` at 284.000 of 284 while
            # lambda climbed. What differentiates the channels is the
            # prediction gradient; L1 only removes what nothing holds up. The
            # falsifier is `ef` TOGETHER WITH `ef_w`, since a uniform L1 does
            # not lower the scale-invariant `ef` at all.
            _karag = self.model.encoder.occ_gate_kara()
            if _karag is None:
                raise RuntimeError(
                    "the gate controller exists but `occ_gate_kara()` "
                    "returned None: the term would be a constant and the log "
                    "would look identical.")
            _termg = _lamg * _karag
            stats["occ_gate_kara"] = float(_karag.detach())
            subordinate = subordinate + _termg
            comp_names.append("occ_gate")
            comp_parts.append(_termg.detach().reshape(()))
            stats["occ_gate_lambda"] = _lamg
            stats["occ_gate_obs"] = _obsg
            stats["occ_gate_cel"] = float(self.occ_gate_controller.target)
            stats["occ_gate_c"] = float(self.occ_gate_controller._last_c)
            stats["occ_gate_sat"] = float(
                self.occ_gate_controller.saturated_fraction())

        # The category head is a subordinate term, not a Nash player: a
        # bargain hands the LARGEST weight to the task with the smallest
        # gradient, and this is an auxiliary head trained on ~20 % of the
        # rows of a batch.
        cat_head = getattr(model, "category", None)
        if (cat_head is not None and out.get("cat_head") is not None
                and batch.cat_target is not None):
            _tu = None
            if bool(getattr(self.cfg.heads.category, "only_dropped_text",
                            True)):
                _tu = out.get("text_used")
                if _tu is None:
                    raise ValueError(
                        "heads.category.only_dropped_text is on but the "
                        "condition encoder returned no per-row text mask, so "
                        "the head would train on rows that HAVE the "
                        "description -- which is the tautology this head "
                        "exists to remove. Refused rather than trained on "
                        "the wrong rows.")
            _cl, _cp = cat_head.loss(
                out["cat_head"], batch.cat_target, text_used=_tu, track=track)
            _cw = float(self.cfg.heads.category.weight)
            _term = _cw * _cl
            subordinate = subordinate + _term
            comp_names.append("category")
            comp_parts.append(_term.detach().reshape(()))
            stats.update({"cat_" + k.replace("cat_", ""): v
                          for k, v in _cp.items()})
        for name, term in out["losses"].items():
            # Weight 1.0 on purpose: these terms are already multiplied inside
            # their own modules (see `subordinate_terms`).
            subordinate = subordinate + term
            comp_names.append(name)
            comp_parts.append(term.detach().reshape(()))
        # One device synchronisation for the whole decomposition.
        for name, value in zip(comp_names, torch.stack(comp_parts).tolist()):
            stats["comp_sub_" + name] = float(value)
            if name != "reconstruction":
                # Kept for continuity with existing logs and parsers.
                stats["term_" + name] = float(value)
        # Encoder scale at the one route onto the hyperboloid. Read straight
        # off the quantiser: `report()` would merge it through a whitelist.
        stats.update(model.runaway_report())
        return players, subordinate, stats, out

    # Step
    def train_step(self, batch: TrainBatch, recompute: bool = True
                   ) -> Dict[str, float]:
        self.model.train()
        self.optimiser.zero_grad(set_to_none=True)
        players, subordinate, stats, _out = self.compute_losses(batch)
        # Condition-path diagnostics: the last (pooled, cond_tokens) pair from
        # TRAINING, detached, so that `format_quantiser_report` can compute a
        # leave-one-channel-out from it without its own forward pass -- every
        # extra forward moves the BatchNorm statistics and the EMA. Cost is
        # ~0.35 MB at B=64, behind the same switch as the report.
        try:
            if (getattr(self.model, "fusion", None) is not None
                    and self.condition_report_enabled()):
                _p, _t = _out.get("pooled"), _out.get("cond_tokens")
                if (isinstance(_p, torch.Tensor)
                        and isinstance(_t, torch.Tensor)):
                    self._cond_probe = (_p.detach(), _t.detach())
        except Exception:                           # never kill the run
            pass
        if not players:
            raise RuntimeError(
                "no player produced a loss on this batch: the update would be "
                "the subordinate terms alone and the log would still report a "
                "step")
        # A hinge satisfied on every pair is exactly zero, and a zero loss has
        # a zero gradient. A bargain equalises utilities, so the player with
        # no gradient would take the MAXIMUM weight; it is dropped from this
        # step, and the drop is counted rather than silent, since a player
        # absent from half the steps is a different experiment. `relu` has
        # subgradient 0 at the hinge in torch, so value and gradient vanish
        # together; for a smooth loss an exact 0.0 in float does not occur.
        zeroed = [n for n, v in players.items() if float(v.detach()) == 0.0]
        for n in zeroed:
            players.pop(n)
            self.zero_gradient_steps[n] = self.zero_gradient_steps.get(n, 0) + 1
            stats["dropped_" + n] = 1.0
        if not players:
            raise RuntimeError(
                "every player's loss is exactly 0.0 on this batch (%r). The "
                "update would be the subordinate terms alone."
                % (sorted(self.zero_gradient_steps),))
        total, alphas = self.weighting.apply(players, subordinate=subordinate,
                                             recompute=recompute)
        gc = float(self.cfg.training.grad_clip)
        if gc > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimiser.param_groups for p in g["params"]],
                gc)
            stats["grad_norm"] = float(norm)
        self._ustaw_lr()
        self.optimiser.step()
        self.step_count += 1
        stats["loss"] = float(total)
        for k, v in alphas.items():
            stats["alpha_" + k] = float(v)
        for k, v in self.weighting.components().items():
            stats["comp_task_" + k] = float(v)
        for k, v in players.items():
            stats["task_" + k] = float(v.detach())
        self.check_closure(stats)
        self.check_divergence(stats)
        return stats

    # Guards
    def components(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Every addend of `loss`, keyed by where it came from.

        Derived from `stats` by prefix: `comp_` keys are written only at the
        two places that build the sum, so the set is closed by construction
        rather than by a list somebody has to remember to extend.
        """
        return {k[len("comp_"):]: v for k, v in stats.items()
                if k.startswith("comp_")}

    def check_closure(self, stats: Dict[str, float]) -> None:
        """Assert that the printed decomposition IS the loss.

        Everything in the sum is printed, so anything not printed breaks this
        assertion and a hidden component costs an exit code rather than forty
        epochs.

        The tolerance is relative to the sum of absolute components. Summing K
        float32 addends carries about K*eps = 3.6e-07 of relative error, so at
        the healthy epoch-1 loss of 7.49 a fixed 1e-6 absolute tolerance would
        fire on a healthy step. Scaling by sum|component| rather than |loss|
        is deliberate: cancellation between large opposite-signed terms sets
        the error, and the total does not see it.
        """
        comps = self.components(stats)
        if not comps:
            raise DivergenceError(
                "no loss components were recorded for step %d. The closure "
                "assertion cannot run, and a guard that cannot run must not "
                "report a pass." % (self.step_count,))
        if not self.weighting.subordinate_included():
            raise DivergenceError(
                "the subordinate terms were computed but NOT added to the "
                "update at step %d: %s. Computing a term, reporting it and "
                "not optimising it is a cost with no effect -- this repo's "
                "recurring failure -- so it stops the run instead."
                % (self.step_count,
                   {k: v for k, v in comps.items() if k.startswith("sub_")}))
        loss = float(stats["loss"])
        got = math.fsum(comps.values())
        scale = max(1.0, abs(loss), math.fsum(abs(v) for v in comps.values()))
        tol = float(self.cfg.training.closure_tol) * scale
        resid = got - loss
        stats["closure_resid"] = resid
        if not math.isfinite(resid) or abs(resid) > tol:
            raise DivergenceError(
                "loss does not equal the sum of its printed components at "
                "step %d: loss %.6e, sum %.6e, residual %.3e > tol %.3e "
                "(rel %.1e x scale %.6e). Components: %s. Either a term was "
                "added to the objective without being recorded where the sum "
                "is built, or one of them is non-finite."
                % (self.step_count, loss, got, resid, tol,
                   self.cfg.training.closure_tol, scale,
                   " ".join("%s=%.6e" % kv
                            for kv in sorted(comps.items()))))

    def check_divergence(self, stats: Dict[str, float]) -> None:
        """Stop on a non-finite loss, a runaway loss, or a runaway radius.

        Three gates in a deliberate order. The radius gate is the early one:
        at a pre-clamp radius of 6 the commitment term is still small, so it
        trips while the loss gate is nowhere near. The loss gate is the
        backstop for every term that is not commitment. The non-finite check
        needs no threshold: there is no batch on which `nan` is a measurement.
        """
        tc = self.cfg.training
        loss = float(stats.get("loss", float("nan")))
        if not math.isfinite(loss):
            raise DivergenceError(
                "loss is %r at step %d (epoch %d). Components: %s"
                % (loss, self.step_count, self.epoch,
                   " ".join("%s=%.6e" % kv
                            for kv in sorted(self.components(stats).items()))))
        if tc.loss_max > 0.0 and loss > tc.loss_max:
            raise DivergenceError(
                "loss %.6e exceeds training.loss_max %.6e at step %d (epoch "
                "%d) -- %.3gx the healthy epoch-1 value of 7.49. The largest "
                "component is %s. A1_dnabert_s7898 ran 40 epochs past this "
                "point and exited 0."
                % (loss, tc.loss_max, self.step_count, self.epoch,
                   loss / 7.49, self._largest_component(stats)))
        raw = stats.get("enc_radius_raw_mean")
        if tc.enc_radius_max > 0.0 and raw is not None \
                and float(raw) > tc.enc_radius_max:
            raise DivergenceError(
                "level-0 encoder radius (PRE-clamp) %.4f exceeds "
                "training.enc_radius_max %.4f at step %d (epoch %d); "
                "|z| mean %.4e max %.4e, clipped fraction %.4f. The latent "
                "works at r0_mean 2.4447, and commitment is 0.25*(cosh(d)-1), "
                "so this is the term that reached 2.662e16 at radius 41.33."
                % (float(raw), tc.enc_radius_max, self.step_count, self.epoch,
                   stats.get("enc_norm_mean", float("nan")),
                   stats.get("enc_norm_max", float("nan")),
                   stats.get("enc_clip_frac", float("nan"))))
        clip = stats.get("enc_clip_frac")
        if tc.enc_clip_frac_max > 0.0 and clip is not None \
                and float(clip) > tc.enc_clip_frac_max:
            raise DivergenceError(
                "%.2f %% of level-0 positions are held at the radius clamp "
                "(model.l0_max_radius) at step %d, above "
                "training.enc_clip_frac_max %.2f %%. On the clamp the radial "
                "gradient is removed and every POST-clamp observable reads a "
                "healthy %.1f while the encoder map keeps growing."
                % (100.0 * float(clip), self.step_count,
                   100.0 * tc.enc_clip_frac_max,
                   float(self.model.ladder.position_quantizer.max_radius)))

    # Formatting
    @staticmethod
    def _g(value: float) -> str:
        """%.4f hides a runaway: 2.662e16 prints as 26620000000000000.0000 and
        1e-30 prints as 0.0000. Both are the cases the log exists for."""
        return "%.4g" % (value,)

    def format_step(self, epoch: int, n: int, stats: Dict[str, float]) -> str:
        """The step line: `loss`, every addend of it, and the residual.

        Ordered largest first, so the term that is running away is the first
        thing on the line rather than the alphabetically lucky one.
        """
        comps = self.components(stats)
        order = sorted(comps, key=lambda k: -abs(comps[k]))
        extra = [k for k in ("rec_full", "recon_lambda",
                             "enc_radius_raw_mean", "enc_clip_frac")
                 if k in stats]
        return ("ep%d step %d | loss %s = %s | resid %.1e | %s"
                % (epoch, n, self._g(float(stats["loss"])),
                   " + ".join("%s %s" % (k, self._g(comps[k]))
                              for k in order),
                   float(stats.get("closure_resid", float("nan"))),
                   " ".join("%s %s" % (k, self._g(float(stats[k])))
                            for k in extra)))

    def format_epoch(self, res: "EpochResult") -> str:
        """The epoch line: the same decomposition, averaged, plus the radius.

        `enc_radius_raw_mean` and `enc_norm_mean` are here because a diverged
        run reached 41.33 and 1.8e18 on them while no line of the log carried
        either. `enc_radius_mean` is the post-clamp value and sits next to the
        raw one on purpose: while the clamp is inactive they are equal, and
        the moment they separate the run is on the bound.
        """
        t, v = res.train, res.val
        comps = {k: t[k] for k in sorted(t) if k.startswith("comp_")}
        order = sorted(comps, key=lambda k: -abs(comps[k]))
        return ("  epoch %d (%.1f s) | train loss %s = %s\n"
                "    latent   enc_norm_mean %s enc_radius_raw_mean %s "
                "enc_radius_mean %s enc_clip_frac %s\n"
                "    val      pcc %.4f scc %.4f pair_acc %.4f auc %.4f "
                "rec_full %.4f | rec_prefix_0 %.4f rec_column %.4f "
                "rec_degree_0 %.4f | ziarna %.0f sel %.5f +- %.5f%s"
                % (res.epoch, res.seconds,
                   self._g(t.get("loss", float("nan"))),
                   " + ".join("%s %s" % (k[len("comp_"):], self._g(comps[k]))
                              for k in order) or "<no components recorded>",
                   self._g(t.get("enc_norm_mean", float("nan"))),
                   self._g(t.get("enc_radius_raw_mean", float("nan"))),
                   self._g(t.get("enc_radius_mean", float("nan"))),
                   self._g(t.get("enc_clip_frac", float("nan"))),
                   v.get("pcc", float("nan")), v.get("scc", float("nan")),
                   v.get("pair_acc", float("nan")),
                   v.get("disc_auc", float("nan")),
                   v.get("rec_full", float("nan")),
                   # `rec_prefix_0` never once appeared in a 36 000-line log
                   # although 0.60 of the view-weight budget works there.
                   # `rec_column` and `rec_degree_0` beside it, because
                   # without them the prefix has nothing to be compared to.
                   v.get("rec_prefix_0", float("nan")),
                   v.get("rec_column", float("nan")),
                   v.get("rec_degree_0", float("nan")),
                   # How many evaluation seeds and how far they spread.
                   # `sel_sd` exactly 0 with more than one seed means the
                   # intervention is broken.
                   v.get("eval_seeds", float("nan")),
                   v.get("sel_mean", float("nan")),
                   v.get("sel_sd", float("nan")),
                   # With a per-cell head `scc` counts cells and is a
                   # different quantity from the single-target runs. The
                   # marginal column is the comparable one and is printed
                   # beside it.
                   ("" if "scc_marginal" not in v else
                    " | MARGINAL scc %.4f pcc %.4f"
                    % (v.get("scc_marginal", float("nan")),
                       v.get("pcc_marginal", float("nan")))))
                + self.format_osie(v) + self.format_ostrosc(t)
                + self.format_bands(t) + self.format_bramy(t)
                + self.format_drabina(t, v))

    def format_osie(self, v: Dict[str, float]) -> str:
        """The per-axis decomposition, as three lines of the epoch report.

        Adding a key to the stats dictionary is not enough to make it
        observable: `format_epoch` builds its lines from a fixed format
        string, so a key absent from that string is invisible however often it
        was computed. Several of these were computed every epoch and reached
        the log only through the checkpoint line, i.e. only in epochs where
        the score improved. The same holds for the other `format_*` helpers.

        `nan` here means "not computed" and is information: on a corpus with
        no `og` table the decomposition does not exist, and the line says so
        rather than vanishing.
        """
        g = lambda k: v.get(k, float("nan"))       # noqa: E731
        return ("\n    osie     ortogrupa r2 %.4f nachyl %.3f pcc %.3f | "
                "GENOM r2 %+.4f nachyl %.3f pcc %.3f r2_kal %.4f "
                "nadmiar %.2fx | warunek r2 %+.4f nachyl %.3f "
                "(gl %+.4f int %+.4f)"
                "\n    wziecia  ortogrupa %.4f warunek %.4f genom_netto "
                "%.4f reszta %.4f var_y %.4f | os_genomu_war %.4f | "
                "domkn %.1e /gen %.1e nieortog %.1e | n %.0f og %.0f "
                "gen %.0f kat %.0f bez_og %.4f"
                "\n    readout   sd_p/(pcc*sd_t) %.4f pred_sd %.4f "
                "target_sd %.4f | pcc_cell %.4f scc_cell %.4f | "
                "cat_gain %+.4f cat_acc %.4f cat_floor %.4f"
                % (g("os_ortogrupa_r2"), g("os_ortogrupa_nachylenie"),
                   g("os_ortogrupa_pcc"),
                   g("os_genomu_r2"), g("os_genomu_nachylenie"),
                   g("os_genomu_pcc"), g("os_genomu_r2_kal"),
                   g("os_genomu_nadmiar"),
                   g("os_warunku_r2"), g("os_warunku_nachylenie"),
                   g("os_warunku_glowny_r2"), g("os_warunku_interakcja_r2"),
                   g("os_wziecie_ortogrupa"), g("os_wziecie_warunek"),
                   g("os_wziecie_genom_netto"), g("os_reszta_var"),
                   g("os_var_y"), g("os_genomu_war"),
                   g("os_domkniecie"), g("os_domkniecie_genom"),
                   g("os_nieortogonalnosc"),
                   g("os_n"), g("os_n_ortogrup"), g("os_n_genomow"),
                   g("os_n_kategorii"), g("os_bez_ortogrupy"),
                   g("kalibracja_sd"), g("pred_sd"), g("target_sd"),
                   g("pcc_cell_readout"), g("scc_cell_readout"),
                   g("cat_gain"), g("cat_acc"), g("cat_floor")))

    def format_ostrosc(self, t: Dict[str, float]) -> str:
        """The sharpening line; empty string when the module is absent.

        `udzial` is the RMS of the correction against the RMS of `pooled`. It
        starts at exactly 0 because the output is zero-initialised, so a 0
        after two epochs means the branch never woke -- a different failure
        from `sd_w` being zero, which means there is nothing to sharpen inside
        the L0 window. `sd_w` against `sd_m` says how much of the level's
        variation sits within a window and how much between windows; pooling
        sees only the latter.
        """
        if not any(k.startswith("ostr_") for k in t):
            return ""
        g = lambda k: t.get(k, float("nan"))          # noqa: E731
        poz = sorted(k[len("ostr_sdw_"):] for k in t
                     if k.startswith("ostr_sdw_"))
        return ("\n    sharpness  udzial %.5f rms %.3e w_norma %.3e "
                "u_rms %.3e | %s"
                % (g("ostr_udzial"), g("ostr_rms"), g("ostr_w_norma"),
                   g("ostr_u_rms"),
                   " ".join("%s sd_w %.3f sd_m %.3f"
                            % (p, g("ostr_sdw_" + p), g("ostr_sdm_" + p))
                            for p in poz)))

    def format_drabina(self, t: Dict[str, float],
                       v: Dict[str, float]) -> str:
        """Coding overhead `r` and the falsifiers of the residual predictor.

        `r` is the sum of achieved code entropies over the 1598.6 bits of the
        sequence, and must be read TOGETHER with `rec_full`: r falls trivially
        when the code dies, so only the pair -- a point on the rate-distortion
        curve -- means anything.

        `pred_r2` is how much the predictor removes from the feature; the
        threshold is more than the linear 0.2043 / 0.0385 / 0.0424 / 0.0658
        obtained by regressing on the single parent point, and below that the
        predictor has collapsed to zero. `udzial` is rms(pred)/rms(feature),
        and a drift towards 1.0 with a falling `uzyte` is the other collapse:
        the predictor becomes the feature, the residual is noise and the
        codebook dies.
        """
        if "r_code" not in t and "bits_total" not in t:
            return ""
        g = lambda k: t.get(k, float("nan"))          # noqa: E731
        lv = sorted((k for k in t if k.startswith("bits_total_l")),
                    key=lambda k: int(k.rsplit("_l", 1)[1]))
        uf = sorted((k for k in t if k.startswith("used_frac_l")),
                    key=lambda k: int(k.rsplit("_l", 1)[1]))
        rb = sorted((k for k in t if k.startswith("rate_bits_l")),
                    key=lambda k: int(k.rsplit("_l", 1)[1]))
        out = ("\n    drabina  r %.4f (bity %.1f / sekw %.1f) | rec_full "
               "%.4f | bity/poziom %s | uzyte %s"
               % (g("r_code"), g("bits_total"), g("sequence_bits"),
                  v.get("rec_full", float("nan")),
                  " ".join("%.0f" % g(k) for k in lv) or "<missing>",
                  " ".join("%.3f" % g(k) for k in uf) or "<missing>"))
        if rb:
            out += " | tempo %s" % " ".join("%.2f" % g(k) for k in rb)
        if t.get("pred_on", 0.0):
            r2 = sorted((k for k in t if k.startswith("pred_r2_l")),
                        key=lambda k: int(k.rsplit("_l", 1)[1]))
            sh = sorted((k for k in t if k.startswith("pred_share_l")),
                        key=lambda k: int(k.rsplit("_l", 1)[1]))
            out += ("\n    predRez  waga %.3f | r2 %s (prog liniowy 0.2043 "
                    "0.0385 0.0424 0.0658) | udzial %s"
                    % (g("pred_weight"),
                       " ".join("%.4f" % g(k) for k in r2) or "<missing>",
                       " ".join("%.3f" % g(k) for k in sh) or "<missing>"))
        return out

    def format_bands(self, t: Dict[str, float]) -> str:
        """The L0 band line; empty string when the module is absent.

        `udzial` is rms(correction) / rms(`pooled` before both modules), and
        should leave its zero start by epoch 2; for scale, sharpening went
        0.166 to 0.403 over seven epochs in the same family. `sd_miedzy` is
        the sd of the band means across bands, and zero means every band is
        equal, so the module carries the same thing as an `L0/mean` slice and
        position is absent -- a different failure from a sleeping branch.
        `prof` is the mean |weight| of the output columns falling on each
        band; mass on the LAST band means the module reproduced `anchor` and
        nothing more, so we paid for a reparametrisation.
        """
        if not any(k.startswith("pasma_") for k in t):
            return ""
        g = lambda k: t.get(k, float("nan"))          # noqa: E731
        prof = sorted((k for k in t if k.startswith("pasma_prof_")),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
        return ("\n    pasmaL0  udzial %.5f rms %.3e w_norma %.3e "
                "sd_miedzy %.3e | prof %s"
                % (g("pasma_udzial"), g("pasma_rms"), g("pasma_w_norma"),
                   g("pasma_sd_miedzy"),
                   " ".join("%.3e" % g(k) for k in prof) or "<missing>"))

    def format_bramy(self, t: Dict[str, float]) -> str:
        """The occupancy-gate line; empty string when there is no gate.

        `ef` is the effective number of occupancy channels let through; for
        scale, an unconstrained run read 280.6 of 284.

        Read `ef` only beside `ef_w`. `ef` is invariant to a rescaling of the
        gate, so the gate can close on channel i while `occ_proj` rebuilds the
        signal by raising column W[:, i]: `ef` falls and the bottleneck
        receives the same thing. `ef_w` is the same statistic on the product
        g_i*||W[:, i]||, and selection is real only when both fall.

        The `lam` segment exists only when the controller is on. Without it
        the gate is under no sparsity pressure, so `ef` at the channel count
        means "there was no reason to" rather than "all of them are needed":
        a readout, not a selection.
        """
        if "occ_gate_ef" not in t:
            return ""
        g = lambda k: t.get(k, float("nan"))          # noqa: E731
        _linia = ("\n    bramaOcc ef %.1f z %.0f | ef_w %.1f | g srednia "
                  "%.4f min %.4f max %.4f | ponizej_0,1 %.4f | masa44 %.4f "
                  "(podloga %.4f)"
                  % (g("occ_gate_ef"), g("occ_gate_kanaly"),
                     g("occ_gate_ef_w"),
                     g("occ_gate_srednia"), g("occ_gate_min"),
                     g("occ_gate_max"), g("occ_gate_ponizej_01"),
                     g("occ_gate_masa44"),
                     (44.0 / g("occ_gate_kanaly"))
                     if g("occ_gate_kanaly") == g("occ_gate_kanaly") else
                     float("nan")))
        # Every observable of the controller, explicitly.
        if "occ_gate_lambda" in t:
            _linia += ("\n    bramaCtl rzadkosc %.4f -> cel %.4f | c %+.5f "
                       "| lambda %.4f w [%.4g, %.4g] | sat %.3f | karaL1 "
                       "%.4f | czlon %.5f | ef_cel %.0f"
                       % (g("occ_gate_obs"), g("occ_gate_cel"),
                          g("occ_gate_c"), g("occ_gate_lambda"),
                          self.cfg.training.occ_gate_constraint.lam_min,
                          self.cfg.training.occ_gate_constraint.lam_max,
                          g("occ_gate_sat"), g("occ_gate_kara"),
                          t.get("comp_sub_occ_gate", float("nan")),
                          self.cfg.training.occ_gate_constraint.cel_kanalow))
        # The profile list is appended here rather than printed separately,
        # for the same reason: the epoch line is built explicitly.
        _nz = self._brama_nazwy()
        if _nz:
            _linia += _nz
        return _linia

    def _brama_nazwy(self, k: int = 24) -> str:
        """The profiles the gate lets through and the ones it suppresses.

        `ef` says how many channels survive, not which, and the deliverable of
        this gate is a list naming the motif profiles that matter.

        The channel order is a contract: `MotifOccupancy.forward` builds
        `cat([frozen(x), de_novo(x)])`, so channels 0..n_frozen-1 are the MEME
        profiles in file order and the rest are de novo filters. Were that
        order to change the list would name the wrong profiles silently, which
        is why a mismatch between the number of names and the number of gates
        raises rather than printing.

        What the list does not show, and must travel with it: a separate
        measurement found 7 profiles of 252 above their own null against
        12.6 +- 3.5 expected by chance, so the bank has no measurable motif
        activity on *Trichoderma* promoters. A profile the gate selects is one
        the MODEL USES, not one that OCCURS in these promoters.
        """
        g = getattr(self.model.encoder, "occ_gate_logit", None)
        if g is None:
            return ""
        occ = getattr(self.model, "occupancy", None)
        nazwy = []
        if occ is not None and getattr(occ, "frozen", None) is not None:
            nazwy.extend(list(occ.frozen.names))
        n_meme = len(nazwy)
        if occ is not None and getattr(occ, "de_novo", None) is not None:
            nazwy.extend("deNovo%02d" % i
                         for i in range(int(occ.de_novo.n_filters)))
        if len(nazwy) != int(g.numel()):
            raise RuntimeError(
                "%d channel names against %d gates: the occupancy channel "
                "order has stopped being the cat([frozen, de_novo]) "
                "contract, so the listing would name the wrong profiles."
                % (len(nazwy), int(g.numel())))
        with torch.no_grad():
            s = torch.sigmoid(g.detach().float()).cpu()
        kol = torch.argsort(s, descending=True).tolist()
        cel = int(self.cfg.training.occ_gate_constraint.cel_kanalow)
        cel = max(1, min(cel, len(kol)))
        # The two ends are taken from disjoint halves: `kol[-k:]` returns the
        # whole list below 2k channels, so SUPPRESSES printed the PASSES line
        # again and the gate read as suppressing what it passes.
        n_gora = min(k, (len(kol) + 1) // 2)
        gora = kol[:n_gora]
        dol = kol[n_gora:][-k:]
        ile_meme = sum(1 for i in kol[:cel] if i < n_meme)
        _f = lambda i: "%s:%.3f" % (nazwy[i], float(s[i]))       # noqa: E731
        return ("\n    bramaProfil PASSES     %s"
                "\n    bramaProfil SUPPRESSES %s"
                "\n    bramaProfil in the %d highest gates: %d MEME profiles "
                "of %d and %d de novo filters of %d"
                % (" ".join(_f(i) for i in gora),
                   " ".join(_f(i) for i in dol),
                   cel, ile_meme, n_meme, cel - ile_meme,
                   len(nazwy) - n_meme))


    # Quantiser diagnostics: everything the ladder and the two quantiser
    # classes already compute on every forward and that no line of the log
    # ever carried. Grepping a 30-epoch production log for `saturation`,
    # `bit_fraction`, `perplexity` or `dead_restarts` returned zero hits, so
    # the run's flagship mechanism was unobservable in its own artefact. The
    # numbers were never missing, only unrendered.
    LADDER_STAT_KEYS = (
        "bits_total", "bits_target_total", "manifold_err", "radius",
        "radius_plan", "radius_gap", "radius_step_measured",
        "aperture_use", "step_norm_in", "step_norm_out",
        "step_clamped_frac", "outside_cone", "angle_max", "angle_ceiling",
        "cone_can_fire", "d_ratio", "crossover", "step_spread",
        "null_frac", "prefix_ready", "missing_report_keys",
        "report_errors")
    QUANTISER_STAT_KEYS = (
        "perplexity", "used_frac", "used_codes", "used_cells",
        "entropy_bits", "entropy_ceiling_bits", "mode_mass",
        "dead_restarts", "dead_restarts_last", "radius_mean", "radius_sd",
        "manifold_error", "wrong_sheet", "dist_ratio", "enc_to_code_dist",
        "code_dist_median", "bits", "target_bits", "bit_fraction",
        "marginal_bits", "redundancy_bits", "effective_cells",
        "saturation", "std_in", "gain", "zero_frac", "expected_zero_frac",
        "zero_frac_ratio", "aperture_use", "quant_error", "entropy_weight",
        "entropy_term", "clamped")
    #: `LatentConditioner._diag`, in a fixed order. The per-channel
    #: `fuse_gate_*` and `fuse_code_rms_*` go on their own line, because how
    #: many of them there are depends on the number of condition channels.
    FUSION_STAT_KEYS = (
        "fuse_gamma_mean", "fuse_gamma_sd", "fuse_beta_rms",
        "fuse_shift_ratio", "fuse_code_rms", "fuse_code_pr",
        "fuse_bottleneck_dim",
        # The last five decompose the shift, which `fuse_shift_ratio` above
        # does not: that one is movement after the damper relative to the
        # latent's own norm. `fuse_udzial_przes` is the energy before the
        # damper and `..._h` after it, and the two need not agree because the
        # damper removes a common scale; `fuse_lat_rms` is the denominator in
        # latent scale and `fuse_mult_rms` the multiplicative numerator.
        # `fuse_beta_skala` is 0/1 for whether the flag ARRIVED -- without it,
        # "flag on" and "flag never reached the module" look identical.
        "fuse_udzial_przes", "fuse_udzial_przes_h",
        "fuse_lat_rms", "fuse_mult_rms", "fuse_beta_skala")
    #: `ConditionEncoder._diag`. The list is not closed -- anything starting
    #: with `ktx_` reaches the line anyway; this tuple only fixes the order of
    #: the keys we already know, so the log can be parsed by column.
    CONDITION_STAT_KEYS = (
        "ktx_udzial_gat", "ktx_udzial_tekst", "ktx_udzial_flankaL",
        "ktx_udzial_flankaR", "ktx_udzial_gen",
        "ktx_tekst_pokrycie", "ktx_flanka_pokrycie",
        "ktx_tekst_dropout", "ktx_species_dropout",
        "ktx_tekst_dev_nad_srednia")

    @staticmethod
    def _stat(value: object) -> str:
        """Render a report value without hiding a runaway, a flag or a list.

        `%.4f` is refused here for the reason given at `_g`. Lists are
        printed whole: these observables are per level, and a mean over
        levels would hide the one level that collapsed.
        """
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return "%.4g" % (value,)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(TrainLoop._stat(v) for v in value) + "]"
        return str(value)

    def format_quantiser_report(self) -> List[str]:
        """The ladder and every quantiser, once per epoch, from TRAINING.

        Must be called before `evaluate()`, for two independent reasons.
        `evaluate()` calls `reset_statistics(self.model)`, so read after it
        `entropy_bits`, `mode_mass`, `dead_restarts` and `steps` describe a
        counter zeroed a moment earlier. And even for the keys that survive a
        reset, the numbers would describe the eval regime: a report read after
        evaluation once had 62 of its 134 keys measured under eval and read as
        training numbers, and `BatchNorm` on the path makes train and eval two
        different models.

        Never raises. A diagnostic that can abort a 4.5 hour job is worse than
        no diagnostic, so the body is wrapped and the exception is printed.
        Every key is read with `.get`, so a rename inside a quantiser removes
        the number loudly, as a named `absent` entry, rather than the line.
        """
        lines: List[str] = []
        try:
            ladder = getattr(self.model, "ladder", None)
            if ladder is None:
                return ["  ladder   <no ladder on this model>"]
            stats = dict(getattr(ladder, "_stats", None) or {})
            shown = [(k, stats[k]) for k in self.LADDER_STAT_KEYS
                     if stats.get(k) is not None]
            absent = [k for k in self.LADDER_STAT_KEYS if k not in stats]
            lines.append("  ladder   " + (" ".join(
                "%s %s" % (k, self._stat(v)) for k, v in shown)
                or "<ladder._stats is empty: no forward has run>"))
            if absent:
                lines.append("  ladder   absent %s" % (",".join(absent),))
            mods = [("L0.position",
                     getattr(ladder, "position_quantizer", None))]
            for i, q in enumerate(
                    list(getattr(ladder, "tangent_quantizers", []) or [])):
                mods.append(("L%d.fsq" % (i + 1), q))
            with torch.no_grad():
                for label, module in mods:
                    if module is None:
                        lines.append("  %-12s <absent>" % (label,))
                        continue
                    fn = getattr(module, "report", None)
                    if not callable(fn):
                        lines.append("  %-12s <no report()>" % (label,))
                        continue
                    try:
                        rep = fn()
                    except Exception as exc:        # never kill the run
                        lines.append("  %-12s report() raised %s: %s"
                                     % (label, type(exc).__name__, exc))
                        continue
                    rep = rep if isinstance(rep, dict) else {}
                    got = [(k, rep[k]) for k in self.QUANTISER_STAT_KEYS
                           if rep.get(k) is not None]
                    lines.append("  %-12s %s" % (label, " ".join(
                        "%s %s" % (k, self._stat(v)) for k, v in got)
                        or "<report() returned no known key>"))
        except Exception as exc:                    # never kill the run
            lines.append("  quantiser report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        # The encoder's own falsifiers. `context_dom_raw` is the one that
        # killed a run (1.85 -> 22.64 over ten epochs); the encoder's
        # threshold for the effective ratio is 0.5. Also carries the occupancy
        # scale spread, the stated risk of replacing the per-sample z-score
        # with a fixed per-profile scale.
        try:
            _enc = getattr(self.model, "encoder", None)
            _fn = getattr(_enc, "encoder_report", None)
            if callable(_fn):
                _er = _fn() or {}
                if isinstance(_er, dict) and _er:
                    lines.append("  encoder      " + " ".join(
                        "%s %s" % (k, self._stat(_er[k]))
                        for k in sorted(_er) if _er[k] is not None))
                else:
                    lines.append("  encoder      <report returned nothing>")
            else:
                lines.append("  encoder      <no encoder_report()>")
        except Exception as exc:                    # never kill the run
            lines.append("  encoder      report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        # The de novo motif bank. Without this the run produces no number
        # about that bank except the penalty, and a penalty near zero means
        # either "the conditions are met" or "the filters are dead" -- two
        # opposite states behind one number. Read from the last TRAINING
        # forward, which is what sets `_last_density` and `_last_max_act`.
        try:
            _occ = getattr(self.model, "occupancy", None)
            _dn = getattr(_occ, "de_novo", None) if _occ is not None else None
            _fr = getattr(_occ, "frozen", None) if _occ is not None else None
            if _dn is None:
                lines.append("  motifs_dn    <de novo bank missing>")
            else:
                with torch.no_grad():
                    _mr = _dn.report(_fr) or {}
                    lines.append("  motifs_dn    " + " ".join(
                        "%s %s" % (k, self._stat(_mr[k]))
                        for k in sorted(_mr) if _mr[k] is not None))
                    _zle = [(n, d) for n, ok, d in _dn.falsifiers(_fr)
                            if not ok]
                    for _n, _d in _zle:
                        lines.append("  motifs_dn    FALSIFIER FAILED: "
                                     "%s -- %s" % (_n, _d))
                    if not _zle:
                        lines.append("  motifs_dn    falsifiers: "
                                     "all passed")
        except Exception as exc:                    # never kill the run
            lines.append("  motifs_dn    report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        # The condition path. Without this block the run produces no number
        # about conditioning at all, so every change to it was unfalsifiable
        # in a run -- and the condition path is the only one that can separate
        # two rows carrying byte-identical sequence, which is 50.5 % of the
        # labelled corpus. Inherits the "before `evaluate()`" requirement.
        try:
            lines.extend(self._condition_report_lines())
        except Exception as exc:                    # never kill the run
            lines.append("  cond_        report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        # The effective receptive field per level. Deliberately last: it is
        # the only block of the report that runs its own forward pass, so
        # higher up it would overwrite the `_diag` of the encoder, the
        # occupancy and the motif bank with the eval regime on eight
        # validation sequences.
        try:
            lines.extend(self._erf_raport())
        except Exception as exc:                    # never kill the run
            lines.append("  erf          report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        return lines

    # Effective receptive field
    #: Output positions at which the field is measured -- fractions of the map
    #: length, away from the edges, where the field is truncated and the
    #: measurement would describe the padding rather than the model.
    ERF_FRAKCJE = (0.3, 0.5, 0.7)
    #: How many validation sequences. Fixed once per run (`self._erf_x`), so
    #: the trajectory across epochs describes the model, not the batch.
    ERF_N = 8

    def _erf_jednej_mapy(self, mk, xx, seq_len):
        """ERF50 and ERF90 of one map, in base pairs.

        Gradient saliency (Luo 2016): for output position p, take
        d(sum |map[:, :, p]|) / dx over the one-hot input, sum |grad| over
        channels and batch, and read off a profile over the 800 positions.
        r50/r90 is the radius containing 50 % / 90 % of the mass around the
        centre of the field, and ERF = 2r.
        """
        import numpy as _np
        if mk is None or mk.dim() != 3:
            return float("nan"), float("nan")
        Lk = int(mk.shape[-1])
        if Lk < 2:
            return float("nan"), float("nan")
        skok = float(seq_len) / float(Lk)
        r50, r90 = [], []
        for f in self.ERF_FRAKCJE:
            p = int(round(float(f) * (Lk - 1)))
            g, = torch.autograd.grad(mk[:, :, p].abs().sum(), xx,
                                     retain_graph=True)
            gg = g.detach().abs().sum(dim=(0, 1)).double().cpu().numpy()
            masa = float(gg.sum())
            if not (masa > 0.0):
                continue
            c = (p + 0.5) * skok
            idx = _np.arange(gg.shape[0], dtype=_np.float64)
            d = _np.abs(idx - c)
            kol = _np.argsort(d)
            cs = _np.cumsum(gg[kol]) / masa
            dk = d[kol]
            i50 = min(int(_np.searchsorted(cs, 0.50)), dk.shape[0] - 1)
            i90 = min(int(_np.searchsorted(cs, 0.90)), dk.shape[0] - 1)
            r50.append(float(dk[i50]))
            r90.append(float(dk[i90]))
        if not r50:
            return float("nan"), float("nan")
        return (2.0 * float(sum(r50) / len(r50)),
                2.0 * float(sum(r90) / len(r90)))

    def _erf_raport(self):
        """Effective receptive field of each encoder level, once per epoch.

        `rf_bp_by_level` and `rf_span_ratio` are integer arithmetic over the
        assembled modules, so they print the same list byte for byte in every
        epoch: a number known before epoch 1 is a receipt from assembly, not a
        falsifier. This measures a quantity that CAN fall, and on one run's
        checkpoints the finest level's ERF90 grew 5.9x between initialisation
        and epoch 8, so it does move.

        The instrument checks itself: the gradient mass outside the nominal
        receptive field is 0.0000 on L1-L4 and 0.0402 on L0 -- zero wherever
        convolution forbids it, non-zero on exactly the one level with a
        global path.

        Cost is 18 backward passes on a fixed batch, once per epoch. The model
        goes into `eval()` for the duration; every floating-point buffer is
        cloned and restored, with `erf_bufory_delta` reporting how much had to
        be put back, and the CPU and GPU RNG states are saved and restored so
        the training stream is bit for bit what it would be without this line.
        """
        m = self.model
        enc = getattr(m, "encoder", None)
        if enc is None:
            return ["  erf          <missing enkodera>"]
        x = getattr(self, "_erf_x", None)
        if x is None:
            for _b in self.maker.epoch("val", shuffle=False, seed=0,
                                       max_steps=1, with_negatives=False):
                x = _b.x[:int(self.ERF_N)].float().to(self.device)
                break
            if x is None:
                return ["  erf          <missing batcha walidacji>"]
            x = x.contiguous()
            self._erf_x = x
        seq = float(x.shape[-1])
        bufory = [(b, b.detach().clone()) for b in m.buffers()
                  if b.dtype.is_floating_point]
        stan_cpu = torch.get_rng_state()
        stan_gpu = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        byl_train = bool(m.training)
        e50, e90, podloga = [], [], None
        try:
            m.eval()
            with torch.enable_grad():
                xx = x.clone().requires_grad_(True)
                zlap = {}
                hak = None
                pre = getattr(enc, "pre", None)
                if pre is not None and len(pre) > 0:
                    hak = pre[-1].register_forward_hook(
                        lambda _mo, _a, o: zlap.__setitem__("pre", o))
                try:
                    occ = m.motif_occupancy(xx)
                    mapy = enc(xx, occupancy=occ, context=None)
                finally:
                    if hak is not None:
                        hak.remove()
                if isinstance(mapy, dict):
                    mapy = mapy.get("levels", list(mapy.values()))
                mapy = list(mapy)
                for mk in mapy:
                    if isinstance(mk, (list, tuple)):
                        mk = mk[-1]
                    a, b2 = self._erf_jednej_mapy(mk, xx, seq)
                    e50.append(a)
                    e90.append(b2)
                if "pre" in zlap:
                    podloga = self._erf_jednej_mapy(zlap["pre"], xx, seq)
                del xx
        finally:
            delta = 0.0
            for b, kopia in bufory:
                try:
                    d = float((b.detach().double()
                               - kopia.double()).abs().max())
                except Exception:                       # noqa: BLE001
                    d = float("nan")
                if d == d and d > delta:
                    delta = d
                b.detach().copy_(kopia)
            torch.set_rng_state(stan_cpu)
            if stan_gpu is not None:
                torch.cuda.set_rng_state_all(stan_gpu)
            m.train(byl_train)
        rap = {}
        rap["erf90_by_level"] = [round(v, 1) for v in e90]
        rap["erf50_by_level"] = [round(v, 1) for v in e50]
        il = [(e90[k] / e90[k + 1]) for k in range(len(e90) - 1)
              if e90[k + 1] > 0.0]
        rap["erf_ratio_by_level"] = [round(v, 3) for v in il]
        rap["erf_ratio_min"] = round(min(il), 3) if il else 0.0
        rap["erf90_finest"] = round(e90[-1], 1) if e90 else 0.0
        if podloga is not None:
            rap["erf90_floor"] = round(podloga[1], 1)
            # How much of the finest level's effective field is the shared
            # stem -- the output of the pre-blocks, before the first stride,
            # inherited bit for bit by every level. One run read 0.86 on L4
            # and 0.72 on L1.
            rap["erf_floor_frac_finest"] = (
                round(podloga[1] / e90[-1], 3)
                if (e90 and e90[-1] > 0.0) else 0.0)
        rap["erf_bufory_delta"] = delta
        rap["erf_n"] = int(x.shape[0])
        out = ["  erf          " + " ".join(
            "%s %s" % (k, self._stat(rap[k])) for k in sorted(rap))]
        # The verdict goes in the log, not in a comment: silence must not be
        # equivalent to a passed gate.
        prog = 1.5
        if il:
            ok = min(il) >= prog
            out.append("  erf          FALSIFIER erf_ratio_min %.3f %s "
                       ">= %.2f | the calibration behind this threshold "
                       "(one instrument throughout, 8 validation sequences, "
                       "EVAL, before epoch 1): baseline 1.035 (ERF90 100.7 "
                       "against 97.3 at neighbouring levels, i.e. the levels "
                       "are copies IN USE) | 1.267 | 1.200 | 1.800 with "
                       "skip_gate0 0.3 + level_gate0 0.9 + level_kernel 5. "
                       "From that configuration on, the threshold is met by "
                       "construction, so the falsifier is the DIRECTION: "
                       "erf_ratio_by_level must rise on the three upper "
                       "pairs and the minimum must not fall to 1.20. The "
                       "L3/L4 pair is structurally bounded: "
                       "erf_floor_frac_finest ~1.0 means the finest level IS "
                       "the shared floor"
                       % (min(il), "ok" if ok else "FAILED", prog))
        if delta > 0.0:
            out.append("  erf          FALSIFIER erf_bufory_delta %.3e "
                       "FAILED > 0: the measurement moved a buffer "
                       "(BatchNorm/EMA) and it was restored, but the report "
                       "lines above may already have seen it" % (delta,))
        return out

    # The condition path
    def condition_report_enabled(self) -> bool:
        """`training.condition_report` (on by default), or `HYPPE_COND_REPORT`.

        Read through `getattr` because the key need not exist in
        `TrainingConfig`: this switches printing, not the model's behaviour.
        The model is bit for bit the same in both positions -- nothing below
        writes a parameter, a buffer, a running statistic or the global RNG.
        """
        env = os.environ.get("HYPPE_COND_REPORT")
        if env is not None and env.strip() != "":
            return env.strip().lower() not in ("0", "false", "no", "off")
        return bool(getattr(getattr(self, "cfg", None), "training", None)
                    and getattr(self.cfg.training, "condition_report", True))

    def _condition_report_lines(self) -> List[str]:
        """The condition path, once per epoch, from training. Never raises.

        Four lines, each of which says loudly when it has nothing:
        `cond_fuse` is `LatentConditioner._diag` from the last training
        forward; `cond_eff` is the leave-one-channel-out `channel_effect` on
        the remembered (pooled, cond_tokens) pair, plus a falsifier on
        `bottleneck.null`; `cond_ktx` is `ConditionEncoder._diag`; and
        `cond_attn` is `cond_H` / `cond_share` of the encoder's cross layer,
        or an explicit "missing" with a reason, since in this model
        `encoder.cond is None` and those keys do not exist at all.

        `ConditionEncoder.falsifiers()` is deliberately NOT called: under
        `cat_head_mode='prototype'` it adds a `__probe__` category to the live
        encoder, which is mutating the model mid-run for one line of log. Its
        two rows with real content per epoch are computed here directly,
        against thresholds taken from `condition.py`.

        `LatentConditioner.falsifiers()` is called: a change of condition
        alone must move the latent above the noise floor of 0.049, and so must
        a change of sequence alone. It overwrites `_diag`, so the snapshot is
        taken before and restored afterwards.
        """
        lines: List[str] = []
        if not self.condition_report_enabled():
            return lines
        fus = getattr(self.model, "fusion", None)

        # 1. Fusion: numbers from the last training forward
        if fus is None:
            lines.append(
                "  cond_fuse    missing: the model has no LatentConditioner "
                "(heads.fusion.enabled=false, or no slicer), so the "
                "condition does NOT reach the latent by this path")
        else:
            snap = {}
            try:
                snap = dict(getattr(fus, "_diag", None) or {})
                if not snap:
                    lines.append(
                        "  cond_fuse    no numbers: _diag is empty, so this "
                        "epoch had NOT ONE pass through the fusion "
                        "(context=None?)")
                else:
                    shown = [(k, snap[k]) for k in self.FUSION_STAT_KEYS
                             if snap.get(k) is not None]
                    absent = [k for k in self.FUSION_STAT_KEYS
                              if k not in snap]
                    lines.append("  cond_fuse    " + " ".join(
                        "%s %s" % (k, self._stat(v)) for k, v in shown))
                    if absent:
                        lines.append("  cond_fuse    absent %s"
                                     % (",".join(absent),))
                    per = [(k, v) for k, v in sorted(snap.items())
                           if k.startswith("fuse_gate_")
                           or k.startswith("fuse_code_rms_")]
                    if per:
                        lines.append("  cond_fuse    " + " ".join(
                            "%s %s" % (k, self._stat(v)) for k, v in per))
            except Exception as exc:                # never kill the run
                lines.append("  cond_fuse    report FAILED %s: %s"
                             % (type(exc).__name__, exc))

            # 2. Leave-one-out, plus the falsifier on `null`
            probe = getattr(self, "_cond_probe", None)
            if probe is None:
                lines.append(
                    "  cond_eff     missing: no training step left a "
                    "(pooled, cond_tokens) pair behind, so leave-one-out has "
                    "nothing to compute on")
            else:
                try:
                    with torch.no_grad():
                        eff = fus.channel_effect(probe[0], probe[1]) or {}
                    e = dict(eff.get("effect") or {})
                    sh = dict(eff.get("share") or {})
                    ga = dict(eff.get("gate") or {})
                    if not e:
                        lines.append("  cond_eff     missing: channel_effect() "
                                     "returned no channel at all")
                    else:
                        lines.append("  cond_eff     " + " ".join(
                            "%s eff %s sh %s g %s"
                            % (n, self._stat(float(e[n])),
                               self._stat(float(sh.get(n, float("nan")))),
                               self._stat(float(ga.get(n, float("nan")))))
                            for n in e))
                        _esr = dict(eff.get("effect_mean") or {})
                        _ssr = dict(eff.get("share_mean") or {})
                        if _esr:
                            lines.append("  cond_eff_sr  " + " ".join(
                                "%s eff %s sh %s"
                                % (n, self._stat(float(_esr[n])),
                                   self._stat(float(_ssr.get(
                                       n, float("nan")))))
                                for n in _esr))
                            lines.append(
                                "  cond_eff_sr  reference: the channel is "
                                "replaced by ITS OWN BATCH MEAN, not by "
                                "`null`. This is the number to quote: `null` "
                                "gets no gradient, so `cond_eff` above "
                                "measures the distance to a constant left "
                                "over from initialisation, while this "
                                "measures what the channel contributes PER "
                                "SAMPLE with the shared component kept")
                        else:
                            lines.append(
                                "  cond_eff_sr  missing: needs batch > 1 "
                                "(the batch mean is the reference here)")
                        _st = list(eff.get("starved") or [])
                        lines.append(
                            "  cond_eff     n %d active_floor %s starved %s "
                            "all_active %s"
                            % (probe[0].shape[0],
                               self._stat(eff.get("active_floor")),
                               (",".join(_st) or "none"),
                               self._stat(bool(eff.get("all_active")))))
                        if _st:
                            lines.append(
                                "  cond_eff     FALSIFIER FAILED: channels "
                                "%s fell below the share threshold -- a "
                                "channel that does not reach the latent "
                                "cannot separate two rows carrying the same "
                                "sequence" % (",".join(_st),))
                except Exception as exc:            # never kill the run
                    lines.append("  cond_eff     report FAILED %s: %s"
                                 % (type(exc).__name__, exc))
            # `null` is the reference of the whole `channel_effect`
            # measurement, so if it is dead every number above is void. Across
            # one run's checkpoints its norm was identical bit for bit in
            # epoch 1 and epoch 14, the only one of 26 fusion tensors that
            # never moved: `null[i]` enters the graph only through
            # `channel_codes(drop=...)`, and `drop` is passed only by
            # `channel_effect`, which is `@torch.no_grad()`.
            try:
                _bn = getattr(fus, "bottleneck", None)
                _null = getattr(_bn, "null", None)
                if _null is None:
                    lines.append("  cond_eff     null <missing parameter "
                                 "bottleneck.null>")
                else:
                    _d = _null.detach()
                    _g = _null.grad
                    _ref = getattr(self, "_cond_null_ref", None)
                    if _ref is None or _ref.shape != _d.shape:
                        self._cond_null_ref = _d.clone()
                        _moved = 0.0
                    else:
                        _moved = float((_d - _ref.to(_d.device)).norm())
                    _gn = (None if _g is None
                           else float(_g.detach().abs().max()))
                    lines.append(
                        "  cond_eff     null_norm %s null_grad %s "
                        "null_moved_od_ep1 %s"
                        % (self._stat(float(_d.norm())),
                           ("None" if _gn is None else self._stat(_gn)),
                           self._stat(_moved)))
                    if _gn is None or _gn == 0.0:
                        lines.append(
                            "  cond_eff     FALSIFIER FAILED: "
                            "fusion.bottleneck.null gets NO gradient "
                            "(grad=%s), so `channel_effect` measures the "
                            "distance to an UNTRAINED random vector rather "
                            "than to a learnt 'this channel is absent'. The "
                            "`eff`/`sh` numbers above are then a reading of "
                            "the channel code's NORM against a constant left "
                            "over from initialisation"
                            % ("None" if _gn is None else "0"))
            except Exception as exc:                # never kill the run
                lines.append("  cond_eff     null report FAILED %s: %s"
                             % (type(exc).__name__, exc))

            # 3. The two separation falsifiers
            try:
                _f = fus.falsifiers() or {}
                lines.append(
                    "  cond_fals    ctx_rel %s seq_rel %s podloga %s | "
                    "G2_rozdziela_te_sama_sekwencje %s "
                    "G3_sekwencja_wciaz_wazna %s"
                    % (self._stat(float(_f.get("context_effect_relative",
                                               float("nan")))),
                       self._stat(float(_f.get("sequence_effect_relative",
                                               float("nan")))),
                       self._stat(float(_f.get("noise_floor", float("nan")))),
                       self._stat(bool(_f.get(
                           "identical_sequence_separates"))),
                       self._stat(bool(_f.get("sequence_still_matters")))))
                for _n in ("identical_sequence_separates",
                           "sequence_still_matters"):
                    if not bool(_f.get(_n)):
                        lines.append("  cond_fals    FALSIFIER FAILED: %s"
                                     % (_n,))
            except Exception as exc:                # never kill the run
                lines.append("  cond_fals    report FAILED %s: %s"
                             % (type(exc).__name__, exc))
            finally:
                # `falsifiers()` pushes its own inputs through `forward`, and
                # `forward` overwrites `_diag`. Without this restore, the next
                # reader of `_diag` would get numbers from this probe's random
                # noise instead of from a training batch.
                try:
                    if snap:
                        fus._diag = dict(snap)
                except Exception:
                    pass

        # 4. The condition encoder: `ktx_*`
        try:
            con = getattr(self.model, "condition", None)
            if con is None:
                lines.append("  cond_ktx     missing: the model has no "
                             "ConditionEncoder")
            else:
                d = dict(getattr(con, "_diag", None) or {})
                if not d:
                    lines.append(
                        "  cond_ktx     no numbers: _diag is empty, so this "
                        "epoch had NOT ONE pass through the condition "
                        "encoder")
                else:
                    kol = [k for k in self.CONDITION_STAT_KEYS if k in d]
                    reszta = sorted(k for k in d
                                    if k.startswith("ktx_") and k not in kol)
                    lines.append("  cond_ktx     " + " ".join(
                        "%s %s" % (k, self._stat(d[k])) for k in kol + reszta))
                    # Every token is normalised to RMS 1 per sample, so each
                    # of the T tokens must come out at 1/T. The 0.20-0.30
                    # window written in `condition.py` was for T=4, i.e.
                    # 1/T +- 0.05; here T is read off, because this model has
                    # a fifth token (gene) and 0.25 would be wrong.
                    _ud = {k: v for k, v in d.items()
                           if k.startswith("ktx_udzial_")}
                    if _ud:
                        _cel = 1.0 / float(len(_ud))
                        _dev = max(abs(float(v) - _cel)
                                   for v in _ud.values())
                        lines.append(
                            "  cond_ktx     tokens %d cel_udzialu %s "
                            "max_odchyl %s (threshold 0.05)"
                            % (len(_ud), self._stat(_cel), self._stat(_dev)))
                        if _dev > 0.05:
                            lines.append(
                                "  cond_ktx     FALSIFIER FAILED: "
                                "RMS balance across tokens (D8) -- deviation "
                                "%s from 1/T; a branch with its own scale has "
                                "taken over the map three times in this "
                                "project (dominance 1.85 -> 22.64)"
                                % (self._stat(_dev),))
                    # Falsifiers of the `xattn` branch. The thresholds live as
                    # the class constants `CoarseConditionAttention.
                    # PROG_POZ_CV` and `SHARE_THRESHOLD`, and nothing used to
                    # watch them. A threshold nobody reads is not a falsifier.
                    _xa = getattr(self.model, "xattn", None)
                    if _xa is not None:
                        _cv = d.get("ktx_xa_poz_cv")
                        _pcv = float(getattr(type(_xa), "PROG_POZ_CV", 0.5))
                        if _cv is not None and float(_cv) < _pcv:
                            lines.append(
                                "  cond_ktx     FALSIFIER FAILED: the "
                                "condition attention is UNIFORM -- "
                                "ktx_xa_poz_cv %s below the threshold %s. "
                                "Random q/k give 0.27, so until the "
                                "threshold is passed the module is an "
                                "expensive FiLM: 64 112 weights for a window "
                                "choice it does not make"
                                % (self._stat(float(_cv)),
                                   self._stat(_pcv)))
                        _uxa = d.get("ktx_xa_udzial")
                        _pud = float(getattr(type(_xa), "SHARE_THRESHOLD", 1.0))
                        if _uxa is not None and float(_uxa) > _pud:
                            lines.append(
                                "  cond_ktx     FALSIFIER FAILED: the xattn "
                                "branch is TAKING OVER the latent -- "
                                "ktx_xa_udzial %s above the threshold %s. A "
                                "branch with its own scale has taken over "
                                "the map three times in this project "
                                "(dominance 1.85 -> 22.64 over ten epochs), "
                                "and the only brake here is weight decay on "
                                "`output.weight`"
                                % (self._stat(float(_uxa)),
                                   self._stat(_pud)))
                    _tdev = d.get("ktx_tekst_dev_nad_srednia")
                    if _tdev is not None and float(_tdev) <= 0.2:
                        lines.append(
                            "  cond_ktx     FALSIFIER FAILED: the text token "
                            "carries no per-sample content -- "
                            "deviation/mean %s against a threshold of 0.2 "
                            "(precedent: 0.0869 dead -> 1.0701 after "
                            "parameter-free centring)"
                            % (self._stat(float(_tdev)),))
        except Exception as exc:                    # never kill the run
            lines.append("  cond_ktx     report FAILED %s: %s"
                         % (type(exc).__name__, exc))

        # 5. `cond_H` / `cond_share` of the cross-attention layer
        try:
            _enc = getattr(self.model, "encoder", None)
            _c = getattr(_enc, "cond", None) if _enc is not None else None
            if _c is None:
                lines.append(
                    "  cond_attn    missing: encoder.cond is None, so cond_H "
                    "and cond_share DO NOT EXIST in this model -- the "
                    "condition does not enter the map, only the latent "
                    "through the fusion. The absence of these keys from the "
                    "log is a property of the build, not lost diagnostics")
            else:
                _r = _c.report() or {}
                if not _r:
                    lines.append("  cond_attn    no numbers: report() empty")
                else:
                    lines.append("  cond_attn    " + " ".join(
                        "%s %s" % (k, self._stat(
                            float(v) if isinstance(v, torch.Tensor)
                            else v))
                        for k, v in sorted(_r.items())))
        except Exception as exc:                    # never kill the run
            lines.append("  cond_attn    report FAILED %s: %s"
                         % (type(exc).__name__, exc))
        return lines

    def _largest_component(self, stats: Dict[str, float]) -> str:
        comps = self.components(stats)
        if not comps:
            return "<none recorded>"
        name = max(comps, key=lambda k: abs(comps[k]))
        total = math.fsum(abs(v) for v in comps.values()) or 1.0
        return "%s = %.6e (%.3f %% of sum|component|)" % (
            name, comps[name], 100.0 * abs(comps[name]) / total)

    # Learning rate
    @torch.no_grad()
    def _zapamietaj_lr(self, opt) -> None:
        """Remember the lr of each parameter group separately.

        `param_groups` builds three groups -- manifold, no-decay, decayed --
        and the manifold group may carry its own lr. The schedule multiplies
        them by a common scalar; writing one value into all of them would
        silently equalise the groups.
        """
        for g in opt.param_groups:
            g["lr_bazowy"] = float(g["lr"])

    def _mnoznik_lr(self) -> float:
        """The common multiplier for every group at this step."""
        tc = self.cfg.training
        if str(getattr(tc, "lr_schedule", "none")) != "cosine":
            return 1.0
        total = int(getattr(self, "_run_steps", 0) or 0)
        if total <= 1:
            return 1.0
        t = min(float(self.step_count), float(total - 1))
        w = float(getattr(tc, "lr_warmup_frac", 0.0)) * float(total)
        if w > 0.0 and t < w:
            return (t + 1.0) / (w + 1.0)
        import math
        p = (t - w) / max(1.0, float(total - 1) - w)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))

    def _ustaw_lr(self) -> None:
        """At 'none' this touches `param_groups` not once -- bit for bit."""
        if str(getattr(self.cfg.training, "lr_schedule", "none")) != "cosine":
            return
        m = self._mnoznik_lr()
        for g in self.optimiser.param_groups:
            g["lr"] = float(g.get("lr_bazowy", g["lr"])) * m
        self._lr_biezacy = float(m)

    def _kod_ortogrup(self):
        """[N] dense orthogroup code per corpus row, computed once.

        `corpus.og` is a list of N strings and `row_index` indexes exactly
        that array. No corpus, or no `og` column, returns None and the axis
        decomposition simply does not exist -- the epoch line then shows `nan`,
        which states "not computed" rather than quietly dropping a row.
        """
        ko = getattr(self, "_og_kod", None)
        if ko is not None:
            return ko
        cor = getattr(self, "maker", None)
        cor = None if cor is None else getattr(cor, "corpus", None)
        og = None if cor is None else getattr(cor, "og", None)
        if og is None or len(og) == 0:
            return None
        _, inv = np.unique(np.asarray(og), return_inverse=True)
        self._og_kod = np.asarray(inv, dtype=np.int64).ravel()
        return self._og_kod

    def _osie_wiersza(self, b: "TrainBatch", sel: np.ndarray, ksztalt):
        """Codes (orthogroup, genome, condition) for the elements `sel` picks.

        `sel` is exactly the mask `mu_p`/`mu_t` go through, so the codes come
        out in the same order and at the same length as the values. Returns
        None when the corpus has no `og` table.
        """
        ko = self._kod_ortogrup()
        if ko is None:
            return None
        n = int(np.asarray(sel).shape[0])
        ri = self._dopelnij(b.row_index, n)
        sp = self._dopelnij(b.species_id, n)
        # -1 means its own group, not a negative index: `ko[-1]` would be the
        # orthogroup of the last corpus row, so the negatives would quietly be
        # assigned a real one.
        og = np.where(ri >= 0, ko[np.clip(ri, 0, ko.size - 1)], -1)
        if len(ksztalt) == 2:
            # Per-cell regime: the condition is the COLUMN, not `cat_target`.
            # The target is a (row x category) matrix, so each element has its
            # own category and taking the row label would substitute one
            # quantity for another.
            K = int(ksztalt[1])
            _og2 = np.repeat(og[:, None], K, axis=1)[sel]
            self._axes_missing = (getattr(self, "_axes_missing", 0)
                               + int(np.count_nonzero(_og2 < 0)))
            self._osie_ile = getattr(self, "_osie_ile", 0) + int(_og2.size)
            return (_og2,
                    np.repeat(sp[:, None], K, axis=1)[sel],
                    np.tile(np.arange(K, dtype=np.int64), (ri.size, 1))[sel])
        ct = (np.full(n, -1, dtype=np.int64)
              if getattr(b, "cat_target", None) is None
              else self._dopelnij(b.cat_target, n))
        _og1 = og[sel]
        self._axes_missing = (getattr(self, "_axes_missing", 0)
                           + int(np.count_nonzero(_og1 < 0)))
        self._osie_ile = getattr(self, "_osie_ile", 0) + int(_og1.size)
        return _og1, sp[sel], ct[sel]

    @staticmethod
    def _dopelnij(v, n: int) -> np.ndarray:
        """[m] -> [n] by padding with -1 on the right, or truncating.

        `row_index` has the length of the POSITIVE rows while `mu_row` covers
        the whole batch including negatives. The padding keeps a 1:1
        correspondence with `mu_p`; how much of it actually reached the metric
        is reported as `os_bez_ortogrupy`.
        """
        a = np.asarray(v.detach().cpu().numpy(), dtype=np.int64).ravel()
        if a.size == n:
            return a
        if a.size > n:
            return a[:n]
        out = np.full(n, -1, dtype=np.int64)
        out[:a.size] = a
        return out

    def evaluate(self, split: str = "val", max_steps: int = 0,
                 seed: int = 0) -> Dict[str, float]:
        """Metrics on one split, with every source of run-to-run drift pinned.

        The mask counter is pinned here. `LatentMask` seeds its block draw
        with `seed * 1000003 + _draws`, and every call to `views()` -- an
        evaluation included -- increments that counter. It is a persistent
        buffer, so a checkpoint carries whatever value it had at save time,
        and the trap is not that it is lost but that it moves: evaluate, save,
        reload and evaluate again, and the second pass starts one evaluation
        further along, masks different columns and reports a different
        reconstruction accuracy for identical weights. Without the pin T5
        would be a test of an RNG rather than of the weights. The counter is
        restored afterwards so training still sees a fresh mask every step.
        """
        self.model.eval()
        draws = int(self.model.mask._draws)
        self.model.mask._draws.fill_(0)
        # The usage histograms are reset and restored, for the same reason.
        # They are exponential averages held in buffers registered
        # `persistent=False`, so every bits/perplexity/usage number was a
        # blend of the validation distribution with whatever training left in
        # the decay: on two evaluations of bit-identical weights, `bits_total`
        # read 145.150 against 118.500. Restored afterwards because the same
        # histograms drive the dead-code restarts in training.
        buffers = {k: v.detach().clone()
                   for k, v in self.model.named_buffers()}
        reset_statistics(self.model)
        for head in (self.model.absolute, self.model.pairwise,
                     self.model.discriminator):
            if head is not None:
                head.reset()
        mu_p, mu_t, d_s, d_l = [], [], [], []
        mu_pm, mu_tm = [], []      # the marginal column only
        # Three int64 codes -- orthogroup, genome, condition -- for every
        # element that enters `mu_p`. Collected in the SAME branch and under
        # the SAME `sel`, because a decomposition computed over a different
        # subset of rows than `pcc` measures a different object. Cost: three
        # [B] copies per batch.
        ax_og, ax_gen, ax_cat = [], [], []
        self._axes_missing = 0
        self._osie_ile = 0
        # The per-cell readout in the per-sample regime. Separate lists rather
        # than `mu_pm`: that one means "the genome-marginal column", this one
        # means "the category columns computed from the same features".
        mu_pc, mu_tc = [], []
        # The same cell readout kept 2-D. `mu_pc` is flattened through the
        # validity mask, which destroys the row structure, and the
        # per-category question -- does the model move along a row at all --
        # cannot be asked of a flat vector. Without these three lists the
        # within-row contrast is not computable from an evaluation pass.
        cell2_p, cell2_t, cell2_v = [], [], []
        cat_acc_sum: Dict[str, float] = {}
        cat_acc_n = 0
        # The input of the cell readout is the centroids of the sample
        # descriptions, NOT the category prototypes. The prototypes are
        # orthogonal to the description cloud (cross-set cosine +0.0103), so a
        # readout from them asked `text_proj` about directions it had never
        # seen: measured 0.4519 from prototypes against 0.6696 from centroids
        # on the same checkpoint.
        _ps = getattr(self.maker.corpus, "per_sample", None)
        _cT = None if _ps is None else getattr(_ps, "cat_centroids", None)
        self._cell_read_T = (None if _cT is None else
                             torch.from_numpy(_cT).to(self.device))
        pair_scores = []
        acc, n_seen, n_batches = {}, 0, 0
        try:
            for batch in self.maker.epoch(split, shuffle=False,
                                          seed=int(seed),
                                          max_steps=max_steps):
                b = batch.to(self.device)
                players, subordinate, stats, out = self.compute_losses(
                    b, track=False)
                n_batches += 1
                n_seen += len(b)
                for k, v in stats.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                for k, v in players.items():
                    acc["task_" + k] = acc.get("task_" + k, 0.0) + \
                        float(v.detach())
                # Per-sample regime: the target is [B], not [B, K]. The
                # headline metric is then computed on per-sample values and is
                # by definition lower than the per-cell one, because the
                # target is noisier (two independent draws of the same cell
                # correlate at 0.8466). The cell readout is collected beside
                # it, on the same rows; without it a comparison with a
                # per-cell run compares units of the metric, not models.
                if b.mu_row is not None:
                    sel = b.mu_row_valid.detach().cpu().numpy().astype(bool)
                    _p = out["absolute"].mu.detach().cpu().numpy()
                    _t = b.mu_row.detach().cpu().numpy()
                    if sel.any():
                        mu_p.append(_p[sel])
                        mu_t.append(_t[sel])
                        _ax = self._osie_wiersza(b, sel, _p.shape)
                        if _ax is not None:
                            ax_og.append(_ax[0])
                            ax_gen.append(_ax[1])
                            ax_cat.append(_ax[2])
                    # Cell readout: the same features, columns generated from
                    # the category descriptions. One matrix product and zero
                    # parameters -- it is the same `text_proj` projection.
                    _ao = out["absolute"]
                    _h = getattr(_ao, "trunk", None)
                    _cell = b.mu_valid.detach().cpu().numpy().astype(bool)
                    if (_h is not None and _cell.any()
                            and self._cell_read_T is not None):
                        with torch.no_grad():
                            _mc = self.model.absolute.columns(
                                _h, text_emb=self._cell_read_T)
                        _mcn = _mc.detach().cpu().numpy()
                        _tc = b.mu.detach().cpu().numpy()
                        mu_pc.append(_mcn[_cell])
                        mu_tc.append(_tc[_cell])
                        cell2_p.append(_mcn[:, :_tc.shape[1]])
                        cell2_t.append(_tc)
                        cell2_v.append(_cell)
                    # The category head on held-out data. `cond_dropout_text`
                    # applies only in training, so in evaluation no row has
                    # its description dropped and the head sees not one
                    # example -- measured as `cat_acc nan`. Without this pass
                    # its falsifier (at chance when untrained, above chance
                    # when trained) is unmeasurable outside training. A second
                    # forward with `text_ok` explicitly zeroed, capped at
                    # `cat_eval_batches`, which is its only cost.
                    if (self.model.category is not None
                            and n_batches <= self.CAT_EVAL_BATCHES
                            and b.cat_target is not None):
                        with torch.no_grad():
                            o2 = self.model(
                                b.x, text_emb=b.text_emb,
                                species_id=b.species_id,
                                flank_l=b.flank_l, flank_r=b.flank_r,
                                flank_ok=b.flank_ok,
                                text_ok=torch.zeros_like(b.text_ok),
                                gene_numeric=getattr(b, "gene_numeric", None),
                                gene_identity=getattr(b, "gene_identity",
                                                      None),
                                gene_text=getattr(b, "gene_text", None),
                                gene_ok=getattr(b, "gene_ok", None),
                                pair_group=b.pair_group,
                                row_text=b.row_text, reconstruct=False)
                            if o2.get("cat_head") is not None:
                                _, _cp2 = self.model.category.loss(
                                    o2["cat_head"], b.cat_target,
                                    text_used=o2.get("text_used"), track=True)
                                for _k, _v in _cp2.items():
                                    cat_acc_sum[_k] = cat_acc_sum.get(
                                        _k, 0.0) + float(_v)
                                cat_acc_n += 1
                    d_s.append(
                        out["discriminator"].logit.detach().cpu().numpy())
                    d_l.append(
                        b.is_promoter.detach().cpu().numpy().astype(float))
                    if b.n_pairs:
                        po = self._pair_output(out["pair_score"], b,
                                               z=out.get("z"))
                        pair_scores.append(
                            (po.delta.detach().cpu().numpy(),
                             b.pair_sign.detach().cpu().numpy()))
                    continue
                sel = b.mu_valid.detach().cpu().numpy().astype(bool)
                _p = out["absolute"].mu.detach().cpu().numpy()
                _t = b.mu.detach().cpu().numpy()
                if sel.any():
                    mu_p.append(_p[sel])
                    mu_t.append(_t[sel])
                    _ax = self._osie_wiersza(b, sel, _p.shape)
                    if _ax is not None:
                        ax_og.append(_ax[0])
                        ax_gen.append(_ax[1])
                        ax_cat.append(_ax[2])
                # With a per-cell head the pooled number above runs over cells
                # and is not comparable with the single-target runs; the
                # marginal column is, so it is collected separately and both
                # are printed. Only when a marginal column exists -- under
                # per-category training it does not, and taking column -1
                # would silently report `other` under the name `marginal`.
                if _p.ndim == 2 and not getattr(
                        self.maker, "per_category", False):
                    _m = sel[:, -1]
                    if _m.any():
                        mu_pm.append(_p[:, -1][_m])
                        mu_tm.append(_t[:, -1][_m])
                d_s.append(out["discriminator"].logit.detach().cpu().numpy())
                d_l.append(b.is_promoter.detach().cpu().numpy().astype(float))
                if b.n_pairs:
                    po = self._pair_output(out["pair_score"], b,
                                           z=out.get("z"))
                    pair_scores.append(
                        (po.delta.detach().cpu().numpy(),
                         b.pair_sign.detach().cpu().numpy()))
        finally:
            self.model.mask._draws.fill_(draws)
            with torch.no_grad():
                for k, v in self.model.named_buffers():
                    v.copy_(buffers[k])
        if not n_batches:
            raise RuntimeError("split %r produced no batch" % (split,))
        met = {k: v / n_batches for k, v in acc.items()}
        met["n_rows"] = float(n_seen)
        met["n_batches"] = float(n_batches)
        if mu_p:
            p = np.concatenate(mu_p)
            t = np.concatenate(mu_t)
            met["pcc"] = pearson(p, t)
            met["scc"] = spearman(p, t)
            met["mse"] = float(np.mean((p - t) ** 2))
            met["pred_sd"] = float(np.std(p))
            met["target_sd"] = float(np.std(t))
        if mu_pm:
            # The genome-marginal column, so a per-cell run can still be put
            # beside the single-target runs. `scc` above counts CELLS.
            pm = np.concatenate(mu_pm)
            tm = np.concatenate(mu_tm)
            met["scc_marginal"] = spearman(pm, tm)
            met["pcc_marginal"] = pearson(pm, tm)
        if cat_acc_n:
            for _k, _v in cat_acc_sum.items():
                met[_k] = _v / cat_acc_n
            met["cat_eval_batches"] = float(cat_acc_n)
        if mu_pc:
            pc_ = np.concatenate(mu_pc)
            tc_ = np.concatenate(mu_tc)
            # These two are the only valid way of comparing a per-sample run
            # with a per-cell one: `pcc` above is computed on the per-sample
            # target, whose ceiling is ~0.89, while these are computed on the
            # same target as before and on the same rows.
            met["pcc_cell_readout"] = pearson(pc_, tc_)
            met["scc_cell_readout"] = spearman(pc_, tc_)
            met["n_cell_readout"] = float(pc_.size)
        # Stashed, not scored here: the within-row contrast belongs to
        # `hyppe.baselines.within_row_contrast`, which the baselines and
        # `score_checkpoint.py` already call, and a second copy of that formula
        # is how two copies drift apart. A caller that wants the per-category
        # component reads this and calls that one function.
        self._readout_2d = ((np.concatenate(cell2_p, axis=0),
                            np.concatenate(cell2_t, axis=0),
                            np.concatenate(cell2_v, axis=0))
                           if cell2_p else None)
        if mu_p and ax_og and len(ax_og) == len(mu_p):
            met.update(rozbicie_osi(
                np.concatenate(mu_p), np.concatenate(mu_t),
                np.concatenate(ax_og), np.concatenate(ax_gen),
                np.concatenate(ax_cat)))
            # The decomposition's own falsifier. `row_index` does not cover
            # the negatives, so if one of them has a valid target it falls
            # into the lumped group -1 and inflates the orthogroup take. 0.0
            # means the axes were computed on corpus rows alone.
            met["os_bez_ortogrupy"] = (
                float(self._axes_missing) / float(self._osie_ile)
                if self._osie_ile else float("nan"))
        # Global calibration as a single number. `pred_sd` and `target_sd`
        # have long been computed and appear in no line of the log, and their
        # ratio to the optimum (pcc * target_sd) drifts: measured 0.976 (ep1)
        # -> 1.027 (ep4) -> 1.187 (ep14). Above 1 the model overshoots the
        # amplitude, paying MSE for correlation it does not have.
        _sd_t = float(met.get("target_sd", 0.0))
        _pcc = float(met.get("pcc", float("nan")))
        _mian = _pcc * _sd_t
        met["kalibracja_sd"] = (float(met.get("pred_sd", float("nan")))
                                / _mian if abs(_mian) > 0.0
                                else float("nan"))
        if pair_scores:
            delta = np.concatenate([a for a, _ in pair_scores])
            sign = np.concatenate([b for _, b in pair_scores])
            met["pair_acc"] = float(np.mean((delta * sign) > 0))
        if d_s:
            met["disc_auc"] = roc_auc(np.concatenate(d_s),
                                      np.concatenate(d_l))
        return met

    # Selection
    def evaluate_srednia(self, split: str = "val", max_steps: int = 0,
                         n_ziaren: int = 1) -> Dict[str, float]:
        """`evaluate` averaged over N evaluation seeds.

        Five seeds of a validation epoch on one checkpoint gave
        `pcc_cell_readout` between 0.68934 and 0.70312, sd 0.00596, while
        `rec_full` had sd 0.00001 and two consecutive evaluations at the same
        seed differed by 0.00e+00. The noise is therefore entirely the
        `maker.epoch` seed -- which mu rows, negatives and ortholog blocks are
        drawn -- and not non-determinism. Selecting by argmax over ~28 such
        draws biases the recorded peak upwards by about 2 sd; averaging over N
        seeds divides that noise by sqrt(N).

        Observables: `eval_seeds`, `sel_mean`, `sel_sd`. A `sel_sd` of exactly
        0.00000 means the seed is not really changing -- a broken intervention,
        not an absence of noise.
        """
        n = max(1, int(n_ziaren))
        runs = [self.evaluate(split, max_steps=max_steps, seed=i)
                 for i in range(n)]
        if n == 1:
            out = dict(runs[0])
            out["eval_seeds"] = 1.0
            try:
                out["sel_mean"] = float(self.selection_metric(out))
            except (KeyError, ValueError):
                out["sel_mean"] = float("nan")
            out["sel_sd"] = 0.0
            return out
        klucze = set()
        for b in runs:
            klucze.update(k for k, v in b.items()
                          if isinstance(v, (int, float))
                          and not isinstance(v, bool))
        out: Dict[str, float] = {}
        for k in sorted(klucze):
            vals = [float(b[k]) for b in runs
                    if k in b and np.isfinite(float(b[k]))]
            if vals:
                out[k] = float(np.mean(vals))
        # Non-numeric keys take their value from seed 0: averaging them makes
        # no sense, and dropping them would change the shape of the dict.
        for k, v in runs[0].items():
            if k not in out:
                out[k] = v
        sel = []
        for b in runs:
            try:
                sel.append(float(self.selection_metric(b)))
            except (KeyError, ValueError):
                pass
        out["eval_seeds"] = float(n)
        out["sel_mean"] = float(np.mean(sel)) if sel else float("nan")
        out["sel_sd"] = (float(np.std(sel, ddof=1)) if len(sel) > 1
                         else float("nan"))
        out["sel_min"] = float(min(sel)) if sel else float("nan")
        out["sel_max"] = float(max(sel)) if sel else float("nan")
        if sel and len(sel) > 1 and out["sel_sd"] == 0.0:
            print("  WARNING: sd across %d evaluation seeds = 0.00000. The "
                  "seed is not really being changed -- this is a BROKEN "
                  "INTERVENTION, not an absence of noise (the measured "
                  "order is 0.006)." % (n,))
        return out

    def selection_metric(self, val: Dict[str, float]) -> float:
        """The raw `select_on` metric, unchanged."""
        key = self.cfg.training.select_on
        table = {"val_pairwise": "pair_acc", "val_pcc": "pcc",
                 "val_scc": "scc", "val_auc": "disc_auc"}
        name = table.get(key, key)
        if name not in val:
            raise KeyError(
                "training.select_on=%r resolves to metric %r, which this "
                "evaluation does not produce (have: %s). A selection key with "
                "no receiver silently selects the last epoch."
                % (key, name, ", ".join(sorted(val))))
        return float(val[name])

    #: How many evaluation batches get the category head's second forward
    #: pass, which is its entire cost. At batch 96 this is ~768 rows, so an
    #: accuracy SE of ~0.018: enough to tell a gain over chance from zero, too
    #: little to matter in wall-clock time.
    CAT_EVAL_BATCHES: int = 8

    #: Chance level per selection metric. `pair_acc` and `disc_auc` are
    #: two-class orderings, so their floor is 0.5; correlations start at 0.
    #: Stated, not assumed: normalising `pair_acc` from 0 would make the task
    #: factor read 0.73 where the model is only 0.45 of the way above chance,
    #: and the composite would then be dominated by a constant.
    METRIC_FLOOR = {"pair_acc": 0.5, "disc_auc": 0.5,
                    "pcc": 0.0, "scc": 0.0}

    def selection_score(self, val: Dict[str, float]) -> float:
        """Composite: the geometric mean of task quality and reconstruction.

        Selecting on the task metric alone once saved a checkpoint at
        pair_acc 0.7254 with rec_full 0.3139 -- against a floor of 0.2626, that
        is 7 % of the achievable reconstruction. The objective is a record SCC
        *and* good reconstruction together, so a rule that cannot see one of
        them cannot implement it.

        A geometric mean rather than a weighted sum, because a sum lets one
        factor be bought back by the other, while a geometric mean goes to
        zero with either, so a collapsed reconstruction cannot be traded for
        pairwise accuracy at any weight. Both quantities are first put on a
        common scale -- the fraction of the achievable range above their own
        floor -- because rec_full lives in [0.2626, 1] and pair_acc in
        [0.5, 1], and a geometric mean of two differently offset numbers
        weights them by their offsets rather than by their quality.

        `select_recon_weight: 0.0` reproduces the old score exactly; the
        admission gate in `admit_checkpoint` is separate and still applies.
        """
        metric = self.selection_metric(val)
        w = float(self.cfg.training.select_recon_weight)
        if w <= 0.0:
            return metric
        key = self.cfg.training.select_on
        name = {"val_pairwise": "pair_acc", "val_pcc": "pcc",
                "val_scc": "scc", "val_auc": "disc_auc"}.get(key, key)
        floor = self.METRIC_FLOOR.get(name, 0.0)
        rec = float(val.get("rec_full", float("nan")))
        if not math.isfinite(rec):
            raise KeyError(
                "training.select_recon_weight=%.3f asks for a composite "
                "score, but this evaluation produced no finite `rec_full` "
                "(have: %s). Falling back to the task metric alone would "
                "silently restore exactly the rule that saved rec_full "
                "0.3139." % (w, ", ".join(sorted(val))))
        q_task = max(0.0, (metric - floor) / max(1e-9, 1.0 - floor))
        q_rec = max(0.0, (rec - RECONSTRUCTION_FLOOR)
                    / (1.0 - RECONSTRUCTION_FLOOR))
        return float(q_task ** (1.0 - w) * q_rec ** w)

    def admit_checkpoint(self, val: Dict[str, float]) -> Tuple[bool, str]:
        """Hard admission, independent of the score. `(ok, why)`.

        Separate from the score on purpose: a score can always be improved by
        a large enough gain on the other factor, while a floor cannot be
        traded against anything. `training.recon_admit` defaults to 0.90,
        below the 0.9486 an unaided epoch 1 reached and far above the 0.2626
        floor, so it rejects a collapsed reconstruction without ever rejecting
        a healthy one.

        The refusal is printed with its numbers, because a missing file that
        reads like "not yet" is indistinguishable from waiting.
        """
        rec = float(val.get("rec_full", float("nan")))
        floor = float(self.cfg.training.recon_admit)
        if not math.isfinite(rec):
            return False, "rec_full is not finite"
        if rec < floor:
            span = (rec - RECONSTRUCTION_FLOOR) / (1.0 - RECONSTRUCTION_FLOOR)
            return False, (
                "rec_full %.4f < training.recon_admit %.4f (that is %.1f %% "
                "of the range above the %.4f floor; epoch 1 of A1 reached "
                "0.9486)" % (rec, floor, 100.0 * span, RECONSTRUCTION_FLOOR))
        return True, "rec_full %.4f >= %.4f" % (rec, floor)

    # Checkpoints
    def checkpoint_path(self, tag: str) -> str:
        return os.path.join(self.settings.out_dir,
                            "%s_%s.pt" % (self.settings.name, tag))

    def save_checkpoint(self, path: str, metrics: Dict[str, float]) -> str:
        state = {
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "config": to_mapping(self.cfg),
            "metrics": dict(metrics),
            "epoch": self.epoch,
            "step": self.step_count,
            "method": self.method,
            "task_weights": dict(self.task_weights),
            # The controller's whole state, `ema` included: restoring lambda
            # without the running average restarts the controller's memory at
            # the first batch after a resume, and a controller whose memory
            # restarts overshoots exactly once per resume.
            "constraint": (None if self.controller is None else
                           {"lam": self.controller.lam,
                            "ema": self.controller._avg,
                            "steps": self.controller._steps}),
            # Also inside `model`'s state_dict (it is a persistent buffer);
            # written out separately so that a reader of the checkpoint can
            # see the value without loading the weights.
            "prefix_constraint": (
                None if self.prefix_controller is None else
                {"lam": self.prefix_controller.lam,
                 "ema": self.prefix_controller._avg,
                 "steps": self.prefix_controller._steps}),
            # With its average, for the same reason as the first controller.
            "occ_gate_constraint": (
                None if self.occ_gate_controller is None else
                {"lam": self.occ_gate_controller.lam,
                 "ema": self.occ_gate_controller._avg,
                 "steps": self.occ_gate_controller._steps,
                 "target": self.occ_gate_controller.target}),
            "mask_draws": int(self.model.mask._draws),
            # Run identity. None of the keys above come from RunSettings, so
            # without these four the checkpoint could not say which corpus
            # rows it had been allowed to see and the filename was the only
            # carrier -- and a filename is not a contract. Anyone fine-tuning
            # this needs the split, to avoid training on rows it already saw.
            "name": str(self.settings.name),
            "splits_path": str(self.settings.splits_path),
            "split_column": str(self.settings.split_column),
            "source_mtimes": _source_mtimes(),
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(state["model"],
                                                         strict=strict)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint does not match the model: %d missing, %d "
                "unexpected keys (%r / %r). A partially loaded checkpoint "
                "reports the metric of a partially random model."
                % (len(missing), len(unexpected), list(missing)[:5],
                   list(unexpected)[:5]))
        if state.get("constraint") and self.controller is not None:
            self.controller.load_state(ConstraintState(
                lam=state["constraint"]["lam"],
                ema=state["constraint"]["ema"],
                steps=int(state["constraint"]["steps"])))
        self.model.mask._draws.fill_(int(state.get("mask_draws", 0)))
        self.epoch = int(state.get("epoch", 0))
        self.step_count = int(state.get("step", 0))
        return state

    # Gates
    def wiring_gates(self) -> int:
        report = run_wiring_gates(self.weighting_factory())
        print(report.format())
        return report.exit_code()

    def measure_orthogonality(self, batch: TrainBatch) -> Dict[str, object]:
        """The Gram off-diagonal, measured on the real model in epoch 1."""
        self.model.train()
        self.optimiser.zero_grad(set_to_none=True)
        players, _sub, _stats, _out = self.compute_losses(batch, track=False)
        scaled = {n: self.task_weights[n] * v for n, v in players.items()}
        info = orthogonality_gate(scaled, self.shared)
        self.optimiser.zero_grad(set_to_none=True)
        self.orthogonality = info
        return info

    # Run
    def header(self) -> str:
        tc = self.cfg.training
        lines = [self.model.header(),
                 self.weighting.header(len(self.task_weights)),
                 "device %s | method %r | %d shared tensors (%d weights) in "
                 "scope %r"
                 % (self.device, self.method, len(self.shared),
                    sum(p.numel() for p in self.shared),
                    tc.shared_scope_for(self.method))]
        if self.controller is not None:
            lines.append(self.controller.describe())
        if self.prefix_controller is not None:
            lines.append(self.prefix_controller.describe())
        if self.occ_gate_controller is not None:
            lines.append(self.occ_gate_controller.describe())
        else:
            lines.append(
                "reconstruction: FIXED weight, effective %.4f for the `full` "
                "view (%.4f x %.4f) -- the coefficient that collapsed 0.9998 "
                "-> 0.5549 in one epoch on the sibling track is 0.0600"
                % (tc.effective_recon_coefficient("full"), tc.recon_weight,
                   tc.view_weights.get("full", 0.0)))
        for term in self.subordinate_terms():
            lines.append("  subordinate " + term.describe())
        lines.append(self.optimiser_header())
        lines.append(self.guard_header())
        return "\n".join(lines)

    def guard_header(self) -> str:
        """Print every guard, and say when one is off.

        A disabled guard must be as visible as an enabled one: `loss_max: 0`
        reads identically to a healthy run in every other line of the log,
        which is how a guard gets switched off in the config that needs it.
        """
        tc = self.cfg.training
        def _on(v: float, fmt: str) -> str:
            return (fmt % v) if v > 0 else "OFF"
        return ("guards: closure_tol %.1e (relative) | loss_max %s | "
                "enc_radius_max %s (pre-clamp; hard clamp at "
                "model.l0_max_radius %.1f) | enc_clip_frac_max %s\n"
                "        checkpoint: admit rec_full >= %.4f (floor %.4f), "
                "score = geometric mean of %s and rec_full at weight %.2f"
                % (tc.closure_tol, _on(tc.loss_max, "%.3g"),
                   _on(tc.enc_radius_max, "%.2f"),
                   float(self.cfg.model.l0_max_radius),
                   _on(tc.enc_clip_frac_max, "%.3g"),
                   tc.recon_admit, RECONSTRUCTION_FLOOR,
                   tc.select_on, tc.select_recon_weight))

    def optimiser_header(self) -> str:
        """Name the optimiser that was actually built, and what is on the
        manifold.

        The config report used to call `training.riemannian` inert on the
        grounds that no parameter lives on the manifold, while
        `_build_optimiser` did return `RiemannianAdam`. For this model that
        claim is false: `genome.points` is (20, 4) float64 with requires_grad
        true, measured max|grad| 9.9e-01 against a model-wide median of
        7.0e-04, in the `manifold` group at weight_decay 0. Neither claim was
        checkable from the log, so the header reads the BUILT optimiser and
        the BUILT group rather than the flag.
        """
        opt = self.optimiser
        man = [g for g in opt.param_groups if g.get("manifold")]
        n_ten = sum(len(g["params"]) for g in man)
        n_par = sum(int(p.numel()) for g in man for p in g["params"])
        ids = {id(p) for g in man for p in g["params"]}
        names = [n for n, p in self.model.named_parameters() if id(p) in ids]
        decays = sorted({float(g.get("weight_decay", 0.0)) for g in man})
        return ("optimiser %s | manifold group: %d tensor%s / %d params "
                "(%s) | weight_decay on that group %s | "
                "training.riemannian %s"
                % (type(opt).__name__, n_ten, "" if n_ten == 1 else "s",
                   n_par, ", ".join(names) or "EMPTY",
                   ",".join("%.4g" % v for v in decays) or "n/a",
                   bool(self.cfg.training.riemannian)))

    def _steps_per_epoch(self) -> int:
        """Steps in one training epoch, counted the way batches are built.

        `maker.steps_per_epoch` computes ceil(n_train / batch_size) and knows
        nothing about ortholog blocks. `BatchMaker.epoch` reserves
        `int(batch_size * ortholog_fraction)` rounded DOWN to a multiple of
        `ortholog_block`, and strides over the ordinary rows by the remainder.
        At batch 64 with fraction 0.30 and block 4 the reserve is 16 and the
        stride 48, which is 33 % more steps than the naive count.

        Without this the cosine schedule received a horizon a quarter too
        short and the whole of epoch 1 ran on it; the correction made in epoch
        2 does not undo an epoch that has already happened.

        `epoch` skips the last batch when fewer than two rows remain, since a
        pair needs two, so a remainder of 1 subtracts one step and a remainder
        of 0 subtracts nothing. Every fallback branch returns
        `maker.steps_per_epoch`: a worse estimate beats an exception in the
        first line of a run.
        """
        mk = self.maker
        mx = int(getattr(self.settings, "max_steps", 0) or 0)
        zapas = int(mk.steps_per_epoch("train", mx))
        # The log's decomposition gets the estimate WITHOUT `max_steps`:
        # `zapas` is already capped, so on a smoke run with `--max-steps 3`
        # the "old estimate" field would read 3, which is untrue of the
        # mechanism that line describes.
        szac = int(mk.steps_per_epoch("train", 0))
        try:
            n = int(mk.corpus.indices("train").size)
            bs = int(getattr(mk, "batch_size", 0) or 0)
        except Exception:
            return zapas
        if n <= 0 or bs <= 0:
            return zapas
        n_blk = 0
        frac = float(getattr(mk, "ortholog_fraction", 0.0) or 0.0)
        if frac > 0.0:
            try:
                bloki = mk._ortholog_blocks("train")
            except Exception:
                bloki = []
            if bloki:
                blk = max(2, int(getattr(mk, "ortholog_block", 4) or 4))
                n_blk = int(bs * frac)
                n_blk -= n_blk % blk
                n_blk = max(0, min(n_blk, bs - 2))
        step = max(1, bs - n_blk)
        ile = -(-n // step)
        if n % step == 1:
            ile -= 1
        ile = max(1, ile)
        self._decomposition_steps = (n, bs, n_blk, step, ile, szac)
        return min(ile, mx) if mx else ile

    def run(self, epochs: Optional[int] = None) -> List[EpochResult]:
        s = self.settings
        n_epochs = int(epochs or s.epochs or self.cfg.training.epochs)
        self._run_steps = n_epochs * int(self._steps_per_epoch())
        if str(getattr(self.cfg.training, "lr_schedule",
                       "none")) == "cosine":
            print("  lr: cosine over %d steps (%d epochs), "
                  "warmup %.3f"
                  % (self._run_steps, n_epochs,
                     float(getattr(self.cfg.training,
                                   "lr_warmup_frac", 0.0))))
        # The terms of the arithmetic, not only its result: a step count
        # without its decomposition cannot be checked from the log, and that
        # count was wrong by 33 % for a whole run. The old estimate stays on
        # the line on purpose, so that a future divergence says by how much.
        _rz = getattr(self, "_decomposition_steps", None)
        if _rz is not None:
            print("  lr: steps/epoch %d = ceil(%d rows / step %d) "
                  "| batch %d minus the orthogroup reserve %d "
                  "| old estimate (without blocks) %d"
                  % (_rz[4], _rz[0], _rz[3], _rz[1], _rz[2], _rz[5]))
        # `_best.pt` is no longer produced -- one name, one file. A file of
        # that name on disk comes from an EARLIER run and nothing here will
        # overwrite it, so a probe using the default `_best.pt` path would
        # load somebody else's weights with nothing anywhere to say so.
        # Deleting it is the user's decision, so the log only says it loudly.
        _stary = self.checkpoint_path("best")
        if os.path.exists(_stary):
            print("  WARNING: %s EXISTS and is STALE -- this run writes only "
                  "%s. The file is not deleted, but any probe using the "
                  "default `_best.pt` path is NOW reading an artefact of an "
                  "earlier run."
                  % (os.path.basename(_stary),
                     os.path.basename(self.checkpoint_path("fundational"))))
        _tp0 = (float(self.model.genome.tree_prior)
                if self.model.genome is not None else 0.0)
        _tpn = int(getattr(self.cfg.conditioning,
                           "genome_tree_prior_epochs", 0) or 0)
        for ep in range(1, n_epochs + 1):
            self.epoch = ep
            # Correcting the schedule horizon from measurement. The naive
            # count ignores negatives, so the cosine would reach zero in epoch
            # 10.5 of 14 and the last 3.5 epochs would run at lr exactly 0. At
            # the start of epoch 2, `step_count` is the exact length of epoch
            # 1, so it replaces the estimate.
            if ep == 2 and int(self.step_count) > 0:
                _stare = int(getattr(self, "_run_steps", 0) or 0)
                _nowe = n_epochs * int(self.step_count)
                if _stare != _nowe:
                    self._run_steps = _nowe
                    print("  lr: CORRECTION %d -> %d steps (epoch 1 had %d, "
                          "the estimate assumed %d per epoch)"
                          % (_stare, _nowe, int(self.step_count),
                             _stare // max(1, n_epochs)))
            # The tree prior as an initialiser. At
            # `genome_tree_prior_epochs = 0` this branch moves nothing and the
            # run is bit for bit as before; above 0 the coefficient decays
            # linearly to zero, so the prior PLANTS the points on the tree and
            # then hands them to the data. The reason is measured: the
            # `genome` module's gradient has a between-batch SNR of 33 212, so
            # it is a pure prior, and the channel contributes +0.00003 PCC.
            if self.model.genome is not None:
                if _tpn > 0:
                    _f = max(0.0, 1.0 - float(ep - 1) / float(_tpn))
                    self.model.genome.tree_prior = _tp0 * _f
                # The geometry falsifier, printed every epoch whether or not
                # the prior is annealed: mean |d_L/(s*d_patr) - 1| must stay
                # below 0.10, or the honest move is to drop the tree. A run
                # that does not print this number cannot fail it, and one run
                # failed it by 8.7x where only the annealing made it visible.
                _pf = self.model.genome.prior_fit()
                _mar = float(_pf.get("mean_abs_rel", float("nan")))
                _n_par = int(_pf.get("n_pairs", 0))
                # nan < 0.10 is false, so without this branch a registry with
                # no tree would report a breach rather than "no pairs".
                _f6 = ("missing pairs" if _n_par == 0 else
                       ("OK" if _mar < 0.10 else "EXCEEDED (threshold 0.10)"))
                print("  tree_prior ep%d: %.5f (start %.5f) | par %d | "
                      "mean_abs_rel %.4f max_abs_rel %.4f | F6 %s"
                      % (ep, float(self.model.genome.tree_prior), _tp0,
                         _n_par, _mar,
                         float(_pf.get("max_abs_rel", float("nan"))), _f6))
            t0 = time.time()
            self.weighting.new_epoch()
            if self.controller is not None:
                self.controller.new_epoch()
            if self.prefix_controller is not None:
                self.prefix_controller.new_epoch()
            # `lam` and the running average survive; only the per-epoch
            # counters are zeroed, otherwise every epoch would be the same
            # experiment over again.
            if self.occ_gate_controller is not None:
                self.occ_gate_controller.new_epoch()
            acc: Dict[str, float] = {}
            n = 0
            every = self.cfg.training.refresh_cadence(self.method)
            for batch in self.maker.epoch("train", shuffle=True, seed=ep,
                                          max_steps=s.max_steps):
                b = batch.to(self.device)
                if ep == 1 and n == 0 and not s.skip_gates:
                    info = self.measure_orthogonality(b)
                    print("  orthogonality (epoch 1, first batch): "
                          "offdiag_max %.4f | offdiag_mean %.4f | cos_min "
                          "%.4f | floor %.2f"
                          % (info.get("offdiag_max", float("nan")),
                             info.get("offdiag_mean", float("nan")),
                             info.get("cos_min", float("nan")),
                             ORTHOGONALITY_FLOOR))
                    print("      norms %s" % (info.get("norms"),))
                    print("      " + str(info.get("verdict", "")))
                stats = self.train_step(b, recompute=(n % every == 0))
                n += 1
                for k, v in stats.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                if s.log_every and n % s.log_every == 0:
                    print("    " + self.format_step(ep, n, stats))
            train = {k: v / max(n, 1) for k, v in acc.items()}
            train["steps"] = float(n)
            # `LocalConditionAttention` computes these four numbers on every
            # forward and nothing printed them. The `cond_ktx` line below
            # shows the LAST batch of the epoch; this one is the epoch mean,
            # which is the number to quote.
            #
            # The falsifier is `poz_cv`, the coefficient of variation of the
            # correction's norm ACROSS POSITIONS. An untrained model with a
            # random `output` gives 5.9e-09, i.e. a correction that is
            # constant across positions -- a global mechanism wearing
            # per-position arithmetic. `ktx_lok_H` is NOT a falsifier: its
            # ceiling is ln(T) = 1.6094 and an untrained model reads 1.6092,
            # so "H < ln(T)" is satisfied by construction.
            _lok = {k: v for k, v in train.items()
                    if k.startswith("ktx_lok_")}
            if _lok:
                print("  ktx_lok      " + " ".join(
                    "%s %s" % (k[len("ktx_lok_"):], self._g(v))
                    for k, v in sorted(_lok.items()))
                    + "  (mean over %d steps of the epoch)" % (n,))
            # The amplitude penalty on the genome axis. Gated on the key
            # being present, which keeps a run without the term byte for byte
            # as it was: at `kara_amplitudy: 0.0` the head does not put these
            # keys into `parts`, so the line never appears.
            if "abs_kara_amplitudy" in train:
                _kk = float(train.get("abs_kara_amplitudy", float("nan")))
                _kw = float(train.get("abs_kara_amplitudy_waga", 0.0))
                _kn = float(train.get("abs_kara_amplitudy_n", 0.0))
                _kp = float(train.get("abs_kara_amplitudy_par", 0.0))
                _kg = float(train.get("abs_kara_amplitudy_grup", 0.0))
                _kb = float(train.get("abs_kara_amplitudy_bez_og", 0.0))
                _kt = float(train.get("abs_total", float("nan")))
                # `abs_total` already contains `weight * penalty`, so this is
                # the real share rather than an estimate from the side.
                _ku = (_kw * _kk / _kt) if _kt else float("nan")
                # The first falsifier is about `par`, not `n`: `n` counts
                # valid rows, and the term is zero bit for bit when each of
                # them sits in its own group.
                print("  kara_ampl    kara %s waga %s | n %.1f par %.1f "
                      "grup %.1f bez_og %.1f | abs_total %s udzial %.4f"
                      "\n               F5a par>0 %s | F5b udzial in "
                      "[0.02;0.10] %s  (mean over %d steps of the epoch)"
                      % (self._g(_kk), self._g(_kw), _kn, _kp, _kg, _kb,
                         self._g(_kt), _ku,
                         "OK" if _kp > 0.0 else "FAILS",
                         "OK" if 0.02 <= _ku <= 0.10 else "FAILS", n))
            # The head report is taken BEFORE the evaluation. `evaluate()`
            # calls `head.reset()` on the absolute head, and `observe()` only
            # runs under `track=True`, i.e. in the training loop alone, so a
            # report read afterwards described a counter zeroed a moment
            # earlier and 30 epochs of 30 printed "absolute n 0 ok 0". This
            # one describes the TRAINING epoch. The printing stays below so
            # the order of lines in the log does not change.
            _ah = getattr(self.model, "absolute", None)
            _ar = (_ah.report()
                   if (_ah is not None and hasattr(_ah, "report"))
                   else None)
            # The quantiser report is taken and printed before the evaluation
            # for the same reason, and for one of its own: `evaluate()` calls
            # `reset_statistics(self.model)`.
            for _qline in self.format_quantiser_report():
                print(_qline)
            val = self.evaluate_srednia(
                "val", max_steps=s.max_eval_steps,
                n_ziaren=int(getattr(self.cfg.training, "eval_seeds", 1)))
            res = EpochResult(epoch=ep, train=train, val=val,
                              weighting=self.weighting.report(),
                              constraint=({} if self.controller is None
                                          else self.controller.report()),
                              seconds=time.time() - t0)
            self.history.append(res)
            print(self.format_epoch(res))
            # The absolute head's own report. It existed and was called by no
            # line of `src/`, so a 30-epoch run said nothing about whether the
            # variance head had collapsed to a constant, was sitting on the
            # clamp, or what its calibration slope was -- and the affine
            # correction is mandatory before this predictor is used as a
            # design target, which cannot be checked after the fact.
            if _ar is not None:
                if isinstance(_ar, dict) and _ar:
                    print("  absolute " + " ".join(
                        "%s %.4g" % (k, float(x))
                        for k, x in sorted(_ar.items())
                        if isinstance(x, (int, float))))
                elif _ar:
                    print("  absolute %s" % (_ar,))
            print("  " + self.weighting.format_report())
            if self.controller is not None:
                print("  " + self.controller.format_report())
            if self.prefix_controller is not None:
                print("  " + self.prefix_controller.format_report())
            # The controller's own report, carrying `sat_hi` -- that is what
            # decides whether the run measured the mechanism or the box.
            if self.occ_gate_controller is not None:
                print("  " + self.occ_gate_controller.format_report())
            self.consider_checkpoint(val)
            # After `consider_checkpoint`, because the weight average is to
            # compete with the live weights for the SAME `self.best` rather
            # than in place of them.
            self._rozwaz_srednia_wag()
        if self.best is None:
            # Not a silent finish: every epoch was refused, so there is no
            # `_fundational.pt`, and a missing file reads as "not there yet".
            raise NoAdmissibleCheckpoint(
                "no epoch met training.recon_admit = %.4f, so NO checkpoint "
                "was written in %d epochs. The run trained; it produced "
                "nothing admissible. Last rejection: %s"
                % (self.cfg.training.recon_admit, n_epochs,
                   self.last_rejection or "<none recorded>"))
        return self.history

    def _rozwaz_srednia_wag(self) -> None:
        """SWA: the mean of the last `swa_window` epochs, as a separate
        candidate.

        At `swa_start_epoch = 0` it leaves on the first line, so the run is
        bit for bit as before. An error inside must not kill the run: it is
        caught, printed with its traceback, and disables SWA for the rest of
        the run loudly.
        """
        tc = self.cfg.training
        start = int(getattr(tc, "swa_start_epoch", 0) or 0)
        okno = int(getattr(tc, "swa_window", 0) or 0)
        if start <= 0 or okno < 2 or int(self.epoch) < start:
            return
        if getattr(self, "_swa_martwe", False):
            return
        try:
            self._swa_step(okno)
        except Exception as exc:                       # noqa: BLE001
            import traceback
            self._swa_martwe = True
            traceback.print_exc()
            print("  SWA DISABLED for the rest of the run after an error: %r"
                  % (exc,))

    def _swa_step(self, okno: int) -> None:
        """One SWA epoch: add the weights, average, score, maybe save."""
        from ..model.quantizer import collect_manifold_state_names

        if not hasattr(self, "_swa_stany"):
            self._swa_stany = []
            # Excluded from the average: the L0 codebook points lie on the
            # hyperboloid, and an arithmetic mean of ambient coordinates takes
            # them off the manifold. Measured earlier at
            # max|<x,x>_L + 1| = 5.05e-02 against a tolerance of 1e-16, with a
            # reported reconstruction of 0.9945 against 0.8012 from the
            # checkpoint that was actually written.
            self._swa_zab = set(collect_manifold_state_names(self.model))
            print("  SWA: start (window %d epochs) | manifold tensors "
                  "excluded from the average: %d"
                  % (okno, len(self._swa_zab)))

        zywe = {k: v.detach().to("cpu").clone()
                for k, v in self.model.state_dict().items()}
        self._swa_stany.append(zywe)
        while len(self._swa_stany) > okno:
            self._swa_stany.pop(0)
        n = len(self._swa_stany)
        if n < 2:
            print("  SWA: %d of %d window epochs collected -- there is no "
                  "average yet" % (n, okno))
            return

        sred = {}
        n_sred = 0
        for name, v in zywe.items():
            if name in self._swa_zab or not torch.is_floating_point(v):
                sred[name] = v
                continue
            acc = torch.zeros_like(v, dtype=torch.float64)
            for sd in self._swa_stany:
                acc += sd[name].to(torch.float64)
            sred[name] = (acc / float(n)).to(v.dtype)
            n_sred += 1

        missing, nad = self.model.load_state_dict(sred, strict=False)
        if missing or nad:
            raise RuntimeError(
                "SWA: the averaged state_dict does not match | missing %d %s "
                "| unexpected %d %s" % (len(missing), list(missing)[:4],
                                        len(nad), list(nad)[:4]))
        try:
            val_s = self.evaluate("val")
        finally:
            # The restore is in `finally`: if the evaluation raised, the model
            # would be left holding the averaged weights and the NEXT epoch
            # would train from them -- the run would quietly become a different
            # run.
            b2, n2 = self.model.load_state_dict(zywe, strict=False)
            if b2 or n2:
                raise RuntimeError(
                    "SWA: failed to restore the live weights")

        ok, why = self.admit_checkpoint(val_s)
        score = self.selection_score(val_s)
        print("  SWA(%d epochs, %d tensors averaged): composite %.5f | "
              "%s %.5f | rec_full %.4f | admitted %s"
              % (n, n_sred, score, self.cfg.training.select_on,
                 self.selection_metric(val_s),
                 float(val_s.get("rec_full", float("nan"))), ok))
        if not ok:
            print("  SWA: NOT ADMITTED (%s)" % why)
            return
        if self.best is not None and score <= self.best:
            print("  SWA: composite %.5f <= best %.5f -- not saved"
                  % (score, self.best))
            return
        self.best = score
        self.model.load_state_dict(sred, strict=False)
        try:
            self.save_checkpoint(self.checkpoint_path("fundational"), val_s)
        finally:
            self.model.load_state_dict(zywe, strict=False)
        print("  SWA: NEW BEST composite %.5f -> %s (average of %d epochs, "
              "epochs %d..%d)"
              % (score,
                 os.path.basename(self.checkpoint_path("fundational")),
                 n, int(self.epoch) - n + 1, int(self.epoch)))

    def consider_checkpoint(self, val: Dict[str, float]) -> None:
        """Admission first, score second. Both printed, whichever way it goes.

        The order matters: scoring an inadmissible epoch and then discarding
        it would leave `self.best` describing a model that was never saved,
        and the next admissible epoch would be compared against a bar no file
        on disk ever met.
        """
        ok, why = self.admit_checkpoint(val)
        score = self.selection_score(val)
        if getattr(self.settings, "save_every_epoch", False):
            arch = self.save_checkpoint(
                self.checkpoint_path("ep%03d" % self.epoch), val)
            print("  archive: %s (composite %.5f, admitted %s)"
                  % (os.path.basename(arch), score, ok))
        # The spy track runs on every epoch, admitted on val or not. The two
        # rules answer different questions, so letting the val verdict gate
        # the spy one would silently make spy a subset of val -- and the whole
        # reason for a second track is that the best design model need not be
        # the best foundation model.
        self.consider_spy_checkpoint(val)
        if not ok:
            self.last_rejection = "epoch %d: %s" % (self.epoch, why)
            print("  checkpoint REFUSED (epoch %d): %s | composite %.5f "
                  "would %s the standing best %s"
                  % (self.epoch, why, score,
                     "beat" if (self.best is None or score > self.best)
                     else "not beat",
                     "%.5f" % self.best if self.best is not None else "(none)"))
            return
        if self.best is not None and score <= self.best:
            print("  checkpoint: admitted (%s) but composite %.5f <= best "
                  "%.5f -- kept" % (why, score, self.best))
            return
        self.best = score
        # One write, one name. `_best.pt` used to be a second copy of the same
        # weights under an older name -- measured on a pair from a live run:
        # 460 tensors, 0 different, max|difference| 0.000e+00. Two names for
        # one file are not backward compatibility but a second source of
        # truth, since each probe picked one of them and nothing checked that
        # it picked the same one. `run()` warns at startup if a file of that
        # name is left on disk from an earlier run.
        self.save_checkpoint(self.checkpoint_path("fundational"), val)
        print("  checkpoint: new best composite %.5f (%s %.5f, rec_full "
              "%.4f, recon weight %.2f) -> %s"
              % (score, self.cfg.training.select_on,
                 self.selection_metric(val),
                 float(val.get("rec_full", float("nan"))),
                 float(self.cfg.training.select_recon_weight),
                 os.path.basename(self.checkpoint_path("fundational"))))


# Assembly
def build_corpus(cfg: HyppeConfig, settings: RunSettings,
                 strain_names: Sequence[str]) -> Corpus:
    if settings.synthetic:
        return Corpus.synthetic(n=int(settings.synthetic),
                                seq_len=int(cfg.data.sequence_length),
                                seed=int(cfg.training.seed))
    # CLI over config, config over nothing. `build_corpus` used to read only
    # the settings, so `data.sequences_path` and its siblings were declared in
    # every config file and consulted by no line of `src/`: a key that looks
    # effective. Filling a MISSING setting from the config cannot change a run
    # that already worked, because an empty setting raised.
    resolved = {
        "sequences_path": settings.sequences_path or cfg.data.sequences_path,
        "combat_path": settings.combat_path or cfg.data.combat_path,
        "samples_path": settings.samples_path or cfg.data.samples_path,
        # `splits_path` is deliberately NOT here: DataConfig declares no such
        # key and the loader rejects it, so --splits is the only source.
        "splits_path": settings.splits_path,
    }
    # Optional, and therefore outside `resolved`: an empty path selects the
    # per-cell regime, while `resolved` lists paths whose ABSENCE is an error.
    # Putting it there made the loop refuse to start any run without a
    # per-sample target.
    sample_text_path = str(getattr(cfg.data, "sample_text_path", "") or "")
    rank_path = str(getattr(cfg.data, "rank_path", "") or "")
    if rank_path and not os.path.isfile(rank_path):
        raise ValueError(
            "data.rank_path points at %r, which does not exist. An empty "
            "value takes the pair label from the value matrix on purpose; a "
            "WRONG value must not fall back to it silently, because the two "
            "runs differ in what the pairwise head learns." % (rank_path,))
    if sample_text_path and not os.path.isfile(sample_text_path):
        raise ValueError(
            "data.sample_text_path points at %r, which does not exist. An "
            "empty value selects the per-cell regime on purpose; a WRONG "
            "value must not fall back to it silently, because the two runs "
            "differ in what they optimise and the log would not say which "
            "one ran." % (sample_text_path,))
    for key, value in resolved.items():
        if value and not getattr(settings, key):
            print("  path from config: %-16s %s" % (key, value))
    missing = [k for k, v in resolved.items() if not v]
    if missing:
        raise ValueError(
            "no data paths for %s and --synthetic not set. The loop refuses "
            "to invent a corpus: a run whose provenance is ambiguous is worse "
            "than no run." % (", ".join(sorted(missing)),))
    flank = settings.flank_cache_path or cfg.data.flank_embedding_path
    if flank and not settings.flank_cache_path:
        print("  path from config: %-16s %s" % ("flank_cache", flank))
    # The experiment description, switched by one key:
    # `conditioning.experiment_text.enabled`. False means the loop supplies no
    # text and the corpus report says so; true means the path must exist,
    # because a channel declared on and silently empty is indistinguishable in
    # every log from a channel that does not help -- which once left this
    # channel running on its unknown token for 167 195 rows.
    _tc = getattr(getattr(cfg, "conditioning", None),
                  "experiment_text", None)
    text_path = ""
    text_dim = None
    if _tc is not None and bool(getattr(_tc, "enabled", False)):
        text_path = (settings.text_cache_path
                     or getattr(cfg.data, "text_embedding_path", "")
                     or "")
        text_dim = int(getattr(_tc, "dim", 0)) or None
        if not text_path:
            raise ValueError(
                "conditioning.experiment_text.enabled is true but no "
                "cache path is set. Give --text-cache or set "
                "data.text_embedding_path. The loop refuses to run an "
                "enabled channel on its unknown token: that state is "
                "identical in the log to a channel that carries "
                "nothing.")
        if not settings.text_cache_path:
            print("  path from config: %-16s %s"
                  % ("text_cache", text_path))
    # The gene the promoter drives. Same one-key rule and same refusal as the
    # experiment description above.
    _gc = getattr(getattr(cfg, "conditioning", None), "gene", None)
    orth_path = ""
    if _gc is not None and bool(getattr(_gc, "enabled", False)):
        orth_path = (getattr(cfg.conditioning, "orthology_path", "") or
                     os.path.join(_repo_root(), "data", "final_datasets",
                                  "orthology", "og_table_combined.tsv"))
        if not os.path.isfile(orth_path):
            raise ValueError(
                "conditioning.gene.enabled is true but no orthology table is "
                "at %r. Set conditioning.orthology_path. The loop refuses to "
                "run an enabled channel on its unknown token." % (orth_path,))
        print("  path from config: %-16s %s" % ("orthology", orth_path))
    return Corpus.from_paths(
        sequences_path=resolved["sequences_path"],
        combat_path=resolved["combat_path"],
        samples_path=resolved["samples_path"],
        splits_path=resolved["splits_path"],
        sample_text_path=sample_text_path,
        rank_path=rank_path,
        split_column=settings.split_column,
        strain_names=strain_names,
        seq_len=int(cfg.data.sequence_length),
        max_orthogroups=int(settings.max_orthogroups),
        min_obs_for_sigma=int(cfg.data.min_obs_for_sigma),
        flank_cache_path=flank,
        flank_index_path=settings.flank_index_path,
        text_cache_path=text_path,
        text_index_path=settings.text_index_path,
        text_expect_dim=text_dim,
        # No `all` column when the head predicts categories: it is
        # a genome-specific mixture, not a category. The
        # single-target path still needs it, and still gets it.
        include_marginal=not bool(getattr(
            cfg.data, "target_per_category", False)),
        orthology_path=orth_path)


def build_loop(cfg: HyppeConfig, settings: RunSettings) -> TrainLoop:
    torch.manual_seed(int(cfg.training.seed))
    np.random.seed(int(cfg.training.seed))
    model = HyppeModel.build(cfg)
    corpus = build_corpus(cfg, settings, model.strain_names)
    print(corpus.report.format())
    # The encoder builds `text_proj = Linear(cond_d_lm, cond_d_ctx)` from the
    # category prototype width and the corpus loads vectors of its own. They
    # must agree, and they must agree here -- at construction, with both
    # numbers in the message -- rather than as a matmul error on the first
    # forward.
    if getattr(corpus, "text_cache", None) is not None:
        _dlm = int(model.condition.cfg.cond_d_lm)
        _dtx = int(corpus.text_cache.dim)
        if _dlm != _dtx:
            raise ValueError(
                "the experiment-text cache is %d dims wide and the "
                "condition encoder takes %d "
                "(ConditionConfig.cond_d_lm, set from the category "
                "prototype matrix in assemble.py). Rebuild the cache "
                "with the same sentence encoder."
                % (_dtx, _dlm))
        print("  text channel: %d dims -> text_proj(%d -> %d); "
              "cond_dropout_text = %.2f, drawn PER PAIR GROUP, and it "
              "now drops a REAL vector rather than the unknown token "
              "onto itself"
              % (_dtx, _dlm, int(model.condition.cfg.cond_d_ctx),
                 float(model.condition.cfg.cond_dropout_text)))
    maker = BatchMaker(corpus,
                       batch_size=int(settings.batch_size
                                      or cfg.data.batch_size),
                       negative_fraction=float(settings.negative_fraction),
                       seed=int(cfg.training.seed),
                       per_category=bool(getattr(
                           cfg.data, "target_per_category", False)),
                       # Without `sequences_path` the `coding`/`flank`
                       # pools have nothing to be built from, and the
                       # slot falls back to shuffle -- the
                       # `<kind>_unavailable` counter shows that.
                       sequences_path=str(getattr(
                           cfg.data, "sequences_path", "") or ""),
                       negative_mix=(dict(getattr(
                           cfg.heads.discriminator, "negative_mix", {})
                           or {}) or None),
                       # Cross-genome pair supply. At the default 0.0 this is
                       # the previous behaviour bit for bit; above zero the
                       # batch reserves a slice for orthogroup blocks, which
                       # is the only path by which `task_pairwise` can send
                       # gradient to the species channel.
                       ortholog_fraction=float(getattr(
                           cfg.heads.pairwise, "ortholog_fraction", 0.0)),
                       ortholog_block=int(getattr(
                           cfg.heads.pairwise, "ortholog_block", 4)),
                       nonpromoter_pairs=bool(getattr(
                           cfg.heads.pairwise, "nonpromoter_pairs", True)),
                       # Read from the config, not from a local: `rank_path`
                       # is resolved inside `build_corpus`, which is another
                       # scope. `corpus.mu_rank` is a necessary condition on
                       # the `_build` side anyway, so there is no second
                       # source of truth here.
                       pair_from_rank=bool(
                           str(getattr(cfg.data, "rank_path", "") or "")),
                       min_gap_rank=float(getattr(
                           cfg.data, "min_gap_rank", 0.0)))
    return TrainLoop(cfg, model, maker, settings)


# CLI
def _add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="",
                    help="config file; omitted = the dataclass defaults")
    ap.add_argument("--name", default="smoke")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--method", default="",
                    help="scalarise (default, the baseline) | nash | "
                         "normprior")
    ap.add_argument("--sequences", default="")
    ap.add_argument("--combat", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--splits", default="")
    ap.add_argument("--split-column", default="B_block10")
    ap.add_argument("--flank-cache", default="",
                    help="DNA-LM flank embeddings.npy; its index.tsv is read "
                         "from the same directory unless --flank-index says "
                         "otherwise. Omitted = cfg.data.flank_embedding_path; "
                         "neither = the gene_context channel answers with its "
                         "unknown token and the corpus report says so.")
    ap.add_argument("--flank-index", default="",
                    help="row index for --flank-cache (default: index.tsv "
                         "beside it)")
    ap.add_argument("--text-cache", default="",
                    help="sentence-LM experiment-description cache "
                         ".npz, keyed <genome>|<category>. Read ONLY "
                         "when conditioning.experiment_text.enabled is "
                         "true; omitted = cfg.data.text_embedding_path.")
    ap.add_argument("--text-index", default="",
                    help="optional row index for --text-cache")
    ap.add_argument("--max-orthogroups", type=int, default=0)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="build a synthetic corpus of N rows instead")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-eval-steps", type=int, default=0)
    ap.add_argument("--negative-fraction", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--no-constraint", action="store_true",
                    help="run reconstruction at its FIXED declared weight")
    ap.add_argument("--recon-target", type=float, default=-1.0)
    ap.add_argument("--skip-gates", action="store_true",
                    help="refuses unless --i-know: the gates are the point")
    ap.add_argument("--i-know", action="store_true")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="archive <name>_ep###.pt every epoch so a "
                         "different selection rule can be tested "
                         "without retraining")
    ap.add_argument("--cost-probe", action="store_true",
                    help="time one step of each method and print the ratio")
    ap.add_argument("--allow-euclidean-manifold", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="short run plus T1-T5, the acceptance test")


def _settings_from_args(a: argparse.Namespace) -> RunSettings:
    return RunSettings(
        name=a.name, out_dir=a.out_dir, device=a.device,
        sequences_path=a.sequences, combat_path=a.combat,
        samples_path=a.samples, splits_path=a.splits,
        flank_cache_path=a.flank_cache, flank_index_path=a.flank_index,
        text_cache_path=getattr(a, "text_cache", ""),
        text_index_path=getattr(a, "text_index", ""),
        split_column=a.split_column, max_orthogroups=a.max_orthogroups,
        synthetic=a.synthetic, max_steps=a.max_steps,
        max_eval_steps=a.max_eval_steps, log_every=a.log_every,
        epochs=a.epochs, batch_size=a.batch_size,
        negative_fraction=a.negative_fraction, method=a.method,
        skip_gates=bool(a.skip_gates), cost_probe=bool(a.cost_probe),
        save_every_epoch=bool(a.save_every_epoch),
        allow_euclidean_manifold=bool(a.allow_euclidean_manifold))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hyppe-train",
        description="Train HyppeModel. Scalarisation is the default and the "
                    "baseline; reconstruction is a constraint, not a player.")
    _add_arguments(ap)
    a = ap.parse_args(argv)
    if a.skip_gates and not a.i_know:
        print("--skip-gates without --i-know. The wiring gates are what "
              "distinguishes a balancing method that is wired from one that "
              "is configured; a run without them measures the sum.")
        return EXIT_GATE

    cfg = load_config(a.config) if a.config else HyppeConfig()
    if a.method:
        cfg.training.method = a.method
    if a.no_constraint:
        cfg.training.recon_constraint.enabled = False
    if a.recon_target > 0:
        cfg.training.recon_constraint.target = float(a.recon_target)
    if a.epochs:
        cfg.training.epochs = int(a.epochs)
    cfg.validate()

    settings = _settings_from_args(a)
    print("=" * 78)
    print("config: %s" % (cfg.source or "dataclass defaults",))
    _inert_cfg = list(report_inert_keys(cfg))
    for key, why in _inert_cfg:
        print("  inert  %-38s %s" % (key, why))
    print("=" * 78)

    loop = build_loop(cfg, settings)
    print(loop.header())
    for note in loop.model.inert_keys():
        print("  " + str(note))
    # The model half of the inert-key report. The print above calls
    # `report_inert_keys` WITHOUT the model, because the model is only built
    # in `build_loop`, so every entry found by the parameter names of the
    # BUILT model is lost -- including the only one that detects a dead channel
    # ("nothing in the built model consumes it"), and entries about
    # no_decay/manifold patterns matching no parameter name. Measured on the
    # delivered config: `cfg` alone gives 6 entries, `(cfg, model)` gives 14.
    try:
        _widziane = set(k for k, _ in _inert_cfg)
        for key, why in report_inert_keys(cfg, loop.model):
            if key not in _widziane:
                print("  inert* %-38s %s" % (key, why))
    except Exception as exc:                    # never kill the run
        print("  inert* report FAILED %s: %s" % (type(exc).__name__, exc))

    if not settings.skip_gates:
        code = loop.wiring_gates()
        if code:
            print("wiring gates FAILED: the method is not wired. Stopping.")
            return EXIT_GATE
    if a.gates_only:
        return EXIT_OK

    if settings.cost_probe:
        _cost_probe(loop)

    if a.smoke:
        # The gate runs the real `train_step`, so it can diverge too -- and a
        # gate that diverges must report it as a FAILED GATE, not as a
        # traceback the queue reads as "crashed for unknown reasons".
        try:
            return smoke(loop)
        except DivergenceError as exc:
            print("\nGATE FAILED (divergence during the smoke run): %s"
                  % (exc,))
            return EXIT_DIVERGED
    # A divergence is an exit code, not a traceback on stdout: the queue reads
    # the exit status, and the failure that produced this whole guard was a
    # run that diverged and returned 0, so the queue recorded a success and
    # started the next arm.
    try:
        loop.run()
    except NoAdmissibleCheckpoint as exc:
        print("\n" + "!" * 78)
        print("NO CHECKPOINT: %s" % (exc,))
        print("!" * 78)
        return EXIT_NO_CHECKPOINT
    except DivergenceError as exc:
        print("\n" + "!" * 78)
        print("RUN STOPPED: %s" % (exc,))
        print("!" * 78)
        return EXIT_DIVERGED
    return EXIT_OK


def _cost_probe(loop: TrainLoop) -> None:
    """Price the method on THIS encoder, rather than quoting a cost measured
    on another model."""
    batch = next(iter(loop.maker.epoch("train", shuffle=False, seed=0,
                                       max_steps=1))).to(loop.device)
    base = Scalarisation(dict(loop.task_weights))
    saved = loop.weighting

    def one(method: TaskWeighting) -> float:
        def step():
            loop.optimiser.zero_grad(set_to_none=True)
            players, sub, _s, _o = loop.compute_losses(batch, track=False)
            method.apply(players, subordinate=sub)
            loop.optimiser.zero_grad(set_to_none=True)
        return measure_step_cost(step, n=3, warmup=1)

    t_base = one(base)
    t_run = one(saved) if saved.name != base.name else t_base
    print("  step cost: scalarise %.3f s | %s %.3f s | ratio %.2fx"
          % (t_base, saved.name, t_run, t_run / max(t_base, 1e-9)))


# Acceptance
def smoke(loop: TrainLoop) -> int:
    """T1-T5. Every one of them prints a number and its threshold.

    T1  the loss falls (first third against last third of the steps);
    T2  F1/F2/F3 passed -- already run in `main`, restated with the numbers;
    T3  the Gram off-diagonal is printed, with its verdict;
    T4  reconstruction stays above `RECONSTRUCTION_FLOOR` and the multiplier
        moved, since a controller pinned to a bound is a constant weight;
    T5  the metric of the reloaded checkpoint equals the metric of the live
        model: exactly for every evaluation metric, and to
        `T5_COMPONENT_TOL` for the per-batch loss components.
    """
    s = loop.settings
    print("\n" + "=" * 78)
    print("SMOKE RUN: %d step(s) max, batch %d, split sizes %s"
          % (s.max_steps, loop.maker.batch_size,
             loop.maker.corpus.report.per_split))
    print("=" * 78)
    losses: List[float] = []
    tasks: List[float] = []
    recon: List[float] = []
    lam: List[float] = []
    glue = float("nan")
    every = loop.cfg.training.refresh_cadence(loop.method)
    n = 0
    loop.weighting.new_epoch()
    if loop.controller is not None:
        loop.controller.new_epoch()
    loop.epoch = 1
    for batch in loop.maker.epoch("train", shuffle=True, seed=1,
                                  max_steps=s.max_steps):
        b = batch.to(loop.device)
        if n == 0:
            info = loop.measure_orthogonality(b)
            loop.model.eval()
            with torch.no_grad():
                out = loop.model(b.x, species_id=b.species_id,
                                 flank_l=b.flank_l, flank_r=b.flank_r,
                                 flank_ok=b.flank_ok,
                                 text_emb=b.text_emb,
                                 text_ok=b.text_ok,
                                 gene_numeric=(b.gene_numeric
                                               if loop.model.gene is not None
                                               else None),
                                 gene_ok=(b.gene_ok
                                          if loop.model.gene is not None
                                          else None),
                                 pair_group=b.pair_group, reconstruct=False)
                glue = loop.check_pair_glue(out, b, out.get("context"))
        st = loop.train_step(b, recompute=(n % every == 0))
        losses.append(st["loss"])
        tasks.append(sum(v for k, v in st.items() if k.startswith("task_")))
        recon.append(st.get("rec_full", float("nan")))
        lam.append(st.get("recon_lambda", float("nan")))
        n += 1
        if s.log_every and n % s.log_every == 0:
            # The same formatter as the real run: a gate that prints a
            # different decomposition from the run it gates is a gate on a
            # different quantity.
            print("  " + loop.format_step(1, n, st))
    if n < 6:
        print("FAIL: %d steps is too few to read a trend" % (n,))
        return EXIT_FALSIFIER

    fails: List[str] = []
    third = max(1, n // 3)
    head = float(np.mean(tasks[:third]))
    tail = float(np.mean(tasks[-third:]))
    lag_head = float(np.mean(losses[:third]))
    lag_tail = float(np.mean(losses[-third:]))
    ok1 = tail < head
    # T1 is read on the prediction objective, not on the Lagrangian. `loss` is
    # L_pred + lambda * L_recon with lambda rising while the constraint is
    # violated, so the total is not a descent quantity -- the controller
    # increases it on purpose. On the first synthetic run of this loop, lambda
    # went 0.15 -> 100 in ten steps: the task losses fell while the total went
    # 5.31 -> 145.63, and a criterion read on the total would have called a
    # working controller a broken loop.
    print("\nT1  prediction objective (sum of the player losses) falls: "
          "first %d steps %.4f -> last %d steps %.4f (delta %+.4f) ... %s"
          % (third, head, third, tail, tail - head,
             "PASS" if ok1 else "FAIL"))
    print("    Lagrangian total %.4f -> %.4f (delta %+.4f) -- REPORTED, not "
          "the criterion: lambda rises while the constraint is violated"
          % (lag_head, lag_tail, lag_tail - lag_head))
    if not ok1:
        fails.append("T1")

    print("T2  wiring gates F1/F2/F3: see the report above (exit code %d "
          "means all passed) ... %s"
          % (0, "PASS" if not s.skip_gates else "NOT RUN"))
    if s.skip_gates:
        fails.append("T2")
    # NaN used to read as a passed gate, so a branch that never ran passed
    # silently. The gate requires a NUMBER.
    ok_glue = (not math.isnan(glue)) and glue <= TrainLoop.PAIR_GLUE_TOL
    if math.isnan(glue):
        print("    pair glue: NOT COMPUTED (no pairs in the control batch). "
              "Silence is not a pass.")
    print("    pair glue |indexed score - head.score(row)| / scale = %.3e "
          "(tol %.0e: the two differ only by the matmul reduction order at a "
          "different batch size; a pair built from the wrong rows moves this "
          "by O(1)) ... %s"
          % (glue, TrainLoop.PAIR_GLUE_TOL, "PASS" if ok_glue else "FAIL"))
    if not ok_glue:
        fails.append("T2-glue")

    if loop.zero_gradient_steps:
        print("    players dropped for an exactly zero loss (a satisfied "
              "hinge has no gradient, and a zero-gradient player takes the "
              "MAXIMUM share of a bargain): %s of %d steps"
              % (", ".join("%s %d" % kv
                           for kv in sorted(loop.zero_gradient_steps.items())),
                 n))

    info = loop.orthogonality
    mx = float(info.get("offdiag_max", float("nan")))
    print("T3  Gram offdiag_max %.4f (floor %.2f), offdiag_mean %.4f, "
          "cos_min %.4f ... %s"
          % (mx, ORTHOGONALITY_FLOOR,
             float(info.get("offdiag_mean", float("nan"))),
             float(info.get("cos_min", float("nan"))),
             "PRINTED" if mx == mx else "FAIL"))
    print("    " + str(info.get("verdict", "")))
    if mx != mx:
        fails.append("T3")

    # T4 is three questions and only the second is about the floor. An
    # untrained decoder starts BELOW the unigram base rate of 0.2626, not
    # having learnt yet that the four bases are unequal, so "never below the
    # floor" is unsatisfiable at step 1 by construction -- and a falsifier that
    # cannot be satisfied is not a falsifier. The claim is that reconstruction
    # does not collapse while the controller runs, so the three questions are
    # asked separately.
    measured_floor, composition = loop.maker.corpus.base_floor("train")
    floor = float(s.recon_min) if s.recon_min > 0 else measured_floor
    worst = float(np.nanmin(recon)) if recon else float("nan")
    last = recon[-1] if recon else float("nan")
    span = (max(lam) / min(lam)) if lam and min(lam) > 0 else float("nan")
    ok4a = last >= recon[0]
    ok4b = last >= floor
    print("T4  reconstruction: first %.4f -> last %.4f (worst %.4f)"
          % (recon[0], last, worst))
    print("    T4a no collapse: last >= first ... %s (delta %+.4f)"
          % ("PASS" if ok4a else "FAIL", last - recon[0]))
    print("    T4b above the base rate %.4f MEASURED on this corpus "
          "(composition %s; the repo constant for the real promoter corpus "
          "is %.4f) ... %s"
          % (floor, " ".join("%s %.3f" % kv
                             for kv in sorted(composition.items())),
             RECONSTRUCTION_FLOOR, "PASS" if ok4b else "FAIL"))
    if loop.controller is not None:
        r = loop.controller.report()
        print("    lambda %.4f, span %.2fx over the run, sat %.3f "
              "(hi %.3f / lo %.3f); a controller on a bound IS a constant "
              "weight" % (r.get("lam", float("nan")), span,
                          r.get("sat", float("nan")),
                          r.get("sat_hi", float("nan")),
                          r.get("sat_lo", float("nan"))))
        warm = int(loop.cfg.training.recon_constraint.warmup_steps)
        ok4c = n > warm and span > 1.0
        print("    T4c the controller ACTED: lambda span %.3fx over %d steps "
              "past a %d-step warm-up ... %s (a multiplier that never moved "
              "is a constant weight and reads identically in every other "
              "number here)"
              % (span, n, warm, "PASS" if ok4c else "FAIL"))
        if not ok4c:
            fails.append("T4c")
            if n <= warm:
                print("      the run is SHORTER than the warm-up, so T4c "
                      "could not be satisfied by construction: give "
                      "--max-steps > %d or lower "
                      "training.recon_constraint.warmup_steps" % (warm,))
        if r.get("violations"):
            print("    controller violations: " + " ; ".join(r["violations"]))
    else:
        print("    no controller: reconstruction ran at the FIXED effective "
              "coefficient %.4f"
              % (loop.cfg.training.effective_recon_coefficient("full"),))
    if not ok4a:
        fails.append("T4a")
    if not ok4b:
        fails.append("T4b")

    # T5
    live = loop.evaluate("val", max_steps=s.max_eval_steps or 2)
    path = loop.save_checkpoint(loop.checkpoint_path("smoke"), live)
    loop.load_checkpoint(path)
    again = loop.evaluate("val", max_steps=s.max_eval_steps or 2)
    keys = sorted(set(live) & set(again))
    deltas = {k: abs(float(live[k]) - float(again[k])) for k in keys
              if not math.isnan(float(live[k]))}
    # The metrics are held to zero and the per-batch loss components to
    # `T5_COMPONENT_TOL`; see that constant for the measurement behind the
    # split. `worst_d` keeps its old meaning -- the worst delta anywhere -- so
    # the json field and every stored gate stay comparable.
    m_del = {k: v for k, v in deltas.items() if not k.startswith("comp_")}
    c_del = {k: v for k, v in deltas.items() if k.startswith("comp_")}
    worst_k = max(deltas, key=deltas.get) if deltas else ""
    worst_d = deltas.get(worst_k, float("nan"))
    m_key = max(m_del, key=m_del.get) if m_del else ""
    m_worst = m_del.get(m_key, 0.0)
    c_key = max(c_del, key=c_del.get) if c_del else ""
    c_worst = c_del.get(c_key, 0.0)
    ok5m = m_worst == 0.0
    ok5c = c_worst <= T5_COMPONENT_TOL
    ok5 = ok5m and ok5c
    print("T5  checkpoint round-trip over %d metrics: max|live - reloaded| = "
          "%.3e on %r ... %s"
          % (len(deltas), worst_d, worst_k, "PASS" if ok5 else "FAIL"))
    print("    T5a %d evaluation metrics, tolerance ZERO: max %.3e on %r "
          "... %s" % (len(m_del), m_worst, m_key or "<none>",
                      "PASS" if ok5m else "FAIL"))
    print("    T5b %d loss components, tolerance %.1e: max %.3e on %r ... %s "
          "(the components are float32 reductions over the evaluation "
          "batches and are NOT bit-reproducible on this device: measured "
          "2.235e-09 between two evaluations with no checkpoint between "
          "them)"
          % (len(c_del), T5_COMPONENT_TOL, c_worst, c_key or "<none>",
             "PASS" if ok5c else "FAIL"))
    print("    live     pcc %.5f scc %.5f pair_acc %.5f auc %.5f rec_full %.5f"
          % (live.get("pcc", float("nan")), live.get("scc", float("nan")),
             live.get("pair_acc", float("nan")),
             live.get("disc_auc", float("nan")),
             live.get("rec_full", float("nan"))))
    print("    reloaded pcc %.5f scc %.5f pair_acc %.5f auc %.5f rec_full %.5f"
          % (again.get("pcc", float("nan")), again.get("scc", float("nan")),
             again.get("pair_acc", float("nan")),
             again.get("disc_auc", float("nan")),
             again.get("rec_full", float("nan"))))
    if not ok5m:
        fails.append("T5a")
    if not ok5c:
        fails.append("T5b")

    summary = {"steps": n, "task_loss_first": head, "task_loss_last": tail,
               "total_first": lag_head, "total_last": lag_tail,
               "recon_first": recon[0], "recon_last": last,
               "lambda_span": span, "recon_worst": worst, "offdiag_max": mx,
               "checkpoint_delta": worst_d,
               # Split out, because one number cannot say which half moved: a
               # metric moving at 1e-9 is a leak, a component moving at 1e-9
               # is the GPU.
               "checkpoint_delta_metrics": m_worst,
               "checkpoint_delta_components": c_worst,
               "checkpoint_delta_component_tol": T5_COMPONENT_TOL,
               "zero_gradient_steps": dict(loop.zero_gradient_steps),
               "method": loop.method, "failed": fails}
    with open(os.path.join(s.out_dir, "%s_smoke.json" % s.name), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nSMOKE %s%s" % ("PASSED" if not fails else "FAILED: ",
                            "" if not fails else ", ".join(fails)))
    return EXIT_OK if not fails else EXIT_FALSIFIER


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
