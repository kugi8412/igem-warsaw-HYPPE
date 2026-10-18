"""Choose the checkpoint on SPY, and say what the choice was bought with.

Selection does not maximise pairwise accuracy on `val`, for two reasons.
`val` is 3 725 orthogroups of general promoters while the deliverable is ten
sequences for one gene in one strain; and pairwise accuracy cannot see a
design failure, because a model can order two natural promoters correctly and
be blind to a two-base edit -- and every one of the ten designs IS an edit.
The panel measures that directly (`edit_resolution`, `motif_sensitivity`).

The score is a geometric mean, and the gates are separate. A weighted sum
lets one factor buy back another: the precedent in this repo is a checkpoint
saved at pair_acc 0.7254 with rec_full 0.3139, i.e. 7 % of the achievable
reconstruction. A geometric mean goes to zero with either factor, so a
collapsed component cannot be traded at any weight. Each component is first
mapped to the fraction of its own achievable range above its own floor,
because a geometric mean of numbers with different offsets weights them by
the offsets.

The hard gates are not part of the score, because a score can always be
raised somewhere else while a floor cannot be traded. They are `rec_full`
above `training.recon_admit`, since the design route is decode(codes) and a
model that cannot reconstruct cannot be inverted; `edit_moves`, since a model
blind to a two-base edit cannot rank its own designs; and `anchor_order`,
since four of the ten wet slots ask for a promoter BETWEEN tef1 and pks1.
Every refusal is printed with its numbers, so that a missing `_full.pt` is
never indistinguishable from one that is merely not written yet.

A high spy score proves less than it looks. Spy is 249 sequences, 45 of them
with a measured mu, so the standard error of a Spearman is roughly 0.15 and
`is_tie()` says so rather than letting an argmax pretend otherwise. Selecting
on spy also SPENDS spy: the only untouched number at the end is `test`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import RECONSTRUCTION_FLOOR
from ..data.spy_panel import SCORED_NATIVE_CLASSES, SpyPanel

__all__ = [
    "SPY_COMPONENTS",
    "SpyScorer",
    "SelectionPolicy",
    "SelectionVerdict",
]

#: component -> (floor, ceiling, weight). The floor is CHANCE for that
#: quantity, not zero: normalising an AUC from 0 would make a coin flip read
#: 0.5 of the way to perfect and the composite would be dominated by a
#: constant. Weights are the project's own priorities, written down once:
#: ranking is the oracle's job, motif sensitivity is what makes the designs
#: steerable, reconstruction is what makes them invertible.
SPY_COMPONENTS: Dict[str, Tuple[float, float, float]] = {
    #: Strength order over the 65 scored natives, and the largest weight
    #: because it is the first priority. SE under H0 at n = 65 is 0.125,
    #: which is also the resolution of the whole selection rule.
    "spy_scc":          (0.0, 1.0, 0.34),
    #: Promoter against GENE BODY (`gen_kontrola`, 300 records).
    "spy_auc_coding":   (0.5, 1.0, 0.20),
    #: Natives against `losowa_euler4` (76), a dinucleotide-preserving Euler
    #: shuffle on the SAME strand. `reverse` is deliberately excluded: it is
    #: the other strand, and a linear 5-mer probe separates it at AUC 0.9998,
    #: so it would inflate the number without the model's help.
    "spy_auc_euler":    (0.5, 1.0, 0.16),
    #: Whether single mutations move the score. Weight 0: the component is
    #: measured and printed but does not weigh the composite, because it has
    #: read 0.38-0.58 in every epoch of every run against a chance-level
    #: floor. It did harm at a non-zero weight, since a hard component
    #: sitting at zero contributes a constant factor of 0.501 that halves
    #: the spread of the composite against an ABSOLUTE tie threshold. The
    #: property it protected is covered by `require_edit_resolution`.
    #: Falsifier: a value above the floor in any epoch means the weight
    #: should come back. The floor is computed, not chosen:
    #: 0.500 + z(0.05/14)*0.089 = 0.739, i.e. chance plus Bonferroni over
    #: the 14 epochs of a run at a family-wise alpha of 0.05.
    "motif_auc_small":  (0.74, 1.0, 0.00),
    #: Where the anchors are placed: pks1 and tef1 sit at percentile 92.6
    #: and 99.9 in the P1 corpus. Before this component existed the model
    #: placed pks1 at ~19-28.
    #:
    #: It cannot be a hard gate: with two anchors the percentile has
    #: sd = 0.204 under H0, so a "chance + z(0.05/16)*SE" floor comes out
    #: above one. The predecessor `spy_anchor_fit` failed that way, reading
    #: 0.0000 in 33 epochs of 33 from a floor 2.53 sd above random.
    #:
    #: The floor is 0.0 and not the random level 0.536, because the model
    #: measures 0.5327 / 0.3418 / 0.3848, i.e. BELOW random, and a floor at
    #: the random level would zero the component in every epoch. The price
    #: is that a random model is credited 0.536 here while `spy_scc` credits
    #: it 0.0, so the component scales are not commensurate; the panel
    #: recomputes `spy_anchor_pct_floor` every run so the distance from
    #: chance stays readable.
    "spy_anchor_pct":   (0.0, 1.0, 0.15),
    #: Design goes through `decode(codes)`, so a model that cannot rebuild a
    #: sequence from its codes is unfit however well it predicts.
    "rec_full":         (RECONSTRUCTION_FLOOR, 1.0, 0.15),
    # Removed: `spy_anchor_fit` (weight 0.15) read exactly 0.0000 in 33
    # epochs of 33, from a floor 2.53 sd above random; `spy_confidence`
    # (weight 0.10) was always negative, because the variance head carries
    # no per-sample information against a constant-variance baseline. Both
    # sat on the geometric floor, so 0.25 of the weight took no part in the
    # choice.
}

#: Standard error of a Spearman on n points, ~1/sqrt(n-1). At the panel's 47
#: scored natives that is 0.148, which is the resolution of this whole rule.
def spearman_se(n: int) -> float:
    return float("inf") if n < 3 else 1.0 / math.sqrt(max(n - 1, 1))


# Running the model over the panel
class SpyScorer(object):
    """Model + panel -> the panel metrics, in eval mode, with nothing pinned.

    The panel is scored with the conditioning the DEPLOYMENT has, not the one
    training had: the species token of the design strain, no flanks (a plasmid
    is 5.13 sd outside the genomic flank distribution, so the unknown token is
    the honest value), and the gene token of the target orthogroup. Scoring
    the panel with genomic flanks would flatter every number by a channel the
    wet construct does not have.
    """

    def __init__(self, panel: SpyPanel, species_id: int = 0,
                 gene_features=None, design_og: str = "",
                 gene_numeric: Optional[np.ndarray] = None,
                 gene_text: Optional[np.ndarray] = None,
                 text_emb: Optional[np.ndarray] = None,
                 batch_size: int = 32, seq_len: int = 800,
                 use_flanks: bool = False,
                 category: str = "control",
                 categories: Optional[Sequence[str]] = None) -> None:
        self.panel = panel
        self.species_id = int(species_id)
        # Resolved BY NAME against the head's category list, so a reordering of
        # the list cannot silently re-point the panel at another condition.
        from ..data.labels import DEFAULT_CATEGORIES
        cats = tuple(categories or DEFAULT_CATEGORIES)
        if category not in cats:
            raise ValueError(
                "spy category %r is not one of %r. Refusing to fall back to "
                "column 0: the panel's mu is a `control` measurement (pks1 "
                "anchor 7.4235 against a measured control of 7.355) and "
                "scoring it against another condition compares two different "
                "quantities." % (category, cats))
        self.category = str(category)
        self.category_index = cats.index(category)
        self.categories = cats
        self.gene_numeric = gene_numeric
        self.gene_text = gene_text

        # One gene vector per RECORD, not one for the panel. The 65 scored
        # natives are 65 different orthogroups and the two anchors are
        # different again, so handing every record the design target's gene
        # says "all of these drive pks1" -- false for 66 of them, and a
        # combination the model never saw, since each training row is paired
        # with its own orthogroup. Measured consequence of getting it wrong:
        # val scc 0.69-0.71 while spy_scc sat at 0.11 / 0.05 / -0.16 / 0.12
        # with a negative anchor gap in every epoch.
        #
        # `design_og` is the fallback only, for the designed and mutated
        # sequences that have no orthogroup of their own.
        self.gene_rows: Optional[np.ndarray] = None
        self.gene_ok_rows: Optional[np.ndarray] = None
        self.gene_coverage: Dict[str, float] = {}
        if gene_features is not None:
            ix = gene_features.index()
            width = int(gene_features.numeric.shape[1])
            n = len(panel.records)
            rows = np.zeros((n, width), dtype=np.float32)
            ok = np.zeros(n, dtype=np.float32)
            fb = ix.get(str(design_og), -1)
            own = 0
            fell = 0
            for i, r in enumerate(panel.records):
                j = ix.get(str(getattr(r, "og_id", "") or ""), -1)
                if j >= 0:
                    own += 1
                else:
                    j = fb
                    if j >= 0:
                        fell += 1
                if j >= 0:
                    rows[i] = gene_features.numeric[j]
                    ok[i] = 1.0
            self.gene_rows = rows
            self.gene_ok_rows = ok
            self.gene_coverage = {
                "own_orthogroup": float(own),
                "fell_back_to_design": float(fell),
                "unknown_token": float(n - own - fell),
                "distinct": float(len({getattr(r, "og_id", "")
                                       for r in panel.records
                                       if getattr(r, "og_id", "")})),
            }
        self.text_emb = text_emb
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.use_flanks = bool(use_flanks)
        self._eye = np.eye(4, dtype=np.float32)
        self._code = {c: i for i, c in enumerate("ACGT")}

    # Encoding
    def one_hot(self, seqs: Sequence[str]) -> torch.Tensor:
        """[n, 4, L] in ACGT order. Ambiguous bases become 0.25 everywhere.

        A row containing N cannot be dropped the way the corpus drops it:
        these are the sequences that were ASKED about, and returning fewer
        predictions than sequences is how a panel metric ends up computed on
        a different set from the one it names.

        Short sequences are padded at the FRONT. Position L is the base
        immediately before the ATG and the core promoter sits at the 3' end,
        so padding the tail would shift the region `LatentPooling`'s anchor
        statistic reads.
        """
        n = len(seqs)
        out = np.full((n, 4, self.seq_len), 0.25, dtype=np.float32)
        for i, s in enumerate(seqs):
            s = s.upper()[-self.seq_len:]
            off = self.seq_len - len(s)
            for j, ch in enumerate(s):
                k = self._code.get(ch)
                if k is not None:
                    out[i, :, off + j] = 0.0
                    out[i, k, off + j] = 1.0
        return torch.from_numpy(out)

    # Scoring
    def _column(self, t: torch.Tensor, b: int) -> np.ndarray:
        """One value per SEQUENCE, from a head that emits one per CATEGORY.

        With `data.target_per_category` the absolute head emits `[B, 7]`, so
        flattening and taking the first `b` numbers silently returns the
        seven category outputs of the first nine sequences under the names of
        the first sixty-four.

        Which column is not a guess: the panel's `mu` is the `control`
        measurement (the pks1 anchor carries 7.4235 against a measured
        control of 7.355), so `category` defaults to "control", is resolved
        BY NAME against the head's own list, and raises if the name is absent
        rather than falling back to column 0.
        """
        a = t.detach().cpu().numpy()
        if a.ndim == 1:
            return a[:b]
        if a.ndim != 2 or a.shape[0] != b:
            raise RuntimeError(
                "absolute head returned %r for a batch of %d; the spy scorer "
                "cannot align it to panel names" % (a.shape, b))
        k = int(self.category_index)
        if not 0 <= k < a.shape[1]:
            raise RuntimeError(
                "category index %d outside the head's %d outputs"
                % (k, a.shape[1]))
        return a[:, k]

    @torch.no_grad()
    def predict(self, model) -> Tuple[Dict[str, float], Dict[str, float],
                                      Dict[str, float]]:
        """`(strength, promoterness, variance)` keyed by panel member name.

        The third dict is the head's own predicted variance, `exp(log_var)`,
        which lets a selection rule ask how sure the model is.
        """
        was = model.training
        model.eval()
        names = [r.name for r in self.panel.records]
        seqs = self.panel.sequences()
        strength: Dict[str, float] = {}
        realness: Dict[str, float] = {}
        variance: Dict[str, float] = {}
        device = next(model.parameters()).device
        for start in range(0, len(seqs), self.batch_size):
            chunk = seqs[start:start + self.batch_size]
            x = self.one_hot(chunk).to(device)
            b = x.shape[0]
            kw: Dict[str, object] = {
                "species_id": torch.full((b,), self.species_id,
                                         dtype=torch.long, device=device),
                "reconstruct": False,
            }
            if self.text_emb is not None:
                kw["text_emb"] = torch.from_numpy(
                    np.repeat(self.text_emb[None, :], b, axis=0)).to(device)
            if self.gene_rows is not None and model.gene is not None:
                # The slice for THESE records, in panel order. Asserted
                # rather than assumed, because a misalignment here would pair
                # each promoter with a neighbour's gene and look like a model
                # that is merely worse.
                sl = self.gene_rows[start:start + b]
                if sl.shape[0] != b:
                    raise RuntimeError(
                        "gene rows %d for a batch of %d: the panel order and "
                        "the gene table have come apart" % (sl.shape[0], b))
                kw["gene_numeric"] = torch.from_numpy(
                    np.ascontiguousarray(sl)).to(device)
                kw["gene_ok"] = torch.from_numpy(
                    np.ascontiguousarray(
                        self.gene_ok_rows[start:start + b])).to(device)
            elif self.gene_numeric is not None and model.gene is not None:
                kw["gene_numeric"] = torch.from_numpy(
                    np.repeat(self.gene_numeric[None, :], b, axis=0)).to(device)
                kw["gene_ok"] = torch.ones(b, device=device)
            if self.gene_text is not None and model.gene is not None:
                kw["gene_text"] = torch.from_numpy(
                    np.repeat(self.gene_text[None, :], b, axis=0)).to(device)
            out = model(x, **kw)
            if out.get("absolute") is not None:
                ao = out["absolute"]
                mu = self._column(ao.mu, b)
                lv = getattr(ao, "log_var", None)
                lvv = None if lv is None else self._column(lv, b)
                for i, nm in enumerate(names[start:start + b]):
                    strength[nm] = float(mu[i])
                    if lvv is not None and i < lvv.size:
                        variance[nm] = float(np.exp(lvv[i]))
            if out.get("discriminator") is not None:
                lg = out["discriminator"].logit.detach().cpu().numpy().ravel()
                for i, nm in enumerate(names[start:start + b]):
                    realness[nm] = float(lg[i])
        model.train(was)
        return strength, realness, variance

    # The two components that need sigma, or the panel's own labels
    def anchor_and_confidence(self, strength: Mapping[str, float],
                              variance: Mapping[str, float]
                              ) -> Dict[str, float]:
        """`spy_anchor_fit` and `spy_confidence`, both scale-free.

        Neither is a raw difference. Predictions in this family are
        compressed towards the mean -- the measured calibration slope is
        0.8153-0.8414 -- and the intercept was never measured, so
        `|mu_pred(tef1) - 12.3542|` would mostly measure the compression and
        rank an uncalibrated model above a better-ordered one. Both are
        therefore computed after an affine calibration fitted on the SCORED
        NATIVES ONLY; fitting it on the anchors would fit the anchor error
        away and leave the component measuring nothing.

        `spy_anchor_fit` is 1 - (mean |calibrated prediction - measured mu|
        over pks1 and tef1) / sd(measured mu over the natives), so 1.0 means
        both anchors land where the panel's own scale says they should and
        0.0 means off by a full panel standard deviation.

        `spy_confidence` is the Gaussian log-likelihood of the scored natives
        under the model's own calibrated mean and predicted variance,
        relative to a constant-variance baseline. It rewards a small sigma
        only where the error is genuinely small; mean sigma alone would be
        gameable, since the head can always shrink sigma.
        """
        out: Dict[str, float] = {}
        recs = [r for r in self.panel.by_class(*SCORED_NATIVE_CLASSES)
                if r.name in strength and np.isfinite(r.mu)]
        if len(recs) < 4:
            return out
        p = np.array([strength[r.name] for r in recs], dtype=np.float64)
        y = np.array([r.mu for r in recs], dtype=np.float64)
        sd_y = float(y.std())
        if sd_y <= 1e-9 or float(p.std()) <= 1e-9:
            return out
        # affine calibration on the NATIVES, anchors held out of the fit
        slope = float(np.cov(p, y, bias=True)[0, 1] / max(p.var(), 1e-12))
        inter = float(y.mean() - slope * p.mean())
        out["spy_calib_slope"] = slope

        # A negative or collapsed slope makes both components below
        # meaningless: the affine map that "calibrates" the predictions
        # reverses them, so an anchor error measured through it is a number
        # about a flipped model. Measured once: slope -0.0719 gave
        # spy_confidence -166.28, which clamps to 0 in the score but reads in
        # a log as if it had been measured. An `abs(slope)` test would let
        # that through, so the test is on the SIGNED slope, and below the
        # threshold both components are omitted and reported as not scored.
        if not (slope >= 0.05):
            out["spy_calib_degenerate"] = 1.0
            return out

        # Anchors
        errs = []
        for cls in ("kotwica_pks1", "kotwica_tef1"):
            got = self.panel.by_class(cls)
            if got and got[0].name in strength and np.isfinite(got[0].mu):
                cal = slope * float(strength[got[0].name]) + inter
                e = abs(cal - float(got[0].mu))
                errs.append(e)
                out["anchor_err_%s" % cls.replace("kotwica_", "")] = float(e)
        if errs:
            out["spy_anchor_err"] = float(np.mean(errs))
            out["spy_anchor_fit"] = float(
                max(0.0, 1.0 - np.mean(errs) / sd_y))

        # Confidence
        v = np.array([variance.get(r.name, np.nan) for r in recs],
                     dtype=np.float64)
        if np.isfinite(v).all() and (v > 0).all():
            resid = (slope * p + inter) - y
            # The head's variance is in the PREDICTION's units; calibration
            # rescales the mean by `slope`, so the variance rescales by
            # slope^2. Skipping this would compare a residual in label units
            # against a variance in prediction units.
            vc = v * (slope ** 2)
            nll = 0.5 * np.mean(resid ** 2 / vc + np.log(vc))
            base_v = float(np.mean(resid ** 2))
            nll0 = 0.5 * (1.0 + math.log(max(base_v, 1e-12)))
            out["spy_nll"] = float(nll)
            out["spy_sigma_mean"] = float(np.sqrt(vc).mean())
            # >0 means the per-sample variance beats one constant variance
            # fitted to the same residuals, i.e. the confidence is
            # INFORMATIVE rather than merely small.
            out["spy_confidence"] = float(nll0 - nll)
        return out

    def metrics(self, model) -> Dict[str, float]:
        """The aggregated metrics alone, for older call sites."""
        met, _raw = self.metrics_and_raw(model)
        return met

    def metrics_and_raw(self, model):
        """`(met, raw)`, where `raw` holds the panel's RAW predictions.

        The paired tie bootstrap has to resample the panel and recompute the
        composite for both candidates on the same resample, which aggregated
        metrics cannot support: an aggregate does not decompose back into
        records, and recomputing it would mean a second forward pass, which
        disturbs the EMA.
        """
        strength, realness, variance = self.predict(model)
        met = self.panel.score(strength, realness or None)
        met.update(self.anchor_and_confidence(strength, variance))
        # A constant predictor scores a defensible-looking correlation of NaN
        # and an AUC of exactly 0.5; the spread is what separates "no signal"
        # from "no output".
        vals = np.array(list(strength.values()), dtype=np.float64)
        met["spy_pred_sd"] = float(vals.std()) if vals.size else 0.0
        met["spy_n_predicted"] = float(len(strength))
        raw = {"strength": dict(strength),
               "realness": dict(realness or {}),
               "variance": dict(variance or {})}
        self.last_raw = raw
        return met, raw


# The policy
@dataclass
class SelectionVerdict:
    admitted: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        head = ("ADMITTED score %.5f" % (self.score,) if self.admitted
                else "REFUSED (score would be %.5f)" % (self.score,))
        parts = " ".join("%s %.4f->%.3f" % (k, self.components.get(k, float("nan")),
                                            self.quality.get(k, float("nan")))
                         for k in sorted(self.quality))
        return head + " | " + parts + (
            " | " + "; ".join(self.reasons) if self.reasons else "")


@dataclass
class SelectionPolicy:
    """Hard gates, then a geometric mean over the panel components."""

    components: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: dict(SPY_COMPONENTS))
    recon_admit: float = 0.90
    require_edit_resolution: bool = True
    require_anchor_order: bool = True
    #: Whether both anchors must be CALLED PROMOTERS before an epoch may
    #: take the `_pks` slot. Measured against the median of `losowa_euler4`
    #: rather than an absolute threshold, because an AUC of 0.916 does not
    #: carry over to any single cut point.
    require_anchor_promoters: bool = True
    #: Two checkpoints closer than the panel can resolve are a TIE, and a tie
    #: must not be broken by an argmax that reads like a measurement.
    tie_margin: float = 0.0
    #: Quality floor inside the geometric mean. See the comment in `score`.
    geom_floor: float = 1e-2

    #: Components whose value BELOW the floor still carries order.
    #:
    #: Measured need for this: in one 30-epoch run three of six components
    #: sat on the floor in every epoch (`spy_anchor_fit` at exactly 0,
    #: `spy_confidence` always negative, `motif_auc_small` straddling its
    #: floor). With a geometric floor of 1e-6, 0.40 of the weight sitting
    #: there contributes 0.40*ln(1e-6) = -5.53 to the exponent, so the whole
    #: score was the floor raised to a power. The range over 30 epochs came
    #: to 0.0157 against this rule's own tie threshold of 0.038, so the rule
    #: could choose nothing: 29 ties and one "new best" in epoch 1, while
    #: epoch 11 scored 6.8x higher.
    #:
    #: Below the floor these map MONOTONICALLY instead of clipping, so an
    #: error of 1.1 sd and one of 5.4 sd no longer read the same.
    MIEKKIE: ClassVar[Tuple[str, ...]] = ("spy_anchor_fit", "spy_confidence")

    #: The counts the `motif_auc_small` floor was computed from. If the panel
    #: changes, `check_motif_floor()` reports it rather than quietly using a
    #: threshold derived for another set.
    MOTIF_N: ClassVar[Tuple[int, int]] = (60, 60)

    def check_motif_floor(self, n_hit: int, n_keep: int) -> Dict[str, float]:
        """Whether the `motif_auc_small` floor matches the panel's counts."""
        se = math.sqrt((n_hit + n_keep + 1.0)
                       / max(12.0 * n_hit * n_keep, 1e-9))
        floor = float(self.components["motif_auc_small"][0])
        return {
            "se_h0": se,
            "podloga": floor,
            "podloga_w_se": (floor - 0.5) / max(se, 1e-9),
            "licznosci_zgodne": float((n_hit, n_keep) == self.MOTIF_N),
        }

    def waga_efektywna(self, name: str,
                       met: "Optional[Mapping[str, float]]" = None) -> float:
        """A component's weight as `score()` sees it: AFTER normalisation.

        `score()` divides each weight by the sum of the weights actually
        scored in that epoch, and only that number is an exponent of the
        geometric mean. With the full set (0.34 + 0.20 + 0.16 + 0.00 + 0.15
        = 0.85) the effective exponent of `spy_scc` is 0.4000, not 0.34, so
        a tie threshold built from the raw weight is 17.6 % too narrow.
        """
        if name not in self.components:
            return 0.0
        if met is None:
            total = sum(float(w) for _f, _c, w in self.components.values())
        else:
            total = sum(float(w) for n, (_f, _c, w) in self.components.items()
                        if n in met and np.isfinite(float(met[n])))
        if total <= 0.0:
            return 0.0
        return float(self.components[name][2]) / total

    def quality(self, name: str, value: float) -> float:
        floor, ceil, _w = self.components[name]
        if not np.isfinite(value):
            return 0.0
        q = (value - floor) / max(1e-9, ceil - floor)
        if name not in self.MIEKKIE:
            return float(max(0.0, min(1.0, q)))
        # Monotone, and continuous at zero. An earlier version used a sigmoid
        # for q < 0 and clipping above, which was NOT monotone: a value of
        # -0.50 scored 0.3775 while 0.00 scored 0.0000, so the worse value
        # scored higher. Both branches now meet at `geom_floor` -- below it
        # an exponential decay that keeps the order, above it linear to 1 --
        # and `check_quality_monotone` measures the continuity.
        if q < 0.0:
            # The exponent is bounded at -4: four normalised units below
            # chance, further degradation carries no information about which
            # epoch to keep, and without the bound one component could
            # compress the composite arbitrarily. The floor is therefore
            # geom_floor * e^-4 = 1.83e-4.
            return float(self.geom_floor * math.exp(max(q, -4.0)))
        return float(self.geom_floor
                     + (1.0 - self.geom_floor) * min(q, 1.0))

    def check_quality_monotone(self, name: str, n: int = 400
                               ) -> Dict[str, float]:
        """Falsifier for the map: monotone, and continuous at zero."""
        floor, ceil, _w = self.components[name]
        span = max(1e-9, ceil - floor)
        xs = [floor + span * (-3.0 + 4.0 * i / (n - 1)) for i in range(n)]
        qs = [self.quality(name, x) for x in xs]
        spadki = sum(1 for a, b in zip(qs, qs[1:]) if b < a - 1e-12)
        return {
            "spadkow": float(spadki),
            "skok_w_zerze": abs(self.quality(name, floor - 1e-9)
                                - self.quality(name, floor + 1e-9)),
            "w_zerze": self.quality(name, floor),
            "min": min(qs), "max": max(qs),
        }

    def gates(self, met: Mapping[str, float]) -> List[str]:
        """Reasons to refuse. Empty means admitted."""
        bad: List[str] = []
        rec = float(met.get("rec_full", float("nan")))
        if not np.isfinite(rec):
            bad.append("rec_full is not finite")
        elif rec < self.recon_admit:
            span = ((rec - RECONSTRUCTION_FLOOR)
                    / (1.0 - RECONSTRUCTION_FLOOR))
            bad.append("rec_full %.4f < %.4f (%.1f %% of the range above the "
                       "%.4f floor); the design route is decode(codes), so a "
                       "model that cannot reconstruct cannot be inverted"
                       % (rec, self.recon_admit, 100.0 * span,
                          RECONSTRUCTION_FLOOR))
        if float(met.get("spy_pred_sd", 0.0)) <= 1e-6:
            bad.append("the strength prediction is CONSTANT across the panel "
                       "(sd %.3e): an untrained model in eval measured 3.19e-08 "
                       "and a constant predictor scores a defensible-looking "
                       "loss" % (float(met.get("spy_pred_sd", 0.0)),))
        if self.require_edit_resolution:
            if float(met.get("edit_moves_prediction", 0.0)) <= 0.0:
                bad.append("a 2-base edit does not move the prediction at "
                           "all: the latent has no resolution below the block "
                           "level and every one of the ten designs IS an edit")
            elif "edit_resolution_ok" in met and \
                    float(met["edit_resolution_ok"]) <= 0.0:
                bad.append("88 substitutions move the prediction LESS than 2 "
                           "do (ratio %.3f): the head is responding to noise, "
                           "so the in-motif comparison means nothing"
                           % (float(met.get("edit_resolution", float("nan"))),))
        if self.require_anchor_promoters and "anchor_promoters_ok" in met:
            if float(met["anchor_promoters_ok"]) <= 0.0:
                bad.append(
                    "the anchor is NOT recognised as a promoter against its "
                    "own composition control (`losowa_euler4`, median "
                    "%.4f): pks1 %.4f (margin %+.4f), tef1 %.4f "
                    "(margin %+.4f). The ten designed promoters land in "
                    "the pks1 locus, and a model that does not "
                    "recognise pks1 as a promoter has nothing to ground "
                    "either realness or strength on"
                    % (float(met.get("anchor_real_neg_med", float("nan"))),
                       float(met.get("anchor_real_pks1", float("nan"))),
                       float(met.get("anchor_real_margin_pks1", float("nan"))),
                       float(met.get("anchor_real_tef1", float("nan"))),
                       float(met.get("anchor_real_margin_tef1",
                                     float("nan")))))
        if self.require_anchor_order and "anchor_order_ok" in met:
            if float(met["anchor_order_ok"]) <= 0.0:
                bad.append("tef1 scores BELOW pks1 (gap %.4f); four of the ten "
                           "wet slots ask for a promoter between them and a "
                           "model that inverts the pair cannot place one"
                           % (float(met.get("anchor_gap", float("nan"))),))
        return bad

    def score(self, met: Mapping[str, float]) -> Tuple[float, Dict[str, float],
                                                       Dict[str, float]]:
        comps: Dict[str, float] = {}
        qual: Dict[str, float] = {}
        total_w = 0.0
        for name, (_f, _c, w) in self.components.items():
            if name not in met or not np.isfinite(float(met[name])):
                continue
            comps[name] = float(met[name])
            qual[name] = self.quality(name, float(met[name]))
            total_w += float(w)
        if not qual or total_w <= 0.0:
            return 0.0, comps, qual
        acc = 0.0
        for name, q in qual.items():
            w = float(self.components[name][2]) / total_w
            # Log-space geometric mean, with a floor so one zero does not
            # make the whole score -inf and therefore unorderable between two
            # equally-failed checkpoints. The floor is 1e-2 rather than 1e-6:
            # at 1e-6 one dead component of weight 0.15 contributes -2.07 to
            # the exponent, damping the score 7.9x and three of them 330x, at
            # which point the score is the floor raised to a power rather
            # than a mean. At 1e-2 the same dead component costs a factor of
            # 2.0 -- the penalty stays, the annihilation goes.
            #
            # Only for the HARD components. `MIEKKIE` ones already carry
            # their own lower floor from `quality()`, which keeps the order
            # below zero; clipping them a second time here cancelled exactly
            # what that map exists for, and was measured as an identical
            # composite (0.115471) for four different values.
            podloga = 0.0 if name in self.MIEKKIE else self.geom_floor
            acc += w * math.log(max(q, podloga, 1e-12))
        return float(math.exp(acc)), comps, qual

    def check_soft_survives_score(self) -> Dict[str, float]:
        """Whether the order of the MIEKKIE components survives `score()`.

        Measured through `score()` and not through `quality()`: testing
        `quality()` in isolation passes while `score()` clips its result back
        onto the floor, leaving the composite identical for four different
        values.

        S1  the composite must RISE with a soft component below zero;
        S2  control: a HARD component below its floor must leave the
            composite constant, because the clipping there is intended;
        S3  the exponent bound works: -10 and -4 give the same answer.
        """
        baza = {"spy_scc": 0.40, "motif_auc_small": 0.47, "spy_auc": 0.88,
                "rec_full": 0.997, "spy_confidence": 0.0}

        def s_przy(name, v):
            met = dict(baza)
            met[name] = v
            return self.score(met)[0]

        miekkie = [s_przy("spy_anchor_fit", v)
                   for v in (-3.0, -1.5, -0.5, 0.0)]
        twarde = [s_przy("motif_auc_small", v) for v in (0.20, 0.35, 0.45)]
        rosnie = all(b > a + 1e-12 for a, b in zip(miekkie, miekkie[1:]))
        return {
            "s1_miekka_rosnie": float(rosnie),
            "s1_rozstep": float(max(miekkie) - min(miekkie)),
            "s2_twarda_stala": float(max(twarde) - min(twarde) < 1e-12),
            "s3_wykladnik_ograniczony": float(
                abs(s_przy("spy_anchor_fit", -10.0)
                    - s_przy("spy_anchor_fit", -4.0)) < 1e-12),
        }

    def judge(self, met: Mapping[str, float]) -> SelectionVerdict:
        bad = self.gates(met)
        s, comps, qual = self.score(met)
        return SelectionVerdict(admitted=not bad, score=s, reasons=bad,
                                components=comps, quality=qual)

    def is_tie(self, a: float, b: float, n_scored: int,
               q_scc: float = 0.0) -> bool:
        """True when two scores are inside what the panel can resolve.

        Compared IN LOG SPACE, because the composite is a weighted geometric
        mean. Comparing the absolute difference against `SE * weight` is
        correct only when the composite is of order 1: at a quality of ~0.1
        every difference shrinks proportionally while the threshold does not,
        and a sweep of `spy_scc` over 1.6 panel SE moved the composite by
        0.0147 against a threshold of 0.0375, declaring a real change a tie.

        `q_scc` is the favourite's quality on `spy_scc`, without which the
        panel SE cannot be converted to log units, since `ln(1 + SE/q)`
        depends on the working point. At `q_scc = 0` it falls back to
        `q = SE`, the widest threshold this formula gives, so a missing
        argument cannot quietly narrow it.

        The test is scale-invariant, and `check_tie_scale()` measures that.
        """
        se = self.tie_margin or spearman_se(int(n_scored))
        w = self.waga_efektywna("spy_scc")
        if not math.isfinite(se):
            return True
        if a <= 0.0 or b <= 0.0:
            return abs(a - b) < se * w
        q = float(q_scc) if float(q_scc) > 0.0 else se
        prog = w * math.log1p(se / max(q, 1e-9))
        return abs(math.log(a) - math.log(b)) < prog

    #: The bootstrap's unit of independence: the orthogroup, not the record.
    #: The 65 scored natives are 65 orthogroups, but the panel's mutants all
    #: come from one parent (pks1) and are therefore one unit, not sixty.
    #: Resampling records would inflate n and narrow the threshold.
    def _jednostki(self, panel):
        grupy = {}
        for i, r in enumerate(panel.records):
            klucz = getattr(r, "og_id", "") or ("klasa:" + r.klasa)
            grupy.setdefault(klucz, []).append(i)
        return grupy

    def _kompozyt_na_probie(self, panel, recs, raw, met_pelny):
        """The composite on a SUB-PANEL, with the rest of `met` carried over.

        Components the sub-panel does not reproduce (`rec_full` is a property
        of the model, not of the panel) keep their full-panel value. If they
        dropped out, `score()` would renormalise the weights and every
        resample would measure a different composite.
        """
        from ..data.spy_panel import SpyPanel
        sub = SpyPanel(records=recs, source=getattr(panel, "source", ""))
        met = dict(met_pelny)
        met.update(sub.score(raw["strength"], raw["realness"] or None))
        return float(self.score(met)[0])

    def bootstrap_tie(self, panel, raw_a, raw_b, met_a, met_b,
                      n_boot: int = 1000, seed: int = 7898,
                      z: float = 1.0) -> Dict[str, float]:
        """Tie test by PAIRED BOOTSTRAP, from the raw predictions.

        Not `spearman_se(n) = 1/sqrt(n-1)`: that is the SE of a Spearman
        under H0 for ONE sample, whereas the measured quantity is the
        DIFFERENCE between two models scored on the SAME panel. A paired
        difference is less variable, because "which sequences are in the
        panel" cancels: measured, a paired SE of 0.0100 against an unpaired
        0.0264, so the formula came out 33 % too wide and declared ties on
        differences the panel resolves.

        Returns the observables; `remis` is the verdict.
        """
        grupy = self._jednostki(panel)
        klucze = sorted(grupy)
        rng = np.random.default_rng(int(seed))
        s_a = float(self.score(met_a)[0])
        s_b = float(self.score(met_b)[0])
        eps = 1e-12
        delta = math.log(max(s_a, eps)) - math.log(max(s_b, eps))
        da, la, lb = [], [], []
        for _ in range(int(n_boot)):
            wyb = rng.integers(0, len(klucze), size=len(klucze))
            recs = []
            for j in wyb:
                for i in grupy[klucze[int(j)]]:
                    recs.append(panel.records[i])
            ka = self._kompozyt_na_probie(panel, recs, raw_a, met_a)
            kb = self._kompozyt_na_probie(panel, recs, raw_b, met_b)
            xa = math.log(max(ka, eps))
            xb = math.log(max(kb, eps))
            la.append(xa)
            lb.append(xb)
            da.append(xa - xb)
        arr = np.asarray(da, dtype=np.float64)
        se_par = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        se_niep = float(np.sqrt(np.var(la, ddof=1) + np.var(lb, ddof=1))) \
            if arr.size > 1 else float("nan")
        prog = float(z) * se_par
        return {
            "delta_log": float(delta),
            "se_parowy": se_par,
            "se_nieparowy": se_niep,
            # At a paired SE of exactly zero (an arithmetic identity) the
            # ratio is infinite, not 2e+10: a number that looks like a
            # measurement is worse than an explicit infinity.
            "zysk_parowania": (float("inf") if se_par <= 0.0
                               else float(se_niep / se_par)),
            "prog": prog,
            "prog_wzorem": self.waga_efektywna("spy_scc")
            * math.log1p(spearman_se(int(met_a.get("spy_n", 0) or 65))
                         / max(self.quality(
                             "spy_scc", float(met_a.get("spy_scc", 0.0))),
                             1e-9)),
            "n_jednostek": float(len(klucze)),
            "n_boot": float(n_boot),
            "jednostka": "ortogrupa",
            # The inequality is NOT strict. Caught by this test's own
            # positive control: one model against itself gives delta 0.0 and
            # a paired SE of 0.0, and `0.0 < 0.0` is false, so the test
            # declared a DIFFERENCE between two identical candidates.
            "remis": float(abs(delta) <= prog),
        }

    def check_tie_bootstrap(self, panel, raw, met, n_boot: int = 200
                            ) -> Dict[str, float]:
        """Falsifiers for the bootstrap.

        B1 positive control (an arithmetic identity): one model against
           itself must give delta exactly 0 and the verdict TIE.
        B2 the paired SE must be smaller than the unpaired one, or the
           pairing was not performed.
        B3 the unpaired SE must be positive: an exact 0 means the resampling
           does not move the panel, i.e. a broken intervention rather than
           an absence of noise.
        """
        w = self.bootstrap_tie(panel, raw, raw, met, met, n_boot=n_boot)
        return {
            "b1_tozsamosc_delta": abs(w["delta_log"]),
            "b1_tozsamosc_remis": w["remis"],
            "b2_parowanie_zwezylo": float(w["se_parowy"] < w["se_nieparowy"]),
            "b3_se_niezerowe": float(w["se_nieparowy"] > 0.0),
            "zysk_parowania": w["zysk_parowania"],
        }

    def check_tie_scale(self, n_scored: int = 65,
                        q_scc: float = 0.42) -> Dict[str, float]:
        """Falsifiers for the tie test.

        T1  scale invariance: multiplying both scores by 100 must not change
            the verdict. The absolute-difference test failed this by
            construction.
        T2  positive control: a difference worth 2 SE of `spy_scc` must not
            be a tie.
        T3  negative control: a difference worth 0.2 SE must be a tie, or the
            threshold does not protect against noise.
        """
        se = self.tie_margin or spearman_se(int(n_scored))
        w = self.waga_efektywna("spy_scc")
        baza = 0.10

        def after_change(k):
            return baza * math.exp(w * math.log1p(k * se / q_scc))

        t1_a = self.is_tie(baza, after_change(2.0), n_scored, q_scc)
        t1_b = self.is_tie(100 * baza, 100 * after_change(2.0), n_scored, q_scc)
        return {
            "t1_skala_zgodna": float(t1_a == t1_b),
            "t2_dwa_se_nie_remis": float(
                not self.is_tie(baza, after_change(2.0), n_scored, q_scc)),
            "t3_male_jest_remisem": float(
                self.is_tie(baza, after_change(0.2), n_scored, q_scc)),
            "prog_log": w * math.log1p(se / q_scc),
            "se_panelu": se,
        }
