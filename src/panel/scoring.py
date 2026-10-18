"""Scoring. The one route from a candidate to a number, for all five strategies.

Three rules this module enforces rather than documents.

A candidate is scored as DNA, never as codes. `encode(decode(idx)) == idx` is
refuted on this model (per-level round-trip agreement 0.3025, 0.4612, 0.7975,
0.9884, 0.9959), so a code does not identify a molecule; `score_codes()`
decodes first and scores the same object the wet laboratory will order.

Both constructs, combined by the minimum. pAN7-1 and pHYPPE differ in the
flank channel and the model is measurably sensitive to it, so a promoter that
must work in both is limited by its worse construct, and a sum or a mean would
let one construct pay for the other. `combine=` makes the choice visible and
records it.

The experiment description has no default. Inventing a sentence here would
condition a reported number on text nobody in the wet laboratory wrote.
Omitted, the model is scored at the learned unknown text token -- a trained
state, not a zero vector -- and every record says so through
`text_is_unknown`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import (Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import anchors as K  # noqa: E402
import torch  # noqa: E402
from model_io import ROOT, SRC, Provenance  # noqa: E402

from hyppe.data.labels import SamplePolicy  # noqa: E402
from hyppe.model.species import StrainRegistry  # noqa: E402

BASES = "ACGT"
CATEGORIES: Tuple[str, ...] = tuple(SamplePolicy().categories)
CONSTRUCTS: Tuple[str, ...] = ("pHYPPE", "pAN7-1")
DESIGN_FLANKS = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "design_flanks.npz")
CONTEXT_TOLERANCE = 1e-4


def _load_design_panel():
    """Import `src/scripts/design_panel.py` by path and reuse its helpers.

    By path, not by package name: `src/scripts` is not a package and putting
    it on `sys.path` would place a directory of single-purpose scripts ahead
    of everything else; its `__main__` guard means importing runs no design.
    The three helpers encode the contract between the query side and the
    cached rows, and a second spelling of that contract is how a query lands
    in a different space from the training data and still returns a plausible
    cosine.
    """
    path = os.path.join(SRC, "scripts", "design_panel.py")
    if not os.path.exists(path):
        raise SystemExit(
            "REFUSED: %s is missing; its helpers are the contract for the "
            "condition channel.\nThis repository carries the MODEL, not the "
            "design pipeline, so scoring a candidate in the two reporter "
            "constructs needs design_panel.py from the companion design "
            "repository. Training, diagnostics of the ladder itself and the "
            "unit tests do not." % (path,))
    spec = importlib.util.spec_from_file_location("_hyppe_design_panel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("build_design_text", "build_design_flanks", "design_context"):
        if not hasattr(mod, name):
            raise SystemExit("REFUSED: design_panel.py has no %r; this module "
                             "reuses it rather than re-deriving it" % (name,))
    return mod


class _LazyDesignPanel:
    """Deferred import of the three construct-context helpers.

    Eager loading made this module unimportable in a checkout that carries the
    model and not the design pipeline, which is what this repository is. The
    helpers are reached only when a candidate is scored in a reporter
    construct, so they must not be a load-time dependency.
    """

    _mod = None

    def __getattr__(self, name):
        if _LazyDesignPanel._mod is None:
            _LazyDesignPanel._mod = _load_design_panel()
        return getattr(_LazyDesignPanel._mod, name)


DP = _LazyDesignPanel()


def one_hot(seqs: Sequence[str]) -> torch.Tensor:
    """[N, 4, L] in ACGT order. Rows with a non-ACGT base are refused.

    Not mapped to 0.25 in four channels: the argmax of such a row is a base
    that is not in the sequence, so an N would travel through the whole
    pipeline as a real nucleotide.
    """
    if not seqs:
        raise ValueError("no sequences")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError("sequences of different lengths: %s"
                         % (sorted(lengths),))
    code = {c: i for i, c in enumerate(BASES)}
    n, L = len(seqs), len(seqs[0])
    x = torch.zeros(n, 4, L)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s.upper()):
            k = code.get(ch)
            if k is None:
                raise ValueError(
                    "sequence %d has %r at position %d; only ACGT is "
                    "accepted here" % (i, ch, j))
            x[i, k, j] = 1.0
    return x


def codes_to_dna(model, indices: Sequence[torch.Tensor]) -> List[str]:
    """Decode ladder indices to DNA: the gate between codes and reality."""
    with torch.no_grad():
        logits = model.decode_codes([t for t in indices])
    arr = logits.argmax(dim=1).cpu().numpy()
    return ["".join(BASES[int(i)] for i in row) for row in arr]


@dataclass
class Context:
    """One fully resolved conditioning state, plus what it is made of."""
    key: str                                   # "pHYPPE" | "pAN7-1" | "none"
    species: str
    species_row: int
    text_is_unknown: bool
    text_note: str
    flank_note: str
    vector: torch.Tensor                       # [1, ctx_dim]
    text_emb: Optional[torch.Tensor]
    text_ok: Optional[torch.Tensor]
    flank_l: Optional[torch.Tensor]
    flank_r: Optional[torch.Tensor]
    flank_ok: Optional[torch.Tensor]

    def note(self) -> str:
        head = ("context %r | species %s (row %d)" %
                (self.key, self.species, self.species_row))
        warn = ("\n  !! TEXT = UNKNOWN TOKEN: no experiment description was "
                "given, so this score is NOT conditioned on the competition "
                "experiment. The description must come from the WetLab team; "
                "none is invented here." if self.text_is_unknown else "")
        return "%s\n  %s\n  %s%s" % (head, self.text_note, self.flank_note,
                                     warn)


@dataclass
class ConstructScore:
    """Everything the heads say about one sequence in one construct."""
    construct: str
    mu: np.ndarray                       # [7] one per category column
    log_var: np.ndarray                  # [7] or [1]
    sigma: np.ndarray                    # [7] or [1] = exp(0.5 * log_var)
    log_sigma_obs: Optional[np.ndarray]  # measured-sigma head, or None
    realness: float                      # discriminator logit
    pair_score: float                    # s(z, C); an ORDER, not a level
    pair_vs: Dict[str, float] = field(default_factory=dict)   # delta per anchor
    pair_prob_vs: Dict[str, float] = field(default_factory=dict)

    def mu_of(self, category: str) -> float:
        return float(self.mu[CATEGORIES.index(category)])


@dataclass
class CandidateScore:
    """One candidate, scored in every construct, with the combined objective."""
    dna: str
    category: str
    category_index: int
    per_construct: Dict[str, ConstructScore]
    combined: float
    combiner: str
    text_is_unknown: bool
    text_note: str
    provenance: Dict[str, object] = field(default_factory=dict)
    codes_source: str = "dna"            # "dna" or "decoded from codes"

    def limiting_construct(self) -> str:
        """The construct the combined objective is limited by."""
        return min(self.per_construct,
                   key=lambda c: self.per_construct[c].mu_of(self.category))

    def table(self) -> str:
        rows = ["candidate (%d bp)  category %r (column %d)  combined %s = "
                "%+.4f" % (len(self.dna), self.category, self.category_index,
                           self.combiner, self.combined)]
        if self.text_is_unknown:
            rows.append("  !! scored at the UNKNOWN description token -- "
                        "not conditioned on the WetLab experiment")
        for c in CONSTRUCTS:
            s = self.per_construct.get(c)
            if s is None:
                continue
            rows.append("  %-8s mu[%s] %+.4f | sigma %.4f | realness %+.4f "
                        "| pair %+.4f" %
                        (c, self.category, s.mu_of(self.category),
                         float(np.mean(s.sigma)), s.realness, s.pair_score))
            rows.append("           mu all: " +
                        " ".join("%s=%+.3f" % (n[:4], v)
                                 for n, v in zip(CATEGORIES, s.mu)))
            if s.pair_vs:
                rows.append("           pairwise delta: " +
                            " ".join("vs %s %+.4f (P=%.3f)"
                                     % (k, v, s.pair_prob_vs.get(k, float("nan")))
                                     for k, v in sorted(s.pair_vs.items())))
        rows.append("  limited by: %s" % (self.limiting_construct(),))
        return "\n".join(rows)


def combine_min(values: Sequence[float]) -> float:
    """The default combiner. A design must survive its worse construct."""
    return float(min(values))


def combine_mean(values: Sequence[float]) -> float:
    """Provided so that choosing it is an explicit act, and recorded as one."""
    return float(np.mean(values))


class Scorer(object):
    """Model + resolved contexts + anchors. The only place a number is made.

    `model` and `provenance` come from `model_io.load_model`.
    `experiment_text` is a required keyword with no default that means
    something: pass the wet laboratory's own description of the experiment, or
    pass `""` to state that there is none yet and be scored at the unknown
    token. There is no third option and no invented sentence.
    """

    def __init__(self, model, provenance: Provenance,
                 experiment_text: Union[str, Mapping[str, str], None] = None,
                 species: str = "P1",
                 constructs: Sequence[str] = CONSTRUCTS,
                 design_flanks: str = DESIGN_FLANKS,
                 anchors: Optional[K.AnchorSequences] = None) -> None:
        if experiment_text is None:
            raise TypeError(
                "Scorer(experiment_text=...) has NO DEFAULT. Pass the "
                "WetLab's description of the experiment, or pass '' to state "
                "that there is none and accept scoring at the UNKNOWN text "
                "token. A default sentence here would condition every number "
                "in the panel on text nobody wrote.")
        if isinstance(experiment_text, Mapping):
            missing = [c for c in constructs if c not in experiment_text]
            if missing:
                raise KeyError(
                    "experiment_text is a mapping but has no entry for %s; "
                    "constructs=%s. Every construct needs its own description "
                    "or an explicit '' -- a missing key must not fall back to "
                    "another construct's sentence."
                    % (", ".join(map(repr, missing)), list(constructs)))
            _teksty = {c: experiment_text[c] for c in constructs}
        elif isinstance(experiment_text, str):
            _teksty = {c: experiment_text for c in constructs}
        else:
            raise TypeError(
                "experiment_text must be a str or a {construct: str} mapping, "
                "got %r" % (type(experiment_text).__name__,))
        for _c, _t in _teksty.items():
            if not isinstance(_t, str):
                raise TypeError("experiment_text[%r] must be a str, got %r"
                                % (_c, type(_t).__name__))
        self.model = model
        self.prov = provenance
        self.device = next(model.parameters()).device
        self.species = species
        self.species_row = StrainRegistry.default().names.index(species)
        self.design_flanks = design_flanks
        self.anchor_seqs = anchors or K.load_anchor_sequences()

        self._tekst_wg_klucza = {}
        for _c, _t in _teksty.items():
            _e, _o, _n = DP.build_design_text(model, _t)
            self._tekst_wg_klucza[_c] = (
                None if _e is None else _e.to(self.device),
                None if _o is None else _o.to(self.device),
                _n, _e is None)
        self.experiment_text = experiment_text
        self.experiment_texts = dict(_teksty)
        _pierwszy = _teksty[list(constructs)[0]] if list(constructs) else ""
        _jednorodny = len(set(_teksty.values())) <= 1
        _e0, _o0, _n0, _u0 = self._tekst_wg_klucza[list(constructs)[0]]
        self.text_emb = _e0 if _jednorodny else None
        self.text_ok = _o0 if _jednorodny else None
        self.text_note = _n0 if _jednorodny else " | ".join(
            "%s: %s" % (c, self._tekst_wg_klucza[c][2]) for c in constructs)
        self.text_is_unknown = any(self._tekst_wg_klucza[c][3]
                                   for c in constructs)
        self.text_per_construct = not _jednorodny

        self.contexts: Dict[str, Context] = {}
        for c in constructs:
            self.contexts[c] = self._build_context(c)

        self._anchor_cache: Dict[str, Dict[str, ConstructScore]] = {}
        self.last_context_error = 0.0

    def _build_context(self, key: str) -> Context:
        fl, fr, fok, fnote = DP.build_design_flanks(
            self.model, key, self.design_flanks)
        move = lambda t: None if t is None else t.to(self.device)
        fl, fr, fok = move(fl), move(fr), move(fok)
        t_emb, t_ok, t_note, t_unk = self._tekst_wg_klucza[key]
        vec = DP.design_context(self.model, self.species_row,
                                t_emb, t_ok,
                                flank_l=fl, flank_r=fr, flank_ok=fok)
        return Context(key=key, species=self.species,
                       species_row=self.species_row,
                       text_is_unknown=t_unk,
                       text_note=t_note, flank_note=fnote,
                       vector=vec.detach(), text_emb=t_emb,
                       text_ok=t_ok, flank_l=fl, flank_r=fr,
                       flank_ok=fok)

    def context_note(self, key: str) -> str:
        return self.contexts[key].note()

    @staticmethod
    def category_index(name: str) -> int:
        if name not in CATEGORIES:
            raise KeyError("category %r is not one of %s"
                           % (name, list(CATEGORIES)))
        return CATEGORIES.index(name)

    def context_delta(self) -> Dict[str, float]:
        """max |ctx_a - ctx_b| for every pair of constructs, plus the null.

        The null is a construct against itself, computed through the SAME
        subtraction as every treatment rather than through an equality test.
        `torch.equal(v, v)` was a tautology for any finite vector, so the
        control could not fail; going through the arithmetic means a vector
        carrying NaN or inf surfaces as NaN here instead of as a clean zero.

        What it proves is therefore narrow and worth stating: the context
        vectors are finite and the difference path is sound. It does NOT
        prove the contexts were built independently -- the same object is
        read twice -- so a treatment of 0.0 still has to be read as "these
        two constructs carry the same context", not as a broken probe.
        """
        out: Dict[str, float] = {}
        keys = list(self.contexts)
        for i, a in enumerate(keys):
            v = self.contexts[a].vector
            out["NULL %s vs %s" % (a, a)] = float((v - v).abs().max())
            for b in keys[i + 1:]:
                out["%s vs %s" % (a, b)] = float(
                    (self.contexts[a].vector - self.contexts[b].vector)
                    .abs().max())
        return out

    @torch.no_grad()
    def _forward(self, seqs: Sequence[str], key: str,
                 batch: int = 16) -> Dict[str, np.ndarray]:
        """Run the model on the DNA, in one construct: all heads, no losses."""
        ctx = self.contexts[key]
        mu_l, lv_l, ls_l, re_l, ps_l = [], [], [], [], []
        ctx_err = 0.0
        for i in range(0, len(seqs), batch):
            x = one_hot(list(seqs[i:i + batch])).to(self.device)
            B = int(x.shape[0])
            rep = lambda t: None if t is None else t.expand(B, *t.shape[1:])
            out = self.model(
                x,
                text_emb=rep(ctx.text_emb), text_ok=rep(ctx.text_ok),
                species_id=torch.full((B,), self.species_row,
                                      dtype=torch.long, device=self.device),
                flank_l=rep(ctx.flank_l), flank_r=rep(ctx.flank_r),
                flank_ok=rep(ctx.flank_ok),
                reconstruct=False)
            ctx_err = max(ctx_err, float(
                (out["context"] - ctx.vector.expand_as(out["context"]))
                .abs().max()))
            if B == 1 and ctx_err != 0.0:
                raise SystemExit(
                    "REFUSED: at batch 1 the context built inside forward "
                    "must be BIT-IDENTICAL to the one design_panel built "
                    "(measured 0.000000e+00 on HYPPE_ep003); it differs by "
                    "%.3e. That is a changed expression, not float noise."
                    % (ctx_err,))
            ab = out["absolute"]
            mu_l.append(np.atleast_2d(ab.mu.detach().cpu().numpy()))
            lv_l.append(np.atleast_2d(ab.log_var.detach().cpu().numpy()))
            ls_l.append(None if ab.log_sigma_obs is None else
                        np.atleast_2d(ab.log_sigma_obs.detach().cpu().numpy()))
            d = out.get("discriminator")
            re_l.append(np.full(B, np.nan) if d is None
                        else d.logit.detach().cpu().numpy().reshape(-1))
            p = out.get("pair_score")
            ps_l.append(np.full(B, np.nan) if p is None
                        else p.detach().cpu().numpy().reshape(-1))
        if ctx_err > CONTEXT_TOLERANCE:
            raise SystemExit(
                "REFUSED: the context built inside forward differs from the "
                "one design_panel.design_context built by %.3e, above the "
                "measured batch-size float32 noise (%.1e). In eval there is "
                "no flank dropout, so the two are the same expression; a "
                "difference this size means anchors and candidates are being "
                "scored under two different conditions."
                % (ctx_err, CONTEXT_TOLERANCE))
        self.last_context_error = ctx_err
        res = {"mu": np.concatenate(mu_l, axis=0),
               "log_var": np.concatenate(lv_l, axis=0),
               "realness": np.concatenate(re_l),
               "pair_score": np.concatenate(ps_l)}
        if all(v is not None for v in ls_l):
            res["log_sigma_obs"] = np.concatenate(ls_l, axis=0)
        return res

    def raw(self, seqs: Sequence[str], key: str,
            batch: int = 16) -> Dict[str, np.ndarray]:
        """Head outputs for DNA in one construct, straight from `_forward`.

        mu, log_var, realness, pair_score, and log_sigma_obs where the head
        has one. Public because `panel_gates.realness_reference` needs the
        discriminator on a few hundred real promoters, and reaching into a
        private method for it is how a second scoring path gets written.
        """
        return self._forward(list(seqs), key, batch=batch)

    def mu_columns(self, seqs: Sequence[str], key: str) -> np.ndarray:
        """[N, 7]. The duck-typed method `anchors.measure_anchors` calls."""
        return self._forward(list(seqs), key)["mu"]

    def anchor_scores(self, key: str) -> Dict[str, ConstructScore]:
        """pks1 and tef1 scored in this construct, cached per construct."""
        if key not in self._anchor_cache:
            seqs = [self.anchor_seqs.pks1, self.anchor_seqs.tef1]
            raw = self._forward(seqs, key)
            self._anchor_cache[key] = {
                name: self._row(raw, i, key)
                for i, name in enumerate(("pks1", "tef1"))}
        return self._anchor_cache[key]

    def anchors(self, category: str, key: str) -> K.Anchors:
        """`anchors.Anchors` recomputed in the construct the design uses."""
        return K.measure_anchors(self, key, category,
                                 sequences=self.anchor_seqs)

    def _row(self, raw: Dict[str, np.ndarray], i: int,
             key: str) -> ConstructScore:
        lv = np.asarray(raw["log_var"][i], dtype=np.float64).reshape(-1)
        ls = raw.get("log_sigma_obs")
        return ConstructScore(
            construct=key,
            mu=np.asarray(raw["mu"][i], dtype=np.float64).reshape(-1),
            log_var=lv, sigma=np.exp(0.5 * lv),
            log_sigma_obs=(None if ls is None else
                           np.asarray(ls[i], dtype=np.float64).reshape(-1)),
            realness=float(raw["realness"][i]),
            pair_score=float(raw["pair_score"][i]))

    def score_dna(self, seqs: Sequence[str], category: str,
                  combine: Callable[[Sequence[float]], float] = combine_min,
                  codes_source: str = "dna") -> List[CandidateScore]:
        """Score DNA in every construct and combine. The entry point."""
        seqs = [s.upper() for s in seqs]
        k = self.category_index(category)
        per_key = {key: self._forward(seqs, key) for key in self.contexts}
        temp = float(getattr(getattr(self.model, "pairwise", None),
                             "temperature", 1.0) or 1.0)
        out: List[CandidateScore] = []
        for i, s in enumerate(seqs):
            rows: Dict[str, ConstructScore] = {}
            for key, raw in per_key.items():
                row = self._row(raw, i, key)
                anc = self.anchor_scores(key)
                for name, a in anc.items():
                    d = row.pair_score - a.pair_score
                    row.pair_vs[name] = float(d)
                    row.pair_prob_vs[name] = float(
                        1.0 / (1.0 + np.exp(-d / temp)))
                rows[key] = row
            vals = [rows[key].mu[k] for key in sorted(rows)]
            out.append(CandidateScore(
                dna=s, category=category, category_index=k,
                per_construct=rows,
                combined=float(combine(vals)),
                combiner=getattr(combine, "__name__", str(combine)),
                text_is_unknown=any(
                    self._tekst_wg_klucza[_k][3] for _k in rows),
                text_note=" | ".join(
                    "%s: %s" % (_k, self._tekst_wg_klucza[_k][2])
                    for _k in sorted(rows)),
                codes_source=codes_source,
                provenance=dict(self.prov.as_dict(),
                                constructs=sorted(rows),
                                species=self.species)))
        return out

    def score_codes(self, indices: Sequence[torch.Tensor], category: str,
                    combine: Callable[[Sequence[float]], float] = combine_min
                    ) -> List[CandidateScore]:
        """Decode first, then score the DNA: the only sanctioned code route.

        The decoded string is carried in the record, so what was scored and
        what would be ordered from the synthesis house are the same object.
        """
        dna = codes_to_dna(self.model, indices)
        return self.score_dna(dna, category, combine=combine,
                              codes_source="decoded from codes")

    @torch.no_grad()
    def roundtrip_report(self, seqs: Sequence[str]) -> Dict[str, object]:
        """Per-level agreement of `encode(decode(encode(x)))` with `encode(x)`.

        The measurement that forbids scoring in code space, kept next to the
        scorer so a notebook can print it rather than cite it. Not a gate a
        candidate passes: real P1 promoters give [0.236, 0.462, 0.814, 0.992,
        0.995] too, which is a property of the encoder at the coarse levels
        and not of the candidate.
        """
        x = one_hot(list(seqs)).to(self.device)
        idx = self.model.encode_codes(x)
        dna = codes_to_dna(self.model, idx)
        idx2 = self.model.encode_codes(one_hot(dna).to(self.device))
        agree = [float((a == b).float().mean()) for a, b in zip(idx, idx2)]
        return {"per_level_agreement": agree,
                "n": len(seqs),
                "exact_levels": int(sum(1 for a in agree if a == 1.0)),
                "note": "a code does not identify a molecule; score the DNA"}

    @torch.no_grad()
    def codes_vs_dna_consistency(self, seqs: Sequence[str], key: str
                                 ) -> Dict[str, float]:
        """Are "run the model on the DNA" and "pin the codes" one function?

        Three numbers, because any one of them alone would hide the answer.
        `z_max_abs_diff` compares the encoder's z with
        `latent_from_codes(indices)` over every ladder level and must be
        exactly 0; that is the route comparison. `mu_same_context` compares mu
        from forward with `predict_from_codes` given forward's own context,
        and is also exactly 0. `mu_batch1_context` uses the context built at
        batch 1 by `design_panel.design_context`: 0 at batch 1 and ~1e-6 above
        it, which is the batch-size float32 reduction behind
        CONTEXT_TOLERANCE, not a second function.

        Reporting only the third would make a float32 reduction look like a
        broken contract; reporting only the first would hide that a score
        depends on batch composition at the 1e-6 level.
        """
        ctx = self.contexts[key]
        x = one_hot(list(seqs)).to(self.device)
        B = int(x.shape[0])
        rep = lambda t: None if t is None else t.expand(B, *t.shape[1:])
        out = self.model(
            x, text_emb=rep(ctx.text_emb), text_ok=rep(ctx.text_ok),
            species_id=torch.full((B,), self.species_row, dtype=torch.long,
                                  device=self.device),
            flank_l=rep(ctx.flank_l), flank_r=rep(ctx.flank_r),
            flank_ok=rep(ctx.flank_ok), reconstruct=False)
        idx = self.model.encode_codes(x)
        z_cod, _pts = self.model.latent_from_codes(idx)
        z_err = max(float((a - b).abs().max())
                    for a, b in zip(out["z"], z_cod))
        same = self.model.predict_from_codes(
            idx, context=out["context"])["absolute"].mu
        b1 = self.model.predict_from_codes(
            idx, context=ctx.vector.expand(B, -1))["absolute"].mu
        mu = out["absolute"].mu
        return {"n": float(B),
                "z_max_abs_diff": z_err,
                "mu_same_context": float((mu - same).abs().max()),
                "mu_batch1_context": float((mu - b1).abs().max())}


__all__ = ["CATEGORIES", "CONSTRUCTS", "CONTEXT_TOLERANCE", "Context", "ConstructScore",
           "CandidateScore", "Scorer", "one_hot", "codes_to_dna",
           "combine_min", "combine_mean"]
