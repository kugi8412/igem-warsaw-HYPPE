"""Assembly: the one class that composes the modules into a runnable model.

This is the only place the modules of ``hyppe.model``, ``hyppe.heads`` and
``hyppe.geometry`` are built next to each other, so it is the only place a
module that passes its own smoke test and breaks on wiring gets caught.

The order of the data::

    MotifOccupancy      [B, 4, 800] -> [B, M, 800]   frozen + de-novo
    PromoterEncoder     + anchor    -> five maps, coarsest first
    level_proj (1x1)    encoder width -> ModelConfig.channels per level
    ConeLadder          -> z / indices / points, geodesic composition
    LatentMask          -> K views glued along the batch axis
    CodeDecoder         -> logits [K*B, 4, 800], from codes only
    LatentPooling       z -> [B, D]
    ConditionEncoder    -> context tokens, flattened for the heads
    four heads          absolute / pairwise / discriminator / category

Conditioning sits behind the quantiser (decision S-1). In front of it the codes
would be a function of (sequence, condition), so ``decode_codes(indices)`` would
no longer be a function of the indices alone and ``predict_from_codes`` could
not reproduce a forward pass from a sampled code state. ``PromoterEncoder`` is
built with ``cond_dim=0`` and raises if a context is passed to it, so the
decision is structural rather than a convention.

The encoder contains BatchNorm, so ``encode_codes`` is a function of the batch
composition in train mode. The contract covers only ``decode_codes`` and
``predict_from_codes``, which carry no batch statistic at all, and every gate
below runs in ``eval`` -- the mode sampling runs in.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import HyppeConfig, fsq_grid
from ..data.gene_features import build_gene_features
from ..data.sequences import NUCLEOTIDE_ORDER
from ..heads import (
    AbsoluteStrengthHead,
    CategoryHead,
    LatentPooling,
    PairComparison,
    PairwiseRankingHead,
    PromoterDiscriminator,
)
from ..heads.decomposition import DecompositionHeads, TokenSlicer

# `_rms_normalise` is imported rather than re-implemented: the flank-dropout
# blend below has to reproduce the module's own normalisation bit for bit.
from .condition import (
    CategoryVocabulary,
    CentredLatentConditioner,
    CoarseConditionAttention,
    ConditionConfig,
    ConditionEncoder,
    ConditionOutput,
    LocalConditionAttention,
    _rms_normalise,
    hash_embedding,
)
from .decoder import CodeDecoder
from .encoder import PromoterEncoder
from .fusion import FusionConfig, LatentConditioner
from .gene import GeneConfig, GeneEncoder
from .ladder import ConeLadder, LadderOutput

#: Parameter-ledger note for the residual predictor. A separate constant
#: because the note has to state its own parameter delta: a missing delta is
#: the only thing that exposes a config key with no code behind it.
_NOTA_PRED = (
    "ON: %d parameters, window %d, hidden %d, MSE weight %.3f. The level "
    "k+1 quantiser receives the FEATURE MINUS A PREDICTION from a window of "
    "parent points, and the prediction is added back on the synthesis side; "
    "without that return the code would still have to specify the WHOLE "
    "step and the subtraction would remove no bits at all. The diffusion "
    "contract holds: the parent is a function of the indices, so "
    "decode_indices computes the same prediction. FALSIFIERS on the epoch "
    "line drabina/predRez: r (currently 1.59-1.64) must FALL at rec_full "
    ">= 0.98; pred_r2 must rise above the linear 0.2043/0.0385/0.0424/"
    "0.0658; used_frac must NOT fall (prediction collapsing into the "
    "feature). Rate term fsq_rate_weight = %.4g."
)
_NOTA_TEMPO = (
    "ON: %.4g, zero parameters. This is the CONTROL ARM -- the rate term "
    "without the residual predictor. On its own it lowers r trivially "
    "(Goodhart); the number that means something is the difference in r "
    "against the drab_pred arm at the same rec_full."
)
from .attention_readout import AttentionReadout
from .bands_l0 import BandsL0
from .condition_cone import ConditionCone
from .masking import LatentMask
from .motifs import DeNovoMotifBank, MemeMotifBank, MotifOccupancy, read_meme
from .quantizer import HyperbolicCodebook, ScalarFSQ
from .sharpening import LevelSharpening
from .species import OUTGROUP_TAXA, GenomeEmbedding, StrainRegistry, default_tree_paths

__all__ = ["HyppeModel", "AssemblyNote", "subcategory_names"]


class AssemblyNote(tuple):
    """(key, reason) -- something that is inert, substituted or degraded.

    A plain tuple subclass so it prints as one line in a report.
    """

    __slots__ = ()

    def __new__(cls, key: str, reason: str):
        return super().__new__(cls, (str(key), str(reason)))

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return "%-28s %s" % (self[0], self[1])


def subcategory_names(path: str) -> Tuple[str, ...]:
    """Subcategory labels from `data/conditions/condition_taxonomy.tsv`.

    Multi-label: the column holds a comma-separated list. Returned sorted, so
    the order is reproducible from the file alone -- the index of a category is
    its logit, and set iteration order would silently renumber them.
    """
    import csv

    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw = (row.get("subcategories") or "").strip()
            for part in raw.split(","):
                part = part.strip()
                if part:
                    seen.add(part)
    return tuple(sorted(seen))


def _text_embeddings(config, categories, notes):
    """Frozen LLM embeddings of the experiment descriptions, in column order.

    No silent fallback: if the file is missing or does not cover every category
    the channel is switched off and an assembly note says so. A hash-embedded
    stand-in would train, take a share of the context and carry nothing.
    """
    import numpy as _np
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(os.environ.get("HYPPE_DATA_ROOT", _repo_root),
                        "data/category_text_emb.npz")
    try:
        z = _np.load(path, allow_pickle=True)
        names = [str(x) for x in z["categories"]]
        V = _np.asarray(z["vectors"], dtype="float32")
    except Exception as exc:                       # noqa: BLE001
        notes.append(AssemblyNote(
            "absolute.text_emb",
            "OFF: %s. The seven outputs fall back to free weights, so a new "
            "experiment cannot be added by describing it." % (exc,)))
        return None
    idx = {n: i for i, n in enumerate(names)}
    missing = [c for c in categories if c not in idx]
    if missing:
        notes.append(AssemblyNote(
            "absolute.text_emb",
            "OFF: no description for %s" % (", ".join(missing),)))
        return None
    out = _np.stack([V[idx[c]] for c in categories]).astype("float32")
    sim = out @ out.T
    off = sim[~_np.eye(len(categories), dtype=bool)]
    notes.append(AssemblyNote(
        "absolute.text_emb",
        "%d descriptions x %d dims from %s; pairwise cosine min %.4f mean "
        "%.4f max %.4f (near 1.0 would mean interchangeable descriptions)"
        % (out.shape[0], out.shape[1], str(z["backend"]),
           float(off.min()), float(off.mean()), float(off.max()))))
    return out


def _category_prototypes(config, vocab, notes):
    """Frozen-LM prototypes for every entry of the condition vocabulary (D2).

    Either every category carries a real language-model vector or none does and
    the note says HASH. A matrix that is real for the seven parent categories
    and hashed for the thirty-three subcategories would be worse than either:
    nothing downstream could tell which rows mean anything, so the head would
    learn to trust all forty equally.

    The dimension travels with the matrix. `set_category_prototypes` demands an
    exact [n_cat, cond_d_lm] shape and the artefact is 768-dimensional while the
    historical default was 384, so returning the config next to the vectors
    makes the mismatch unconstructible rather than a ValueError raised later.

    Returns (ConditionConfig, [n_cat, d_lm] float32).
    """
    import numpy as _np
    names = list(vocab.names)
    phrases = vocab.phrase_list()
    path = os.path.join(_repo_root(), "data", "condition_proto_emb.npz")

    def _substitute(reason):
        cfg = ConditionConfig()
        notes.append(AssemblyNote(
            "C2 category prototypes",
            "HASH embeddings, not the frozen LM: %s. Deterministic substitute "
            "so the prototype head is non-degenerate, but the %d category "
            "directions are ARBITRARY: the head trains, spends "
            "cond_cat_aux_weight, and carries no semantics, and D2 (specify an "
            "experiment by DESCRIBING it) does not hold for this build. "
            "Rebuild the artefact with poprawki/sapbert_proto_patch.py before "
            "any run whose numbers are quoted." % (reason, len(names))))
        return cfg, _np.stack([hash_embedding(p, cfg.cond_d_lm)
                               for p in phrases]).astype(_np.float32)

    try:
        z = _np.load(path, allow_pickle=True)
        have = [str(x) for x in z["categories"]]
        V = _np.asarray(z["vectors"], dtype="float32")
        model_id = str(z["model_id"])
        backend = str(z["backend"])
        revision = str(z["revision"])
        source = [str(x) for x in z["source"]]
    except Exception as exc:                       # noqa: BLE001
        return _substitute("%s: %s" % (type(exc).__name__, exc))

    idx = {n: i for i, n in enumerate(have)}
    missing = [n for n in names if n not in idx]
    if missing:
        return _substitute(
            "%s covers %d of %d categories, missing %s"
            % (os.path.basename(path), len(names) - len(missing), len(names),
               ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")))

    M = _np.stack([V[idx[n]] for n in names]).astype(_np.float32)
    src = [source[idx[n]] for n in names]
    n_desc = sum(1 for s in src if s.startswith("description"))
    cfg = ConditionConfig(cond_d_lm=int(M.shape[1]),
                          cond_text_model=model_id,
                          cond_text_revision=(revision or None))
    F = M.astype("float64")
    sim = F @ F.T
    off = sim[~_np.eye(len(names), dtype=bool)]
    notes.append(AssemblyNote(
        "C2 category prototypes",
        "REAL frozen-LM embeddings, %d of %d categories covered, %d dims, "
        "from %s (revision %s): %d rows are the corpus-grounded experiment "
        "descriptions, copied bit for bit from the artefact the absolute head "
        "reads, and %d are the vocabulary phrase. Pairwise cosine min %.4f "
        "mean %.4f max %.4f -- near 1.0 would mean interchangeable "
        "categories. cond_d_lm %d -> %d, so text_proj is Linear(%d, %d) and a "
        "checkpoint written under the hash stand-in will NOT load here."
        % (len(names), len(names), M.shape[1], backend, revision or "unpinned",
           n_desc, len(names) - n_desc, float(off.min()), float(off.mean()),
           float(off.max()), ConditionConfig().cond_d_lm, M.shape[1],
           M.shape[1], cfg.cond_d_ctx)))

    ch = getattr(getattr(config, "conditioning", None), "experiment_text", None)
    if ch is not None and int(ch.dim) != int(M.shape[1]):
        # Raises rather than notes when the channel is enabled: the loop does
        # supply text_emb behind that key, so a width disagreement would be a
        # matmul error on the first forward.
        if bool(ch.enabled):
            raise ValueError(
                "conditioning.experiment_text.dim = %d but the condition "
                "encoder takes %d: `text_proj` is Linear(cond_d_lm, "
                "cond_d_ctx) and cond_d_lm comes from the category "
                "prototype matrix, which is %d wide. The channel is "
                "ENABLED, so this would be a shape error at the first "
                "forward. Set the key to %d, or rebuild the prototypes."
                % (int(ch.dim), int(M.shape[1]), int(M.shape[1]),
                   int(M.shape[1])))
        notes.append(AssemblyNote(
            "conditioning.experiment_text.dim",
            "declares %d while the condition encoder now takes %d. The "
            "channel is enabled=%r, so the mismatch is INERT today -- "
            "the first run that switches the channel on RAISES here "
            "instead of reaching a forward."
            % (int(ch.dim), int(M.shape[1]), bool(ch.enabled))))
    return cfg, M


def _category_weights(config, strain_names, n_cat, notes):
    """Per-genome category proportions, aligned to the strain registry.

    Measured, not chosen: these do as well as the exact per-pair `n_obs` at
    reproducing the marginal (R^2 0.999849 either way) and, unlike `n_obs`,
    exist at design time; uniform weights give up 0.0083 of R^2, which is the
    control showing the comparison has resolution. Failure is reported rather
    than silent, because a silent fallback here would look exactly like a
    correct derivation.
    """
    import numpy as _np
    try:
        from ..data.labels import load_samples as _ls
        path = str(getattr(config.data, "samples_path", "") or "")
        if not path:
            raise IOError("data.samples_path is empty")
        tab = _ls(path)
        keep = _np.asarray(tab.keep, dtype=bool)
        cat = _np.asarray(tab.category, dtype=_np.float64)[keep]
        gen = _np.asarray(tab.genome)[keep]
    except Exception as exc:                       # noqa: BLE001
        notes.append(AssemblyNote(
            "absolute.cell_weights",
            "UNIFORM fallback: %s. The derived marginal is then the plain "
            "mean of the seven categories, which measured R^2 0.9915 against "
            "0.9998 for per-genome proportions." % (exc,)))
        return _np.ones((1, n_cat), dtype=_np.float32)
    w = _np.zeros((len(strain_names), n_cat), dtype=_np.float32)
    seen = 0
    for i, name in enumerate(strain_names):
        sel = gen == name
        if not bool(sel.any()):
            continue
        seen += 1
        w[i] = (cat[sel] > 0).sum(axis=0)[:n_cat]
    notes.append(AssemblyNote(
        "absolute.cell_weights",
        "per-genome category proportions for %d of %d registry rows; rows "
        "with no samples fall back to the corpus profile inside the head"
        % (seen, len(strain_names))))
    return w


class HyppeModel(nn.Module):
    """Encoder + cone ladder + decoder + condition + four heads, in one object.

    Construct with :meth:`build` (which is the G1 gate) or directly from a
    :class:`~hyppe.config.HyppeConfig`.

    Capabilities and where each one lives::

        C1 species         ``GenomeEmbedding`` -> ``species_tangent`` token
        C2 experiment      ``ConditionEncoder`` text token + multi-label
                           ``cat_logits`` over parent and subcategories
        C3 promoter/not    ``PromoterDiscriminator``
        C4 known motifs    ``MemeMotifBank`` (frozen log-odds, both strands)
        C5 de-novo motifs  ``DeNovoMotifBank`` (learned width, sparsity)
        C6 siamese rank    ``PairwiseRankingHead``; ``forward`` returns the
                           per-sequence score, pairing is the loop's job
        C7 variance        ``AbsoluteStrengthHead`` (mu, log_var, log_sigma_obs)
        C8 diffusion API   ``encode_codes`` / ``decode_codes`` /
                           ``predict_from_codes``
        C9 task balance    ``nash_players()`` / ``subordinate_terms()``
    """

    #: Token order of the flattened context handed to the heads. A checkpoint
    #: contract: the first `d_ctx` columns of every head's first layer are the
    #: species token.
    CONTEXT_TOKENS = ("species", "text", "flank_l", "flank_r", "gene")

    # Build
    @classmethod
    def build(cls, config: Optional[HyppeConfig] = None, **kwargs
              ) -> "HyppeModel":
        """G1: construct from the default config. Raises, never degrades."""
        return cls(config or HyppeConfig(), **kwargs)

    def __init__(self,
                 config: HyppeConfig,
                 meme_path: Optional[str] = None,
                 tree_path: Optional[str] = None,
                 taxonomy_path: Optional[str] = None,
                 de_novo_filters: int = 32,
                 encoder_dim: Optional[int] = None,
                 # Fallback only: `conditioning.gene_context.dropout` is
                 # read below and wins whenever that channel is enabled.
                 flank_drop_prob: float = 0.25,
                 context_fuse: str = "flatten",
                 require_motifs: bool = True) -> None:
        super().__init__()
        config.validate()
        self.cfg = config
        self.notes: List[AssemblyNote] = []
        mc, dc, hc = config.model, config.data, config.heads
        self.seq_len = int(dc.sequence_length)
        self.n_levels = mc.n_levels()
        self.level_lengths = mc.map_lengths(self.seq_len)   # coarsest first
        self.channels = [int(c) for c in mc.channels]
        if len(self.channels) != self.n_levels:
            raise ValueError("model.channels has %d entries, %d levels"
                             % (len(self.channels), self.n_levels))
        if context_fuse not in ("flatten", "mean"):
            raise ValueError("context_fuse must be flatten|mean, got %r"
                             % (context_fuse,))
        self.context_fuse = context_fuse

        root = _repo_root()

        # C4 / C5 motifs
        _mc = getattr(config.model, "motifs", None)
        meme = (meme_path
                or (str(getattr(_mc, "bank_path", "") or "") if _mc else "")
                or os.path.join(root, "data", "motifs.meme"))
        frozen: Optional[MemeMotifBank] = None
        if os.path.isfile(meme):
            _profile = read_meme(meme)
            _n0 = len(_profile)
            _profile, _nota = _odsiej_motywy(_profile, _mc)
            if len(_profile) == 0:
                raise ValueError(
                    "the model.motifs criteria removed ALL %d profiles from "
                    "%r. An empty bank is not a bank -- the run would have "
                    "no motif channel at all while still being named as one "
                    "with exclusions." % (_n0, meme))
            frozen = MemeMotifBank(_profile)
            if _nota:
                self.notes.append(AssemblyNote("model.motifs", _nota))
        elif require_motifs:
            raise FileNotFoundError(
                "no MEME file at %r. A model built without the frozen bank is "
                "a different model (C4 absent) and would report the same "
                "headline numbers, so this is an error and not a fallback. "
                "Pass require_motifs=False to build the de-novo-only variant "
                "deliberately." % (meme,))
        else:
            self.notes.append(AssemblyNote(
                "motifs.frozen", "no MEME file at %r: C4 (known motifs) is "
                "ABSENT from this build" % (meme,)))
        de_novo = (DeNovoMotifBank(n_filters=int(de_novo_filters))
                   if int(de_novo_filters) > 0 else None)
        if de_novo is None:
            self.notes.append(AssemblyNote(
                "motifs.de_novo", "0 filters: C5 (de-novo motifs) is absent"))
        self.occupancy = (MotifOccupancy(frozen=frozen, de_novo=de_novo)
                          if (frozen is not None or de_novo is not None)
                          else None)
        if self.occupancy is not None:
            # Which scale made the encoder's input: `profile` divides each
            # channel by its own PWM's background moments, so occupancy at i
            # does not depend on j, while `sequence` z-scores over all 800
            # positions and was measured to move 800/800 of them under a single
            # flipped base. Nothing in the weights records which one ran.
            self.notes.append(AssemblyNote(
                "motifs.occupancy_scale",
                "occupancy scaled by %r (normalise=%s). 'profile' = fixed "
                "per-PWM background mean/sd, local; 'sequence' = per-sample "
                "z-score over all positions, GLOBAL. Checkpoints from the two "
                "are not comparable and the state_dict does not record it."
                % (getattr(self.occupancy, "scale", "sequence"),
                   bool(self.occupancy.normalise))))
        occ_ch = 0 if self.occupancy is None else int(self.occupancy.n_channels)
        if occ_ch != int(mc.occupancy_channels):
            # The config key is a declaration; the bank is the fact.
            self.notes.append(AssemblyNote(
                "model.occupancy_channels",
                "config says %d, the assembled bank has %d (%s frozen + %d "
                "de-novo); the bank wins and the encoder is built for %d"
                % (int(mc.occupancy_channels), occ_ch,
                   0 if frozen is None else frozen.n_motifs,
                   0 if de_novo is None else de_novo.n_filters, occ_ch)))
        self.occupancy_channels = occ_ch

        # Encoder
        enc_dim = int(encoder_dim) if encoder_dim else int(max(self.channels))
        def _level_dilations(v):
            """Flat list -> shared by the tower; list of lists -> per level."""
            if not v:
                return (1,)
            v = list(v)
            if all(isinstance(e, (list, tuple)) for e in v):
                return tuple(tuple(int(d) for d in e) for e in v)
            return tuple(int(d) for d in v)
        self.encoder = PromoterEncoder(
            dim=enc_dim,
            strides=tuple(mc.strides),
            occupancy_channels=occ_ch,
            dropout_kind=str(getattr(mc, "dropout_kind", "element")
                             or "element"),
            level_blocks=(list(getattr(mc, "level_blocks", 0))
                          if isinstance(getattr(mc, "level_blocks", 0),
                                        (list, tuple))
                          else int(getattr(mc, "level_blocks", 0) or 0)),
            level_kernel=int(getattr(mc, "level_kernel", 3) or 3),
            level_dilations=_level_dilations(
                getattr(mc, "level_dilations", (1,))),
            # Consumer of `model.level_gate0`; without this line the key is
            # inert and the constructor default wins.
            level_gate0=(list(getattr(mc, "level_gate0", 0.1))
                         if isinstance(getattr(mc, "level_gate0", 0.1),
                                       (list, tuple))
                         else float(getattr(mc, "level_gate0", 0.1) or 0.1)),
            skip_gate0=(list(getattr(mc, "skip_gate0", 1.0))
                        if isinstance(getattr(mc, "skip_gate0", 1.0),
                                      (list, tuple))
                        else float(getattr(mc, "skip_gate0", 1.0) or 1.0)),
            occupancy_bottleneck=int(
                getattr(mc, "occupancy_bottleneck", 0) or 0),
            # Consumers of `model.occupancy_gate`.
            occupancy_gate=bool(getattr(mc, "occupancy_gate", False)),
            occupancy_gate_init=float(
                getattr(mc, "occupancy_gate_init", 4.0)),
            anchor=bool(mc.position_channel),
            stem_kernel=int(mc.stem_kernel),
            pre_blocks=int(mc.n_res_blocks),
            pre_kernel=int(mc.res_kernel),
            # Consumer of `model.pre_dilations`.
            pre_dilations=tuple(int(d) for d in mc.pre_dilations),
            billboard_layers=1 if occ_ch else 0,
            # Component 5, the depthwise context on the coarsest map.
            # `receptive_field()` reports `context_covers_map`, False for every
            # k < 2W-1 = 49 at 800 bp: the coarse map does not see the promoter.
            context_kernel=int(mc.context_kernel),
            # Decision S-1: `cond_dim=0` is what makes `encode_codes` a
            # function of the sequence alone.
            cond_dim=0)
        self.notes.append(AssemblyNote(
            "S-1 conditioning site",
            "cond_dim=0 in the encoder: the condition enters BEHIND the "
            "quantiser (heads), so decode_codes/predict_from_codes stay "
            "functions of the indices alone"))
        # Per-level 1x1 projections. `PromoterEncoder` has one width for every
        # level while `ModelConfig.channels` declares five, so the declaration
        # is realised here rather than ignored.
        self.level_proj = nn.ModuleList(
            [nn.Identity() if c == enc_dim else nn.Conv1d(enc_dim, c, 1)
             for c in self.channels])

        # Ladder
        l0_dim = int(mc.l0_dim)
        self.l0_dim = l0_dim
        position_q = HyperbolicCodebook(
            dim=l0_dim, codebook_size=int(mc.l0_codes),
            decay=float(mc.vq_ema_decay),
            commit_weight=float(config.training.commitment_weight),
            max_radius=float(mc.l0_max_radius),
            commit_measure=str(mc.l0_commit_measure),
            lift_norm=str(mc.l0_lift_norm),
            lift_squash=float(mc.l0_lift_squash),
            geodesic_target=bool(mc.vq_frechet),
            dead_restart=bool(mc.vq_dead_restart),
            learnable=bool(mc.vq_learnable))
        tangent_qs = []
        self.level_bits: List[float] = [math.log2(float(mc.l0_codes))]
        for b in mc.fsq_bits:
            dims, _cells, real = fsq_grid(int(b))
            tangent_qs.append(ScalarFSQ(
                dim=l0_dim, levels=(3,) * dims,
                gain=float(mc.fsq_gain),
                entropy_weight=float(mc.fsq_entropy_weight),
                rate_weight=float(mc.fsq_rate_weight),
                rate_seq_len=float(self.seq_len)))
            self.level_bits.append(real)
        if len(tangent_qs) != self.n_levels - 1:
            raise ValueError("model.fsq_bits has %d entries, the ladder needs "
                             "%d refinement levels"
                             % (len(tangent_qs), self.n_levels - 1))
        aperture = list(mc.ladder_aperture)
        if len(aperture) == 1:
            aperture = aperture * (self.n_levels - 1)
        if len(aperture) != self.n_levels - 1:
            # Broadcasting a two-entry aperture over four levels would silently
            # reinterpret a measured pair of numbers.
            last = aperture[-1]
            aperture = aperture + [last] * (self.n_levels - 1 - len(aperture))
            self.notes.append(AssemblyNote(
                "model.ladder_aperture",
                "declared %r for %d refinement levels; the last value is "
                "repeated, which is the CONSERVATIVE direction (a too-wide "
                "aperture would show as step_clamped_frac near 0)"
                % (list(mc.ladder_aperture), self.n_levels - 1)))
        self.ladder = ConeLadder(
            position_quantizer=position_q,
            tangent_quantizers=tangent_qs,
            dim=l0_dim,
            in_channels=self.channels,
            step_length=aperture,
            radius_step=float(mc.ladder_step_radius),
            seq_len=self.seq_len,
            seed=int(config.training.seed),
            pred_reszty=bool(mc.drab_pred),
            pred_okno=int(mc.drab_pred_okno),
            pred_ukryte=int(mc.drab_pred_ukryte),
            pred_waga=float(mc.drab_pred_waga))
        _rp = getattr(self.ladder, "residual_pred", None)
        _rp_n = 0 if _rp is None else sum(p.numel() for p in _rp.parameters())
        if _rp is not None:
            self.notes.append(AssemblyNote(
                "model.drab_pred",
                _NOTA_PRED % (_rp_n, int(mc.drab_pred_okno),
                              int(mc.drab_pred_ukryte),
                              float(mc.drab_pred_waga),
                              float(mc.fsq_rate_weight))))
        elif float(mc.fsq_rate_weight) > 0.0:
            self.notes.append(AssemblyNote(
                "model.fsq_rate_weight",
                _NOTA_TEMPO % float(mc.fsq_rate_weight)))

        # Decoder
        z_dims = [l0_dim] * self.n_levels
        self.z_dims = z_dims
        if not mc.decoder_sees_codes_only or mc.allow_continuous_decoder:
            raise ValueError(
                "decoder_sees_codes_only=%r / allow_continuous_decoder=%r: a "
                "continuous bypass gives the decoder a lossless copy of the "
                "encoder and makes the codes optional (measured in the "
                "sibling track: reported recon 0.9984 with the bypass, 0.29 "
                "from codes alone against a floor of 0.25). This assembly "
                "does not implement it."
                % (mc.decoder_sees_codes_only, mc.allow_continuous_decoder))
        self.decoder = CodeDecoder(level_dims=z_dims,
                                   level_lengths=self.level_lengths,
                                   seq_len=self.seq_len,
                                   mixing_max_len=int(mc.band_width))
        # U1 is asserted by the decoder's own constructor; repeated here so a
        # later edit to this file cannot quietly drop it.
        self.decoder.assert_no_batch_statistics()

        # Masking
        self.mask = LatentMask(level_dims=z_dims,
                               level_lengths=self.level_lengths,
                               seq_len=self.seq_len,
                               level_bits=self.level_bits,
                               weights=_expand_view_weights(
                                   config.training.view_weights,
                                   self.n_levels),
                               seed=int(config.training.seed))

        # C1 species. `conditioning.genome.source` selects the tree.
        # "embedding" is the ten-leaf supermatrix keyed by accession: it places
        # seven registry genomes and collapses thirteen onto one centroid,
        # including six strains whose measured control spans 0.039 to 7.624.
        # "tree20" is the in-house NJ tree keyed by strain name and places all
        # twenty.
        _src = str(getattr(config.conditioning.genome, "source", "") or "")
        _paths = default_tree_paths()
        if _src == "tree20":
            registry = StrainRegistry.by_leaf_name()
            tree = tree_path or _paths.get("inhouse20")
            _outgrp = None          # this tree carries no outgroup
            if not tree:
                raise FileNotFoundError(
                    "conditioning.genome.source='tree20' but "
                    "data/trees/external/inhouse_nj_20genomes_aasubst.tre "
                    "was not found. Refusing to fall back to the ten-leaf "
                    "tree: that would silently collapse 13 of 20 genomes "
                    "onto one point and the run would look healthy.")
        else:
            registry = StrainRegistry.default()
            tree = tree_path or _paths["supermatrix"]
            _outgrp = OUTGROUP_TAXA
        self.strain_names = registry.names
        genome_dim = int(config.conditioning.genome.dim)
        self.genome: Optional[GenomeEmbedding] = None
        if config.conditioning.genome.enabled and tree:
            # F6 (`mean |d_L/(s*d_patr) - 1|`, threshold 0.10) on the 20-leaf
            # tree: 0.1673 at 10 restarts, 0.1050 at 40, while the ten-leaf
            # tree already reaches 0.0128 at 10. More dimensions are worse
            # (0.179 at 4, 0.240 at 6): the limit is the fit's convergence.
            _rest = 40 if _src == "tree20" else 10
            self.genome = GenomeEmbedding.from_tree(
                registry, tree_path=tree, out_dim=genome_dim,
                seed=int(config.training.seed), outgroup=_outgrp,
                restarts=_rest,
                tree_prior=float(config.conditioning.genome_tree_prior),
                prior_denom_floor=float(
                    config.conditioning.genome_prior_denom_floor))
            # Strains the tree cannot tell apart. I1237/P3041/P3116 have
            # d_patr = 0 and their mutual pairs fall below PRIOR_MIN_DIST, yet
            # after nine epochs they had separated by 6e-05 against +0.049 on
            # bound pairs: the tree gives each the same distance to every other
            # leaf, so binding them to the rest pins them together indirectly.
            # The flag keeps one representative per class in the prior.
            if bool(getattr(config.conditioning,
                            "genome_prior_free_collapsed", False)):
                _przed = int(self.genome._prior_i.numel())
                self.genome.prior_free_collapsed = True
                self.genome._set_prior_pairs(self.genome._patristic,
                                             self.genome._patristic_taxa)
                _po = int(self.genome._prior_i.numel())
                _wolnych = len(getattr(self.genome, "_prior_wolne", ()))
                self.notes.append(AssemblyNote(
                    "conditioning.genome_prior_free_collapsed",
                    "strains in classes at patristic distance 0 are taken "
                    "out of the prior except for one representative per "
                    "class: %d rows freed, prior pairs %d -> %d. Without "
                    "this the prior ties every member of a class identically "
                    "to 16 other genomes and so pins the class to a single "
                    "point -- a measured separation of 6e-05 over 9 epochs "
                    "against +0.049 on the pairs that are tied."
                    % (_wolnych, _przed, _po)))
            _npl = sum(1 for r in registry if r.placed and r.taxon)
            self.notes.append(AssemblyNote(
                "conditioning.genome.tree",
                "%s: %d of %d genomes placed by name/taxon"
                % (os.path.basename(tree), _npl, len(registry))))
            _pf = self.genome.prior_fit()
            self.notes.append(AssemblyNote(
                "conditioning.genome.prior",
                "%d pairs, denominator floor %.3g softens %d of them; "
                "stiffness max/median %.0fx. The prior weights a pair as "
                "1/d^2, so without a floor it holds the phylogenetically "
                "closest genomes hardest -- which are the strains the "
                "channel exists to tell apart."
                % (int(_pf["n_pairs"]), float(_pf["denom_floor"]),
                   int(_pf["softened"]),
                   float(_pf["stiff_max_over_median"]))))
        elif config.conditioning.genome.enabled:
            # An all-at-origin embedding has logmap0 == 0 for every genome:
            # a channel that is present, costs parameters and carries nothing.
            self.notes.append(AssemblyNote(
                "conditioning.genome",
                "no tree file: the hyperbolic genome channel would be "
                "identically zero, so the learned strain TABLE is used "
                "instead and C1 has no geometry in this build"))
        else:
            self.notes.append(AssemblyNote(
                "conditioning.genome", "disabled in the config"))

        # C2 condition
        vocab = CategoryVocabulary()
        tax = taxonomy_path or os.path.join(root, "data", "conditions",
                                            "condition_taxonomy.tsv")
        self.n_parent_categories = len(vocab)
        if os.path.isfile(tax):
            subs = subcategory_names(tax)
            for name in subs:
                if name not in vocab.names:
                    vocab = vocab.add(name, name.replace("_", " "))
            self.subcategories = subs
        else:
            self.subcategories = ()
            self.notes.append(AssemblyNote(
                "C2 subcategories",
                "no taxonomy at %r: the category head is single-level "
                "(%d parent categories) and the measured two-level "
                "taxonomy is absent" % (tax, len(vocab))))
        # One call for both, because `set_category_prototypes` demands an exact
        # [n_cat, cond_d_lm] shape and the two numbers must agree.
        cond_cfg, _proto = _category_prototypes(config, vocab, self.notes)
        # Species-token dropout, when the config states one. A negative value
        # leaves the ConditionConfig default in force.
        _sd = float(getattr(config.conditioning, "species_dropout", -1.0))
        if _sd >= 0.0:
            cond_cfg.cond_dropout_species = _sd
        # Text-token dropout trains `unknown_token_text`, the representation a
        # zero-shot inference on an unseen description falls back to.
        _et = getattr(config.conditioning, "experiment_text", None)
        # The sentinel is 0.0, not a negative value: `ChannelConfig.validate`
        # rejects negatives, so a negative branch would be unreachable from a
        # config. Silence and an explicit 0.0 are therefore indistinguishable,
        # and silence keeps the ConditionConfig default.
        _td = float(getattr(_et, "dropout", 0.0)) if _et is not None else 0.0
        _td_stary = float(cond_cfg.cond_dropout_text)
        if _td > 0.0:
            cond_cfg.cond_dropout_text = _td
            self.notes.append(AssemblyNote(
                "conditioning.experiment_text.dropout",
                "text token dropout set from the config: %.3f (the "
                "ConditionConfig default is %.3f). It trains "
                "`unknown_token_text`, the token a zero-shot inference on "
                "an unseen experiment description falls back to; at 0.0 "
                "that token is never trained." % (_td, _td_stary)))
        else:
            self.notes.append(AssemblyNote(
                "conditioning.experiment_text.dropout",
                "config is SILENT (0.0); the ConditionConfig default %.3f "
                "stays in force and this run does NOT declare it."
                % (_td_stary,)))
            self.notes.append(AssemblyNote(
                "conditioning.species_dropout",
                "species token dropout set from the config: %.3f (the "
                "ConditionConfig default is %.3f). Measured reason: "
                "substituting the species channel globally moves val PCC by "
                "-0.0026 at the peak epoch and +0.0046 at epoch 30, so past "
                "the peak the channel HURTS on held-out orthogroups."
                % (_sd, 0.15)))

        # C12: the gene the promoter drives, as an appended fifth context
        # token. Admitted on a measurement, per `ConditioningConfig.gene`'s own
        # rule: ridge on the 44-column phyletic block, fit on 10 168 TRAIN
        # orthogroups and scored on 1 155 held-out ones (the split is
        # orthogroup-disjoint, so every scored gene is unseen) gives R^2
        # +0.3163 against a features-permuted control of -0.0069 +- 0.0033.
        # The -0.1279 recorded elsewhere for "the gene channel" is a different
        # feature block scored on the model residual.
        self.gene_features = None
        self.gene: Optional[GeneEncoder] = None
        gcfg = config.conditioning.gene
        if gcfg.enabled:
            # A path inside a checkpoint is the path of the machine that
            # trained it, so on a fresh clone this branch used to raise. The
            # configured path still wins when it resolves; only otherwise do we
            # look where the file actually ships.
            _default = os.path.join(root, "data", "final_datasets",
                                    "orthology", "og_table_combined.tsv")
            _cand = [p for p in (config.conditioning.orthology_path,
                                 _default) if p]
            if config.conditioning.orthology_path:
                # also try the configured path read against the data root,
                # which covers a config written with a relative path
                _cand.insert(1, os.path.join(
                    root, config.conditioning.orthology_path.lstrip("/\\")))
            opath = next((p for p in _cand if os.path.isfile(p)), None)
            if opath is None:
                raise FileNotFoundError(
                    "conditioning.gene is enabled but the orthology table was "
                    "not found. Tried: %s. Refusing to build an empty gene "
                    "channel: it would report a context share and carry "
                    "nothing, which is the failure this channel was just "
                    "rescued from. Set HYPPE_DATA_ROOT if your data lives "
                    "outside the repository." % ", ".join(repr(p)
                                                          for p in _cand))
            self.gene_features = build_gene_features(opath)
            _spec = self.gene_features.spec
            self.gene = GeneEncoder(
                GeneConfig(d_ctx=int(cond_cfg.cond_d_ctx),
                           d_lm=int(cond_cfg.cond_d_lm),
                           identity_dim=int(
                               config.conditioning.gene_identity_dim),
                           identity_buckets=int(
                               config.conditioning.gene_identity_buckets),
                           use_text=bool(config.conditioning.gene_use_text),
                           p_drop_numeric=float(
                               config.conditioning.gene_drop_numeric),
                           p_drop_text=float(
                               config.conditioning.gene_drop_text),
                           p_drop_all=float(gcfg.dropout)),
                n_numeric=int(_spec.n_numeric()), spec=_spec)
            self.notes.append(AssemblyNote(
                "C12 gene channel",
                "token 4 of %d: %d orthogroups x %d numeric columns from %s. "
                "identity_dim=%d (0 = zero-shot: the split is "
                "orthogroup-disjoint, so an identity row of any evaluated "
                "gene would receive no gradient), use_text=%s"
                % (len(ConditionOutput.TOKEN_NAMES),
                   len(self.gene_features.og_ids), _spec.n_numeric(),
                   os.path.basename(opath),
                   int(config.conditioning.gene_identity_dim),
                   bool(config.conditioning.gene_use_text))))
            if config.conditioning.gene_use_text:
                self.notes.append(AssemblyNote(
                    "conditioning.gene_use_text",
                    "true, but `build_gene_features` reports the text column "
                    "as the PHYLETIC DESCRIPTION -- a function of the numeric "
                    "block, so it adds no independent information in this "
                    "build. It costs an LM embedding per row and a projection."))
        else:
            self.notes.append(AssemblyNote(
                "conditioning.gene",
                "disabled: the model has no key for WHICH GENE the promoter "
                "drives, so a design objective phrased 'make this strong for "
                "pks1' has nothing to bind to"))

        self.condition = ConditionEncoder(
            config=cond_cfg, vocab=vocab, n_strains=len(registry),
            d_species=(genome_dim if self.genome is not None else None),
            strain_names=registry.names,
            category_embeddings=_proto,
            gene=self.gene)
        if self.genome is not None:
            # `context()` passes species_id=None whenever the Lorentz registry
            # exists, so the flat strain table would never be indexed. It used
            # to be built anyway, with a measured gradient of exactly
            # 0.000e+00 from the absolute loss.
            _se = getattr(self.condition, "species_emb", None)
            self.notes.append(AssemblyNote(
                "condition.species_emb",
                ("NOT BUILT under this wiring: the Lorentz registry supplies "
                 "the species tangent, so the flat strain table would never "
                 "be indexed. It used to be built anyway -- measured "
                 "gradient exactly 0.000e+00 from the absolute loss -- and "
                 "was weight-decayed and checkpointed for nothing.")
                if _se is None else
                ("REACHABLE (%d params): no species tangent is supplied, so "
                 "the flat strain table IS the species channel"
                 % int(_se.weight.numel()))))
        #: Context channels withheld from the pairwise and discriminator
        #: branches. An empty tuple means every head sees the same tokens.
        self.head_withhold: Tuple[str, ...] = tuple(
            str(x) for x in getattr(config.heads, "head_withhold", ()) or ())
        self.d_ctx = int(self.condition.d_ctx)
        self.n_context_tokens = int(getattr(self.condition, "n_tokens", 4))
        self.context_tokens = tuple(
            self.CONTEXT_TOKENS[:self.n_context_tokens])
        self.context_dim = (self.n_context_tokens * self.d_ctx
                            if context_fuse == "flatten" else self.d_ctx)
        #: Names the heads can slice the flattened context by. `None` under
        #: `context_fuse="mean"`, where the per-token structure is gone.
        self.slicer = (TokenSlicer(self.context_tokens, self.d_ctx)
                       if context_fuse == "flatten" else None)

        # Flank dropout. `conditioning.gene_context.dropout` names this
        # quantity; an explicit kwarg still wins, for probes and ablations.
        _flank_drop_from_cfg = flank_drop_prob
        _gc = getattr(config.conditioning, "gene_context", None)
        if _gc is not None and bool(getattr(_gc, "enabled", False)):
            _flank_drop_from_cfg = float(getattr(_gc, "dropout",
                                                 flank_drop_prob))
        self.flank_drop_prob = float(_flank_drop_from_cfg)
        if self.flank_drop_prob <= 0.0:
            self.notes.append(AssemblyNote(
                "flank_drop_prob",
                "0.0: the 'unknown flank' token is never trained, so an "
                "inference on a plasmid (measured 5.13 sd outside the genomic "
                "flank distribution) would use an untrained token"))

        # C6 / C7 / C3 heads
        self.pooling = LatentPooling(channels=z_dims,
                                     stats=tuple(hc.pooling.stats),
                                     anchor_frac=float(hc.pooling.anchor_frac))
        d_lat = self.pooling.out_dim
        self.latent_dim = d_lat
        if d_lat != config.latent_dim():
            # `HyppeConfig.latent_dim()` sums the encoder widths. The heads
            # read `z`, which is `l0_dim` wide at every level because z is the
            # spatial part of a point of H^{l0_dim}; quoting the config number
            # would overstate the head input fourfold.
            self.notes.append(AssemblyNote(
                "HyppeConfig.latent_dim()",
                "returns %d (sum of encoder channels x %d stats) but the "
                "pooled latent the heads receive is %d (%d levels x l0_dim %d "
                "x %d stats) -- the config helper describes the encoder, not "
                "the latent"
                % (config.latent_dim(), len(hc.pooling.stats), d_lat,
                   self.n_levels, l0_dim, len(hc.pooling.stats))))

        # One output per labelled cell when the target carries a category axis.
        # The head emits every cell at once; a single output over a sampled
        # cell would average the categories and report the average.
        _n_cells = 1
        _derive = False
        _cell_w = None
        self.category_names: Tuple[str, ...] = ()
        if bool(getattr(config.data, "target_per_category", False)):
            from ..data.labels import SamplePolicy as _SP
            # Seven category columns. The eighth -- the genome marginal -- is
            # derived inside the head: there is no `mean` experiment to predict
            # and a free column for it lets the head contradict its categories.
            _n_cells = len(_SP().categories)
            # Deliberately off: the mixture a derived mean averages over is
            # genome-specific and an artefact of which experiments exist.
            _derive = False
            _cell_w = None
            _text = _text_embeddings(config, _SP().categories,
                                     self.notes)
            # Column names in the order the head emits them; without them a
            # [B, K] output is K numbers with no way to name the categories.
            self.category_names = tuple(str(c) for c in _SP().categories)
        self.absolute = (AbsoluteStrengthHead(
            n_outputs=_n_cells,
            # A1. With a per-sample target every row carries its own
            # description, so `row_text` is never None and the per-category
            # branch of `forward` is unreachable from the config; the head then
            # skips `lv_layer`/`ls_layer` (3 598 weights with no gradient).
            per_row_only=bool(getattr(
                config.data, "target_per_category", False)),
            derive_marginal=_derive, cell_weights=_cell_w,
            text_emb=(_text if _n_cells > 1 else None),
            in_dim=d_lat, hidden=hc.absolute.hidden, depth=hc.absolute.depth,
            dropout=hc.absolute.dropout,
            context_dim=self._ctx_dim_for(hc.absolute.context_mode),
            context_mode=hc.absolute.context_mode,
            context_center=hc.absolute.context_center,
            beta=hc.absolute.beta, beta_mu=hc.absolute.beta_mu,
            lv_min=hc.absolute.lv_min,
            lv_max=hc.absolute.lv_max, clamp_mode=hc.absolute.clamp_mode,
            predict_sigma=hc.absolute.predict_sigma,
            sigma_weight=hc.absolute.sigma_weight,
            sigma_floor=hc.absolute.sigma_floor,
            tie_sigma_to_var=hc.absolute.tie_sigma_to_var,
            pooling=self.pooling, seed=int(config.training.seed))
            if hc.absolute.enabled else None)
        self.pairwise = (PairwiseRankingHead(
            in_dim=d_lat, hidden=hc.pairwise.hidden, depth=hc.pairwise.depth,
            dropout=hc.pairwise.dropout,
            context_dim=self._ctx_dim_for(hc.pairwise.context_mode),
            context_mode=hc.pairwise.context_mode,
            context_center=hc.pairwise.context_center,
            temperature=hc.pairwise.temperature,
            nonprom_weight=hc.pairwise.nonprom_weight,
            pooling=self.pooling, seed=int(config.training.seed))
            if hc.pairwise.enabled else None)
        # Pre-pooling comparison, built unconditionally whenever the pairwise
        # head exists: at `pair_term_weight: 0.0` the pass is bit for bit the
        # same, but `state_dict` keeps one shape in every regime. Called from
        # the loop, which knows which rows form a pair; `_heads` does not.
        self.pair_compare = (PairComparison(
            channels=tuple(z_dims), hidden=hc.pairwise.pair_term_hidden,
            weight=hc.pairwise.pair_term_weight,
            dropout=hc.pairwise.pair_term_dropout,
            levels=tuple(hc.pairwise.pair_term_levels))
            if hc.pairwise.enabled else None)
        # Categoriser: FiLM plus the regression trunk -> category. It reads the
        # regression head's `h` and its own columns, not the description
        # vector, so without `target_per_category` it is not built. `n_cat`
        # counts categories and `n_columns` the regression axis; they are equal
        # today and separate once that axis moves to description clusters.
        self.category = (CategoryHead(
            in_dim=hc.absolute.hidden,
            n_cat=len(self.category_names) or _n_cells,
            n_columns=_n_cells,
            hidden=hc.category.hidden, dropout=hc.category.dropout,
            use_columns=bool(hc.category.use_columns))
            if (hc.category.enabled and self.absolute is not None
                and _n_cells > 1) else None)
        self.discriminator = (PromoterDiscriminator(
            in_dim=d_lat, hidden=hc.discriminator.hidden,
            depth=hc.discriminator.depth, dropout=hc.discriminator.dropout,
            context_dim=self._ctx_dim_for(hc.discriminator.context_mode),
            context_mode=hc.discriminator.context_mode,
            context_center=hc.discriminator.context_center,
            pos_weight=hc.discriminator.pos_weight,
            label_smoothing=hc.discriminator.label_smoothing,
            negative_kinds=tuple(hc.discriminator.negative_kinds),
            pooling=self.pooling, seed=int(config.training.seed))
            if hc.discriminator.enabled else None)
        if hc.discriminator.enabled and hc.discriminator.context_mode == "none":
            self.notes.append(AssemblyNote(
                "heads.discriminator.context_mode",
                "'none': the promoter/non-promoter head is UNCONDITIONED by "
                "design (realness is not a property of the growth condition)"))

        # C11 fusion. The context is fused into the pooled latent and the heads
        # read the result with no separate context input. `_ctx_dim_for`
        # returns 0 while this is on, so a stale `context_mode: concat` in a
        # config cannot resurrect the bypass this block exists to remove.
        fc = hc.fusion
        # `film_center` picks a class, not a branch in forward: at "none" the
        # same `LatentConditioner` is built with the same arguments at the same
        # point of the RNG stream, so the model is bit for bit the previous one.
        _fkonf = FusionConfig(
            d_channel=int(fc.d_channel),
            d_bottleneck=int(fc.d_bottleneck),
            hidden=int(fc.hidden), dropout=float(fc.dropout),
            film_range=float(fc.film_range),
            init_scale=float(fc.init_scale),
            gate_floor=float(fc.gate_floor),
            active_floor=float(fc.active_floor),
            floor_weight=float(fc.floor_weight),
            # `FusionConfig` is built field by field, so a field not named here
            # keeps its default whatever the YAML says.
            beta_w_skali_latentu=bool(fc.beta_w_skali_latentu))
        _fsrodek = str(getattr(fc, "film_center", "none") or "none")
        if fc.enabled and self.slicer is not None:
            if _fsrodek == "none":
                self.fusion = LatentConditioner(
                    latent_dim=d_lat, ctx_dim=self.d_ctx,
                    token_names=self.context_tokens, config=_fkonf)
            else:
                self.fusion = CentredLatentConditioner(
                    latent_dim=d_lat, ctx_dim=self.d_ctx,
                    token_names=self.context_tokens, config=_fkonf,
                    tryb=_fsrodek)
        else:
            self.fusion = None
        if self.fusion is not None and _fsrodek != "none":
            self.notes.append(AssemblyNote(
                "heads.fusion.film_center",
                "modulation CENTRED across the batch in mode %r: gamma has "
                "mean 1 and beta mean 0 across rows, with no learnt "
                "parameters (two statistics buffers). Measured on the "
                "anchor: after subtracting the identity, the shared "
                "component of gamma had norm 2.93 against its own 1.94, so "
                "about 59 %% of the modulation energy was an affine "
                "constant the head absorbs anyway. NOTE: at head_withhold = "
                "%r this module is called TWICE per pass and both calls "
                "update the buffers"
                % (_fsrodek, tuple(self.head_withhold))))
        if self.fusion is not None:
            self.notes.append(AssemblyNote(
                "C11 latent fusion",
                "%d channels %r -> bottleneck of %d -> FiLM on the latent. "
                "The heads receive ONE vector and no separate context, so "
                "every prediction is a function of the latent and an edit to "
                "the codes reaches all of it; the condition can still move it "
                "on its own, which 50.50 %% of labelled rows require because "
                "they share their 800 bp with another row. Gate floor %.3f, "
                "starvation floor %.4f"
                % (self.n_context_tokens, self.context_tokens,
                   int(fc.d_bottleneck), float(fc.gate_floor),
                   self.fusion.active_floor)))
        elif fc.enabled:
            self.notes.append(AssemblyNote(
                "heads.fusion",
                "context_fuse=%r destroys the per-token structure the fusion "
                "block reads, so it is ABSENT and the heads keep the "
                "concatenated-context bypass" % (context_fuse,)))

        # C10 per-channel heads
        self.decomposition = None
        if hc.decomposition.enabled and self.slicer is not None:
            self.decomposition = DecompositionHeads(
                in_dim=d_lat, slicer=self.slicer,
                hidden=hc.decomposition.hidden,
                depth=hc.decomposition.depth,
                dropout=hc.decomposition.dropout,
                context_mode=hc.decomposition.context_mode,
                context_center=hc.decomposition.context_center,
                targets=tuple(hc.decomposition.targets),
                weights=hc.decomposition.weights(),
                huber_delta=float(hc.decomposition.huber_delta),
                pooling=self.pooling)
            if self.decomposition.skipped:
                self.notes.append(AssemblyNote(
                    "heads.decomposition.targets",
                    "%s requested but their context token is absent from this "
                    "build, so they were dropped rather than silently "
                    "re-pointed at another token"
                    % (", ".join(self.decomposition.skipped),)))
            self.notes.append(AssemblyNote(
                "C10 per-channel heads",
                "%s -- each reads ONLY its own context token, which is what "
                "turns the species and text channels from decoration into the "
                "single route to their target"
                % (", ".join("%s<-%s" % (t, self.decomposition.token_for[t])
                             for t in self.decomposition.targets) or "none",)))

        for key, reason in self.encoder.inert_keys():
            self.notes.append(AssemblyNote("encoder." + key, reason))
        self._check_channel_order()

        # X1: the condition meets the sequence before pooling, behind the
        # quantiser on z[0], so the codes stay a function of the sequence
        # alone. Built at the very end of `__init__`, because every `nn.Linear`
        # draws from the RNG stream and a module inserted earlier shifts the
        # initialisation of everything after it -- "on" would then differ from
        # "off" by the whole draw rather than by one mechanism. The attention
        # output is zero-initialised, so `xattn_glowy: 4` is bit for bit `0` at
        # step zero, which the gate measures rather than assumes.
        self.xattn_local = None
        self.xattn: Optional[CoarseConditionAttention] = None
        _xg = int(getattr(hc.fusion, "xattn_glowy", 0) or 0)
        if _xg > 0:
            if self.fusion is None:
                # Accepting this silently would give parameters without a
                # mechanism: the attention output is added to the latent that
                # enters FiLM, and without fusion that latent goes nowhere.
                raise ValueError(
                    "heads.fusion.xattn_glowy = %d, but no fusion block was "
                    "built (heads.fusion.enabled = %r, context_fuse = %r). "
                    "The condition attention adds its output to the latent "
                    "just before FiLM, so without the fusion there is "
                    "nowhere to add it."
                    % (_xg, bool(hc.fusion.enabled), context_fuse))
            self.xattn = CoarseConditionAttention(
                d_ctx=self.d_ctx, d_mapa=int(self.z_dims[0]),
                latent_dim=d_lat, n_glow=_xg,
                token_names=self.context_tokens)
            self.notes.append(AssemblyNote(
                "heads.fusion.xattn_glowy",
                "X1 ON: %d heads x dk %d; queries from %d condition tokens, "
                "keys and values from the level-0 map (%d channels x %d "
                "positions, stride %d bp). Cost %d parameters (%s). The "
                "output is zero-initialised, so at the start the model is "
                "bit for bit the previous one -- observable ONLY through "
                "`ktx_xa_wyj_norma`"
                % (_xg, CoarseConditionAttention.DK, self.n_context_tokens,
                   int(self.z_dims[0]), int(self.level_lengths[0]),
                   int(self.seq_len // max(int(self.level_lengths[0]), 1)),
                   self.xattn.parameter_count(), self.xattn.rachunek())))
        else:
            self.notes.append(AssemblyNote(
                "heads.fusion.xattn_glowy",
                "0: the condition meets the sequence ONLY as FiLM on the "
                "pooled latent, i.e. after averaging over positions. "
                "Measured consequences: gamma is 87 %% shared between "
                "samples, and the within-orthogroup residual axis stalls "
                "after two epochs (ceiling 0.6950, the model takes 0.29)"))

        # The reversed direction: positions query the metadata. Built after the
        # `xattn_glowy` block, because both routes arrive at the same place
        # (`_with_condition`) and are interchangeable.
        _xl = int(getattr(hc.fusion, "xattn_lokalny", 0) or 0)
        if _xl > 0:
            if not bool(hc.fusion.enabled):
                raise ValueError(
                    "heads.fusion.xattn_lokalny = %d with the fusion "
                    "disabled: the attention output would have nowhere to "
                    "go." % (_xl,))
            self.xattn_local = LocalConditionAttention(
                d_mapa=int(self.z_dims[0]), d_token=int(self.d_ctx),
                d_wyj=int(d_lat), glowy=_xl)
            self.notes.append(AssemblyNote(
                "heads.fusion.xattn_lokalny",
                "ON: %d heads, queries from EVERY POSITION of the z[0] map, "
                "keys and values from %d condition tokens -- the reverse of "
                "xattn_glowy, where the condition received ONE summary "
                "vector per sample. Cost %d parameters. The output is "
                "zero-initialised, so at the start the model is bit for bit "
                "the previous one; waking up shows in ktx_lok_norma. "
                "LOCALITY is measured by ktx_lok_poz_cv: a flat distribution "
                "means the mechanism is global despite the per-position "
                "arithmetic, and that is a falsifier which CAN fail."
                % (_xl, len(self.context_tokens),
                   self.xattn_local.parameter_count())))
        else:
            self.notes.append(AssemblyNote(
                "heads.fusion.xattn_lokalny",
                "0: positions do NOT query the metadata."))


        # Condition cone: the context tokens as geodesic refinements of the
        # species point. Built only when enabled, because a module
        # built-then-ignored would still consume the RNG stream. Measured with
        # the cone wired just after `self.slicer`: the context was bit for bit
        # the same and yet max|d mu| was 8.037e-01 and `encode_codes` moved by
        # 131 indices, because everything built later got different weights.
        self.cone: Optional[ConditionCone] = None
        _stc = config.conditioning
        if bool(getattr(_stc, "cone", False)):
            if self.genome is None:
                raise ValueError(
                    "conditioning.cone is on but no GenomeEmbedding was "
                    "built, so every cone anchor would be the origin and the "
                    "hierarchy would silently lose its middle floor.")
            _kan = tuple(str(x) for x in _stc.cone_channels)
            _missing = [k for k in _kan if k not in self.context_tokens]
            if _missing:
                raise ValueError(
                    "conditioning.cone_channels names %r, which are not "
                    "among the %d context tokens %r. A silently skipped "
                    "channel would do nothing and read in the log exactly "
                    "like a working one."
                    % (_missing, len(self.context_tokens), self.context_tokens))
            self.cone = ConditionCone(
                d_ctx=self.d_ctx, d_m=int(_stc.cone_dim), channels=_kan,
                # The species manifold dimension comes from the module, not the
                # config: `conditioning.genome.dim` is the width of the
                # Euclidean projection `genome.proj`, while the points live in
                # H^3. Passing 32 would build without error and anchor in zeros.
                d_species=int(self.genome.dim),
                aperture=float(_stc.cone_aperture),
                zero_init=bool(_stc.cone_zero_init),
                korzenie=bool(_stc.cone_roots),
                diag_co=int(_stc.cone_diag_every))
            self.notes.append(AssemblyNote(
                "conditioning.cone",
                "ON: %d parameters, channels %r, aperture %.3f, H^%d, %d "
                "roots on the manifold. The codes remain a function of the "
                "sequence alone -- this module works on the CONDITION "
                "TOKENS, i.e. behind the quantiser."
                % (self.cone.parameter_count(), _kan,
                   float(_stc.cone_aperture), int(_stc.cone_dim),
                   len(self.cone.manifold_parameters()))))
        else:
            self.notes.append(AssemblyNote(
                "conditioning.cone",
                "off: the condition tokens are Euclidean, and the model's "
                "only Lorentz point (`genome.points`) leaves the manifold "
                "immediately through `logmap0` and passes through two linear "
                "layers, so curvature takes part in no conditioning "
                "operation at all"))

        # B2: the fine levels sharpen the picture built on L0. Built after the
        # cone, for the same RNG-stream reason.
        self.sharpness: Optional[LevelSharpening] = None
        if bool(getattr(config.model, "sharpening", False)):
            _poz = tuple(int(v) for v in getattr(
                config.model, "sharpen_levels", (1, 2, 3, 4)))
            self.sharpness = LevelSharpening(
                channels=tuple(self.z_dims),
                output=int(self.pooling.out_dim),
                levels=_poz,
                dim=int(getattr(config.model, "sharpen_dim", 24)),
                stats=tuple(config.heads.pooling.stats),
                anchor_frac=float(config.heads.pooling.anchor_frac),
                bands=int(getattr(config.model, "sharpen_bands", 4)))
            self.notes.append(AssemblyNote(
                "model.sharpening",
                "ON: %d parameters, levels %r, dim %d. A correction to "
                "`pooled` computed from fine-level statistics WITHIN A "
                "LEVEL-0 WINDOW (at 800 bp one L0 window is 32 bp, i.e. "
                "2/4/8/16 cells of L1/L2/L3/L4). The output is ZEROED, so "
                "the model starts bit for bit as it would without the "
                "module, yet `d(loss)/d(W_out)` is non-zero -- there is no "
                "multiplicative gate here, which together with a zero value "
                "would leave the branch dead from the first step."
                % (sum(p.numel() for p in self.sharpness.parameters()),
                   list(_poz),
                   int(getattr(config.model, "sharpen_dim", 32)))))
        else:
            self.notes.append(AssemblyNote(
                "model.sharpening",
                "off: the heads see levels L1..L4 ONLY through three "
                "statistics taken over the whole length of each level. "
                "Measured: MI against strength is 0.26 / 0.14 / 0.09 / "
                "0.05 % of the ceiling on L1..L4 against 4.24 % on L0, and "
                "the `absolute` gradient energy in the pooling subspace "
                "sits at its ceiling of 0.8"))

        # B3: positional readout of level 0, built after `self.sharpness` for
        # the same RNG-stream reason.
        self.bands_l0: Optional[BandsL0] = None
        if bool(getattr(config.model, "bands_l0", False)):
            _p = int(getattr(config.model, "bands_l0_p", 8) or 8)
            self.bands_l0 = BandsL0(
                channels=tuple(self.z_dims),
                output=int(self.pooling.out_dim),
                bands=_p,
                anchor_frac=float(config.heads.pooling.anchor_frac))
            self.notes.append(AssemblyNote(
                "model.bands_l0",
                "ON: %d parameters, P = %d bands over the level-0 windows "
                "(%d channels). A correction to `pooled` through a ZEROED "
                "linear layer, so the model starts bit for bit as it would "
                "without the module while `d(loss)/d(W_out)` is non-zero -- "
                "there is no multiplicative gate here. Band boundaries are "
                "fractional, so the module does not know the sequence "
                "length. The expected identity `band_P == anchor` is FALSE "
                "at anchor_frac 0.125 with 25 windows (4 cells against 3); "
                "the positive control for the grid is `kontrola_siatki()`: "
                "the width-weighted mean of the bands == the `L0/mean` "
                "slice."
                % (sum(p.numel() for p in self.bands_l0.parameters()),
                   _p, int(self.z_dims[0]))))
        else:
            self.notes.append(AssemblyNote(
                "model.bands_l0",
                "off: level 0 reaches the heads only through `mean` and `sd` "
                "(permutation-invariant by definition) and `anchor` (TWO "
                "regions). `sharpening` does not read level 0 by "
                "construction, so 4.24-6.83 % of the MI ceiling sits behind "
                "a readout with a resolution of two regions"))

        # Content-weighted readout, wired after `bands_l0` so that all three
        # additions to `pooled` share one `reference` in the `*_udzial`
        # observables. What it does not promise: the positional axis is worth
        # dPCC(perm_pos) = -0.04205 +- 0.00790 against -0.52165 for a global
        # shuffle, i.e. 8.1 % of what the latent contributes, while the A/B
        # resolution on one seed is 2 sigma ~ 0.016. It can show harm, or that
        # the mechanism is alive, but not that it helps.
        self.readout: Optional[AttentionReadout] = None
        if bool(getattr(config.model, "attention_readout", False)):
            _poz = tuple(int(x) for x in (
                getattr(config.model, "readout_levels", None) or (0, 1, 2)))
            self.readout = AttentionReadout(
                channels=tuple(self.z_dims),
                output=int(self.pooling.out_dim),
                levels=_poz,
                dim=int(getattr(config.model, "readout_dim", 64) or 64),
                zapytania=int(
                    getattr(config.model, "readout_queries", 8) or 8))
            self.notes.append(AssemblyNote(
                "model.attention_readout",
                "ON: %d parameters, levels %r, dim %d, %d queries. Position "
                "weights are a FUNCTION OF CONTENT (softmax over "
                "|h - mean_l(h)|), unlike `pooling`, which yields only FIXED "
                "functionals (mean and sd are permutation-invariant BY "
                "DEFINITION). The centring is PER SEQUENCE, with no batch "
                "statistic, so the prediction stays a function of (codes, "
                "condition) and the diffusion contract holds. The output is "
                "ZEROED and there is no multiplicative gate. MECHANISM "
                "FALSIFIER: `odcz_skup_l*` > 0 and rising -- under uniform "
                "attention the output is EXACTLY zero (an identity, "
                "`kontrola()` K1), so `odcz_rms` > 0 proves the positions "
                "are weighted non-uniformly."
                % (sum(p.numel() for p in self.readout.parameters()),
                   _poz, int(getattr(config.model, "readout_dim", 64) or 64),
                   int(getattr(config.model, "readout_queries", 8) or 8))))
        else:
            self.notes.append(AssemblyNote(
                "model.attention_readout",
                "off: every path from the latent to the absolute head "
                "weights positions by a CONSTANT (mean, sd, anchor, bands), "
                "so the 8.1 % of the latent's contribution that lies in "
                "POSITION is reachable only through `sharpness` and "
                "`bands_l0`, which also use fixed slices"))

    # Internals
    def _ctx_dim_for(self, mode: str) -> int:
        """0 when a head declares no conditioning.

        A non-zero `context_dim` on a head whose mode is "none" builds a
        `ConditionedTrunk` that stores the width and never reads it.
        """
        # Read from the config, not from `self.fusion`: the heads are built
        # before the fusion block, so a check on the module would still be
        # None here and every head would expect a context it never receives.
        if self.cfg.heads.fusion.enabled and \
                self.context_fuse == "flatten":
            return 0
        return 0 if mode == "none" else int(self.context_dim)

    def _check_channel_order(self) -> None:
        """G6, asserted rather than eyeballed."""
        if NUCLEOTIDE_ORDER != "ACGT":
            raise RuntimeError(
                "hyppe.data.sequences.NUCLEOTIDE_ORDER is %r. Every module "
                "here assumes ACGT, in which the complement is a REVERSAL of "
                "the channel axis. The sibling track's A,C,T,G convention "
                "(complement = (i+2) %% 4) does not hold in this repo."
                % (NUCLEOTIDE_ORDER,))

    # Occupancy
    def motif_occupancy(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """[B, 4, L] -> [B, M, L] normalised occupancy, or None."""
        if self.occupancy is None:
            return None
        return self.occupancy(x)

    # Context
    def context(self, batch: int,
                text_emb: Optional[torch.Tensor] = None,
                species_id: Optional[torch.Tensor] = None,
                flank_l: Optional[torch.Tensor] = None,
                flank_r: Optional[torch.Tensor] = None,
                flank_ok: Optional[torch.Tensor] = None,
                text_ok: Optional[torch.Tensor] = None,
                gene_numeric: Optional[torch.Tensor] = None,
                gene_identity: Optional[torch.Tensor] = None,
                gene_text: Optional[torch.Tensor] = None,
                gene_ok: Optional[torch.Tensor] = None,
                pair_group: Optional[torch.Tensor] = None):
        """Build the conditioning tokens and the flattened head context.

        `pair_group` shares every dropout draw between rows carrying the same
        id. The siamese head scores a pair as `s(z_A, C)` and `s(z_B, C)`: one
        condition, two promoters. A per-row draw hands the two rows of a pair
        different C with probability 2p(1-p) = 18 % at p = 0.1, measured as
        17.2 % broken pairs over 400 forwards.
        """
        species_tangent = None
        if self.genome is not None and species_id is not None:
            species_tangent = self.genome(species_id)
        out = self.condition(
            text_emb=text_emb,
            species_id=(species_id if self.genome is None else None),
            species_tangent=species_tangent,
            flank_l=flank_l, flank_r=flank_r, flank_ok=flank_ok,
            text_ok=text_ok,
            gene_numeric=gene_numeric, gene_identity=gene_identity,
            gene_text=gene_text, gene_ok=gene_ok,
            batch_size=batch, group=pair_group)
        # Cone, before the flank dropout, so that `cond_tokens` keeps meaning
        # "tokens after the condition encoder, before the dropout".
        _st_diag: Dict[str, float] = {}
        if self.cone is not None:
            _sty = None
            if self.genome is not None and species_id is not None:
                # `genome.tangent()` is `logmap0(points)`, [n, 3] float64: the
                # hyperboloid point read as a tangent at the origin. Not
                # `self.genome(species_id)`, which has already passed through
                # `proj` and is a Euclidean feature with no curvature in it.
                _idx = species_id.reshape(-1).long().to(
                    self.genome.points.device)
                _sty = self.genome.tangent()[_idx]
            out = dataclasses.replace(
                out, tokens=self.cone(out.tokens, self.context_tokens,
                                        _sty))
            _st_diag = self.cone.report()
            # The `cond_ktx` log line takes its numbers only from
            # `condition._diag`, which `ConditionEncoder` overwrites on every
            # pass, so appending after the call is the only way the cone's
            # numbers reach the log at all.
            if _st_diag:
                _cd = getattr(self.condition, "_diag", None)
                if isinstance(_cd, dict):
                    _cd.update(_st_diag)
        tokens, drop_frac = self._flank_dropout(out.tokens, pair_group)
        ctx = (tokens.reshape(tokens.shape[0], -1)
               if self.context_fuse == "flatten" else tokens.mean(dim=1))
        diag = dict(out.diagnostics)
        diag["ktx_flanka_dropout"] = float(drop_frac)
        diag.update(_st_diag)
        return ctx, out, diag

    def _flank_dropout(self, tokens: torch.Tensor,
                       group: Optional[torch.Tensor]
                       ) -> Tuple[torch.Tensor, float]:
        """Replace BOTH flank tokens by the learned unknown token, per sample.

        Both at once because the deployment case is a plasmid, whose left and
        right flank are simultaneously out of distribution (measured 5.13 sd
        outside the genomic flank distribution); dropping one side at a time
        would train a token for an event that never occurs.

        The blend is exact: `ConditionEncoder` computes
        `rms(keep*x + (1-keep)*null)` and `keep` is binary here, so
        `keep*rms(x) + (1-keep)*rms(null)` is the same tensor row by row.
        `check_flank_unknown_equivalence` measures that instead of asserting it.
        """
        p = self.flank_drop_prob
        if not self.training or p <= 0.0:
            return tokens, 0.0
        B = tokens.shape[0]
        device, dtype = tokens.device, tokens.dtype
        if group is None:
            keep = (torch.rand(B, device=device) >= p)
        else:
            g = group.to(device=device).reshape(-1).long()
            uniq, inv = torch.unique(g, return_inverse=True)
            keep = (torch.rand(uniq.numel(), device=device) >= p)[inv]
        k = keep.to(dtype).reshape(B, 1)
        null = _rms_normalise(
            self.condition.unknown_token_flank.to(device=device, dtype=dtype)
            .unsqueeze(0).expand(B, -1))
        out = tokens.clone()
        out[:, 2] = k * tokens[:, 2] + (1.0 - k) * null
        out[:, 3] = k * tokens[:, 3] + (1.0 - k) * null
        return out, float(1.0 - k.mean())

    def _restrict_tokens(self, tokens: torch.Tensor,
                         zdejmij: Sequence[str]) -> torch.Tensor:
        """Copy of the token stack with the named channels replaced by UNKNOWN.

        The absolute head learns a VALUE and has to see everything; the
        pairwise head learns which promoter is stronger and sees only what is a
        property of the sequence in its surroundings -- the flanks and the gene.
        A pair is drawn inside one (genome, category) cell, so the species and
        text tokens are constant within it: they cannot change the sign of the
        difference, only shift both scores equally, which makes them a route to
        fitting the cell rather than the task. Measured: substituting the
        species channel moves dPCC by -0.0026 at the peak epoch and +0.0046 at
        epoch 30.

        Substituted by UNKNOWN, not by zero: `text_proj` has a bias, so zero
        input is a learned constant reported as the channel's share rather than
        an absent channel.
        """
        if not zdejmij:
            return tokens
        nazwy = list(self.context_tokens)
        device, dtype = tokens.device, tokens.dtype
        B = tokens.shape[0]
        out = tokens.clone()
        for nm in zdejmij:
            if nm not in nazwy:
                raise ValueError(
                    "channel %r is not one of the %d context tokens %r; a "
                    "silently ignored name would restrict nothing and read "
                    "in the log as if it had" % (nm, len(nazwy), nazwy))
            i = nazwy.index(nm)
            if nm == "species":
                u = self.condition.unknown_token_species
            elif nm == "text":
                u = self.condition.unknown_token_text
            elif nm in ("flank_l", "flank_r"):
                u = self.condition.unknown_token_flank
            else:
                raise ValueError(
                    "channel %r has no unknown token in ConditionEncoder, so "
                    "it cannot be withheld without inventing one. Withholding "
                    "it as zeros would be a learned constant, not an absent "
                    "channel." % (nm,))
            out[:, i] = _rms_normalise(
                u.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1))
        return out

    @torch.no_grad()
    def check_restrict_equivalence(self, batch: int = 4) -> Dict[str, float]:
        """Falsifiers for channel routing, measured on tensors.

        R1 withholding an empty list returns the same tensor (max|delta| = 0);
        R2 a withheld channel is constant across rows (sd over the batch = 0);
        R3 positive control -- a channel that was NOT withheld is not constant,
        without which R2 passes trivially on input that was constant anyway.
        """
        t = torch.randn(batch, self.n_context_tokens, self.d_ctx)
        r0 = self._restrict_tokens(t, ())
        r1 = self._restrict_tokens(t, ("species", "text"))
        nazwy = list(self.context_tokens)
        i_sp = nazwy.index("species")
        i_gn = nazwy.index("gene") if "gene" in nazwy else 0
        return {
            "r1_pusta_lista": float((r0 - t).abs().max()),
            "r2_zdjety_staly": float(r1[:, i_sp].std(dim=0).max()),
            "r3_niezdjety_zmienny": float(r1[:, i_gn].std(dim=0).max()),
        }

    @torch.no_grad()
    def check_flank_unknown_equivalence(self, batch: int = 4,
                                        seed: int = 0) -> float:
        """max |blended tokens - tokens built with flank=None|. Must be 0.

        A falsifier satisfiable by construction: the two routes are
        arithmetically the same expression, so anything but an exact zero means
        the blend stopped matching the module.
        """
        gen = torch.Generator().manual_seed(int(seed))
        d_dna = int(self.condition.cfg.cond_d_dna)
        fl = torch.randn(batch, d_dna, generator=gen)
        fr = torch.randn(batch, d_dna, generator=gen)
        was = self.training
        self.eval()
        ref = self.condition(flank_l=None, flank_r=None,
                             batch_size=batch).tokens
        live = self.condition(flank_l=fl, flank_r=fr,
                              batch_size=batch).tokens
        self.train(was)
        keep = torch.zeros(batch, 1, dtype=live.dtype)
        null = _rms_normalise(
            self.condition.unknown_token_flank.detach()
            .unsqueeze(0).expand(batch, -1))
        blended = live.clone()
        blended[:, 2] = keep * live[:, 2] + (1.0 - keep) * null
        blended[:, 3] = keep * live[:, 3] + (1.0 - keep) * null
        return float((blended[:, 2:] - ref[:, 2:]).abs().max())

    # Encode
    def encode(self, x: torch.Tensor,
               occupancy: Optional[torch.Tensor] = None) -> LadderOutput:
        """[B, 4, L] -> LadderOutput (z, indices, points, stats, losses)."""
        if occupancy is None:
            occupancy = self.motif_occupancy(x)
        maps = self.encoder(x, occupancy=occupancy, context=None)
        maps = [proj(m) for proj, m in zip(self.level_proj, maps)]
        return self.ladder(maps)

    def encode_codes(self, x: torch.Tensor,
                     occupancy: Optional[torch.Tensor] = None
                     ) -> List[torch.Tensor]:
        """C8: [B, 4, L] -> list of index tensors [B, L_k], coarsest first."""
        return [i.detach() for i in self.encode(x, occupancy).indices]

    def runaway_report(self) -> Dict[str, float]:
        """Scale of the encoder map at the one route onto the hyperboloid.

        `enc_norm_*` and `enc_radius_raw_*` are taken before the radius clamp
        and are the only place a runaway is visible: with `l0_max_radius`
        active the post-clamp radius is bounded by construction and would read
        a healthy 8.0 for ever. These have reached 1.846e18 and 41.33 while
        nothing in the epoch line moved.
        """
        return self.ladder.runaway_report()

    def latent_from_codes(self, indices: Sequence[torch.Tensor]):
        """C8: indices -> (z per level, points per level). No encoder tensor."""
        return self.ladder.decode_indices(indices)

    def decode_codes(self, indices: Sequence[torch.Tensor]) -> torch.Tensor:
        """C8: indices -> base logits [B, 4, seq_len]. The diffusion contract."""
        z, _points = self.latent_from_codes(indices)
        return self.decoder(z)

    def decode_to_onehot(self, indices: Sequence[torch.Tensor]
                         ) -> torch.Tensor:
        """Hard one-hot from the decoded logits, in the ACGT channel order."""
        logits = self.decode_codes(indices)
        return torch.nn.functional.one_hot(
            logits.argmax(dim=1), num_classes=logits.shape[1]
        ).permute(0, 2, 1).to(logits.dtype)

    def species_id_is_live(self) -> bool:
        """True only where `species_id` can change an output of `_heads`.

        Exactly one expression reads it: `AbsoluteStrengthHead.forward` selects
        a row of `cell_weights` inside the `derive_marginal` branch. With
        `derive_marginal` off -- which is what this file builds -- the argument
        is accepted and discarded, so a caller is refused rather than quietly
        served a species-free number.
        """
        head = getattr(self, "absolute", None)
        if head is None or not bool(getattr(head, "derive_marginal", False)):
            return False
        w = getattr(head, "cell_weights", None)
        # A single shared row makes the index_select a no-op as surely as a
        # missing branch does: `forward` falls through to `w.mean(dim=0)`.
        return w is not None and int(w.shape[0]) > 1

    def predict_from_codes(self, indices: Sequence[torch.Tensor],
                           context: Optional[torch.Tensor] = None,
                           species_id: Optional[torch.Tensor] = None,
                           row_text: Optional[torch.Tensor] = None
                           ) -> Dict[str, object]:
        """C8: prediction as a function of the CODES, not of a sequence.

        Not `self(decode_to_onehot(indices))`: the encoder would compute its
        own codes from the decoded sequence. For codes that came from a
        sequence those agree (the bijection holds), but for generated codes --
        the only case that occurs in sampling -- they need not, and the fixed
        point is not the identity. The code state is pinned, not recomputed.
        """
        if species_id is not None and not self.species_id_is_live():
            _dm = (None if getattr(self, "absolute", None) is None
                   else bool(getattr(self.absolute, "derive_marginal", False)))
            raise ValueError(
                "predict_from_codes(species_id=...) cannot act on this "
                "build, so it is refused rather than ignored. The argument "
                "reaches exactly one expression -- the `derive_marginal` "
                "branch of AbsoluteStrengthHead.forward -- and this model "
                "was assembled with derive_marginal=%r. MEASURED here: "
                "species 0 against species 1 at a fixed context differ by "
                "0.000000e+00. Put the species in the CONTEXT instead:\n"
                "    ctx, _out, _diag = model.context(batch, "
                "species_id=sp)\n"
                "    model.predict_from_codes(indices, context=ctx)\n"
                "A silent no-op and a working argument must not be "
                "indistinguishable." % (_dm,))
        z, points = self.latent_from_codes(indices)
        return self._heads(z, context, species_id=species_id,
                           row_text=row_text)

    # Heads
    def _heads(self, z: Sequence[torch.Tensor],
               context: Optional[torch.Tensor],
               species_id: Optional[torch.Tensor] = None,
               row_text: Optional[torch.Tensor] = None
               ) -> Dict[str, object]:
        """Pool, fuse, then predict. With fusion on, the heads see ONE vector.

        The context is not handed to the heads alongside the latent; it is
        fused into it and the heads are called with `context=None`. That is the
        difference between conditioning a representation and giving the
        prediction a second, parallel input -- and only the first is reachable
        by an edit to the codes, which is what the design loop does.
        """
        pooled = self.pooling(list(z))
        # Reference for the `*_udzial` observables: `pooled` before all three
        # additions. A per-module denominator would make F5 (separating the
        # contributions) unreadable from the log.
        _odn = pooled
        if self.sharpness is not None:
            # Here rather than in `forward`: both routes -- `forward` and
            # `predict_from_codes` -- go through `_heads`, so the prediction
            # stays a function of (codes, condition) by construction. The
            # module reads `z`, the latent after the quantiser, not the
            # encoder map. Same for the two blocks below.
            pooled = pooled + self.sharpness(list(z), reference=_odn)
        if self.bands_l0 is not None:
            pooled = pooled + self.bands_l0(list(z), reference=_odn)
        if self.readout is not None:
            # The readout centres over positions within the sequence, so no
            # batch statistic enters here either.
            pooled = pooled + self.readout(list(z), reference=_odn)
        out: Dict[str, object] = {"pooled": pooled}
        feats = pooled
        head_ctx = context
        if self.fusion is not None and context is not None:
            tokens = context.reshape(context.shape[0], self.n_context_tokens,
                                     self.d_ctx)
            # `z[0]` is the coarsest map (25 positions at 800 bp), the only
            # level a probe sees anything on (L0 0.5853 against L3 0.1919 and
            # L4 0.2948). The result is added to the latent before FiLM. `z` is
            # the same object in `forward` and in `predict_from_codes`, so the
            # diffusion contract holds -- the gate measures 0.000e+00.
            lat_pelny = self._with_condition(pooled, tokens, z)
            if lat_pelny is not pooled:
                # Exported because `pooled` is no longer the whole input to the
                # fusion: `loop.py` computes `fusion.channel_effect` from the
                # pair (`pooled`, `cond_tokens`), and with attention on, that
                # number describes only part of the condition's route.
                out["latent_warunkowany"] = lat_pelny
            fused = self.fusion(lat_pelny, tokens)
            feats = fused.latent
            head_ctx = None
            out["fusion"] = fused
            out["condition_code"] = fused.code
        out["features"] = feats
        if self.absolute is not None:
            out["absolute"] = self.absolute(feats, self._ctx(
                head_ctx, self.cfg.heads.absolute.context_mode),
                species_id=species_id, row_text=row_text)
        if self.category is not None and out.get("absolute") is not None:
            _ao = out["absolute"]
            # The column profile, not `mu`: in the per-sample regime `mu` is
            # [B] and carries no profile. The per-category read-out exists in
            # both regimes -- the same `text_proj` with category prototypes
            # instead of the row's description, one matmul and no parameters.
            _mu = _ao.mu if _ao.mu.dim() == 2 else None
            if _mu is None and self.category.use_columns:
                _mu = self.absolute.columns(_ao.trunk)
            out["cat_head"] = self.category(_ao.trunk, _mu)
        # Narrower-condition branch. The pairwise head -- and the
        # discriminator, since realness is not a property of the strain -- gets
        # a latent conditioned only on channels that belong to the sequence in
        # its surroundings. With `withhold` empty this is `feats` itself.
        feats_waski = feats
        if (self.fusion is not None and context is not None
                and self.head_withhold):
            _tok = context.reshape(context.shape[0], self.n_context_tokens,
                                   self.d_ctx)
            # Second fusion call on a restricted token stack, a different input
            # distribution, so it must not enter FiLM's running statistics:
            # without this gate `film_steps` rises 0 -> 2 per pass.
            _poprz = getattr(self.fusion, "aktualizuj_statystyki", None)
            if _poprz is not None:
                self.fusion.aktualizuj_statystyki = False
            # The condition attention is restricted too: on the full token
            # stack the species and text channels would return to the pairwise
            # head by a side route and `head_withhold` would have no effect.
            # Its diagnostics are suspended for the same reason as FiLM's.
            _tokw = self._restrict_tokens(_tok, self.head_withhold)
            _xz = getattr(self.xattn, "zbieraj", None)
            if _xz is not None:
                self.xattn.zbieraj = False
            # Same for the local direction: `_with_condition` calls
            # `self.xattn_local` in both branches, so the narrow branch would
            # overwrite the `_diag` written by the full one.
            _xlz = getattr(getattr(self, "xattn_local", None), "zbieraj", None)
            if _xlz is not None:
                self.xattn_local.zbieraj = False
            try:
                _fw = self.fusion(self._with_condition(pooled, _tokw, z), _tokw)
            finally:
                if _poprz is not None:
                    self.fusion.aktualizuj_statystyki = _poprz
                if _xz is not None and self.xattn is not None:
                    self.xattn.zbieraj = _xz
                if (_xlz is not None
                        and getattr(self, "xattn_local", None) is not None):
                    self.xattn_local.zbieraj = _xlz
            feats_waski = _fw.latent
            out["fusion_narrow"] = _fw
        if self.pairwise is not None:
            out["pair_score"] = self.pairwise.score(feats_waski, self._ctx(
                head_ctx, self.cfg.heads.pairwise.context_mode))
        if self.discriminator is not None:
            out["discriminator"] = self.discriminator(feats_waski, self._ctx(
                head_ctx, self.cfg.heads.discriminator.context_mode))
        if self.decomposition is not None and context is not None:
            # These read the RAW per-token context on purpose: each exists to
            # give ONE channel a target only that channel can reach, and the
            # fused latent has already mixed them.
            out["decomposition"] = self.decomposition(pooled, context)
        if self.xattn is not None:
            _xd = dict(getattr(self.xattn, "_diag", None) or {})
            if _xd:
                out["xattn_diag"] = _xd
                # Route to the log: `loop.py` prints every `ktx_` key of
                # `condition._diag`, which `condition.forward` rebuilds before
                # `_heads`, so nothing here survives as a stale number. Without
                # this the module produces no number at all in a run, and a
                # starved branch looks exactly like a disabled one.
                self.condition._diag.update(_xd)
        if getattr(self, "xattn_local", None) is not None:
            # Twin of the block above. That one reaches only `self.xattn`,
            # which is None at `xattn_glowy: 0`, so `ktx_lok_*` was computed
            # into its own `_diag` and read by nobody.
            _xld = dict(getattr(self.xattn_local, "_diag", None) or {})
            if _xld:
                out["xattn_lok_diag"] = _xld
                self.condition._diag.update(_xld)
        return out

    def _with_condition(self, pooled: torch.Tensor, tokens: torch.Tensor,
                     z: Sequence[torch.Tensor]) -> torch.Tensor:
        """`pooled` plus the condition cross-attention on the level-0 map.

        Separate because it is called twice, for the full branch and the narrow
        one, and a drift between two copies of this arithmetic would be silent:
        both return the same shape and the difference would surface only as a
        worse pairwise score. With `xattn = None` it returns `pooled` itself,
        not a copy.
        """
        _dod = None
        if getattr(self, "xattn_local", None) is not None:
            _dod = self.xattn_local(z[0], tokens)
        if self.xattn is None:
            return pooled if _dod is None else pooled + _dod
            return pooled
        _y = pooled + self.xattn(tokens, z[0], reference=pooled)
        return _y if _dod is None else _y + _dod

    @staticmethod
    def _ctx(context: Optional[torch.Tensor], mode: str
             ) -> Optional[torch.Tensor]:
        return None if mode == "none" else context

    # Forward
    def forward(self, x: torch.Tensor,
                occupancy: Optional[torch.Tensor] = None,
                text_emb: Optional[torch.Tensor] = None,
                species_id: Optional[torch.Tensor] = None,
                flank_l: Optional[torch.Tensor] = None,
                flank_r: Optional[torch.Tensor] = None,
                flank_ok: Optional[torch.Tensor] = None,
                text_ok: Optional[torch.Tensor] = None,
                gene_numeric: Optional[torch.Tensor] = None,
                gene_identity: Optional[torch.Tensor] = None,
                gene_text: Optional[torch.Tensor] = None,
                gene_ok: Optional[torch.Tensor] = None,
                pair_group: Optional[torch.Tensor] = None,
                row_text: Optional[torch.Tensor] = None,
                reconstruct: bool = True) -> Dict[str, object]:
        """One pass. Returns everything the losses need, nothing pre-reduced.

        The returned dict carries `z` / `indices` / `points` (the latent per
        level, coarsest first), `recon_logits` [K*B, 4, seq_len] over the mask
        views with `view_names` / `view_masks` / `view_weights`, the head
        outputs `absolute` / `pair_score` / `discriminator` / `cat_logits`,
        `context` / `cond_tokens`, `losses` (subordinate terms, each a tensor)
        and `stats` (merged reports, floats only).

        `row_text` [B, d_lm] puts the absolute head in the per-sample regime:
        one value per row, its column generated from that row's own experiment
        description, so the target can be the COMBAT value of one (orthogroup,
        sample) pair. Left None the head returns [B, K] over the category
        prototypes. It does not reach the condition tokens -- the description
        enters the conditioning through `text_emb` -- so passing one without
        the other conditions the latent on nothing and reads a column no
        channel produced.
        """
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError("x must be [B, 4, L] in ACGT order, got %s"
                             % (tuple(x.shape),))
        B = int(x.shape[0])
        if occupancy is None:
            occupancy = self.motif_occupancy(x)
        ladder_out = self.encode(x, occupancy)
        ctx, cond_out, cond_diag = self.context(
            B, text_emb=text_emb, species_id=species_id,
            flank_l=flank_l, flank_r=flank_r, flank_ok=flank_ok,
            text_ok=text_ok,
            gene_numeric=gene_numeric, gene_identity=gene_identity,
            gene_text=gene_text, gene_ok=gene_ok,
            pair_group=pair_group)

        out: Dict[str, object] = {
            "z": ladder_out.z,
            "indices": ladder_out.indices,
            "points": ladder_out.points,
            "total_bits": ladder_out.total_bits,
            "context": ctx,
            "cond_tokens": cond_out.tokens,
            "cat_logits": cond_out.cat_logits,
            # [B] 1.0 = the description token reached this row's conditioning.
            # The categoriser trains on rows with 0.0; without this mask it
            # would predict the category from the description that encodes it.
            "text_used": cond_out.text_used,
            "occupancy": occupancy,
        }
        out.update(self._heads(ladder_out.z, ctx,
                               species_id=species_id, row_text=row_text))

        losses: Dict[str, torch.Tensor] = {}
        if ladder_out.loss is not None:
            losses["commitment"] = ladder_out.loss
        if self.occupancy is not None and self.occupancy.de_novo is not None:
            # Second scan of the de-novo bank: the penalties need the raw
            # activation and `MotifOccupancy` returns the normalised
            # concatenation. 32 filters over 800 bp is the cheap half of the
            # occupancy, so this is paid deliberately.
            act = self.occupancy.de_novo(x)
            pen = self.occupancy.de_novo.penalties(
                act, frozen=self.occupancy.frozen)
            losses["motif"] = pen["total"]
            out["motif_penalties"] = {k: float(v.detach())
                                      for k, v in pen.items()}
        if self.genome is not None:
            losses["tree_prior"] = self.genome.tree_prior_loss()

        stats: Dict[str, float] = {}
        stats.update({k: v for k, v in ladder_out.stats.items()
                      if isinstance(v, (int, float))})
        stats.update(cond_diag)
        # `cond_diag` is built in `self.context(...)`, before `_heads`, so the
        # condition-attention numbers cannot already be in it.
        _xd = out.get("xattn_diag")
        if isinstance(_xd, dict):
            stats.update(_xd)
        # Same for the local direction; without it the numbers reach
        # `condition._diag` but not `out["stats"]`.
        _xld = out.get("xattn_lok_diag")
        if isinstance(_xld, dict):
            stats.update(_xld)
        if self.sharpness is not None:
            # Without these three lines `ostr_*`, `pasma_*` and `odcz_*` stay
            # in each module's own `_diag` and reach nothing.
            stats.update(self.sharpness.report())
        if self.bands_l0 is not None:
            stats.update(self.bands_l0.report())
        if self.readout is not None:
            # `odcz_skup_l*` is the only falsifier of the readout: without it a
            # run with the module is indistinguishable from one whose branch is
            # frozen at zero.
            stats.update(self.readout.report())
        _gr = getattr(self.encoder, "occ_gate_report", None)
        if callable(_gr):
            # The gate lives in the encoder while `stats` is built here.
            stats.update(_gr())

        if reconstruct:
            views = self.mask.views(ladder_out.z)
            logits = self.decoder(views.views)
            out["recon_logits"] = logits
            out["view_names"] = views.names
            out["view_masks"] = views.masks
            out["view_weights"] = views.weights
            out["recon_logits_full"] = logits.chunk(self.mask.n_views,
                                                    dim=0)[0]
            stats.update(self.decoder.report())

        out["losses"] = losses
        out["stats"] = stats
        return out

    def reconstruction_loss(self, out: Dict[str, object],
                            target_idx: torch.Tensor,
                            per_view=None):
        """Weighted CE over the views -> one term, one subordinate player."""
        return self.mask.reconstruction_loss(
            out["recon_logits"], target_idx, out["view_masks"],
            per_view=per_view)

    # C9 bookkeeping
    def nash_players(self) -> Tuple[str, ...]:
        """Heads that are equal players in the bargain.

        Reconstruction is not here. A saturated task takes the largest share of
        a Nash bargain precisely because its gradient is smallest: measured in
        the sibling track, reconstruction at 0.9998 took alpha 2.31 of 3.0 and
        squeezed the real targets to 0.39 / 0.31. It enters through
        `subordinate_terms` with its declared weight instead.
        """
        return tuple(n for n in self.cfg.training.nash.players
                     if getattr(self, n, None) is not None)

    def subordinate_terms(self) -> Dict[str, float]:
        """Weight of every non-player term, as it reaches the loss."""
        eff = self.cfg.training.effective_recon_coefficients()
        out = {"recon[%s]" % k: v for k, v in eff.items()}
        out["commitment"] = float(self.cfg.training.commitment_weight)
        out["motif"] = 1.0
        if self.genome is not None:
            out["tree_prior"] = float(self.genome.tree_prior)
        return out

    def param_groups(self, lr: float = 3e-4, weight_decay: float = 0.01
                     ) -> List[Dict[str, object]]:
        """manifold / no-decay / decayed, by identity not by name substring.

        A name-substring exemption list has already lost a tensor in this
        project (`maska_wektor` exempt, `wektor_maski` not). The manifold set
        comes from the modules that own the points.
        """
        manifold = list(self.ladder.manifold_parameters())
        if self.genome is not None:
            manifold += list(self.genome.manifold_parameters())
        if getattr(self, "cone", None) is not None:
            # By identity, like ladder and genome. `roots_manifold` has ndim 2,
            # so without this line it lands in `decayed` at weight_decay 0.05
            # -- a factor 1.8e-5 over 34 epochs on hyperboloid points, which
            # squeezes the root to the origin with no trace in the loss.
            manifold += list(self.cone.manifold_parameters())
        mid = {id(p) for p in manifold}
        no_decay, decayed = [], []
        patterns = tuple(self.cfg.training.no_decay_patterns)
        for name, p in self.named_parameters():
            if id(p) in mid or not p.requires_grad:
                continue
            if p.ndim <= 1 or any(s in name for s in patterns):
                no_decay.append(p)
            else:
                decayed.append(p)
        return [
            {"params": manifold, "lr": lr, "weight_decay": 0.0,
             "manifold": True, "name": "manifold"},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0,
             "name": "no_decay"},
            {"params": decayed, "lr": lr, "weight_decay": float(weight_decay),
             "name": "decayed"},
        ]

    # Reports
    def parameter_report(self) -> Dict[str, int]:
        parts = {
            "occupancy": self.occupancy, "encoder": self.encoder,
            "level_proj": self.level_proj, "ladder": self.ladder,
            "decoder": self.decoder, "mask": self.mask,
            "genome": self.genome, "condition": self.condition,
            "absolute": self.absolute, "pairwise": self.pairwise,
            "discriminator": self.discriminator,
            "cone": getattr(self, "cone", None),
            "sharpening": getattr(self, "sharpness", None),
            "bands_l0": getattr(self, "bands_l0", None),
        }
        out: Dict[str, int] = {}
        for name, mod in parts.items():
            if mod is None:
                out[name] = 0
                continue
            out[name] = sum(p.numel() for p in mod.parameters()
                            if p.requires_grad)
        out["total"] = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        out["frozen"] = sum(p.numel() for p in self.parameters()
                            if not p.requires_grad)
        return out

    def bit_budget(self) -> Dict[str, object]:
        return self.cfg.model.bit_budget(self.seq_len)

    def header(self) -> str:
        p = self.parameter_report()
        b = self.bit_budget()
        return ("HyppeModel | %d bp | %d levels %r | latent %d | context %d "
                "| occupancy %d | %d params | code budget %.0f bits vs %.1f "
                "measured (%.2fx)"
                % (self.seq_len, self.n_levels, self.level_lengths,
                   self.latent_dim, self.context_dim, self.occupancy_channels,
                   p["total"], b["total_bits"], b["sequence_bits"],
                   b["reserve"]))

    def inert_keys(self) -> List[AssemblyNote]:
        return list(self.notes)


# Helpers
def _repo_root() -> str:
    env = os.environ.get("HYPPE_DATA_ROOT")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _expand_view_weights(weights: Dict[str, float], n_levels: int
                         ) -> Optional[Dict[str, float]]:
    """Turn the config's aggregate `prefix_k` into per-prefix weights.

    `TrainingConfig.view_weights` uses one key for all prefixes; `LatentMask`
    names them `prefix_0 ... prefix_{K-2}` and rejects an unknown key. Passing
    the config through unchanged raises, and dropping it silently would make a
    declared weight inert. The split is the module's own (1/(i+1), normalised,
    remainder on the last), so the two agree by construction.
    """
    if not weights:
        return None
    out = {k: float(v) for k, v in weights.items() if k != "prefix_k"}
    share = float(weights.get("prefix_k", 0.0))
    n = n_levels - 1
    if share and n > 0:
        raw = [1.0 / (i + 1) for i in range(n)]
        total = sum(raw)
        vals = [share * r / total for r in raw]
        vals[-1] = share - sum(vals[:-1])
        for i in range(n):
            out["prefix_%d" % i] = vals[i]
    return out


def _odsiej_motywy(profile, mc):
    """Filter the motif bank by `model.motifs`. Returns (profiles, note).

    Order matters and is explicit here: the quantitative thresholds (bits,
    width) first, then the name lists. The other way round a threshold could
    remove a profile the user named in `keep_names`, so the config would say
    one thing and the bank hold another.

    `strict_names` makes a name absent from the bank raise; a typo would
    otherwise give a run with the full bank under the name of a run with an
    exclusion.
    """
    if mc is None:
        return profile, ""
    n0 = len(profile)
    czesci = []

    _mb = float(getattr(mc, "min_bits", 0.0) or 0.0)
    if _mb > 0.0:
        przed = len(profile)
        profile = [m for m in profile if m.information_bits() >= _mb]
        czesci.append("min_bits %.2f removed %d" % (_mb, przed - len(profile)))

    _mw = int(getattr(mc, "max_width", 0) or 0)
    if _mw > 0:
        przed = len(profile)
        profile = [m for m in profile if m.width <= _mw]
        czesci.append("max_width %d removed %d" % (_mw, przed - len(profile)))

    _dostepne = set(m.name for m in profile)
    _strict = bool(getattr(mc, "strict_names", True))

    _drop = tuple(getattr(mc, "drop_names", ()) or ())
    if _drop:
        _missing = [n for n in _drop if n not in _dostepne]
        if _missing and _strict:
            raise ValueError(
                "model.motifs.drop_names lists %d names that are NOT in the "
                "bank: %r. With strict_names=true this is an error, because "
                "passing over them silently would give a run with the full "
                "bank under the name of a run with exclusions."
                % (len(_missing), _missing[:8]))
        przed = len(profile)
        profile = [m for m in profile if m.name not in set(_drop)]
        czesci.append("drop_names removed %d" % (przed - len(profile)))

    _keep = tuple(getattr(mc, "keep_names", ()) or ())
    if _keep:
        _missing = [n for n in _keep if n not in _dostepne]
        if _missing and _strict:
            raise ValueError(
                "model.motifs.keep_names lists %d names that are NOT in "
                "the bank: %r." % (len(_missing), _missing[:8]))
        przed = len(profile)
        profile = [m for m in profile if m.name in set(_keep)]
        czesci.append("keep_names kept %d of %d" % (len(profile), przed))

    if not czesci:
        return profile, ""
    return profile, ("bank %d -> %d profiles (%s)"
                     % (n0, len(profile), "; ".join(czesci)))
