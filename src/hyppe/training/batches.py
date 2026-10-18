"""What one training step is fed, and where every field of it comes from.

One forward pass serves three tasks. Only the per-row strength (`absolute`) is
a supervised target; the ranking score and the realness logit are defined on
structure the batch has to carry. A ranking pair is built by indexing the
scores of two rows, never by a second forward pass, which would draw a second
flank dropout for the same condition. A negative is a dinucleotide-preserving
shuffle of a real row, so a head that beats it has not simply learnt GC
content; it enters `absolute` with `mu_valid = False` and `pairwise` as a
guaranteed loser against its source.

Pairs are drawn within one genome, because "which of these two is stronger" is
only defined if the two share the condition. The two rows of a pair also share
a `pair_group` id, which makes the flank-dropout draw identical for both: a
per-row draw would hand them different conditions with probability 27 % at
p = 0.15.

Properties of the source files that this code has to respect:

* `GROUP == "Promotor"` is exactly 800 bp; other lengths are counted and
  dropped, never padded.
* 7.4 % of bases are soft-masked lowercase, so nothing may read `SEQUENCE`
  directly; `iter_sequence_tsv` upper-cases through `CasePolicy.UPPERCASE`.
* the file is sorted by genome, so any prefix is a genome-biased sample, and
  sub-sampling goes by a hash of the orthogroup over the whole stream.
* `rank` imputes a missing orthogroup as exactly 1.0 on 33.26 % of cells, so
  the validity mask comes from `combat` and never from rank.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.conditioning import FlankEmbeddingCache
from ..data.labels import build_targets, load_samples
from ..data.negatives import assert_shuffle_preserves, euler_shuffle
from ..data.per_sample import PerSampleTargets, build_per_sample
from ..data.sequences import CasePolicy, iter_sequence_tsv
from ..data.splits import DEFAULT_COLUMN, load_splits
from ..data.text_cache import ExperimentTextCache, resolve_cells

__all__ = ["TrainBatch", "Corpus", "CorpusReport", "BatchMaker",
           "NEGATIVE_KIND", "NEGATIVE_KIND_CHIMERA", "BASES"]

#: The negative class. One spelling, shared by the batch, the discriminator's
#: `kind` argument and the report; a second would split the per-kind AUC into
#: two half-empty buckets. It must be declared in
#: `heads.discriminator.negative_kinds`, which the head checks.
NEGATIVE_KIND = "dinucleotide_shuffle"

#: Two promoters of the SAME genome spliced at a random breakpoint. Measured
#: composition ceiling 0.5195 and positional ceiling 0.5217, the only
#: construction in the survey at chance for both: every position carries real
#: promoter sequence from the same distance to the ATG, and only coherence
#: between positions differs.
NEGATIVE_KIND_CHIMERA = "chimera"
#: Three further classes declared in `discriminator.NEGATIVE_KINDS`.
NEGATIVE_KIND_CODING = "coding"
NEGATIVE_KIND_FLANK = "flank"
NEGATIVE_KIND_REVERSE = "reverse"

#: ACGT, the order of THIS repo (`hyppe.data.sequences.NUCLEOTIDE_ORDER`), in
#: which the complement is a reversal of the channel axis. The sibling track's
#: A,C,T,G convention with complement = (i+2) % 4 does not hold here.
BASES = "ACGT"
_CODE = {c: i for i, c in enumerate(BASES)}

#: Code of an unknown base, the same number as `data.kmer_index.N_CODE` and
#: `dataset.encode_codes`: a second value would make the contamination filter
#: and the one-hot disagree about what N is.
N_CODE = 4
#: Decoding alphabet. `BASES` stays four letters because it is the contract of
#: the one-hot channels; N is not a channel, it is the absence of one.
_BASES_N = BASES + "N"
#: One-hot column value for N, equal to `DataConfig.n_value`.
N_VALUE = 0.25
#: Reconstruction target at an N position. `reconstruction_loss` runs
#: `F.cross_entropy` over four classes, so code 4 is an index error, not a
#: worse target. The position therefore gets a fabricated 'A'; the choice is
#: arbitrary and only has to be CONSTANT, since the decoder input there is a
#: uniform column and "uniform column -> A" is learnable.
N_TARGET_CODE = 0


# Batch
@dataclass
class TrainBatch:
    """One step's worth of data. Every tensor is [B] unless stated.

    `x`             [B, 4, L] one-hot, ACGT.
    `target_idx`    [B, L] int64, built from the CODES and not from
                    `x.argmax()`, so an N cannot silently become an A.
    `is_promoter`   False for the shuffled rows.
    `mu_valid`      False wherever there is no measured strength: every
                    shuffled row, and every empty (orthogroup, genome) cell.
    `sigma_valid`   additionally False where n_obs < 2.
    `weight`        sqrt(n_obs): reliability is already 0.994-0.999 at the low
                    end, so linear weighting would hand `reesei` (386 samples)
                    16x the pull of `P1` (24) for no measured gain.
    `pair_*`        [P] over pairs; `pair_a`/`pair_b` index rows of this batch.
    `pair_group`    [B] equal ids for the two rows of a pair, so the flank
                    dropout draws once per pair.
    `flank_l/r`     [B, d_dna] frozen DNA-LM vectors of the LEFT and RIGHT
                    neighbourhood, or None. Two DIFFERENT vectors; one copied
                    into both is the failure "two tokens, one number".
    `flank_ok`      [B] bool, False where the promoter is not in the cache.
                    Those rows must reach `unknown_token_flank`, not a zero
                    vector: `flank_proj` has a bias, so a zero input is a
                    learned constant and the channel would report a context
                    share while carrying no information.
    `text_emb`      [B, d_lm] frozen sentence-LM vector of the experiment
                    description of the cell this row is scored in, or None.
                    Equal within a `pair_group` by construction, so the
                    siamese contract `s(z_A, C)` / `s(z_B, C)` holds.
    `text_ok`       [B] 1.0 where a vector was found; same rule as `flank_ok`.
    """

    x: torch.Tensor
    target_idx: torch.Tensor
    is_promoter: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    mu_valid: torch.Tensor
    sigma_valid: torch.Tensor
    weight: torch.Tensor
    species_id: torch.Tensor
    pair_group: torch.Tensor
    pair_a: torch.Tensor
    pair_b: torch.Tensor
    pair_sign: torch.Tensor
    pair_delta_y: torch.Tensor
    pair_nonprom: torch.Tensor
    row_index: torch.Tensor
    #: Per-sample regime; all None means the cell regime. `mu_row` is the
    #: ComBat value of this (orthogroup, SAMPLE), `sigma_row` its sd over
    #: replicates, and `cat_target` the row's category index or -1. In this
    #: regime `row_text` and `text_emb` are the SAME vector in two roles --
    #: conditioning the latent and generating the readout column -- because
    #: different ones would read the column of a description the model was
    #: not conditioned on, with no metric saying so.
    mu_row: Optional[torch.Tensor] = None
    mu_row_valid: Optional[torch.Tensor] = None
    sigma_row: Optional[torch.Tensor] = None
    sigma_row_valid: Optional[torch.Tensor] = None
    row_text: Optional[torch.Tensor] = None
    cat_target: Optional[torch.Tensor] = None
    negative_kind: Tuple[str, ...] = ()
    flank_l: Optional[torch.Tensor] = None
    flank_r: Optional[torch.Tensor] = None
    flank_ok: Optional[torch.Tensor] = None
    text_emb: Optional[torch.Tensor] = None
    text_ok: Optional[torch.Tensor] = None
    #: `gene_numeric` [B, n_numeric] is the phyletic block of the ORTHOGROUP
    #: this promoter belongs to; `gene_ok` [B] is 0 where the orthogroup is
    #: absent from the table, and those rows reach `unknown_token_gene`.
    gene_numeric: Optional[torch.Tensor] = None
    gene_ok: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.pair_a.numel())

    def to(self, device: torch.device) -> "TrainBatch":
        def mv(t):
            return t.to(device) if torch.is_tensor(t) else t
        return TrainBatch(
            x=mv(self.x), target_idx=mv(self.target_idx),
            is_promoter=mv(self.is_promoter), mu=mv(self.mu),
            sigma=mv(self.sigma), mu_valid=mv(self.mu_valid),
            sigma_valid=mv(self.sigma_valid), weight=mv(self.weight),
            species_id=mv(self.species_id), pair_group=mv(self.pair_group),
            pair_a=mv(self.pair_a), pair_b=mv(self.pair_b),
            pair_sign=mv(self.pair_sign), pair_delta_y=mv(self.pair_delta_y),
            pair_nonprom=mv(self.pair_nonprom), row_index=mv(self.row_index),
            mu_row=mv(self.mu_row), mu_row_valid=mv(self.mu_row_valid),
            sigma_row=mv(self.sigma_row),
            sigma_row_valid=mv(self.sigma_row_valid),
            row_text=mv(self.row_text),
            cat_target=mv(self.cat_target),
            negative_kind=self.negative_kind,
            flank_l=mv(self.flank_l), flank_r=mv(self.flank_r),
            flank_ok=mv(self.flank_ok),
            text_emb=mv(self.text_emb), text_ok=mv(self.text_ok),
            gene_numeric=mv(self.gene_numeric), gene_ok=mv(self.gene_ok))


# Corpus
@dataclass
class CorpusReport:
    """Everything that was dropped, and why. Printed before the first step."""

    rows_seen: int = 0
    promoter_rows: int = 0
    kept: int = 0
    dropped_length: int = 0
    dropped_ambiguous: int = 0
    dropped_no_split: int = 0
    #: Disjoint subsets that together make up `dropped_ambiguous`, split
    #: because "too much N" follows the `max_n_frac` threshold while "another
    #: IUPAC code" is a property of the file (0 such rows in 238 460).
    dropped_n_excess: int = 0
    dropped_iupac: int = 0
    #: Rows kept DESPITE an N, and how many N positions they contributed.
    kept_with_n: int = 0
    n_positions: int = 0
    #: The threshold that actually applied, and where it came from.
    max_n_frac: float = 0.0
    max_n_source: str = "default"
    dropped_no_target: int = 0
    dropped_unknown_genome: int = 0
    dropped_subsample: int = 0
    per_split: Dict[str, int] = field(default_factory=dict)
    genomes: Tuple[str, ...] = ()
    orthogroups: int = 0
    with_sigma: int = 0
    flank_cache: str = ""
    flank_covered: int = 0
    text_cache: str = ""
    text_backend: str = ""
    text_dim: int = 0
    text_covered: int = 0
    text_stats: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            "corpus: %d rows of GROUP=Promotor streamed (the other three "
            "groups are filtered inside the reader), %d kept"
            % (self.promoter_rows, self.kept),
            "  dropped: length!=800 %d | non-ACGT %d | no split %d | "
            "no combat cell %d | genome not in registry %d | sub-sample %d"
            % (self.dropped_length, self.dropped_ambiguous,
               self.dropped_no_split, self.dropped_no_target,
               self.dropped_unknown_genome, self.dropped_subsample),
            "  non-ACGT SPLIT: N above data.max_n_frac=%.4f (%s) %d | other "
            "IUPAC %d | kept WITH N %d rows carrying %d N positions (%.3e of "
            "all positions). A kept N is code %d -> a uniform %.2f one-hot "
            "column, and its reconstruction target is 'A', a base the "
            "sequence does not contain: that position counts as an error in "
            "rec_* unless the decoder learns the convention."
            % (self.max_n_frac, self.max_n_source, self.dropped_n_excess,
               self.dropped_iupac, self.kept_with_n, self.n_positions,
               (self.n_positions / float(self.kept * 800))
               if self.kept else 0.0, N_CODE, N_VALUE),
            "  splits: %s | %d orthogroups | %d genomes | %d rows carry a "
            "measured sigma (n_obs >= 2)"
            % (", ".join("%s=%d" % kv for kv in sorted(self.per_split.items())),
               self.orthogroups, len(self.genomes), self.with_sigma),
        ]
        if self.flank_cache:
            frac = (self.flank_covered / float(self.kept)) if self.kept else 0.0
            lines.append(
                "  gene_context: DNA-LM cache %s, %d of %d kept rows covered "
                "(%.4f); the rest answer with the unknown flank token"
                % (self.flank_cache, self.flank_covered, self.kept, frac))
        if self.text_cache:
            tf = ((self.text_covered / float(self.kept))
                  if self.kept else 0.0)
            lines.append(
                "  experiment_text: %s via %s, %d dims, %d of %d kept "
                "rows resolve a vector on a category cell (%.4f), %d "
                "distinct vectors, degenerate_frac %.4f; the rest answer "
                "with the unknown text token"
                % (self.text_cache, self.text_backend or "?",
                   self.text_dim, self.text_covered, self.kept, tf,
                   int(self.text_stats.get("distinct", 0)),
                   float(self.text_stats.get("degenerate_frac",
                                             0.0))))
        lines += ["  NOTE " + n for n in self.notes]
        return "\n".join(lines)


class Corpus(object):
    """Promoters with their strength labels and their split, in memory.

    Sequences are stored as base CODES (uint8, 0-3), not as strings: the codes
    are already the reconstruction target, and one-hot is then an indexing
    operation rather than a per-character Python loop.
    """

    def __init__(self, codes: np.ndarray, og: List[str], genome: List[str],
                 species_id: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 n_obs: np.ndarray, split: List[str],
                 report: CorpusReport,
                 ids: Optional[List[str]] = None,
                 flank_cache: Optional[FlankEmbeddingCache] = None,
                 flank_rows: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 text_cache: Optional[ExperimentTextCache] = None,
                 text_rows: Optional[np.ndarray] = None,
                 gene_table: Optional[np.ndarray] = None,
                 per_sample: Optional[PerSampleTargets] = None,
                 mu_rank: Optional[np.ndarray] = None,
                 rank_sign: float = 0.0
                 ) -> None:
        self.codes = codes                     # [N, L] uint8
        self.og = og
        self.genome = genome
        self.species_id = species_id           # [N] int64, registry row
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        self.n_obs = n_obs.astype(np.float32)
        self.split = split
        self.report = report
        # The promoter ID is the join key of every per-row cache; without it a
        # cache path can be declared and read by nothing.
        self.ids: List[str] = list(ids) if ids is not None else []
        self.flank_cache = flank_cache
        n = int(codes.shape[0]) if codes.size else 0
        if flank_rows is None:
            self.flank_row_l = np.full(n, -1, dtype=np.int64)
            self.flank_row_r = np.full(n, -1, dtype=np.int64)
        else:
            self.flank_row_l, self.flank_row_r = flank_rows
        self.flank_ok = self.flank_row_l >= 0
        # `text_row` is [N, C + 1]: one column per category plus the genome
        # marginal LAST, C + 1 whatever the label's column count is, so the
        # text key does not silently follow `include_marginal`.
        #: Per-sample target, or None for the cell regime.
        self.per_sample: Optional[PerSampleTargets] = per_sample
        #: Second target, [N, C+1], for the PAIR LABEL only. None means the
        #: pair label comes from the same matrix as the value.
        self.mu_rank = (None if mu_rank is None
                        else np.asarray(mu_rank, dtype=np.float32))
        #: Sign multiplier for pairs labelled from `mu_rank`, +1 or -1,
        #: established by MEASUREMENT at build time; 0.0 means no second
        #: target.
        self.rank_sign = float(rank_sign)
        self.text_cache = text_cache
        if text_rows is None:
            self.text_row = np.full((n, 1), -1, dtype=np.int64)
            self.text_n_cat = 0
        else:
            self.text_row = np.asarray(text_rows, dtype=np.int64)
            if self.text_row.shape[0] != n:
                raise ValueError(
                    "text_rows has %d rows for a corpus of %d"
                    % (int(self.text_row.shape[0]), n))
            self.text_n_cat = int(self.text_row.shape[1]) - 1
        # Gene channel, PER ROW rather than per unique orthogroup: `og` is
        # already a per-row list, and a second index is one more thing to keep
        # aligned for 29 MB.
        if gene_table is None:
            self.gene_numeric = None
            self.gene_ok = np.zeros(n, dtype=bool)
        else:
            self.gene_numeric = np.asarray(gene_table, dtype=np.float32)
            if self.gene_numeric.shape[0] != n:
                raise ValueError(
                    "gene_table has %d rows for a corpus of %d"
                    % (int(self.gene_numeric.shape[0]), n))
            self.gene_ok = np.isfinite(self.gene_numeric).all(axis=1) & \
                (np.abs(self.gene_numeric).sum(axis=1) > 0)
        self.seq_len = int(codes.shape[1]) if codes.size else 0
        #: Which rows carry an N, and whether any does. `any_n` gates every
        #: branch that `data.max_n_frac` adds, so at a threshold of 0.0
        #: exactly the original code path runs.
        self.has_n = ((codes == N_CODE).any(axis=1) if codes.size
                      else np.zeros(0, dtype=bool))
        self.any_n = bool(self.has_n.any())
        self._by_split: Dict[str, np.ndarray] = {}
        for name in sorted(set(split)):
            self._by_split[name] = np.array(
                [i for i, s in enumerate(split) if s == name], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.codes.shape[0])

    def indices(self, split: str) -> np.ndarray:
        if split not in self._by_split:
            raise KeyError("no rows in split %r; have %s"
                           % (split, sorted(self._by_split)))
        return self._by_split[split]

    def sequence(self, row: int) -> str:
        # `_BASES_N`, not `BASES`: code 4 is legal once `data.max_n_frac > 0`,
        # and `BASES[4]` would be an IndexError in the first batch that drew
        # such a row.
        return "".join(_BASES_N[c] for c in self.codes[row])

    def base_floor(self, split: str = "") -> Tuple[float, Dict[str, float]]:
        """(accuracy of always writing the most frequent base, composition).

        Measured on THIS corpus, not quoted. `RECONSTRUCTION_FLOOR = 0.2626`
        in `hyppe.config` is the same quantity on the real promoter corpus,
        and on uniform random DNA the floor is 0.25, so a synthetic run
        judged against 0.2626 fails a threshold its data cannot reach.
        """
        idx = self.indices(split) if split else np.arange(len(self))
        if idx.size == 0:
            return float("nan"), {}
        flat = self.codes[idx].reshape(-1)
        # N is not part of the composition: the floor is "the most frequent
        # BASE", and left in the denominator an N would lower `frac.max()`.
        flat = flat[flat < 4]
        if flat.size == 0:
            return float("nan"), {}
        counts = np.bincount(flat, minlength=4)
        frac = counts / float(counts.sum())
        return float(frac.max()), {BASES[i]: float(frac[i]) for i in range(4)}

    # Building
    @classmethod
    def from_paths(cls, sequences_path: str, combat_path: str,
                   samples_path: str, splits_path: str,
                   split_column: str = DEFAULT_COLUMN,
                   strain_names: Sequence[str] = (),
                   seq_len: int = 800,
                   max_orthogroups: int = 0,
                   min_obs_for_sigma: int = 2,
                   flank_cache_path: str = "",
                   flank_index_path: str = "",
                   text_cache_path: str = "",
                   text_index_path: str = "",
                   text_expect_dim: Optional[int] = None,
                   include_marginal: bool = True,
                   orthology_path: str = "",
                   sample_text_path: str = "",
                   rank_path: str = "",
                   max_n_frac: Optional[float] = None) -> "Corpus":
        """Stream the four source files into one aligned table.

        `max_orthogroups` sub-samples by a HASH of the orthogroup id over the
        whole stream. Not by truncation: the sequence table is sorted by
        genome, so its first N rows are one or two strains and a run on them
        measures a strain, not a sub-sample.
        """
        rep = CorpusReport()
        # Argument beats config beats default, and the report says which of
        # the three applied: a "0.0" in the log otherwise cannot distinguish
        # a config value from an unread config.
        if max_n_frac is None:
            _cfg = None
            try:
                from ..config import active_config as _ac
                _cfg = _ac()
            except Exception:                              # noqa: BLE001
                _cfg = None
            if _cfg is not None and hasattr(_cfg, "data"):
                rep.max_n_frac = float(getattr(_cfg.data, "max_n_frac", 0.0))
                rep.max_n_source = "config %s" % (
                    str(getattr(_cfg, "source", "")) or "<mapping>",)
            else:
                rep.max_n_frac = 0.0
                rep.max_n_source = "default (no config loaded)"
        else:
            rep.max_n_frac = float(max_n_frac)
            rep.max_n_source = "argument"
        if not 0.0 <= rep.max_n_frac <= 1.0:
            raise ValueError("max_n_frac is a fraction, got %r"
                             % (rep.max_n_frac,))
        _max_n = float(rep.max_n_frac)
        splits = load_splits(splits_path, split_column)
        samples = load_samples(samples_path)
        targets = build_targets(combat_path, samples,
                                min_obs=int(min_obs_for_sigma),
                                include_marginal=bool(include_marginal))
        rep.notes.append(
            "split column %r from %s; %d orthogroups assigned"
            % (split_column, splits.source.split("/")[-1], len(splits)))
        rep.notes.append(
            "labels: %d kept samples of %d, %d (orthogroup, genome) cells "
            "with a mean" % (samples.n_kept(), len(samples.sample_ids),
                             int(targets.valid.sum())))

        # A second target, for the PAIR LABEL only: the continuous head
        # learns a value and the pairwise head an order, and the matrices
        # differ exactly there. Spearman rho against log2(TPM+1) within a
        # sample is 0.9956 for rank against 0.8670 for combat.
        targets_rank = None
        rank_sign = 0.0
        if rank_path:
            # Through `build_rank_targets`, NOT `build_targets`: `rank` fills
            # missing cells with exactly 1.0, which is also a legal value for
            # a silent gene (9.22 % of cells, a 667x spike). `rank_targets`
            # takes the VALUES from rank and the VALIDITY from ComBat.
            from ..data.rank_targets import build_rank_targets
            _rt = build_rank_targets(rank_path, samples, targets,
                                     min_obs=int(min_obs_for_sigma))

            class _RankAdapter(object):
                pass
            targets_rank = _RankAdapter()
            targets_rank.mu = np.where(_rt.valid, _rt.y, np.nan)
            targets_rank.valid = _rt.valid
            _r = _rt.report()
            rep.notes.append(
                "rank targets via build_rank_targets: values from %s, "
                "VALIDITY from %s. imputed_kept=%s, imputed_masked=%s -- the "
                "first is the number of cells that survived the combat mask "
                "at exactly the imputed 1.0, i.e. the number that says "
                "whether the join worked."
                % (rank_path.split("/")[-1], combat_path.split("/")[-1],
                   _r.get("imputed_kept"), _r.get("imputed_masked")))
            if targets_rank.mu.shape != targets.mu.shape:
                raise ValueError(
                    "rank targets are %s and value targets are %s; the two "
                    "matrices must yield the same cell table or a pair label "
                    "would be read off a different cell than its value"
                    % (targets_rank.mu.shape, targets.mu.shape))
            # The direction is MEASURED, not assumed: rank here is descending
            # (1 = silent gene), and a wrong sign is invisible in the
            # metrics, because `pair_acc` comes out as 1 - x and still looks
            # plausible at x ~ 0.3. The two masks are the same by
            # construction, so this product is an identity that stays in
            # order to break here if that stops being true.
            _ok = (targets.valid & targets_rank.valid)
            _a = np.asarray(targets.mu)[_ok].ravel()
            _b = np.asarray(targets_rank.mu)[_ok].ravel()
            _f = np.isfinite(_a) & np.isfinite(_b)
            if int(_f.sum()) < 1000:
                raise ValueError(
                    "only %d cells are finite in BOTH the value and the rank "
                    "target; the direction of the rank scale cannot be "
                    "established and a guessed sign would train the head on "
                    "the reversed order" % int(_f.sum()))
            _r = float(np.corrcoef(_a[_f], _b[_f])[0, 1])
            if abs(_r) < 0.30:
                raise ValueError(
                    "value and rank targets correlate at %.4f over %d cells. "
                    "That is too weak to be a sign convention, so the two "
                    "matrices disagree about the ordering itself and the "
                    "pair label cannot be taken from one while the value "
                    "comes from the other." % (_r, int(_f.sum())))
            rank_sign = 1.0 if _r > 0 else -1.0
            rep.notes.append(
                "pair labels come from %s, values from %s. Direction MEASURED "
                "on %d shared cells: corr(value, rank) = %+.4f, so the pair "
                "sign is multiplied by %+.0f. A rank scale that is ascending "
                "instead of descending would flip this number, and the guard "
                "would follow it instead of silently training the reversed "
                "order."
                % (rank_path.split("/")[-1], combat_path.split("/")[-1],
                   int(_f.sum()), _r, rank_sign))

        keep_og = None
        if max_orthogroups and max_orthogroups > 0:
            ranked = sorted(
                splits.assignment,
                key=lambda o: hashlib.blake2b(o.encode("utf-8"),
                                              digest_size=8).hexdigest())
            # Sub-sample INSIDE each split, so a capped run still has a
            # validation set: a global hash cut can empty a 10 % split, and
            # an empty validation set reads as "no metric".
            want: Dict[str, int] = {}
            per = max(1, int(max_orthogroups) // max(len(set(
                splits.assignment.values())), 1))
            keep_og = set()
            for o in ranked:
                s = splits.assignment[o]
                if want.get(s, 0) < per:
                    want[s] = want.get(s, 0) + 1
                    keep_og.add(o)
            rep.notes.append(
                "SUB-SAMPLED to <= %d orthogroups per split by blake2b(og): a "
                "prefix of the sequence file would be one genome, not a sample"
                % (per,))

        strain_index = {n: i for i, n in enumerate(strain_names)}
        mur_out: List[np.ndarray] = []
        rows_codes: List[np.ndarray] = []
        id_out: List[str] = []
        og_out: List[str] = []
        gen_out: List[str] = []
        spc_out: List[int] = []
        mu_out: List[float] = []
        sd_out: List[float] = []
        nob_out: List[float] = []
        split_out: List[str] = []
        lut = np.full(256, 255, dtype=np.uint8)
        for c, i in _CODE.items():
            lut[ord(c)] = i

        for rec in iter_sequence_tsv(sequences_path, groups=("Promotor",),
                                     policy=CasePolicy.UPPERCASE):
            rep.rows_seen += 1
            rep.promoter_rows += 1
            og = str(rec.meta["og_id"])
            genome = str(rec.meta["genome"])
            if len(rec.seq) != int(seq_len):
                rep.dropped_length += 1
                continue
            sp = splits.of(og)
            if sp is None:
                rep.dropped_no_split += 1
                continue
            if keep_og is not None and og not in keep_og:
                rep.dropped_subsample += 1
                continue
            if genome not in strain_index:
                rep.dropped_unknown_genome += 1
                continue
            j = targets.residual_index.get((og, genome))
            if j is None or not bool(np.asarray(targets.valid[j]).any()):
                rep.dropped_no_target += 1
                continue
            arr = lut[np.frombuffer(rec.seq.encode("ascii", "replace"),
                                    dtype=np.uint8)]
            _bad = (arr == 255)
            _nbad = int(_bad.sum())
            if _nbad:
                # Two reasons, counted separately. An IUPAC code other than N
                # always goes, because a uniform column is a lie about R,
                # which means A or G. N goes only above `data.max_n_frac`,
                # because a uniform one-hot over much of a row is sequence
                # the model cannot see under a label at full weight.
                if set(rec.seq) - set(BASES) - {"N"}:
                    rep.dropped_ambiguous += 1
                    rep.dropped_iupac += 1
                    continue
                if _nbad > _max_n * len(rec.seq):
                    rep.dropped_ambiguous += 1
                    rep.dropped_n_excess += 1
                    continue
                arr = arr.copy()
                arr[_bad] = N_CODE
                rep.kept_with_n += 1
                rep.n_positions += _nbad
            rows_codes.append(arr)
            id_out.append(str(rec.name))
            og_out.append(og)
            gen_out.append(genome)
            spc_out.append(strain_index[genome])
            # One row per (orthogroup, genome); the CATEGORY is an axis of
            # the label, not of the row, and is drawn per batch.
            mu_out.append(np.asarray(targets.mu[j], dtype=np.float32))
            if targets_rank is not None:
                mur_out.append(np.asarray(targets_rank.mu[j],
                                          dtype=np.float32))
            sd_out.append(np.asarray(targets.sigma[j], dtype=np.float32))
            nob_out.append(np.asarray(targets.n_obs[j], dtype=np.float32))
            split_out.append(sp)

        if not rows_codes:
            raise RuntimeError(
                "no usable promoter rows. %s" % (rep.format(),))
        codes = np.stack(rows_codes)
        rep.kept = int(codes.shape[0])
        for s in split_out:
            rep.per_split[s] = rep.per_split.get(s, 0) + 1
        rep.genomes = tuple(sorted(set(gen_out)))
        rep.orthogroups = len(set(og_out))
        sd = np.asarray(sd_out, dtype=np.float32)
        # A ROW carries a measured sigma if ANY of its cells does; counting
        # cells would inflate the number by the category axis.
        rep.with_sigma = int(np.isfinite(sd).any(axis=1).sum()
                             if sd.ndim == 2 else np.isfinite(sd).sum())
        flank = None
        flank_rows = None
        if flank_cache_path:
            flank = FlankEmbeddingCache.load(flank_cache_path,
                                             flank_index_path)
            lrow, rrow = flank.rows_for(id_out)
            flank_rows = (lrow, rrow)
            rep.flank_cache = flank_cache_path
            rep.flank_covered = int((lrow >= 0).sum())
            rep.notes.append(flank.report.format().replace("\n", "\n  "))
        else:
            rep.notes.append(
                "condition channels flank_l / flank_r are NOT supplied by "
                "this loop (no DNA-LM embeddings given), so the condition "
                "encoder uses its UNKNOWN token for them")

        # Experiment description (C2), keyed by the CELL `<genome>|<category>`
        # rather than by row: a per-row aggregate would be a function of the
        # genome alone. Measured eta^2 of the embedding is 0.1348 on the
        # 19-level genome factor and 0.3317 on the 66-level cell factor, and
        # `degenerate_frac` below is the falsifier for that. The per-sample
        # target needs a SAMPLE-keyed artefact, which `build_per_sample`
        # checks.
        per_sample = None
        if sample_text_path:
            per_sample = build_per_sample(combat_path, samples,
                                          sample_text_path)
            _psr = per_sample.report()
            rep.notes.append(
                "PER-SAMPLE target: %d samples across %d genomes, "
                "descriptions per genome median %d (min %d, max %d), "
                "(genome, description) cells %d, of which %d carry a "
                "sigma; samples with a replicate in their own genome "
                "%.1f %%. The readout column is generated from THIS "
                "sample's description, so grouping by category, cluster "
                "or subcategory stays a free readout and is NOT done "
                "here."
                % (_psr["n_samples"], _psr["n_genomes"],
                   _psr["opisow_na_genom_mediana"],
                   _psr["opisow_na_genom_min"], _psr["opisow_na_genom_max"],
                   _psr["komorek_genom_x_opis"], _psr["komorek_z_sigma"],
                   100.0 * _psr["frakcja_probek_z_replika_W_GENOMIE"]))
        text = None
        text_rows = None
        if text_cache_path:
            text = ExperimentTextCache.load(
                text_cache_path, text_index_path,
                expect_dim=text_expect_dim)
            text_rows, tstats = resolve_cells(text, gen_out,
                                              targets.categories)
            rep.text_cache = text_cache_path
            rep.text_dim = int(text.dim)
            rep.text_backend = str(text.report.backend)
            rep.text_stats = dict(tstats)
            rep.text_covered = int(
                tstats["rows_with_any_category"])
            # A cache that loads and joins NOTHING is the silent failure:
            # enabled, costed, and carrying the unknown token on every row.
            if (int(tstats["rows_with_any_category"]) == 0
                    and int(tstats["rows_with_marginal"]) == 0):
                raise ValueError(
                    "%s loaded %d keys and matched NONE of the %d kept "
                    "rows. The join is on `<genome>|<category>` "
                    "(dataset.ConditionResolver.cell_key). The cache "
                    "offers %s; "
                    "this corpus asks for %s. Every row would reach "
                    "unknown_token_text, which is indistinguishable in "
                    "the log from a channel that does not help."
                    % (text_cache_path, int(text.report.n_keys),
                       rep.kept, list(text.report.sample_keys),
                       [_k for _k in (
                           "%s|%s" % (gen_out[0], targets.categories[0]),
                           "%s|*" % (gen_out[0],))]))
            rep.notes.append(text.report.format().replace("\n", "\n  "))
            rep.notes.append(
                "condition channel text_emb IS supplied by this loop: %d "
                "vectors of %d dims keyed <genome>|<category> (the "
                "spelling of dataset.ConditionResolver.cell_key); %d of %d kept "
                "rows resolve a vector on at least one category cell and "
                "%d on the genome marginal, %d distinct vectors are "
                "reachable from this corpus. degenerate_frac = %.4f is the "
                "falsifier: 1.0 means every category column of a row "
                "resolves to ONE vector, i.e. the artefact is keyed per "
                "genome after all and H(text | species) = 0. Rows with no "
                "key reach unknown_token_text through `text_ok`, an "
                "availability mask applied to the OPERAND, never "
                "text_proj(zeros)"
                % (int(text.report.n_keys), int(text.dim),
                   int(tstats["rows_with_any_category"]), rep.kept,
                   int(tstats["rows_with_marginal"]),
                   int(tstats["distinct"]),
                   float(tstats["degenerate_frac"])))
        else:
            rep.notes.append(
                "condition channel text_emb is NOT supplied by this loop "
                "(conditioning.experiment_text.enabled is false, or no "
                "cache path is set), so the condition encoder uses its "
                "UNKNOWN token for it and C2 is at its unknown value in "
                "every run built from this corpus")
        # Gene channel: the phyletic block of each row's orthogroup, built
        # here because it is a per-row join on the corpus's own `og` column
        # and the model has no row index.
        gene_table = None
        if orthology_path:
            from ..data.gene_features import build_gene_features
            gfeat = build_gene_features(orthology_path)
            gix = gfeat.index()
            rows_g = np.array([gix.get(str(o), -1) for o in og_out],
                              dtype=np.int64)
            gene_table = np.zeros((len(og_out), gfeat.numeric.shape[1]),
                                  dtype=np.float32)
            hit = rows_g >= 0
            gene_table[hit] = gfeat.numeric[rows_g[hit]]
            rep.notes.append(
                "gene channel: %d of %d rows joined to the orthology table "
                "(%d orthogroups, %d numeric columns); the rest reach the "
                "unknown gene token"
                % (int(hit.sum()), len(og_out), len(gfeat.og_ids),
                   int(gfeat.numeric.shape[1])))
            if not hit.any():
                raise ValueError(
                    "conditioning.gene is enabled and %r was read, but NOT "
                    "ONE corpus row joined to it. The orthogroup ids do not "
                    "match; a channel that is unknown for every row would "
                    "train a constant and report a context share."
                    % (orthology_path,))
        return cls(codes, og_out, gen_out,
                   np.array(spc_out, dtype=np.int64),
                   np.array(mu_out, dtype=np.float32), sd,
                   np.array(nob_out, dtype=np.float32), split_out, rep,
                   ids=id_out, flank_cache=flank, flank_rows=flank_rows,
                   text_cache=text, text_rows=text_rows,
                   per_sample=per_sample,
                   mu_rank=(np.stack(mur_out).astype(np.float32)
                            if mur_out else None),
                   rank_sign=float(rank_sign),
                   gene_table=gene_table)

    # Synthetic
    @classmethod
    def synthetic(cls, n: int = 512, seq_len: int = 800, n_genomes: int = 4,
                  seed: int = 7898) -> "Corpus":
        """A corpus with a KNOWN signal, for wiring runs only.

        The strength is a linear function of the count of one 6-mer plus
        noise, so a loop that is wired at all must move the metric. Every
        report built from it says `synthetic`.
        """
        rng = np.random.default_rng(seed)
        codes = rng.integers(0, 4, size=(n, seq_len), dtype=np.int64)
        motif = np.array([_CODE[c] for c in "TATAAA"], dtype=np.int64)
        hits = np.zeros(n, dtype=np.float32)
        for i in range(n):
            k = int(rng.integers(0, 6))
            for _ in range(k):
                p = int(rng.integers(0, seq_len - motif.size))
                codes[i, p:p + motif.size] = motif
            hits[i] = k
        mu = hits + rng.normal(0.0, 0.3, size=n).astype(np.float32)
        og = ["OGSYN%05d" % (i // 2) for i in range(n)]
        genome = ["G%d" % (i % n_genomes) for i in range(n)]
        species = np.array([i % n_genomes for i in range(n)], dtype=np.int64)
        split = ["train" if i % 10 < 8 else "val" for i in range(n)]
        rep = CorpusReport(rows_seen=n, promoter_rows=n, kept=n,
                           per_split={"train": sum(1 for s in split
                                                   if s == "train"),
                                      "val": sum(1 for s in split
                                                 if s == "val")},
                           genomes=tuple(sorted(set(genome))),
                           orthogroups=len(set(og)), with_sigma=n)
        rep.notes.append("SYNTHETIC corpus: strength = count of TATAAA + "
                         "N(0, 0.3). Wiring only; no biological claim.")
        return cls(codes.astype(np.uint8), og, genome, species,
                   mu.astype(np.float32),
                   np.full(n, 0.3, dtype=np.float32),
                   np.full(n, 4.0, dtype=np.float32), split, rep)


# Per-orthogroup sample weighting
#: Values of `data.sample_weight`. See `hyppe.config.DataConfig` for the
#: measurements behind the exponent, and for why the correction lives in the
#: SAMPLER rather than in the loss weight.
SAMPLE_WEIGHT_DEFAULT = "sqrt_n_obs"
SAMPLE_WEIGHT_PER_OG = "sqrt_n_obs_per_og"
SAMPLE_WEIGHTS = (SAMPLE_WEIGHT_DEFAULT, SAMPLE_WEIGHT_PER_OG)

#: Cap on how often one corpus row may enter one balanced epoch. On the train
#: split the per-orthogroup mass CV falls 0.3407 / 0.2880 / 0.2110 / 0.1796 at
#: caps 3, 6, 12, 15, and only at 15 does the cap stop binding; the residual
#: is the spread of sqrt(n_obs) between groups, which is meant to stay. The
#: cap is a guard for a corpus with a larger N/G, and `raport_wazenia` prints
#: the realised maximum.
OG_MAX_KOPII = 15


# Batch maker
class BatchMaker(object):
    """Corpus -> `TrainBatch`, including the negatives and the pairs.

    Parameters
    ----------
    negative_fraction
        Shuffled rows per promoter row. They are the discriminator's only
        negative class, so 0.0 disables the head's task rather than making it
        easier, and the loop reports it as a dropped player.
    min_gap
        Pairs whose measured difference is below this are DROPPED, not given
        an arbitrary sign: a ranking label inside the label's own noise is
        noise with a direction attached.
    """

    def __init__(self, corpus: Corpus, batch_size: int = 16,
                 negative_fraction: float = 0.25, min_gap: float = 0.05,
                 seed: int = 7898, verify_shuffles: int = 4,
                 shuffle_order: int = 4,
                 chimera_fraction: float = 0.5,
                 negative_mix: Optional[Dict[str, float]] = None,
                 sequences_path: str = "",
                 aux_per_genome: int = 1200,
                 flank_kmer: int = 31,
                 per_category: bool = False,
                 min_cell_obs: int = 2,
                 ortholog_fraction: float = 0.0,
                 ortholog_block: int = 4,
                 nonpromoter_pairs: bool = True,
                 pair_from_rank: bool = False,
                 min_gap_rank: float = 0.0,
                 sample_weight: Optional[str] = None) -> None:
        self.corpus = corpus
        self.batch_size = int(batch_size)
        self.negative_fraction = float(negative_fraction)
        self.min_gap = float(min_gap)
        #: Share of each batch reserved for ORTHOGROUP BLOCKS -- rows of one
        #: gene from several genomes, so the siamese head can be given a pair
        #: that varies the genome and holds everything else. A random batch
        #: of 96 holds 0.60 cross-genome pairs, so at 0.0 the rule is inert.
        self.ortholog_fraction = float(ortholog_fraction)
        #: Whether negatives enter the pairwise head's pairs. They are 38.16 %
        #: of the pair budget and their sign is constant, so the shuffled-sign
        #: control scores 1.0000 instead of ~0.5 and the layer separates no
        #: two methods. They stay with the discriminator instead.
        self.nonpromoter_pairs = bool(nonpromoter_pairs)
        #: Whether the pair label comes from the RANK matrix instead of the
        #: value matrix.
        self.pair_from_rank = bool(pair_from_rank)
        #: Noise threshold IN RANK UNITS. `min_gap` is in value units, where
        #: the median |delta| is 1.8439 against 0.2490 on ranks, so 0.05
        #: keeps 98.50 % of pairs against 89.09 %. It must be given
        #: explicitly, because the share of rejected pairs is the task.
        self.min_gap_rank = float(min_gap_rank)
        self.ortholog_block = max(2, int(ortholog_block))
        self._og_blocks: Dict[str, List[np.ndarray]] = {}
        self.seed = int(seed)
        self.verify_shuffles = int(verify_shuffles)
        # Order 4, not 2 (ceiling 0.6305 against 0.9136) and not 5 (13.4 % of
        # promoters cannot be shuffled there, and the failures are biased).
        self.shuffle_order = int(shuffle_order)
        self.chimera_fraction = float(chimera_fraction)
        #: Weights of the negative classes. None keeps the older behaviour
        #: (chimera against shuffle by `chimera_fraction`) bit for bit.
        self.negative_mix = (dict(negative_mix) if negative_mix
                             else None)
        if self.negative_mix:
            _s = float(sum(max(0.0, float(v))
                           for v in self.negative_mix.values()))
            if _s <= 0:
                raise ValueError("negative_mix sums to zero")
            self.negative_mix = {k: max(0.0, float(v)) / _s
                                 for k, v in self.negative_mix.items()}
        self.sequences_path = str(sequences_path or "")
        self.aux_per_genome = int(aux_per_genome)
        self.flank_kmer = int(flank_kmer)
        #: Pools loaded LAZILY: building one costs a pass over the file.
        self._pools: Dict[str, object] = {}
        self._pool_failed: Dict[str, str] = {}
        #: Draw a CATEGORY cell per row instead of using the genome marginal.
        self.per_category = bool(per_category)
        #: A cell needs this many observations before it may be drawn. A mean
        #: of one sample is not a mean, and P1 has cells of size 1.
        self.min_cell_obs = int(min_cell_obs)
        #: The category's stand-in for a frozen sentence-LM embedding: one
        #: FIXED row per category, drawn once from this seed so a checkpoint
        #: trained against it can be reloaded. Built lazily, because its
        #: width comes from the model's condition config.
        self._cat_emb = None
        self._shuffled: Dict[int, np.ndarray] = {}
        self._verified = 0
        # FIVE rows, not four: row `N_CODE` is the uniform `N_VALUE` column.
        # At `max_n_frac = 0.0` no code 4 is ever made and the fifth row is
        # never indexed; without it the first N would be an IndexError.
        self._eye = np.concatenate(
            [np.eye(4, dtype=np.float32),
             np.full((1, 4), N_VALUE, dtype=np.float32)], axis=0)
        #: What was actually built. Printed by the loop; without it a class
        #: could silently degrade to order 2 for part of the corpus.
        self.neg_stats: Dict[str, int] = {}
        #: Pair groups whose members were checked to resolve ONE
        #: experiment-text row. 0 until the first batch with a cache.
        self.text_pair_groups: int = 0
        # `data.sample_weight`, resolved as `max_n_frac` is in
        # `Corpus.from_paths`, and PRINTED. `build_loop` passes named
        # arguments and does not know this key, so without the bridge it
        # would be a declaration with no reader.
        if sample_weight is None:
            _cfg = None
            try:
                from ..config import active_config as _ac
                _cfg = _ac()
            except Exception:                              # noqa: BLE001
                _cfg = None
            if _cfg is not None and hasattr(_cfg, "data"):
                self.sample_weight = str(getattr(
                    _cfg.data, "sample_weight", SAMPLE_WEIGHT_DEFAULT))
                self.sample_weight_source = "config %s" % (
                    str(getattr(_cfg, "source", "")) or "<mapping>",)
            else:
                self.sample_weight = SAMPLE_WEIGHT_DEFAULT
                self.sample_weight_source = "default (no config loaded)"
        else:
            self.sample_weight = str(sample_weight)
            self.sample_weight_source = "argument"
        if self.sample_weight not in SAMPLE_WEIGHTS:
            raise ValueError(
                "data.sample_weight = %r; znane %r"
                % (self.sample_weight, SAMPLE_WEIGHTS))
        #: Caches for the observable and the epoch plan. Computed once.
        self._og_grupy_cache: Dict[str, List[np.ndarray]] = {}
        self._og_kod_cache: Optional[np.ndarray] = None
        self._masa_cache: Optional[np.ndarray] = None
        print("  sample_weight: %r (%s) -- %s"
              % (self.sample_weight, self.sample_weight_source,
                 ("epoch BALANCED by orthogroup, at most %d copies of a row; "
                  "the 1/k correction acts in the SAMPLER, because the "
                  "absolute head is given weight=None in the per-sample "
                  "regime"
                  % (OG_MAX_KOPII,))
                 if self.sample_weight == SAMPLE_WEIGHT_PER_OG
                 else "one pass over the split rows, with no orthogroup "
                      "weighting (the behaviour so far)"))

    # Negatives
    def negative_codes(self, row: int) -> np.ndarray:
        """Dinucleotide-preserving shuffle of one row, cached and seeded.

        The first few are checked with `assert_shuffle_preserves`, which
        fails both on a changed 3-mer composition and on a shuffle that
        returned its input -- a "negative" identical to its source trains the
        discriminator to call a promoter a non-promoter.
        """
        got = self._shuffled.get(row)
        if got is not None:
            return got
        src = self.corpus.sequence(row)
        # `euler_shuffle` refuses a sequence containing N, since an Eulerian
        # path over k-mer vertices is undefined outside the alphabet, so
        # without this branch the first N-bearing source raises mid-epoch.
        # Shuffling the ACGT subsequence and restoring the N positions keeps
        # the composition exact and the gaps where they were.
        _pos_n = ()
        if self.corpus.any_n and "N" in src:
            _pos_n = tuple(i for i, ch in enumerate(src) if ch == "N")
            src = src.replace("N", "")
            self.neg_stats["shuffle_z_n"] = (
                self.neg_stats.get("shuffle_z_n", 0) + 1)
        # The fallback ladder is not caution. At order 4 about 4.9 % of
        # promoters admit no Eulerian path preserving their 5-mer counts,
        # and the failures are homopolymer-rich (longest run 11.30 against
        # 6.56), so dropping them would let the head separate on homopolymer
        # length alone. The counter says how often the ladder was used.
        arr = None
        for order in range(self.shuffle_order, 1, -1):
            try:
                out = euler_shuffle(
                    src, order=order,
                    rng=random.Random(self.seed * 1000003 + row))
                if self._verified < self.verify_shuffles:
                    assert_shuffle_preserves(src, out, order=order)
                    self._verified += 1
            except (ValueError, RuntimeError, AssertionError):
                continue
            key = "shuffle_order_%d" % order
            self.neg_stats[key] = self.neg_stats.get(key, 0) + 1
            if _pos_n:
                _buf = list(out)
                for _i in _pos_n:
                    _buf.insert(_i, "N")
                out = "".join(_buf)
            arr = np.array([_CODE.get(c, N_CODE) for c in out],
                           dtype=np.uint8)
            break
        if arr is None:
            # Nothing down to order 2 worked; refuse rather than return the
            # input unchanged.
            raise RuntimeError(
                "row %d admits no dinucleotide-preserving shuffle at any "
                "order from %d down to 2" % (row, self.shuffle_order))
        self._shuffled[row] = arr
        return arr

    def chimera_codes(self, row_a: int, row_b: int,
                      rng: np.random.Generator) -> np.ndarray:
        """Two promoters of the same genome, spliced at a random breakpoint.

        The breakpoint is drawn rather than fixed at the midpoint, because a
        constant breakpoint is itself a learnable feature. The edges are
        excluded so that both parents contribute a real stretch.
        """
        a = self.corpus.codes[row_a]
        b = self.corpus.codes[row_b]
        n = int(a.shape[0])
        lo, hi = max(1, n // 5), min(n - 1, (4 * n) // 5)
        bp = int(rng.integers(lo, hi))
        out = np.concatenate([a[:bp], b[bp:]]).astype(np.uint8)
        assert out.shape[0] == n, (out.shape, n)
        self.neg_stats["chimera"] = self.neg_stats.get("chimera", 0) + 1
        return out

    def reverse_codes(self, row: int) -> np.ndarray:
        """Reverse complement of the row: the same GC content, reversed.

        This is a correct negative and not a copy of the positive because the
        corpus is consistently oriented (mirror test -0.03 against +0.99 for
        a synthetic 50/50 mixture) and `data.random_rc` is false: the model
        never sees RC as an augmentation, so the other strand is not the same
        object to it. BASES = "ACGT", so the complement is `3 - i`, a
        different convention from the sibling track's `(i+2) % 4`.
        """
        src = self.corpus.codes[row]
        _rev = src[::-1].astype(np.int16)
        # The complement of N is N. Without this, `3 - 4 = -1` wraps in uint8
        # to 255, a code outside the alphabet.
        out = np.where(_rev >= N_CODE, N_CODE, 3 - _rev).astype(np.uint8)
        self.neg_stats["reverse"] = self.neg_stats.get("reverse", 0) + 1
        return np.ascontiguousarray(out)

    def _pool(self, kind: str):
        """The `coding` or `flank` pool, loaded once. None = unavailable."""
        if kind in self._pools:
            return self._pools[kind]
        if kind in self._pool_failed:
            return None
        if not self.sequences_path:
            self._pool_failed[kind] = "missing sequences_path"
            print("POOL %s UNAVAILABLE: BatchMaker was not given "
                  "sequences_path, so this class of negatives will NOT "
                  "be produced (the slot falls back to shuffle)"
                  % (kind,))
            return None
        from .neg_pools import AuxPool
        grupy = (("Gene",) if kind == NEGATIVE_KIND_CODING
                 else ("FlankLeft", "FlankRight"))
        gen = sorted(set(self.corpus.genome))
        dl = int(self.corpus.codes.shape[1])
        try:
            pool = AuxPool.load(self.sequences_path, grupy, kind,
                                length=dl, genomes=gen,
                                max_per_genome=self.aux_per_genome,
                                seed=self.seed)
        except Exception as exc:                       # noqa: BLE001
            self._pool_failed[kind] = str(exc)
            print("PULA %s NIE ZBUDOWANA (%s) -- slot spadnie na "
                  "shuffle i licznik to pokaze" % (kind, exc))
            return None
        if kind == NEGATIVE_KIND_FLANK:
            # 5.1 % of flanks ARE a promoter and 36.3 % contain promoter
            # sequence, so without this filter the class teaches the head
            # that a promoter is a negative, at full weight. Promoters are a
            # uniform 800 bp, which is why `Corpus` keeps no length vector
            # and why the assertion stays. TRAIN ONLY: built from the whole
            # corpus it would let held-out sequence decide which negatives
            # exist, biasing the final TEST disc_auc.
            _rows = self.corpus.indices("train")
            if _rows.size == 0:
                raise RuntimeError(
                    "the flank contamination filter has no TRAIN rows to build "
                    "from. Refusing rather than falling back to the whole "
                    "corpus: the fallback is exactly the leak this guard "
                    "exists to stop, and it would be invisible in the log.")
            _codes = np.ascontiguousarray(self.corpus.codes[_rows])
            _n, _w = _codes.shape
            assert _w == int(self.corpus.codes.shape[1]), (_n, _w)
            _dl = np.full(_n, _w, dtype=np.int64)
            print("  contamination filter built from %d TRAIN rows of %d "
                  "in the corpus (held-out is NOT read)"
                  % (_n, int(self.corpus.codes.shape[0])))
            pool.drop_contaminated(_codes, _dl,
                                   k=self.flank_kmer, prom_stride=8)
        print(pool.format())
        self._pools[kind] = pool
        return pool

    def _pool_codes(self, kind: str, row: int,
                    rng: np.random.Generator):
        pool = self._pool(kind)
        if pool is None:
            return None
        got = pool.draw(str(self.corpus.genome[row]),
                        int(self.corpus.codes.shape[1]), rng)
        if got is None:
            return None
        self.neg_stats[kind] = self.neg_stats.get(kind, 0) + 1
        return got

    # Batch
    def _build(self, rows: np.ndarray, rng: np.random.Generator,
               with_negatives: bool, n_block: int = 0) -> TrainBatch:
        cor = self.corpus
        n_pos = int(rows.size)
        n_neg = (int(round(self.negative_fraction * n_pos))
                 if with_negatives else 0)
        n_neg = min(n_neg, n_pos)
        neg_src = (rng.choice(n_pos, size=n_neg, replace=False)
                   if n_neg else np.zeros(0, dtype=np.int64))

        codes = [cor.codes[r] for r in rows]
        # Whether this batch can hold two copies of one row; see the chimera
        # partner guard below.
        _bez_dubli = (self.sample_weight == SAMPLE_WEIGHT_PER_OG)
        # Each negative keeps its SOURCE row, so genome, flanks and the
        # `promoter outranks non-promoter` pair stay keyed on `neg_src`.
        neg_kinds: List[str] = []
        by_spc: Dict[int, List[int]] = {}
        for i in range(n_pos):
            by_spc.setdefault(int(cor.species_id[rows[i]]), []).append(i)
        for s in neg_src:
            i = int(s)
            # Duplicate-row guard: in the balanced regime two batch POSITIONS
            # can be the same corpus row, and a chimera of a row with itself
            # returns the source DNA -- a "negative" that is a real promoter.
            # Pairs do not need the guard: dy = 0 fails `abs(dy) < prog_par`.
            partners = [j for j in by_spc.get(int(cor.species_id[rows[i]]), ())
                        if j != i
                        and not (_bez_dubli
                                 and int(rows[j]) == int(rows[i]))]
            got = None
            kind = None
            if self.negative_mix:
                # Weighted draw of the class. Sorted, so that one seed gives
                # one sequence whatever the key order in the YAML.
                _k = sorted(self.negative_mix)
                _w = np.array([self.negative_mix[x] for x in _k],
                              dtype=np.float64)
                kind = str(_k[int(rng.choice(len(_k), p=_w))])
                if kind == NEGATIVE_KIND_CHIMERA:
                    if partners:
                        j = int(partners[int(rng.integers(len(partners)))])
                        got = self.chimera_codes(int(rows[i]),
                                                 int(rows[j]), rng)
                elif kind == NEGATIVE_KIND_REVERSE:
                    got = self.reverse_codes(int(rows[i]))
                elif kind in (NEGATIVE_KIND_CODING, NEGATIVE_KIND_FLANK):
                    got = self._pool_codes(kind, int(rows[i]), rng)
                elif kind == NEGATIVE_KIND:
                    got = self.negative_codes(int(rows[i]))
                else:
                    raise ValueError(
                        "negative_mix zawiera nieznana klase %r; "
                        "dozwolone: %r" % (kind, (NEGATIVE_KIND,
                                                 NEGATIVE_KIND_CHIMERA,
                                                 NEGATIVE_KIND_CODING,
                                                 NEGATIVE_KIND_FLANK,
                                                 NEGATIVE_KIND_REVERSE)))
                if got is None:
                    # The fallback is counted, not silent.
                    _key = "%s_unavailable" % (kind,)
                    self.neg_stats[_key] = self.neg_stats.get(_key, 0) + 1
                    got = self.negative_codes(int(rows[i]))
                    kind = NEGATIVE_KIND
                codes.append(got)
                neg_kinds.append(kind)
            else:
                want_chimera = (partners
                                and rng.random() < self.chimera_fraction)
                if want_chimera:
                    j = int(partners[int(rng.integers(len(partners)))])
                    codes.append(self.chimera_codes(int(rows[i]),
                                                    int(rows[j]), rng))
                    neg_kinds.append(NEGATIVE_KIND_CHIMERA)
                else:
                    # No same-genome partner in this batch: fall back to the
                    # shuffle rather than splicing across genomes, which
                    # would give the head a genome cue.
                    codes.append(self.negative_codes(int(rows[i])))
                    neg_kinds.append(NEGATIVE_KIND)
                    if not partners:
                        self.neg_stats["chimera_no_partner"] = (
                            self.neg_stats.get("chimera_no_partner", 0)
                            + 1)
        code_arr = np.stack(codes).astype(np.int64)
        x = torch.from_numpy(
            self._eye[code_arr].transpose(0, 2, 1).copy())     # [B, 4, L]
        # An N position gets target 0, a base the sequence does not contain,
        # and counts as an error in `rec_*` like any other. What excuses it
        # is size: 1.2e-05 of all corpus positions at `max_n_frac = 0.05`.
        # The test asks the ARRAY, not the corpus, because a negative can
        # bring a code 4 from a pool while `cor.any_n` covers promoters.
        target_idx = torch.from_numpy(
            np.where(code_arr > 3, N_TARGET_CODE, code_arr)
            if int(code_arr.max()) > 3 else code_arr)

        B = n_pos + n_neg
        is_prom = np.zeros(B, dtype=bool)
        is_prom[:n_pos] = True
        two_d = (np.asarray(cor.mu).ndim == 2) and self.per_category
        n_col = int(np.asarray(cor.mu).shape[1]) if two_d else 1
        shape = (B, n_col) if two_d else (B,)
        mu = np.zeros(shape, dtype=np.float32)
        sd = np.full(shape, np.nan, dtype=np.float32)
        nob = np.zeros(shape, dtype=np.float32)
        spc = np.zeros(B, dtype=np.int64)
        # The label is EVERY cell of the row, with the mask saying which were
        # measured. Sampling one cell per draw would show the model one
        # sequence under different labels: the marginal with extra noise.
        if two_d:
            mu[:n_pos] = np.asarray(cor.mu)[rows]
            sd[:n_pos] = np.asarray(cor.sigma)[rows]
            nob[:n_pos] = np.asarray(cor.n_obs)[rows]
        else:
            mu[:n_pos] = (cor.mu[rows] if np.asarray(cor.mu).ndim == 1
                          else np.asarray(cor.mu)[rows, -1])
            sd[:n_pos] = (cor.sigma[rows]
                          if np.asarray(cor.sigma).ndim == 1
                          else np.asarray(cor.sigma)[rows, -1])
            nob[:n_pos] = (cor.n_obs[rows]
                           if np.asarray(cor.n_obs).ndim == 1
                           else np.asarray(cor.n_obs)[rows, -1])
        spc[:n_pos] = cor.species_id[rows]

        # `cat_row` picks which cell a row is PAIRED in; it does not restrict
        # what the head is trained on.
        cat_row = np.full(B, -1, dtype=np.int64)
        if two_d:
            usable = (np.isfinite(np.asarray(cor.mu)[rows])
                      & (np.asarray(cor.n_obs)[rows] >= self.min_cell_obs))
            # The marginal is not a category and cannot host a pair -- but
            # only if the corpus still carries one. Under
            # `include_marginal=False` every column IS a category, and
            # dropping the last silently removed `other` (0 pairs of 4064).
            # The guard is therefore on the SHAPE, not on a fixed index.
            from ..data.labels import SamplePolicy as _SP_pair
            if n_col > len(_SP_pair().categories):
                usable[:, n_col - 1] = False   # more columns than categories
            for i in range(n_pos):
                opts = np.flatnonzero(usable[i])
                if opts.size:
                    cat_row[i] = int(opts[rng.integers(opts.size)])
        if n_neg:
            # A shuffled row keeps the GENOME and the CATEGORY of its source:
            # the condition is the same, only the sequence is destroyed.
            spc[n_pos:] = cor.species_id[rows[neg_src]]
            cat_row[n_pos:] = cat_row[np.asarray(neg_src, dtype=np.int64)]

        # The same rule governs the flanks, the description and the gene: a
        # negative inherits all three from its source, because an unknown
        # value would hand the discriminator a second, easier cue.
        flank_l = flank_r = flank_ok = None
        if cor.flank_cache is not None:
            base = np.asarray(rows, dtype=np.int64)
            src_rows = (np.concatenate([base, base[neg_src]])
                        if n_neg else base)
            assert src_rows.size == B, (src_rows.size, B)
            lrow = cor.flank_row_l[src_rows]
            rrow = cor.flank_row_r[src_rows]
            flank_l = torch.from_numpy(cor.flank_cache.gather(lrow))
            flank_r = torch.from_numpy(cor.flank_cache.gather(rrow))
            flank_ok = torch.from_numpy((lrow >= 0).astype(np.float32))

        text_emb = text_ok = None
        # Per-sample target, drawn HERE and not in the corpus: each epoch is
        # meant to draw a different sample for the same (orthogroup, genome)
        # row, which is the whole difference from the cell target.
        mu_row = mu_row_valid = sigma_row = sigma_row_valid = None
        row_text = None
        if getattr(cor, "per_sample", None) is not None:
            base_ps = np.asarray(rows, dtype=np.int64)
            src_ps = (np.concatenate([base_ps, base_ps[neg_src]])
                      if n_neg else base_ps)
            d = cor.per_sample.draw([cor.og[i] for i in src_ps.tolist()],
                                    [cor.genome[i] for i in src_ps.tolist()],
                                    rng)
            mu_row = torch.from_numpy(np.nan_to_num(d.mu, nan=0.0))
            # A negative has no measured strength and cannot reach the
            # continuous head, or the regression would learn the source
            # promoter's value for its own shuffled copy.
            ok_ps = d.valid.copy()
            if n_neg:
                ok_ps[n_pos:] = False
            mu_row_valid = torch.from_numpy(ok_ps)
            sigma_row = torch.from_numpy(np.nan_to_num(d.sigma, nan=0.0))
            sv = d.sigma_valid.copy()
            if n_neg:
                sv[n_pos:] = False
            sigma_row_valid = torch.from_numpy(sv)
            row_text = torch.from_numpy(cor.per_sample.gather(d.text_row))

        trow = None
        if cor.text_cache is not None:
            base = np.asarray(rows, dtype=np.int64)
            src_rows = (np.concatenate([base, base[neg_src]])
                        if n_neg else base)
            assert src_rows.size == B, (src_rows.size, B)
            ncat = int(cor.text_n_cat)
            # A row that drew no category takes the genome marginal column,
            # which is the correct key for "no category". There is NO
            # fallback from a missing (genome, category) key to the marginal:
            # such a row reaches the unknown token and is counted, because
            # the marginal is a different quantity.
            col = np.where((cat_row >= 0) & (cat_row < ncat),
                           cat_row, ncat)
            trow = cor.text_row[src_rows, col]
            text_emb = torch.from_numpy(cor.text_cache.gather(trow))
            text_ok = torch.from_numpy(
                (trow >= 0).astype(np.float32))
        if row_text is not None:
            # In the per-sample regime the conditioning must be the
            # description of the very sample whose value is the target, or
            # the model conditions on a category description while reading a
            # column generated from a sample description, with no metric
            # showing it. Overwriting the cell column is deliberate.
            text_emb = row_text
            text_ok = torch.from_numpy(
                (np.asarray(d.text_row) >= 0).astype(np.float32))

        gene_numeric = gene_ok = None
        if cor.gene_numeric is not None:
            base = np.asarray(rows, dtype=np.int64)
            src_rows = (np.concatenate([base, base[neg_src]])
                        if n_neg else base)
            assert src_rows.size == B, (src_rows.size, B)
            gene_numeric = torch.from_numpy(
                np.ascontiguousarray(cor.gene_numeric[src_rows]))
            gene_ok = torch.from_numpy(
                cor.gene_ok[src_rows].astype(np.float32))

        if two_d:
            # A cell is a target only where it was MEASURED, or the head
            # would be trained towards NaN on every absent cell -- and the
            # corpus has 65 non-empty cells of a possible 133.
            mu_valid = is_prom[:, None] & np.isfinite(mu)
        else:
            mu_valid = is_prom.copy() & np.isfinite(mu)
        # `sd == 0` is a tie between two identical ComBat values, not a
        # measurement that the condition has no variability: left in, those
        # 3.35 % of cells carry 46.11 % of the sigma term.
        sigma_valid = (mu_valid & np.isfinite(sd) & (nob >= 2.0)
                       & (sd > 0.0))
        weight = np.sqrt(np.maximum(nob, 0.0)).astype(np.float32)
        weight[n_pos:] = 1.0

        # Pairs
        pair_a: List[int] = []
        pair_b: List[int] = []
        pair_sign: List[float] = []
        pair_dy: List[float] = []
        pair_np: List[bool] = []
        group = np.arange(B, dtype=np.int64)
        gid = B
        # A pair must share a cell: across categories the sign of a
        # difference answers a different question. The orthogroup block --
        # the last `n_block` rows -- is excluded here and paired across
        # genomes below instead, because a row carries only one `pair_group`
        # and letting it into both pairings would silently drop one of the
        # two shared-dropout guarantees.
        by_genome: Dict[Tuple[int, int], List[int]] = {}
        # Source of the pair label. By default the same matrix as the value.
        mu_par, zn_par, prog_par = mu, 1.0, self.min_gap
        if getattr(cor, "mu_rank", None) is not None and self.pair_from_rank:
            # Same shape as `mu`, filled for POSITIVE rows only: negatives
            # have no order just as they have no value, and `pair_np` keeps
            # them out of this loop.
            mu_par = np.zeros(shape, dtype=np.float32)
            MR = np.asarray(cor.mu_rank)
            if two_d:
                mu_par[:n_pos] = MR[rows]
            else:
                mu_par[:n_pos] = (MR[rows] if MR.ndim == 1
                                  else MR[rows, -1])
            zn_par = float(cor.rank_sign)
            prog_par = float(self.min_gap_rank)
            if zn_par == 0.0:
                raise ValueError(
                    "pair labels were asked to come from the rank target but "
                    "the measured direction is 0.0, which means the corpus "
                    "was built without establishing it. Refusing rather than "
                    "assuming a sign.")

        for i in range(n_pos - int(n_block)):
            by_genome.setdefault((int(spc[i]), int(cat_row[i])), []).append(i)
        for _g, members in sorted(by_genome.items()):
            order = list(members)
            rng.shuffle(order)
            _c = (int(_g[1]) if isinstance(_g, tuple) else -1)
            if _c < 0:
                _c = n_col - 1
            for a, b in zip(order[0::2], order[1::2]):
                dy = (float(mu_par[a, _c] - mu_par[b, _c]) if two_d
                      else float(mu_par[a] - mu_par[b]))
                # A non-finite `dy` is rejected BEFORE the threshold, because
                # `abs(nan) < prog_par` is False and `dy > 0` is False too,
                # so such a pair would reach the head with `pair_sign` -1.0
                # always: a ranking label made out of nothing, at full
                # weight. No such pairs exist on today's corpus, but
                # `mu[a, c]` is sometimes NaN once unlabelled rows enter.
                if not np.isfinite(dy):
                    continue
                if abs(dy) < prog_par:
                    continue
                pair_a.append(a)
                pair_b.append(b)
                pair_sign.append(zn_par if dy > 0 else -zn_par)
                pair_dy.append(dy)
                pair_np.append(False)
                group[a] = group[b] = gid
                gid += 1
        # Cross-genome pairs: one gene, one category, two genomes. These are
        # the only pairs in which the head must decide between two sequences
        # of the SAME gene, which is what `ortholog_fraction` buys. They feed
        # the species channel no gradient under `head_withhold: [species,
        # text]`, where the pairwise branch sees one unknown token expanded
        # over the batch. Supply is 0.706 per batch at n_blk = 4 and m = 7.
        og_gids: List[int] = []
        if int(n_block) > 0:
            by_og: Dict[Tuple[object, int], List[int]] = {}
            for i in range(n_pos - int(n_block), n_pos):
                by_og.setdefault((cor.og[rows[i]], int(cat_row[i])),
                                 []).append(i)
            for _k in sorted(by_og, key=lambda z: (str(z[0]), int(z[1]))):
                order = list(by_og[_k])
                rng.shuffle(order)
                _c = int(_k[1])
                if _c < 0:
                    _c = n_col - 1
                for a, b in zip(order[0::2], order[1::2]):
                    if int(spc[a]) == int(spc[b]):
                        continue        # same genome: not this term's job
                    dy = (float(mu_par[a, _c] - mu_par[b, _c]) if two_d
                          else float(mu_par[a] - mu_par[b]))
                    if not np.isfinite(dy):
                        continue
                    if abs(dy) < prog_par:
                        continue
                    pair_a.append(a)
                    pair_b.append(b)
                    pair_sign.append(zn_par if dy > 0 else -zn_par)
                    pair_dy.append(dy)
                    pair_np.append(False)
                    # The invariant of this pair type, asserted rather than
                    # trusted: one gene, one category, two genomes.
                    if (cor.og[rows[a]] != cor.og[rows[b]]
                            or int(cat_row[a]) != int(cat_row[b])
                            or int(spc[a]) == int(spc[b])):
                        raise RuntimeError(
                            "cross-genome pair is not one gene in two "
                            "genomes: og %r/%r cat %d/%d genome %d/%d"
                            % (cor.og[rows[a]], cor.og[rows[b]],
                               int(cat_row[a]), int(cat_row[b]),
                               int(spc[a]), int(spc[b])))
                    group[a] = group[b] = gid
                    og_gids.append(gid)
                    gid += 1
        for k, s in enumerate(neg_src if self.nonpromoter_pairs else ()):
            a = int(s)                      # the promoter
            b = n_pos + k                   # its shuffle
            pair_a.append(a)
            pair_b.append(b)
            pair_sign.append(1.0)           # a real promoter outranks a shuffle
            pair_dy.append(float("nan"))    # no measured difference exists
            pair_np.append(True)
            group[b] = group[a]

        if trow is not None and B:
            # One condition, two promoters: checked, not assumed, because a
            # per-row condition breaks 17.2 % of pairs silently. Cross-genome
            # pairs are two experiments by design and are excluded.
            order = np.argsort(group, kind="stable")
            gs, ts = group[order], trow[order]
            same = gs[1:] == gs[:-1]
            if og_gids:
                _krzyz = np.isin(gs, np.asarray(og_gids, dtype=group.dtype))
                same = same & ~_krzyz[1:] & ~_krzyz[:-1]
            bad = int((same & (ts[1:] != ts[:-1])).sum())
            if bad:
                raise RuntimeError(
                    "%d pair-group neighbours resolve different "
                    "experiment-text rows. Pairs are built inside one "
                    "(species_id, cat_row) bucket and a negative "
                    "inherits both from its source, so this can only "
                    "mean one of those two invariants changed. Two "
                    "conditions in one pair is the failure the `group` "
                    "argument of ConditionEncoder._resolve exists to "
                    "prevent." % (bad,))
            self.text_pair_groups = int(np.unique(group).size)

        t = torch.tensor
        return TrainBatch(
            x=x, target_idx=target_idx,
            cat_target=torch.from_numpy(cat_row.astype(np.int64)),
            mu_row=mu_row, mu_row_valid=mu_row_valid,
            sigma_row=sigma_row, sigma_row_valid=sigma_row_valid,
            row_text=row_text,
            is_promoter=t(is_prom),
            mu=t(mu), sigma=t(sd),
            mu_valid=t(mu_valid), sigma_valid=t(sigma_valid),
            weight=t(weight), species_id=t(spc), pair_group=t(group),
            pair_a=t(pair_a, dtype=torch.long),
            pair_b=t(pair_b, dtype=torch.long),
            pair_sign=t(pair_sign, dtype=torch.float32),
            pair_delta_y=t(pair_dy, dtype=torch.float32),
            pair_nonprom=t(pair_np, dtype=torch.bool),
            row_index=t(np.asarray(rows, dtype=np.int64)),
            negative_kind=tuple(neg_kinds),
            flank_l=flank_l, flank_r=flank_r, flank_ok=flank_ok,
            text_emb=text_emb, text_ok=text_ok,
            gene_numeric=gene_numeric, gene_ok=gene_ok)

    def _ortholog_blocks(self, split: str) -> List[np.ndarray]:
        """Rows grouped by orthogroup, keeping groups several genomes carry.

        Built from `indices(split)` alone, which is the whole leakage
        argument: a block drawn for a training batch can only contain
        training rows.
        """
        got = self._og_blocks.get(split)
        if got is not None:
            return got
        cor = self.corpus
        buck: Dict[object, List[int]] = {}
        for r in cor.indices(split):
            buck.setdefault(cor.og[int(r)], []).append(int(r))
        out: List[np.ndarray] = []
        for rs in buck.values():
            if len(rs) < 2:
                continue
            if len({int(cor.species_id[r]) for r in rs}) < 2:
                continue                # one genome: no cross-genome pair
            out.append(np.asarray(rs, dtype=np.int64))
        self._og_blocks[split] = out
        return out

    # Per-orthogroup sample weighting
    def _og_kody(self) -> np.ndarray:
        """[N] int64: the orthogroup id of every CORPUS row, computed once.

        The observable runs `np.unique` over this rather than over 137 k
        strings every epoch, for the same number.
        """
        if self._og_kod_cache is None:
            _, inv = np.unique(np.asarray(self.corpus.og), return_inverse=True)
            self._og_kod_cache = inv.astype(np.int64)
        return self._og_kod_cache

    def _og_grupy(self, split: str) -> List[np.ndarray]:
        """Rows of one split, grouped by orthogroup.

        Built from `indices(split)` alone, so k is the orthogroup's size IN
        THIS SPLIT -- the quantity that enters k_eff = k/(1+(k-1)r) -- and a
        row of another split cannot appear here.
        """
        got = self._og_grupy_cache.get(split)
        if got is not None:
            return got
        idx = np.asarray(self.corpus.indices(split), dtype=np.int64)
        kod = self._og_kody()[idx]
        porzadek = np.argsort(kod, kind="stable")
        idx = idx[porzadek]
        kod = kod[porzadek]
        _u, start = np.unique(kod, return_index=True)
        out = [np.ascontiguousarray(g)
               for g in np.split(idx, start[1:])] if idx.size else []
        self._og_grupy_cache[split] = out
        return out

    def _masa_wiersza(self) -> np.ndarray:
        """[N] float64: `sqrt(n_obs)` summed over the MEASURED cells.

        The mass a row contributes to the denominator of `masked_mean` in
        the cell regime. For the observable and the gate only: `_build`
        never reads this array, so this method cannot change training.
        """
        if self._masa_cache is None:
            mu = np.asarray(self.corpus.mu)
            w = np.sqrt(np.maximum(np.asarray(self.corpus.n_obs), 0.0))
            w = np.where(np.isfinite(mu), w, 0.0)
            if w.ndim == 2:
                w = w.sum(axis=1)
            self._masa_cache = np.asarray(w, dtype=np.float64)
        return self._masa_cache

    def _plan_zbalansowany(self, split: str) -> Tuple[int, int, int]:
        """(M, rows in the epoch, orthogroups) -- deterministic, no rng.

        M = round(N/G) draws per orthogroup, clipped to `OG_MAX_KOPII * k`
        so that a one-row orthogroup does not enter the epoch M times.
        """
        grupy = self._og_grupy(split)
        n = int(self.corpus.indices(split).size)
        G = len(grupy)
        if not G:
            return 0, n, 0
        M = max(1, int(round(float(n) / float(G))))
        total = int(sum(min(M, OG_MAX_KOPII * int(g.size)) for g in grupy))
        return M, total, G

    def _order_zbalansowany(self, split: str, seed: int) -> np.ndarray:
        """An epoch order in which every orthogroup gets the same step count.

        Equivalent to weighting the loss by 1/k, since drawing with p ~ 1/k
        has the same expected gradient. The draw is STRATIFIED rather than an
        `rng.choice` over the whole split, so the step count per orthogroup
        is exact and not merely expected. Repetitions are ordered -- first
        `m // k` whole copies of the group, then the remainder without
        replacement -- so no row enters more than `ceil(m/k)` times.

        Uses its own `rng` stream, because the epoch stream is consumed
        further on by the orthogroup blocks and the negatives.
        """
        grupy = self._og_grupy(split)
        M, _tot, _G = self._plan_zbalansowany(split)
        rng = np.random.default_rng(self.seed * 104729 + 31 * int(seed) + 17)
        czesci: List[np.ndarray] = []
        for g in grupy:
            k = int(g.size)
            if k <= 0:
                continue
            m = min(M, OG_MAX_KOPII * k)
            pelne, reszta = divmod(m, k)
            if pelne:
                czesci.append(np.tile(g, pelne))
            if reszta:
                czesci.append(rng.choice(g, size=reszta, replace=False))
        order = (np.concatenate(czesci) if czesci
                 else np.zeros(0, dtype=np.int64))
        rng.shuffle(order)
        return np.ascontiguousarray(order.astype(np.int64))

    def raport_wazenia(self, split: str, order: np.ndarray) -> str:
        """One run-log line: whether per-orthogroup weighting took effect.

        The per-orthogroup mass CV is the falsifier: it is 0.4640 on the
        train split under `sqrt_n_obs` and must fall below 0.20 under
        `sqrt_n_obs_per_og`. It is computed from the order just built, so a
        wrong implementation shows in epoch 1.
        """
        order = np.asarray(order, dtype=np.int64)
        if order.size == 0:
            return "  sample_weight[%s] %r: EMPTY ordering" % (
                split, self.sample_weight)
        masa = self._masa_wiersza()
        kod = self._og_kody()[order]
        _u, inv = np.unique(kod, return_inverse=True)
        suma = np.bincount(inv, weights=masa[order])
        sr = float(suma.mean())
        cv = float(suma.std() / sr) if sr > 0 else float("nan")
        _r, cnt = np.unique(order, return_counts=True)
        n_split = max(1, int(self.corpus.indices(split).size))
        return ("  sample_weight[%s] %r: rows %d (%.3fx of the split) | "
                "orthogroups %d of %d | weight mass per orthogroup CV %.4f "
                "(max/min %.2f) | row copies max %d, rows with >1 copy "
                "%d (%.2f %%)"
                % (split, self.sample_weight, int(order.size),
                   float(order.size) / float(n_split), int(_u.size),
                   len(self._og_grupy(split)), cv,
                   float(suma.max() / max(float(suma.min()), 1e-12)),
                   int(cnt.max()), int((cnt > 1).sum()),
                   100.0 * float((cnt > 1).mean())))

    def epoch(self, split: str, shuffle: bool = True, seed: int = 0,
              max_steps: int = 0, with_negatives: bool = True
              ) -> Iterator[TrainBatch]:
        idx = self.corpus.indices(split)
        rng = np.random.default_rng(self.seed * 7919 + int(seed))
        order = idx.copy()
        # Under `sqrt_n_obs_per_og` the epoch order is not a permutation of
        # the rows but a draw giving every orthogroup the same number of
        # gradient steps. The default branch, and the `rng` stream, are
        # untouched by it.
        if shuffle and self.sample_weight == SAMPLE_WEIGHT_PER_OG:
            order = self._order_zbalansowany(split, int(seed))
        elif shuffle:
            rng.shuffle(order)
        if shuffle:
            # Computed on the order that was just built, not on the plan;
            # otherwise the line would report what was meant to happen.
            print(self.raport_wazenia(split, order))
        # Rows reserved for orthogroup blocks, rounded DOWN to a whole number
        # of blocks so a batch never carries a half block that cannot pair.
        blocks: List[np.ndarray] = []
        n_blk = 0
        if self.ortholog_fraction > 0.0:
            blocks = self._ortholog_blocks(split)
            if blocks:
                n_blk = int(self.batch_size * self.ortholog_fraction)
                n_blk -= n_blk % self.ortholog_block
                n_blk = max(0, min(n_blk, self.batch_size - 2))
        step = self.batch_size - n_blk
        n = 0
        for start in range(0, order.size, step):
            rows = order[start:start + step]
            if rows.size < 2:
                continue                    # a pair needs two rows
            if n_blk:
                extra: List[int] = []
                guard = 0
                while len(extra) < n_blk and guard < 4 * n_blk:
                    guard += 1
                    cand = blocks[int(rng.integers(len(blocks)))]
                    ile = min(self.ortholog_block, n_blk - len(extra),
                              int(cand.size))
                    if ile < 2:
                        continue
                    sel = rng.choice(int(cand.size), size=ile, replace=False)
                    extra.extend(int(cand[s]) for s in sel)
                if extra:
                    rows = np.concatenate(
                        [rows, np.asarray(extra, dtype=order.dtype)])
                    yield self._build(rows, rng, with_negatives,
                                      n_block=len(extra))
                    n += 1
                    if max_steps and n >= max_steps:
                        return
                    continue
            yield self._build(rows, rng, with_negatives)
            n += 1
            if max_steps and n >= max_steps:
                return

    def steps_per_epoch(self, split: str, max_steps: int = 0) -> int:
        size = int(self.corpus.indices(split).size)
        if self.sample_weight == SAMPLE_WEIGHT_PER_OG:
            # A balanced epoch has a different length, and a deterministic
            # one (`_plan_zbalansowany` uses no rng), so the LR schedule can
            # start from the right number rather than correct itself later.
            size = int(self._plan_zbalansowany(split)[1])
        n = int(np.ceil(size / self.batch_size))
        return min(n, max_steps) if max_steps else n
