"""Zero-shot: score for a species, an experiment or a gene never seen.

The ten designs go to synthesis with no GFP measurement from P1 behind them,
so nothing may be fitted on the target: every quantity used at design time is
either learned from other organisms and conditions, or specified in words.

Species is a point on the Lorentz tree, so an unseen strain is placed by its
phylogenetic neighbours and not by a lookup row. The experiment is free text
through a frozen LM: categories reach at most 24.81 % of the reproducible
conditional signal and for P1 are negative (-0.0115) against 0.1812 for text.
The gene is a phyletic profile plus an English description, both of which
exist for a gene with no measurements. Flanks are absent -- a plasmid's
neighbourhood is 5.13 sd outside the genomic flank distribution, so the honest
value is the learned unknown token that `flank_drop_prob` trains.

Three protocols, each falsifying one claim. `leave_one_species_out` holds out
one genome: if the held-out ranking survives substituting the species token,
that channel is not carrying the transfer. `leave_one_study_out` holds out one
study and is the honest split for text -- a study one-hot alone scores 0.28885
under a random split and 0.00304 under a study split, ComBat having removed
batch means and not the batch x promoter interaction. `leave_one_block_out`
holds out a leakage block and is already realised as the spy split.

None of this predicts GFP fluorescence: three domain shifts sit between the
training label and the wet readout -- mRNA to protein, native locus to
plasmid, rank within a library to one absolute value. `disclaimer()` returns
that sentence so it travels with the number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "ZeroShotQuery",
    "ZeroShotScorer",
    "leave_one_out_groups",
    "transfer_report",
]

_CODE = {c: i for i, c in enumerate("ACGT")}


# A query
@dataclass
class ZeroShotQuery:
    """Everything the model is told about a design that has no measurement.

    An absent field resolves to the channel's learned unknown token, not to a
    zero vector: a projection has a bias, so a zero input is a learned
    constant that reports a context share while carrying nothing.
    """

    #: Strain name as it appears in the registry, or None for "any".
    species: Optional[str] = None
    #: Growth condition in English, through the same normalisation contract
    #: and frozen LM as the training text, so a condition the corpus never
    #: had is still placeable.
    condition_text: Optional[str] = None
    #: Orthogroup of the gene the promoter will drive.
    orthogroup: Optional[str] = None
    #: English description of the gene, for a gene outside the orthology table.
    gene_text: Optional[str] = None
    #: Defaults to False deliberately: the wet construct is a plasmid.
    #: Read only by `describe()`. `ZeroShotScorer.resolve()` and `.score()`
    #: never consult it and never pass a flank tensor, so setting it True
    #: changes the printed description and not the forward pass -- the flank
    #: tokens stay at their unknown value either way. Wiring it would mean
    #: supplying genomic flanks through the scorer, which this class does
    #: not do; the field is kept because the panel quotes it in its report.
    use_flanks: bool = False

    def describe(self) -> str:
        parts = []
        parts.append("species=%s" % (self.species or "UNKNOWN token"))
        parts.append("condition=%s" % (("%r" % self.condition_text)
                                       if self.condition_text
                                       else "UNKNOWN token"))
        parts.append("gene=%s" % (self.orthogroup or self.gene_text
                                  or "UNKNOWN token"))
        parts.append("flanks=%s" % ("genomic" if self.use_flanks
                                    else "UNKNOWN token (plasmid)"))
        return ", ".join(parts)

    @staticmethod
    def disclaimer() -> str:
        return (
            "This is a predicted ORDER on the scale the model was trained on: "
            "the rank of mRNA abundance at a native locus. The wet readout is "
            "GFP protein from a promoter::GFP plasmid, which differs in "
            "quantity (transcript -> protein, through stability, translation "
            "initiation and maturation), in context (native neighbourhood and "
            "chromatin -> integration site) and in task (rank within a library "
            "-> one absolute value in one condition). No number here is a "
            "predicted fluorescence or a predicted fold change.")


# Scoring
class ZeroShotScorer(object):
    """Turn a `ZeroShotQuery` plus sequences into predictions.

    Holds the resolvers -- strain registry, frozen text encoder, gene feature
    table -- so a caller supplies words and gets a prediction, which is the
    interface the design loop and the wet-lab hand-off both need.
    """

    def __init__(self, model, text_encoder=None, gene_features=None,
                 seq_len: int = 800, batch_size: int = 32) -> None:
        self.model = model
        self.text_encoder = text_encoder
        self.gene_features = (gene_features
                              if gene_features is not None
                              else getattr(model, "gene_features", None))
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.notes: List[str] = []

    def resolve(self, q: ZeroShotQuery) -> Dict[str, object]:
        """Query -> the tensors `HyppeModel.forward` takes, plus a report.

        Anything that cannot be resolved is left as None and recorded. A query
        naming a strain the registry does not have must not quietly become
        strain 0, which is a different organism.
        """
        out: Dict[str, object] = {}
        notes: List[str] = []

        names = list(getattr(self.model, "strain_names", ()) or ())
        if q.species is not None:
            if q.species in names:
                out["species_index"] = names.index(q.species)
            else:
                notes.append(
                    "strain %r is not in the registry (%d strains). The "
                    "species token falls back to its UNKNOWN value, which is "
                    "the honest answer for an unplaced strain -- scoring it as "
                    "strain 0 would be scoring a different organism."
                    % (q.species, len(names)))

        if q.condition_text is not None:
            if self.text_encoder is None:
                notes.append(
                    "a condition text was given but no frozen text encoder is "
                    "attached, so the text token is its UNKNOWN value and the "
                    "condition was NOT used")
            else:
                out["text_emb"] = self.text_encoder.encode(
                    [q.condition_text])[0].astype(np.float32)

        if q.orthogroup is not None and self.gene_features is not None:
            num, _ident, ok = self.gene_features.vectors_for([q.orthogroup])
            if bool(ok[0]):
                out["gene_numeric"] = num[0]
            else:
                notes.append(
                    "orthogroup %r is not in the orthology table; the gene "
                    "token falls back to its UNKNOWN value unless a "
                    "gene_text is supplied" % (q.orthogroup,))
        if q.gene_text is not None and self.text_encoder is not None:
            out["gene_text_emb"] = self.text_encoder.encode(
                [q.gene_text])[0].astype(np.float32)

        out["notes"] = notes
        out["described"] = q.describe()
        self.notes = notes
        return out

    def one_hot(self, seqs: Sequence[str]) -> torch.Tensor:
        n = len(seqs)
        out = np.full((n, 4, self.seq_len), 0.25, dtype=np.float32)
        for i, s in enumerate(seqs):
            s = s.upper()[-self.seq_len:]
            off = self.seq_len - len(s)
            for j, ch in enumerate(s):
                k = _CODE.get(ch)
                if k is not None:
                    out[i, :, off + j] = 0.0
                    out[i, k, off + j] = 1.0
        return torch.from_numpy(out)

    @torch.no_grad()
    def score(self, sequences: Sequence[str], query: ZeroShotQuery
              ) -> Dict[str, np.ndarray]:
        """`{mu, sigma, log_var, promoterness}` arrays aligned to `sequences`.

        `sigma` is `exp(0.5 * log_var)`, the model's own uncertainty, which
        the "highly stable promoter" design slot reads directly.

        With a condition text the head is driven per row and `mu` is one
        number per sequence. Without one it emits [B, K] over the category
        prototypes, and that case returns `mu_by_category` and
        `category_names` but no `mu`: a flattened [B, K] block is K times
        longer than `sequences` and misaligned after the first row, which a
        caller reading `res["mu"]` could not tell from a valid answer.
        """
        r = self.resolve(query)
        model = self.model
        _head = getattr(model, "absolute", None)
        per_row = ("text_emb" in r
                   and bool(getattr(_head, "text_conditioned", False)))
        was = model.training
        model.eval()
        device = next(model.parameters()).device
        mu: List[np.ndarray] = []
        lv: List[np.ndarray] = []
        pr: List[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            chunk = list(sequences[start:start + self.batch_size])
            x = self.one_hot(chunk).to(device)
            b = x.shape[0]
            kw: Dict[str, object] = {"reconstruct": False}
            if "species_index" in r:
                kw["species_id"] = torch.full((b,), int(r["species_index"]),
                                              dtype=torch.long, device=device)
            if "text_emb" in r:
                _t = torch.from_numpy(
                    np.repeat(np.asarray(r["text_emb"])[None, :], b, 0)
                ).to(device)
                kw["text_emb"] = _t
                # The same embedding twice, in two roles: `text_emb`
                # conditions the latent through FiLM, `row_text` generates the
                # output column. Passing only the first leaves the head
                # reading K category columns for a query that named one.
                if per_row:
                    kw["row_text"] = _t
            if "gene_numeric" in r and getattr(model, "gene", None) is not None:
                kw["gene_numeric"] = torch.from_numpy(
                    np.repeat(np.asarray(r["gene_numeric"])[None, :], b, 0)
                ).to(device)
                kw["gene_ok"] = torch.ones(b, device=device)
            if "gene_text_emb" in r and getattr(model, "gene", None) is not None:
                kw["gene_text"] = torch.from_numpy(
                    np.repeat(np.asarray(r["gene_text_emb"])[None, :], b, 0)
                ).to(device)
            out = model(x, **kw)
            if out.get("absolute") is not None:
                _mu = out["absolute"].mu.detach().cpu().numpy()
                if per_row and _mu.shape[0] != b:
                    raise RuntimeError(
                        "the per-row regime returned %s for %d sequences; "
                        "the head did not take row_text" % (_mu.shape, b))
                mu.append(_mu.reshape(b) if per_row else _mu.reshape(b, -1))
                if getattr(out["absolute"], "log_var", None) is not None:
                    # The variance keeps the shape of the mean it belongs to.
                    # Pinning it to [B] would reinterpret K category
                    # variances as K sequences.
                    _lv = out["absolute"].log_var.detach().cpu().numpy()
                    lv.append(_lv.reshape(b) if per_row
                              else _lv.reshape(b, -1))
            if out.get("discriminator") is not None:
                pr.append(out["discriminator"].logit.detach().cpu()
                          .numpy().ravel())
        model.train(was)
        res: Dict[str, np.ndarray] = {}
        if mu:
            _M = np.concatenate(mu, axis=0)
            if per_row:
                res["mu"] = _M
            else:
                res["mu_by_category"] = _M
                res["category_names"] = np.asarray(
                    getattr(model, "category_names", ()) or
                    ["col%d" % i for i in range(_M.shape[1])])
                self.notes.append(
                    "no condition text was resolved, so the head could not "
                    "generate a column and returned one value per CATEGORY. "
                    "The result carries `mu_by_category` [%d, %d] and no "
                    "`mu`." % _M.shape)
        if lv:
            _L = np.concatenate(lv, axis=0)
            _k = "log_var" if per_row else "log_var_by_category"
            res[_k] = _L
            res["sigma" if per_row else "sigma_by_category"] = np.exp(0.5 * _L)
        if pr:
            res["promoterness"] = np.concatenate(pr)
        return res

    def report(self, query: ZeroShotQuery) -> Dict[str, object]:
        return {"query": query.describe(),
                "unresolved": list(self.notes),
                "disclaimer": ZeroShotQuery.disclaimer()}


# Protocols
def leave_one_out_groups(keys: Sequence[str], min_size: int = 20
                         ) -> List[Tuple[str, np.ndarray]]:
    """`[(group, row indices)]` for every group big enough to be scored.

    Groups below `min_size` are dropped rather than merged into an "other"
    bucket: on eight rows the standard error exceeds the effect, and an
    "other" group is a mixture rather than a held-out condition.
    """
    k = np.asarray([str(x) for x in keys])
    out: List[Tuple[str, np.ndarray]] = []
    for g in sorted(set(k.tolist())):
        idx = np.nonzero(k == g)[0]
        if idx.size >= int(min_size):
            out.append((g, idx))
    return out


def transfer_report(held_out: Mapping[str, float],
                    substituted: Mapping[str, float],
                    metric: str = "scc") -> Dict[str, object]:
    """Does the channel carry the transfer, or is the sequence doing it?

    `held_out` is the metric on the held-out group with the correct channel
    value, `substituted` the same rows scored with another group's. Equal
    numbers mean the channel is not what made the transfer work -- the one
    comparison that separates "generalises to a new species" from "ignores
    species, and the promoters happen to be similar".
    """
    a = float(held_out.get(metric, float("nan")))
    b = float(substituted.get(metric, float("nan")))
    gap = a - b
    verdict = "channel carries the transfer"
    if not np.isfinite(gap):
        verdict = "not measurable (a metric is missing)"
    elif abs(gap) < 0.01:
        verdict = ("channel is NOT carrying the transfer: substituting it "
                   "changes the held-out metric by %.4f, so the result is a "
                   "property of the sequences" % (gap,))
    elif gap < 0:
        verdict = ("substituting the channel IMPROVED the held-out metric by "
                   "%.4f: the channel is actively misleading on this group"
                   % (-gap,))
    return {"metric": metric, "with_correct_channel": a,
            "with_substituted_channel": b, "gap": gap, "verdict": verdict}
