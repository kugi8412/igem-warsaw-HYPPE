"""Frozen DNA language-model embeddings for genomic neighbourhood context.

Paired ablations over 80 fixed batches (baseline SCC 0.74885) rank the
conditioning channels unambiguously:

    channel                 zeroed     shuffled    substituted
    gene context            -0.0875    -0.1692     -0.1819
    latent                  -0.0109    -0.0730       --
    genome (species tree)   -0.0020    +0.0000     -0.0024
    experiment description  -0.0029    +0.0000     -0.0024

Substituting a different gene costs 24 % of the metric, an order of magnitude
more than any other channel, so the genomic neighbourhood is the one worth
precomputing first. Two warnings carry over from those measurements, both
having produced wrong conclusions before. The `+0.00000` entries are an empty
intervention rather than a dead channel -- batches are genome-homogeneous, so
shuffling the genome within one changes nothing by construction, and a
conditioning channel must be ablated by substitution. And "the predictor does
not need the latent" does not carry over from the DeepSTARR track: here
shuffling the latent costs 6.7x more than zeroing it.

The model choice is constrained by the environment rather than preference.
DNABERT-2-117M and the nucleotide transformer both fail to load on the target
machine (incompatible remote code; a missing `rope_theta`), while
`zhihan1996/DNA_bert_6` and `_3` load with a standard `BertModel` and no
remote code. DNABERT v1 6-mer is therefore the default, and `model_id` stays
configurable so a newer encoder can replace it without touching this file.

Windowing is forced by the 512-token limit: DNABERT v1 encodes overlapping
k-mers, so a 1000 bp flank is 995 tokens and truncating would silently discard
half of it. The sequence is covered by overlapping windows whose pooled
vectors are averaged, and the window count and overlap are reported.

Mean-pooled transformer embeddings are famously anisotropic: they can collapse
so that every input maps to nearly the same direction, which measures nothing
while looking healthy downstream. :meth:`DnaLmEmbedder.embedding_report`
therefore reports `cos_mean` and `cos_p95` over distinct sequences (near 1.0
means collapsed), `eff_rank`, the exp of the spectral entropy (near 1 means
one direction explains everything), and `cos_rc` against the reverse
complement, which must be clearly below 1.0 or orientation is not represented
at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Reverse complement over the ACGTN alphabet, case preserved."""
    return seq.translate(_COMPLEMENT)[::-1]


def kmerise(seq: str, k: int) -> List[str]:
    """Overlapping k-mers, the tokenisation DNABERT v1 was trained with."""
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


@dataclass
class DnaLmConfig:
    """Everything this module reads. No hidden defaults elsewhere."""

    model_id: str = "zhihan1996/DNA_bert_6"
    kmer: int = 6
    max_tokens: int = 512          # hard model limit, not a preference
    window_overlap_tokens: int = 32
    batch_size: int = 64
    dtype: str = "float16"         # storage dtype for the output matrix
    device: str = "cuda"
    pooling: str = "mean"          # "mean" (masked) or "cls"
    uppercase: bool = True         # soft-masked lowercase is still sequence
    # Sequences shorter than this are reported and embedded as-is rather than
    # padded, because silent padding would invent context that does not exist.
    min_length: int = 32

    def __post_init__(self) -> None:
        if self.pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls', got %r"
                             % (self.pooling,))
        if self.window_overlap_tokens >= self.max_tokens:
            raise ValueError(
                "window_overlap_tokens (%d) must be smaller than max_tokens "
                "(%d), otherwise windows never advance"
                % (self.window_overlap_tokens, self.max_tokens))
        if self.kmer < 1:
            raise ValueError("kmer must be >= 1")


@dataclass
class WindowPlan:
    """How one sequence is covered. Reported so truncation is never silent."""

    n_windows: int
    token_starts: List[int]
    n_tokens: int
    truncated: bool = False


def plan_windows(n_tokens: int, cfg: DnaLmConfig) -> WindowPlan:
    """Cover `n_tokens` with overlapping windows of at most `max_tokens`.

    The last window is anchored to the end of the sequence rather than left
    where the stride lands, so the terminal bases are always seen: for a flank
    that terminus is the junction with the promoter, and dropping it would
    bias the context vector.
    """
    body = cfg.max_tokens - 2                      # [CLS] and [SEP]
    if n_tokens <= body:
        return WindowPlan(1, [0], n_tokens)
    stride = body - cfg.window_overlap_tokens
    starts = list(range(0, n_tokens - body, stride))
    last = n_tokens - body                         # end-anchored start
    # Append the end-anchored window only if it advances materially. Without
    # this, a 995-token flank gives starts [0, 478, 485], whose last two
    # windows overlap by 98 %: a third of the compute wasted and, worse, the
    # tail double-weighted in the average. Dropping it instead leaves the tail
    # uncovered, so the last window is slid to the end. `starts[-1] > 0`
    # guards the case where sliding the only window would uncover the start.
    min_advance = max(1, cfg.window_overlap_tokens)
    if starts and starts[-1] + body < n_tokens:
        if starts[-1] > 0 and last - starts[-1] < min_advance:
            starts[-1] = last
        else:
            starts.append(last)
    elif not starts:
        starts.append(last)
    return WindowPlan(len(starts), starts, n_tokens)


class DnaLmEmbedder:
    """Frozen DNA LM producing one pooled vector per input sequence."""

    def __init__(self, cfg: Optional[DnaLmConfig] = None) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg or DnaLmConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModel.from_pretrained(self.cfg.model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.hidden_size = int(self.model.config.hidden_size)
        # Reported, not assumed: a mismatch here silently truncates every input.
        self.model_max_tokens = int(
            getattr(self.model.config, "max_position_embeddings", 512))
        self.stats = {"n_sequences": 0, "n_windows": 0, "n_short": 0,
                      "n_multi_window": 0, "n_truncated": 0}

    # Core
    def _encode_windows(self, kmers: Sequence[str],
                        plan: WindowPlan) -> List[str]:
        body = self.cfg.max_tokens - 2
        return [" ".join(kmers[s:s + body]) for s in plan.token_starts]

    @torch.no_grad()
    def _forward_batch(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=self.cfg.max_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state")
                  else out[0])
        if self.cfg.pooling == "cls":
            return hidden[:, 0].float()
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        # Masked mean: without it a short window is pulled towards the
        # padding embedding, i.e. averaged over positions that do not exist.
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (summed / denom).float()

    @torch.no_grad()
    def embed(self, sequences: Iterable[str]) -> np.ndarray:
        """Embed sequences in order. Returns [N, hidden] in cfg.dtype."""
        seqs = list(sequences)
        texts: List[str] = []
        owner: List[int] = []          # which sequence each window belongs to
        for i, raw in enumerate(seqs):
            s = raw.upper() if self.cfg.uppercase else raw
            if len(s) < self.cfg.min_length:
                self.stats["n_short"] += 1
            kmers = kmerise(s, self.cfg.kmer)
            if not kmers:
                # Cannot be embedded at all; left as zeros so the caller sees
                # a missing row instead of a plausible wrong one.
                continue
            plan = plan_windows(len(kmers), self.cfg)
            if plan.n_windows > 1:
                self.stats["n_multi_window"] += 1
            self.stats["n_windows"] += plan.n_windows
            for t in self._encode_windows(kmers, plan):
                texts.append(t)
                owner.append(i)
        self.stats["n_sequences"] += len(seqs)

        acc = torch.zeros(len(seqs), self.hidden_size, dtype=torch.float32)
        cnt = torch.zeros(len(seqs), dtype=torch.float32)
        bs = max(1, int(self.cfg.batch_size))
        for a in range(0, len(texts), bs):
            chunk = texts[a:a + bs]
            vec = self._forward_batch(chunk).cpu()
            for j, o in enumerate(owner[a:a + bs]):
                acc[o] += vec[j]
                cnt[o] += 1.0
        out = acc / cnt.clamp_min(1.0).unsqueeze(-1)
        return out.numpy().astype(self.cfg.dtype)

    # Falsifiers
    @torch.no_grad()
    def embedding_report(self, sequences: Sequence[str],
                         n_sample: int = 256) -> dict:
        """Anisotropy and orientation checks. See the module docstring."""
        seqs = list(sequences)[:n_sample]
        if len(seqs) < 8:
            return {"error": "need at least 8 sequences for a report"}
        emb = self.embed(seqs).astype(np.float32)
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        cos = norm @ norm.T
        iu = np.triu_indices(len(seqs), k=1)
        off = cos[iu]
        # Effective rank: exp of the entropy of the normalised spectrum.
        centred = emb - emb.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(centred, compute_uv=False)
        p = (sv ** 2)
        p = p / max(p.sum(), 1e-12)
        p = p[p > 0]
        eff_rank = float(math.exp(-(p * np.log(p)).sum()))
        rc = self.embed([reverse_complement(s) for s in seqs[:32]]
                        ).astype(np.float32)
        fw = emb[:32]
        rc_n = rc / (np.linalg.norm(rc, axis=1, keepdims=True) + 1e-9)
        fw_n = fw / (np.linalg.norm(fw, axis=1, keepdims=True) + 1e-9)
        cos_rc = float((fw_n * rc_n).sum(axis=1).mean())
        return {
            "n": len(seqs),
            "hidden": self.hidden_size,
            "cos_mean": float(off.mean()),
            "cos_p95": float(np.percentile(off, 95)),
            "cos_min": float(off.min()),
            "eff_rank": eff_rank,
            "eff_rank_frac": eff_rank / float(self.hidden_size),
            "cos_rc": cos_rc,
            "stats": dict(self.stats),
        }
