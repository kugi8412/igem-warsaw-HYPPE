"""Frozen DNABERT embeddings for HYPPE promoters, with a 6-mer baseline.

The baseline is a condition and not an add-on: this repository has already
measured frozen foundation-model embeddings (D3) at 0.2348 in a linear probe
against 0.4716 for plain 6-mer composition, so the free feature was twice as
good. A DNABERT run without that control on the same sequences and the same
split would be unassessable, and `compute` therefore always writes 6-mer
shards beside the embeddings and refuses to start if it cannot.

Measured properties of the input (`--data-report`): Promotor is 238 831 rows
of which 99.84 % are exactly 800 bp -- the "1000 bp" figure that circulates
here belongs to the flanks. 950 of 237 780 IDs repeat, so deduplication is
explicit and reported. The alphabet carries 7.4 % lower-case letters (repeat
soft-masking) and 257 737 'N'. The DNABERT-2 vocabulary has no lower case, so
without `.upper()` 7.4 % of the corpus becomes [UNK] and scrambles the BPE
merges around it; `unk_fraction` is in the index to catch that if `upper()`
ever disappears.

Measured with the DNABERT-2 tokenizer (`--tokenizer-report`): 800 bp gives a
median of 167 tokens and a maximum of 176 (+2 special) against a budget of
512, so the whole promoter fits in one pass with no chunking. A single SNP
moves 1.80 tokens on average over 8.74 bp and changes the token count in
21.5 % of cases -- BPE is not local, and this script does not fix that, only
measures whether the frozen features contribute anything.

This module does not make a split. It reads the canonical cluster split
through `splits_build.load_split()`; the `split` column in `index.tsv` is a
copy, and on a conflict the truth is in `data/splits/homology_clusters/`.

`ONE_HOT_ORDER` goes into the manifest so that no consumer has to guess it,
because two sources in this repository disagree: `hyppe/dna.py` says "ACTG"
and `hyppe/data/sequences.py` says "ACGT". The legacy "ACTG" is kept, because
the existing artefacts were written with it and `analysis.promoter_boundary`
decodes `cache/ml_dataset_800.npz` that way. A ridge probe is invariant to
the choice (it permutes feature columns), but a tensor handed to another
module is not, so the smoke test prints both rather than picking quietly.

Usage: build the split once with `python -m hyppe.analysis.splits_build`,
then `embeddings smoke` for a tiny CPU check and `embeddings compute` for the
full run. Weight licences are in `MODELS` below.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import paths
from . import splits_build as splits


# Model catalogue; licences checked in the LICENSE file on HuggingFace.
@dataclass(frozen=True)
class Variant:
    name: str
    repo: str
    licence: str
    licence_source: str
    tokenisation: str
    token_limit: int
    hidden: int
    #: how many bp fit in `token_limit`; None = measure with the tokenizer
    bp_per_token: float | None
    notes: str


MODELS: dict[str, Variant] = {
    # Recommended: 800 bp is a median of 167 tokens, so the whole promoter
    # fits without chunking, and ALiBi makes 512 a soft limit.
    "dnabert2": Variant(
        name="DNABERT-2-117M",
        repo="zhihan1996/DNABERT-2-117M",
        licence="Apache-2.0",
        licence_source="https://huggingface.co/zhihan1996/DNABERT-2-117M/raw/main/LICENSE",
        tokenisation="BPE, vocab 4096",
        token_limit=512,
        hidden=768,
        bp_per_token=4.79,
        notes=("ALiBi (alibi_starting_size 512) -> the limit is soft. "
               "The pooler is NOT in the checkpoint (random initialisation) -- "
               "we use the [CLS] hidden state, NEVER pooler_output."),
    ),
    # DNABERT-2's tokenizer and architecture, contrastively fine-tuned to
    # distinguish species -- for us the strain is a confounder, so this is
    # the wrong inductive bias.  An arm for testing that claim, not a
    # candidate.
    "dnabert-s": Variant(
        name="DNABERT-S",
        repo="zhihan1996/DNABERT-S",
        licence="Apache-2.0",
        licence_source="https://huggingface.co/api/models/zhihan1996/DNABERT-S (cardData.license)",
        tokenisation="BPE, vocab 4096 (the same as DNABERT-2)",
        token_limit=512,
        hidden=768,
        bp_per_token=4.79,
        notes=("the contrastive objective is species separation; for us the "
               "genome is a confounder, not signal.  Falsifier: if these "
               "embeddings predict the GENOME better than DNABERT-2 does, "
               "while not predicting expression better -- they carry the "
               "confounder."),
    ),
    # DNABERT v1, k=6, stride 1: 800 bp is 797 tokens against a hard limit of
    # 512 (absolute positions, no ALiBi), so this one must chunk.
    "dnabert1-6": Variant(
        name="DNABERT (v1) k=6",
        repo="zhihan1996/DNA_bert_6",
        licence="Apache-2.0",
        licence_source="https://huggingface.co/zhihan1996/DNA_bert_6/raw/main/LICENSE",
        tokenisation="k-mer k=6, stride 1, vocab 4101",
        token_limit=512,
        hidden=768,
        bp_per_token=1.0,
        notes=("max_position_embeddings=512 with ABSOLUTE positions -> a hard "
               "limit.  510 tokens = 515 bp, i.e. 64 % of the promoter in one "
               "window.  Chunking here is explicit (`--chunking`), never the "
               "default."),
    ),
}

DEFAULT_MODEL = "dnabert2"
SEED = 7898                              # the project seed (CLAUDE.md)

# One-hot channel order; two sources in this repository disagree and the
# smoke test reports both.  See the module docstring.
ONE_HOT_ORDER = "ACTG"
ONE_HOT_SOURCE_LEGACY = ("hyppe/dna.py", 20)          # legacy tree, "ACTG"
ONE_HOT_SOURCE_CANONICAL = ("hyppe/data/sequences.py", 60)   # this tree, "ACGT"

# The 6-mer index uses A,C,G,T -- arbitrary, and different from the one-hot
# order, so it goes into the manifest.
KMER_ORDER = "ACGT"
K = 6
KMER_DIM = 4 ** K                        # 4096


# Reading the sequences
@dataclass
class Corpus:
    ids: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    genome: list[str] = field(default_factory=list)
    og: list[str] = field(default_factory=list)
    gene: list[str] = field(default_factory=list)
    seq: list[str] = field(default_factory=list)
    # What was dropped and why, for the manifest.
    n_group_rows: int = 0
    n_duplicates: int = 0
    n_too_short: int = 0
    n_lower_case: int = 0
    n_bases_N: int = 0

    def __len__(self) -> int:
        return len(self.ids)


def load_corpus(path: str | None = None, group: str = "Promotor",
                min_bp: int = 50, limit: int | None = None,
                every_n: int = 1) -> Corpus:
    """Streaming read of one group from sequences.tsv.gz.

    Deduplication by ID keeps the first occurrence in file order; the rest
    are counted into `n_duplicates`, which goes into the manifest.

    `every_n` takes every nth row of the group, and is a separate knob from
    `limit` because the file is ordered by genome: the first 20 000 promoters
    are 2 genomes of 19, so `limit` alone gives a prefix and not a sample.
    """
    p = paths.require(path or paths.SEQUENCES, "sequences.tsv.gz")
    k = Corpus()
    seen: set[str] = set()
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        pos = {n: i for i, n in enumerate(hdr)}
        for needed in ("ID", "SPECIES", "GENOME", "SEQUENCE", "GROUP",
                       "OG_ID", "GENE_ID"):
            if needed not in pos:
                raise KeyError("sequences.tsv.gz has no column %r; header=%s"
                               % (needed, hdr))
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if c[pos["GROUP"]] != group:
                continue
            k.n_group_rows += 1
            if every_n > 1 and (k.n_group_rows - 1) % every_n:
                continue
            ident = c[pos["ID"]]
            if ident in seen:
                k.n_duplicates += 1
                continue
            s = c[pos["SEQUENCE"]]
            n_lower = sum(1 for ch in s if ch.islower())
            s = s.upper()
            if len(s) < min_bp:
                k.n_too_short += 1
                continue
            seen.add(ident)
            k.n_lower_case += n_lower
            k.n_bases_N += s.count("N")
            k.ids.append(ident)
            k.species.append(c[pos["SPECIES"]])
            k.genome.append(c[pos["GENOME"]])
            k.og.append(c[pos["OG_ID"]])
            k.gene.append(c[pos["GENE_ID"]])
            k.seq.append(s)
            if limit is not None and len(k.ids) >= limit:
                break
    if not len(k):
        raise RuntimeError("group %r has not a single row in %s" % (group, p))
    return k


# The split has one source: hyppe/analysis/splits_build.py.  An earlier
# version of this module drew its own three-way split by OG_ID, which closed
# orthologues but left the 18 764 cross-orthogroup pairs at Jaccard >= 0.05
# open, and had nowhere to put the distinction between choosing a
# hyperparameter and choosing between models.  That manifest is archived
# under `data/splits/homology_clusters/archive/` rather than deleted.
SPLIT_DIR = splits.SPLIT_DIR


def load_split(directory: str | None = None,
               allow_incomplete: bool = False) -> dict:
    """ID -> canonical part, from the canonical cluster split.

    Returns the dictionary `compute` uses: labels, clusters, manifest,
    directory, splits, grouping.  Part semantics are in
    `splits_build.SPLIT_ROLES`.

    `splits_build.load_split` already guards the manifest; the guard is
    called again here so this module does not depend on nobody ever
    loosening it.
    """
    p = splits.load_split(directory, allow_incomplete=allow_incomplete)
    splits.guard_manifest(p.manifest, directory or splits.MANIFEST_JSON,
                          allow_incomplete=allow_incomplete)
    present = tuple(sorted(set(p.rows["split"])))
    if present != tuple(sorted(splits.SPLITS)):
        raise ValueError(
            "the split has parts %s, and must have four: %s"
            % (list(present), list(splits.SPLITS)))
    return {
        "labels": p.id_to_split(),
        "clusters": p.id_to_cluster(),
        "manifest": p.manifest,
        "directory": directory or splits.SPLIT_DIR,
        "splits": splits.SPLITS,
        "grouping": "homology_cluster",
    }


def _sha1_of_file(p: str, header_only: bool = False) -> str:
    """SHA1 of a file; `header_only` uses the first 8 MiB + the size, because
    a full hash of a 281 MB gz takes ~2 s on every run."""
    h = hashlib.sha1()
    h.update(str(os.path.getsize(p)).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(8 << 20) if header_only else fh.read())
    return h.hexdigest()


# The 6-mer baseline
_KMER_MAP = np.full(256, -1, dtype=np.int16)
for _i, _c in enumerate(KMER_ORDER):
    _KMER_MAP[ord(_c)] = _i


def kmer_counts(s: str, k: int = K) -> np.ndarray:
    """Vector of k-mer counts, length 4**k, dtype uint16.

    Windows containing anything outside ACGT are skipped.  The maximum count
    for an 800 bp promoter is 795, so uint16 is exact: the baseline is the
    control and must not be handicapped by quantisation.

    The vector has 4**k = 4096 entries and is not collapsed onto the reverse
    complement.  The HYPPE corpus is consistently oriented (mirror test
    -0.03 against +0.99), so an RC collapse would erase strand information.
    """
    code = _KMER_MAP[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
    n = code.shape[0] - k + 1
    if n <= 0:
        return np.zeros(4 ** k, dtype=np.uint16)
    windows = np.lib.stride_tricks.sliding_window_view(code, k)     # [n, k]
    good = (windows >= 0).all(axis=1)
    if not good.any():
        return np.zeros(4 ** k, dtype=np.uint16)
    weights = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    idx = (windows[good].astype(np.int64) * weights).sum(axis=1)
    out = np.bincount(idx, minlength=4 ** k)
    if out.max() > np.iinfo(np.uint16).max:
        raise OverflowError("a k-mer count of %d exceeds uint16 -- the "
                            "sequence is longer than assumed" % out.max())
    return out.astype(np.uint16)


def kmer_block(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), KMER_DIM), dtype=np.uint16)
    for i, s in enumerate(sequences):
        out[i] = kmer_counts(s)
    return out


# The model
def _local_environment_workaround():
    """Explicit workaround for two environment faults.  Off by default.

    (a) A torchvision built against a different torch raises on import, and
        transformers imports torchvision at module level.
    (b) DNABERT's `bert_layers.py` imports `.flash_attn_triton`, which
        imports triton at module level.  `check_imports` walks that file
        recursively and demands triton even though the code has a pytorch
        fallback, and there is no triton wheel on Windows.

    Neither is needed on a clean Linux env; this is only the path that lets
    the smoke test pass on a development machine.
    """
    import importlib.machinery
    import types

    print("!! --workaround: substituting stubs for torchvision and triton.",
          file=sys.stderr)
    print("!! This is a LOCAL-MACHINE workaround, not part of the method. "
          "On dendrite it must be OFF.", file=sys.stderr)

    class _LooseMeta(type):
        def __getattr__(cls, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return n

    class _LooseModule(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return _LooseMeta(n, (), {})

    def _mk(name):
        m = _LooseModule(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []
        sys.modules[name] = m
        return m

    if "torchvision" not in sys.modules:
        tv = _mk("torchvision")
        tv.__version__ = "0.0.0-stub"
        for s in ("io", "ops", "transforms", "transforms.functional",
                  "transforms.v2", "transforms.v2.functional",
                  "utils", "datasets", "models"):
            m = _mk("torchvision." + s)
            parent = (sys.modules["torchvision." + s.rsplit(".", 1)[0]]
                      if "." in s else tv)
            object.__setattr__(parent, s.rsplit(".", 1)[-1], m)


def _triton_stub():
    """Must be called AFTER importing transformers, otherwise it poisons
    torch.inductor."""
    import importlib.machinery
    import types
    if "triton" in sys.modules:
        return
    m = types.ModuleType("triton")
    m.__spec__ = importlib.machinery.ModuleSpec("triton", None)
    m.__version__ = "0.0.0-stub"
    # Deliberately without `triton.language`: the import in
    # flash_attn_triton.py then raises ModuleNotFoundError, which is an
    # ImportError and lands in bert_layers' pytorch fallback.
    sys.modules["triton"] = m


def load_model(variant: Variant, device: str, workaround: bool):
    """Returns (tokenizer, model, info).  Raises if anything is off."""
    if workaround:
        _local_environment_workaround()
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if workaround:
        _triton_stub()

    tok = AutoTokenizer.from_pretrained(variant.repo, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(variant.repo, trust_remote_code=True)
    # transformers >= 4.29 no longer injects pad_token_id into a remote
    # config, while `bert_layers.BertEmbeddings.__init__` reads it, so it is
    # set explicitly from the tokenizer rather than left to a default.
    if getattr(cfg, "pad_token_id", None) is None:
        if tok.pad_token_id is None:
            raise RuntimeError("tokenizer %s has no pad_token_id"
                               % variant.repo)
        cfg.pad_token_id = tok.pad_token_id
    # `output_loading_info` gives the list of missing keys.  Catching this
    # through `warnings` does not work -- transformers announces "newly
    # initialized" through the logger, so `catch_warnings` sees nothing and
    # the manifest would claim the pooler was in the checkpoint when it was
    # not.  Ask about the fact, not about the message.
    model, loading = AutoModel.from_pretrained(
        variant.repo, trust_remote_code=True, config=cfg,
        output_loading_info=True)
    missing = list(loading.get("missing_keys", []))
    unexpected = list(loading.get("unexpected_keys", []))
    model = model.eval().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    hidden = int(getattr(cfg, "hidden_size"))
    if hidden != variant.hidden:
        raise ValueError("hidden_size %d != expected %d for %s"
                         % (hidden, variant.hidden, variant.name))
    info = {
        "repo": variant.repo,
        "parameters": n_par,
        "hidden_size": hidden,
        "licence": variant.licence,
        "licence_source": variant.licence_source,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "pooler_in_checkpoint": not any("pooler" in k for k in missing),
    }
    not_pooler = [k for k in missing if "pooler" not in k]
    if not_pooler:
        raise RuntimeError(
            "checkpoint %s did not supply %d weights OUTSIDE the pooler: %s. "
            "A randomly initialised layer in the middle of the encoder gives "
            "features that are not this model's features -- not something one "
            "may pass over in silence."
            % (variant.repo, len(not_pooler), not_pooler[:8]))
    if not info["pooler_in_checkpoint"]:
        print("NOTE: the pooler is NOT in the checkpoint (%s) -- it is random. "
              "That is why we pool the [CLS] HIDDEN STATE and NEVER "
              "pooler_output." % [k for k in missing if "pooler" in k])
    return tok, model, info


def _pools(h, mask):
    """(cls, mean, max) from [B, L, D] and a mask [B, L].

    `mean` and `max` run over content tokens only; [CLS] and [SEP] are
    excluded, otherwise the mean mixes a sentence token with content.
    """
    import torch
    m = mask.clone()
    m[:, 0] = 0                                   # [CLS]
    ln = mask.sum(dim=1) - 1                      # index of [SEP]
    m[torch.arange(m.shape[0], device=m.device), ln.clamp(min=0)] = 0
    if (m.sum(dim=1) == 0).any():
        raise RuntimeError("a sequence with no content token at all")
    w = m.unsqueeze(-1).to(h.dtype)
    cls = h[:, 0, :]
    mean = (h * w).sum(dim=1) / w.sum(dim=1)
    mx = h.masked_fill(w == 0, float("-inf")).max(dim=1).values
    return cls, mean, mx


# The run
POOLS = ("cls", "mean", "max")


def _shard_file(d, model, pool, nr):
    return os.path.join(d, "emb_%s_%s_%05d.npy" % (model, pool, nr))


def _kmer_file(d, nr):
    return os.path.join(d, "kmer%d_%05d.npy" % (K, nr))


def _save_atomic(path: str, arr: np.ndarray):
    """Write through a temporary file and rename, so a shard that exists is
    complete.  Otherwise resuming after a killed run would load a truncated
    array with no way to notice.

    `np.save` appends '.npy' when the name lacks it, so the temporary name
    carries the extension explicitly.
    """
    if not path.endswith(".npy"):
        raise ValueError("shards are saved as .npy, got %r" % path)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _shard_complete(d, model, nr, n_rows, dim) -> bool:
    """A shard is done only when every pool and the 6-mers exist and have the
    right shape; the existence of a file is not enough."""
    files = [_shard_file(d, model, p, nr) for p in POOLS] + [_kmer_file(d, nr)]
    shapes = [(n_rows, dim)] * len(POOLS) + [(n_rows, KMER_DIM)]
    for p, sh in zip(files, shapes):
        if not os.path.exists(p):
            return False
        try:
            k = np.load(p, mmap_mode="r").shape
        except Exception as e:                     # a corrupted file
            raise RuntimeError("shard %s cannot be loaded (%s) -- delete it "
                               "and repeat the run" % (p, e))
        if tuple(k) != sh:
            raise RuntimeError("shard %s has shape %s, expected %s -- the "
                               "output directory mixes two different runs"
                               % (p, k, sh))
    return True


def estimate(n_seq: int, variant: Variant, batch: int, seq_per_s: float,
             n_map: int, tok_len: int) -> dict:
    """VRAM, time and disk estimate, before the run starts."""
    # DNABERT-S is DNABERT-2's architecture fine-tuned, so it carries the same
    # 117M weights; matching on "DNABERT-2" alone sized it as the 110M v1.
    weights_gib = (117.07e6 * 4 / 2 ** 30
                   if variant.name.startswith(("DNABERT-2", "DNABERT-S"))
                   else 110e6 * 4 / 2 ** 30)
    # Activations measured locally: peak 1.17 GiB at B=64, L=177 with fp32
    # weights, i.e. 11.5 MiB per sequence.
    act_gib = batch * 11.5 / 1024 * (tok_len / 177.0)
    d = variant.hidden

    def gib(b):
        return b / 2 ** 30

    return {
        "vram_weights_GiB": round(weights_gib, 3),
        "vram_activations_GiB": round(act_gib, 3),
        "vram_total_GiB": round(weights_gib + act_gib, 2),
        "time_s": round(n_seq / seq_per_s, 1),
        "time_readable": _human_time(n_seq / seq_per_s),
        "throughput_seq_s": round(seq_per_s, 1),
        "disk_pools_GB": round(gib(len(POOLS) * n_seq * d * 2) * 1.0737, 3),
        "disk_kmer_GB": round(gib(n_seq * KMER_DIM * 2) * 1.0737, 3),
        "disk_map_subsample_GB": round(gib(n_map * tok_len * d * 2) * 1.0737, 3),
        "disk_map_FULL_if_it_were_GB": round(
            gib(n_seq * tok_len * d * 2) * 1.0737, 1),
    }


def _human_time(s: float) -> str:
    if s < 90:
        return "%.0f s" % s
    if s < 5400:
        return "%.1f min" % (s / 60)
    return "%.2f h" % (s / 3600)


def compute(args) -> int:
    import torch

    variant = MODELS[args.model]
    paths.ensure_dir(args.out)

    split = load_split(args.split, allow_incomplete=args.allow_incomplete_split)
    labels = split["labels"]
    print("SPLIT  cluster-based, %d parts: %s   (%s)"
          % (len(split["splits"]), ", ".join(split["splits"]),
             split["directory"]))

    print("=" * 74)
    print("MODEL   %s  (%s)" % (variant.name, variant.repo))
    print("LICENCE %s   source: %s" % (variant.licence, variant.licence_source))
    print("TOKENISATION %s, limit %d tokens" % (variant.tokenisation,
                                                variant.token_limit))
    print("=" * 74)

    kor = load_corpus(args.sequences, args.group, args.min_bp, args.max,
                      getattr(args, "every_n", 1))
    missing = [i for i in kor.ids if i not in labels]
    if missing:
        raise KeyError(
            "%d corpus IDs are not in the split manifest (e.g. %s).  The "
            "manifest was made on a different file/group/min_bp than this run "
            "-- not something one may substitute around."
            % (len(missing), missing[:3]))
    print("corpus: %d sequences of group %r (from %d rows)"
          % (len(kor), args.group, kor.n_group_rows))
    print("  skipped %d duplicate IDs, %d shorter than %d bp"
          % (kor.n_duplicates, kor.n_too_short, args.min_bp))
    print("  lower-case letters (soft masking) in the source: %d  |  'N': %d"
          % (kor.n_lower_case, kor.n_bases_N))
    ln = np.array([len(s) for s in kor.seq])
    print("  length bp: min %d  median %d  max %d  fraction==800 %.4f"
          % (ln.min(), int(np.median(ln)), ln.max(), float((ln == 800).mean())))

    if args.kmer_only:
        tok = model = None
        info = {"repo": None,
                "note": "--kmer-only run, the model was not loaded"}
        tok_len = 0
    else:
        t0 = time.time()
        tok, model, info = load_model(variant, args.device, args.workaround)
        print("model loaded in %.1f s: %s parameters"
              % (time.time() - t0, "{:,}".format(info["parameters"])))

        # Calibration before the run: token length and throughput.
        sample = kor.seq[:min(len(kor), 4 * args.batch)]
        enc = tok(sample, return_tensors=None)["input_ids"]
        nt = np.array([len(e) for e in enc])
        tok_len = int(nt.max())
        print("tokenisation ON THESE sequences (n=%d): median %d, max %d "
              "tokens against a limit of %d"
              % (len(nt), int(np.median(nt)), tok_len, variant.token_limit))
        if tok_len > variant.token_limit:
            if not args.chunking:
                raise RuntimeError(
                    "max %d tokens > limit %d for %s.  Chunking MUST be "
                    "enabled explicitly (--chunking) so that it does not "
                    "happen silently.  At k=6/stride 1 a 510-token window is "
                    "515 bp, i.e. %.0f %% of the promoter -- and the result is "
                    "then about 515 bp, not about 800."
                    % (tok_len, variant.token_limit, variant.name,
                       100 * 515 / 800))
            print("!! CHUNKING ENABLED: every sequence truncated to %d tokens. "
                  "This is a result about the initial fragment, not about the "
                  "whole promoter." % variant.token_limit)
            tok_len = variant.token_limit

        seq_per_s = _calibrate(model, tok, kor.seq, args.batch, args.device,
                               variant, args.chunking)

        est = estimate(len(kor), variant, args.batch, seq_per_s,
                       args.map_subsample, tok_len)
        print("-" * 74)
        print("ESTIMATE BEFORE THE RUN (calibrated on %d batches of this "
              "hardware)" % _N_CALIBRATION)
        print("  VRAM:  weights %.2f + activations %.2f = %.2f GiB"
              % (est["vram_weights_GiB"], est["vram_activations_GiB"],
                 est["vram_total_GiB"]))
        print("  time:  %d sequences / %.0f seq/s = %s"
              % (len(kor), est["throughput_seq_s"], est["time_readable"]))
        print("  disk:  pools %.2f GB + 6-mers %.2f GB + map subsample %.2f GB"
              % (est["disk_pools_GB"], est["disk_kmer_GB"],
                 est["disk_map_subsample_GB"]))
        print("         (a full position map for ALL of them would be %.1f GB "
              "-- hence a subsample)" % est["disk_map_FULL_if_it_were_GB"])
        print("-" * 74)
        if args.estimate_only:
            return 0

    # --- subsample for the full position map, deterministically -----------
    rng = np.random.default_rng(args.seed)
    n_map = min(args.map_subsample, len(kor))
    map_idx = (set(rng.choice(len(kor), size=n_map, replace=False).tolist())
               if n_map > 0 else set())

    # --- the main loop -----------------------------------------------------
    index_rows = []
    map_X, map_id, map_off = [], [], []
    n_shards = (len(kor) + args.shard - 1) // args.shard
    t_start = time.time()
    processed = 0
    batch_now = args.batch
    batch_min = args.batch
    for nr in range(n_shards):
        a, b = nr * args.shard, min((nr + 1) * args.shard, len(kor))
        n_w = b - a
        if not args.kmer_only and _shard_complete(args.out, args.model, nr,
                                                  n_w, variant.hidden):
            print("[shard %d/%d] already computed -- skipping"
                  % (nr + 1, n_shards))
            _append_index(index_rows, kor, a, b, nr, labels, None)
            continue
        block = kor.seq[a:b]
        km = kmer_block(block)
        _save_atomic(_kmer_file(args.out, nr), km)
        if args.kmer_only:
            _append_index(index_rows, kor, a, b, nr, labels, None)
            processed += n_w
            print("[shard %d/%d] 6-mers %s" % (nr + 1, n_shards, km.shape))
            continue

        out = {p: np.empty((n_w, variant.hidden), dtype=np.float16)
               for p in POOLS}
        n_tok = np.zeros(n_w, dtype=np.int32)
        n_unk = np.zeros(n_w, dtype=np.int32)
        i = 0
        while i < n_w:
            j = min(i + batch_now, n_w)
            extra = ({"truncation": True, "max_length": variant.token_limit}
                     if args.chunking else {})
            x = tok(block[i:j], return_tensors="pt", padding=True, **extra)
            ii = x["input_ids"].to(args.device)
            mm = x["attention_mask"].to(args.device)
            try:
                with torch.no_grad(), torch.autocast(
                        args.device, dtype=torch.float16,
                        enabled=(args.device == "cuda")):
                    res = model(input_ids=ii, attention_mask=mm)
                h = (res[0] if isinstance(res, (tuple, list))
                     else res.last_hidden_state)
                h = h.float()
            except torch.OutOfMemoryError:
                # A reduced batch lands in the manifest (`batch_reduced_to`),
                # because otherwise the run would quietly change its own
                # conditions.
                del ii, mm
                torch.cuda.empty_cache()
                if batch_now <= 1:
                    raise
                batch_now = max(1, batch_now // 2)
                batch_min = min(batch_min, batch_now)
                print("!! OOM at batch %d -> reducing to %d and repeating this "
                      "chunk" % (2 * batch_now, batch_now))
                continue
            cls, mean, mx = _pools(h, mm)
            out["cls"][i:j] = cls.cpu().numpy().astype(np.float16)
            out["mean"][i:j] = mean.cpu().numpy().astype(np.float16)
            out["max"][i:j] = mx.cpu().numpy().astype(np.float16)
            n_tok[i:j] = mm.sum(dim=1).cpu().numpy()
            n_unk[i:j] = (ii == tok.unk_token_id).sum(dim=1).cpu().numpy()
            for local in range(j - i):
                glob = a + i + local
                if glob in map_idx:
                    real = int(mm[local].sum())
                    map_X.append(h[local, :real].cpu().numpy()
                                 .astype(np.float16))
                    map_id.append(kor.ids[glob])
                    map_off.append(real)
            del h, res, ii, mm, cls, mean, mx
            i = j
        if args.device == "cuda":
            torch.cuda.empty_cache()
        for p in POOLS:
            _save_atomic(_shard_file(args.out, args.model, p, nr), out[p])
        _append_index(index_rows, kor, a, b, nr, labels, (n_tok, n_unk))
        processed += n_w
        done = b
        rate = processed / max(time.time() - t_start, 1e-9)
        print("[shard %d/%d] %d seq | %.0f seq/s | remaining %s"
              % (nr + 1, n_shards, n_w, rate,
                 _human_time((len(kor) - done) / max(rate, 1e-9))))

    # --- position map for the subsample ------------------------------------
    map_path = None
    M = None
    if map_X:
        L = max(x.shape[0] for x in map_X)
        M = np.zeros((len(map_X), L, variant.hidden), dtype=np.float16)
        for i, x in enumerate(map_X):
            M[i, :x.shape[0]] = x
        map_path = os.path.join(args.out, "position_map_%s.npz" % args.model)
        np.savez(map_path + ".tmp.npz", X=M,
                 lengths=np.array(map_off, dtype=np.int32),
                 ids=np.array(map_id))
        os.replace(map_path + ".tmp.npz", map_path)
        print("position map (subsample): %s  shape %s  %.2f GB"
              % (map_path, M.shape, M.nbytes / 1e9))

    # --- index + manifest ---------------------------------------------------
    idx_path = os.path.join(args.out, "index.tsv")
    with open(idx_path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("ID\tSPECIES\tGENOME\tOG_ID\tGENE_ID\tlen_bp\tn_tok\tn_unk"
                 "\tshard\trow\tsplit\n")
        for r in index_rows:
            fh.write("\t".join(str(v) for v in r) + "\n")
    os.replace(idx_path + ".tmp", idx_path)

    total = 0
    for f in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, f)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    man = {
        "version": 1,
        "model": args.model,
        "variant": dataclasses.asdict(variant),
        "model_info": info,
        "seed": args.seed,
        "group": args.group,
        "min_bp": args.min_bp,
        "every_n": getattr(args, "every_n", 1),
        "max": args.max,
        "batch": args.batch,
        "batch_reduced_to": (None if batch_min == args.batch else batch_min),
        "shard": args.shard,
        "chunking": bool(args.chunking),
        "pools": list(POOLS),
        "kmer_k": K,
        "kmer_dim": KMER_DIM,
        "kmer_order": KMER_ORDER,
        "kmer_dtype": "uint16 (counts, exact)",
        "one_hot_order": ONE_HOT_ORDER,
        "one_hot_order_source": "%s:%d (legacy)" % ONE_HOT_SOURCE_LEGACY,
        "one_hot_order_canonical_disagrees": "%s:%d says ACGT"
                                             % ONE_HOT_SOURCE_CANONICAL,
        "split_directory": os.path.relpath(split["directory"], paths.ROOT),
        "split_grouping": split["grouping"],
        "split_parts": list(split["splits"]),
        "split_built": split["manifest"].get("built"),
        "split_seed": split["manifest"].get("seed"),
        "n_sequences": len(kor),
        "n_group_rows": kor.n_group_rows,
        "n_duplicate_ids": kor.n_duplicates,
        "n_too_short": kor.n_too_short,
        "n_lower_case_in_source": kor.n_lower_case,
        "n_bases_N": kor.n_bases_N,
        "position_map": os.path.basename(map_path) if map_path else None,
        "position_map_n": len(map_X),
        "bytes_total": total,
        "GB_total": round(total / 1e9, 3),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    man_path = os.path.join(args.out, "manifest_%s.json" % args.model)
    with open(man_path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=True, indent=1, default=str)
    os.replace(man_path + ".tmp", man_path)

    # --- final report -------------------------------------------------------
    print("=" * 74)
    print("REPORT")
    print("  sequences            : %d" % len(kor))
    print("  shards               : %d (of %d)" % (n_shards, args.shard))
    for p in POOLS:
        f0 = _shard_file(args.out, args.model, p, 0)
        if os.path.exists(f0):
            k = np.load(f0, mmap_mode="r").shape
            print("  emb_%-4s             : %d x %d  (fp16, %.3f GB total)"
                  % (p, len(kor), k[1], len(kor) * k[1] * 2 / 1e9))
    print("  kmer%d                : %d x %d  (uint16, %.3f GB)"
          % (K, len(kor), KMER_DIM, len(kor) * KMER_DIM * 2 / 1e9))
    if map_path:
        print("  position map         : %d x %d x %d  (fp16, %.3f GB)"
              % (M.shape[0], M.shape[1], M.shape[2], M.nbytes / 1e9))
    print("  DIRECTORY TOTAL      : %.3f GB" % (total / 1e9))
    print("  index                : %s" % idx_path)
    print("  manifest             : %s" % man_path)
    print("  run time             : %s" % _human_time(time.time() - t_start))
    print("=" * 74)
    return 0


_N_CALIBRATION = 3


def _calibrate(model, tok, seqs, batch, device, variant, chunking) -> float:
    """Measure throughput ON THIS hardware instead of guessing it."""
    import torch
    sample = seqs[:batch]
    if len(sample) < batch:
        sample = (sample * (batch // max(len(sample), 1) + 1))[:batch]
    extra = ({"truncation": True, "max_length": variant.token_limit}
             if chunking else {})
    x = tok(sample, return_tensors="pt", padding=True, **extra)
    ii = x["input_ids"].to(device)
    mm = x["attention_mask"].to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16,
                                         enabled=(device == "cuda")):
        model(input_ids=ii, attention_mask=mm)          # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(_N_CALIBRATION):
            model(input_ids=ii, attention_mask=mm)
        if device == "cuda":
            torch.cuda.synchronize()
    dt = (time.time() - t) / _N_CALIBRATION
    if device == "cuda":
        print("  peak VRAM during calibration: %.2f GiB"
              % (torch.cuda.max_memory_allocated() / 2 ** 30))
    return batch / dt


def _append_index(buf, kor, a, b, nr, labels, tokens):
    """Append one index row per window. `tokens=None` writes -1 counts.

    -1 means "not counted on this pass", not "zero tokens". Both the
    `--kmer-only` branch and the resume branch pass None, so an index built
    over several runs carries real counts for the shards computed in the
    last one and -1 for the shards skipped. `unk_fraction` is the observable
    that would catch a missing `.upper()`, so read it only over rows whose
    counts are non-negative, and recompute the index from scratch if you
    need it over the whole corpus.
    """
    for i in range(a, b):
        nt = int(tokens[0][i - a]) if tokens else -1
        nu = int(tokens[1][i - a]) if tokens else -1
        buf.append((kor.ids[i], kor.species[i], kor.genome[i], kor.og[i],
                    kor.gene[i], len(kor.seq[i]), nt, nu, nr, i - a,
                    labels[kor.ids[i]]))


# Measurement reports: the source of the numbers in the module docstring.
def data_report(args) -> int:
    import collections
    p = paths.require(args.sequences or paths.SEQUENCES, "sequences.tsv.gz")
    lens = collections.defaultdict(list)
    alpha = collections.Counter()
    idc = collections.Counter()
    n = 0
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        pos = {x: i for i, x in enumerate(hdr)}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            g, s = c[pos["GROUP"]], c[pos["SEQUENCE"]]
            lens[g].append(len(s))
            n += 1
            if g == args.group:
                alpha.update(s)
                idc[c[pos["ID"]]] += 1
    print("file: %s" % p)
    print("rows total: %d" % n)
    for g, v in sorted(lens.items()):
        v = np.array(v)
        print("  %-11s n=%6d  min=%5d  median=%5d  max=%6d  fraction==800 %.4f"
              % (g, len(v), v.min(), int(np.median(v)), v.max(),
                 float((v == 800).mean())))
    print("alphabet of group %r: %s" % (args.group, alpha.most_common()))
    print("ID: %d rows, %d unique, %d repeated (max multiplicity %d)"
          % (sum(idc.values()), len(idc),
             sum(1 for v in idc.values() if v > 1),
             max(idc.values()) if idc else 0))
    return 0


def tokenizer_report(args) -> int:
    """Measure token length and the locality of tokenisation under one SNP."""
    if args.workaround:
        _local_environment_workaround()
    from transformers import AutoTokenizer
    if args.workaround:
        _triton_stub()
    variant = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(variant.repo, trust_remote_code=True)
    kor = load_corpus(args.sequences, args.group, args.min_bp,
                      limit=args.n_tok)
    full = [s for s in kor.seq if len(s) == 800 and "N" not in s]
    if not full:
        raise RuntimeError("no 800 bp sequence without N in the sample")
    enc = tok(full, return_tensors=None, add_special_tokens=False)["input_ids"]
    nt = np.array([len(e) for e in enc])
    print("n=%d sequences of 800 bp without N" % len(full))
    print("tokens: min %d median %d mean %.1f max %d  -> %.2f bp/token"
          % (nt.min(), int(np.median(nt)), nt.mean(), nt.max(),
             800 / nt.mean()))
    print("with [CLS]/[SEP]: max %d  against a limit of %d  -> %s"
          % (nt.max() + 2, variant.token_limit,
             "the WHOLE promoter in one pass"
             if nt.max() + 2 <= variant.token_limit else "CHUNKING REQUIRED"))

    rng = np.random.default_rng(args.seed)
    changed, spread, dnt = [], [], []
    f = tok.backend_tokenizer if hasattr(tok, "backend_tokenizer") else None
    if f is None:
        print("(the tokenizer is not 'fast' -- the locality measurement needs "
              "offsets)")
        return 0
    for s in full[:args.n_snp_seq]:
        e0 = f.encode(s, add_special_tokens=False)
        a = set(zip([o[0] for o in e0.offsets], e0.tokens))
        for _ in range(5):
            q = int(rng.integers(100, 700))
            alt = [c for c in "ACGT" if c != s[q]][int(rng.integers(0, 3))]
            e1 = f.encode(s[:q] + alt + s[q + 1:], add_special_tokens=False)
            b = set(zip([o[0] for o in e1.offsets], e1.tokens))
            d = (a - b) | (b - a)
            changed.append(len(a - b))
            spread.append((max(x[0] + len(x[1]) for x in d)
                           - min(x[0] for x in d)) if d else 0)
            dnt.append(len(e1.ids) - len(e0.ids))
    changed, spread, dnt = map(np.array, (changed, spread, dnt))
    print("\nSNP -> tokenisation (n=%d, seed %d):" % (len(changed), args.seed))
    print("  tokens changed: mean %.2f  median %.1f  max %d"
          % (changed.mean(), np.median(changed), changed.max()))
    print("  spread [bp]:    mean %.2f  median %.1f  max %d"
          % (spread.mean(), np.median(spread), spread.max()))
    print("  the NUMBER of tokens changes in %.4f of cases"
          % float((dnt != 0).mean()))
    print("  analytical control, k-mer k=6 stride 1: 6 tokens, 6 bp, the "
          "number of tokens changes by 0 ALWAYS")
    return 0


# Smoke test
def smoke(args) -> int:
    """A tiny input, on CPU.  Every point is a falsifier, not a 'looks ok'."""
    ok = True

    def check(name, condition, detail=""):
        nonlocal ok
        ok = ok and bool(condition)
        print("  [%s] %-46s %s" % ("OK " if condition else "BAD", name, detail))

    print("SMOKE TEST -- %s" % time.strftime("%Y-%m-%d %H:%M"))

    # 1. The nucleotide-order convention, against both sources.
    def _read_line(rel, lineno):
        for base in (paths.ROOT, os.path.join(paths.ROOT, "src")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    for i, l in enumerate(fh, 1):
                        if i == lineno:
                            return p, l.strip()
        return None, ""

    p_leg, line_leg = _read_line(*ONE_HOT_SOURCE_LEGACY)
    p_can, line_can = _read_line(*ONE_HOT_SOURCE_CANONICAL)
    print("  one-hot order in use: %r" % ONE_HOT_ORDER)
    print("     legacy    %s:%d -> %r" % (ONE_HOT_SOURCE_LEGACY[0],
                                          ONE_HOT_SOURCE_LEGACY[1], line_leg))
    print("     canonical %s:%d -> %r" % (ONE_HOT_SOURCE_CANONICAL[0],
                                          ONE_HOT_SOURCE_CANONICAL[1],
                                          line_can))
    # Reported, not checked pass/fail: a falsifier has to be able to pass,
    # and this one cannot while the two sources genuinely disagree.  An
    # unresolved decision as a permanently failing test would make the whole
    # smoke test read FAILED forever and stop meaning anything.
    found = sorted({o for o in ("ACTG", "ACGT")
                    if o in line_leg or o in line_can})
    print("     orders readable in this tree: %s" % (found or "none"))
    if ONE_HOT_ORDER not in found:
        print("     UNRESOLVED: the order in use (%r) is NOT readable in this"
              % ONE_HOT_ORDER)
        print("     tree -- %s is a LEGACY-TREE file and was not migrated."
              % ONE_HOT_SOURCE_LEGACY[0])
        print("     The canonical tree declares %r instead.  A ridge probe is"
              % (line_can.split('"')[1] if '"' in line_can else "?"))
        print("     invariant to this (it permutes feature columns), but any")
        print("     tensor handed between modules is not.  Decision pending.")
    elif len(found) > 1:
        print("     UNRESOLVED: the two sources DISAGREE (legacy ACTG, "
              "canonical ACGT).")
        print("     This module keeps the legacy order, because the existing "
              "artefacts were written with it.")

    # 2. 6-mer: the counts sum to the number of valid windows.
    s = "ACGT" * 200
    v = kmer_counts(s)
    check("kmer: sum of counts == len-k+1", int(v.sum()) == len(s) - K + 1,
          "sum=%d expected=%d" % (v.sum(), len(s) - K + 1))
    check("kmer: dimension 4**6", v.shape == (KMER_DIM,), str(v.shape))
    # Windows with N are skipped, not silently zeroed.
    s2 = "ACGT" * 100 + "N" + "ACGT" * 100
    v2 = kmer_counts(s2)
    check("kmer: windows with N skipped (exactly k)",
          int(v2.sum()) == len(s2) - K + 1 - K,
          "sum=%d, expected=%d" % (v2.sum(), len(s2) - K + 1 - K))
    # Positive control: a SNP must move the vector.
    s3 = list(s)
    s3[400] = "G" if s3[400] != "G" else "T"
    check("kmer: a single SNP moves the vector",
          not np.array_equal(v, kmer_counts("".join(s3))))

    # 3. Split: the canonical one loads and a non-cluster one is rejected.
    #    Checking that `default_rng(SEED).permutation` is deterministic, as
    #    an earlier version did, is a property of numpy and passes for every
    #    split including a bad one.
    try:
        p = load_split(args.split,
                       allow_incomplete=args.allow_incomplete_split)
        check("split: the canonical one loads, 4 parts",
              tuple(p["splits"]) == splits.SPLITS,
              "%s, n=%d IDs" % (", ".join(p["splits"]), len(p["labels"])))
        check("split: the atom is a homology cluster",
              p["grouping"] == "homology_cluster", p["grouping"])
    except Exception as e:
        check("split: the canonical one loads", False,
              "%s: %s" % (type(e).__name__, str(e).split("\n")[0]))
    # Positive control: the guard must reject a manifest keyed on OG_ID
    # alone, otherwise "the split loaded" means nothing.
    _arch = os.path.join(splits.SPLIT_DIR, "archive",
                         "og_promoter_split_seed7898.json")
    if os.path.exists(_arch):
        try:
            splits.guard_manifest(json.load(open(_arch, encoding="utf-8")),
                                  _arch)
            check("split: the guard rejects an OG_ID manifest", False,
                  "the guard let %s through" % os.path.basename(_arch))
        except ValueError:
            check("split: the guard rejects an OG_ID manifest", True,
                  "positive control on %s" % os.path.basename(_arch))
    else:
        # A missing control file is reported, not passed over: a positive
        # control that does not run is not a positive control.
        check("split: positive control of the guard is available", False,
              "no %s" % _arch)

    # 4. The corpus loads and has its measured properties.
    try:
        kor = load_corpus(args.sequences, args.group, args.min_bp, limit=256)
        ln = np.array([len(x) for x in kor.seq])
        check("corpus: %d sequences, no lower case after upper()" % len(kor),
              all(s == s.upper() for s in kor.seq))
        check("corpus: median length 800 bp (not 1000)",
              int(np.median(ln)) == 800, "median=%d" % np.median(ln))
    except Exception as e:
        check("corpus loaded", False, "%s: %s" % (type(e).__name__, e))
        kor = None

    # 5. The model: import, forward, falsifiers.
    if args.no_model:
        print("  [--] model skipped (--no-model)")
    else:
        try:
            variant = MODELS[args.model]
            tok, model, info = load_model(variant, "cpu", args.workaround)
            check("model loaded", True,
                  "%s parameters" % "{:,}".format(info["parameters"]))
            check("licence is permissive (not ND/NC)",
                  "ND" not in info["licence"] and "NC" not in info["licence"],
                  info["licence"])

            import torch
            sample = (kor.seq[:4] if kor else ["ACGT" * 200] * 4)
            x = tok(sample, return_tensors="pt", padding=True)
            with torch.no_grad():
                res = model(input_ids=x["input_ids"],
                            attention_mask=x["attention_mask"])
            h = (res[0] if isinstance(res, (tuple, list))
                 else res.last_hidden_state)
            check("forward: shape [B, L, %d]" % variant.hidden,
                  h.shape[0] == len(sample) and h.shape[2] == variant.hidden,
                  str(tuple(h.shape)))
            check("the whole promoter in one pass (no chunking)",
                  int(x["attention_mask"].sum(1).max()) <= variant.token_limit,
                  "max %d tokens / limit %d"
                  % (int(x["attention_mask"].sum(1).max()),
                     variant.token_limit))

            # Falsifier A: padding must not change the result.
            a1 = tok([sample[0]], return_tensors="pt")
            b1 = tok([sample[0], sample[1] + "ACGT" * 30],
                     return_tensors="pt", padding=True)
            with torch.no_grad():
                ha = model(input_ids=a1["input_ids"],
                           attention_mask=a1["attention_mask"])[0]
                hb = model(input_ids=b1["input_ids"],
                           attention_mask=b1["attention_mask"])[0]
            nn_ = int(a1["attention_mask"].sum())
            d = float((ha[0, :nn_] - hb[0, :nn_]).abs().max())
            check("padding invariance max|delta| < 1e-3", d < 1e-3, "%.3e" % d)

            # Falsifier B: the model must not return a constant.
            cls, mean, mx = _pools(h, x["attention_mask"])
            spread = float(mean.std(dim=0).mean())
            check("the mean pool is not constant (sd > 1e-4)", spread > 1e-4,
                  "sd=%.4e" % spread)

            # Falsifier C: [CLS] != mean, otherwise two pools are one.
            r = float((cls - mean).abs().mean())
            check("cls != mean (two pools carry different things)", r > 1e-3,
                  "|cls-mean|=%.4f" % r)

            # Falsifier D: a single SNP moves the embedding.
            s0 = sample[0]
            q = len(s0) // 2
            alt = "G" if s0[q] != "G" else "T"
            xs = tok([s0, s0[:q] + alt + s0[q + 1:]], return_tensors="pt",
                     padding=True)
            with torch.no_grad():
                hs = model(input_ids=xs["input_ids"],
                           attention_mask=xs["attention_mask"])[0]
            c2, m2_, _ = _pools(hs, xs["attention_mask"])
            dsnp = float((m2_[0] - m2_[1]).abs().mean())
            check("a single SNP moves the mean pool", dsnp > 1e-4,
                  "|delta|=%.4e" % dsnp)

            # 6. A tiny end-to-end run into a temporary directory.
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                mini = argparse.Namespace(
                    model=args.model, out=td, sequences=args.sequences,
                    group=args.group, min_bp=args.min_bp, max=8, batch=4,
                    shard=4, seed=SEED, device="cpu", every_n=1,
                    map_subsample=2, chunking=False, workaround=args.workaround,
                    kmer_only=False, estimate_only=False, split=args.split,
                    allow_incomplete_split=args.allow_incomplete_split)
                try:
                    rc = compute(mini)
                    files = sorted(os.listdir(td))
                    check("end-to-end run (8 sequences) wrote shards",
                          rc == 0 and any(f.startswith("emb_") for f in files)
                          and any(f.startswith("kmer") for f in files),
                          "%d files" % len(files))
                    # Resumability: a second pass must skip the shards.
                    rc2 = compute(mini)
                    check("resumability: the second pass exits OK", rc2 == 0)
                except FileNotFoundError as e:
                    check("end-to-end run", False,
                          "no split -- run `python -m "
                          "hyppe.analysis.splits_build`: %s"
                          % str(e).split("\n")[0])
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("model / run", False, "%s: %s" % (type(e).__name__, e))

    print("\nSMOKE TEST RESULT: %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# CLI
def _common(p):
    p.add_argument("--sequences", default=None,
                   help="default: paths.SEQUENCES")
    p.add_argument("--group", default="Promotor",
                   choices=["Promotor", "Gene", "FlankLeft", "FlankRight"])
    p.add_argument("--min-bp", type=int, default=50, dest="min_bp")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    p.add_argument("--workaround", "--environment-workaround",
                   action="store_true", dest="workaround",
                   help="EXPLICIT workaround for a broken torchvision / "
                        "missing triton on a development machine. Do NOT use "
                        "on dendrite.")


def _split_args(p):
    p.add_argument("--split", default=None,
                   help="directory of the canonical cluster split (default: "
                        "data/splits/homology_clusters/).  This is NOT a path "
                        "to your own manifest -- the split has one source: "
                        "hyppe/analysis/splits_build.py")
    p.add_argument("--allow-incomplete-split", action="store_true",
                   dest="allow_incomplete_split",
                   help="accept a split built before the version-2 fixes.  "
                        "EXPLICIT opt-in: such a split passes every row-level "
                        "test and still leaks.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="hyppe.analysis.embeddings",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # There is no `split` subcommand here and none may be added: the split is
    # built only by `python -m hyppe.analysis.splits_build`, and two places
    # producing a split are two different splits.

    p = sub.add_parser("compute", help="compute embeddings + the 6-mer baseline")
    _common(p)
    _split_args(p)
    p.add_argument("--out", default=os.path.join(paths.CACHE, "embeddings"))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--shard", type=int, default=8192)
    p.add_argument("--max", type=int, default=None,
                   help="limit the number of sequences (debug). NOTE: this is "
                        "a PREFIX, and the file is ordered by genome -- the "
                        "first 20 000 promoters are 2 genomes of 19 "
                        "(measured).  For a SAMPLE use --every-n.")
    p.add_argument("--every-n", type=int, default=1, dest="every_n",
                   help="take every nth row of the group -- a sample spread "
                        "over the whole file, i.e. over all genomes")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--map-subsample", type=int, default=2000,
                   dest="map_subsample",
                   help="how many sequences get a FULL position map [N,L,768]")
    p.add_argument("--chunking", action="store_true",
                   help="allow truncating sequences to the model's limit")
    p.add_argument("--kmer-only", action="store_true", dest="kmer_only")
    p.add_argument("--estimate-only", action="store_true",
                   dest="estimate_only",
                   help="compute the VRAM/time estimate and exit")
    p.set_defaults(fn=compute)

    p = sub.add_parser("smoke", help="tiny input, CPU, falsifiers")
    _common(p)
    _split_args(p)
    p.add_argument("--no-model", action="store_true", dest="no_model")
    p.set_defaults(fn=smoke)

    p = sub.add_parser("data-report", help="length distribution and alphabet")
    _common(p)
    p.set_defaults(fn=data_report)

    p = sub.add_parser("tokenizer-report",
                       help="token length and locality under a SNP")
    _common(p)
    p.add_argument("--n-tok", type=int, default=400, dest="n_tok")
    p.add_argument("--n-snp-seq", type=int, default=40, dest="n_snp_seq")
    p.set_defaults(fn=tokenizer_report)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
