"""Loader for `data/experiment_text_emb.npz`, the query side of channel C2.

The join key is `<genome>|<category>`, with `<genome>|*` for the genome
marginal, spelled as in `dataset.ConditionResolver.cell_key`. It keys the cell
and not the row: `labels.build_targets` backs an (orthogroup, genome) target
with a sample list independent of the orthogroup, so a per-row aggregate would
be a function of the genome alone -- which `species_id` already carries.

A row is the mean of the SapBERT embeddings of the `condition` strings behind
its cell, re-normalised -- 85 keys, [85, 768] float32 at unit norm, centred on
the stored `mean_vector` so a new description lands in the same space. It is a
cell-level token: differences between experiments within one category are
real, but a cell mean discards them.

An uncovered cell is safe rather than silent: `rows_for` gives -1, `gather`
gives zeros, and the corpus pairs that with `text_ok = 0`, so the row reaches
`unknown_token_text` and never `text_proj(zeros)` -- a learned constant that
would report a context share while carrying nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["DescriptionTextCache", "DEFAULT_PATH", "MARGINAL_CATEGORY",
           "cell_key", "embed_description", "nearest_keys"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(
    os.environ.get("HYPPE_DATA_ROOT", _REPO_ROOT),
    "data/experiment_text_emb.npz")

#: Genome marginal, as in `dataset.ConditionResolver.cell_key`.
MARGINAL_CATEGORY = "*"

#: The frozen encoder the artefact was built with. Changing it invalidates
#: every vector in the file, so it is read back from the artefact, not assumed.
MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def cell_key(genome: str, category: Optional[str] = None) -> str:
    """`<genome>|<category>`; `<genome>|*` for the genome marginal."""
    return "%s|%s" % (genome, category if category else MARGINAL_CATEGORY)


@dataclass
class DescriptionTextCache(object):
    """Frozen description vectors, one per conditioning cell.

    `text_cache.ExperimentTextCache.load` discovers this class by name and
    reads `keys` and `vectors` off it, so those two names are a contract with
    that module. The rest is provenance for a row's own vector.
    """

    keys: List[str]
    vectors: np.ndarray                  # [n_keys, dim] float32, unit norm
    texts: List[str] = field(default_factory=list)
    genome: List[str] = field(default_factory=list)
    category: List[str] = field(default_factory=list)
    n_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, "int32"))
    samples: List[str] = field(default_factory=list)
    mean_vector: Optional[np.ndarray] = None
    backend: str = ""
    model_id: str = ""
    aggregate: str = ""
    path: str = ""

    @classmethod
    def load(cls, path: str = DEFAULT_PATH,
             index_path: str = "") -> "DescriptionTextCache":
        """Read the artefact. `index_path` is accepted and ignored.

        `ExperimentTextCache.load` calls a sibling loader as `fn(path,
        index_path)` and catches the TypeError to retry `fn(path)`, so
        refusing the argument would silently change which branch ran.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no experiment-text artefact at %s. Build it with "
                "poprawki/build_experiment_text_emb.py --write, or leave "
                "conditioning.experiment_text.enabled at false: a stand-in "
                "would report a context share while carrying no information."
                % (path,))
        z = np.load(path, allow_pickle=False)
        need = ("keys", "vectors")
        missing = [n for n in need if n not in list(z.files)]
        if missing:
            raise ValueError(
                "%s lacks %r (has %r). A key array and a vector array are "
                "both required: a matrix alone cannot be joined to a corpus "
                "row." % (path, missing, list(z.files)))
        keys = [str(k) for k in z["keys"].tolist()]
        V = np.ascontiguousarray(np.asarray(z["vectors"]), dtype=np.float32)
        if V.ndim != 2 or len(keys) != int(V.shape[0]):
            raise ValueError(
                "%s: %d keys against vectors %s. The join is positional, so a "
                "length mismatch means the two arrays describe different "
                "objects." % (path, len(keys), (V.shape,)))
        if not bool(np.isfinite(V).all()):
            raise ValueError(
                "%s: %d non-finite entries. A NaN in a frozen embedding "
                "poisons every downstream statistic silently."
                % (path, int((~np.isfinite(V)).sum())))
        dup = len(keys) - len(set(keys))
        if dup:
            raise ValueError(
                "%s: %d duplicate keys. Which row a cell resolves to would "
                "depend on iteration order." % (path, dup))

        def col(name):
            return ([str(x) for x in z[name].tolist()] if name in z.files
                    else [])

        return cls(
            keys=keys, vectors=V, texts=col("texts"), genome=col("genome"),
            category=col("category"),
            n_samples=(np.asarray(z["n_samples"], dtype=np.int64)
                       if "n_samples" in z.files
                       else np.zeros(len(keys), dtype=np.int64)),
            samples=col("samples"),
            mean_vector=(np.asarray(z["mean_vector"], dtype=np.float64)
                         if "mean_vector" in z.files else None),
            backend=(str(z["backend"]) if "backend" in z.files else ""),
            model_id=(str(z["model_id"]) if "model_id" in z.files
                      else MODEL_ID),
            aggregate=(str(z["aggregate"]) if "aggregate" in z.files else ""),
            path=str(path))

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def index(self) -> Dict[str, int]:
        return {k: i for i, k in enumerate(self.keys)}

    def rows_for(self, keys: Sequence[str]) -> np.ndarray:
        """Row per key, -1 where absent. -1 must become `text_ok = 0`."""
        ix = self.index()
        return np.array([ix.get(str(k), -1) for k in keys], dtype=np.int64)

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """[n, dim] float32. A -1 row comes back as zeros, which `text_proj`
        turns into a learned constant: pair this with the `rows >= 0` mask.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.zeros((rows.size, self.dim), dtype=np.float32)
        hit = rows >= 0
        if bool(hit.any()):
            out[hit] = self.vectors[rows[hit]]
        return out

    def describe(self, key: str) -> str:
        """One line of provenance for a key, for a report or a notebook."""
        ix = self.index().get(key)
        if ix is None:
            return "%s: NOT IN THIS ARTEFACT" % (key,)
        n = int(self.n_samples[ix]) if len(self.n_samples) > ix else -1
        txt = self.texts[ix] if ix < len(self.texts) else ""
        return "%s: %d samples averaged | %s" % (key, n, txt[:200])


def embed_description(texts, cache: Optional[DescriptionTextCache] = None,
                      path: str = DEFAULT_PATH,
                      device: str = "cpu") -> np.ndarray:
    """Embed new free text into the same space as the artefact's rows.

    The design-time entry point: describe a wet-lab experiment in words and
    hand the result to the condition encoder as `text_emb`. Centring uses the
    artefact's stored mean, never the input's own, which would land the query
    in a different space from every row of the cache.

    Returns [n, dim] float32, unit norm.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("no text given")
    cache = cache or DescriptionTextCache.load(path)
    if cache.mean_vector is None:
        raise ValueError(
            "%s carries no `mean_vector`, so a new description cannot be put "
            "in the same space as its rows. Rebuild the artefact with "
            "poprawki/build_experiment_text_emb.py --write." % (cache.path,))
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_id = cache.model_id or MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_id,
                                    local_files_only=True).eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hid = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
            out.append(((hid * m).sum(1)
                        / m.sum(1).clamp_min(1e-9)).float().cpu().numpy())
    A = np.concatenate(out, axis=0).astype("float64")
    if A.shape[1] != cache.dim:
        raise ValueError(
            "%s produced %d dims but the artefact holds %d: the two sides of "
            "the channel would not live in one space."
            % (model_id, A.shape[1], cache.dim))
    B = A - np.asarray(cache.mean_vector, dtype="float64").reshape(1, -1)
    n = np.linalg.norm(B, axis=1, keepdims=True)
    if float(n.min()) < 1e-9:
        raise ValueError("a description collapsed onto the centring "
                         "reference and has no direction")
    return (B / n).astype("float32")


def nearest_keys(vector: np.ndarray, cache: DescriptionTextCache,
                 k: int = 5) -> List[Tuple[str, float]]:
    """The `k` cells a query lands nearest, with their cosines.

    The condition encoder's prototype head asks the same question, so a query
    landing somewhere unrelated is visible before anything is generated.
    """
    v = np.asarray(vector, dtype="float64").reshape(-1)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    sim = cache.vectors.astype("float64") @ v
    order = np.argsort(-sim)[:max(1, int(k))]
    return [(cache.keys[int(i)], float(sim[int(i)])) for i in order]
