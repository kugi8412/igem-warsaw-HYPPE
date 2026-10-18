"""TF-IDF + SVD: the honest offline text encoder. Replaces `hash_embedding`.

`FrozenTextEncoder` falls back to `hash_embedding(string)` when no language
model and no cache are available. That is deterministic and runs offline, and
those are its only virtues: it maps each string to an unrelated random
direction, so "growth on glucose" and "glucose as sole carbon source" get
vectors whose expected cosine is zero. Nothing generalises across paraphrase,
which is the whole point of putting the condition in as text rather than as a
category index. A model trained on hash vectors still fits -- 487 distinct
conditions give 487 distinct directions and a large head memorises the lookup
table -- and then fails on the first unseen condition, silently, because the
training curve looked healthy.

TF-IDF + SVD is not a placeholder for the LM but the measured baseline it has
to beat. Out-of-sample R^2 on the conditional deviation, ridge per
orthogroup over 11 323 orthogroups:

    feature                     random split   STUDY split
    cat7 (7 multi-hot)             0.10601       0.02067
    txt32 (TF-IDF + SVD)           0.36302       0.07853
    txt64                          0.42206       0.09303
    cond1hot (lookup table)        0.43448       0.08615
    CONTROL cat7+txt32_SHUF        0.05388       0.01195

`txt64` reaches 97.1 % of what an oracle lookup table achieves in-vocabulary,
so the case for a frozen language model is entirely out-of-vocabulary: it
earns its place only above 0.09303 under the study split.

Two strings sharing a token share a coordinate before the projection, so
their cosine is positive by construction and after the SVD they share a
subspace. :meth:`similarity_report` measures exactly that, and a fitted
encoder whose two numbers are equal has failed.

Determinism is a contract here: the vocabulary is sorted, the SVD sign is
fixed by making each component's largest-magnitude entry positive, and the
fitted state is serialisable. Without the sign fix -- `svd` returns either
sign and numpy's choice can flip between BLAS builds -- the same corpus gives
mirrored embeddings on two machines and every cached vector disagrees with
every checkpoint.

What it cannot do is handle a word it never saw. An unseen condition whose
tokens are all out of vocabulary projects to the zero vector, reported as
`oov_fraction` rather than returned as a plausible-looking direction. That is
the one case a real language model handles, and the case the design brief
needs, so `oov_fraction > 0` on a design query is a reason to fetch the LM.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "TFIDF_VERSION",
    "tokenise",
    "TfidfSvdEncoder",
]

TFIDF_VERSION = "hyppe-tfidf-svd-1"

_TOKEN = re.compile(r"[a-z0-9%+.\-]+")
_EPS = 1e-12


def tokenise(text: str, ngram_max: int = 2) -> List[str]:
    """Lowercase word tokens plus contiguous n-grams up to `ngram_max`.

    Bigrams are included because the most informative distinctions in this
    corpus are two-word ("carbon source", "heat shock") and a unigram model
    scores "light" the same in "light response" and "light microscopy". They
    cost vocabulary, which is cheap over 505 distinct normalised strings.
    """
    words = _TOKEN.findall(str(text).lower())
    out: List[str] = list(words)
    for n in range(2, max(2, int(ngram_max)) + 1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i:i + n]))
    return out


@dataclass
class TfidfSvdEncoder:
    """Fit on a corpus of strings, then embed any string deterministically."""

    dim: int = 64
    ngram_max: int = 2
    min_df: int = 1
    max_df_ratio: float = 0.9
    sublinear_tf: bool = True

    vocabulary: Dict[str, int] = field(default_factory=dict)
    idf: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None      # [dim, |V|]
    singular_values: Optional[np.ndarray] = None
    n_documents: int = 0
    _fit_hash: str = ""

    # Fit
    def fit(self, texts: Sequence[str]) -> "TfidfSvdEncoder":
        docs = [tokenise(t, self.ngram_max) for t in texts]
        n_docs = len(docs)
        if n_docs < 2:
            raise ValueError(
                "TF-IDF needs at least two documents; an IDF over one "
                "document is zero everywhere and every embedding would be the "
                "zero vector")
        df: Dict[str, int] = {}
        for d in docs:
            for tok in set(d):
                df[tok] = df.get(tok, 0) + 1
        # min_df=1 keeps hapaxes on purpose: min_df=2 is what silently dropped
        # the accession numbers in an earlier probe and made a leaky field
        # look harmless. `ConditionTextSpec` removes them at the source, where
        # the removal can be audited.
        hi = int(math.ceil(float(self.max_df_ratio) * n_docs))
        vocab = sorted(t for t, c in df.items()
                       if int(self.min_df) <= c <= hi)
        if not vocab:
            raise ValueError(
                "empty vocabulary after min_df=%d / max_df_ratio=%.2f over %d "
                "documents" % (self.min_df, self.max_df_ratio, n_docs))
        index = {t: i for i, t in enumerate(vocab)}
        idf = np.zeros(len(vocab), dtype=np.float64)
        for t, i in index.items():
            idf[i] = math.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0

        X = self._matrix(docs, index, idf)
        # Thin SVD on the document-term matrix, uncentred: TF-IDF rows are
        # sparse and non-negative, and centring makes them dense for no
        # measured gain.
        k = int(min(self.dim, min(X.shape) - 1))
        if k < 1:
            raise ValueError("cannot take %d components from a %r matrix"
                             % (self.dim, X.shape))
        _u, s, vt = np.linalg.svd(X, full_matrices=False)
        comp = vt[:k]
        # Sign fix: see the module docstring. Without it two BLAS builds give
        # mirrored embeddings for the same corpus.
        for i in range(comp.shape[0]):
            j = int(np.argmax(np.abs(comp[i])))
            if comp[i, j] < 0:
                comp[i] *= -1.0
        self.vocabulary = index
        self.idf = idf
        self.components = comp
        self.singular_values = s[:k]
        self.n_documents = n_docs
        self._fit_hash = hashlib.sha256(
            json.dumps({"version": TFIDF_VERSION, "vocab": vocab,
                        "dim": k, "ngram_max": self.ngram_max,
                        "sublinear_tf": bool(self.sublinear_tf)},
                       sort_keys=True).encode("utf-8")).hexdigest()
        return self

    def _matrix(self, docs: Sequence[Sequence[str]],
                index: Dict[str, int], idf: np.ndarray) -> np.ndarray:
        X = np.zeros((len(docs), len(index)), dtype=np.float64)
        for r, d in enumerate(docs):
            counts: Dict[int, int] = {}
            for tok in d:
                j = index.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                tf = (1.0 + math.log(c)) if self.sublinear_tf else float(c)
                X[r, j] = tf * idf[j]
            n = np.linalg.norm(X[r])
            if n > _EPS:
                X[r] /= n
        return X

    # Encode
    @property
    def fitted(self) -> bool:
        return self.components is not None and bool(self.vocabulary)

    def fingerprint(self) -> str:
        return self._fit_hash

    def encode(self, texts: Sequence[str], l2_normalise: bool = True
               ) -> np.ndarray:
        """`[n, dim]` float32. An all-OOV string returns the ZERO vector.

        Zero rather than a random direction, because a zero vector is
        detectable downstream and a plausible direction is not: the hash
        encoder's defining flaw was that its failure mode looked exactly like
        its success mode.
        """
        if not self.fitted:
            raise RuntimeError("encoder is not fitted; call fit(corpus) first")
        assert self.idf is not None and self.components is not None
        docs = [tokenise(t, self.ngram_max) for t in texts]
        X = self._matrix(docs, self.vocabulary, self.idf)
        Z = X @ self.components.T
        if l2_normalise:
            n = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = np.divide(Z, np.maximum(n, _EPS),
                          out=np.zeros_like(Z), where=n > _EPS)
        return Z.astype(np.float32)

    def oov_report(self, texts: Sequence[str]) -> Dict[str, float]:
        """How much of the input the vocabulary did not cover.

        Reported so that an all-OOV string is a decision -- fetch the language
        model -- rather than a silent zero vector.
        """
        tot = 0
        miss = 0
        empty = 0
        for t in texts:
            toks = tokenise(t, self.ngram_max)
            if not toks:
                empty += 1
                continue
            hit = sum(1 for x in toks if x in self.vocabulary)
            tot += len(toks)
            miss += len(toks) - hit
            if hit == 0:
                empty += 1
        return {
            "n_texts": float(len(texts)),
            "oov_fraction": (float(miss) / tot) if tot else 1.0,
            "all_oov_texts": float(empty),
            "all_oov_fraction": float(empty) / max(len(texts), 1),
        }

    # Falsifiers
    def similarity_report(self, texts: Sequence[str],
                          max_pairs: int = 20000,
                          seed: int = 7898) -> Dict[str, float]:
        """Cosine for token-sharing pairs against disjoint pairs.

        The falsifier for this whole module. A hash encoder scores the same
        number for both, which is what "carries no semantics" means
        numerically; a working encoder scores clearly higher for pairs sharing
        tokens. A `separation` near 0 means there is no reason to prefer this
        to the hash it replaced.
        """
        Z = self.encode(texts)
        toks = [set(tokenise(t, self.ngram_max)) for t in texts]
        rng = np.random.default_rng(int(seed))
        n = len(texts)
        if n < 4:
            return {"separation": float("nan"), "n_pairs": 0.0}
        shared: List[float] = []
        disjoint: List[float] = []
        for _ in range(int(max_pairs)):
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i == j:
                continue
            c = float(np.dot(Z[i], Z[j]))
            (shared if toks[i] & toks[j] else disjoint).append(c)
        a = float(np.mean(shared)) if shared else float("nan")
        b = float(np.mean(disjoint)) if disjoint else float("nan")
        return {
            "cos_sharing_tokens": a,
            "cos_disjoint": b,
            "separation": a - b,
            "n_sharing": float(len(shared)),
            "n_disjoint": float(len(disjoint)),
            "carries_semantics": bool(np.isfinite(a) and np.isfinite(b)
                                      and (a - b) > 0.05),
        }

    def report(self) -> Dict[str, object]:
        return {
            "version": TFIDF_VERSION,
            "fitted": self.fitted,
            "dim": int(self.components.shape[0]) if self.fitted else 0,
            "vocabulary": len(self.vocabulary),
            "n_documents": int(self.n_documents),
            "ngram_max": int(self.ngram_max),
            "fingerprint": self._fit_hash,
            "measured_ceiling_note": (
                "txt64 reached 0.42206 out-of-sample R^2 against a "
                "lookup-table ceiling of 0.43448 under a random split "
                "(97.1 %); a frozen LM earns its place only above 0.09303 "
                "under the STUDY split"),
        }

    # Serialisation
    def state(self) -> Dict[str, object]:
        if not self.fitted:
            raise RuntimeError("nothing to serialise: encoder is not fitted")
        assert self.idf is not None and self.components is not None
        inv = sorted(self.vocabulary, key=lambda t: self.vocabulary[t])
        return {
            "version": TFIDF_VERSION,
            "vocabulary": inv,
            "idf": self.idf.astype(np.float32).tolist(),
            "components": self.components.astype(np.float32).tolist(),
            "singular_values": (self.singular_values.astype(np.float32)
                                .tolist() if self.singular_values is not None
                                else []),
            "dim": int(self.components.shape[0]),
            "ngram_max": int(self.ngram_max),
            "min_df": int(self.min_df),
            "max_df_ratio": float(self.max_df_ratio),
            "sublinear_tf": bool(self.sublinear_tf),
            "n_documents": int(self.n_documents),
            "fingerprint": self._fit_hash,
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "TfidfSvdEncoder":
        if str(state.get("version")) != TFIDF_VERSION:
            raise ValueError(
                "encoder state was written by %r and this build is %r. The "
                "vocabulary and the projection define the embedding, so a "
                "version mismatch means every cached vector is a different "
                "function of the same string."
                % (state.get("version"), TFIDF_VERSION))
        enc = cls(dim=int(state["dim"]), ngram_max=int(state["ngram_max"]),
                  min_df=int(state["min_df"]),
                  max_df_ratio=float(state["max_df_ratio"]),
                  sublinear_tf=bool(state["sublinear_tf"]))
        vocab = [str(t) for t in state["vocabulary"]]
        enc.vocabulary = {t: i for i, t in enumerate(vocab)}
        enc.idf = np.asarray(state["idf"], dtype=np.float64)
        enc.components = np.asarray(state["components"], dtype=np.float64)
        enc.singular_values = np.asarray(state.get("singular_values", []),
                                         dtype=np.float64)
        enc.n_documents = int(state.get("n_documents", 0))
        enc._fit_hash = str(state.get("fingerprint", ""))
        return enc

    def save(self, path: str) -> str:
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f)
        return path

    @classmethod
    def load(cls, path: str) -> "TfidfSvdEncoder":
        with io.open(path, encoding="utf-8") as f:
            return cls.from_state(json.load(f))
