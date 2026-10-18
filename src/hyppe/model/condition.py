"""Condition encoder C: free text is the channel, categories are a target.

Variance of the ComBat matrix splits into a promoter x genome main effect of
85.22 % and a conditional deviation of 14.78 %, of which 73.92 % is
reproducible between replicates -- 10.92 % of total variance, and that is
the whole budget here. Nothing in this module may be described as improving
the headline PCC/SCC: at 10.92 % reachable the effect on a pooled metric
sits inside seed noise, so it is reported on a within-group metric or not at
all, and always under the STUDY split (D4), since ComBat left the batch x
promoter interaction and a random split inflates everything about 4x.

Text is the input and categories are not (D1): out-of-sample R^2 on the
conditional deviation is 0.09303 for a text vector against 0.02067 for the
seven categories, and for P1 -- the strain the wet evaluation runs on -- the
categories are negative. They enter as a multi-label head predicted FROM the
text vector, so adding one changes no dimension of the conditioning path and
costs no parameters (D2): a category is named in English, pushed through the
same frozen LM, and scored against a prototype row.

`logit_scale` and `logit_bias` read as trained and are not: `category_loss`
is called from no line of `training/` and `cat_logits` enters no term of the
total loss, so `falsifiers()` carries a row that fails while that holds.
Those `cat_logits` are a 40-column score over `CategoryVocabulary`, whereas
the 7-class `cat_gain` comes from `heads/category.py` and reads the
regression trunk -- two read-outs that have been confused.

`experiment_title` is a sample key, not a description (D3): 1491 of 1498
kept values are unique and 65.3 % contain a digit, which TF-IDF dropped
through min_df=2 and a language model would embed instead. Only `condition`
and `study_title` go in, after the accession regex. The normalisation
function is a contract of the same rank as a channel order, because a frozen
LM is a pure function of the string it is given, so
`ConditionTextSpec.fingerprint()` hashes the rules and the field list and
the cache refuses vectors built under another fingerprint.

No task of this layer is a Nash player (D5): the conditional target reaches
R^2 ~0.03-0.10, so its gradient on the shared trunk is small, and NashMTL
hands the largest alpha to the smallest gradient. The module does not own
the fusion block -- `ConditionCrossAttention` lives in
`hyppe.model.encoder` -- and nothing here canonicalises a strand, since
promoters are not RC-invariant.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..geometry.lorentz import DEFAULT_NO_DECAY_PATTERNS
from .fusion import LatentConditioner

__all__ = [
    "TEXT_NORMALISATION_VERSION",
    "DEFAULT_TEXT_FIELDS",
    "FORBIDDEN_TEXT_FIELDS",
    "DEFAULT_CATEGORY_PHRASES",
    "CONDITION_NO_DECAY_SUBSTRINGS",
    "ConditionTextSpec",
    "normalise_text",
    "build_condition_text",
    "normalisation_ladder",
    "CategoryVocabulary",
    "ConditionEmbeddingCache",
    "hash_embedding",
    "FrozenTextEncoder",
    "ConditionConfig",
    "ConditionOutput",
    "ConditionEncoder",
    "context_attention_stats",
    "average_precision",
    "participation_ratio",
    "falsifiers",
    "report",
    "CentredLatentConditioner",
]

_EPS = 1e-6

#: Bumped whenever a rule below changes. A checkpoint that carries a different
#: version was built on different strings and its predictions are not
#: comparable, even though every weight matches.
TEXT_NORMALISATION_VERSION = "hyppe-cond-norm-1"

#: Fields that go to the language model, in this order (D3).
DEFAULT_TEXT_FIELDS: Tuple[str, ...] = ("condition", "study_title")

#: Fields that must NEVER reach the language model, with the measurement that
#: put them here. `falsifiers()` checks the spec against this tuple.
FORBIDDEN_TEXT_FIELDS: Tuple[str, ...] = ("experiment_title", "sample",
                                          "sample_name", "geo_accessions",
                                          "bioproject", "study", "label")

#: English phrasings used to embed a category through the SAME frozen LM (D2).
#: Adding a category is adding an entry here plus a name in the vocabulary --
#: no new parameters. The keys must match hyppe.data.labels.DEFAULT_CATEGORIES.
DEFAULT_CATEGORY_PHRASES: Dict[str, str] = {
    "control": "control condition, untreated reference culture",
    "carbon source": "carbon source, growth on a defined carbon substrate",
    "biotic antagonism": "biotic antagonism, confrontation with another fungus "
                         "or mycoparasitism",
    "light response": "light response, illumination or photoreceptor "
                      "stimulation",
    "plant association": "plant association, interaction with plant roots or "
                         "plant tissue",
    "mechanical injury": "mechanical injury, wounding of the mycelium",
    "other": "other experimental condition, not covered by the named "
             "categories",
}

#: Weight decay exemptions for THIS module; the trainer must use this tuple,
#: not the project default, which does not cover `unknown_token_*`. Decaying
#: a learned "condition unknown" token drives it to zero, a legal point of
#: the space the model has already learned about: lambda 0.0762 over 34
#: epochs is a factor 1.8e-5 and once erased a gate.
CONDITION_NO_DECAY_SUBSTRINGS: Tuple[str, ...] = tuple(
    DEFAULT_NO_DECAY_PATTERNS) + ("unknown_token", "side_token", "logit_")


# Small numerical helpers
def _rms_normalise(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Scale each SAMPLE to unit RMS over the feature dimension.

    Per sample, not per batch (D8): a batch statistic would make the context,
    and through it the prediction, a function of who else was in the batch.
    """
    return x / x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve for one binary column.

    Used instead of accuracy because the categories run 547 to 33 over 1498
    samples. Ties are collapsed to their expected precision: a head emitting
    one score for everything would otherwise score 1.0000 or 0.3312 by where
    the positives happen to sit.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.shape != y.shape:
        raise ValueError("scores %s and labels %s differ" % (s.shape, y.shape))
    pos = float(y.sum())
    if pos <= 0.0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s = s[order]
    y = y[order]
    cum = np.cumsum(y)
    n = np.arange(1, y.size + 1, dtype=np.float64)
    # Group boundaries of equal scores; inside a group every ordering is
    # equally legitimate, so use the precision at the end of the group.
    ends = np.nonzero(np.r_[s[1:] != s[:-1], True])[0]
    starts = np.r_[0, ends[:-1] + 1]
    total = 0.0
    for a, b in zip(starts, ends):
        k = float(cum[b])                     # positives up to the group end
        m = float(n[b])                       # items up to the group end
        hits = float(y[a:b + 1].sum())
        if hits > 0.0:
            total += (k / m) * hits
    return float(total / pos)


def participation_ratio(x: np.ndarray) -> float:
    """Effective rank of a set of vectors: (sum l)^2 / sum l^2 over covariance.

    Reference point: the frozen DNABERT flank channel measures 13.9 of 768.
    A text channel below ~20 is in the same regime, its context keys reduced
    to two almost parallel directions.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    cov = (a.T @ a) / float(a.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    s1 = float(ev.sum())
    s2 = float((ev ** 2).sum())
    if s2 <= 0.0:
        return float("nan")
    return (s1 * s1) / s2


# Text normalisation: ordered, literal rules, hashed into the fingerprint.
# Editing a pattern invalidates every cached embedding built under the old
# one, which is the intent: a changed regex is a changed model.
_ACCESSION_RE = re.compile(
    r"\b(?:SR[XRSP]|ER[XRSP]|DR[XRSP]|PRJ[A-Z]{1,2}|GS[MEDX]|SAM[NDE]A?|"
    r"GC[AF])[0-9]{3,}(?:\.[0-9]+)?\b", re.IGNORECASE)
_REPLICATE_RE = re.compile(
    r"\b(?:biological\s+|technical\s+)?"
    r"(?:replicates?|replic|repl|rep|runs?|samples?|lanes?)\s*"
    r"[#\-_.:]?\s*[0-9]+\b", re.IGNORECASE)
_TRAILING_REP_RE = re.compile(r"\b[rR][0-9]{1,2}\b")
_TIMEPOINT_RE = re.compile(
    r"\b(?:time\s*points?|timepoints?|tp|t|days?|d|hours?|hrs?|h|mins?|"
    r"minutes?|groups?|grp|batches|batch|sets?|conditions?)\s*"
    r"[#\-_.:]?\s*[0-9]+(?:\.[0-9]+)?\b", re.IGNORECASE)
_TIMEVALUE_RE = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:h|hr|hrs|hours?|d|days?|min|mins|minutes?)\b",
    re.IGNORECASE)
_INTEGER_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b")
# Both cases on purpose: a lowercase-only class silently deletes every
# capital under `lowercase=False`, so "Control Culture" would normalise to
# "ontrol ulture".
_PUNCT_RE = re.compile(r"[^a-zA-Z0-9%+/\-\. ]+")
_SPACE_RE = re.compile(r"\s+")

#: level -> (name, description). The measured ladder on the `condition` column
#: of the 1498 kept samples: 1025 -> 505 -> 469 -> 373 unique strings.
NORMALISATION_LEVELS: Dict[int, Tuple[str, str]] = {
    0: ("raw", "case folding, punctuation and whitespace only"),
    1: ("accession+replicate",
        "also drops accessions and 'replicate N'/'run N'/'sample N'"),
    2: ("+timepoint", "also drops 'timepoint N', '24 h', 'group N'"),
    3: ("+integers", "drops every number; MERGES 25C with 28C, use knowingly"),
}


@dataclass(frozen=True)
class ConditionTextSpec:
    """How a metadata row becomes the one string the frozen LM sees.

    A checkpoint artefact, not a preference: a changed rule moves every
    embedding while every weight stays equal, which `fingerprint()` makes
    checkable. Level 1 is the default, being the last level that cannot
    merge two different experiments ("25C" with "28C").
    """

    fields: Tuple[str, ...] = DEFAULT_TEXT_FIELDS
    level: int = 1
    separator: str = " | "
    lowercase: bool = True
    max_chars: int = 512
    empty_text: str = "condition unknown"

    def __post_init__(self) -> None:
        if self.level not in NORMALISATION_LEVELS:
            raise ValueError("level must be one of %s, got %r"
                             % (sorted(NORMALISATION_LEVELS), self.level))
        bad = [f for f in self.fields if f in FORBIDDEN_TEXT_FIELDS]
        if bad:
            raise ValueError(
                "fields %r are sample identifiers, not descriptions, and must "
                "not reach the language model: experiment_title is unique for "
                "1491 of 1498 kept samples, 66.9 %% of its token types are "
                "hapaxes and 65.3 %% contain a digit" % (bad,))

    def rules(self) -> Tuple[Tuple[str, str], ...]:
        """The ordered, literal rules, as (name, pattern) for the fingerprint."""
        out: List[Tuple[str, str]] = [("lowercase", str(bool(self.lowercase)))]
        if self.level >= 1:
            out.append(("accession", _ACCESSION_RE.pattern))
            out.append(("replicate", _REPLICATE_RE.pattern))
            out.append(("trailing_rep", _TRAILING_REP_RE.pattern))
        if self.level >= 2:
            out.append(("timepoint", _TIMEPOINT_RE.pattern))
            out.append(("timevalue", _TIMEVALUE_RE.pattern))
        if self.level >= 3:
            out.append(("integers", _INTEGER_RE.pattern))
        out.append(("punct", _PUNCT_RE.pattern))
        out.append(("space", _SPACE_RE.pattern))
        return tuple(out)

    def fingerprint(self) -> str:
        """sha256 over version, fields, separator and every literal rule."""
        payload = json.dumps(
            {
                "version": TEXT_NORMALISATION_VERSION,
                "fields": list(self.fields),
                "separator": self.separator,
                "level": int(self.level),
                "max_chars": int(self.max_chars),
                "empty_text": self.empty_text,
                "rules": [list(r) for r in self.rules()],
            },
            sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        d = {
            "version": TEXT_NORMALISATION_VERSION,
            "fields": list(self.fields),
            "level": int(self.level),
            "level_name": NORMALISATION_LEVELS[self.level][0],
            "separator": self.separator,
            "lowercase": bool(self.lowercase),
            "max_chars": int(self.max_chars),
            "empty_text": self.empty_text,
            "rules": [list(r) for r in self.rules()],
            "fingerprint": self.fingerprint(),
            "forbidden_fields": list(FORBIDDEN_TEXT_FIELDS),
        }
        return d

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "ConditionTextSpec":
        """Rebuild from a checkpoint and VERIFY the fingerprint still matches."""
        spec = ConditionTextSpec(
            fields=tuple(d.get("fields", DEFAULT_TEXT_FIELDS)),
            level=int(d.get("level", 1)),
            separator=str(d.get("separator", " | ")),
            lowercase=bool(d.get("lowercase", True)),
            max_chars=int(d.get("max_chars", 512)),
            empty_text=str(d.get("empty_text", "condition unknown")),
        )
        want = d.get("fingerprint")
        if want is not None and str(want) != spec.fingerprint():
            raise ValueError(
                "text normalisation fingerprint mismatch: checkpoint %s, this "
                "code %s. The regexes changed, so every cached embedding was "
                "built from a different string." % (want, spec.fingerprint()))
        return spec


def normalise_text(text: str, spec: Optional[ConditionTextSpec] = None) -> str:
    """Apply the ordered rules of `spec` to one string.

    Returns `spec.empty_text` for an input that normalises away: an empty
    string is not a neutral input to a language model but a specific point of
    its output space.
    """
    sp = spec or ConditionTextSpec()
    s = text or ""
    if sp.lowercase:
        s = s.lower()
    if sp.level >= 1:
        s = _ACCESSION_RE.sub(" ", s)
        s = _REPLICATE_RE.sub(" ", s)
        s = _TRAILING_REP_RE.sub(" ", s)
    if sp.level >= 2:
        s = _TIMEPOINT_RE.sub(" ", s)
        s = _TIMEVALUE_RE.sub(" ", s)
    if sp.level >= 3:
        s = _INTEGER_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s.lower() if sp.lowercase else s)
    s = _SPACE_RE.sub(" ", s).strip(" .-")
    if not s:
        return sp.empty_text
    if len(s) > sp.max_chars:
        s = s[: sp.max_chars].rsplit(" ", 1)[0]
    return s


def build_condition_text(row: Mapping[str, object],
                         spec: Optional[ConditionTextSpec] = None) -> str:
    """Join the allowed fields of one metadata row and normalise the result.

    Missing fields are skipped, not filled: a row without `study_title` must
    map to the same string as the same row with an empty one, or one
    experiment gets two embeddings.
    """
    sp = spec or ConditionTextSpec()
    parts: List[str] = []
    for f in sp.fields:
        v = row.get(f, "")
        v = "" if v is None else str(v).strip()
        if v:
            parts.append(v)
    joined = sp.separator.join(parts)
    return normalise_text(joined, sp)


def normalisation_ladder(texts: Sequence[str],
                         levels: Sequence[int] = (0, 1, 2, 3),
                         fields: Tuple[str, ...] = ("condition",),
                         ) -> Dict[str, object]:
    """How many distinct strings survive each level. A measurement, not a knob.

    Reference values on the `condition` column: 1025 raw, 505 / 469 / 373 at
    levels 1-3. A normaliser far below 505 at level 1 is merging experiments.
    """
    out: Dict[str, object] = {"n_texts": len(texts)}
    for lv in levels:
        sp = ConditionTextSpec(fields=fields, level=int(lv))
        norm = [normalise_text(t, sp) for t in texts]
        uniq = len(set(norm))
        out["level_%d" % lv] = {
            "name": NORMALISATION_LEVELS[int(lv)][0],
            "unique": uniq,
            "samples_per_string": (float(len(texts)) / uniq) if uniq else
            float("nan"),
            "singletons": int(sum(1 for _, c in _counts(norm).items()
                                  if c == 1)),
        }
    return out


def _counts(items: Iterable[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for it in items:
        d[it] = d.get(it, 0) + 1
    return d


# Categories
@dataclass
class CategoryVocabulary:
    """Category names in a FIXED order plus the English phrase for each.

    The order is a checkpoint contract: reordering it changes what every
    saved logit means. `add()` is the whole mechanism behind "a new condition
    without retraining", but on unseen conditions the text model reaches R^2
    0.03430 against a ceiling of 0.7392: the interface generalises, the
    accuracy does not.
    """

    names: Tuple[str, ...] = tuple(DEFAULT_CATEGORY_PHRASES)
    phrases: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_PHRASES))

    def __post_init__(self) -> None:
        missing = [n for n in self.names if n not in self.phrases]
        if missing:
            raise ValueError("no English phrase for categories %r; a category "
                             "without a phrase cannot be embedded by the "
                             "frozen LM (D2)" % (missing,))

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def phrase_list(self) -> List[str]:
        return [self.phrases[n] for n in self.names]

    def add(self, name: str, phrase: str) -> "CategoryVocabulary":
        """Append a category. New names go at the END; existing indices hold."""
        if name in self.names:
            raise ValueError("category %r already present at index %d"
                             % (name, self.index(name)))
        ph = dict(self.phrases)
        ph[name] = phrase
        return CategoryVocabulary(names=tuple(self.names) + (name,), phrases=ph)

    def to_dict(self) -> Dict[str, object]:
        return {"names": list(self.names),
                "phrases": {n: self.phrases[n] for n in self.names}}


# Text embeddings: cache, hash substitute, frozen LM
def _key_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def hash_embedding(text: str, dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic SUBSTITUTE embedding of a string. Carries no semantics.

    It exists so an offline smoke gate measures shapes, wiring and
    normalisation rather than nothing. Equal strings map to equal vectors,
    which exercises every code path and is exactly why it must never be read
    as evidence about the text channel; `substitute_fraction` counts it.
    """
    d = int(dim)
    digest = hashlib.sha256(("%d|%s" % (int(seed), text)).encode("utf-8"))
    state = int.from_bytes(digest.digest()[:8], "big") % (2 ** 32)
    rng = np.random.RandomState(state)
    v = rng.normal(0.0, 1.0, size=d).astype(np.float32)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


class ConditionEmbeddingCache:
    """Normalised string -> frozen embedding, with the LM identity attached.

    Checkpoint item 7: a different snapshot of the same repo id shifts every
    vector silently, so `load()` raises on a metadata mismatch rather than
    serving a plausible wrong vector.
    """

    def __init__(self, meta: Mapping[str, object],
                 keys: Optional[Sequence[str]] = None,
                 vectors: Optional[np.ndarray] = None) -> None:
        self.meta: Dict[str, object] = dict(meta)
        ks = list(keys or [])
        if vectors is None:
            dim = int(self.meta.get("dim", 0))
            vectors = np.zeros((0, dim), dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(ks):
            raise ValueError("cache has %d keys and %d vectors"
                             % (len(ks), vectors.shape[0]))
        self._index: Dict[str, int] = {k: i for i, k in enumerate(ks)}
        self._keys: List[str] = ks
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size or \
            self.vectors.shape[1] else int(self.meta.get("dim", 0))

    def get(self, text: str) -> Optional[np.ndarray]:
        i = self._index.get(_key_of(text))
        return None if i is None else self.vectors[i]

    def put(self, text: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        k = _key_of(text)
        if k in self._index:
            self.vectors[self._index[k]] = v[0]
            return
        if self.vectors.shape[0] == 0:
            self.vectors = v.copy()
        else:
            if v.shape[1] != self.vectors.shape[1]:
                raise ValueError("vector has %d dims, cache has %d"
                                 % (v.shape[1], self.vectors.shape[1]))
            self.vectors = np.concatenate([self.vectors, v], axis=0)
        self._index[k] = len(self._keys)
        self._keys.append(k)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        np.savez(path, keys=np.array(self._keys, dtype=object),
                 vectors=self.vectors,
                 meta=np.array(json.dumps(self.meta, sort_keys=True),
                               dtype=object))
        return path

    @staticmethod
    def load(path: str, expect: Optional[Mapping[str, object]] = None
             ) -> "ConditionEmbeddingCache":
        with np.load(path, allow_pickle=True) as z:
            keys = [str(k) for k in z["keys"].tolist()]
            vectors = np.asarray(z["vectors"], dtype=np.float32)
            meta = json.loads(str(z["meta"].item()))
        cache = ConditionEmbeddingCache(meta, keys, vectors)
        if expect:
            cache.assert_compatible(expect)
        return cache

    def assert_compatible(self, expect: Mapping[str, object]) -> None:
        """Raise unless model id, revision, pooling, dim and fingerprint match.

        `dim` is in the list because it is the one field a wrong cache
        carries through the constructor, to fail later inside `encode` where
        the traceback no longer names the cache.
        """
        for k in ("model_id", "revision", "pooling", "l2_normalise",
                  "dim", "text_fingerprint"):
            if k not in expect:
                continue
            have, want = self.meta.get(k), expect[k]
            if k == "dim":
                # A cache that does not record its dim still has one; read it
                # off the vectors rather than skipping the only check that
                # catches a 768-dim flank cache handed to a 384-dim encoder.
                if have is None and self.vectors.shape[0] > 0:
                    have = int(self.vectors.shape[1])
                if have is None:
                    continue
                have, want = int(have), int(want)
            if have != want:
                raise ValueError(
                    "embedding cache %s=%r but this run asks for %r. A cache "
                    "from another model or another normalisation rule moves "
                    "every vector without moving a single weight."
                    % (k, have, want))


class FrozenTextEncoder:
    """Normalised string -> vector, from cache, then a frozen LM, then a hash.

    Counts per source are reported: the cache is the only one that proves
    training and inference used the same model, `live` imports
    `transformers` inside the call so this file needs no network, and `hash`
    carries nothing. Whether a frozen LM beats a 442-token TF-IDF is open --
    TF-IDF reaches 97.1 % of the lookup-table ceiling in vocabulary, so the
    deciding threshold is emb64 > 0.09303 under the study split, below which
    TF-IDF + SVD(64) is the honest implementation of this layer.
    """

    def __init__(self, spec: Optional[ConditionTextSpec] = None,
                 dim: int = 384,
                 model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 revision: Optional[str] = None,
                 pooling: str = "mean",
                 l2_normalise: bool = True,
                 cache: Optional[ConditionEmbeddingCache] = None,
                 allow_live: bool = False,
                 allow_hash: bool = True,
                 hash_seed: int = 0) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls', got %r"
                             % (pooling,))
        self.spec = spec or ConditionTextSpec()
        self.dim = int(dim)
        self.model_id = str(model_id)
        self.revision = revision
        self.pooling = pooling
        self.l2_normalise = bool(l2_normalise)
        self.allow_live = bool(allow_live)
        self.allow_hash = bool(allow_hash)
        self.hash_seed = int(hash_seed)
        self.cache = cache
        self._lm = None
        self._tok = None
        self.counts: Dict[str, int] = {"cache": 0, "live": 0, "hash": 0}
        if cache is not None:
            cache.assert_compatible(self.identity())

    def identity(self) -> Dict[str, object]:
        """Everything that decides the value of the returned vector.

        Checkpoint item 1. `revision=None` is reported as a warning: a repo
        that moves changes a frozen checkpoint's predictions.
        """
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "l2_normalise": self.l2_normalise,
            "dim": self.dim,
            "text_fingerprint": self.spec.fingerprint(),
            "text_version": TEXT_NORMALISATION_VERSION,
            "revision_pinned": self.revision is not None,
        }

    def _load_live(self) -> None:
        if self._lm is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433
        kw = {} if self.revision is None else {"revision": self.revision}
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
        self._lm = AutoModel.from_pretrained(self.model_id, **kw).eval()
        for p in self._lm.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _embed_live(self, texts: Sequence[str]) -> np.ndarray:
        self._load_live()
        batch = self._tok(list(texts), padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        out = self._lm(**batch).last_hidden_state
        if self.pooling == "cls":
            v = out[:, 0]
        else:
            m = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            v = (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        if self.l2_normalise:
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return v.float().cpu().numpy()

    def encode(self, texts: Sequence[str], normalise: bool = True
               ) -> np.ndarray:
        """Embed a list of strings; returns [n, dim] float32.

        `normalise=True` applies the contract rules first; the cache is keyed
        by the NORMALISED string, so raw text misses every hit silently.
        """
        items = [normalise_text(t, self.spec) if normalise else t
                 for t in texts]
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        pending: List[int] = []
        for i, t in enumerate(items):
            v = self.cache.get(t) if self.cache is not None else None
            if v is not None:
                if v.shape[-1] != self.dim:
                    raise ValueError("cache vector has %d dims, encoder %d"
                                     % (v.shape[-1], self.dim))
                out[i] = v
                self.counts["cache"] += 1
            else:
                pending.append(i)
        if pending and self.allow_live:
            vals = self._embed_live([items[i] for i in pending])
            if vals.shape[1] != self.dim:
                raise ValueError(
                    "live model returns %d dims, encoder configured for %d"
                    % (vals.shape[1], self.dim))
            for j, i in enumerate(pending):
                out[i] = vals[j]
                self.counts["live"] += 1
                if self.cache is not None:
                    self.cache.put(items[i], vals[j])
            pending = []
        if pending:
            if not self.allow_hash:
                raise RuntimeError(
                    "%d strings have no cached embedding, the live model is "
                    "disabled and the hash substitute is disabled too"
                    % (len(pending),))
            for i in pending:
                out[i] = hash_embedding(items[i], self.dim, self.hash_seed)
                self.counts["hash"] += 1
        return out

    def encode_categories(self, vocab: CategoryVocabulary) -> np.ndarray:
        """Embed the English phrases (D2). Category names are never one-hot."""
        return self.encode(vocab.phrase_list())

    def report(self, vectors: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Source counts plus, when vectors are given, the collapse checks.

        `cos_mean` and `eff_rank` say whether the text space is usable at
        all; thresholds set beforehand are < 0.95 and > 20.
        """
        total = sum(self.counts.values())
        rep: Dict[str, object] = {
            "counts": dict(self.counts),
            "n_encoded": total,
            "substitute_fraction": (float(self.counts["hash"]) / total)
            if total else float("nan"),
            "cache_size": (len(self.cache) if self.cache is not None else 0),
        }
        rep.update(self.identity())
        if vectors is not None and len(vectors) > 1:
            v = np.asarray(vectors, dtype=np.float64)
            n = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
            g = n @ n.T
            iu = np.triu_indices(g.shape[0], k=1)
            rep["cos_mean"] = float(g[iu].mean())
            rep["cos_p95"] = float(np.percentile(g[iu], 95))
            rep["eff_rank"] = participation_ratio(v)
            rep["dim_used"] = int(v.shape[1])
        return rep


# Configuration
@dataclass
class ConditionConfig:
    """Config keys of this layer. Defaults reproduce the specification.

    `nash_delta_player=False` is not a preference: a task with 14.78 % of
    the variance has a small gradient on the shared trunk, and NashMTL gives
    the largest alpha to the smallest gradient. Enabling it means measuring
    gradient norms on `model.codes` first and holding alpha_delta <= 1.2.
    """

    cond_text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cond_text_revision: Optional[str] = None       # REQUIRED for a real run
    cond_text_fields: Tuple[str, ...] = DEFAULT_TEXT_FIELDS
    cond_text_level: int = 1
    cond_d_lm: int = 384
    cond_d_dna: int = 768
    cond_d_ctx: int = 224                          # = trunk_dim
    cond_heads: int = 8
    cond_level: int = 2                            # ladder level of the fusion
    cond_gate_init: float = 0.02
    cond_share: float = 0.3                       # fixed share of the context
    cond_cat_aux_weight: float = 0.1
    #: Text channel dropout; chosen, not measured. Live only when
    #: `conditioning.experiment_text.enabled` is true, otherwise `_resolve`
    #: returns at its `x is None` branch. It replaces a real vector by
    #: `unknown_token_text` on 20 % of PAIR GROUPS, which is what calibrates
    #: classifier-free guidance.
    cond_dropout_text: float = 0.2
    #: Species channel dropout; the value is chosen, the reason measured:
    #: this was a literal 0.0 at the species `_resolve` call, so
    #: `unknown_token_species` trained only on rows that have no species.
    cond_dropout_species: float = 0.15
    cond_flank_rank: Optional[int] = None          # None = full rank
    cat_head_mode: str = "prototype"               # prototype | linear
    delta_weight: float = 0.15                     # subordinate term (D5)
    nash_delta_player: bool = False                # D5

    def to_dict(self) -> Dict[str, object]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ConditionOutput:
    """What the encoder returns.

    `tokens` is [B, T, d_ctx] in the order (species, text, flank left, flank
    right) -- fixed, because it is the key order of the fusion softmax.
    """

    tokens: torch.Tensor
    cat_logits: torch.Tensor
    text_vector: torch.Tensor
    diagnostics: Dict[str, float]

    #: [B], 1.0 where the description token really reached this row's
    #: conditioning, 0.0 where it was dropped (`cond_dropout_text`) or
    #: unavailable. The categoriser trains only on rows with 0.0; on the rest
    #: the category is already on the input and the task is a tautology.
    text_used: Optional[torch.Tensor] = None

    #: Token order, a contract: it is the key order of the fusion softmax.
    #: "gene" is appended at index 4 and present only when that channel is
    #: built, so readers must slice by NAME through `TokenSlicer` -- with a
    #: four-token checkpoint and a five-token build a literal offset
    #: silently returns the wrong channel.
    TOKEN_NAMES: ClassVar[Tuple[str, ...]] = ("species", "text", "flank_l",
                                              "flank_r", "gene")


# The encoder
class ConditionEncoder(nn.Module):
    """Four context tokens plus an auxiliary category head.

    Inputs, all optional, a missing one resolving to its learned unknown
    token::

        text_emb        [B, d_lm]    frozen LM output, precomputed and cached
        species_id      [B]          index into the strain table, or
        species_tangent [B, d_sp]    logmap0 of the Lorentz strain point
        flank_l/flank_r [B, d_dna]   frozen DNABERT, cached
        text_ok         [B]          1.0 where a description exists for this
                                     row; a 0 row reaches the unknown token,
                                     never `text_proj(zeros)`

    Output is `tokens` [B, T, d_ctx], RMS normalised per source and per
    sample, and `cat_logits` [B, n_cat], which is never an input.

    C cannot depend on the promoter: `forward` takes no sequence argument, so
    the siamese contract ("both elements of a pair see the same C") holds
    structurally rather than by discipline. The one batch-composition
    dependence left is `text_center` in train mode, where both elements of a
    pair are in the same forward anyway; eval uses the running statistics,
    which it must keep because a single new condition at inference has no
    batch to centre against.

    The unknown tokens are learned rather than zero, because zero is a legal
    point of the space the model has already learned something about.
    `cond_dropout_text` trains the text one by replacing the token per
    sample, which also calibrates classifier-free guidance at sampling time.
    """

    def __init__(self, config: Optional[ConditionConfig] = None,
                 vocab: Optional[CategoryVocabulary] = None,
                 n_strains: int = 20,
                 d_species: Optional[int] = None,
                 species_init: Optional[np.ndarray] = None,
                 strain_names: Optional[Sequence[str]] = None,
                 unplaced_to: Optional[Mapping[str, str]] = None,
                 category_embeddings: Optional[np.ndarray] = None,
                 gene: Optional[nn.Module] = None) -> None:
        super().__init__()
        cfg = config or ConditionConfig()
        if cfg.cat_head_mode not in ("prototype", "linear"):
            raise ValueError("cat_head_mode must be 'prototype' or 'linear', "
                             "got %r" % (cfg.cat_head_mode,))
        self.cfg = cfg
        self.vocab = vocab or CategoryVocabulary()
        d = int(cfg.cond_d_ctx)
        self.d_ctx = d

        # Gene path, appended at index 4. `gene` is a fully built
        # `GeneEncoder` or None, injected rather than constructed here because
        # it needs the orthology feature spec, a data artefact: keeping that
        # lookup in `assemble` leaves this module importable without files.
        self.gene = gene
        if self.gene is not None and int(getattr(self.gene, "d_ctx", d)) != d:
            raise ValueError(
                "the gene encoder emits a %d-wide token but the condition "
                "encoder is %d-wide. The tokens are STACKED, so a width "
                "mismatch is not broadcastable and a silent pad would put "
                "zeros where a channel should be."
                % (int(self.gene.d_ctx), d))
        self.n_tokens = 4 + (1 if self.gene is not None else 0)
        #: The names THIS build emits, in stacking order. `TokenSlicer` and the
        #: fusion block read this; nothing may index the tokens by a literal.
        self.token_names: Tuple[str, ...] = \
            ConditionOutput.TOKEN_NAMES[:self.n_tokens]

        # Text path. affine=False means zero learnable parameters, but
        # running_mean and running_var are saved in the state dict, which is
        # the point.
        self.text_proj = nn.Linear(int(cfg.cond_d_lm), d)
        self.text_center = nn.BatchNorm1d(d, affine=False)

        # Flank path, shared by both sides plus a side token. The frozen
        # DNABERT flank space is nearly collapsed, so a rank-32 factorisation
        # loses almost nothing; full rank is the default only because the
        # specification says Linear(d_dna, d_ctx).
        if cfg.cond_flank_rank:
            r = int(cfg.cond_flank_rank)
            self.flank_proj = nn.Sequential(
                nn.Linear(int(cfg.cond_d_dna), r, bias=False),
                nn.Linear(r, d))
        else:
            self.flank_proj = nn.Linear(int(cfg.cond_d_dna), d)
        self.side_token = nn.Parameter(torch.randn(2, d) * 0.02)

        self.n_strains = int(n_strains)
        self.d_species = int(d_species) if d_species else None
        # Species path. The strain table is built only without a tangent
        # path: `forward` prefers `species_tangent` and `GenomeEmbedding`
        # always supplies one when it exists, so with `d_species` the
        # `elif species_id` branch is unreachable and the table was dead
        # (measured gradient exactly 0.000e+00).
        self.species_emb: Optional[nn.Embedding] = None
        self.species_proj: Optional[nn.Linear] = None
        if self.d_species:
            self.species_proj = nn.Linear(self.d_species, d)
        else:
            self.species_emb = nn.Embedding(self.n_strains, d)
            nn.init.normal_(self.species_emb.weight, std=0.02)
        if species_init is not None:
            self.init_species_from_tangents(species_init)
        self.strain_names: Tuple[str, ...] = tuple(strain_names or ())
        self.unplaced_to: Dict[str, str] = dict(unplaced_to or {})

        # Learned unknown tokens (checkpoint item 5). Small but non-zero, and
        # one per source: a single shared null would make "no text" and "no
        # flank" the same event.
        self.unknown_token_text = nn.Parameter(torch.randn(d) * 0.02)
        self.unknown_token_species = nn.Parameter(torch.randn(d) * 0.02)
        self.unknown_token_flank = nn.Parameter(torch.randn(d) * 0.02)

        # Category head
        n_cat = len(self.vocab)
        if cfg.cat_head_mode == "linear":
            self.cat_linear: Optional[nn.Linear] = nn.Linear(d, n_cat)
        else:
            self.cat_linear = None
        # Shared scalars: their count does not depend on the number of
        # categories, which is what keeps `add_category` free. The literals
        # live in module constants so `falsifiers()` compares against the
        # same numbers the constructor writes, not a copy that can drift.
        self.logit_scale = nn.Parameter(torch.tensor(_CAT_SCALE_INIT))
        self.logit_bias = nn.Parameter(torch.tensor(_CAT_BIAS_INIT))
        proto = (np.zeros((n_cat, int(cfg.cond_d_lm)), dtype=np.float32)
                 if category_embeddings is None
                 else np.asarray(category_embeddings, dtype=np.float32))
        if proto.shape != (n_cat, int(cfg.cond_d_lm)):
            raise ValueError("category_embeddings must be [%d, %d], got %s"
                             % (n_cat, cfg.cond_d_lm, proto.shape))
        self.register_buffer("category_lm", torch.from_numpy(proto.copy()))
        self._has_prototypes = bool(category_embeddings is not None)
        # Per-class offset, zero parameters (D2). Zeros mean "no offset", so
        # a build that never calls `set_category_prior` is bit identical.
        # Non-persistent: it is a corpus statistic, and a persistent buffer
        # would give every existing checkpoint a missing key.
        self.register_buffer("category_log_prior", torch.zeros(n_cat),
                             persistent=False)
        self._has_prior = False

        #: [B] mask of the LAST channel passed through `_resolve`; read it
        #: immediately after the text call, since the species and flank calls
        #: overwrite it.
        self._last_keep: Optional[torch.Tensor] = None
        self._diag: Dict[str, float] = {}

    def init_species_from_tangents(self, tangents: np.ndarray) -> float:
        """Seed the strain table from logmap0 of the Lorentz tree points.

        The geometry is owned by the species module; this only places its
        tangent vectors into the first coordinates, scaled to unit RMS. The
        returned Pearson correlation of the two pairwise distance matrices is
        the only way to notice an initialisation that silently did nothing.
        """
        if self.species_emb is None:
            raise ValueError(
                "init_species_from_tangents() has no table to seed: this "
                "encoder was built with d_species=%r, so the species channel "
                "reads the tangent DIRECTLY through species_proj and the "
                "strain table does not exist. Seeding it would have written "
                "into a tensor that forward() never reads -- which is what "
                "this build used to do silently." % (self.d_species,))
        t = np.asarray(tangents, dtype=np.float32)
        if t.ndim != 2 or t.shape[0] != self.n_strains:
            raise ValueError("species tangents must be [%d, d], got %s"
                             % (self.n_strains, t.shape))
        k = min(t.shape[1], self.d_ctx)
        rms = float(np.sqrt((t[:, :k] ** 2).mean())) or 1.0
        with torch.no_grad():
            self.species_emb.weight[:, :k] = torch.from_numpy(t[:, :k] / rms)
        a = t[:, :k] / rms
        b = self.species_emb.weight.detach().cpu().numpy()
        da = _pairwise(a)
        db = _pairwise(b)
        if da.size < 2:
            return float("nan")
        return float(np.corrcoef(da, db)[0, 1])

    def set_category_prototypes(self, embeddings: np.ndarray) -> None:
        """Install the LM embeddings of the category phrases (D2)."""
        e = np.asarray(embeddings, dtype=np.float32)
        if e.shape != (len(self.vocab), int(self.cfg.cond_d_lm)):
            raise ValueError("prototypes must be [%d, %d], got %s"
                             % (len(self.vocab), self.cfg.cond_d_lm, e.shape))
        self.category_lm = torch.from_numpy(e.copy()).to(
            self.category_lm.device)
        self._has_prototypes = True

    def set_category_prior(self, counts) -> None:
        """Install the per-class offset from CORPUS COUNTS. Zero parameters.

        The offset is `log(p_c) - mean_c log(p_c)`, centred so that it moves
        the columns relative to each other and leaves `logit_bias` meaning
        what it meant; a zero count is floored at half a row, since `log 0`
        would make that column unreachable for good. It fixes the part that
        is arithmetic: the class 54.5x commoner wins the argmax on frequency.
        """
        c = np.asarray(counts, dtype=np.float64).reshape(-1)
        if c.shape[0] != len(self.vocab):
            raise ValueError(
                "category counts must have one entry per category: got %d "
                "for %d categories. Passing the 7 PARENT counts to a vocab "
                "that carries parents AND subcategories is the shape this "
                "guard exists for -- it would have silently offset the first "
                "seven columns and left the rest at zero."
                % (c.shape[0], len(self.vocab)))
        if (c < 0).any():
            raise ValueError("category counts must be non-negative, got min "
                             "%r" % (float(c.min()),))
        if c.sum() <= 0:
            raise ValueError("category counts are all zero: there is no "
                             "prior to install and a uniform one is what the "
                             "buffer already holds")
        p = np.maximum(c, 0.5) / max(float(np.maximum(c, 0.5).sum()), 1e-12)
        lp = np.log(p)
        lp = lp - lp.mean()
        self.category_log_prior = torch.from_numpy(
            lp.astype(np.float32)).to(self.category_log_prior.device)
        self._has_prior = True

    def add_category(self, name: str, phrase: str,
                     embedding: Optional[np.ndarray] = None) -> int:
        """Append a category IN PLACE and return its index.

        Under `cat_head_mode="prototype"` this adds a buffer row and no
        parameters, which is the mechanical content of D2; under `"linear"`
        it raises rather than reinitialising a trained head.
        """
        if self.cfg.cat_head_mode == "linear":
            raise RuntimeError(
                "cat_head_mode='linear' encodes the category count in the head "
                "weights; adding a category there means retraining the head. "
                "Use the default 'prototype' head for D2.")
        if embedding is None and self._has_prototypes:
            # A zero row is not a neutral prototype: `text_proj(0)` is the
            # bias, so the new category's logit becomes a constant.
            raise ValueError(
                "category %r needs its LM embedding: this encoder already "
                "carries prototypes, and a zero row would make the new "
                "category's logit a constant. Pass "
                "FrozenTextEncoder.encode([phrase])[0]." % (name,))
        self.vocab = self.vocab.add(name, phrase)
        row = (np.zeros((1, int(self.cfg.cond_d_lm)), dtype=np.float32)
               if embedding is None
               else np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        if row.shape[1] != int(self.cfg.cond_d_lm):
            raise ValueError("category embedding must have %d dims, got %d"
                             % (self.cfg.cond_d_lm, row.shape[1]))
        self.category_lm = torch.cat(
            [self.category_lm, torch.from_numpy(row).to(
                self.category_lm.device, self.category_lm.dtype)], dim=0)
        # The offset grows with the vocabulary, or the next forward would
        # broadcast a short row against a wider logit matrix; a new category
        # has no corpus count, so its offset is the neutral 0.0.
        self.category_log_prior = torch.cat(
            [self.category_log_prior,
             torch.zeros(1, device=self.category_log_prior.device,
                         dtype=self.category_log_prior.dtype)], dim=0)
        return len(self.vocab) - 1

    def strain_index(self, name: str) -> Tuple[int, bool]:
        """Resolve a strain name to a row, following the unplaced-strain rule.

        Returns (index, placed). The rule lives in the checkpoint: the trees
        carry 10 taxa against 20 strains, so a strain without a leaf is the
        norm rather than the exception.
        """
        if name in self.strain_names:
            return self.strain_names.index(name), True
        donor = self.unplaced_to.get(name)
        if donor is not None and donor in self.strain_names:
            return self.strain_names.index(donor), False
        raise KeyError(
            "strain %r has neither a row nor an `unplaced_to` rule; refusing "
            "to guess (the fallback must be recorded in the checkpoint)"
            % (name,))

    # The forward pass
    def _centre_text(self, x: torch.Tensor, use_running: bool
                     ) -> torch.Tensor:
        if not use_running:
            return self.text_center(x)
        m = self.text_center.running_mean
        v = self.text_center.running_var
        return (x - m) / torch.sqrt(v + self.text_center.eps)

    def _resolve(self, x: Optional[torch.Tensor], unknown: torch.Tensor,
                 batch: int, drop_p: float, device: torch.device,
                 dtype: torch.dtype,
                 group: Optional[torch.Tensor] = None,
                 avail: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fill in a missing source and apply dropout, per GROUP not per row.

        The operands are masked, not the result: `torch.where` on the output
        still backpropagates through the discarded branch. `group` shares the
        draw between rows with the same id, because the siamese head scores
        one condition against two promoters and a per-row draw would hand a
        pair different C with probability 2p(1-p) = 18 % at p = 0.1.
        """
        null = unknown.to(dtype).unsqueeze(0).expand(batch, -1)
        zero = torch.zeros((), device=device, dtype=dtype)
        if x is None:
            # Every exit sets `_last_keep`; this one did not, and the readout
            # then met an attribute that did not exist yet. An absent channel
            # reached no row, so the mask is zeros -- which is the correct
            # answer for the categoriser, not a missing one.
            self._last_keep = torch.zeros(batch, device=device, dtype=dtype)
            return null, zero + 1.0
        x = x.to(dtype)
        # `avail` is per row and is not a dropout: dropout teaches the model
        # to work without a channel it has, availability says the channel
        # does not exist for this row. A row at avail=0 must reach the
        # unknown token, not `proj(zeros)` -- the projection has a bias, so
        # zero input is a learned constant, i.e. a channel reporting a share
        # while carrying no information.
        a = (None if avail is None
             else avail.to(device=device, dtype=dtype).reshape(-1, 1))
        if self.training and drop_p > 0.0:
            if group is None:
                keep = (torch.rand(batch, 1, device=device) >= drop_p).to(dtype)
            else:
                g = group.to(device=device).reshape(-1).long()
                uniq, inv = torch.unique(g, return_inverse=True)
                draw = (torch.rand(uniq.numel(), device=device) >= drop_p)
                keep = draw[inv].to(dtype).unsqueeze(1)
            if a is not None:
                keep = keep * a
            self._last_keep = keep.detach().reshape(-1)
            return keep * x + (1.0 - keep) * null, (1.0 - keep.mean())
        if a is not None:
            self._last_keep = a.detach().reshape(-1)
            return a * x + (1.0 - a) * null, (1.0 - a.mean())
        self._last_keep = torch.ones(batch, device=device, dtype=dtype)
        return x, zero

    def forward(self, text_emb: Optional[torch.Tensor] = None,
                species_id: Optional[torch.Tensor] = None,
                species_tangent: Optional[torch.Tensor] = None,
                flank_l: Optional[torch.Tensor] = None,
                flank_r: Optional[torch.Tensor] = None,
                flank_ok: Optional[torch.Tensor] = None,
                text_ok: Optional[torch.Tensor] = None,
                gene_numeric: Optional[torch.Tensor] = None,
                gene_identity: Optional[torch.Tensor] = None,
                gene_text: Optional[torch.Tensor] = None,
                gene_ok: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None,
                group: Optional[torch.Tensor] = None) -> ConditionOutput:
        if self.gene is None and any(t is not None for t in
                                     (gene_numeric, gene_identity, gene_text,
                                      gene_ok)):
            # Dropping them silently is how a channel becomes decoration: the
            # caller would see a context vector, a share, and no gene in it.
            raise ValueError(
                "gene inputs were given but this encoder was built without a "
                "gene channel (conditioning.gene.enabled is false). Enable "
                "the channel or stop passing the inputs -- the tokens are "
                "stacked positionally and there is nowhere to put these.")
        ref = next((t for t in (text_emb, species_tangent, flank_l, flank_r,
                                species_id, gene_numeric, gene_text)
                    if t is not None), None)
        if ref is None and batch_size is None:
            raise ValueError("no input and no batch_size: nothing says how "
                             "many samples to build unknown tokens for")
        B = int(batch_size if ref is None else ref.shape[0])
        device = self.text_proj.weight.device
        dtype = self.text_proj.weight.dtype

        raw_rms: Dict[str, float] = {}

        # Text
        if text_emb is not None:
            te = text_emb.to(device=device, dtype=dtype)
            raw_rms["text"] = float(te.pow(2).mean().sqrt().item())
            t_proj = self.text_proj(te)
            t_proj = self._centre_text(t_proj, use_running=not self.training)
        else:
            t_proj = None
        # `text_used` is read immediately after this call, because `_resolve`
        # runs again for species and flanks and `_last_keep` always describes
        # the last channel. The categoriser needs it: trained on rows that
        # saw the description, it would predict the category from text that
        # encodes it -- a tautology the present `cat_logits` exemplifies
        # (untrained model 0.9295 AUC against 0.8032 trained).
        t_tok, text_dropped = self._resolve(
            t_proj, self.unknown_token_text, B,
            float(self.cfg.cond_dropout_text), device, dtype, group=group,
            avail=text_ok)
        text_used = self._last_keep
        text_vector = t_tok
        t_tok = _rms_normalise(t_tok)

        # Species
        if species_tangent is not None:
            if self.species_proj is None:
                raise ValueError(
                    "species_tangent given but the encoder was built without "
                    "d_species; construct with d_species=<tangent dim>")
            sv = species_tangent.to(device=device, dtype=dtype)
            raw_rms["species"] = float(sv.pow(2).mean().sqrt().item())
            s_proj = self.species_proj(sv)
        elif species_id is not None and self.species_emb is not None:
            idx = species_id.to(device=device).long()
            s_proj = self.species_emb(idx)
            raw_rms["species"] = float(s_proj.pow(2).mean().sqrt().item())
        else:
            s_proj = None
        # `group` is not cosmetic here: with a per-row draw the two rows of a
        # siamese pair get different conditions with probability
        # 2p(1-p) = 25.5 % at p = 0.15.
        s_tok, species_dropped = self._resolve(
            s_proj, self.unknown_token_species, B,
            float(self.cfg.cond_dropout_species), device, dtype,
            group=group)
        s_tok = _rms_normalise(s_tok)

        # Flanks. Coverage is reported per forward because the shares cannot
        # report it: every token is RMS normalised before the share is taken,
        # so they are 0.25 by construction whatever the inputs were. Coverage
        # and the raw input RMS separate a live cache from the unknown token.
        f_toks: List[torch.Tensor] = []
        cover = (1.0 if flank_ok is None
                 else float(flank_ok.to(dtype=torch.float32).mean().item()))
        for side, fx in enumerate((flank_l, flank_r)):
            if fx is not None:
                fv = fx.to(device=device, dtype=dtype)
                raw_rms["flank_l" if side == 0 else "flank_r"] = float(
                    fv.pow(2).mean().sqrt().item())
                p = self.flank_proj(fv) + self.side_token[side].unsqueeze(0)
            else:
                p = None
            tok, _ = self._resolve(p, self.unknown_token_flank, B, 0.0,
                                   device, dtype, avail=flank_ok)
            f_toks.append(_rms_normalise(tok))

        # Gene, appended at index 4
        stack = [s_tok, t_tok, f_toks[0], f_toks[1]]
        if self.gene is not None:
            g_out = self.gene(numeric=gene_numeric, identity=gene_identity,
                              text_emb=gene_text, available=gene_ok,
                              batch_size=B, group=group)
            # `GeneEncoder` already RMS-normalises its token; a second pass
            # would be a no-op on a unit vector and a lie if that module ever
            # stopped. The share printed below is what checks it.
            stack.append(g_out.token.to(device=device, dtype=dtype))
            raw_rms.update({"gene_" + k: v
                            for k, v in g_out.diagnostics.items()
                            if isinstance(v, float)})
        tokens = torch.stack(stack, dim=1)

        # Auxiliary category head
        cat_logits = self._category_logits(text_vector)

        with torch.no_grad():
            per_tok = tokens.pow(2).mean(dim=-1).sqrt()          # [B, T]
            share = (per_tok / per_tok.sum(dim=1, keepdim=True).clamp_min(
                _EPS)).mean(dim=0)
            names = ("gat", "tekst", "flankaL", "flankaR", "gen")[
                :tokens.shape[1]]
            self._diag = {"ktx_udzial_%s" % n: float(share[i].item())
                          for i, n in enumerate(names)}
            for k, v in raw_rms.items():
                self._diag["ktx_rms_we_%s" % k] = v
            self._diag["ktx_tekst_dropout"] = float(text_dropped)
            # Coverage of the text cache on this forward; 0.0 here with
            # text_emb given means the channel resolved to its unknown token
            # on every row.
            self._diag["ktx_tekst_pokrycie"] = (
                0.0 if text_emb is None else
                (1.0 if text_ok is None else float(
                    text_ok.to(dtype=torch.float32).mean().item())))
            # Realised species drop fraction of this forward; without it the
            # probability would be unobservable in the log.
            self._diag["ktx_species_dropout"] = float(species_dropped)
            self._diag["ktx_flanka_pokrycie"] = (
                cover if (flank_l is not None or flank_r is not None) else 0.0)
            self._diag["ktx_tekst_dev_nad_srednia"] = float(
                _deviation_over_mean(text_vector))
            # `ktx_*` keys reach the epoch line by prefix, which is the only
            # route this module has without touching `format_epoch`.
            # `ktx_cat_martwa` is a falsifier: 1.0 means both scalars stand
            # exactly on the constructor's literals, i.e. no gradient.
            self._diag["ktx_cat_skala"] = float(self.logit_scale.detach())
            self._diag["ktx_cat_bias"] = float(self.logit_bias.detach())
            self._diag["ktx_cat_martwa"] = float(
                float(self.logit_scale.detach()) == _CAT_SCALE_INIT
                and float(self.logit_bias.detach()) == _CAT_BIAS_INIT)
            # Spread of the logits across classes. At zero spread the argmax
            # is random and the reported AP is computed on a constant -- the
            # uninstalled-prototype failure seen from the other side.
            self._diag["ktx_cat_sd_klas"] = float(
                cat_logits.detach().float().std(dim=-1).mean())
            self._diag["ktx_cat_prior_zakres"] = float(
                self.category_log_prior.detach().max()
                - self.category_log_prior.detach().min())
        return ConditionOutput(tokens=tokens, cat_logits=cat_logits,
                               text_vector=text_vector,
                               diagnostics=dict(self._diag),
                               text_used=text_used)

    def _category_logits(self, text_vector: torch.Tensor) -> torch.Tensor:
        if self.cfg.cat_head_mode == "linear":
            assert self.cat_linear is not None
            return self.cat_linear(text_vector)
        # Prototype head. The prototypes go through the same projection and
        # centring as the samples, but always with the RUNNING statistics:
        # seven prototype rows inside a batch of 96 would move the running
        # mean that a single-sample inference then depends on.
        proto = self.category_lm.to(dtype=text_vector.dtype,
                                    device=text_vector.device)
        p = self._centre_text(self.text_proj(proto), use_running=True)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        q = text_vector / text_vector.norm(dim=-1, keepdim=True).clamp_min(
            _EPS)
        # `logit_bias` is one number for all columns and cannot express the
        # 54.5x class spread; `category_log_prior` is one per column and can,
        # at zero parameters. It defaults to zeros, so this line is an
        # identity until `set_category_prior` is called.
        prior = self.category_log_prior.to(dtype=text_vector.dtype,
                                           device=text_vector.device)
        return (self.logit_scale * (q @ p.t()) + self.logit_bias
                + prior[None, :])

    # Losses and their status in the optimisation
    def category_loss(self, cat_logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """Multi-label BCE over the category vector.

        Multi-label, not softmax: 66 of the 1498 kept samples carry two
        categories. Refuses to run on an uninstalled prototype head, where
        every row projects to the bias of `text_proj` and the head emits
        n_cat identical logits -- a cost with no effect, invisible in the log.
        """
        if self.cfg.cat_head_mode == "prototype" and not self._has_prototypes:
            raise RuntimeError(
                "the prototype category head has no prototypes installed, so "
                "all %d logits are identical (measured sd across categories "
                "0.000e+00) and this loss is a cost with no gradient direction "
                "between categories; call set_category_prototypes(...) with "
                "the LM embeddings of vocab.phrase_list()" % (len(self.vocab),))
        return nn.functional.binary_cross_entropy_with_logits(
            cat_logits, targets.to(cat_logits.dtype))

    def nash_players(self) -> Tuple[str, ...]:
        """Tasks of this layer that enter the Nash bargain. Empty by default.

        With the flag on, `delta` becomes a player AND leaves
        `subordinate_terms()`, so a trainer summing both cannot double count.
        """
        return ("delta",) if self.cfg.nash_delta_player else ()

    def subordinate_terms(self) -> Dict[str, float]:
        """Fixed-weight terms this layer contributes to the total loss.

        delta_weight = 0.15 comes from the 85.22/14.78 variance split, not
        from a sweep.
        """
        terms = {"cond_cat_aux": float(self.cfg.cond_cat_aux_weight)}
        if not self.cfg.nash_delta_player:
            terms["delta"] = float(self.cfg.delta_weight)
        return terms

    def no_decay_names(self) -> List[str]:
        """Parameters that must be exempt from weight decay, by name.

        The project default substring list does not cover `unknown_token_*`,
        so the trainer has to use `CONDITION_NO_DECAY_SUBSTRINGS`; whether it
        does is reported rather than assumed.
        """
        out = []
        for n, _ in self.named_parameters():
            low = n.lower()
            if ("unknown_token" in low or "side_token" in low
                    or "logit_" in low or "bias" in low):
                out.append(n)
        return out

    # Observables
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def report(self, cat_logits: Optional[torch.Tensor] = None,
               cat_targets: Optional[torch.Tensor] = None,
               attention: Optional[Mapping[str, float]] = None
               ) -> Dict[str, object]:
        """Everything the log needs. Keys follow the specification's contract.

        `ktx_*` names are the logging contract from the specification: a run
        without them is unobservable, and renaming one breaks every parser.
        """
        rep: Dict[str, object] = {
            "n_parameters": self.parameter_count(),
            "n_categories": len(self.vocab),
            "categories": list(self.vocab.names),
            "cat_head_mode": self.cfg.cat_head_mode,
            "has_category_prototypes": bool(self._has_prototypes),
            "d_ctx": self.d_ctx,
            "n_tokens": self.n_tokens,
            # The tokens THIS build emits, not the declared list: with the
            # gene channel off there are four, and reporting five would
            # describe a model that was not made.
            "token_order": list(self.token_names),
            "nash_players": list(self.nash_players()),
            "subordinate_terms": self.subordinate_terms(),
            "no_decay_names": self.no_decay_names(),
            "no_decay_covered_by_default": all(
                any(p in n.lower() for p in DEFAULT_NO_DECAY_PATTERNS)
                for n in self.no_decay_names()),
            "unknown_token_norm": {
                "text": float(self.unknown_token_text.detach().norm().item()),
                "species": float(
                    self.unknown_token_species.detach().norm().item()),
                "flank": float(self.unknown_token_flank.detach().norm().item()),
            },
            "text_center_running_saved": bool(
                "text_center.running_mean" in self.state_dict()),
        }
        rep.update({k: float(v) for k, v in self._diag.items()})
        if attention:
            rep.update({k: float(v) for k, v in attention.items()})
        if cat_logits is not None and cat_targets is not None:
            s = cat_logits.detach().float().cpu().numpy()
            y = cat_targets.detach().float().cpu().numpy()
            aps = [average_precision(s[:, j], y[:, j])
                   for j in range(min(s.shape[1], y.shape[1]))]
            finite = [a for a in aps if not math.isnan(a)]
            rep["cat_aux_ap"] = float(np.mean(finite)) if finite else float(
                "nan")
            rep["cat_aux_ap_per_class"] = [float(a) for a in aps]
            rep["cat_aux_prevalence"] = [float(v) for v in y.mean(axis=0)]
        return rep

    def contract(self, text_encoder: Optional[FrozenTextEncoder] = None
                 ) -> Dict[str, object]:
        """The eight things a checkpoint needs for a NEW condition to work.

        Written as data, because every item has a recorded failure mode: the
        LM identity, the literal normalisation rule, `text_proj` weights AND
        the `text_center` running statistics, the category order and phrases,
        the unknown tokens, the strain table with its no-leaf rule, the
        embedding cache with its hash, and the per-strain calibration fitted
        after training and therefore stored apart from the weights.
        """
        spec = (text_encoder.spec if text_encoder is not None
                else ConditionTextSpec(fields=self.cfg.cond_text_fields,
                                       level=self.cfg.cond_text_level))
        return {
            "1_lm_identity": (text_encoder.identity() if text_encoder
                              is not None else
                              {"model_id": self.cfg.cond_text_model,
                               "revision": self.cfg.cond_text_revision,
                               "revision_pinned":
                                   self.cfg.cond_text_revision is not None}),
            "2_text_normalisation": spec.to_dict(),
            "3_text_path": ["text_proj.weight", "text_proj.bias",
                            "text_center.running_mean",
                            "text_center.running_var",
                            "text_center.num_batches_tracked"],
            "4_categories": self.vocab.to_dict(),
            "5_unknown_tokens": ["unknown_token_text", "unknown_token_species",
                                 "unknown_token_flank"],
            "6_strains": {"names": list(self.strain_names),
                          "unplaced_to": dict(self.unplaced_to),
                          "n_rows": self.n_strains},
            "7_embedding_cache": {
                "keyed_by": "sha1(normalised string)",
                "fingerprint": spec.fingerprint(),
                "size": (len(text_encoder.cache) if text_encoder is not None
                         and text_encoder.cache is not None else 0)},
            "8_calibration": {"per_strain_quantile_map": None,
                              "note": "fitted after training, stored apart "
                                      "from the weights"},
            "config": self.cfg.to_dict(),
        }

    def falsifiers(self, output: Optional[ConditionOutput] = None,
                   attention: Optional[Mapping[str, float]] = None
                   ) -> List[Tuple[str, bool, str]]:
        """(name, passed, detail). A check that cannot fail is not a check."""
        return falsifiers(self, output=output, attention=attention)

    def extra_repr(self) -> str:
        return ("d_ctx=%d, tokens=%d, categories=%d, head=%s, "
                "dropout_text=%.2f, dropout_species=%.2f"
                % (self.d_ctx, self.n_tokens,
                   len(self.vocab),
                   self.cfg.cat_head_mode,
                   self.cfg.cond_dropout_text,
                   self.cfg.cond_dropout_species))


#: The constructor's literals for the two shared scalars of the prototype
#: head, kept here so `falsifiers()` compares against the SAME numbers
#: `__init__` writes: a second copy is how a dead-knob check quietly stops
#: checking when one of the two is edited.
_CAT_SCALE_INIT = 10.0
_CAT_BIAS_INIT = -2.0


def _pairwise(a: np.ndarray) -> np.ndarray:
    d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(a.shape[0], k=1)
    return d[iu]


def _deviation_over_mean(x: torch.Tensor) -> float:
    """Per-sample deviation norm divided by the norm of the batch mean.

    Below ~0.2 the token is close to a constant and cross-attention over it
    is an expensive bias; a projection 91 % shared across samples measured
    0.0869, and 1.0701 after parameter-free centring.
    """
    if x.shape[0] < 2:
        return float("nan")
    mu = x.mean(dim=0, keepdim=True)
    dev = (x - mu).norm(dim=-1).mean()
    return float((dev / mu.norm().clamp_min(_EPS)).item())


# Fusion diagnostics; the fusion block itself lives in hyppe.model.encoder
@torch.no_grad()
def context_attention_stats(xattn: nn.Module, h: torch.Tensor,
                            context: torch.Tensor,
                            token_names: Sequence[str] =
                            ConditionOutput.TOKEN_NAMES) -> Dict[str, float]:
    """Attention mass per context token, and its spread along the sequence.

    Three falsifiers live in these numbers. F4: cross-attention must be
    POSITIONAL or it is an expensive FiLM, since with four keys the only
    thing gained over FiLM is that the mixture changes along L. F5: the text
    token must not be crowded out by the flanks, which ablate ~75x stronger,
    so its mean mass must be >= 0.05 and its spread across samples >= 0.01 --
    a constant mass means the token is ignored, only more expensively.

    F6, dominance, is NOT read from `cond_dom`: that module rescales its
    contribution to the map's RMS per sample before measuring rms(y)/rms(h),
    so the number is identically 1.0 and can falsify nothing. The live
    observable for a concatenate-and-merge fusion is `ktx_dom`, the norm
    ratio of the two halves of the merge kernel, ~1.0 at init by symmetry,
    so its bound is a training-time bound.

    The arithmetic mirrors `ConditionCrossAttention.forward`; the entropy is
    returned so it can be compared against that module's own `cond_H`, which
    is what makes a divergence between the two visible.
    """
    B, _, L = h.shape
    q = xattn.q(h).view(B, xattn.h, xattn.dk, L).transpose(-1, -2)
    c = context.to(h.dtype)
    k = xattn.k(c).view(B, c.shape[1], xattn.h, xattn.dk).transpose(1, 2)
    att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(xattn.dk), -1)
    # att: [B, heads, L, T]
    per_token = att.mean(dim=(1, 2))                      # [B, T]
    along_pos = att.mean(dim=1).std(dim=1)                # [B, T] over L
    out: Dict[str, float] = {}
    for i, name in enumerate(token_names[: att.shape[-1]]):
        out["ktx_masa_%s_srednia" % name] = float(per_token[:, i].mean())
        out["ktx_masa_%s_sd" % name] = float(per_token[:, i].std())
        out["ktx_poz_sd_%s" % name] = float(along_pos[:, i].mean())
    ti = list(token_names).index("text") if "text" in token_names else 1
    out["ktx_masa_tekst_srednia"] = float(per_token[:, ti].mean())
    out["ktx_masa_tekst_sd"] = float(per_token[:, ti].std())
    out["ktx_poz_sd_max"] = float(along_pos.mean(dim=0).max())
    ent = -(att * (att + 1e-12).log2()).sum(-1).mean()
    out["ktx_H"] = float(ent)
    out["ktx_H_max"] = float(math.log2(att.shape[-1]))
    merge = getattr(xattn, "merge", None)
    if merge is not None and hasattr(merge, "weight"):
        w = merge.weight
        dim = int(getattr(xattn, "dim", w.shape[1] // 2))
        wmap = float(w[:, :dim].norm())
        wctx = float(w[:, dim:].norm())
        out["ktx_dom"] = wctx / max(wmap, 1e-9)
        out["ktx_dom_wmap"] = wmap
        out["ktx_dom_wctx"] = wctx
    return out


# Module-level falsifiers and report
def falsifiers(encoder: ConditionEncoder,
               output: Optional[ConditionOutput] = None,
               attention: Optional[Mapping[str, float]] = None,
               text_encoder: Optional[FrozenTextEncoder] = None,
               text_report: Optional[Mapping[str, object]] = None,
               ) -> List[Tuple[str, bool, str]]:
    """Structural and measured checks on the condition channel.

    Anything not measured is reported as NOT PASSED with the reason, never
    omitted: silence is not a passed gate.
    """
    rows: List[Tuple[str, bool, str]] = []
    cfg = encoder.cfg

    # Structural
    n0 = encoder.parameter_count()
    if cfg.cat_head_mode == "prototype":
        vocab_before = encoder.vocab
        proto_before = encoder.category_lm.clone()
        # EVERY buffer the probe grows has to be restored, not just the one
        # that existed when this check was written: with
        # `category_log_prior` left one row longer than the vocabulary, the
        # next forward broadcasts a wider offset against the logits, and
        # calling `falsifiers()` would be what broke the model.
        prior_before = encoder.category_log_prior.clone()
        encoder.add_category(
            "__probe__",
            "a probe category added by the falsifier and removed again",
            embedding=np.zeros((1, int(cfg.cond_d_lm)), dtype=np.float32))
        n1 = encoder.parameter_count()
        encoder.vocab = vocab_before
        encoder.category_lm = proto_before
        encoder.category_log_prior = prior_before
        assert int(encoder.category_log_prior.numel()) == len(encoder.vocab)
        rows.append((
            "adding a category costs zero parameters (D2)", n0 == n1,
            "%d -> %d parameters, %d categories; the head is a prototype head "
            "so a category is a string plus a buffer row"
            % (n0, n1, len(encoder.vocab))))
    else:
        rows.append((
            "adding a category costs zero parameters (D2)", False,
            "cat_head_mode='linear' encodes the category count in the head "
            "weights (%d params); this is the specification's literal "
            "Linear(d_ctx -> 7) and it fails D2 by construction" % n0))

    scale = float(encoder.logit_scale.detach())
    bias = float(encoder.logit_bias.detach())
    rows.append((
        "the prototype head's shared scalars are trained",
        not (scale == _CAT_SCALE_INIT and bias == _CAT_BIAS_INIT),
        "logit_scale=%r, logit_bias=%r against the constructor's %r / %r. "
        "Equality BIT FOR BIT means zero gradient, and the cause is "
        "structural: category_loss() is called from no line of training/ and "
        "cat_logits enters no term of the total loss. Measured on "
        "runs/HYPPE3_ep014.pt, with the difference against ep001 exactly 0.0 "
        "for both. This row passes by itself once the loss is wired."
        % (scale, bias, _CAT_SCALE_INIT, _CAT_BIAS_INIT)))

    prior = encoder.category_log_prior.detach()
    rows.append((
        "the per-class offset has one row per category (D2)",
        int(prior.numel()) == len(encoder.vocab),
        "category_log_prior has %d rows for %d categories, range %.4f, "
        "installed=%r; it is a buffer, so it does not enter sum(p.numel()) "
        "and the row above still passes. Zeros = identity, i.e. a build that "
        "never calls set_category_prior() is the previous model bit for bit"
        % (int(prior.numel()), len(encoder.vocab),
           float(prior.max() - prior.min()) if prior.numel() else 0.0,
           bool(getattr(encoder, "_has_prior", False)))))

    sig_ok = all(k not in ConditionEncoder.forward.__code__.co_varnames
                 for k in ("x", "seq", "sequence", "onehot", "promoter"))
    rows.append((
        "C cannot depend on the promoter (siamese contract, structural)",
        sig_ok,
        "forward() arguments: %s" % (list(
            ConditionEncoder.forward.__code__.co_varnames[
                1:ConditionEncoder.forward.__code__.co_argcount]),)))

    rows.append((
        "no task of this layer is a Nash player (D5)",
        encoder.nash_players() == (),
        "nash_players=%r, subordinate weights=%r; a task with 14.78 %% of the "
        "variance and R^2 0.03-0.10 has the smallest gradient and Nash hands "
        "the largest alpha to exactly that player (measured: alpha 2.31 of 3.0 "
        "at gradient norm 0.0432)"
        % (encoder.nash_players(), encoder.subordinate_terms())))

    rows.append((
        "nash_delta_player is off by default (D5)",
        not cfg.nash_delta_player,
        "nash_delta_player=%r, delta_weight=%.3f from the 85.22/14.78 variance "
        "split" % (cfg.nash_delta_player, cfg.delta_weight)))

    spec = (text_encoder.spec if text_encoder is not None
            else ConditionTextSpec(fields=cfg.cond_text_fields,
                                   level=cfg.cond_text_level))
    rows.append((
        "experiment_title never reaches the language model (D3)",
        all(f not in FORBIDDEN_TEXT_FIELDS for f in spec.fields),
        "fields=%r; experiment_title is unique for 1491/1498 kept samples, "
        "66.9 %% hapax token types, 65.3 %% with a digit" % (spec.fields,)))

    rows.append((
        "unknown tokens are learned, not zero (checkpoint item 5)",
        min(float(encoder.unknown_token_text.detach().norm()),
            float(encoder.unknown_token_species.detach().norm()),
            float(encoder.unknown_token_flank.detach().norm())) > 1e-4,
        "norms text/species/flank = %.4f / %.4f / %.4f"
        % (float(encoder.unknown_token_text.detach().norm()),
           float(encoder.unknown_token_species.detach().norm()),
           float(encoder.unknown_token_flank.detach().norm()))))

    sd = encoder.state_dict()
    have_stats = ("text_center.running_mean" in sd
                  and "text_center.running_var" in sd)
    rows.append((
        "text centring statistics are saved (checkpoint item 3)", have_stats,
        "state_dict carries running_mean/var: %r; without them a single new "
        "condition at inference has nothing to centre against" % (have_stats,)))

    nd = encoder.no_decay_names()
    covered = all(any(p in n.lower() for p in CONDITION_NO_DECAY_SUBSTRINGS)
                  for n in nd)
    default_cov = all(any(p in n.lower() for p in DEFAULT_NO_DECAY_PATTERNS)
                      for n in nd)
    rows.append((
        "every decay-exempt parameter matches CONDITION_NO_DECAY_SUBSTRINGS",
        covered,
        "names=%r; covered by the project default list: %r (if False the "
        "trainer MUST pass CONDITION_NO_DECAY_SUBSTRINGS)"
        % (nd, default_cov)))

    rows.append((
        "the frozen LM revision is pinned (checkpoint item 1)",
        (text_encoder.revision if text_encoder is not None
         else cfg.cond_text_revision) is not None,
        "revision=%r; an unpinned repo moves every embedding without moving a "
        "weight" % (text_encoder.revision if text_encoder is not None
                    else cfg.cond_text_revision,)))

    # Measured on a forward pass
    if output is None:
        rows.append(("context tokens are RMS balanced (D8)", False,
                     "NOT MEASURED: pass a ConditionOutput"))
        rows.append(("text token carries per-sample content", False,
                     "NOT MEASURED: pass a ConditionOutput"))
    else:
        d = output.diagnostics
        shares = [d.get("ktx_udzial_%s" % n, float("nan"))
                  for n in ("gat", "tekst", "flankaL", "flankaR")]
        ok = all(0.20 <= s <= 0.30 for s in shares)
        rows.append((
            "context tokens are RMS balanced (D8)", ok,
            "shares gat/tekst/flankaL/flankaR = %s; per-source per-sample RMS "
            "normalisation should put each at 0.25, and a branch with its own "
            "scale took over the map three times in this project (dominance "
            "1.85 -> 22.64)" % (["%.4f" % s for s in shares],)))
        dev = d.get("ktx_tekst_dev_nad_srednia", float("nan"))
        rows.append((
            "text token carries per-sample content", dev > 0.2,
            "deviation/mean = %.4f; measured precedent 0.0869 (dead) -> 1.0701 "
            "after parameter-free centring" % (dev,)))

    if attention is None:
        rows.append(("F4 cross-attention is positional, not a costly FiLM",
                     False, "NOT MEASURED: pass context_attention_stats(...)"))
        rows.append(("F5 the text token is not crowded out by the flanks",
                     False, "NOT MEASURED: pass context_attention_stats(...)"))
        rows.append(("F6 context does not dominate the map", False,
                     "NOT MEASURED: pass context_attention_stats(...)"))
    else:
        pos_sd = float(attention.get("ktx_poz_sd_max", float("nan")))
        rows.append((
            "F4 cross-attention is positional, not a costly FiLM",
            pos_sd > 0.02,
            "max sd of attention along positions = %.4f (threshold 0.02); with "
            "4 keys positional dependence is the ONLY advantage over FiLM, and "
            "201 600 weights ride on it" % (pos_sd,)))
        m = float(attention.get("ktx_masa_tekst_srednia", float("nan")))
        msd = float(attention.get("ktx_masa_tekst_sd", float("nan")))
        rows.append((
            "F5 the text token is not crowded out by the flanks",
            (m >= 0.05) and (msd >= 0.01),
            "mean mass %.4f (>= 0.05), sd across samples %.4f (>= 0.01); the "
            "flank channel measured ~75x stronger in ablation (-0.1819 vs "
            "-0.0024)" % (m, msd)))
        dom = attention.get("ktx_dom")
        if dom is None:
            rows.append(("F6 context does not dominate the map", False,
                         "NOT MEASURED: no ktx_dom in the report (and "
                         "`cond_dom` is NOT a substitute: the fusion block "
                         "rescales its contribution to the map RMS before "
                         "measuring it, so that number is 1.0 by construction)"))
        else:
            rows.append((
                "F6 context does not dominate the map", float(dom) < 3.0,
                "ktx_dom = %.4f (merge kernel: context half / map half). The "
                "specification's bound 0.5 was set for a gated ADDITIVE branch "
                "starting at gate 0.02 and does not transfer to a "
                "concatenate-and-merge block, whose symmetric initialisation "
                "sits at ~1.0; the failure this guards against is the measured "
                "1.85 -> 15.24 -> 22.64 drift, hence 3.0. Read it on a TRAINED "
                "model" % (float(dom),)))

    # The text source
    tr = dict(text_report or {})
    if text_encoder is not None and not tr:
        tr = text_encoder.report()
    if tr:
        sub = float(tr.get("substitute_fraction", float("nan")))
        rows.append((
            "no embedding came from the hash substitute",
            sub == 0.0,
            "substitute_fraction = %.4f, counts = %r; the hash path runs "
            "offline and carries no semantics" % (sub, tr.get("counts"))))
        if "cos_mean" in tr:
            rows.append((
                "text space is not collapsed (cos_mean < 0.95)",
                float(tr["cos_mean"]) < 0.95,
                "cos_mean = %.4f; the frozen DNABERT flank channel measures "
                "0.958, i.e. nearly collapsed" % (float(tr["cos_mean"]),)))
        if "eff_rank" in tr:
            rows.append((
                "text space has effective rank > 20",
                float(tr["eff_rank"]) > 20.0,
                "eff_rank = %.2f of %s dims; DNABERT flanks measure 13.9/768"
                % (float(tr["eff_rank"]), tr.get("dim_used"))))
    else:
        rows.append(("no embedding came from the hash substitute", False,
                     "NOT MEASURED: pass text_encoder or text_report"))
    return rows


def report(encoder: ConditionEncoder,
           output: Optional[ConditionOutput] = None,
           cat_targets: Optional[torch.Tensor] = None,
           attention: Optional[Mapping[str, float]] = None,
           text_encoder: Optional[FrozenTextEncoder] = None) -> Dict[str, object]:
    """One dictionary with everything the log and the checkpoint need."""
    rep = encoder.report(
        cat_logits=None if output is None else output.cat_logits,
        cat_targets=cat_targets, attention=attention)
    if text_encoder is not None:
        rep["text_source"] = text_encoder.report()
    rep["contract"] = encoder.contract(text_encoder)
    rows = falsifiers(encoder, output=output, attention=attention,
                      text_encoder=text_encoder)
    rep["falsifiers"] = [{"name": n, "passed": bool(p), "detail": d}
                         for n, p, d in rows]
    rep["falsifiers_failed"] = [n for n, p, _ in rows if not p]
    return rep


# Centred FiLM modulation -- key `heads.fusion.film_center`
class CentredLatentConditioner(LatentConditioner):
    """FiLM whose modulation has zero common component. No learnt parameters.

    Measured against the correct centre -- gamma's identity is 1, not 0 --
    about 59 % of the modulation energy is an affine constant on the latent,
    identical for every promoter, and the head's first layer is linear and
    absorbs it. That capacity shifts a LEVEL instead of reweighting features.

    Centring gamma after the tanh would break the invariant `fusion.py`
    exists for -- gamma must never reach zero, because a closing
    multiplicative gate orphans its own source -- since `gamma - mean + 1`
    lies in [-0.8, 2.8] at `film_range` 0.9. The centre is removed in the
    tanh domain instead, scaled to keep the image inside (-1, 1):

        t_c = (tanh(raw_g) - m) / (1 + |m|),    gamma = 1 + r * t_c

    so `mean_B(gamma)` is exactly 1 and `mean_B(beta)` exactly 0, at the cost
    of an amplitude smaller by 1/(1+|m|), about 4 % at the measured |m|.

    `film_center: batch` is BatchNorm semantics; `film_center: ema` uses the
    running mean in both modes, so a row's output never depends on who else
    was in the batch -- which matters because the pairwise head draws its
    pairs inside a batch. With a non-empty `head_withhold` this module is
    called twice per pass and both calls update the buffers; the drift
    between the two input distributions is measured, not assumed small.
    """

    #: The order is a contract of the diagnostics (`fuse_center_tryb`).
    TRYBY: ClassVar[Tuple[str, ...]] = ("batch", "ema")
    #: As in `BatchNorm1d`, deliberately not a config key: the BatchNorm
    #: default is the only reference value anyone has measured here.
    MOMENTUM: ClassVar[float] = 0.1

    def __init__(self, latent_dim: int, ctx_dim: int,
                 token_names: Sequence[str],
                 config=None, tryb: str = "batch") -> None:
        super().__init__(latent_dim=latent_dim, ctx_dim=ctx_dim,
                         token_names=token_names, config=config)
        t = str(tryb)
        if t not in self.TRYBY:
            raise ValueError(
                "heads.fusion.film_center = %r; allowed are %r or 'none' "
                "(which builds a plain LatentConditioner)"
                % (t, list(self.TRYBY)))
        self.tryb = t
        # Buffers, not parameters, so `sum(p.numel())` does not move. They
        # are persistent: a checkpoint without them would load in eval with a
        # zero mean, that is, silently without centring.
        self.register_buffer("film_mean_t", torch.zeros(self.latent_dim))
        self.register_buffer("film_mean_b", torch.zeros(self.latent_dim))
        self.register_buffer("film_steps", torch.zeros((), dtype=torch.long))
        #: Whether THIS pass updates the buffers. The withheld-channel pass
        #: has a different input distribution, so letting it into the running
        #: mean would mix two distributions in the statistic that mode `ema`
        #: reads in eval. A plain attribute: it must not reach the checkpoint.
        self.aktualizuj_statystyki = True

    def _srodki(self, t: torch.Tensor, beta: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Centres used for centring; updates the buffers during training."""
        partia = bool(self.training) and int(t.shape[0]) >= 2
        # The batch centre is always computed -- it is a property of this
        # pass -- but only the unrestricted pass updates the buffers.
        aktualizuj = partia and bool(getattr(self, "aktualizuj_statystyki",
                                             True))
        # Buffers read BEFORE the update: if mode `ema` read them after the
        # current batch had been folded in, a row's output would again depend
        # on who else was in the batch, which is what the mode removes.
        srodek = (self.film_mean_t.detach().clone().to(t.dtype),
                  self.film_mean_b.detach().clone().to(beta.dtype))
        if partia:
            m_t = t.mean(dim=0)
            m_b = beta.mean(dim=0)
            with torch.no_grad():
              if aktualizuj:
                  p = float(self.MOMENTUM)
                  self.film_mean_t.mul_(1.0 - p).add_(
                      m_t.detach().to(self.film_mean_t.dtype), alpha=p)
                  self.film_mean_b.mul_(1.0 - p).add_(
                      m_b.detach().to(self.film_mean_b.dtype), alpha=p)
                  self.film_steps += 1
            if self.tryb == "batch":
                return m_t, m_b
        return srodek

    def modulate(self, latent: torch.Tensor, code: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gb = self.film(code)
        raw_g, beta = gb[:, :self.latent_dim], gb[:, self.latent_dim:]
        t = torch.tanh(raw_g)
        m_t, m_b = self._srodki(t, beta)
        # |t - m| < 1 + |m|, so |t_c| < 1 and gamma never reaches zero.
        t_c = (t - m_t) / (1.0 + m_t.abs())
        gamma = 1.0 + float(self.cfg.film_range) * t_c
        beta_c = beta - m_b
        h = gamma * latent + beta_c
        # Same per-sample RMS damping as the base class, written out rather
        # than called, and kept bit identical: two copies of one formula are
        # exactly how these drift apart.
        h = h / h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
        return h, gamma, beta_c

    def forward(self, latent: torch.Tensor, tokens: torch.Tensor):
        out = super().forward(latent, tokens)
        with torch.no_grad():
            d = out.gamma.detach().double() - 1.0
            wsp = d.mean(dim=0)
            extra = {
                "fuse_center_tryb": float(self.TRYBY.index(self.tryb)),
                "fuse_center_kroki": float(self.film_steps),
                # Without these a run reports nothing about whether the
                # centring works, and a blind alarm is not an alarm.
                "fuse_gamma_wspolna": float(wsp.norm()),
                "fuse_gamma_wlasna": float((d - wsp).norm(dim=1).mean()),
                "fuse_beta_wspolna": float(
                    out.beta.detach().double().mean(dim=0).norm()),
            }
        self._diag.update(extra)
        out.diagnostics.update(extra)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + (", film_center=%s" % self.tryb)


# The condition meets the sequence before pooling -- key
# `heads.fusion.xattn_glowy`
class CoarseConditionAttention(nn.Module):
    """Condition tokens query the coarsest level of the ladder.

    Elsewhere the condition meets the sequence only as FiLM on the POOLED
    latent, where position is already gone, so it can reweight features and
    shift a level but cannot say WHICH WINDOW counts. This module grants
    exactly that one permission, and a probe cascade closes the alternatives:
    quantiser, pooling and head each move a paired probe by at most 0.0009.
    The coarsest level is chosen because the prediction lives there -- L0
    probes at 0.5853 against 0.19-0.29 on L2-L4 and alone is as good as all
    five together -- and because its 25 positions make a fine level 16x
    dearer. Keys come only from the map, so selection is by content, not
    position: a positional anchor already exists in `LatentPooling`, and a
    second one here would measure the sum of two changes. `ktx_xa_poz_sd` is
    the falsifier -- without spread along positions the module is an
    expensive bias, and rank-1 FiLM would do the same more cheaply.

    `output.weight` and `.bias` start at zero, so the forward pass is bit
    identical to the module being absent. A zero init can starve a branch, so
    the arithmetic is written out rather than assumed: with `y = W_o @ a`,
    `dL/dW_o = (dL/dy) (x) a` is non-zero at `W_o = 0` because `a != 0`,
    while `dL/da = W_o^T (dL/dy)` is zero. Step 0 trains only `output` and
    from step 1 q/k/v receive gradient -- a one-step delay, unlike a
    zero-initialised gating scalar, which would orphan `W_o` too.

    Aggregation over tokens is a SUM: a mean would silently scale down every
    existing token when a channel is added, and would break the positive
    control that replacing one token moves the output by that token's share.
    The diffusion contract is untouched, since keys and values come from the
    quantiser output, which `latent_from_codes` rebuilds from indices alone.
    """

    #: Head width. Deliberately not a config key: with 25 keys the number of
    #: heads is the only knob that means anything here.
    DK: ClassVar[int] = 32
    #: Threshold on `ktx_xa_poz_cv` = `poz_sd * L`, which unlike `poz_sd` is
    #: invariant in L: 0 for uniform, sqrt(L-1) = 4.90 for one-point at
    #: L = 25, and 0.5 means the weights differ from uniform by half its
    #: value. The 0.02 of F4 in `context_attention_stats` cannot be reused --
    #: that softmax has four keys and a uniform weight of 0.25, this one 25
    #: keys and 0.04. It is a TRAINING threshold: random q/k, the point a run
    #: starts from, already measure 0.2681. The build falsifier is the gate's
    #: control, where a map with zeroed positional deviation must give
    #: exactly uniform attention.
    PROG_POZ_CV: ClassVar[float] = 0.5
    #: Alarm level for `ktx_xa_udzial` = rms(y) / rms(pooled latent). This
    #: branch has its own scale, bounded only by weight decay on `output`,
    #: deliberately: a zero-initialised multiplicative gate would orphan its
    #: source, and it is the SOURCE that should be warmed up, not the scale.
    #: The price is that taking over the map is possible and must therefore
    #: be observable -- in the sibling track such a branch took over three
    #: times. 1.0 means the attention contributes as much as the whole
    #: pooled latent.
    SHARE_THRESHOLD: ClassVar[float] = 1.0

    def __init__(self, d_ctx: int, d_mapa: int, latent_dim: int,
                 n_glow: int, token_names: Sequence[str] = ()) -> None:
        super().__init__()
        if int(n_glow) < 1:
            raise ValueError(
                "CoarseConditionAttention built with n_glow = %r; a module "
                "that computes not one head is parameters without a "
                "mechanism -- disable it by NOT BUILDING it "
                "(heads.fusion.xattn_glowy = 0)" % (n_glow,))
        self.d_ctx = int(d_ctx)
        self.d_mapa = int(d_mapa)
        self.latent_dim = int(latent_dim)
        self.glowy = int(n_glow)
        self.dk = int(self.DK)
        self.d_wew = self.glowy * self.dk
        self.names: Tuple[str, ...] = tuple(str(n) for n in token_names)

        self.q = nn.Linear(self.d_ctx, self.d_wew)
        self.k = nn.Linear(self.d_mapa, self.d_wew)
        self.v = nn.Linear(self.d_mapa, self.d_wew)
        self.output = nn.Linear(self.d_wew, self.latent_dim)
        # Bit identical at start; see the gradient arithmetic in the class
        # docstring for why this is the only zero init that does not orphan
        # itself.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        #: Whether THIS pass leaves numbers behind. The module is called
        #: twice when `heads.head_withhold` is non-empty, and two input
        #: distributions in one statistic cannot be read. A plain attribute:
        #: it must not reach the checkpoint.
        self.zbieraj = True
        self._diag: Dict[str, float] = {}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def rachunek(self) -> str:
        """Parameter cost derived, so the gate has something to compare to."""
        q = self.d_ctx * self.d_wew + self.d_wew
        k = self.d_mapa * self.d_wew + self.d_wew
        o = self.d_wew * self.latent_dim + self.latent_dim
        return ("q %d + k %d + v %d + output %d = %d"
                % (q, k, k, o, q + 2 * k + o))

    def weights_(self, tokens: torch.Tensor, mapa: torch.Tensor) -> torch.Tensor:
        """The attention map alone, [B, heads, T, L]; used by the gate.

        Factored out rather than reimplemented in the gate, because two
        copies of one formula are exactly how a gate stops testing the model.
        """
        att, _v = self._att(tokens, mapa)
        return att

    def _att(self, tokens: torch.Tensor, mapa: torch.Tensor):
        if tokens.dim() != 3 or int(tokens.shape[-1]) != self.d_ctx:
            raise ValueError("the tokens must be [B, T, %d], got %r"
                             % (self.d_ctx, tuple(tokens.shape)))
        if mapa.dim() != 3 or int(mapa.shape[1]) != self.d_mapa:
            raise ValueError("the map must be [B, %d, L], got %r"
                             % (self.d_mapa, tuple(mapa.shape)))
        dtype = self.q.weight.dtype
        device = self.q.weight.device
        t = tokens.to(device=device, dtype=dtype)
        B, T = int(t.shape[0]), int(t.shape[1])
        m = mapa.to(device=device, dtype=dtype).transpose(1, 2)   # [B, L, C]
        if int(m.shape[0]) != B:
            raise ValueError("the tokens have batch %d, the map %d"
                             % (B, int(m.shape[0])))
        L = int(m.shape[1])
        # Remove the component shared along positions, then one scale per
        # sequence. Without the subtraction the attention comes out exactly
        # uniform: positions of `z[0]` share nearly all of their direction
        # (cosine between keys ~0.95, dropping to 0.0543 once the mean is
        # removed), and that mean is already a `LatentPooling` statistic, so
        # the attention should carry what pooling throws away.
        #
        # The scale is taken over the WHOLE map, not per position: dividing
        # per position would lift the noise of a position near the mean to
        # the norm of an outlier's content. `z` is also the spatial part of a
        # hyperboloid point, so the raw scale is a geometric quantity and
        # must not enter a softmax.
        m = m - m.mean(dim=1, keepdim=True)
        m = m / m.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(_EPS)
        q = self.q(t).view(B, T, self.glowy, self.dk).transpose(1, 2)
        k = self.k(m).view(B, L, self.glowy, self.dk).transpose(1, 2)
        v = self.v(m).view(B, L, self.glowy, self.dk).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-1, -2))
                            / math.sqrt(float(self.dk)), dim=-1)
        return att, v

    def forward(self, tokens: torch.Tensor, mapa: torch.Tensor,
                reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, T, d_ctx] x [B, C, L] -> [B, latent_dim], added to the latent."""
        att, v = self._att(tokens, mapa)              # [B,h,T,L] [B,h,L,dk]
        B, T = int(att.shape[0]), int(att.shape[2])
        ctx = (att @ v).transpose(1, 2).reshape(B, T, self.d_wew)
        per_token = self.output(ctx)                 # [B, T, latent]
        y = per_token.sum(dim=1)                      # [B, latent]
        if self.zbieraj:
            self._diag = self._numbers(att, per_token, y, reference)
            self._diag["ktx_xa_klucz_cos"] = self._cos_kluczy(mapa)
        return y

    @torch.no_grad()
    def _cos_kluczy(self, mapa: torch.Tensor) -> float:
        """Mean cosine between KEYS of different positions of one sequence.

        Collinear keys cannot be told apart by any softmax, so a value near
        1.0 means centring along positions has stopped being enough and
        there is no selection whatever the head does.
        """
        m = mapa.to(dtype=self.k.weight.dtype).transpose(1, 2)
        m = m - m.mean(dim=1, keepdim=True)
        m = m / m.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(_EPS)
        k = self.k(m)                                   # [B, L, d_wew]
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        g = k @ k.transpose(1, 2)
        L = int(g.shape[-1])
        if L < 2:
            return float("nan")
        iu = torch.triu_indices(L, L, offset=1, device=g.device)
        return float(g[:, iu[0], iu[1]].mean())

    @torch.no_grad()
    def _numbers(self, att: torch.Tensor, per_token: torch.Tensor,
                y: torch.Tensor, reference: Optional[torch.Tensor]
                ) -> Dict[str, float]:
        """Eight numbers, every one of which can fail.

        `output` starts at zero, so a starved branch looks exactly like a
        disabled module in every headline metric; `ktx_xa_wyj_norma` is the
        only number that separates the two states.
        """
        a = att.double()
        L = int(a.shape[-1])
        out: Dict[str, float] = {}
        # Spread of the weights along POSITIONS. A mixture constant along L
        # is FiLM by a more expensive route.
        out["ktx_xa_poz_sd"] = float(a.std(dim=-1).mean())
        # Dimensionless; the threshold stands on this, not on `poz_sd`. See
        # `PROG_POZ_CV`: 0.0 uniform, sqrt(L-1) one-point.
        out["ktx_xa_poz_cv"] = float(out["ktx_xa_poz_sd"] * L)
        out["ktx_xa_poz_max"] = float(a.max(dim=-1).values.mean())
        ent = -(a * (a + 1e-12).log2()).sum(-1)
        out["ktx_xa_H"] = float(ent.mean())
        out["ktx_xa_H_max"] = float(math.log2(L))
        # Do different tokens ask about different windows? Zero means the T
        # queries are one query and there is no per-channel selection.
        if int(a.shape[2]) > 1:
            d = (a.unsqueeze(2) - a.unsqueeze(3)).abs().sum(-1) * 0.5
            n = int(a.shape[2])
            out["ktx_xa_rozn_tokenow"] = float(
                d.sum() / (n * (n - 1) * d.shape[0] * d.shape[1]))
        # Does the map differ BETWEEN ROWS? Zero means the attention is the
        # same for every sample, i.e. the branch is a constant bias.
        if int(a.shape[0]) > 1:
            out["ktx_xa_rozn_probek"] = float(
                (a - a.mean(dim=0, keepdim=True)).abs().sum(-1).mean() * 0.5)
        out["ktx_xa_rms"] = float(y.pow(2).mean().sqrt())
        # Has the branch left zero at all? Without this a starved branch
        # looks exactly like a disabled module.
        out["ktx_xa_wyj_norma"] = float(self.output.weight.norm())
        if reference is not None and reference.numel():
            r = float(reference.pow(2).mean().sqrt())
            out["ktx_xa_udzial"] = float(out["ktx_xa_rms"] / max(r, _EPS))
        for i, nm in enumerate(self.names[: int(per_token.shape[1])]):
            out["ktx_xa_wklad_%s" % nm] = float(
                per_token[:, i].pow(2).mean().sqrt())
        return out

    def no_decay_names(self) -> List[str]:
        """Nothing here is exempt from decay, and that is a decision.

        Exemptions are for tokens and gates, whose zero is an ABSORBING
        point. Zero on `output` is a STARTING point that one gradient step
        leaves, so an exemption would be one taken just in case.
        """
        return []

    def extra_repr(self) -> str:
        return ("d_ctx=%d, d_mapa=%d, latent=%d, glowy=%d, dk=%d, tokeny=%r"
                % (self.d_ctx, self.d_mapa, self.latent_dim, self.glowy,
                   self.dk, self.names))


class LocalConditionAttention(nn.Module):
    """Map POSITIONS query the metadata, the reverse of the coarse module.

    Q comes from each position of `z[0]`, K and V from the condition tokens,
    so a position without content gives near-uniform weights and a position
    carrying a motif picks the token relevant to it. It runs AFTER the
    quantiser, so the codes stay a function of the sequence alone.
    """

    def __init__(self, d_mapa: int, d_token: int, d_wyj: int,
                 glowy: int = 4, dk: int = 32) -> None:
        super().__init__()
        self.glowy = int(glowy)
        self.dk = int(dk)
        wew = self.glowy * self.dk
        self.q = nn.Linear(int(d_mapa), wew, bias=False)
        self.k = nn.Linear(int(d_token), wew, bias=False)
        self.v = nn.Linear(int(d_token), wew, bias=False)
        self.output = nn.Linear(wew, int(d_wyj), bias=False)
        # Zero init: bit identical to the previous model at start. It also
        # hides a broken wiring, so the gate measures after steps, not on a
        # fresh model.
        nn.init.zeros_(self.output.weight)
        self._diag = {}
        self.zbieraj = True

    def forward(self, mapa: torch.Tensor, tokeny: torch.Tensor
                ) -> torch.Tensor:
        # mapa [B, C, L] -> [B, L, C];  tokeny [B, T, d_token]
        m = mapa.transpose(1, 2)
        B, L, _ = m.shape
        T = tokeny.shape[1]
        q = self.q(m).view(B, L, self.glowy, self.dk).transpose(1, 2)
        k = self.k(tokeny).view(B, T, self.glowy, self.dk).transpose(1, 2)
        v = self.v(tokeny).view(B, T, self.glowy, self.dk).transpose(1, 2)
        w = torch.softmax(q @ k.transpose(-1, -2) / (self.dk ** 0.5), dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, L, -1)
        # Without this non-linearity `output` (a bias-free Linear) commutes
        # with the sum over positions, so only the column masses of the
        # attention matrix reached the output:
        # `sum_p W(y_p) = W(sum_t (sum_p w[p,t]) v_t)`. Of 4x25x5 = 500
        # numbers, 20 aggregates got through and per-position selectivity had
        # no route at all. Measured before the fix:
        # `max|forward - folded_to_masses| = 5.722e-06` at output scale
        # 26.13. `silu` does not commute with the sum.
        y = torch.nn.functional.silu(y)
        popr = self.output(y)                       # [B, L, d_wyj]
        if self.zbieraj:
            with torch.no_grad():
                n = popr.norm(dim=-1)                # [B, L]
                sr = n.mean(dim=1).clamp_min(1e-12)
                # Coefficient of variation over positions: the measure of
                # LOCALITY. A flat profile means every position gets the
                # same, i.e. the mechanism is global despite the
                # per-position arithmetic.
                self._diag['ktx_lok_poz_cv'] = float(
                    (n.std(dim=1) / sr).mean())
                self._diag['ktx_lok_norma'] = float(n.mean())
                # Entropy of the weights over TOKENS: low means a position
                # picks particular metadata, high means it asks all equally.
                p = w.mean(dim=(1, 2)).clamp_min(1e-9)
                self._diag['ktx_lok_H'] = float(
                    -(p * p.log()).sum(dim=-1).mean())
                self._diag['ktx_lok_Hmax'] = float(
                    torch.log(torch.tensor(float(T))))
        # Mean, not sum. Summing over L map positions measured
        # ||contribution||/||pooled|| = 1.6640, i.e. the branch was already
        # 1.66x louder than what it is added to. The mean removes the factor
        # L and makes the scale independent of map length, which matters
        # because the range stratification changes that length.
        return popr.mean(dim=1)                      # [B, d_wyj]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
