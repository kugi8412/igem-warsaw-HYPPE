"""A promoter-identity k-mer index that fits in memory at corpus scale.

`negatives.PromoterKmerIndex` keeps its codes in a Python `set` of ints, which
is fine for a few hundred sequences and does not survive the real corpus: one
genome of ~12 000 promoters at 800 bp is 9.2 M 31-mers, and a CPython `set` of
boxed ints costs ~60 B per element -- 550 MB for one genome, 11 GB for twenty.
The filter would then run on a subsample, and a filter that runs on a
subsample is not a filter. The same codes as a sorted `uint64` array cost 8 B
each, 74 MB for the largest genome, and lookup is `np.searchsorted`.

This is a deliberate drop-in for the other index: `negatives.build_pool` and
`negatives.clean_windows` duck-type it on `hit_count`, `best_window` and
`verify`, implemented here with identical semantics and identical failure
behaviour, so no negative-sampling code had to change to gain the memory.
:meth:`GenomeKmerIndex.verify` keeps its assertion in particular: the symptom
of a broken case mapping was 66 k-mers found instead of 770, which scales
every contamination percentage down threefold while still looking like a
working filter.

The index is built from every promoter of the genome, from every part of the
split, and that is not a leak: rejecting a candidate negative removes it and
never adds information to a training example. An index built from the training
part alone would pass flank windows that overlap a validation promoter
verbatim, which is the contamination the filter exists to remove -- FlankRight
contains promoter sequence in 36.3 % of windows and is a promoter in 5.1 %.

A window containing anything outside ACGT produces no k-mer, as in
`negatives._encode_kmers`. Mapping N to A would invent 31-mers that are not in
the genome and make the index report matches that do not exist.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Union

import numpy as np

#: Code of any base outside ACGT. Must match `dataset.encode_codes`.
N_CODE = 4
_ACGT = "ACGT"
_TO_CODE = np.full(256, N_CODE, dtype=np.uint8)
for _i, _c in enumerate(_ACGT):
    _TO_CODE[ord(_c)] = _i
    _TO_CODE[ord(_c.lower())] = _i


def encode_codes(seq: str) -> np.ndarray:
    """ASCII text -> uint8 codes, A=0 C=1 G=2 T=3, everything else 4.

    Lowercase maps to the same code as uppercase: here it is soft-masking
    (7.2600 % of promoter bases), not an alphabet of its own.
    """
    a = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    return _TO_CODE[a]


def kmer_matrix(codes: np.ndarray, k: int = 31
                ) -> "tuple[np.ndarray, np.ndarray]":
    """Rolling 2-bit k-mer codes of every row. Returns (codes, valid).

    `codes` is uint8 [n, L] (or [L]); the result is uint64 [n, L-k+1] with a
    boolean mask that is False wherever the window touched a non-ACGT base.

    The loop runs over positions rather than sequences: at L = 800 and
    n = 12 000 that is 800 vector operations instead of 12 000.
    """
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)
    n, L = codes.shape
    k = int(k)
    if k <= 0 or k > 32:
        raise ValueError("k must be in 1..32 to fit a uint64, got %d" % k)
    m = L - k + 1
    if m <= 0:
        return (np.zeros((n, 0), dtype=np.uint64),
                np.zeros((n, 0), dtype=bool))
    mask = np.uint64((1 << (2 * k)) - 1) if k < 32 else np.uint64(~0)
    cur = np.zeros(n, dtype=np.uint64)
    run = np.zeros(n, dtype=np.int32)
    out = np.empty((n, m), dtype=np.uint64)
    ok = np.empty((n, m), dtype=bool)
    two = np.uint64(2)
    for i in range(L):
        c = codes[:, i]
        bad = c >= N_CODE
        cur = ((cur << two) | c.astype(np.uint64)) & mask
        run = np.where(bad, 0, run + 1).astype(np.int32)
        if bad.any():
            cur = np.where(bad, np.uint64(0), cur)
        if i >= k - 1:
            j = i - k + 1
            out[:, j] = cur
            ok[:, j] = run >= k
    return out, ok


def kmer_vector(codes: np.ndarray, k: int = 31
                ) -> "tuple[np.ndarray, np.ndarray]":
    """Same as `kmer_matrix` for ONE sequence, but in one vector operation.

    `kmer_matrix` loops over positions, which is the right shape for thousands
    of sequences; for a single 800 bp candidate it is 800 numpy calls on a
    length-1 array, and that candidate is the inner loop of `clean_windows`.

    No overflow: a valid code is <= 3, so the largest k-mer value is
    4^31 - 1 < 2^62; an invalid code (4) can push a term to 2^62 and the sum
    stays inside uint64, and those positions are masked out anyway.
    """
    c = np.asarray(codes).reshape(-1)
    k = int(k)
    m = c.shape[0] - k + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=bool)
    w = np.lib.stride_tricks.sliding_window_view(c, k)
    ok = (w < N_CODE).all(axis=1)
    powers = (np.uint64(4) ** np.arange(k - 1, -1, -1, dtype=np.uint64))
    km = (w.astype(np.uint64) * powers).sum(axis=1)
    return km, ok


class GenomeKmerIndex:
    """Sorted uint64 set of the 31-mers of one genome's promoters.

    Interface-compatible with `negatives.PromoterKmerIndex`.
    """

    def __init__(self, k: int = 31) -> None:
        self.k = int(k)
        self.n_promoters = 0
        self._codes = np.zeros(0, dtype=np.uint64)
        self._parts: list = []

    # Building
    def add_codes(self, codes: np.ndarray) -> "GenomeKmerIndex":
        """Add a [n, L] uint8 code matrix. Cheap; `finish()` does the sort."""
        km, ok = kmer_matrix(codes, self.k)
        if km.size:
            self._parts.append(np.unique(km[ok]))
        self.n_promoters += int(codes.shape[0] if codes.ndim == 2 else 1)
        return self

    def build(self, promoter_seqs: Iterable[str]) -> "GenomeKmerIndex":
        """Same signature as `PromoterKmerIndex.build`, for strings."""
        buf: list = []
        for s in promoter_seqs:
            buf.append(encode_codes(s))
            if len(buf) >= 512:
                self._add_ragged(buf)
                buf = []
        if buf:
            self._add_ragged(buf)
        return self.finish()

    def _add_ragged(self, seqs: Sequence[np.ndarray]) -> None:
        L = max(int(s.shape[0]) for s in seqs)
        mat = np.full((len(seqs), L), N_CODE, dtype=np.uint8)
        for i, s in enumerate(seqs):
            mat[i, :s.shape[0]] = s
        self.add_codes(mat)

    def finish(self) -> "GenomeKmerIndex":
        if self._parts:
            self._codes = np.unique(np.concatenate(self._parts))
            self._parts = []
        return self

    @property
    def n_kmers(self) -> int:
        return int(self._codes.shape[0])

    def nbytes(self) -> int:
        return int(self._codes.nbytes)

    # Querying
    def _query(self, seq: Union[str, np.ndarray]) -> np.ndarray:
        codes = encode_codes(seq) if isinstance(seq, str) else seq
        km, ok = kmer_vector(codes, self.k)
        return km[ok]

    def hit_mask(self, seq: Union[str, np.ndarray]) -> np.ndarray:
        """Per-position boolean: is the k-mer starting here in the index?

        The window scan of `clean_windows` asks this about overlapping windows
        of one candidate, so the answer is computed once and the per-window
        counts become a cumulative sum.
        """
        if self._parts:
            self.finish()
        codes = encode_codes(seq) if isinstance(seq, str) else seq
        km, ok = kmer_vector(codes, self.k)
        out = np.zeros(km.shape[0], dtype=bool)
        if km.size == 0 or self._codes.size == 0:
            return out
        pos = np.minimum(np.searchsorted(self._codes, km),
                         self._codes.shape[0] - 1)
        out[ok] = (self._codes[pos] == km)[ok]
        return out

    def clean_window_starts(self, seq: Union[str, np.ndarray],
                            window: int = 800, stride: int = 20,
                            max_hits: int = 0) -> np.ndarray:
        """Start offsets of every `window` whose promoter-identity hit count
        is <= `max_hits`. One pass, not one pass per window."""
        codes = encode_codes(seq) if isinstance(seq, str) else seq
        L = int(np.asarray(codes).reshape(-1).shape[0])
        if L < window:
            return np.zeros(0, dtype=np.int64)
        hits = self.hit_mask(codes).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(hits)])
        per = window - self.k + 1
        starts = np.arange(0, L - window + 1, max(stride, 1))
        counts = cs[starts + per] - cs[starts]
        return starts[counts <= max_hits].astype(np.int64)

    def hit_count(self, seq: Union[str, np.ndarray]) -> int:
        """How many of the sequence's k-mers occur in the index."""
        if self._parts:
            self.finish()
        q = self._query(seq)
        if q.size == 0 or self._codes.size == 0:
            return 0
        pos = np.searchsorted(self._codes, q)
        pos = np.minimum(pos, self._codes.shape[0] - 1)
        return int((self._codes[pos] == q).sum())

    def best_window(self, seq: Union[str, np.ndarray], window: int = 800,
                    stride: int = 20) -> int:
        """Largest hit count over any `window`-long stretch."""
        codes = encode_codes(seq) if isinstance(seq, str) else seq
        L = int(codes.shape[0])
        if L <= window:
            return self.hit_count(codes)
        best = 0
        for i in range(0, L - window + 1, max(stride, 1)):
            best = max(best, self.hit_count(codes[i:i + window]))
        return best

    def verify(self, inside: Union[str, np.ndarray],
               outside: Union[str, np.ndarray]) -> Dict[str, object]:
        """A promoter FROM the index must hit fully; one outside must not.

        An assertion rather than a log line, the same contract as
        `negatives.PromoterKmerIndex.verify`.
        """
        n_in = self.hit_count(inside)
        n_out = self.hit_count(outside)
        len_in = len(inside) if isinstance(inside, str) else int(
            inside.shape[0])
        len_out = len(outside) if isinstance(outside, str) else int(
            outside.shape[0])
        expect = max(len_in - self.k + 1, 0)
        if n_in != expect or n_out != 0:
            raise AssertionError(
                "k-mer index verification failed: a promoter from the index "
                "hit %d of %d expected, one outside hit %d of %d (both must "
                "be %d and 0). A case-mapping bug here silently scales every "
                "contamination figure by ~3x."
                % (n_in, expect, n_out, max(len_out - self.k + 1, 0), expect))
        return {"inside_hits": n_in, "expected": expect,
                "outside_hits": n_out, "n_promoters": self.n_promoters,
                "n_kmers": self.n_kmers}

    def report(self) -> Dict[str, object]:
        return {"k": self.k, "n_promoters": self.n_promoters,
                "n_kmers": self.n_kmers,
                "megabytes": round(self.nbytes() / 1e6, 2)}
