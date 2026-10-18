"""Negative pools from outside the Promotor group: genes and flanks.

`BatchMaker`'s existing negatives, `dinucleotide_shuffle` and `chimera`, are
derived from the positive row. `coding` and `flank` are real DNA of the same
genome that is not a promoter, so they need a second data source rather than a
transformation. The raw material was already there -- `sequences.tsv.gz` holds
Gene, Promotor, FlankLeft and FlankRight, ~100 thousand rows each, and
`iter_sequence_tsv` takes a `groups` argument that until now was always
`("Promotor",)` -- while `NEGATIVE_KINDS` declared both classes and nothing in
`batches.py` ever emitted them.

FlankRight contains promoter sequence in 36.3 % of records and is a promoter
in 5.1 %, so an unfiltered flank negative injects wrong labels at full weight;
`contamination_mask` rejects every pool row sharing a 31-mer with any promoter
of the corpus. The stride is on the promoter side only: striding both sides is
unsatisfiable whenever the overlapping block starts at an offset `d` with
`d % N != 0`, which would let contamination through as a function of the
offset. Scanning the pool at every position, with a contiguous overlap longer
than `k + stride`, guarantees the hit.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.sequences import CasePolicy, iter_sequence_tsv

__all__ = ["AuxPool", "kmer_hashes", "contamination_mask"]

_BASES = "ACGT"
_CODE = {c: i for i, c in enumerate(_BASES)}


def kmer_hashes(codes: np.ndarray, k: int = 31,
                stride: int = 1) -> np.ndarray:
    """Codes 0-3 -> packed k-mers as uint64, every `stride` positions.

    The packing is a bijection for k <= 31 (2 bits per base, 62 bits), so
    these are not hashes and equal values mean equal k-mers; the name stays
    because that is how they are used.
    """
    a = np.asarray(codes, dtype=np.uint64).ravel()
    n = int(a.size)
    if n < k:
        return np.zeros(0, dtype=np.uint64)
    out = np.zeros(n - k + 1, dtype=np.uint64)
    for i in range(k):
        out = (out << np.uint64(2)) | a[i:n - k + 1 + i]
    return out[::stride] if stride > 1 else out


def contamination_mask(pool: np.ndarray, pool_len: np.ndarray,
                       prom: np.ndarray, prom_len: np.ndarray,
                       k: int = 31, prom_stride: int = 8,
                       chunk: int = 4096) -> np.ndarray:
    """True for pool rows sharing a k-mer with any promoter: True = reject."""
    ref: List[np.ndarray] = []
    jedno = (int(np.min(prom_len)) == int(np.max(prom_len))
             if prom_len.size else False)
    for i in range(0, int(prom.shape[0]), chunk):
        blok = prom[i:i + chunk]
        dl = prom_len[i:i + chunk]
        if jedno:
            # Promoters are uniformly 800 long, so a block takes 31 numpy
            # operations instead of 4096 Python calls. One 4096 x 770 uint64
            # block is 25 MB; the full set (925 MB) never exists at once.
            a = blok[:, :int(dl[0])].astype(np.uint64)
            n = int(a.shape[1])
            if n < k:
                continue
            acc = np.zeros((a.shape[0], n - k + 1), dtype=np.uint64)
            for t in range(k):
                acc = (acc << np.uint64(2)) | a[:, t:n - k + 1 + t]
            ref.append(np.unique(acc[:, ::prom_stride].ravel()))
        else:
            for j in range(blok.shape[0]):
                ref.append(kmer_hashes(blok[j, :int(dl[j])], k=k,
                                       stride=prom_stride))
    if not ref:
        return np.zeros(int(pool.shape[0]), dtype=bool)
    tab = np.unique(np.concatenate(ref))
    zle = np.zeros(int(pool.shape[0]), dtype=bool)
    for i in range(int(pool.shape[0])):
        h = kmer_hashes(pool[i, :int(pool_len[i])], k=k, stride=1)
        if h.size == 0:
            continue
        pos = np.searchsorted(tab, h)
        pos[pos >= tab.size] = 0
        zle[i] = bool(np.any(tab[pos] == h))
    return zle


class AuxPool(object):
    """Sequences of one or several GROUP groups, grouped by genome.

    `by_genome` maps a genome name to row indices in `codes`. A row shorter
    than `length` is rejected; a longer one is cropped randomly at draw time,
    not at load time, so one load yields many different negatives.
    """

    def __init__(self, codes: np.ndarray, length: np.ndarray,
                 genome: List[str], kind: str,
                 report: Optional[Dict[str, object]] = None) -> None:
        self.codes = codes
        self.length = length
        self.genome = list(genome)
        self.kind = str(kind)
        self.report = dict(report or {})
        self.by_genome: Dict[str, List[int]] = {}
        for i, g in enumerate(self.genome):
            self.by_genome.setdefault(g, []).append(i)

    def __len__(self) -> int:
        return int(self.codes.shape[0])

    # Load
    @classmethod
    def load(cls, path: str, groups: Sequence[str], kind: str,
             length: int = 800, genomes: Optional[Sequence[str]] = None,
             max_per_genome: int = 1200, seed: int = 7898,
             min_length: Optional[int] = None) -> "AuxPool":
        want = set(genomes) if genomes else None
        lut = np.full(256, 255, dtype=np.uint8)
        for c, i in _CODE.items():
            lut[ord(c)] = i
        keep_min = int(min_length if min_length is not None else length)
        per: Dict[str, int] = {}
        rows: List[np.ndarray] = []
        lens: List[int] = []
        gen: List[str] = []
        seen = 0
        drop_short = 0
        drop_amb = 0
        drop_genome = 0
        drop_quota = 0
        for rec in iter_sequence_tsv(path, groups=tuple(groups),
                                     policy=CasePolicy.UPPERCASE):
            seen += 1
            g = str(rec.meta["genome"])
            if want is not None and g not in want:
                drop_genome += 1
                continue
            if per.get(g, 0) >= max_per_genome:
                drop_quota += 1
                continue
            s = rec.seq
            if len(s) < keep_min:
                drop_short += 1
                continue
            arr = lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
            if int((arr == 255).sum()):
                # As for promoters: a row with an N or an IUPAC code is
                # rejected, not mapped. `one_hot` would write 0.25 into four
                # channels, whose argmax is a base not in the sequence.
                drop_amb += 1
                continue
            rows.append(arr.astype(np.uint8))
            lens.append(int(arr.size))
            gen.append(g)
            per[g] = per.get(g, 0) + 1
        if not rows:
            raise RuntimeError(
                "AuxPool(%s): zero rows from groups %r in %s (%d seen)"
                % (kind, tuple(groups), path, seen))
        szer = max(lens)
        buf = np.zeros((len(rows), szer), dtype=np.uint8)
        for i, a in enumerate(rows):
            buf[i, :a.size] = a
        rep = {"seen": seen, "kept": len(rows), "groups": tuple(groups),
               "drop_short": drop_short, "drop_ambiguous": drop_amb,
               "drop_genome": drop_genome, "drop_quota": drop_quota,
               "width": int(szer), "genomes": len(set(gen))}
        return cls(buf, np.asarray(lens, dtype=np.int64), gen, kind, rep)

    # Filtering
    def drop_contaminated(self, prom: np.ndarray, prom_len: np.ndarray,
                          k: int = 31, prom_stride: int = 8) -> int:
        """Drops rows sharing a k-mer with a promoter. Returns how many."""
        zle = contamination_mask(self.codes, self.length, prom, prom_len,
                                 k=k, prom_stride=prom_stride)
        n0 = int(self.codes.shape[0])
        keep = ~zle
        self.codes = self.codes[keep]
        self.length = self.length[keep]
        self.genome = [g for g, ok in zip(self.genome, keep) if ok]
        self.by_genome = {}
        for i, g in enumerate(self.genome):
            self.by_genome.setdefault(g, []).append(i)
        usun = n0 - int(self.codes.shape[0])
        self.report["contaminated_dropped"] = usun
        self.report["contaminated_frac"] = (usun / float(n0)) if n0 else 0.0
        self.report["kept_after_filter"] = int(self.codes.shape[0])
        return usun

    # Draw
    def draw(self, genome: str, length: int,
             rng: np.random.Generator) -> Optional[np.ndarray]:
        """A random `length` window of a random row of this genome, or None."""
        idx = self.by_genome.get(genome)
        if not idx:
            return None
        i = int(idx[int(rng.integers(len(idx)))])
        n = int(self.length[i])
        if n < length:
            return None
        off = int(rng.integers(0, n - length + 1)) if n > length else 0
        return self.codes[i, off:off + length].astype(np.uint8)

    def format(self) -> str:
        r = self.report
        s = ("  pool %-8s groups %s: %d seen, %d kept from %d genomes "
             "(short %d, ambiguous %d, outside the registry %d, quota %d)"
             % (self.kind, ",".join(r.get("groups", ())), r.get("seen", 0),
                r.get("kept", 0), r.get("genomes", 0), r.get("drop_short", 0),
                r.get("drop_ambiguous", 0), r.get("drop_genome", 0),
                r.get("drop_quota", 0)))
        if "contaminated_dropped" in r:
            s += ("\n    31-mer contamination filter: dropped %d (%.4f), "
                  "%d left"
                  % (r["contaminated_dropped"], r["contaminated_frac"],
                     r["kept_after_filter"]))
        return s
