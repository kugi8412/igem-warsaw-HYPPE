"""The canonical HYPPE split into four parts: train / val / spy / test.

One source of truth: every downstream module reads the split through
`load_split()` and never draws its own.

Not to be confused with `hyppe/data/splits.py`, which reads and audits
`data/splits/SPLIT_FINAL.tsv`, a split whose unit is a leakage block of
orthogroups joined by shared canonical 25-mers. The two splits of this corpus
do not agree, so a number measured under one is not comparable with a number
measured under the other and every reported metric must name its split.

The four roles are disjoint. `train` fits weights; `val` selects the epoch
within one run; `spy` chooses BETWEEN models, opened once per comparison and
never inside the training loop; `test` is opened once, at the end. The sd of
the evaluation alone is 0.0064 over eight passes on the same weights, and
selection under a metric that noisy picked a result +1.62 sd above the mean,
so choosing an epoch and choosing a model on one set inflates twice.

The atom is a homology cluster, not a row and not an orthogroup, because the
end task is zero-shot. Orthologues are not independent examples: within an
orthogroup the median Jaccard is 0.0633 and 15.9 % of pairs are identical,
against a cross-orthogroup median of 0.0000 and p99.99 of 0.00581. Splitting
by OG_ID closes orthologues but neither paralogues nor gene families, so the
orthogroup is only a seed and a cluster arises by merging seeds above the
threshold. The threshold stands on a percentile and not on a maximum --
0.05 is 8.6x the cross-orthogroup p99.99 -- because a cross-orthogroup pair
is sometimes genuinely homologous, which is why merging exists at all.

Similarity is Jaccard over sets of forward 11-mers; the corpus is
consistently oriented (mirror test -0.03 against +0.99) so strands are not
canonicalised. It approximates identity and is not an alignment: at identity
p the Jaccard falls like p^k/(2-p^k), so a detected similarity is real while
anything below ~75 % identity at k=11 is indistinguishable from noise.

Samples are the other axis, and design C of `baselines.py` -- a new promoter
in a new context -- needs both held at once. Genome and study are confounded,
I(genome; study)/H(genome) = 0.9446, so the column split is stratified per
genome. The column atom is a connected component of the graph joining samples
by `study` or by (condition, genome), because `study` alone is not enough.

The manifest and the two TSVs use English keys; `migrate_legacy_split()`
converts the legacy Polish-schema artefacts through an explicit rename table.
`load_split()` reads only the new schema and raises on the old one, naming
the migrator, because accepting both would be the silent fallback this module
otherwise spends a hundred lines preventing.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths

# Build parameters; all of them land in the JSON manifest.
SEED = 20260911

K = 11                      # k-mer length
SCALE = 16                  # FracMinHash: keep ~1/16 of the k-mers
CAP_DF = 1000               # a k-mer present in >CAP_DF distinct sequences
                            # generates no candidates (repeats, low complexity)
MIN_SHARED = 2              # candidate = >=2 shared sampled k-mers
MERGE_THRESHOLD = 0.05      # exact Jaccard at which orthogroups are merged

# CAP_DF = 100 was a real leak, not caution: 2.03 % of sampled k-mers sat
# above it and carried 9.06 % of the postings, and 17 evaluation rows with a
# `train` partner at J >= 0.05 passed through that blind spot, one of them
# from the target strain P1.  Separated by execution: cap=1000 catches all 9
# affected `test` rows and so does removing the cap, while MIN_SHARED=1 at
# cap=100 catches none -- the culprit is the cap, not the sketch gate.
# cap=1000 collects 87 % of what no cap collects, for 1.41x the work.

# The independent falsifier -- these are not build parameters.  It may not
# share the build's sketch: a pair the candidate stage never generates would
# then be invisible to the merging and to the control at once, making
# "crossing the boundary = 0" true by construction.  So the probe draws its
# own k-mers -- denser (1/8), no df cap, a different hash salt.
#
# Both blind spots computed exactly.  A pair at J = 0.05 with |A| = |B| = 786
# shares I = 75 k-mers, and the shared SAMPLED count is Binom(I, 1/SCALE):
#   build  SCALE 16, MIN_SHARED 2: P(c <= 1) = 4.90e-02
#   probe  SCALE  8, MIN_SHARED 2: P(c <= 1) = 4.14e-04   (118x tighter)
# The probe is thus an order of magnitude wider than a literal "cap_df=None
# and min_shared=1", and cheaper, because a denser sketch cuts the number of
# random candidates.
PROBE_SCALE = 8
PROBE_MIN_SHARED = 2
PROBE_CAP_DF = None
PROBE_SALT = 0x5BD1E995      # hash salt -- a different draw from the build
PROBE_MAX_ROUNDS = 3         # at most this many leakage-closing rounds

SPLITS = ("train", "val", "spy", "test")
PROPORTIONS = {"train": 0.70, "val": 0.10, "spy": 0.10, "test": 0.10}

# Downstream CLIs accept a few spellings; the translation lives here and only
# here, so that no module grows its own `if name == "training"`.
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "validation": "val",
    "spy": "spy",
    "test": "test",
}

# What each part is for.  Not decorative: `spy` and `test` differ only by
# this sentence, being numerically the same size (20 385 against 20 372
# sequences).
SPLIT_ROLES = {
    "train": "fitting weights",
    "val": "choosing a hyperparameter within one frame (e.g. the ridge lambda)",
    "spy": "choosing BETWEEN models/representations; opened once per comparison",
    "test": "opened ONCE, at the end, for the report",
}


def canonical_split(x: str) -> str:
    """Alias -> canonical name.  Raises on an unknown name, never guesses."""
    try:
        return SPLIT_ALIASES[x]
    except KeyError:
        raise ValueError(
            "unknown split %r -- allowed %s (aliases: %s)"
            % (x, list(SPLITS), sorted(SPLIT_ALIASES))) from None


GROUP = "Promotor"          # which sequence block the split is about
TARGET_STRAIN = "P1"

SPLIT_DIR = os.path.join(paths.SPLITS, "homology_clusters")
ROWS_TSV = os.path.join(SPLIT_DIR, "rows.tsv")
COLUMNS_TSV = os.path.join(SPLIT_DIR, "columns.tsv")
MANIFEST_JSON = os.path.join(SPLIT_DIR, "manifest.json")
CACHE_DIR = os.path.join(paths.CACHE, "splits")

#: Directory that must be on PYTHONPATH for the worker subprocesses.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _subprocess_env() -> dict:
    """Environment for a worker subprocess: `hyppe` importable from src/.

    The path is passed explicitly rather than relying on the current
    directory, because the package lives under `src/` and not at the
    repository root.
    """
    env = dict(os.environ)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_DIR + (os.pathsep + old if old else "")
    return env


# k-mers
_MAP = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _MAP[ord(_c)] = _i
    _MAP[ord(_c.lower())] = _i


def kmers(seq: str, k: int = K) -> np.ndarray:
    """Sorted set of forward k-mers, 2-bit encoded.

    A letter outside ACGT breaks the k-mer; 'N' is not substituted by 'A',
    because that would invent a k-mer the sequence does not contain.
    """
    a = _MAP[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = a.shape[0]
    if n < k:
        return np.empty(0, dtype=np.uint64)
    ok = a != 255
    b = a.astype(np.uint64)
    b[~ok] = 0
    code = np.zeros(n - k + 1, dtype=np.uint64)
    for i in range(k):
        code = (code << np.uint64(2)) | b[i:n - k + 1 + i]
    csum = np.concatenate([[0], np.cumsum(~ok)])
    good = (csum[k:] - csum[:-k]) == 0
    return np.unique(code[good[:code.shape[0]]])


_M1 = np.uint64(0x9E3779B97F4A7C15)


def mix(x: np.ndarray, salt: int = 0) -> np.ndarray:
    """splitmix64 -- deterministic, with no dependence on an RNG.

    `salt` shifts the input before mixing, giving an independent draw of
    k-mers.  `salt = 0` reproduces the build sketch bit for bit; the
    falsifier uses `PROBE_SALT` so that its blind spot is not the build's.
    """
    x = (x + _M1 + np.uint64(salt)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return (x ^ (x >> np.uint64(31))).astype(np.uint64)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    idx = np.searchsorted(a, b)
    idx[idx >= a.shape[0]] = 0
    w = int((a[idx] == b).sum())
    return w / (a.shape[0] + b.shape[0] - w)


# Input
def load_promoters(path: str | None = None) -> pd.DataFrame:
    """Rows with GROUP=Promotor from sequences.tsv.gz.

    Returns ID, GENOME, OG_ID, GENE_ID, SEQ.  ID is not a key: 950 of 237 780
    IDs occur more than once (different gene models with the same ID), so the
    key is the row number.
    """
    p = paths.require(path or paths.SEQUENCES, "sequences.tsv.gz")
    ID, GEN, OG, GID, SEQ = [], [], [], [], []
    with gzip.open(p, "rt") as fh:
        h = fh.readline().rstrip("\n").split("\t")
        pos = {n: i for i, n in enumerate(h)}
        for needed in ("ID", "GENOME", "OG_ID", "GENE_ID", "SEQUENCE", "GROUP"):
            if needed not in pos:
                raise ValueError("missing column %r in %s" % (needed, p))
        iG, iID, iGE, iOG, iGI, iS = (pos["GROUP"], pos["ID"], pos["GENOME"],
                                      pos["OG_ID"], pos["GENE_ID"],
                                      pos["SEQUENCE"])
        for line in fh:
            q = line.rstrip("\n").split("\t")
            if q[iG] != GROUP:
                continue
            ID.append(q[iID]); GEN.append(q[iGE]); OG.append(q[iOG])
            GID.append(q[iGI]); SEQ.append(q[iS])
    df = pd.DataFrame({"ID": ID, "GENOME": GEN, "OG_ID": OG, "GENE_ID": GID,
                       "SEQ": SEQ})
    if (df["OG_ID"].isin(["", "NA", "nan"])).any():
        # The orthogroup is the seed of a cluster, so an empty OG_ID must get
        # its own seed explicitly rather than share one silently.
        df.loc[df["OG_ID"].isin(["", "NA", "nan"]), "OG_ID"] = \
            "NO_OG:" + df.loc[df["OG_ID"].isin(["", "NA", "nan"]), "ID"]
    return df


def target_per_gene(scope: str = "all_genes_scope",
                    matrix: str = "combat") -> pd.DataFrame:
    """Stratification target: the mean of `combat` over a row's finite cells.

    `rank` is DESCENDING (rank 1 = a silent gene) and is unusable as a target
    without inverting it; `combat` is expression after batch correction.
    """
    ID, MEAN, NN = [], [], []
    with gzip.open(paths.matrix_path(scope, matrix), "rt") as fh:
        h = fh.readline().rstrip("\n").split("\t")
        if h[:4] != ["ID", "gene_id", "genome", "og_id"]:
            raise ValueError("unexpected header %s" % h[:4])
        for line in fh:
            q = line.rstrip("\n").split("\t")
            v = [x for x in q[4:] if x]
            ID.append(q[0])
            if v:
                a = np.asarray(v, dtype=np.float64)
                MEAN.append(float(a.mean())); NN.append(len(v))
            else:
                MEAN.append(np.nan); NN.append(0)
    return pd.DataFrame({"ID": ID, "target": MEAN, "n_measurements": NN})


# Measuring the merge threshold
def measure_threshold(df: pd.DataFrame, n_pairs: int = 4000,
                      n_null: int = 60000, seed: int = SEED) -> dict:
    """Similarity of within-orthogroup pairs against random pairs.

    The merge threshold must sit where these two distributions separate.
    Every returned number goes into the manifest.
    """
    rng = np.random.default_rng(seed)
    seq = df["SEQ"].tolist()
    og = df["OG_ID"].tolist()
    gen = df["GENOME"].tolist()
    n = len(seq)
    by_og: dict[str, list[int]] = {}
    for i, o in enumerate(og):
        by_og.setdefault(o, []).append(i)
    multi = [v for v in by_og.values() if len(v) >= 2]

    cache: dict[int, np.ndarray] = {}

    def km(i: int) -> np.ndarray:
        v = cache.get(i)
        if v is None:
            v = kmers(seq[i])
            if len(cache) > 20000:      # 20k * 6.3 KiB = a 126 MiB ceiling
                cache.clear()
            cache[i] = v
        return v

    Jw, diff_gen = [], []
    for _ in range(n_pairs):
        grp = multi[int(rng.integers(len(multi)))]
        i, j = rng.choice(len(grp), size=2, replace=False)
        a, b = grp[int(i)], grp[int(j)]
        Jw.append(jaccard(km(a), km(b)))
        diff_gen.append(gen[a] != gen[b])
    Jw = np.asarray(Jw)
    diff_gen = np.asarray(diff_gen)

    Jr = []
    ia = rng.integers(n, size=n_null)
    ib = rng.integers(n, size=n_null)
    for a, b in zip(ia, ib):
        if og[int(a)] == og[int(b)]:
            continue
        Jr.append(jaccard(km(int(a)), km(int(b))))
    Jr = np.asarray(Jr)

    pk = [1, 5, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
    return {
        "k": K,
        "n_pairs_within_orthogroup": int(Jw.shape[0]),
        "n_pairs_random": int(Jr.shape[0]),
        "within_mean": float(Jw.mean()),
        "within_percentiles": {str(p): float(v) for p, v in
                               zip(pk, np.percentile(Jw, pk))},
        "within_fraction_identical": float((Jw >= 0.999).mean()),
        "within_different_genomes_mean": float(Jw[diff_gen].mean()),
        "random_mean": float(Jr.mean()),
        "random_percentiles": {str(p): float(v) for p, v in
                               zip(pk, np.percentile(Jr, pk))},
        # Reported, but not the maximum of the null: a cross-orthogroup pair
        # is sometimes genuinely homologous, so the threshold is justified by
        # percentiles instead.
        "random_maximum": float(Jr.max()),
        "random_largest_5": [float(v) for v in np.sort(Jr)[-5:][::-1]],
        "random_above_0_01": int((Jr >= 0.01).sum()),
        "fraction_random_above_threshold": float((Jr >= MERGE_THRESHOLD).mean()),
        "fraction_orthologous_above_threshold": float(
            (Jw >= MERGE_THRESHOLD).mean()),
        "threshold": MERGE_THRESHOLD,
        "threshold_in_multiples_of_random_p9999": float(
            MERGE_THRESHOLD / max(float(np.percentile(Jr, 99.99)), 1e-12)),
    }


def gate_sensitivity(df: pd.DataFrame, n_pairs: int = 3000,
                     seed: int = SEED + 1) -> dict:
    """Falsifier of the candidate stage: does the sketch gate let through the
    pairs an exact Jaccard would consider mergeable?

    The gate (">= MIN_SHARED shared sampled k-mers") is random with respect
    to k-mers, so its sensitivity must be measured, not assumed.
    """
    rng = np.random.default_rng(seed)
    seq = df["SEQ"].tolist()
    og = df["OG_ID"].tolist()
    by_og: dict[str, list[int]] = {}
    for i, o in enumerate(og):
        by_og.setdefault(o, []).append(i)
    multi = [v for v in by_og.values() if len(v) >= 2]
    h_threshold = np.uint64((1 << 64) // SCALE)
    buckets = {"0.05-0.10": [0, 0], "0.10-0.30": [0, 0], ">=0.30": [0, 0]}
    for _ in range(n_pairs):
        grp = multi[int(rng.integers(len(multi)))]
        i, j = rng.choice(len(grp), size=2, replace=False)
        ka, kb = kmers(seq[grp[int(i)]]), kmers(seq[grp[int(j)]])
        J = jaccard(ka, kb)
        if J < MERGE_THRESHOLD:
            continue
        sa = ka[mix(ka) < h_threshold]
        sb = kb[mix(kb) < h_threshold]
        shared = int(np.isin(sa, sb).sum())
        key = ("0.05-0.10" if J < 0.10 else "0.10-0.30" if J < 0.30
               else ">=0.30")
        buckets[key][0] += 1
        buckets[key][1] += int(shared >= MIN_SHARED)
    return {k: {"pairs": v[0], "passed": v[1],
                "sensitivity": (v[1] / v[0]) if v[0] else float("nan")}
            for k, v in buckets.items()}


def full_sensitivity(df: pd.DataFrame, n_pairs: int = 6000,
                     seed: int = SEED + 2, log=print) -> dict:
    """Sensitivity of the whole candidate stage: the sketch gate and the df cut.

    `gate_sensitivity` measures only the first condition.  The second -- a
    k-mer in more than CAP_DF distinct sequences generates no candidates at
    all -- costs sensitivity exactly where the family is large, i.e. where
    merging is needed most.

    A limitation to report with the result: homologous pairs are drawn from
    inside orthogroups, the only place many of them can be drawn, while pairs
    between orthogroups sit in larger families where the df cut bites harder.
    """
    rng = np.random.default_rng(seed)
    seq = df["SEQ"].tolist()
    og = df["OG_ID"].tolist()
    distinct, _assign = _distinct_sequences(seq)
    h_threshold = np.uint64((1 << 64) // SCALE)
    PK = []
    for sq in distinct:
        full = kmers(sq)
        if full.shape[0]:
            kk = full[mix(full) < h_threshold]
            if kk.shape[0]:
                PK.append(kk)
    PK = np.concatenate(PK)
    PK.sort()
    bounds = np.flatnonzero(np.concatenate([[True], PK[1:] != PK[:-1], [True]]))
    kmer = PK[bounds[:-1]]
    dfk = np.diff(bounds)
    allowed = kmer[dfk <= CAP_DF]
    del PK, kmer, dfk

    by_og: dict[str, list[int]] = {}
    for i, o in enumerate(og):
        by_og.setdefault(o, []).append(i)
    multi = [v for v in by_og.values() if len(v) >= 2]
    buckets = {"0.05-0.10": [0, 0, 0], "0.10-0.30": [0, 0, 0],
               ">=0.30": [0, 0, 0]}
    for _ in range(n_pairs):
        grp = multi[int(rng.integers(len(multi)))]
        i, j = rng.choice(len(grp), size=2, replace=False)
        ka, kb = kmers(seq[grp[int(i)]]), kmers(seq[grp[int(j)]])
        J = jaccard(ka, kb)
        if J < MERGE_THRESHOLD:
            continue
        sa = ka[mix(ka) < h_threshold]
        sb = kb[mix(kb) < h_threshold]
        shared = np.intersect1d(sa, sb, assume_unique=True)
        b_ = ("0.05-0.10" if J < 0.10 else "0.10-0.30" if J < 0.30
              else ">=0.30")
        buckets[b_][0] += 1
        buckets[b_][1] += int(shared.shape[0] >= MIN_SHARED)
        buckets[b_][2] += int(int(np.isin(shared, allowed).sum()) >= MIN_SHARED)
    return {"fraction_kmers_allowed": float(allowed.shape[0] /
                                            max(bounds.shape[0] - 1, 1)),
            "buckets": {k_: {"pairs": v[0], "gate_only": v[1],
                             "gate_and_df": v[2],
                             "sensitivity": (v[2] / v[0]) if v[0]
                             else float("nan")}
                        for k_, v in buckets.items()}}


# Homology clusters
class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra < rb:
            self.p[rb] = ra
        else:
            self.p[ra] = rb
        return True


def _candidates(distinct: list[str], log=print
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pairs of distinct sequences sharing >= MIN_SHARED sampled k-mers.

    The index sits at the level of sequences rather than orthogroup unions,
    and that follows from a measurement: the sampled 11-mer space is
    saturated (262 722 observed against 4^11/16 = 262 144 possible), so
    E[shared] = |A||B|/U.  Two unions of ~4000 k-mers give E = 61, more than
    a pair of identical members contributes (~50), so an index at the union
    level has no resolution where it is needed.  At sequence level
    E = 50*50/262144 = 0.0095, and the condition >= 2 has p ~ 4.5e-5.
    """
    t = time.time()
    nU = len(distinct)
    h_threshold = np.uint64((1 << 64) // SCALE)
    PK, PI = [], []
    for j, s in enumerate(distinct):
        full = kmers(s)
        kk = full[mix(full) < h_threshold] if full.shape[0] else full
        if kk.shape[0]:
            PK.append(kk)
            PI.append(np.full(kk.shape[0], j, dtype=np.int32))
    PK = np.concatenate(PK)
    PI = np.concatenate(PI)
    ordr = np.argsort(PK, kind="stable")
    PK, PI = PK[ordr], PI[ordr]
    bounds = np.flatnonzero(np.concatenate([[True], PK[1:] != PK[:-1], [True]]))
    df_ = np.diff(bounds)
    sel = np.flatnonzero((df_ >= 2) & (df_ <= CAP_DF))
    count = int((df_[sel].astype(np.int64) * (df_[sel] - 1) // 2).sum())
    log("   postings %d | k-mers %d | used %d | pairs %.3e | %.1f s"
        % (PK.shape[0], df_.shape[0], sel.shape[0], count, time.time() - t))
    P = max(1, int(np.ceil(count / 40_000_000)))
    A_, B_, C_ = [], [], []
    for p in range(P):
        chunks = []
        for gi in sel:
            a, b = bounds[gi], bounds[gi + 1]
            grp = np.sort(PI[a:b].astype(np.int64))
            ii, jj = np.triu_indices(grp.shape[0], 1)
            code = grp[ii] * nU + grp[jj]
            if P > 1:
                code = code[(code % P) == p]
            if code.shape[0]:
                chunks.append(code)
        code = np.concatenate(chunks)
        del chunks
        code.sort()
        g2 = np.flatnonzero(np.concatenate([[True], code[1:] != code[:-1],
                                            [True]]))
        c = np.diff(g2).astype(np.int32)
        u = code[g2[:-1]]
        del code
        m = c >= MIN_SHARED
        A_.append((u[m] // nU).astype(np.int32))
        B_.append((u[m] % nU).astype(np.int32))
        C_.append(c[m])
        log("   partition %d/%d -> candidates %d (%.1f s)"
            % (p + 1, P, int(m.sum()), time.time() - t))
    A = np.concatenate(A_)
    B = np.concatenate(B_)
    C = np.concatenate(C_)
    info = {"postings": int(PK.shape[0]),
            "kmers_sampled": int(df_.shape[0]),
            "kmers_used": int(sel.shape[0]),
            "pairs_generated": count,
            "partitions": P,
            "candidates": int(A.shape[0]),
            "sampled_space": int(4 ** K // SCALE),
            "space_saturation": float(df_.shape[0] / (4 ** K // SCALE))}
    return A, B, C, info


def _distinct_sequences(seq: list[str]) -> tuple[list[str], np.ndarray]:
    """(list of distinct sequences, row -> distinct-sequence index)."""
    uniq: dict[str, int] = {}
    assign = np.empty(len(seq), dtype=np.int32)
    for i, s in enumerate(seq):
        j = uniq.get(s)
        if j is None:
            j = len(uniq)
            uniq[s] = j
        assign[i] = j
    distinct: list[str] = [""] * len(uniq)
    for s, j in uniq.items():
        distinct[j] = s
    return distinct, assign


def _orthogroup_map(og: list[str], assign: np.ndarray, n_distinct: int):
    ogu = sorted(set(og))
    og_nr = {o: i for i, o in enumerate(ogu)}
    ogi = np.array([og_nr[o] for o in og], dtype=np.int32)
    seq_og: list[set] = [set() for _ in range(n_distinct)]
    for i in range(len(og)):
        seq_og[int(assign[i])].add(int(ogi[i]))
    return ogu, ogi, seq_og


def CACHE_PATH() -> str:
    return os.path.join(CACHE_DIR,
                        "candidates_k%d_s%d_df%d.npz" % (K, SCALE, CAP_DF))


# The independent probe: shared machinery of the falsifier and the
# sensitivity measurement.  Everything below has its own sketch
# (PROBE_SCALE, PROBE_SALT, no df cap) and its own cache files; none of these
# functions reads CACHE_PATH().
def PROBE_CACHE_PATH(round_: int) -> str:
    return os.path.join(CACHE_DIR, "falsifier_k%d_s%d_mw%d_salt%x_r%d.npz"
                        % (K, PROBE_SCALE, PROBE_MIN_SHARED, PROBE_SALT,
                           round_))


def SENSITIVITY_CACHE_PATH() -> str:
    return os.path.join(CACHE_DIR,
                        "cross_og_sensitivity_k%d_s%d.npz" % (K, PROBE_SCALE))


def SPLIT_ASSIGNMENT_FILE() -> str:
    """Per-ROW split assignment, handed to the probe subprocess."""
    return os.path.join(CACHE_DIR, "_split_for_falsifier.tsv")


def EXTRA_MERGES_FILE() -> str:
    """Pairs of distinct sequences merged in after falsifier rounds."""
    return os.path.join(CACHE_DIR, "_extra_merges.npz")


def load_extra_merges() -> np.ndarray:
    """Pairs reported by the falsifier in previous builds.

    Kept on disk because they are the result of a measurement, not a
    parameter: otherwise every build would start from zero and the first
    falsifier round would fire again on the same pairs.  Indices refer to the
    distinct-sequence list of `_distinct_sequences(load_promoters())`, which
    is deterministic given the input file; `build_clusters` checks the range.
    """
    p = EXTRA_MERGES_FILE()
    if not os.path.exists(p):
        return np.zeros((0, 2), dtype=np.int32)
    z = np.load(p)
    return z["pairs"]


def save_extra_merges(pairs: np.ndarray) -> None:
    paths.ensure_dir(CACHE_DIR)
    p = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    code = np.unique(p[:, 0].astype(np.int64) * (1 << 32) + p[:, 1])
    p = np.stack([(code >> 32).astype(np.int32),
                  (code & 0xFFFFFFFF).astype(np.int32)], axis=1)
    np.savez(EXTRA_MERGES_FILE(), pairs=p)


def _kmer_arena(distinct: list[str], log=print) -> tuple[np.ndarray, np.ndarray]:
    """Full k-mer sets of all distinct sequences in one array.

    Verification in `candidate_stage` costs 193 us per pair because `kmers()`
    is recomputed on every miss of a 20 000-entry cache; the arena (1.04 GiB)
    computes each sequence's k-mers exactly once.  Two passes rather than
    one, so the memory peak is 1.04 GiB and not 2.1 GiB.
    """
    t = time.time()
    lens = np.empty(len(distinct) + 1, dtype=np.int64)
    lens[0] = 0
    for i, s in enumerate(distinct):
        lens[i + 1] = kmers(s).shape[0]
    off = np.cumsum(lens)
    data = np.empty(int(off[-1]), dtype=np.uint64)
    for i, s in enumerate(distinct):
        a, b = int(off[i]), int(off[i + 1])
        if b > a:
            data[a:b] = kmers(s)
    log("   arena: %d k-mers, %.2f GiB (%.1f s)"
        % (data.shape[0], data.nbytes / 2 ** 30, time.time() - t))
    return data, off


def _sampled_index(distinct: list[str], scale: int, salt: int, log=print):
    """Inverted index of sampled k-mers, in two orderings.

    Returns (PI, bounds, dfk, kmer_u, seq_slot, seq_bounds):
      PI/bounds -- postings sorted by k-mer (gathering partners),
      seq_slot/seq_bounds -- k-mer numbers sorted by sequence (one target's
                   k-mer list, with no search),
      kmer_u/dfk -- distinct k-mers and their df (needed to judge the cap).
    """
    t = time.time()
    h_threshold = np.uint64((1 << 64) // scale)
    PK, PI = [], []
    for j, s in enumerate(distinct):
        full = kmers(s)
        kk = full[mix(full, salt) < h_threshold] if full.shape[0] else full
        if kk.shape[0]:
            PK.append(kk)
            PI.append(np.full(kk.shape[0], j, dtype=np.int32))
    PK = np.concatenate(PK)
    PI = np.concatenate(PI)
    o = np.argsort(PK, kind="stable")
    PK, PI = PK[o], PI[o]
    del o
    bounds = np.flatnonzero(np.concatenate([[True], PK[1:] != PK[:-1], [True]]))
    dfk = np.diff(bounds).astype(np.int32)
    kmer_u = PK[bounds[:-1]].copy()
    del PK
    slot = np.zeros(PI.shape[0], dtype=np.int32)
    slot[bounds[1:-1]] = 1
    np.cumsum(slot, out=slot)
    o2 = np.argsort(PI, kind="stable")
    seq_slot = slot[o2]
    del o2, slot
    counts = np.bincount(PI, minlength=len(distinct)).astype(np.int64)
    seq_bounds = np.concatenate([[0], np.cumsum(counts)])
    log("   index 1/%d (salt %#x): postings %d | k-mers %d | df max %d (%.1f s)"
        % (scale, salt, PI.shape[0], dfk.shape[0], int(dfk.max()),
           time.time() - t))
    return PI, bounds, dfk, kmer_u, seq_slot, seq_bounds


def _partners_wide(PI, bounds, seq_slot, seq_bounds, targets, min_shared,
                   n_distinct: int, split_of: np.ndarray | None = None,
                   log=print, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Pairs (a < b) of distinct sequences sharing >= `min_shared` sampled
    k-mers, always without a df cap.

    With `split_of` given, only pairs across the part boundary are kept --
    not an optimisation but the definition of what is being measured.
    """
    t = time.time()
    A_, B_ = [], []
    count = 0
    for q_ in targets:
        q = int(q_)
        a0, a1 = int(seq_bounds[q]), int(seq_bounds[q + 1])
        if a1 <= a0:
            continue
        chunks = [PI[bounds[x]:bounds[x + 1]] for x in seq_slot[a0:a1]]
        v = np.concatenate(chunks)
        v.sort()
        g = np.flatnonzero(np.concatenate([[True], v[1:] != v[:-1], [True]]))
        u = v[g[:-1]]
        c = np.diff(g)
        m = (c >= min_shared) & (u != q)
        if split_of is not None and m.any():
            m &= split_of[u] != split_of[q]
        if not m.any():
            continue
        uu = u[m].astype(np.int64)
        A_.append(np.minimum(uu, q))
        B_.append(np.maximum(uu, q))
        count += uu.shape[0]
        if count > 120_000_000:
            raise RuntimeError(
                "the probe generated >1.2e8 pairs -- this will not fit in "
                "memory; raise PROBE_MIN_SHARED or PROBE_SCALE")
    if not A_:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    A = np.concatenate(A_)
    B = np.concatenate(B_)
    del A_, B_
    code = np.unique(A * n_distinct + B)
    A = (code // n_distinct).astype(np.int32)
    B = (code % n_distinct).astype(np.int32)
    log("   [%s] targets %d | pairs %d -> distinct %d (%.1f s)"
        % (label, len(targets), count, A.shape[0], time.time() - t))
    return A, B


def _exact_jaccard(data, off, A, B, log=print, label: str = "") -> np.ndarray:
    """Exact Jaccard of the pairs, grouped by the first element.

    One `searchsorted` for a whole group instead of one per pair: the numpy
    call overhead (~10 us/pair) is the dominant cost here.
    """
    t = time.time()
    if A.shape[0] == 0:
        return np.zeros(0, np.float64)
    o = np.argsort(A, kind="stable")
    A2, B2 = A[o], B[o]
    J = np.empty(A2.shape[0], dtype=np.float64)
    gr = np.flatnonzero(np.concatenate([[True], A2[1:] != A2[:-1], [True]]))
    for g in range(gr.shape[0] - 1):
        i0, i1 = int(gr[g]), int(gr[g + 1])
        a = int(A2[i0])
        ka = data[off[a]:off[a + 1]]
        bs = B2[i0:i1]
        dl = (off[bs + 1] - off[bs]).astype(np.int64)
        if ka.shape[0] == 0:
            J[i0:i1] = 0.0
            continue
        kb = np.concatenate([data[off[b]:off[b + 1]] for b in bs])
        if kb.shape[0] == 0:
            J[i0:i1] = 0.0
            continue
        pos = np.searchsorted(ka, kb)
        pos[pos >= ka.shape[0]] = 0
        hit = (ka[pos] == kb).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(hit)])
        edges = np.concatenate([[0], np.cumsum(dl)])
        w = cum[edges[1:]] - cum[edges[:-1]]
        J[i0:i1] = w / (ka.shape[0] + dl - w)
    out = np.empty_like(J)
    out[o] = J
    log("   [%s] exact Jaccard of %d pairs (%.1f s)"
        % (label, A.shape[0], time.time() - t))
    return out


def _round_salt(round_: int) -> int:
    """Every round draws k-mers differently, otherwise round 2 would only
    confirm round 1's blind spot."""
    return int((PROBE_SALT + round_ * 0x9E3779B97F4A7C15) % (1 << 64))


def _probe_process(round_: int, log=print) -> str:
    """The probe pass (a subprocess): reads SPLIT_ASSIGNMENT_FILE(), writes npz.

    It does not read CACHE_PATH().  A pair the build's candidate stage never
    generated is visible here precisely because the sketch is different --
    denser (PROBE_SCALE), differently salted (`_round_salt`), no df cap.
    """
    paths.ensure_dir(CACHE_DIR)
    df = load_promoters()
    cz = pd.read_csv(SPLIT_ASSIGNMENT_FILE(), sep="\t",
                     dtype=str)["split"].to_numpy()
    if cz.shape[0] != len(df):
        raise RuntimeError("SPLIT_ASSIGNMENT_FILE has %d rows, promoters "
                           "number %d" % (cz.shape[0], len(df)))
    distinct, assign = _distinct_sequences(df["SEQ"].tolist())
    n = len(distinct)
    code = {c: i for i, c in enumerate(SPLITS)}
    split_of = np.full(n, -1, dtype=np.int8)
    for i in range(cz.shape[0]):
        j = int(assign[i])
        k = code[cz[i]]
        if split_of[j] < 0:
            split_of[j] = k
        elif split_of[j] != k:
            raise RuntimeError(
                "the same sequence (distinct no. %d) is in two parts -- "
                "identity merging is broken" % j)
    log("   distinct sequences %d | per part %s"
        % (n, {c: int((split_of == code[c]).sum()) for c in SPLITS}))

    salt = _round_salt(round_)
    data, off = _kmer_arena(distinct, log=log)
    PI, bounds, dfk, _ku, seq_slot, seq_bounds = _sampled_index(
        distinct, PROBE_SCALE, salt, log=log)
    A, B = _partners_wide(PI, bounds, seq_slot, seq_bounds, range(n),
                          PROBE_MIN_SHARED, n, split_of=split_of, log=log,
                          label="round %d" % round_)
    del PI, bounds, seq_slot
    J = _exact_jaccard(data, off, A, B, log=log, label="round %d" % round_)

    m = J >= MERGE_THRESHOLD
    # Below J = 0.01 is background we would not merge anyway, and the file is
    # meant to stay readable.
    keep = J >= 0.01
    weights = np.bincount(assign, minlength=n)
    info = {
        "round": round_, "salt": "%#x" % salt, "scale": PROBE_SCALE,
        "min_shared": PROBE_MIN_SHARED, "cap_df": PROBE_CAP_DF,
        "distinct_sequences": n,
        "cross_boundary_candidates": int(A.shape[0]),
        "cross_boundary_above_threshold": int(m.sum()),
        "probe_df_max": int(dfk.max()),
    }
    for lo, hi in ((MERGE_THRESHOLD, 1.01), (0.03, MERGE_THRESHOLD),
                   (0.02, 0.03), (0.01, 0.02)):
        mm = (J >= lo) & (J < hi)
        touched = (np.unique(np.concatenate([A[mm], B[mm]])) if mm.any()
                   else np.zeros(0, np.int32))
        info["J_%.2f_%.2f" % (lo, hi)] = {
            "pairs": int(mm.sum()),
            "distinct_sequences_touched": int(touched.shape[0]),
            "rows_touched": (int(weights[touched].sum())
                             if touched.shape[0] else 0)}
    np.savez(PROBE_CACHE_PATH(round_), A=A[keep], B=B[keep], J=J[keep],
             above=m[keep], n=np.int64(n), info=json.dumps(info))
    log("   across the boundary above %.2f: %d pairs"
        % (MERGE_THRESHOLD, int(m.sum())))
    return PROBE_CACHE_PATH(round_)


def leakage_falsifier(row_split: pd.Series, round_: int = 1, log=print) -> dict:
    """Runs the probe in a separate process and returns its report.

    The returned `pairs` are distinct-sequence indices (not rows) of pairs
    that crossed the merge threshold despite the part boundary: the leakage
    the build's candidate stage did not see.
    """
    paths.ensure_dir(CACHE_DIR)
    pd.DataFrame({"split": row_split.to_numpy()}).to_csv(
        SPLIT_ASSIGNMENT_FILE(), sep="\t", index=False)
    path = PROBE_CACHE_PATH(round_)
    if os.path.exists(path):
        os.remove(path)          # the round must be recomputed from scratch
    r = subprocess.run([sys.executable, "-u", "-m",
                        "hyppe.analysis.splits_build",
                        "--falsifier", str(round_)],
                       cwd=_SRC_DIR, env=_subprocess_env())
    if r.returncode != 0 or not os.path.exists(path):
        raise RuntimeError("the falsifier failed (rc=%s)" % r.returncode)
    z = np.load(path)
    info = json.loads(str(z["info"]))
    above = z["above"]
    info["pairs"] = (np.stack([z["A"][above], z["B"][above]], axis=1)
                     if int(above.sum()) else np.zeros((0, 2), dtype=np.int32))
    return info


def cross_orthogroup_sensitivity(n_targets: int = 20000,
                                 seed: int = SEED + 7,
                                 log=print, cache: bool = True) -> dict:
    """Sensitivity of the build's narrow gate on cross-orthogroup pairs.

    `gate_sensitivity` and `full_sensitivity` draw pairs from inside
    orthogroups, the only place many of them can be drawn -- but the
    population that matters for the split is pairs between orthogroups, since
    orthologues share a seed anyway.  Here the population comes from the wide
    probe (PROBE_SCALE, no cap) and the measured quantity is the fraction the
    narrow gate (SCALE, CAP_DF, MIN_SHARED) would also have generated.  Both
    populations go into the manifest separately; quoting the within-orthogroup
    number alone overstates the candidate stage.
    """
    path = SENSITIVITY_CACHE_PATH()
    if cache and os.path.exists(path):
        z = np.load(path)
        return json.loads(str(z["info"]))
    df = load_promoters()
    distinct, assign = _distinct_sequences(df["SEQ"].tolist())
    og = df["OG_ID"].tolist()
    _ogu, _ogi, seq_og = _orthogroup_map(og, assign, len(distinct))
    del df, og
    data, off = _kmer_arena(distinct, log=log)
    PI, bounds, dfk, ku, seq_slot, seq_bounds = _sampled_index(
        distinct, PROBE_SCALE, PROBE_SALT, log=log)
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(len(distinct),
                                 size=min(n_targets, len(distinct)),
                                 replace=False))
    A, B = _partners_wide(PI, bounds, seq_slot, seq_bounds, targets,
                          PROBE_MIN_SHARED, len(distinct), log=log,
                          label="sensitivity")
    del PI, bounds, seq_slot, dfk, ku
    J = _exact_jaccard(data, off, A, B, log=log, label="sensitivity")
    m = J >= MERGE_THRESHOLD
    A, B, J = A[m], B[m], J[m]
    log("   pairs above the threshold %d" % A.shape[0])

    # The build's narrow sketch -- the same one `_candidates` uses.
    _PI2, bounds2, dfk2, ku2, _ss2, _sb2 = _sampled_index(
        distinct, SCALE, 0, log=log)
    del _PI2, _ss2, _sb2, bounds2
    h_narrow = np.uint64((1 << 64) // SCALE)
    # [within_og_pairs, within_og_caught, cross_og_pairs, cross_og_caught]
    buckets = {"0.05-0.10": [0, 0, 0, 0], "0.10-0.30": [0, 0, 0, 0],
               ">=0.30": [0, 0, 0, 0]}
    for a, b, j in zip(A, B, J):
        a, b = int(a), int(b)
        ka = data[off[a]:off[a + 1]]
        kb = data[off[b]:off[b + 1]]
        sa = ka[mix(ka) < h_narrow]
        sb = kb[mix(kb) < h_narrow]
        shared = np.intersect1d(sa, sb, assume_unique=True)
        if shared.shape[0]:
            pos = np.searchsorted(ku2, shared)
            pos[pos >= ku2.shape[0]] = 0
            ok = (ku2[pos] == shared) & (dfk2[pos] <= CAP_DF)
            caught = int(ok.sum()) >= MIN_SHARED
        else:
            caught = False
        kl = ("0.05-0.10" if j < 0.10 else "0.10-0.30" if j < 0.30
              else ">=0.30")
        same_og = bool(seq_og[a] & seq_og[b])
        buckets[kl][0 if same_og else 2] += 1
        buckets[kl][(1 if same_og else 3)] += int(caught)
    out = {
        "n_targets": int(targets.shape[0]), "seed": seed,
        "probe": {"scale": PROBE_SCALE, "min_shared": PROBE_MIN_SHARED,
                  "cap_df": PROBE_CAP_DF, "salt": "%#x" % PROBE_SALT},
        "build_gate": {"scale": SCALE, "min_shared": MIN_SHARED,
                       "cap_df": CAP_DF},
        "pairs_above_threshold": int(A.shape[0]),
        "buckets": {k_: {
            "within_og_pairs": v[0], "within_og_caught": v[1],
            "within_og_sensitivity": (v[1] / v[0]) if v[0] else float("nan"),
            "cross_og_pairs": v[2],
            "cross_og_caught": v[3],
            "cross_og_sensitivity": (v[3] / v[2]) if v[2] else float("nan"),
        } for k_, v in buckets.items()},
    }
    tot_w = sum(v[0] for v in buckets.values())
    got_w = sum(v[1] for v in buckets.values())
    tot_c = sum(v[2] for v in buckets.values())
    got_c = sum(v[3] for v in buckets.values())
    out["within_og_sensitivity_overall"] = ((got_w / tot_w) if tot_w
                                            else float("nan"))
    out["cross_og_sensitivity_overall"] = ((got_c / tot_c) if tot_c
                                           else float("nan"))
    np.savez(path, A=A, B=B, J=J, info=json.dumps(out))
    return out


def candidate_stage(log=print) -> str:
    """The expensive stage: candidates + exact verification -> disk cache.

    Run as a separate process (see `build_clusters`): one process doing
    everything grew to 10.3 GiB of private memory on a 15.8 GiB machine and
    started thrashing.  Millions of small numpy allocations are returned to
    the system only when the process exits.
    """
    paths.ensure_dir(CACHE_DIR)
    path = CACHE_PATH()
    df = load_promoters()
    seq = df["SEQ"].tolist()
    og = df["OG_ID"].tolist()
    del df
    distinct, assign = _distinct_sequences(seq)
    del seq
    log("   sequences %d -> distinct %d" % (assign.shape[0], len(distinct)))
    _ogu, _ogi, seq_og = _orthogroup_map(og, assign, len(distinct))
    del og, assign

    A, B, C, info = _candidates(distinct, log=log)
    # Pairs inside one orthogroup drop out before verification: an orthogroup
    # is already one seed, so such a pair cannot merge anything.
    first = np.array([min(s) for s in seq_og], dtype=np.int32)
    many = np.array([len(s) > 1 for s in seq_og])
    cross = (first[A] != first[B]) | many[A] | many[B]
    info["candidates_between_distinct_orthogroups"] = int(cross.sum())
    log("   candidates %d, between DISTINCT orthogroups %d"
        % (A.shape[0], int(cross.sum())))
    A, B, C = A[cross], B[cross], C[cross]

    ordr = np.argsort(A, kind="stable")
    A, B, C = A[ordr], B[ordr], C[ordr]
    t = time.time()
    J = np.empty(A.shape[0], dtype=np.float32)
    ka = None
    last = -1
    mem: dict[int, np.ndarray] = {}
    for p in range(A.shape[0]):
        a = int(A[p])
        if a != last:
            ka = kmers(distinct[a])
            last = a
        b = int(B[p])
        kb = mem.get(b)
        if kb is None:
            kb = kmers(distinct[b])
            if len(mem) > 20000:
                mem.clear()
            mem[b] = kb
        J[p] = jaccard(ka, kb)
    log("   exact verification of %d pairs in %.1f s"
        % (A.shape[0], time.time() - t))
    for th in (0.02, 0.05, 0.10, 0.20):
        info["pairs_above_%.2f" % th] = int((J >= th).sum())
    np.savez(path, A=A, B=B, C=C, J=J, info=json.dumps(info))
    log("   cache -> %s" % path)
    return path


def build_clusters(df: pd.DataFrame, log=print, cache: bool = True,
                   extra_pairs: np.ndarray | None = None
                   ) -> tuple[np.ndarray, dict]:
    """Homology clusters: the orthogroup as a seed, plus merging on similarity.

    `extra_pairs` is an (M, 2) array of distinct-sequence indices that the
    independent falsifier found above the threshold and the build's narrow
    gate missed.  The loop closes this way rather than by loosening the gate,
    because otherwise the falsifier would again be testing its own input.

    Returns (cluster number per row, report).
    """
    paths.ensure_dir(CACHE_DIR)
    path = CACHE_PATH()
    seq = df["SEQ"].tolist()
    og = df["OG_ID"].tolist()
    distinct, assign = _distinct_sequences(seq)
    log("   sequences %d -> distinct %d" % (len(seq), len(distinct)))

    if not (cache and os.path.exists(path)):
        log("   no cache -- candidate stage in a separate process ...")
        r = subprocess.run([sys.executable, "-u", "-m",
                            "hyppe.analysis.splits_build", "--candidates"],
                           cwd=_SRC_DIR, env=_subprocess_env())
        if r.returncode != 0 or not os.path.exists(path):
            raise RuntimeError("the candidate stage failed (rc=%s)"
                               % r.returncode)
    z = np.load(path)
    A, B, C, J = z["A"], z["B"], z["C"], z["J"]
    info = json.loads(str(z["info"]))
    log("   candidates from %s: %d pairs"
        % (os.path.basename(path), A.shape[0]))

    ogu, ogi, seq_og = _orthogroup_map(og, assign, len(distinct))

    uf = UnionFind(len(ogu))
    # 1. The same sequence in two orthogroups: unconditional merge.
    n_ident = 0
    for s in seq_og:
        if len(s) > 1:
            v = sorted(s)
            for x in v[1:]:
                n_ident += int(uf.union(v[0], x))
    # 2. Similarity above the threshold.
    m = J >= MERGE_THRESHOLD
    n_sim = 0
    for a, b in zip(A[m], B[m]):
        for x in seq_og[int(a)]:
            for y in seq_og[int(b)]:
                n_sim += int(uf.union(x, y))
    # 3. Pairs reported by the independent falsifier in previous rounds.
    n_extra = 0
    n_extra_pairs = 0
    if extra_pairs is not None and len(extra_pairs):
        n_extra_pairs = int(len(extra_pairs))
        for a, b in extra_pairs:
            if int(a) >= len(distinct) or int(b) >= len(distinct):
                raise RuntimeError(
                    "extra merge pair (%d,%d) out of range of %d distinct "
                    "sequences -- the indices come from a different input"
                    % (a, b, len(distinct)))
            for x in seq_og[int(a)]:
                for y in seq_og[int(b)]:
                    n_extra += int(uf.union(x, y))
    root = np.array([uf.find(i) for i in range(len(ogu))], dtype=np.int32)
    _, cluster_og = np.unique(root, return_inverse=True)
    cluster = cluster_og[ogi].astype(np.int32)

    sizes = np.bincount(cluster)
    sizes_og = np.bincount(cluster_og, minlength=cluster_og.max() + 1)
    report = dict(info)
    report.update({
        "distinct_sequences": len(distinct),
        "seed_orthogroups": len(ogu),
        "pairs_above_threshold": int(m.sum()),
        "merges_from_identity": n_ident,
        "merges_from_similarity": n_sim,
        "pairs_from_falsifier": n_extra_pairs,
        "merges_from_falsifier": n_extra,
        "clusters": int(sizes.shape[0]),
        "largest_cluster_sequences": int(sizes.max()),
        "largest_cluster_orthogroups": int(sizes_og.max()),
        "single_sequence_clusters": int((sizes == 1).sum()),
        "clusters_of_several_orthogroups": int((sizes_og > 1).sum()),
        "fraction_sequences_in_largest": float(sizes.max() / sizes.sum()),
    })
    return cluster, report


# Row split (clusters), stratified
def _quotas(n: int, prop: dict[str, float],
            debt: dict[str, float] | None = None) -> dict[str, int]:
    """Largest-remainder method; the quotas sum to exactly n.

    `debt` (modified in place) carries the fractional shortfall between
    strata.  Without it, putting the genome in the stratum key fragments the
    strata from 50 to ~950 and every single-element stratum hands its whole
    cluster to `train`, since at 0.70/0.10/0.10/0.10 the largest remainder
    belongs to the largest quota.  With the debt the proportions hold
    globally, and |debt| stays bounded because sum(debt) = 0 by construction.
    `debt=None` reproduces the unweighted behaviour, which is how the column
    split calls it.
    """
    d = {c: 0.0 for c in SPLITS} if debt is None else debt
    raw = {c: n * prop[c] + d[c] for c in SPLITS}
    base = {c: max(0, int(np.floor(raw[c]))) for c in SPLITS}
    missing = n - sum(base.values())
    if missing > 0:
        rest = sorted(SPLITS, key=lambda c: (-(raw[c] - base[c]), c))
        for i in range(missing):
            base[rest[i % len(SPLITS)]] += 1
    elif missing < 0:
        rest = sorted(SPLITS, key=lambda c: ((raw[c] - base[c]), c))
        i = 0
        while missing < 0:
            c = rest[i % len(SPLITS)]
            if base[c] > 0:
                base[c] -= 1
                missing += 1
            i += 1
            if i > 64 * len(SPLITS):
                raise RuntimeError("quotas do not converge to n -- the debt "
                                   "has drifted")
    if debt is not None:
        for c in SPLITS:
            debt[c] = raw[c] - base[c]
    return base


def assign_clusters(kl: pd.DataFrame, prop=PROPORTIONS, seed: int = SEED,
                    log=print) -> pd.Series:
    """Assignment of clusters to parts.

    The quota is computed over clusters and met exactly within each stratum.
    Stratum = (size bucket) x (target quartile) x (contains the target
    strain) x (dominant genome), the four things that account for most of the
    compositional difference between parts.  Within a stratum the assignment
    is greedy on the deficit in number of sequences and in P1 sequences, so
    the parts are comparable by volume and not only by cluster count.

    Genome belongs in the key: without it the TVD of the genome mixture was
    0.0094-0.0297 against a permutation null of 0.0012-0.0054, i.e. 3-6x,
    though the largest absolute gap was only 0.68 pp.  The manifest carries
    the post-split TVD so that the correction can be checked.
    """
    rng = np.random.default_rng(seed)
    bucket = pd.cut(kl["n_seq"], [0, 1, 5, 20, 50, 10 ** 9],
                    labels=["1", "2-5", "6-20", "21-50", ">50"])
    q = pd.qcut(kl["target"].rank(method="first", na_option="keep"), 4,
                labels=["q1", "q2", "q3", "q4"])
    q = q.cat.add_categories(["none"]).fillna("none")
    if "dominant_genome" not in kl.columns:
        raise RuntimeError("no 'dominant_genome' column -- the genome is a "
                           "stratifying variable and may not enter silently "
                           "as missing")
    kl = kl.assign(_bucket=bucket.astype(str), _q=q.astype(str),
                   _p1=(kl["n_p1"] > 0).astype(int),
                   _gen=kl["dominant_genome"].astype(str))
    split = pd.Series(index=kl.index, dtype=object)

    tot_s = float(kl["n_seq"].sum())
    tot_p1 = float(max(kl["n_p1"].sum(), 1))
    state_s = {c: 0.0 for c in SPLITS}
    state_p1 = {c: 0.0 for c in SPLITS}
    # Stratum order is part of the algorithm: the largest clusters are placed
    # first, while all parts are empty, otherwise they land in whichever part
    # happens to lag behind (a cluster of 3125 sequences is 13.3 % of `val`
    # and 1.8 % of `train`).
    groups = sorted(kl.groupby(["_bucket", "_q", "_p1", "_gen"], sort=True,
                               observed=True),
                    key=lambda kg: (-float(kg[1]["n_seq"].max()), str(kg[0])))
    n_strata = 0
    debt = {c: 0.0 for c in SPLITS}
    for _key, grp in groups:
        n_strata += 1
        idx = grp.index.to_numpy()
        idx = idx[rng.permutation(idx.shape[0])]
        # Largest clusters first: they shift the balance hardest.
        idx = idx[np.argsort(-kl.loc[idx, "n_seq"].to_numpy(), kind="stable")]
        kw = _quotas(idx.shape[0], prop, debt)
        for i in idx:
            candidates = [c for c in SPLITS if kw[c] > 0]

            def deficit(c):
                return ((prop[c] * tot_s - state_s[c]) / tot_s
                        + (prop[c] * tot_p1 - state_p1[c]) / tot_p1)

            c = max(candidates, key=lambda c: (deficit(c), c))
            split.at[i] = c
            kw[c] -= 1
            state_s[c] += float(kl.loc[i, "n_seq"])
            state_p1[c] += float(kl.loc[i, "n_p1"])
    log("   strata: %d | final debt %s"
        % (n_strata, {c: round(debt[c], 3) for c in SPLITS}))
    if split.isna().any():
        raise RuntimeError("a cluster with no part -- the assignment is "
                           "incomplete")
    return split


# Column split (studies)
def _condition_key(samples: pd.DataFrame) -> pd.Series:
    """The second key of the column axis: (condition, genome).

    The same culture of the same strain described in two submissions is the
    same observation, even when `study` differs.
    """
    for col in ("condition", "genome"):
        if col not in samples.columns:
            raise ValueError("no column %r in samples_metadata" % col)
    return (samples["condition"].fillna("").astype(str) + "\t"
            + samples["genome"].fillna("").astype(str))


def column_components(samples: pd.DataFrame) -> pd.Series:
    """The atom of the column axis: a connected component of the sample graph.

    An edge joins samples with the same `study` or the same
    (condition, genome) key.  `study` alone is not enough: 69.98 % of studies
    are single-sample, so for 14.49 % of the panel holding the study holds
    nothing, and a guard checking `study` disjointness alone let 112 pairs
    across the part boundary at a median expression correlation of 0.9523.
    """
    idx = list(samples.index)
    pos = {s: i for i, s in enumerate(idx)}
    uf = UnionFind(len(idx))
    for col in (samples["study"].astype(str), _condition_key(samples)):
        first: dict[str, int] = {}
        for s, v in zip(idx, col.to_numpy()):
            p = first.get(v)
            if p is None:
                first[v] = pos[s]
            else:
                uf.union(p, pos[s])
    root = np.array([uf.find(i) for i in range(len(idx))], dtype=np.int64)
    _, nr = np.unique(root, return_inverse=True)
    return pd.Series(nr.astype(np.int64), index=samples.index,
                     name="component")


def assign_columns(samples: pd.DataFrame, prop=PROPORTIONS, seed: int = SEED,
                   log=print) -> tuple[pd.Series, dict]:
    """Assignment of components to parts, stratified by dominant genome.

    A genome with a single component goes wholly to `train`, not by a
    hand-written exception but because at 0.70/0.10/0.10/0.10 the
    largest-remainder method gives the single unit to the largest quota.

    Returns (part per sample, report about the components).
    """
    rng = np.random.default_rng(seed + 1)
    comp = column_components(samples)
    s = samples.assign(_comp=comp.to_numpy())
    sizes = s.groupby("_comp").size()
    # Dominant genome of a component; the atom stays the component, not the
    # component-genome pair.
    g = (s.groupby(["_comp", "genome"]).size().reset_index(name="n")
         .sort_values(["_comp", "n", "genome"], ascending=[True, False, True]))
    dom = g.drop_duplicates("_comp").set_index("_comp")
    split_comp = pd.Series(index=sizes.index, dtype=object)
    debt = {c: 0.0 for c in SPLITS}
    for _genome, grp in dom.groupby("genome"):
        st = grp.index.to_numpy()
        st = st[rng.permutation(st.shape[0])]
        st = st[np.argsort(-sizes.loc[st].to_numpy(), kind="stable")]
        kw = _quotas(st.shape[0], prop, debt)
        tot = float(sizes.loc[st].sum())
        state = {c: 0.0 for c in SPLITS}
        for x in st:
            candidates = [c for c in SPLITS if kw[c] > 0]
            c = max(candidates,
                    key=lambda c: ((prop[c] * tot - state[c]) / max(tot, 1), c))
            split_comp.at[x] = c
            kw[c] -= 1
            state[c] += float(sizes.loc[x])
    if split_comp.isna().any():
        raise RuntimeError("a component with no part")
    studies_per_comp = s.groupby("_comp")["study"].nunique()
    report = {
        "atom": "connected component of the graph (study OR (condition, genome))",
        "samples": int(len(s)),
        "studies": int(s["study"].nunique()),
        "single_sample_studies": int((s.groupby("study").size() == 1).sum()),
        "components": int(len(sizes)),
        "components_joining_several_studies": int((studies_per_comp > 1).sum()),
        "merges_across_studies": int(s["study"].nunique() - len(sizes)),
        "largest_component_samples": int(sizes.max()),
        "largest_component_studies": int(studies_per_comp.max()),
        "single_sample_components": int((sizes == 1).sum()),
        "genomes_with_one_component": int(
            (dom.reset_index().groupby("genome")["_comp"].nunique() == 1).sum()),
    }
    log("   studies %d -> components %d (merges across studies %d); "
        "largest component %d samples / %d studies"
        % (report["studies"], report["components"],
           report["merges_across_studies"],
           report["largest_component_samples"],
           report["largest_component_studies"]))
    return (split_comp.reindex(comp.to_numpy()).set_axis(samples.index),
            report)


def column_confounding(samples: pd.DataFrame, sample_split: pd.Series) -> dict:
    """How much a study split shifts the genome mixture, as a number."""
    def H(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    pg = samples["genome"].value_counts(normalize=True)
    Hg = H(pg.values)
    Hgs = 0.0
    for _st, grp in samples.groupby("study"):
        Hgs += (len(grp) / len(samples)) * H(
            grp["genome"].value_counts(normalize=True).values)
    Hgc = 0.0
    for _c, grp in samples.groupby(sample_split.reindex(samples.index).values):
        Hgc += (len(grp) / len(samples)) * H(
            grp["genome"].value_counts(normalize=True).values)
    tvd = {}
    for c in SPLITS:
        m = sample_split.reindex(samples.index).values == c
        q = (samples.loc[m, "genome"].value_counts(normalize=True)
             .reindex(pg.index).fillna(0.0))
        tvd[c] = float(0.5 * np.abs(q.values - pg.values).sum())
    gps = samples.groupby("study")["genome"].nunique()
    return {
        "H_genome_bits": Hg,
        "H_genome_given_study_bits": Hgs,
        "I_genome_study_over_H": (Hg - Hgs) / Hg,
        "H_genome_given_part_bits": Hgc,
        "I_genome_part_over_H": (Hg - Hgc) / Hg,
        "TVD_genome_mixture_vs_whole": tvd,
        "multi_genome_studies": int((gps > 1).sum()),
        "studies": int(len(gps)),
        "samples_in_multi_genome_studies": int(samples["study"].isin(
            gps[gps > 1].index).sum()),
        "genomes_with_fewer_than_4_studies": int(
            (samples.groupby("genome")["study"].nunique() < 4).sum()),
    }


def row_confounding(df: pd.DataFrame, cluster_split: pd.Series,
                    seed: int = SEED + 11, n_perm: int = 200) -> dict:
    """TVD of the genome mixture per part against the whole, with a
    permutation null.

    The null permutes cluster labels, not rows: the cluster is the atom, so
    its indivisibility alone contributes TVD and a row-wise null would
    overstate the effect.  Both numbers go into the manifest, because the
    stratification shows up as a fall in their ratio.
    """
    rng = np.random.default_rng(seed)
    gen = df["GENOME"].to_numpy()
    kl = df["cluster"].to_numpy()
    genu, gi = np.unique(gen, return_inverse=True)
    p = np.bincount(gi, minlength=genu.shape[0]).astype(float)
    p /= p.sum()
    klu = cluster_split.index.to_numpy()
    et = cluster_split.to_numpy()
    pos = {int(k): i for i, k in enumerate(klu)}
    kl_i = np.array([pos[int(k)] for k in kl], dtype=np.int64)

    def tvd(labels):
        out = {}
        cz = labels[kl_i]
        for c in SPLITS:
            m = cz == c
            q = np.bincount(gi[m], minlength=genu.shape[0]).astype(float)
            q = q / max(q.sum(), 1.0)
            out[c] = float(0.5 * np.abs(q - p).sum())
        return out

    obs = tvd(et)
    null = {c: [] for c in SPLITS}
    for _ in range(n_perm):
        v = tvd(et[rng.permutation(et.shape[0])])
        for c in SPLITS:
            null[c].append(v[c])
    return {
        "TVD_observed": obs,
        "TVD_permutation_null_p50": {c: float(np.percentile(null[c], 50))
                                     for c in SPLITS},
        "TVD_permutation_null_p95": {c: float(np.percentile(null[c], 95))
                                     for c in SPLITS},
        "ratio_obs_to_null_p50": {
            c: float(obs[c] / max(float(np.percentile(null[c], 50)), 1e-12))
            for c in SPLITS},
        "n_permutations": n_perm,
        "largest_gap_percentage_points": float(
            100 * max(obs[c] for c in SPLITS)),
    }


def leakage_description(df: pd.DataFrame, log=print) -> dict:
    """A description of the leakage we accept.  This is not a falsifier.

    It loads CACHE_PATH(), exactly the candidate file `build_clusters` made
    its merges from, so a pair the candidate stage never generated is
    invisible to the merging and to this control at once and "crossing the
    boundary = 0" is true by construction.  The falsifier is
    `leakage_falsifier`, with its own sketch, salt and cache.

    What is meaningful here is the description of leaks below the threshold,
    which we knowingly do not merge and which are legitimately measured with
    the index that let them through.
    """
    path = CACHE_PATH()
    if not os.path.exists(path):
        raise FileNotFoundError("no candidate cache: %s" % path)
    z = np.load(path)
    A, B, J = z["A"], z["B"], z["J"]
    distinct, assign = _distinct_sequences(df["SEQ"].tolist())
    cz = df["split"].to_numpy()
    seq_split: list[set] = [set() for _ in range(len(distinct))]
    for i in range(cz.shape[0]):
        seq_split[int(assign[i])].add(cz[i])
    n_eval = int((cz != "train").sum())
    out = {}
    for lo, hi in ((MERGE_THRESHOLD, 1.01), (0.03, MERGE_THRESHOLD),
                   (0.02, 0.03), (0.01, 0.02)):
        m = (J >= lo) & (J < hi)
        crossing = 0
        touched: set[int] = set()
        for a_, b_ in zip(A[m], B[m]):
            sa, sb = seq_split[int(a_)], seq_split[int(b_)]
            if sa != sb or len(sa) > 1:
                crossing += 1
            if "train" in sa and (sb - {"train"}):
                touched.add(int(b_))
            if "train" in sb and (sa - {"train"}):
                touched.add(int(a_))
        rows = int((np.isin(assign, list(touched)) & (cz != "train")).sum())
        out["J_%.2f_%.2f" % (lo, hi)] = {
            "pairs": int(m.sum()), "crossing_boundary": crossing,
            "distinct_eval_sequences_with_train_partner": len(touched),
            "eval_rows": rows,
            "fraction_of_eval_set": rows / max(n_eval, 1)}
        log("   J in [%.2f,%.2f): pairs %6d | crossing %5d | eval rows %5d "
            "(%.4f)" % (lo, hi, int(m.sum()), crossing, rows,
                        rows / max(n_eval, 1)))
    key = "J_%.2f_%.2f" % (MERGE_THRESHOLD, 1.01)
    out["note"] = ("a description, not a falsifier: computed from the same "
                   "cache the merges were made from -- the zero in the first "
                   "row is true by construction")
    if out[key]["crossing_boundary"] != 0:
        # Not a falsifier (it would be a tautology), but a non-zero value
        # means the merging did not use its own candidates: a code bug, not
        # a blind spot of the measurement.
        raise RuntimeError(
            "%d pairs with J >= %.2f from OUR OWN cache cross the boundary -- "
            "the merging did not use its own candidates"
            % (out[key]["crossing_boundary"], MERGE_THRESHOLD))
    return out


# Guards
#: Manifest sections present only in a version-2 build.  Each records a
#: measurement a version-1 split never made, and their absence is invisible
#: to every row-level test.
V2_EVIDENCE = {
    "falsifier_rounds":
        "the independent leakage falsifier (own sketch, own salt, no df cap)",
    "cross_orthogroup_sensitivity":
        "gate sensitivity on CROSS-ORTHOGROUP pairs (defect 5)",
    "genome_row_confounding":
        "TVD of the genome mixture against a permutation null (defect 4)",
    "column_components":
        "the column atom as a connected component, not a study (defect 3)",
}


def manifest_missing_evidence(man: dict) -> list[str]:
    """Which version-2 evidence sections this manifest does not carry."""
    return [("%s (%s)" % (k, v)) for k, v in sorted(V2_EVIDENCE.items())
            if k not in man]


def guard_disjoint(w: pd.DataFrame) -> None:
    """Raises when a cluster occurs in more than one part.

    It raises rather than warning because leakage between train and test
    changes no dimension and breaks no shape assertion: it shows up only as
    an inflated result.
    """
    bad = w.groupby("cluster")["split"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise RuntimeError(
            "LEAKAGE: %d clusters occur in >1 part, e.g. %s"
            % (len(bad), list(bad.index[:5])))
    bad2 = w.groupby("OG_ID")["split"].nunique()
    bad2 = bad2[bad2 > 1]
    if len(bad2):
        raise RuntimeError(
            "LEAKAGE: %d orthogroups occur in >1 part, e.g. %s"
            % (len(bad2), list(bad2.index[:5])))
    for c in SPLITS:
        if not (w["split"] == c).any():
            raise RuntimeError("part %r is empty" % c)
        if not ((w["split"] == c) & (w["GENOME"] == TARGET_STRAIN)).any():
            raise RuntimeError("part %r does not contain strain %s"
                               % (c, TARGET_STRAIN))


def guard_manifest(man: dict, where: str = "manifest.json",
                   allow_incomplete: bool = False) -> None:
    """Raises when the manifest does not describe a cluster split in four parts.

    Separate from `guard_disjoint` because rows are also disjoint when the
    atom was an orthogroup, and an OG_ID split closes orthologues but neither
    paralogues nor gene families: 18 764 pairs of distinct sequences from
    distinct orthogroups sit at J >= 0.05.  Such a split passes every
    disjointness test and leaks anyway, and the only trace is what the atom
    was, so this guard asks about that directly.
    """
    if "klastrowanie" in man or "rozmiary_wierszy" in man:
        raise ValueError(
            "%s is in the LEGACY POLISH schema.  Convert it once with "
            "`python -m hyppe.analysis.splits_build --migrate <old dir>` -- "
            "reading both schemas silently is exactly the fallback this "
            "module exists to prevent." % where)
    kl = man.get("clustering")
    if not isinstance(kl, dict) or "clusters" not in kl:
        raise ValueError(
            "%s is not a manifest of a CLUSTER split (no `clustering` "
            "section).  A split by OG_ID alone closes neither paralogues nor "
            "gene families and may not be substituted here -- build the "
            "canonical one:  python -m hyppe.analysis.splits_build" % where)
    prop = (man.get("parameters") or {}).get("proportions") or {}
    missing = [c for c in SPLITS if c not in prop]
    if missing:
        raise ValueError(
            "%s does not have the four parts %s -- missing %s.  The part "
            "`spy` (choosing BETWEEN models) and `test` (opened once) are "
            "disjoint because of the measured sd of the evaluation alone, "
            "0.0064 (hyppe/szum_oceny.py); a three-part split has nowhere to "
            "separate that." % (where, list(SPLITS), missing))
    r = man.get("row_sizes") or {}
    empty = [c for c in SPLITS if int((r.get(c) or {}).get("sequences", 0)) <= 0]
    if empty:
        raise ValueError("%s: parts %s are empty" % (where, empty))
    missing_evidence = manifest_missing_evidence(man)
    if missing_evidence and not allow_incomplete:
        raise ValueError(
            "%s describes a split built BEFORE the version-2 fixes: it is "
            "missing %s.\nThat means: no INDEPENDENT leakage falsifier ran "
            "(the only control that does not read the build's own cache), the "
            "cross-orthogroup gate sensitivity was never measured, and the "
            "genome was not a stratifying variable.  A disjointness test "
            "passes on such a split anyway -- that is precisely why this "
            "check looks at the evidence and not at the rows.\nRebuild with "
            "`python -m hyppe.analysis.splits_build`, or pass "
            "allow_incomplete=True to use it knowingly."
            % (where, missing_evidence))


def guard_columns(k: pd.DataFrame) -> None:
    """Raises when either of the two column-axis keys crosses a boundary.

    `study` alone constrains nothing for the 69.98 % of studies that are
    single-sample, and 112 pairs sharing (condition, genome) crossed the
    boundary at a median expression correlation of 0.9523.  It raises rather
    than warning: a warning in a training log is indistinguishable from its
    absence.
    """
    bad = k.groupby("study")["split"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise RuntimeError("COLUMN LEAKAGE: a study in >1 part: %s"
                           % list(bad.index[:5]))
    if "condition" in k.columns:
        kw = _condition_key(k)
        bad2 = k.assign(_kw=kw.to_numpy()).groupby("_kw")["split"].nunique()
        bad2 = bad2[bad2 > 1]
        if len(bad2):
            raise RuntimeError(
                "COLUMN LEAKAGE: %d (condition, genome) keys in >1 part, "
                "e.g. %s" % (len(bad2),
                             [v.replace("\t", " | ") for v in bad2.index[:3]]))
    else:
        # A column manifest without `condition` cannot be checked on the
        # second key, so a missing column is an error and not a skipped
        # condition.
        raise RuntimeError(
            "the column manifest has no 'condition' column -- the second key "
            "of the column axis cannot be checked; rebuild the split.\n"
            "MEASURED: the split artefacts shipped in the legacy tree are "
            "exactly like this (columns: sample, genome, study, "
            "condition_category, split, is_target_strain), so the legacy "
            "`splity.wczytaj_podzial()` raises on its own output too -- "
            "checked by execution.  The artefacts predate the defect-3 fix "
            "that introduced this key.")
    for c in SPLITS:
        m = (k["split"] == c) & (k["genome"] == TARGET_STRAIN)
        if not m.any():
            raise RuntimeError("the columns of part %r do not contain %s"
                               % (c, TARGET_STRAIN))


# Build
def build(seed: int = SEED, log=print, cache: bool = True) -> dict:
    """The full, deterministic build of the split + writing the manifests."""
    t0 = time.time()
    paths.ensure_dir(SPLIT_DIR)
    log("1. loading promoters ...")
    df = load_promoters()
    log("   rows %d | unique IDs %d | orthogroups %d | genomes %d"
        % (len(df), df["ID"].nunique(), df["OG_ID"].nunique(),
           df["GENOME"].nunique()))

    log("2. MEASURING the merge threshold ...")
    thr = measure_threshold(df, seed=seed)
    log("   within orthogroup: median %.4f | identical %.3f | >=threshold %.3f"
        % (thr["within_percentiles"]["50"],
           thr["within_fraction_identical"],
           thr["fraction_orthologous_above_threshold"]))
    log("   random: p99 %.5f | p99.99 %.5f | max %.5f | >=threshold %d of %d"
        % (thr["random_percentiles"]["99"], thr["random_percentiles"]["99.99"],
           thr["random_maximum"],
           int(round(thr["fraction_random_above_threshold"]
                     * thr["n_pairs_random"])),
           thr["n_pairs_random"]))
    log("   threshold %.2f = %.1fx the 99.99th percentile of random pairs"
        % (MERGE_THRESHOLD, thr["threshold_in_multiples_of_random_p9999"]))
    gate = gate_sensitivity(df)
    for k_, v in gate.items():
        log("   sketch gate sensitivity %-10s %s/%s"
            % (k_, v["passed"], v["pairs"]))
    gate_full = full_sensitivity(df, log=log)
    log("   k-mers allowed (df<=%d): %.4f"
        % (CAP_DF, gate_full["fraction_kmers_allowed"]))
    for k_, v in gate_full["buckets"].items():
        log("   FULL sensitivity (gate+df) %-10s %d/%d = %.4f"
            % (k_, v["gate_and_df"], v["pairs"], v["sensitivity"]))

    log("3. homology clusters ...")
    extra = load_extra_merges()
    if len(extra):
        log("   pairs reported by the falsifier in previous rounds: %d"
            % len(extra))
    cluster, rep_cl = build_clusters(df, log=log, cache=cache,
                                     extra_pairs=extra)
    log("   clusters %d (from %d orthogroups), largest %d sequences"
        % (rep_cl["clusters"], rep_cl["seed_orthogroups"],
           rep_cl["largest_cluster_sequences"]))

    log("4. target per gene (for stratification) ...")
    tgt = target_per_gene()
    df = df.merge(tgt, on="ID", how="left")
    log("   without a target: %d of %d rows"
        % (int(df["target"].isna().sum()), len(df)))

    def _deal(cluster_v):
        """Cluster frame + part assignment, once per falsifier round: an
        extra merge changes the clusters and hence the strata."""
        df["cluster"] = cluster_v
        g = df.groupby("cluster")
        klr = pd.DataFrame({
            "n_seq": g.size(),
            "n_og": g["OG_ID"].nunique(),
            "n_genomes": g["GENOME"].nunique(),
            "n_p1": df.assign(
                _p1=(df["GENOME"] == TARGET_STRAIN).astype(int)
            ).groupby("cluster")["_p1"].sum(),
            "target": g["target"].mean(),
        })
        klr["dominant_genome"] = (df.groupby(["cluster", "GENOME"]).size()
                                  .reset_index(name="n")
                                  .sort_values(["cluster", "n", "GENOME"],
                                               ascending=[True, False, True])
                                  .drop_duplicates("cluster")
                                  .set_index("cluster")["GENOME"])
        cz = assign_clusters(klr, seed=seed, log=log)
        df["split"] = cz.reindex(df["cluster"]).to_numpy()
        return klr, cz

    log("5. splitting the clusters ...")
    _kl, cluster_split = _deal(cluster)

    log("5b. INDEPENDENT leakage falsifier (own sketch, own cache) ...")
    rounds = []
    for round_ in range(1, PROBE_MAX_ROUNDS + 1):
        rep_f = leakage_falsifier(df["split"], round_=round_, log=log)
        pairs = rep_f.pop("pairs")
        rounds.append(rep_f)
        log("   round %d (salt %s): cross-boundary pairs %d, above %.2f: %d"
            % (round_, rep_f["salt"], rep_f["cross_boundary_candidates"],
               MERGE_THRESHOLD, len(pairs)))
        if len(pairs) == 0:
            break
        if round_ == PROBE_MAX_ROUNDS:
            raise RuntimeError(
                "the falsifier still finds %d pairs above the threshold after "
                "%d rounds -- the closure does not converge, the split is not "
                "written" % (len(pairs), PROBE_MAX_ROUNDS))
        extra = pairs if not len(extra) else np.concatenate([extra, pairs])
        save_extra_merges(extra)
        log("   merging in %d pairs and REBUILDING the clusters ..."
            % len(extra))
        cluster, rep_cl = build_clusters(df, log=log, cache=cache,
                                         extra_pairs=extra)
        _kl, cluster_split = _deal(cluster)
    rep_cl["falsifier_rounds"] = len(rounds)

    log("5c. TVD of the genome mixture (defect 4) ...")
    conf_rows = row_confounding(df, cluster_split)
    for c in SPLITS:
        log("   %-5s TVD %.4f | null p50 %.4f | ratio %.2f"
            % (c, conf_rows["TVD_observed"][c],
               conf_rows["TVD_permutation_null_p50"][c],
               conf_rows["ratio_obs_to_null_p50"][c]))

    log("6. column split (components of study / (condition, genome)) ...")
    samples = pd.read_csv(paths.samples_metadata_path(), sep="\t", dtype=str)
    samples = samples.set_index("sample", drop=False)
    samples["component"] = column_components(samples).to_numpy()
    sample_split, rep_col = assign_columns(samples, seed=seed, log=log)
    samples["split"] = sample_split.to_numpy()
    conf = column_confounding(samples, pd.Series(samples["split"].to_numpy(),
                                                 index=samples.index))
    log("   I(genome;study)/H = %.4f | I(genome;part)/H = %.4f"
        % (conf["I_genome_study_over_H"], conf["I_genome_part_over_H"]))

    log("7. gate sensitivity on CROSS-ORTHOGROUP pairs (defect 5) ...")
    sens_cross = cross_orthogroup_sensitivity(log=log, cache=cache)
    for k_, v in sens_cross["buckets"].items():
        log("   %-10s within og %d/%d = %.4f | CROSS %d/%d = %.4f"
            % (k_, v["within_og_caught"], v["within_og_pairs"],
               v["within_og_sensitivity"], v["cross_og_caught"],
               v["cross_og_pairs"], v["cross_og_sensitivity"]))

    log("8. guards ...")
    guard_disjoint(df)
    guard_columns(samples)
    leak = leakage_description(df, log=log)
    log("   OK")

    log("9. writing ...")
    w = df[["ID", "GENE_ID", "OG_ID", "GENOME", "cluster", "split"]]
    w.to_csv(ROWS_TSV, sep="\t", index=False)
    col = samples[["sample", "genome", "study", "component",
                   "condition_category", "condition", "split"]].copy()
    col["is_target_strain"] = (col["genome"] == TARGET_STRAIN).astype(int)
    col.to_csv(COLUMNS_TSV, sep="\t", index=False)

    sizes = df.groupby("split").agg(
        sequences=("ID", "size"), genes=("ID", "nunique"),
        orthogroups=("OG_ID", "nunique"), clusters=("cluster", "nunique"))
    p1 = df[df["GENOME"] == TARGET_STRAIN].groupby("split").size()
    colr = samples.groupby("split").agg(samples=("sample", "size"),
                                        studies=("study", "nunique"),
                                        components=("component", "nunique"))
    colp1 = samples[samples["genome"] == TARGET_STRAIN].groupby("split").size()

    man = {
        "schema": "hyppe.analysis.splits_build/1 (English keys)",
        "version": 2,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "parameters": {"k": K, "fracminhash_scale": SCALE, "cap_df": CAP_DF,
                       "min_shared_kmers": MIN_SHARED,
                       "merge_threshold_jaccard": MERGE_THRESHOLD,
                       "sequence_group": GROUP, "proportions": PROPORTIONS,
                       "target_strain": TARGET_STRAIN,
                       "stratification_target":
                           "all_genes_scope/combat, row mean",
                       "row_stratum": ["size_bucket", "target_quartile",
                                       "p1_flag", "dominant_genome"],
                       "column_atom": ("connected component: study OR "
                                       "(condition, genome)"),
                       "falsifier": {"scale": PROBE_SCALE,
                                     "min_shared": PROBE_MIN_SHARED,
                                     "cap_df": PROBE_CAP_DF,
                                     "base_salt": "%#x" % PROBE_SALT,
                                     "max_rounds": PROBE_MAX_ROUNDS,
                                     "own_cache": True}},
        "input": {"sequences": os.path.basename(paths.SEQUENCES),
                  "rows": int(len(df)),
                  "unique_ids": int(df["ID"].nunique()),
                  "duplicated_ids": int(len(df) - df["ID"].nunique()),
                  "orthogroups": int(df["OG_ID"].nunique()),
                  "genomes": int(df["GENOME"].nunique())},
        "threshold_measurement": thr,
        "sketch_gate_sensitivity": gate,
        "full_candidate_stage_sensitivity": gate_full,
        "clustering": rep_cl,
        "row_sizes": {c: {k2: int(v2) for k2, v2 in sizes.loc[c].items()}
                      for c in sizes.index},
        "p1_sequences": {c: int(p1.get(c, 0)) for c in SPLITS},
        "column_sizes": {c: {k2: int(v2) for k2, v2 in colr.loc[c].items()}
                         for c in colr.index},
        "p1_samples": {c: int(colp1.get(c, 0)) for c in SPLITS},
        "study_genome_confounding": conf,
        "column_components": rep_col,
        "genome_row_confounding": conf_rows,
        "cross_orthogroup_sensitivity": sens_cross,
        "falsifier_rounds": rounds,
        "boundary_leakage_DESCRIPTION_NOT_FALSIFIER": leak,
        "p1_columns": sorted(samples.loc[samples["genome"] == TARGET_STRAIN,
                                         "sample"].tolist()),
        "files": {"rows": os.path.basename(ROWS_TSV),
                  "columns": os.path.basename(COLUMNS_TSV)},
        "design_C": ("rows of part X x columns of part X; A = columns only, "
                     "B = rows only (see baselines.py)"),
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    log("   %s\n   %s\n   %s" % (ROWS_TSV, COLUMNS_TSV, MANIFEST_JSON))
    log("done in %.1f s" % (time.time() - t0))
    return man


# Migration of the legacy Polish-schema split
#: Manifest key renames.  Explicit and asserted: `migrate_legacy_split`
#: raises on any key it does not know, so a schema change cannot pass
#: unnoticed.
_MANIFEST_RENAMES = {
    "wersja": "version", "zbudowano": "built", "ziarno": "seed",
    "parametry": "parameters", "wejscie": "input",
    "prog_pomiar": "threshold_measurement",
    "czulosc_bramki_szkicu": "sketch_gate_sensitivity",
    "czulosc_pelna_etapu_kandydatow": "full_candidate_stage_sensitivity",
    "klastrowanie": "clustering", "rozmiary_wierszy": "row_sizes",
    "sekwencji_P1": "p1_sequences", "rozmiary_kolumn": "column_sizes",
    "probek_P1": "p1_samples",
    "konfund_badanie_genom": "study_genome_confounding",
    "kolumny_skladowe": "column_components",
    "konfund_genom_wiersze": "genome_row_confounding",
    "czulosc_miedzy_ortogrupami": "cross_orthogroup_sensitivity",
    "falsyfikator_rundy": "falsifier_rounds",
    "przeciek_przez_granice_OPIS_NIE_FALSYFIKATOR":
        "boundary_leakage_DESCRIPTION_NOT_FALSIFIER",
    # v1 spelling of the same section.
    "przeciek_przez_granice":"boundary_leakage_DESCRIPTION_NOT_FALSIFIER",
    "kolumny_P1": "p1_columns", "pliki": "files", "projekt_C": "design_C",
}

_PARAM_RENAMES = {
    "k": "k", "skala_fracminhash": "fracminhash_scale", "cap_df": "cap_df",
    "min_wspolnych_kmerow": "min_shared_kmers",
    "prog_scalania_jaccard": "merge_threshold_jaccard",
    "grupa_sekwencji": "sequence_group", "proporcje": "proportions",
    "szczep_docelowy": "target_strain",
    "cel_stratyfikacji": "stratification_target",
    "strata_wierszy": "row_stratum", "atom_kolumn": "column_atom",
    "falsyfikator": "falsifier",
}

_CLUSTERING_RENAMES = {
    "postingow": "postings", "kmerow_probkowanych": "kmers_sampled",
    "kmerow_uzytych": "kmers_used", "par_wygenerowanych": "pairs_generated",
    "partycji": "partitions", "kandydatow": "candidates",
    "przestrzen_probkowana": "sampled_space",
    "wysycenie_przestrzeni": "space_saturation",
    "kandydatow_miedzy_roznymi_ortogrupami":
        "candidates_between_distinct_orthogroups",
    "roznych_sekwencji": "distinct_sequences",
    "ortogrup_ziarnowych": "seed_orthogroups",
    "par_nad_progiem": "pairs_above_threshold",
    "scalen_z_identycznosci": "merges_from_identity",
    "scalen_z_podobienstwa": "merges_from_similarity",
    "par_od_falsyfikatora": "pairs_from_falsifier",
    "scalen_od_falsyfikatora": "merges_from_falsifier",
    "klastrow": "clusters",
    "klaster_najwiekszy_sekwencji": "largest_cluster_sequences",
    "klaster_najwiekszy_ortogrup": "largest_cluster_orthogroups",
    "klastrow_jednosekwencyjnych": "single_sequence_clusters",
    "klastrow_zlozonych_z_wielu_ortogrup": "clusters_of_several_orthogroups",
    "frakcja_sekwencji_w_najwiekszym": "fraction_sequences_in_largest",
    "rund_falsyfikatora": "falsifier_rounds",
}

_CLUSTERING_RENAMES.update({
    # v1 emitted the threshold histogram with these literal keys
    "par_nad_0.02": "pairs_above_0.02", "par_nad_0.05": "pairs_above_0.05",
    "par_nad_0.10": "pairs_above_0.10", "par_nad_0.20": "pairs_above_0.20",
})

_THRESHOLD_RENAMES = {
    "k": "k",
    "n_par_w_ortogrupie": "n_pairs_within_orthogroup",
    "n_par_losowych": "n_pairs_random",
    "w_ortogrupie_srednia": "within_mean",
    "w_ortogrupie_percentyle": "within_percentiles",
    "w_ortogrupie_frakcja_identycznych": "within_fraction_identical",
    "w_ortogrupie_rozne_genomy_srednia": "within_different_genomes_mean",
    "losowe_srednia": "random_mean",
    "losowe_percentyle": "random_percentiles",
    "losowe_maksimum": "random_maximum",
    "losowe_najwieksze_5": "random_largest_5",
    "losowe_nad_0_01": "random_above_0_01",
    "frakcja_losowych_nad_progiem": "fraction_random_above_threshold",
    "frakcja_ortologicznych_nad_progiem": "fraction_orthologous_above_threshold",
    "prog": "threshold",
    "prog_w_krotnosciach_p9999_losowych":
        "threshold_in_multiples_of_random_p9999",
}

_GATE_BUCKET_RENAMES = {"par": "pairs", "przepuszczonych": "passed",
                        "czulosc": "sensitivity"}
_FULL_BUCKET_RENAMES = {"par": "pairs", "sama_bramka": "gate_only",
                        "bramka_i_df": "gate_and_df", "czulosc": "sensitivity"}
_FULL_RENAMES = {"kmerow_dozwolonych_frakcja": "fraction_kmers_allowed",
                 "kosze": "buckets"}

_COL_CONF_RENAMES = {
    "H_genom_bity": "H_genome_bits",
    "H_genom_dane_badanie_bity": "H_genome_given_study_bits",
    "I_genom_badanie_przez_H": "I_genome_study_over_H",
    "H_genom_dana_czesc_bity": "H_genome_given_part_bits",
    "I_genom_czesc_przez_H": "I_genome_part_over_H",
    "TVD_mieszanki_genomow_wobec_calosci": "TVD_genome_mixture_vs_whole",
    "badan_wielogenomowych": "multi_genome_studies",
    "badan": "studies",
    "probek_w_badaniach_wielogenomowych": "samples_in_multi_genome_studies",
    "genomow_z_mniej_niz_4_badaniami": "genomes_with_fewer_than_4_studies",
}

_LEAK_BUCKET_RENAMES = {
    "par": "pairs",
    "przecinajacych_granice": "crossing_boundary",
    "roznych_sekwencji_ewal_z_partnerem_w_train":
        "distinct_eval_sequences_with_train_partner",
    "wierszy_ewal": "eval_rows",
    "frakcja_zbioru_ewaluacyjnego": "fraction_of_eval_set",
    "uwaga": "note",
}

_ROW_SIZE_RENAMES = {"sekwencje": "sequences", "geny": "genes",
                     "ortogrupy": "orthogroups", "klastry": "clusters"}
_COL_SIZE_RENAMES = {"probki": "samples", "badania": "studies",
                     "skladowe": "components"}


def _rename_keys(d: dict, table: dict, where: str) -> dict:
    unknown = [k for k in d if k not in table]
    if unknown:
        raise KeyError("%s: unknown legacy keys %s -- extend the rename table "
                       "rather than dropping them" % (where, unknown))
    return {table[k]: v for k, v in d.items()}


def migrate_legacy_split(src_dir: str, dst_dir: str | None = None,
                         log=print) -> str:
    """Convert a legacy Polish-schema split directory to the English schema.

    Rebuilding from scratch would take hours and, worse, produce a different
    split, because the falsifier rounds accumulate merges on disk.  So the
    artefacts are converted and not re-derived: only key names change, every
    rename is in an explicit table, and an unknown key raises.
    """
    dst = paths.ensure_dir(dst_dir or SPLIT_DIR)
    rows = pd.read_csv(os.path.join(src_dir, "podzial_wierszy.tsv"), sep="\t",
                       dtype={"ID": str, "GENE_ID": str, "OG_ID": str,
                              "GENOME": str, "klaster": np.int64,
                              "czesc": str})
    rows = rows.rename(columns={"klaster": "cluster", "czesc": "split"})
    rows.to_csv(os.path.join(dst, os.path.basename(ROWS_TSV)), sep="\t",
                index=False)

    cols = pd.read_csv(os.path.join(src_dir, "podzial_kolumn.tsv"), sep="\t",
                       dtype=str)
    cols = cols.rename(columns={"skladowa": "component", "czesc": "split",
                                "czy_szczep_docelowy": "is_target_strain"})
    cols.to_csv(os.path.join(dst, os.path.basename(COLUMNS_TSV)), sep="\t",
                index=False)

    man = json.load(open(os.path.join(src_dir, "manifest.json"),
                         encoding="utf-8"))
    out = _rename_keys(man, _MANIFEST_RENAMES, "manifest")
    out["parameters"] = _rename_keys(out["parameters"], _PARAM_RENAMES,
                                     "manifest.parametry")
    fals = out["parameters"].get("falsifier") or {}
    out["parameters"]["falsifier"] = {
        {"skala": "scale", "min_wsp": "min_shared", "cap_df": "cap_df",
         "sol_bazowa": "base_salt", "rund_najwyzej": "max_rounds",
         "wlasny_cache": "own_cache"}.get(k, k): v for k, v in fals.items()}
    out["clustering"] = _rename_keys(out["clustering"], _CLUSTERING_RENAMES,
                                     "manifest.klastrowanie")
    out["row_sizes"] = {c: _rename_keys(v, _ROW_SIZE_RENAMES,
                                        "manifest.rozmiary_wierszy")
                        for c, v in out["row_sizes"].items()}
    out["column_sizes"] = {c: _rename_keys(v, _COL_SIZE_RENAMES,
                                           "manifest.rozmiary_kolumn")
                           for c, v in out["column_sizes"].items()}
    out["input"] = {
        {"sekwencje": "sequences", "wierszy": "rows",
         "ID_unikalnych": "unique_ids", "ID_zduplikowanych": "duplicated_ids",
         "ortogrup": "orthogroups", "genomow": "genomes"}.get(k, k): v
        for k, v in out["input"].items()}
    # Nested measurement payloads are renamed too: a half-translated manifest
    # leaves a reader unable to tell which half they are looking at.
    out["threshold_measurement"] = _rename_keys(
        out["threshold_measurement"], _THRESHOLD_RENAMES,
        "manifest.prog_pomiar")
    out["sketch_gate_sensitivity"] = {
        b: _rename_keys(v, _GATE_BUCKET_RENAMES,
                        "manifest.czulosc_bramki_szkicu[%s]" % b)
        for b, v in out["sketch_gate_sensitivity"].items()}
    full = _rename_keys(out["full_candidate_stage_sensitivity"], _FULL_RENAMES,
                        "manifest.czulosc_pelna_etapu_kandydatow")
    full["buckets"] = {
        b: _rename_keys(v, _FULL_BUCKET_RENAMES,
                        "manifest.czulosc_pelna...kosze[%s]" % b)
        for b, v in full["buckets"].items()}
    out["full_candidate_stage_sensitivity"] = full
    out["study_genome_confounding"] = _rename_keys(
        out["study_genome_confounding"], _COL_CONF_RENAMES,
        "manifest.konfund_badanie_genom")
    leak = out["boundary_leakage_DESCRIPTION_NOT_FALSIFIER"]
    out["boundary_leakage_DESCRIPTION_NOT_FALSIFIER"] = {
        b: (_rename_keys(v, _LEAK_BUCKET_RENAMES, "manifest.przeciek[%s]" % b)
            if isinstance(v, dict) else v)
        for b, v in leak.items()}
    out["files"] = {"rows": os.path.basename(ROWS_TSV),
                    "columns": os.path.basename(COLUMNS_TSV)}
    out["schema"] = "hyppe.analysis.splits_build/1 (English keys)"
    out["migrated_from"] = os.path.abspath(src_dir)
    out["migration_note"] = (
        "converted from the legacy Polish schema; only key names changed, "
        "every value is carried over unmodified")
    gaps = manifest_missing_evidence(out)
    out["missing_v2_evidence"] = gaps
    if gaps:
        log("   WARNING: this is a version-%s artefact and is MISSING %d of "
            "the version-2 evidence sections:" % (out.get("version"), len(gaps)))
        for g in gaps:
            log("     - %s" % g)
        log("   `load_split()` will REFUSE it unless allow_incomplete=True.")
    if "condition" not in cols.columns:
        log("   WARNING: columns.tsv has no `condition` column, so the second "
            "key of the column axis (condition, genome) cannot be checked. "
            "`guard_columns` raises on this -- as it does on the legacy "
            "artefact, verified by execution.")
    with open(os.path.join(dst, os.path.basename(MANIFEST_JSON)), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    log("migrated %s -> %s" % (src_dir, dst))
    return dst


# Reading: the only route for downstream modules
@dataclass
class Split:
    rows: pd.DataFrame             # ID, GENE_ID, OG_ID, GENOME, cluster, split
    columns: pd.DataFrame          # sample, genome, study, split
    manifest: dict

    def ids_of(self, split: str) -> np.ndarray:
        split = canonical_split(split)
        return self.rows.loc[self.rows["split"] == split, "ID"].to_numpy()

    def id_to_split(self) -> dict[str, str]:
        """Sequence ID -> canonical part.

        A function, not a relation: 950 IDs repeat in `sequences.tsv.gz` (up
        to 5x), so the check that no repeated ID got two parts is explicit.
        On the current split none did, though 2 IDs lie in two clusters --
        the cluster is not a function of the ID, but the part is.
        """
        g = self.rows.groupby("ID")["split"].nunique()
        bad = g[g > 1]
        if len(bad):
            raise RuntimeError(
                "%d IDs have two different parts, e.g. %s -- the split is not "
                "a function of the ID" % (len(bad), list(bad.index[:5])))
        return dict(zip(self.rows["ID"], self.rows["split"]))

    def id_to_cluster(self) -> dict[str, int]:
        """ID -> its smallest cluster number (2 IDs lie in two clusters).

        For the leakage guard in downstream modules: the cluster is the atom
        of the split, so it, and not OG_ID, is the right unit of control.
        """
        g = self.rows.groupby("ID")["cluster"].min()
        return {str(i): int(k) for i, k in g.items()}

    def orthogroups_of(self, split: str) -> np.ndarray:
        """Orthogroups of a part: the row key of `og_scope`.

        The guard guarantees no orthogroup occurs in two parts -- the cluster
        is the atom and the orthogroup its seed -- so og_scope inherits the
        split with no translation.
        """
        split = canonical_split(split)
        return np.unique(self.rows.loc[self.rows["split"] == split,
                                       "OG_ID"].to_numpy())

    def columns_of(self, split: str) -> np.ndarray:
        split = canonical_split(split)
        return self.columns.loc[self.columns["split"] == split,
                                "sample"].to_numpy()

    def design_c(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(rows, columns) for design C of baselines.py: a new promoter in a
        new context, both dimensions held at once."""
        return self.ids_of(split), self.columns_of(split)

    def p1_columns(self) -> np.ndarray:
        return self.columns.loc[self.columns["genome"] == TARGET_STRAIN,
                                "sample"].to_numpy()


def load_split(directory: str | None = None,
               allow_incomplete: bool = False) -> Split:
    """The only source of truth about the split for the whole project.

    Three guards, all raising, because each of these failures is silent:
      * `guard_manifest` -- the atom must be a homology cluster and there
                            must be four parts (train/val/spy/test);
      * `guard_disjoint` -- no cluster or orthogroup in two parts;
      * `guard_columns`  -- no study in two parts.

    No module draws a split on the fly: one made quietly at every run would
    be a different split at every run, and the difference leaves no trace in
    any metric except an inflated result.
    """
    d = directory or SPLIT_DIR
    tsv = os.path.join(d, os.path.basename(ROWS_TSV))
    kol = os.path.join(d, os.path.basename(COLUMNS_TSV))
    js = os.path.join(d, os.path.basename(MANIFEST_JSON))
    for p in (tsv, kol, js):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "missing %s -- run `python -m hyppe.analysis.splits_build`, "
                "or migrate a legacy split with `--migrate <old dir>`" % p)
    w = pd.read_csv(tsv, sep="\t", dtype={"ID": str, "GENE_ID": str,
                                          "OG_ID": str, "GENOME": str,
                                          "cluster": np.int64, "split": str})
    k = pd.read_csv(kol, sep="\t", dtype=str)
    man = json.load(open(js, encoding="utf-8"))
    # The guard runs at read time too.  `allow_incomplete` is an explicit
    # opt-in and never a default: a version-1 split passes every row-level
    # test and still leaks.
    guard_manifest(man, js, allow_incomplete=allow_incomplete)
    guard_disjoint(w)
    guard_columns(k)
    return Split(w, k, man)


def main():
    man = build()
    print("\n" + "=" * 70)
    print("SIZES")
    print("=" * 70)
    print("%-6s %10s %8s %10s %9s %8s %7s | %7s %8s %10s %6s"
          % ("part", "sequences", "genes", "orthogrp", "clusters", "P1",
             "P1 %", "samples", "studies", "components", "P1col"))
    for c in SPLITS:
        r = man["row_sizes"][c]
        k = man["column_sizes"][c]
        print("%-6s %10d %8d %10d %9d %8d %6.2f%% | %7d %8d %10d %6d"
              % (c, r["sequences"], r["genes"], r["orthogroups"], r["clusters"],
                 man["p1_sequences"][c],
                 100.0 * man["p1_sequences"][c] / max(r["sequences"], 1),
                 k["samples"], k["studies"], k["components"],
                 man["p1_samples"][c]))
    print()
    print("seed %d | falsifier: %d rounds, last one %d pairs above threshold"
          % (man["seed"], len(man["falsifier_rounds"]),
             man["falsifier_rounds"][-1]["cross_boundary_above_threshold"]))


if __name__ == "__main__":
    if "--candidates" in sys.argv:
        candidate_stage()
    elif "--falsifier" in sys.argv:
        _r = int(sys.argv[sys.argv.index("--falsifier") + 1])
        _probe_process(_r)
    elif "--sensitivity" in sys.argv:
        print(json.dumps(cross_orthogroup_sensitivity(cache=False), indent=2))
    elif "--migrate" in sys.argv:
        _src = sys.argv[sys.argv.index("--migrate") + 1]
        migrate_legacy_split(_src)
    else:
        main()
