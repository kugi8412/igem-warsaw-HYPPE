"""Linear probe: DNABERT vs 6-mers vs one-hot -- does the embedding enter at all.

An earlier measurement here gave a foundation model (D3) 0.2348 in a linear
probe where 6-mer composition alone gave 0.4716. This repeats it for DNABERT,
on our promoters and our split, so the decision rests on a number.

Falsifier, set before the measurement: DNABERT enters if and only if

    SCC(best DNABERT pool) > SCC(6-mers) + 2 * SE(difference)

with SE from a paired bootstrap over homology clusters rather than rows --
orthologues are not independent examples (median within-orthogroup k-mer
Jaccard 0.0633, J = 1.0 for 15.9 % of pairs) and the cluster, not the
orthogroup, is the atom of the split. Both sides are computed on the same
sequences with the same estimator, so the threshold is satisfiable by
construction.

Three controls carry the result. `constant` must give SCC ~0 (Spearman is
undefined for a constant prediction; we report 0 and say so). `permutation`
shuffles the target in the evaluation split, and every frame must fall to ~0
or the leak is in the pipe rather than in the features. `genome` is the
one-hot of the 19 strains, a confounder that hands any strain-aware model part
of the SCC for free -- which is why every frame is scored twice, the second
time against the target with the genome mean removed, those means estimated on
train only.

The target is the mean `rank` of a gene over the samples of its own genome,
averaged over its own block of the block-sparse matrix rather than over the
whole row. `rank` is descending in this release, so a high value is a weak
promoter; SCC is invariant to monotone transforms, and `--invert-target` only
flips the sign for readability.

The split is read from `hyppe.analysis.splits_build`, not from the `split`
column of `index.tsv`, which is a copy taken when the embeddings were computed
-- the probe joins on ID and reports how many rows changed part. The four
roles are disjoint: train fits the weights, val chooses lambda and nothing
else, `spy` chooses BETWEEN representations, `test` is opened once for the
report and needs an explicit `--open-test`. val and spy must differ because
the sd of the evaluation alone is 0.0064 here, and selection under a metric
that noisy once picked a result 1.62 sd above the mean.

    python -m hyppe.analysis.embedding_probe \\
        --embeddings cache/embeddings --model dnabert2 [--eval-split spy]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

from . import paths
from . import splits_build as splits
from .embeddings import ONE_HOT_ORDER, POOLS, SEED, K, _kmer_file, _shard_file

LAMBDAS = np.array([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])


# 1. The index and the features
def load_index(d: str):
    p = os.path.join(d, "index.tsv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "missing %s -- compute the embeddings first:\n"
            "    python -m hyppe.analysis.embeddings compute --out %s"
            % (p, d))
    with open(p, encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        rows = [l.rstrip("\n").split("\t") for l in fh]
    out = {k: [w[i] for w in rows] for i, k in enumerate(cols)}
    out["shard"] = np.array(out["shard"], dtype=np.int32)
    out["row"] = np.array(out["row"], dtype=np.int32)
    out["len_bp"] = np.array(out["len_bp"], dtype=np.int32)
    return out


def assemble_shards(d: str, model: str | None, pool: str | None,
                    shards: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Glue the shards into one array in index order.

    The order is checked rather than assumed: a wrong row range would permute
    the features against the target, which leaves no trace anywhere except a
    lower score.
    """
    parts, positions = [], []
    for nr in np.unique(shards):
        m = shards == nr
        p = (_kmer_file(d, int(nr)) if pool is None
             else _shard_file(d, model, pool, int(nr)))
        if not os.path.exists(p):
            raise FileNotFoundError("missing shard %s" % p)
        arr = np.load(p)
        w = rows[m]
        if w.min() < 0 or w.max() >= arr.shape[0]:
            raise IndexError("shard %d: rows %d..%d outside the array %s"
                             % (nr, w.min(), w.max(), arr.shape))
        parts.append(arr[w])
        positions.append(np.flatnonzero(m))
    out = np.empty((int(shards.shape[0]), parts[0].shape[1]),
                   dtype=parts[0].dtype)
    for c, p in zip(parts, positions):
        out[p] = c
    return out


def has_kmers_mask(d: str, shards: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """True when a sequence has at least one valid k-mer.

    A sequence of pure 'N' passes the `--min-bp` filter yet has no valid
    k-mer, so its fractions would be 0/0. Such rows drop out of all frames at
    once: dropping them from the 6-mer frame alone would stop the comparison
    being paired.
    """
    ok = np.ones(shards.shape[0], dtype=bool)
    for nr in np.unique(shards):
        m = shards == nr
        arr = np.load(_kmer_file(d, int(nr)), mmap_mode="r")
        ok[m] = np.asarray(arr[rows[m]]).sum(axis=1) > 0
    return ok


def one_hot_from_sequences(ids: list[str], path=None, group="Promotor",
                           length=800) -> np.ndarray:
    """One-hot [N, length*4] in the channel order `embeddings.ONE_HOT_ORDER`.

    Shorter sequences are padded with zeros rather than with bases: a position
    with no base is empty in all four channels instead of pretending to be an
    'A'. The channel order is whatever `embeddings` declares -- for a ridge it
    is a permutation of columns and does not move the score.
    """
    wanted = {i: n for n, i in enumerate(ids)}
    X = np.zeros((len(ids), length, 4), dtype=np.float32)
    table = np.full(256, -1, dtype=np.int16)
    for i, c in enumerate(ONE_HOT_ORDER):
        table[ord(c)] = i
    with gzip.open(paths.require(path or paths.SEQUENCES,
                                 "sequences.tsv.gz"), "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        pos = {n: i for i, n in enumerate(hdr)}
        for l in fh:
            c = l.rstrip("\n").split("\t")
            if c[pos["GROUP"]] != group:
                continue
            n = wanted.pop(c[pos["ID"]], None)
            if n is None:
                continue
            s = c[pos["SEQUENCE"]].upper()[:length]
            code = table[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
            good = code >= 0
            X[n, np.flatnonzero(good), code[good]] = 1.0
    if wanted:
        raise KeyError("%d IDs not found in the sequence file (e.g. %s)"
                       % (len(wanted), list(wanted)[:3]))
    return X.reshape(len(ids), length * 4)


# 2. The target
def load_target(scope="all_genes_scope", matrix="rank",
                cache_dir=None) -> dict[str, float]:
    """Gene ID -> mean value over the samples of its own genome; cached."""
    cache_dir = paths.ensure_dir(cache_dir or paths.CACHE)
    c = os.path.join(cache_dir, "target_%s_%s.npz" % (scope, matrix))
    if os.path.exists(c):
        d = np.load(c, allow_pickle=False)
        return dict(zip(d["ids"].tolist(), d["y"].tolist()))

    import pandas as pd
    sm = pd.read_csv(paths.samples_metadata_path("og_scope"), sep="\t",
                     dtype=str)
    genome_of = dict(zip(sm["sample"], sm["genome"]))
    p = paths.matrix_path(scope, matrix)
    ids, ys, ns = [], [], []
    t = time.time()
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        meta = [x for x in hdr if x in ("ID", "gene_id", "genome", "og_id")]
        if "genome" not in meta:
            raise ValueError("%s has no `genome` column -- a per-GENE target "
                             "needs the gene scope, not the orthogroup one" % p)
        cols = hdr[len(meta):]
        kg = np.array([genome_of.get(c, "?") for c in cols])
        i_gen, i_id = hdr.index("genome"), hdr.index("ID")
        for l in fh:
            c = l.rstrip("\n").split("\t")
            v = np.array(c[len(meta):], dtype=object)
            if v.shape[0] < kg.shape[0]:
                v = np.concatenate([v, np.array([""] * (kg.shape[0] - v.shape[0]),
                                                dtype=object)])
            m = (v[:kg.shape[0]] != "") & (kg == c[i_gen])
            if not m.any():
                continue
            ids.append(c[i_id])
            ys.append(float(np.mean(v[m].astype(np.float64))))
            ns.append(int(m.sum()))
    print("target: %d genes from %s in %.0f s (samples per gene: median %d, "
          "min %d)" % (len(ids), os.path.basename(p), time.time() - t,
                       int(np.median(ns)), int(np.min(ns))))
    np.savez(c, ids=np.array(ids), y=np.array(ys, dtype=np.float64))
    return dict(zip(ids, ys))


# 3. Ridge and metrics
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0                     # a constant prediction -> rho undefined
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class Ridge:
    """Ridge by eigendecomposition of X'X: one decomposition, the whole grid.

    Standardisation is computed on the training set only, and a column of
    zero variance gets scale 1 rather than 0 so the division cannot silently
    produce NaN.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.dead = int((sd <= 1e-12).sum())
        Z = (X - self.mu) / self.sd
        self.y0 = float(y.mean())
        yc = y - self.y0
        A = Z.T @ Z
        b = Z.T @ yc
        w, V = np.linalg.eigh(A)
        self.w = np.clip(w, 0.0, None)
        self.Vb = V.T @ b
        self.V = V

    def weights(self, lam: float) -> np.ndarray:
        return self.V @ (self.Vb / (self.w + lam))

    def predict(self, X: np.ndarray, lam: float) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return Z @ self.weights(lam) + self.y0


def fit_and_score(Xtr, ytr, Xva, yva, Xev, yev, lambdas=LAMBDAS):
    """Lambda is chosen on validation, the numbers fall on the eval split.

    With `Xva is None` lambda is chosen on the evaluation split instead; the
    result is then optimistic and is flagged `lambda_from_eval=True` so it
    cannot be quoted as clean.
    """
    r = Ridge(Xtr, ytr)
    Xsel, ysel, from_eval = ((Xva, yva, False) if Xva is not None
                             else (Xev, yev, True))
    scores = [(spearman(ysel, r.predict(Xsel, l)), l) for l in lambdas]
    best_scc, lam = max(scores, key=lambda t: t[0])
    p = r.predict(Xev, lam)
    return {
        "scc": spearman(yev, p),
        "pcc": pearson(yev, p),
        "lambda": lam,
        "lambda_at_grid_edge": bool(lam in (lambdas[0], lambdas[-1])),
        "lambda_from_eval": from_eval,
        "selection_scc": best_scc,
        "dead_columns": r.dead,
        "pred": p,
    }


def bootstrap_by_group(y, p1, p2, groups, n=200, seed=SEED):
    """SE of the SCC difference, resampling by group: paired, same resample."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    buckets = [np.flatnonzero(inv == i) for i in range(len(ug))]
    d, s1, s2 = [], [], []
    for _ in range(n):
        pick = rng.integers(0, len(ug), size=len(ug))
        idx = np.concatenate([buckets[i] for i in pick])
        a, b = spearman(y[idx], p1[idx]), spearman(y[idx], p2[idx])
        s1.append(a)
        s2.append(b)
        d.append(a - b)
    return (float(np.std(d, ddof=1)), float(np.std(s1, ddof=1)),
            float(np.std(s2, ddof=1)))


# 4. The probe
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hyppe.analysis.embedding_probe",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--embeddings",
                    default=os.path.join(paths.CACHE, "embeddings"))
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--train-split", default="train", dest="split_tr",
                    help="the part used to fit the weights (default train)")
    ap.add_argument("--selection-split", default="val", dest="split_va",
                    help="the part used to choose LAMBDA (default val); "
                         "'none' = choose on the evaluation split (an "
                         "optimistic result, reported as such)")
    ap.add_argument("--eval-split", default="spy", dest="split_ev",
                    help="the part the reported numbers fall on.  Default "
                         "`spy` -- the part for choosing BETWEEN frames.  "
                         "`test` requires --open-test")
    ap.add_argument("--open-test", action="store_true", dest="open_test",
                    help="EXPLICIT consent to use the `test` part.  Without "
                         "this flag `--eval-split test` is an error rather "
                         "than a silent spending of the one set that may be "
                         "opened once")
    ap.add_argument("--split", default=None,
                    help="directory of the cluster split (default: "
                         "data/splits/homology_clusters/)")
    ap.add_argument("--allow-incomplete-split", action="store_true",
                    dest="allow_incomplete_split",
                    help="accept a split built before the version-2 fixes "
                         "(EXPLICIT opt-in)")
    ap.add_argument("--n-train", type=int, default=40000, dest="n_tr",
                    help="subsample of the training set (0 = everything); a "
                         "ridge on 4096 features is fond of memory")
    ap.add_argument("--n-eval", type=int, default=0, dest="n_ev")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-onehot", action="store_true", dest="no_onehot")
    ap.add_argument("--invert-target", action="store_true", dest="invert")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--target-scope", default="all_genes_scope")
    ap.add_argument("--target-matrix", default="rank")
    a = ap.parse_args(argv)

    # `test` is gated before any data is read: a guard further down would
    # already have read it by the time an hour-long run was interrupted.
    # Alias translation stays in splits_build.py, in one place.
    a.split_tr = splits.canonical_split(a.split_tr)
    a.split_ev = splits.canonical_split(a.split_ev)
    if a.split_va != "none":
        a.split_va = splits.canonical_split(a.split_va)
    used = [a.split_tr, a.split_ev] + ([] if a.split_va == "none"
                                       else [a.split_va])
    if "test" in used and not a.open_test:
        print("ERROR: the `test` part was requested as %s, but without "
              "--open-test.\n"
              "  `test` is opened ONCE, at the end, for the report.  Frame "
              "comparison is what `spy` is for (--eval-split spy, the default)."
              % ", ".join(n for n, v in (("training", a.split_tr),
                                         ("lambda selection", a.split_va),
                                         ("evaluation", a.split_ev))
                          if v == "test"),
              file=sys.stderr)
        return 3
    if len(set(used)) != len(used):
        print("ERROR: the parts must be disjoint, they are %s -- fitting and "
              "scoring on one part measures nothing" % used, file=sys.stderr)
        return 3

    rng = np.random.default_rng(a.seed)
    idx = load_index(a.embeddings)
    man_p = os.path.join(a.embeddings, "manifest_%s.json" % a.model)
    man = (json.load(open(man_p, encoding="utf-8"))
           if os.path.exists(man_p) else {})

    print("=" * 78)
    print("LINEAR PROBE -- DNABERT (%s) vs %d-mers vs one-hot" % (a.model, K))
    print("FALSIFIER (set before the measurement): DNABERT enters only when")
    print("   SCC(best pool) > SCC(%d-mers) + 2*SE(difference),  SE by a PAIRED"
          % K)
    print("   bootstrap over HOMOLOGY CLUSTERS (%d replicates)." % a.bootstrap)
    print("=" * 78)
    if man:
        print("manifest: model=%s, licence=%s, n=%s, chunking=%s"
              % (man.get("model"), man.get("model_info", {}).get("licence"),
                 man.get("n_sequences"), man.get("chunking")))

    y_map = load_target(a.target_scope, a.target_matrix)
    has_target = np.array([i in y_map for i in idx["ID"]])
    has_km = has_kmers_mask(a.embeddings, idx["shard"], idx["row"])
    if (~has_km).any():
        print("dropped from ALL frames: %d sequences with not one valid "
              "%d-mer (all 'N')" % (int((~has_km).sum()), K))
    has_target = has_target & has_km
    print("sequences in the index %d, with a target %d (%.1f %%)"
          % (len(has_target), int(has_target.sum()), 100 * has_target.mean()))
    if has_target.sum() < 1000:
        raise RuntimeError("fewer than 1000 sequences have a target -- the "
                           "sequence IDs do not match the matrix IDs; check "
                           "the scope (%s)" % a.target_scope)

    sp = splits.load_split(a.split, allow_incomplete=a.allow_incomplete_split)
    id_split = sp.id_to_split()                  # cluster-based / 4 parts
    id_cluster = sp.id_to_cluster()
    missing = [i for i in idx["ID"] if i not in id_split]
    if missing:
        raise KeyError(
            "%d index IDs are not in the cluster split (e.g. %s).  The "
            "embeddings were computed on a different sequence file or a "
            "different group from the split -- not something one may "
            "substitute around." % (len(missing), missing[:3]))
    split = np.array([id_split[i] for i in idx["ID"]])
    cluster = np.array([id_cluster[i] for i in idx["ID"]])
    old = np.array(idx.get("split", [""] * len(split)))
    print("split      : cluster-based, 4 parts, %s"
          % os.path.relpath(a.split or splits.SPLIT_DIR, paths.ROOT))
    print("  roles    : train=fitting | val=lambda | spy=frame choice | "
          "test=report (once)")
    print("  counts in this index: %s"
          % {c: int((split == c).sum()) for c in splits.SPLITS})
    if old.size and set(old) - {""}:
        agree = int((old == split).sum())
        print("  the `split` column in index.tsv comes from a DIFFERENT split "
              "(%s): %d of %d agree (%.1f %%).  The cluster split is the one "
              "used." % ("/".join(sorted(set(old))), agree, len(split),
                         100 * agree / len(split)))
    genome = np.array(idx["GENOME"])
    og = np.array(idx["OG_ID"])
    y_all = np.array([y_map.get(i, np.nan) for i in idx["ID"]],
                     dtype=np.float64)
    if a.invert:
        y_all = -y_all

    def pick(name, n_max):
        m = (split == name) & has_target
        w = np.flatnonzero(m)
        if w.size == 0:
            raise ValueError("split %r is empty" % name)
        if n_max and w.size > n_max:
            w = np.sort(rng.choice(w, size=n_max, replace=False))
        return w

    i_tr = pick(a.split_tr, a.n_tr)
    i_ev = pick(a.split_ev, a.n_ev)
    i_va = None if a.split_va == "none" else pick(a.split_va, 20000)
    # The leakage guard is on the cluster, the atom of the split. An
    # orthogroup is only the cluster's seed, so a split by OG_ID alone passes
    # the orthogroup test and still lets paralogues through.
    for name, ii in (("evaluation", i_ev),) + (() if i_va is None
                                               else (("lambda selection", i_va),)):
        shared = set(cluster[i_tr]) & set(cluster[ii])
        if shared:
            raise RuntimeError(
                "%d homology clusters occur both in training and in %s (e.g. "
                "%s) -- the split leaks"
                % (len(shared), name, list(shared)[:3]))
        shared_og = set(og[i_tr]) & set(og[ii])
        if shared_og:
            raise RuntimeError(
                "%d orthogroups occur both in training and in %s (e.g. %s)"
                % (len(shared_og), name, list(shared_og)[:3]))
    print("fitting   %-5s n=%d" % (a.split_tr, len(i_tr)))
    print("lambda    %-5s n=%s" % (a.split_va,
                                   "none" if i_va is None else len(i_va)))
    print("EVAL      %-5s n=%d  (orthogroups %d, clusters %d)  <-- every "
          "number below falls here"
          % (a.split_ev, len(i_ev), len(set(og[i_ev])),
             len(set(cluster[i_ev]))))

    # The target and its genome-free variant
    y_tr, y_ev = y_all[i_tr], y_all[i_ev]
    y_va = None if i_va is None else y_all[i_va]
    genome_mean = {g: y_all[i_tr][genome[i_tr] == g].mean()
                   for g in np.unique(genome[i_tr])}
    glob = float(y_tr.mean())

    def subtract(ii):
        return np.array([genome_mean.get(g, glob) for g in genome[ii]])

    yr_tr, yr_ev = y_tr - subtract(i_tr), y_ev - subtract(i_ev)
    yr_va = None if i_va is None else y_va - subtract(i_va)
    print("target: mean `%s` per gene | sd %.4f | after removing the genome "
          "mean sd %.4f (the genome explains %.1f %% of the variance)"
          % (a.target_matrix, y_ev.std(), yr_ev.std(),
             100 * (1 - yr_ev.var() / y_ev.var())))

    # The feature frames
    def features(name):
        if name.startswith("dnabert_"):
            p = name.split("_", 1)[1]

            def f(ii):
                return assemble_shards(a.embeddings, a.model, p,
                                       idx["shard"][ii], idx["row"][ii]
                                       ).astype(np.float64)
        elif name == "kmer%d" % K:
            def f(ii):
                c = assemble_shards(a.embeddings, None, None, idx["shard"][ii],
                                    idx["row"][ii]).astype(np.float64)
                s = c.sum(axis=1, keepdims=True)
                if (s == 0).any():
                    raise ValueError("a sequence with not one valid %d-mer" % K)
                return c / s                      # fractions, not counts
        elif name == "onehot":
            def f(ii):
                return one_hot_from_sequences(
                    [idx["ID"][k] for k in ii]).astype(np.float64)
        elif name == "genome":
            ug = sorted(set(genome))
            mp = {g: i for i, g in enumerate(ug)}

            def f(ii):
                X = np.zeros((len(ii), len(ug)))
                X[np.arange(len(ii)), [mp[g] for g in genome[ii]]] = 1.0
                return X
        else:
            raise ValueError(name)
        return f

    frames = ["dnabert_%s" % p for p in POOLS] + ["kmer%d" % K, "genome"]
    if not a.no_onehot:
        frames.append("onehot")

    results = {}
    for name in frames:
        t = time.time()
        try:
            f = features(name)
            Xtr, Xev = f(i_tr), f(i_ev)
            Xva = None if i_va is None else f(i_va)
        except FileNotFoundError as e:
            print("  %-16s SKIPPED: %s" % (name, e))
            continue
        r = fit_and_score(Xtr, y_tr, Xva, y_va, Xev, y_ev)
        rr = fit_and_score(Xtr, yr_tr, Xva, yr_va, Xev, yr_ev)
        r["scc_without_genome"] = rr["scc"]
        r["pred_without_genome"] = rr["pred"]
        r["d"] = Xtr.shape[1]
        r["time"] = time.time() - t
        results[name] = r
        print("  %-16s d=%5d  SCC %.4f  PCC %.4f  (without genome SCC %.4f)  "
              "lambda %.0e%s  %.0f s"
              % (name, r["d"], r["scc"], r["pcc"], r["scc_without_genome"],
                 r["lambda"], " AT THE GRID EDGE!" if r["lambda_at_grid_edge"]
                 else "", r["time"]))

    # Controls
    print("-" * 78)
    print("CONTROLS")
    y_perm = y_ev.copy()
    rng.shuffle(y_perm)
    for name, r in results.items():
        print("  target permutation, %-14s SCC %+.4f  (should be ~0)"
              % (name, spearman(y_perm, r["pred"])))
    print("  constant prediction            SCC %+.4f  (0 by definition, rho "
          "undefined)" % spearman(y_ev, np.full_like(y_ev, y_ev.mean())))

    # Verdict
    print("=" * 78)
    dna = [n for n in results if n.startswith("dnabert_")]
    km = "kmer%d" % K
    if not dna or km not in results:
        print("VERDICT: undecidable -- a frame is missing (%s / %s)"
              % (dna, km))
        return 2
    best = max(dna, key=lambda n: results[n]["scc"])
    # Resampling by cluster, not by orthogroup: a bootstrap over orthogroups
    # would count two orthogroups of one cluster as two observations,
    # understating the SE and making the falsifier easier to pass.
    se_d, se1, se2 = bootstrap_by_group(
        y_ev, results[best]["pred"], results[km]["pred"], cluster[i_ev],
        n=a.bootstrap, seed=a.seed)
    difference = results[best]["scc"] - results[km]["scc"]
    threshold = 2 * se_d
    print("best DNABERT pool      : %-14s SCC %.4f (SE %.4f)"
          % (best, results[best]["scc"], se1))
    print("%d-mers                 : %-14s SCC %.4f (SE %.4f)"
          % (K, km, results[km]["scc"], se2))
    print("difference             : %+.4f  |  2*SE(difference) = %.4f  "
          "(bootstrap over %d homology clusters, part `%s`)"
          % (difference, threshold, len(set(cluster[i_ev])), a.split_ev))
    passed = difference > threshold
    print("VERDICT: DNABERT %s (falsifier: difference > 2*SE)"
          % ("ENTERS" if passed else "DOES NOT ENTER"))
    if not passed and difference < -threshold:
        print("  ...and this is not a tie: the 6-mers are SIGNIFICANTLY "
              "better, just as with D3 (0.2348 against 0.4716).")

    # The file name carries the evaluation part: a result on `spy` and one on
    # `test` are different numbers and neither may overwrite the other.
    out = os.path.join(a.embeddings,
                       "probe_%s_%s.json" % (a.model, a.split_ev))
    json.dump({n: {k: v for k, v in r.items() if not k.startswith("pred")}
               for n, r in results.items()} |
              {"_verdict": {"best_pool": best, "difference": difference,
                            "threshold_2SE": threshold, "passed": bool(passed),
                            "n_eval": len(i_ev), "seed": a.seed,
                            "train_split": a.split_tr,
                            "selection_split": a.split_va,
                            "eval_split": a.split_ev,
                            "split": "cluster (hyppe/analysis/splits_build.py)",
                            "split_built": sp.manifest.get("built"),
                            "bootstrap_unit": "homology_cluster"}},
              open(out, "w", encoding="utf-8"), indent=1, default=float)
    print("written: %s" % out)
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
