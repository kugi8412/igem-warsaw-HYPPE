"""Falsify the split from `splits_build.py` by independent measurement.

The point is to measure the split again rather than read it back.

The working assumption is that the split leaks, and this module looks for the
proof.  It has to exist alongside `splits_build.leakage_description`, because
that function loads the same candidate cache the clusters were made from: its
"0 pairs with J >= 0.05 across the boundary" is true by construction, since a
pair the candidate stage never generated is invisible to the merging and to
the control at once.

Candidates are therefore generated independently and more widely than in the
build: the same FracMinHash 1/16 sketch, but with no df cut at all,
MIN_SHARED = 1, and test x train pairs ranked top-M by shared k-mers.  How
much the cut costs is printed by section 0 from the live
`splits_build.CAP_DF` rather than quoted here, because the cap has already
moved once (100 -> 1000) and a figure measured under the old one would read
as if it still held.  The measure is identical to the build's -- exact Jaccard
of full forward 11-mer sets, `splits_build.jaccard` -- because comparing
against a different measure would compare two different quantities.

The positive control, without which the result is void, is the same
measurement through a random row split: an audit that cannot tell a random
split from a cluster split is itself broken.

Section 5 keys repeated conditions on (condition, genome), and two copies of
the sample table are in circulation whose `condition` column differs in 2300
of 2332 rows.  The release copy (20 columns) gives 91 keys and 290 samples
across the boundary; the legacy copy (18 columns) gives 56 and 153, and those
are the numbers quoted in the report.  This module defaults to the release
copy and `--metadata <path>` reproduces the old log; the verdict is FAILED
either way (0.124 and 0.066 of the panel, both above the 0.05 threshold), so
only the magnitude moves.  Every run prints the path, the column count and
the md5 of the table it used.

Run with:
  python -m hyppe.analysis.splits_audit
"""
from __future__ import annotations

import argparse
import hashlib
import io as _io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import paths
from . import splits_build as sb

AUDIT_SEED = 20260911
TOP_M = 20                  # how many best candidates are computed exactly
N_PERM = 200                # permutations in the part-composition test
POSTINGS_CACHE = os.path.join(paths.CACHE, "splits", "audit_postings.npz")

_t0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - _t0), *a, flush=True)


def header(s):
    print("\n" + "=" * 78, flush=True)
    print(s, flush=True)
    print("=" * 78, flush=True)


VERDICTS: list[dict] = []


def verdict(name, result, threshold, ok, source):
    VERDICTS.append({"name": name, "result": result, "threshold": threshold,
                     "verdict": "PASSED" if ok else "FAILED", "source": source})
    print("  -> %-7s %s\n     result:    %s\n     threshold: %s"
          % ("PASSED" if ok else "FAILED", name, result, threshold), flush=True)


# Input
def load(split_dir: str | None = None
         ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    df = sb.load_promoters()
    log("sequences: %d rows" % len(df))
    d = split_dir or sb.SPLIT_DIR
    w = pd.read_csv(paths.require(os.path.join(d, os.path.basename(sb.ROWS_TSV)),
                                  "the split rows table"),
                    sep="\t", dtype={"cluster": np.int64})
    if len(w) != len(df):
        raise RuntimeError("rows.tsv has %d rows, the sequences %d"
                           % (len(w), len(df)))
    # The key is the row number (ID is not unique: 1051 duplicates), so the
    # order has to match row by row, and that is checked rather than assumed.
    bad = int((w["ID"].to_numpy() != df["ID"].to_numpy()).sum())
    if bad:
        raise RuntimeError("the row order has drifted: %d mismatched IDs" % bad)
    bad = int((w["OG_ID"].to_numpy() != df["OG_ID"].to_numpy()).sum())
    if bad:
        raise RuntimeError("OG_ID has drifted in %d rows" % bad)
    log("rows.tsv: %d rows, consistent in ID and OG_ID" % len(w))
    distinct, assign = sb._distinct_sequences(df["SEQ"].tolist())
    log("distinct sequences: %d" % len(distinct))
    return df, w, distinct, assign


# An independent index of sampled k-mers
def postings(distinct: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(PI sorted by k-mer, bounds, dfk).  Cached, because it takes 30 s.

    The index is a function of `sb.K` and `sb.SCALE` as well as of the corpus,
    and the file name carries neither, so both are stored beside `n` and
    compared; a cache written before they were is treated as stale rather than
    reused under values it was not built with.  `sb.CAP_DF` is not among them:
    it is applied when querying `dfk`, not when building the index.
    """
    if os.path.exists(POSTINGS_CACHE):
        z = np.load(POSTINGS_CACHE)
        if (int(z["n"]) == len(distinct) and "k" in z and "scale" in z
                and int(z["k"]) == sb.K and int(z["scale"]) == sb.SCALE):
            log("postings from cache: %s" % POSTINGS_CACHE)
            return z["PI"], z["bounds"], z["dfk"]
        log("postings cache ignored (built under other parameters): %s"
            % POSTINGS_CACHE)
    h_threshold = np.uint64((1 << 64) // sb.SCALE)
    PK, PI = [], []
    for j, s in enumerate(distinct):
        full = sb.kmers(s)
        kk = full[sb.mix(full) < h_threshold] if full.shape[0] else full
        if kk.shape[0]:
            PK.append(kk)
            PI.append(np.full(kk.shape[0], j, dtype=np.int32))
    PK = np.concatenate(PK)
    PI = np.concatenate(PI)
    o = np.argsort(PK, kind="stable")
    PK, PI = PK[o], PI[o]
    bounds = np.flatnonzero(np.concatenate([[True], PK[1:] != PK[:-1], [True]]))
    dfk = np.diff(bounds)
    paths.ensure_dir(os.path.dirname(POSTINGS_CACHE))
    np.savez_compressed(POSTINGS_CACHE, PI=PI, bounds=bounds, dfk=dfk,
                        n=np.int64(len(distinct)), k=np.int64(sb.K),
                        scale=np.int64(sb.SCALE))
    log("postings computed: %d postings, %d k-mers"
        % (PI.shape[0], dfk.shape[0]))
    return PI, bounds, dfk


def posting_slot(bounds: np.ndarray, n_post: int) -> np.ndarray:
    """The k-mer number of each posting (postings are sorted by k-mer)."""
    s = np.zeros(n_post, dtype=np.int32)
    s[bounds[1:-1]] = 1
    return np.cumsum(s, dtype=np.int32)


def max_similarity(PI, bounds, dfk, slot, full_kmers,
                   source_mask: np.ndarray, target_mask: np.ndarray,
                   distinct: list[str], cap_df: int | None, min_shared: int,
                   label: str, top_m: int = TOP_M,
                   return_partner: bool = False):
    """For each distinct sequence in `target`, the maximum exact Jaccard
    against a sequence in `source`.

    Returns (indices of the target's distinct sequences, max J).  The Jaccard
    comes from `splits_build.jaccard` on full 11-mer sets, the same measure
    as the cluster build uses.
    """
    t = time.time()
    nk = dfk.shape[0]
    allowed = np.ones(nk, dtype=bool) if cap_df is None else (dfk <= cap_df)

    # Source postings, grouped by k-mer.
    mz = source_mask[PI] & allowed[slot]
    iz = np.flatnonzero(mz)
    PI_s = PI[iz]
    slot_s = slot[iz]
    count_s = np.bincount(slot_s, minlength=nk).astype(np.int64)
    bounds_s = np.concatenate([[0], np.cumsum(count_s)])

    # Target postings, grouped by sequence.
    mc = target_mask[PI] & allowed[slot]
    ic = np.flatnonzero(mc)
    PI_t = PI[ic]
    slot_t = slot[ic]
    o = np.argsort(PI_t, kind="stable")
    PI_t = PI_t[o]
    slot_t = slot_t[o]
    gr = np.flatnonzero(np.concatenate([[True], PI_t[1:] != PI_t[:-1], [True]]))
    targets = PI_t[gr[:-1]]

    # Stage 1: candidates by the number of shared sampled k-mers.
    pair_t, pair_s = [], []
    for q in range(targets.shape[0]):
        sl = slot_t[gr[q]:gr[q + 1]]
        chunks = [PI_s[bounds_s[x]:bounds_s[x + 1]] for x in sl]
        if not chunks:
            continue
        v = np.concatenate(chunks)
        if v.shape[0] == 0:
            continue
        u, c = np.unique(v, return_counts=True)
        m = c >= min_shared
        if not m.any():
            continue
        u, c = u[m], c[m]
        if u.shape[0] > top_m:
            sel = np.argpartition(-c, top_m)[:top_m]
            u = u[sel]
        pair_t.append(np.full(u.shape[0], targets[q], dtype=np.int32))
        pair_s.append(u.astype(np.int32))
    if pair_t:
        pair_t = np.concatenate(pair_t)
        pair_s = np.concatenate(pair_s)
    else:
        pair_t = np.zeros(0, np.int32)
        pair_s = np.zeros(0, np.int32)
    log("   [%s] targets %d | pairs for exact evaluation %d (%.1f s)"
        % (label, targets.shape[0], pair_t.shape[0], time.time() - t))

    # Stage 2: exact Jaccard, grouped by source so k-mers are computed once.
    maxJ = np.zeros(targets.shape[0], dtype=np.float64)
    arg = np.full(targets.shape[0], -1, dtype=np.int64)
    pos = {int(c): i for i, c in enumerate(targets)}
    if pair_t.shape[0]:
        o = np.argsort(pair_s, kind="stable")
        pair_t, pair_s = pair_t[o], pair_s[o]
        gr2 = np.flatnonzero(np.concatenate([[True],
                                             pair_s[1:] != pair_s[:-1], [True]]))
        for b in range(gr2.shape[0] - 1):
            sid = int(pair_s[gr2[b]])
            ks = sb.kmers(distinct[sid])
            for tid in pair_t[gr2[b]:gr2[b + 1]]:
                J = sb.jaccard(ks, full_kmers[int(tid)])
                i = pos[int(tid)]
                if J > maxJ[i]:
                    maxJ[i] = J
                    arg[i] = sid
    log("   [%s] done (%.1f s)" % (label, time.time() - t))
    if return_partner:
        return targets, maxJ, arg
    return targets, maxJ


def describe(maxJ: np.ndarray, weights: np.ndarray, name: str) -> dict:
    """Quantiles of the distribution, weighted by rows: a row is the unit of
    evaluation."""
    o = np.argsort(maxJ)
    v, w = maxJ[o], weights[o].astype(np.float64)
    cw = np.cumsum(w) / w.sum()

    def qt(p):
        return float(v[np.searchsorted(cw, p / 100.0)])

    d = {"n_distinct": int(maxJ.shape[0]), "n_rows": int(weights.sum()),
         "p50": qt(50), "p90": qt(90), "p99": qt(99), "p99.9": qt(99.9),
         "max": float(maxJ.max()) if maxJ.size else 0.0,
         "mean": float(np.average(maxJ, weights=weights))}
    for th in (0.05, 0.10, 0.30, 0.999):
        m = maxJ >= th
        d["rows_above_%.3f" % th] = int(weights[m].sum())
        d["fraction_rows_above_%.3f" % th] = float(weights[m].sum()
                                                   / weights.sum())
    print("  %s: p50 %.4f | p90 %.4f | p99 %.4f | p99.9 %.4f | max %.4f"
          % (name, d["p50"], d["p90"], d["p99"], d["p99.9"], d["max"]),
          flush=True)
    print("     rows above 0.05: %d (%.4f) | above 0.30: %d | identical: %d"
          % (d["rows_above_0.050"], d["fraction_rows_above_0.050"],
             d["rows_above_0.300"], d["rows_above_0.999"]), flush=True)
    return d


# Main
def _md5(path: str) -> str:
    h = hashlib.md5()
    with _io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metadata", default=None,
                    help="sample table for section 5.  Default: the release "
                         "copy.  Point it at the legacy 18-column copy to "
                         "reproduce the 56/153 in the report; see the module "
                         "docstring for the measured difference.")
    ap.add_argument("--split", default=None,
                    help="split directory (default: the cluster split)")
    args = ap.parse_args([] if argv is None else argv)

    results: dict = {}
    df, w, distinct, assign = load(args.split)
    n_distinct = len(distinct)
    cz = w["split"].to_numpy()
    PI, bounds, dfk = postings(distinct)
    slot = posting_slot(bounds, PI.shape[0])

    header("0. BLIND SPOT OF THE CANDIDATE STAGE "
           "(why the manifest's 'zero' is true by construction)")
    over = dfk > sb.CAP_DF
    fr_post = float(dfk[over].sum() / dfk.sum())
    print("  sampled k-mers %d | with df > %d: %d (%.4f)"
          % (dfk.shape[0], sb.CAP_DF, int(over.sum()), float(over.mean())))
    print("  postings inside the CUT k-mers: %d of %d = %.4f"
          % (int(dfk[over].sum()), int(dfk.sum()), fr_post))
    print("  df: max %d | p99 %d | p99.9 %d"
          % (dfk.max(), np.percentile(dfk, 99), np.percentile(dfk, 99.9)))
    print("  leakage_description reads THE SAME cache that built the clusters,")
    print("  so ungenerated pairs are invisible to BOTH.")
    results["blind_spot"] = {"kmers_cut": int(over.sum()),
                             "fraction_kmers": float(over.mean()),
                             "fraction_postings_cut": fr_post,
                             "df_max": int(dfk.max())}

    header("1. HOMOLOGY LEAKAGE: max J(test -> train), same measure as the build")
    m_test = np.zeros(n_distinct, dtype=bool)
    m_train = np.zeros(n_distinct, dtype=bool)
    m_test[assign[cz == "test"]] = True
    m_train[assign[cz == "train"]] = True
    log("distinct sequences: test %d | train %d | in both %d"
        % (int(m_test.sum()), int(m_train.sum()),
           int((m_test & m_train).sum())))

    full: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(m_test):
        full[int(i)] = sb.kmers(distinct[int(i)])
    log("full target k-mers computed (%d sequences)" % len(full))

    weights_test = np.bincount(assign[cz == "test"], minlength=n_distinct)

    print("\n  (a) candidates AS IN THE BUILD (cap_df=%d, min_shared=%d)"
          % (sb.CAP_DF, sb.MIN_SHARED))
    c1, J1 = max_similarity(PI, bounds, dfk, slot, full, m_train, m_test,
                            distinct, sb.CAP_DF, sb.MIN_SHARED, "build")
    d1 = describe(J1, weights_test[c1], "test->train [build regime]")

    print("\n  (b) candidates INDEPENDENTLY (NO df cap, min_shared=1)")
    c2, J2 = max_similarity(PI, bounds, dfk, slot, full, m_train, m_test,
                            distinct, None, 1, "independent")
    d2 = describe(J2, weights_test[c2], "test->train [independent]")

    results["max_J_test_train_build_regime"] = d1
    results["max_J_test_train_independent"] = d2

    # Scale control: the same Jaccard on within-orthogroup pairs.
    print("\n  (c) scale control -- pairs in THE SAME orthogroup, fresh measurement")
    rng = np.random.default_rng(AUDIT_SEED + 77)
    og = df["OG_ID"].to_numpy()
    by_og: dict[str, list[int]] = {}
    for i, o in enumerate(og):
        by_og.setdefault(o, []).append(i)
    multi = [v for v in by_og.values() if len(v) >= 2]
    Jo = []
    for _ in range(2000):
        grp = multi[int(rng.integers(len(multi)))]
        i, j = rng.choice(len(grp), size=2, replace=False)
        Jo.append(sb.jaccard(sb.kmers(df["SEQ"].iloc[grp[int(i)]]),
                             sb.kmers(df["SEQ"].iloc[grp[int(j)]])))
    Jo = np.asarray(Jo)
    print("     within orthogroup (n=2000): median %.4f | p75 %.4f | J=1.0 in %.4f"
          % (np.median(Jo), np.percentile(Jo, 75), float((Jo >= 0.999).mean())))
    results["within_orthogroup_control"] = {
        "n": 2000, "median": float(np.median(Jo)),
        "p75": float(np.percentile(Jo, 75)),
        "fraction_J1": float((Jo >= 0.999).mean())}
    # The identity control converted onto the same scale.
    for p, nm in ((0.655, "orthologues at 0.655 identity"),
                  (0.252, "random pairs at 0.252 identity")):
        jp = p ** sb.K / (2 - p ** sb.K)
        print("     %s  ->  J(k=11) ~ %.5f  (substitution model p^k/(2-p^k))"
              % (nm, jp))
        results.setdefault("identity_conversion", {})[nm] = float(jp)

    # (d) Sensitivity of the audit itself: does the top-M cut lose pairs?
    print("\n  (d) control of my own audit: top-M = 20 against 100")
    c2b, J2b = max_similarity(PI, bounds, dfk, slot, full, m_train, m_test,
                              distinct, None, 1, "top100", top_m=100)
    d2b = describe(J2b, weights_test[c2b],
                   "test->train [independent, top-M=100]")
    results["max_J_test_train_top100"] = d2b
    if d2b["rows_above_0.050"] > d2["rows_above_0.050"]:
        print("     WARNING: top-M = 20 LOSES pairs.  The number in (b) is a "
              "LOWER bound on the leakage.")

    # (e) Diagnosis: did the build's candidate stage see these pairs?
    print("\n  (e) pairs above the threshold -- the cause")
    ogv = df["OG_ID"].to_numpy()
    klv = w["cluster"].to_numpy()
    genv = df["GENOME"].to_numpy()
    seq_desc: dict[int, tuple] = {}
    for i in range(len(df)):
        j = int(assign[i])
        if j not in seq_desc:
            seq_desc[j] = (ogv[i], int(klv[i]), genv[i], cz[i])
    z = np.load(sb.CACHE_PATH())
    A, B, Jc = z["A"], z["B"], z["J"]
    code = A.astype(np.int64) * n_distinct + B.astype(np.int64)
    o = np.argsort(code)
    code_s = code[o]
    J_s = Jc[o]
    detail = []
    _, _, arg2 = max_similarity(PI, bounds, dfk, slot, full, m_train, m_test,
                                distinct, None, 1, "partners", top_m=100,
                                return_partner=True)
    for i in np.flatnonzero(J2b >= sb.MERGE_THRESHOLD):
        t_ = int(c2b[i])
        s_ = int(arg2[i])
        J_ = float(J2b[i])
        kl_ = min(t_, s_) * n_distinct + max(t_, s_)
        p = np.searchsorted(code_s, kl_)
        in_cache = (float(J_s[p]) if p < code_s.shape[0] and code_s[p] == kl_
                    else None)
        ot, os_ = seq_desc[t_], seq_desc[s_]
        detail.append({"target": t_, "partner": s_, "J": J_,
                       "pair_in_build_cache": in_cache,
                       "target_OG": ot[0], "target_cluster": ot[1],
                       "target_genome": ot[2], "partner_OG": os_[0],
                       "partner_cluster": os_[1], "partner_genome": os_[2],
                       "rows": int(weights_test[t_])})
        print("     %6d (%s/%s/cl %d) <-> train %6d (%s/%s/cl %d) J=%.4f | "
              "pair in the build cache: %s"
              % (t_, ot[2], ot[0], ot[1], s_, os_[2], os_[0], os_[1], J_,
                 "ABSENT" if in_cache is None else "%.4f" % in_cache))
    n_absent = sum(1 for s in detail if s["pair_in_build_cache"] is None)
    print("     pairs above the threshold: %d, of which ABSENT from the build "
          "cache: %d" % (len(detail), n_absent))
    print("     -> %s"
          % ("the cause is the BLIND SPOT of the candidate stage (cap_df / "
             "min_shared), not broken merging"
             if n_absent == len(detail) else
             "SOME pairs were in the cache and still did not merge -- broken "
             "merging, which is more serious"))
    results["pairs_above_threshold_detail"] = detail
    to_merge = sorted({(min(s["target_cluster"], s["partner_cluster"]),
                        max(s["target_cluster"], s["partner_cluster"]))
                       for s in detail})
    results["clusters_to_merge_test"] = [list(x) for x in to_merge]
    print("     minimal repair for test: merge %d cluster pairs: %s"
          % (len(to_merge), to_merge))

    # (f) The remaining evaluation parts, same measurement.
    print("\n  (f) val and spy -- the same measurement (leakage is not only "
          "about test)")
    others = {}
    for c in ("val", "spy"):
        mc_ = np.zeros(n_distinct, dtype=bool)
        mc_[assign[cz == c]] = True
        full_c = {int(i): sb.kmers(distinct[int(i)])
                  for i in np.flatnonzero(mc_)}
        wg = np.bincount(assign[cz == c], minlength=n_distinct)
        cc, JJ, aa = max_similarity(PI, bounds, dfk, slot, full_c, m_train,
                                    mc_, distinct, None, 1, c, top_m=100,
                                    return_partner=True)
        others[c] = describe(JJ, wg[cc], "%s->train [independent]" % c)
        pairs_c = []
        for i in np.flatnonzero(JJ >= sb.MERGE_THRESHOLD):
            t_, s_ = int(cc[i]), int(aa[i])
            pairs_c.append((seq_desc[t_][1], seq_desc[s_][1]))
        others[c]["clusters_to_merge"] = sorted({(min(x), max(x))
                                                 for x in pairs_c})
        print("     %s: clusters to merge %d"
              % (c, len(others[c]["clusters_to_merge"])))
    results["max_J_val_spy"] = {
        k_: {kk: (vv if not isinstance(vv, list) else [list(y) for y in vv])
             for kk, vv in v_.items()} for k_, v_ in others.items()}

    ok1 = (d2b["rows_above_0.050"] == 0
           and others["val"]["rows_above_0.050"] == 0
           and others["spy"]["rows_above_0.050"] == 0)
    verdict("1. homology leakage eval->train (J >= merge threshold 0.05)",
            "test %d rows | val %d | spy %d (independent candidate stage, "
            "top-M=100); test: p99 = %.4f, max = %.4f; the build regime sees 0 "
            "at max %.4f"
            % (d2b["rows_above_0.050"], others["val"]["rows_above_0.050"],
               others["spy"]["rows_above_0.050"], d2b["p99"], d2b["max"],
               d1["max"]),
            "0 rows; the cluster is the atom, so a pair >= 0.05 cannot cross "
            "the boundary",
            ok1, "hyppe/analysis/splits_audit.py section 1(b)")

    header("2. POSITIVE CONTROL: the same measurement on a RANDOM ROW split")
    rng = np.random.default_rng(AUDIT_SEED)
    n = len(df)
    rnd = np.array(["train"] * n, dtype=object)
    perm = rng.permutation(n)
    kw = [int(round(sb.PROPORTIONS[c] * n)) for c in sb.SPLITS]
    kw[0] = n - sum(kw[1:])
    pos = 0
    for c, k in zip(sb.SPLITS, kw):
        rnd[perm[pos:pos + k]] = c
        pos += k
    lm_test = np.zeros(n_distinct, dtype=bool)
    lm_train = np.zeros(n_distinct, dtype=bool)
    lm_test[assign[rnd == "test"]] = True
    lm_train[assign[rnd == "train"]] = True
    log("random: distinct test %d | train %d | in both %d"
        % (int(lm_test.sum()), int(lm_train.sum()),
           int((lm_test & lm_train).sum())))
    full_r: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(lm_test):
        full_r[int(i)] = sb.kmers(distinct[int(i)])
    weights_rnd = np.bincount(assign[rnd == "test"], minlength=n_distinct)
    c3, J3 = max_similarity(PI, bounds, dfk, slot, full_r, lm_train, lm_test,
                            distinct, None, 1, "random")
    d3 = describe(J3, weights_rnd[c3], "RANDOM test->train [independent]")
    results["max_J_random"] = d3

    diff = d3["fraction_rows_above_0.050"] - d2["fraction_rows_above_0.050"]
    ok2 = d3["fraction_rows_above_0.050"] >= 0.10 and diff >= 0.05
    verdict("2. positive control of the audit",
            "random split: %.4f of test rows have a partner in train with "
            "J >= 0.05 (%d rows), identical %d; cluster split: %.4f.  "
            "Difference %.4f"
            % (d3["fraction_rows_above_0.050"], d3["rows_above_0.050"],
               d3["rows_above_0.999"], d2["fraction_rows_above_0.050"], diff),
            "the audit must show >= 0.10 leakage on a random split and a "
            "difference >= 0.05 against the cluster split; otherwise it "
            "measures nothing",
            ok2, "hyppe/analysis/splits_audit.py section 2")

    header("3. DISJOINTNESS")
    r3 = {}
    for col in ("cluster", "OG_ID", "GENE_ID"):
        g = w.groupby(col)["split"].nunique()
        bad = g[g > 1]
        r3[col] = int(len(bad))
        print("  %-9s in >1 part: %d  (of %d)" % (col, len(bad), len(g)))
        if len(bad):
            print("     e.g. %s" % list(bad.index[:5]))
    # GENE_ID: the same gene, or a collision between genomes?
    g2 = w.assign(_k=w["GENOME"] + "|" + w["GENE_ID"].astype(str))
    gg = g2.groupby("_k")["split"].nunique()
    r3["GENOME_GENE_ID"] = int((gg > 1).sum())
    gid = w.groupby("GENE_ID")["GENOME"].nunique()
    r3["GENE_ID_in_several_genomes"] = int((gid > 1).sum())
    print("  GENOME|GENE_ID in >1 part: %d  (of %d)"
          % (int((gg > 1).sum()), len(gg)))
    print("  GENE_IDs occurring in >1 GENOME: %d  -> GENE_ID is not a global key"
          % int((gid > 1).sum()))
    # Exact duplicate sequences across the boundary.
    seq_split = {}
    for i in range(n):
        seq_split.setdefault(int(assign[i]), set()).add(cz[i])
    dup_across = [s for s, v in seq_split.items() if len(v) > 1]
    r3["identical_sequences_in_2_parts"] = len(dup_across)
    print("  IDENTICAL sequences present in >1 part: %d" % len(dup_across))
    # Duplicate rows in the file.
    dup_id = int(w.duplicated(subset=["ID"]).sum())
    dup_full = int(w.duplicated().sum())
    dup_seq_row = int(pd.Series(assign).duplicated().sum())
    r3["duplicated_ids"] = dup_id
    r3["identical_rows_in_file"] = dup_full
    r3["rows_with_repeated_sequence"] = dup_seq_row
    print("  repeated IDs: %d | identical rows: %d | rows with a repeated "
          "SEQUENCE: %d" % (dup_id, dup_full, dup_seq_row))
    results["disjointness"] = r3
    ok3 = (r3["cluster"] == 0 and r3["OG_ID"] == 0
           and r3["GENOME_GENE_ID"] == 0 and len(dup_across) == 0)
    verdict("3. disjointness of cluster / OG_ID / gene / sequence",
            "cluster %d | OG_ID %d | GENOME|GENE_ID %d | bare GENE_ID %d "
            "(of which %d GENE_IDs live in >1 genome, so a bare GENE_ID is not "
            "a gene key) | identical sequence in 2 parts %d"
            % (r3["cluster"], r3["OG_ID"], r3["GENOME_GENE_ID"], r3["GENE_ID"],
               r3["GENE_ID_in_several_genomes"], len(dup_across)),
            "cluster, OG_ID, GENOME|GENE_ID and identical sequence: each 0",
            ok3, "hyppe/analysis/splits_audit.py section 3")

    header("4. COMPARABILITY OF THE PARTS "
           "(null = permutation of CLUSTERS, not of rows)")
    tgt = sb.target_per_gene()
    log("target loaded: %d matrix rows" % len(tgt))
    wc = w.merge(tgt, on="ID", how="left")
    if len(wc) != len(w):
        raise RuntimeError("the target merge changed the row count: %d -> %d"
                           % (len(w), len(wc)))
    kl = w["cluster"].to_numpy()
    nkl = int(kl.max()) + 1
    genomes = sorted(w["GENOME"].unique())
    gi = pd.Categorical(w["GENOME"], categories=genomes).codes
    M = np.zeros((nkl, len(genomes)), dtype=np.int64)
    np.add.at(M, (kl, gi), 1)
    n_seq = M.sum(1)
    n_p1 = np.bincount(kl,
                       weights=(w["GENOME"] == "P1").to_numpy().astype(float),
                       minlength=nkl)
    c_ok = np.isfinite(wc["target"].to_numpy())
    s_tgt = np.bincount(kl, weights=np.where(c_ok, wc["target"].to_numpy(), 0.0),
                        minlength=nkl)
    n_tgt = np.bincount(kl, weights=c_ok.astype(float), minlength=nkl)
    kl_split = pd.Series(cz).groupby(kl).first().reindex(range(nkl)).to_numpy()

    whole = M.sum(0) / M.sum()

    def stats(assignment):
        out = {}
        for c in sb.SPLITS:
            m = assignment == c
            share = M[m].sum(0) / max(M[m].sum(), 1)
            out[c] = {
                "TVD_genomes": float(0.5 * np.abs(share - whole).sum()),
                "max_pp": float(100 * np.abs(share - whole).max()),
                "mean_target": float(s_tgt[m].sum() / max(n_tgt[m].sum(), 1)),
                "p1_share": float(n_p1[m].sum() / max(M[m].sum(), 1)),
                "median_size": float(np.median(n_seq[m])),
                "max_size": int(n_seq[m].max()),
                "sequences": int(M[m].sum()),
                "clusters": int(m.sum()),
            }
        return out

    obs = stats(kl_split)
    print("  %-6s %8s %8s %7s %7s %7s %7s %6s" %
          ("part", "seq", "clust", "TVD", "maxpp", "target", "P1%", "med"))
    for c in sb.SPLITS:
        o = obs[c]
        print("  %-6s %8d %8d %7.4f %7.2f %7.3f %7.3f %6.0f"
              % (c, o["sequences"], o["clusters"], o["TVD_genomes"],
                 o["max_pp"], o["mean_target"], 100 * o["p1_share"],
                 o["median_size"]))
    print("  largest cluster per part: " +
          " | ".join("%s %d" % (c, obs[c]["max_size"]) for c in sb.SPLITS))

    rngp = np.random.default_rng(AUDIT_SEED + 5)
    perm_tvd = {c: [] for c in sb.SPLITS}
    perm_tgt = {c: [] for c in sb.SPLITS}
    perm_p1 = {c: [] for c in sb.SPLITS}
    for _ in range(N_PERM):
        p = kl_split[rngp.permutation(nkl)]
        st = stats(p)
        for c in sb.SPLITS:
            perm_tvd[c].append(st[c]["TVD_genomes"])
            perm_tgt[c].append(st[c]["mean_target"])
            perm_p1[c].append(st[c]["p1_share"])
    log("permutations: %d" % N_PERM)

    r4 = {"observed": obs, "permutation_null": {}}
    print("\n  against the PERMUTATION NULL (%d cluster permutations, seed %d):"
          % (N_PERM, AUDIT_SEED + 5))
    min_p = 1.0
    min_desc = ""
    for c in sb.SPLITS:
        a = np.asarray(perm_tvd[c])
        b = np.asarray(perm_tgt[c])
        e = np.asarray(perm_p1[c])
        p_tvd = float((a >= obs[c]["TVD_genomes"]).mean())
        p_tgt = float((np.abs(b - b.mean())
                       >= abs(obs[c]["mean_target"] - b.mean())).mean())
        p_p1 = float((np.abs(e - e.mean())
                      >= abs(obs[c]["p1_share"] - e.mean())).mean())
        r4["permutation_null"][c] = {
            "TVD_null_median": float(np.median(a)),
            "TVD_null_p95": float(np.percentile(a, 95)),
            "p_TVD": p_tvd, "p_target": p_tgt, "p_P1": p_p1}
        print("     %-6s TVD obs %.4f (null med %.4f, p95 %.4f) p=%.3f | "
              "target p=%.3f | P1 p=%.3f"
              % (c, obs[c]["TVD_genomes"], np.median(a),
                 np.percentile(a, 95), p_tvd, p_tgt, p_p1))
        for nm, pv in (("genome TVD", p_tvd), ("mean target", p_tgt),
                       ("P1 share", p_p1)):
            if pv < min_p:
                min_p, min_desc = pv, "%s in %s" % (nm, c)
    # Effect size, not only significance: which genome, and by how much.
    print("\n  EFFECT SIZE (at 22 159 clusters, significance catches every mote):")
    max_pp = 0.0
    for c in sb.SPLITS:
        m = kl_split == c
        share = M[m].sum(0) / M[m].sum()
        d_ = share - whole
        j = int(np.argmax(np.abs(d_)))
        print("     %-6s largest gap: %s  %.3f %% against %.3f %% (%+.2f pp)"
              % (c, genomes[j], 100 * share[j], 100 * whole[j], 100 * d_[j]))
        max_pp = max(max_pp, abs(100 * d_[j]))
    # Is the TVD excess the work of the largest cluster?
    biggest = int(np.argmax(n_seq))
    m2 = kl_split.copy()
    m2[biggest] = None
    st2 = {}
    for c in sb.SPLITS:
        mm = m2 == c
        share = M[mm].sum(0) / M[mm].sum()
        st2[c] = float(0.5 * np.abs(share - whole).sum())
    print("     after REMOVING the largest cluster (%d seq.): TVD %s"
          % (n_seq[biggest],
             " | ".join("%s %.4f" % (c, st2[c]) for c in sb.SPLITS)))
    r4["max_genome_gap_pp"] = max_pp
    r4["TVD_without_largest_cluster"] = st2
    results["comparability"] = r4
    ok4 = min_p >= 0.05 / 12
    verdict("4. comparability of the parts",
            "smallest permutation p: %.4f (%s); largest genome TVD %.4f (%s) "
            "against a null median of %.4f; SIZE: largest genome-share gap "
            "%.2f pp, P1 %.2f %% in train against %.2f %% in test"
            % (min_p, min_desc,
               max(obs[c]["TVD_genomes"] for c in sb.SPLITS),
               max(sb.SPLITS, key=lambda c: obs[c]["TVD_genomes"]),
               float(np.median(perm_tvd["test"])), max_pp,
               100 * obs["train"]["p1_share"], 100 * obs["test"]["p1_share"]),
            "p >= 0.05/12 (Bonferroni over 12 comparisons) against a null that "
            "is a RANDOM permutation of clusters -- a distribution this split "
            "cannot beat indefinitely",
            ok4, "hyppe/analysis/splits_audit.py section 4")

    header("5. LEAKAGE THROUGH THE COLUMNS")
    k = pd.read_csv(os.path.join(args.split or sb.SPLIT_DIR,
                                 os.path.basename(sb.COLUMNS_TSV)), sep="\t")
    meta_path = args.metadata or paths.samples_metadata_path()
    paths.require(meta_path, "samples metadata")
    meta = pd.read_csv(meta_path, sep="\t")
    # Printed, not assumed: section 5 is a function of this file's `condition`
    # column, and more than one version of it exists.
    print("  metadata: %s" % meta_path)
    print("            %d columns | md5 %s" % (meta.shape[1], _md5(meta_path)))
    results["metadata_source"] = {"path": meta_path,
                                  "columns": int(meta.shape[1]),
                                  "md5": _md5(meta_path)}
    mk = k.merge(meta[["sample", "condition"]], on="sample", how="left")
    if len(mk) != len(k):
        raise RuntimeError("the metadata merge changed the sample count")
    r5 = {}
    for name, columns in (
            ("study+condition+genome (the key from baselines.sanity 3)",
             ["study", "condition", "genome"]),
            ("condition+genome (WITHOUT study -- may cross the boundary)",
             ["condition", "genome"]),
            ("condition_category+genome",
             ["condition_category", "genome"])):
        key = mk[columns].fillna("").astype(str).agg("|".join, axis=1)
        dup = key.duplicated(keep=False)
        g = mk.assign(_k=key)[dup].groupby("_k")["split"].nunique()
        across = int((g > 1).sum())
        samples_across = int(
            mk.assign(_k=key)[dup & key.isin(g[g > 1].index)].shape[0])
        r5[name] = {"samples_in_duplicates": int(dup.sum()),
                    "duplicated_keys": int(len(g)),
                    "keys_across_boundary": across,
                    "samples_in_keys_across_boundary": samples_across}
        print("  %-62s dup. samples %4d | keys %4d | across %3d (%d samples)"
              % (name, int(dup.sum()), len(g), across, samples_across))
    print("  NOTE: the key from `baselines.sanity 3` contains `study`, and")
    print("  `study` is an ATOM of the column split (guard_columns) -- so that")
    print("  falsifier is satisfied BY CONSTRUCTION and carries no information")
    print("  on its own.  The information is in the key WITHOUT `study`.")
    results["column_leakage"] = r5
    sharp = r5["condition+genome (WITHOUT study -- may cross the boundary)"]
    ok5 = sharp["samples_in_keys_across_boundary"] / len(mk) < 0.05
    verdict("5. leakage through the columns",
            "key study+condition+genome: %d keys across the boundary "
            "(satisfied by construction, study is the atom).  Key "
            "condition+genome: %d keys across, %d samples = %.4f of the panel"
            % (r5["study+condition+genome (the key from baselines.sanity 3)"]
               ["keys_across_boundary"],
               sharp["keys_across_boundary"],
               sharp["samples_in_keys_across_boundary"],
               sharp["samples_in_keys_across_boundary"] / len(mk)),
            "< 0.05 of the panel in repeated conditions across the boundary "
            "(the key WITHOUT study)",
            ok5, "hyppe/analysis/splits_audit.py section 5")

    header("SUMMARY")
    for v in VERDICTS:
        print("  %-7s %s" % (v["verdict"], v["name"]))
    results["verdicts"] = VERDICTS
    out = os.path.join(sb.SPLIT_DIR, "audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("\n  written: %s" % out)
    return 0 if all(v["verdict"] == "PASSED" for v in VERDICTS) else 3


if __name__ == "__main__":
    sys.exit(main())
