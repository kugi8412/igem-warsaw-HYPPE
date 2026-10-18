"""Blocker 0.3 -- measure the measurement-noise ceiling on final_datasets.

`baselines.py` scores a model by the mean per-column Spearman: for one sample
column, rank the orthogroup rows. The ceiling must be computed on the same
axis -- two sample columns correlated across rows -- never as a global
correlation over all cells. The label there is one column, i.e. one SRX, so
the reliability that binds is that of a single replicate, `sqrt(r_pair)`. The
Spearman-Brown number `sqrt(n*r/(1+(n-1)*r))` is printed beside it because it
is what would bind after a switch to a replicate-averaged label.

`condition` is free text carrying the replicate index inside it ("..., rep 2",
"... - sample 49"), and the raw string gives 2237 singleton groups, so the
index has to be stripped. Stripping too much merges different biology
(timepoints, carbon sources) and deflates the ceiling, hence three policies:
STRICT strips only explicit replicate words, keying on study + genome + mutant
+ normalised condition; LOOSE also strips "sample N" / "run N" /
"experiment N"; NOSTUDY is LOOSE without `study`, so its groups may span
batches. STRICT is the headline, because it never merges two strings that
differ in any word but the replicate token.

Which copy of `samples_metadata.tsv` is read moves the result: two copies are
in circulation whose `condition` differs in 2300 of 2332 rows, in vocabulary
rather than formatting, giving different groups and a different ceiling
(release 0.9858, legacy 0.9868 on `combat`, raw, STRICT). The default is the
release copy; ``--metadata`` reproduces the legacy number. Every run prints
the path, column count and md5 of the table it used.

Negative control: correlating two columns across orthogroups is dominated by
the orthogroup main effect (a row-mean baseline already reaches ~0.89), so
non-replicates correlate highly too, and if replicates do not separate from
them no ceiling has been measured and the number must not be quoted. Both
controls hold genome fixed, since replicates are always same-genome: C1 is
same genome and different study, C2 same study and different condition, the
tighter of the two. Everything is recomputed on the row-centred matrix as
well, which is design A' of `baselines.py`.

    python -m hyppe.analysis.noise_ceiling
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import matrices as data
from . import paths

SCOPE = "og_scope"
MIN_OVERLAP = 200          # rows finite in both columns
N_CONTROL_PAIRS = 3000     # sampled per control, per matrix
RNG = np.random.default_rng(0)

# Condition normalisation. Explicit replicate token: "rep 1", "rep.2",
# "biological replicate 3", "biological replicte 1" (the typo is in the
# release), "replicate a".
REP_STRICT = re.compile(
    r"\b(?:biological|technical|bio|tech)?\s*"
    r"(?:replicates?|replicat|replicte|replica|reps?)"
    r"\s*[._#-]?\s*(?:[0-9]+|[a-f])\b", re.I)
# Index-like tokens, usually a replicate index but possibly a real factor;
# used by LOOSE only.
REP_LOOSE = re.compile(r"\b(?:sample|run|experiment|expt)\s*[._#-]?\s*[0-9]+\b",
                       re.I)


def _tidy(s: str) -> str:
    return re.sub(r"[\s,;:._#-]+", " ", s).strip().lower()


def norm_strict(s: str) -> str:
    return _tidy(REP_STRICT.sub(" ", s))


def norm_loose(s: str) -> str:
    return _tidy(REP_LOOSE.sub(" ", REP_STRICT.sub(" ", s)))


def build_groups(meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Three grouping keys, as Series aligned to meta.index."""
    cs = meta["condition"].map(norm_strict)
    cl = meta["condition"].map(norm_loose)
    sep = "|"
    return {
        "STRICT": meta["study"] + sep + meta["genome"] + sep + meta["mutant"]
        + sep + cs,
        "LOOSE": meta["study"] + sep + meta["genome"] + sep + meta["mutant"]
        + sep + cl,
        "NOSTUDY": meta["genome"] + sep + meta["mutant"] + sep + cl,
    }


# Pair statistics
def auc_separation(rep: np.ndarray, ctrl: np.ndarray) -> float:
    """P(a random replicate pair correlates higher than a random control pair).

    Mann-Whitney U / (n*m), ties counted as half: 0.5 means replicates do not
    stand out from non-replicates, 1.0 that the distributions are disjoint.
    Reported beside the quantile test, which is sharp enough that failing it
    does not yet mean there is no separation.
    """
    r = np.asarray(rep, float)
    c = np.asarray(ctrl, float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size == 0 or c.size == 0:
        return float("nan")
    allv = np.concatenate([r, c])
    rk = rankdata(allv)
    u = rk[:r.size].sum() - r.size * (r.size + 1) / 2.0
    return float(u / (r.size * c.size))


def pair_stats(a: np.ndarray, b: np.ndarray,
               mask: np.ndarray | None = None) -> tuple[float, float, int]:
    """(Spearman, Pearson, n) between two columns, over rows finite in both.

    NaN is a result, not an error, when the overlap is too small or a column
    is constant on it: a constant column cannot rank anything, and dropping
    such pairs silently would hide how many there are.
    """
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = m & mask
    n = int(m.sum())
    if n < MIN_OVERLAP:
        return float("nan"), float("nan"), n
    x, y = a[m], b[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), n
    p = float(np.corrcoef(x, y)[0, 1])
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan"), p, n
    s = float(np.corrcoef(rx, ry)[0, 1])
    return s, p, n


def replicate_pairs(groups: pd.Series, cols: list[str]) -> list[tuple]:
    """(i, j, group_key) for every within-group pair present in the matrix."""
    pos = {c: i for i, c in enumerate(cols)}
    g = groups[[c for c in cols if c in groups.index]]
    out = []
    for key, idx in g.groupby(g).groups.items():
        members = [pos[c] for c in idx if c in pos]
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                out.append((members[a], members[b], key))
    return out


def control_pairs(meta: pd.DataFrame, cols: list[str], kind: str,
                  groups: pd.Series, n_want: int) -> list[tuple]:
    """Non-replicate pairs, genome held fixed."""
    pos = {c: i for i, c in enumerate(cols)}
    sub = meta.loc[[c for c in cols if c in meta.index]]
    out: list[tuple] = []
    seen: set[tuple[int, int]] = set()
    by_genome = {g: d for g, d in sub.groupby("genome")}
    genomes = [g for g, d in by_genome.items() if len(d) >= 4]
    if not genomes:
        return out
    tries = 0
    while len(out) < n_want and tries < n_want * 200:
        tries += 1
        g = genomes[RNG.integers(len(genomes))]
        d = by_genome[g]
        i, j = RNG.integers(len(d), size=2)
        if i == j:
            continue
        ra, rb = d.iloc[i], d.iloc[j]
        if kind == "C1":                      # different study
            if ra["study"] == rb["study"]:
                continue
        elif kind == "C2":                    # same study, different condition
            if ra["study"] != rb["study"]:
                continue
        else:
            raise ValueError(kind)
        # A real replicate pair must never leak into a control.
        if groups.get(ra["sample"]) == groups.get(rb["sample"]):
            continue
        if norm_loose(ra["condition"]) == norm_loose(rb["condition"]):
            continue
        pa, pb = pos[ra["sample"]], pos[rb["sample"]]
        k = (min(pa, pb), max(pa, pb))
        if k in seen:
            continue
        seen.add(k)
        out.append((pa, pb, kind))
    return out


def summarise(vals: np.ndarray, keys: list | None = None) -> dict:
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    d = {"n_pairs": int(v.size), "n_valid": int(ok.sum())}
    if ok.sum() == 0:
        return d
    w = v[ok]
    d.update({
        "min": float(w.min()), "q05": float(np.percentile(w, 5)),
        "q25": float(np.percentile(w, 25)), "median": float(np.median(w)),
        "q75": float(np.percentile(w, 75)), "q95": float(np.percentile(w, 95)),
        "max": float(w.max()), "mean": float(w.mean()),
    })
    z = np.arctanh(np.clip(w, -0.999999, 0.999999))
    d["fisher_z"] = float(np.tanh(z.mean()))
    if keys is not None:
        kk = np.asarray(keys, dtype=object)[ok]
        per = pd.Series(w).groupby(pd.Series(kk)).mean()
        d["mean_over_groups"] = float(per.mean())
        d["n_groups"] = int(per.size)
    return d


def fmt(d: dict) -> str:
    if d.get("n_valid", 0) == 0:
        return "  NO VALID PAIRS (n_pairs=%d)" % d["n_pairs"]
    s = ("  n_pairs %5d (valid %5d) | median %+.4f | mean %+.4f | "
         "fisher-z %+.4f\n"
         "    quantiles  min %+.4f  q05 %+.4f  q25 %+.4f  q75 %+.4f  "
         "q95 %+.4f  max %+.4f"
         % (d["n_pairs"], d["n_valid"], d["median"], d["mean"],
            d["fisher_z"], d["min"], d["q05"], d["q25"], d["q75"],
            d["q95"], d["max"]))
    if "mean_over_groups" in d:
        s += ("\n    mean over GROUPS (not over pairs): %+.4f  (%d groups)"
              % (d["mean_over_groups"], d["n_groups"]))
    return s


def spearman_brown(r: float, n: int) -> float:
    if not np.isfinite(r):
        return float("nan")
    return n * r / (1.0 + (n - 1) * r)


def ceiling(r: float) -> float:
    """corr(truth, one measurement) = sqrt(reliability); negative -> nan."""
    if not np.isfinite(r) or r <= 0:
        return float("nan")
    return float(np.sqrt(min(r, 1.0)))


RAW = "raw"
CENTRED = "row-centred (design A')"
REPLICATES = "REPLICATES"
CTRL1 = "control C1"
CTRL2 = "control C2"


def run_matrix(name: str, meta: pd.DataFrame, groups_all: dict,
               headline_policy: str = "STRICT",
               aux_finite: np.ndarray | None = None,
               aux_label: str = "") -> dict:
    print("\n" + "=" * 78)
    print("MATRIX %s  (scope=%s)" % (name, SCOPE))
    print("=" * 78)
    ids, cols, M = data.load_orthogroup_matrix(SCOPE, name)
    print("shape %s | finite cells %.2f %%"
          % (M.shape, 100 * np.isfinite(M).mean()))

    variants = [(RAW, M, None)]
    Mc = M - np.nanmean(M, axis=1, keepdims=True)
    variants.append((CENTRED, Mc, None))
    if aux_finite is not None:
        variants.append(("%s, mask %s" % (RAW, aux_label), M, aux_finite))
        variants.append(("centred, mask %s" % aux_label, Mc, aux_finite))

    res: dict = {}
    for pol in groups_all:
        rp_ = replicate_pairs(groups_all[pol], cols)
        print("\npolicy %-8s : replicate pairs in matrix = %d"
              % (pol, len(rp_)))
        res.setdefault("pairs", {})[pol] = len(rp_)

    g_head = groups_all[headline_policy]
    rp = replicate_pairs(g_head, cols)
    c1 = control_pairs(meta, cols, "C1", g_head, N_CONTROL_PAIRS)
    c2 = control_pairs(meta, cols, "C2", g_head, N_CONTROL_PAIRS)

    for vname, V, aux in variants:
        print("\n" + "-" * 78)
        print("variant: %s" % vname)
        print("-" * 78)
        for label, pairs, use_keys in (
                ("%s (%s)" % (REPLICATES, headline_policy), rp, True),
                ("%s (same genome, DIFFERENT study)" % CTRL1, c1, False),
                ("%s (same study, DIFFERENT condition)" % CTRL2, c2, False)):
            sp, pe, ns, ks = [], [], [], []
            for a, b, k in pairs:
                m = None if aux is None else (aux[:, a] & aux[:, b])
                s, p, n = pair_stats(V[:, a], V[:, b], m)
                sp.append(s)
                pe.append(p)
                ns.append(n)
                ks.append(k)
            ds = summarise(sp, ks if use_keys else None)
            dp = summarise(pe, ks if use_keys else None)
            print("\n%s" % label)
            print("  [Spearman across rows]")
            print(fmt(ds))
            print("  [Pearson across rows]")
            print(fmt(dp))
            if ns:
                print("    row overlap: median %d, min %d"
                      % (int(np.median(ns)), int(np.min(ns))))
            res.setdefault(vname, {})[label.split(" (")[0]] = {
                "spearman": ds, "pearson": dp,
                "_v_spearman": np.asarray(sp, float),
                "_v_pearson": np.asarray(pe, float)}
    res["_cols"] = cols
    res["_rp"] = rp
    return res


def ceiling_table(tag: str, rep: dict, ctrl1: dict, ctrl2: dict,
                  group_sizes: pd.Series) -> bool:
    print("\n" + "=" * 78)
    print("CEILING -- %s" % tag)
    print("=" * 78)
    ok = True
    for stat in ("spearman", "pearson"):
        r_med = rep[stat]["median"]
        r_grp = rep[stat].get("mean_over_groups", float("nan"))
        c1_95 = ctrl1[stat]["q95"]
        c2_95 = ctrl2[stat]["q95"]
        c1_med = ctrl1[stat]["median"]
        c2_med = ctrl2[stat]["median"]
        sep = min(r_med - c1_med, r_med - c2_med)
        a1 = auc_separation(rep["_v_" + stat], ctrl1["_v_" + stat])
        a2 = auc_separation(rep["_v_" + stat], ctrl2["_v_" + stat])
        strict = (r_med > c1_95) and (r_med > c2_95)
        auc_ok = min(a1, a2) >= 0.75
        passed = auc_ok
        ok = ok and passed
        print("\n[%s]" % stat)
        print("  r_replicates  median %+.4f | mean over groups %+.4f"
              % (r_med, r_grp))
        print("  control C1 median %+.4f  q95 %+.4f | AUC rep>C1 %.4f"
              % (c1_med, c1_95, a1))
        print("  control C2 median %+.4f  q95 %+.4f | AUC rep>C2 %.4f"
              % (c2_med, c2_95, a2))
        print("  separation (replicate median - control median): %+.4f" % sep)
        print("  sharp test (replicate median > control q95): %s"
              % ("YES" if strict else "NO -- the distributions overlap"))
        print("  NEGATIVE CONTROL (criterion: min AUC >= 0.75): %s"
              % ("PASSED" if passed
                 else "FAILED -- no ceiling was measured, do not quote"))
        if not passed:
            continue
        for r_name, r in (("pair median", r_med),
                          ("mean over groups", r_grp)):
            print("  -- from r = %+.4f (%s)" % (r, r_name))
            print("     label = a SINGLE sample (baselines.py protocol):"
                  "  ceiling = sqrt(r) = %.4f" % ceiling(r))
            for n in sorted({2, 3, int(round(group_sizes.mean()))}):
                rn = spearman_brown(r, n)
                print("     label = MEAN of n=%d replicates: r_n = %.4f, "
                      "ceiling = %.4f" % (n, rn, ceiling(rn)))
    return ok


def _md5(path: str) -> str:
    h = hashlib.md5()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metadata", default=None,
                    help="sample table to key replicates on.  Default: the "
                         "release copy.  Point it at the legacy 18-column "
                         "copy to reproduce the 0.9868 in the report; see "
                         "the module docstring for the measured difference.")
    args = ap.parse_args(argv)

    meta_path = args.metadata or paths.samples_metadata_path(SCOPE)
    paths.require(meta_path, "samples metadata")
    meta = pd.read_csv(meta_path, sep="\t", dtype=str).set_index("sample",
                                                                 drop=False)
    meta = meta.fillna("")
    print("=" * 78)
    print("BLOCKER 0.3 -- MEASUREMENT NOISE CEILING, final_datasets/%s" % SCOPE)
    print("=" * 78)
    # Printed, not assumed: the grouping, and so the whole ceiling, is a
    # function of this file's `condition` column, and more than one version
    # of that column exists.
    print("metadata : %s\n           %d columns | md5 %s"
          % (meta_path, meta.shape[1], _md5(meta_path)))
    print("samples in metadata: %d | studies %d | genomes %d"
          % (len(meta), meta["study"].nunique(), meta["genome"].nunique()))

    # 1-2. Replicate groups
    groups = build_groups(meta)
    print("\n" + "-" * 78)
    print("1-2. REPLICATE GROUPS")
    print("-" * 78)
    raw_key = (meta["study"] + "|" + meta["genome"] + "|" + meta["mutant"]
               + "|" + meta["condition"])
    rvc = raw_key.value_counts()
    print("baseline control: a key on the RAW `condition` gives %d groups, "
          "of which %d have size 1" % (rvc.size, int((rvc == 1).sum())))
    print("  -> the replicate index lives inside the condition text; without "
          "normalisation no replicates are visible at all.")
    sizes = {}
    for pol, g in groups.items():
        vc = g.value_counts()
        sizes[pol] = vc
        dist = vc.value_counts().sort_index()
        print("\n%s: groups %d | groups >=2: %d | samples in groups >=2: %d"
              % (pol, vc.size, int((vc >= 2).sum()), int(vc[vc >= 2].sum())))
        print("  size distribution (size: how many groups): %s"
              % ", ".join("%d:%d" % (k, v) for k, v in dist.items()))
    head_sizes = sizes["STRICT"][sizes["STRICT"] >= 2]
    print("\nSTRICT, groups >=2: mean size %.2f | median %d | max %d"
          % (head_sizes.mean(), int(head_sizes.median()),
             int(head_sizes.max())))

    # 3. Correlations
    out = {}
    out["combat"] = run_matrix("combat", meta, groups)
    # `rank` has no NaN: a missing orthogroup is written as 1.0, a silent
    # gene. Unmasked, ~29 % of rows are then the same constant in both
    # columns and any two columns correlate higher by construction, so the
    # mask is the cells finite in `combat`.
    _ids, _cols, C = data.load_orthogroup_matrix(SCOPE, "combat")
    fin_c = np.isfinite(C)
    _ids_r, _cols_r, R = data.load_orthogroup_matrix(SCOPE, "rank")
    assert _cols_r == _cols, "rank and combat must have the same columns"
    r_one = (R == 1.0)
    print("\nmask `combat-finite`: %.2f %% of cells kept.\n"
          "  `rank` is NaN in %.2f %% of cells and exactly 1.0 in %.2f %%;\n"
          "  share of combat-NaN cells that are 1.0 in `rank`: %.2f %% "
          "-> a filler constant, not a measurement."
          % (100 * fin_c.mean(), 100 * (~np.isfinite(R)).mean(),
             100 * r_one.mean(), 100 * r_one[~fin_c].mean()))
    del R, r_one
    out["rank"] = run_matrix("rank", meta, groups, aux_finite=fin_c,
                             aux_label="combat-finite")

    # 4-5. Ceiling and control
    verdicts = {}
    todo = [("combat", RAW),
            ("combat", CENTRED),
            ("rank", RAW),
            ("rank", CENTRED),
            ("rank", "%s, mask combat-finite" % RAW),
            ("rank", "centred, mask combat-finite")]
    for name, vname in todo:
        blk = out[name][vname]
        verdicts[(name, vname)] = ceiling_table(
            "%s / %s" % (name, vname),
            blk[REPLICATES], blk[CTRL1], blk[CTRL2], head_sizes)

    # 6. Sensitivity to the grouping policy. Falsifier of the normaliser: if
    # it merged different biology, r_replicates would fall under a looser
    # policy, and without this the ceiling could be an artefact of a regex.
    print("\n" + "=" * 78)
    print("6. SENSITIVITY TO THE GROUPING POLICY (combat, centred)")
    print("=" * 78)
    Cc = C - np.nanmean(C, axis=1, keepdims=True)
    for pol, g in groups.items():
        prs = replicate_pairs(g, _cols)
        sp = [pair_stats(Cc[:, a], Cc[:, b])[0] for a, b, _k in prs]
        d = summarise(sp)
        print("  %-8s pairs %5d | median Spearman %+.4f | mean %+.4f"
              % (pol, len(prs), d["median"], d["mean"]))
    print("  -> a drop of the median under LOOSE/NOSTUDY = the normalisation\n"
          "     merges different biology and deflates the ceiling; no drop = "
          "the groups are replicates.")

    # 7. Breakdown per genome. One number for the release hides that genomes
    # differ in depth and quality, and since the model is scored per column
    # and columns belong to genomes, the per-genome ceiling is what binds.
    print("\n" + "=" * 78)
    print("7. CEILING PER GENOME (combat, STRICT, >=20 pairs)")
    print("=" * 78)
    gmeta = meta.loc[[c for c in _cols if c in meta.index], "genome"]
    prs_all = replicate_pairs(groups["STRICT"], _cols)
    pos = {c: i for i, c in enumerate(_cols)}
    gof = {pos[c]: g for c, g in gmeta.items() if c in pos}
    by_g: dict[str, list] = {}
    for a, b, _k in prs_all:
        by_g.setdefault(gof.get(a, "?"), []).append((a, b))
    print("  %-16s %5s | %-24s | %-24s"
          % ("genome", "pairs", "RAW r / ceiling", "CENTRED r / ceiling"))
    for g in sorted(by_g, key=lambda x: -len(by_g[x])):
        prs = by_g[g]
        if len(prs) < 20:
            continue
        sr = np.median([pair_stats(C[:, a], C[:, b])[0] for a, b in prs])
        sc = np.median([pair_stats(Cc[:, a], Cc[:, b])[0] for a, b in prs])
        print("  %-16s %5d | %+.4f / %.4f          | %+.4f / %.4f"
              % (g, len(prs), sr, ceiling(sr), sc, ceiling(sc)))

    # P1 is the strain the HYPPE track stands on (ML_Dataset, GSE66835).
    # GSE66835 is not part of this release, checked below, so this is the
    # closest available object rather than the same one.
    print("\n  membership control: GSE66835 samples in the release: %d"
          % int((meta["study"] == "GSE66835").sum()))
    p1 = [(a, b) for a, b, _k in replicate_pairs(groups["NOSTUDY"], _cols)
          if gof.get(a) == "P1" and gof.get(b) == "P1"]
    if p1:
        sr = np.median([pair_stats(C[:, a], C[:, b])[0] for a, b in p1])
        sc = np.median([pair_stats(Cc[:, a], Cc[:, b])[0] for a, b in p1])
        print("  P1 (NOSTUDY policy, because every P1 sample has its OWN "
              "study): pairs %d\n    raw      r %+.4f -> ceiling %.4f\n"
              "    centred  r %+.4f -> ceiling %.4f"
              % (len(p1), sr, ceiling(sr), sc, ceiling(sc)))

    print("\n" + "=" * 78)
    print("NEGATIVE-CONTROL VERDICT")
    print("=" * 78)
    for k, v in verdicts.items():
        print("  %-46s %s" % ("%s / %s" % k, "PASSED" if v else "FAILED"))
    print("\nbaselines.py protocol: TARGET=%r, one column = one sample (SRX), "
          "metric = mean per-column Spearman across rows.\n"
          "The binding ceiling is therefore the SINGLE-replicate one, "
          "sqrt(r).\nThe Spearman-Brown number would bind only after the "
          "label changes to the MEAN of n replicates." % "combat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
