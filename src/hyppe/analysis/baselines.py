"""Sanity checks, evaluation protocol and baselines.

The task is "given a context (condition + species), which promoter is best",
which ranks rows within a column, so the headline metric is the mean
per-column Spearman. A global correlation is dominated by the orthogroup main
effect -- 79.2 % of total variance in the EDA -- which says nothing about
choosing between promoters for one experiment.

Three held-out designs, asking different questions. A holds out whole studies
(columns), rows seen: can we score a promoter in an experiment never run? B
holds out whole orthogroups (rows), columns seen: can we score a promoter
never measured? C holds out both at once, which is the one that matches
designing a new promoter for a new condition, and the hard one.

    python -m hyppe.analysis.baselines
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from . import matrices as data  # noqa: E402
from . import paths

SCOPE = "og_scope"
TARGET = "combat"          # `rank` is descending; `combat` is expression
#: One generator for the whole module, consumed by `split_columns_by_study`
#: and `split_rows` in call order. That makes a run reproducible only as
#: long as `main()` is unchanged: adding, removing or reordering a design
#: shifts every later split, and with it every number below it. Deliberately
#: not reseeded per call, because that would silently give two designs the
#: same split and hide a difference between them. The published baseline
#: table comes from `scripts/run_baselines.py`, which uses `hyppe.baselines`
#: rather than this module, so those figures do not depend on this.
RNG = np.random.default_rng(0)


# Metrics
def column_spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Spearman over columns: 'rank the promoters within a context'."""
    out = []
    for j in range(y.shape[1]):
        m = np.isfinite(y[:, j]) & np.isfinite(p[:, j])
        if m.sum() < 20 or np.std(p[m, j]) < 1e-12:
            continue
        out.append(spearmanr(y[m, j], p[m, j]).statistic)
    return float(np.nanmean(out)) if out else float("nan")


def row_spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Spearman over rows: 'rank the contexts for one promoter'."""
    out = []
    for i in range(y.shape[0]):
        m = np.isfinite(y[i]) & np.isfinite(p[i])
        if m.sum() < 20 or np.std(p[i, m]) < 1e-12:
            continue
        out.append(spearmanr(y[i, m], p[i, m]).statistic)
    return float(np.nanmean(out)) if out else float("nan")


def global_pcc(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 10 or np.std(p[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y[m], p[m])[0, 1])


def _fmt(v, const):
    """NaN here is a result, not a failure: a prediction that is constant
    along the axis being ranked cannot rank at all."""
    if np.isfinite(v):
        return "%+.4f" % v
    return "  n/a  " if not const else "CANNOT RANK"


def report(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    d = {"baseline": name, "col_spearman": column_spearman(y, p),
         "row_spearman": row_spearman(y, p), "pcc": global_pcc(y, p)}
    const_c = bool(np.nanmax(np.nanstd(p, axis=0)) < 1e-12)
    const_r = bool(np.nanmax(np.nanstd(p, axis=1)) < 1e-12)
    print("  %-34s col-Spearman %-11s | row-Spearman %-11s | PCC %s"
          % (name, _fmt(d["col_spearman"], const_c),
             _fmt(d["row_spearman"], const_r), _fmt(d["pcc"], False)))
    return d


# Sanity checks
def sanity(M: np.ndarray, cols: list[str], samples: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("SANITY CHECKS")
    print("=" * 74)
    meta = samples.loc[cols]

    nan = ~np.isfinite(M)
    print("1. missingness: %.2f %% of cells are NaN" % (100 * nan.mean()))
    per_col = nan.mean(0)
    by_genome = pd.Series(per_col, index=cols).groupby(meta["genome"].values).mean()
    print("   NaN rate per genome (top 5):")
    print(by_genome.sort_values(ascending=False).head(5).to_string())
    print("   -> if NaN concentrates in a few genomes it is STRUCTURAL "
          "(gene absent from that strain), not random, and imputing it with a "
          "column mean invents expression for a gene that does not exist.")

    print("\n2. constant rows/columns")
    sd_r = np.nanstd(M, axis=1)
    sd_c = np.nanstd(M, axis=0)
    print("   rows with sd == 0: %d | columns with sd == 0: %d"
          % (int((sd_r == 0).sum()), int((sd_c == 0).sum())))

    print("\n3. duplicate samples (same study+condition+genome)")
    key = (meta["study"].fillna("") + "|" + meta["condition"].fillna("")
           + "|" + meta["genome"].fillna(""))
    dup = key.duplicated(keep=False)
    print("   %d of %d samples share a key with another sample"
          % (int(dup.sum()), len(key)))
    print("   -> duplicates across a train/test boundary are leakage: the "
          "model can copy a near-identical column it has already seen.")

    print("\n4. does the study split actually separate anything?")
    st = meta["study"]
    print("   studies %d | samples %d | largest study %d samples"
          % (st.nunique(), len(st), st.value_counts().iloc[0]))
    genomes_per_study = meta.groupby("study")["genome"].nunique()
    print("   studies spanning >1 genome: %d"
          % int((genomes_per_study > 1).sum()))
    print("   -> a study is almost always one genome, so holding out studies "
          "ALSO shifts the genome mix; that confound must be reported.")

    print("\n5. sequence sanity")
    # A prefix, not a sample, knowingly: the file is ordered by genome, so
    # these 40 000 rows are one strain and four record types. Only the claim
    # about shape (are lengths uniform?) survives that bias, and nothing else
    # is read off it; corpus statistics live in `sequence_audit`.
    rows = list(data.stream_sequences(limit=40000))
    lens = np.array([len(r[3]) for r in rows])
    bad = int((lens < 100).sum())
    print("   sampled %d | length median %d | shorter than 100 bp: %d (%.2f %%)"
          % (len(lens), int(np.median(lens)), bad, 100 * bad / len(lens)))
    print("   longest %d bp -- lengths are NOT uniform, so the model needs a "
          "fixed window (crop/pad) and the choice must be justified."
          % lens.max())


# Splits
def split_columns_by_study(cols, samples, frac=0.2):
    st = samples.loc[cols, "study"].values
    uniq = pd.unique(st)
    held = set(RNG.choice(uniq, size=max(1, int(len(uniq) * frac)),
                          replace=False))
    test = np.array([s in held for s in st])
    return ~test, test


def split_rows(n_rows, frac=0.2):
    idx = RNG.permutation(n_rows)
    k = int(n_rows * frac)
    test = np.zeros(n_rows, dtype=bool)
    test[idx[:k]] = True
    return ~test, test


# Baselines
def fit_predict(train: np.ndarray, shape, kind: str,
                col_group: np.ndarray | None = None) -> np.ndarray:
    """All baselines are fitted on the train cells only."""
    g = np.nanmean(train)
    row = np.nanmean(train, axis=1, keepdims=True)
    col = np.nanmean(train, axis=0, keepdims=True)
    row = np.where(np.isfinite(row), row, g)
    col = np.where(np.isfinite(col), col, g)
    if kind == "global mean":
        return np.full(shape, g, dtype=np.float32)
    if kind == "row mean (orthogroup)":
        return np.broadcast_to(row, shape).copy()
    if kind == "column mean (sample)":
        return np.broadcast_to(col, shape).copy()
    if kind == "row + column (additive)":
        return (row + col - g).astype(np.float32)
    if kind == "row + genome":
        out = np.broadcast_to(row, shape).copy()
        for v in pd.unique(col_group):
            m = col_group == v
            eff = np.nanmean(train[:, m] - row)
            if np.isfinite(eff):
                out[:, m] += eff
        return out
    raise ValueError(kind)


BASELINES = ("global mean", "row mean (orthogroup)", "column mean (sample)",
             "row + column (additive)", "row + genome")


def evaluate_design(name, M, cols, samples, hold_rows, hold_cols):
    print("\n" + "-" * 74)
    print("DESIGN %s" % name)
    print("-" * 74)
    tr_r, te_r = (split_rows(M.shape[0]) if hold_rows
                  else (np.ones(M.shape[0], bool), np.ones(M.shape[0], bool)))
    tr_c, te_c = (split_columns_by_study(cols, samples) if hold_cols
                  else (np.ones(M.shape[1], bool), np.ones(M.shape[1], bool)))
    train = M.copy()
    if hold_rows:
        train[te_r, :] = np.nan
    if hold_cols:
        train[:, te_c] = np.nan
    y = M[np.ix_(te_r, te_c)]
    print("train cells %d | test block %s | test cells finite %d"
          % (int(np.isfinite(train).sum()), y.shape, int(np.isfinite(y).sum())))
    cg = samples.loc[cols, "genome"].values
    out = []
    for b in BASELINES:
        p_full = fit_predict(train, M.shape, b, cg)
        out.append(report(b, y, p_full[np.ix_(te_r, te_c)]))
    for d in out:
        d["design"] = name
    return out


def main():
    samples = data.load_samples(SCOPE)
    ids, cols, M = data.load_orthogroup_matrix(SCOPE, TARGET)
    print("target=%s | matrix %s | finite %.2f %%"
          % (TARGET, M.shape, 100 * np.isfinite(M).mean()))
    sanity(M, cols, samples)

    print("\n" + "=" * 74)
    print("BASELINES  (headline metric = col-Spearman: rank promoters within "
          "a context)")
    print("=" * 74)
    rows = []
    rows += evaluate_design("A. new context (hold out studies)", M, cols,
                            samples, hold_rows=False, hold_cols=True)

    # The 0.89 above is the constitutive effect: some orthogroups are simply
    # expressed more than others everywhere, which a promoter designer
    # already knows. Design A is repeated on row-centred values to see what
    # is left after it -- the context-specific part.
    print("\n" + "-" * 74)
    print("DESIGN A' -- same split, target ROW-CENTRED "
          "(constitutive level removed)")
    print("-" * 74)
    Mc = M - np.nanmean(M, axis=1, keepdims=True)
    rows += [dict(r, design="A'. new context, row-centred")
             for r in evaluate_design("A'. context-specific only", Mc, cols,
                                      samples, hold_rows=False,
                                      hold_cols=True)]
    rows += evaluate_design("B. new promoter (hold out orthogroups)", M, cols,
                            samples, hold_rows=True, hold_cols=False)
    rows += evaluate_design("C. both (new promoter AND new context)", M, cols,
                            samples, hold_rows=True, hold_cols=True)

    df = pd.DataFrame(rows)
    paths.ensure_dir(paths.FIGURES)
    piv = df.pivot(index="baseline", columns="design", values="col_spearman")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    piv.plot.bar(ax=ax)
    ax.set_ylabel("mean per-column Spearman")
    ax.set_title("baselines by held-out design (target=%s)" % TARGET)
    ax.axhline(0, color="k", lw=0.8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(paths.FIGURES, "06_baselines.png"), dpi=130)
    plt.close(fig)
    print("\n    figure -> %s"
          % os.path.relpath(os.path.join(paths.FIGURES,
                                         "06_baselines.png"), paths.ROOT))
    print("\nWHAT THE MODEL MUST BEAT: the strongest baseline of design C. "
          "Anything that only beats design A or B is exploiting a main effect "
          "that a promoter designer already has for free.")


if __name__ == "__main__":
    main()
