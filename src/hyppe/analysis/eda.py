"""Exploratory data analysis for the final HYPPE release.

Measured rather than assumed, in six sections: the cohort (genomes, samples,
studies, conditions); the row object of each scope and how values are laid
out; the targets, `rank` against `combat`, and which way rank runs; how much
of the variance is study or batch rather than biology; whether the species
tree covers the genomes we have; and promoter sequence statistics, which are
delegated to :mod:`hyppe.analysis.sequence_audit`.

    python -m hyppe.analysis.eda        # figures to doc/figures
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import matrices as data  # noqa: E402
from . import paths

GENE_SCOPE = "all_genes_scope"
OG_SCOPE = "og_scope"


def _save(fig, name):
    p = os.path.join(paths.ensure_dir(paths.FIGURES), name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("    figure -> %s" % os.path.relpath(p, paths.ROOT))


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def cohort(samples: pd.DataFrame) -> None:
    section("1. COHORT")
    print("samples %d | genomes %d | studies %d | unique condition texts %d"
          % (len(samples), samples["genome"].nunique(),
             samples["study"].nunique(), samples["condition"].nunique()))

    per_genome = samples["genome"].value_counts()
    print("\nsamples per genome:")
    print(per_genome.to_string())
    print("\nimbalance: largest genome holds %.1f %% of all samples; "
          "the 10 smallest hold %.1f %% together"
          % (100 * per_genome.iloc[0] / len(samples),
             100 * per_genome.tail(10).sum() / len(samples)))

    print("\ncondition categories:")
    print(samples["condition_category"].value_counts().to_string())
    print("\nmutant backgrounds (top 8):")
    print(samples["mutant"].value_counts().head(8).to_string())

    ss = samples["study"].value_counts()
    print("\nstudy sizes: median %d, mean %.1f, max %d; %d of %d studies have "
          "a SINGLE sample" % (ss.median(), ss.mean(), ss.max(),
                               int((ss == 1).sum()), len(ss)))
    print("-> with 1 sample in %.0f %% of studies, a study-held-out split "
          "leaves many conditions with no within-study replicate at all."
          % (100 * (ss == 1).mean()))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    per_genome.plot.bar(ax=ax[0], color="#3b6ea5")
    ax[0].set_title("samples per genome")
    ax[0].set_ylabel("samples")
    samples["condition_category"].value_counts().plot.barh(
        ax=ax[1], color="#a55b3b")
    ax[1].set_title("condition category")
    ax[2].hist(ss.values, bins=40, color="#4f8a4f")
    ax[2].set_yscale("log")
    ax[2].set_title("study size")
    ax[2].set_xlabel("samples per study")
    _save(fig, "01_cohort.png")


def layout(samples: pd.DataFrame) -> tuple[pd.Index, list[str], np.ndarray]:
    section("2. WHAT EACH SCOPE ACTUALLY CONTAINS")
    for sc in (GENE_SCOPE, OG_SCOPE):
        hdr = data.matrix_header(sc, "rank")
        meta, cols = data.split_columns(hdr)
        print("%-16s row object = %-11s | metadata cols %s | samples %d"
              % (sc, data.row_object(sc), meta, len(cols)))

    print("\n-- gene-level scope: is it block sparse? --")
    bs = data.probe_block_structure(GENE_SCOPE, "rank", max_rows=20000)
    print("probed %d gene rows x %d sample columns" % (bs.n_rows, bs.n_samples))
    print("fraction of filled cells whose sample genome == gene genome: %.6f"
          % bs.own_genome_fraction)
    print("filled cells OUTSIDE the gene's own genome: %d" % bs.off_genome_cells)
    print("-> %s" % ("BLOCK SPARSE CONFIRMED: a gene is only measured in "
                     "samples of its own genome."
                     if bs.own_genome_fraction > 0.999 else
                     "NOT block sparse -- cross-genome cells exist."))

    print("\n-- orthogroup scope: how dense is it? --")
    ids, cols, M = data.load_orthogroup_matrix(OG_SCOPE, "rank")
    filled = np.isfinite(M)
    print("orthogroups %d x samples %d | filled %.2f %%"
          % (M.shape[0], M.shape[1], 100 * filled.mean()))
    per_og = filled.sum(1)
    print("samples per orthogroup: min %d median %d max %d"
          % (per_og.min(), int(np.median(per_og)), per_og.max()))
    print("orthogroups present in ALL samples: %d (%.1f %%)"
          % (int((per_og == M.shape[1]).sum()),
             100 * (per_og == M.shape[1]).mean()))
    print("-> this is the CROSS-SPECIES object: one row comparable across "
          "the whole phylogeny, which the gene-level scope can never be.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(per_og, bins=50, color="#3b6ea5")
    ax[0].set_title("samples measured per orthogroup")
    ax[0].set_xlabel("samples")
    gsz = pd.Series([len(v) for v in bs.filled_per_row.values()],
                    index=list(bs.filled_per_row))
    gsz.sort_values(ascending=False).plot.bar(ax=ax[1], color="#a55b3b")
    ax[1].set_title("genes probed per genome (first 20k rows)")
    _save(fig, "02_layout.png")
    return ids, cols, M


def targets(samples, og_ids, cols, R):
    section("3. TARGETS: rank vs combat")
    ids_c, cols_c, C = data.load_orthogroup_matrix(OG_SCOPE, "combat")
    assert list(og_ids) == list(ids_c) and cols == cols_c, \
        "rank and combat must be aligned before they are compared"

    ok = np.isfinite(R) & np.isfinite(C)
    rho = np.corrcoef(R[ok], C[ok])[0, 1]
    print("rank  : min %.4f max %.4f mean %.4f | NaN %.2f %%"
          % (np.nanmin(R), np.nanmax(R), np.nanmean(R), 100 * np.isnan(R).mean()))
    print("combat: min %.4f max %.4f mean %.4f | NaN %.2f %%"
          % (np.nanmin(C), np.nanmax(C), np.nanmean(C), 100 * np.isnan(C).mean()))
    print("\nPearson(rank, combat) over %d cells = %+.4f" % (ok.sum(), rho))

    # The sign of a correlation is easy to misread as a direction, so read
    # the direction off the data: what do rows at each end of the rank scale
    # actually express?
    r_flat, c_flat = R[ok], C[ok]
    lo = c_flat[r_flat <= 0.05].mean()
    hi = c_flat[r_flat >= 0.95].mean()
    print("mean combat at rank <= 0.05 : %+7.3f" % lo)
    print("mean combat at rank >= 0.95 : %+7.3f" % hi)
    if hi < lo:
        print("-> rank is DESCENDING: rank 1 is the SILENT row, rank 0 the "
              "most expressed one.  A loss written as if rank were expression "
              "would optimise the wrong direction.  This agrees with the "
              "project memory `rank in final_datasets is descending`.")
    else:
        print("-> rank is ASCENDING: rank 1 is the most expressed row.  NOTE: "
              "this CONTRADICTS the project memory, so it must be resolved "
              "before any target is used.")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(R[ok], bins=60, color="#3b6ea5")
    ax[0].set_title("rank")
    ax[1].hist(C[ok], bins=60, color="#a55b3b")
    ax[1].set_title("combat")
    idx = np.random.default_rng(0).choice(np.flatnonzero(ok.ravel()),
                                          size=min(60000, int(ok.sum())),
                                          replace=False)
    ax[2].hexbin(R.ravel()[idx], C.ravel()[idx], gridsize=60, bins="log",
                 cmap="viridis")
    ax[2].set_xlabel("rank")
    ax[2].set_ylabel("combat")
    ax[2].set_title("rank vs combat  (r = %+.3f)" % rho)
    _save(fig, "03_targets.png")
    return C


def variance_decomposition(samples, cols, C) -> None:
    section("4. WHERE THE VARIANCE SITS")
    meta = samples.loc[cols]
    ok = np.isfinite(C)
    row_mean = np.nanmean(C, axis=1, keepdims=True)
    total = float(np.nanvar(C[ok]))
    centred = C - row_mean
    resid = float(np.nanvar(centred[ok]))
    print("total variance of combat        %.4f" % total)
    print("after removing the ROW mean     %.4f  -> the orthogroup identity "
          "alone explains %.1f %%" % (resid, 100 * (1 - resid / total)))

    shares = {}
    for key in ("genome", "study", "condition_category", "mutant"):
        lab = meta[key].fillna("?").values
        acc = np.zeros_like(centred)
        for v in pd.unique(lab):
            m = lab == v
            if m.sum() < 2:
                continue
            acc[:, m] = np.nanmean(centred[:, m], axis=1, keepdims=True)
        expl = 1 - float(np.nanvar((centred - acc)[ok])) / resid
        shares[key] = 100 * expl
        print("  %-20s explains %5.1f %% of the row-centred variance"
              % (key, 100 * expl))
    print("-> `study` is the batch axis.  If it explains much more than "
          "`condition_category`, the model can score well by memorising "
          "batches, and the split MUST hold out whole studies.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    pd.Series(shares).plot.bar(ax=ax[0], color="#6a4fa5")
    ax[0].set_ylabel("% of row-centred variance")
    ax[0].set_title("variance explained by each factor")
    cats = meta["condition_category"].fillna("?")
    order = list(cats.value_counts().index)
    ax[1].boxplot([np.nanmean(C[:, (cats == c).values], axis=1) for c in order],
                  labels=order, showfliers=False)
    ax[1].set_ylabel("mean combat")
    ax[1].set_title("expression by condition category")
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
    _save(fig, "04_variance.png")


def phylogeny(samples: pd.DataFrame) -> None:
    section("5. SPECIES TREE COVERAGE")
    ann = data.load_tree_annotation()
    nwk = data.read_newick()
    leaves = [t.split(":")[0].strip() for t in
              nwk.replace("(", ",").replace(")", ",").split(",")
              if "_protein" in t]
    print("tree leaves: %d" % len(leaves))
    print(ann.to_string(index=False))
    genomes = sorted(samples["genome"].unique())
    print("\ngenomes in the expression data: %d" % len(genomes))
    print(", ".join(genomes))
    print("\n-> %d leaves for %d genomes.  Genomes are STRAINS; several "
          "strains collapse onto one leaf and some have no leaf at all. "
          "A phylogenetic embedding needs an explicit strain -> leaf map, "
          "and strains sharing a leaf get IDENTICAL coordinates unless the "
          "model separates them another way." % (len(leaves), len(genomes)))


def sequences() -> None:
    """A5 lives in :mod:`hyppe.analysis.sequence_audit`; this delegates.

    The inline version was invalid three times over (a prefix of one strain
    instead of a sample, no `GROUP` filter, GC counted on upper case only)
    and was deleted rather than kept to be fixed later: a wrong measurement
    that still runs is worse than none.
    """
    section("6. PROMOTER SEQUENCES  (delegated to sequence_audit)")
    from . import sequence_audit
    rc = sequence_audit.main([])
    if rc:
        print("A5 falsifier fired (exit %d) -- see the log." % rc)


def main() -> None:
    samples = data.load_samples(OG_SCOPE)
    cohort(samples)
    og_ids, cols, R = layout(samples)
    C = targets(samples, og_ids, cols, R)
    variance_decomposition(samples, cols, C)
    phylogeny(samples)
    sequences()
    print("\nDone.  Figures in %s" % os.path.relpath(paths.FIGURES, paths.ROOT))


if __name__ == "__main__":
    main()
