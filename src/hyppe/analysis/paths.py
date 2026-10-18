"""Canonical locations of the final HYPPE dataset.

Everything downstream reads its paths from here, so a moved dataset is a
one-line change. Every accessor goes through :func:`require`: a missing
dataset must not be representable as a string, or the failure surfaces
hundreds of lines later as an empty table.
"""
from __future__ import annotations

import os

#: Repository root ``.../HYPPE``; this file is ``ROOT/src/hyppe/analysis``.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

DATA = os.path.join(ROOT, "data")
FINAL = os.path.join(DATA, "final_datasets")
TREES = os.path.join(DATA, "trees")
SPLITS = os.path.join(DATA, "splits")
CACHE = os.path.join(ROOT, "cache")
LOGS = os.path.join(ROOT, "logs")
FIGURES = os.path.join(ROOT, "doc", "figures")
TARGET = os.path.join(ROOT, "target")

# `og_scope` keeps only genes belonging to an orthogroup shared across
# strains, `all_genes_scope` every annotated gene: they differ in row count,
# not in column meaning.
SCOPES = ("og_scope", "all_genes_scope")

# Every matrix has the layout `ID, gene_id, genome, og_id` followed by one
# column per SRX sample.
MATRICES = ("rank", "combat", "counts", "anchor_tef1", "anchor_3genes")

SEQUENCES = os.path.join(FINAL, "sequences", "sequences.tsv.gz")
SEQUENCES_3KB = os.path.join(FINAL, "sequences", "sequences_3kb.tsv.gz")

GENE_INDEX = os.path.join(FINAL, "orthology", "gene_index.tsv")
OG_TABLE = os.path.join(FINAL, "orthology", "og_table_combined.tsv")

# `supertree.tre` carries topology only -- all leaf branches 0.0, three of the
# 45 leaf pairs at distance exactly 0, so any distance-based embedding is
# degenerate; `supermatrix.treefile` has real lengths (0.0014 .. 0.7374).
# Neither ships, and the delivered model reads neither: it is configured with
# `conditioning.genome.source: tree20`.
SPECIES_TREE = os.path.join(TREES, "supermatrix.treefile")
SPECIES_TOPOLOGY_ONLY = os.path.join(TREES, "supertree.tre")
TREE_ANNOTATION = os.path.join(TREES, "annotation.tsv")

#: The 800 bp pks1 promoter, the design target. Not the same file as
#: ``target/pks1_promotor_gen.fasta`` (8202 bp: promoter plus gene body).
TARGET_FASTA = os.path.join(TARGET, "target.fasta")
CLASSICAL_PROMOTERS = os.path.join(TARGET, "classical_promoters.fasta")

#: Panel handed to the hackathon jury; the construct filter calibrates on it.
JURY_SEQUENCES = os.path.join(DATA, "hackathon", "sekwencje_jury.csv")

#: Legacy 800 bp ML dataset, kept because ``promoter_boundary.section_mld``
#: compares it against ``final_datasets``.
ML_DATASET_800 = os.path.join(CACHE, "ml_dataset_800.npz")


def require(path: str, what: str = "") -> str:
    """Return ``path`` if it exists, else raise naming what was wanted."""
    if not os.path.exists(path):
        hint = ""
        # A tree is the one missing input a reader can produce themselves, so
        # say how rather than only that it is absent.
        if os.path.dirname(path) == TREES and path.endswith((".tre",
                                                             ".treefile")):
            hint = ("\nTrees under data/trees/ are built by "
                    "pipelines/species_tree/ (see its README). Note that the "
                    "delivered model does not read this one: it uses "
                    "conditioning.genome.source='tree20', i.e. "
                    "data/trees/external/inhouse_nj_20genomes_aasubst.tre.")
        raise FileNotFoundError(
            "%s not found at %s\n(HYPPE root resolved to %s -- if that is "
            "wrong, this file has been moved out of src/hyppe/analysis/)%s"
            % (what or os.path.basename(path), path, ROOT, hint))
    return path


def matrix_path(scope: str, name: str) -> str:
    """Path to one expression matrix, e.g. ``matrix_path("og_scope", "rank")``."""
    if scope not in SCOPES:
        raise ValueError("scope must be one of %s, got %r" % (SCOPES, scope))
    if name not in MATRICES:
        raise ValueError("matrix must be one of %s, got %r" % (MATRICES, name))
    return require(os.path.join(FINAL, scope, name + ".tsv.gz"),
                   "expression matrix %s/%s" % (scope, name))


def samples_metadata_path(scope: str = "og_scope") -> str:
    """The canonical per-sample table ``data/samples_metadata.tsv``.

    `scope` is accepted and ignored -- the table describes samples, the same
    2332 columns in both gene scopes -- but a wrong scope name still raises
    rather than silently select a different table. The table is the superset
    of three earlier copies that disagreed on ``condition`` in 2300 of 2332
    rows. New code should prefer ``hyppe.data.metadata.load()``.
    """
    if scope not in SCOPES:
        raise ValueError("scope must be one of %s, got %r" % (SCOPES, scope))
    canonical = os.path.join(DATA, "samples_metadata.tsv")
    if not os.path.exists(canonical):
        raise FileNotFoundError(
            "canonical sample metadata not found at %s\n(HYPPE root resolved "
            "to %s)\nBuild it with:  python src/scripts/build_samples_metadata.py"
            % (canonical, ROOT))
    return canonical


def samples_metadata_legacy_path() -> str:
    """The 18-column copy in the old tree; superseded, for reproduction only.

    The reported noise ceiling and the batch- and strain-effect numbers were
    measured on it; for new work ``hyppe.data.metadata.legacy_column()``
    returns its four divergent columns out of the canonical table.
    """
    return require(os.path.abspath(os.path.join(
        ROOT, "..", "HYperbolicPromoterProgrammingEngine", "final_datasets",
        "og_scope", "samples_metadata.tsv")), "legacy samples_metadata.tsv")


#: Where the superseded sample-metadata copies went. Outside the HYPPE tree
#: on purpose: exactly one such table may exist under HYPPE.
ARCHIVE = os.path.abspath(os.path.join(
    ROOT, "..", "_archiwum", "2026-09-12", "samples_metadata"))


def samples_metadata_curated_path() -> str:
    """Removed: the curated copy no longer exists under HYPPE, so this raises.

    A raising stub rather than a deletion, so a caller still asking for it
    gets the replacement named instead of a path that silently resolves to a
    different table.
    """
    raise FileNotFoundError(
        "samples_metadata_curated.tsv no longer exists under HYPPE (archived "
        "2026-09-12 to %s).\nIts columns are all in the canonical table %s -- "
        "read `condition_category_curated` with "
        "hyppe.data.metadata.load(['sample', 'condition_category_curated'])."
        % (ARCHIVE, os.path.join(DATA, "samples_metadata.tsv")))


def ensure_dir(path: str) -> str:
    """Create a write destination (figures, cache, logs) and return it.

    Separate from :func:`require` on purpose: creating a directory we are
    about to read from would turn a missing dataset into an empty result.
    """
    os.makedirs(path, exist_ok=True)
    return path
