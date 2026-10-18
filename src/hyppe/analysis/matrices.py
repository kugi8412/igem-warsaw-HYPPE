"""Loaders for the final HYPPE dataset, as the measurement code needs them.

The expression matrices are block sparse: a gene row and a sample column each
belong to one genome, and a cell carries a value only when the two agree.
Loading one densely would materialise 239k x 2332 mostly empty cells, so
:func:`load_expression_block` streams the file once and returns one dense
sub-matrix per genome; :func:`probe_block_structure` measures that layout
rather than assuming it.

:func:`load_samples` duplicates ``hyppe.data.labels.load_samples`` on purpose:
that one applies the sample policy and returns what the training heads may
see, this one the raw 2332 rows. The noise ceiling keys replicate groups on
``mutant``, which the policy has already removed.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths

# The two scopes do not share a row object: `all_genes_scope` rows are genes
# (ID, gene_id, genome, og_id; block sparse), `og_scope` rows are orthogroups
# (og_id alone, and one row can carry values across every genome). Metadata
# columns are therefore detected, not hard-coded.
KNOWN_META = ("ID", "gene_id", "genome", "og_id")


# Metadata
def load_samples(scope: str = "og_scope") -> pd.DataFrame:
    """Per-sample metadata table, indexed by SRX accession; raw, unfiltered."""
    df = pd.read_csv(paths.samples_metadata_path(scope), sep="\t", dtype=str)
    df = df.set_index("sample", drop=False)
    if "mapping_rate" in df.columns:
        df["mapping_rate"] = pd.to_numeric(df["mapping_rate"], errors="coerce")
    return df


def load_gene_index() -> pd.DataFrame:
    """Gene ID -> (genome, gene_id)."""
    return pd.read_csv(paths.require(paths.GENE_INDEX, "gene_index.tsv"),
                       sep="\t", dtype=str)


def load_tree_annotation() -> pd.DataFrame:
    """Tree leaf label -> readable organism name."""
    return pd.read_csv(paths.require(paths.TREE_ANNOTATION, "annotation.tsv"),
                       sep="\t", dtype=str)


def read_newick(path: str | None = None) -> str:
    return io.open(paths.require(path or paths.SPECIES_TREE, "species tree"),
                   encoding="utf-8").read().strip()


# Expression matrices
def matrix_header(scope: str, name: str) -> list[str]:
    """Column names without reading the body."""
    with gzip.open(paths.matrix_path(scope, name), "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")


def split_columns(header: list[str]) -> tuple[list[str], list[str]]:
    """(metadata columns, sample columns) for either scope layout."""
    meta = [c for c in header if c in KNOWN_META]
    return meta, [c for c in header if c not in KNOWN_META]


def sample_columns(scope: str, name: str) -> list[str]:
    return split_columns(matrix_header(scope, name))[1]


def row_object(scope: str, name: str = "rank") -> str:
    """"gene" if the matrix is gene-level, "orthogroup" otherwise."""
    return "gene" if "genome" in matrix_header(scope, name) else "orthogroup"


@dataclass
class BlockStructure:
    """How values are distributed over the (gene genome x sample) grid."""

    genomes: list[str]
    n_rows: int
    n_samples: int
    filled_per_row: dict[str, np.ndarray] = field(default_factory=dict)
    own_genome_fraction: float = float("nan")
    off_genome_cells: int = 0


def probe_block_structure(scope: str = "all_genes_scope", name: str = "rank",
                          max_rows: int | None = 20000) -> BlockStructure:
    """Measure -- not assume -- which cells carry values.

    The default is the gene-level scope, the only one this question applies
    to: `og_scope` rows are orthogroups and the raise below is unconditional
    for them, so advertising it in the signature promised a call that cannot
    run.

    Falsifier: `own_genome_fraction == 1.0` proves the block layout; anything
    less means genes are measured across genomes and
    :func:`load_expression_block` silently drops data.
    """
    samples = load_samples(scope)
    header = matrix_header(scope, name)
    _meta, cols = split_columns(header)
    # Before `col_genome`, which the guard exists to protect.
    if "genome" not in header:
        raise ValueError(
            "scope %r is %s-level and has no `genome` column; the block "
            "question only applies to the gene-level scope"
            % (scope, row_object(scope, name)))
    col_genome = np.array([samples["genome"].get(c, "?") for c in cols])
    idx = {c: i for i, c in enumerate(header)}
    g_at = idx["genome"]
    n_meta = len(_meta)

    per_genome: dict[str, list[int]] = {}
    own_hits = 0
    off_hits = 0
    n_rows = 0
    with gzip.open(paths.matrix_path(scope, name), "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            genome = parts[g_at]
            vals = np.array(parts[n_meta:], dtype=object)
            if vals.shape[0] < col_genome.shape[0]:      # ragged tail
                vals = np.concatenate(
                    [vals, np.array([""] * (col_genome.shape[0] - vals.shape[0]),
                                    dtype=object)])
            filled = vals[:col_genome.shape[0]] != ""
            own = filled & (col_genome == genome)
            own_hits += int(own.sum())
            off_hits += int((filled & ~own).sum())
            per_genome.setdefault(genome, []).append(int(filled.sum()))
            n_rows += 1
            if max_rows is not None and n_rows >= max_rows:
                break
    total = own_hits + off_hits
    return BlockStructure(
        genomes=sorted(per_genome),
        n_rows=n_rows,
        n_samples=len(cols),
        filled_per_row={g: np.array(v) for g, v in per_genome.items()},
        own_genome_fraction=(own_hits / total) if total else float("nan"),
        off_genome_cells=off_hits,
    )


def load_expression_block(genome: str, scope: str = "og_scope",
                          name: str = "rank") -> tuple[pd.Index, list[str], np.ndarray]:
    """Dense sub-matrix for one genome: (gene ids, sample ids, values).

    float32, NaN for missing cells, rows the genes of `genome` and columns
    its samples.
    """
    samples = load_samples(scope)
    header = matrix_header(scope, name)
    meta_cols, cols = split_columns(header)
    keep = [i for i, c in enumerate(cols)
            if samples["genome"].get(c, "?") == genome]
    keep_names = [cols[i] for i in keep]
    idx = {c: i for i, c in enumerate(header)}
    g_at, id_at = idx["genome"], idx["ID"]

    ids: list[str] = []
    rows: list[np.ndarray] = []
    with gzip.open(paths.matrix_path(scope, name), "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[g_at] != genome:
                continue
            vals = parts[len(meta_cols):]
            row = np.full(len(keep), np.nan, dtype=np.float32)
            for j, k in enumerate(keep):
                if k < len(vals) and vals[k] != "":
                    row[j] = float(vals[k])
            ids.append(parts[id_at])
            rows.append(row)
    return pd.Index(ids, name="ID"), keep_names, np.vstack(rows) if rows else \
        np.zeros((0, len(keep)), dtype=np.float32)


# Sequences
def stream_sequences(path: str | None = None, limit: int | None = None):
    """Yield (ID, species, genome, sequence, group, og_id, gene_id).

    Kept verbatim from the legacy tree: new code should call
    ``hyppe.data.sequences.iter_sequence_tsv``, but `eda` and `baselines`
    reported numbers from a reader that applies no ``.upper()`` (7.4 % of
    bases are soft-masked, enough to bias a GC count and reorder a ranking)
    and no ``GROUP`` filter, while the file interleaves FlankLeft / Promotor
    / Gene / FlankRight. `limit` is a prefix of a genome-ordered file, so it
    is a debugging convenience and never a sample.
    """
    p = paths.require(path or paths.SEQUENCES, "sequences.tsv.gz")
    with gzip.open(p, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pos = {n: i for i, n in enumerate(header)}
        # A missing column is an error: `pos.get("GROUP", 0)` would hand the
        # caller column 0 (`ID`) in the group slot, and every
        # `group == "Promotor"` filter would silently match nothing.
        missing = [c for c in ("ID", "SPECIES", "GENOME", "SEQUENCE",
                               "GROUP", "OG_ID", "GENE_ID") if c not in pos]
        if missing:
            raise KeyError("%s lacks column(s) %s; header = %s"
                           % (p, missing, header))
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            yield (parts[pos["ID"]], parts[pos["SPECIES"]], parts[pos["GENOME"]],
                   parts[pos["SEQUENCE"]], parts[pos["GROUP"]],
                   parts[pos["OG_ID"]], parts[pos["GENE_ID"]])


def sequence_lengths(path: str | None = None, limit: int = 50000) -> np.ndarray:
    return np.array([len(r[3]) for r in stream_sequences(path, limit)],
                    dtype=np.int32)


# Orthogroup-level view, the cross-species one
def load_orthogroup_matrix(scope: str = "og_scope", name: str = "rank",
                           max_rows: int | None = None):
    """Dense orthogroup x sample matrix.

    Not block sparse, unlike the gene-level scope: an orthogroup is present
    in many strains, so one row is comparable across the phylogeny.
    """
    header = matrix_header(scope, name)
    meta_cols, cols = split_columns(header)
    ids: list[str] = []
    rows: list[np.ndarray] = []
    n_meta = len(meta_cols)
    with gzip.open(paths.matrix_path(scope, name), "rt") as fh:
        fh.readline()
        for n, line in enumerate(fh):
            if max_rows is not None and n >= max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            vals = parts[n_meta:]
            row = np.full(len(cols), np.nan, dtype=np.float32)
            for j in range(min(len(vals), len(cols))):
                if vals[j] != "":
                    row[j] = float(vals[j])
            rows.append(row)
    return pd.Index(ids, name=meta_cols[0]), cols, np.vstack(rows)
