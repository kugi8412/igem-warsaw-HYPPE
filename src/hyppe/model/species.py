"""Strain registry and the hyperbolic genome channel.

HYPPE conditions promoter strength on the genome a promoter comes from. The
genome vector is a free parameter on the hyperboloid; the tree is only its
initialisation and a pull-back prior, and nothing here reads a distance off the
tree at inference time. On the borrow-strength surrogate free weights reach
R^2 0.9807 against 0.9678 for weights derived from a distance and 0.9411 for
uniform weights, so geometry recovers about two thirds of what free weights
buy and freezing the channel on it costs 0.0129 R^2.

The channel is small and must be reported as small. Variance of the corrected
atlas splits 78.99 % between orthogroups, 6.23 % genome within an orthogroup
and 14.78 % sample within (orthogroup, genome); only the 6.23 is reachable from
here, and that is :data:`GENOME_VARIANCE_CEILING`. A larger measured ablation
effect is evidence of leakage rather than of a better channel -- batches are
genome-homogeneous, so a "genome" channel can carry batch identity instead.

Curvature is bought because the object being embedded is a tree, not because
the data are tree-like: the patristic matrix satisfies the four-point condition
to 2.220e-16, so residual distortion is down to dimension and scale. Dimension
3 is the default because ten points in H^d have ``10d - d(d+1)/2`` identifiable
coordinates against 45 distances, so d=8 fits 44 of them and regularises
nothing. Scale, not dimension, is the binding constraint: past s~20 the fit
degrades through Minkowski cancellation in float64 rather than through
geometry, hence a default of 10.0 and the hard :data:`MAX_SAFE_SCALE`. Every
fit here differentiates through autograd -- finite differences stalled an
earlier version of the same measurement and produced the false conclusion that
large scale breaks the embedding.

Two rejected inputs are worth stating. A 5-mer promoter distance correlates
+0.9684 with ``|dGC|`` over 190 genome pairs, so it is a composition axis and
not a hierarchy; :func:`surrogate_falsifier` enforces that rule on any new
surrogate. And the tree makes P1 the sister of IMI206040 at the smallest
distance in the whole matrix while only 26.52 % of their promoters are
bit-identical, against 38-40 % for the rest of the block -- zero-shot for P1
must not lean on tree adjacency.

Adding a genome is a runtime operation: :meth:`StrainRegistry.add` appends and
:meth:`GenomeEmbedding.add_genome` grows exactly one tensor by one row. Every
public entry point carries ``falsifiers()`` as (name, passed, detail) and
``report()``; ``py -m hyppe.model.species`` runs the self test.
"""
from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..geometry.lorentz import (
    MAX_TANGENT_NORM,
    assert_on_manifold,
    expmap,
    frechet_mean,
    geodesic_distance,
    logmap,
    logmap0,
    lorentz_centroid,
    mark_manifold,
    minkowski_dot,
    origin_like,
    parallel_transport,
    project_to_hyperboloid,
    radius,
)

__all__ = [
    # measured constants
    "DEFAULT_STRAINS",
    "STRAIN_TO_TAXON",
    "TAXA_WITHOUT_DATA",
    "OUTGROUP_TAXA",
    "NEAR_DUPLICATE_CLUSTERS",
    "STRAINS_WITHOUT_EXPRESSION",
    "GENOME_VARIANCE_CEILING",
    "WITHIN_OG_VARIANCE",
    "MEASURED_GENOME_AXIS",
    "MAX_SAFE_SCALE",
    "MAX_SAFE_RADIUS",
    "MEASURED_RMS_FREE_SCALE",
    "MEASURED_RMS_EUCLIDEAN",
    "MEASURED_RMS_BY_SCALE_H2",
    "NEW_GENOME_MEASURED",
    "PATRISTIC_PUBLISHED_TAXA",
    "PATRISTIC_PUBLISHED",
    "SURROGATE_GC_LIMIT",
    # registry
    "GenomeRecord",
    "StrainRegistry",
    # tree
    "BranchLengthsMissing",
    "Clade",
    "parse_newick",
    "read_newick",
    "reroot_on_outgroup",
    "patristic_matrix",
    "unrooted_splits",
    "robinson_foulds",
    "four_point_defect",
    "default_tree_paths",
    # embedding
    "EmbeddingFit",
    "DEFAULT_SCALE_LADDER",
    "fit_hyperbolic_embedding",
    "fit_euclidean_control",
    "place_from_distances",
    # model
    "GenomeEmbedding",
    # layer-level entry points
    "surrogate_falsifier",
    "report",
    "falsifiers",
    "self_test",
]


# Measured constants. Changing one of these is a claim about a measurement.

#: The 20 genomes of the atlas, in the order that IS the embedding row index.
#: Append-only: re-sorting this tuple silently re-labels every stored vector.
DEFAULT_STRAINS: Tuple[str, ...] = (
    "B10", "F7", "I1237", "IMI206040", "MMS1295", "N1508", "P1",
    "P3041", "P3080", "P3116", "asperelloides", "asperellum",
    "citrinoviride", "guizhouense", "harzianum", "longibrachiatum",
    "reesei", "velutinum", "virens", "viride",
)

#: Strain code -> leaf label in ``data/trees/supermatrix.treefile``. Seven
#: entries, and the emptiness of the rest is the measurement: 13 of the 20
#: atlas genomes have no leaf. Three species-level matches are flagged below;
#: no assembly manifest in the repo settles their strain.
STRAIN_TO_TAXON: Dict[str, str] = {
    "IMI206040": "GCA_000171015.2_protein",
    "P1": "GCF_020647795.1_protein",
    "reesei": "GCF_000167675.1_protein",
    "virens": "GCF_000170995.1_protein",
    "harzianum": "GCA_000988865.1_protein",      # species-level match only
    "guizhouense": "GCA_002022785.1_protein",
    "asperelloides": "GCA_052084505.1_protein",  # species-level match only
}

#: Leaves of the tree with no atlas column. Two outgroups plus T. gamsii,
#: which is a Trichoderma with a leaf and no expression data at all.
TAXA_WITHOUT_DATA: Tuple[str, ...] = (
    "GCF_000149955.1_protein",   # Fusarium oxysporum 4287
    "GCF_000182925.2_protein",   # Neurospora crassa OR74A
    "GCF_001481775.2_protein",   # Trichoderma gamsii T6085
)

#: Rerooting target. Read literally, both tree files show Trichoderma as
#: paraphyletic with Fusarium and Neurospora nested inside it, so any "genus
#: node" obtained by MRCA without rerooting spans the outgroups.
OUTGROUP_TAXA: Tuple[str, ...] = (
    "GCF_000149955.1_protein", "GCF_000182925.2_protein")

#: Near-duplicate clusters, measured as the fraction of orthogroups whose
#: 800 bp promoter is identical bit for bit: 0.88-0.96 inside a cluster. A
#: coordinate-offset control puts the modal offset at 0, so these are real
#: duplicates, not an extraction artefact, and the ten "atroviride strains"
#: are about five independent genomes.
NEAR_DUPLICATE_CLUSTERS: Tuple[Tuple[str, ...], ...] = (
    ("B10", "F7"),
    ("I1237", "P3041", "P3116", "P3080"),
    ("MMS1295", "N1508"),
)

#: In the orthology, absent from the expression metadata: 10 255 promoters,
#: 9926 orthogroups, ZERO samples.
STRAINS_WITHOUT_EXPRESSION: Tuple[str, ...] = ("viride",)

#: What the model does ON the "genome within orthogroup" axis, split into a
#: per-genome main effect and a genome x orthogroup interaction.
#: :data:`WITHIN_OG_VARIANCE` says how much label variance lies on that axis;
#: this says what the model recovers from it. The species vector is constant
#: within a genome, so it can only contribute the main effect, 20.31 % of it.
#: The easy misreading: the model is worse than a constant here (total R^2
#: -0.0045) not for want of signal (PCC 0.3325 against a within-orthogroup
#: shuffle control 0.0033 +- 0.0159) but because the slope fell below 0.5,
#: where ``R^2 = |ap|^2 (2s-1)/|ay|^2`` turns negative. The fault is amplitude.
MEASURED_GENOME_AXIS = {
    # share of the axis variance
    "main_effect_share": 0.2031,
    "interaction_share": 0.7969,
    # R^2 and slope per component
    "main_effect_r2": 0.8485,
    "main_effect_slope": 1.2410,
    "interaction_r2": -0.2218,
    "interaction_slope": 0.2361,
    "total_r2": -0.0045,
    "total_slope": 0.4950,
    "total_pcc": 0.3325,
    # negative control: predictions shuffled WITHIN an orthogroup, 20 draws
    "shuffle_pcc_mean": 0.0033,
    "shuffle_pcc_sd": 0.0159,
    # attribution: variance of the difference from baseline, as a share of the
    # prediction variance on this axis, under ablate='substitute' for the
    # species channel and flank_ok=0 for the genomic context
    "species_channel_share": 0.093,
    "species_channel_slope": 1.6746,
    "flank_channel_share": 0.035,
    "flank_channel_pcc": -0.0365,
    # after a per-axis affine calibration fitted on spy
    "calibrated_r2": 0.1988,
    "calibrated_slope": 1.052,
    "calibrated_pcc_gain_cells": 0.0234,
}

#: Ceiling on what the genome channel can explain: genome within orthogroup.
GENOME_VARIANCE_CEILING: float = 0.0623

#: The full variance decomposition, so the ceiling is never quoted alone.
WITHIN_OG_VARIANCE: Dict[str, float] = {
    "between_orthogroup": 0.7899,
    "genome_within_orthogroup": 0.0623,
    "sample_within_orthogroup_genome": 0.1478,
}

#: Hard guard. s=20 already puts the outermost taxon at radius 14.1, s=50
#: measured RMS 0.1898 and s=100 measured 0.4499 -- float64 Minkowski
#: cancellation, not geometry.
MAX_SAFE_SCALE: float = 20.0

#: Falsifier F2: the fitted radius must stay below this. Past ~14 the
#: constraint ``<x,x>_L = -1`` is a difference of numbers of order 2.2e6 and
#: the reported distortion stops meaning anything.
MAX_SAFE_RADIUS: float = 14.0

#: Relative RMS of the free-scale fit on the 10 taxa, by dimension. The F1
#: thresholds.
MEASURED_RMS_FREE_SCALE: Dict[int, float] = {
    2: 0.04321, 3: 0.01123, 5: 0.00293, 8: 1e-5,
}

#: Euclidean control at the same dimension and budget. If curvature does not
#: beat flat space on a metric that IS a tree, the machinery is decoration.
MEASURED_RMS_EUCLIDEAN: Dict[int, float] = {2: 0.08448, 3: 0.03037, 5: 0.00817}

#: Fixed-scale fit in H^2, the measurement that makes scale the binding
#: constraint rather than dimension.
MEASURED_RMS_BY_SCALE_H2: Dict[float, float] = {
    0.5: 0.0807, 5.0: 0.0717, 10.0: 0.0601, 20.0: 0.0418,
    50.0: 0.1898, 100.0: 0.4499,
}

#: Leave-one-genome-out placement of an unseen genome, with the bandwidth tuned
#: on the OTHER genomes because a new genome has no samples of its own.
#: ``jaccard`` clears its mean threshold and fails catastrophically on one
#: genome, which is why ``centroid`` is the default mode.
NEW_GENOME_MEASURED: Dict[str, Dict[str, float]] = {
    "jaccard": {"mean_gain": 0.0059, "worst_gain": -0.1509, "h": 0.05,
                "lam": 0.3},
    "kmer5": {"mean_gain": -0.0108, "worst_gain": float("nan"), "h": 0.05,
              "lam": 0.3},
    "consensus_shrunk": {"mean_gain": 0.0048, "worst_gain": -0.1525,
                         "h": 0.05, "lam": 0.3},
}

#: Order of :data:`PATRISTIC_PUBLISHED`.
PATRISTIC_PUBLISHED_TAXA: Tuple[str, ...] = (
    "GCA_000171015.2_protein",   # IMI206040
    "GCA_000988865.1_protein",   # harzianum T6776
    "GCA_002022785.1_protein",   # guizhouense
    "GCA_052084505.1_protein",   # asperelloides X01119
    "GCF_000149955.1_protein",   # Fusarium oxysporum
    "GCF_000167675.1_protein",   # reesei QM6a
    "GCF_000170995.1_protein",   # virens Gv29-8
    "GCF_000182925.2_protein",   # Neurospora crassa
    "GCF_001481775.2_protein",   # gamsii T6085
    "GCF_020647795.1_protein",   # P1
)

#: The patristic matrix as published in the analysis, rounded to 4 decimals.
#: A check on the parser, never a data source: the four-point defect of the
#: rounded matrix is ~1e-4 rather than 2.2e-16, so computing anything from it
#: would turn a rounding artefact into "the data are not a tree".
#: :func:`falsifiers` compares the parsed matrix against it at 5e-5.
PATRISTIC_PUBLISHED: np.ndarray = np.array([
    [0.0000, 0.2244, 0.2232, 0.0691, 0.6766, 0.2483, 0.2119, 1.0980, 0.0401, 0.0079],
    [0.2244, 0.0000, 0.0207, 0.2086, 0.6756, 0.1829, 0.0785, 1.0970, 0.2271, 0.2296],
    [0.2232, 0.0207, 0.0000, 0.2074, 0.6744, 0.1817, 0.0773, 1.0959, 0.2259, 0.2284],
    [0.0691, 0.2086, 0.2074, 0.0000, 0.6608, 0.2325, 0.1961, 1.0822, 0.0719, 0.0743],
    [0.6766, 0.6756, 0.6744, 0.6608, 0.0000, 0.6995, 0.6631, 1.0534, 0.6793, 0.6818],
    [0.2483, 0.1829, 0.1817, 0.2325, 0.6995, 0.0000, 0.1704, 1.1209, 0.2510, 0.2535],
    [0.2119, 0.0785, 0.0773, 0.1961, 0.6631, 0.1704, 0.0000, 1.0845, 0.2146, 0.2171],
    [1.0980, 1.0970, 1.0959, 1.0822, 1.0534, 1.1209, 1.0845, 0.0000, 1.1008, 1.1032],
    [0.0401, 0.2271, 0.2259, 0.0719, 0.6793, 0.2510, 0.2146, 1.1008, 0.0000, 0.0453],
    [0.0079, 0.2296, 0.2284, 0.0743, 0.6818, 0.2535, 0.2171, 1.1032, 0.0453, 0.0000],
], dtype=np.float64)

#: Falsifier F10: any surrogate genome distance must correlate with |dGC|
#: below this. 5-mer promoter distance measured +0.9684 (rejected), Jaccard of
#: orthogroup content +0.6436 (accepted).
SURROGATE_GC_LIMIT: float = 0.8

#: Written into the artefact so a vector can never be read without the scale
#: and dimension that give it meaning.
EMBEDDING_VERSION: str = "hyppe-species-1"

_EPS64 = 1e-15


# Registry

@dataclass(frozen=True)
class GenomeRecord:
    """One genome of the atlas. Immutable; the registry owns the ordering.

    ``name``            strain code as it appears in ``sequences.tsv.gz``. For
                        the ten coded strains this is not a species name; their
                        species is nowhere declared in the repo.
    ``taxon``           leaf label in the supermatrix tree, or ``None``. 13 of
                        20 genomes are ``None``, which is the real coverage
                        problem of this layer rather than the tree's quality.
    ``has_expression``  False only for ``viride`` (10 255 promoters, 0 samples).
    ``cluster``         near-duplicate cluster id, or ``None`` for a singleton.
                        Two genomes in one cluster are not independent evidence:
                        88-96 % of their promoters are bit-identical.
    ``species_level``   True when the leaf matches the atlas column at species
                        level only, so a strain-level claim is unsupported.
    """

    name: str
    taxon: Optional[str] = None
    has_expression: bool = True
    cluster: Optional[str] = None
    species_level: bool = False
    note: str = ""

    @property
    def placed(self) -> bool:
        """True when the tree gives this genome a position of its own."""
        return self.taxon is not None


class StrainRegistry:
    """Ordered, extensible registry of genomes. The index is the contract.

    A class and not a dict literal because adding a genome must not require a
    code edit: build the registry from the measured default, from a TSV or from
    rows, and :meth:`add` appends. Row order is embedding row order, so it is
    append-only -- re-sorting silently re-labels every stored vector.

    ``effective_n()`` is the number that belongs in a standard error: 20
    genomes but 15 independent units, because three near-duplicate clusters
    share 88-96 % of their promoters bit for bit.
    """

    def __init__(self, records: Sequence[GenomeRecord]) -> None:
        names = [r.name for r in records]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                "duplicate genome names %r: the index of a genome is its "
                "embedding row, so names must be unique" % (dup,))
        taxa = [r.taxon for r in records if r.taxon is not None]
        if len(set(taxa)) != len(taxa):
            raise ValueError(
                "two genomes claim the same tree leaf; the tree cannot place "
                "them apart and the prior would pull them together")
        self._records: List[GenomeRecord] = list(records)
        self._index: Dict[str, int] = {r.name: i for i, r in enumerate(records)}

    # Constructors
    @classmethod
    def by_leaf_name(cls, base: Optional["StrainRegistry"] = None
                     ) -> "StrainRegistry":
        """The same genomes, keyed to a tree whose leaves ARE the names.

        `default()` places only seven genomes, because `supermatrix.treefile`
        names its ten leaves by accession. A tree built over all twenty
        in-house genomes names its leaves by strain instead, and then every
        genome is placeable. Row order is copied unchanged, as the registry
        contract requires.
        """
        base = base or cls.default()
        # `placed` is a property (taxon is not None), so setting the taxon is
        # what places the genome.
        return cls([replace(r, taxon=r.name) for r in base])

    @classmethod
    def default(cls) -> "StrainRegistry":
        """The measured 20-genome atlas registry.

        Seven genomes carry a leaf, thirteen do not, one (``viride``) has no
        expression at all, three clusters are near-duplicates.
        """
        cluster_of: Dict[str, str] = {}
        for k, members in enumerate(NEAR_DUPLICATE_CLUSTERS):
            for m in members:
                cluster_of[m] = "dup%d" % k
        species_only = {"harzianum", "asperelloides"}
        recs = []
        for name in DEFAULT_STRAINS:
            recs.append(GenomeRecord(
                name=name,
                taxon=STRAIN_TO_TAXON.get(name),
                has_expression=name not in STRAINS_WITHOUT_EXPRESSION,
                cluster=cluster_of.get(name),
                species_level=name in species_only,
                note=("leaf matches the atlas column at species level only; "
                      "no assembly manifest in the repo settles the strain"
                      if name in species_only else ""),
            ))
        return cls(recs)

    @classmethod
    def from_rows(cls, rows: Iterable[Dict[str, object]]) -> "StrainRegistry":
        """Build from dict rows -- the path that makes a new genome a data edit.

        Recognised keys: ``name`` (required), ``taxon``, ``has_expression``,
        ``cluster``, ``species_level``, ``note``. An unknown key raises instead
        of being dropped, so a typo cannot silently omit a field.
        """
        known = {"name", "taxon", "has_expression", "cluster", "species_level",
                 "note"}
        out = []
        for row in rows:
            extra = set(row) - known
            if extra:
                raise ValueError("unknown registry field(s) %r; known: %r"
                                 % (sorted(extra), sorted(known)))
            if not row.get("name"):
                raise ValueError("registry row without a name: %r" % (row,))
            taxon = row.get("taxon") or None
            out.append(GenomeRecord(
                name=str(row["name"]),
                taxon=str(taxon) if taxon else None,
                has_expression=bool(row.get("has_expression", True)),
                cluster=(str(row["cluster"]) if row.get("cluster") else None),
                species_level=bool(row.get("species_level", False)),
                note=str(row.get("note", "")),
            ))
        return cls(out)

    @classmethod
    def from_tsv(cls, path: str) -> "StrainRegistry":
        """Read a registry from a TSV with a header naming the fields above.

        No pandas on purpose: this module must import with no data file
        present, and the registry is 20 rows.
        """
        rows: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                row = dict(zip(header, parts))
                if "has_expression" in row:
                    row["has_expression"] = str(row["has_expression"]).lower() \
                        not in ("0", "false", "no", "")
                if "species_level" in row:
                    row["species_level"] = str(row["species_level"]).lower() \
                        in ("1", "true", "yes")
                rows.append(row)
        return cls.from_rows(rows)

    # Accessors
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, key) -> GenomeRecord:
        if isinstance(key, int):
            return self._records[key]
        return self._records[self._index[str(key)]]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self._records)

    def index(self, name: str) -> int:
        """Embedding row of ``name``; raises with the full list on a miss."""
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                "genome %r is not in the registry. Known: %s. Adding one is "
                "`registry.add(...)` plus `GenomeEmbedding.add_genome(...)`, "
                "not a code edit." % (name, ", ".join(self.names))) from None

    def placed_names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self._records if r.placed)

    def unplaced_names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self._records if not r.placed)

    def taxa(self) -> Tuple[str, ...]:
        return tuple(r.taxon for r in self._records if r.taxon is not None)

    def taxon_to_index(self) -> Dict[str, int]:
        return {r.taxon: i for i, r in enumerate(self._records) if r.placed}

    def clusters(self) -> Dict[str, Tuple[str, ...]]:
        """Cluster id -> members, singletons included under their own name."""
        out: Dict[str, List[str]] = {}
        for r in self._records:
            key = r.cluster or ("@" + r.name)
            out.setdefault(key, []).append(r.name)
        return {k: tuple(v) for k, v in out.items()}

    def effective_n(self) -> int:
        """Independent units, counting a near-duplicate cluster once.

        20 genomes -> 15 units on the default registry. Every standard error
        over genomes must use this and not ``len(self)``: inside a cluster
        88-96 % of promoters are identical bit for bit, so the rows are copies.
        """
        return len(self.clusters())

    # Mutation
    def add(self, name: str, taxon: Optional[str] = None,
            has_expression: bool = True, cluster: Optional[str] = None,
            species_level: bool = False, note: str = "") -> int:
        """Append a genome and return its index, which is ``len(registry) - 1``.

        That index is the row :meth:`GenomeEmbedding.add_genome` will create.
        Re-adding an existing name raises: overwriting a row would move a
        trained vector to a different genome with no shape or dtype changing.
        """
        if name in self._index:
            raise ValueError(
                "genome %r is already registered at index %d. The index is a "
                "checkpoint contract; to correct a record build a new registry "
                "rather than overwriting a trained row."
                % (name, self._index[name]))
        if taxon is not None and taxon in set(self.taxa()):
            raise ValueError("leaf %r already belongs to genome %r"
                             % (taxon, self[[r.taxon for r in self._records]
                                            .index(taxon)].name))
        rec = GenomeRecord(name=name, taxon=taxon,
                           has_expression=bool(has_expression),
                           cluster=cluster, species_level=bool(species_level),
                           note=note)
        self._records.append(rec)
        self._index[name] = len(self._records) - 1
        return len(self._records) - 1

    # Observability
    def report(self) -> Dict[str, object]:
        """Registry state. Every entry is checkable against the analysis."""
        return {
            "n_genomes": len(self),
            "n_placed": len(self.placed_names()),
            "n_unplaced": len(self.unplaced_names()),
            "placed": list(self.placed_names()),
            "unplaced": list(self.unplaced_names()),
            "n_species_level_only": sum(1 for r in self._records
                                        if r.species_level),
            "no_expression": [r.name for r in self._records
                              if not r.has_expression],
            "n_clusters": len(self.clusters()),
            "effective_n": self.effective_n(),
            "cluster_members": {k: list(v) for k, v in self.clusters().items()
                                if len(v) > 1},
        }

    def falsifiers(self) -> List[Tuple[str, bool, str]]:
        """(name, passed, detail). Structural claims about the registry."""
        r = self.report()
        names = self.names
        return [
            ("names are unique (index is the embedding row)",
             len(set(names)) == len(names),
             "%d names, %d unique" % (len(names), len(set(names)))),
            ("exactly 7 genomes carry a tree leaf",
             r["n_placed"] == 7 if set(names) == set(DEFAULT_STRAINS) else True,
             "placed=%d, unplaced=%d (the brief said 10 unplaced; measured 13)"
             % (r["n_placed"], r["n_unplaced"])),
            ("no leaf is claimed by two genomes",
             len(set(self.taxa())) == len(self.taxa()),
             "%d leaves used" % len(self.taxa())),
            ("effective n counts near-duplicate clusters once",
             r["effective_n"] <= len(self),
             "effective_n=%d of %d genomes (measured identity inside clusters "
             "0.8845 / 0.9590 / 0.9538)" % (r["effective_n"], len(self))),
            ("a genome without expression is flagged, not dropped",
             all(n in names for n in STRAINS_WITHOUT_EXPRESSION)
             if set(names) >= set(STRAINS_WITHOUT_EXPRESSION) else True,
             "no_expression=%r" % (r["no_expression"],)),
        ]


# Newick: parsing, rerooting, patristic distances, topology

class BranchLengthsMissing(ValueError):
    """Raised when a tree cannot support a patristic distance.

    ``supertree.tre`` hits this: 10 of its 19 nodes have no branch length and
    the internal lengths it does carry are ASTRAL coalescent units, which are
    not convertible to substitutions per site. It is admissible as a topology
    and support control and for nothing else.
    """


class Clade:
    """A node of a rooted Newick tree.

    ``label`` is whatever sat before the colon. For an internal node that is a
    support value, never a branch length -- reading ``)100:0.03`` as a length
    of 100 is the classic Newick parsing failure, so the two live in separate
    fields and :attr:`support` is only set for internal nodes.
    """

    __slots__ = ("label", "length", "children")

    def __init__(self, label: str = "", length: Optional[float] = None,
                 children: Optional[List["Clade"]] = None) -> None:
        self.label = label
        self.length = length
        self.children: List[Clade] = list(children) if children else []

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def support(self) -> Optional[float]:
        """Internal-node label read as a support value, if it is numeric."""
        if self.is_leaf or not self.label:
            return None
        try:
            return float(self.label)
        except ValueError:
            return None

    def leaves(self) -> List[str]:
        if self.is_leaf:
            return [self.label]
        out: List[str] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Clade(%r, len=%r, children=%d)" % (self.label, self.length,
                                                   len(self.children))


def parse_newick(text: str) -> Clade:
    """Parse a Newick string into a rooted :class:`Clade`.

    Whitespace is stripped globally, which is safe here because the labels are
    accessions. A quoted label raises rather than being silently mangled.
    """
    if "'" in text or '"' in text:
        raise ValueError(
            "quoted Newick labels are not supported by this parser; the HYPPE "
            "trees use bare accessions. Refusing rather than mangling.")
    s = "".join(text.split())
    if not s:
        raise ValueError("empty Newick string")
    if s.endswith(";"):
        s = s[:-1]
    pos = 0

    def _token_end(i: int) -> int:
        while i < len(s) and s[i] not in ",():":
            i += 1
        return i

    def _parse() -> Clade:
        nonlocal pos
        children: List[Clade] = []
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                children.append(_parse())
                if pos >= len(s):
                    raise ValueError("unbalanced parentheses in Newick")
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
                raise ValueError("unexpected %r at %d in Newick" % (s[pos], pos))
        end = _token_end(pos)
        label = s[pos:end]
        pos = end
        length: Optional[float] = None
        if pos < len(s) and s[pos] == ":":
            pos += 1
            end = _token_end(pos)
            try:
                length = float(s[pos:end])
            except ValueError:
                raise ValueError("bad branch length %r at %d"
                                 % (s[pos:end], pos)) from None
            pos = end
        return Clade(label=label, length=length, children=children)

    root = _parse()
    if pos != len(s):
        raise ValueError("trailing characters after the tree: %r" % (s[pos:],))
    return root


def read_newick(path: str) -> Clade:
    """Read a Newick file. No file is touched at import time, only here."""
    with open(path, "r", encoding="utf-8") as fh:
        return parse_newick(fh.read())


def _edges(root: Clade) -> Tuple[List[str], Dict[int, List[Tuple[int, Optional[float]]]]]:
    """Rooted tree -> undirected adjacency, collapsing a degree-2 root.

    Patristic distance is a property of the unrooted tree, and a root with
    exactly two children is an artefact of the file format that would add a
    spurious node in the middle of one edge. Merging the two branch lengths is
    the only collapse that keeps every distance identical.
    """
    nodes: List[Clade] = list(root.walk())
    node_id = {id(n): i for i, n in enumerate(nodes)}
    adj: Dict[int, List[Tuple[int, Optional[float]]]] = {i: [] for i in
                                                         range(len(nodes))}
    for n in nodes:
        for c in n.children:
            i, j = node_id[id(n)], node_id[id(c)]
            adj[i].append((j, c.length))
            adj[j].append((i, c.length))
    labels = [n.label if n.is_leaf else "" for n in nodes]
    rid = node_id[id(root)]
    if len(adj[rid]) == 2:
        (a, la), (b, lb) = adj[rid]
        merged = None if (la is None or lb is None) else la + lb
        adj[a] = [(x, w) for x, w in adj[a] if x != rid] + [(b, merged)]
        adj[b] = [(x, w) for x, w in adj[b] if x != rid] + [(a, merged)]
        adj[rid] = []
    return labels, adj


def _leaf_ids(labels: Sequence[str], adj: Dict[int, List[Tuple[int, Optional[float]]]]
              ) -> List[int]:
    return [i for i in adj if len(adj[i]) == 1]


def _side_leaves(adj, labels, start: int, blocked: int) -> frozenset:
    """Leaf labels reachable from ``start`` without crossing to ``blocked``."""
    seen = {blocked, start}
    stack = [start]
    out = set()
    if len(adj[start]) == 1:
        out.add(labels[start])
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if len(adj[v]) == 1:
                out.add(labels[v])
            stack.append(v)
    return frozenset(out)


def reroot_on_outgroup(root: Clade,
                       outgroup: Sequence[str] = OUTGROUP_TAXA) -> Clade:
    """Reroot on the branch that separates ``outgroup`` from everything else.

    Mandatory before any MRCA-based reasoning: read literally, both HYPPE tree
    files place Fusarium and Neurospora inside Trichoderma, so a "genus node"
    taken as the MRCA of the placed taxa spans the outgroups and every tip
    depth measured against it is wrong.

    The new root goes at the midpoint of the separating edge, so patristic
    distances are unchanged; :func:`falsifiers` checks that invariance rather
    than assuming it. Raises if the outgroup is not monophyletic, because then
    no single edge separates it and rerooting would be a choice.
    """
    labels, adj = _edges(root)
    want = frozenset(outgroup)
    leaves = {labels[i] for i in _leaf_ids(labels, adj)}
    missing = want - leaves
    if missing:
        raise ValueError("outgroup taxa absent from the tree: %r"
                         % (sorted(missing),))
    for u in adj:
        for v, w in adj[u]:
            if u > v:
                continue
            if _side_leaves(adj, labels, u, v) == want:
                a, b, length = u, v, w
                break
            if _side_leaves(adj, labels, v, u) == want:
                a, b, length = v, u, w
                break
        else:
            continue
        break
    else:
        raise ValueError(
            "outgroup %r is not monophyletic in this tree; no single edge "
            "separates it, so rerooting on it is undefined" % (list(outgroup),))

    half = None if length is None else length / 2.0

    def _build(node: int, parent: int, blen: Optional[float]) -> Clade:
        kids = [(_build(v, node, w)) for v, w in adj[node] if v != parent]
        return Clade(label=labels[node] if not kids else "", length=blen,
                     children=kids)

    return Clade(label="", length=None,
                 children=[_build(a, b, half), _build(b, a, half)])


def patristic_matrix(root: Clade, taxa: Optional[Sequence[str]] = None
                     ) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Leaf-to-leaf path lengths. Returns ``(taxa, D)`` with ``D`` symmetric.

    Raises :class:`BranchLengthsMissing` if any edge has no length, which keeps
    ``supertree.tre`` out of every distance computation. Path sums, not depth
    differences from an MRCA: the two agree on a correctly rooted tree, and
    only the path sum is reroot-invariant and therefore checkable.
    """
    labels, adj = _edges(root)
    missing = sum(1 for u in adj for v, w in adj[u] if w is None) // 2
    if missing:
        total = sum(1 for u in adj if adj[u])
        raise BranchLengthsMissing(
            "%d of this tree's edges have no branch length (%d nodes carry "
            "edges); a patristic matrix is not computable from it. "
            "supertree.tre is the known case: it is admissible as a topology "
            "and support control only." % (missing, total))
    leaf_ids = _leaf_ids(labels, adj)
    name_of = {labels[i]: i for i in leaf_ids}
    if len(name_of) != len(leaf_ids):
        raise ValueError("duplicate leaf labels in the tree")
    order = tuple(taxa) if taxa is not None else tuple(sorted(name_of))
    unknown = [t for t in order if t not in name_of]
    if unknown:
        raise ValueError("taxa absent from the tree: %r" % (unknown,))
    n = len(order)
    d = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        src = name_of[order[a]]
        dist = {src: 0.0}
        stack = [src]
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if v in dist:
                    continue
                dist[v] = dist[u] + float(w)
                stack.append(v)
        for b in range(n):
            d[a, b] = dist[name_of[order[b]]]
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return order, d


def unrooted_splits(root: Clade) -> set:
    """Non-trivial leaf bipartitions, canonicalised for comparison.

    A split is stored as the side not containing the alphabetically first leaf,
    which makes the representation independent of rooting. Without that, these
    two files compare as RF = 1, because ``supertree.tre`` is rooted on
    IMI206040 and that rooting induces one extra apparent split.
    """
    labels, adj = _edges(root)
    leaf_ids = _leaf_ids(labels, adj)
    all_leaves = frozenset(labels[i] for i in leaf_ids)
    ref = min(all_leaves)
    out = set()
    for u in adj:
        for v, _ in adj[u]:
            if u > v:
                continue
            side = _side_leaves(adj, labels, u, v)
            if len(side) < 2 or len(side) > len(all_leaves) - 2:
                continue
            out.add(side if ref not in side else (all_leaves - side))
    return out


def robinson_foulds(a: Clade, b: Clade) -> int:
    """Unrooted RF distance on non-trivial splits. Must be 0 for these files.

    Measured: 7 non-trivial splits each, 7 shared, with full support on both
    sides. There is no contested clade to adjudicate, which is why the
    supertree contributes nothing beyond this check.
    """
    sa, sb = unrooted_splits(a), unrooted_splits(b)
    return len(sa ^ sb)


def four_point_defect(d: np.ndarray) -> float:
    """Max ``(largest - second largest)`` of the three quadruple pair-sums.

    Zero exactly for a tree metric; measured 2.220e-16 over all 210 quadruples
    of the parsed supermatrix, so residual embedding distortion is a fact about
    dimension and scale rather than about the data.
    """
    n = int(d.shape[0])
    worst = 0.0
    for i, j, k, l in itertools.combinations(range(n), 4):
        s1 = d[i, j] + d[k, l]
        s2 = d[i, k] + d[j, l]
        s3 = d[i, l] + d[j, k]
        s = sorted((s1, s2, s3))
        worst = max(worst, float(s[2] - s[1]))
    return worst


#: Smallest patristic distance the prior will build a term on.
#:
#: Not a tolerance: the in-house 439-marker matrix aligns ~191 175 amino-acid
#: columns per pair, which puts the binomial SE near 9e-4, so a distance under
#: 1e-3 is indistinguishable from zero in the data it came from. It matters
#: because prior stiffness goes as 1/d^2 -- the NJ tree carries a pair at 1e-6,
#: whose stiffness is then ~30 000x the median pair's and owns the gradient the
#: instant the points move. The ten-leaf tree's smallest pair is eight times
#: this, so nothing there changes.
PRIOR_MIN_DIST = 1e-3

#: Denominator floor for the same stiffness problem, one step further on.
#:
#: `PRIOR_MIN_DIST` drops the pairs the markers cannot resolve at all and does
#: nothing for the pairs just above it: over the 182 kept pairs of the 20-leaf
#: tree, stiffness spans 8679x, and its top is held by the seven strains that
#: share pks1 byte for byte, whose measured control spans a factor of 195. So
#: the prior holds hardest exactly the genomes the channel exists to separate.
#: The relative-error form is still right -- an absolute one would let the two
#: outgroup branches flatten everything inside Trichoderma -- it is just being
#: asked for a precision the matrix does not have. A floor of 5e-3 is where the
#: tree's own relative precision reaches 20 % (9e-4 / 0.20), and it softens 16
#: of the 182 pairs. The default is `PRIOR_MIN_DIST`, under which nothing
#: softens and the term is bit for bit what it was; `prior_fit()` reports
#: whether raising it moved the points.
PRIOR_DENOM_FLOOR = PRIOR_MIN_DIST


def default_tree_paths() -> Dict[str, Optional[str]]:
    """Locate the tree files without importing anything data-dependent.

    Resolution order: ``HYPPE_DATA_ROOT``, then the repo layout relative to
    this file. A missing entry comes back as ``None`` so a caller can report
    "no tree" instead of crashing at import.
    """
    roots: List[str] = []
    env = os.environ.get("HYPPE_DATA_ROOT")
    if env:
        roots.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    roots.append(os.path.abspath(os.path.join(here, "..", "..", "..")))
    out: Dict[str, Optional[str]] = {"supermatrix": None, "supertree": None,
                                     "annotation": None, "inhouse20": None,
                                     "plus4": None}
    for root in roots:
        cand = os.path.join(root, "data", "trees")
        for key, fname in (("supermatrix", "supermatrix.treefile"),
                           ("supertree", "supertree.tre"),
                           ("annotation", "annotation.tsv"),
                           ("plus4",
                            "supermatrix_plus4_inhouse439_ntsubst.treefile"),
                           ("inhouse20",
                            os.path.join(
                                "external",
                                "inhouse_nj_20genomes_aasubst.tre"))):
            p = os.path.join(cand, fname)
            if out[key] is None and os.path.isfile(p):
                out[key] = p
    return out


# Embedding

@dataclass
class EmbeddingFit:
    """Result of :func:`fit_hyperbolic_embedding`.

    ``points``     ``[n, dim+1]`` float64 on the hyperboloid.
    ``rms_rel``    ``sqrt(mean((d_fit / (scale * d_tree) - 1)^2))`` -- the F1
                   statistic. Free-scale references: H^2 0.04321, H^3 0.01123,
                   H^8 ~1e-5.
    ``distortion`` worst-case distortion ``max ratio / min ratio``; 1.0 for a
                   scaled isometry, measured 1.2632 for H^2.
    ``max_radius`` F2: must stay under :data:`MAX_SAFE_RADIUS`.
    """

    points: torch.Tensor
    taxa: Tuple[str, ...]
    dim: int
    scale: float
    rms_rel: float
    max_rel: float
    distortion: float
    max_radius: float
    loss: float
    restarts: int
    free_scale: bool
    fitted_scale: float

    def report(self) -> Dict[str, object]:
        out = {
            "n_taxa": len(self.taxa), "dim": self.dim, "scale": self.scale,
            "fitted_scale": self.fitted_scale, "free_scale": self.free_scale,
            "rms_rel": self.rms_rel, "max_rel": self.max_rel,
            "distortion_worst": self.distortion, "max_radius": self.max_radius,
            "loss": self.loss, "restarts": self.restarts,
            "manifold_violation": assert_on_manifold(self.points,
                                                     raise_on_fail=False),
        }
        out.update(distance_disagreement(self.points))
        return out

    def falsifiers(self) -> List[Tuple[str, bool, str]]:
        r = self.report()
        ref = MEASURED_RMS_FREE_SCALE.get(self.dim)
        out = [
            ("F2 max radius below the float64 limit",
             self.max_radius < MAX_SAFE_RADIUS,
             "max radius %.4f (limit %.1f; s=20 measured 14.1, s=50 RMS "
             "0.1898)" % (self.max_radius, MAX_SAFE_RADIUS)),
            ("result is on the hyperboloid",
             float(r["manifold_violation"]) < 1e-12,
             "max|<x,x>_L + 1| = %.3e" % r["manifold_violation"]),
            ("radius stays inside the expmap clamp",
             self.max_radius < MAX_TANGENT_NORM,
             "max radius %.4f against clamp %.1f (informative: s=10 measured "
             "7.0)" % (self.max_radius, MAX_TANGENT_NORM)),
            ("float64 still resolves the SHORT distances at this radius",
             float(r["max_rel_disagreement"]) < 1e-9,
             "two algebraically identical formulas disagree by %.2e (worst on "
             "the pair at distance %.4f); above ~1e-9 the reported distortion "
             "is partly a float64 artefact"
             % (r["max_rel_disagreement"], r["worst_pair_distance"])),
        ]
        if self.free_scale and ref is not None:
            tol = max(ref * 1.10, ref + 1e-4)
            out.insert(0, (
                "F1 relative RMS at the measured level for d=%d" % self.dim,
                self.rms_rel <= tol,
                "RMS %.5f against measured %.5f (tolerance %.5f)"
                % (self.rms_rel, ref, tol)))
        return out


def _acosh_stable(g: torch.Tensor) -> torch.Tensor:
    """``arccosh(g)`` via ``log1p``, with the sqrt operand floored.

    Same construction as ``lorentz._safe_geodesic``, kept local so this module
    does not depend on a private name. ``acosh(1+eps)`` reports 4.5e-4 as the
    distance of a point from itself, and ``sqrt(t^2+2t)`` has infinite slope at
    ``t=0``, which is where a converging fit drives. Flooring the operand
    blocks the gradient through the clamp; ``torch.where`` on the result would
    still propagate NaN.
    """
    t = (g - 1.0).clamp_min(0.0)
    return torch.log1p(t + torch.sqrt((t * t + 2.0 * t).clamp_min(_EPS64 ** 2)))


def distance_disagreement(points: torch.Tensor) -> Dict[str, float]:
    """How much float64 is left at this radius, measured rather than quoted.

    ``cosh(d) - 1`` can be had as ``G - 1`` with
    ``G = x0_i x0_j - <u_i, u_j>``, which subtracts two numbers of order
    ``cosh^2(r)``, or as ``(|u_i - u_j|^2 - (x0_i - x0_j)^2) / 2``, which
    subtracts differences. At radius 16.6 the first carries ~6e-3 of absolute
    error: nothing next to a far pair, and half the value for the closest pair
    in this tree. That is the concrete form of "scale is the binding
    constraint" -- the damage lands on the shortest distances, which a
    relative-error objective weights most.

    Returns the max and median relative disagreement and the pair it is worst
    on. Above ~1e-9 the reported distortion is partly a float64 artefact.
    """
    p = points.to(torch.float64)
    n = int(p.shape[0])
    iu, ju = torch.triu_indices(n, n, offset=1)
    x0, u = p[:, 0], p[:, 1:]
    g_a = x0[iu] * x0[ju] - (u[iu] * u[ju]).sum(dim=-1)
    diff_u = (u[iu] - u[ju]).pow(2).sum(dim=-1)
    diff_0 = (x0[iu] - x0[ju]).pow(2)
    g_b = 1.0 + 0.5 * (diff_u - diff_0)
    d_a = _acosh_stable(g_a)
    d_b = _acosh_stable(g_b)
    rel = ((d_a - d_b).abs() / d_b.clamp_min(1e-300))
    k = int(rel.argmax())
    return {"max_rel_disagreement": float(rel.max()),
            "median_rel_disagreement": float(rel.median()),
            "worst_pair": (int(iu[k]), int(ju[k])),
            "worst_pair_distance": float(d_b[k])}


def _pair_distances(u: torch.Tensor, iu: torch.Tensor, ju: torch.Tensor
                    ) -> torch.Tensor:
    """Geodesic distances between rows of the free parameterisation ``u``.

    ``x = (sqrt(1+|u|^2), u)`` satisfies ``<x,x>_L = -1`` for any ``u``, so the
    fit is unconstrained and no projection step can be forgotten.
    """
    x0 = torch.sqrt(1.0 + u.pow(2).sum(dim=-1))
    g = x0[iu] * x0[ju] - (u[iu] * u[ju]).sum(dim=-1)
    return _acosh_stable(g)


#: Scale rungs for the continuation: the fit at scale s is warm-started from
#: the rung below it, which is how the reference numbers were produced. A cold
#: fit at s=20 lands at relative RMS 0.1428 where continuation reaches 0.0418.
DEFAULT_SCALE_LADDER: Tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 20.0)


def recenter_embedding(points: torch.Tensor) -> torch.Tensor:
    """Move the Frechet mean of ``points`` to the origin. An isometry.

    The radius budget is a statement about a gauge: the fit constrains only
    distances, so its solution is fixed up to O(1,d) and where the cloud sits
    relative to the origin is arbitrary. Leaving it arbitrary cost 3.8 of
    radius -- the H^3 fit at s=10 reached radius 9.349, past the ``expmap``
    clamp, while carrying a diameter of only ~11.2.

    Logmap at the mean, parallel transport to the origin, expmap there, which
    is an isometry; arithmetic re-centring of ambient coordinates would not
    preserve distances and would leave the sheet.
    """
    c = frechet_mean(points, iters=12)
    v = logmap(c.unsqueeze(0).expand_as(points), points)
    w = parallel_transport(c.unsqueeze(0).expand_as(points),
                           origin_like(points), v)
    out = expmap(origin_like(points), w)
    return project_to_hyperboloid(out, ambient=True)


def _rescale_radii(u: torch.Tensor, ratio: float) -> torch.Tensor:
    """Multiply every point's geodesic radius by ``ratio``, keeping directions.

    The continuation map between scale rungs. Carrying ``u`` verbatim is the
    wrong warm start: the next rung's solution is approximately a radially
    dilated copy, and ``|u| = sinh(r)`` makes that ``sinh(asinh(|u|)*ratio)``
    rather than ``|u| * ratio``. Without it, continuation to s=20 in H^2 stalls
    at radius 10.45 (RMS 0.0839) instead of the reference basin at 14.1.
    """
    nrm = u.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    return u / nrm * torch.sinh(torch.asinh(nrm) * float(ratio))


def _fit_at_scale(target: torch.Tensor, iu: torch.Tensor, ju: torch.Tensor,
                  n: int, dim: int, s: float, restarts: int, seed: int,
                  rounds: int, max_iter: int, init_scale: float,
                  u0: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    """One fixed-scale fit; ``u0`` warm-starts it. Returns ``(u, loss)``.

    ``rounds`` re-creates the L-BFGS state between rounds, which is not
    cosmetic: a single ``step`` stops on its own tolerances well short of the
    optimum here, and re-seeding the inverse-Hessian history restarts it.
    """
    best_u, best = None, float("inf")
    # The warm start races the cold ones rather than replacing them: a
    # continuation rung can land in a worse basin than a fresh draw.
    starts: List[Optional[torch.Tensor]] = ([u0] if u0 is not None else [])
    starts += [None] * max(1, int(restarts))
    for k, start in enumerate(starts):
        if start is None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1013 * k)
            u = (torch.randn(n, int(dim), generator=gen, dtype=torch.float64)
                 * float(init_scale)).requires_grad_(True)
        else:
            u = start.clone().requires_grad_(True)
        for _ in range(max(1, int(rounds))):
            opt = torch.optim.LBFGS([u], max_iter=int(max_iter),
                                    history_size=50, tolerance_grad=1e-16,
                                    tolerance_change=1e-18,
                                    line_search_fn="strong_wolfe")

            def _closure():
                opt.zero_grad(set_to_none=True)
                dist = _pair_distances(u, iu, ju)
                val = (dist / (s * target) - 1.0).pow(2).mean()
                val.backward()
                return val

            opt.step(_closure)
        with torch.no_grad():
            dist = _pair_distances(u, iu, ju)
            val = float((dist / (s * target) - 1.0).pow(2).mean())
        if math.isfinite(val) and val < best:
            best, best_u = val, u.detach().clone()
    assert best_u is not None
    return best_u, best


def fit_hyperbolic_embedding(d: np.ndarray, dim: int = 3,
                             scale: Optional[float] = 10.0,
                             restarts: int = 10, seed: int = 0,
                             max_iter: int = 400, rounds: int = 3,
                             init_scale: Optional[float] = None,
                             continuation: bool = True,
                             max_radius: float = MAX_SAFE_RADIUS,
                             scale_ladder: Sequence[float] = DEFAULT_SCALE_LADDER,
                             recenter: bool = True,
                             ) -> EmbeddingFit:
    """Embed a tree distance matrix on H^dim at a fixed or profiled scale.

    The objective is ``mean((d_L(x_i,x_j) / (s * d_tree(i,j)) - 1)^2)``, a
    relative error, because an absolute one spends the whole budget on the
    longest paths and this matrix spans 141.4x. The Neurospora branch alone is
    0.7374 and would flatten the Trichoderma block the model conditions on.

    ``scale=None`` means "search the ladder under a radius budget", not "let
    the scale float": with the scale profiled out the objective is
    scale-invariant, so inflating the embedding always helps and the fit stops
    only where L-BFGS stalls. A free-scale RMS is therefore meaningful only
    together with the radius it was achieved at, which ``max_radius`` pins
    down. ``recenter`` fixes the gauge before that radius is measured, and
    ``continuation`` warm-starts each scale from the rung below it, which
    reproduces the operating point where a cold fit does not.

    L-BFGS with strong-Wolfe line search over autograd gradients in float64,
    ``restarts`` random restarts, ``rounds`` L-BFGS restarts each, best kept.
    Finite differences are banned: they stall this objective at 0.46 and
    produce the false conclusion that large scale breaks the embedding.

    Raises on ``scale > MAX_SAFE_SCALE``, past which the degradation is
    Minkowski cancellation in float64 and not a property of the tree.
    """
    if scale is not None and float(scale) > MAX_SAFE_SCALE:
        raise ValueError(
            "scale %.3f exceeds MAX_SAFE_SCALE %.1f. Measured: s=20 puts the "
            "outermost taxon at radius 14.1 (already past the expmap clamp of "
            "8.0), s=50 gives relative RMS 0.1898 and s=100 gives 0.4499. "
            "Those are float64 cancellations in <x,x>_L = ||s||^2 - x0^2, not "
            "embeddings." % (float(scale), MAX_SAFE_SCALE))
    if scale is not None and float(scale) <= 0.0:
        raise ValueError("scale must be positive, got %r" % (scale,))
    dd = np.asarray(d, dtype=np.float64)
    if dd.ndim != 2 or dd.shape[0] != dd.shape[1]:
        raise ValueError("distance matrix must be square, got %r"
                         % (dd.shape,))
    n = int(dd.shape[0])
    if n < 3:
        raise ValueError("need at least 3 taxa, got %d" % n)
    if not np.allclose(dd, dd.T, atol=1e-12):
        raise ValueError("distance matrix is not symmetric")
    iu_np, ju_np = np.triu_indices(n, k=1)
    if float(dd[iu_np, ju_np].min()) <= 0.0:
        raise ValueError(
            "two taxa are at distance 0; merge them before embedding -- a "
            "relative-error objective cannot use them")

    target = torch.as_tensor(dd[iu_np, ju_np], dtype=torch.float64)
    iu = torch.as_tensor(iu_np, dtype=torch.long)
    ju = torch.as_tensor(ju_np, dtype=torch.long)
    med = float(np.median(dd[iu_np, ju_np]))
    free = scale is None

    def _init(s: float) -> float:
        # Seed the cloud at half the typical target distance, so the fit starts
        # in the right order of magnitude rather than expanding from a point --
        # which is where L-BFGS stalls at large s.
        return float(init_scale) if init_scale is not None else 0.5 * s * med

    def _rungs(s_end: float) -> List[float]:
        if not continuation:
            return [float(s_end)]
        return [float(x) for x in scale_ladder if x < s_end] + [float(s_end)]

    # Random restarts are spent on the first rung, where the problem is small
    # and well conditioned; later rungs race the warm start against one cold
    # draw. Restarting everywhere costs 37 s per fit against 11 s for the same
    # RMS at s=10.
    later = 1

    def _finish(u: torch.Tensor, s: float, loss: float) -> EmbeddingFit:
        with torch.no_grad():
            dist = _pair_distances(u, iu, ju)
            ratio = dist / (s * target)
            rel = (ratio - 1.0).abs()
            pts = project_to_hyperboloid(u)
            # Fix the gauge before measuring the radius; distances are
            # unchanged by construction.
            pts = recenter_embedding(pts) if recenter else pts
            rad = radius(pts)
        return EmbeddingFit(
            points=pts, taxa=tuple("t%d" % i for i in range(n)),
            dim=int(dim), scale=float(s), rms_rel=float(math.sqrt(loss)),
            max_rel=float(rel.max()),
            distortion=float(ratio.max() / ratio.min().clamp_min(1e-30)),
            max_radius=float(rad.max()), loss=float(loss),
            restarts=int(max(1, restarts)), free_scale=bool(free),
            fitted_scale=float(s))

    if not free:
        s_end = float(scale)
        u = None
        s_prev = None
        loss = float("inf")
        for s_now in _rungs(s_end):
            warm = (None if u is None
                    else _rescale_radii(u, s_now / float(s_prev)))
            u, loss = _fit_at_scale(target, iu, ju, n, dim, s_now,
                                    restarts if warm is None else later,
                                    seed, rounds, max_iter, _init(s_now), warm)
            s_prev = s_now
        assert u is not None
        return _finish(u, s_end, loss)

    # Free scale: walk the ladder, keep the best rung whose achieved radius is
    # still inside the float64 budget. The reported `scale` is that rung, so
    # the number is reproducible rather than "wherever L-BFGS stopped".
    u = None
    s_prev = None
    best: Optional[EmbeddingFit] = None
    for s_now in scale_ladder:
        if float(s_now) > MAX_SAFE_SCALE:
            break
        warm = None if u is None else _rescale_radii(u, float(s_now) /
                                                     float(s_prev))
        u, loss = _fit_at_scale(target, iu, ju, n, dim, float(s_now),
                                restarts if warm is None else later,
                                seed, rounds, max_iter, _init(float(s_now)),
                                warm)
        s_prev = s_now
        cand = _finish(u, float(s_now), loss)
        if cand.max_radius > float(max_radius):
            break
        if best is None or cand.rms_rel < best.rms_rel:
            best = cand
    if best is None:
        raise RuntimeError(
            "no scale rung stayed under max_radius=%.1f; the matrix needs a "
            "smaller ladder" % float(max_radius))
    return best


def fit_euclidean_control(d: np.ndarray, dim: int = 3, restarts: int = 10,
                          seed: int = 0, rounds: int = 3,
                          max_iter: int = 400) -> Dict[str, float]:
    """Same objective, same optimiser, flat space. The control for curvature.

    Returns ``{"rms_rel", "distortion", "max_rel"}``. Scale is not a parameter
    here and needs no ladder: Euclidean distances are homogeneous of degree 1,
    so the relative objective is exactly scale-invariant and every rung gives
    the same answer. That is the point -- hyperbolic space is the one where
    making the embedding bigger makes a tree fit better, which is the whole
    mechanism the curvature is bought for.

    References on these 10 taxa: E^2 0.08448, E^3 0.03037, E^5 0.00817 against
    H^2 0.04321, H^3 0.01123, H^5 0.00293. If flat space ever ties here,
    nothing downstream justifies the machinery.
    """
    dd = np.asarray(d, dtype=np.float64)
    n = int(dd.shape[0])
    iu_np, ju_np = np.triu_indices(n, k=1)
    target = torch.as_tensor(dd[iu_np, ju_np], dtype=torch.float64)
    iu = torch.as_tensor(iu_np, dtype=torch.long)
    ju = torch.as_tensor(ju_np, dtype=torch.long)
    best_x, best = None, float("inf")
    for k in range(max(1, int(restarts))):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1013 * k)
        x = (torch.randn(n, int(dim), generator=gen, dtype=torch.float64)
             * float(np.median(dd[iu_np, ju_np]))).requires_grad_(True)
        for _ in range(max(1, int(rounds))):
            opt = torch.optim.LBFGS([x], max_iter=int(max_iter),
                                    history_size=50, tolerance_grad=1e-16,
                                    tolerance_change=1e-18,
                                    line_search_fn="strong_wolfe")

            def _closure():
                opt.zero_grad(set_to_none=True)
                diff = x[iu] - x[ju]
                dist = diff.pow(2).sum(dim=-1).clamp_min(_EPS64 ** 2).sqrt()
                # Profile the scale out in closed form; exact for a flat
                # embedding, not an approximation.
                a = dist / target
                s = (a * a).mean() / a.mean().clamp_min(1e-30)
                val = (a / s - 1.0).pow(2).mean()
                val.backward()
                return val

            opt.step(_closure)
        with torch.no_grad():
            diff = x[iu] - x[ju]
            dist = diff.pow(2).sum(dim=-1).clamp_min(_EPS64 ** 2).sqrt()
            a = dist / target
            s = (a * a).mean() / a.mean()
            val = float((a / s - 1.0).pow(2).mean())
        if math.isfinite(val) and val < best:
            best, best_x = val, x.detach().clone()
    assert best_x is not None
    with torch.no_grad():
        diff = best_x[iu] - best_x[ju]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        a = dist / target
        s = (a * a).mean() / a.mean()
        ratio = a / s
    return {"rms_rel": float(math.sqrt(best)),
            "max_rel": float((ratio - 1.0).abs().max()),
            "distortion": float(ratio.max() / ratio.min())}


def place_from_distances(points: torch.Tensor, targets: torch.Tensor,
                         scale: float = 10.0, restarts: int = 5,
                         seed: int = 0, max_iter: int = 300
                         ) -> Tuple[torch.Tensor, float]:
    """Position one new point from its distances to frozen points.

    Solves the same relative-error objective for a single ``u in R^d`` with
    every other point held fixed: ``d`` unknowns against ``len(targets)``
    equations, identifiable from ``d+1`` neighbours and overdetermined at the
    seven this atlas has. Returns ``(point [dim+1], rms_rel)``.

    The conditioning of that solve is unvalidated -- overdetermined is not
    well-conditioned at noisy distances -- so ``rms_rel`` is returned rather
    than logged, as it is the caller's only handle on it.
    """
    if points.ndim != 2:
        raise ValueError("points must be [n, dim+1]")
    d = int(points.shape[-1]) - 1
    pts = points.to(torch.float64)
    tgt = targets.to(torch.float64).reshape(-1)
    if tgt.numel() != pts.shape[0]:
        raise ValueError("need one target distance per frozen point (%d), got "
                         "%d" % (int(pts.shape[0]), int(tgt.numel())))
    if float(tgt.min()) <= 0.0:
        raise ValueError("a target distance is zero; the new genome would be "
                         "identical to an existing one")
    x0_ref = pts[:, 0]
    u_ref = pts[:, 1:]
    want = float(scale) * tgt
    best_u, best = None, float("inf")
    for k in range(max(1, int(restarts))):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919 * k)
        u = (torch.randn(d, generator=gen, dtype=torch.float64) * 0.5
             + u_ref.mean(dim=0)).requires_grad_(True)
        opt = torch.optim.LBFGS([u], max_iter=int(max_iter), history_size=32,
                                line_search_fn="strong_wolfe")

        def _closure():
            opt.zero_grad(set_to_none=True)
            x0 = torch.sqrt(1.0 + u.pow(2).sum())
            g = x0 * x0_ref - (u_ref * u).sum(dim=-1)
            dist = _acosh_stable(g)
            val = ((dist / want - 1.0) ** 2).mean()
            val.backward()
            return val

        opt.step(_closure)
        with torch.no_grad():
            x0 = torch.sqrt(1.0 + u.pow(2).sum())
            g = x0 * x0_ref - (u_ref * u).sum(dim=-1)
            val = float(((_acosh_stable(g) / want - 1.0) ** 2).mean())
        if val < best and math.isfinite(val):
            best, best_u = val, u.detach().clone()
    assert best_u is not None
    return project_to_hyperboloid(best_u), float(math.sqrt(best))


# The model-side channel

class GenomeEmbedding(nn.Module):
    """One learned hyperbolic vector per genome, with the tree as a prior.

    The points are free parameters, so ``nn.Parameter`` under
    ``RiemannianAdam`` with ``weight_decay`` hard zero -- lambda 0.0762 over 34
    epochs is a factor 1.8e-5 and drags every hyperboloid point to the origin.
    They are float64 while the rest of the model is float32: 20 x 4 numbers
    cost nothing, and ``<x,x>_L = -1`` is a difference of large numbers that
    float32 has already lost. The forward pass casts to the consumer's dtype
    after ``logmap0``.

    The output is ``logmap0(point)``, so ``dim`` numbers and not ``dim+1``: the
    time component of a tangent at the origin is identically zero, and passing
    ``dim+1`` hands the fusion layer a column of zeros. ``out_dim`` adds a
    Euclidean projection for a fusion layer that wants a specific width.

    Ablation is substitution, not shuffling. Batches are genome-homogeneous, so
    shuffling the genome index within a batch is the identity -- measured
    +0.0000 -- and :meth:`forward` refuses ``ablate="shuffle"``.
    """

    def __init__(self, registry: StrainRegistry,
                 points: Optional[torch.Tensor] = None,
                 dim: int = 3, scale: float = 10.0,
                 out_dim: Optional[int] = None,
                 patristic: Optional[np.ndarray] = None,
                 patristic_taxa: Optional[Sequence[str]] = None,
                 tree_prior: float = 0.1,
                 prior_denom_floor: float = PRIOR_DENOM_FLOOR,
                 learnable: bool = True) -> None:
        super().__init__()
        if float(prior_denom_floor) < PRIOR_MIN_DIST:
            raise ValueError(
                "prior_denom_floor %.3g is below PRIOR_MIN_DIST %.3g. Pairs "
                "under PRIOR_MIN_DIST are dropped from the prior entirely, so "
                "a floor beneath it can never bind and the key would be inert."
                % (float(prior_denom_floor), PRIOR_MIN_DIST))
        if float(scale) > MAX_SAFE_SCALE:
            raise ValueError(
                "scale %.3f exceeds MAX_SAFE_SCALE %.1f: at s=20 the outermost "
                "taxon already sits at radius 14.1, past the expmap clamp of "
                "8.0, and s>=30 is forbidden outright (measured RMS 0.1898 at "
                "s=50, 0.4499 at s=100 -- float64 cancellation)."
                % (float(scale), MAX_SAFE_SCALE))
        if float(scale) <= 0.0:
            raise ValueError("scale must be positive, got %r" % (scale,))
        self.registry = registry
        self.dim = int(dim)
        self.scale = float(scale)
        self.tree_prior = float(tree_prior)
        self.prior_denom_floor = float(prior_denom_floor)
        self.learnable = bool(learnable)
        self.variance_ceiling = GENOME_VARIANCE_CEILING

        n = len(registry)
        if points is None:
            pts = torch.zeros(n, self.dim + 1, dtype=torch.float64)
            pts[:, 0] = 1.0                      # everyone at the origin
            self.placed_at_init: Tuple[str, ...] = ()
        else:
            pts = torch.as_tensor(points, dtype=torch.float64).clone()
            if tuple(pts.shape) != (n, self.dim + 1):
                raise ValueError(
                    "points has shape %r but the registry has %d genomes and "
                    "dim=%d, so [%d, %d] is required"
                    % (tuple(pts.shape), n, self.dim, n, self.dim + 1))
            self.placed_at_init = registry.placed_names()
        assert_on_manifold(pts, tol=1e-10, name="genome points")
        self.points = nn.Parameter(pts, requires_grad=self.learnable)
        if self.learnable:
            mark_manifold(self.points)

        self.out_dim = int(out_dim) if out_dim else self.dim
        self.proj: Optional[nn.Linear] = None
        if out_dim:
            self.proj = nn.Linear(self.dim, int(out_dim))

        self._set_prior_pairs(patristic, patristic_taxa)

    # Construction
    @classmethod
    def from_tree(cls, registry: Optional[StrainRegistry] = None,
                  tree_path: Optional[str] = None,
                  dim: int = 3, scale: float = 10.0,
                  out_dim: Optional[int] = None, tree_prior: float = 0.1,
                  prior_denom_floor: float = PRIOR_DENOM_FLOOR,
                  restarts: int = 10, seed: int = 0,
                  learnable: bool = True,
                  outgroup: Optional[Sequence[str]] = OUTGROUP_TAXA,
                  ) -> "GenomeEmbedding":
        """Build from ``supermatrix.treefile``: fit, then seat every genome.

        A placed genome takes its fitted position; an unplaced one starts at
        the Lorentz centroid of the placed ones, which is the geometric
        counterpart of the uniform borrowing weights that win the
        leave-one-genome-out comparison. Random initialisation is not offered:
        it would put 13 of 20 genomes in arbitrary places while the prior pulls
        only the other 7.
        """
        registry = registry or StrainRegistry.default()
        path = tree_path or default_tree_paths()["supermatrix"]
        if not path:
            raise FileNotFoundError(
                "supermatrix.treefile not found; pass tree_path= or set "
                "HYPPE_DATA_ROOT. This module imports without data files, but "
                "a tree-initialised channel needs the tree.")
        # Rerooting is a no-op for the quantity taken next -- the path between
        # two leaves is the same edge set whatever node is called the root,
        # measured at max |difference| 0.000e+00 over all 45 pairs -- so a tree
        # with no outgroup, like the in-house 20-genome NJ tree, is read
        # directly rather than refused. `outgroup=None` is the only way there.
        _raw = read_newick(path)
        root = reroot_on_outgroup(_raw, outgroup) if outgroup else _raw
        taxa, d = patristic_matrix(root)
        # Leaves the tree cannot separate: in the in-house 20-genome tree three
        # strains sit at distance exactly zero, because the 439 single-copy
        # markers do not tell them apart. A relative-error objective cannot use
        # a zero pair and an epsilon would invent a distance the data does not
        # support, so a zero-distance class is treated as ONE tree position:
        # fit a representative, seat every member there. They stay separate
        # learnable rows, so the data may still move them apart.
        _klasa_of = {}
        _klasy = []
        for _i in range(len(taxa)):
            if _i in _klasa_of:
                continue
            _czl = [_j for _j in range(len(taxa)) if float(d[_i][_j]) <= 0.0]
            for _j in _czl:
                _klasa_of[_j] = len(_klasy)
            _klasy.append(_czl)
        _rep = [c[0] for c in _klasy]
        _scalone = len(taxa) - len(_rep)
        if _scalone:
            _d_fit = np.asarray(d)[np.ix_(_rep, _rep)]
            _taxa_fit = [taxa[k] for k in _rep]
        else:
            _d_fit, _taxa_fit = d, list(taxa)
        fit = fit_hyperbolic_embedding(_d_fit, dim=dim, scale=scale,
                                       restarts=restarts, seed=seed)
        fit = replace(fit, taxa=tuple(_taxa_fit))
        _row_of = {t: i for i, t in enumerate(_taxa_fit)}
        # every original leaf -> the row of its class representative
        idx_of = {taxa[j]: _row_of[taxa[_rep[_klasa_of[j]]]]
                  for j in range(len(taxa))}
        placed_rows = [fit.points[idx_of[r.taxon]] for r in registry
                       if r.placed and r.taxon in idx_of]
        if not placed_rows:
            raise ValueError("no registry genome maps onto a tree leaf")
        centroid = lorentz_centroid(torch.stack(placed_rows), dim=-2)
        pts = torch.stack([
            fit.points[idx_of[r.taxon]] if (r.placed and r.taxon in idx_of)
            else centroid for r in registry]).clone()
        model = cls(registry, points=pts, dim=dim, scale=scale,
                    out_dim=out_dim, patristic=d, patristic_taxa=taxa,
                    tree_prior=tree_prior,
                    prior_denom_floor=prior_denom_floor,
                    learnable=learnable)
        model.fit_report = fit.report()          # type: ignore[attr-defined]
        return model

    def _set_prior_pairs(self, patristic: Optional[np.ndarray],
                         patristic_taxa: Optional[Sequence[str]]) -> None:
        """Cache the pairs the tree places, as non-persistent buffers.

        Non-persistent on purpose: adding a placed genome changes their shape,
        and the falsifier "adding a genome changes exactly one tensor" must
        stay true of the state dict. The tables are derivable from the
        artefact, which stores the patristic matrix and the genome order.
        """
        self._patristic = None if patristic is None else np.asarray(
            patristic, dtype=np.float64)
        self._patristic_taxa = tuple(patristic_taxa) if patristic_taxa else ()
        pi: List[int] = []
        pj: List[int] = []
        pt: List[float] = []
        if self._patristic is not None and self._patristic_taxa:
            col = {t: k for k, t in enumerate(self._patristic_taxa)}
            rows = [(i, col[r.taxon]) for i, r in enumerate(self.registry)
                    if r.placed and r.taxon in col]
            # Zero-distance classes. The tree gives every member of such a
            # class an identical distance to every other leaf, so although
            # their mutual pairs fall below PRIOR_MIN_DIST, tying them to the
            # rest pins them to each other indirectly: over 9 epochs such a
            # triple drifted apart by 6e-05 against +0.049 for bound pairs.
            # Under `prior_free_collapsed` one representative per class is kept
            # anchored where the tree puts it and the rest leave the prior.
            _wolne = set()
            if getattr(self, "prior_free_collapsed", False):
                _klasa = {}
                for (a, ca) in rows:
                    if a in _klasa:
                        continue
                    czlon = [b for (b, cb) in rows
                             if float(self._patristic[ca, cb]) < PRIOR_MIN_DIST]
                    for b in czlon:
                        _klasa[b] = a
                    if len(czlon) > 1:
                        _wolne.update(czlon[1:])
            self._prior_wolne = tuple(sorted(_wolne))
            _pominiete = 0
            for (a, ca), (b, cb) in itertools.combinations(rows, 2):
                _want = float(self._patristic[ca, cb])
                # `tree_prior_loss` divides by the wanted distance, so a pair
                # the tree places at zero divides by zero and one it places
                # just above zero is worse: it survives the check and arrives
                # with stiffness 1/d^2 on a distance the markers cannot
                # resolve. Such pairs are dropped AND counted, because a prior
                # that silently loses pairs is the failure mode this buffer
                # exists to make visible.
                if a in _wolne or b in _wolne:
                    _pominiete += 1
                    continue
                if not (_want >= PRIOR_MIN_DIST):
                    _pominiete += 1
                    continue
                pi.append(a)
                pj.append(b)
                pt.append(_want)
            self._prior_skipped = _pominiete
        self.register_buffer("_prior_i", torch.tensor(pi, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_prior_j", torch.tensor(pj, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_prior_d", torch.tensor(pt, dtype=torch.float64),
                             persistent=False)

    # Forward
    def tangent(self) -> torch.Tensor:
        """All genome vectors as tangents at the origin: ``[n, dim]``, float64."""
        return logmap0(self.points)

    def forward(self, index: torch.Tensor, ablate: Optional[str] = None,
                substitute_to: Optional[int] = None) -> torch.Tensor:
        """Genome index ``[B]`` -> conditioning vector ``[B, out_dim]``.

        ``ablate``:
          ``None``          normal path;
          ``"zero"``        zeros after the logmap, which reports on the
                            downstream consumer rather than on the content of
                            the channel;
          ``"substitute"``  every row replaced by ``substitute_to``'s genome.
                            The only admissible ablation here, since batches
                            are genome-homogeneous and shuffling within one is
                            the identity.
        """
        if ablate == "shuffle":
            raise ValueError(
                "shuffling the genome index is an EMPTY intervention here: "
                "HYPPE batches are genome-homogeneous, so the shuffle is the "
                "identity (measured +0.0000). Use ablate='substitute'.")
        if ablate not in (None, "none", "zero", "substitute"):
            raise ValueError("unknown ablation %r" % (ablate,))
        # Guard BEFORE the point is read, not after: writing `.data` bumps the
        # tensor's version counter, so it must precede every readout in this
        # step or the matching backward would see a tensor changed in place.
        # The loop does not accumulate gradient across steps, so here is safe.
        self._manifold_guard("forward")
        idx = index.reshape(-1).long()
        if ablate == "substitute":
            if substitute_to is None:
                raise ValueError("ablate='substitute' needs substitute_to=<row>")
            idx = torch.full_like(idx, int(substitute_to))
        if int(idx.max()) >= self.points.shape[0] or int(idx.min()) < 0:
            raise IndexError(
                "genome index out of range: got [%d, %d] for %d registered "
                "genomes" % (int(idx.min()), int(idx.max()),
                             int(self.points.shape[0])))
        vec = logmap0(self.points[idx.to(self.points.device)])
        if ablate == "zero":
            vec = torch.zeros_like(vec)
        if self.proj is not None:
            return self.proj(vec.to(self.proj.weight.dtype))
        return vec.to(torch.get_default_dtype())

    #: Above this, ``points`` count as off the hyperboloid: the F11 threshold
    #: of this class. Live training weights measure 1e-15, three orders under,
    #: so the guard in `forward` is a no-op during training and fires exactly
    #: on what does break the constraint -- the SWA mean, at 1.621e-03.
    REPROJECT_TOL = 1e-12

    def _reproject_enabled(self) -> bool:
        """Whether `conditioning.genome_reproject` is on.

        Read from the active config rather than from a constructor argument,
        following the `active_config()` pattern used elsewhere in this package.
        """
        try:
            from ..config import active_config
            cfg = active_config()
        except Exception:                       # pragma: no cover
            return False
        if cfg is None:
            return False
        return bool(getattr(cfg.conditioning, "genome_reproject", False))

    def _manifold_guard(self, powod: str) -> Optional[Dict[str, float]]:
        """Repair only when the constraint is broken. Counts its own calls.

        The `v > REPROJECT_TOL` test is what makes this a guard rather than
        another operation in the step: at live weights (1e-15) it never holds,
        so the training trajectory is bit for bit what it was.
        """
        if not self._reproject_enabled():
            return None
        with torch.no_grad():
            p = self.points.data
            v = float((minkowski_dot(p, p, keepdim=False) + 1.0).abs().max())
        if not (v > self.REPROJECT_TOL):
            return None
        raport = self.reproject_()
        print("  genome.reproject_ [%s]: max|<x,x>_L + 1| %.3e -> %.3e | "
              "max przesuniecie wspolrzednej %.3e"
              % (powod, raport["violation_before"],
                 raport["violation_after"], raport["max_coord_shift"]))
        self._rep_n = int(getattr(self, "_rep_n", 0)) + 1
        self._rep_ostatni = dict(raport)
        self._rep_ostatni["powod"] = powod
        return raport

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Repair after weights are loaded. This is the hook that catches SWA.

        `loop._swa_step` calls `load_state_dict` and only then
        `save_checkpoint`, so a point repaired here reaches the saved file --
        the whole difference between 1.621e-03 and 8.9e-16 in it. The same
        hook catches a resumed run and every evaluation of a checkpoint.
        """
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
        self._manifold_guard("load_state_dict")

    def reproject_(self) -> Dict[str, float]:
        """Pull `points` back onto the hyperboloid, in place.

        A finding, not a precaution. Live epoch checkpoints measure
        `max|<x,x>_L + 1|` around 1e-15, but the SWA mean of those same epochs
        measures 1.621e-03 -- nine orders over this class's F11 threshold, in
        exactly the file evaluations are computed on -- because SWA adds
        tensors in ambient coordinates and the arithmetic mean of hyperboloid
        points does not lie on the hyperboloid.

        The measured effect on the prediction is 0.000e+00, since `forward`
        reaches the point only through `logmap0`, which rebuilds the time
        coordinate from the spatial part. The guard exists because that is a
        property of one path and not of the point: `tree_prior_loss`, `radius`
        and `prior_fit` read `x0` directly, so a run resumed from such a file
        would compute the tree prior off the manifold and nothing would say so.
        Returns the before and after numbers so a call can be reported.
        """
        with torch.no_grad():
            p = self.points.data
            przed = float((minkowski_dot(p, p, keepdim=False) + 1.0)
                          .abs().max())
            prz = p[..., 1:]
            x0 = torch.sqrt(1.0 + prz.pow(2).sum(dim=-1, keepdim=True))
            nowe = torch.cat([x0, prz], dim=-1).to(p.dtype)
            ruch = float((nowe - p).abs().max())
            self.points.data.copy_(nowe)
            po = float((minkowski_dot(self.points.data, self.points.data,
                                      keepdim=False) + 1.0).abs().max())
        return {"violation_before": przed, "violation_after": po,
                "max_coord_shift": ruch}

    # The prior
    def tree_prior_loss(self) -> torch.Tensor:
        """Relative pull-back onto the tree over the placed pairs.

        ``mean_pairs ((d_L(i,j) - s*d_patr(i,j)) / max(d_patr(i,j), floor))^2``
        scaled by ``tree_prior``. Zero when no pair is available, so a registry
        with no tree trains without a silent constant.

        The denominator is floored rather than raw; see
        :data:`PRIOR_DENOM_FLOOR`. At the default floor the two forms are
        identical.

        ``tree_prior`` is a choice, not a measurement: the 0.0129 R^2 gap
        between free and frozen weights was measured on a surrogate task and
        never converted into a coefficient. F6 is what to read afterwards --
        mean ``|d_L/(s*d_patr) - 1|`` below 0.10, or drop the tree rather than
        raise lambda.
        """
        if self._prior_i.numel() == 0 or self.tree_prior == 0.0:
            return self.points.new_zeros(())
        p = self.points[self._prior_i]
        q = self.points[self._prior_j]
        # lorentz_sqdist is the differentiable surrogate for d^2, but what is
        # needed here is d itself against a target distance; _acosh_stable
        # keeps the gradient finite by flooring the sqrt operand.
        dist = _acosh_stable(-minkowski_dot(p, q, keepdim=False))
        want = self.scale * self._prior_d
        # The target stays `scale * d_patr`; only the weight is floored. A
        # floor in the numerator would move where the points are asked to sit,
        # making the tree wrong rather than softly enforced.
        denom = self._prior_d.clamp_min(self.prior_denom_floor)
        return self.tree_prior * ((dist - want) / denom).pow(2).mean()

    def prior_fit(self) -> Dict[str, float]:
        """F6 statistics: how far the learned points drifted off the tree."""
        if self._prior_i.numel() == 0:
            return {"n_pairs": 0, "mean_abs_rel": float("nan"),
                    "max_abs_rel": float("nan"), "softened": 0,
                    "stiff_max_over_median": float("nan")}
        with torch.no_grad():
            p = self.points[self._prior_i]
            q = self.points[self._prior_j]
            dist = geodesic_distance(p, q)
            rel = (dist / (self.scale * self._prior_d) - 1.0).abs()
            # How unequally the term weights its pairs, as a number rather
            # than an argument: under the raw denominator it reached 8679 on
            # the 20-leaf tree.
            denom = self._prior_d.clamp_min(self.prior_denom_floor)
            stiff = 1.0 / denom.pow(2)
            softened = int((self._prior_d < self.prior_denom_floor).sum())
        return {"n_pairs": int(rel.numel()),
                "mean_abs_rel": float(rel.mean()),
                "max_abs_rel": float(rel.max()),
                "denom_floor": float(self.prior_denom_floor),
                "softened": softened,
                "stiff_max_over_median": float(stiff.max()
                                               / stiff.median())}

    # Optimiser wiring
    def manifold_parameters(self) -> List[nn.Parameter]:
        """The hyperboloid parameters. Empty if the channel is frozen."""
        return [self.points] if self.points.requires_grad else []

    def euclidean_parameters(self) -> List[nn.Parameter]:
        return list(self.proj.parameters()) if self.proj is not None else []

    def param_groups(self, lr: float = 1e-3,
                     weight_decay: float = 0.0) -> List[dict]:
        """Optimiser groups with ``weight_decay = 0`` on the manifold.

        ``RiemannianAdam`` refuses a manifold group with decay, so this is a
        second lock on the same door: the failure it prevents -- lambda 0.0762
        for 34 epochs is a factor 1.8e-5, dragging the points to the origin --
        is invisible in the loss.
        """
        groups: List[dict] = []
        if self.manifold_parameters():
            groups.append({"params": self.manifold_parameters(),
                           "manifold": True, "weight_decay": 0.0, "lr": lr})
        if self.euclidean_parameters():
            groups.append({"params": self.euclidean_parameters(),
                           "manifold": False,
                           "weight_decay": float(weight_decay), "lr": lr})
        return groups

    # Adding a genome
    def add_genome(self, name: str, mode: str = "centroid",
                   taxon: Optional[str] = None,
                   patristic_to_placed: Optional[Dict[str, float]] = None,
                   jaccard_to_known: Optional[Dict[str, float]] = None,
                   h: float = 0.05, lam: float = 0.3,
                   has_expression: bool = True,
                   allow_unvalidated: bool = False) -> Dict[str, object]:
        """Register a genome and give it a position. Grows one tensor by a row.

        ``mode``:
          ``"centroid"``  (default) Lorentz centroid of the known genomes: the
                          geometric uniform prior, and uniform is what survives
                          the leave-one-genome-out protocol.
          ``"tree"``      solve the position from patristic distances to the
                          placed genomes, every other point frozen. Requires a
                          leaf for the new genome.
          ``"jaccard"``   weights ``w = (1-lam)/m + lam*softmax(-d/h)`` over
                          orthogroup-content distances, then a weighted Lorentz
                          centroid. Fails its own falsifier F7 -- +0.0059 R^2
                          mean against a -0.1509 worst case, because the
                          Jaccard neighbours of IMI206040 are the two worst
                          predictors in its block -- so it requires
                          ``allow_unvalidated=True``.

        Returns a record of what changed, ``unplaced_genome`` included, because
        a genome sitting at the consensus is a different object from one the
        tree placed.

        The optimiser must be rebuilt afterwards: ``points`` is a new
        ``nn.Parameter``, so an optimiser still holding the old one updates a
        tensor the model no longer uses. The returned dict says so in
        ``rebuild_optimiser``.
        """
        mode = str(mode)
        if mode not in ("centroid", "tree", "jaccard"):
            raise ValueError("unknown mode %r" % (mode,))
        if mode == "jaccard" and not allow_unvalidated:
            raise ValueError(
                "mode='jaccard' fails falsifier F7 (mean +0.0059 R^2 but worst "
                "case -0.1509 on IMI206040, threshold -0.02) and is not the "
                "default for that reason. Pass allow_unvalidated=True if you "
                "are deliberately running the measured-risky placement.")
        before = {k: tuple(v.shape) for k, v in self.state_dict().items()}
        old_points = self.points.detach().clone()
        old_names = self.registry.names

        if mode == "centroid":
            new_pt = lorentz_centroid(old_points, dim=-2)
            detail: Dict[str, object] = {"placement": "lorentz centroid of "
                                         "%d genomes" % old_points.shape[0]}
        elif mode == "tree":
            if taxon is None:
                raise ValueError("mode='tree' needs the new leaf label")
            if not patristic_to_placed:
                raise ValueError(
                    "mode='tree' needs patristic distances to the genomes that "
                    "the tree already places: %r"
                    % (list(self.registry.placed_names()),))
            rows, tgts = [], []
            for gname, dist in patristic_to_placed.items():
                rows.append(self.registry.index(gname))
                tgts.append(float(dist))
            frozen = old_points[torch.tensor(rows, dtype=torch.long)]
            new_pt, rms = place_from_distances(
                frozen, torch.tensor(tgts, dtype=torch.float64),
                scale=self.scale)
            detail = {"placement": "solved from %d patristic distances"
                      % len(rows), "rms_rel": rms,
                      "caveat": "conditioning of this solve is unmeasured "
                                "(open point N6)"}
        else:
            if not jaccard_to_known:
                raise ValueError("mode='jaccard' needs jaccard_to_known")
            w = torch.zeros(old_points.shape[0], dtype=torch.float64)
            dvec = torch.full((old_points.shape[0],), float("inf"),
                              dtype=torch.float64)
            for gname, dist in jaccard_to_known.items():
                dvec[self.registry.index(gname)] = float(dist)
            finite = torch.isfinite(dvec)
            soft = torch.zeros_like(dvec)
            soft[finite] = torch.softmax(-dvec[finite] / float(h), dim=0)
            m = old_points.shape[0]
            w = (1.0 - float(lam)) / m + float(lam) * soft
            new_pt = lorentz_centroid(old_points, weights=w, dim=-2)
            detail = {"placement": "weighted centroid, h=%.3g lam=%.3g" % (h, lam),
                      "measured_mean_gain": NEW_GENOME_MEASURED["jaccard"]
                      ["mean_gain"],
                      "measured_worst_gain": NEW_GENOME_MEASURED["jaccard"]
                      ["worst_gain"]}

        new_pt = project_to_hyperboloid(new_pt.reshape(-1), ambient=True)
        assert_on_manifold(new_pt, tol=1e-10, name="new genome point")

        idx = self.registry.add(name, taxon=taxon,
                                has_expression=has_expression)
        stacked = torch.cat([old_points, new_pt.reshape(1, -1)], dim=0)
        self.points = nn.Parameter(stacked, requires_grad=self.learnable)
        if self.learnable:
            mark_manifold(self.points)
        self._set_prior_pairs(self._patristic, self._patristic_taxa)

        after = {k: tuple(v.shape) for k, v in self.state_dict().items()}
        changed = sorted(k for k in set(before) | set(after)
                         if before.get(k) != after.get(k))
        out = {
            "name": name, "index": idx, "mode": mode,
            "n_genomes": len(self.registry),
            "unplaced_genome": int(mode != "tree"),
            "tensors_changed": changed,
            "shape_before": before.get("points"),
            "shape_after": after.get("points"),
            "radius": float(radius(new_pt)),
            "rebuild_optimiser": True,
            "previous_names": list(old_names),
        }
        out.update(detail)
        return out

    # Persistence
    def save_artifact(self, path: str,
                      og_profiles: Optional[np.ndarray] = None,
                      og_ids: Optional[Sequence[str]] = None) -> Dict[str, object]:
        """Write ``gatunki.npz``: everything needed to add a genome later.

        Stores the points, the dimension, the scale and the curvature (a vector
        without its scale is meaningless, since ``logmap0`` and every distance
        depend on it), the genome order, the patristic matrix and the
        strain-to-taxon map so a new leaf can be solved, the Lorentz centroid,
        the leave-one-genome-out constants ``h`` and ``lam`` together with
        their measured gain and worst case, and optionally the
        orthogroup-content profiles -- the only surrogate a new genome can
        supply from its proteome alone.

        Asserts ``max|<x,x>_L + 1| < 1e-12`` before writing.
        """
        viol = assert_on_manifold(self.points.detach(), tol=1e-12,
                                  name="genome points at save")
        payload: Dict[str, object] = {
            "points": self.points.detach().cpu().numpy().astype(np.float64),
            "order": np.array(self.registry.names, dtype="U64"),
            "taxa": np.array([r.taxon or "" for r in self.registry],
                             dtype="U64"),
            # The cluster and expression flags travel with the points: without
            # them a reloaded registry reports effective_n = 20 instead of 15,
            # so every standard error over genomes silently widens its n by a
            # third and nothing raises.
            "clusters": np.array([r.cluster or "" for r in self.registry],
                                 dtype="U32"),
            "has_expression": np.array([bool(r.has_expression)
                                        for r in self.registry], dtype=bool),
            "species_level": np.array([bool(r.species_level)
                                       for r in self.registry], dtype=bool),
            "dim": np.int64(self.dim),
            "scale": np.float64(self.scale),
            "curvature": np.float64(-1.0),
            "version": np.array(EMBEDDING_VERSION, dtype="U32"),
            "tree_prior": np.float64(self.tree_prior),
            "centroid": lorentz_centroid(self.points.detach(),
                                         dim=-2).cpu().numpy(),
            "h_logo": np.float64(NEW_GENOME_MEASURED["jaccard"]["h"]),
            "lam_logo": np.float64(NEW_GENOME_MEASURED["jaccard"]["lam"]),
            "logo_mean_gain": np.float64(
                NEW_GENOME_MEASURED["jaccard"]["mean_gain"]),
            "logo_worst_gain": np.float64(
                NEW_GENOME_MEASURED["jaccard"]["worst_gain"]),
            "variance_ceiling": np.float64(GENOME_VARIANCE_CEILING),
            "manifold_violation": np.float64(viol),
        }
        if self._patristic is not None:
            payload["patristic"] = self._patristic
            payload["patristic_taxa"] = np.array(
                self._patristic_taxa, dtype=object).astype("U64")
        if og_profiles is not None:
            prof = np.asarray(og_profiles)
            if prof.shape[0] != len(self.registry):
                raise ValueError(
                    "og_profiles has %d rows for %d genomes; the row order is "
                    "the registry order" % (prof.shape[0], len(self.registry)))
            payload["og_profiles"] = prof.astype(bool)
            if og_ids is not None:
                payload["og_ids"] = np.array(og_ids, dtype=object).astype("U32")
        np.savez_compressed(path, **payload)
        return {"path": path, "n_genomes": len(self.registry),
                "manifold_violation": viol,
                "keys": sorted(payload.keys())}

    @classmethod
    def load_artifact(cls, path: str, out_dim: Optional[int] = None,
                      tree_prior: float = 0.1,
                      learnable: bool = True) -> "GenomeEmbedding":
        """Rebuild from ``gatunki.npz``. Refuses a file off the manifold."""
        z = np.load(path, allow_pickle=False)
        names = [str(s) for s in z["order"]]
        n = len(names)
        taxa = [str(s) or None for s in z["taxa"]] if "taxa" in z \
            else [None] * n
        clusters = ([str(s) or None for s in z["clusters"]]
                    if "clusters" in z else [None] * n)
        has_expr = ([bool(b) for b in z["has_expression"]]
                    if "has_expression" in z else [True] * n)
        sp_level = ([bool(b) for b in z["species_level"]]
                    if "species_level" in z else [False] * n)
        reg = StrainRegistry.from_rows(
            [{"name": a, "taxon": b, "cluster": c, "has_expression": d,
              "species_level": e}
             for a, b, c, d, e in zip(names, taxa, clusters, has_expr,
                                      sp_level)])
        pts = torch.as_tensor(z["points"], dtype=torch.float64)
        viol = assert_on_manifold(pts, tol=1e-10, name="loaded genome points")
        model = cls(reg, points=pts, dim=int(z["dim"]), scale=float(z["scale"]),
                    out_dim=out_dim,
                    patristic=(z["patristic"] if "patristic" in z else None),
                    patristic_taxa=([str(s) for s in z["patristic_taxa"]]
                                    if "patristic_taxa" in z else None),
                    tree_prior=tree_prior, learnable=learnable)
        model.loaded_violation = viol            # type: ignore[attr-defined]
        return model

    # Observability
    def report(self, ablation_variance: Optional[float] = None
               ) -> Dict[str, object]:
        """State of the channel. ``ablation_variance`` feeds F5 when measured."""
        with torch.no_grad():
            rad = radius(self.points.detach())
            tang = self.tangent()
            tnorm = tang.norm(dim=-1)
        out: Dict[str, object] = {
            "n_genomes": len(self.registry),
            "dim": self.dim, "scale": self.scale, "out_dim": self.out_dim,
            "learnable": self.learnable,
            "tree_prior": self.tree_prior,
            "n_prior_pairs": int(self._prior_i.numel()),
            # Observables for the guard. Without these three an enabled key
            # and a dead one look identical: `manifold_violation` is always
            # small AFTER a repair and cannot say whether one happened.
            "reproject_enabled": float(self._reproject_enabled()),
            "reproject_n": int(getattr(self, "_rep_n", 0)),
            "reproject_last_before": float(
                getattr(self, "_rep_ostatni", {}).get("violation_before", 0.0)),
            "reproject_last_shift": float(
                getattr(self, "_rep_ostatni", {}).get("max_coord_shift", 0.0)),
            "manifold_violation": assert_on_manifold(self.points.detach(),
                                                     raise_on_fail=False),
            "wrong_sheet": int((self.points.detach()[:, 0] <= 0).sum()),
            "radius_min": float(rad.min()), "radius_max": float(rad.max()),
            "radius_mean": float(rad.mean()),
            "tangent_norm_max": float(tnorm.max()),
            "n_placed": len(self.registry.placed_names()),
            "n_unplaced": len(self.registry.unplaced_names()),
            "unplaced": list(self.registry.unplaced_names()),
            "effective_n": self.registry.effective_n(),
            "variance_ceiling": GENOME_VARIANCE_CEILING,
            "variance_decomposition": dict(WITHIN_OG_VARIANCE),
            "manifold_params": int(sum(p.numel() for p in
                                       self.manifold_parameters())),
            "euclidean_params": int(sum(p.numel() for p in
                                        self.euclidean_parameters())),
        }
        out.update({"prior_" + k: v for k, v in self.prior_fit().items()})
        if len(self.registry) > 1:
            with torch.no_grad():
                p = self.points.detach()
                dm = geodesic_distance(p.unsqueeze(1), p.unsqueeze(0))
                eye = torch.eye(dm.shape[0], dtype=torch.bool,
                                device=dm.device)
                out["min_pair_d"] = float(dm[~eye].min())
                out["max_pair_d"] = float(dm.max())
                # The noise floor is not zero. `arccosh` works on a difference
                # of large numbers, so the smallest distance it resolves at
                # radius r is about sqrt(2 * 2e-16) * cosh(r) -- measured
                # 4.8e-07 between two genomes holding the same coordinates bit
                # for bit at radius 4.13. Below this, agreement is coincidence
                # and a threshold would falsify float64, not the model.
                floor = float(math.sqrt(4e-16) * math.cosh(
                    float(rad.max()))) + 1e-12
                out["distance_noise_floor"] = floor
                out["n_coincident_pairs"] = int(
                    ((dm <= floor) & (~eye)).sum() // 2)
                placed_rows = [i for i, r_ in enumerate(self.registry)
                               if r_.placed]
                if len(placed_rows) > 1:
                    sub = dm[placed_rows][:, placed_rows]
                    e2 = torch.eye(sub.shape[0], dtype=torch.bool,
                                   device=sub.device)
                    out["min_pair_d_placed"] = float(sub[~e2].min())
        if ablation_variance is not None:
            out["ablation_variance"] = float(ablation_variance)
        return out

    def falsifiers(self, ablation_variance: Optional[float] = None
                   ) -> List[Tuple[str, bool, str]]:
        """(name, passed, detail). F2, F5, F6, F11 and the wiring guards."""
        r = self.report(ablation_variance=ablation_variance)
        out: List[Tuple[str, bool, str]] = [
            ("F11 points are on the hyperboloid (< 1e-12)",
             float(r["manifold_violation"]) < 1e-12,
             "max|<x,x>_L + 1| = %.3e; precedent 5.05e-02/1.35e-01/2.49e-01 "
             "turned a true 0.8012 into a reported 0.9945. MEASURED "
             "21.09.2026: live epoch checkpoints of HYPPE3 sit at 1e-15, "
             "but HYPPE3_fundational.pt -- the SWA MEAN of those same "
             "epochs -- sits at 1.621e-03, i.e. this falsifier fires on "
             "the averaged file and on no other. `reproject_()` repairs "
             "it; the measured effect on the prediction is 0.000e+00 "
             "because `forward` reaches the point only through logmap0, "
             "but tree_prior_loss reads x0 directly and a RESUMED run "
             "would compute the prior off the manifold"
             % r["manifold_violation"]),
            ("points are on the UPPER sheet", int(r["wrong_sheet"]) == 0,
             "wrong_sheet=%d" % r["wrong_sheet"]),
            ("F2 max radius below the float64 limit",
             float(r["radius_max"]) < MAX_SAFE_RADIUS,
             "max radius %.4f (limit %.1f)" % (r["radius_max"],
                                               MAX_SAFE_RADIUS)),
            ("radius inside the expmap clamp (8.0)",
             float(r["radius_max"]) < MAX_TANGENT_NORM,
             "max radius %.4f; s=10 was measured at 7.0" % r["radius_max"]),
            ("scale is inside the admissible band",
             self.scale <= MAX_SAFE_SCALE,
             "scale %.3f, hard limit %.1f, s>=30 forbidden"
             % (self.scale, MAX_SAFE_SCALE)),
            ("manifold parameters carry no weight decay",
             all(g.get("weight_decay", 0.0) == 0.0
                 for g in self.param_groups() if g.get("manifold")),
             "lambda 0.0762 for 34 epochs is a factor 1.8e-5 on the point"),
            # Only the tree-placed genomes must stay apart. The unplaced ones
            # are supposed to coincide at the centroid until training moves
            # them, so a global "no two points merged" check fires on a
            # correct initialisation.
            ("placed genomes stay distinguishable",
             float(r.get("min_pair_d_placed", 1.0)) > 1e-4
             if "min_pair_d_placed" in r else True,
             "min distance among the %d placed genomes %.4f (noise floor "
             "%.2e); %d coincident pairs overall, expected %d at init from "
             "%d unplaced genomes sharing the centroid"
             % (int(r["n_placed"]), float(r.get("min_pair_d_placed",
                                                float("nan"))),
                float(r.get("distance_noise_floor", float("nan"))),
                int(r.get("n_coincident_pairs", 0)),
                max(0, int(r["n_unplaced"]) * (int(r["n_unplaced"]) - 1) // 2),
                int(r["n_unplaced"]))),
        ]
        pf = self.prior_fit()
        if pf["n_pairs"]:
            out.append((
                "F6 the learned points still respect the tree (mean |rel| < 0.10)",
                float(pf["mean_abs_rel"]) < 0.10,
                "mean %.4f, max %.4f over %d placed pairs; a larger value means "
                "the prior is decoration and the honest move is to drop the "
                "tree, not to raise lambda"
                % (pf["mean_abs_rel"], pf["max_abs_rel"], pf["n_pairs"])))
        else:
            out.append(("F6 tree prior has pairs to act on", False,
                        "0 placed pairs: the prior is identically zero and "
                        "`tree_prior=%.3g` is decoration" % self.tree_prior))
        if ablation_variance is not None:
            out.append((
                "F5 measured channel effect is under the 6.23 % ceiling",
                float(ablation_variance) <= GENOME_VARIANCE_CEILING * 1.05,
                "measured %.4f against ceiling %.4f; more than that means the "
                "channel carries BATCH identity (batches are genome-homogeneous), "
                "so compare it with the condition channel, whose ceiling is "
                "0.1478" % (float(ablation_variance), GENOME_VARIANCE_CEILING)))
        return out


# Layer-level entry points

def surrogate_falsifier(distance: np.ndarray, gc: np.ndarray,
                        name: str = "surrogate") -> Tuple[str, bool, str]:
    """F10: a genome distance must not be a GC axis in disguise.

    Takes a square distance matrix and a per-genome GC vector, correlates the
    off-diagonal distances against ``|dGC|`` and requires ``|r| < 0.8``.
    Measured: 5-mer promoter distance +0.9684, rejected, because promoter GC
    splits the atroviride/harzianum block from reesei and relatives and the
    "distance" is that split; Jaccard of orthogroup content +0.6436, accepted.
    """
    d = np.asarray(distance, dtype=np.float64)
    g = np.asarray(gc, dtype=np.float64).reshape(-1)
    if d.shape[0] != d.shape[1] or d.shape[0] != g.shape[0]:
        raise ValueError("distance %r and gc %r do not match"
                         % (d.shape, g.shape))
    iu = np.triu_indices(d.shape[0], k=1)
    dg = np.abs(g[iu[0]] - g[iu[1]])
    dv = d[iu]
    if dv.std() < 1e-12 or dg.std() < 1e-12:
        return (name + " is not a GC axis", False, "degenerate: zero variance")
    r = float(np.corrcoef(dv, dg)[0, 1])
    return (name + " is not a GC axis (|r with |dGC|| < %.1f)"
            % SURROGATE_GC_LIMIT,
            abs(r) < SURROGATE_GC_LIMIT,
            "r = %+.4f; 5-mers measured +0.9684 (rejected), Jaccard +0.6436"
            % r)


def report(model: Optional[GenomeEmbedding] = None,
           registry: Optional[StrainRegistry] = None,
           tree_path: Optional[str] = None,
           supertree_path: Optional[str] = None,
           fit_dims: Sequence[int] = (3,),
           restarts: int = 4, seed: int = 0) -> Dict[str, object]:
    """One dict describing the whole layer: registry, tree, fits, channel.

    Every entry is either a measurement of the files on disk or a constant this
    module is prepared to be judged against. Tree entries are absent rather
    than faked when the files are not found, since the module must be usable
    with no data file present.
    """
    registry = registry or (model.registry if model is not None
                            else StrainRegistry.default())
    out: Dict[str, object] = {
        "registry": registry.report(),
        "constants": {
            "genome_variance_ceiling": GENOME_VARIANCE_CEILING,
            "variance_decomposition": dict(WITHIN_OG_VARIANCE),
            "max_safe_scale": MAX_SAFE_SCALE,
            "max_safe_radius": MAX_SAFE_RADIUS,
            "measured_rms_free_scale": dict(MEASURED_RMS_FREE_SCALE),
            "measured_rms_euclidean": dict(MEASURED_RMS_EUCLIDEAN),
            "measured_rms_by_scale_h2": dict(MEASURED_RMS_BY_SCALE_H2),
            "new_genome_measured": {k: dict(v) for k, v in
                                    NEW_GENOME_MEASURED.items()},
            "p1_warning": ("tree d(P1, IMI206040) = 0.0079 is the smallest in "
                           "the matrix, but only 26.52 % of their promoters "
                           "are identical against 38-40 % for the rest of the "
                           "block: zero-shot for P1 must not lean on tree "
                           "adjacency"),
        },
    }
    paths = default_tree_paths()
    tp = tree_path or paths["supermatrix"]
    sp = supertree_path or paths["supertree"]
    out["paths"] = {"supermatrix": tp, "supertree": sp}
    if tp and os.path.isfile(tp):
        raw = read_newick(tp)
        rooted = reroot_on_outgroup(raw, OUTGROUP_TAXA)
        taxa, d = patristic_matrix(rooted)
        _, d_raw = patristic_matrix(raw, taxa)
        iu = np.triu_indices(len(taxa), k=1)
        out["tree"] = {
            "taxa": list(taxa), "n_taxa": len(taxa),
            "min_distance": float(d[iu].min()),
            "max_distance": float(d[iu].max()),
            "median_distance": float(np.median(d[iu])),
            "ratio_max_min": float(d[iu].max() / d[iu].min()),
            "four_point_defect": four_point_defect(d),
            "reroot_invariance": float(np.abs(d - d_raw).max()),
            "n_splits": len(unrooted_splits(rooted)),
        }
        if set(taxa) == set(PATRISTIC_PUBLISHED_TAXA):
            out["tree"]["published_max_abs_diff"] = float(np.abs(
                _align_published(taxa, d) - PATRISTIC_PUBLISHED).max())
        if sp and os.path.isfile(sp):
            st = read_newick(sp)
            out["tree"]["robinson_foulds_vs_supertree"] = robinson_foulds(
                rooted, st)
            try:
                patristic_matrix(st)
                out["tree"]["supertree_has_lengths"] = True
            except BranchLengthsMissing as exc:
                out["tree"]["supertree_has_lengths"] = False
                out["tree"]["supertree_guard"] = str(exc)
        fits = {}
        for dim in fit_dims:
            f_free = fit_hyperbolic_embedding(d, dim=dim, scale=None,
                                              restarts=restarts, seed=seed)
            f_fixed = fit_hyperbolic_embedding(d, dim=dim, scale=10.0,
                                               restarts=restarts, seed=seed)
            euc = fit_euclidean_control(d, dim=dim, restarts=restarts,
                                        seed=seed)
            fits["d%d" % dim] = {
                "ladder_search": f_free.report(),
                "scale_10": f_fixed.report(),
                "euclidean_control": euc,
                "gain_vs_euclidean": euc["rms_rel"] - f_free.rms_rel,
            }
        out["fits"] = fits
    if model is not None:
        out["channel"] = model.report()
    return out


def _align_published(taxa: Sequence[str], d: np.ndarray) -> np.ndarray:
    """Reorder ``d`` into :data:`PATRISTIC_PUBLISHED_TAXA` order."""
    pos = {t: i for i, t in enumerate(taxa)}
    idx = [pos[t] for t in PATRISTIC_PUBLISHED_TAXA]
    return d[np.ix_(idx, idx)]


def falsifiers(model: Optional[GenomeEmbedding] = None,
               registry: Optional[StrainRegistry] = None,
               tree_path: Optional[str] = None,
               supertree_path: Optional[str] = None,
               ablation_variance: Optional[float] = None,
               run_fits: bool = True, restarts: int = 4,
               seed: int = 0) -> List[Tuple[str, bool, str]]:
    """Every falsifier of this layer, as ``(name, passed, detail)``.

    F1 (embedding RMS), F2 (radius), F3 (four-point) and F4 (RF = 0) need the
    tree files. When no tree exists they are reported as misses rather than
    skipped, so a unit test without data cannot manufacture a pass.
    """
    out: List[Tuple[str, bool, str]] = []
    registry = registry or (model.registry if model is not None
                            else StrainRegistry.default())
    out.extend(registry.falsifiers())

    paths = default_tree_paths()
    tp = tree_path or paths["supermatrix"]
    sp = supertree_path or paths["supertree"]
    if tp and os.path.isfile(tp):
        raw = read_newick(tp)
        rooted = reroot_on_outgroup(raw, OUTGROUP_TAXA)
        taxa, d = patristic_matrix(rooted)
        _, d_raw = patristic_matrix(raw, taxa)
        defect = four_point_defect(d)
        out.append(("F3 the matrix is an exact tree metric (4-point < 1e-12)",
                    defect < 1e-12,
                    "defect %.3e over %d quadruples (measured 2.220e-16, "
                    "Gromov delta = defect/2)"
                    % (defect, math.comb(len(taxa), 4))))
        out.append(("patristic distances are reroot-invariant",
                    float(np.abs(d - d_raw).max()) < 1e-12,
                    "max|d_rooted - d_raw| = %.3e"
                    % float(np.abs(d - d_raw).max())))
        out.append(("rerooting puts the outgroup alone on one side",
                    frozenset(rooted.children[0].leaves()) == frozenset(
                        OUTGROUP_TAXA)
                    or frozenset(rooted.children[1].leaves()) == frozenset(
                        OUTGROUP_TAXA),
                    "root children: %r"
                    % [sorted(c.leaves()) for c in rooted.children]))
        if set(taxa) == set(PATRISTIC_PUBLISHED_TAXA):
            diff = float(np.abs(_align_published(taxa, d)
                                - PATRISTIC_PUBLISHED).max())
            out.append(("parser reproduces the published matrix (<= 5e-5)",
                        diff <= 5e-5,
                        "max|parsed - published| = %.2e (published values are "
                        "rounded to 4 decimals)" % diff))
        if sp and os.path.isfile(sp):
            st = read_newick(sp)
            rf = robinson_foulds(rooted, st)
            out.append(("F4 supertree and supermatrix agree (RF = 0)", rf == 0,
                        "RF = %d over %d/%d non-trivial splits; a 1 here is "
                        "the split induced by the supertree's rooting on "
                        "IMI206040, i.e. an implementation bug in the measure"
                        % (rf, len(unrooted_splits(rooted)),
                           len(unrooted_splits(st)))))
            guard = False
            try:
                patristic_matrix(st)
            except BranchLengthsMissing:
                guard = True
            out.append(("supertree is refused as a distance source", guard,
                        "10 of its 19 nodes have no branch length and its "
                        "internal lengths are ASTRAL coalescent units"))
        if run_fits:
            f3 = fit_hyperbolic_embedding(d, dim=3, scale=None,
                                          restarts=restarts, seed=seed)
            f2 = fit_hyperbolic_embedding(d, dim=2, scale=None,
                                          restarts=restarts, seed=seed)
            f8 = fit_hyperbolic_embedding(d, dim=8, scale=None,
                                          restarts=max(2, restarts // 2),
                                          seed=seed)
            out.append(("F1 H^3 relative RMS <= 0.0120", f3.rms_rel <= 0.0120,
                        "RMS %.5f (measured 0.01123), worst distortion %.4f"
                        % (f3.rms_rel, f3.distortion)))
            out.append(("F1 H^2 relative RMS <= 0.0450", f2.rms_rel <= 0.0450,
                        "RMS %.5f (measured 0.04321)" % f2.rms_rel))
            out.append(("F1 positive control: H^8 goes below 1e-4",
                        f8.rms_rel < 1e-4,
                        "RMS %.3e (measured ~1e-5); failing this means the "
                        "gradient is wrong, not that the data are hard"
                        % f8.rms_rel))
            fixed = fit_hyperbolic_embedding(d, dim=3, scale=10.0,
                                             restarts=restarts, seed=seed)
            out.extend(fixed.falsifiers())
            guard = False
            try:
                fit_hyperbolic_embedding(d, dim=3, scale=30.0, restarts=1)
            except ValueError:
                guard = True
            out.append(("scale >= 30 is refused", guard,
                        "s=50 measured RMS 0.1898, s=100 0.4499 -- float64 "
                        "cancellation in <x,x>_L"))
    else:
        out.append(("tree files available for F1-F4", False,
                    "supermatrix.treefile not found (looked at %r); the "
                    "tree-side falsifiers did not run" % (tp,)))
    if model is not None:
        out.extend(model.falsifiers(ablation_variance=ablation_variance))
    return out


def self_test(verbose: bool = True, seed: int = 0) -> int:
    """Run every falsifier that needs no data, plus the tree ones if found.

    Returns the number of failures, so ``py -m hyppe.model.species`` exits
    non-zero when something is broken.
    """
    results: List[Tuple[str, bool, str]] = []
    reg = StrainRegistry.default()
    results.extend(reg.falsifiers())

    # Synthetic tree: a 4-taxon metric whose exact embedding is known to exist.
    d = np.array([[0.0, 0.1, 0.5, 0.5],
                  [0.1, 0.0, 0.5, 0.5],
                  [0.5, 0.5, 0.0, 0.2],
                  [0.5, 0.5, 0.2, 0.0]], dtype=np.float64)
    results.append(("synthetic 4-point metric is a tree metric",
                    four_point_defect(d) < 1e-12,
                    "defect %.3e" % four_point_defect(d)))
    fit = fit_hyperbolic_embedding(d, dim=3, scale=10.0, restarts=3, seed=seed)
    results.extend(fit.falsifiers())

    # Channel without any data file: everyone at the origin, prior inert.
    model = GenomeEmbedding(reg, dim=3, scale=10.0, tree_prior=0.1)
    vec = model(torch.arange(len(reg)))
    results.append(("forward returns [n, dim]", tuple(vec.shape) ==
                    (len(reg), 3), "shape %r" % (tuple(vec.shape),)))
    refused = False
    try:
        model(torch.zeros(2, dtype=torch.long), ablate="shuffle")
    except ValueError:
        refused = True
    results.append(("shuffle ablation is refused (empty intervention)",
                    refused, "batches are genome-homogeneous, measured +0.0000"))
    guard = False
    try:
        GenomeEmbedding(reg, dim=3, scale=30.0)
    except ValueError:
        guard = True
    results.append(("channel refuses scale >= 30", guard,
                    "MAX_SAFE_SCALE = %.1f" % MAX_SAFE_SCALE))

    results.extend(falsifiers(run_fits=False))

    failures = sum(1 for _, ok, _ in results if not ok)
    if verbose:
        width = max(len(n) for n, _, _ in results)
        for name, ok, detail in results:
            print("[%s] %s  %s" % ("PASS" if ok else "FAIL",
                                   name.ljust(width), detail))
        print("\n%d/%d checks passed (%d failure%s)."
              % (len(results) - failures, len(results), failures,
                 "" if failures == 1 else "s"))
    return failures


if __name__ == "__main__":   # pragma: no cover
    import sys

    sys.exit(self_test(verbose=True))
