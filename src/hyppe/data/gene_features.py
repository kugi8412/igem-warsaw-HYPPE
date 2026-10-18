"""Gene channel: what the promoter is IN FRONT OF, without reading its label.

`ConditioningConfig.gene_context` is named for the gene but its source is
`dna_lm` over the left and right flanks, the neighbourhood. Nothing under
`model/` or `heads/` ever saw an orthogroup id, an ortholog or any property of
the coding sequence the promoter drives, so the design queries "make this
strong for pks1" and "keep it between pks1 and tef1" had no channel to be
conditioned on.

The leakage this channel must not commit: the obvious gene feature is how
strongly the gene is expressed on average, which is exactly the gene head's
target, so feeding it in copies the label into the input. Every feature here
comes from orthology and sequence only, never from `combat`, `counts`, `tpm`
or `rank`. :func:`forbidden_sources` names the files, and
:meth:`GeneFeatureSpec.fingerprint` goes into the checkpoint, so a later edit
adding an expression column invalidates every cached vector instead of
silently improving the metric.

Identity embedding is off by default because the split is orthogroup disjoint:
every orthogroup in validation, test or spy -- pks1 among them -- has an
embedding row that received exactly zero gradient. Such a table raises the
training metric, cannot raise the validation metric, and hides that the
channel does nothing transferable. The default is content only: the phyletic
profile (which strains carry the orthogroup), the copy structure (genes per
strain, single-copy flag, n_strains) and an optional free-text description
through the same frozen LM as the condition channel, all three of which exist
for a gene the model has never seen. `identity_dim > 0` switches the table on
for the fine-tuning regime, and `report()` then carries `identity_enabled`, so
the number is never quoted as zero-shot.

Conservation is not a proxy for strength, and that is the point: the phyletic
profile says how widely a gene is kept, not how loudly it is read, which is
what lets it transfer to a gene with no measurements.
:meth:`GeneFeatures.audit_against` reports its correlation with the gene
head's target, so a suspiciously high value is visible rather than celebrated.

A gene absent from the orthology table resolves to `availability = 0` and the
model reaches `unknown_token_gene`, not a zero vector: the projection has a
bias, so zeros are a learned constant and the channel would report a share
while carrying nothing.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GENE_FEATURE_VERSION",
    "FORBIDDEN_GENE_SOURCES",
    "GeneFeatureSpec",
    "GeneFeatures",
    "OrthogroupRecord",
    "read_orthology_table",
    "build_gene_features",
    "gene_description",
    "forbidden_sources",
]

#: Bumped whenever the feature construction changes. A checkpoint carrying a
#: different version was conditioned on different numbers while every weight
#: still matches.
GENE_FEATURE_VERSION = "hyppe-gene-1"

#: Files whose columns are the LABEL. Reading any of them into this channel
#: turns conditioning into copying, and the resulting metric is unfalsifiable.
FORBIDDEN_GENE_SOURCES: Tuple[str, ...] = (
    "combat.tsv.gz", "counts.tsv.gz", "tpm.tsv.gz", "rank.tsv.gz",
    "xpn.tsv.gz", "anchor_tef1.tsv.gz", "anchor_3genes.tsv.gz")

_ACCESSION_SPLIT = re.compile(r"\s*,\s*")


def forbidden_sources() -> Tuple[str, ...]:
    """The expression matrices, named so a reviewer can grep for them."""
    return FORBIDDEN_GENE_SOURCES


# The spec
@dataclass(frozen=True)
class GeneFeatureSpec:
    """How an orthogroup becomes a feature vector. A checkpoint artefact."""

    #: Strain columns of the orthology table, IN ORDER. Append-only: the index
    #: of a strain is a column of every cached vector.
    strains: Tuple[str, ...] = ()
    #: Width of the learned identity table. 0 is content only: the default,
    #: and the only setting under which a number may be called zero-shot.
    identity_dim: int = 0
    #: Buckets the orthogroup id is hashed into when `identity_dim > 0`.
    identity_buckets: int = 4096
    #: Include the free-text description channel.
    use_text: bool = True
    #: Cap on log1p(genes per strain), so one exploded family cannot dominate
    #: the scale of a channel shared with binary presence flags.
    copies_cap: float = 4.0

    def fingerprint(self) -> str:
        payload = json.dumps({
            "version": GENE_FEATURE_VERSION,
            "strains": list(self.strains),
            "identity_dim": int(self.identity_dim),
            "identity_buckets": int(self.identity_buckets),
            "use_text": bool(self.use_text),
            "copies_cap": float(self.copies_cap),
            "forbidden": list(FORBIDDEN_GENE_SOURCES),
        }, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def n_numeric(self) -> int:
        """presence per strain + copies per strain + 4 summary scalars."""
        return 2 * len(self.strains) + 4

    def column_names(self) -> List[str]:
        out = ["present:%s" % s for s in self.strains]
        out += ["copies:%s" % s for s in self.strains]
        out += ["n_strains_frac", "log_n_genes", "single_copy",
                "mean_copies"]
        return out

    def to_dict(self) -> Dict[str, object]:
        d = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in self.__dict__.items()}
        d["version"] = GENE_FEATURE_VERSION
        d["fingerprint"] = self.fingerprint()
        d["n_numeric"] = self.n_numeric()
        return d


# Reading the orthology table
@dataclass
class OrthogroupRecord:
    og_id: str
    copies: Dict[str, int]
    n_strains: int
    n_genes: int
    single_copy_all: bool

    def accessions(self, strain: str) -> int:
        return int(self.copies.get(strain, 0))


def read_orthology_table(path: str, strains: Sequence[str] = ()
                         ) -> Tuple[List[OrthogroupRecord], Tuple[str, ...]]:
    """Parse `og_table_combined.tsv` into per-orthogroup copy counts.

    The strain columns hold comma-separated protein accessions, and an empty
    cell means the strain does not carry the orthogroup. `n_strains`,
    `n_genes` and `single_copy_all` are read from the table when present and
    recomputed otherwise, never half of each: a mix of stated and derived
    counts is how two rows of one table come to disagree.
    """
    base = os.path.basename(path)
    if base in FORBIDDEN_GENE_SOURCES:
        raise ValueError(
            "%r is an expression matrix and is the gene head's TARGET. "
            "Feeding it to the gene channel copies the label into the input; "
            "the forbidden list is %r." % (base, FORBIDDEN_GENE_SOURCES))
    meta_cols = {"orthogroup", "n_strains", "n_genes", "single_copy_all"}
    out: List[OrthogroupRecord] = []
    with io.open(path, encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        if header[0] != "orthogroup":
            raise ValueError("expected first column 'orthogroup', got %r"
                             % (header[0],))
        cols = [c for c in header if c not in meta_cols]
        use = tuple(strains) if strains else tuple(cols)
        unknown = [s for s in use if s not in header]
        if unknown:
            raise ValueError(
                "strain column(s) %r absent from %s. The strain order is a "
                "checkpoint contract, so a missing column must stop the build "
                "rather than shift every later column by one."
                % (unknown, path))
        ix = {c: header.index(c) for c in header}
        for row in rdr:
            if not row or not row[0]:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            copies: Dict[str, int] = {}
            for s in use:
                cell = row[ix[s]].strip()
                copies[s] = (0 if not cell
                             else len([a for a in _ACCESSION_SPLIT.split(cell)
                                       if a]))
            n_str = sum(1 for v in copies.values() if v > 0)
            n_gen = sum(copies.values())
            if "n_strains" in ix and row[ix["n_strains"]].strip():
                n_str = int(float(row[ix["n_strains"]]))
            if "n_genes" in ix and row[ix["n_genes"]].strip():
                n_gen = int(float(row[ix["n_genes"]]))
            sc = (row[ix["single_copy_all"]].strip().lower() == "true"
                  if "single_copy_all" in ix
                  else all(v == 1 for v in copies.values() if v > 0))
            out.append(OrthogroupRecord(row[0], copies, n_str, n_gen, sc))
    return out, use


# The feature table
@dataclass
class GeneFeatures:
    """Numeric features, optional text, and availability -- per orthogroup."""

    spec: GeneFeatureSpec
    og_ids: List[str]
    numeric: np.ndarray                  # [n_og, spec.n_numeric()] float32
    identity: np.ndarray                 # [n_og] int64 hash bucket, or -1
    text: List[str]                      # "" where no description exists
    available: np.ndarray                # [n_og] bool
    notes: List[str] = field(default_factory=list)

    def index(self) -> Dict[str, int]:
        return {o: i for i, o in enumerate(self.og_ids)}

    # Lookup
    def rows_for(self, og_ids: Sequence[str]) -> np.ndarray:
        """Row per requested orthogroup, -1 where the gene is unknown."""
        ix = self.index()
        return np.array([ix.get(str(o), -1) for o in og_ids], dtype=np.int64)

    def vectors_for(self, og_ids: Sequence[str]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(numeric [n, d], identity [n], available [n]).

        Unknown genes get a zero numeric row with `available = False`, and the
        zeros never reach a projection: `GeneEncoder` masks the operand and
        substitutes the learned unknown token.
        """
        rows = self.rows_for(og_ids)
        ok = rows >= 0
        num = np.zeros((rows.size, self.numeric.shape[1]), dtype=np.float32)
        ident = np.full(rows.size, -1, dtype=np.int64)
        if ok.any():
            num[ok] = self.numeric[rows[ok]]
            ident[ok] = self.identity[rows[ok]]
        return num, ident, ok

    def texts_for(self, og_ids: Sequence[str]) -> List[str]:
        ix = self.index()
        return [self.text[ix[str(o)]] if str(o) in ix else ""
                for o in og_ids]

    # Falsifiers
    def audit_against(self, og_ids: Sequence[str], gene_effect: np.ndarray,
                      valid: Optional[np.ndarray] = None
                      ) -> Dict[str, float]:
        """Correlation of each feature with the gene head's TARGET.

        Read in both directions: a near-zero maximum means the channel cannot
        help and the gene head will be driven by the sequence alone, while a
        high one means a feature is close to being the label, which for a
        purely orthology-derived feature needs explaining before the number is
        used. Returns the largest absolute correlation and the column carrying
        it, plus the multiple R^2 of a least-squares fit of the target on the
        whole numeric block, which is this channel's ceiling.
        """
        rows = self.rows_for(og_ids)
        y = np.asarray(gene_effect, dtype=np.float64).ravel()
        ok = rows >= 0
        if valid is not None:
            ok = ok & np.asarray(valid, dtype=bool).ravel()
        ok = ok & np.isfinite(y)
        if int(ok.sum()) < 8:
            return {"max_abs_corr": float("nan"), "r2_multiple": float("nan"),
                    "n": float(ok.sum())}
        X = self.numeric[rows[ok]].astype(np.float64)
        t = y[ok]
        sd = X.std(axis=0)
        live = sd > 1e-9
        corr = np.zeros(X.shape[1], dtype=np.float64)
        if live.any() and t.std() > 1e-9:
            Z = (X[:, live] - X[:, live].mean(0)) / sd[live]
            corr[live] = (Z * ((t - t.mean()) / t.std())[:, None]).mean(0)
        names = self.spec.column_names()
        j = int(np.argmax(np.abs(corr))) if corr.size else 0
        A = np.c_[X[:, live], np.ones(X.shape[0])]
        beta, *_ = np.linalg.lstsq(A, t, rcond=None)
        resid = t - A @ beta
        r2 = 1.0 - float(resid.var()) / float(t.var() or 1.0)
        return {
            "max_abs_corr": float(np.abs(corr[j])) if corr.size else 0.0,
            "max_abs_corr_column": names[j] if names else "",
            "r2_multiple": float(r2),
            "n": float(ok.sum()),
            "n_live_columns": float(int(live.sum())),
        }

    def report(self) -> Dict[str, object]:
        cover = float(self.available.mean()) if self.available.size else 0.0
        return {
            "n_orthogroups": len(self.og_ids),
            "coverage": cover,
            "identity_enabled": bool(self.spec.identity_dim > 0),
            "identity_buckets": int(self.spec.identity_buckets),
            "text_coverage": (float(np.mean([bool(t) for t in self.text]))
                              if self.text else 0.0),
            "n_numeric": int(self.numeric.shape[1]) if self.numeric.size else 0,
            "spec": self.spec.to_dict(),
            "forbidden_sources": list(FORBIDDEN_GENE_SOURCES),
            "notes": list(self.notes),
        }


def _bucket(og_id: str, buckets: int) -> int:
    h = hashlib.blake2b(og_id.encode("utf-8"), digest_size=8).digest()
    return int(int.from_bytes(h, "big") % max(1, int(buckets)))


def gene_description(rec: OrthogroupRecord, strains: Sequence[str],
                     annotation: str = "") -> str:
    """One English sentence per orthogroup, for the frozen LM.

    Through the same encoder as the condition text: describing a new object in
    words costs zero parameters, while a new index costs a row of a trained
    table and cannot be added at inference. A gene the corpus has never seen
    can therefore be specified at design time in words.

    `annotation` is whatever functional text is available. It goes first,
    because it is the only part distinguishing two orthogroups with the same
    phyletic profile and a frozen LM truncates from the end.
    """
    parts: List[str] = []
    if annotation:
        parts.append(annotation.strip())
    present = [s for s in strains if rec.copies.get(s, 0) > 0]
    if present:
        parts.append("conserved in %d of %d strains"
                     % (len(present), len(strains)))
    parts.append("single copy" if rec.single_copy_all
                 else "%d genes in the family" % (rec.n_genes,))
    return "gene family, " + ", ".join(parts)


def build_gene_features(orthology_path: str,
                        spec: Optional[GeneFeatureSpec] = None,
                        annotations: Optional[Mapping[str, str]] = None,
                        strains: Sequence[str] = ()) -> GeneFeatures:
    """Read the orthology table into the channel's feature block.

    `annotations` maps orthogroup id to free-text function, and is absent for
    this corpus today. The parameter exists so that conditioning on a gene
    described in words needs no code change, and so that a channel whose text
    is empty for every row says so in `report()` instead of looking like a
    text channel.
    """
    recs, cols = read_orthology_table(orthology_path, strains)
    sp = spec or GeneFeatureSpec(strains=tuple(cols))
    if not sp.strains:
        sp = GeneFeatureSpec(strains=tuple(cols),
                             identity_dim=sp.identity_dim,
                             identity_buckets=sp.identity_buckets,
                             use_text=sp.use_text, copies_cap=sp.copies_cap)
    missing = [s for s in sp.strains if s not in cols]
    if missing:
        raise ValueError("spec names strains %r that the table lacks"
                         % (missing,))
    n_str = len(sp.strains)
    ann = dict(annotations or {})
    og_ids: List[str] = []
    num = np.zeros((len(recs), sp.n_numeric()), dtype=np.float32)
    ident = np.zeros(len(recs), dtype=np.int64)
    text: List[str] = []
    for i, r in enumerate(recs):
        og_ids.append(r.og_id)
        c = np.array([r.copies.get(s, 0) for s in sp.strains],
                     dtype=np.float64)
        num[i, :n_str] = (c > 0).astype(np.float32)
        num[i, n_str:2 * n_str] = np.minimum(
            np.log1p(c), float(sp.copies_cap)).astype(np.float32)
        live = c[c > 0]
        num[i, 2 * n_str + 0] = float(r.n_strains) / max(1, n_str)
        num[i, 2 * n_str + 1] = float(min(np.log1p(r.n_genes),
                                          2.0 * sp.copies_cap))
        num[i, 2 * n_str + 2] = 1.0 if r.single_copy_all else 0.0
        num[i, 2 * n_str + 3] = float(live.mean()) if live.size else 0.0
        ident[i] = (_bucket(r.og_id, sp.identity_buckets)
                    if sp.identity_dim > 0 else -1)
        text.append(gene_description(r, sp.strains, ann.get(r.og_id, ""))
                    if sp.use_text else "")
    notes = ["orthology source: %s" % (os.path.basename(orthology_path),),
             "NO expression matrix was read; forbidden: %s"
             % (", ".join(FORBIDDEN_GENE_SOURCES),)]
    if sp.identity_dim > 0:
        notes.append(
            "identity table ENABLED (%d buckets): the split is orthogroup "
            "disjoint, so every val/test/spy orthogroup -- pks1 included -- "
            "has an untrained row. Numbers from this build are NOT zero-shot."
            % (sp.identity_buckets,))
    if not ann:
        notes.append(
            "no functional annotations supplied: the text column is the "
            "phyletic description only, which is a function of the numeric "
            "block and adds no independent information in THIS build")
    return GeneFeatures(spec=sp, og_ids=og_ids, numeric=num, identity=ident,
                        text=text,
                        available=np.ones(len(recs), dtype=bool), notes=notes)
