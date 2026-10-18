"""Everything the per-channel heads need, attached to a corpus after the fact.

`Corpus` is one row per (orthogroup, genome) carrying the mean over
conditions, while the conditional term is one value per (orthogroup, sample):
a row has one `mu` and up to 386 values of `cond`. Widening `Corpus` would
mean duplicating every sequence per condition -- the 697 525-row layout this
project rejected -- or a ragged column, so the bank keeps the decomposition in
its own shape and resolves one condition per row at batch time. Within an
epoch a promoter is then seen under several conditions and the text token
varies while the sequence does not, which is the contrast the `cond` head
needs. `draw_conditions` takes an explicit `rng`: the batch rng in training, a
freshly seeded constant one in evaluation, so the validation metric does not
depend on how many batches were drawn before it.

No expression matrix reaches the gene channel; `gene_features` refuses those
files by name and this module only forwards what it produced. Text embeddings
are optional and their absence is loud: `Corpus.from_paths` has never supplied
`text_emb`, so every run so far put the unknown token in the text slot, where
a `cond` head can only learn a constant. `report()["text_coverage"]` is 0.0 in
that case, and has to appear beside any claim about the experiment channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.decomposition import Decomposition
from ..data.gene_features import GeneFeatures

__all__ = ["TextEmbeddingCache", "ChannelBank", "attach_channels"]


# Text cache
@dataclass
class TextEmbeddingCache:
    """Frozen-LM vectors keyed by an id (a sample id, or an orthogroup id)."""

    keys: List[str]
    vectors: np.ndarray                  # [n, d] float32
    fingerprint: str = ""
    source: str = ""

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def index(self) -> Dict[str, int]:
        return {k: i for i, k in enumerate(self.keys)}

    def rows_for(self, keys: Sequence[str]) -> np.ndarray:
        ix = self.index()
        return np.array([ix.get(str(k), -1) for k in keys], dtype=np.int64)

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """Vectors for `rows`; a -1 row returns zeros and must be masked.

        The zeros never reach a projection -- the caller passes an
        availability mask and the encoder substitutes its unknown token. A
        zero into a biased projection is a learned constant, which reports a
        context share while carrying no information.
        """
        r = np.asarray(rows, dtype=np.int64)
        out = np.zeros((r.size, self.dim), dtype=np.float32)
        ok = r >= 0
        if ok.any():
            out[ok] = self.vectors[r[ok]]
        return out

    @classmethod
    def load(cls, path: str) -> "TextEmbeddingCache":
        z = np.load(path, allow_pickle=True)
        need = ("keys", "vectors")
        missing = [k for k in need if k not in z]
        if missing:
            raise ValueError(
                "%r is not a text embedding cache: missing %r. Expected an "
                "npz with `keys` (ids) and `vectors` [n, d], optionally "
                "`fingerprint` (the ConditionTextSpec hash the strings were "
                "built under)." % (path, missing))
        fp = str(z["fingerprint"]) if "fingerprint" in z else ""
        return cls(keys=[str(k) for k in z["keys"]],
                   vectors=np.asarray(z["vectors"], dtype=np.float32),
                   fingerprint=fp, source=path)

    def save(self, path: str) -> str:
        np.savez_compressed(path,
                            keys=np.array(self.keys, dtype=object),
                            vectors=self.vectors,
                            fingerprint=np.array(self.fingerprint,
                                                 dtype=object))
        return path


# The bank
@dataclass
class ChannelBank:
    """Per-row targets and features for the gene / species / text channels."""

    #: [n_rows] target of the gene head, and its mask.
    gene_target: np.ndarray
    gene_valid: np.ndarray
    #: [n_rows] target of the strain head (genome main effect + interaction).
    strain_target: np.ndarray
    strain_valid: np.ndarray

    #: [n_rows, n_numeric] phyletic block, and whether the gene was found.
    gene_numeric: np.ndarray
    gene_ok: np.ndarray
    #: [n_rows] row into the gene text cache, -1 where absent.
    gene_text_row: np.ndarray

    #: Per row, the usable columns of `decomposition.cond`. A list of arrays
    #: rather than a matrix: 28.6 % of cells are empty, and padding them with
    #: a sentinel is how a sentinel gets trained on.
    cond_columns: List[np.ndarray]
    cond_values: List[np.ndarray]
    #: [n_samples] row into the sample text cache, -1 where absent.
    sample_text_row: np.ndarray

    gene_text: Optional[TextEmbeddingCache] = None
    sample_text: Optional[TextEmbeddingCache] = None
    notes: List[str] = field(default_factory=list)

    # Draw
    def draw_conditions(self, rows: np.ndarray, rng: np.random.Generator
                        ) -> Dict[str, np.ndarray]:
        """One condition per row: its `cond` value, mask and text vector.

        A row with no usable condition returns ``valid = False`` and a zero
        text vector with ``text_ok = 0``; it must reach the unknown token.
        """
        r = np.asarray(rows, dtype=np.int64)
        n = int(r.size)
        val = np.zeros(n, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        srow = np.full(n, -1, dtype=np.int64)
        for i, row in enumerate(r):
            cols = self.cond_columns[int(row)]
            if cols.size == 0:
                continue
            j = int(rng.integers(cols.size))
            val[i] = float(self.cond_values[int(row)][j])
            ok[i] = True
            srow[i] = int(self.sample_text_row[int(cols[j])])
        out: Dict[str, np.ndarray] = {
            "cond_target": val,
            "cond_valid": ok,
            "text_ok": (srow >= 0).astype(np.float32),
        }
        if self.sample_text is not None:
            out["text_emb"] = self.sample_text.gather(srow)
        return out

    def gene_inputs(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        r = np.asarray(rows, dtype=np.int64)
        out: Dict[str, np.ndarray] = {
            "gene_numeric": self.gene_numeric[r],
            "gene_ok": self.gene_ok[r].astype(np.float32),
        }
        if self.gene_text is not None:
            out["gene_text"] = self.gene_text.gather(self.gene_text_row[r])
        return out

    def targets(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        r = np.asarray(rows, dtype=np.int64)
        return {
            "gene_target": self.gene_target[r],
            "gene_valid": self.gene_valid[r],
            "strain_target": self.strain_target[r],
            "strain_valid": self.strain_valid[r],
        }

    # Report
    def report(self) -> Dict[str, object]:
        n = int(self.gene_valid.size)
        n_cond = sum(1 for c in self.cond_columns if c.size)
        per = np.array([c.size for c in self.cond_columns], dtype=np.float64)
        out: Dict[str, object] = {
            "n_rows": n,
            "gene_target_coverage": float(self.gene_valid.mean()) if n else 0.0,
            "strain_target_coverage": (float(self.strain_valid.mean())
                                       if n else 0.0),
            "cond_row_coverage": (n_cond / float(n)) if n else 0.0,
            "conditions_per_row_median": (float(np.median(per))
                                          if per.size else 0.0),
            "conditions_per_row_max": float(per.max()) if per.size else 0.0,
            "gene_feature_coverage": (float(self.gene_ok.mean())
                                      if self.gene_ok.size else 0.0),
            "gene_text_coverage": (float((self.gene_text_row >= 0).mean())
                                   if self.gene_text_row.size else 0.0),
            "text_coverage": (float((self.sample_text_row >= 0).mean())
                              if self.sample_text_row.size else 0.0),
            "notes": list(self.notes),
        }
        if self.sample_text is None:
            out["WARNING_text"] = (
                "no sentence-LM cache: the text token is the UNKNOWN token in "
                "every row, so the `cond` head is regressing a conditional "
                "deviation from a CONSTANT context and can only learn its "
                "mean. No claim about the experiment channel may be made from "
                "such a run. Build one with "
                "src/scripts/precompute_condition_text.py.")
        return out

    def format(self) -> str:
        r = self.report()
        lines = [
            "channels: gene target %.4f | strain target %.4f | cond rows "
            "%.4f (median %d conditions/row, max %d)"
            % (r["gene_target_coverage"], r["strain_target_coverage"],
               r["cond_row_coverage"], int(r["conditions_per_row_median"]),
               int(r["conditions_per_row_max"])),
            "  gene features %.4f covered | gene text %.4f | sample text %.4f"
            % (r["gene_feature_coverage"], r["gene_text_coverage"],
               r["text_coverage"]),
        ]
        lines += ["  NOTE " + n for n in self.notes]
        if "WARNING_text" in r:
            lines.append("  WARNING " + str(r["WARNING_text"]))
        return "\n".join(lines)


# Construction
def attach_channels(og: Sequence[str], genome: Sequence[str],
                    dec: Decomposition,
                    gene_features: Optional[GeneFeatures] = None,
                    sample_text: Optional[TextEmbeddingCache] = None,
                    gene_text: Optional[TextEmbeddingCache] = None,
                    max_conditions_per_row: int = 0) -> ChannelBank:
    """Align a decomposition and a feature table to the corpus rows.

    `og` and `genome` are the corpus's own per-row columns and the alignment
    is by key, not position: the decomposition comes from the ComBat matrix
    and the corpus from the sequence table, sorted differently. A positional
    join here would be silent and total.
    """
    n = len(og)
    if len(genome) != n:
        raise ValueError("og and genome columns differ in length (%d vs %d)"
                         % (n, len(genome)))
    if gene_text is not None and gene_features is None:
        # `gene_text` is read only on the branch that has a feature table, so
        # this combination would leave every `gene_text_row` at -1 and hand
        # `gene_inputs` an all-zero text vector with no availability flag --
        # a zero into a biased projection, i.e. a learned constant.
        raise ValueError(
            "gene_text was supplied without gene_features. The text rows are "
            "resolved alongside the feature table, so this pair would give "
            "every row the zero vector with no flag saying it is absent, and "
            "the gene head would report a context share while carrying no "
            "information. Pass gene_features too, or drop gene_text.")
    oi = dec.og_index()
    gi = dec.genome_index()
    notes: List[str] = []

    gene_t = np.zeros(n, dtype=np.float32)
    gene_v = np.zeros(n, dtype=bool)
    strain_t = np.zeros(n, dtype=np.float32)
    strain_v = np.zeros(n, dtype=bool)
    cond_cols: List[np.ndarray] = []
    cond_vals: List[np.ndarray] = []

    # Columns of `cond` belonging to each genome, computed once.
    by_genome: Dict[int, np.ndarray] = {
        j: np.nonzero(dec.sample_genome == j)[0] for j in range(len(dec.genomes))
    }
    miss_og = miss_gen = 0
    for i in range(n):
        a = oi.get(str(og[i]))
        b = gi.get(str(genome[i]))
        if a is None:
            miss_og += 1
        if b is None:
            miss_gen += 1
        if a is None or b is None:
            cond_cols.append(np.zeros(0, dtype=np.int64))
            cond_vals.append(np.zeros(0, dtype=np.float32))
            continue
        if bool(dec.gene_ok[a]):
            gene_t[i] = float(dec.gene[a])
            gene_v[i] = True
        if bool(dec.inter_ok[a, b]) and bool(dec.genome_ok[b]):
            # The strain head owns the genome main effect and the interaction
            # together: split, one head would get a target constant across a
            # genome-homogeneous batch, which produces no usable gradient.
            strain_t[i] = float(dec.genome_eff[b]) + float(dec.inter[a, b])
            strain_v[i] = True
        cols = by_genome.get(b, np.zeros(0, dtype=np.int64))
        usable = cols[dec.cond_ok[a, cols]] if cols.size else cols
        if max_conditions_per_row and usable.size > int(max_conditions_per_row):
            # Thinned by a hash of the column index, not truncated: the
            # columns follow the metadata file, so a prefix is one study.
            keep = np.argsort((usable * 2654435761) % 1000003)[
                :int(max_conditions_per_row)]
            usable = np.sort(usable[keep])
        cond_cols.append(usable.astype(np.int64))
        cond_vals.append(dec.cond[a, usable].astype(np.float32)
                         if usable.size else np.zeros(0, dtype=np.float32))
    if miss_og:
        notes.append("%d of %d corpus rows have no row in the decomposition "
                     "(orthogroup absent from the ComBat matrix)" % (miss_og, n))
    if miss_gen:
        notes.append("%d of %d corpus rows name a genome the decomposition "
                     "does not have" % (miss_gen, n))

    # Gene features
    if gene_features is not None:
        num, _ident, ok = gene_features.vectors_for([str(o) for o in og])
        g_text_row = (gene_text.rows_for([str(o) for o in og])
                      if gene_text is not None
                      else np.full(n, -1, dtype=np.int64))
        notes.append("gene features: %s" % (gene_features.report()["spec"]
                                            ["fingerprint"][:16],))
    else:
        num = np.zeros((n, 0), dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        g_text_row = np.full(n, -1, dtype=np.int64)
        notes.append("no gene feature table: the gene channel reads its "
                     "unknown token for every row")

    s_text_row = (sample_text.rows_for(dec.sample_ids)
                  if sample_text is not None
                  else np.full(len(dec.sample_ids), -1, dtype=np.int64))

    return ChannelBank(
        gene_target=gene_t, gene_valid=gene_v,
        strain_target=strain_t, strain_valid=strain_v,
        gene_numeric=num, gene_ok=ok, gene_text_row=g_text_row,
        cond_columns=cond_cols, cond_values=cond_vals,
        sample_text_row=s_text_row,
        gene_text=gene_text, sample_text=sample_text, notes=notes)
