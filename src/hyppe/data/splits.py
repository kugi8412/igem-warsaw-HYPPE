"""Train / val / test / Spy assignment, and the audit that makes it credible.

The split has to achieve two things that pull against each other: 80/10/10
evenly across all 20 strains, counted both in promoters and in labelled
promoters -- 167 716 labels against 238 831 promoters, so the two are not
proportional and the label count is the one that trains the heads -- and no
homology across the boundary, so that a validation score measures
generalisation rather than recall.

The unit is a leakage block, not an orthogroup. Splitting by orthogroup
leaves 31.87 % of homologous promoter pairs sharing >= 169 bp of exact
sequence on opposite sides of the boundary, because orthology assignment and
sequence identity are different relations and the second is the one that
leaks. A block is a connected component of the graph whose vertices are
orthogroups and whose edges join two sharing >= 10 canonical 25-mers: 23 832
blocks from 26 327 orthogroups, cross-boundary leakage 0.0000, at a cost of
0.0026 in evenness.

Canonical k-mers belong to the audit and never to the model. The model must
not canonicalise, a promoter being directional and its reverse complement a
different object; the audit must, because 90.7 % of the sequence shared
between orthogroups sits in the opposite orientation, divergent gene pairs
sharing one intergenic region read from both ends. A forward-only scan
under-reported leakage tenfold.

Spy is the whole block, not the one promoter. pks1 is the design target and
its promoter is in the training corpus byte for byte; holding out only
P1_G009919 leaves six byte-identical copies in training, so a "zero-shot"
number would be a memory read. Spy is therefore the whole leakage block of
OG0000996: 81 orthogroups, 896 promoters, 461 labels. An exhaustive scan
confirms sufficiency -- the 13 promoters sharing an exact 25-mer with pks1 are
all inside that block.

`viride` contributes 10 255 promoters with no label at all, the strain being
in the orthology's 20 genomes but not the metadata's 19. It is full-value
material for reconstruction and the discriminator, and must be absent from
every PCC/SCC table or the denominator lies.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

SPLIT_NAMES = ("train", "val", "test", "spy")
#: Orthogroup of pks1 (XP_013949124 -> P1_G009919 -> TrAtP1_009995).
PKS1_ORTHOGROUP = "OG0000996"
#: Column of SPLIT_FINAL.tsv to use. `A_og` is kept as a control that still
#: leaks; `C_block5` is the stricter variant.
DEFAULT_COLUMN = "B_block10"
#: Strain present in the orthology but absent from the expression metadata.
UNLABELLED_GENOMES = ("viride",)


@dataclass
class SplitTable:
    """orthogroup -> split, plus the provenance needed to report it."""

    assignment: Dict[str, str]
    column: str = DEFAULT_COLUMN
    source: str = ""
    columns_available: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.assignment)

    def of(self, orthogroup: str, default: Optional[str] = None
           ) -> Optional[str]:
        return self.assignment.get(orthogroup, default)

    def orthogroups(self, split: str) -> List[str]:
        return sorted(k for k, v in self.assignment.items() if v == split)

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in SPLIT_NAMES}
        for v in self.assignment.values():
            out[v] = out.get(v, 0) + 1
        return out

    def report(self) -> Dict[str, object]:
        c = self.counts()
        total = max(sum(c.values()), 1)
        return {
            "column": self.column,
            "source": os.path.basename(self.source),
            "columns_available": list(self.columns_available),
            "n_orthogroups": len(self.assignment),
            "counts": c,
            "fractions": {k: round(v / total, 4) for k, v in c.items()},
            "spy_is_block_not_single_orthogroup": c.get("spy", 0) > 1,
        }


def load_splits(path: str, column: str = DEFAULT_COLUMN) -> SplitTable:
    """Read SPLIT_FINAL.tsv.

    The file is the artefact of a deterministic procedure -- the greedy order
    is size-descending with an index tie-break -- so the same inputs reproduce
    it. It is read rather than recomputed because recomputing needs the full
    203 MB promoter dump and a FracMinHash pass, and a split that silently
    differed between runs would invalidate every comparison across them.
    """
    rows = list(io.open(path, encoding="utf-8"))
    if not rows:
        raise ValueError("empty split file: %s" % path)
    header = rows[0].rstrip("\n").split("\t")
    if header[0] != "og":
        raise ValueError("first column of %s must be 'og', found %r"
                         % (path, header[0]))
    if column not in header:
        raise ValueError(
            "column %r not in %s; available: %s"
            % (column, os.path.basename(path), ", ".join(header[1:])))
    col = header.index(column)
    assignment: Dict[str, str] = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        value = f[col]
        if value not in SPLIT_NAMES:
            raise ValueError("unknown split %r for %s" % (value, f[0]))
        assignment[f[0]] = value
    return SplitTable(assignment=assignment, column=column, source=path,
                      columns_available=tuple(header[1:]))


# Audits
def canonical_kmers(sequence: str, k: int = 25) -> Set[str]:
    """Canonical k-mers -- min(window, its reverse complement).

    For auditing only; the model never canonicalises a strand. Leakage is a
    property of the DNA rather than of the model's reading of it.
    """
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    out: Set[str] = set()
    s = sequence.upper()
    for i in range(0, max(len(s) - k + 1, 0)):
        w = s[i:i + k]
        if any(ch not in comp for ch in w):
            continue
        rc = "".join(comp[ch] for ch in reversed(w))
        out.add(min(w, rc))
    return out


def leakage_report(sequences: Sequence[Tuple[str, str, str]],
                   splits: SplitTable, k: int = 25,
                   min_shared: int = 10) -> Dict[str, object]:
    """Cross-boundary homology, measured rather than assumed.

    `sequences` is [(promoter_id, orthogroup, sequence)]. Returns the number
    of promoter pairs sharing at least `min_shared` canonical k-mers while
    sitting in different splits -- the quantity that must be 0 for a
    validation score to mean anything -- with the by-orthogroup control
    alongside, since that rule looks fine and leaks 31.87 %.
    """
    index: Dict[str, List[int]] = {}
    items = list(sequences)
    for i, (_pid, _og, seq) in enumerate(items):
        for km in canonical_kmers(seq, k):
            index.setdefault(km, []).append(i)

    shared: Dict[Tuple[int, int], int] = {}
    for km, hits in index.items():
        # Groups above 40 are repetitive elements rather than homology (71 of
        # them here); counting them would report the genome's repeat content
        # as leakage.
        if len(hits) < 2 or len(hits) > 40:
            continue
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                i, j = hits[a], hits[b]
                if items[i][1] == items[j][1]:
                    continue
                key = (i, j) if i < j else (j, i)
                shared[key] = shared.get(key, 0) + 1

    crossing = 0
    homologous = 0
    for (i, j), n in shared.items():
        if n < min_shared:
            continue
        homologous += 1
        si = splits.of(items[i][1])
        sj = splits.of(items[j][1])
        if si is not None and sj is not None and si != sj:
            crossing += 1
    return {
        "n_promoters": len(items),
        "n_homologous_pairs": homologous,
        "n_crossing_boundary": crossing,
        "fraction_crossing": (crossing / homologous) if homologous else 0.0,
        "k": k, "min_shared_kmers": min_shared,
        "approx_shared_bp": k + min_shared - 1,
    }


def evenness_report(promoters: Iterable[Tuple[str, str]], splits: SplitTable,
                    labelled: Optional[Set[Tuple[str, str]]] = None,
                    ratio: Sequence[float] = (0.80, 0.10, 0.10)
                    ) -> Dict[str, object]:
    """Per-strain deviation from 80/10/10, in promoters AND in labels.

    Both are reported because they are not proportional and balancing one
    costs the other: balancing promoters alone gives max|dev| 0.00017 for
    promoters and 0.01268 for labels, while balancing both gives 0.00335 and
    0.00576. Labels train the heads, so the second is allowed to win.

    `promoters` is [(orthogroup, genome)]; `labelled` is the set of
    (orthogroup, genome) pairs carrying a usable combat value.
    """
    tot: Dict[str, Dict[str, float]] = {}
    lab: Dict[str, Dict[str, float]] = {}
    for og, genome in promoters:
        sp = splits.of(og)
        if sp is None or sp == "spy":
            continue
        tot.setdefault(genome, {}).setdefault(sp, 0.0)
        tot[genome][sp] += 1.0
        if labelled is not None and (og, genome) in labelled:
            lab.setdefault(genome, {}).setdefault(sp, 0.0)
            lab[genome][sp] += 1.0

    def deviations(table: Dict[str, Dict[str, float]]
                   ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        worst = 0.0
        frac: Dict[str, Dict[str, float]] = {}
        for genome, counts in table.items():
            n = sum(counts.values())
            if n <= 0:
                continue
            frac[genome] = {}
            for i, sp in enumerate(("train", "val", "test")):
                f = counts.get(sp, 0.0) / n
                frac[genome][sp] = round(f, 4)
                worst = max(worst, abs(f - ratio[i]))
        return worst, frac

    worst_t, frac_t = deviations(tot)
    out: Dict[str, object] = {
        "n_genomes": len(tot),
        "max_dev_promoters": round(worst_t, 5),
        "fractions_promoters": frac_t,
    }
    if labelled is not None:
        worst_l, frac_l = deviations(lab)
        out["max_dev_labels"] = round(worst_l, 5)
        out["fractions_labels"] = frac_l
        out["genomes_without_labels"] = sorted(set(tot) - set(lab))
    return out


def falsifiers(splits: SplitTable,
               leakage: Optional[Dict[str, object]] = None,
               evenness: Optional[Dict[str, object]] = None
               ) -> List[Tuple[str, bool, str]]:
    """(name, passed, detail). Each one can fail on real data."""
    c = splits.counts()
    n = max(sum(c.values()), 1)
    out = [
        ("Spy is a whole block, not one orthogroup",
         c.get("spy", 0) > 1,
         "spy holds %d orthogroups; holding out only pks1 leaves 6 "
         "byte-identical copies in training" % c.get("spy", 0)),
        # Not a threshold on orthogroup counts: the ratio is balanced over
        # promoters and labels, and orthogroups hold 1 to 82 promoters, so
        # 71.2 % of orthogroups carry 80.06 % of promoters and asserting 0.80
        # here would fail a correct split. The thresholds live on the
        # quantities that were actually balanced, in `evenness`.
        ("every split is non-empty and Spy is smallest",
         all(c.get(k, 0) > 0 for k in SPLIT_NAMES)
         and c.get("spy", 0) == min(c.get(k, 0) for k in SPLIT_NAMES),
         "orthogroups train %d / val %d / test %d / spy %d (fractions "
         "%.4f / %.4f / %.4f -- NOT expected to be 80/10/10; that ratio "
         "is balanced over promoters and labels)"
         % (c.get("train", 0), c.get("val", 0), c.get("test", 0),
            c.get("spy", 0), c.get("train", 0) / n, c.get("val", 0) / n,
            c.get("test", 0) / n)),
    ]
    if leakage is not None:
        out.append(("no homology crosses the boundary",
                    leakage["n_crossing_boundary"] == 0,
                    "%d of %d homologous pairs cross (by-orthogroup splitting "
                    "measured 31.87 %%)" % (leakage["n_crossing_boundary"],
                                            leakage["n_homologous_pairs"])))
    if evenness is not None:
        for i, sp in enumerate(("train", "val", "test")):
            want = (0.80, 0.10, 0.10)[i]
            got = [f[sp] for f in evenness["fractions_promoters"].values()]
            if got:
                out.append(
                    ("promoter fraction in %s is %.2f in EVERY strain"
                     % (sp, want),
                     max(abs(g - want) for g in got) < 0.01,
                     "range %.4f..%.4f over %d strains"
                     % (min(got), max(got), len(got))))
        out.append(("per-strain evenness within 0.01 in promoters",
                    evenness["max_dev_promoters"] < 0.01,
                    "max|dev| %.5f" % evenness["max_dev_promoters"]))
        if "max_dev_labels" in evenness:
            out.append(("per-strain evenness within 0.02 in labels",
                        evenness["max_dev_labels"] < 0.02,
                        "max|dev| %.5f (labels, not promoters -- these are "
                        "not proportional)" % evenness["max_dev_labels"]))
    return out
