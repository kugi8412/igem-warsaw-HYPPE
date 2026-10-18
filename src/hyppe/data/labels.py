"""Expression labels: sample policy, validity mask, and the three targets.

The decisions here rest on measurements over og_scope (11 323 orthogroups x
2332 samples), kept so they can be re-checked.

A per-genome variance decomposition of `combat` puts the promoter main effect
at 79.7-92.7 %, the sample main effect at 0.4-1.7 % (ComBat removed it) and
the promoter x condition interaction at 6.8-18.6 %. The interaction is
reproducible across biological replicates, reliability 0.68-0.84, so it is
signal -- but a minority of the variance, which is why the example is a
promoter and the condition is sampled per step rather than enumerated.

Target reliability by split-half with Spearman-Brown is 0.994-0.999 for the
mean over conditions and 0.805-0.931 for a single (promoter, sample) cell, so
label noise is not the bottleneck here. Orthologue correlation between
genomes is 0.845-0.980 and an orthogroup spans 14.81 genomes, giving an
effective multiplicity of 1.09: report metrics against an effective n of
about 12 400, not against the 167 724 rows.

The missingness rule is exact rather than statistical: `combat` holds a value
for (orthogroup, genome) if and only if that genome has exactly one gene in
that orthogroup, with no partial blocks. Missingness is a property of the
pair, so the validity mask is an (orthogroup x genome) boolean matrix, and it
must come from `combat`: `rank` imputes missing cells with exactly 1.0, and
since rank is descending and 1.0 means "silent", rank alone conflates "no
such gene in this strain" with "gene switched off" on 28.6 % of cells.

The corrected category column is adopted for its exclusions and relabellings
with one override. `exclude_mutant` (793) is adopted, because mutants add no
new promoter sequence and over-expression experiments replace the promoter
outright, so the label then belongs to a sequence that was not physically
there. `exclude_control` (41) and the ten relabellings are adopted.
`exclude_multicategory` (66) is overridden: it collapses every
multi-category sample into one bucket, destroying the information the project
wants to demonstrate, so those samples are kept as a multi-hot vector. Since
the corrected column erases composite categories, the vector is parsed from
`condition_category` on commas and the relabellings applied on top, giving
seven atomic categories rather than sixteen.

Honest limit wherever this capability is claimed: only 66 of 2332 samples
(2.8 %) carry more than one category. The conditioning is multi-hot by
construction, so any combination can be specified at inference without
retraining, but the evidence that combinations are learned rests on those 66.
Eleven further samples have a non-WT `mutant` value yet are not flagged
`exclude_mutant`; `sample_policy_report()` lists them rather than following
the discrepancy silently.
"""
from __future__ import annotations

import collections
import csv
import gzip
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Measured atomic categories, in a fixed order so the multi-hot vector is
# stable across runs and configs. Adding a category appends to this list via
# config; it must never be reordered, or saved checkpoints change meaning.
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "control",
    "carbon source",
    "biotic antagonism",
    "light response",
    "plant association",
    "mechanical injury",
    "other",
)

EXCLUDE_PREFIX = "exclude_"
# The one exclusion we override, and the reason, kept next to the constant.
KEEP_DESPITE_EXCLUSION: Tuple[str, ...] = ("exclude_multicategory",)


@dataclass
class SamplePolicy:
    """How samples are filtered and how categories are built."""

    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    category_column: str = "condition_category"
    #: The name the shipped file uses. The `*_curated` spelling belonged to
    #: an archived copy and survives only through the alias in
    #: :func:`load_samples`, which is why the contract gate once reported this
    #: column as read by nothing: it was read under a name no artefact carries.
    corrected_column: str = "condition_category_olek"
    keep_despite_exclusion: Tuple[str, ...] = KEEP_DESPITE_EXCLUSION
    #: A non-WT `mutant` value is never rescued by `keep_despite_exclusion`.
    #: The corrected column folds several reasons into one label, so without
    #: this, eleven mutants enter a corpus the experiment specifies as wild
    #: type -- in reesei (9) and IMI206040 (2), none in P1.
    wt_only: bool = True
    #: Relabellings that are plain category changes rather than exclusions are
    #: applied by replacing the parsed vector.
    apply_relabellings: bool = True

    def category_index(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.categories)}


@dataclass
class SampleTable:
    """Parsed sample metadata with the policy applied."""

    sample_ids: List[str]
    genome: List[str]
    keep: np.ndarray                 # bool [n_samples]
    category: np.ndarray             # float32 [n_samples, n_categories]
    excluded_reason: List[str]
    condition: List[str]
    study: List[str]
    text: List[str]                  # free text for the language encoder
    warnings: List[str] = field(default_factory=list)

    def n_kept(self) -> int:
        return int(self.keep.sum())


def _split_categories(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def load_samples(path: str, policy: Optional[SamplePolicy] = None
                 ) -> SampleTable:
    """Read a sample table and apply the documented policy.

    Pass ``hyppe.data.metadata.metadata_path()``, the one canonical table.
    Against the archived curated copy every decision this function makes is
    bit-identical; the single difference is ``text`` on SRX20732858, whose
    ``experiment_title`` the release copy had dropped.
    """
    pol = policy or SamplePolicy()
    cidx = pol.category_index()
    with io.open(path, encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        need = ("sample", "genome", pol.category_column, "condition", "study",
                "mutant")
        missing = [c for c in need if c not in header]
        if missing:
            raise ValueError("sample metadata lacks columns %r" % (missing,))
        ix = {c: header.index(c) for c in header}
        # Canonical name <- the name an older artefact might carry. A file
        # with the pre-rename header is accepted and the substitution is
        # reported, so that a cosmetic rename is not data loss.
        _legacy = {"condition_category_olek": "condition_category_curated",
                   "olek_keep": "curated_keep",
                   "olek_exclude_reason": "curated_exclude_reason"}
        _legacy_used = []
        for _new, _old in _legacy.items():
            if _new not in ix and _old in ix:
                ix[_new] = ix[_old]
                _legacy_used.append((_old, _new))
        has_corr = pol.corrected_column in ix

        sids, gens, keeps, cats, reasons = [], [], [], [], []
        conds, studies, texts, warns = [], [], [], []
        odd_mutants = []
        legacy_notes = ["header uses the pre-rename column %r, read as %r"
                        % (o, n) for o, n in _legacy_used]
        # What the table SAYS the mask is, kept beside what this loop computes.
        stated_keep, stated_reason = [], []
        for row in rdr:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            sid = row[ix["sample"]]
            raw_cat = row[ix[pol.category_column]]
            corr = row[ix[pol.corrected_column]] if has_corr else ""
            mut = row[ix["mutant"]].strip()

            keep = True
            reason = ""
            if corr.startswith(EXCLUDE_PREFIX):
                # The override saves multi-category samples, not everything
                # the corrected column folded into that label: eleven of them
                # are non-WT mutants, and the experiment is specified on
                # wild-type P1, so a mutant is excluded whatever else it is.
                _is_wt = mut in ("", "WT", "wt")
                if corr in pol.keep_despite_exclusion and (
                        _is_wt or not pol.wt_only):
                    reason = "kept_despite:%s" % corr
                elif corr in pol.keep_despite_exclusion:
                    keep = False
                    reason = "mutant_under:%s" % corr
                else:
                    keep = False
                    reason = corr

            names = _split_categories(raw_cat)
            if (pol.apply_relabellings and corr
                    and not corr.startswith(EXCLUDE_PREFIX)
                    and corr in cidx and len(names) == 1
                    and names[0] != corr):
                names = [corr]

            vec = np.zeros(len(pol.categories), dtype=np.float32)
            for nm in names:
                j = cidx.get(nm)
                if j is None:
                    warns.append("sample %s: unknown category %r" % (sid, nm))
                else:
                    vec[j] = 1.0

            is_mut = bool(mut) and mut.upper() not in ("WT", "NA", "NONE", "")
            if is_mut and keep:
                odd_mutants.append(sid)

            if "model_keep" in ix:
                stated_keep.append(row[ix["model_keep"]].strip() == "True")
            if "model_exclude_reason" in ix:
                stated_reason.append(row[ix["model_exclude_reason"]])

            sids.append(sid)
            gens.append(row[ix["genome"]])
            keeps.append(keep)
            cats.append(vec)
            reasons.append(reason)
            conds.append(row[ix["condition"]])
            studies.append(row[ix["study"]])
            # The C2 text fields and only those. `experiment_title` is in
            # `model/condition.py`'s FORBIDDEN_TEXT_FIELDS -- unique on 1491
            # of the kept samples, 66.9 % of its token types hapaxes -- and
            # the ban has to hold where the string is built, not only where
            # the spec is validated. Fields are looked up explicitly, so a
            # missing one contributes nothing instead of falling back to
            # column 0, the sample id.
            texts.append(" | ".join(
                x for x in (row[ix["condition"]],
                            row[ix["study_title"]] if "study_title" in ix
                            else "")
                if x))
    # The table's MODEL block states the mask and this function computes it;
    # disagreement stops the run. `model_keep` is a column of its own because
    # it is not `olek_keep`: the latter reads the curation literally and comes
    # to 1432, while this policy overrides `exclude_multicategory` and then
    # drops the 11 mutants among them under `wt_only`, keeping 1487.
    if len(stated_keep) == len(sids) and len(stated_reason) == len(sids):
        bad_keep = [s for s, a, b in zip(sids, keeps, stated_keep) if a != b]
        bad_reason = [s for s, a, b in zip(sids, reasons, stated_reason)
                      if a != b]
        if bad_keep or bad_reason:
            raise ValueError(
                "%s disagrees with this policy: model_keep differs on %d rows "
                "(e.g. %r), model_exclude_reason on %d (e.g. %r). Rebuild the "
                "table with src/scripts/build_samples_metadata.py, or change "
                "SamplePolicy and the builder together -- a mask that is "
                "stated in one place and computed in another, differently, is "
                "how 1432 gets trained on and 1498 gets reported."
                % (path, len(bad_keep), bad_keep[:3],
                   len(bad_reason), bad_reason[:3]))
    if odd_mutants:
        warns.append(
            "%d samples carry a non-WT `mutant` value but are NOT flagged "
            "exclude_mutant; reported, not silently followed: %s"
            % (len(odd_mutants), odd_mutants[:12]))
    return SampleTable(sids, gens, np.array(keeps, dtype=bool),
                       np.stack(cats) if cats else np.zeros((0, 7), "float32"),
                       reasons, conds, studies, texts, warns)


def sample_policy_report(tab: SampleTable) -> Dict[str, object]:
    """Everything a reader needs to audit the filtering decision."""
    by_reason = collections.Counter(r for r in tab.excluded_reason if r)
    per_cat = tab.category[tab.keep].sum(axis=0).astype(int).tolist()
    multi = int((tab.category[tab.keep].sum(axis=1) > 1).sum())
    return {
        "n_samples": len(tab.sample_ids),
        "n_kept": tab.n_kept(),
        "by_reason": dict(by_reason),
        "kept_per_category": per_cat,
        "kept_multicategory": multi,
        "kept_genomes": len(set(g for g, k in zip(tab.genome, tab.keep) if k)),
        "warnings": tab.warnings,
    }


# The matrix
def read_matrix(path: str, index_name: str = "og_id"
                ) -> Tuple[List[str], List[str], np.ndarray]:
    """Read a gzipped og x sample matrix. Missing cells become NaN."""
    op = gzip.open if path.endswith(".gz") else io.open
    rows: List[str] = []
    data: List[np.ndarray] = []
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header[0] != index_name:
            raise ValueError("expected first column %r, got %r"
                             % (index_name, header[0]))
        cols = header[1:]
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            rows.append(p[0])
            vals = np.full(len(cols), np.nan, dtype=np.float32)
            for j, v in enumerate(p[1:len(cols) + 1]):
                if v and v not in ("NA", "nan", "NaN"):
                    try:
                        vals[j] = float(v)
                    except ValueError:
                        pass
            data.append(vals)
    return rows, cols, np.stack(data) if data else np.zeros((0, len(cols)),
                                                            "float32")


@dataclass
class Targets:
    """The three learning targets, per (orthogroup, genome) and per category.

    The arrays are [n_pairs, C + 1]. Columns 0..C-1 follow
    `SamplePolicy.categories`; column C is the genome marginal over every kept
    sample of that genome, selected by ``category_ix = -1``. A sample counted
    in two categories contributes to both: the category is a multi-hot matrix
    and the cells are different questions about the same sample, not a double
    count.
    """

    og_ids: List[str]
    genomes: List[str]
    mu: np.ndarray                   # [n_pairs, C+1] mean combat per cell
    sigma: np.ndarray                # [n_pairs, C+1] sd over those samples
    n_obs: np.ndarray                # [n_pairs, C+1] samples behind each cell
    valid: np.ndarray                # [n_pairs, C+1] bool
    residual_index: Dict[Tuple[str, str], int]
    categories: Tuple[str, ...] = ()

    #: Index of the marginal column within the last axis.
    @property
    def marginal_column(self) -> int:
        return int(self.mu.shape[1]) - 1

    def column_of(self, category_ix: int) -> int:
        """`category_ix < 0` means the genome marginal."""
        return self.marginal_column if category_ix < 0 else int(category_ix)

    def weight(self) -> np.ndarray:
        """Sample-count weight, sqrt rather than linear.

        Reliability is already 0.994 at the low end, so extra samples buy very
        little once a handful are present and linear weighting would hand
        `reesei` (386 samples) 16x the pull of `P1` (24) for no measured gain.
        """
        return np.sqrt(np.maximum(self.n_obs, 0.0))


def build_targets(combat_path: str, samples: SampleTable,
                  min_obs: int = 2,
                  include_marginal: bool = True) -> Targets:
    """mu / sigma / validity per (orthogroup, genome), from ComBat values.

    `min_obs` guards sigma: a standard deviation from a single observation is
    not a measurement. Pairs below it stay valid for `mu` but their `sigma`
    is NaN, and the caller must mask the variance loss accordingly.
    """
    og_ids, cols, mat = read_matrix(combat_path)
    pos = {s: j for j, s in enumerate(cols)}
    genomes = sorted(set(g for g, k in zip(samples.genome, samples.keep) if k))
    by_genome: Dict[str, List[int]] = {g: [] for g in genomes}
    for sid, gen, keep in zip(samples.sample_ids, samples.genome,
                              samples.keep):
        if not keep:
            continue
        j = pos.get(sid)
        if j is not None:
            by_genome[gen].append(j)

    # Column layout: one per category, then the genome marginal last.
    cats = tuple(SamplePolicy().categories)
    C = len(cats)
    # Which kept sample columns back each (genome, category) cell. A sample in
    # two categories appears in both lists, by construction.
    by_cell: Dict[Tuple[str, int], List[int]] = {}
    for sid, gen, keep, catrow in zip(samples.sample_ids, samples.genome,
                                      samples.keep, samples.category):
        if not keep:
            continue
        j = pos.get(sid)
        if j is None:
            continue
        for c in range(min(C, int(np.asarray(catrow).shape[0]))):
            if float(catrow[c]) > 0.0:
                by_cell.setdefault((gen, c), []).append(j)

    og_out, gen_out = [], []
    mu, sd, nob, val = [], [], [], []
    for gi, og in enumerate(og_ids):
        row = mat[gi]
        for gen in genomes:
            cidx_marginal = by_genome[gen]
            if not cidx_marginal:
                continue
            og_out.append(og)
            gen_out.append(gen)
            m_row = np.full(C + 1, np.nan, dtype=np.float64)
            s_row = np.full(C + 1, np.nan, dtype=np.float64)
            n_row = np.zeros(C + 1, dtype=np.int32)
            v_row = np.zeros(C + 1, dtype=bool)
            for col in range(C + 1):
                cidx = (cidx_marginal if col == C
                        else by_cell.get((gen, col), ()))
                if not len(cidx):
                    continue
                sub = row[list(cidx)]
                ok = ~np.isnan(sub)
                n = int(ok.sum())
                n_row[col] = n
                if n == 0:
                    continue
                m_row[col] = float(sub[ok].mean())
                if n >= min_obs:
                    s_row[col] = float(sub[ok].std(ddof=1))
                v_row[col] = True
            mu.append(m_row)
            sd.append(s_row)
            nob.append(n_row)
            val.append(v_row)
    idx = {(o, g): i for i, (o, g) in enumerate(zip(og_out, gen_out))}
    _mu = np.asarray(mu, dtype="float32")
    _sd = np.asarray(sd, dtype="float32")
    _nob = np.asarray(nob, dtype="int32")
    _val = np.asarray(val, dtype=bool)
    if not include_marginal:
        # There is no `all` category. The marginal is a mean over every
        # sample of a genome, and the mixture differs per genome (P1 is
        # 70.8 % control, reesei 46.5 % carbon source), so its composition
        # is an artefact of which experiments were run rather than a
        # property of the promoter. Dropping the column here rather than
        # downstream means no caller can select it by accident.
        _C = len(cats)
        _mu, _sd = _mu[:, :_C], _sd[:, :_C]
        _nob, _val = _nob[:, :_C], _val[:, :_C]
    return Targets(og_out, gen_out, _mu, _sd, _nob, _val,
                   idx, cats)
