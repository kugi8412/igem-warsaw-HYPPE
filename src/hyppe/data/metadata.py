"""The per-sample metadata loader: one canonical table, no fallbacks.

Four files named ``samples_metadata*.tsv`` once held three distinct contents
and disagreed on ``condition`` in 2300 of 2332 rows, so a replicate key built
from that column depended on which module you were standing in. Nothing here
falls back to another path, accepts a short table or returns an empty frame;
every failure mode raises, naming the file and what was wrong with it.

The table is three blocks and the ORDER is part of the contract: ``KEY`` (1) |
``MODEL`` (9) | ``AUDIT`` (35). Read it through :func:`model_view` rather than
:func:`load` -- that is the block the training pipeline is allowed to see, and
:func:`assert_contract` raises on a missing column and on a permuted header
alike. The sample mask is :func:`model_mask`, 1487 of 2332, not ``olek_keep``.

``batch_corrected_og`` does not mean "this sample is a column of
``combat.tsv.gz``" -- all 2332 of them are. It means ComBat moved the sample.
The 338 rows flagged ``False`` are raw ``log2(TPM+1)`` sitting inside the
corrected matrix, 14.5 % of it, because their study has one sample and a
singleton batch has no within-batch variance to estimate. Use
:func:`corrected_mask` before any statement about batch on that matrix.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Optional, Sequence

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
#: ``.../HYPPE``: this file is ``ROOT/src/hyppe/data/metadata.py``.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

#: The one canonical table.  Built by ``src/scripts/build_samples_metadata.py``.
CANONICAL = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "samples_metadata.tsv")
MANIFEST = os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "samples_metadata.manifest.json")

#: The release ships 2332 samples and every matrix has exactly these columns.
#: A table with any other length is a different dataset, not a newer one.
N_SAMPLES = 2332

#: Present in every historical copy; a table lacking one of these is not this
#: dataset and must not be silently accepted.
REQUIRED = ("sample", "genome", "label", "study", "condition", "condition_category",
            "mutant", "bioproject", "batch_corrected_og", "batch_corrected_all_genes",
            "condition_category_olek")

#: Columns stored as the strings ``True``/``False``.
BOOL_COLUMNS = ("batch_corrected_og", "batch_corrected_all_genes",
                "condition_category_is_multi", "olek_keep", "model_keep")

#: Columns reconciled between the legacy and release copies.  Each has a
#: ``_source`` and a ``_legacy`` companion.
RECONCILED = ("condition", "condition_category", "mutant", "experiment_title")

# These tuples repeat the builder's own lists on purpose: a loader that
# imports its expectations from the thing it is checking checks nothing.
# A column is MODEL if and only if some line under src/hyppe/{data,training,
# model} reads its value; naming a column in order to ban it is not a read.
KEY_COLUMNS: tuple = ("sample",)

MODEL_COLUMNS: tuple = (
    "genome",
    "study",
    "mutant",
    "condition",
    "study_title",
    "condition_category",
    # `*_olek`, not the `*_curated` of the archived copy this contract was
    # first written against. While that name was stale the four contract
    # functions all raised, making the MODEL/AUDIT split unreachable.
    "condition_category_olek",
    "model_keep",
    "model_exclude_reason",
)

AUDIT_COLUMNS: tuple = (
    # release fields no pipeline line reads
    "label", "instrument", "layout", "bioproject", "experiment_title",
    "library_strategy", "library_selection", "library_source", "sample_name",
    "geo_accessions", "mapping_rate",
    "batch_corrected_og", "batch_corrected_all_genes",
    # provenance
    "condition_source", "condition_legacy",
    "condition_category_source", "condition_category_legacy",
    "experiment_title_source", "experiment_title_legacy",
    "mutant_source", "mutant_legacy",
    # derived convenience
    "condition_category_norm", "condition_category_primary",
    # `*_olek`, not `*_curated`: same rename as above.
    "condition_category_is_multi", "olek_keep", "olek_exclude_reason",
    "study_n_samples", "genome_study_n_samples",
    # denormalised copy of data/conditions/condition_taxonomy.tsv
    "subcategories", "n_subcategories", "axes", "matched_terms",
    "match_field", "confidence", "separating_subcategories",
)

#: The full column order, which the file must reproduce exactly.
CONTRACT_COLUMNS: tuple = KEY_COLUMNS + MODEL_COLUMNS + AUDIT_COLUMNS

#: Measured on the file: 1487, not the 1498 of older comments, which was the
#: count before ``wt_only``. ``olek_keep`` reads the curation literally and
#: gives 1432; the policy then overrides ``exclude_multicategory`` and
#: ``wt_only`` removes the 11 of those 66 samples that are mutants.
#: :func:`model_view` raises when the count disagrees.
N_MODEL_KEEP = 1487
N_OLEK_KEEP = 1432

#: Where the superseded HYPPE-side copies went. Outside the HYPPE tree, so
#: that exactly one ``samples_metadata*`` file exists under it.
ARCHIVE = os.path.abspath(os.path.join(
    ROOT, "..", "_archiwum", "2026-09-12", "samples_metadata", "HYPPE"))

#: The superseded copies, so :func:`assert_not_legacy` can name them. The
#: first three no longer exist; their paths stay here anyway, because a script
#: still carrying one hard-coded must be caught by name rather than by a
#: FileNotFoundError from somewhere else.
SUPERSEDED = (
    # gone from HYPPE, still rejected by name
    os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "final_datasets", "og_scope", "samples_metadata.tsv"),
    os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "final_datasets", "og_scope", "samples_metadata_curated.tsv"),
    os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "samples_metadata_curated.tsv"),
    # where two of them now live
    os.path.join(ARCHIVE, "data", "final_datasets", "og_scope", "samples_metadata.tsv"),
    os.path.join(ARCHIVE, "data", "samples_metadata_curated.tsv"),
    # never inside HYPPE; still the source of the reconciled `_legacy` columns
    os.path.abspath(os.path.join(ROOT, "..", "HYperbolicPromoterProgrammingEngine",
                                 "final_datasets", "og_scope", "samples_metadata.tsv")),
)


class MetadataError(RuntimeError):
    """Raised for every failure mode of this loader.  There is no fallback."""


def metadata_path() -> str:
    """Absolute path to the canonical table, or raise saying how to build it."""
    if not os.path.exists(CANONICAL):
        raise MetadataError(
            "canonical sample metadata not found at %s\n"
            "(HYPPE root resolved to %s)\n"
            "Build it with:  python src/scripts/build_samples_metadata.py"
            % (CANONICAL, ROOT))
    return CANONICAL


def _to_bool(series: pd.Series, column: str) -> pd.Series:
    s = series.astype(str).str.strip()
    ok = s.isin(("True", "False"))
    if not bool(ok.all()):
        bad = sorted(set(s[~ok]))[:5]
        raise MetadataError(
            "column %r must hold only 'True'/'False', found %r. Refusing to "
            "coerce: in Python every non-empty string is truthy, so a silent "
            "cast here would make 'false' mean True." % (column, bad))
    return s == "True"


def load(columns: Optional[Sequence[str]] = None,
         path: Optional[str] = None,
         typed_bools: bool = True) -> pd.DataFrame:
    """Load the canonical table as strings, indexed by position, validated.

    ``columns`` selects a subset, and a name that does not exist raises rather
    than coming back as NaN. ``path`` overrides the canonical file, for
    reproducing an old number against a superseded copy; it is never
    defaulted to. ``typed_bools`` converts the flag columns to real ``bool``
    and should be turned off only when writing the table back out unchanged.
    Raises :class:`MetadataError` on a missing file, a wrong row count, a
    duplicate or empty ``sample``, or a missing or unknown column.
    """
    p = path or metadata_path()
    if not os.path.exists(p):
        raise MetadataError("sample metadata not found at %s" % p)

    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_filter=False)

    if len(df) != N_SAMPLES:
        raise MetadataError(
            "%s has %d rows, expected %d. This is not the HYPPE release sample "
            "table; nothing downstream is keyed to it." % (p, len(df), N_SAMPLES))

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise MetadataError(
            "%s lacks required columns %r. It looks like a superseded copy -- "
            "the canonical table is %s, built by "
            "src/scripts/build_samples_metadata.py" % (p, missing, CANONICAL))

    if df["sample"].duplicated().any():
        dup = sorted(df.loc[df["sample"].duplicated(), "sample"])[:5]
        raise MetadataError("%s has duplicate sample ids, e.g. %r" % (p, dup))
    if (df["sample"].str.strip() == "").any():
        raise MetadataError("%s has empty sample ids" % p)

    if typed_bools:
        for c in BOOL_COLUMNS:
            if c in df.columns:
                df[c] = _to_bool(df[c], c)

    if columns is not None:
        unknown = [c for c in columns if c not in df.columns]
        if unknown:
            raise MetadataError(
                "columns %r are not in %s. Available: %s"
                % (unknown, p, ", ".join(df.columns)))
        df = df.loc[:, list(columns)]
    return df


def load_indexed(columns: Optional[Sequence[str]] = None,
                 path: Optional[str] = None) -> pd.DataFrame:
    """:func:`load`, indexed by ``sample`` -- the shape most callers want."""
    cols = None if columns is None else (
        list(columns) if "sample" in columns else ["sample"] + list(columns))
    return load(cols, path).set_index("sample")


def load_rows(path: Optional[str] = None) -> List[dict]:
    """The same table as a list of dicts, for the csv-only callers."""
    import csv
    p = path or metadata_path()
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) != N_SAMPLES:
        raise MetadataError("%s has %d rows, expected %d" % (p, len(rows), N_SAMPLES))
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise MetadataError("%s lacks required columns %r" % (p, missing))
    return rows


def reorder_to(samples: Iterable[str], path: Optional[str] = None) -> pd.DataFrame:
    """Metadata aligned to a matrix's column order, raising on any mismatch.

    A plain reindex fills unknown samples with NaN and carries on, so the
    misalignment surfaces as a weak correlation three functions later. Here a
    mismatch in either direction raises.
    """
    want = list(samples)
    df = load_indexed(path=path)
    missing = [s for s in want if s not in df.index]
    if missing:
        raise MetadataError(
            "%d of %d requested samples are not in the metadata, e.g. %r"
            % (len(missing), len(want), missing[:5]))
    extra = len(df) - len(set(want))
    if extra:
        raise MetadataError(
            "metadata holds %d samples not requested; pass all %d matrix columns "
            "or slice deliberately after loading" % (extra, N_SAMPLES))
    return df.loc[want]


# The contract: KEY | MODEL | AUDIT
def assert_contract(path: Optional[str] = None) -> List[str]:
    """Raise unless the table's header is :data:`CONTRACT_COLUMNS`, in order.

    Order is checked, not just membership: a header whose columns are all
    present but permuted reads as a correct table and is not one. Returns the
    header it accepted, so a caller can log it.
    """
    p = path or metadata_path()
    with open(p, encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\n").split("\t")

    want = list(CONTRACT_COLUMNS)
    if header == want:
        return header

    missing = [c for c in want if c not in header]
    extra = [c for c in header if c not in want]
    if missing or extra:
        raise MetadataError(
            "%s does not satisfy the column contract.\n"
            "  missing: %r\n"
            "  unexpected: %r\n"
            "The contract is data/samples_metadata.contract.md; rebuild with "
            "python src/scripts/build_samples_metadata.py. Nothing may be "
            "dropped from this table -- if a column is genuinely new, it has "
            "to be added to KEY_COLUMNS / MODEL_COLUMNS / AUDIT_COLUMNS here "
            "and to the builder, in the same order, in one change."
            % (p, missing, extra))

    first = next(i for i, (a, b) in enumerate(zip(header, want)) if a != b)
    raise MetadataError(
        "%s has the right columns in the WRONG ORDER: position %d is %r, the "
        "contract says %r. Column order is part of the contract -- readers "
        "that slice by position, and every diff of this file, depend on it."
        % (p, first, header[first], want[first]))


def model_view(path: Optional[str] = None,
               typed_bools: bool = True) -> pd.DataFrame:
    """The MODEL block and nothing else: ``sample`` + :data:`MODEL_COLUMNS`.

    Indexed by ``sample``, with ``model_keep`` as a real ``bool``. Raises if a
    contract column is missing or the order moved, so a table that drifted
    stops the run here rather than producing a quietly different model. AUDIT
    columns are unreachable by design: adding one to the model should be a
    visible change to :data:`MODEL_COLUMNS`, not a string at a call site.
    """
    assert_contract(path)
    df = load(list(KEY_COLUMNS + MODEL_COLUMNS), path=path,
              typed_bools=typed_bools).set_index("sample")

    n_keep = int(df["model_keep"].sum()) if typed_bools else \
        int((df["model_keep"] == "True").sum())
    if n_keep != N_MODEL_KEEP:
        raise MetadataError(
            "model_keep selects %d samples, the contract says %d. Either the "
            "curation changed or the override of exclude_multicategory was "
            "dropped; both change what every reported number means."
            % (n_keep, N_MODEL_KEEP))
    return df


def audit_view(path: Optional[str] = None,
               typed_bools: bool = True) -> pd.DataFrame:
    """The AUDIT block: provenance, derived columns and the taxonomy copy.

    Separate from :func:`model_view` so that "the model reads this" and "this
    is here so a correction can be shown" are different calls and grep
    differently. Nothing here may enter training without first moving to
    :data:`MODEL_COLUMNS`.
    """
    assert_contract(path)
    return load(list(KEY_COLUMNS + AUDIT_COLUMNS), path=path,
                typed_bools=typed_bools).set_index("sample")


def model_mask(path: Optional[str] = None) -> pd.Series:
    """The sample mask the pipeline applies -- 1487 of 2332, not ``olek_keep``.

    The 66 ``exclude_multicategory`` samples are kept on purpose: they are the
    only evidence in this corpus that condition COMBINATIONS are learnable.
    """
    return model_view(path)["model_keep"]


def corrected_mask(scope: str = "og_scope", path: Optional[str] = None) -> pd.Series:
    """Boolean per sample: did ComBat actually MOVE this sample, in this scope?

    ``combat.tsv.gz`` carries all 2332 samples as columns in both scopes, but
    338 of them (og_scope) are bit-identical to ``log2(TPM+1)`` -- rms
    difference 2.9e-05 against 6.2e-01 for the lowest corrected sample, with no
    overlap. Any claim that the batch effect is not visible on combat must be
    made on ``df[corrected_mask()]`` or state that 14.5 % of the matrix is raw.
    """
    col = {"og_scope": "batch_corrected_og",
           "all_genes_scope": "batch_corrected_all_genes"}.get(scope)
    if col is None:
        raise MetadataError("scope must be 'og_scope' or 'all_genes_scope', got %r" % scope)
    df = load_indexed(path=path)
    return df[col]


def legacy_column(name: str, path: Optional[str] = None) -> pd.Series:
    """Recover a superseded column exactly, for reproducing an older number.

    ``legacy_column('condition')`` returns the 18-column copy's ``condition``
    byte for byte: the ``_legacy`` value where the two disagreed, the canonical
    value where they did not.
    """
    if name not in RECONCILED:
        raise MetadataError(
            "%r was identical in every copy, so there is no legacy version of "
            "it. Reconciled columns are %r" % (name, list(RECONCILED)))
    df = load_indexed(path=path)
    alt = df[name + "_legacy"]
    return alt.where(alt != "", df[name])


def assert_not_legacy(path: str) -> None:
    """Raise if ``path`` is one of the superseded copies.

    For call sites that accept a ``--metadata`` override: one pointing at a
    superseded file is usually a stale default rather than an intent.
    """
    p = os.path.abspath(path)
    if any(os.path.normcase(p) == os.path.normcase(s) for s in SUPERSEDED):
        raise MetadataError(
            "%s is a SUPERSEDED copy of the sample table. The canonical table is "
            "%s. If you are deliberately reproducing an older number, use "
            "load(path=...) and say so in the log." % (p, CANONICAL))


def fingerprint(path: Optional[str] = None) -> str:
    """``sha256[:16]`` of the table actually read, for the run log."""
    p = path or metadata_path()
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


__all__ = ["CANONICAL", "MANIFEST", "ARCHIVE", "N_SAMPLES", "REQUIRED", "BOOL_COLUMNS",
           "RECONCILED", "SUPERSEDED", "MetadataError", "metadata_path", "load",
           "load_indexed", "load_rows", "reorder_to", "corrected_mask",
           "legacy_column", "assert_not_legacy", "fingerprint",
           "KEY_COLUMNS", "MODEL_COLUMNS", "AUDIT_COLUMNS", "CONTRACT_COLUMNS",
           "N_MODEL_KEEP", "N_OLEK_KEEP",
           "assert_contract", "model_view", "audit_view", "model_mask"]
