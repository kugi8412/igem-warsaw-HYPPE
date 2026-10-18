"""Sequence loading with an explicit, measured lowercase policy.

Lowercase means three different things across this project's files, and a
single `.upper()` or a single "strip lowercase" corrupts two of the three:

    source                      lowercase        meaning         action
    sequences.tsv.gz            interior         soft-mask       upper
    target/pks1_promotor_gen    4 interior       soft-mask       upper
    target/classical_promoters  2 terminal       flank note      trim
    target/flanki_GFP           1 terminal       homology arm    trim or keep

Stripping lowercase from `pks1_promoter` yields 727 bp instead of 800;
uppercasing `classical_promoters` yields 820 instead of 800 (= 9 + 800 + 11).
Both are the same bug, one convention applied where another holds, and both
look plausible. The policy is therefore declared per source, never inferred
silently; `AUTO` exists but decides by measurement, reports the decision, and
is validated by the expected length.

`psor_classical` is a known exception at 789 bp = 9 + 769 + 11, only 769 bp of
promoter against the 800 bp contract. Padding it would invent 31 bases, so the
loader reports it and `LengthPolicy` makes the caller's decision explicit.

The training alphabet is ACGNT plus lowercase. N is not an error and must not
be mapped to A: :func:`one_hot` gives it a uniform 0.25 vector, the only
encoding that adds no information, and :func:`n_fraction` is reported so a
caller can drop sequences that are mostly unknown.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Channel order is a contract, stated once and used everywhere. Note that the
# sibling DeepSTARR track uses A,C,T,G.
NUCLEOTIDE_ORDER = "ACGT"
_IDX = {c: i for i, c in enumerate(NUCLEOTIDE_ORDER)}
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class CasePolicy(Enum):
    """What lowercase means in a given source."""

    UPPERCASE = "uppercase"        # soft-mask: keep the bases, drop the case
    TRIM_TERMINAL = "trim"         # terminal lowercase is flank annotation
    AUTO = "auto"                  # decide by measurement, then report


class LengthPolicy(Enum):
    """What to do when a record does not match the expected length."""

    STRICT = "strict"              # raise
    REPORT = "report"              # keep, record in `warnings`
    DROP = "drop"                  # skip the record, record in `warnings`


@dataclass
class Record:
    """One sequence plus everything needed to trace it back."""

    name: str
    seq: str
    source: str
    raw_length: int
    trimmed_left: int = 0
    trimmed_right: int = 0
    n_fraction: float = 0.0
    meta: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


def reverse_complement(seq: str) -> str:
    """Reverse complement. NOT used for augmentation in this project.

    Promoters are directional, so a promoter and its reverse complement are
    not the same object and both RC augmentation and RC test-time averaging
    are forbidden here, unlike for enhancers. Motif scanning still uses both
    strands, because binding sites do occur on either: that is occupancy
    symmetry, not sequence symmetry.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def lowercase_blocks(seq: str) -> List[Tuple[int, int]]:
    """Contiguous runs of lowercase as (start, end_inclusive)."""
    out: List[Tuple[int, int]] = []
    start = None
    for i, c in enumerate(seq):
        if c.islower():
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(seq) - 1))
    return out


def classify_case(seq: str, edge_tolerance: int = 2) -> str:
    """'terminal', 'interior', 'both' or 'none' -- the basis for AUTO."""
    blocks = lowercase_blocks(seq)
    if not blocks:
        return "none"
    n = len(seq)
    terminal = any(a <= edge_tolerance or b >= n - 1 - edge_tolerance
                   for a, b in blocks)
    interior = any(a > edge_tolerance and b < n - 1 - edge_tolerance
                   for a, b in blocks)
    if terminal and interior:
        return "both"
    return "terminal" if terminal else "interior"


def _trim_terminal_lowercase(seq: str, edge_tolerance: int = 2
                             ) -> Tuple[str, int, int]:
    """Remove ONLY the leading and trailing lowercase runs."""
    blocks = lowercase_blocks(seq)
    if not blocks:
        return seq, 0, 0
    n = len(seq)
    left = 0
    right = 0
    if blocks[0][0] <= edge_tolerance:
        left = blocks[0][1] + 1
    if blocks[-1][1] >= n - 1 - edge_tolerance:
        right = n - blocks[-1][0]
    if left + right >= n:
        raise ValueError("trimming would remove the whole sequence (%d bp)" % n)
    return seq[left:n - right], left, right


def apply_case_policy(seq: str, policy: CasePolicy,
                      expected_length: Optional[int] = None,
                      edge_tolerance: int = 2
                      ) -> Tuple[str, int, int, str]:
    """Returns (sequence, trimmed_left, trimmed_right, decision)."""
    if policy is CasePolicy.UPPERCASE:
        return seq.upper(), 0, 0, "uppercase"
    if policy is CasePolicy.TRIM_TERMINAL:
        s, a, b = _trim_terminal_lowercase(seq, edge_tolerance)
        return s.upper(), a, b, "trim"
    # AUTO: decide by what the lowercase looks like, then let the expected
    # length arbitrate. A decision that misses the expected length is
    # reported by the caller rather than forced.
    kind = classify_case(seq, edge_tolerance)
    if kind in ("none", "interior"):
        return seq.upper(), 0, 0, "auto:uppercase(%s)" % kind
    s, a, b = _trim_terminal_lowercase(seq, edge_tolerance)
    if expected_length is not None and len(s) != expected_length:
        if len(seq) == expected_length:
            # Trimming moved away from the contract while the raw record
            # already met it, so the lowercase was soft-mask after all.
            return seq.upper(), 0, 0, "auto:uppercase(raw matched length)"
    return s.upper(), a, b, "auto:trim(%s)" % kind


def one_hot(seq: str, dtype=np.float32) -> np.ndarray:
    """[4, L] one-hot over A,C,G,T. N and anything unknown becomes 0.25."""
    L = len(seq)
    out = np.full((4, L), 0.25, dtype=dtype)
    for i, c in enumerate(seq.upper()):
        j = _IDX.get(c)
        if j is not None:
            out[:, i] = 0.0
            out[j, i] = 1.0
    return out


def n_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    return sum(1 for c in s if c not in _IDX) / float(len(s))


# FASTA
def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence). Headers keep everything after '>'."""
    name = None
    parts: List[str] = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name = line[1:]
                parts = []
            else:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts)


def load_fasta(path: str, policy: CasePolicy,
               expected_length: Optional[int] = None,
               length_policy: LengthPolicy = LengthPolicy.REPORT,
               allow_short: Sequence[str] = ()
               ) -> Tuple[List[Record], List[str]]:
    """Load a FASTA under an explicit case policy. Returns (records, warnings).

    `allow_short` lists record-name prefixes whose length mismatch is known
    and accepted (e.g. 'psor'), so that a NEW mismatch is still an error.
    """
    records: List[Record] = []
    warnings: List[str] = []
    for name, raw in read_fasta(path):
        seq, a, b, decision = apply_case_policy(
            raw, policy, expected_length=expected_length)
        rec = Record(name=name, seq=seq, source=path, raw_length=len(raw),
                     trimmed_left=a, trimmed_right=b,
                     n_fraction=n_fraction(seq),
                     meta={"case_decision": decision})
        if expected_length is not None and len(seq) != expected_length:
            known = any(name.startswith(p) for p in allow_short)
            msg = ("%s: %d bp after %s (raw %d), expected %d%s"
                   % (name.split("|")[0], len(seq), decision, len(raw),
                      expected_length, " [known]" if known else ""))
            if length_policy is LengthPolicy.STRICT and not known:
                raise ValueError(msg)
            warnings.append(msg)
            if length_policy is LengthPolicy.DROP and not known:
                continue
        records.append(rec)
    return records, warnings


# TSV
def iter_sequence_tsv(path: str, groups: Optional[Sequence[str]] = None,
                      policy: CasePolicy = CasePolicy.UPPERCASE
                      ) -> Iterator[Record]:
    """Stream the project's sequences.tsv[.gz].

    The default is UPPERCASE because 674 795 of these sequences carry
    lowercase only in the interior and only 1 275 carry it only at an edge:
    here lowercase is soft-masking, not annotation.
    """
    want = set(groups) if groups else None
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        need = ("ID", "SPECIES", "GENOME", "SEQUENCE", "GROUP", "OG_ID",
                "GENE_ID")
        missing = [c for c in need if c not in header]
        if missing:
            raise ValueError("sequence table lacks columns %r (has %r)"
                             % (missing, header))
        ix = {c: header.index(c) for c in need}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            grp = p[ix["GROUP"]]
            if want is not None and grp not in want:
                continue
            raw = p[ix["SEQUENCE"]]
            seq, a, b, decision = apply_case_policy(raw, policy)
            yield Record(
                name=p[ix["ID"]], seq=seq, source=path, raw_length=len(raw),
                trimmed_left=a, trimmed_right=b, n_fraction=n_fraction(seq),
                meta={"species": p[ix["SPECIES"]], "genome": p[ix["GENOME"]],
                      "group": grp, "og_id": p[ix["OG_ID"]],
                      "gene_id": p[ix["GENE_ID"]], "case_decision": decision})
