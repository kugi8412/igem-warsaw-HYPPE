"""The anchors: pks1 and tef1, resolved from the corpus and scored in context.

Two of the ten delivered promoters are defined only relative to these two
sequences: Max is "at least as strong as tef1" and Mid is "strictly between
pks1 and tef1, with a margin". Neither target means anything as an absolute
number, because the model's strength scale is squeezed (measured calibration
slope 0.8153-0.8414, so predictions are 16-18 % compressed toward the mean)
and has no independent unit. The anchors are the unit.

Both are resolved from the corpus by record id, not from a FASTA:

    pks1 = P1_G009919   the design target; byte-identical to target/target.fasta
    tef1 = P1_G002124   the native tef1 of P1 (OG0000518, TPM 5782.9, 99.9th pct)
"""

from __future__ import annotations

import gzip
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.environ.get("HYPPE_DATA_ROOT",
                      os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

PKS1_ID = "P1_G009919"
TEF1_ID = "P1_G002124"

CALIBRATION_SLOPE_RANGE: Tuple[float, float] = (0.8153, 0.8414)

UNITS = "model mu units (absolute head, per category column)"


# Corpus
def _corpus_path() -> str:
    cand = [os.path.join(ROOT, "final_datasets", "sequences",
                         "sequences.tsv.gz"),
            os.path.join(os.environ.get("HYPPE_DATA_ROOT", ROOT), "data", "final_datasets", "sequences",
                         "sequences.tsv.gz")]
    for c in cand:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("no corpus at any of: %s" % (", ".join(cand),))


_CORPUS_MEMO: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}


def corpus_promoters(genome: str = "P1", limit: int = 0,
                     want: Optional[Sequence[str]] = None
                     ) -> List[Tuple[str, str]]:
    """`(id, sequence)` for promoter records of one genome.

    `want` ids are always returned whatever `limit` is, so a caller can take a
    cheap slice of the corpus and still be sure the anchors are inside it.
    """
    key = (genome, int(limit))
    if not want and key in _CORPUS_MEMO:
        return _CORPUS_MEMO[key]
    need = set(want or ())
    out: List[Tuple[str, str]] = []
    got: Dict[str, str] = {}
    with gzip.open(_corpus_path(), "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        iI, iG, iS, iGRP = (hdr.index("ID"), hdr.index("GENOME"),
                            hdr.index("SEQUENCE"), hdr.index("GROUP"))
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[iGRP] != "Promotor" or f[iG] != genome:
                continue
            if f[iI] in need:
                got[f[iI]] = f[iS].upper()
            if not limit or len(out) < limit:
                out.append((f[iI], f[iS].upper()))
            if limit and len(out) >= limit and len(got) == len(need):
                break
    have = {i for i, _ in out}
    for k, v in got.items():
        if k not in have:
            out.append((k, v))
    missing = need - {i for i, _ in out}
    if missing:
        raise SystemExit("REFUSED: corpus has no record(s) %s"
                         % (sorted(missing),))
    if not want:
        _CORPUS_MEMO[key] = out
    return out


@dataclass
class AnchorSequences:
    """The two anchor promoters and the evidence that they are the right two."""
    pks1: str
    tef1: str
    pks1_sha256: str
    tef1_sha256: str
    target_fasta: str
    target_header: str
    identity_checked: bool

    def note(self) -> str:
        return ("anchors from the corpus: pks1=%s (%d bp, sha %s), "
                "tef1=%s (%d bp, sha %s) | pks1 == %s (%s): %s"
                % (PKS1_ID, len(self.pks1), self.pks1_sha256[:12],
                   TEF1_ID, len(self.tef1), self.tef1_sha256[:12],
                   os.path.basename(self.target_fasta), self.target_header,
                   "VERIFIED byte-identical" if self.identity_checked
                   else "NOT CHECKED"))


def _read_fasta_one(path: str) -> Tuple[str, str]:
    head, parts = "", []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if head:
                    break
                head = line[1:]
            else:
                parts.append(line)
    return head, "".join(parts).upper()


def load_anchor_sequences(target_fasta: str = "") -> AnchorSequences:
    """Resolve both anchors and prove the pks1 one is the design target."""
    _cand_tf = [os.path.join(ROOT, "data", "target",
                             "design_target.fasta"),
                os.path.join(ROOT, "target", "target.fasta")]
    tf = target_fasta or next(
        (c for c in _cand_tf if os.path.exists(c)), _cand_tf[0])
    rows = dict(corpus_promoters("P1", limit=1, want=(PKS1_ID, TEF1_ID)))
    pks1, tef1 = rows[PKS1_ID], rows[TEF1_ID]
    head, seq = "", ""
    checked = False
    if os.path.exists(tf):
        head, seq = _read_fasta_one(tf)
        if seq != pks1:
            raise SystemExit(
                "REFUSED: corpus record %s (%d bp) is NOT byte-identical to "
                "%s (%d bp, header %r). The design target and the anchor "
                "would then be two different molecules and every Mid/Max "
                "threshold computed here would be anchored on the wrong one."
                % (PKS1_ID, len(pks1), tf, len(seq), head))
        checked = True
    else:
        raise SystemExit(
            "REFUSED: %s does not exist, so the claim 'the pks1 corpus record "
            "IS the design target' cannot be checked. It is checked, not "
            "assumed." % (tf,))
    return AnchorSequences(
        pks1=pks1, tef1=tef1,
        pks1_sha256=hashlib.sha256(pks1.encode()).hexdigest(),
        tef1_sha256=hashlib.sha256(tef1.encode()).hexdigest(),
        target_fasta=tf, target_header=head, identity_checked=checked)


# Anchors in context
@dataclass
class Anchors:
    """mu of both anchors, in one named context, for one category column.

    There is no constructor that takes numbers from a table. The only way to
    build this is `measure_anchors`, which measures them through the scorer
    the candidates will go through.
    """
    mu_pks1: float
    mu_tef1: float
    mu_pks1_columns: np.ndarray          # [7], all categories, for audit
    mu_tef1_columns: np.ndarray          # [7]
    category: str
    category_index: int
    context_key: str                     # e.g. "pHYPPE" / "pAN7-1"
    context_note: str                    # the full context description
    units: str = UNITS
    calibration_slope_range: Tuple[float, float] = CALIBRATION_SLOPE_RANGE
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """tef1 - pks1, in model mu units. May be <= 0; see `ordering_ok`."""
        return float(self.mu_tef1 - self.mu_pks1)

    @property
    def ordering_ok(self) -> bool:
        """True when the model puts tef1 above pks1 in this context.

        Not asserted at construction: on an early-epoch checkpoint it can be
        false, which is a measurement to report, not a reason for the module
        to become unimportable. `mid_window` is where it refuses.
        """
        return self.gap > 0.0

    def max_target(self, above_tef1: float = 0.0) -> float:
        """Max: mu >= mu(tef1), plus an optional margin, in model units.

        Not "maximise without end": a prediction outside the range of the
        training data is an extrapolation the model was never validated for,
        so the target is an anchor and not an optimum.
        """
        return float(self.mu_tef1 + max(0.0, float(above_tef1)))

    def mid_window(self, margin_frac: float = 0.15
                   ) -> Tuple[float, float, float]:
        """Mid: `(lo, hi, margin)` strictly inside the anchors, model units.

        Mid exists so as not to poison the fungus with runaway expression, so
        the margin is a biological safety allowance and is stated rather than
        implied: `margin = margin_frac * gap`, cut off both ends.
        """
        if not self.ordering_ok:
            raise SystemExit(
                "REFUSED to define a Mid window: in context %r, category %r, "
                "the model puts tef1 at %.4f and pks1 at %.4f, i.e. NOT "
                "above it (gap %.4f). 'Between the two' has no meaning when "
                "the order is inverted. Report the inversion; do not pick a "
                "window." % (self.context_key, self.category, self.mu_tef1,
                             self.mu_pks1, self.gap))
        m = float(margin_frac) * self.gap
        return (float(self.mu_pks1 + m), float(self.mu_tef1 - m), float(m))

    def satisfies_max(self, mu: float, above_tef1: float = 0.0) -> bool:
        return float(mu) >= self.max_target(above_tef1)

    def satisfies_mid(self, mu: float, margin_frac: float = 0.15) -> bool:
        lo, hi, _m = self.mid_window(margin_frac)
        return bool(lo < float(mu) < hi)

    def describe(self, margin_frac: float = 0.15) -> str:
        lines = [
            "ANCHORS  category %r (column %d)  context %r"
            % (self.category, self.category_index, self.context_key),
            "  %s" % (self.context_note,),
            "  units: %s" % (self.units,),
            "  pks1 %s = %+.4f | tef1 %s = %+.4f | gap %+.4f"
            % (PKS1_ID, self.mu_pks1, TEF1_ID, self.mu_tef1, self.gap),
        ]
        if not self.ordering_ok:
            lines.append(
                "  ORDER INVERTED: tef1 is NOT above pks1 in this context. "
                "Max is still defined (it is an anchor, not an ordering "
                "claim); Mid is NOT, and mid_window() refuses.")
        else:
            lo, hi, m = self.mid_window(margin_frac)
            lines.append("  MAX: mu >= %+.4f" % (self.max_target(),))
            lines.append("  MID: %+.4f < mu < %+.4f   (margin %.0f %% of the "
                         "gap = %.4f model units on EACH side)"
                         % (lo, hi, 100 * margin_frac, m))
        lo_s, hi_s = self.calibration_slope_range
        lines.append(
            "  CALIBRATION: measured slope %.4f-%.4f (predictions compressed "
            "toward the mean by %.0f-%.0f %%). These thresholds are in MODEL "
            "units. A difference here equals difference/slope in measured "
            "units -- use difference_to_measured_units(); an absolute LEVEL "
            "does not convert, because the intercept was not measured."
            % (lo_s, hi_s, 100 * (1 - hi_s), 100 * (1 - lo_s)))
        return "\n".join(lines)


def measure_anchors(scorer, context_key: str, category: str,
                    sequences: Optional[AnchorSequences] = None) -> Anchors:
    """Score both anchors through the scorer, in the context given.

    `scorer` must expose `mu_columns(seqs, context_key) -> [N, 7]`,
    `category_index(name) -> int` and `context_note(context_key) -> str`,
    which is exactly `scoring.Scorer`. The duck type keeps this module free of
    any import from `scoring`, so `scoring` may import it for the anchor
    sequences, and it makes an `Anchors` impossible to build from anything
    other than a live scorer.
    """
    seqs = sequences or load_anchor_sequences()
    mu = np.asarray(scorer.mu_columns([seqs.pks1, seqs.tef1], context_key),
                    dtype=np.float64)
    if mu.ndim != 2 or mu.shape[0] != 2:
        raise ValueError("scorer.mu_columns returned %r, expected [2, K]"
                         % (mu.shape,))
    k = int(scorer.category_index(category))
    return Anchors(
        mu_pks1=float(mu[0, k]), mu_tef1=float(mu[1, k]),
        mu_pks1_columns=mu[0].copy(), mu_tef1_columns=mu[1].copy(),
        category=category, category_index=k,
        context_key=str(context_key),
        context_note=str(scorer.context_note(context_key)),
        extra={"pks1_id": PKS1_ID, "tef1_id": TEF1_ID,
               "pks1_sha256": seqs.pks1_sha256,
               "tef1_sha256": seqs.tef1_sha256,
               "identity_checked": seqs.identity_checked})


# Units
def difference_to_measured_units(difference_model_units: float,
                                 slope: Optional[float] = None
                                 ) -> Tuple[float, float]:
    """A difference in model units -> the interval it spans in measured units.

    Only differences convert: the calibration is `pred = slope * truth +
    intercept`, the intercept was not measured here, so an absolute level has
    no image on the measured scale while a difference does, at
    `difference / slope`. The interval comes from the whole measured slope
    range, because reporting one number from a range turns a measurement into
    a claim.
    """
    d = float(difference_model_units)
    if slope is not None:
        s = float(slope)
        return (d / s, d / s)
    lo, hi = CALIBRATION_SLOPE_RANGE
    a, b = d / hi, d / lo
    return (min(a, b), max(a, b))


def target_in_model_units(target_measured_difference: float,
                          slope: Optional[float] = None
                          ) -> Tuple[float, float]:
    """The inverse: a wanted difference in measured units -> model units.

    Because predictions are compressed, asking the model for a difference of
    `d` when `d` was stated in measured units means aiming at `slope * d`.
    """
    d = float(target_measured_difference)
    if slope is not None:
        return (d * float(slope), d * float(slope))
    lo, hi = CALIBRATION_SLOPE_RANGE
    return (d * lo, d * hi)


__all__ = ["PKS1_ID", "TEF1_ID", "CALIBRATION_SLOPE_RANGE", "UNITS",
           "AnchorSequences", "Anchors", "corpus_promoters",
           "load_anchor_sequences", "measure_anchors",
           "difference_to_measured_units", "target_in_model_units"]
