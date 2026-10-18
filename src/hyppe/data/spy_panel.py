"""SPY: the validation set the model is selected on, synthetics included.

`data/splits/SPY_SET.fasta` is 249 sequences built around the design target,
and its composition is what makes it a selection set rather than a second
validation split: 45 native P1 promoters with a measured mu, one block per
strength quintile; 25 of the same on eight other Trichoderma species, for
transfer; the pks1 and tef1 anchors; classical promoters from the literature,
with the ones that leak into training kept separately and never scored; four
blocks of 20 pks1 mutants (2 or 88 substitutions, inside or outside a known
motif); 30 chimeras of two P1 promoters; 30 order-4 Euler shuffles, which
keep composition to 5-mers and destroy architecture; and 30 windows of coding
sequence. Spy holds out the whole leakage block of pks1 -- 81 orthogroups,
896 promoters -- because holding out the target alone would leave six
byte-identical copies in training and make any "zero-shot" number a memory
read.

Selecting on spy is legitimate; reporting test on it is not. Spy is looked at
once per epoch and decides which checkpoint is kept, so it is spent exactly as
`val` is: inside the selection budget, outside the training budget.

Four contrasts. `spy_scc` is Spearman of predicted strength against measured
mu on the native P1 promoters. `motif_sensitivity` compares in-motif and
out-of-motif mutants carrying the same number of substitutions from the same
parent, so the only difference is whether a known motif was hit; no measured y
is needed, because the claim is about sensitivity and not accuracy.
`edit_resolution` asks whether two substitutions in 800 bp move the prediction
at all, with the 88-substitution mutants as the control: if the large set does
not move more, the head is responding to noise. `spy_auc` requires the
shuffles and the coding windows to score below the natives, the Euler set
being the falsifier, since a discriminator can reach 0.99 against random
negatives having learned composition alone.

The panel cannot validate absolute strength: the target scale has no physical
unit and the wet readout is GFP protein while the labels are mRNA rank. Every
number here is an order or a sensitivity.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .sequences import read_fasta

__all__ = [
    "SPY_CLASSES",
    "SCORED_NATIVE_CLASSES",
    "NEGATIVE_CLASSES",
    "SpyRecord",
    "SpyPanel",
    "load_spy_panel",
    "mann_whitney_u",
]

#: Every class the builder emits, with what it is. Keys are the literal
#: `klasa=` values in the FASTA; they are Polish because the artefact is, and
#: renaming them here would mean this module and the file disagree.
SPY_CLASSES: Dict[str, str] = {
    "nat_q1": "native P1 promoter, strength quintile 1 (weakest)",
    "nat_q2": "native P1 promoter, quintile 2",
    "nat_q3": "native P1 promoter, quintile 3",
    "nat_q4": "native P1 promoter, quintile 4",
    "nat_q5": "native P1 promoter, quintile 5 (strongest)",
    "natobcy_q1": "native promoter of another species, quintile 1",
    "natobcy_q2": "native promoter of another species, quintile 2",
    "natobcy_q3": "native promoter of another species, quintile 3",
    "natobcy_q4": "native promoter of another species, quintile 4",
    "natobcy_q5": "native promoter of another species, quintile 5",
    "kotwica_pks1": "the design target, native pks1 promoter",
    "kotwica_tef1": "the upper reference, native tef1 promoter",
    "kotwica_klasyczna": "classical promoter from the literature, clean",
    "kotwica_klasyczna_skazona": "classical promoter that LEAKS into training",
    "mut_zachowane": "pks1 + 2 substitutions in a free block (motifs intact)",
    "mut_zniszczone": "pks1 + 2 substitutions inside a known motif",
    "mutL_zachowane": "pks1 + 88 substitutions, none in a motif",
    "mutL_zniszczone": "pks1 + 88 substitutions hitting motifs",
    "chimera": "two P1 promoters spliced at a random breakpoint",
    "losowa_euler4": "order-4 Euler shuffle: 5-mer composition kept",
    "gen_kontrola": "800 bp of coding sequence -- not a promoter",
}

#: Classes that carry a measured `mu` and may enter a ranking metric.
SCORED_NATIVE_CLASSES: Tuple[str, ...] = (
    "nat_q1", "nat_q2", "nat_q3", "nat_q4", "nat_q5",
    "kotwica_pks1", "kotwica_tef1")
#: The same for the transfer metric, on species other than P1.
FOREIGN_NATIVE_CLASSES: Tuple[str, ...] = (
    "natobcy_q1", "natobcy_q2", "natobcy_q3", "natobcy_q4", "natobcy_q5")
#: Classes that are not promoters and must score below the natives.
NEGATIVE_CLASSES: Tuple[str, ...] = ("losowa_euler4", "gen_kontrola")
#: Excluded from every scored metric, by name and with the reason attached.
EXCLUDED_CLASSES: Dict[str, str] = {
    "kotwica_klasyczna_skazona":
        "these classical promoters share exact 25-mers with training "
        "sequences, so a score on them is a memory read; the builder kept "
        "them so the contamination is visible rather than dropped",
}

_ATTR = re.compile(r"(\w+)=([^;\s]*)")


# Records
@dataclass
class SpyRecord:
    """One panel member, with everything its header declared."""

    name: str
    sequence: str
    klasa: str
    n_substitutions: int = 0
    retention: float = float("nan")
    parent: str = ""
    genome: str = "P1"
    og_id: str = ""
    motif: str = ""
    positions: Tuple[int, ...] = ()
    mu: float = float("nan")
    quantile: int = 0
    leak25: float = float("nan")
    raw: str = ""

    @property
    def is_mutant(self) -> bool:
        return self.klasa.startswith("mut")

    @property
    def in_motif(self) -> bool:
        """True for the mutants that were aimed at a motif.

        Read from the `motyw` header field and not from the class name: the
        name is a label, the field is the fact.
        """
        return bool(self.motif) and self.motif != "-"


#: Anchor weights in `spy_anchor_pct`, deliberately unequal. Over the 18 P1
#: control samples tef1 sits at mean 99.8, sd 0.1 -- practically constant --
#: while pks1, the target, is mean 90.7, sd 10.6. An even mean would measure
#: mostly tef1.
ANCHOR_PCT_WEIGHTS = {"pks1": 0.75, "tef1": 0.25}


def _parse_header(header: str) -> Dict[str, str]:
    """`>name klasa=... a=b;c=d` -> flat dict. Later keys win.

    The builder writes two attribute styles in one line, space-separated and
    semicolon-separated after a tag. One regex over the whole line reads
    both, which keeps this parser independent of the tag.
    """
    return {k: v for k, v in _ATTR.findall(header)}


def _int(d: Mapping[str, str], key: str, default: int = 0) -> int:
    v = d.get(key, "")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(d: Mapping[str, str], key: str) -> float:
    v = d.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _positions(d: Mapping[str, str]) -> Tuple[int, ...]:
    raw = d.get("pozycje", "")
    if not raw or raw == "-":
        return ()
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip().rstrip(".")
        if part.isdigit():
            out.append(int(part))
    return tuple(out)


# The panel
@dataclass
class SpyPanel:
    """The 249 sequences, indexed by class, with the contrasts precomputed."""

    records: List[SpyRecord]
    source: str = ""
    #: A statement about the budget, not about the file.
    role: str = "selection (validation budget) -- never training, never test"
    notes: List[str] = field(default_factory=list)

    # Indexing
    def by_class(self, *classes: str) -> List[SpyRecord]:
        want = set(classes)
        return [r for r in self.records if r.klasa in want]

    def index(self) -> Dict[str, int]:
        return {r.name: i for i, r in enumerate(self.records)}

    def sequences(self) -> List[str]:
        return [r.sequence for r in self.records]

    def parent(self) -> Optional[SpyRecord]:
        """The pks1 anchor: every mutant's reference point."""
        got = self.by_class("kotwica_pks1")
        return got[0] if got else None

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records:
            out[r.klasa] = out.get(r.klasa, 0) + 1
        return out

    # The four contrasts
    def score(self, prediction: Mapping[str, float],
              promoterness: Optional[Mapping[str, float]] = None
              ) -> Dict[str, float]:
        """Every panel metric from one dict of `name -> predicted strength`.

        `promoterness` is the discriminator logit, if the caller has one.
        Without it `spy_auc` is absent rather than approximated from the
        strength prediction: substituting one head for the other is how a
        number for a head that was never run gets into a table.
        """
        out: Dict[str, float] = {}
        p = dict(prediction)

        # 1. Ranking on the natives
        for tag, classes in (("spy", SCORED_NATIVE_CLASSES),
                             ("spy_foreign", FOREIGN_NATIVE_CLASSES)):
            recs = [r for r in self.by_class(*classes)
                    if r.name in p and np.isfinite(r.mu)]
            if len(recs) >= 4:
                a = np.array([p[r.name] for r in recs], dtype=np.float64)
                b = np.array([r.mu for r in recs], dtype=np.float64)
                out["%s_scc" % tag] = _spearman(a, b)
                out["%s_pcc" % tag] = _pearson(a, b)
                out["%s_n" % tag] = float(len(recs))

        # 2. The anchor ordering, which the whole project rests on
        pk, tf = self.by_class("kotwica_pks1"), self.by_class("kotwica_tef1")
        if pk and tf and pk[0].name in p and tf[0].name in p:
            out["anchor_gap"] = float(p[tf[0].name] - p[pk[0].name])
            # tef1 (mu 12.3542) is stronger than pks1 (mu 7.4235). A model
            # that inverts this cannot place a design between them, which is
            # four of the ten wet-lab slots.
            out["anchor_order_ok"] = float(out["anchor_gap"] > 0.0)

            # 2b. Are the anchors placed where they actually sit? Selection
            # never asked, and the model put pks1 at percentile 19-28 where
            # the data say 92.6. The score is accuracy and not height:
            # rewarding height would reward placing pks1 at 99.
            _nat = [r for r in self.by_class(*SCORED_NATIVE_CLASSES)
                    if r.name in p and np.isfinite(r.mu)]
            if len(_nat) >= 20:
                _pp = np.array([p[r.name] for r in _nat], dtype=np.float64)
                _tt = np.array([r.mu for r in _nat], dtype=np.float64)
                _error = 0.0
                for _w, _rec, _tag in ((ANCHOR_PCT_WEIGHTS["pks1"], pk[0],
                                        "pks1"),
                                       (ANCHOR_PCT_WEIGHTS["tef1"], tf[0],
                                        "tef1")):
                    _prz = float((_pp < p[_rec.name]).mean())
                    _praw = float((_tt < _rec.mu).mean())
                    # Both percentiles are over the same set, so comparable.
                    out["anchor_pct_pred_%s" % _tag] = _prz
                    out["anchor_pct_true_%s" % _tag] = _praw
                    out["anchor_pct_err_%s" % _tag] = abs(_prz - _praw)
                    _error += _w * abs(_prz - _praw)
                out["spy_anchor_pct"] = float(1.0 - _error)
                # Random floor, computed from these same percentiles rather
                # than chosen: E|U - t| = t^2/2 + (1-t)^2/2.
                _pod = 0.0
                for _w, _tag in ((ANCHOR_PCT_WEIGHTS["pks1"], "pks1"),
                                 (ANCHOR_PCT_WEIGHTS["tef1"], "tef1")):
                    _t = out["anchor_pct_true_%s" % _tag]
                    _pod += _w * (_t * _t / 2.0 + (1.0 - _t) ** 2 / 2.0)
                out["spy_anchor_pct_floor"] = float(1.0 - _pod)

        # 3. Motif awareness and edit resolution
        par = self.parent()
        if par is not None and par.name in p:
            base = float(p[par.name])
            for size, keep_cls, hit_cls in (("small", "mut_zachowane",
                                             "mut_zniszczone"),
                                            ("large", "mutL_zachowane",
                                             "mutL_zniszczone")):
                keep = [abs(p[r.name] - base) for r in self.by_class(keep_cls)
                        if r.name in p]
                hit = [abs(p[r.name] - base) for r in self.by_class(hit_cls)
                       if r.name in p]
                if keep:
                    out["shift_%s_outside" % size] = float(np.mean(keep))
                if hit:
                    out["shift_%s_in_motif" % size] = float(np.mean(hit))
                if keep and hit:
                    # Same substitution count, same parent, so the motif is
                    # the only difference. A ratio for readability and a rank
                    # statistic because that survives the tails of 20 samples.
                    out["motif_sensitivity_%s" % size] = float(
                        np.mean(hit) / max(np.mean(keep), 1e-9))
                    out["motif_auc_%s" % size] = mann_whitney_u(hit, keep)
            small = out.get("shift_small_outside")
            large = out.get("shift_large_outside")
            if small is not None and large is not None:
                # 2 substitutions against 88, both outside motifs. Below 1
                # the head is reading noise and the in-motif comparison
                # above means nothing.
                out["edit_resolution"] = float(large / max(small, 1e-9))
                out["edit_resolution_ok"] = float(large > small)
            out["edit_moves_prediction"] = float(
                (out.get("shift_small_in_motif", 0.0)
                 + out.get("shift_small_outside", 0.0)) > 0.0)

            # 3b. Motif contrast after matching on position. The raw
            # `in_motif - outside` contrast is confounded by position: mean
            # substitution position is 460.5 outside a motif and 306.0
            # inside, and corr(position, |dP|) with no motif involved is
            # +0.379 to +0.395. Matching each in-motif mutant to its K
            # positionally nearest out-of-motif mutants (mean distance
            # 37.3 bp) makes the effect vanish, -0.038 -> +0.006 +- 0.004.
            # A separate key rather than a replacement, because this is a
            # difference of |dP| and not an AUC.
            for size, keep_cls, hit_cls in (("small", "mut_zachowane",
                                             "mut_zniszczone"),
                                            ("large", "mutL_zachowane",
                                             "mutL_zniszczone")):
                out.update(_kontrast_dopasowany(
                    self, p, base, size, keep_cls, hit_cls))

        # 4. Promoterness
        src = promoterness if promoterness is not None else None
        if src is not None:
            # The anchors must themselves read as promoters: an epoch may be
            # chosen only if it calls both pks1 and tef1 promoters. Not a
            # formality -- the model has been measured scoring pks1 1.67
            # below its own shuffled version.
            #
            # The threshold is the composition-matched control, not a
            # constant: `losowa_euler4` keeps composition to 5-mers, so an
            # anchor that fails to beat it is a promoter compositionally
            # only. An absolute cut-off would be wrong because an AUC of
            # 0.916 does not transfer to a single cut point.
            _neg_e = sorted(float(src[r.name])
                            for r in self.by_class("losowa_euler4")
                            if r.name in src)
            if _neg_e:
                _mid = _neg_e[len(_neg_e) // 2] if len(_neg_e) % 2 else \
                    0.5 * (_neg_e[len(_neg_e) // 2 - 1]
                           + _neg_e[len(_neg_e) // 2])
                out["anchor_real_neg_med"] = float(_mid)
                _ile_ok = 0
                _maja = 0
                for _cls, _tag in (("kotwica_pks1", "pks1"),
                                   ("kotwica_tef1", "tef1")):
                    _g = self.by_class(_cls)
                    if _g and _g[0].name in src:
                        _maja += 1
                        _r = float(src[_g[0].name])
                        out["anchor_real_%s" % _tag] = _r
                        # The margin, not just the fact: a 0/1 flag cannot
                        # show whether an epoch passed by a hair or with
                        # room to spare.
                        out["anchor_real_margin_%s" % _tag] = float(_r - _mid)
                        if _r > _mid:
                            _ile_ok += 1
                if _maja == 2:
                    out["anchor_promoters_ok"] = float(_ile_ok == 2)
            pos = [float(src[r.name])
                   for r in self.by_class(*SCORED_NATIVE_CLASSES)
                   if r.name in src]
            for tag, classes in (("spy_auc", NEGATIVE_CLASSES),
                                 ("spy_auc_euler", ("losowa_euler4",)),
                                 ("spy_auc_coding", ("gen_kontrola",))):
                neg = [float(src[r.name]) for r in self.by_class(*classes)
                       if r.name in src]
                if pos and neg:
                    out[tag] = mann_whitney_u(pos, neg)
        return out

    # Reporting
    def report(self) -> Dict[str, object]:
        c = self.counts()
        scored = sum(c.get(k, 0) for k in SCORED_NATIVE_CLASSES)
        return {
            "source": os.path.basename(self.source),
            "n": len(self.records),
            "role": self.role,
            "counts": c,
            "n_with_measured_mu": int(sum(1 for r in self.records
                                          if np.isfinite(r.mu))),
            "n_scored_native": scored,
            "n_synthetic": int(sum(1 for r in self.records
                                   if r.n_substitutions > 0)),
            "excluded": dict(EXCLUDED_CLASSES),
            "unknown_classes": sorted(set(c) - set(SPY_CLASSES)),
            "notes": list(self.notes),
        }

    def format(self) -> str:
        r = self.report()
        lines = ["spy panel: %d sequences from %s (%d with a measured mu, "
                 "%d synthetic)"
                 % (r["n"], r["source"], r["n_with_measured_mu"],
                    r["n_synthetic"]),
                 "  role: %s" % (r["role"],),
                 "  classes: " + ", ".join(
                     "%s=%d" % kv for kv in sorted(r["counts"].items()))]
        if r["unknown_classes"]:
            lines.append("  UNKNOWN classes (not scored): %s"
                         % (", ".join(r["unknown_classes"]),))
        lines += ["  NOTE " + n for n in r["notes"]]
        return "\n".join(lines)


def load_spy_panel(path: str, drop_excluded: bool = True) -> SpyPanel:
    """Read `SPY_SET.fasta`. Contaminated classes are dropped by default."""
    recs: List[SpyRecord] = []
    notes: List[str] = []
    dropped = 0
    for header, seq in read_fasta(path):
        d = _parse_header(header)
        klasa = d.get("klasa", "")
        if drop_excluded and klasa in EXCLUDED_CLASSES:
            dropped += 1
            continue
        name = header.split()[0] if header else ""
        recs.append(SpyRecord(
            name=name, sequence=seq.upper(), klasa=klasa,
            n_substitutions=_int(d, "n_podstawien"),
            retention=_float(d, "retencja"),
            parent=d.get("rodzic", ""),
            genome=d.get("genom", "P1"),
            og_id=d.get("og", ""),
            motif=d.get("motyw", ""),
            positions=_positions(d),
            mu=_float(d, "mu"),
            quantile=_int(d, "kwantyl"),
            leak25=_float(d, "wyciek25"),
            raw=header))
    if dropped:
        notes.append("%d sequences dropped: %s"
                     % (dropped, "; ".join(EXCLUDED_CLASSES.values())))
    unknown = sorted({r.klasa for r in recs} - set(SPY_CLASSES))
    if unknown:
        notes.append("classes not in SPY_CLASSES are carried but NOT scored: "
                     + ", ".join(unknown))
    lens = {len(r.sequence) for r in recs}
    if len(lens) > 1:
        notes.append("sequence lengths are not uniform: %s -- the model takes "
                     "a fixed 800 bp, so the caller must pad or refuse"
                     % (sorted(lens),))
    return SpyPanel(records=recs, source=path, notes=notes)


# Statistics
def _srednia_pozycja(rec) -> float:
    """Mean substitution position of a record; NaN if the header gave none."""
    poz = getattr(rec, "positions", ()) or ()
    return float(np.mean(poz)) if len(poz) else float("nan")


def _kontrast_dopasowany(panel, p, base, size, keep_cls, hit_cls, k=5):
    """|dP| in a motif minus |dP| outside it, after matching on position.

    Each in-motif mutant is paired with the `k` out-of-motif mutants whose
    mean substitution position is nearest, and the difference is taken
    pairwise. Both numbers are reported, raw and matched: without the raw one
    it is invisible how much of the effect position ate, and without the SE
    it is invisible whether what remains is anything.
    """
    out = {}
    keep = [(r, _srednia_pozycja(r)) for r in panel.by_class(keep_cls)
            if r.name in p]
    hit = [(r, _srednia_pozycja(r)) for r in panel.by_class(hit_cls)
           if r.name in p]
    keep = [(r, x) for r, x in keep if np.isfinite(x)]
    hit = [(r, x) for r, x in hit if np.isfinite(x)]
    if not keep or not hit:
        return out
    kpos = np.asarray([x for _r, x in keep], dtype=np.float64)
    kval = np.asarray([abs(p[r.name] - base) for r, _x in keep],
                      dtype=np.float64)
    hval = np.asarray([abs(p[r.name] - base) for r, _x in hit],
                      dtype=np.float64)
    kk = int(min(int(k), kpos.size))
    roznice, odleglosci = [], []
    for (r, x) in hit:
        d = np.abs(kpos - x)
        idx = np.argsort(d)[:kk]
        roznice.append(abs(p[r.name] - base) - float(kval[idx].mean()))
        odleglosci.append(float(d[idx].mean()))
    a = np.asarray(roznice, dtype=np.float64)
    out["motif_contrast_%s" % size] = float(hval.mean() - kval.mean())
    out["motif_contrast_%s_matched" % size] = float(a.mean())
    out["motif_contrast_%s_matched_se" % size] = (
        float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1
        else float("nan"))
    out["motif_match_dist_%s" % size] = float(np.mean(odleglosci))
    out["motif_match_k_%s" % size] = float(kk)
    out["motif_pos_in_%s" % size] = float(np.mean([x for _r, x in hit]))
    out["motif_pos_out_%s" % size] = float(kpos.mean())
    return out


def mann_whitney_u(positive: Sequence[float], negative: Sequence[float]
                   ) -> float:
    """P(a random positive > a random negative), ties counted as one half.

    AUC computed from ranks, written out rather than imported because the tie
    rule is the whole point: an untrained model in eval returns a constant
    (measured sd 3.19e-08), and with ties broken by position a constant scorer
    gets 1.0 or 0.0 depending on the order. At one half it gets exactly 0.5.
    """
    a = np.asarray(positive, dtype=np.float64).ravel()
    b = np.asarray(negative, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    s = allv[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    ra = ranks[:a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(x.size, dtype=np.float64)
    s = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and s[j + 1] == s[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_ranks(a), _ranks(b))
