"""The one place where an edit made in the latent becomes surgical in DNA."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

R = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(R, "src", "panel"), os.path.join(R, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import panel_spec as ZP  # noqa: E402
import scoring  # noqa: E402

ZASADY = "".join(scoring.BASES)
assert sorted(ZASADY) == sorted(set(ZASADY)) == list("ACGT"), ZASADY


GEOM_DLUGOSCI = (25, 50, 100, 200, 400)
MARGINES_KOMOREK = 1
ROZLEW_POZA_SLADEM = {3: 0.000, 4: 0.075}

Przedzial = Tuple[int, int]


class NotSurgical(AssertionError):
    """The candidate differs from the original outside its own trace."""


def geometry_from_codes(indeksy: Sequence[torch.Tensor]) -> Tuple[int, ...]:
    """Cells per level, read off the actual code tensors."""
    return tuple(int(t.shape[-1]) for t in indeksy)


def check_geometry(indeksy: Sequence[torch.Tensor]) -> Tuple[int, ...]:
    """Raise unless the real codes match GEOM_DLUGOSCI.

    Called wherever codes are available, so the constant is never what decides
    how many bp a cell is worth.
    """
    g = geometry_from_codes(indeksy)
    if g != tuple(GEOM_DLUGOSCI):
        raise ValueError(
            "ladder geometry %s does not match GEOM_DLUGOSCI %s -- the bp per "
            "cell that every trace is built from would be wrong. Pass "
            "n_komorek explicitly, or fix the constant."
            % (g, tuple(GEOM_DLUGOSCI)))
    return g


def bp_na_komorke(poziom: int, dlugosc: int,
                  n_komorek: Optional[int] = None) -> float:
    n = int(GEOM_DLUGOSCI[poziom] if n_komorek is None else n_komorek)
    if n <= 0:
        raise ValueError("level %d has %d cells" % (poziom, n))
    return float(dlugosc) / float(n)


def slad(poziom: int, komorki: Iterable[int], dlugosc: int,
         n_komorek: Optional[int] = None,
         margines: int = MARGINES_KOMOREK) -> Optional[Przedzial]:
    """Footprint of one level's changed cells, in bp: (lo, hi), hi open.

    `komorki` may be scattered and the returned span encloses all of them,
    which is the conservative direction but for a scattered edit can cover
    most of the sequence; `slad_z_kodow` returns the tight disjoint intervals
    instead. None for an empty cell set, and `sklej` then reproduces the
    original bit for bit.
    """
    k = sorted({int(c) for c in np.asarray(list(komorki), dtype=np.int64).reshape(-1)})
    if not k:
        return None
    n = int(GEOM_DLUGOSCI[poziom] if n_komorek is None else n_komorek)
    if k[0] < 0 or k[-1] >= n:
        raise IndexError("cells %s outside level %d with %d cells"
                         % ((k[0], k[-1]), poziom, n))
    bp = bp_na_komorke(poziom, dlugosc, n)
    lo = int(math.floor((k[0] - margines) * bp))
    hi = int(math.ceil((k[-1] + 1 + margines) * bp))
    return (max(0, lo), min(int(dlugosc), hi))


def komorki_zmienione(idx_przed: Sequence[torch.Tensor],
                      idx_po: Sequence[torch.Tensor],
                      wiersz: int = 0) -> Dict[int, np.ndarray]:
    """Per level, which cells the edit actually moved. Measured against the
    codes before the edit, never against what the caller meant to mask -- a
    masked cell resampled to its own old symbol moved nothing and must not
    widen the trace."""
    if len(idx_przed) != len(idx_po):
        raise ValueError("different number of levels: %d vs %d"
                         % (len(idx_przed), len(idx_po)))
    out = {}
    for lv, (a, b) in enumerate(zip(idx_przed, idx_po)):
        aa = a[wiersz].reshape(-1).cpu().numpy()
        bb = b[wiersz].reshape(-1).cpu().numpy()
        if aa.shape != bb.shape:
            raise ValueError("level %d shape %s vs %s" % (lv, aa.shape, bb.shape))
        out[lv] = np.flatnonzero(aa != bb).astype(np.int64)
    return out


def scal(przedzialy: Iterable[Przedzial]) -> List[Przedzial]:
    """Sorted, merged, half-open intervals. Touching intervals merge, because
    [0,10) and [10,20) describe one continuous stretch of held-out bases."""
    xs = sorted((int(a), int(b)) for a, b in przedzialy if b > a)
    out: List[Przedzial] = []
    for a, b in xs:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def slad_z_kodow(idx_przed: Sequence[torch.Tensor],
                 idx_po: Sequence[torch.Tensor], dlugosc: int,
                 margines: int = MARGINES_KOMOREK,
                 wiersz: int = 0) -> List[Przedzial]:
    """The tight trace of a multi-level edit: one +-margin interval per moved
    cell, merged. Scattered edits need this, because the enclosing span of
    scattered cells is the whole sequence."""
    zm = komorki_zmienione(idx_przed, idx_po, wiersz=wiersz)
    n_kom = geometry_from_codes(idx_przed)
    prz: List[Przedzial] = []
    for lv, cells in zm.items():
        if cells.size == 0:
            continue
        bp = bp_na_komorke(lv, dlugosc, n_kom[lv])
        for c in cells.tolist():
            lo = int(math.floor((c - margines) * bp))
            hi = int(math.ceil((c + 1 + margines) * bp))
            prz.append((max(0, lo), min(int(dlugosc), hi)))
    return scal(prz)


def _przedzialy(slad_arg) -> List[Przedzial]:
    """Accept None, one (lo, hi), or an iterable of them. One normaliser, so
    `sklej`, the guard and the report cannot disagree about what the trace was
    -- a guard reading a different trace than the splice used would pass
    vacuously."""
    if slad_arg is None:
        return []
    if isinstance(slad_arg, tuple) and len(slad_arg) == 2 \
            and all(isinstance(v, (int, np.integer)) for v in slad_arg):
        return scal([(int(slad_arg[0]), int(slad_arg[1]))])
    if isinstance(slad_arg, (list, tuple)):
        if len(slad_arg) == 0:
            return []
        if all(isinstance(v, (int, np.integer)) for v in slad_arg) \
                and len(slad_arg) == 2:
            return scal([(int(slad_arg[0]), int(slad_arg[1]))])
        return scal([(int(a), int(b)) for a, b in slad_arg])
    raise TypeError("trace must be None, (lo, hi) or a list of them, not %r"
                    % (type(slad_arg),))


def maska_sladu(slad_arg, dlugosc: int) -> np.ndarray:
    """Boolean [L]: True inside the trace."""
    m = np.zeros(int(dlugosc), dtype=bool)
    for a, b in _przedzialy(slad_arg):
        if a < 0 or b > dlugosc:
            raise IndexError("interval (%d, %d) outside 0..%d" % (a, b, dlugosc))
        m[a:b] = True
    return m


def _bajty(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8)


def sklej(oryginal: str, zdekodowane: str, slad_arg) -> str:
    """Original bases outside the trace, decoded bases inside it.

    This is what makes the edit surgical by construction rather than by hope:
    afterwards every base that differs from the original lies inside the
    trace, and `surgical_guard` proves it. An empty trace returns the original
    bit for bit and the full trace returns the decode unchanged; both are
    negative controls in the smoke test.
    """
    if len(oryginal) != len(zdekodowane):
        raise ValueError("lengths differ: %d vs %d"
                         % (len(oryginal), len(zdekodowane)))
    m = maska_sladu(slad_arg, len(oryginal))
    a = _bajty(oryginal).copy()
    b = _bajty(zdekodowane)
    a[m] = b[m]
    return a.tobytes().decode()


def roznice(a: str, b: str) -> np.ndarray:
    return np.flatnonzero(_bajty(a) != _bajty(b))


def surgical_guard(oryginal: str, kandydat: str, slad_arg,
                          name: str = "candidate") -> int:
    """Raise unless the candidate equals the original outside the trace.

    The test that the word "surgical" is earned. Returns the number of changed
    bases inside the trace, so a caller that wants the number does not have to
    recompute it with a second, slightly different comparison.
    """
    if len(oryginal) != len(kandydat):
        raise NotSurgical("%s has %d bp, original has %d"
                              % (name, len(kandydat), len(oryginal)))
    m = maska_sladu(slad_arg, len(oryginal))
    d = roznice(oryginal, kandydat)
    poza = d[~m[d]]
    if poza.size:
        raise NotSurgical(
            "%s changes %d bases OUTSIDE the trace %s (positions %s%s) -- the "
            "edit is NOT surgical. Either the trace is too narrow, or the "
            "sequence did not go through sklej()."
            % (name, int(poza.size), _przedzialy(slad_arg),
               list(poza[:20]), " ..." if poza.size > 20 else ""))
    return int(d.size)


def konstrukty(scorer) -> Tuple[str, ...]:
    return tuple(sorted(scorer.contexts))


def anchor_units(scorer, seqs: Sequence[str], kategoria: Optional[str] = None,
                     batch: int = 32) -> np.ndarray:
    """Anchor units: 0 = pks1, 1 = tef1, combined over constructs by minimum.

    Minimum and not mean, because a design must work in both constructs and is
    limited by its worse one; `panel_spec.LACZENIE` is asserted here so the
    two cannot drift apart. Anchors are re-measured on this scorer, with its
    own contexts and category; no stored anchor value is used anywhere.
    """
    assert ZP.LACZENIE == "min", ZP.LACZENIE
    kat = ZP.KATEGORIA if kategoria is None else kategoria
    kidx = scorer.category_index(kat)
    per = []
    for k in konstrukty(scorer):
        a = scorer.anchor_scores(k)
        mu_p, mu_t = a["pks1"].mu_of(kat), a["tef1"].mu_of(kat)
        if not mu_t > mu_p:
            raise ValueError("construct %s: tef1 %.4f is not above pks1 %.4f, "
                             "anchor units would be inverted" % (k, mu_t, mu_p))
        mu = scorer.raw(list(seqs), k, batch=batch)["mu"][:, kidx]
        per.append((np.asarray(mu, dtype=np.float64) - mu_p) / (mu_t - mu_p))
    return np.min(np.stack(per, axis=0), axis=0)


def okno_celu(j: float) -> str:
    """Which target window a value in anchor units falls in."""
    if j >= ZP.OKNO_MAX[0]:
        return "MAX"
    if ZP.OKNO_MID[0] <= j <= ZP.OKNO_MID[1]:
        return "MID"
    return "ponizej_MID" if j < ZP.OKNO_MID[0] else "miedzy"


@torch.no_grad()
def decode_codes(model, indeksy: Sequence[torch.Tensor]) -> List[str]:
    """Codes -> DNA through `decode_to_onehot`, the verified contract.

    `decode_to_onehot(encode_codes(x))` agrees with the model's own
    reconstruction (`recon_logits` view 0) on 1.000000 of 800 positions. Use
    it rather than `recon_logits`, which is a stack of seven masked views,
    shape (7, 4, 800); view 0 is the full one.
    """
    oh = model.decode_to_onehot([t for t in indeksy])
    if oh.dim() != 3 or int(oh.shape[1]) != len(ZASADY):
        raise ValueError(
            "decode_to_onehot returned %s; expected [B, %d, L]. If this is a "
            "stack of masked views, take view 0." % (tuple(oh.shape), len(ZASADY)))
    arr = oh.argmax(dim=1).cpu().numpy()
    return ["".join(ZASADY[int(i)] for i in row) for row in arr]


@dataclass
class RoundTrip:
    """decode(encode(original)) and everything the report needs from it.

    An edit's effect has to be quotable against two baselines and they are
    different numbers: `cel_oryginal` is the real molecule (0.0 when the
    original is pks1, by definition of anchor units), `cel` is what the
    decoder alone costs. Reporting one as the other is the mistake this
    dataclass exists to prevent.
    """
    oryginal: str
    dna: str
    indeksy: List[torch.Tensor]
    cel_oryginal: float
    cel: float
    zle_zasady: int
    pozycje: np.ndarray
    kategoria: str
    konstrukty: Tuple[str, ...]
    anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def dlugosc(self) -> int:
        return len(self.oryginal)

    def description(self) -> str:
        p = self.pozycje
        zak = ("%d..%d, median gap %.1f bp"
               % (p.min(), p.max(), float(np.median(np.diff(p))))
               if p.size > 1 else ("%d" % p[0] if p.size else "none"))
        return ("round-trip: %d wrong bases out of %d (%.2f %%), "
                "positions %s\n"
                "  objective of the original %+.5f | objective of the "
                "round trip %+.5f  "
                "(anchor units, category %r, combined by min over %s)"
                % (self.zle_zasady, self.dlugosc,
                   100.0 * self.zle_zasady / self.dlugosc, zak,
                   self.cel_oryginal, self.cel, self.kategoria,
                   "+".join(self.konstrukty)))


def reference(model, scorer, oryginal: str, kategoria: Optional[str] = None,
                batch: int = 32, sprawdz_model: bool = True) -> RoundTrip:
    """The reference point for every edit of `oryginal`.

    Computed once and passed around: it costs a forward pass per construct,
    and recomputing it per candidate would also make the baseline depend on
    batch composition at the 1e-6 mu level (`scoring.CONTEXT_TOLERANCE`).
    """
    kat = ZP.KATEGORIA if kategoria is None else kategoria
    if sprawdz_model and os.environ.get("HYPPE_CHECKPOINT") is None:
        sha = getattr(scorer.prov, "checkpoint_sha256", "")
        if sha != ZP.CHECKPOINT_SHA256:
            raise ValueError(
                "scorer carries checkpoint sha %s, the competition model is "
                "%s. Set HYPPE_CHECKPOINT to say the substitution is "
                "deliberate." % (sha[:16], ZP.CHECKPOINT_SHA256[:16]))
    x = scoring.one_hot([oryginal]).to(scorer.device)
    with torch.no_grad():
        idx = [t.clone() for t in model.encode_codes(x)]
    check_geometry(idx)
    dna = decode_codes(model, [t.clone() for t in idx])[0]
    j = anchor_units(scorer, [oryginal, dna], kategoria=kat, batch=batch)
    kot = {}
    for k in konstrukty(scorer):
        a = scorer.anchor_scores(k)
        kot[k] = (a["pks1"].mu_of(kat), a["tef1"].mu_of(kat))
    return RoundTrip(oryginal=oryginal, dna=dna, indeksy=idx,
                     cel_oryginal=float(j[0]), cel=float(j[1]),
                     zle_zasady=int(roznice(oryginal, dna).size),
                     pozycje=roznice(oryginal, dna),
                     kategoria=kat, konstrukty=konstrukty(scorer), anchors=kot)


KOLUMNY = ("zmienione_zasady_w_oknie", "zmienione_zasady_poza_oknem",
           "cel_surowo", "cel_sklejone", "cel_round_trip",
           "efekt_edycji_bezwzgledny", "efekt_edycji_wzgledny",
           "przezycie_rejonu")


def raporty(scorer, odn: RoundTrip, kandydaci: Sequence[Tuple[str, object]],
            batch: int = 32, kategoria: Optional[str] = None
            ) -> List[Dict[str, object]]:
    """Report a batch of (raw decode, trace) pairs.

    Batched on purpose: raw and spliced are scored in one call each, so every
    candidate sits in the same batch composition and the 1e-6 mu batch effect
    is common to all of them. Each report splices and then runs the guard on
    the splice, reading the same normalised trace `sklej` used, so the guard
    cannot pass vacuously.
    """
    kat = odn.kategoria if kategoria is None else kategoria
    L = odn.dlugosc
    surowe, sklejone, slady, maski = [], [], [], []
    for dna, sl in kandydaci:
        if len(dna) != L:
            raise ValueError("candidate has %d bp, original %d" % (len(dna), L))
        prz = _przedzialy(sl)
        skl = sklej(odn.oryginal, dna, prz)
        surgical_guard(odn.oryginal, skl, prz, name="spliced candidate")
        surowe.append(dna)
        sklejone.append(skl)
        slady.append(prz)
        maski.append(maska_sladu(prz, L))

    j_sur = anchor_units(scorer, surowe, kategoria=kat, batch=batch)
    j_skl = anchor_units(scorer, sklejone, kategoria=kat, batch=batch)

    out: List[Dict[str, object]] = []
    for i, (dna, skl, prz, m) in enumerate(zip(surowe, sklejone, slady, maski)):
        d_skl = roznice(odn.oryginal, skl)
        d_sur = roznice(odn.oryginal, dna)
        poza_n = int((~m).sum())
        przez = (float((_bajty(odn.oryginal)[~m] == _bajty(dna)[~m]).mean())
                 if poza_n else float("nan"))
        erased = int((~m[roznice(odn.dna, dna)]).sum()) if len(odn.dna) == L else -1
        rec = {
            "zmienione_zasady_w_oknie": int(d_skl.size),
            "zmienione_zasady_poza_oknem": int((~m[d_skl]).sum()),
            "cel_surowo": float(j_sur[i]),
            "cel_sklejone": float(j_skl[i]),
            "cel_round_trip": float(odn.cel),
            "efekt_edycji_bezwzgledny": float(j_skl[i] - odn.cel_oryginal),
            "efekt_edycji_wzgledny": float(j_sur[i] - odn.cel),
            "przezycie_rejonu": przez,
            "zmienione_zasady_surowo": int(d_sur.size),
            "zasady_skasowane_przez_sklejenie": erased,
            "slad_bp": int(m.sum()),
            "slad_przedzialy": list(prz),
            "okno_sklejone": okno_celu(float(j_skl[i])),
            "kategoria": kat,
            "dna_sklejone": skl,
        }
        assert rec["zmienione_zasady_poza_oknem"] == 0, rec
        out.append(rec)
    return out


def raport(scorer, odn: RoundTrip, dna_surowe: str, slad_arg,
           batch: int = 32, kategoria: Optional[str] = None) -> Dict[str, object]:
    """One candidate. A thin wrapper over `raporty`, so there is one path."""
    return raporty(scorer, odn, [(dna_surowe, slad_arg)], batch=batch,
                   kategoria=kategoria)[0]


def wiersz_tsv(rec: Dict[str, object], kolumny: Sequence[str] = KOLUMNY) -> str:
    def f(v):
        if isinstance(v, float):
            return "%.5f" % v
        if isinstance(v, (list, tuple)):
            return ";".join("%d-%d" % (a, b) for a, b in v)
        return str(v)
    return "\t".join(f(rec[k]) for k in kolumny)


def naglowek_tsv(kolumny: Sequence[str] = KOLUMNY) -> str:
    return "\t".join(kolumny)


def module_description() -> str:
    return ("surgery: alphabet %r (from scoring.BASES), margin %d cell(s), "
            "geometry %s, spill beyond the trace L3 %.1f %% / L4 %.1f %%"
            % (ZASADY, MARGINES_KOMOREK, list(GEOM_DLUGOSCI),
               100 * ROZLEW_POZA_SLADEM[3], 100 * ROZLEW_POZA_SLADEM[4]))


if __name__ == "__main__":
    print(module_description())
    print("report columns: %s" % (list(KOLUMNY),))
