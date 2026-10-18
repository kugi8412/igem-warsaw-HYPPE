"""The panel specification: one source of truth for the ten sequences."""
from __future__ import annotations

DESC_pAN7_1 = (
    "Trichoderma atroviride P1 protoplasts transformed with pAN7-1, carrying a "
    "synthetic promoter integrated at the pks1 locus; potato dextrose agar; "
    "hygromycin B selection; no cellulose discs; cells grown at 28C in darkness"
)

DESC_pHYPPE = (
    "Trichoderma atroviride P1 protoplasts transformed with pHYPPE, a plasmid "
    "integrating at a random genomic locus, carrying a synthetic promoter "
    "driving EGFP and the constitutive tef1 promoter driving mCherry as an "
    "internal reference; potato dextrose broth in 96-well plates; geneticin "
    "(G418) selection; no cellulose discs; cells grown at 28C in darkness"
)

KONSTRUKTY = {
    "pAN7-1": {
        "description": DESC_pAN7_1,
        "flanki": "pAN7-1",
        "kontekst": "native genomic locus of pks1",
    },
    "pHYPPE": {
        "description": DESC_pHYPPE,
        "flanki": "pHYPPE",
        "kontekst": "random genomic integration, EGFP reporter with tef1-mCherry internal reference",
    },
}

MASKGIT_EDIT_LEVELS = (3, 4)
MASKGIT_FORBIDDEN_LEVELS = (0, 1, 2)

PANEL = [
    # (id, arm, category, variants)
    ("2", "surgery", "multi-objective gradient on a relaxed sequence", ("MID", "MAX")),
    ("3", "surgery", "genetic algorithm, minimal edits in base space", ("MID", "MAX")),
    ("4", "surgery", "MaskGIT resampling of fine levels only (L3, L4)", ("MID", "MAX")),
    ("1", "de_novo", "de novo from the model, a single unguided draw", ("MID", "MAX")),
    ("5", "de_novo", "discrete diffusion with an absorbing state", ("MID", "MAX")),
]

N_DE_NOVO = sum(len(v) for _i, r, _k, v in PANEL if r == "de_novo")
N_SURGICAL = sum(len(v) for _i, r, _k, v in PANEL if r == "surgery")
assert N_SURGICAL == 6, N_SURGICAL
assert N_DE_NOVO == 4, N_DE_NOVO
assert N_DE_NOVO + N_SURGICAL == 10

MARGINES_MID = 0.15
LACZENIE = "min"
NACHYLENIE_KALIBRACJI = (0.8153, 0.8414)


def description(konstrukt: str) -> str:
    """The experiment description for a construct, or raise.

    No default: a silent fallback to the unknown text token would hide the
    design condition, and the channel is live.
    """
    if konstrukt not in KONSTRUKTY:
        raise KeyError("unknown construct %r; known: %s"
                       % (konstrukt, ", ".join(sorted(KONSTRUKTY))))
    return KONSTRUKTY[konstrukt]["description"]


def podsumowanie() -> str:
    out = ["PANEL: %d surgical edits + %d de novo = %d"
           % (N_SURGICAL, N_DE_NOVO,
              N_DE_NOVO + N_SURGICAL)]
    for i, ramie, kat, war in PANEL:
        out.append("  %s  %-8s %-48s %s" % (i, ramie, kat, "+".join(war)))
    out.append("constructs: %s" % ", ".join(sorted(KONSTRUKTY)))
    out.append("objective combines constructs by: %s" % LACZENIE)
    return "\n".join(out)


import os as _os

CHECKPOINT = _os.environ.get(
    "HYPPE_CHECKPOINT", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "runs", "hyppe_foundational.pt"))
CHECKPOINT_POPRZEDNI = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "runs", "hyppe_foundational_previous.pt")
CHECKPOINT_SHA256_POPRZEDNI = (
    "cf17b79b3be372d01aadf833166113cdd0c65ad7d802ff132693352eae3089f4")
CHECKPOINT_SHA256 = (
    "c484dff676e55c63b30cae84438dc0c7ab9aac1b4a1818a80ddf79791b473b45")
EPOKA_POPRZEDNIA = 10
EPOKA = 7
ZESPOL_EPOK = (EPOKA,)
MODEL_JEDEN = True
EPOKI_NIEPEWNOSCI_POPRZEDNIE = (5, 6)
EPOKI_NIEPEWNOSCI = ()
ZESPOL_SCIEZKI = (CHECKPOINT,)


def uncertainty_description() -> str:
    """What the uncertainty column means, or that there is none.

    `EPOKI_NIEPEWNOSCI` is empty for this release. Formatting the empty
    tuple produced "over epochs " with nothing after it, which reads as a
    missing value rather than as a deliberate absence.
    """
    if not EPOKI_NIEPEWNOSCI:
        return ("no prediction-uncertainty ensemble for this release: the "
                "panel is scored by ep%03d alone (an earlier panel averaged "
                "%s)"
                % (EPOKA,
                   ", ".join("ep%03d" % e
                             for e in EPOKI_NIEPEWNOSCI_POPRZEDNIE)))
    return ("prediction uncertainty computed over epochs %s (it is NOT an "
            "objective and NOT a criterion; the model that goes to the "
            "WetLab is ep%03d)"
            % (", ".join("ep%03d" % e for e in EPOKI_NIEPEWNOSCI), EPOKA))

TEST_SCC = 0.68051
TEST_SCC_SE = 0.01749


def checkpoint_description() -> str:
    return ("competition model: %s\n  epoch %d, sha256 %s\n  scoring "
            "ensemble: %s"
            % (CHECKPOINT, EPOKA, CHECKPOINT_SHA256[:16],
               ", ".join("ep%03d" % e for e in ZESPOL_EPOK)))


KATEGORIA = "control"
KATEGORIE_CIEKAWOSTKA = (
    "carbon source", "biotic antagonism", "light response",
    "plant association", "mechanical injury", "other",
)
ANCHOR_GAP = {
    "control": 7.560, "carbon source": 7.348, "biotic antagonism": 7.862,
    "light response": 7.585, "plant association": 8.132,
    "mechanical injury": 7.746, "other": 6.996,
}
ROZKLAD_KORPUSU_POPRZEDNI = {
    5: -0.0407, 25: 0.1572, 50: 0.2815, 75: 0.4135,
    90: 0.6013, 95: 0.7600, 99: 1.0387,
}
ROZKLAD_KORPUSU = {
    5: -0.1490, 25: -0.0371, 50: 0.1454, 75: 0.3112, 90: 0.4501, 95: 0.5771, 99: 0.9368
}
PERCENTYL_PKS1 = 30.1
PERCENTYL_PKS1_LACZONA = 29.3
PERCENTYL_TEF1 = 99.0
PERCENTYL_TEF1_JEST_PROGIEM = True
OKNO_MAX = (ROZKLAD_KORPUSU[90], float("inf"))     # >= +0.4501 (p90 HYPPE6)
OKNO_MID = (MARGINES_MID, 0.50)                    # anchor units
ODNIESIENIE_TEF1 = 1.0
SUFIT_EDYCJI_PUNKTOWEJ = 0.6393
SUFIT_EDYCJI_PUNKTOWEJ_NIEAKTUALNY = True
SUFIT_EDYCJI_PUNKTOWEJ_MODEL = "previous (before HYPPE6)"


def category_description() -> str:
    return ("design category: %r (anchor gap %.3f mu on the limiting "
            "construct)\n  MID window %s, MAX window %s in anchor units"
            "\n  pks1 = percentile %.1f of the corpus (%.1f under the "
            "combined score), tef1 >= %.0f"
            "\n  NOT 'the bottom 10 %% of the genome' -- that was the "
            "previous model's number (9.6)"
            "\n  curiosity, computed and reported, never optimised: %s"
            % (KATEGORIA, ANCHOR_GAP[KATEGORIA], OKNO_MID, OKNO_MAX,
               PERCENTYL_PKS1, PERCENTYL_PKS1_LACZONA, PERCENTYL_TEF1,
               ", ".join(KATEGORIE_CIEKAWOSTKA)))

ATG_OGON_NT = 10
ATG_OGON_TWARDY = True
ATG_OGON_EDYTOWALNY = False
CHIR_BUDZET_MID = 8
CHIR_BUDZET_MAX = 40
CHIR_BUDZET_ZASAD = CHIR_BUDZET_MAX
SUFIT_PRZY_BUDZECIE = {8: 0.2606, 40: 0.4806}
SUFIT_PRZY_BUDZECIE_NIEAKTUALNY = True
CHIR_BUDZET_TWARDY = True
CHIR_OKNO_MID = (ROZKLAD_KORPUSU[25], ROZKLAD_KORPUSU[50])    # -0.037 .. 0.145
CHIR_OKNO_MAX = (ROZKLAD_KORPUSU[50], float("inf"))           # >= 0.1454
DENOVO_OKNO_MID = (0.15, 0.50)
DENOVO_OKNO_MAX = (ROZKLAD_KORPUSU[90], float("inf"))         # >= 0.4501
RAMIE_METODY = {
    "01_de_novo": "de_novo",
    "02_gradient": "surgery",
    "03_genetyczny": "surgery",
    "04_maskgit": "surgery",
    "05_dyfuzja": "de_novo",
}
SUFIT_EKSTRAPOLACJI = ROZKLAD_KORPUSU[99]                     # 0.9368


def okna(metoda: str):
    """(MID, MAX, budget) for a method.

    Raises on an unknown method rather than defaulting: a method judged by the
    wrong arm's rule is the failure this file exists to prevent.
    """
    ramie = RAMIE_METODY.get(metoda)
    if ramie is None:
        raise KeyError("unknown method %r; known: %s"
                       % (metoda, ", ".join(sorted(RAMIE_METODY))))
    if ramie == "surgery":
        return CHIR_OKNO_MID, CHIR_OKNO_MAX, CHIR_BUDZET_ZASAD
    return DENOVO_OKNO_MID, DENOVO_OKNO_MAX, None


def arm_description() -> str:
    return (
        "SURGICAL ARM (02, 03, 04): budget %d bases out of 800, HARD.\n"
        "  MID %s = corpus p25..p50 | MAX %s = from p50 upwards\n"
        "  greedy ceiling 0.3360: OUT OF DATE, measured on the previous "
        "model\n"
        "DE NOVO ARM (01, 05): no budget.\n"
        "  MID %s | MAX %s = corpus p90\n"
        "  extrapolation ceiling %.4f (p99) -- above it, only report"
        % (CHIR_BUDZET_ZASAD, CHIR_OKNO_MID, CHIR_OKNO_MAX,
           DENOVO_OKNO_MID, DENOVO_OKNO_MAX, SUFIT_EKSTRAPOLACJI))


def budzet_dla(metoda, wariant):
    """Edit budget for one (method, variant). None for the de novo arm.

    Raises on an unknown variant rather than defaulting to the larger budget:
    a MID candidate judged by the MAX budget would be a 40-base edit filed as
    a 1 % touch, which is the confusion the budgets exist to prevent.
    """
    ramie = RAMIE_METODY.get(metoda)
    if ramie is None:
        raise KeyError("unknown method %r" % (metoda,))
    if ramie != "surgery":
        return None
    w = str(wariant).upper()
    if w == "MID":
        return CHIR_BUDZET_MID
    if w == "MAX":
        return CHIR_BUDZET_MAX
    raise KeyError("unknown variant %r; expected MID or MAX" % (wariant,))


def budget_description():
    linie = [
        "SURGICAL ARM -- the budget DEPENDS ON THE VARIANT:",
        "  MID <= %d bases (%.1f %% of 800), window %s = p25..p50, ceiling "
        "%.4f (OUT OF DATE)"
        % (CHIR_BUDZET_MID, 100.0 * CHIR_BUDZET_MID / 800.0,
           CHIR_OKNO_MID, SUFIT_PRZY_BUDZECIE[8]),
        "  MAX <= %d bases (%.1f %% of 800), window >= %.4f = p50, ceiling "
        "%.4f (OUT OF DATE)"
        % (CHIR_BUDZET_MAX, 100.0 * CHIR_BUDZET_MAX / 800.0,
           CHIR_OKNO_MAX[0], SUFIT_PRZY_BUDZECIE[40]),
        "  NOTE: the MID ceiling (%.4f, previous model) LIES ABOVE the MAX "
        "threshold (%.4f, HYPPE6)"
        % (SUFIT_PRZY_BUDZECIE[8], CHIR_OKNO_MAX[0]),
        "  -- the separation of the windows is NO LONGER arithmetic; it "
        "was, under the previous distribution",
        "DE NOVO ARM -- no budget:",
        "  MID %s | MAX >= %.4f = corpus p90 | extrapolation ceiling %.4f"
        % (DENOVO_OKNO_MID, DENOVO_OKNO_MAX[0], SUFIT_EKSTRAPOLACJI),
    ]
    return "\n".join(linie)


if __name__ == "__main__":
    print(podsumowanie())
    for k in sorted(KONSTRUKTY):
        print()
        print(k)
        print("  " + description(k))
