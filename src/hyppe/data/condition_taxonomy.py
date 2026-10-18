"""Two-level, multi-label taxonomy of experimental conditions.

`RULES` is the regex rule set, i.e. the definition. It is applied once by
`src/scripts/build_condition_taxonomy.py`, which writes
`data/conditions/condition_taxonomy.tsv`; `load_taxonomy()` reads that table
and every consumer reads it and nothing else. A missing table raises rather
than falling back to re-running the regexes, because a table regenerated
behind the caller's back would differ from the one the falsifier was run
against.

Four measurements shaped the rules. Matching is on `condition` alone, never
on study or experiment titles: one study's title carries "red" and "far-red"
on all 21 of its samples where only 9 are red, 9 far-red and 3 darkness
controls. The light axis is declared across all parents, because darkness is
filed under `control` on 139 of its 148 samples while the lit arm sits under
`light response`, so an axis confined to one parent cannot see the strongest
light contrast in the corpus. The subcategory is `darkness` and not
`dark_recovery`, since nothing in this corpus matches /recover/ and the name
must not claim a time-course that was never run. And "14h photoperiod" is
growth boilerplate rather than a light treatment, which is the main precision
decision on that axis.

The encoding is a parent embedding plus zero-initialised per-subcategory
offsets, with one `axis_stated` bit per axis, rather than a flat one-hot. A
flat class is not available: 371 of 2332 samples carry two or three
subcategories at once. Axes are orthogonal to parents, so subcategories are
not nested under one -- `light_darkness` spans three parents -- and an added
offset can express that where a selected child cannot. Zero initialisation
makes an unseen or frozen subcategory exactly its parent, and the
`axis_stated` bit is what separates "no regime stated" from "not that kind of
experiment" for the 1016 samples carrying no subcategory at all.

The budget for a subcategory is its number of independent studies, not its
number of samples: one confined to a single study is the same variable as
that study's batch. `support()` recomputes this and sets `offset_policy` to
`learn` at three or more studies, `shrink` at two, and `freeze` at one, where
the offset is held at zero and the model is told only the parent. That rule
is the most important line in this file, because the permutation falsifier
cannot catch the case: a perfectly batch-driven group separates beautifully,
and blue against red is p = 0.0009 while being 100 % study-disjoint.

Field values copied from the source metadata keep their original spelling.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Rule",
    "RULES",
    "AXES",
    "PARENT_CATEGORIES",
    "SampleLabels",
    "TAXONOMY_PATH",
    "load_taxonomy",
    "apply_rules",
    "subcategory_matrix",
    "axis_stated_matrix",
    "Support",
    "support",
]

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
TAXONOMY_PATH = os.path.join(_REPO, "data", "conditions", "condition_taxonomy.tsv")

#: The seven parent categories, exactly as they appear in
#: og_scope/samples_metadata.tsv column `condition_category`.
PARENT_CATEGORIES: Tuple[str, ...] = (
    "control",
    "carbon source",
    "other",
    "biotic antagonism",
    "light response",
    "mechanical injury",
    "plant association",
)


# Rule definition
@dataclass(frozen=True)
class Rule:
    """One subcategory rule.

    `axis` names the axis the subcategory belongs to: subcategories on the
    same axis are alternatives, those on different axes are meant to co-occur.
    `pattern` is a case-insensitive regex matched against `condition` only,
    and `parent_scope` lists the parents it may fire in, `None` meaning any.
    `requires` and `excludes` name subcategories that must or must not have
    matched first, which is what keeps `blue_pulse` a refinement of `blue` and
    stops the generic mycoparasitism fallback firing when a taxon was named.
    `note` says why the rule is written the way it is; read it before widening
    one.
    """

    axis: str
    subcategory: str
    pattern: str
    parent_scope: Optional[Tuple[str, ...]] = None
    requires: Tuple[str, ...] = ()
    excludes: Tuple[str, ...] = ()
    note: str = ""

    def regex(self) -> "re.Pattern[str]":
        return re.compile(self.pattern, re.IGNORECASE)


_BIOTIC = ("biotic antagonism",)

RULES: Tuple[Rule, ...] = (
    # Light. Scope is all parents on purpose.
    Rule(
        axis="light_regime",
        subcategory="light_blue",
        pattern=r"blue[\s-]*light(?!\s*receptor)|blue[\s-]*7|light\s*pulse",
        note="`blue7` is a light-response gene name that only ever appears on "
             "blue-pulse samples in this corpus; `light pulse` samples are "
             "blue in all matched rows (verified in the precision audit). "
             "The `(?!\\s*receptor)` lookahead is a FIX FOUND BY THE AUDIT: "
             "`Trichoderma Blr1 (Blue light receptor) mutant grown in light` "
             "is a WHITE-light sample whose parenthetical gene annotation "
             "names the colour. 1 row, and it was the only blue false "
             "positive in the corpus.",
    ),
    Rule(
        axis="light_regime",
        subcategory="light_blue_pulse",
        pattern=r"puls(?:e|ed|ing)\b",
        requires=("light_blue",),
        note="Refinement of light_blue, never standalone. `requires` stops the "
             "word `pulsed` from labelling the sorbitol-pulse osmotic samples "
             "in the carbon-source parent (2 rows) as light.",
    ),
    Rule(
        axis="light_regime",
        subcategory="light_red",
        pattern=r"(?<!far )(?<!far-)\bred\s*light",
        note="Negative lookbehind, because every red sample in the corpus comes "
             "from a study that also ran far-red.",
    ),
    Rule(
        axis="light_regime",
        subcategory="light_far_red",
        pattern=r"far[\s-]?red",
    ),
    Rule(
        axis="light_regime",
        subcategory="light_white",
        pattern=r"\bin\s+(?:constant\s+)?light\b|grown\s+in\s+light\b"
                r"|growth\s+on\s+\S+\s+in\s+light\b|photoinduc",
        excludes=("light_blue", "light_red", "light_far_red"),
        note="Unspecified / broad-spectrum illumination. Excluded when a colour "
             "was named, so the axis stays a partition of illuminations. Does "
             "NOT match `14h photoperiod` -- see fact 4.",
    ),
    Rule(
        axis="light_regime",
        subcategory="light_darkness",
        pattern=r"\bdark(?:ness)?\b",
        note="Constant darkness, used as the control arm. NOT `dark recovery`: "
             "nothing in this corpus matches /recover/ -- see fact 3.",
    ),
    # Antagonist kingdom
    Rule(
        axis="antagonist_kingdom",
        subcategory="antagonist_oomycete",
        pattern=r"pythium|globisporangium|oomycete",
        parent_scope=_BIOTIC,
        note="Oomycetes are not fungi. Kept separate from antagonist_fungus "
             "because the cell wall is cellulosic, not chitinous, and the "
             "mycoparasitic programme should not be assumed identical.",
    ),
    Rule(
        axis="antagonist_kingdom",
        subcategory="antagonist_bacterium",
        pattern=r"bacillus|velezensis|\bbacteri",
        parent_scope=_BIOTIC,
        note="6 samples. See the falsifier report: this subcategory is BELOW "
             "the permutation floor and is reported as untestable, not as "
             "non-separating.",
    ),
    Rule(
        axis="antagonist_kingdom",
        subcategory="antagonist_animal",
        pattern=r"fungivor|larva|melanogaster|folsomi|collembol"
                r"|bursaphelenchus|nematod",
        parent_scope=_BIOTIC,
        note="Grazing animals: Drosophila larvae (insect), Orthonychiurus "
             "folsomi (springtail), Bursaphelenchus xylophilus (nematode). "
             "The supervisor's phrasing was `insect larva`; the corpus also "
             "contains a springtail and a nematode, so the label is `animal`.",
    ),
    Rule(
        axis="antagonist_kingdom",
        subcategory="antagonist_fungus",
        pattern=r"rhizoctonia|\br\.\s*solani|botrytis|\bb\.\s*cinerea"
                r"|\balternaria|\ba\.\s*alternata|fusarium|\bf\.\s*graminearum"
                r"|foc4|nrrl\s*\d+|sclerotinia|sclerot|\bs\.\s*rolfsii"
                r"|cronartium|fusaric|colletotrichum"
                r"|\bc\.\s*musae|schizophyllum|laccaria|\bl\.\s*bicolor",
        parent_scope=_BIOTIC,
        note="Named fungal genera and the abbreviations actually used in this "
             "corpus. `nrrl\\d+` are Fusarium oxysporum strain codes used in "
             "the volatile-exposure study, where the partner is named only by "
             "strain code.",
    ),
    Rule(
        axis="antagonist_kingdom",
        subcategory="antagonist_fungus",
        pattern=r"mycoparasit|pathogen|patholog|fungi\b|fungal",
        parent_scope=_BIOTIC,
        excludes=("antagonist_oomycete", "antagonist_bacterium",
                  "antagonist_animal", "antagonist_fungus"),
        note="Generic fallback, fires only when no taxon was named. `excludes` "
             "is what keeps `pathogene Globisporangium ultimum` out of the "
             "fungal bucket.",
    ),
    # Antagonist mode
    Rule(
        axis="antagonist_mode",
        subcategory="challenge_inanimate",
        pattern=r"cell\s*wall?\b|inactivated|fusaric acid|necromass"
                r"|mycelial powder",
        parent_scope=_BIOTIC,
        note="The supervisor's `fungus vs inanimate opponent`: purified cell "
             "wall, autoclaved biomass, a purified antagonist metabolite. "
             "`wall?` because the source metadata spells it `cell wal` on the "
             "3 Sclerotinia-cell-wall samples; without the optional l they "
             "fell into challenge_confrontation, which the audit caught.",
    ),
    Rule(
        axis="antagonist_mode",
        subcategory="challenge_volatile",
        pattern=r"volatile|\bvocs?\b",
        parent_scope=_BIOTIC,
        note="Live partner, no physical contact: only the headspace is shared.",
    ),
    Rule(
        axis="antagonist_mode",
        subcategory="challenge_confrontation",
        pattern=r".",
        parent_scope=_BIOTIC,
        excludes=("challenge_inanimate", "challenge_volatile"),
        note="Default arm of the mode axis: direct plate confrontation with a "
             "live partner. Pattern `.` with `excludes` makes the axis total "
             "inside the parent, so `confrontation` is never silently empty.",
    ),
    # Contact stage
    Rule(
        axis="contact_stage",
        subcategory="before_contact",
        pattern=r"before\s+(?:plant\s+)?contact",
        note="Scope is ALL parents, on the same measured grounds as the light "
             "axis: all 103 `before contact` samples are filed under parent "
             "`control`, and 22 of the `after contact` ones too. Scoping this "
             "axis to `biotic antagonism` makes the before-arm EMPTY -- "
             "measured, 0 rows, in the first build of this table.",
    ),
    Rule(
        axis="contact_stage",
        subcategory="at_contact",
        pattern=r"\bat\s+contact|initial\s+contact",
    ),
    Rule(
        axis="contact_stage",
        subcategory="after_contact",
        pattern=r"after\s+(?:plant\s+)?contact",
    ),
    # Carbon class
    Rule(
        axis="carbon_class",
        subcategory="carbon_chitin",
        pattern=r"chitin",
        parent_scope=("carbon source",),
    ),
    Rule(
        axis="carbon_class",
        subcategory="carbon_starvation",
        pattern=r"no\s+carbon\s+source|carbon\s+starv|without\s+carbon",
        parent_scope=("carbon source",),
    ),
    Rule(
        axis="carbon_class",
        subcategory="carbon_lignocellulose",
        pattern=r"cellulos|avicel|wheat\s*(?:straw|bran)|\bwb\b|bagasse"
                r"|sugar\s?cane|xylan|xyloglucan|\bsbh\b|corn\s*stover"
                r"|soy\s?bean\s+hull|\bstraw\b",
        parent_scope=("carbon source",),
        note="Insoluble / polymeric substrate. `WB` = wheat bran and `SBH` = "
             "sugarcane bagasse hydrolysate are the source's own abbreviations.",
    ),
    Rule(
        axis="carbon_class",
        subcategory="carbon_soluble_inducer",
        pattern=r"sophorose|lactose",
        parent_scope=("carbon source",),
        note="Soluble cellulase inducers. Kept apart from simple sugars "
             "because the regulatory outcome is the opposite one.",
    ),
    Rule(
        axis="carbon_class",
        subcategory="carbon_simple_sugar",
        pattern=r"glucose|glycerol|glyceron|mannitol|\bd-?xylose\b|xylose"
                r"|fructose|sucrose",
        parent_scope=("carbon source",),
        note="Repressing or neutral monomers/polyols. `glyceron` is a typo for "
             "glycerol present in the source metadata and is matched on "
             "purpose -- the alternative is losing 3 samples.",
    ),
    # Injury time
    Rule(
        axis="injury_time",
        subcategory="injury_early",
        pattern=r"\b(?:30\s*min|90\s*min)\b",
        parent_scope=("mechanical injury",),
    ),
    Rule(
        axis="injury_time",
        subcategory="injury_late",
        pattern=r"\b(?:4\s*h|8\s*h)\b",
        parent_scope=("mechanical injury",),
    ),
    # Other stressor
    Rule(
        axis="other_stressor",
        subcategory="stress_oxidative",
        pattern=r"hydrogen\s+peroxide|h2o2|oxidative|menadione|paraquat",
        parent_scope=("other",),
    ),
    Rule(
        axis="other_stressor",
        subcategory="sexual_development",
        pattern=r"\bmating\b|sexual|perithec|ascospore",
        parent_scope=("other",),
    ),
    Rule(
        axis="other_stressor",
        subcategory="metal_exposure",
        pattern=r"copper|cadmium|\bzinc\b|heavy\s+metal",
        parent_scope=("other",),
    ),
    Rule(
        axis="other_stressor",
        subcategory="xenobiotic",
        pattern=r"dichlorvos|polyethylene\s+glycol|fungicide|pesticid",
        parent_scope=("other",),
    ),
    Rule(
        axis="other_stressor",
        subcategory="nitrogen_source",
        pattern=r"low\s+n\s+medium|nitrogen|ammonium\s+sulfate|peptone"
                r"|methionine",
        parent_scope=("other",),
    ),
)

#: axis -> ordered subcategories on that axis
AXES: Dict[str, Tuple[str, ...]] = {}
for _r in RULES:
    _cur = AXES.get(_r.axis, ())
    if _r.subcategory not in _cur:
        AXES[_r.axis] = _cur + (_r.subcategory,)
del _r, _cur


# Applying the rules (the builder script only)
def apply_rules(row: Dict[str, str],
                rules: Sequence[Rule] = RULES) -> Tuple[List[str], List[str]]:
    """Label one metadata row.

    Returns ``(subcategories, matched_terms)``, both in rule order. Rules are
    evaluated in declaration order so that `requires`/`excludes` see the
    decisions already taken on the same row.
    """
    text = (row.get("condition") or "")
    got: List[str] = []
    terms: List[str] = []
    for rule in rules:
        if rule.parent_scope is not None:
            if row.get("condition_category") not in rule.parent_scope:
                continue
        if any(x in got for x in rule.excludes):
            continue
        if rule.requires and not all(x in got for x in rule.requires):
            continue
        m = rule.regex().search(text)
        if m is None:
            continue
        if rule.subcategory in got:
            continue
        got.append(rule.subcategory)
        terms.append(m.group(0).strip().lower() or rule.subcategory)
    return got, terms


# Reading the table (everything else)
@dataclass(frozen=True)
class SampleLabels:
    sample: str
    genome: str
    study: str
    parent_category: str
    parent_category_curated: str
    subcategories: Tuple[str, ...]
    matched_terms: Tuple[str, ...]
    match_field: str
    #: Measured precision of the WEAKEST rule that fired on this row, from the
    #: manual audit in data/conditions/condition_taxonomy_precision.tsv.
    #: 1.0 when no rule fired (the empty label is trivially correct).
    confidence: float
    #: Subcategories that passed the permutation falsifier. A consumer that
    #: wants only load-bearing dimensions should use this, not `subcategories`.
    separating: Tuple[str, ...]


def load_taxonomy(path: str = TAXONOMY_PATH) -> Dict[str, SampleLabels]:
    """Read the one source of truth. Raises if it is not there.

    No regex is run here on purpose: a consumer must not be able to get labels
    that differ from the ones the falsifier was run against.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"condition taxonomy table not found at {path}. "
            "Generate it with src/scripts/build_condition_taxonomy.py -- "
            "this reader deliberately does NOT fall back to re-running the "
            "regex rules."
        )
    out: Dict[str, SampleLabels] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"sample", "genome", "study", "parent_category",
                    "subcategories", "matched_terms", "match_field",
                    "confidence", "separating_subcategories"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            subs = tuple(s for s in row["subcategories"].split("|") if s)
            sep = tuple(s for s in row["separating_subcategories"].split("|") if s)
            out[row["sample"]] = SampleLabels(
                sample=row["sample"],
                genome=row["genome"],
                study=row["study"],
                parent_category=row["parent_category"],
                parent_category_curated=row.get("parent_category_curated", ""),
                subcategories=subs,
                matched_terms=tuple(t for t in row["matched_terms"].split("|") if t),
                match_field=row["match_field"],
                confidence=float(row["confidence"]),
                separating=sep,
            )
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


@dataclass(frozen=True)
class Support:
    subcategory: str
    axis: str
    n_samples: int
    n_genomes: int
    n_studies: int
    #: Rule from the module docstring: "learn" / "shrink" / "freeze".
    offset_policy: str


def support(path: str = TAXONOMY_PATH) -> Dict[str, Support]:
    """Per-subcategory support, recomputed from the table.

    `offset_policy` implements the rule in the module docstring: a
    subcategory confined to ONE study is indistinguishable from that study's
    batch, so its embedding offset must be frozen at zero rather than learned.
    """
    tax = load_taxonomy(path)
    acc: Dict[str, Dict[str, set]] = {}
    for lab in tax.values():
        for sub in lab.subcategories:
            e = acc.setdefault(sub, {"s": set(), "g": set(), "st": set()})
            e["s"].add(lab.sample)
            e["g"].add(lab.genome)
            e["st"].add(lab.study)
    out: Dict[str, Support] = {}
    for sub, e in acc.items():
        ns = len(e["st"])
        policy = "learn" if ns >= 3 else ("shrink" if ns == 2 else "freeze")
        axis = next((a for a, m in AXES.items() if sub in m), "")
        out[sub] = Support(sub, axis, len(e["s"]), len(e["g"]), ns, policy)
    return out


def axis_stated_matrix(samples: Iterable[str], path: str = TAXONOMY_PATH):
    """One bit per axis: "this sample states a value on this axis".

    Returns ``(matrix, axis_names)``. Requires numpy.
    """
    import numpy as np

    tax = load_taxonomy(path)
    samples = list(samples)
    axes = list(AXES)
    mat = np.zeros((len(samples), len(axes)), dtype=np.float32)
    for i, s in enumerate(samples):
        lab = tax.get(s)
        if lab is None:
            raise KeyError(f"sample {s!r} is not in {path}")
        got = set(lab.subcategories)
        for j, ax in enumerate(axes):
            if got & set(AXES[ax]):
                mat[i, j] = 1.0
    return mat, axes


def subcategory_matrix(samples: Iterable[str],
                       vocabulary: Optional[Sequence[str]] = None,
                       path: str = TAXONOMY_PATH,
                       separating_only: bool = True):
    """Multi-hot indicator matrix, shape ``(len(samples), len(vocabulary))``.

    `separating_only=True` (default) keeps only subcategories that passed the
    permutation falsifier, so a dimension without measured expression signal
    cannot silently enter the model.

    Returns ``(matrix, vocabulary)``. Requires numpy.
    """
    import numpy as np

    tax = load_taxonomy(path)
    samples = list(samples)
    if vocabulary is None:
        seen: List[str] = []
        for s in samples:
            lab = tax.get(s)
            if lab is None:
                continue
            for c in (lab.separating if separating_only else lab.subcategories):
                if c not in seen:
                    seen.append(c)
        vocabulary = sorted(seen)
    index = {c: i for i, c in enumerate(vocabulary)}
    mat = np.zeros((len(samples), len(vocabulary)), dtype=np.float32)
    for i, s in enumerate(samples):
        lab = tax.get(s)
        if lab is None:
            raise KeyError(f"sample {s!r} is not in {path}")
        for c in (lab.separating if separating_only else lab.subcategories):
            j = index.get(c)
            if j is not None:
                mat[i, j] = 1.0
    return mat, list(vocabulary)
