"""What disqualifies a sequence from the order and from the construct.

A panel of 10 promoters that cannot be synthesised or cloned is worth zero, so
this module turns vendor requirements, the cloning scheme and host biology
into predicates with numeric thresholds that a generator can call in a loop.

The thresholds come from three different places. Synthesis rules come from
vendor documentation (IDT, Twist, GenScript) and breaking one breaks the
order, not the biology: the fragment is flagged high-complexity, priced
higher, rejected or delivered with errors. Cloning rules cover type IIS sites
and six-cutters; the vector is not known, so the set is minimal-safe -- four
type IIS enzymes used by MoClo, GoldenBraid and TrichoGate, plus six
six-cutters that sit in the MCS of most fungal vectors. Host rules cover what
a sequence->expression model cannot see: it predicts mRNA rank while the lab
reads GFP protein, and a uORF, a cryptic intron or premature polyadenylation
each lower protein at unchanged or higher mRNA.

Every threshold is calibrated against the 13 567 natural P1 promoters of
`sequences.tsv.gz` (`--calibrate`), because a threshold that rejects most of
the host's natural promoters rejects the pattern the task asks us to imitate.
Each rule carries the measured fraction that breaks it in nature.

The number to know before wiring this in: all 16 evaluated rules at once are
satisfied by 86 of 13 567 = 0.63 % of natural P1 promoters. The seventeenth,
K3, needs the Gibson adapters and cannot fail without them, so it is reported
as not evaluated rather than as passed; `check_adapters` scores it once the
adapters are known. This is not a sieve for
generator candidates -- as a post hoc filter it would reject 99.4 % of them.
It belongs inside a loop of repair, re-prediction and re-check; the repair is
a median of 1 type IIS site plus 2 six-cutter sites plus the removal of tail
ATGs, and every one of those substitutions changes the model's input.

Two rules are deliberately absent. "No accidental ORFs" as a global rule is
unsatisfiable, since an ORF of >= 10 aa appears in 98.68 % of natural P1
promoters, so G2 covers the tail only, which is the part that is transcribed.
And a run of S bases (G or C, mixed) is a different object from a vendor's
homopolymer (a run of the same base): on the 113 jury promoters an S run >= 7
breaks 85.8 % against 12.4 % for a run of one identical base, and the vendor
thresholds are about the second.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys

from . import paths

COMP = str.maketrans("ACGT", "TGCA")

# Threshold constants; each carries its source.

MODEL_LEN = 800                # length of the model input (corpus: 13 567/13 568)
TAIL = 150                     # leader zone at the 3' end, see TAIL_RATIONALE

TAIL_RATIONALE = """\
The `Promotor` fragment ends immediately before the gene's ATG codon: 99.93 %
of P1 genes start with ATG, and the composition at promoter position -3 is
A 0.556 against a background of 0.253 (positions -400..-391).  The TSS
therefore lies INSIDE the 800 bp and the tail of the fragment reaches the
transcript.  The tail length of 150 bp is taken from A. fumigatus: mean 5'UTR
126 nt, median 49 nt (Cohen et al., RNA 32:250, PMID 41326228 /
bioRxiv 2025.07.04.662582).  The natural median distance from the 3' end to
the nearest ATG in P1 is 103 bp -- so the tail rules are not absurdly
restrictive with respect to the host."""

# Synthesis.  IDT gBlocks reject runs of >= 10 A/T or >= 6 G/C:
# https://www.idtdna.com/pages/support/faqs/what-types-of-sequence-motifs-should-be-avoided-when-ordering-gblocks-gene-fragments-
MAX_RUN_AT = 9
MAX_RUN_GC = 5
# Twist, a hard rule for Gene Fragments and Clonal Genes alike:
# "Avoid homopolymers >= 14bp".
# https://www.twistbioscience.com/faq/gene-synthesis/are-there-any-sequence-limitationsdesign-guidelines-genes-which-i-should-follow
MAX_RUN_ANY = 13
# GenScript Gene Fragments: GC 25-65 % per 100 bp window, repeats under 20 bp.
# https://www.genscript.com/gentitan-gene-fragments.html
GC_WINDOW = 100
GC_WINDOW_MIN, GC_WINDOW_MAX = 0.25, 0.65
MAX_REPEAT = 19                # reject >= 20
# Twist "High Complexity": local GC per 50 bp window outside 10-90 %, or a
# spread between the highest and lowest window above 50 points.
GC_W50_MIN, GC_W50_MAX, GC_W50_DELTA = 0.10, 0.90, 0.50
# Global GC narrowed from the wide vendor range (IDT 25-75 %) to the host's
# own distribution (P1: p1 = 0.383, p99 = 0.616).
GC_GLOBAL_MIN, GC_GLOBAL_MAX = 0.35, 0.65
# Inverted repeat: from the corpus calibration (p99 = 18, breaks 0.52 %).  The
# vendor threshold is an order of magnitude looser, so this is a design rule
# and not an ordering rule.
MAX_INVERTED = 19              # reject >= 20

# Cloning.  TrichoGate (Nogueira-Lopez et al. 2019,
# doi:10.3389/fmicb.2019.02794) uses only BsaI; MoClo and GoldenBraid add
# BsmBI, BbsI and SapI.
TYPE_IIS = {"BsaI": "GGTCTC", "BsmBI": "CGTCTC", "BbsI": "GAAGAC",
            "SapI": "GCTCTTC"}
SIX_CUTTERS = {"EcoRI": "GAATTC", "HindIII": "AAGCTT", "BamHI": "GGATCC",
               "XbaI": "TCTAGA", "XhoI": "CTCGAG", "KpnI": "GGTACC"}
# Gibson / NEBuilder HiFi: overlap 15-25 bp, Tm >= 48 C.
# https://www.neb.com/en/faqs/how-does-overlap-length-affect-nebuilder-hifi-dna-assembly-efficiency
GIBSON_MIN, GIBSON_MAX, GIBSON_TM = 15, 25, 48.0

# Host.  A. oryzae 3'-end processing: AAUGAA dominates at 6 % of transcripts,
# AAUAAA at 4-5 %.  Tanaka et al., DNA Research 18:189.
POLYA = ("AATAAA", "AATGAA", "ATTAAA", "AATAAT")
# Fungal intron: 5'SS GT[AG]AG, branch point CT[AG]A[CT] (>90 % of Fusarium
# introns), 3'SS [CT]AG, under 100 bp in filamentous fungi.
RE_5SS = re.compile(r"GT[AG]AG")
RE_BP = re.compile(r"CT[AG]A[CT]")
RE_3SS = re.compile(r"[CT]AG")
INTRON_MIN, INTRON_MAX = 50, 150
# Kozak context measured on 13 568 P1 promoters: position -3 = A at 0.556 is
# the only one departing from background by more than 2x.
KOZAK_RECOMMENDED = "GCCACC"
STOPS = ("TAA", "TAG", "TGA")


# Primitives

def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def gc(s: str) -> float:
    return (s.count("G") + s.count("C")) / max(1, len(s))


def gc_windows(s: str, w: int):
    n = len(s)
    if n < w:
        return gc(s), gc(s)
    pref = [0] * (n + 1)
    for i, ch in enumerate(s):
        pref[i + 1] = pref[i] + (1 if ch in "GC" else 0)
    v = [(pref[i + w] - pref[i]) / w for i in range(n - w + 1)]
    return min(v), max(v)


def max_run_identical(s: str, bases: str) -> int:
    """Longest run of one base, that base being one of `bases`.

    The vendor's definition.  A mixed run, G next to C, is a different
    measurement; see `max_run_class`."""
    best = cur = 0
    prev = ""
    for ch in s:
        cur = cur + 1 if (ch == prev and ch in bases) else (1 if ch in bases else 0)
        prev = ch
        if cur > best:
            best = cur
    return best


def max_run_class(s: str, klass: str) -> int:
    """Longest run of bases from a class, e.g. S = G|C mixed.  Not a
    homopolymer in the vendors' sense; kept so that the two definitions can
    be compared in one place."""
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch in klass else 0
        if cur > best:
            best = cur
    return best


def longest_repeat(s: str, maximum: int = 60) -> int:
    lo, hi, best = 1, min(maximum, len(s) // 2), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        seen, hit = set(), False
        for i in range(len(s) - mid + 1):
            k = s[i:i + mid]
            if k in seen:
                hit = True
                break
            seen.add(k)
        if hit:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def longest_inverted(s: str, maximum: int = 60) -> int:
    lo, hi, best = 1, min(maximum, len(s) // 2), 0
    r = rc(s)
    while lo <= hi:
        mid = (lo + hi) // 2
        pool = {s[i:i + mid] for i in range(len(s) - mid + 1)}
        hit = any(r[i:i + mid] in pool for i in range(len(r) - mid + 1))
        if hit:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def count_sites(s: str, motif: str) -> int:
    """Occurrences on both strands; a palindrome counted once."""
    m2 = rc(motif)
    return s.count(motif) + (s.count(m2) if m2 != motif else 0)


def sites_detail(s: str, table: dict) -> dict:
    return {n: count_sites(s, m) for n, m in table.items() if count_sites(s, m)}


def orf_to_end(s: str, tail: int = TAIL) -> int:
    """ATGs in the tail whose frame reaches the 3' end with no stop codon.

    After fusion such an ATG reads on into the reporter: in frame it gives an
    N-terminal extension, out of frame no GFP at all."""
    n, count = len(s), 0
    for i in range(max(0, n - tail), n - 2):
        if s[i:i + 3] != "ATG":
            continue
        if not any(s[j:j + 3] in STOPS for j in range(i + 3, n - 2, 3)):
            count += 1
    return count


def cryptic_intron(s: str, tail: int = TAIL) -> int:
    o = s[-tail:]
    for m5 in RE_5SS.finditer(o):
        a = m5.start()
        for mb in RE_BP.finditer(o, a + 20):
            b = mb.start()
            if b - a > INTRON_MAX:
                break
            for m3 in RE_3SS.finditer(o, b + 3):
                if INTRON_MIN <= (m3.start() + 3 - a) <= INTRON_MAX:
                    return 1
    return 0


def simple_tm(s: str) -> float:
    """Wallace rule, 2 C per A/T and 4 C per G/C, as in NEB's overlap
    guideline."""
    return 2.0 * (s.count("A") + s.count("T")) + 4.0 * (s.count("G") + s.count("C"))


def longest_common(a: str, b: str, maximum: int = 400) -> int:
    """Longest exact common substring (both strands of b)."""
    b2 = b + "|" + rc(b)
    lo, hi, best = 1, min(maximum, len(a)), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        pool = {b2[i:i + mid] for i in range(len(b2) - mid + 1)}
        hit = any(a[i:i + mid] in pool for i in range(len(a) - mid + 1))
        if hit:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# Rules: (code, description, predicate -> (ok, value), consequence).

def _r(code, description, fn, consequence, layer):
    return {"code": code, "description": description, "fn": fn,
            "consequence": consequence, "layer": layer}


RULES = [
    # Synthesis
    _r("S1", "alphabet = {A,C,G,T}, no N and no degenerate bases",
       lambda s: (set(s) <= set("ACGT"),
                  len(s) - sum(s.count(b) for b in "ACGT")),
       "Twist does not accept degenerate bases, and the model encodes N as 0.25 "
       "in each of the four channels (hyppe/dna.py, UNKNOWN_ENCODING='uniform') "
       "-- a distribution it never saw in training.", "synthesis"),
    _r("S2", "insert length == 800 bp (the model's regime); ordered fragment "
             "125-3000 bp (IDT) / 300-5000 (Twist) / 200-3000 (GenScript)",
       lambda s: (len(s) == MODEL_LEN, len(s)),
       "A different length is a different input from the one the model was "
       "trained on; the prediction stops being comparable with the reference "
       "points (tef1 0.9983, pks1 0.9437).", "synthesis"),
    _r("S3", "global GC in [0.35; 0.65]",
       lambda s: (GC_GLOBAL_MIN <= gc(s) <= GC_GLOBAL_MAX, round(gc(s), 4)),
       "Outside the host's band (P1: p1 = 0.383, p99 = 0.616) and closer to "
       "IDT's rejection threshold (<25 % / >75 %).  In the jury corpus the "
       "three sequences outside the band had strength 0.000.", "synthesis"),
    _r("S4", "every 100 bp window has GC in [0.25; 0.65] (GenScript)",
       lambda s: (gc_windows(s, GC_WINDOW)[0] >= GC_WINDOW_MIN
                  and gc_windows(s, GC_WINDOW)[1] <= GC_WINDOW_MAX,
                  tuple(round(x, 3) for x in gc_windows(s, GC_WINDOW))),
       "The sequence trips GenScript's complexity filter; the order is routed "
       "to the more expensive clonal path or rejected.", "synthesis"),
    _r("S5", "every 50 bp window has GC in [0.10; 0.90] and the spread of 50 bp "
             "windows <= 0.50 (Twist)",
       lambda s: (gc_windows(s, 50)[0] >= GC_W50_MIN
                  and gc_windows(s, 50)[1] <= GC_W50_MAX
                  and (gc_windows(s, 50)[1] - gc_windows(s, 50)[0]) <= GC_W50_DELTA,
                  round(gc_windows(s, 50)[1] - gc_windows(s, 50)[0], 3)),
       "Twist flags the fragment as 'High Complexity': longer turnaround, "
       "higher error rate, possible refusal to synthesise.", "synthesis"),
    _r("S6", "homopolymer of an identical base: A<=9, T<=9, G<=5, C<=5 (IDT)",
       lambda s: (max_run_identical(s, "AT") <= MAX_RUN_AT
                  and max_run_identical(s, "GC") <= MAX_RUN_GC,
                  (max_run_identical(s, "AT"), max_run_identical(s, "GC"))),
       "Polymerase slippage during oligo assembly: deletions/insertions in the "
       "delivered fragment, i.e. a sequence different from the one ordered and "
       "different from the one the model scored.", "synthesis"),
    _r("S7", "homopolymer of any base <= 13 (Twist hard rule)",
       lambda s: (max_run_identical(s, "ACGT") <= MAX_RUN_ANY,
                  max_run_identical(s, "ACGT")),
       "A hard rejection on Twist's side ('Avoid homopolymers >= 14bp').",
       "synthesis"),
    _r("S8", "longest exact direct repeat <= 19 bp (GenScript)",
       lambda s: (longest_repeat(s) <= MAX_REPEAT, longest_repeat(s)),
       "Recombination/slippage during assembly and during propagation in "
       "E. coli; the fragment comes back shorter or with a deletion between "
       "the repeats.", "synthesis"),
    _r("S9", "longest inverted repeat <= 19 bp",
       lambda s: (longest_inverted(s) <= MAX_INVERTED, longest_inverted(s)),
       "A hairpin during synthesis and PCR: truncated products, and in the "
       "cell a potential instability of the insert.  The threshold is a design "
       "one (P1 corpus: p99 = 18), not an ordering one.", "synthesis"),

    # Cloning
    _r("K1", "zero BsaI/BsmBI/BbsI/SapI sites on either strand",
       lambda s: (sum(count_sites(s, m) for m in TYPE_IIS.values()) == 0,
                  sites_detail(s, TYPE_IIS)),
       "Golden Gate cuts INSIDE the insert: instead of one fragment there are "
       "two, and assembly yields a truncated construct or none.  TrichoGate "
       "(Trichoderma) uses BsaI.", "cloning"),
    _r("K2", "zero EcoRI/HindIII/BamHI/XbaI/XhoI/KpnI sites (minimal set for an "
             "unknown vector)",
       lambda s: (sum(count_sites(s, m) for m in SIX_CUTTERS.values()) == 0,
                  sites_detail(s, SIX_CUTTERS)),
       "If any of these enzymes linearises the vector or excises the cassette, "
       "the insert gets cut.  The rule is insurance: once the vector is fixed "
       "it can be narrowed.", "cloning"),
    _r("K3", "no exact common substring >= 20 bp between insert and Gibson "
             "adapters",
       lambda s: (True, "requires the adapters to be supplied"),
       "Mispriming and assembly in the wrong place; checked only once the "
       "adapters are known (see `check_adapters`).", "cloning"),

    # Host and construct
    _r("G1", "zero ATG in the last 150 bp (sense strand)",
       lambda s: (s[-TAIL:].count("ATG") == 0, s[-TAIL:].count("ATG")),
       "Every such ATG is a uORF in the transcript leader: the ribosome "
       "initiates before GFP, there is less protein at UNCHANGED mRNA -- i.e. "
       "the model's objective is satisfied and the experiment is not.", "host"),
    _r("G2", "zero frames from the tail reaching the 3' end without a STOP",
       lambda s: (orf_to_end(s) == 0, orf_to_end(s)),
       "Such a frame runs into GFP: in frame it gives an N-terminal extension "
       "(changed brightness, possible mislocalisation), out of frame it gives "
       "no GFP at all.", "host"),
    _r("G3", "position -3 = A (the host's Kozak context)",
       lambda s: (s[-3] == "A", s[-3]),
       "A weaker start context -> some ribosomes skip the GFP ATG (leaky "
       "scanning); the fluorescence signal drops with no relation to promoter "
       "strength.  Measured in P1: A at -3 in 0.556 of cases against a "
       "background of 0.253.", "host"),
    _r("G4", "zero AATAAA/AATGAA/ATTAAA/AATAAT signals in the last 150 bp",
       lambda s: (sum(s[-TAIL:].count(m) for m in POLYA) == 0,
                  sum(s[-TAIL:].count(m) for m in POLYA)),
       "Premature cleavage and polyadenylation in the leader: the transcript "
       "ends before GFP.  In fungi the hexamer is not canonical (AAUGAA 6 %, "
       "AAUAAA 4-5 % of A. oryzae transcripts), so the rule is deliberately "
       "cautious.", "host"),
    _r("G5", "no 5'SS ... branch point ... 3'SS arrangement (50-150 nt) in the "
             "last 150 bp",
       lambda s: (cryptic_intron(s) == 0, cryptic_intron(s)),
       "A cryptic intron splices out part of the leader or the start of GFP; "
       "no protein is made, while the mRNA still counts as present.", "host"),
]

# Rules whose predicate cannot fail on the insert alone, so a verdict on them
# is not a measurement: they are excluded from the "all rules" tally and
# labelled, instead of inflating it by one rule that always passes.
NOT_EVALUATED = ("K3",)


# API

def check(seq: str) -> list:
    """List of verdicts for one sequence.  Each: code, ok, value, consequence."""
    s = seq.upper().strip()
    out = []
    for r in RULES:
        ok, v = r["fn"](s)
        out.append({"code": r["code"], "layer": r["layer"],
                    "description": r["description"], "ok": bool(ok),
                    "value": v, "consequence": r["consequence"]})
    return out


def passes(seq: str) -> bool:
    return all(w["ok"] for w in check(seq))


def report(seq: str, name: str = "") -> str:
    w = check(seq)
    ev = [x for x in w if x["code"] not in NOT_EVALUATED]
    lines = ["%s  (%d/%d evaluated rules passed, %d not evaluated)"
             % (name or "sequence", sum(x["ok"] for x in ev), len(ev),
                len(w) - len(ev))]
    for x in w:
        lines.append("  %-4s %-6s %-28s %s"
                     % (x["code"],
                        "N/EVAL" if x["code"] in NOT_EVALUATED
                        else ("OK" if x["ok"] else "BREAK"),
                        str(x["value"])[:28], x["description"][:70]))
    return "\n".join(lines)


def check_adapters(insert: str, adapter5: str, adapter3: str) -> list:
    """Rules that require the cloning adapters to be known.

    The vendor scans the ordered sequence, i.e. the insert together with its
    adapters, so every `synthesis` rule is recomputed on the joined
    sequence."""
    out = []
    for name, a in (("5'", adapter5), ("3'", adapter3)):
        if not a:
            continue
        out.append({"code": "K3a", "ok": GIBSON_MIN <= len(a) <= GIBSON_MAX,
                    "value": len(a),
                    "description": "overlap length %s in [15, 25] bp "
                                   "(NEBuilder HiFi)" % name})
        out.append({"code": "K3b", "ok": simple_tm(a) >= GIBSON_TM,
                    "value": simple_tm(a),
                    "description": "overlap Tm %s >= 48 C (2 C A/T, 4 C G/C)"
                                   % name})
        out.append({"code": "K3c", "ok": longest_common(insert, a) < 20,
                    "value": longest_common(insert, a),
                    "description": "no common substring >= 20 bp "
                                   "insert<->adapter %s" % name})
    joined = (adapter5 or "") + insert + (adapter3 or "")
    for r in RULES:
        if r["layer"] != "synthesis" or r["code"] == "S2":
            continue
        ok, v = r["fn"](joined)
        out.append({"code": r["code"] + "*", "ok": bool(ok), "value": v,
                    "description": r["description"]
                    + "  [computed on the JOINED sequence, because that is "
                      "what is ordered]"})
    return out


def check_panel(sequences: dict, references: dict = None,
                max_panel: int = 50, max_native: int = 100) -> list:
    """Rules at the level of a panel of 10.

    P1: two designs may not share an exact substring >= 50 bp, or they can
    recombine during a joint transformation and stop being independent points
    at interpretation time.

    P2: no design may carry >= 100 bp of exact identity with a native host
    sequence, because such a stretch acts as a homology arm and directs
    integration to the native locus rather than at random, mixing the GFP
    readout with damage to the native gene.  The threshold is an estimate:
    minimal homology arms in filamentous fungi run from about 50 bp to 1 kb.
    """
    out = []
    names = list(sequences)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = longest_common(sequences[names[i]], sequences[names[j]])
            out.append({"code": "P1", "ok": d < max_panel, "value": d,
                        "description": "%s vs %s: common substring < %d bp"
                                       % (names[i], names[j], max_panel)})
    for n, s in sequences.items():
        for rn, rs in (references or {}).items():
            d = longest_common(s, rs)
            out.append({"code": "P2", "ok": d < max_native, "value": d,
                        "description": "%s vs native %s: identity < %d bp"
                                       % (n, rn, max_native)})
    return out


# Calibration on the corpus

def _p1_corpus(limit=None):
    out = []
    with gzip.open(paths.require(paths.SEQUENCES, "sequences.tsv.gz"),
                   "rt") as f:
        h = f.readline().rstrip("\n").split("\t")
        iG, iS, iGR = h.index("GENOME"), h.index("SEQUENCE"), h.index("GROUP")
        for line in f:
            r = line.rstrip("\n").split("\t")
            if (len(r) > iGR and r[iGR] == "Promotor" and r[iG] == "P1"
                    and len(r[iS]) == MODEL_LEN):
                out.append(r[iS].upper())
                if limit and len(out) >= limit:
                    break
    return out


def calibrate(limit=None):
    seqs = _p1_corpus(limit)
    print("corpus: %d natural P1 promoters of %d bp" % (len(seqs), MODEL_LEN))
    print(TAIL_RATIONALE)
    broken = {r["code"]: 0 for r in RULES}
    tested = [r for r in RULES if r["code"] not in NOT_EVALUATED]
    passed = 0
    for s in seqs:
        ok_all = True
        for r in tested:
            if not r["fn"](s)[0]:
                broken[r["code"]] += 1
                ok_all = False
        passed += ok_all
    print("\n=== HOW MANY NATURAL P1 PROMOTERS BREAK EACH RULE ===")
    for r in RULES:
        if r["code"] in NOT_EVALUATED:
            print("%-4s %-10s %8s   %s"
                  % (r["code"], r["layer"], "NOT EVAL", r["description"][:64]))
            continue
        print("%-4s %-10s %6.2f %%   %s"
              % (r["code"], r["layer"], 100 * broken[r["code"]] / len(seqs),
                 r["description"][:64]))
    print("\nsatisfies ALL %d EVALUATED rules: %d / %d = %.2f %%"
          % (len(tested), passed, len(seqs), 100 * passed / len(seqs)))
    print("   %s of %d rules not evaluated (%s): the predicate cannot fail "
          "on the insert alone, so counting it as passed would inflate the "
          "line above." % (len(RULES) - len(tested), len(RULES),
                           ", ".join(NOT_EVALUATED)))
    print("-> the list is NOT a post hoc filter; it has to run inside a "
          "repair-and-recompute loop.")
    return broken


def jury():
    """Do the rules fight promoter strength?

    The jury corpus (113 sequences with a `sila_prawdziwa` column) is the only
    set here with a ground truth on the same scale as the model's target.  Per
    rule: how many break it, and whether the two groups differ in strength
    (Mann-Whitney, two-sided).

    `sila_prawdziwa` is an mRNA rank while a uORF acts on translation, so the
    relation of G1 and G2 to strength is an association and not a mechanism.
    The direction still matters, because it rules out the objection that the
    uORF rule costs strength.

    `sila_prawdziwa`, `sekwencja` and `gatunek` are field names in an external
    file and are deliberately not translated.
    """
    import csv

    import numpy as np
    from scipy.stats import mannwhitneyu, spearmanr

    p = paths.require(paths.JURY_SEQUENCES, "jury sequence panel")
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8"), delimiter=";")
            if r["sila_prawdziwa"] not in (None, "")
            and len(r["sekwencja"]) == MODEL_LEN]
    strength = np.array([float(r["sila_prawdziwa"]) for r in rows])
    seqs = [r["sekwencja"].upper() for r in rows]
    print("jury: %d sequences of 800 bp with `sila_prawdziwa`" % len(rows))

    print("\n=== TWO DEFINITIONS OF A HOMOPOLYMER (THESE ARE TWO DIFFERENT "
          "MEASUREMENTS) ===")
    print("run of the IDENTICAL base G or C >= 7 : %5.1f %%" %
          (100 * np.mean([max_run_identical(x, "GC") >= 7 for x in seqs])))
    print("run of S bases (G|C mixed)       >= 7 : %5.1f %%   <-- a different "
          "object, the vendor threshold does NOT cover it" %
          (100 * np.mean([max_run_class(x, "GC") >= 7 for x in seqs])))
    print("run of the IDENTICAL base A or T >= 10: %5.1f %%" %
          (100 * np.mean([max_run_identical(x, "AT") >= 10 for x in seqs])))
    print("run of W bases (A|T mixed)       >= 10: %5.1f %%" %
          (100 * np.mean([max_run_class(x, "AT") >= 10 for x in seqs])))

    print("\n=== RULE vs STRENGTH ===")
    for r in RULES:
        # A rule that is not evaluated has no verdict to report. Printing it
        # at 0.0 % broken reads as a rule that was tested and passed.
        if r["code"] in NOT_EVALUATED:
            print("%-4s NOT EVALUATED (needs the adapters to be supplied)"
                  % (r["code"],))
            continue
        ok = np.array([bool(r["fn"](x)[0]) for x in seqs])
        if ok.all() or (~ok).all():
            print("%-4s breaks %5.1f %%  (no contrast)"
                  % (r["code"], 100 * (~ok).mean()))
            continue
        pv = mannwhitneyu(strength[ok], strength[~ok],
                          alternative="two-sided")[1]
        print("%-4s breaks %5.1f %%  strength(pass)=%.3f  strength(break)=%.3f "
              " p=%.4f"
              % (r["code"], 100 * (~ok).mean(), strength[ok].mean(),
                 strength[~ok].mean(), pv))

    print("\n=== COMPOSITION vs STRENGTH (does the GC threshold fight "
          "strength) ===")
    print("spearman(global GC, strength)           = %.4f  p=%.3g"
          % spearmanr([gc(x) for x in seqs], strength))
    print("spearman(GC, strength) T. atroviride only = %.4f  p=%.3g  (n=%d)" % (
        *spearmanr([gc(seqs[i]) for i, r in enumerate(rows)
                    if "atroviride" in r["gatunek"]],
                   strength[[i for i, r in enumerate(rows)
                             if "atroviride" in r["gatunek"]]]),
        sum(1 for r in rows if "atroviride" in r["gatunek"])))
    print("spearman(number of type IIS sites, strength) = %.4f  p=%.3g"
          % spearmanr([sum(count_sites(x, m) for m in TYPE_IIS.values())
                       for x in seqs], strength))
    print("-> the positive GC correlation vanishes within a species, so GC is "
          "NOT a design lever;")
    print("   the positive correlation of type IIS sites is the same "
          "composition effect (GGTCTC/CGTCTC are GC-rich).")


def domestication_cost(limit=3000):
    """How many substitutions it costs just to remove the restriction sites."""
    import statistics
    seqs = _p1_corpus(limit)
    iis = [sum(count_sites(s, m) for m in TYPE_IIS.values()) for s in seqs]
    six = [sum(count_sites(s, m) for m in SIX_CUTTERS.values()) for s in seqs]
    both = [a + b for a, b in zip(iis, six)]
    print("\n=== DOMESTICATION COST (n=%d natural P1 promoters) ===" % len(seqs))
    for name, c in (("type IIS (4 enzymes)", iis), ("six-cutters (6)", six),
                    ("together", both)):
        print("%-22s fraction with >=1: %.4f  mean %.2f  median %d  p95 %d  "
              "max %d" % (name, sum(1 for x in c if x > 0) / len(c),
                          statistics.mean(c), statistics.median(c),
                          sorted(c)[int(0.95 * len(c))], max(c)))
    print("-> every site is >= 1 substitution, and every substitution changes "
          "the model's input.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true",
                    help="measure how many natural P1 promoters break each rule")
    ap.add_argument("--jury", action="store_true",
                    help="do the rules fight promoter strength (jury corpus, "
                         "113 sequences)")
    ap.add_argument("--cost", action="store_true",
                    help="how many substitutions domesticating the restriction "
                         "sites costs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fasta", default=None, help="check sequences from a FASTA")
    a = ap.parse_args(argv)
    if a.fasta:
        name, buf = None, []
        seqs = {}
        for line in open(a.fasta):
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line[1:], []
            elif line:
                buf.append(line.upper())
        if name:
            seqs[name] = "".join(buf)
        for n, s in seqs.items():
            print(report(s, n))
            print()
        if len(seqs) > 1:
            print("=== PANEL ===")
            for w in check_panel(seqs):
                print("  %-4s %-6s %-6s %s"
                      % (w["code"], "OK" if w["ok"] else "BREAK", w["value"],
                         w["description"]))
    if a.jury:
        jury()
    if a.cost:
        domestication_cost(a.limit or 3000)
    if a.calibrate or not (a.fasta or a.jury or a.cost):
        calibrate(a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
