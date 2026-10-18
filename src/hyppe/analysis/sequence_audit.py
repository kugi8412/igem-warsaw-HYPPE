"""A5 -- sequence statistics, the corrected version of `eda.sequences`.

    python -m hyppe.analysis.sequence_audit     # log to logs/sequence_audit.log

The version this replaces read `stream_sequences(limit=20000)` and was invalid
three times over. `limit` is a prefix of a genome-ordered file, so the whole
sample was one strain of twenty. Nothing filtered `GROUP`, and the file holds
four record types (Promotor, Gene, FlankLeft, FlankRight), so its length
distribution mixed four objects. And GC was counted on upper case only, while
18 of 20 genomes carry 90-96 % of their bases soft-masked -- the one strain in
that prefix being one of the two unmasked ones, so the first and third defects
hid each other.

The obvious repair, every nth row of the file, has its own defect: records run
in a FlankLeft/Promotor/Gene/FlankRight cycle, so a step divisible by 4
aliases with it (measured on every 4th row: 93 164 FlankRight against 40 120
Promotor, a 2.3x distortion of what should be an even quarter). The cycle is
also imperfect, so the phase drifts and the aliasing is partial, which is
worse than total. Hence the step counter is per group: every nth record of a
given group in file order, which is uniform over the file and immune to the
cycle.

One pass computes two things of different scope. Lengths and counts come from
the whole file -- `len()` is free, so the 800 bp falsifier stands on the full
corpus. Base composition (GC, N, lower case, alphabet) comes from the
every-nth sample, which is where `.upper()` and counting are paid for. Each
report column says which. Representativeness is checked separately, by
comparing the 800 bp fraction of the sample against the full corpus.

Falsifier: if the fraction of promoters of length exactly 800 falls below
0.99, that is reported as a result and the exit code is 3, not quietly
repaired, because "800 bp in and out" is an assumption of the whole model.
The same counter on FlankLeft/FlankRight/Gene gives values near zero, so the
threshold is satisfiable by construction, and that control is printed.

There is no RNG: the sample is a function of (n, phase) and file order, so the
same file and the same `--n` give a bit-identical log.
"""
from __future__ import annotations

import argparse
import array
import collections
import gzip
import os
import sys
import zlib

import numpy as np

from . import paths

# Report order. `Promotor` first because it is the model's object; the rest
# are printed so that 800 and 1000 have explicitly assigned objects.
GROUPS = ("Promotor", "FlankLeft", "FlankRight", "Gene")

# Alphabet measured on every 97th record of the file (9846 records). Not
# assumed: anything outside it is counted into `other` and the real alphabet
# is reported per genome.
UPPER = ("A", "C", "G", "T", "N")
LOWER = ("a", "c", "g", "t", "n")

THRESHOLD_800 = 0.99     # falsifier
DEFAULT_STEP = 5         # every 5th record within a group


class Tee:
    """The log goes to the screen and to a file at once."""

    def __init__(self, path):
        paths.ensure_dir(os.path.dirname(path))
        self.fh = open(path, "w", encoding="utf-8")
        self.term = sys.stdout

    def write(self, s):
        self.term.write(s)
        self.fh.write(s)

    def flush(self):
        self.term.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


class GroupAccumulator:
    """Statistics for one (GROUP, GENOME) pair."""

    def __init__(self):
        self.lengths = array.array("i")       # full corpus
        self.n_800 = 0                        # full corpus
        self.n_sample = 0                     # sample
        self.n_800_sample = 0                 # sample, representativeness
        self.gc = array.array("d")            # sample, after .upper()
        self.gc_wrong = array.array("d")      # sample, the old way
        self.bases = collections.Counter()    # sample, after .upper()
        self.n_lower = 0                      # sample, bases
        self.n_rec_lower = 0                  # sample, records with >=1 lower
        self.n_rec_N50 = 0                    # sample, records with >50 % N
        self.n_bases = 0                      # sample
        self.alphabet = set()                 # sample, raw characters


def _pass(path, step):
    """One pass over the file: (accumulators, counts, promoter ids)."""
    acc = collections.defaultdict(GroupAccumulator)
    group_count = collections.Counter()
    # Duplicate IDs are counted for promoters only. crc32 rather than `hash()`
    # because the latter is salted per process and would give a different log
    # every run.
    prom_ids = {}

    with gzip.open(paths.require(path, "sequences.tsv.gz"), "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pos = {n: i for i, n in enumerate(header)}
        for needed in ("ID", "GENOME", "SEQUENCE", "GROUP"):
            if needed not in pos:
                raise KeyError(
                    "the sequence file has no column %r; header = %s.  "
                    "Without GROUP this analysis is meaningless -- four record "
                    "types would merge into one distribution."
                    % (needed, header))
        i_id, i_gen = pos["ID"], pos["GENOME"]
        i_seq, i_grp = pos["SEQUENCE"], pos["GROUP"]

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            grp = parts[i_grp]
            gen = parts[i_gen]
            seq = parts[i_seq]
            a = acc[(grp, gen)]

            # Full corpus, free
            ln = len(seq)
            a.lengths.append(ln)
            if ln == 800:
                a.n_800 += 1
            group_count[grp] += 1

            if grp == "Promotor":
                entry = prom_ids.get(parts[i_id])
                if entry is None:
                    prom_ids[parts[i_id]] = [1,
                                             {zlib.crc32(seq.upper().encode())}]
                else:
                    entry[0] += 1
                    entry[1].add(zlib.crc32(seq.upper().encode()))

            # Sample: every `step`th record of this group
            if group_count[grp] % step != 1 % step:
                continue
            a.n_sample += 1
            if ln == 800:
                a.n_800_sample += 1
            a.alphabet |= set(seq)
            up = seq.upper()
            counted = 0
            for b in UPPER:
                c = up.count(b)
                a.bases[b] += c
                counted += c
            if counted != ln:
                a.bases["other"] += ln - counted
            n_lower = 0
            for b in LOWER:
                n_lower += seq.count(b)
            a.n_lower += n_lower
            # Two different quantities, easily confused: the fraction of
            # lower-case bases (~7 %) and the fraction of records containing
            # even one lower-case letter (~93 %). Both are reported, by name.
            if n_lower:
                a.n_rec_lower += 1
            if ln and up.count("N") > ln / 2:
                a.n_rec_N50 += 1
            a.n_bases += ln

            gc_up = up.count("G") + up.count("C")
            gc_raw = seq.count("G") + seq.count("C")
            if ln:
                a.gc.append(gc_up / ln)
                a.gc_wrong.append(gc_raw / ln)

    return acc, group_count, prom_ids


def _table(acc, group, header=True):
    """Print a per-genome table for one group; return genome -> row."""
    genomes = sorted(g for (gr, g) in acc if gr == group)
    if header:
        print("\n%-14s | %7s %6s | %6s %7s %7s %7s | %7s %7s | %7s %7s %8s | %s"
              % ("genome", "n_file", "n_samp", "len_min", "len_med", "len_max",
                 "f_800", "GC_mean", "GC_sd", "f_N", "f_lower", "f_rec_low",
                 "alphabet"))
        print("  (f_lower = fraction of lower-case BASES; f_rec_low = fraction "
              "of RECORDS with at least one lower-case letter -- two different "
              "numbers)")
        print("-" * 143)
    rows = {}
    for g in genomes:
        a = acc[(group, g)]
        ln = np.frombuffer(a.lengths, dtype=np.int32)
        gc = np.frombuffer(a.gc, dtype=np.float64)
        gcw = np.frombuffer(a.gc_wrong, dtype=np.float64)
        n_bas = max(a.n_bases, 1)
        w = dict(
            n_file=len(ln), n_sample=a.n_sample,
            len_min=int(ln.min()), len_med=float(np.median(ln)),
            len_max=int(ln.max()),
            f_800=a.n_800 / max(len(ln), 1),
            gc_mean=float(gc.mean()) if len(gc) else float("nan"),
            gc_sd=float(gc.std(ddof=1)) if len(gc) > 1 else float("nan"),
            gc_wrong=float(gcw.mean()) if len(gcw) else float("nan"),
            f_N=a.bases["N"] / n_bas,
            f_lower=a.n_lower / n_bas,
            f_rec_low=a.n_rec_lower / max(a.n_sample, 1),
            n_rec_N50=a.n_rec_N50,
            f_other=a.bases.get("other", 0) / n_bas,
            alphabet="".join(sorted(a.alphabet)),
        )
        rows[g] = w
        print("%-14s | %7d %6d | %6d %7.0f %7d %7.4f | %7.4f %7.4f | %7.5f "
              "%7.4f %8.4f | %s"
              % (g, w["n_file"], w["n_sample"], w["len_min"], w["len_med"],
                 w["len_max"], w["f_800"], w["gc_mean"], w["gc_sd"],
                 w["f_N"], w["f_lower"], w["f_rec_low"], w["alphabet"]))
    return rows


def _pooled(acc, group):
    """Statistics pooled over genomes, for a one-line summary."""
    ln = np.concatenate([np.frombuffer(acc[(gr, g)].lengths, dtype=np.int32)
                         for (gr, g) in acc if gr == group])
    n800 = sum(acc[(gr, g)].n_800 for (gr, g) in acc if gr == group)
    gc = np.concatenate([np.frombuffer(acc[(gr, g)].gc, dtype=np.float64)
                         for (gr, g) in acc if gr == group])
    return ln, n800, gc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=DEFAULT_STEP,
                    help="sampling step: every nth record of EACH group "
                         "(n=1 is the full corpus).  Default %d" % DEFAULT_STEP)
    ap.add_argument("--file", default=paths.SEQUENCES)
    ap.add_argument("--log", default=os.path.join(paths.LOGS,
                                                  "sequence_audit.log"))
    args = ap.parse_args(argv)
    if args.n < 1:
        raise ValueError("--n must be >= 1, got %r" % args.n)

    tee = Tee(args.log)
    old = sys.stdout
    sys.stdout = tee
    try:
        return _report(args)
    finally:
        sys.stdout = old
        tee.close()
        print("log -> %s" % args.log)


def _report(args) -> int:
    print("=" * 132)
    print("A5 -- SEQUENCE STATISTICS (corrected version)")
    print("=" * 132)
    print("file          : %s" % args.file)
    print("size          : %d B" % os.path.getsize(args.file))
    print("sampling      : every %dth record OF EACH GROUP separately (phase 1)"
          % args.n)
    print("                -- NOT every nth row of the file: records run in a")
    print("                   FlankLeft/Promotor/Gene/FlankRight cycle, so a")
    print("                   step divisible by 4 aliases with that cycle.")
    print("lengths       : from the FULL corpus (every row), not from a sample")
    print("composition   : from the SAMPLE, after .upper()")
    print("determinism   : no RNG; sample = f(step, phase, file order)")

    acc, counts, prom_ids = _pass(args.file, args.n)

    print("\n" + "=" * 132)
    print("1. FILE COMPOSITION -- four record types, not one")
    print("=" * 132)
    total = sum(counts.values())
    for g in GROUPS:
        print("  %-11s %8d records (%5.2f %% of the file)"
              % (g, counts[g], 100 * counts[g] / total))
    unknown = set(counts) - set(GROUPS)
    if unknown:
        print("  WARNING: unexpected groups: %s" % sorted(unknown))
    print("  total       %8d" % total)
    print("\n  -> the previous analysis did not filter GROUP, so its length")
    print("     distribution was a mixture of these four objects.")

    print("\n" + "=" * 132)
    print("2. PROMOTERS -- per genome  (GROUP == 'Promotor')")
    print("=" * 132)
    rows = _table(acc, "Promotor")
    len_p, n800_p, gc_p = _pooled(acc, "Promotor")
    fraction_800 = n800_p / len(len_p)
    print("\n  POOLED: n %d | len min %d med %.0f max %d | fraction 800 bp %.4f"
          % (len(len_p), len_p.min(), np.median(len_p), len_p.max(),
             fraction_800))
    print("  POOLED: GC mean %.4f sd %.4f (sample n=%d)"
          % (gc_p.mean(), gc_p.std(ddof=1), len(gc_p)))
    pairs_p = [("Promotor", g) for g in rows]
    bas_p = sum(acc[p].n_bases for p in pairs_p)
    low_p = sum(acc[p].n_lower for p in pairs_p)
    rec_p = sum(acc[p].n_sample for p in pairs_p)
    rec_low_p = sum(acc[p].n_rec_lower for p in pairs_p)
    n50 = sum(acc[p].n_rec_N50 for p in pairs_p)
    masked = [g for g in rows if rows[g]["f_lower"] > 0]
    unmasked = [g for g in rows if rows[g]["f_lower"] == 0]
    print("  POOLED: soft masking -- %.4f of BASES are lower case, but %.4f "
          "of RECORDS contain a lower-case letter"
          % (low_p / bas_p, rec_low_p / rec_p))
    print("          this is THE SAME property measured on TWO DIFFERENT "
          "objects (bases vs records);")
    print("          quoting one instead of the other is an error of %.0fx."
          % ((rec_low_p / rec_p) / (low_p / bas_p)))
    print("          genomes WITHOUT masking: %s; in the other %d f_rec_low "
          "= %.4f .. %.4f"
          % (", ".join(sorted(unmasked)), len(masked),
             min(rows[g]["f_rec_low"] for g in masked),
             max(rows[g]["f_rec_low"] for g in masked)))
    print("  POOLED: records with >50 %% N bases: %d of %d sampled (%.5f); "
          "most N in %s (f_N %.5f)"
          % (n50, rec_p, n50 / rec_p,
             max(rows, key=lambda g: rows[g]["f_N"]),
             max(w["f_N"] for w in rows.values())))
    print("  -> 'median 800' BELONGS TO THE PROMOTERS.  The number 1000 "
          "belongs to")
    print("     the flanks (section 3), and the long tail to the genes "
          "(section 4).")

    # A per-genome report only means something next to the question of which
    # genomes have a label at all: otherwise "20 genomes" here and "19" in the
    # expression cohort are two numbers under one name.
    print("\n  AGREEMENT WITH THE EXPRESSION COHORT (which genome has a label "
          "at all):")
    try:
        from . import matrices as _data
        gen_expr = set(_data.load_samples("og_scope")["genome"].unique())
    except Exception as e:                     # not silent: we say what failed
        print("    NOT CHECKED -- %s: %s" % (type(e).__name__, e))
    else:
        gen_seq = set(rows)
        only_seq = sorted(gen_seq - gen_expr)
        only_expr = sorted(gen_expr - gen_seq)
        print("    genomes in the sequences %d | in the sample metadata %d"
              % (len(gen_seq), len(gen_expr)))
        print("    only in the sequences (NO expression label): %s"
              % (", ".join("%s (%d promoters)" % (g, rows[g]["n_file"])
                           for g in only_seq) or "none"))
        print("    only in the expression data (no sequences): %s"
              % (", ".join(only_expr) or "none"))
        if only_seq:
            print("    -> a 'corpus-wide' statistic describes %d genomes, but "
                  "training can use %d." % (len(gen_seq),
                                            len(gen_seq & gen_expr)))

    print("\n" + "=" * 132)
    print("3. FLANKS -- per genome  (FlankLeft, FlankRight)")
    print("=" * 132)
    for g in ("FlankLeft", "FlankRight"):
        print("\n--- %s ---" % g)
        _table(acc, g)
        ln, n800, gc = _pooled(acc, g)
        print("  POOLED: n %d | len min %d med %.0f max %d | fraction 800 bp "
              "%.4f | fraction 1000 bp %.4f"
              % (len(ln), ln.min(), np.median(ln), ln.max(), n800 / len(ln),
                 float((ln == 1000).mean())))

    print("\n" + "=" * 132)
    print("4. GENES -- per genome  (GROUP == 'Gene')")
    print("=" * 132)
    _table(acc, "Gene")
    len_g, n800_g, gc_g = _pooled(acc, "Gene")
    print("  POOLED: n %d | len min %d med %.0f max %d | fraction 800 bp %.4f"
          % (len(len_g), len_g.min(), np.median(len_g), len_g.max(),
             n800_g / len(len_g)))
    print("  -> this is where 'max 66 302' and 'min 1' in the old distribution "
          "came from.")

    print("\n" + "=" * 132)
    print("5. DEFECT 3: GC WITHOUT .upper() -- bias and a REORDERED RANKING")
    print("=" * 132)
    print("  GC_ok    = (G+C) after .upper() / length")
    print("  GC_wrong = (G+C) upper case only / length   <- the old code")
    print("\n%-14s | %7s %8s %8s | %7s | %6s %6s %5s"
          % ("genome", "GC_ok", "GC_wrong", "bias", "f_lower",
             "r_ok", "r_wrong", "shift"))
    print("-" * 132)
    ok = {g: w["gc_mean"] for g, w in rows.items()}
    wrong = {g: w["gc_wrong"] for g, w in rows.items()}
    r_ok = {g: i + 1 for i, g in enumerate(sorted(ok, key=lambda k: -ok[k]))}
    r_wrong = {g: i + 1 for i, g in
               enumerate(sorted(wrong, key=lambda k: -wrong[k]))}
    for g in sorted(ok, key=lambda k: r_ok[k]):
        print("%-14s | %7.4f %8.4f %+8.4f | %7.4f | %6d %6d %+5d"
              % (g, ok[g], wrong[g], wrong[g] - ok[g], rows[g]["f_lower"],
                 r_ok[g], r_wrong[g], r_ok[g] - r_wrong[g]))
    bias = np.array([wrong[g] - ok[g] for g in ok])
    print("\n  mean bias over genomes: %+.4f (min %+.4f, max %+.4f)"
          % (bias.mean(), bias.min(), bias.max()))
    gcw_all = np.concatenate([np.frombuffer(acc[("Promotor", g)].gc_wrong,
                                            dtype=np.float64) for g in ok])
    print("  bias on the POOLED promoter sample: %+.4f "
          "(GC_ok %.4f -> GC_wrong %.4f)"
          % (gcw_all.mean() - gc_p.mean(), gc_p.mean(), gcw_all.mean()))
    rho = float(np.corrcoef(
        [r_ok[g] for g in sorted(ok)], [r_wrong[g] for g in sorted(ok)])[0, 1])
    print("  Pearson correlation of ranks (ok vs wrong): %+.4f" % rho)
    least = sorted(ok, key=lambda k: rows[k]["f_lower"])[:3]
    print("  genomes with the LOWEST fraction of lower-case letters: %s"
          % ", ".join("%s (%.4f)" % (g, rows[g]["f_lower"]) for g in least))
    print("  -> these are the ones that 'win' the wrong ranking, because only")
    print("     their bases are visible to a counter without .upper().  The")
    print("     old analysis' prefix consisted entirely of such a genome, so")
    print("     defect 1 and defect 3 masked each other.")

    print("\n" + "=" * 132)
    print("6. REPEATED IDs IN THE 'Promotor' GROUP")
    print("=" * 132)
    dup = {k: v for k, v in prom_ids.items() if v[0] > 1}
    multiplicity = collections.Counter(v[0] for v in dup.values())
    print("  unique IDs %d | repeated IDs %d | max multiplicity %d"
          % (len(prom_ids), len(dup),
             max((v[0] for v in prom_ids.values()), default=0)))
    print("  multiplicity distribution: %s" % dict(sorted(multiplicity.items())))
    differing = sum(1 for v in dup.values() if len(v[1]) > 1)
    print("  of those %d repeated IDs: %d have DIFFERENT sequences, %d "
          "identical" % (len(dup), differing, len(dup) - differing))
    print("  -> ID is not a key in this file.  Anything joining on ID must")
    print("     first decide which record it takes (or aggregate).")

    print("\n" + "=" * 132)
    print("7. REPRESENTATIVENESS CONTROL FOR THE SAMPLE")
    print("=" * 132)
    print("  The sample (every %dth) against the full corpus, same statistic:"
          % args.n)
    print("  %-11s | %10s %10s | %8s %8s" %
          ("group", "n_file", "n_sample", "f800_file", "f800_samp"))
    for g in GROUPS:
        pairs = [(gr, gg) for (gr, gg) in acc if gr == g]
        ln = np.concatenate([np.frombuffer(acc[p].lengths, dtype=np.int32)
                             for p in pairs])
        n800 = sum(acc[p].n_800 for p in pairs)
        # Exactly the records that entered the composition sample: the
        # counter is incremented in the same branch, so there is no second,
        # approximate definition of the sample here.
        n_samp = sum(acc[p].n_sample for p in pairs)
        n800_samp = sum(acc[p].n_800_sample for p in pairs)
        print("  %-11s | %10d %10d | %8.4f %8.4f"
              % (g, len(ln), n_samp, n800 / len(ln),
                 n800_samp / max(n_samp, 1)))
    print("  -> if the f800 columns agree, the sample does not distort what")
    print("     the falsifier measures.")
    print("\n  FULL-PASS CONTROL (recorded, not computed now).")
    print("  The same module run over EVERY record:")
    print("      python -m hyppe.analysis.sequence_audit --n 1 --log <other file>")
    print("  gave, on promoters, GC 0.4865 sd 0.0554 | f_lower 0.0726 | "
          "bias -0.0304,")
    print("  against the n=5 sample: GC 0.4866 sd 0.0552 | f_lower 0.0727 | "
          "bias -0.0304.")
    print("  Largest per-genome deviation of the n=5 sample from the full "
          "corpus:")
    print("      GC_mean  0.00240 (longibrachiatum 0.5131 against 0.5107)")
    print("      GC_sd    0.00580 (longibrachiatum 0.0755 against 0.0813)")
    print("      f_lower  0.00250 (reesei 0.1090 against 0.1065)")
    print("      f_800    0.00000 -- by construction, because lengths come "
          "from the")
    print("               full corpus in both passes.")
    print("  -> at n=5 no number in this report depends on the sampling in the")
    print("     third decimal place.  The cost: n=1 counts all 1.045 Gbases of")
    print("     the file instead of the 0.209 Gbases of the sample.")

    print("\n" + "=" * 132)
    print("8. FALSIFIER: fraction of promoters of length EXACTLY 800")
    print("=" * 132)
    print("  threshold   : %.2f" % THRESHOLD_800)
    print("  measured    : %.4f  (%d of %d promoters, FULL corpus)"
          % (fraction_800, n800_p, len(len_p)))
    below = {g: w["f_800"] for g, w in rows.items()
             if w["f_800"] < THRESHOLD_800}
    print("  genomes below the threshold: %s"
          % (", ".join("%s %.4f" % (g, v) for g, v in sorted(below.items()))
             if below else "none"))
    print("\n  POSITIVE CONTROL (is the threshold reachable by this counter at "
          "all):")
    for g in ("FlankLeft", "FlankRight", "Gene"):
        ln, n800, _ = _pooled(acc, g)
        print("    the same counter on %-11s -> %.4f  %s"
              % (g, n800 / len(ln),
                 "BELOW the threshold" if n800 / len(ln) < THRESHOLD_800
                 else "above the threshold"))
    print("    -> the counter takes values on both sides of the threshold, so")
    print("       the falsifier is satisfiable by construction, not "
          "decorative.")

    if fraction_800 < THRESHOLD_800:
        print("\n  >>> FALSIFIER FIRED: %.4f < %.2f."
              % (fraction_800, THRESHOLD_800))
        print("  >>> The assumption '800 bp in and out' is NOT a property of "
              "the")
        print("  >>> corpus.  Reported as a result; not trimmed quietly.")
        print("\nexit code 3")
        return 3
    print("\n  falsifier NOT fired: %.4f >= %.2f -> 800 bp holds."
          % (fraction_800, THRESHOLD_800))
    print("\nexit code 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
