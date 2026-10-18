"""Where does the "Promotor" fragment in final_datasets end -- at the TSS or at ATG?

The question blocks design work: if the fragment ends at the START codon then
its 3' end contains the 5'UTR, and every ATG sitting there is a candidate uORF
that suppresses GFP translation at an unchanged mRNA level -- which is exactly
where a sequence->expression model is blind.  If it ends at the TSS, none of
those ATGs reaches the transcript and a uORF scan does not apply.

Settling it needs no literature, because `sequences.tsv.gz` carries **four rows
per gene** (FlankLeft / Promotor / Gene / FlankRight).  If the `Gene` row is a
CDS (starts with ATG, length divisible by 3, ends in a stop codon), then
`Promotor` is the window immediately before the START codon, not before the TSS.

Sections, selected with --section:
  structure    -- lengths per GROUP; is Gene a CDS
  atg          -- distribution of the last ATG, ATG at the very end, ATG density
                  in windows along 800 bp + a trinucleotide control
  gc           -- GC profile in windows along 800 bp (Promotor, Gene, FlankLeft)
  expression   -- ATG in the last 200 bp vs expression rank (P1)
  target       -- target.fasta against the data
  all

Run with:
  python -m hyppe.analysis.promoter_boundary --section all
"""
from __future__ import annotations

import argparse
import gzip
import os

import numpy as np

from . import matrices as pdata
from . import paths

STOPS = ("TAA", "TAG", "TGA")


def _find_all(s: str, pattern: str) -> list[int]:
    out, i = [], s.find(pattern)
    while i != -1:
        out.append(i)
        i = s.find(pattern, i + 1)
    return out


def _rows(path, limit=None):
    """Stream of dicts; local, because `stream_sequences` yields a tuple and
    gives no access to the header.

    SEQUENCE is upper-cased.  The data are soft-masked: 7.25 % of promoter
    bases are lower case and the masked share rises towards the 3' end
    (6.8 % -> 9.8 % in the 720..759 window), so a case-sensitive search for
    "ATG" would manufacture an apparent depletion exactly where this script
    makes its claim.

    Rows with fewer fields than the header are skipped and COUNTED, and the
    count is printed when the stream ends: every claim here is a count of
    rows per GROUP, so a silent drop moves the denominator of the claim.
    """
    short = 0
    n_fields = 0
    try:
        with gzip.open(paths.require(path, "sequences.tsv.gz"), "rt") as fh:
            h = fh.readline().rstrip("\n").split("\t")
            pos = {n: i for i, n in enumerate(h)}
            i_seq = pos["SEQUENCE"]
            n_fields = len(h)
            for n, line in enumerate(fh):
                if limit is not None and n >= limit:
                    break
                p = line.rstrip("\n").split("\t")
                if len(p) < n_fields:
                    short += 1
                    continue
                p[i_seq] = p[i_seq].upper()
                yield {k: p[i] for k, i in pos.items()}
    finally:
        # In `finally`, so an early `break` by the caller reports too.
        if short:
            print("  [_rows] skipped %d malformed row(s) (fewer than %d "
                  "fields)" % (short, n_fields))


def section_structure(limit=None):
    print("=" * 72)
    print("A. STRUCTURE: lengths per GROUP, and whether the Gene row is a CDS")
    print("=" * 72)
    lengths = {}
    gene_atg = gene_tot = gene_mod3 = gene_stop = 0
    gene_len = []
    prom_tot = prom_800 = 0
    genomes = {}
    for r in _rows(paths.SEQUENCES, limit):
        g, s = r["GROUP"], r["SEQUENCE"]
        d = lengths.setdefault(g, {})
        d[len(s)] = d.get(len(s), 0) + 1
        genomes[r["GENOME"]] = genomes.get(r["GENOME"], 0) + 1
        if g == "Gene":
            gene_tot += 1
            gene_len.append(len(s))
            if s[:3] == "ATG":
                gene_atg += 1
            if len(s) % 3 == 0:
                gene_mod3 += 1
            if s[-3:] in STOPS:
                gene_stop += 1
        elif g == "Promotor":
            prom_tot += 1
            if len(s) == 800:
                prom_800 += 1

    print("\nGROUP -> most frequent lengths (top 3) and row count")
    for g, d in sorted(lengths.items()):
        n = sum(d.values())
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        desc = ", ".join("%d bp: %d (%.4f)" % (L, c, c / n) for L, c in top)
        print("  %-18s n=%7d  distinct lengths=%6d  | %s"
              % (g, n, len(d), desc))

    print("\nThe Gene row as a CDS  (n=%d):" % gene_tot)
    if gene_tot:
        print("  starts with 'ATG'            : %7d / %7d = %.6f"
              % (gene_atg, gene_tot, gene_atg / gene_tot))
        print("  length divisible by 3        : %7d / %7d = %.6f"
              % (gene_mod3, gene_tot, gene_mod3 / gene_tot))
        print("  ends in a stop codon         : %7d / %7d = %.6f"
              % (gene_stop, gene_tot, gene_stop / gene_tot))
        gl = np.array(gene_len)
        print("  Gene length: min=%d  median=%d  max=%d"
              % (gl.min(), int(np.median(gl)), gl.max()))
    print("\nPromotor exactly 800 bp: %d / %d = %.6f"
          % (prom_800, prom_tot, prom_800 / max(prom_tot, 1)))
    print("\ngenomes in the file: %d  ->  %s" % (len(genomes), sorted(genomes)))
    return lengths


def section_atg(limit=None, window=20):
    print("=" * 72)
    print("B. ATG: where the last ATG sits, what is at the very end, ATG profile")
    print("=" * 72)
    nb = 800 // window
    atg_bin = np.zeros(nb, dtype=np.int64)      # promoter
    cat_bin = np.zeros(nb, dtype=np.int64)      # control: CAT (ATG on the - strand)
    ttg_bin = np.zeros(nb, dtype=np.int64)      # control: TTG, close in composition
    gene_atg_bin = np.zeros(40, dtype=np.int64)  # first 800 bp of the CDS, window 20
    gene_n = 0
    last = []
    ends_atg = 0
    atg_in_last = {1: 0, 3: 0, 10: 0, 30: 0}
    n = 0
    no_atg = 0
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] == "Gene":
            s = r["SEQUENCE"]
            if len(s) >= 800:
                for i in _find_all(s[:800], "ATG"):
                    gene_atg_bin[i // 20] += 1
                gene_n += 1
            continue
        if r["GROUP"] != "Promotor":
            continue
        s = r["SEQUENCE"]
        if len(s) != 800:
            continue
        n += 1
        at = _find_all(s, "ATG")
        for i in at:
            atg_bin[i // window] += 1
        for i in _find_all(s, "CAT"):
            cat_bin[i // window] += 1
        for i in _find_all(s, "TTG"):
            ttg_bin[i // window] += 1
        if not at:
            no_atg += 1
        else:
            last.append(at[-1])
        if s[-3:] == "ATG":
            ends_atg += 1
        for k in atg_in_last:
            # is there an ATG starting within the last k positions
            if any(p >= 800 - 2 - k for p in at):
                atg_in_last[k] += 1

    o = np.array(last)
    print("\n800 bp promoters: %d   (with no ATG at all: %d = %.5f)"
          % (n, no_atg, no_atg / max(n, 1)))
    print("\nDIRECT TEST -- does the fragment end on a START codon:")
    print("  last three bases == 'ATG' : %d / %d = %.6f"
          % (ends_atg, n, ends_atg / max(n, 1)))
    print("  expectation under a random ACGT composition ~ 1/64 = 0.015625")
    for k in sorted(atg_in_last):
        print("  an ATG starts within the last %2d bp : %.6f"
              % (k, atg_in_last[k] / max(n, 1)))
    print("\nPosition of the LAST ATG (0 = start of fragment, 797 = maximum):")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print("  p%02d = %6.1f" % (q, np.percentile(o, q)))
    print("  mean = %.1f   median = %.1f   max = %d"
          % (o.mean(), np.median(o), o.max()))
    print("  distance from the end (797 - position): median = %.1f"
          % (797 - np.median(o)))

    print("\nATG PROFILE along the promoter, window %d bp (per sequence):"
          % window)
    print("  bin   from..to   ATG/seq   CAT/seq   TTG/seq   ATG/CAT")
    for b in range(nb):
        a, c, t = atg_bin[b] / n, cat_bin[b] / n, ttg_bin[b] / n
        print("  %3d  %3d..%3d   %7.4f   %7.4f   %7.4f   %6.3f"
              % (b, b * window, (b + 1) * window - 1, a, c, t,
                 a / max(c, 1e-9)))
    middle = slice(0, nb - 10)   # everything except the last 200 bp
    tail = slice(nb - 10, nb)
    a_mid = atg_bin[middle].sum() / (atg_bin[middle].size * n)
    a_tail = atg_bin[tail].sum() / (atg_bin[tail].size * n)
    c_mid = cat_bin[middle].sum() / (cat_bin[middle].size * n)
    c_tail = cat_bin[tail].sum() / (cat_bin[tail].size * n)
    print("\n  ATG per window: first 600 bp = %.4f | last 200 bp = %.4f"
          "  (ratio %.4f)" % (a_mid, a_tail, a_tail / a_mid))
    print("  CAT per window: first 600 bp = %.4f | last 200 bp = %.4f"
          "  (ratio %.4f)" % (c_mid, c_tail, c_tail / c_mid))
    if gene_n:
        print("\nPOSITIVE CONTROL -- ATG profile in the first 800 bp of the "
              "CDS (n=%d), window 20 bp, first 10 windows:" % gene_n)
        for b in range(10):
            print("  %3d  %3d..%3d   ATG/seq=%.4f"
                  % (b, b * 20, b * 20 + 19, gene_atg_bin[b] / gene_n))
    return atg_bin, cat_bin, n


def section_gc(limit=None, window=20):
    print("=" * 72)
    print("C. GC PROFILE along the fragments")
    print("=" * 72)
    acc = {}
    count = {}
    dl = {"Promotor": 800, "Gene": 800, "FlankLeft": 1000, "FlankRight": 1000}
    for r in _rows(paths.SEQUENCES, limit):
        g, s = r["GROUP"], r["SEQUENCE"]
        if g not in dl or len(s) < dl[g]:
            continue
        s = s[:dl[g]] if g != "Promotor" else s[:800]
        if g == "Promotor" and len(r["SEQUENCE"]) != 800:
            continue
        b = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        gc = ((b == 71) | (b == 67)).astype(np.float64)
        nb = dl[g] // window
        v = gc[:nb * window].reshape(nb, window).mean(axis=1)
        if g not in acc:
            acc[g] = np.zeros(nb)
            count[g] = 0
        acc[g] += v
        count[g] += 1
    for g in ("FlankLeft", "Promotor", "Gene", "FlankRight"):
        if g not in acc:
            continue
        p = acc[g] / count[g]
        print("\n%s (n=%d), window %d bp:" % (g, count[g], window))
        s = "  "
        for i, v in enumerate(p):
            s += "%5.1f" % (100 * v)
            if (i + 1) % 10 == 0:
                s += "\n  "
        print(s)
        if g == "Promotor":
            print("  GC first 640 bp = %.2f%%   last 160 bp = %.2f%%"
                  "   delta = %+.2f pts"
                  % (100 * p[:-160 // window].mean(),
                     100 * p[-160 // window:].mean(),
                     100 * (p[-160 // window:].mean()
                            - p[:-160 // window].mean())))
            print("  GC last 40 bp  = %.2f%%   GC first 40 bp = %.2f%%"
                  % (100 * p[-40 // window:].mean(),
                     100 * p[:40 // window].mean()))
    return acc, count


def _mean_rank_p1():
    """Mean expression rank per gene for genome P1 (all_genes_scope/rank)."""
    p = paths.matrix_path("all_genes_scope", "rank")
    out = {}
    with gzip.open(p, "rt") as fh:
        h = fh.readline().rstrip("\n").split("\t")
        meta = [c for c in h if c in pdata.KNOWN_META]
        nm = len(meta)
        i_gen = h.index("genome")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[i_gen] != "P1":
                continue
            vals = [v for v in parts[nm:] if v != ""]
            if not vals:
                continue
            a = np.array(vals, dtype=np.float64)
            out[parts[0]] = (a.mean(), a.size)
    return out


def section_expression(limit=None, tail=200):
    print("=" * 72)
    print("D. ATG in the last %d bp vs EXPRESSION RANK (genome P1)" % tail)
    print("=" * 72)
    rank = _mean_rank_p1()
    print("P1 genes with a rank: %d" % len(rank))
    v = np.array([x[0] for x in rank.values()])
    print("rank: min=%.4f  p05=%.4f  median=%.4f  p95=%.4f  max=%.4f"
          % (v.min(), np.percentile(v, 5), np.median(v),
             np.percentile(v, 95), v.max()))
    for ref, name in (("P1_G002124", "tef1"),):
        if ref in rank:
            print("  %s (%s): mean rank = %.4f  (n samples %d)"
                  % (ref, name, rank[ref][0], rank[ref][1]))

    ids, n_atg, n_atg_full, y = [], [], [], []
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] != "Promotor" or r["GENOME"] != "P1":
            continue
        s = r["SEQUENCE"]
        if len(s) != 800:
            continue
        rid = r["ID"]
        if rid not in rank:
            continue
        ids.append(rid)
        # an ATG starting within the last `tail` bp of the fragment
        n_atg.append(len(_find_all(s[-tail:], "ATG")))
        n_atg_full.append(len(_find_all(s, "ATG")))
        y.append(rank[rid][0])
    n_atg = np.array(n_atg)
    n_atg_full = np.array(n_atg_full)
    y = np.array(y)
    print("\npaired P1 promoters: %d" % len(y))
    print("ATG in the last %d bp: mean=%.3f  median=%.1f  "
          "fraction with >=1: %.4f" % (tail, n_atg.mean(), np.median(n_atg),
                                       (n_atg >= 1).mean()))
    print("ATG over the whole 800 bp: mean=%.3f  fraction with >=1: %.4f"
          % (n_atg_full.mean(), (n_atg_full >= 1).mean()))

    from scipy import stats
    for name, x in (("ATG in the last %d bp" % tail, n_atg),
                    ("ATG over the whole 800 bp", n_atg_full)):
        rho, p = stats.spearmanr(x, y)
        print("\n%s vs rank:  Spearman rho = %+.4f  (p = %.3g)"
              % (name, rho, p))
    # control: GC of the last 200 bp (a known confounder of ATG)
    print("\nRank quintiles -> mean number of ATG in the last %d bp:" % tail)
    q = np.quantile(y, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (y >= q[i]) & (y <= q[i + 1] if i == 4 else y < q[i + 1])
        print("  quintile %d  rank %.3f..%.3f  n=%6d  ATG=%.3f  "
              "fraction>=1 = %.4f"
              % (i + 1, q[i], q[i + 1], m.sum(), n_atg[m].mean(),
                 (n_atg[m] >= 1).mean()))
    # Mind the sign: `rank` decreases with expression (section D0 measures
    # Spearman(rank, tpm) = -0.9844), so a low rank is a strong promoter and
    # quintile 1 above is the strongest.
    strong = y <= np.percentile(y, 5)
    weak = y >= np.percentile(y, 95)
    print("\n  STRONGEST 5%% (rank<=p05): ATG=%.3f   "
          "WEAKEST 5%% (rank>=p95): ATG=%.3f"
          % (n_atg[strong].mean(), n_atg[weak].mean()))
    return n_atg, y


def section_target(limit=None):
    print("=" * 72)
    print("E. target.fasta against the data")
    print("=" * 72)
    fp = paths.require(paths.TARGET_FASTA, "target.fasta (800 bp pks1 promoter)")
    header, seq = None, []
    for ln in open(fp):
        if ln.startswith(">"):
            header = ln.strip()
        else:
            seq.append(ln.strip())
    seq = "".join(seq).upper()
    print("header  : %s" % header)
    print("length  : %d bp" % len(seq))
    print("last 30 bp: %s" % seq[-30:])
    print("last 3 bp : %r   (is it 'ATG'? %s)" % (seq[-3:], seq[-3:] == "ATG"))
    # coordinates from the header
    try:
        coord = header[1:].split("/")[0]
        sc, rng = coord.split(":")
        a, b = (int(x) for x in rng.split("-"))
        print("coordinates: %s %d-%d  ->  length from header = %d bp"
              % (sc, a, b, b - a + 1))
    except Exception as e:
        print("could not parse the coordinates: %r" % e)
    at = _find_all(seq, "ATG")
    print("ATG total: %d, last one at position %d (%d bp from the end)"
          % (len(at), at[-1], len(seq) - at[-1]))
    print("ATG in the last 200 bp: %d" % len(_find_all(seq[-200:], "ATG")))

    # is this sequence in the corpus as a Promotor?
    hit = None
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] != "Promotor":
            continue
        if r["SEQUENCE"].upper() == seq:
            hit = r
            break
    if hit:
        print("\nEXACT HIT in sequences.tsv.gz: ID=%s GENOME=%s "
              "OG=%s GENE_ID=%s" % (hit["ID"], hit["GENOME"],
                                    hit["OG_ID"], hit["GENE_ID"]))
        for r in _rows(paths.SEQUENCES, limit):
            if r["ID"] == hit["ID"] and r["GROUP"] == "Gene":
                s = r["SEQUENCE"]
                print("  Gene row of the same ID: length=%d  "
                      "first 12 bp=%r  last 3 bp=%r"
                      % (len(s), s[:12], s[-3:]))
                print("  junction Promotor[-9:] + Gene[:9] = %r"
                      % (seq[-9:] + s[:9]))
                break
    else:
        print("\nNO exact hit (searched up to the limit).")
        # partial: longest shared suffix/prefix with P1 promoters
        best = (0, None)
        for r in _rows(paths.SEQUENCES, limit):
            if r["GROUP"] != "Promotor" or r["GENOME"] != "P1":
                continue
            s = r["SEQUENCE"].upper()
            if s[-60:] == seq[-60:]:
                best = (60, r)
                break
        if best[1]:
            print("  but the last 60 bp match ID=%s" % best[1]["ID"])
    return seq


def section_orientation():
    """Does the `rank` column rise with expression, or fall?

    Measured by correlating against `tpm` and `counts` on the same genes and
    samples, not inferred from the description.  Without it the sign of the
    ATG-expression correlation in section D is a guess.
    """
    print("=" * 72)
    print("D0. ORIENTATION OF THE `rank` COLUMN -- measured, not assumed")
    print("=" * 72)
    from scipy import stats

    def means(name):
        # Not via paths.matrix_path: `tpm` exists on disk but is not in
        # paths.MATRICES, so matrix_path rejects it.
        p = paths.require(
            os.path.join(paths.FINAL, "all_genes_scope", name + ".tsv.gz"),
            "all_genes_scope/%s.tsv.gz" % name)
        out = {}
        with gzip.open(p, "rt") as fh:
            h = fh.readline().rstrip("\n").split("\t")
            nm = len([c for c in h if c in pdata.KNOWN_META])
            ig = h.index("genome")
            for line in fh:
                q = line.rstrip("\n").split("\t")
                if q[ig] != "P1":
                    continue
                v = [x for x in q[nm:] if x != ""]
                if v:
                    out[q[0]] = float(np.mean(np.array(v, dtype=np.float64)))
        return out

    r = means("rank")
    a_all = np.array(list(r.values()))
    for ref_name in ("tpm", "counts"):
        t = means(ref_name)
        common = sorted(set(r) & set(t))
        a = np.array([r[k] for k in common])
        b = np.array([t[k] for k in common])
        rho, p = stats.spearmanr(a, b)
        print("P1 genes shared by rank/%s: %d" % (ref_name, len(common)))
        print("Spearman(rank, %s) = %+.4f   (p = %.3g)" % (ref_name, rho, p))
        for ref, name in (("P1_G002124", "tef1"),):
            if ref in r and ref in t:
                print("  %s (%s): rank=%.4f (percentile %.2f%%)  %s=%.2f "
                      "(percentile %.2f%%)"
                      % (ref, name, r[ref], 100 * float((a < r[ref]).mean()),
                         ref_name, t[ref], 100 * float((b < t[ref]).mean())))
    print("\n  rho < 0  ->  a LOW `rank` value = HIGH expression "
          "(the column is descending)")
    print("  rho > 0  ->  a high `rank` value = high expression")
    print("  rank distribution (P1, n=%d): p05=%.4f median=%.4f p95=%.4f"
          % (len(a_all), np.percentile(a_all, 5), np.median(a_all),
             np.percentile(a_all, 95)))
    return r


def section_controls(limit=None, tail=200):
    """Is the ATG-expression signal a uORF, or just base composition?

    Three controls, each able to kill it: CAT, the same trinucleotide on the
    opposite strand with identical mononucleotide composition and no
    translational meaning; ATG in the first 200 bp, the same motif in a
    region almost certainly outside the transcript; and GC, a known
    confounder of both, stratified by deciles of tail GC.
    """
    print("=" * 72)
    print("D1. CONTROLS: is ATG in the last %d bp a uORF or composition" % tail)
    print("=" * 72)
    from scipy import stats
    rank = _mean_rank_p1()
    rec = []
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] != "Promotor" or r["GENOME"] != "P1":
            continue
        s = r["SEQUENCE"]
        if len(s) != 800 or r["ID"] not in rank:
            continue
        tl = s[-tail:]
        hd = s[:tail]
        b = np.frombuffer(tl.encode("ascii"), dtype=np.uint8)
        rec.append((len(_find_all(tl, "ATG")),
                    len(_find_all(tl, "CAT")),
                    len(_find_all(hd, "ATG")),
                    float(((b == 71) | (b == 67)).mean()),
                    rank[r["ID"]][0]))
    A = np.array(rec, dtype=np.float64)
    atg, cat, atg_head, gc, y = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    print("n = %d P1 promoters\n" % len(y))
    print("  feature                        mean      Spearman vs rank")
    for name, x in (("ATG in the last %d bp" % tail, atg),
                    ("CAT in the last %d bp   [control]" % tail, cat),
                    ("ATG in the FIRST %d bp  [control]" % tail, atg_head),
                    ("GC in the last %d bp    [confounder]" % tail, gc)):
        rho, p = stats.spearmanr(x, y)
        print("  %-32s %7.3f   %+.4f  (p=%.3g)" % (name, x.mean(), rho, p))

    print("\nStratified by tail-GC deciles (rho ATG vs rank WITHIN a decile):")
    q = np.quantile(gc, np.linspace(0, 1, 11))
    rhos = []
    for i in range(10):
        m = (gc >= q[i]) & ((gc <= q[i + 1]) if i == 9 else (gc < q[i + 1]))
        if m.sum() < 50:
            continue
        rho, _ = stats.spearmanr(atg[m], y[m])
        rhoc, _ = stats.spearmanr(cat[m], y[m])
        rhos.append(rho)
        print("  decile %2d  GC %.3f..%.3f  n=%5d   rho(ATG)=%+.4f   "
              "rho(CAT)=%+.4f" % (i + 1, q[i], q[i + 1], m.sum(), rho, rhoc))
    print("  mean rho(ATG) within GC deciles = %+.4f" % float(np.mean(rhos)))
    return A


def section_frame(limit=None, tail=200):
    """A falsifier with zero degrees of freedom in base composition.

    If the end of the fragment lies inside the transcript and is scanned by
    the ribosome, an ATG out of frame with the CDS is far worse than an
    in-frame one -- it opens an ORF overlapping the gene rather than
    extending the protein by a few residues -- so selection should strip
    frames 1 and 2 harder than frame 0.  Composition does not know about
    frame, so the asymmetry cannot come from it; the CAT control is computed
    the same way.  If the fragment ends at the TSS, all three frames must
    come out equal.
    """
    print("=" * 72)
    print("F. FRAME relative to the START codon (last %d bp of the fragment)"
          % tail)
    print("=" * 72)
    atg_f = np.zeros(3, dtype=np.int64)
    cat_f = np.zeros(3, dtype=np.int64)
    atg_f_far = np.zeros(3, dtype=np.int64)   # control: first 200 bp
    cat_f_far = np.zeros(3, dtype=np.int64)
    # An ATG with no stop codon before the end of the fragment is an ORF
    # running into the gene.
    through = np.zeros(3, dtype=np.int64)
    stopped = np.zeros(3, dtype=np.int64)
    # The same accounting for CAT, as the null for "does this position and
    # frame reach the end without a stop codon".  Without it the read-through
    # fraction compares frames of differing stop density.
    through_c = np.zeros(3, dtype=np.int64)
    stopped_c = np.zeros(3, dtype=np.int64)
    n = 0
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] != "Promotor":
            continue
        s = r["SEQUENCE"]
        if len(s) != 800:
            continue
        n += 1
        for i in _find_all(s, "ATG"):
            d = 800 - i           # distance to the 'A' of the START codon
            f = d % 3
            if i >= 800 - tail:
                atg_f[f] += 1
                stop = None
                for j in range(i + 3, 798, 3):
                    if s[j:j + 3] in STOPS:
                        stop = j
                        break
                if stop is None:
                    through[f] += 1
                else:
                    stopped[f] += 1
            elif i < tail:
                atg_f_far[f] += 1
        for i in _find_all(s, "CAT"):
            f = (800 - i) % 3
            if i >= 800 - tail:
                cat_f[f] += 1
                stop = None
                for j in range(i + 3, 798, 3):
                    if s[j:j + 3] in STOPS:
                        stop = j
                        break
                if stop is None:
                    through_c[f] += 1
                else:
                    stopped_c[f] += 1
            elif i < tail:
                cat_f_far[f] += 1

    print("\nn = %d promoters of 800 bp" % n)
    print("\nframe  meaning                   ATG      CAT   ATG/CAT")
    desc = {0: "IN FRAME with the CDS", 1: "shifted +1", 2: "shifted +2"}
    for f in range(3):
        print("  %d    %-22s %8d %8d    %.4f"
              % (f, desc[f], atg_f[f], cat_f[f],
                 atg_f[f] / max(cat_f[f], 1)))
    ratio = np.array([atg_f[f] / max(cat_f[f], 1) for f in range(3)])
    print("\n  ratio frame0 / mean(frame1, frame2) = %.4f"
          % (ratio[0] / max(ratio[1:].mean(), 1e-9)))
    print("\nCONTROL -- the same frames in the FIRST %d bp (a region outside "
          "the transcript):" % tail)
    for f in range(3):
        print("  %d    %-22s %8d %8d    %.4f"
              % (f, desc[f], atg_f_far[f], cat_f_far[f],
                 atg_f_far[f] / max(cat_f_far[f], 1)))
    ratio_far = np.array([atg_f_far[f] / max(cat_f_far[f], 1)
                          for f in range(3)])
    print("\n  ratio frame0 / mean(frame1, frame2) = %.4f"
          % (ratio_far[0] / max(ratio_far[1:].mean(), 1e-9)))
    print("\nFate of an ORF opened by an ATG in the last %d bp "
          "(null = the same positions counted for CAT):" % tail)
    print("  frame   ATG through  fraction | CAT through  fraction | "
          "ATG/CAT for read-throughs only")
    for f in range(3):
        tot = through[f] + stopped[f]
        totc = through_c[f] + stopped_c[f]
        print("    %d     %8d   %.4f  | %8d   %.4f  |  %.4f"
              % (f, through[f], through[f] / max(tot, 1),
                 through_c[f], through_c[f] / max(totc, 1),
                 through[f] / max(through_c[f], 1)))
    ratio_thru = np.array([through[f] / max(through_c[f], 1) for f in range(3)])
    print("\n  ONLY ORFs running into the gene -- here frame 0 is an N-terminal")
    print("  extension (mild), while frames 1 and 2 overlap the gene (severe).")
    print("  ratio frame0 / mean(frame1, frame2) = %.4f"
          % (ratio_thru[0] / max(ratio_thru[1:].mean(), 1e-9)))
    return atg_f, cat_f


def section_mld():
    """Test the claim in the `hyppe/safety.py` docstring on its own object.

    safety.py justifies `tss_at_end=True` by "measured on ML_Dataset the TSS
    is at the 3' end (GC rises 50.5 % -> 52.4 % across the final 160 bp)".
    Checked here on `cache/ml_dataset_800.npz`, against final_datasets.
    """
    print("=" * 72)
    print("G. ML_Dataset -- does the number in the safety.py docstring hold")
    print("=" * 72)
    p = paths.ML_DATASET_800
    if not os.path.exists(p):
        print("missing %s" % p)
        return
    z = np.load(p, allow_pickle=True)
    print("keys: %s" % list(z.keys()))
    for k in z.keys():
        a = z[k]
        print("  %s: shape=%s dtype=%s" % (k, getattr(a, "shape", None), a.dtype))
    idx = z["sequences"]
    if idx.ndim != 2 or idx.shape[1] != 800:
        print("unexpected shape %s" % (idx.shape,))
        return
    # hyppe/dna.py:20  NUCLEOTIDE_ORDER = "ACTG"  ->  0=A 1=C 2=T 3=G
    A_, C_, T_, G_ = 0, 1, 2, 3
    N = idx.shape[0]
    print("\n%d sequences x 800, index encoding per hyppe/dna.py:20 "
          "('ACTG')" % N)
    print("  base shares: A=%.4f C=%.4f T=%.4f G=%.4f"
          % tuple((idx == b).mean() for b in (A_, C_, T_, G_)))
    gc = ((idx == C_) | (idx == G_)).mean(axis=0)
    window = 20
    pr = gc.reshape(800 // window, window).mean(axis=1)
    s = "  "
    for i, v in enumerate(pr):
        s += "%5.1f" % (100 * v)
        if (i + 1) % 10 == 0:
            s += "\n  "
    print("\nGC profile, window %d bp:" % window)
    print(s)
    print("  GC first 640 bp = %.2f%%  last 160 bp = %.2f%%  "
          "delta = %+.2f pts"
          % (100 * pr[:-8].mean(), 100 * pr[-8:].mean(),
             100 * (pr[-8:].mean() - pr[:-8].mean())))
    print("  the safety.py:36 docstring says: 50.5%% -> 52.4%%, i.e. +1.9 pts")
    atg = ((idx[:, :-2] == A_) & (idx[:, 1:-1] == T_) & (idx[:, 2:] == G_))
    cat = ((idx[:, :-2] == C_) & (idx[:, 1:-1] == A_) & (idx[:, 2:] == T_))
    ab = atg.sum(axis=0)[:780].reshape(39, 20).sum(axis=1) / N
    cb = cat.sum(axis=0)[:780].reshape(39, 20).sum(axis=1) / N
    print("\nATG per 20 bp window along ML_Dataset (last 20 bp omitted, "
          "the trinucleotide does not fit):")
    print("  first 200 bp : ATG=%.4f  CAT=%.4f  ATG/CAT=%.4f"
          % (ab[:10].mean(), cb[:10].mean(), ab[:10].mean() / cb[:10].mean()))
    print("  last 180 bp  : ATG=%.4f  CAT=%.4f  ATG/CAT=%.4f"
          % (ab[-9:].mean(), cb[-9:].mean(), ab[-9:].mean() / cb[-9:].mean()))
    print("\n  last 5 windows of 20 bp (ATG | CAT):")
    for b in range(34, 39):
        print("    %3d..%3d  %.4f | %.4f" % (b * 20, b * 20 + 19, ab[b], cb[b]))
    # are the last three bases ATG?
    end = ((idx[:, 797] == A_) & (idx[:, 798] == T_) & (idx[:, 799] == G_))
    print("\n  last three bases == ATG: %d / %d = %.6f (chance 1/64 "
          "= 0.0156)" % (end.sum(), N, end.mean()))


def section_mask():
    """How much of the data is soft-masked and where: a precondition for any
    motif counting on this file."""
    print("=" * 72)
    print("H. SOFT-MASKING (lower case) in promoters")
    print("=" * 72)
    n = low = has_any = 0
    prof = np.zeros(800)
    with gzip.open(paths.require(paths.SEQUENCES, "sequences.tsv.gz"),
                   "rt") as fh:
        h = fh.readline().rstrip("\n").split("\t")
        i_s, i_g = h.index("SEQUENCE"), h.index("GROUP")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(h) or p[i_g] != "Promotor":
                continue
            s = p[i_s]
            if len(s) != 800:
                continue
            n += 1
            b = np.frombuffer(s.encode("ascii"), dtype=np.uint8) >= 97
            prof += b
            low += int(b.sum())
            if b.any():
                has_any += 1
    print("800 bp promoters: %d" % n)
    print("fraction of lower-case bases: %.4f" % (low / (800.0 * n)))
    print("promoters with at least one lower-case base: %d = %.4f"
          % (has_any, has_any / n))
    p = prof / n
    print("masking profile, window 40 bp (%):")
    print("  " + " ".join("%.1f" % (100 * p[i:i + 40].mean())
                          for i in range(0, 800, 40)))
    print("\nThis is why `_rows` upper-cases SEQUENCE: the mask thickens "
          "towards the 3' end (6.8 -> 9.8 %), i.e. exactly where we measure "
          "ATG depletion.")


def section_kozak(limit=None):
    """Base composition at positions -5..-1 of the fragment against background.

    This fixes the offset to a single base: if the last base of the fragment
    is at -1 from the 'A' of the START codon, position -3 must carry the
    fungal Kozak context (a preference for A), and a shift of one base in
    either direction would move that peak.
    """
    print("=" * 72)
    print("I. KOZAK CONTEXT at positions -5..-1 of the fragment")
    print("=" * 72)
    import collections
    positions = [795, 796, 797, 798, 799, 400]
    cnt = {k: {p: collections.Counter() for p in positions}
           for k in ("P1", "ALL")}
    gs = {"P1": [0, 0], "ALL": [0, 0]}
    nearest = {"P1": [], "ALL": []}
    for r in _rows(paths.SEQUENCES, limit):
        g, s, gen = r["GROUP"], r["SEQUENCE"], r["GENOME"]
        ks = ["ALL", "P1"] if gen == "P1" else ["ALL"]
        if g == "Gene":
            for k in ks:
                gs[k][1] += 1
                if s[:3] == "ATG":
                    gs[k][0] += 1
            continue
        if g != "Promotor" or len(s) != 800:
            continue
        for k in ks:
            for p in positions:
                cnt[k][p][s[p]] += 1
            pl = _find_all(s, "ATG")
            if pl:
                nearest[k].append(800 - pl[-1])
    for k in ("P1", "ALL"):
        print("\n--- %s ---  the Gene row starts with 'ATG': %d / %d = %.5f"
              % (k, gs[k][0], gs[k][1], gs[k][0] / gs[k][1]))
        for p in positions:
            c = cnt[k][p]
            n = sum(c.values())
            print("   pos %3d (%+d rel. to ATG):  A=%.4f C=%.4f G=%.4f T=%.4f"
                  % (p, p - 800, c["A"] / n, c["C"] / n, c["G"] / n,
                     c["T"] / n))
        a = np.array(nearest[k])
        print("   distance of the NEAREST uATG to the START codon: "
              "median=%.0f p25=%.0f p75=%.0f bp"
              % (np.median(a), np.percentile(a, 25), np.percentile(a, 75)))


def section_design(limit=None):
    """The numbers a panel of 10 designs has to defend itself against.

    The distribution of uATG burden in NATURAL P1 promoters, broken down by
    promoter strength, plus where `target.fasta` sits in that distribution.
    """
    print("=" * 72)
    print("J. uATG BURDEN IN NATURE -- thresholds for the design panel")
    print("=" * 72)
    windows = (800, 200, 150, 100, 50)
    rank = _mean_rank_p1()
    tgt = "".join(l.strip() for l in
                  open(paths.require(paths.TARGET_FASTA, "target.fasta"))
                  if not l.startswith(">")).upper()
    T = [len(_find_all(tgt if w == 800 else tgt[-w:], "ATG")) for w in windows]
    ALL, B10, P1, ids = [], [], [], []
    for r in _rows(paths.SEQUENCES, limit):
        if r["GROUP"] != "Promotor":
            continue
        s = r["SEQUENCE"]
        if len(s) != 800:
            continue
        v = [len(_find_all(s if w == 800 else s[-w:], "ATG")) for w in windows]
        ALL.append(v)
        if r["GENOME"] == "B10":
            B10.append(v)
        if r["GENOME"] == "P1" and r["ID"] in rank:
            P1.append(v + [rank[r["ID"]][0]])
            ids.append(r["ID"])
    ALL, B10, P1 = np.array(ALL), np.array(B10), np.array(P1, dtype=float)
    print("\ntarget.fasta against the distribution of natural promoters:")
    for i, w in enumerate(windows):
        print("  ATG in the last %3d bp: target=%2d | all (n=%d): "
              "mean=%.3f percentile=%.1f%% | B10 (n=%d): percentile=%.1f%%"
              % (w, T[i], len(ALL), ALL[:, i].mean(),
                 100 * (ALL[:, i] < T[i]).mean(), len(B10),
                 100 * (B10[:, i] < T[i]).mean()))
    y = P1[:, len(windows)]
    print("\nP1 by promoter strength (rank DECREASES with expression -- see D0):")
    head = "  %-24s %6s" % ("group", "n")
    for w in windows[1:]:
        head += "  ATG-%-3d zero%%" % w
    print(head)
    groups = (("STRONGEST 10%%", y <= np.percentile(y, 10)),
              ("STRONGEST 1%%", y <= np.percentile(y, 1)),
              ("median +-5%%", (y >= np.percentile(y, 45))
               & (y <= np.percentile(y, 55))),
              ("WEAKEST 10%%", y >= np.percentile(y, 90)))
    for name, m in groups:
        row = "  %-24s %6d" % (name, m.sum())
        for i in range(1, len(windows)):
            row += "   %.3f  %5.1f" % (P1[m, i].mean(),
                                       100 * (P1[m, i] == 0).mean())
        print(row)
    if "P1_G002124" in ids:
        i = ids.index("P1_G002124")
        print("\n  tef1 P1_G002124 (rank %.4f): %s"
              % (P1[i, len(windows)],
                 "  ".join("ATG-%d=%d" % (w, P1[i, j])
                           for j, w in enumerate(windows))))


def section_mld_vs_final(limit=None):
    """Is `TrAtP1_*` from ML_Dataset the same gene as in final_datasets?"""
    print("=" * 72)
    print("K. ML_Dataset against final_datasets, matched on gene identifier")
    print("=" * 72)
    p = paths.ML_DATASET_800
    if not os.path.exists(p):
        print("missing %s" % p)
        return
    z = np.load(p, allow_pickle=True)
    ORDER = "ACTG"          # hyppe/dna.py:20
    COMP = str.maketrans("ACGTN", "TGCAN")
    ml = {str(g): "".join(ORDER[i] if i < 4 else "N" for i in row)
          for g, row in zip(z["gene_ids"], z["sequences"])}
    gi = {}
    for ln in open(paths.require(paths.GENE_INDEX, "gene_index.tsv")):
        q = ln.rstrip("\n").split("\t")
        if len(q) >= 3 and q[1] == "P1":
            gi[q[2]] = q[0]
    fd = {}
    for r in _rows(paths.SEQUENCES, limit):
        if (r["GROUP"] == "Promotor" and r["GENOME"] == "P1"
                and len(r["SEQUENCE"]) == 800):
            fd[r["ID"]] = r["SEQUENCE"]
    eq = rc = other = 0
    rest = []
    for g, s in ml.items():
        t = fd.get(gi.get(g))
        if t is None:
            continue
        if s == t:
            eq += 1
        elif s == t.translate(COMP)[::-1]:
            rc += 1
        else:
            other += 1
            rest.append(max(
                sum(a == b for a, b in zip(s, t)) / 800.0,
                sum(a == b for a, b in zip(s, t.translate(COMP)[::-1])) / 800.0))
    n = eq + rc + other
    print("paired on gene_id: %d" % n)
    print("  identical bit for bit        : %d = %.4f" % (eq, eq / max(n, 1)))
    print("  reverse complementary        : %d" % rc)
    print("  neither                      : %d = %.4f"
          % (other, other / max(n, 1)))
    if rest:
        print("  for those last ones, best identity: mean=%.4f "
              "median=%.4f   (chance level ~0.25)"
              % (float(np.mean(rest)), float(np.median(rest))))


SECTIONS = ("structure", "atg", "gc", "orientation", "expression", "controls",
            "frame", "mld", "target", "mask", "kozak", "design",
            "mld_vs_final", "all")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", default="all", choices=SECTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="how many FILE ROWS to read (4 per gene).  A PREFIX "
                         "of a genome-ordered file, so never a sample: use it "
                         "for a smoke test, not for a statistic.")
    a = ap.parse_args(argv)
    s = a.section
    if s in ("structure", "all"):
        section_structure(a.limit)
    if s in ("atg", "all"):
        section_atg(a.limit)
    if s in ("gc", "all"):
        section_gc(a.limit)
    if s in ("orientation", "all"):
        section_orientation()
    if s in ("expression", "all"):
        section_expression(a.limit)
    if s in ("controls", "all"):
        section_controls(a.limit)
    if s in ("frame", "all"):
        section_frame(a.limit)
    if s in ("mld", "all"):
        section_mld()
    if s in ("target", "all"):
        section_target(a.limit)
    if s in ("mask", "all"):
        section_mask()
    if s in ("kozak", "all"):
        section_kozak(a.limit)
    if s in ("design", "all"):
        section_design(a.limit)
    if s in ("mld_vs_final", "all"):
        section_mld_vs_final(a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
