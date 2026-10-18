"""Does this corpus contain sequence variation within a condition?

    export HYPPE_DATA_ROOT=/path/to/hyppe-data
    python scripts/corpus_variation.py

The documented claim is that it does not: one promoter sequence per gene per
strain, expression measured across conditions, so the sequence is fixed while
the measurement moves and nothing can teach the model which base carries a
response. That was asserted rather than measured. The expression matrix is
indexed by orthogroup and its columns are samples, each carrying a genome and
a condition, so the same gene appears in many strains with its own promoter
under many conditions. This counts whether that is usable variation.

Reads the training corpus, which does not ship with the repository.
"""
import argparse
import collections
import gzip
import itertools
import os
import random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find(name, override=""):
    """Locate one corpus table, or say exactly what to set."""
    if override:
        if not os.path.isfile(override):
            raise SystemExit("no file at %s" % override)
        return override
    roots = [r for r in (os.environ.get("HYPPE_DATA_ROOT"), _REPO) if r]
    tried = [os.path.join(r, "data", "final_datasets", "FINAL", name)
             for r in roots]
    for p in tried:
        if os.path.isfile(p):
            return p
    raise SystemExit(
        "cannot find %s. This probe reads the training corpus, which is not "
        "part of the repository.\nSet HYPPE_DATA_ROOT to the data root, or "
        "pass the file explicitly.\nTried:\n  %s" % (name, "\n  ".join(tried)))


_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--sequences", default="")
_ap.add_argument("--samples", default="")
_args = _ap.parse_args()

SEQ = _find("sequences_final.tsv.gz", _args.sequences)
SAMP = _find("samples_final.tsv", _args.samples)

# Sequences
seqs = collections.defaultdict(dict)        # og -> genome -> promoter
with gzip.open(SEQ, "rt") as fh:
    head = fh.readline().rstrip("\n").split("\t")
    ci = {c: i for i, c in enumerate(head)}
    for line in fh:
        f = line.rstrip("\n").split("\t")
        seqs[f[ci["og_id"]]][f[ci["genome"]]] = f[ci["promoter"]]

n_og = len(seqs)
per = [len(v) for v in seqs.values()]
multi = [og for og, v in seqs.items() if len(v) >= 2]
print("orthogroups with a promoter: %d" % n_og)
print("  in >= 2 strains: %d (%.1f%%)" % (len(multi), 100.0*len(multi)/n_og))
print("  strains per orthogroup: median %d, max %d"
      % (sorted(per)[len(per)//2], max(per)))

# How different are the promoters of one gene between strains?
random.seed(7898)


def ident(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


sample_ogs = random.sample(multi, min(400, len(multi)))
within, ident_all = [], []
for og in sample_ogs:
    g = list(seqs[og].values())
    for a, b in itertools.islice(itertools.combinations(g, 2), 10):
        ident_all.append(ident(a, b))
        within.append(1 if a != b else 0)
print("\npromoters of the SAME gene in two different strains (%d pairs):"
      % len(ident_all))
print("  identical base for base: %.1f%%"
      % (100.0 * (1 - sum(within) / len(within))))
srt = sorted(ident_all)
print("  positional identity: median %.3f, 10th pct %.3f, 90th pct %.3f"
      % (srt[len(srt)//2], srt[len(srt)//10], srt[9*len(srt)//10]))

# Samples
rows = [l.rstrip("\n").split("\t")
        for l in open(SAMP, encoding="utf-8").read().splitlines() if l.strip()]
sh, sb = rows[0], rows[1:]
si = {c: i for i, c in enumerate(sh)}
samp_genome = {r[si["sample"]]: r[si["genome"]] for r in sb}
samp_cat = {r[si["sample"]]: r[si["category"]] for r in sb}
cats = collections.Counter(samp_cat.values())
print("\nsamples: %d, strains %d, categories %d"
      % (len(sb), len(set(samp_genome.values())), len(cats)))
for c, n in cats.most_common():
    print("   %-28s %5d samples, %2d strains"
          % (c, n, len({samp_genome[s] for s in samp_genome
                        if samp_cat[s] == c})))

# For one condition: how many strains are measured, and do their promoters
# for a shared gene differ? That is variation within a condition.
print("\nwithin ONE condition, strains measured and genes shared between "
      "them:")
by_cat = collections.defaultdict(set)
for s, c in samp_cat.items():
    by_cat[c].add(samp_genome[s])
for c, gs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
    if len(gs) < 2:
        print("   %-28s only %d strain -- no within-condition variation"
              % (c, len(gs)))
        continue
    shared = [og for og in multi if len(set(seqs[og]) & gs) >= 2]
    diff = 0
    for og in random.sample(shared, min(200, len(shared))):
        g = [seqs[og][x] for x in (set(seqs[og]) & gs)]
        if len(set(g)) >= 2:
            diff += 1
    frac = diff / max(1, min(200, len(shared)))
    print("   %-28s %2d strains, %5d shared genes, %.0f%% differ in sequence"
          % (c, len(gs), len(shared), 100 * frac))
