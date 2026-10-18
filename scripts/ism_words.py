"""Which short words does the model actually weigh? Asked of the corpus.

    python scripts/ism_words.py --n-promoters 200 --k 6

Scanning one promoter for motifs somebody else published asks only whether
that sequence contains their sites; the model is not in the question. This
script mutates every position of many real promoters, records how far each
substitution moves the predicted strength, and asks which short words sit
under the positions that moved it most -- a ranking the model produced on its
own, readable against the literature rather than the other way round.

Two controls are load-bearing. Importance is not flat along a promoter: it
rises towards the start codon (Spearman -0.36 on pks1 alone), so any word near
the 3' end would beat a null drawn uniformly along the sequence whatever it
meant. Importance is therefore standardised locally first -- each window
against the windows within a few dozen bases of it -- and only then pooled.
Second, a word with few occurrences has a noisy mean and the extremes of a
ranking over 4^k words are mostly the rarest ones, so the null is drawn at the
word's own occurrence count.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "panel")):
    sys.path.insert(0, os.path.abspath(_p))

import model_io  # noqa: E402

from hyppe.config import from_mapping  # noqa: E402

# Imported, never retyped: the sibling track orders the channels A,C,T,G and
# this one A,C,G,T, so a local copy silently swaps G with T while still
# producing plausible numbers.
from hyppe.data.sequences import NUCLEOTIDE_ORDER as BASES  # noqa: E402

SEQ_LEN = 800


def load_promoters(path, genome, split, n, rng):
    """Promoters of one strain from one split, sampled without replacement."""
    keep = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(head)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[ci["genome"]] != genome or f[ci["split"]] != split:
                continue
            s = f[ci["promoter"]].upper()
            if len(s) == SEQ_LEN and set(s) <= set("ACGT"):
                keep.append((f[ci["og_id"]], s))
    if not keep:
        raise SystemExit("no %s promoters in split %r" % (genome, split))
    idx = rng.choice(len(keep), size=min(n, len(keep)), replace=False)
    return [keep[int(i)] for i in idx]


class Scorer:
    def __init__(self, ckpt, strain, device, batch):
        self.model, prov = model_io.load_model(ckpt, device=device)
        self.model.eval()
        print(prov.one_line(), file=sys.stderr)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        species = list(from_mapping(blob["config"]).conditioning.species)
        if strain not in species:
            raise SystemExit("strain %r not in registry: %s"
                             % (strain, ", ".join(species)))
        self.row, self.device, self.batch = species.index(strain), device, batch
        self.idx = {b: i for i, b in enumerate(BASES)}

    def _enc(self, ss):
        x = torch.zeros(len(ss), 4, SEQ_LEN)
        for j, s in enumerate(ss):
            for i, ch in enumerate(s):
                c = self.idx.get(ch)
                if c is None:
                    x[j, :, i] = 0.25
                else:
                    x[j, c, i] = 1.0
        return x.to(self.device)

    @torch.no_grad()
    def __call__(self, ss):
        out = []
        for s in range(0, len(ss), self.batch):
            ch = ss[s:s + self.batch]
            o = self.model(self._enc(ch),
                           species_id=torch.full((len(ch),), self.row,
                                                 dtype=torch.long,
                                                 device=self.device),
                           reconstruct=False)
            mu = o["absolute"].mu
            out.append((mu.mean(-1) if mu.dim() > 1 else mu)
                       .float().cpu().numpy())
        return np.concatenate(out)


def ism(seq, score):
    """Per position, the largest move any of the three substitutions makes."""
    base = float(score([seq])[0])
    variants, where = [], []
    for i, ch in enumerate(seq):
        for b in BASES:
            if b != ch:
                variants.append(seq[:i] + b + seq[i + 1:])
                where.append(i)
    vals = score(variants)
    imp = np.zeros(len(seq))
    np.maximum.at(imp, where, np.abs(vals - base))
    return imp


def local_z(imp, k, radius):
    """Window importance, standardised against its own neighbourhood.

    The step that removes the positional trend. `w[p]` is the mean importance
    of the k-mer starting at p, expressed in standard deviations of the w
    values within `radius` of it, which makes a window at the 3' end and one
    at the 5' end comparable.
    """
    w = np.convolve(imp, np.ones(k) / k, mode="valid")
    n = len(w)
    c1 = np.concatenate([[0.0], np.cumsum(w)])
    c2 = np.concatenate([[0.0], np.cumsum(w * w)])
    lo = np.maximum(0, np.arange(n) - radius)
    hi = np.minimum(n, np.arange(n) + radius + 1)
    cnt = (hi - lo).astype(float)
    mu = (c1[hi] - c1[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    sd = np.sqrt(var)
    # A neighbourhood of constant importance carries no local information;
    # zero keeps it out of the ranking instead of at the top of it.
    return np.where(sd > 1e-9, (w - mu) / np.maximum(sd, 1e-9), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # The shipped fixture, not the training corpus: 348 P1 validation
    # promoters, enough to run this probe from a fresh clone. The full corpus
    # is reached through HYPPE_DATA_ROOT and is not in the repository.
    ap.add_argument("--sequences",
                    default=os.path.join(_ROOT, "data", "probe",
                                         "p1_val_promoters.tsv"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "weights",
                                         "hyppe_foundational.pt"))
    ap.add_argument("--strain", default="P1")
    ap.add_argument("--split", default="val",
                    help="never 'test': this repository reads test once")
    ap.add_argument("--n-promoters", type=int, default=200)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--radius", type=int, default=50,
                    help="half-width of the neighbourhood a window is "
                         "standardised against")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7898)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results",
                                                  "ism_words.json"))
    ap.add_argument("--save-tracks", default="")
    ap.add_argument("--from-tracks", default="",
                    help="re-analyse saved importance tracks instead of "
                         "recomputing them; the mutagenesis is the whole "
                         "cost, so a second word length should not pay it")
    args = ap.parse_args()
    if args.split == "test":
        raise SystemExit("test is read once, under the epoch-selection "
                         "protocol, and not by an interpretability probe.")
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(args.seed)

    if args.from_tracks:
        blob = np.load(args.from_tracks, allow_pickle=False)
        tracks = [np.asarray(t, dtype=float) for t in blob["tracks"]]
        seqs = [str(s) for s in blob["seqs"]]
        ogs = [str(o) for o in blob["og_ids"]]
        proms = list(zip(ogs, seqs))
        print("%d promoters from %s, %d-mers"
              % (len(proms), args.from_tracks, args.k))
    else:
        proms = load_promoters(args.sequences, args.strain, args.split,
                               args.n_promoters, rng)
        print("%d promoters, strain %s, split %s, %d-mers"
              % (len(proms), args.strain, args.split, args.k))

        score = Scorer(args.checkpoint, args.strain, args.device,
                       args.batch_size)
        tracks, seqs = [], []
        for i, (og, s) in enumerate(proms):
            tracks.append(ism(s, score))
            seqs.append(s)
            if (i + 1) % 10 == 0 or i + 1 == len(proms):
                print("  ISM %d/%d" % (i + 1, len(proms)), file=sys.stderr)

    zs = [local_z(t, args.k, args.radius) for t in tracks]

    # Positive control: after local standardisation the trend must be gone,
    # or every word near the start codon is still advantaged and the ranking
    # below is a ranking of positions wearing words' names.
    raw = np.array([np.corrcoef(t, np.arange(len(t)))[0, 1] for t in tracks])
    std = np.array([np.corrcoef(z, np.arange(len(z)))[0, 1] for z in zs])
    print("\npositional trend (correlation of importance with position)")
    print("  before standardising  mean %+.4f" % raw.mean())
    print("  after                 mean %+.4f" % std.mean())
    if abs(std.mean()) > 0.05:
        raise SystemExit("standardisation left a positional trend of %+.4f; "
                         "the word ranking would be reading position."
                         % std.mean())

    pooled_z = np.concatenate(zs)
    words = {}
    for j, s in enumerate(seqs):
        z = zs[j]
        for p in range(len(z)):
            words.setdefault(s[p:p + args.k], []).append(z[p])

    keep = {w: np.array(v) for w, v in words.items()
            if len(v) >= args.min_count}
    print("\n%d distinct %d-mers, %d seen at least %d times"
          % (len(words), args.k, len(keep), args.min_count))

    # The null for a word seen c times is the mean of c windows drawn at
    # random, whose spread is the pooled spread over sqrt(c). That is an
    # approximation, so it is checked rather than assumed: resampling at three
    # occurrence counts must reproduce the analytic spread. It is done
    # analytically because thousands of words carry hundreds of distinct
    # counts between them, and resampling each separately costs an hour.
    p_mu, p_sd = float(pooled_z.mean()), float(pooled_z.std())
    counts = sorted({len(v) for v in keep.values()})
    print("\nnull check: resampled spread against pooled_sd / sqrt(n)")
    for c in (counts[0], counts[len(counts) // 2], counts[-1]):
        m = np.concatenate([
            rng.choice(pooled_z, size=(2000, c), replace=True).mean(axis=1)
            for _ in range(max(1, args.draws // 2000))])
        emp, ana = float(m.std()), p_sd / np.sqrt(c)
        print("  n=%-5d resampled %.5f   analytic %.5f   ratio %.3f"
              % (c, emp, ana, emp / ana))
        if not 0.9 < emp / ana < 1.1:
            raise SystemExit("the analytic null is off by %.0f%% at n=%d; "
                             "word p-values would be wrong."
                             % (abs(emp / ana - 1) * 100, c))

    from math import erfc, sqrt
    rows = []
    for w, v in keep.items():
        obs, c = float(v.mean()), len(v)
        sd = p_sd / sqrt(c)
        z = (obs - p_mu) / sd if sd > 0 else 0.0
        rows.append({"word": w, "n": int(c), "mean_z": obs, "null_sd": sd,
                     "z": z, "p": 0.5 * erfc(z / sqrt(2.0))})
    rows.sort(key=lambda r: -r["z"])

    # Benjamini-Hochberg: we tested thousands of words and the top of any such
    # ranking is populated by chance unless the multiplicity is paid for.
    n_t = len(rows)
    srt = sorted(rows, key=lambda r: r["p"])
    best = 1.0
    for i in range(n_t - 1, -1, -1):
        best = min(best, srt[i]["p"] * n_t / (i + 1))
        srt[i]["q"] = min(1.0, best)

    sig = [r for r in rows if r["q"] < 0.05]
    print("%d words pass a 5%% false-discovery rate\n" % len(sig))
    print("%-8s %6s %8s %7s %9s %9s" % ("word", "n", "mean z", "z", "p", "q"))
    print("-" * 52)
    for r in rows[:25]:
        print("%-8s %6d %8.3f %7.2f %9.5f %9.5f"
              % (r["word"], r["n"], r["mean_z"], r["z"], r["p"], r["q"]))

    res = {"strain": args.strain, "split": args.split, "k": args.k,
           "n_promoters": len(proms), "radius": args.radius,
           "og_ids": [og for og, _ in proms],
           "trend_before": float(raw.mean()), "trend_after": float(std.mean()),
           "n_tested": n_t, "n_significant": len(sig), "words": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote %s" % args.out)

    if args.save_tracks:
        np.savez_compressed(args.save_tracks,
                            tracks=np.array(tracks),
                            seqs=np.array(seqs),
                            og_ids=np.array([og for og, _ in proms]))
        print("wrote %s" % args.save_tracks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
