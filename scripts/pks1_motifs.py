"""What is actually in the pks1 promoter, and does the model look there?

    python scripts/pks1_motifs.py --attention

The model's own motif channels are a fungal TF-binding bank with Neurospora
names, so this script asks the other question: do the sites of the regulators
the published literature associates with this organism occur here, and does
the model attend to them?

Three readings are kept separate, because conflating them is how a motif scan
turns into a story. Presence says nothing on its own -- a short degenerate
string occurs in 800 bp by chance. Enrichment is measured against a
dinucleotide-preserving shuffle, since dinucleotide bias alone reproduces most
short-motif counts. Attention, whether the model's own in-silico mutagenesis
puts weight on those positions, is the only one of the three that is about
our model.

Evidence tiers travel with the table and are printed: the strength of the
Trichoderma evidence differs enormously between these motifs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T",
         "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
         "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
         "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]"}

#: (key, consensus, regulator, what it does, evidence tier)
#: Tier 1 = binding or footprinting demonstrated IN Trichoderma.
#: Tier 2 = consensus established in Trichoderma, but on unrelated targets.
#: Tier 3 = extrapolated from Neurospora/Aspergillus orthologues.
MOTIFS = [
    ("CRE1", "SYGGRG", "CRE1 / CreA",
     "carbon catabolite repression; functional sites are usually CLOSE PAIRS",
     1),
    ("CCAAT", "CCAAT", "Hap2/3/5 complex",
     "general promoter element, constitutively bound in T. reesei", 1),
    ("XYR1", "GGCTAA", "XYR1",
     "cellulase/xylanase activator; no published reason to sit at a PKS", 2),
    ("XYR1w", "GGCWWW", "XYR1, relaxed",
     "the GGC(A/T)3 variant from the same work", 2),
    # The light elements are the rows this promoter is predicted to carry:
    # pks1 makes 6-PP and 6-PP is made in the dark. The consensus is CGATB,
    # taken verbatim from the phr-1 work ("five CGATB motifs" in a region of
    # about 50 bp); a secondary summary renders it GATCB, which is a different
    # string and scores differently.
    ("LRE", "CGATB", "BLR1/BLR2 complex",
     "light element; the mapped region held FIVE copies in ~50 bp", 1),
    ("WCC", "GATCGA", "white-collar complex",
     "Neurospora WCC consensus; NOT shown in Trichoderma", 3),
    ("GATC", "GATC", "GATA-type repeat core",
     "the repeat unit of Neurospora light-response elements", 3),
]


def compile_motif(cons):
    return re.compile("".join(IUPAC[c] for c in cons))


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def find_all(seq, cons):
    """Positions on both strands; overlapping matches included."""
    rx = compile_motif(cons)
    hits = set()
    for m in re.finditer("(?=(%s))" % rx.pattern, seq):
        hits.add(m.start())
    rc = revcomp(seq)
    n = len(seq)
    for m in re.finditer("(?=(%s))" % rx.pattern, rc):
        hits.add(n - m.start() - len(cons))
    return sorted(h for h in hits if 0 <= h <= len(seq) - len(cons))


def dinuc_shuffle(seq, rng):
    """Preserve dinucleotide composition exactly (Altschul-Erikson)."""
    edges = {}
    for a, b in zip(seq, seq[1:]):
        edges.setdefault(a, []).append(b)
    for k in edges:
        rng.shuffle(edges[k])
    # Build an Eulerian path, retrying until the walk uses every edge.
    for _ in range(64):
        out, cur, used = [seq[0]], seq[0], {k: 0 for k in edges}
        ok = True
        for _ in range(len(seq) - 1):
            if used[cur] >= len(edges[cur]):
                ok = False
                break
            nxt = edges[cur][used[cur]]
            used[cur] += 1
            out.append(nxt)
            cur = nxt
        if ok and len(out) == len(seq):
            return "".join(out)
        for k in edges:
            rng.shuffle(edges[k])
    return seq


def read_fasta(path):
    name, chunks, out = None, [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name:
                    out[name] = "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            elif line.strip():
                chunks.append(line.strip())
    if name:
        out[name] = "".join(chunks).upper()
    return out


def pair_density(pos, window=50):
    """Largest number of hits inside any `window`-bp stretch.

    The two Trichoderma-grounded motifs here are about architecture, not
    presence: CRE1 works as closely spaced pairs and the one mapped light
    element carried five copies in about 50 bp. A total count would score five
    scattered hits the same as five stacked ones.
    """
    if not pos:
        return 0
    best = 1
    for i, p in enumerate(pos):
        j = i
        while j < len(pos) and pos[j] - p <= window:
            j += 1
        best = max(best, j - i)
    return best


def ism_importance(seq, ckpt, strain, device, batch=256):
    """Per-position |change in predicted strength| under substitution.

    The maximum over the three alternative bases, the standard in-silico
    mutagenesis read-out: how much this position could matter, not how much
    an average mutation there does.
    """
    import torch
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    sys.path.insert(0, os.path.join(_ROOT, "src", "panel"))
    import model_io

    from hyppe.config import from_mapping
    from hyppe.data.sequences import NUCLEOTIDE_ORDER as BASES

    model, prov = model_io.load_model(ckpt, device=device)
    model.eval()
    print(prov.one_line(), file=sys.stderr)
    species = list(from_mapping(
        torch.load(ckpt, map_location="cpu",
                   weights_only=False)["config"]).conditioning.species)
    row = species.index(strain)

    def enc(ss):
        idx = {b: i for i, b in enumerate(BASES)}
        x = torch.zeros(len(ss), 4, len(seq))
        for j, s in enumerate(ss):
            for i, ch in enumerate(s):
                c = idx.get(ch)
                if c is None:
                    x[j, :, i] = 0.25
                else:
                    x[j, c, i] = 1.0
        return x.to(device)

    @torch.no_grad()
    def pred(ss):
        out = []
        for s in range(0, len(ss), batch):
            ch = ss[s:s + batch]
            o = model(enc(ch),
                      species_id=torch.full((len(ch),), row,
                                            dtype=torch.long, device=device),
                      reconstruct=False)
            mu = o["absolute"].mu
            out.append((mu.mean(-1) if mu.dim() > 1 else mu)
                       .float().cpu().numpy())
        return np.concatenate(out)

    base = float(pred([seq])[0])
    variants, where = [], []
    for i, ch in enumerate(seq):
        for b in BASES:
            if b != ch:
                variants.append(seq[:i] + b + seq[i + 1:])
                where.append(i)
    vals = pred(variants)
    imp = np.zeros(len(seq))
    for i, v in zip(where, vals):
        imp[i] = max(imp[i], abs(v - base))
    print("ISM: baseline %+.4f | %d variants | mean |delta| %.4f"
          % (base, len(variants), imp.mean()), file=sys.stderr)
    return imp


def attention_at(imp, positions, width, rng, draws=10000):
    """Is importance higher on motif positions than on matched random ones?

    The control preserves the number of blocks and their width, not just the
    number of positions. Importance is autocorrelated along a promoter, so a
    scatter of single positions is an easier null than the same count packed
    into blocks, and using it would inflate every result here.
    """
    if not positions:
        return None
    cov = sorted({p + k for p in positions for k in range(width)
                  if 0 <= p + k < len(imp)})
    obs = float(imp[cov].mean())
    n_blocks, hi = len(positions), len(imp) - width
    null = np.empty(draws)
    for d in range(draws):
        starts = rng.integers(0, hi + 1, size=n_blocks)
        idx = sorted({int(s) + k for s in starts for k in range(width)})
        null[d] = imp[idx].mean()
    mu, sd = null.mean(), null.std()
    return {"observed": obs, "null_mean": float(mu), "null_sd": float(sd),
            "z": float((obs - mu) / sd) if sd > 0 else 0.0,
            "p_empirical": float((int((null >= obs).sum()) + 1) / (draws + 1)),
            "n_positions": len(cov)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fasta", default=os.path.join(_ROOT, "data",
                                                    "final_promoters.fasta"))
    ap.add_argument("--target", default="pks1")
    ap.add_argument("--draws", type=int, default=2000,
                    help="dinucleotide-preserving shuffles for the null")
    ap.add_argument("--seed", type=int, default=7898)
    ap.add_argument("--attention", action="store_true",
                    help="also ask whether the model's in-silico mutagenesis "
                         "puts weight on the motif positions")
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "weights",
                                         "hyppe_foundational.pt"))
    ap.add_argument("--strain", default="P1")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results",
                                                  "pks1_motifs.json"))
    args = ap.parse_args()
    if not args.device:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = read_fasta(args.fasta)
    if args.target not in seqs:
        raise SystemExit("%r not in %s" % (args.target, args.fasta))
    target = seqs[args.target]
    rng = np.random.default_rng(args.seed)

    print("target %s, %d bp, GC %.3f"
          % (args.target, len(target),
             (target.count("G") + target.count("C")) / len(target)))
    print("null: %d dinucleotide-preserving shuffles\n" % args.draws)

    nulls = [dinuc_shuffle(target, rng) for _ in range(args.draws)]
    # The shuffle must preserve composition, or the null is the wrong null.
    assert all(sorted(s) == sorted(target) for s in nulls[:5]), \
        "shuffle changed base composition"

    res = {"target": args.target, "length": len(target), "draws": args.draws,
           "motifs": {}}
    print("%-7s %-8s %5s %8s %7s %7s  %-5s %s"
          % ("key", "consensus", "n", "null", "z", "p", "tier", "max/50bp"))
    print("-" * 86)
    for key, cons, reg, what, tier in MOTIFS:
        pos = find_all(target, cons)
        n = len(pos)
        nd = np.array([len(find_all(s, cons)) for s in nulls], dtype=float)
        mu, sd = nd.mean(), nd.std()
        z = (n - mu) / sd if sd > 0 else 0.0
        # One-sided empirical p; the +1 keeps it from reading exactly 0.
        p =(int((nd >= n).sum()) + 1) / (len(nd) + 1)
        dens = pair_density(pos)
        print("%-7s %-8s %5d %8.1f %7.2f %7.4f  %-5d %d"
              % (key, cons, n, mu, z, p, tier, dens))
        res["motifs"][key] = {
            "consensus": cons, "regulator": reg, "role": what, "tier": tier,
            "n_observed": n, "null_mean": float(mu), "null_sd": float(sd),
            "z": float(z), "p_empirical": float(p),
            "max_in_50bp": int(dens), "positions": pos,
        }

    print("\ntier 1 = binding or footprinting shown IN Trichoderma")
    print("tier 2 = consensus established in Trichoderma, unrelated targets")
    print("tier 3 = extrapolated from Neurospora or Aspergillus\n")

    for key, cons, reg, what, tier in MOTIFS:
        print("  %-7s %-14s %s" % (key, reg, what))

    if args.attention:
        print("\n\nDoes the model look where the literature points?")
        print("in-silico mutagenesis on %s, strain %s\n"
              % (args.target, args.strain))
        imp = ism_importance(target, args.checkpoint, args.strain, args.device)
        res["ism"] = {"mean": float(imp.mean()), "max": float(imp.max()),
                      "per_position": [round(float(v), 6) for v in imp]}

        # Positive control, not a result: the top 5 % of positions by
        # importance must beat this null, or the null is too hard and every
        # negative reading below is uninterpretable.
        k = max(1, len(imp) // 20)
        top = [int(i) for i in np.argsort(-imp)[:k]]
        ctrl = attention_at(imp, top, 1, np.random.default_rng(args.seed))
        print("%-10s %9s %9s %7s %8s" % ("", "observed", "null", "z", "p"))
        print("-" * 48)
        print("%-10s %9.5f %9.5f %7.2f %8.4f   <- positive control"
              % ("top 5%", ctrl["observed"], ctrl["null_mean"],
                 ctrl["z"], ctrl["p_empirical"]))
        if ctrl["p_empirical"] > 0.01:
            raise SystemExit(
                "positive control failed (p=%.4f): the most important 5%% of "
                "positions do not beat the matched null, so this null cannot "
                "detect anything and the rows below mean nothing."
                % ctrl["p_empirical"])
        res["ism"]["positive_control"] = ctrl

        for key, cons, reg, what, tier in MOTIFS:
            r = res["motifs"][key]
            a = attention_at(imp, r["positions"], len(cons),
                             np.random.default_rng(args.seed + 1))
            r["attention"] = a
            if a is None:
                print("%-10s %9s %9s %7s %8s   (no sites to look at)"
                      % (key, "-", "-", "-", "-"))
            else:
                print("%-10s %9.5f %9.5f %7.2f %8.4f"
                      % (key, a["observed"], a["null_mean"],
                         a["z"], a["p_empirical"]))
        print("\nThe null draws the same number of blocks of the same width "
              "at random\npositions, because importance is autocorrelated "
              "along the promoter and\nscattered single positions would be "
              "an easier null than packed blocks.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
