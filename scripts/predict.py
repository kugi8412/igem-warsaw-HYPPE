"""Score promoters with the delivered model. FASTA in, a table out.

    python scripts/predict.py
    python scripts/predict.py --fasta my_promoters.fa --judge

Every other entry point here needs the training corpus or a split file; this
one needs neither, since the weights and the ten delivered promoters both
ship.

The strength column is the model's own scale. It is not fluorescence and is
not calibrated to any wet-lab read-out, so comparisons between rows are
meaningful and the absolute level is not. The judge column, when asked for,
comes from the separate ranking model that decides panel membership, on a
scale of its own again. See the model documentation in `docs/` for the three
domain shifts that sit between this number and a measurement.
"""
from __future__ import annotations

import argparse
import csv
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
# this one A,C,G,T, and a hardcoded copy here silently swapped G and T while
# producing plausible numbers.
from hyppe.data.sequences import NUCLEOTIDE_ORDER as BASES  # noqa: E402

SEQ_LEN = 800


def read_fasta(path):
    """{name: sequence}, uppercased, in file order."""
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


def one_hot(seqs, device):
    """[N, 4, 800]. Anything outside ACTG becomes a uniform column.

    A uniform column is the same "no evidence" the training loader gives an N,
    so an ambiguous base does not silently become an A.
    """
    idx = {b: i for i, b in enumerate(BASES)}
    x = torch.zeros(len(seqs), 4, SEQ_LEN)
    for j, s in enumerate(seqs):
        for i, ch in enumerate(s):
            c = idx.get(ch)
            if c is None:
                x[j, :, i] = 0.25
            else:
                x[j, c, i] = 1.0
    return x.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fasta",
                    default=os.path.join(_ROOT, "data",
                                         "final_promoters.fasta"),
                    help="promoters to score; default is the delivered panel")
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "weights",
                                         "hyppe_foundational.pt"))
    ap.add_argument("--judge", action="store_true",
                    help="also score with the fine-tuned ranking head")
    ap.add_argument("--judge-checkpoint",
                    default=os.path.join(_ROOT, "weights", "hyppe_judge.pt"))
    ap.add_argument("--strain", default="P1",
                    help="which strain to condition on; must be in the "
                         "registry the checkpoint was trained with")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="write TSV here as well")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    if not seqs:
        raise SystemExit("no sequences in %s" % args.fasta)
    bad = {k: len(v) for k, v in seqs.items() if len(v) != SEQ_LEN}
    if bad:
        raise SystemExit(
            "this model reads exactly %d bp, counted from the base before the "
            "start codon. Wrong lengths: %s" % (SEQ_LEN, bad))

    model, prov = model_io.load_model(args.checkpoint, device=args.device)
    model.eval()
    print(prov.one_line(), file=sys.stderr)

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    species = list(from_mapping(blob["config"]).conditioning.species)
    if args.strain not in species:
        raise SystemExit("strain %r is not in this checkpoint's registry. "
                         "Known: %s" % (args.strain, ", ".join(species)))
    row = species.index(args.strain)
    print("strain %s -> registry row %d of %d"
          % (args.strain, row, len(species)), file=sys.stderr)

    names = list(seqs)

    @torch.no_grad()
    def score(net, head):
        """`head` picks which output is read, and the two are not the same.

        Both heads live in the same network, and the judge checkpoint differs
        from the foundation one only in the ranking head's six tensors, so
        reading the strength head out of the judge returns the foundation
        model's own numbers identically.
        """
        out = []
        for s in range(0, len(names), args.batch_size):
            chunk = [seqs[n] for n in names[s:s + args.batch_size]]
            x = one_hot(chunk, args.device)
            sp = torch.full((len(chunk),), row, dtype=torch.long,
                            device=args.device)
            o = net(x, species_id=sp, reconstruct=False)
            if head == "absolute":
                # One number per sequence. The model emits one column per
                # growth condition, and with no condition asked about, the
                # mean over them is the honest summary.
                v = o["absolute"].mu
                v = v.mean(-1) if v.dim() > 1 else v
            else:
                v = o["pair_score"]
                v = v.squeeze(-1) if v.dim() > 1 else v
            out.append(v.float().cpu().numpy())
        return np.concatenate(out)

    strength = score(model, "absolute")

    judge = None
    if args.judge:
        jmodel, jprov = model_io.load_model(args.judge_checkpoint,
                                            device=args.device)
        jmodel.eval()
        print(jprov.one_line(), file=sys.stderr)
        judge = score(jmodel, "pairwise")

    rows = []
    order = np.argsort(-strength)
    head = ["name", "strength"] + (["judge"] if judge is not None else [])
    widths = "%-30s %10s" + ("%10s" if judge is not None else "")
    print(widths % tuple(head))
    print("-" * (42 + (10 if judge is not None else 0)))
    for i in order:
        r = {"name": names[i], "strength": "%.4f" % strength[i]}
        line = "%-30s %10.4f" % (names[i], strength[i])
        if judge is not None:
            r["judge"] = "%.4f" % judge[i]
            line += " %9.4f" % judge[i]
        rows.append(r)
        print(line)

    print("\nThis scale is the model's own. It is not fluorescence, and only "
          "the ordering is meant to transfer.", file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
