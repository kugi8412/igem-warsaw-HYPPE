#! /usr/bin/env python3

"""Report whether a Newick tree can serve as the model's strain prior.

    python check_tree.py ../../data/trees/supermatrix.treefile
    python check_tree.py \
        ../../data/trees/external/inhouse_nj_20genomes_aasubst.tre --registry

The prior pulls strain vectors toward the tree's patristic distances under a
relative error, so what matters is the distances and not the topology. Both
failure modes have occurred here: a tree with all branch lengths at zero
places every strain at one point, and a pair at distance near zero makes the
relative error blow up, which the model answers with a floor.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "src"))

from hyppe.analysis.phylogeny import parse_newick, patristic_matrix  # noqa

try:
    from hyppe.model.species import PRIOR_MIN_DIST
except ImportError:                                   # pragma: no cover
    PRIOR_MIN_DIST = 1e-3


def registry_names():
    """The strains the delivered model was configured with."""
    import yaml
    with open(os.path.join(_REPO, "config", "default.yaml"),
              encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["conditioning"]["species"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tree")
    ap.add_argument("--registry", action="store_true",
                    help="also require the tips to be exactly the strains in "
                         "config/default.yaml")
    ap.add_argument("--floor", type=float, default=PRIOR_MIN_DIST)
    args = ap.parse_args()

    root = parse_newick(open(args.tree, encoding="utf-8").read())
    names, D = patristic_matrix(root)
    n = len(names)
    print("%s\ntips: %d" % (args.tree, n))

    fails = []
    if len(set(names)) != n:
        fails.append("duplicate tip labels: %s"
                     % sorted({x for x in names if names.count(x) > 1}))

    iu = np.triu_indices(n, 1)
    d = D[iu]
    if not np.all(np.isfinite(d)):
        fails.append("%d distances are not finite"
                     % int((~np.isfinite(d)).sum()))
    if np.any(d < 0):
        fails.append("%d distances are negative" % int((d < 0).sum()))

    zero = int((d <= 0).sum())
    below = int((d < args.floor).sum())
    print("pairs           : %d" % len(d))
    print("distance        : min %.6g, median %.6g, max %.6g"
          % (d.min(), float(np.median(d)), d.max()))
    print("exactly zero    : %d (%.1f%%)" % (zero, 100.0 * zero / len(d)))
    print("below the floor : %d (%.1f%%), floor %g"
          % (below, 100.0 * below / len(d), args.floor))

    if zero == len(d):
        fails.append("every pair is at distance zero, so this is a topology "
                     "only tree and no distance based embedding can use it")
    elif below > len(d) / 2:
        print("NOTE: over half the pairs are clamped to the floor, so the "
              "prior carries little signal.")

    collapsed = sorted({names[i] for i, j in zip(*iu) if D[i, j] <= 0}
                       | {names[j] for i, j in zip(*iu) if D[i, j] <= 0})
    if collapsed:
        print("tips no distance separates: %d (%s)"
              % (len(collapsed), ", ".join(collapsed)))

    if args.registry:
        want, have = set(registry_names()), set(names)
        print("\nregistry        : %d strains" % len(want))
        miss, extra = sorted(want - have), sorted(have - want)
        if miss:
            fails.append("the model expects these strains and the tree has "
                         "no tip for them: %s" % ", ".join(miss))
        if extra:
            fails.append("the tree has tips the model does not know: %s"
                         % ", ".join(extra))
        if not miss and not extra:
            print("tips match the registry exactly")

    print()
    if fails:
        print("NOT USABLE as a strain prior:")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("usable as a strain prior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
