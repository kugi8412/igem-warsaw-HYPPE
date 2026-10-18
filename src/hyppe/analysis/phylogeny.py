"""Species tree -> hyperbolic coordinates for every genome in the data.

A genome cannot simply be a leaf. The tree has 10 leaves against 19 genomes;
ten of those genomes are *T. atroviride* strains while the tree carries two
atroviride leaves; four species in the data (asperellum, citrinoviride,
longibrachiatum, velutinum) have no leaf at all; and three leaves (T. gamsii,
Fusarium oxysporum, Neurospora crassa) have no expression data and act as the
outgroups that set the scale. So an anchored genome takes its leaf coordinate,
a sibling strain takes its species' leaf plus a learned tangent offset, and an
unplaced species is a free point initialised at the centroid of the
Trichoderma clade. `describe()` prints which genome fell into which case: a
silent fallback here is a model that believes it knows a phylogeny it does
not.

Two primitives are frozen below as `_legacy_*` and used only by
:func:`fit_tree_embedding` and :meth:`PhylogeneticEmbedding.coordinates`;
everything else uses :mod:`hyppe.geometry`. `hyppe.geometry.expmap0`
truncates a tangent step at MAX_TANGENT_NORM = 8.0, rightly in general
because past that the hyperboloid constraint degrades to ~1e-10 in float64,
but this embedding wants radius 14.1, so the clamp binds and the same fit
lands at 0.0360 instead of the published 0.0118. :func:`main` runs the
canonical backend alongside and prints both, so the gap stays visible.

The published figure is softer than four decimals suggest: the objective
oscillates rather than converging (last 500 of 4000 steps: 0.00623 to 0.02579,
sd 0.0037), so 0.0118 carries about +/- 0.004. The 0.0003 between two distance
formulas is inside that noise; the 0.024 from the clamp is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import geometry as H
from . import matrices as data

# Leaf label -> species key, read off `trees/annotation.tsv`.
LEAF_SPECIES = {
    "GCA_000171015.2_protein": ("atroviride", "IMI206040"),
    "GCF_020647795.1_protein": ("atroviride", "P1"),
    "GCA_000988865.1_protein": ("harzianum", None),
    "GCA_002022785.1_protein": ("guizhouense", None),
    "GCA_052084505.1_protein": ("asperelloides", None),
    "GCF_000167675.1_protein": ("reesei", None),
    "GCF_000170995.1_protein": ("virens", None),
    "GCF_001481775.2_protein": ("gamsii", None),
    "GCF_000149955.1_protein": ("fusarium", None),
    "GCF_000182925.2_protein": ("neurospora", None),
}

# Genome -> species, inferred from the organism names in the experiment
# descriptions by `infer_genome_species` and pinned here so the mapping is
# reviewable rather than re-derived on every run.
GENOME_SPECIES = {
    "B10": "atroviride", "F7": "atroviride", "I1237": "atroviride",
    "IMI206040": "atroviride", "MMS1295": "atroviride", "N1508": "atroviride",
    "P1": "atroviride", "P3041": "atroviride", "P3080": "atroviride",
    "P3116": "atroviride",
    "asperelloides": "asperelloides", "asperellum": "asperellum",
    "citrinoviride": "citrinoviride", "guizhouense": "guizhouense",
    "harzianum": "harzianum", "longibrachiatum": "longibrachiatum",
    "reesei": "reesei", "velutinum": "velutinum", "virens": "virens",
}


def _on_manifold(x: torch.Tensor) -> float:
    """max ``|<x, x>_L + 1|``: worth printing, never assumed."""
    return float((H.minkowski_dot(x, x, keepdim=False) + 1.0).abs().max())


# Frozen arithmetic: these two exist so the module reproduces the published
# 0.0118. They are not a second geometry library and nothing outside this
# file may import them; new code uses `hyppe.geometry`.
def _legacy_expmap0(v: torch.Tensor) -> torch.Tensor:
    """``expmap0`` without the ``MAX_TANGENT_NORM = 8.0`` truncation.

    The cost of not clamping is reported, not hidden: on-manifold drifts from
    3.55e-15 to the ~1e-10 that float64 allows at radius 14.1.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return H.project_to_hyperboloid(torch.sinh(norm) * v / norm)


def _legacy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``acosh(-<x,y>_L)`` with the eps-clamp, as published.

    `hyppe.geometry.geodesic_distance` is better -- its ``log1p`` form is
    exact at coincidence, where this one reports 4.5e-4 -- and the two agree
    to ~1e-15 on every leaf pair here. They still end the trajectory 0.0003
    apart, because it oscillates with sd 0.0037 and is chaotic at that level.
    """
    inner = -H.minkowski_dot(x, y, keepdim=False)
    return torch.acosh(inner.clamp_min(1.0 + 1e-7))


# Newick
@dataclass
class Node:
    name: str = ""
    length: float = 0.0
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(text: str) -> Node:
    """Small recursive-descent parser; enough for the supertree we ship."""
    text = text.strip().rstrip(";")
    pos = 0

    def parse_node() -> Node:
        nonlocal pos
        node = Node()
        if text[pos] == "(":
            pos += 1
            while True:
                node.children.append(parse_node())
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break
                raise ValueError("unexpected %r at %d" % (text[pos], pos))
        m = re.match(r"[^,:\)]*", text[pos:])
        node.name = m.group(0)
        pos += m.end()
        if pos < len(text) and text[pos] == ":":
            pos += 1
            m = re.match(r"-?[\d.eE+-]+", text[pos:])
            node.length = float(m.group(0))
            pos += m.end()
        return node

    return parse_node()


def leaves(node: Node) -> list[Node]:
    return [node] if node.is_leaf else [l for c in node.children
                                        for l in leaves(c)]


def patristic_matrix(root: Node) -> tuple[list[str], np.ndarray]:
    """Leaf names and the matrix of summed branch lengths between them."""
    names = [l.name for l in leaves(root)]
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    D = np.zeros((n, n), dtype=np.float64)

    def walk(node: Node) -> dict[str, float]:
        """Return {leaf: distance from this node}, filling D on the way up."""
        if node.is_leaf:
            return {node.name: 0.0}
        subs = [walk(c) for c in node.children]
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                for la, da in subs[a].items():
                    for lb, db in subs[b].items():
                        d = da + node.children[a].length + \
                            db + node.children[b].length
                        D[index[la], index[lb]] = d
                        D[index[lb], index[la]] = d
        out: dict[str, float] = {}
        for c, sub in zip(node.children, subs):
            for l, d in sub.items():
                out[l] = d + c.length
        return out

    walk(root)
    return names, D


# Embedding
def fit_tree_embedding(D: np.ndarray, dim: int = 4, steps: int = 4000,
                       lr: float = 0.05, seed: int = 0, verbose: bool = True):
    """Fit Lorentz coordinates whose geodesic distances match `D`.

    Optimises mean |d_hyp / (s * d_tree) - 1| with one global scale `s`:
    tree distance is in substitutions and geodesic distance is
    dimensionless, so an unscaled comparison would measure the unit.
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    tgt = torch.tensor(D, dtype=torch.float64)
    iu = torch.triu_indices(n, n, offset=1)
    t = tgt[iu[0], iu[1]]
    # A zero target distance makes the relative error infinite, which is
    # exactly what a topology-only tree file produces. Fail loudly rather
    # than return an `inf` distortion nobody reads.
    n_zero = int((t <= 0).sum())
    if n_zero:
        raise ValueError(
            "%d of %d leaf pairs have tree distance 0 -- this tree carries "
            "topology without branch lengths, so distances are undefined. "
            "Use `paths.SPECIES_TREE` (supermatrix.treefile), not the "
            "topology-only supertree." % (n_zero, t.numel()))
    v = torch.randn(n, dim, dtype=torch.float64) * 0.1
    v.requires_grad_(True)
    log_s = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v, log_s], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        x = _legacy_expmap0(v)
        d = _legacy_distance(x[iu[0]], x[iu[1]])
        rel = d / (log_s.exp() * t).clamp_min(1e-9)
        loss = (rel - 1.0).abs().mean()
        loss.backward()
        opt.step()
        if verbose and (i + 1) % 1000 == 0:
            print("    step %5d  mean |d_hyp/(s*d_tree) - 1| = %.5f"
                  % (i + 1, float(loss.detach())))
    with torch.no_grad():
        x = _legacy_expmap0(v)
        d = _legacy_distance(x[iu[0]], x[iu[1]])
        rel = (d / (log_s.exp() * t)).numpy()
    return x.detach(), {"mean_distortion": float(np.abs(rel - 1).mean()),
                        "max_distortion": float(np.abs(rel - 1).max()),
                        "scale": float(log_s.exp()),
                        "on_manifold": _on_manifold(x.detach())}


class PhylogeneticEmbedding:
    """Genome -> hyperbolic coordinate, with the fallback made explicit."""

    def __init__(self, dim: int = 4, steps: int = 4000, verbose: bool = True):
        root = parse_newick(data.read_newick())
        self.leaf_names, self.D = patristic_matrix(root)
        if verbose:
            print("  fitting %d leaves in %d hyperbolic dimensions"
                  % (len(self.leaf_names), dim))
        self.points, self.quality = fit_tree_embedding(
            self.D, dim=dim, steps=steps, verbose=verbose)
        self.species_to_leaf = {}
        for leaf, (sp, strain) in LEAF_SPECIES.items():
            if leaf in self.leaf_names:
                self.species_to_leaf.setdefault(sp, leaf)
                if strain:
                    self.species_to_leaf["strain:" + strain] = leaf
        self.dim = dim

    def placement(self, genome: str) -> tuple[str, str]:
        """(case, leaf) for one genome.  `case` is one of
        'anchored' / 'sibling strain' / 'unplaced species'."""
        key = "strain:" + genome
        if key in self.species_to_leaf:
            return "anchored", self.species_to_leaf[key]
        sp = GENOME_SPECIES.get(genome)
        if sp and sp in self.species_to_leaf:
            return ("sibling strain" if GENOME_SPECIES.get(genome) ==
                    "atroviride" else "anchored"), self.species_to_leaf[sp]
        return "unplaced species", ""

    def coordinates(self, genomes: list[str]) -> torch.Tensor:
        """Base coordinates, before any learned per-strain offset."""
        idx = {n: i for i, n in enumerate(self.leaf_names)}
        tri = [i for n, i in idx.items()
               if LEAF_SPECIES.get(n, ("", None))[0]
               not in ("fusarium", "neurospora")]
        # `_legacy_expmap0`, not `H.expmap0`: the leaves sit at radius up to
        # 14.1, so the mean tangent vector is well past MAX_TANGENT_NORM, and
        # the canonical exp map would truncate it -- a point on the right ray
        # at the wrong radius, with no exception and no change of shape.
        centroid = _legacy_expmap0(
            H.logmap0(self.points[tri]).mean(0, keepdim=True))
        out = []
        for g in genomes:
            _, leaf = self.placement(g)
            out.append(self.points[idx[leaf]:idx[leaf] + 1] if leaf
                       else centroid)
        return torch.cat(out, dim=0)

    def describe(self, genomes: list[str]) -> None:
        print("\n  tree embedding quality: mean distortion %.4f | max %.4f "
              "| scale %.4f | on-manifold %.2e"
              % (self.quality["mean_distortion"], self.quality["max_distortion"],
                 self.quality["scale"], self.quality["on_manifold"]))
        cases: dict[str, list[str]] = {}
        for g in genomes:
            case, leaf = self.placement(g)
            cases.setdefault(case, []).append(g)
        print("\n  %-18s %-4s %s" % ("case", "n", "genomes"))
        print("  " + "-" * 70)
        for case in ("anchored", "sibling strain", "unplaced species"):
            gs = cases.get(case, [])
            print("  %-18s %-4d %s" % (case, len(gs), ", ".join(gs)[:60]))
        unused = [n for n in self.leaf_names
                  if n not in {self.placement(g)[1] for g in genomes}]
        print("\n  leaves with no expression data (outgroups / absent): %s"
              % ", ".join(LEAF_SPECIES.get(u, (u, None))[0] for u in unused))
        print("  -> 'sibling strain' genomes share a coordinate until a "
              "learned offset separates them; 'unplaced species' genomes "
              "carry NO phylogenetic information and the model must not be "
              "credited with any for them.")


def infer_genome_species(samples) -> dict[str, str]:
    """Re-derive `GENOME_SPECIES` from the free-text descriptions.

    Kept so the pinned table can be audited against the data, not trusted.
    """
    import collections
    txt = (samples["study_title"].fillna("") + " "
           + samples["experiment_title"].fillna("") + " "
           + samples["condition"].fillna(""))
    out = {}
    for g, grp in samples.groupby("genome"):
        joined = " ".join(txt.loc[grp.index].tolist())
        found = re.findall(r"\bTrichoderma\s+([a-z]{4,})", joined)
        counts = collections.Counter(found)
        for name, _ in counts.most_common():
            if name not in ("grown", "during", "with", "withh", "species",
                            "before", "after"):
                out[g] = name
                break
    return out


def _canonical_fit(D: np.ndarray, dim: int = 4, steps: int = 4000,
                   lr: float = 0.05, seed: int = 0):
    """:func:`fit_tree_embedding` with :mod:`hyppe.geometry` primitives.

    Not the reported path. It exists so the cost of the canonical clamp is
    printed on every run rather than remembered.
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    tgt = torch.tensor(D, dtype=torch.float64)
    iu = torch.triu_indices(n, n, offset=1)
    t = tgt[iu[0], iu[1]]
    v = torch.randn(n, dim, dtype=torch.float64) * 0.1
    v.requires_grad_(True)
    log_s = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v, log_s], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = H.expmap0(v)
        d = H.geodesic_distance(x[iu[0]], x[iu[1]])
        loss = (d / (log_s.exp() * t).clamp_min(1e-9) - 1.0).abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = H.expmap0(v)
        d = H.geodesic_distance(x[iu[0]], x[iu[1]])
        rel = (d / (log_s.exp() * t)).numpy()
        return x, {"mean_distortion": float(np.abs(rel - 1).mean()),
                   "max_distortion": float(np.abs(rel - 1).max()),
                   "scale": float(log_s.exp()),
                   "on_manifold": _on_manifold(x),
                   "max_tangent_norm": float(v.norm(dim=-1).max())}


def fit_euclidean_embedding(D: np.ndarray, dim: int = 4, steps: int = 4000,
                            lr: float = 0.05, seed: int = 0):
    """Same objective, same budget, flat space: the control.

    A hyperbolic distortion of 0.012 means nothing until the identical fit
    in Euclidean space of the same dimension is beside it.
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    tgt = torch.tensor(D, dtype=torch.float64)
    iu = torch.triu_indices(n, n, offset=1)
    t = tgt[iu[0], iu[1]]
    v = torch.randn(n, dim, dtype=torch.float64) * 0.1
    v.requires_grad_(True)
    log_s = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v, log_s], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        d = (v[iu[0]] - v[iu[1]]).norm(dim=-1)
        loss = (d / (log_s.exp() * t).clamp_min(1e-9) - 1.0).abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = (v[iu[0]] - v[iu[1]]).norm(dim=-1)
        rel = (d / (log_s.exp() * t)).numpy()
    return {"mean_distortion": float(np.abs(rel - 1).mean()),
            "max_distortion": float(np.abs(rel - 1).max())}


def main() -> None:
    samples = data.load_samples("og_scope")
    genomes = sorted(samples["genome"].unique())
    emb = PhylogeneticEmbedding(dim=4, steps=4000)
    emb.describe(genomes)

    print("\n  CONTROL -- same objective, same budget, flat space:")
    print("  %-8s %-22s %-22s" % ("dim", "hyperbolic mean dist.",
                                  "euclidean mean dist."))
    print("  " + "-" * 56)
    for d in (2, 3, 4, 8):
        hyp, q = fit_tree_embedding(emb.D, dim=d, steps=3000, verbose=False)
        euc = fit_euclidean_embedding(emb.D, dim=d, steps=3000)
        print("  %-8d %-22.5f %-22.5f" % (d, q["mean_distortion"],
                                          euc["mean_distortion"]))
    print("  -> curvature is worth its cost only where the hyperbolic column "
          "is clearly lower.  With 10 leaves the tree is small, so a flat "
          "space may well be enough -- and if it is, say so.")

    print("\n  SAME FIT ON THE CANONICAL BACKEND (hyppe.geometry), for the record:")
    _xc, qc = _canonical_fit(emb.D, dim=4, steps=4000)
    print("  mean distortion %.4f | max %.4f | scale %.4f | on-manifold %.2e"
          % (qc["mean_distortion"], qc["max_distortion"], qc["scale"],
             qc["on_manifold"]))
    print("  -> the gap against the frozen backend above is the "
          "MAX_TANGENT_NORM = 8.0 clamp biting at radius %.1f, not a bug "
          "in either one." % qc["max_tangent_norm"])

    print("\n  audit of the pinned genome -> species table:")
    derived = infer_genome_species(samples)
    bad = {g: (GENOME_SPECIES.get(g), derived.get(g)) for g in genomes
           if GENOME_SPECIES.get(g) != derived.get(g)}
    print("  mismatches: %s" % (bad if bad else "none"))


if __name__ == "__main__":
    main()
