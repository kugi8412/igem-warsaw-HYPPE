"""Per-channel targets: a conditioning channel with no target is decoration.

`build_targets()` returns one number per (orthogroup, genome) pair, the mean
of `combat` over the kept samples of that genome, and every head regresses it.
Being a mean over conditions it is constant in the condition by construction,
so the experiment-text channel is asked to predict a quantity that cannot
depend on it. That is why the species and experiment channels measured
-0.0020 / -0.0029 under zeroing while gene-context measured -0.1819: a channel
is weak when the loss never asks it for anything.

For orthogroup ``o``, genome ``g`` and sample ``s`` (each sample belongs to
exactly one genome) the ComBat cell is written as

    y[o, s] = grand + gene[o] + genome[g] + inter[o, g] + cond[o, s]

with, by construction,

    gene[o]      = mean_s y[o, s] - grand
    genome[g]    = mean_o mean_{s in g} y[o, s] - grand
    inter[o, g]  = mean_{s in g} y[o, s] - grand - gene[o] - genome[g]
    cond[o, s]   = y[o, s] - mean_{s' in g(s)} y[o, s']

Each term is the target of exactly one channel: `gene` of the gene channel,
`genome` of the species channel, `inter` of species x sequence, `cond` of the
experiment channel. `mu`, the quantity the absolute head already regresses, is
recovered exactly as ``grand + gene + genome + inter``, so the decomposition
adds three targets and replaces none.

Earlier measurements put the ceiling of `cond` near 10.92 % of total variance
and of `genome` near 2.31 %; :func:`component_report` recomputes both on the
data it is given, because a ceiling quoted from memory is how a dead channel
looks healthy. These targets therefore cannot move pooled PCC/SCC much, those
being dominated by the remaining 85 %. They exist so the channels become
steerable at design time, and :func:`per_group_metrics` is the metric that
can see them.

``cond`` is centred within (orthogroup, genome), not within the orthogroup
alone: the latter would fold the strain effect into the condition target and
train two channels on overlapping signal. The centring group is the same cell
the absolute head is trained on, so `cond` is exactly the residual that head
cannot reach, whatever it learns.

`combat` is 28.6 % empty, and an absent cell means "this strain has no such
gene", not "this gene is silent". Every term carries its own boolean mask,
masks are ANDed with the cell mask, and `sanitise()` in the heads layer is
still required because a masked NaN multiplied by zero is NaN.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .labels import SampleTable, read_matrix

__all__ = [
    "COMPONENTS",
    "Decomposition",
    "decompose",
    "component_report",
    "per_group_metrics",
    "reproducibility",
    "save_decomposition",
    "load_decomposition",
]

#: The four additive terms, in the order every array in this module uses.
#: Append-only: the index of a component is written into cached files.
COMPONENTS: Tuple[str, ...] = ("gene", "genome", "inter", "cond")

_EPS = 1e-12


# The container
@dataclass
class Decomposition:
    """Additive terms of the ComBat matrix, each with its own mask.

    Arrays indexed by ``i`` run over orthogroups (`og_ids`), by ``j`` over
    genomes (`genomes`), by ``k`` over kept samples (`sample_ids`).
    """

    og_ids: List[str]
    genomes: List[str]
    sample_ids: List[str]
    sample_genome: np.ndarray            # [n_samples] int32 -> genome index

    grand: float
    gene: np.ndarray                     # [n_og]      float32
    gene_ok: np.ndarray                  # [n_og]      bool
    genome_eff: np.ndarray               # [n_genomes] float32
    genome_ok: np.ndarray                # [n_genomes] bool
    inter: np.ndarray                    # [n_og, n_genomes] float32
    inter_ok: np.ndarray                 # [n_og, n_genomes] bool
    cond: np.ndarray                     # [n_og, n_samples] float32
    cond_ok: np.ndarray                  # [n_og, n_samples] bool

    mu: np.ndarray                       # [n_og, n_genomes] cell mean
    mu_ok: np.ndarray                    # [n_og, n_genomes] bool
    n_obs: np.ndarray                    # [n_og, n_genomes] int32

    notes: List[str] = field(default_factory=list)

    # Indexing helpers
    def og_index(self) -> Dict[str, int]:
        return {o: i for i, o in enumerate(self.og_ids)}

    def genome_index(self) -> Dict[str, int]:
        return {g: j for j, g in enumerate(self.genomes)}

    def sample_index(self) -> Dict[str, int]:
        return {s: k for k, s in enumerate(self.sample_ids)}

    def samples_of_genome(self, genome: str) -> np.ndarray:
        j = self.genome_index()[genome]
        return np.nonzero(self.sample_genome == j)[0]

    # The reconstruction identity
    def rebuild_mu(self) -> np.ndarray:
        """``grand + gene + genome + inter`` -- must equal `mu` where valid."""
        return (self.grand + self.gene[:, None] + self.genome_eff[None, :]
                + self.inter)

    def closure_error(self) -> float:
        """max |rebuild_mu - mu| over valid cells; zero up to float32 rounding.

        Anything else means a mask and its array stopped describing the same
        cells, which is what makes a term silently regress a quantity other
        than the one its name claims.
        """
        m = self.mu_ok
        if not m.any():
            return float("nan")
        d = np.abs(self.rebuild_mu()[m] - self.mu[m])
        return float(d.max())

    def report(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "n_orthogroups": len(self.og_ids),
            "n_genomes": len(self.genomes),
            "n_samples": len(self.sample_ids),
            "grand": float(self.grand),
            "cells_valid": int(self.mu_ok.sum()),
            "cells_total": int(self.mu_ok.size),
            "cond_cells_valid": int(self.cond_ok.sum()),
            "closure_error": self.closure_error(),
            "notes": list(self.notes),
        }
        out.update(component_report(self))
        return out


# The decomposition
def decompose(combat_path: str, samples: SampleTable,
              min_samples_per_genome: int = 1,
              max_orthogroups: int = 0) -> Decomposition:
    """Split `combat` into gene / genome / interaction / condition terms.

    Only samples with ``keep == True`` take part; the policy is the one in
    `labels.SamplePolicy` and is not re-derived here. `max_orthogroups`
    truncates for smoke runs and takes a hash-ordered subset rather than a
    prefix: the matrix is sorted by orthogroup id, so a prefix is a contiguous
    block of the ortholog clustering and not a sample of it.
    """
    og_ids, cols, mat = read_matrix(combat_path)
    if not og_ids:
        raise ValueError("no rows read from %r" % (combat_path,))

    pos = {s: j for j, s in enumerate(cols)}
    keep_ids, keep_col, keep_gen = [], [], []
    for sid, gen, keep in zip(samples.sample_ids, samples.genome,
                              samples.keep):
        if not keep:
            continue
        j = pos.get(sid)
        if j is None:
            continue
        keep_ids.append(sid)
        keep_col.append(j)
        keep_gen.append(gen)
    if not keep_ids:
        raise ValueError(
            "no kept sample of %r appears as a column of %r. The join key is "
            "the `sample` column of the metadata against the matrix header; a "
            "silent empty intersection here would produce an all-NaN "
            "decomposition that still has the right shape."
            % (getattr(samples, "source", "<sample table>"), combat_path))

    notes: List[str] = []
    genomes = sorted(set(keep_gen))
    # A genome with one sample has cond == 0 in every cell by construction
    # (the cell mean is that sample), so keeping it would add rows whose
    # target is zero and whose mask says "valid".
    counts = {g: keep_gen.count(g) for g in genomes}
    thin = [g for g in genomes if counts[g] < max(1, int(min_samples_per_genome))]
    if thin:
        notes.append(
            "genomes with < %d kept samples keep their mu/genome terms but "
            "their `cond` mask is False, because a deviation from a one-sample "
            "mean is identically zero: %s"
            % (int(min_samples_per_genome), ", ".join(sorted(thin))))

    gi = {g: j for j, g in enumerate(genomes)}
    sample_genome = np.array([gi[g] for g in keep_gen], dtype=np.int32)
    col = np.array(keep_col, dtype=np.int64)

    if max_orthogroups and max_orthogroups > 0:
        import hashlib
        order = sorted(range(len(og_ids)),
                       key=lambda i: hashlib.blake2b(
                           og_ids[i].encode("utf-8"), digest_size=8).digest())
        pick = np.array(sorted(order[:int(max_orthogroups)]), dtype=np.int64)
        og_ids = [og_ids[i] for i in pick]
        mat = mat[pick]
        notes.append("SUB-SAMPLED to %d orthogroups by blake2b(og_id)"
                     % (len(og_ids),))

    y = mat[:, col].astype(np.float64)              # [n_og, n_kept_samples]
    ok = np.isfinite(y)
    n_og, n_s = y.shape
    n_g = len(genomes)

    # Cell means (orthogroup x genome)
    mu = np.full((n_og, n_g), np.nan, dtype=np.float64)
    n_obs = np.zeros((n_og, n_g), dtype=np.int32)
    for j in range(n_g):
        sel = np.nonzero(sample_genome == j)[0]
        if sel.size == 0:
            continue
        sub_ok = ok[:, sel]
        cnt = sub_ok.sum(axis=1)
        tot = np.where(sub_ok, y[:, sel], 0.0).sum(axis=1)
        n_obs[:, j] = cnt.astype(np.int32)
        nz = cnt > 0
        mu[nz, j] = tot[nz] / cnt[nz]
    mu_ok = n_obs > 0

    # Grand / gene / genome / interaction. Every mean is unweighted over the
    # valid cells of its group: weighting by sample count would make `gene[o]`
    # a statement about which strains were sequenced most (reesei 386 samples
    # against P1 24) rather than a property of the gene.
    grand = float(np.nanmean(mu[mu_ok])) if mu_ok.any() else float("nan")

    gene = np.full(n_og, np.nan, dtype=np.float64)
    cnt_o = mu_ok.sum(axis=1)
    sum_o = np.where(mu_ok, np.nan_to_num(mu), 0.0).sum(axis=1)
    gene[cnt_o > 0] = sum_o[cnt_o > 0] / cnt_o[cnt_o > 0] - grand
    gene_ok = cnt_o > 0

    genome_eff = np.full(n_g, np.nan, dtype=np.float64)
    cnt_g = mu_ok.sum(axis=0)
    sum_g = np.where(mu_ok, np.nan_to_num(mu), 0.0).sum(axis=0)
    genome_eff[cnt_g > 0] = sum_g[cnt_g > 0] / cnt_g[cnt_g > 0] - grand
    genome_ok = cnt_g > 0

    inter = np.zeros((n_og, n_g), dtype=np.float64)
    fit = grand + gene[:, None] + genome_eff[None, :]
    inter[mu_ok] = mu[mu_ok] - fit[mu_ok]
    inter_ok = mu_ok & gene_ok[:, None] & genome_ok[None, :]

    # Conditional deviation
    cond = np.zeros((n_og, n_s), dtype=np.float64)
    cell_mu = mu[:, sample_genome]                  # [n_og, n_s] broadcast
    cell_n = n_obs[:, sample_genome]
    cond_ok = ok & np.isfinite(cell_mu) & (cell_n >= 2)
    cond[cond_ok] = y[cond_ok] - cell_mu[cond_ok]
    if thin:
        thin_cols = np.isin(sample_genome, [gi[g] for g in thin])
        cond_ok[:, thin_cols] = False

    d = Decomposition(
        og_ids=list(og_ids), genomes=list(genomes), sample_ids=list(keep_ids),
        sample_genome=sample_genome,
        grand=grand,
        gene=gene.astype(np.float32), gene_ok=gene_ok,
        genome_eff=genome_eff.astype(np.float32), genome_ok=genome_ok,
        inter=inter.astype(np.float32), inter_ok=inter_ok,
        cond=cond.astype(np.float32), cond_ok=cond_ok,
        mu=mu.astype(np.float32), mu_ok=mu_ok, n_obs=n_obs,
        notes=notes)
    err = d.closure_error()
    if np.isfinite(err) and err > 1e-3:
        raise RuntimeError(
            "grand + gene + genome + inter differs from the cell mean by "
            "%.3e. The four terms are an algebraic identity, so a nonzero "
            "residual means a mask and its array describe different cells and "
            "each head would regress something other than its name." % (err,))
    return d


# What each channel can possibly reach
def component_report(d: Decomposition) -> Dict[str, float]:
    """Share of total variance carried by each term, on this data.

    A ceiling, not a result: a channel cannot explain more of the target than
    its term carries, so a channel reporting a larger gain than its share here
    is reading another channel's signal. The shares do not sum to 1 -- the
    terms are orthogonal only in the balanced case, and `combat` is 28.6 %
    empty -- so the residual is reported as `share_crossterm` rather than
    distributed silently.
    """
    out: Dict[str, float] = {}
    obs = []
    parts = {k: [] for k in COMPONENTS}
    n_og, n_s = d.cond.shape
    for k in range(n_s):
        j = int(d.sample_genome[k])
        sel = d.cond_ok[:, k]
        if not sel.any():
            continue
        obs.append(d.mu[sel, j] + d.cond[sel, k])
        parts["gene"].append(d.gene[sel])
        parts["genome"].append(np.full(int(sel.sum()), d.genome_eff[j]))
        parts["inter"].append(d.inter[sel, j])
        parts["cond"].append(d.cond[sel, k])
    if not obs:
        return {"share_%s" % c: float("nan") for c in COMPONENTS}
    y = np.concatenate(obs).astype(np.float64)
    total = float(y.var()) or float("nan")
    acc = 0.0
    for c in COMPONENTS:
        v = float(np.concatenate(parts[c]).astype(np.float64).var())
        share = v / total if np.isfinite(total) and total > 0 else float("nan")
        out["share_%s" % c] = share
        if np.isfinite(share):
            acc += share
    out["share_crossterm"] = 1.0 - acc
    out["total_variance"] = total
    out["n_cells_scored"] = float(y.size)
    return out


def reproducibility(d: Decomposition, samples: SampleTable,
                    min_replicates: int = 2) -> Dict[str, float]:
    """How much of `cond` survives between replicates of the same condition.

    A deterministic function of the condition is constant inside a set of
    replicates, so whatever differs between replicates of one condition is
    unreachable by any condition encoder. This is the number that turns
    "14.78 % of the variance" into "10.92 % is achievable". Replicates are
    metadata rows sharing (genome, study, condition).
    """
    idx = {s: k for k, s in enumerate(d.sample_ids)}
    groups: Dict[Tuple[str, str, str], List[int]] = {}
    for sid, gen, study, cond in zip(samples.sample_ids, samples.genome,
                                     samples.study, samples.condition):
        k = idx.get(sid)
        if k is None:
            continue
        groups.setdefault((gen, study, cond), []).append(k)
    usable = [v for v in groups.values() if len(v) >= int(min_replicates)]
    if not usable:
        return {"reproducible_fraction": float("nan"), "n_groups": 0.0,
                "n_replicated_samples": 0.0}
    within, total = [], []
    for cols in usable:
        sub = d.cond[:, cols]
        mask = d.cond_ok[:, cols].all(axis=1)
        if not mask.any():
            continue
        s = sub[mask].astype(np.float64)
        within.append(((s - s.mean(axis=1, keepdims=True)) ** 2).ravel())
        total.append(s.ravel())
    if not within:
        return {"reproducible_fraction": float("nan"),
                "n_groups": float(len(usable)), "n_replicated_samples": 0.0}
    w = float(np.concatenate(within).mean())
    t = float(np.concatenate(total).var())
    frac = 1.0 - (w / t) if t > _EPS else float("nan")
    return {
        "reproducible_fraction": float(max(0.0, min(1.0, frac))),
        "within_replicate_variance": w,
        "conditional_variance": t,
        "n_groups": float(len(usable)),
        "n_replicated_samples": float(sum(len(v) for v in usable)),
    }


# The metric that can see a conditional channel
def per_group_metrics(pred: np.ndarray, target: np.ndarray,
                      group: np.ndarray,
                      min_size: int = 3) -> Dict[str, float]:
    """Correlation computed WITHIN each group, then averaged over groups.

    A pooled correlation on a conditional target is dominated by the
    between-group spread, which the conditional term does not carry, being
    centred within exactly that group: a channel worth 14.78 % of the variance
    moves a pooled metric by less than one seed's noise (test SE ~0.029 at
    1179 promoters) and is then written up as dead. Groups smaller than
    `min_size` are dropped rather than scored, because a Spearman over two
    points is +-1 whatever the model did.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    g = np.asarray(group).ravel()
    if not (p.shape == t.shape == g.shape):
        raise ValueError("pred %s, target %s and group %s must align"
                         % (p.shape, t.shape, g.shape))
    finite = np.isfinite(p) & np.isfinite(t)
    p, t, g = p[finite], t[finite], g[finite]
    pcc, scc, sizes = [], [], []
    for key in np.unique(g):
        sel = g == key
        n = int(sel.sum())
        if n < int(min_size):
            continue
        a, b = p[sel], t[sel]
        if a.std() <= _EPS or b.std() <= _EPS:
            # A constant prediction inside a group is the failure this metric
            # exists to catch, so it scores 0 rather than NaN: NaN would be
            # dropped by the mean and the collapse would be invisible.
            pcc.append(0.0)
            scc.append(0.0)
            sizes.append(n)
            continue
        pcc.append(float(np.corrcoef(a, b)[0, 1]))
        scc.append(float(np.corrcoef(_ranks(a), _ranks(b))[0, 1]))
        sizes.append(n)
    if not pcc:
        return {"within_pcc": float("nan"), "within_scc": float("nan"),
                "n_groups": 0.0, "n_scored": 0.0}
    w = np.array(sizes, dtype=np.float64)
    return {
        "within_pcc": float(np.mean(pcc)),
        "within_scc": float(np.mean(scc)),
        "within_pcc_weighted": float(np.average(pcc, weights=w)),
        "within_scc_weighted": float(np.average(scc, weights=w)),
        "n_groups": float(len(pcc)),
        "n_scored": float(w.sum()),
        "median_group_size": float(np.median(w)),
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties AVERAGED -- a constant scores 0, not +-1."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    out = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


# Cache
def save_decomposition(d: Decomposition, path: str) -> str:
    """npz plus a sidecar json. The json is what a human reads in a review."""
    np.savez_compressed(
        path,
        og_ids=np.array(d.og_ids, dtype=object),
        genomes=np.array(d.genomes, dtype=object),
        sample_ids=np.array(d.sample_ids, dtype=object),
        sample_genome=d.sample_genome,
        grand=np.array([d.grand], dtype=np.float32),
        gene=d.gene, gene_ok=d.gene_ok,
        genome_eff=d.genome_eff, genome_ok=d.genome_ok,
        inter=d.inter, inter_ok=d.inter_ok,
        cond=d.cond, cond_ok=d.cond_ok,
        mu=d.mu, mu_ok=d.mu_ok, n_obs=d.n_obs,
        components=np.array(COMPONENTS, dtype=object))
    with io.open(path + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(d.report(), indent=2, sort_keys=True,
                           default=float))
    return path


def load_decomposition(path: str) -> Decomposition:
    z = np.load(path, allow_pickle=True)
    stored = tuple(str(c) for c in z["components"])
    if stored != COMPONENTS:
        # The component order is written into every cached array's meaning.
        raise ValueError(
            "cached decomposition lists components %r, this build uses %r. "
            "The order is a contract: a reordered cache would hand the gene "
            "head the genome target with no shape error."
            % (stored, COMPONENTS))
    return Decomposition(
        og_ids=[str(o) for o in z["og_ids"]],
        genomes=[str(g) for g in z["genomes"]],
        sample_ids=[str(s) for s in z["sample_ids"]],
        sample_genome=z["sample_genome"],
        grand=float(z["grand"][0]),
        gene=z["gene"], gene_ok=z["gene_ok"],
        genome_eff=z["genome_eff"], genome_ok=z["genome_ok"],
        inter=z["inter"], inter_ok=z["inter_ok"],
        cond=z["cond"], cond_ok=z["cond_ok"],
        mu=z["mu"], mu_ok=z["mu_ok"], n_obs=z["n_obs"],
        notes=["loaded from %s" % (path,)])
