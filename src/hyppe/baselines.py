"""Baselines the full architecture has to beat to justify itself.

Every component of the model -- the hyperbolic quantiser, the motif bank, the
phylogenetic conditioning, the reconstruction constraint -- is a claim that it
earns its place, and such a claim is only testable against something simpler
trained and evaluated identically. `kmer_ridge` is a bag of k-mers plus the
strain as a one-hot, in closed form: what a model has learned if it has learned
composition and nothing else. `cnn_onehot` is a small convolution under the
same supervision, isolating what the discrete ladder buys over an ordinary CNN.

Both read the same `Corpus`, use the same orthogroup-grouped split, predict the
same per-(orthogroup, genome, category) cell and are scored by the same pooled
Pearson and Spearman as the full model's `pcc` / `scc`; a baseline
hyperparameter is chosen on `val`, never on `test`. Neither sees the experiment
text, the DNA-LM flank vectors or the orthology features, so the gap to the
full model is the measured value of everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BaselineResult",
    "kmer_matrix",
    "species_onehot",
    "fit_ridge",
    "ridge_normals",
    "solve_ridge",
    "evaluate_cells",
    "axis_report",
    "within_row_contrast",
    "targets_of",
    "run_kmer_ridge",
    "run_cnn_onehot",
]

#: Ridge penalties swept on `val`. Wide and logarithmic, because the feature
#: count moves three orders of magnitude between k = 3 and k = 6 and a single
#: hand-picked lambda would favour one of them by accident.
DEFAULT_LAMBDAS: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


@dataclass
class BaselineResult:
    """One baseline, scored the way the full model is scored."""

    name: str
    n_features: int
    n_params: int
    metrics: Dict[str, float] = field(default_factory=dict)
    chosen: Dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0

    def row(self) -> str:
        m = self.metrics
        return ("%-22s %10d %12d  pcc %.5f  scc %.5f  n %6d"
                % (self.name, self.n_features, self.n_params,
                   m.get("pcc", float("nan")), m.get("scc", float("nan")),
                   int(m.get("n_cells", 0))))


# Features
def kmer_matrix(codes: np.ndarray, k: int = 4,
                normalise: bool = True) -> np.ndarray:
    """Counts of every k-mer, one row per sequence.

    Base 4 over A, C, G, T. A position holding N (code 4) invalidates every
    window covering it, and those windows are skipped rather than mapped to a
    neighbouring k-mer, which would put a count where no observation was.
    k-mers are not collapsed onto their reverse complement, because the model
    does not canonicalise either.
    """
    n, length = codes.shape
    n_feat = 4 ** k
    out = np.zeros((n, n_feat), dtype=np.float32)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    for i in range(n):
        row = codes[i].astype(np.int64)
        if length < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(row, k)
        good = (windows < 4).all(axis=1)
        if not good.any():
            continue
        idx = (windows[good] * powers).sum(axis=1)
        np.add.at(out[i], idx, 1.0)
    if normalise:
        total = out.sum(axis=1, keepdims=True)
        np.divide(out, np.maximum(total, 1.0), out=out)
    return out


def species_onehot(species_id: np.ndarray,
                   n_species: Optional[int] = None) -> np.ndarray:
    """The strain as a one-hot channel.

    A one-hot and not an embedding: the strain enters as an unstructured
    categorical with no phylogeny, so that the full model's Lorentz strain
    registry has something to be compared to.
    """
    n_species = int(n_species or (species_id.max() + 1))
    out = np.zeros((species_id.shape[0], n_species), dtype=np.float32)
    out[np.arange(species_id.shape[0]), species_id.astype(np.int64)] = 1.0
    return out


# Ridge
def ridge_normals(x: np.ndarray, y: np.ndarray,
                  valid: np.ndarray) -> List[Optional[Tuple]]:
    """Per-column centred normal equations, computed once.

    The Gram matrix does not depend on the penalty, so everything
    lambda-independent is hoisted here rather than rebuilt once per lambda.

    The mask is per cell and not per row: a promoter may have a measured value
    under 'carbon source' and none under 'light response', so dropping whole
    rows would discard most of the corpus and filling the gaps with a mean
    would train the model to predict that mean.
    """
    n_col = y.shape[1]
    out: List[Optional[Tuple]] = []
    for c in range(n_col):
        m = valid[:, c]
        if int(m.sum()) < 2:
            out.append(None)
            continue
        xc = x[m]                              # float32 view-copy, not float64
        yc = y[m, c].astype(np.float64)
        mx = xc.mean(axis=0, dtype=np.float64)
        my = float(yc.mean())
        # Centred Gram without materialising the centred matrix:
        #   (X - mx)^T (X - mx) = X^T X - n mx mx^T
        gram = (xc.T @ xc).astype(np.float64)
        n = float(xc.shape[0])
        gram -= n * np.outer(mx, mx)
        xty = (xc.T.astype(np.float64) @ yc) - n * mx * my
        out.append((gram, xty, mx, my))
    return out


def solve_ridge(normals: Sequence[Optional[Tuple]], n_feat: int,
                lam: float) -> np.ndarray:
    """Solve the hoisted normal equations at one penalty.

    The intercept is carried by the centring and is never penalised.
    """
    w = np.zeros((n_feat + 1, len(normals)), dtype=np.float64)
    for c, item in enumerate(normals):
        if item is None:
            continue
        gram, xty, mx, my = item
        a = gram.copy()
        a.flat[:: n_feat + 1] += lam
        try:
            coef = np.linalg.solve(a, xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(a, xty, rcond=None)[0]
        w[:n_feat, c] = coef
        w[n_feat, c] = my - mx @ coef
    return w


def fit_ridge(x: np.ndarray, y: np.ndarray, valid: np.ndarray,
              lam: float) -> np.ndarray:
    """Ridge with a per-cell mask. Convenience wrapper over the two above."""
    return solve_ridge(ridge_normals(x, y, valid), x.shape[1], float(lam))


def _apply(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ w[:-1] + w[-1]


# Metrics
def targets_of(corpus, n_categories: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """`(mu, valid)` as 2-D arrays, whatever shape the corpus carries.

    A per-category corpus has `mu` of shape [N, C(+1)], a single-target one
    [N]. Both are real inputs, so the shape is normalised once here rather
    than guessed by each baseline.
    """
    mu = np.asarray(corpus.mu, dtype=np.float32)
    if mu.ndim == 1:
        mu = mu[:, None]
    if n_categories is not None:
        mu = mu[:, :int(n_categories)]
    return mu, np.isfinite(mu)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        return r
    return _pearson(rank(a), rank(b))


def within_row_contrast(pred: np.ndarray, target: np.ndarray,
                        valid: np.ndarray,
                        min_categories: int = 2) -> Dict[str, float]:
    """How much of the per-category structure a model actually captures.

    A pooled correlation over cells rewards getting the row level right --
    "this gene is highly expressed" -- which is most of the variance and has
    nothing to do with the growth condition. Subtracting each row's mean over
    its measured categories from both prediction and target leaves only the
    part that varies with the condition, which makes the null explicit: a
    model constant along the row scores exactly zero here, however good its
    pooled number is. `within_fraction` says how much of the total cell
    variance this component holds, so a blind model may still be useful.
    """
    py, ty = [], []
    rows_used = 0
    for i in range(target.shape[0]):
        m = valid[i]
        if int(m.sum()) < int(min_categories):
            continue
        t = target[i, m].astype(np.float64)
        p = pred[i, m].astype(np.float64)
        ty.append(t - t.mean())
        py.append(p - p.mean())
        rows_used += 1
    if rows_used < 2:
        return {"within_row_pcc": float("nan"), "within_row_rows": 0.0}
    tv = np.concatenate(ty)
    pv = np.concatenate(py)
    all_t = target[valid].astype(np.float64)
    total_var = float(np.var(all_t)) if all_t.size else float("nan")
    return {
        "within_row_pcc": _pearson(pv, tv),
        "within_row_scc": _spearman(pv, tv),
        "within_row_rows": float(rows_used),
        "within_row_cells": float(tv.size),
        # Exactly 0 means constant across categories, and then the score
        # above is 0 by construction rather than by failure.
        "within_row_pred_sd": float(np.std(pv)),
        "within_row_target_sd": float(np.std(tv)),
        "within_fraction": (float(np.var(tv)) / total_var
                            if total_var else float("nan")),
    }


def axis_report(pred: np.ndarray, target: np.ndarray, valid: np.ndarray,
                og: Sequence[str], genome: Sequence[str],
                rows: np.ndarray) -> Dict[str, float]:
    """Split the score onto the three axes the full model also reports.

    A pooled correlation is dominated by one axis: promoters of different
    genes differ far more than one gene does between strains or conditions, so
    two models can sit within 0.03 of each other pooled while one predicts the
    condition and the other does not predict it at all.

        os_ortogrupa   which gene        mean_og(y) - mean(y)
        os_genomu      which ortholog    y - mean_og(y)
        os_warunku     which condition   the part of that the category explains

    Calls the training loop's own `rozbicie_osi`, so the numbers come from the
    same code that produced the model's.
    """
    try:
        from hyppe.training.loop import rozbicie_osi
    except Exception:                                  # noqa: BLE001
        return {}
    p, t, og_c, gen_c, cat_c = [], [], [], [], []
    for j, row in enumerate(rows):
        for c in range(target.shape[1]):
            if not valid[j, c]:
                continue
            p.append(float(pred[j, c]))
            t.append(float(target[j, c]))
            og_c.append(str(og[row]))
            gen_c.append(str(genome[row]))
            cat_c.append(int(c))
    if len(t) < 8:
        return {}
    out = rozbicie_osi(np.asarray(p), np.asarray(t), np.asarray(og_c),
                       np.asarray(gen_c), np.asarray(cat_c))
    return {k: float(v) for k, v in out.items()
            if isinstance(v, (int, float))}


def evaluate_cells(pred: np.ndarray, target: np.ndarray,
                   valid: np.ndarray) -> Dict[str, float]:
    """Pearson and Spearman over every valid cell, pooled.

    The full model's `pcc` / `scc`: cells, not rows, concatenated across
    categories. Scoring per category and averaging is a different question
    and gives a different number.
    """
    p = pred[valid].astype(np.float64)
    t = target[valid].astype(np.float64)
    return {"pcc": _pearson(p, t), "scc": _spearman(p, t),
            "mse": float(np.mean((p - t) ** 2)) if p.size else float("nan"),
            "pred_sd": float(np.std(p)) if p.size else float("nan"),
            "target_sd": float(np.std(t)) if t.size else float("nan"),
            "n_cells": float(p.size)}


# Baseline 1: k-mer ridge
def run_kmer_ridge(corpus, k: int = 4, with_species: bool = True,
                   lambdas: Sequence[float] = DEFAULT_LAMBDAS,
                   eval_split: str = "val",
                   n_categories: Optional[int] = None) -> BaselineResult:
    """k-mer counts (+ strain one-hot) -> ridge, one column per category.

    `lambdas` is swept on `val` and the winner reported on `eval_split`. When
    those are the same split the number is optimistic by exactly the selection
    it performed; pass `eval_split="test"` for a clean read, once.
    """
    import time
    t0 = time.time()

    feats = [kmer_matrix(corpus.codes, k)]
    if with_species:
        feats.append(species_onehot(corpus.species_id))
    x = np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]

    mu, valid = targets_of(corpus, n_categories)

    tr = corpus.indices("train")
    va = corpus.indices("val")
    ev = corpus.indices(eval_split)

    normals = ridge_normals(x[tr], np.nan_to_num(mu[tr]), valid[tr])
    best = (float("-inf"), None, None)
    for lam in lambdas:
        w = solve_ridge(normals, x.shape[1], float(lam))
        m = evaluate_cells(_apply(x[va], w), mu[va], valid[va])
        score = m["pcc"] if m["pcc"] == m["pcc"] else float("-inf")
        if score > best[0]:
            best = (score, float(lam), w)

    _, lam, w = best
    pred_ev = _apply(x[ev], w)
    metrics = evaluate_cells(pred_ev, mu[ev], valid[ev])
    metrics.update(axis_report(pred_ev, mu[ev], valid[ev],
                               corpus.og, corpus.genome, ev))
    metrics.update(within_row_contrast(pred_ev, mu[ev], valid[ev]))
    name = "kmer%d%s_ridge" % (k, "+species" if with_species else "")
    return BaselineResult(
        name=name, n_features=int(x.shape[1]), n_params=int(w.size),
        metrics=metrics,
        chosen={"lambda": lam, "val_pcc": best[0], "k": k,
                "with_species": with_species, "eval_split": eval_split},
        seconds=time.time() - t0)


# Baseline 2: plain CNN
def run_cnn_onehot(corpus, epochs: int = 8, batch_size: int = 64,
                   lr: float = 1e-3, width: int = 64, depth: int = 4,
                   with_species: bool = True, seed: int = 42,
                   device: str = "cpu", eval_split: str = "val",
                   n_categories: Optional[int] = None) -> BaselineResult:
    """A plain CNN on one-hot sequence, predicting the category row.

    No reconstruction, no quantiser, no motif bank, no text. The strain, when
    included, is appended as constant one-hot channels along the sequence --
    the crudest possible way to give a convolution a categorical, deliberately
    so. The loss is a masked mean squared error over measured cells, and the
    epoch reported is the one that scored best on `val`.
    """
    import time

    import torch
    from torch import nn

    t0 = time.time()
    torch.manual_seed(int(seed))
    dev = torch.device(device)

    mu, valid = targets_of(corpus, n_categories)
    n_out = mu.shape[1]

    codes = torch.from_numpy(np.asarray(corpus.codes, dtype=np.int64))
    sp = torch.from_numpy(species_onehot(corpus.species_id))
    y = torch.from_numpy(np.nan_to_num(mu))
    m = torch.from_numpy(valid.astype(np.float32))
    n_sp = int(sp.shape[1])
    in_ch = 4 + (n_sp if with_species else 0)

    layers: List[nn.Module] = []
    c_in = in_ch
    for i in range(depth):
        layers += [nn.Conv1d(c_in, width, 7, padding=3, dilation=1),
                   nn.BatchNorm1d(width), nn.SiLU(),
                   nn.MaxPool1d(2)]
        c_in = width
    net = nn.Sequential(*layers).to(dev)
    head = nn.Sequential(nn.Linear(2 * width, 128), nn.SiLU(),
                         nn.Linear(128, n_out)).to(dev)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.01)
    n_params = sum(int(p.numel()) for p in params)

    def forward(idx: torch.Tensor) -> torch.Tensor:
        c = codes[idx].to(dev)
        # Code 4 is N and becomes an all-zero column, which is the same
        # "no evidence" signal the full model gives it as a uniform 0.25.
        x = torch.zeros(c.shape[0], 4, c.shape[1], device=dev)
        onehot = (c < 4)
        x.scatter_(1, c.clamp(max=3).unsqueeze(1), onehot.unsqueeze(1).float())
        if with_species:
            s = sp[idx].to(dev).unsqueeze(-1).expand(-1, -1, c.shape[1])
            x = torch.cat([x, s], dim=1)
        h = net(x)
        # mean and max: the pair the full model's pooling also reduces to.
        return head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    tr = torch.from_numpy(corpus.indices("train"))
    va = torch.from_numpy(corpus.indices("val"))
    ev = torch.from_numpy(corpus.indices(eval_split))
    gen = torch.Generator().manual_seed(int(seed))

    def predict(idx: torch.Tensor) -> np.ndarray:
        net.eval(); head.eval()
        out = []
        with torch.no_grad():
            for s in range(0, idx.numel(), 256):
                out.append(forward(idx[s:s + 256]).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, n_out))

    best = (float("-inf"), -1, None)
    for ep in range(1, int(epochs) + 1):
        net.train(); head.train()
        order = tr[torch.randperm(tr.numel(), generator=gen)]
        for s in range(0, order.numel(), batch_size):
            b = order[s:s + batch_size]
            if b.numel() < 2:
                continue
            pred = forward(b)
            mb, yb = m[b].to(dev), y[b].to(dev)
            denom = mb.sum().clamp_min(1.0)
            loss = (((pred - yb) ** 2) * mb).sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        mv = evaluate_cells(predict(va), mu[va.numpy()], valid[va.numpy()])
        score = mv["pcc"] if mv["pcc"] == mv["pcc"] else float("-inf")
        if score > best[0]:
            best = (score, ep,
                    ({k_: v.detach().cpu().clone()
                      for k_, v in net.state_dict().items()},
                     {k_: v.detach().cpu().clone()
                      for k_, v in head.state_dict().items()}))

    if best[2] is not None:
        net.load_state_dict(best[2][0]); head.load_state_dict(best[2][1])
    _evn = ev.numpy()
    pred_ev = predict(ev)
    metrics = evaluate_cells(pred_ev, mu[_evn], valid[_evn])
    metrics.update(axis_report(pred_ev, mu[_evn], valid[_evn],
                               corpus.og, corpus.genome, _evn))
    metrics.update(within_row_contrast(pred_ev, mu[_evn], valid[_evn]))
    name = "cnn_onehot%s" % ("+species" if with_species else "")
    return BaselineResult(
        name=name, n_features=int(in_ch), n_params=n_params, metrics=metrics,
        chosen={"best_epoch": best[1], "val_pcc": best[0], "epochs": epochs,
                "width": width, "depth": depth, "lr": lr, "seed": seed,
                "with_species": with_species, "eval_split": eval_split},
        seconds=time.time() - t0)
