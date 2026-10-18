"""Baselines: the features are what they claim, and a known signal is found."""

from __future__ import annotations

import numpy as np

from hyppe.baselines import (
    evaluate_cells,
    fit_ridge,
    kmer_matrix,
    run_kmer_ridge,
    species_onehot,
    targets_of,
)
from hyppe.training.batches import Corpus


# FEATURES
def test_kmer_counts_are_counts():
    # "ACGT" repeated: every 2-mer of the cycle appears, nothing else does.
    codes = np.array([[0, 1, 2, 3] * 4], dtype=np.uint8)
    x = kmer_matrix(codes, k=2, normalise=False)
    assert x.sum() == codes.shape[1] - 1          # every window counted once
    assert (x > 0).sum() == 4                     # AC, CG, GT, TA


def test_kmer_rows_normalise_to_one():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(8, 200)).astype(np.uint8)
    x = kmer_matrix(codes, k=3)
    assert np.allclose(x.sum(axis=1), 1.0)
    assert x.shape[1] == 64


def test_n_windows_are_skipped_not_remapped():
    """Code 4 is N. A window covering it must contribute nothing at all."""
    clean = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.uint8)
    withn = clean.copy()
    withn[0, 3] = 4
    a = kmer_matrix(clean, k=3, normalise=False)
    b = kmer_matrix(withn, k=3, normalise=False)
    assert b.sum() == a.sum() - 3                 # three windows covered it
    assert b.shape[1] == 64                       # and no 5th base appeared


def test_kmers_are_not_canonicalised():
    """The model does not collapse strands, so neither may the baseline."""
    fwd = np.array([[0, 0, 0, 1]], dtype=np.uint8)          # AAAC
    rev = np.array([[2, 3, 3, 3]], dtype=np.uint8)          # GTTT = rc(AAAC)
    assert not np.allclose(kmer_matrix(fwd, 2), kmer_matrix(rev, 2))


def test_species_onehot_is_one_hot():
    s = species_onehot(np.array([0, 2, 2, 1]), n_species=4)
    assert s.shape == (4, 4)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert s[1, 2] == 1.0


# RIDGE
def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5)).astype(np.float32)
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    y = (x @ w_true + 7.0).astype(np.float32)[:, None]
    valid = np.ones_like(y, dtype=bool)
    w = fit_ridge(x, y, valid, lam=1e-6)
    assert np.allclose(w[:5, 0], w_true, atol=1e-3)
    assert abs(w[5, 0] - 7.0) < 1e-3


def test_ridge_uses_only_the_valid_cells_of_each_column():
    """A cell mask, not a row mask: columns are measured independently."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    y = np.zeros((200, 2), dtype=np.float32)
    y[:, 0] = x[:, 0] * 5.0
    y[:, 1] = x[:, 1] * -4.0
    valid = np.zeros_like(y, dtype=bool)
    valid[:100, 0] = True                          # disjoint halves
    valid[100:, 1] = True
    w = fit_ridge(x, y, valid, lam=1e-6)
    assert abs(w[0, 0] - 5.0) < 1e-2
    assert abs(w[1, 1] + 4.0) < 1e-2


def test_a_column_with_no_data_stays_zero():
    x = np.ones((10, 2), dtype=np.float32)
    y = np.zeros((10, 1), dtype=np.float32)
    w = fit_ridge(x, y, np.zeros_like(y, dtype=bool), lam=1.0)
    assert np.allclose(w, 0.0)


# METRICS
def test_evaluate_pools_cells_rather_than_averaging_columns():
    pred = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    target = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    valid = np.ones_like(pred, dtype=bool)
    m = evaluate_cells(pred, target, valid)
    assert m["n_cells"] == 6
    assert abs(m["pcc"] - 1.0) < 1e-12


def test_evaluate_ignores_invalid_cells():
    pred = np.array([[1.0, 999.0], [2.0, 999.0]])
    target = np.array([[1.0, -999.0], [2.0, -999.0]])
    valid = np.array([[True, False], [True, False]])
    assert evaluate_cells(pred, target, valid)["n_cells"] == 2


# END-2-END for baselines
def test_kmer_ridge_finds_the_planted_signal():
    """`Corpus.synthetic` makes strength a function of a 6-mer count.

    A bag-of-k-mers baseline must therefore score well above zero on it.
    If this fails, the harness is broken rather than the baseline weak.
    """
    corpus = Corpus.synthetic(n=256, seq_len=400, n_genomes=3, seed=5)
    r = run_kmer_ridge(corpus, k=3, with_species=True, eval_split="val")
    assert r.metrics["n_cells"] > 0
    assert r.metrics["pcc"] > 0.2, r.metrics
    assert r.n_features == 64 + 3
    assert r.chosen["lambda"] in tuple(
        __import__("hyppe.baselines", fromlist=["x"]).DEFAULT_LAMBDAS)


def test_species_only_cannot_explain_a_sequence_signal():
    """Control: the planted signal is in the sequence, not in the strain.

    Strain one-hot alone has three free numbers and must do far worse than
    the k-mer features; without this the previous test could be passing on
    the strain column alone.
    """
    corpus = Corpus.synthetic(n=256, seq_len=400, n_genomes=3, seed=5)
    full = run_kmer_ridge(corpus, k=3, with_species=True, eval_split="val")
    mu, valid = targets_of(corpus)
    sp = species_onehot(corpus.species_id)
    sp_only = fit_ridge(sp, np.nan_to_num(mu), valid, lam=1e-2)
    pred = sp @ sp_only[:-1] + sp_only[-1]
    va = corpus.indices("val")
    m = evaluate_cells(pred[va], mu[va], valid[va])
    assert full.metrics["pcc"] > m["pcc"] + 0.1, (full.metrics, m)
