"""`config/default.yaml` must still describe the delivered model.

These are not style checks. Every value asserted here changes the model, and
a model that differs reports the same headline metric names, so the only way
a drift is noticed is a test that names the number.
"""

from __future__ import annotations

import os

import pytest

from hyppe.config import load_config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(HERE, "config", "default.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_config(DEFAULT)


def test_the_loader_accepts_it_with_no_unknown_or_duplicate_keys(cfg):
    assert not cfg.duplicate_keys, cfg.duplicate_keys
    assert not cfg.unknown_keys, cfg.unknown_keys


def test_sequence_contract(cfg):
    assert cfg.data.sequence_length == 800
    assert cfg.data.alphabet == "ACGT"
    assert cfg.data.n_value == 0.25


def test_promoters_are_not_reverse_complement_invariant(cfg):
    assert cfg.data.random_rc is False
    assert cfg.data.rc_tta is False
    assert cfg.data.canonicalise_strand is False


def test_split_is_grouped_not_row_wise(cfg):
    assert cfg.data.split_by == "orthogroup"
    assert cfg.data.allow_leaky_split is False


def test_ladder_shape_is_the_delivered_one(cfg):
    m = cfg.model
    assert m.l0_codes == 128
    assert m.l0_dim == 16
    assert tuple(m.fsq_bits) == (9, 6, 3, 3)
    assert tuple(m.channels) == (192, 160, 128, 96, 64)
    assert tuple(m.strides) == (2, 2, 2, 2, 2)
    assert m.n_levels() == 5
    assert list(m.map_lengths(800)) == [25, 50, 100, 200, 400]


def test_the_latent_has_headroom_over_the_sequence(cfg):
    budget = cfg.model.bit_budget(800)
    assert 3180.0 < float(budget["total_bits"]) < 3195.0
    assert 1.95 < float(budget["reserve"]) < 2.05
    assert budget["fine_sufficient"] is True
    assert float(budget["levels"][0]["bits"]) == 175.0
    assert abs(float(budget["reconstruction_floor"]) - 0.2626) < 1e-4


def test_the_decoder_sees_codes_only(cfg):
    assert cfg.model.decoder_sees_codes_only is True
    assert cfg.model.allow_continuous_decoder is False


def test_dead_code_restart_is_on(cfg):
    assert cfg.model.vq_dead_restart is True
    assert cfg.model.vq_frechet is True


def test_ranking_has_no_margin(cfg):
    assert not hasattr(cfg.heads.pairwise, "margin")
    assert cfg.heads.pairwise.temperature == 1.0


def test_beta_nll_is_configured(cfg):
    # A plain Gaussian likelihood diverged in three epochs.
    assert cfg.heads.absolute.beta == 0.5
    assert cfg.heads.absolute.beta_mu == 1.0
    assert cfg.heads.absolute.tie_sigma_to_var is False


def test_the_discriminator_keeps_its_own_falsifier(cfg):
    # The dinucleotide shuffle is composition-matched, so a discriminator
    assert "dinucleotide_shuffle" in cfg.heads.discriminator.negative_kinds


def test_reconstruction_is_a_constraint_not_a_player(cfg):
    t = cfg.training
    assert t.method == "normprior"
    assert "reconstruction" not in t.normprior.players
    assert set(t.normprior.players) == {"absolute", "pairwise", "discriminator"}
    assert t.recon_constraint.enabled is True


def test_manifold_tensors_are_optimised_riemannianly(cfg):
    assert cfg.training.riemannian is True
    assert "codebook" in cfg.training.no_decay_patterns


def test_the_delivered_schedule(cfg):
    assert cfg.training.epochs == 16
    assert cfg.training.seed == 42
    assert cfg.training.grad_clip == 1.0


def test_relative_data_paths_resolve_against_the_data_root(monkeypatch):
    """The README promises HYPPE_DATA_ROOT; the loader has to honour it.
    Without this a relative path resolved against the CURRENT DIRECTORY, so
    the same config launched from the repo root and from the data root were
    two different runs, and the symptom was a missing file only when the file
    happened to be missing.
    """
    monkeypatch.setenv("HYPPE_DATA_ROOT", os.path.join("/", "data", "root"))
    cfg = load_config(DEFAULT)
    assert cfg.data.sequences_path.startswith(os.path.join("/", "data", "root"))
    assert cfg.conditioning.orthology_path.startswith(
        os.path.join("/", "data", "root"))
    assert cfg.training.spy_selection.panel_path.startswith(
        os.path.join("/", "data", "root"))


def test_a_clone_resolves_what_ships_without_a_data_root(monkeypatch):
    """A fresh clone has no HYPPE_DATA_ROOT, and must still find the files
    that ship inside the repository.
    The delivered checkpoints record the absolute paths of the machine that
    trained them, all under a home directory that exists nowhere else. With
    no data root set the resolver used to return immediately and leave those
    paths verbatim, so anything reading the orthology table failed on a
    directory nobody could have.
    """
    from hyppe.config import resolve_data_paths
    monkeypatch.delenv("HYPPE_DATA_ROOT", raising=False)
    cfg = load_config(DEFAULT)
    foreign = "/home/somebody/HYPPE/data/final_datasets/orthology/" \
              "og_table_combined.tsv"
    cfg.conditioning.orthology_path = foreign
    resolve_data_paths(cfg)
    assert cfg.conditioning.orthology_path != foreign, \
        "a path from another machine was kept as it stood"
    assert os.path.isfile(cfg.conditioning.orthology_path), \
        "it was rewritten to something that still does not exist"


def test_a_foreign_path_with_nowhere_to_go_is_left_alone(monkeypatch):
    """Rerooting must not invent a location for a file that does not ship.
    The corpus is not in the repository. Silently pointing its paths at
    something that happens to exist would be worse than failing: the error
    has to name the file that was actually wanted.
    """
    from hyppe.config import resolve_data_paths
    monkeypatch.delenv("HYPPE_DATA_ROOT", raising=False)
    cfg = load_config(DEFAULT)
    foreign = "/home/somebody/HYPPE/data/final_datasets/FINAL/nothing.tsv.gz"
    cfg.data.sequences_path = foreign
    resolve_data_paths(cfg)
    assert cfg.data.sequences_path == foreign


def test_absolute_data_paths_are_left_alone(monkeypatch):
    from hyppe.config import resolve_data_paths
    monkeypatch.setenv("HYPPE_DATA_ROOT", os.path.join("/", "data", "root"))
    cfg = load_config(DEFAULT)
    absolute = os.path.join("/", "elsewhere", "sequences.tsv.gz")
    cfg.data.sequences_path = absolute
    resolve_data_paths(cfg)
    assert cfg.data.sequences_path == absolute
