"""The split builder: grouping, determinism, and the leakage it must prevent."""

from __future__ import annotations

import random

import pytest

from hyppe.data.make_splits import (
    SPLIT_NAMES,
    SplitSpec,
    build_splits,
    canonical_kmers,
    cluster_by_similarity,
    falsifiers,
    sketch,
    write_split_table,
)


def _random_seq(rng: random.Random, n: int = 800) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, rng: random.Random, n_changes: int) -> str:
    s = list(seq)
    for _ in range(n_changes):
        i = rng.randrange(len(s))
        s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)


def _corpus(n_families: int = 30, per_family: int = 3, seed: int = 1,
            n_changes: int = 1):
    """Families of near-identical sequences: the leakage this must catch."""
    rng = random.Random(seed)
    records = []
    truth = {}
    for f in range(n_families):
        founder = _random_seq(rng)
        for m in range(per_family):
            rid = "f%02d_m%d" % (f, m)
            records.append((rid, founder if m == 0 else
                            _mutate(founder, rng, n_changes)))
            truth[rid] = f
    return records, truth


# Primitives
def test_canonical_kmers_collapse_the_two_strands():
    fwd = "ACGTACGTAC"
    rev = "GTACGTACGT"          # reverse complement of fwd
    assert canonical_kmers(fwd, 6) == canonical_kmers(rev, 6)


def test_canonical_kmers_skip_non_acgt():
    assert canonical_kmers("ACGTNACGT", 4) == canonical_kmers("ACGT", 4) | \
        canonical_kmers("ACGT", 4)


def test_sketch_is_a_deterministic_subset():
    rng = random.Random(0)
    seq = _random_seq(rng)
    full = canonical_kmers(seq, 25)
    s1 = sketch(seq, 25, 8)
    s2 = sketch(seq, 25, 8)
    assert s1 == s2                      # same input, same sketch
    assert s1 <= full                    # a subset, never an invention
    assert 0 < len(s1) < len(full)       # and a real reduction


def test_sketch_scale_one_keeps_everything():
    seq = _random_seq(random.Random(2))
    assert sketch(seq, 25, 1) == frozenset(canonical_kmers(seq, 25))


# Clusters
def test_near_duplicates_land_in_one_cluster():
    """One SNP in 800 bp: containment stays far above the threshold."""
    records, truth = _corpus(n_changes=1)
    spec = SplitSpec()
    sketches = {r: sketch(s, spec.k, spec.scale, spec.seed) for r, s in records}
    clusters = cluster_by_similarity(sketches, spec)
    for rid, fam in truth.items():
        same = [o for o, f in truth.items() if f == fam]
        assert len({clusters[o] for o in same}) == 1, rid


def test_k_mer_clustering_does_not_reach_diverged_orthologues():

    records, truth = _corpus(n_changes=5)
    spec = SplitSpec()
    sketches = {r: sketch(s, spec.k, spec.scale, spec.seed) for r, s in records}
    clusters = cluster_by_similarity(sketches, spec)
    split_families = sum(
        1 for fam in set(truth.values())
        if len({clusters[o] for o, f in truth.items() if f == fam}) > 1)
    assert split_families > 0


def test_groups_hold_diverged_orthologues_together():
    """What the k-mer graph cannot see, the orthogroup key must carry."""
    records, truth = _corpus(n_changes=5)
    groups = {rid: "og%02d" % fam for rid, fam in truth.items()}
    build = build_splits(records, groups=groups)
    assert set(build.assignment) == {"og%02d" % f for f in set(truth.values())}


def test_unrelated_sequences_do_not_merge():
    rng = random.Random(3)
    records = [("r%d" % i, _random_seq(rng)) for i in range(40)]
    spec = SplitSpec()
    sketches = {r: sketch(s, spec.k, spec.scale, spec.seed) for r, s in records}
    clusters = cluster_by_similarity(sketches, spec)
    assert len(set(clusters.values())) == len(records)


# Assignment
def test_assignment_is_deterministic_across_calls():
    records, _ = _corpus()
    a = build_splits(records).assignment
    b = build_splits(records).assignment
    assert a == b


def test_a_different_seed_gives_a_different_split():
    records, _ = _corpus()
    a = build_splits(records, spec=SplitSpec(seed=1)).assignment
    b = build_splits(records, spec=SplitSpec(seed=2)).assignment
    assert a != b


def test_no_cluster_is_split_across_two_sets():
    """The whole point: a homology cluster must not straddle a boundary."""
    records, truth = _corpus(n_changes=1)
    build = build_splits(records)
    for fam in set(truth.values()):
        members = [r for r, f in truth.items() if f == fam]
        assert len({build.assignment[m] for m in members}) == 1


def test_groups_are_never_separated_even_against_the_sequences():
    rng = random.Random(4)
    records = [("r%d" % i, _random_seq(rng)) for i in range(40)]
    groups = {"r%d" % i: "g%d" % (i // 4) for i in range(40)}
    build = build_splits(records, groups=groups)
    assert set(build.assignment) == {"g%d" % i for i in range(10)}


def test_every_requested_split_is_populated():
    records, _ = _corpus(n_families=60)
    build = build_splits(records)
    for s in SPLIT_NAMES:
        assert build.counts[s] > 0, s


def test_fractions_are_approximately_honoured():
    records, _ = _corpus(n_families=200, per_family=2, seed=5)
    build = build_splits(records)
    obs = build.report()["fractions_observed"]
    for s, want in build.spec.fractions.items():
        assert abs(obs[s] - want) < 0.05, (s, obs[s], want)


def test_a_zero_fraction_split_stays_empty():
    records, _ = _corpus(n_families=40)
    spec = SplitSpec(fractions={"train": 0.9, "val": 0.1,
                                "spy": 0.0, "test": 0.0})
    build = build_splits(records, spec=spec)
    assert build.counts["spy"] == 0 and build.counts["test"] == 0


# ------------------------------------------------------------ the verdict
def test_falsifiers_pass_on_a_sane_corpus():
    records, _ = _corpus(n_families=60)
    rep = falsifiers(build_splits(records))
    assert rep["ok"] is True
    assert rep["clusters_straddling_splits"] == 0


def test_falsifiers_catch_a_degenerate_clustering():
    """One sequence repeated is one cluster, and then nothing is held out."""
    seq = _random_seq(random.Random(6))
    records = [("r%d" % i, seq) for i in range(20)]
    rep = falsifiers(build_splits(records))
    assert rep["largest_cluster_below_half"] is False
    assert rep["ok"] is False


def test_written_table_round_trips(tmp_path):
    import csv
    records, _ = _corpus(n_families=20)
    build = build_splits(records)
    path = tmp_path / "splits.tsv"
    write_split_table(build, str(path), group_column="og",
                      split_column="H_kmer90")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert list(rows[0]) == ["og", "H_kmer90"]
    assert {r["og"]: r["H_kmer90"] for r in rows} == build.assignment


# Validation
def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitSpec(fractions={"train": 0.5, "val": 0.1,
                             "spy": 0.1, "test": 0.1}).validate()


def test_a_missing_split_is_refused():
    with pytest.raises(ValueError):
        SplitSpec(fractions={"train": 0.8, "val": 0.2}).validate()


def test_an_empty_corpus_is_refused():
    with pytest.raises(ValueError):
        build_splits([])
