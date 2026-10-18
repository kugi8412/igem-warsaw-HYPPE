"""Synthetic negatives: each class must destroy what it claims and keep the rest."""

from __future__ import annotations

import random

import pytest

from hyppe.data.make_negatives import (
    NEGATIVE_KINDS,
    chimera,
    composition_falsifier,
    dinucleotide_identical,
    dinucleotide_shuffle,
    make_negatives,
    reverse_complement,
)
from hyppe.data.negatives import kmer_counts


def _seq(rng: random.Random, n: int = 400) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _positives(n: int = 24, seed: int = 1):
    rng = random.Random(seed)
    return [("p%02d" % i, _seq(rng)) for i in range(n)]


# DINUCLEOTIDE SHUFFLE
def test_shuffle_preserves_dinucleotides_exactly():
    s = _seq(random.Random(2))
    out = dinucleotide_shuffle(s, random.Random(3))
    assert kmer_counts(out, 2) == kmer_counts(s, 2)
    assert dinucleotide_identical(s, out)


def test_shuffle_destroys_longer_order():
    """If 6-mer counts survived too, the negative would not be a negative."""
    s = _seq(random.Random(4), 800)
    out = dinucleotide_shuffle(s, random.Random(5))
    assert kmer_counts(out, 6) != kmer_counts(s, 6)


def test_shuffle_never_returns_its_input():
    s = _seq(random.Random(6))
    assert dinucleotide_shuffle(s, random.Random(7)) != s


def test_shuffle_refuses_a_non_acgt_source():
    with pytest.raises(ValueError):
        dinucleotide_shuffle("ACGTN" * 20, random.Random(8))


# CHIMERA Promoter
def test_chimera_takes_each_side_from_the_parent_it_claims():
    a, b = "A" * 100, "C" * 100
    out, bp = chimera(a, b, breakpoint=40)
    assert bp == 40
    assert out == "A" * 40 + "C" * 60
    assert len(out) == 100


def test_chimera_keeps_the_length_so_positions_keep_their_distance():
    rng = random.Random(9)
    a, b = _seq(rng), _seq(rng)
    out, _ = chimera(a, b, rng=rng)
    assert len(out) == len(a)


def test_chimera_breakpoint_stays_away_from_the_ends():
    rng = random.Random(10)
    a, b = _seq(rng, 800), _seq(rng, 800)
    for _ in range(50):
        _, bp = chimera(a, b, rng=rng)
        assert 80 <= bp <= 720


def test_chimera_refuses_parents_of_different_length():
    with pytest.raises(ValueError):
        chimera("A" * 10, "C" * 11)


# Reverse
def test_reverse_complement_is_an_involution():
    s = _seq(random.Random(11))
    assert reverse_complement(reverse_complement(s)) == s


def test_reverse_complement_preserves_composition_but_not_the_sequence():
    s = _seq(random.Random(12), 800)
    rc = reverse_complement(s)
    assert s.count("A") == rc.count("T") and s.count("G") == rc.count("C")
    assert rc != s


# Pool
def test_pool_is_deterministic_in_the_seed():
    pos = _positives()
    a = make_negatives(pos, seed=42)
    b = make_negatives(pos, seed=42)
    assert [x.sequence for x in a] == [x.sequence for x in b]
    c = make_negatives(pos, seed=43)
    assert [x.sequence for x in a] != [x.sequence for x in c]


def test_pool_size_follows_the_fraction():
    pos = _positives(40)
    out = make_negatives(pos, fraction=0.5)
    assert 15 <= len(out) <= 25


def test_every_negative_carries_its_provenance():
    pos = _positives()
    ids = {r for r, _ in pos}
    for neg in make_negatives(pos, kinds=("dinucleotide_shuffle", "chimera")):
        assert neg.kind in NEGATIVE_KINDS
        assert neg.source_id in ids
        if neg.kind == "chimera":
            assert neg.partner_id in ids and neg.partner_id != neg.source_id
            assert neg.breakpoint is not None


def test_chimeras_never_cross_a_group():
    """Crossing genomes would change composition, which is the shortcut."""
    pos = _positives(40)
    group_of = {r: ("g0" if i < 20 else "g1") for i, (r, _) in enumerate(pos)}
    for neg in make_negatives(pos, kinds=("chimera",), fraction=1.0,
                              group_of=group_of):
        assert group_of[neg.source_id] == group_of[neg.partner_id]


def test_the_mix_is_respected():
    pos = _positives(60)
    out = make_negatives(pos, kinds=("dinucleotide_shuffle", "reverse"),
                         mix={"dinucleotide_shuffle": 3.0, "reverse": 1.0},
                         fraction=1.0)
    kinds = [n.kind for n in out]
    assert kinds.count("dinucleotide_shuffle") > 2 * kinds.count("reverse")


def test_the_donor_class_refuses_to_be_invented():
    with pytest.raises(ValueError):
        make_negatives(_positives(), kinds=("donor",))


def test_the_donor_class_uses_the_sequence_it_was_given():
    pos = _positives(8)
    donor = "G" * 4000
    out = make_negatives(pos, kinds=("donor",), fraction=1.0,
                         donors=[("d0", donor)])
    assert out and all(set(n.sequence) == {"G"} for n in out)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        make_negatives(_positives(), kinds=("not_a_kind",))


# The falsifier
def test_composition_falsifier_passes_on_matched_negatives():
    """Shuffles and chimeras must not be separable on composition."""
    pos = _positives(40)
    neg = make_negatives(pos, kinds=("dinucleotide_shuffle", "chimera"),
                         fraction=1.0)
    rep = composition_falsifier([s for _, s in pos],
                                [n.sequence for n in neg])
    assert rep["ok"] is True, rep


def test_composition_falsifier_catches_an_unmatched_pool():
    """A GC-rich 'negative' makes the head a GC detector; this must fail."""
    rng = random.Random(13)
    pos = [_seq(rng, 400) for _ in range(30)]
    neg = ["".join(rng.choice("GC") for _ in range(400)) for _ in range(30)]
    rep = composition_falsifier(pos, neg)
    assert rep["ok"] is False
    assert rep["gc_auc"] < 0.1          # perfectly separable, in one direction
