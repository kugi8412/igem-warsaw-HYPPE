"""The shipped split assignment, and the properties the model relies on.
"""

from __future__ import annotations

import csv
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(HERE, "data", "splits", "splits.tsv")

PKS1_OG = "OG0000996"   # the design target
TEF1_OG = "OG0000518"   # the strong reference


@pytest.fixture(scope="module")
def assignment():
    with open(SPLITS, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {r["og"]: r["H_kmer90"] for r in rows}


def test_the_column_is_the_production_one(assignment):
    with open(SPLITS, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    assert header == ["og", "H_kmer90"]


def test_four_splits_and_nothing_else(assignment):
    assert set(assignment.values()) == {"train", "val", "spy", "test"}


def test_every_orthogroup_is_assigned_exactly_once(assignment):
    with open(SPLITS, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == len(assignment)       # no orthogroup appears twice
    assert len(assignment) == 26327


def test_the_delivered_sizes(assignment):
    counts = {}
    for s in assignment.values():
        counts[s] = counts.get(s, 0) + 1
    assert counts == {"train": 23622, "test": 1322, "val": 1056, "spy": 327}


def test_both_anchors_lie_in_val(assignment):
    """Not trained on, visible to epoch selection, absent from spy and test.
    """
    assert assignment[PKS1_OG] == "val"
    assert assignment[TEF1_OG] == "val"


def test_spy_is_small_but_not_degenerate(assignment):
    """`SPY` chooses BETWEEN models; `VAL` chooses within one run.
    An earlier split column gave spy a single orthogroup, which makes the
    veto unusable while still producing a number.
    """
    n_spy = sum(1 for v in assignment.values() if v == "spy")
    assert n_spy > 100


def test_held_out_is_a_meaningful_fraction(assignment):
    held = sum(1 for v in assignment.values() if v != "train")
    assert 0.08 < held / len(assignment) < 0.15
