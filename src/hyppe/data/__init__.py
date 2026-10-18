"""Data layer: sequences, labels, splits, conditioning, pairs, negatives.

`dataset.build_pipeline()` is the entry point, the only object that composes
the other modules; every choice it makes is recorded in `Pipeline.report()`
rather than left to a default.

Nothing is imported eagerly here. `dataset` pulls in `hyppe.model.species`
for the strain registry, and a data module that cannot be imported without
the model package is one that cannot be tested without it.
"""

__all__ = [
    "conditioning", "condition_taxonomy", "dataset", "dna_lm", "kmer_index",
    "labels", "mutations", "negatives", "pairs", "sequences", "split_guard",
    "splits",
]
