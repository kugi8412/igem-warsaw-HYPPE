# Probe fixture

`p1_val_promoters.tsv` — 348 promoters of *Trichoderma atroviride* P1 taken
from the **validation** split, 800 bp each, with their orthogroup identifiers.

This is a fixture for the interpretability probes, not training data and not
the corpus. It is here so that `scripts/ism_words.py` and the XAI notebook run
from a fresh clone without first rebuilding the corpus. The full corpus is
assembled from public archives, reached through `HYPPE_DATA_ROOT`, and is not
part of this repository — see the README for why.

Validation, not test: this repository reads the test split once, through
`parts/common/ocen_test.py`, and an interpretability probe is not that
occasion. `scripts/ism_words.py` refuses `--split test` outright.

Columns: `og_id`, `genome`, `promoter`, `split`.
