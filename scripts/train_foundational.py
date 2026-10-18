"""CLI entry point for the training loop.

    python scripts/train_foundational.py --help
    python scripts/train_foundational.py --gates-only
    python scripts/train_foundational.py --smoke --synthetic 256 --max-steps 30
    python scripts/train_foundational.py --smoke \
        --sequences .../sequences/sequences.tsv.gz \
        --combat   .../og_scope/combat.tsv.gz \
        --samples  data/samples_metadata.tsv \
        --splits   data/splits/SPLIT_FINAL.tsv \
        --max-orthogroups 400 --max-steps 40 --batch-size 8

Exit codes: 0 all clear, 3 a wiring gate failed (the method is configured but
not wired), 4 an acceptance falsifier failed. A queue reads these; a printed
warning it does not.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# `src`, not the repository root: the package lives under src/ (see
# pyproject.toml). A clone therefore runs without `pip install -e .`.
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from hyppe.training.loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
