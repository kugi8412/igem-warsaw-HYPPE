#!/usr/bin/env bash
# The acceptance suite on a synthetic corpus: no data root, no download, no
# GPU. It builds the real model (252 frozen MEME profiles, the five-level cone
# ladder, all four heads), trains it for 120 steps and then runs T1-T5.
#
# Runtime ~8 minutes on 8 CPU cores. Expected last line: "SMOKE PASSED".


set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/train_foundational.py \
    --smoke \
    --synthetic 512 \
    --max-steps 120 \
    --batch-size 8 \
    --device cpu \
    --out-dir _work/smoke
