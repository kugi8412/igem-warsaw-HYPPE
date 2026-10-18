#!/bin/bash
#
# Species tree by summary coalescent. One maximum likelihood tree per gene,
# then ASTRAL-III over all of them. This is a second opinion on the topology.
# Its branch lengths are in coalescent units and its tips have none, so it
# cannot serve as a distance prior.
#
# Output: data/trees/supertree.tre


set -euo pipefail
shopt -s nullglob

THREADS="${THREADS:-6}"
RUN_DIR="${SPECIES_TREE_RUN:-run1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

CLEAN="${RUN_DIR}/clean_trimmed_alignments"
GENE_TREES="${RUN_DIR}/gene_trees"
TREE_DIR="${RUN_DIR}/species_tree"
GENE_TREE_LIST="${TREE_DIR}/all_genetrees.tre"
ASTRAL_OUT="${TREE_DIR}/trichoderma_astral.tre"

step() { echo; echo "=== $* ==="; }

ASTRAL_BIN=""
for c in astral astral4 ASTRAL; do
    command -v "$c" >/dev/null 2>&1 && { ASTRAL_BIN="$c"; break; }
done
if [ -z "$ASTRAL_BIN" ]; then
    echo "ERROR: no ASTRAL binary found. The conda package is 'aster'." >&2
    exit 1
fi
for c in python3 iqtree3; do
    command -v "$c" >/dev/null 2>&1 || { echo "ERROR: $c not on PATH" >&2; exit 1; }
done

if [ ! -d "$CLEAN" ]; then
    echo "ERROR: ${CLEAN} does not exist. Run build_tree.sh first." >&2
    exit 1
fi

alignments=("$CLEAN"/*.clean.aln)
[ "${#alignments[@]}" -gt 0 ] || { echo "ERROR: no cleaned alignments" >&2; exit 1; }
echo "${#alignments[@]} cleaned alignments"

step "Infer gene trees"
# IQ-TREE writes beside its own input
mkdir -p "$GENE_TREES"
build_one() {
    local aln="$1" dir="$2" threads="$3"
    local base; base="$(basename "$aln" .clean.aln)"
    [ -s "${dir}/${base}.treefile" ] && return 0
    iqtree3 -s "$aln" -m MFP -T "$threads" --prefix "${dir}/${base}" \
        --quiet -redo
}
export -f build_one
if command -v parallel >/dev/null 2>&1; then
    parallel -j "$THREADS" build_one {} "$GENE_TREES" 1 ::: "${alignments[@]}"
else
    echo "GNU parallel not found, running one at a time" >&2
    for aln in "${alignments[@]}"; do
        build_one "$aln" "$GENE_TREES" "$THREADS"
    done
fi

step "Collect gene trees"
mkdir -p "$TREE_DIR"
treefiles=("$GENE_TREES"/*.treefile)
[ "${#treefiles[@]}" -gt 0 ] || { echo "ERROR: no .treefile produced" >&2; exit 1; }
: > "$GENE_TREE_LIST"
for t in "${treefiles[@]}"; do
    # One tree per line
    tr -d '\n' < "$t" >> "$GENE_TREE_LIST"
    printf '\n' >> "$GENE_TREE_LIST"
done
n_trees=$(wc -l < "$GENE_TREE_LIST")
echo "${n_trees} gene trees from ${#alignments[@]} alignments"
if [ "$n_trees" -ne "${#alignments[@]}" ]; then
    echo "WARNING: some alignments produced no tree, so ASTRAL runs on a" >&2
    echo "         different gene sample than the supermatrix did." >&2
fi

step "Run ASTRAL-III"
"$ASTRAL_BIN" -i "$GENE_TREE_LIST" -o "$ASTRAL_OUT"

step "Name the tips"
SPECIES_TREE_RUN="$RUN_DIR" python3 "${HERE}/label_tree.py" "$ASTRAL_OUT" \
    --report "${RUN_DIR}/ncbi_dataset/data/assembly_data_report.jsonl" \
    --out "${REPO}/data/trees/supertree.tre"
# Expected to report that this is topology only
python3 "${HERE}/check_tree.py" "${REPO}/data/trees/supertree.tre" || true

echo
echo "DONE: ${REPO}/data/trees/supertree.tre"
echo "For a distance prior use supermatrix.treefile from build_tree.sh."
