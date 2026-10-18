#!/bin/bash
#
# Species tree by concatenation. Single copy orthologues are aligned,
# trimmed, concatenated into one partitioned supermatrix and fitted with
# IQ-TREE. This produces the tree with real branch lengths, which is the one
# a distance prior can use.
#
# Output: data/trees/supermatrix.treefile

set -euo pipefail
shopt -s nullglob

THREADS="${THREADS:-6}"
RUN_DIR="${SPECIES_TREE_RUN:-run1}"
ACCESSIONS="${SPECIES_TREE_ACCESSIONS:-accession_codes.txt}"
LOCAL="${SPECIES_TREE_LOCAL:-proteomes_local}"
OUTGROUP="${OUTGROUP:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROTEOMES="${RUN_DIR}/proteomes"
DATASET="${RUN_DIR}/ncbi_dataset"
ORTHO="${RUN_DIR}/OrthoFinder"
MSA="${RUN_DIR}/MSA"
TRIMMED="${RUN_DIR}/trimmed_alignments"
CLEAN="${RUN_DIR}/clean_trimmed_alignments"
TREE_DIR="${RUN_DIR}/species_tree"

step() { echo; echo "=== $* ==="; }

missing=()
for c in datasets unzip orthofinder mafft trimal python3 iqtree3 AMAS.py; do
    command -v "$c" >/dev/null 2>&1 || missing+=("$c")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: not on PATH: ${missing[*]}" >&2
    echo "       conda env create -f phylogenetic_pipeline.yaml" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"

if [ ! -d "$DATASET" ]; then
    step "Download public genomes"
    datasets download genome accession --inputfile "$ACCESSIONS" \
        --include genome,protein,gff3 \
        --filename "${RUN_DIR}/trichoderma_complete_data.zip"
    unzip -q "${RUN_DIR}/trichoderma_complete_data.zip" -d "$RUN_DIR"
fi

if [ ! -d "$PROTEOMES" ]; then
    step "Collect proteomes"
    mkdir -p "$PROTEOMES"
    while IFS= read -r -d '' faa; do
        cp "$faa" "${PROTEOMES}/$(basename "$(dirname "$faa")")_$(basename "$faa")"
    done < <(find "$DATASET" -type f -name "*.faa" -print0)
    n_public=$(ls -1 "$PROTEOMES"/*.faa | wc -l)

    n_local=0
    if [ -d "$LOCAL" ]; then
        for faa in "$LOCAL"/*.faa; do
            cp "$faa" "${PROTEOMES}/$(basename "$faa")"
            n_local=$((n_local + 1))
        done
    fi
    echo "${n_public} public, ${n_local} in-house"
    if [ "$n_local" -eq 0 ]; then
        echo "NOTE: no in-house proteomes in ${LOCAL}/, so this builds the"
        echo "      public tree only. The delivered model is conditioned on"
        echo "      the twenty strain tree, which needs them."
    fi
fi

if [ ! -d "$ORTHO" ]; then
    step "OrthoFinder"
    orthofinder -f "${PROTEOMES}/" -t "$THREADS" -o "$ORTHO" \
        > "${RUN_DIR}/orthofinder.log"
fi

RESULTS="$(find "$ORTHO" -maxdepth 1 -type d -name 'Results_*' \
            -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$RESULTS" ] || { echo "ERROR: no OrthoFinder results in $ORTHO" >&2; exit 1; }
SCO="${RESULTS}/Single_Copy_Orthologue_Sequences"
[ -d "$SCO" ] || { echo "ERROR: no single copy orthologues in $RESULTS" >&2; exit 1; }
echo "using $RESULTS ($(ls -1 "$SCO"/*.fa | wc -l) single copy orthologues)"

step "Align with MAFFT"
mkdir -p "$MSA"
for f in "$SCO"/*.fa; do
    out="${MSA}/$(basename "${f%.fa}").aln"
    [ -s "$out" ] || mafft --auto --quiet "$f" > "$out"
done

step "Trim with trimAl"
mkdir -p "$TRIMMED"
for f in "$MSA"/*.aln; do
    out="${TRIMMED}/$(basename "$f" .aln).trim.aln"
    [ -s "$out" ] || trimal -in "$f" -out "$out" -automated1
done

step "Rewrite headers as taxon names"
SPECIES_TREE_RUN="$RUN_DIR" SPECIES_TREE_ACCESSIONS="$ACCESSIONS" \
    SPECIES_TREE_LOCAL="$LOCAL" python3 "${HERE}/clean_headers.py"

step "Concatenate with AMAS"
mkdir -p "$TREE_DIR"
AMAS.py concat -i "$CLEAN"/*.clean.aln -f fasta -d aa -u fasta \
    -t "${TREE_DIR}/concatenated.fasta" -p "${TREE_DIR}/partitions.txt" \
    -c "$THREADS"
# AMAS omits the data type column IQ-TREE needs
sed -i '/[^[:space:]]/s/^/Protein, /' "${TREE_DIR}/partitions.txt"

step "Fit with IQ-TREE"
IQ_ARGS=(-s "${TREE_DIR}/concatenated.fasta" -p "${TREE_DIR}/partitions.txt"
         -B 1000 -T "$THREADS" -m MFP --prefix "${TREE_DIR}/trichoderma")

# Rooting is for reading the tree.
[ -n "$OUTGROUP" ] && IQ_ARGS+=(-o "$OUTGROUP")
iqtree3 "${IQ_ARGS[@]}"

step "Name the tips and check the result"
SPECIES_TREE_RUN="$RUN_DIR" python3 "${HERE}/label_tree.py" \
    "${TREE_DIR}/trichoderma.treefile" \
    --report "${DATASET}/data/assembly_data_report.jsonl" \
    --out "${REPO}/data/trees/supermatrix.treefile"
python3 "${HERE}/check_tree.py" "${REPO}/data/trees/supermatrix.treefile"

echo
echo "DONE: ${REPO}/data/trees/supermatrix.treefile"
echo "If this run included the in-house proteomes, check it against the"
echo "registry as well:"
echo "  python3 check_tree.py ${REPO}/data/trees/supermatrix.treefile --registry"
