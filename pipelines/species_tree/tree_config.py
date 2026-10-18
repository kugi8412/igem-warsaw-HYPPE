#! /usr/bin/env python3

"""Paths for one run of the species tree pipeline.

Proteomes come from two places. Public genomes are downloaded from NCBI by
accession. In-house genomes, which have no accession, are read from a local
directory. Both end up in the same run directory and are treated alike from
OrthoFinder onwards.
"""

from __future__ import annotations

import os

RUN_DIR = os.environ.get("SPECIES_TREE_RUN", "run1")
ACCESSION_FILE = os.environ.get("SPECIES_TREE_ACCESSIONS",
                                "accession_codes.txt")
LOCAL_PROTEOMES = os.environ.get("SPECIES_TREE_LOCAL", "proteomes_local")
STRAIN_NAMES = os.environ.get("SPECIES_TREE_NAMES", "strain_names.tsv")
PROTEOMES_DIR = os.path.join(RUN_DIR, "proteomes")
DATASET_DIR = os.path.join(RUN_DIR, "ncbi_dataset")
MSA_DIR = os.path.join(RUN_DIR, "MSA")
ALN_DIRECTORY = os.path.join(RUN_DIR, "trimmed_alignments")
CLEAN_OUTPUT_DIRECTORY = os.path.join(RUN_DIR, "clean_trimmed_alignments")
SPECIES_TREE_DIR = os.path.join(RUN_DIR, "species_tree")

# NCBI's description of what it sent, used to name the tips.
ASSEMBLY_REPORT = os.path.join(DATASET_DIR, "data", "assembly_data_report.jsonl")


def proteome_files(proteomes_dir: str = PROTEOMES_DIR) -> list:
    """Every .faa in the run, sorted so that reruns agree."""
    if not os.path.isdir(proteomes_dir):
        raise SystemExit(
            "no proteome directory at %r. Run the download step first, or "
            "set SPECIES_TREE_RUN to the run you meant." % proteomes_dir)
    out = sorted(os.path.join(proteomes_dir, f)
                 for f in os.listdir(proteomes_dir) if f.endswith(".faa"))
    if not out:
        raise SystemExit("no .faa files in %r" % proteomes_dir)
    return out


def accessions(path: str = ACCESSION_FILE) -> list:
    """The public accessions requested, in file order."""
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def local_proteomes(local_dir: str = LOCAL_PROTEOMES) -> list:
    """In-house proteomes, or an empty list when none are provided."""
    if not os.path.isdir(local_dir):
        return []
    return sorted(os.path.join(local_dir, f)
                  for f in os.listdir(local_dir) if f.endswith(".faa"))


def strain_names(path: str = STRAIN_NAMES) -> dict:
    """{accession or file name: registry strain}, from a two column TSV."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.split("#")[0].strip()
            if not line or line.lower().startswith("source\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise SystemExit("%s line %d: expected two tab separated "
                                 "columns, got %r" % (path, n, line))
            out[parts[0].strip()] = parts[1].strip()
    return out


def taxon_label(faa_path: str) -> str:
    """The name a proteome contributes to the alignments.

    Tips carry this until label_tree.py replaces it with a strain name, so
    it only has to be unique and stable.
    """
    return os.path.splitext(os.path.basename(faa_path))[0]
