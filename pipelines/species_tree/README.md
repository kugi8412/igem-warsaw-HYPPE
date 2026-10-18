# Species tree

Builds the phylogeny the strain channel is embedded against, by two
independent methods, from public proteomes and from in-house ones.

## Which tree is which

The repository holds three trees and they are not interchangeable.

| file | built by | branch lengths | used for |
|---|---|---|---|
| `data/trees/supermatrix.treefile` | `build_tree.sh` | substitutions per site | a distance prior |
| `data/trees/supertree.tre` | `build_supertree.sh` | coalescent units, none on tips | topology only |
| `data/trees/external/inhouse_nj_20genomes_aasubst.tre` | an earlier in-house run | substitutions per site | the delivered model |

The delivered model is configured with `conditioning.genome.source: tree20`
and reads the third file. To rebuild an equivalent of it here you need the
in-house proteomes, because nine of the twenty strains (`P1`, `B10`, `F7`,
`I1237`, `MMS1295`, `N1508`, `P3041`, `P3080`, `P3116`) were sequenced by us
and have no NCBI accession. Without them the pipeline builds the public tree,
which is useful for comparison but cannot place those strains.

That gap is the reason `tree20` exists. With public genomes alone, thirteen
of the twenty strains had nowhere to sit and collapsed onto a single point,
while their measured expression spanned a factor of 195. `check_tree.py`
reports that collapse directly.

## Running it

```bash
conda env create -f phylogenetic_pipeline.yaml
conda activate phylogenetic_pipeline
```

For the public tree:

```bash
bash build_tree.sh                 # supermatrix, the one with distances
bash build_supertree.sh            # ASTRAL, reuses the alignments
```

To include the in-house strains, put their proteomes in `proteomes_local/`
as `.faa` files, name them in `strain_names.tsv`, then run the same commands:

```bash
mkdir -p proteomes_local
cp /path/to/P1_proteins.faa proteomes_local/
# add a line to strain_names.tsv:   P1_proteins<TAB>P1
bash build_tree.sh
python check_tree.py ../../data/trees/supermatrix.treefile --registry
```

The last command is the one that says whether the result can replace the
delivered tree. It passes only when the tips are exactly the twenty strains
in `config/default.yaml`.

`THREADS` and `SPECIES_TREE_RUN` come from the environment, defaulting to 6
and `run1`. Steps skip work already done, so an interrupted run resumes. To
root the tree on the outgroup:

```bash
OUTGROUP=GCF_000149955.1_protein bash build_tree.sh
```

Rooting is for reading the tree. Patristic distances, which is all the prior
uses, do not change with it.

## The steps

1. `datasets` downloads the accessions in `accession_codes.txt`. Proteomes
   from `proteomes_local/` are copied in beside them.
2. **OrthoFinder** finds single copy orthologues, 2596 of them on the public
   set alone.
3. **MAFFT** aligns each one.
4. **trimAl** (`-automated1`) drops poorly aligned columns.
5. `clean_headers.py` replaces every protein accession with the name of the
   proteome it came from. Until this runs, each protein counts as its own
   species.
6. **AMAS** concatenates into a partitioned supermatrix, then **IQ-TREE**
   fits it with model selection and 1000 ultrafast bootstraps. The supertree
   path instead runs IQ-TREE per gene and **ASTRAL-III** over the results.
7. `label_tree.py` renames the tips to strain names.
8. `check_tree.py` reports whether the result can serve as a prior.

## Naming the tips

`strain_names.tsv` maps a proteome to a strain, by accession or by file name.
Public genomes are named automatically from NCBI's assembly report, so they
only need an entry when that name is wrong. In-house genomes can only be
named here. A tip that no source names stops the run, since it would
otherwise enter the tree as a species of its own.

## Input quality

`busco_batch_summary.txt` is the BUSCO report over the ten public proteomes
(`hypocreales_odb10`, 4494 markers). The column that matters is `Single`.
Nine sit at 85 to 99 per cent. The outgroup, *Fusarium oxysporum* f. sp.
*lycopersici* 4287 (`GCF_000149955.1`), is at 67.9 per cent with 27.5 per
cent duplicated, which makes the single copy filter stricter than it would
otherwise be. Fewer orthologues survive, but the ones that do are cleaner,
and 2596 is comfortable.

Run BUSCO on any in-house proteome before adding it. A proteome with a low
`Single` score tightens the filter for every genome in the run.

## Checking a tree without building one

`check_tree.py` works on any Newick file:

```bash
python check_tree.py ../../data/trees/external/inhouse_nj_20genomes_aasubst.tre --registry
```

On the delivered tree it reports 20 tips matching the registry, 3 tips that
no distance separates (`I1237`, `P3041`, `P3116`, which 439 protein markers
do not resolve) and 8 of 190 pairs below the prior's floor. Those are the
numbers quoted in the model documentation, recomputed from the file rather
than copied.

It parses Newick with `hyppe.analysis.phylogeny` and reads the floor from
`hyppe.model.species.PRIOR_MIN_DIST`, so it cannot drift away from what the
model does.
