"""Measurement work: what the data says before any model is trained.

``paths``              dataset locations; every accessor raises rather than
                       returning a path to a file that is not there.
``matrices``           raw loaders for the expression matrices and metadata.
                       Raw on purpose: ``hyppe.data.labels`` applies the
                       sample policy, and these analyses are statements
                       about the full corpus.
``eda``                cohort, layout, targets, variance, sequences.
``baselines``          what a model must beat, per experimental design.
``phylogeny``          species tree -> Lorentz coordinates, with a flat
                       control at equal budget.
``noise_ceiling``      replicate reliability -> the PCC a perfect model
                       could reach.
``promoter_boundary``  does the `Promotor` window end at the TSS or at ATG.
``panel_power``        how many designs, at what effect size, to call a win.
``spy_power``          the same question for the held-out pks1 block.
``construct_filter``   synthesis / cloning / biology rules a design must pass.
``embeddings``         frozen-LM and k-mer representations of the corpus.
``embedding_probe``    do those representations carry expression signal.
``splits_build``       build train/val/test/spy from leakage blocks.
``splits_audit``       and prove the split is what it claims to be.
``sequence_audit``     whole-corpus sequence statistics, per GROUP, with
                       `.upper()` -- the corrective for the EDA shortcuts.
``seed_variance``      how much of a reported difference is seed noise.
``verification/``      one-shot scripts that checked the claims above.
"""
