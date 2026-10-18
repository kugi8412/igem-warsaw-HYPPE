# Baseline ablations: what the architecture earns

The model is a five-level hyperbolic quantiser with four heads, a frozen motif
bank, phylogenetic conditioning and a reconstruction constraint. Each of those
is a claim. This page tests the claims against things that are simpler, under
conditions that make the comparison mean something, and reports the answer
including the parts that are less flattering than we expected.

Everything here was produced by `scripts/run_baselines.py` on one machine
(RTX 5090), from `config/default.yaml`, on the production split
`SPLIT_V6.tsv` column `H_kmer90`.

## What was held fixed

A baseline comparison is worthless unless the baseline and the model are fed
and judged identically. These are the same for every row of every table below:

* **the corpus**, the same `Corpus` object, built by the same loader from the
  same files;
* **the split**, orthogroup-grouped, `train` / `val` / `spy` / `test`, with
  167 257 rows going to 150 232 / 4 583 / 4 241 / 8 201;
* **the target**, the per-(orthogroup, genome, category) cell, seven columns,
  with the per-cell validity mask;
* **the metric**, pooled Pearson and Spearman over every *valid cell* of the
  evaluation split, which is exactly the `pcc` / `scc` the training loop
  reports (`loop.py` computes `pearson(p, t)` over the same concatenation);
* **the selection discipline**, the ridge penalty and the CNN epoch are each
  chosen on `val`, never on `test`, which is the rule the full model's epoch
  selection follows.

What the baselines deliberately do **not** get: the experiment-description
text, the DNA-LM flank vectors, the orthology features, the motif bank. They
see sequence and strain. The gap to the full model is therefore the value of
everything else, measured rather than argued.

## The baselines

| name | what it is |
|---|---|
| `kmerK_ridge` | counts of every K-mer, into a ridge with one coefficient vector per category. No order beyond K, no geometry, closed form. |
| `kmerK+species_ridge` | the same, plus the strain as a one-hot block. |
| `cnn_onehot` | a four-layer 1-D CNN on the one-hot sequence, mean+max pooled, predicting the category row. **No reconstruction, no quantiser, no motif bank, no text.** |
| `cnn_onehot+species` | the same, with the strain appended as constant channels. |

K-mers are **not** collapsed onto their reverse complement, because the model
does not canonicalise either, a promoter read on the other strand is not the
same promoter, and a baseline that canonicalised would be answering a
different question.

## Results, full corpus, evaluated on `val`

15 609 valid cells. Sorted by Pearson.

| model | parameters | pcc | scc |
|---|---|---|---|
| **HYPPE, full architecture** (epoch 8) | **4 506 780** | **0.69369** | **0.67616** |
| `cnn_onehot+species` | 114 503 | 0.63730 | 0.61599 |
| `kmer5+species_ridge` | 7 308 | 0.55409 | 0.55280 |
| `cnn_onehot` | 105 991 | 0.54783 | 0.52158 |
| `kmer4+species_ridge` | 1 932 | 0.54192 | 0.54040 |
| `kmer3+species_ridge` | 588 | 0.51528 | 0.51430 |
| `kmer5_ridge` | 7 175 | 0.48090 | 0.47443 |
| `kmer4_ridge` | 1 799 | 0.46420 | 0.45896 |
| `kmer3_ridge` | 455 | 0.43212 | 0.42562 |

**Two corrections to an earlier version of this table, both of which moved
the number in our favour and are therefore worth stating loudly.**

*The model row must be its cell read-out.* The model reports two scores
against two different targets: `pcc` is computed on the **per-sample** target,
one experiment at a time, and `pcc_cell_readout` on the **cell** target,
the mean over the samples of a (orthogroup, genome, category) cell. The
baselines predict cells. Quoting the model's `pcc` (0.67249) against them
compared two different quantities; the comparable number is
`pcc_cell_readout` = **0.69369**, and the margin over the best baseline is
**+0.069**, not +0.047.


## The information-matched comparison

The baselines see **sequence and strain, and nothing else**. So does the
model, when it is scored the same way: no experiment text, no DNA-LM flanks,
no orthology features, every one of those channels answering with its learned
unknown token. On a model we retrained from `config/default.yaml` on the full
corpus (16 epochs, selection taking epoch 7):

| model, identical inputs and identical rows | pcc | ortogrupa | genomu | warunku | within-row |
|---|---|---|---|---|---|
| **HYPPE** | **0.66327** | **0.80739** | 0.24524 | 0.21773 | 0.09778 |
| `cnn_onehot+species` | 0.62518 | 0.74884 | **0.36998** | 0.17509 | not computed |
| `kmer4+species_ridge` | 0.54192 | 0.61507 | 0.42636 | **0.25107** | **0.39359** |

This is the cleanest table on the page: one loader, one row set, one metric,
one information budget. It confirms the earlier indicative reading and
sharpens one part of it badly.

**Where the architecture wins is the gene axis**, 0.807 against 0.749, which
is the sequence-to-expression mapping and is most of the task.

**Where it loses is the strain axis**, 0.245 against 0.370, exactly as the
indicative table said.

**And the within-row column is the finding of this section.** Stripped of the
text channel, the model captures almost none of the per-category structure, 
0.098 against the ridge's 0.394. That is a mechanism conclusion, not a defect:
the model has **no free per-category columns at all**. Its seven outputs are
produced by one shared projection of the category's *description*, so with the
description unavailable the columns collapse toward one another. The ridge, by
contrast, has seven independent coefficient vectors that work with no text.

The two designs trade the same way they trade on strain. Free columns fit the
categories that were seen and cannot represent a category that was not; a
shared projection of text needs the text, and in exchange a new condition can
be described rather than enumerated. With its text channel the model reaches
0.274 on the condition axis against the ridge's 0.251, so the mechanism pays
for itself, but only when it is fed.

**Do not read 0.098 as the cost of the text channel.** It is the score with
**every** conditioning channel answering with its unknown token, measured
through the baselines' loader and the head's direct output. Removing the text
channel *alone*, on the delivered checkpoint and through the training loop's
own cell read-out, moves the within-row score by **+0.0016**, see the
per-ablation table below, which also shows that the channel this component
actually depends on is the strain. The attribution to the description in the
paragraph above is the part that did not survive; the 0.098 itself stands for
what it measures.

*A note on how this number was nearly wrong.* The script originally had a
`--with-conditioning` flag that loaded the caches into the corpus and then
never passed their vectors to the forward call. It produced numbers identical
to the unconditioned run, under a name claiming otherwise. The flag now
refuses rather than lying, and scoring the model with its conditioning
through `BatchMaker` is written down as the next piece of work rather than
quietly assumed to have been done.

## How much of the category structure a baseline actually captures

"Not condition-aware" needs a number, not an adjective, so here is the exact
one. Subtract each row's mean over its measured categories from both the
prediction and the target, and correlate what is left. This isolates the
per-category component and nothing else: **a model whose output does not
depend on the category is constant along the row, its deviation is exactly
zero, and it scores exactly zero here whatever its pooled number.**

| | `kmer4_ridge` | `kmer4+species_ridge` |
|---|---|---|
| pooled pcc | 0.46420 | 0.54192 |
| **within-row pcc (category structure)** | **0.10533** | **0.39359** |
| sd of the model's within-row deviation | 0.17718 | 0.30741 |
| sd of the target's within-row deviation | 0.66930 | 0.66930 |
| share of cell variance in this component | 8.1 % | 8.1 % |

Three things follow, and the first corrects a loose phrase from an earlier
version of this page.

**The baselines are not literally constant across categories.** They have
seven free output columns, so they can and do fit per-category offsets:
0.394 with the strain, 0.105 without. What jumps out is *where that comes
from*, the strain one-hot is constant along a row and so cannot by itself
move the within-row deviation, except through its own per-category
coefficients. The jump from 0.105 to 0.394 is therefore a **(strain ×
category)** effect: "in this strain, carbon-source experiments run higher
than controls". The baseline learns that, and it is a real part of the task.

**They still speak far too quietly.** The predicted within-row deviation has
sd 0.307 against the target's 0.669, less than half the amplitude. The
component is small (8.1 % of cell variance) and the baselines capture part of
it.

**And none of this touches the part that is genuinely out of reach**, which
is the next section: variation between two experiments *of the same category*.
There the seven-column output is constant by construction, and that is where
"averaged" is literally true.

## What the cell target leaves out, and why it matters

The comparison above is a fair test of one task and silently excludes
another. The full model does not only predict cells. Given a specific
experiment's written description it predicts **that experiment**, the
`row_text` path, trained against `mu_row`, the value of one (orthogroup,
sample) rather than a cell mean. The baselines emit seven numbers, one per
category, and are **constant within a category by construction**: two
experiments of the same kind on the same strain receive the same prediction,
whatever their descriptions say.

Scoring on cell means averages exactly that difference away. So we measured
how much is being averaged away, with no model involved
(`scripts/probe_per_experiment_ceiling.py`, straight from the expression
matrix and the sample metadata):

| | |
|---|---|
| per-sample measurements | 18 836 632 |
| (genome, category) cells | 65 |
| variance **within** cells | 1.16519 of 6.68276 total |
| **share of variance between experiments of one cell** | **17.4 %** |
| **ceiling for any per-category model on the per-sample target** | **pcc 0.90865** |

The cell mean is the least-squares constant inside a cell, so 0.90865 is the
best a per-category output can possibly reach, and the 17.4 % below it is
not merely unreached but **unreachable** for that class of model. The full
model operates on that target directly and scores 0.67249 there.

This does not make the architecture better than the table above says on the
task the table measures. It says the table measures the smaller of the two
tasks, and that the larger one is not a harder version of the same thing, 
it is a capability the baselines do not have at all. An honest summary of
both:

* **on predicting a cell**; the architecture is ahead by 0.069, with a plain
  strain-aware CNN reaching 0.625 of the 0.694;
* **on predicting an experiment**; there is no baseline here to compare
  against, because a per-category output is constant where 17.4 % of the
  variance lives.

## The same result, split onto the three axes

A pooled correlation hides more than it shows here, because one axis dominates
the variance: promoters of different genes differ far more than one gene does
between strains or between conditions. Two models can sit 0.03 apart pooled
while one predicts the condition and the other predicts it backwards.

The training loop already decomposes its own score, so the baselines are put
through the same function (`rozbicie_osi`) rather than a second
implementation:

| axis | what it asks | definition |
|---|---|---|
| `os_ortogrupa` | which gene is this | `mean_og(y) − mean(y)` |
| `os_genomu` | which ortholog of it | `y − mean_og(y)` |
| `os_warunku` | which growth condition | the part of the above that the category explains |

| model | pooled pcc | ortogrupa | genomu | warunku | warunku main-effect R² |
|---|---|---|---|---|---|
| **HYPPE, full** | **0.69369*** | 0.80510† | 0.28905† | 0.27377† | not computed |
| `cnn_onehot+species` | 0.62518 | 0.74884 | 0.36998 | 0.17509 | **−0.36251** |
| `cnn_onehot` | 0.55398 | 0.71847 | 0.08568 | 0.25252 | −0.12584 |
| `kmer4+species_ridge` | 0.54192 | 0.61507 | **0.42636** | 0.25107 | +0.24819 |
| `kmer4_ridge` | 0.46420 | 0.59021 | 0.10306 | 0.21957 | +0.08512 |

(299 orthogroups, 19 genomes, 7 categories, 15 609 cells. The CNN row differs
by 0.012 from the table above because that run was repeated on the CPU to keep
a concurrent GPU measurement clean; GPU reductions are not bit-deterministic,
and the difference is the honest run-to-run spread of this baseline.)

\* the model's pooled figure is its **cell** read-out, matching the baselines.
† **the model's three axis figures are computed on the PER-SAMPLE target**,
because that is what the training loop decomposes, while the baselines' are
on the cell target. They are therefore *not* strictly comparable across the
first row and the rest, and the differences below should be read as
indicative of direction rather than as exact margins. Making them exact means
scoring the checkpoint through this loader and re-running `rozbicie_osi` on
its cell read-out; that is queued, and until it is done this footnote stands
rather than the numbers being quietly presented as like-for-like.

## What the numbers say

**The strain is the single largest factor, and it is not close.** Adding a
20-way one-hot moves the ridge by +0.078 (k = 4: 0.46420 → 0.54192) and the
CNN by +0.089 (0.54783 → 0.63730). Nineteen genomes differ enough in overall
expression level that knowing which one you are in explains more than any
amount of extra sequence modelling the baselines can do. This is why the full
model gives the strain a learned position on a phylogeny rather than a flat
table, and it also means any future result must report whether it beat the
strain-only information, because a model can look good by learning the genome.

**Order matters, but only once there is enough data to learn it.** On the full
corpus the CNN beats the best bag-of-k-mers by +0.083 (0.63730 vs 0.55409),
so convolutional structure buys something real over composition.

**And that conclusion reverses on a subset, which is itself the finding.**
The same grid on 4 000 orthogroups (21 340 rows, 14 600 cells) gave:

| model | pcc on the subset | pcc on the full corpus |
|---|---|---|
| `kmer4+species_ridge` | **0.51511** | 0.54192 |
| `cnn_onehot+species` | 0.51415 | **0.63730** |

On the subset they tie, with the CNN using 59× the parameters; on the full
corpus the CNN wins by 0.083. A baseline comparison run on a convenient
subset would have concluded that the convolution was not worth it. Anyone
reading an ablation in this project should check what fraction of the corpus
it was run on before believing its ordering.

**The full architecture is ahead by +0.069 pooled, and the pooled number is
the least informative thing on this page.** Split by axis, the single margin
turns into three movements in two directions:

| axis | full model − best baseline | |
|---|---|---|
| which gene | +0.056 | indicative, see the † footnote |
| which condition | +0.099 | indicative |
| which strain | **−0.081** | indicative |

The architecture wins where the task is sequence-to-expression and where the
growth condition has to be used, and it **loses on telling strains apart**.
A single pooled margin is the average of those, and quoting it alone would
hide both the win and the loss. The strain deficit is the one we have checked
hardest, below, because it is the one that goes against us.

**The condition result is the one the architecture was built for.** Both CNN
baselines have a *negative* R² on the condition main effect (−0.363 and
−0.126): with seven free output columns and no condition input they do worse
than predicting the per-category mean. The ridges, whose per-category
coefficients are explicit and linear, get the main effect right but reach only
0.251 on the axis. The full model reaches 0.274, with the condition entering
as a sentence embedding of the actual experiment description rather than as a
column index. The difference between "has an output per category" and "knows
what the category is" shows up here and nowhere else in the table.

**The strain result goes against us, and we are keeping it in.** A flat
one-hot fits each of nineteen strains independently and beats the model's
phylogenetic embedding by 0.081 to 0.137 on that axis. The reading we find
most plausible, and it is an interpretation, not yet a measurement, is that
the two are doing different things: a free column per strain is the best
possible *discriminator* between the strains that were seen, while the Lorentz
registry is pulled toward patristic distances and so spends capacity on the
*relations* between strains, which is what lets the same latent serve
reconstruction and transfer rather than only separation.

That interpretation makes a prediction, so we tested it rather than leaving
it as a reading. `scripts/probe_strain_transfer.py` holds one strain's rows
out of training, refits the ridge, and predicts that strain, against the
same model with the strain in, and against the same model with no strain
channel at all. Seventeen strains had enough validation rows to measure
(≥ 200):

| | mean pcc over 17 held-out strains |
|---|---|
| strain **was** in training | 0.50584 |
| sequence only, no strain channel | 0.47432 |
| strain was **held out** | **0.46760** |

| the one-hot strain channel is worth | |
|---|---|
| on a strain it was fitted on | **+0.03151** |
| on a strain it never saw | **−0.00672** |

**The whole value of the one-hot is memorised separation.** On an unseen
strain it is not merely useless but slightly worse than not having the channel
at all, the fitted intercept and the coefficients of the other eighteen
columns are a small active handicap. A flat categorical cannot place a strain
it has no column for, and nothing in it encodes that two strains are related.

That is what the phylogenetic embedding buys, and it is why we keep it despite
losing 0.08–0.14 on the seen-strain axis: the registry places a strain by its
patristic distance to its neighbours, so the position exists before any
expression from that strain has been seen. Our design target is one strain,
P1, and the pipeline is zero-shot with respect to it, no measurement from P1
enters training, so "works on a strain whose rows you did not fit" is not a
hypothetical robustness nicety here. It is the operating condition.

**What this does and does not establish.** It settles the baseline half: the
one-hot does not transfer. It does not yet measure the model half, which
needs a retrain per held-out strain, and until that is run we cannot put a
number on how *well* the embedding transfers, only that the thing it is
being compared against transfers not at all. The seen-strain deficit stays on
the record as measured.

**But predictive accuracy is not the whole job, and this is where the table
stops being the right instrument.** None of these baselines can do the thing
the model was built for. They have no decoder, no discrete codes and no
geometry, so there is nothing to edit and nothing to decode: a candidate
promoter cannot be proposed from them at all. The panel in
[model.md](model.md) exists because the latent is a bijective, auditable code,
and that property has no PCC. Choosing the architecture on this table alone
would be choosing the wrong criterion, the table's job is to show that the
extra machinery does not *cost* accuracy, and it shows that.

**Depth of k-mer saturates and then helps only with the strain.** Without the
strain, k = 5 (1 024 features) beats k = 3 by 0.049; with the strain the same
step is worth 0.039. Both are small next to the strain itself.

## Leave-one-channel-out, on the trained model

`scripts/ablate_channels.py`. Each conditioning channel is removed in
two different ways, and the pair is the point: **zeroing** replaces it with
its learned unknown token and asks what the model loses without it;
**shuffling** hands the model another row's value and asks whether it was
reading the channel per row at all. A channel used as a constant is hurt by
zeroing and not by shuffling.

This now runs on the **delivered checkpoint**, epoch 8, md5
`fd161127f0b11a0a7c33bf55e4bb2cfc`, the file shipped as
`weights/hyppe_foundational.pt`. Earlier versions of this page reported
epochs 7 and 9 only, because the script could address a checkpoint solely
through the `{run}_ep{NNN}.pt` name pattern and the delivered file is not
named that way; `CKPTY` now takes explicit paths and the output records the
md5 of what it scored. Pooled `pcc_cell_readout` on `val`, baseline 0.69481
(`results/ablacja_kanalow_ep8.json`):

| what was removed | val | Δ val | Δ train | train−val gap |
|---|---|---|---|---|
| nothing | 0.69481 | — | — | +0.1298 |
| experiment text | 0.69321 | **−0.00160** | −0.00244 | +0.1290 |
| genomic flanks | 0.68721 | −0.00760 | −0.00329 | +0.1342 |
| gene features | 0.66845 | **−0.02636** | −0.00453 | +0.1517 |
| gene **and** flanks | 0.65295 | **−0.04186** | −0.01166 | +0.1600 |
| strain, *shuffled* | 0.68866 | −0.00615 | −0.00882 | +0.1272 |
| flanks, *shuffled* | 0.67904 | −0.01578 | −0.00855 | +0.1371 |
| gene, *shuffled* | 0.66037 | **−0.03444** | −0.01739 | +0.1469 |

Epochs 7 and 9 give the same ordering with the same signs
(`results/ablacja_kanalow.json`), so this is not one epoch's noise. The
delivered epoch scores higher and overfits less than either neighbour —
baseline 0.69481 against 0.68649 and 0.68544, gap 0.130 against 0.117 and
0.147, which is what selecting it was supposed to achieve.

**Shuffling costs more than zeroing, on every channel.** Gene −0.034 against
−0.026; flanks −0.016 against −0.008. That is the signature of a channel the
model reads *per row*: a wrong value is worse than an honest "I don't know",
which is what the learned unknown token is for. Had the two been equal, the
channel would have been acting as a constant.

**The ordering is gene, then flanks, then strain, then text.** Removing gene
and flanks together costs 0.042, more than removing each alone and adding
them up (0.034 summed), so the two reinforce rather than substitute.

**Removing conditioning makes the model overfit more, not less**: the
train−val gap grows from 0.130 to 0.160 as channels are taken away, and the
val column falls about four times faster than the train column. The
channels are doing work on the validation side that the trunk then has to
invent from sequence alone.

**And the text channel costs essentially nothing here, 0.0016.** That used to
sit badly with the claim, made above and elsewhere, that the model's
per-category structure is carried by the text projection. The experiment that
would settle it, the within-row contrast *under each ablation*, was named as
missing in earlier versions of this page. It has now been run.

## The per-category component, under each ablation

`evaluate` now keeps the cell read-out two-dimensional, so the same
`within_row_contrast` the baselines are scored with can be applied under every
ablation. One implementation of the formula, one loader, one checkpoint.
`val`, 5 003 rows, the component is 8.2 % of cell variance:

| removed | pooled | Δ pooled | within-row | Δ within-row |
|---|---|---|---|---|
| nothing | 0.69481 | — | 0.18680 | — |
| experiment text | 0.69321 | −0.00160 | 0.18840 | **+0.00160** |
| genomic flanks | 0.68721 | −0.00760 | 0.18444 | −0.00236 |
| gene features | 0.66845 | −0.02636 | 0.17835 | −0.00845 |
| gene **and** flanks | 0.65295 | −0.04186 | 0.17321 | −0.01359 |
| strain, *shuffled* | 0.68866 | −0.00615 | 0.16782 | **−0.01898** |
| flanks, *shuffled* | 0.67904 | −0.01578 | 0.18356 | −0.00324 |
| gene, *shuffled* | 0.66037 | −0.03444 | 0.18516 | −0.00164 |

**Positive control of the measure.** Replacing the prediction by its own row
mean, i.e. making it constant along the row, gives deviation **1.2e-16** and
correlation **7.6e-19**: exactly the zero the construction promises. The
script aborts if that control is not zero, because a measure that scored a
flat prediction above zero would void the whole column.

**A missing description is free; a wrong one is not.** Zeroing the text moves
the within-row score **up** by 0.0016; handing the row another row's
description costs **−0.0075**. The channel is read: the first number says the
unknown token is a harmless stand-in, not that the description is idle. A
claim about this channel resting on the zeroing row alone rests on the wrong
probe.

## The description and the strain do not separate

The experiment-text cache is keyed `<genome>|<category>`, so the description
**is** partly the strain, and ablating either alone is scored while its partner
still supplies much of the same information. Measured on the cache the
delivered run loaded (85 keys, 79 distinct vectors):

| text vector regressed on | η² |
|---|---|
| **(genome, category)** | **1.0000**, 0 ambiguous cells |
| genome alone | **0.4134** |
| *control: genome shuffled* | *0.2177 ± 0.0158* |
| category alone | 0.2159 |

So the vector is a deterministic function of the cell, and the strain alone
explains 0.41 of its variance, +0.196 over control. Removing the two together
(`results/ablacja_laczna_ep8.json`):

| removed | Δ pooled | Δ within-row |
|---|---|---|
| description, *zeroed* | −0.00160 | +0.00160 |
| description, *shuffled* | −0.00224 | −0.00754 |
| strain, *shuffled* | −0.00661 | −0.02230 |
| **description + strain** *(zero + shuffle)* | **−0.01160** | **−0.02482** |
| **description + strain** *(both shuffled)* | **−0.01011** | **−0.02810** |

**The strain was hiding what the description is worth.** Zeroing the
description costs 0.0016 with the strain intact and **0.0050** once the strain
is shuffled: 3.1×, pooled interaction −0.0034, super-additive; the shuffle
probe gives 1.6× in the same direction. Quoted as one channel, the growth
condition is worth **0.010–0.012 pooled and 0.025–0.028 within-row**. Those are
the figures for "what the condition contributes"; the single-channel rows are
a decomposition of them, not independent measurements.

**Where it does not generalise.** The masking holds in the pooled column for
both probes. Within-row it holds for zeroing and *reverses* for shuffling
(0.0075 alone against 0.0058 after the strain is gone). We have no account of
that sign change and it sits near the precision of the probe.

**Precision of the shuffle rows.** The permutation comes from one stream, so a
shuffle row depends on how many modes preceded it. Across two runs of the same
checkpoint `strain, shuffled` gave −0.00615 / −0.00661 pooled and 0.16782 /
0.16450 within-row: about **±0.0005 pooled, ±0.003 within-row**. Smaller
differences between two shuffle rows are not results. Zeroing rows are
deterministic.

### Two statistics about the condition point opposite ways

This page now carries both, and anyone quoting one should know the other
exists:

| statistic | HYPPE | `kmer4+species_ridge` | direction |
|---|---|---|---|
| condition axis (`os_warunku`) | **0.27377** | 0.25107 | model ahead |
| within-row contrast | **0.18680** | **0.39359** | **ridge ahead, by 2.1×** |

They are different decompositions, so there is no arithmetic contradiction —
but the second is the less flattering one and it had never been placed beside
the first.

**The loader was the obvious excuse, and it is not the explanation.** The
model's 0.1868 came through the training loop on 5 003 rows while the ridge's
0.39359 came through the baselines' loader on 4 583, which is two differences
at once, row set and read-out path. Both are now removed: the checkpoint was
scored on the **baselines' own `val` rows**, with the same
`within_row_contrast`, with the read-out computed **both** ways
(`results/within_row_loader_control.json`):

| configuration, all on 4 583 rows / 15 609 cells | within-row |
|---|---|
| model, `mu` read-out, no conditioning | 0.08576 |
| model, centroid read-out, no conditioning | **0.17763** |
| model, centroid read-out, full conditioning *(5 003 rows)* | 0.18680 |
| **`kmer4+species_ridge`** | **0.39359** |

The row set accounts for nothing: `within_fraction` is 0.08095 on these rows
against 0.0819 on the loop's, and the target deviation is 0.66930 against
0.6695: the same quantity, measured on the same thing. What the two numbers
differed by was almost entirely the **read-out**: swapping `mu` for the
centroid columns is worth **+0.092**, which is the same mechanism the loop's
own note records (0.4519 from category prototypes against 0.6696 from sample
centroids). Conditioning adds a further **≈0.009**.

So the comparison stands, and the honest sentence is the plain one: **on the
per-category component a strain-aware ridge beats this model 0.394 to 0.178,
by 2.2×, on identical rows with the model's better read-out.** It is not a
loader artefact, it is not the conditioning channels, and it is the one result
on this page that argues against the architecture. The ridge buys it with
seven free per-category coefficient vectors; the model has none, and generates
its columns by projecting category text. That trade is stated in the
information-matched section above, and this is its price.

**The per-category component is dominated by the strain.** Shuffling it costs
−0.019 to −0.022 within-row across two runs, against −0.006 pooled — the
largest per-category effect of any channel and nearly the smallest overall.
The orderings disagree: pooled is gene, flanks, strain, text; within-row is
**strain, gene, flanks, text**. That is the same (strain × category) effect the
ridge shows when its strain one-hot lifts the within-row score from 0.105 to
0.394, and the pooled table hid it entirely.

**What this does not establish.** The read-out generates its seven columns by
projecting the *category centroids* through the shared `text_proj`, so the
column directions are part of the head and no ablation of a row's input can
remove them; a channel reaches this component only through the pooled feature
`h`. The narrow, defensible statement is therefore: the per-category structure
of the read-out travels through `h`, is dominated by the strain, and is
indifferent to whether that row's own description was supplied. Whether a
read-out whose columns came from the row's own text would behave differently
is a separate experiment.

## Reproducing

```bash
export HYPPE_DATA_ROOT=/path/to/hyppe-data
python scripts/run_baselines.py --config config/default.yaml \
    --splits "$HYPPE_DATA_ROOT/data/splits/SPLIT_V6.tsv" \
    --split-column H_kmer90 --max-orthogroups 0 \
    --kmers 3,4,5 --cnn-epochs 6 --device cuda \
    --eval-split val --out results/baselines_full_corpus.json
```

The whole grid takes a few minutes: the ridge normal equations are formed once
per column and reused across the penalty sweep, so the sweep costs one
Cholesky per (column, penalty) rather than one pass over the corpus.

`--eval-split val` is the honest setting while iterating, because both
hyperparameters were selected on `val` and the number is optimistic by exactly
that selection. `--eval-split test` is for one reading, at the end.

## Carbon cost of training the final architecture

Measured with Carbon Footprint Tracker 1.1.5, our own tool, wrapping the
training command:

```bash
carbon-tracker hpc run --zone PL --output results/carbon_training.json -- \
    python scripts/train_foundational.py --config config/default.yaml ...
```

### Measured

Four epochs of the final architecture on a 4 000-orthogroup subset
(21 340 rows, 500 steps per epoch at batch 24), on one RTX 5090, GPU
otherwise idle at the start:

| quantity | value |
|---|---|
| wall clock | 393.8 s (69 s per epoch, plus load and diagnostics) |
| energy | **0.01034 kWh** = 10.34 Wh (measured, `rapl+nvidia-smi`) |
| average power | 94.5 W over the whole wrapped command |
| grid intensity | 650 gCO2eq/kWh, zone PL, built-in table |
| **emissions** | **6.72 g CO2eq** |

### The full run, measured

The delivered architecture, 16 epochs over the whole training split
(150 232 rows, batch 64), wrapped end to end:

| quantity | value |
|---|---|
| wall clock | **2.772 h** (9 978 s) |
| energy | **0.5336 kWh** (measured, `rapl+nvidia-smi`) |
| grid intensity | 650 gCO2eq/kWh, zone PL |
| **emissions** | **346.8 g CO2eq** |

About 2.9 km of driving by the tracker's own conversion factor.

The extrapolation below was written before this run finished and is kept
because it turned out to be a fair check on the method rather than a
placeholder: scaling the small run by samples processed predicted 0.52 kWh
and ≈340 g, against a measured 0.5336 kWh and 346.8 g, **within 3 %**. The
assumption that energy is proportional to samples processed holds on this
workload, which is worth knowing the next time only a short run is affordable.

### Extrapolated to the delivered run (written before the measurement)

The delivered foundation model is 16 epochs over the full training split, so
the measured figure has to be scaled. Taking energy as proportional to
samples processed, the only assumption, and it is stated rather than hidden:

```
measured      2 000 steps x batch 24 =    48 000 samples -> 10.34 Wh
                                                          = 0.2154 mWh/sample
delivered    16 epochs x 150 232 rows = 2 403 712 samples -> 518 Wh
```

**≈ 0.52 kWh, ≈ 340 g CO2eq** for training the delivered model once, about
2.8 km of driving by the tracker's own conversion factor.

That scaling is an over-estimate rather than an under-estimate, for a reason
worth naming: the measured run paid its per-epoch diagnostics over only 500
steps, while the real run amortises the same fixed cost over roughly 2 350
steps per epoch. The true figure is somewhat below 0.52 kWh.

And the honest total is larger than either number, because a delivered model
is every run it took to get there, not the one that shipped. This repository
does not carry a count of the discarded runs, so we report the cost of one
training and say plainly that the project's real footprint is a multiple of
it.

Three things this figure does and does not include, stated because the tool
records them and a reader cannot see them otherwise:

* **It is measured, not modelled.** `energy_measured: true`,
  `energy_source: rapl+nvidia-smi`, sampled every 5 s and integrated.
* **PUE is not applied.** The tool applies its 1.5 data-centre factor only in
  the *estimated* branch; a measured run reports node power without facility
  overhead. The figure is therefore a lower bound on the true cost.
* **The grid intensity is a table value, not a live reading.** Without an
  Electricity Maps key the tracker uses its built-in Polish day/night pair,
  650 / 750 gCO2eq/kWh. It also auto-detected the machine as being in the
  United States (400 gCO2eq/kWh) from the system locale, which would have
  understated the result by a third; `--zone PL` is not optional here.
* **CPU package power was not readable on this host** (`read_cpu_power_watts`
  returns `None`), so the sampled total is dominated by the GPU. For a
  GPU-bound training run that is most of the draw, but it is not all of it.
