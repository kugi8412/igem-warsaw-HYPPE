# Condition metadata: what the model consumes, and how to produce it

Every expression value in the corpus belongs to a cell
`(orthogroup, genome, condition)`. The orthogroup and the genome come from the
upstream pipeline. The **condition** does not: public archives describe an
experiment in free text written by whoever deposited it, in no fixed format,
and frequently not at all. Turning that into something a model can read is
curation, and this page is about how to do it cheaply.

## What the model actually reads: measure this before choosing a method

This matters more than the options below, because it decides which of them is
worth effort.

**The input is free text, not the category.** The experiment description is
embedded once with a frozen sentence model
(`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, mean-pooled, pinned
revision) and cached per cell under the key `<genome>|<category>`. The seven
categories are **not** seven input dimensions. They are the targets of a
multi-label auxiliary head *predicted from* the text vector.

That design is a measurement, not a preference (`src/hyppe/model/condition.py`,
decision D1):

| channel | relative strength |
|---|---|
| free text | **3.6–4.5× the categories**, under both splits |
| categories, on the target strain P1 | **−0.0115**, i.e. negative |
| free text, on P1 | +0.1812 |

Categories can reach at most 24.81 % of the reproducible conditional signal.
The practical consequence for everything below: **a mediocre free-text
description is worth more than a perfectly assigned category.** Effort spent
arguing about category boundaries is largely wasted; effort spent making sure
every sample has *some* faithful sentence is not.

Two further properties worth knowing before you build a pipeline:

* **Granularity is measured.** η² of the text embedding on the 19-level genome
  factor is 0.1348; on the 66-level `(genome, category)` factor it is 0.3317.
  The cell key keeps 2.5× more of the text variance, which is why the cache is
  keyed per cell and not per genome.

  **These two numbers are about the PER-SAMPLE description vectors, before
  aggregation.** The shipped cache holds one vector per *cell*, and on that
  object the same quantities read 0.4134 and **1.0000** — the aggregated
  vector is a deterministic function of its cell, necessarily. Both pairs are
  correct and they describe different objects, which is worth stating in a
  file whose whole subject is the curation step between them;
  [ablations.md](ablations.md) uses the per-cell pair, because that is the
  artefact the model loads.
* **There is no fallback.** A missing `<genome>|<category>` key does *not*
  silently fall back to `<genome>|*`, and it never becomes `text_proj(zeros)`.
  It reaches the model as a learned `unknown` token, flagged by `text_ok`. A
  sample with no usable description is therefore honestly represented as a
  sample with no description, and the model can learn that that is what it is.

## The seven categories

Frozen order; adding one appends, and reordering changes the meaning of every
saved checkpoint.

```
control              untreated reference, standard medium
carbon source        defined or alternative carbon substrate
biotic antagonism    confrontation with another organism or its extracts
light response       light/dark treatment, pulses, circadian series
plant association    growth with or on a plant, root colonisation, exudates
mechanical injury    wounding of the mycelium
other                anything else; say what it is in the description
```

Counts on this corpus: control 974, carbon source 497, other 332, biotic
antagonism 311, light response 160, plant association 62, mechanical injury 62.
Only 66 of 2332 samples (2.8 %) carry more than one category, and those are
kept as multi-hot rather than dropped.

## Six ways to get the metadata, and what each costs

These are the options the team considered. None is good; the question is which
failure is cheapest to live with.

| # | Source | Effort | Main failure |
|---|---|---|---|
| 1 | BioSample `description` field, used verbatim as the condition text | lowest | many records have no such field at all; those that do are inconsistent between submitters and often incomplete |
| 2 | The same, plus a language model that assigns one of the seven categories | low | the category is the weak channel anyway (see above); mostly adds `other` |
| 3 | Experiment **title** as the condition text | lowest | hit or miss: titles often say everything or nothing |
| 4 | Title **concatenated with** `description`, hoping the pair says more than either | low | still inconsistent, but strictly dominates 1 and 3 |
| 5 | **All** metadata fields for the record, handed to a language model | medium | more input, more room for the model to invent; needs a refusal path |
| 6 | The curated spreadsheet produced by the upstream manual pass, handed to a language model | medium | depends on a human step that does not scale |

**Recommendation, given what is measured above.** Option 4 as the text channel,
option 5 as the category assignment, and no effort spent reconciling the two.
Concatenating the title and the description is free and strictly better than
either alone; the category then costs one model call per batch and lands in a
channel that carries at most a quarter of the conditional signal. Options 1 and
3 alone leave too many samples with nothing, and option 6 does not scale past
the organisms a person has already read.

What genuinely matters is not which of these is chosen but that **the choice is
recorded per sample**, so that a later analysis can ask whether the rows whose
description came from a title behave differently from the rows whose
description came from a curated sheet. The pipeline carries a `confidence`
column for exactly that.

Someone put the constraint well in a meeting: *it has to work, not work well.*
That is the right standard here, but only because the measurement above says
this channel is secondary. It would be the wrong standard for the promoter
sequence.

## Rules for a language-model pass

Whichever option is used, the prompt should impose these. They are not style
preferences; each one prevents a failure seen in this corpus.

* **Read a study together, not row by row.** Rows sharing a BioProject come
  from one experiment, and the contrast between its samples is usually visible
  only by comparison. A sample is "control" relative to its own study.
* **Replicates must receive identical text.** Different wording for replicates
  of one condition puts them in different places in embedding space, which is
  pure noise in a channel that is already secondary.
* **One sentence, fixed shape:** `<genotype>, <treatment or growth condition>,
  <time point if any>`, no replicate numbers, no organism name, no instrument
  or library details.
* **Never invent.** If the text does not support a decision, the category is
  `other`, the description says only what is known, and confidence is `low`.
  A fabricated condition is worse than a missing one, because the missing one
  is represented honestly by the unknown token.
* **Forbidden as text sources:** `experiment_title` as a *key*
  (1491 of 1498 values are unique; it is a sample identifier wearing a
  description's clothes), `sample`, `sample_name`, `geo_accessions`,
  `bioproject`, `study`, `label`. The permitted fields are `condition` and
  `study_title`.

## Rebuilding the cache

The embeddings ship as `.npz` under the data root, so training needs neither
`transformers` nor a GPU for this channel. To rebuild after re-curating:

```bash
pip install "hyppe[text]"
export HYPPE_DATA_ROOT=/path/to/hyppe-data
python scripts/build_text_emb.py
```

That writes `data/category_text_emb.npz`, the seven category descriptions,
each a written sentence saying what the condition *is* biologically followed by
the most frequent real `condition` strings recorded for it. A category whose
description is only its own name would teach the model a one-hot with extra
steps, which is the thing the text channel exists to avoid.

The per-cell experiment cache (`experiment_text_emb_*.npz`) is keyed
`<genome>|<category>`. Changing the curation changes the keys, which changes
the model: re-embed and retrain together, and record which curation a
checkpoint was trained against.
