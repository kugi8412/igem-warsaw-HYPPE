"""Real LLM embeddings of the seven experiment descriptions.

    python scripts/build_text_emb.py

Keyed by category, not by (orthogroup, genome). Under the old key the text
could take at most 19 distinct values -- exactly what the species token
already carried, leaving H(text | species) = 0. The text now describes seven
experiments, the same seven for every genome.

Each category gets a description grounded in the corpus -- a sentence saying
what the condition is biologically, followed by the most frequent real
`condition` strings recorded for it. A description that is only the category
name would teach the model a one-hot with extra steps.

The encoder is sentence-transformers/all-MiniLM-L6-v2, mean-pooled over tokens
with the attention mask and L2-normalised, run through plain `transformers`;
384 dimensions, matching `conditioning.experiment_text.dim` in the config.

Falsifier: the seven vectors must not be interchangeable. Every pairwise
cosine near 1.0 means the encoder returned seven copies of one vector and the
channel carries nothing, exactly like the hash backend it replaces.
"""
from __future__ import annotations

import collections
import csv
import io
import os
import sys

import numpy as np

# Resolved against the data root rather than hardcoded, so the artefact a
# checkpoint was trained against can be rebuilt on another machine.
_ROOT = os.environ.get("HYPPE_DATA_ROOT", ".")
SAMPLES = os.environ.get("HYPPE_SAMPLES",
                         os.path.join(_ROOT, "data", "samples_metadata.tsv"))
OUT = os.environ.get("HYPPE_TEXT_EMB_OUT",
                     os.path.join(_ROOT, "data", "category_text_emb.npz"))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# What the condition is, in the language a biologist would use: that is what
# an LLM embedding generalises from, so a new experiment type is added by
# writing one of these rather than by retraining.
DESCRIPTIONS = {
    "control": (
        "Baseline growth of the fungus with no applied stress or inducer. "
        "Cells are cultivated on standard complete medium under standard "
        "laboratory conditions and harvested during normal vegetative growth. "
        "This is the reference state against which every other condition is "
        "compared."),
    "carbon source": (
        "The fungus is grown on a defined carbon source that changes which "
        "catabolic pathways are required, such as cellulose, xylan, lactose, "
        "glycerol, sophorose or glucose. These conditions drive carbon "
        "catabolite repression and the induction of cellulase and hemicellulase "
        "genes."),
    "biotic antagonism": (
        "The fungus is confronted with another living organism, typically a "
        "plant-pathogenic fungus, in direct confrontation assay, dual culture, "
        "or contact with its cell wall material. This is the mycoparasitic "
        "programme: recognition of the prey, secretion of hydrolytic enzymes "
        "and of secondary metabolites, and antagonistic growth."),
    "light response": (
        "The culture is exposed to light, or compared between light and "
        "darkness. Photoreception in Trichoderma controls conidiation, "
        "secondary metabolism and circadian output, and is mediated by the "
        "blue-light regulator complex."),
    "plant association": (
        "The fungus interacts with a living plant or with plant roots, in "
        "co-cultivation, rhizosphere colonisation or endophytic association. "
        "This condition covers the beneficial plant-fungus interaction and the "
        "induction of plant systemic resistance."),
    "mechanical injury": (
        "The mycelium is physically damaged, wounded or sheared before "
        "harvest. Injury triggers a regeneration and wound-response programme, "
        "including oxidative burst and re-initiation of hyphal growth."),
    "other": (
        "An experimental condition that does not fall into the defined "
        "categories: chemical or osmotic stress, nutrient limitation, pH "
        "shift, temperature shift, or a genetic perturbation background."),
}


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp_min(1e-9)


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    with io.open(SAMPLES, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print("samples: %d" % len(rows))

    # Ground each description in the corpus: the real `condition` strings most
    # often recorded under that category.
    by_cat = collections.defaultdict(collections.Counter)
    for r in rows:
        cat = (r.get("condition_category_norm")
               or r.get("condition_category") or "").strip()
        cond = (r.get("condition") or "").strip()
        if not cat or not cond:
            continue
        for part in [c.strip() for c in cat.split(",")]:
            if part:
                by_cat[part][cond] += 1

    cats = list(DESCRIPTIONS)
    texts = []
    for c in cats:
        top = [t for t, _ in by_cat.get(c, collections.Counter()).most_common(6)]
        extra = (" Recorded conditions include: " + "; ".join(top) + "."
                 if top else "")
        texts.append(DESCRIPTIONS[c] + extra)
        print("\n[%s]  %d distinct conditions in corpus" % (c, len(by_cat.get(c, ()))))
        print("   %s" % (texts[-1][:190] + ("..." if len(texts[-1]) > 190 else "")))

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=256,
                  return_tensors="pt")
        out = mdl(**enc)
        v = mean_pool(out.last_hidden_state, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=1)
    V = v.numpy().astype("float32")
    print("\nvectors: %s  (dim must be 384 to match the config)" % (V.shape,))

    # Seven near-identical vectors would be the hash backend in a new costume:
    # a channel that trains, reports a share of the context, and carries
    # nothing.
    S = V @ V.T
    off = S[~np.eye(len(cats), dtype=bool)]
    print("\npairwise cosine: min %.4f | mean %.4f | max %.4f"
          % (off.min(), off.mean(), off.max()))
    print("  (max near 1.0 would mean the encoder returned seven copies)")
    print("\n      %s" % " ".join("%6s" % c[:6] for c in cats))
    for i, c in enumerate(cats):
        print("  %-6s %s" % (c[:6], " ".join("%6.3f" % S[i, j]
                                             for j in range(len(cats)))))
    if off.max() > 0.99:
        raise SystemExit("ABORT: two descriptions are indistinguishable")

    np.savez(OUT, categories=np.array(cats, dtype=object),
             vectors=V, texts=np.array(texts, dtype=object),
             backend=np.array(MODEL))
    print("\nwritten %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
