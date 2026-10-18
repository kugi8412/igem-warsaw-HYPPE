"""Off-target checks and negative controls for promoter design.

Three jobs, all of which exist because the predictor is also the designer and
will happily optimise against its own blind spots. An edit that raises
transcription can introduce an ATG upstream of the real start codon: mRNA
rises, protein falls, and the model's own objective is satisfied while the
experiment fails. Nothing in a sequence-to-expression model can notice that,
nor what else an edit created, nor whether the model reads arrangement at all
rather than composition.
"""

from __future__ import annotations

import numpy as np

STOPS = {"TAA", "TAG", "TGA"}


def find_uorfs(seq, tss_at_end=True, min_len=0):
    """ATGs upstream of the TSS, with the ORF each one opens.

    `tss_at_end` reflects this dataset: on ML_Dataset the TSS is at the 3' end
    (GC rises 50.5 % -> 52.4 % over the final 160 bp), so the whole sequence
    is upstream and every ATG is a potential upstream start. One record per
    ATG: position, distance to the TSS, whether it hits an in-frame stop first
    (a contained uORF, the strongly inhibitory kind) and the peptide length.
    """
    s = seq.upper()
    n = len(s)
    out = []
    for i in range(n - 2):
        if s[i:i + 3] != "ATG":
            continue
        stop = None
        for j in range(i + 3, n - 2, 3):
            if s[j:j + 3] in STOPS:
                stop = j
                break
        length = ((stop - i) // 3) if stop is not None else ((n - i) // 3)
        if length < min_len:
            continue
        out.append({
            "pos": i,
            "dist_to_tss": (n - i) if tss_at_end else i,
            "terminated": stop is not None,
            "peptide_len": length,
            "contained": stop is not None,
        })
    return out


def uorf_risk(seq, tss_at_end=True):
    """One number: the uORF burden this sequence carries near the TSS.

    Weighted by proximity to the TSS, because a start codon 30 bp upstream is
    far more likely to be used than one 700 bp away behind several others.
    """
    n = len(seq)
    risk = 0.0
    for u in find_uorfs(seq, tss_at_end):
        proximity = 1.0 - (u["dist_to_tss"] / max(n, 1))
        risk += proximity ** 2
    return risk


def dinucleotide_shuffle(seq, rng=None, max_tries=100):
    """A shuffle that preserves dinucleotide composition exactly.

    Altschul-Erikson. An ordinary shuffle preserves only base composition, so
    the model separates it from a real promoter on dinucleotide statistics
    alone and the control proves nothing. This one walks an Eulerian path over
    the dinucleotide graph, so only the arrangement changes.

    The walk is Eulerian by construction rather than by rejection: for every
    vertex but the terminal one, one outgoing edge is reserved as the last,
    and those choices form a tree rooted at the terminal vertex, so the exit
    towards the end survives until a vertex's last use and the walk cannot get
    stuck. The remaining edges are shuffled freely. Drawing a random edge
    order and checking afterwards fails essentially always, because at 800 bp
    the graph has 799 edges and those into the terminal vertex must come last.

    `max_tries` only guards a pathological alphabet; the function then raises
    rather than return its input, because a negative control identical to the
    sample reads as "the model cannot tell them apart".
    """
    rng = rng or np.random.default_rng()
    s = seq.upper()
    if len(s) < 4:
        return s

    koniec = s[-1]
    wych = {}
    for x, y in zip(s, s[1:]):
        wych.setdefault(x, []).append(y)

    wierzchy = [v for v in wych if v != koniec]
    for _ in range(int(max_tries)):
        ostatnia = {v: wych[v][rng.integers(len(wych[v]))] for v in wierzchy}
        drzewo = True
        for v in wierzchy:
            widziane, cur = {v}, ostatnia[v]
            while cur != koniec:
                if cur in widziane or cur not in ostatnia:
                    drzewo = False
                    break
                widziane.add(cur)
                cur = ostatnia[cur]
            if not drzewo:
                break
        if not drzewo:
            continue

        kolejka = {}
        for v, lst in wych.items():
            reszta = list(lst)
            if v in ostatnia:
                reszta.remove(ostatnia[v])
                rng.shuffle(reszta)
                reszta.append(ostatnia[v])
            else:
                rng.shuffle(reszta)

            kolejka[v] = reszta[::-1]

        out = [s[0]]
        cur = s[0]
        for _ in range(len(s) - 1):
            lst = kolejka.get(cur)
            if not lst:
                break
            nxt = lst.pop()
            out.append(nxt)
            cur = nxt
        if len(out) == len(s):
            return "".join(out)

    raise RuntimeError(
        "dinucleotide_shuffle: failed to build a tree of last edges in %d "
        "tries (len=%d, alphabet=%s)" % (max_tries, len(s), sorted(wych)))


def dinucleotide_counts(seq):
    s = seq.upper()
    c = {}
    for a, b in zip(s, s[1:]):
        c[a + b] = c.get(a + b, 0) + 1
    return c


def motif_profile(bank, onehot, top_k=10):
    """Per-motif summary of a sequence: peak occupancy, position, count.

    `bank` is a `hyppe.meme.FrozenMotifBank`, `onehot` is `(1, 4, L)`. The
    `top_k` strongest named motifs, so a design decision reads in
    transcription-factor terms rather than as a vector.
    """
    import torch
    with torch.no_grad():
        occ = bank(onehot)[0, :bank.n_motifs]
    peak = occ.amax(-1)
    order = torch.argsort(peak, descending=True)[:top_k]
    return [{
        "motif": bank.names[int(i)],
        "peak": float(peak[int(i)]),
        "pos": int(occ[int(i)].argmax()),
        "n_strong": int((occ[int(i)] > 0.85).sum()),
    } for i in order]


def offtarget_report(bank, before_onehot, after_onehot, threshold=0.85):
    """Motifs an edit created or destroyed.

    The design objective only asks for higher predicted expression; this asks
    what else changed -- a repressor site gained, or an activator site lost
    somewhere the optimiser did not care about.
    """
    import torch
    with torch.no_grad():
        a = bank(before_onehot)[0, :bank.n_motifs]
        b = bank(after_onehot)[0, :bank.n_motifs]
    before_n = (a > threshold).sum(-1)
    after_n = (b > threshold).sum(-1)
    delta = (after_n - before_n)
    changed = (delta != 0).nonzero(as_tuple=False).flatten()
    return [{
        "motif": bank.names[int(i)],
        "before": int(before_n[int(i)]),
        "after": int(after_n[int(i)]),
        "delta": int(delta[int(i)]),
    } for i in changed]
