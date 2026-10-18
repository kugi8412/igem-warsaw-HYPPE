"""Full diagnostics of the foundation model, before the iGEM panel.

    python scripts/diagnose.py [CKPT] [--szybko]

Defaults to `runs/hyppe_foundational.pt`. Every section carries a falsifier
and its own verdict. A failed blocking section exits non-zero, because a
model that does not pass it cannot go on to generate sequences for the wet
laboratory; observational sections report a number and do not block.

The test split is never opened here -- it was read once. Everything is
computed on the anchors (pks1, tef1 of strain P1) and the P1 promoter corpus.
"""
from __future__ import print_function

import io
import json
import os
import sys
import time

import numpy as np

K = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HYPPE_DATA_ROOT", K)
os.environ.setdefault("HYPPE_ROOT", K)
_ARGV = list(sys.argv[1:])
sys.argv = ["diagnostyka_finalna"]
for _p in (K + "/src/panel", K + "/src"):
    sys.path.insert(0, _p)

SZYBKO = "--szybko" in _ARGV
_POZ = [a for a in _ARGV if not a.startswith("--")]
CKPT = _POZ[0] if _POZ else (K + "/runs/hyppe_foundational.pt")
WY = os.environ.get("WY", K + "/results/diagnostyka.json")
ZIARNO = 42

#: How many P1 corpus promoters the distributional statistics use. The SE of a
#: percentile at n = 256 is ~3 points and the differences resolved are tens.
N_KORPUS = 64 if SZYBKO else 256

_RESULTS = {}
_AWARIE = []


def sekcja(tytul):
    print("")
    print("=" * 78)
    print(tytul)
    print("=" * 78)


def verdict(name, ok, tresc, zaporowa=True):
    znak = "OK  " if ok else ("FAIL" if zaporowa else "WARN")
    print("  [%s] %s: %s" % (znak, name, tresc))
    _RESULTS[name] = {"ok": bool(ok), "tresc": tresc, "zaporowa": bool(zaporowa)}
    if not ok and zaporowa:
        _AWARIE.append(name)
    return ok


def main():
    import anchors as KK
    import model_io
    import panel_spec as ZP
    import scoring
    import surgery as CH
    import torch

    t0 = time.time()
    torch.manual_seed(ZIARNO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(ZIARNO)

    sekcja("1. CHECKPOINT AND PROVENANCE")
    model, prov = model_io.load_model(CKPT, device=dev)
    model.eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    epoka = ck.get("epoch")
    step = ck.get("step")
    cfg = ck.get("config") or {}
    tr = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
    n_par = sum(p.numel() for p in model.parameters())
    print("  file      %s" % os.path.basename(CKPT))
    print("  sha256    %s" % prov.checkpoint_sha256[:16])
    print("  epoch     %s of %s (schedule %s)"
          % (epoka, tr.get("epochs"), tr.get("lr_schedule")))
    print("  step      %s | seed %s" % (step, tr.get("seed")))
    print("  parameters %d" % n_par)
    _RESULTS["_meta"] = {"ckpt": CKPT, "sha256": prov.checkpoint_sha256,
                        "epoka": epoka, "step": step,
                        "epochs_cfg": tr.get("epochs"),
                        "seed": tr.get("seed"), "parametry": n_par}
    # `load_model` is strict, so getting this far is already a passed gate.
    # Recorded anyway: no message is not equivalent to a passed gate.
    verdict("wczytanie_strict", True, "no missing/unexpected keys")

    sekcja("2. MANIFOLD INTEGRITY (the EMA/SWA defect)")
    # The codebook points lie on the hyperboloid, and an arithmetic mean of
    # the ambient coordinates (EMA or SWA) takes them off it. At
    # max|<x,x>_L + 1| = 5.05e-02 the reported reconstruction was 0.9945
    # against 0.8012 read back from the checkpoint.
    from hyppe.model.quantizer import collect_manifold_state_names
    nazwy = sorted(collect_manifold_state_names(model))
    sd = model.state_dict()
    najg, ile = 0.0, 0
    for n in nazwy:
        v = sd.get(n)
        if v is None or not torch.is_floating_point(v) or v.ndim < 1:
            continue
        x = v.reshape(-1, v.shape[-1]).to(torch.float64)
        mink = -(x[:, 0] ** 2) + (x[:, 1:] ** 2).sum(dim=1)
        b = float((mink + 1.0).abs().max())
        najg = max(najg, b)
        ile += 1
    print("  tensors on the manifold: %d" % ile)
    verdict("manifold", najg < 1e-9,
            "max|<x,x>_L + 1| = %.3e (threshold 1e-9)" % najg)

    sekcja("3. THE DIFFUSION CONTRACT")
    anchors = KK.load_anchor_sequences()
    PKS1, TEF1 = anchors.pks1, anchors.tef1
    L = len(PKS1)
    print("  pks1 %d bp | tef1 %d bp" % (L, len(TEF1)))
    with torch.no_grad():
        x = scoring.one_hot([PKS1]).to(dev)
        idx_p = [t.clone() for t in model.encode_codes(x)]
        idx_t = [t.clone() for t in model.encode_codes(
            scoring.one_hot([TEF1]).to(dev))]
    geom = CH.geometry_from_codes(idx_p)
    lad = model.ladder
    kwant = [lad.position_quantizer] + list(lad.tangent_quantizers)
    assert len(kwant) == len(idx_p), \
        "quantisers %d, code levels %d" % (len(kwant), len(idx_p))
    # The alphabet is read from the module, not probed with indices: an
    # out-of-range index gives a CUDA device-side assert, which cannot be
    # caught in Python and poisons the whole process.
    alf = [int(q.codebook_size) for q in kwant]
    print("  levels %d | cells %s | alphabets %s"
          % (len(idx_p), list(geom), alf))

    # (a) prediction from codes == the forward pass, under the same condition.
    # `predict_from_codes(idx)` without `context` skips the fusion entirely
    # (`if self.fusion is not None and context is not None`), so it compares
    # two different routes and gives differences of order 0.7. The condition
    # must come from the same pass that `Scorer` computes with.
    KONSTR = tuple(sorted(ZP.KONSTRUKTY))
    sc = scoring.Scorer(model, prov,
                      experiment_text={c: ZP.description(c) for c in KONSTR},
                      species="P1", constructs=KONSTR, anchors=anchors)
    roz_umowa = {}
    for c in KONSTR:
        cx = sc.contexts[c]
        rep = (lambda t: None if t is None else t.expand(1, *t.shape[1:]))
        with torch.no_grad():
            pf = model(x, text_emb=rep(cx.text_emb), text_ok=rep(cx.text_ok),
                       species_id=torch.full((1,), sc.species_row,
                                             dtype=torch.long, device=dev),
                       flank_l=rep(cx.flank_l), flank_r=rep(cx.flank_r),
                       flank_ok=rep(cx.flank_ok), reconstruct=False)
            pk = model.predict_from_codes([t.clone() for t in idx_p],
                                          context=pf["context"])
        r = 0.0
        for kk in ("pair_score",):
            a, b = pf.get(kk), pk.get(kk)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                r = max(r, float((a - b).abs().max()))
        ab_a, ab_b = pf.get("absolute"), pk.get("absolute")
        if ab_a is not None and ab_b is not None:
            r = max(r, float((ab_a.mu - ab_b.mu).abs().max()))
        roz_umowa[c] = r
    najw = max(roz_umowa.values())
    verdict("umowa_predykcji", najw < 1e-4,
            "max|predict_from_codes - forward| = %s"
            % {k: "%.3e" % v for k, v in roz_umowa.items()})

    # (b) bijection encode(decode(idx)) == idx
    with torch.no_grad():
        dna_p = CH.decode_codes(model, [t.clone() for t in idx_p])[0]
        idx_2 = model.encode_codes(scoring.one_hot([dna_p]).to(dev))
    zgod = []
    for lv in range(len(idx_p)):
        zgod.append(float((idx_2[lv] == idx_p[lv]).float().mean()))
    verdict("bijekcja", min(zgod) > 0.99,
            "encode(decode(idx))==idx per level %s"
            % " ".join("%.4f" % z for z in zgod))

    # (c) reconstruction from the codes alone
    zgodne = sum(1 for a, b in zip(dna_p, PKS1) if a == b)
    rec = zgodne / float(L)
    # Composition floor: the chance of hitting a base by drawing from the
    # sequence's own composition. Without it "0.29 against a chance level of
    # 0.25" sounds like a result and carries no information.
    fr = np.array([PKS1.count(b) for b in "ACGT"], dtype=float) / L
    podloga = float((fr ** 2).sum())
    verdict("rekonstrukcja_z_kodow", rec > 0.99,
            "%.4f (composition floor %.4f)" % (rec, podloga))

    sekcja("4. LEVEL EFFECT ON DECODING (what diffusion really moves)")
    print("  substituting a WHOLE level with tef1 codes and counting "
          "changed bases")
    print("  %-8s %10s %12s %12s" % ("level", "cells", "bases", "fraction"))
    dek_wplyw = {}
    for lv in range(len(idx_p)):
        m2 = [t.clone() for t in idx_p]
        m2[lv] = idx_t[lv].clone()
        with torch.no_grad():
            d2 = CH.decode_codes(model, m2)[0]
        nz = sum(1 for a, b in zip(d2, dna_p) if a != b)
        dek_wplyw[lv] = nz
        print("  L%-7d %10d %12d %12.4f"
              % (lv, geom[lv], nz, nz / float(L)))
    _RESULTS["dekodowanie_per_poziom"] = dek_wplyw
    zywe = [lv for lv, n in dek_wplyw.items() if n > 0]
    verdict("poziomy_dekodujace", len(zywe) >= 1,
            "levels that move the sequence: %s (inert: %s)"
            % (zywe, [lv for lv in dek_wplyw if lv not in zywe]),
            zaporowa=False)

    sekcja("5. LEVEL EFFECT ON PREDICTION -- TWO ROUTES")

    def pairs(seqs):
        o = {}
        for c in KONSTR:
            r = sc.raw(list(seqs), c, batch=len(seqs))
            o[c] = np.asarray(r["pair_score"]).reshape(-1)
        return o

    def pairs_from_codes(ind):
        """The same number, read in code space."""
        o = {}
        for c in KONSTR:
            cx = sc.contexts[c]
            rep = (lambda t: None if t is None else t.expand(1, *t.shape[1:]))
            with torch.no_grad():
                ctx = model(x, text_emb=rep(cx.text_emb),
                            text_ok=rep(cx.text_ok),
                            species_id=torch.full((1,), sc.species_row,
                                                  dtype=torch.long,
                                                  device=dev),
                            flank_l=rep(cx.flank_l), flank_r=rep(cx.flank_r),
                            flank_ok=rep(cx.flank_ok),
                            reconstruct=False)["context"]
                p = model.predict_from_codes([t.clone() for t in ind],
                                             context=ctx)
            o[c] = float(np.asarray(p["pair_score"].detach().cpu()).reshape(-1)[0])
        return o

    baza = pairs([dna_p])
    baza_k = pairs_from_codes(idx_p)
    # The two routes are different measurements and must not be mixed.
    # DECODE substitutes codes, decodes, and scores the sequence: the route
    # the panel and the acceptance gate take. CODES substitutes codes and
    # reads `predict_from_codes`: the route diffusion guidance takes. A level
    # that moves the second but not the first steers the sampler in a
    # direction the gate will never see.
    print("  substituting a WHOLE level with tef1 codes")
    print("  %-8s %14s %14s %12s"
          % ("level", "d pair DECODE", "d pair CODES", "d bases"))
    pred_wplyw, pred_kod = {}, {}
    for lv in range(len(idx_p)):
        m2 = [t.clone() for t in idx_p]
        m2[lv] = idx_t[lv].clone()
        with torch.no_grad():
            d2 = CH.decode_codes(model, m2)[0]
        p2 = pairs([d2])
        k2 = pairs_from_codes(m2)
        dd = float(np.mean([abs(p2[c][0] - baza[c][0]) for c in KONSTR]))
        kk2 = float(np.mean([abs(k2[c] - baza_k[c]) for c in KONSTR]))
        pred_wplyw[lv], pred_kod[lv] = dd, kk2
        print("  L%-7d %14.5f %14.5f %12d" % (lv, dd, kk2, dek_wplyw[lv]))
    _RESULTS["predykcja_per_poziom_dekod"] = pred_wplyw
    _RESULTS["predykcja_per_poziom_kody"] = pred_kod
    # Observational: the disjointness is a known limitation of this
    # architecture, measured on every model rather than assumed absent.
    rozl = [lv for lv in pred_kod
            if pred_kod[lv] > 0.05 and pred_wplyw[lv] < 0.01]
    verdict("drogi_zgodne", not rozl,
            "levels that move the PREDICTION FROM CODES but not the "
            "sequence: %s -- diffusion guidance cannot rest on them" % rozl,
            zaporowa=False)

    sekcja("6. ANCHORS: pks1, tef1 AND MID/MAX FEASIBILITY")
    A = pairs([PKS1, TEF1])
    kp = {c: (float(A[c][0]), float(A[c][1])) for c in KONSTR}
    for c in KONSTR:
        print("  %-10s pks1 %+.5f | tef1 %+.5f | gap %+.5f"
              % (c, kp[c][0], kp[c][1], kp[c][1] - kp[c][0]))
    # MID falsifier: tef1 must sit above pks1 in every construct, otherwise
    # the window "pks1 < X < tef1" is empty and MID has no solution, however
    # good the sampler is.
    okno_ok = all(kp[c][1] > kp[c][0] for c in KONSTR)
    # The numbers go into the results dict rather than being parsed back out
    # of a verdict's `tresc`: a regex over it once read `7-1` out of the
    # construct name `pAN7-1`, left the column nan, and still announced a
    # winner because nan compared with nan raises nothing.
    _okno = {c: {"pks1": kp[c][0], "tef1": kp[c][1],
                 "rozstep": kp[c][1] - kp[c][0]} for c in KONSTR}
    _RESULTS["okno_kotwic"] = _okno
    # MID has to hold in every construct, so the narrowest window decides the
    # difficulty, not the mean.
    _RESULTS["okno_najwezsze"] = min(v["rozstep"] for v in _okno.values())
    verdict("okno_MID", okno_ok,
            "tef1 > pks1 in every construct: %s"
            % {c: round(kp[c][1] - kp[c][0], 5) for c in KONSTR})

    # MID feasibility through a single substitution, checked on this model.
    if not SZYBKO:
        ZAS = list(CH.ZASADY)
        poz = rng.choice(L, size=min(96, L), replace=False)
        kand, description = [], []
        for p in poz:
            for b in ZAS:
                if b == PKS1[p]:
                    continue
                kand.append(PKS1[:p] + b + PKS1[p + 1:])
                description.append((int(p), b))
        P = {}
        for c in KONSTR:
            ps = []
            for i in range(0, len(kand), 256):
                cz = kand[i:i + 256]
                ps.append(np.asarray(sc.raw(cz, c, batch=len(cz))["pair_score"]
                                     ).reshape(-1))
            P[c] = np.concatenate(ps)
        nad = np.ones(len(kand), dtype=bool)
        pod = np.ones(len(kand), dtype=bool)
        for c in KONSTR:
            nad &= P[c] > kp[c][0]
            pod &= P[c] < kp[c][1]
        n_mid = int((nad & pod).sum())
        verdict("MID_jedna_edycja", n_mid > 0,
                "%d of %d single substitutions satisfy pks1 < X < tef1"
                % (n_mid, len(kand)))
        _RESULTS["mid_jedna_edycja"] = {"spelnia": n_mid, "zbadane": len(kand)}

    sekcja("7. THE LADDER: code usage and radii")
    raport = None
    f = getattr(lad, "report", None)
    if callable(f):
        try:
            raport = f()
        except Exception as e:                        # noqa: BLE001
            print("  WARNING: ladder.report() raised: %r" % (e,))
    # Code usage is computed on the corpus, not taken from the EMA history:
    # `dr_uzyte` counts cells used ever, and reported 729 of 729 where only
    # 74 codes were really working.
    korp = KK.corpus_promoters(genome="P1", limit=N_KORPUS)
    seqs = [s for s in (korp if isinstance(korp, list) else list(korp))]
    seqs = [s if isinstance(s, str) else s[1] for s in seqs][:N_KORPUS]
    uzyte = [set() for _ in idx_p]
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            cz = seqs[i:i + 32]
            ii = model.encode_codes(scoring.one_hot(cz).to(dev))
            for lv, t in enumerate(ii):
                uzyte[lv].update(int(v) for v in t.reshape(-1).tolist())
    print("  P1 corpus: %d promoters" % len(seqs))
    print("  %-8s %10s %10s %12s %10s"
          % ("level", "alphabet", "used", "fraction", "bits"))
    drab = {}
    for lv in range(len(idx_p)):
        a = alf[lv]
        u = len(uzyte[lv])
        b = np.log2(max(u, 1))
        drab[lv] = {"alfabet": a, "uzyte": u, "bity": float(b)}
        print("  L%-7d %10d %10d %12.4f %10.3f" % (lv, a, u, u / float(a), b))
    _RESULTS["drabina"] = drab
    if raport is not None and isinstance(raport, dict):
        pr = {k: v for k, v in raport.items()
              if "promien" in k or "radius" in k or "d_iloraz" in k}
        if pr:
            print("  ladder report: %s"
                  % " ".join("%s=%s" % (k, np.round(v, 4)
                                        if not isinstance(v, (list, tuple))
                                        else np.round(v, 4))
                             for k, v in sorted(pr.items())))
    # Blocking: one code per level is codebook collapse, and a collapsed
    # model in eval then returns a constant.
    verdict("kodebook_bez_zapadu", all(len(u) > 1 for u in uzyte),
            "codes used per level %s" % [len(u) for u in uzyte])

    sekcja("8. CONDITIONING CHANNELS")
    d = sc.context_delta()
    nul = {k: v for k, v in d.items() if k.startswith("NULL")}
    tre = {k: v for k, v in d.items() if not k.startswith("NULL")}
    zle_nul = {k: v for k, v in nul.items() if v != 0.0}
    verdict("kontrola_zerowa_kontekstu", not zle_nul,
            "NULL control exactly 0 for %d/%d pairs"
            % (len(nul) - len(zle_nul), len(nul)))
    verdict("konteksty_rozne", all(v > 0 for v in tre.values()),
            "max|ctx_a - ctx_b| = %s"
            % {k: round(v, 5) for k, v in tre.items()})

    sekcja("9. SUMMARY")
    zap = [k for k, v in _RESULTS.items()
           if isinstance(v, dict) and v.get("zaporowa") and not v.get("ok")]
    obs = [k for k, v in _RESULTS.items()
           if isinstance(v, dict) and not v.get("zaporowa")
           and v.get("ok") is False]
    print("  blocking sections failed:     %d %s" % (len(zap), zap))
    print("  observational with a warning: %d %s" % (len(obs), obs))
    print("  time %.1f s" % (time.time() - t0))

    try:
        os.makedirs(os.path.dirname(WY), exist_ok=True)
        with io.open(WY, "w", encoding="utf-8") as f:
            f.write(json.dumps(_RESULTS, indent=2, ensure_ascii=False,
                               default=str))
        print("  saved: %s" % WY)
    except Exception as e:                            # noqa: BLE001
        print("  WARNING: JSON not written: %r" % (e,))

    return 1 if zap else 0


if __name__ == "__main__":
    raise SystemExit(main())
