"""Which conditioning channel carries the train-val gap, and which carries PCC.

    python scripts/ablate_channels.py weights/hyppe_foundational.pt

A question of price, not of mechanism: a channel that gives a lot on training
and nothing, or less than nothing, on validation carries memorisation, and
removing it raises validation by as much as comes out here, before anybody
retrains from scratch.

Validation is used in full, against a random subsample of training of the same
size -- not the first N rows, because the corpus is laid out by genome.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

K = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, K + "/src")
os.environ.setdefault("HYPPE_DATA_ROOT", K)

from hyppe.baselines import within_row_contrast  # noqa: E402
from hyppe.config import from_mapping  # noqa: E402
from hyppe.training.loop import RunSettings, build_loop  # noqa: E402

EPOKI = [s.strip() for s in
         os.environ.get("EPOKI", "004,014").split(",") if s.strip()]
WY = os.environ.get("WY", K + "/results/ablacja_kanalow.json")
RUN = os.environ.get("RUN", "HYPPE_FIN")
CKPTY = [a for a in sys.argv[1:] if not a.startswith("-")] or \
    [s.strip() for s in os.environ.get("CKPTY", "").split(",") if s.strip()]
TRYBY = [s.strip() for s in os.environ.get(
    "TRYBY",
    "baza,bez_gene,bez_flank,bez_text,bez_gene_flank,tas_flank,tas_gene,tas_species"
).split(",") if s.strip()]

_TRYB = {"v": "baza"}


def _sciezki():
    """[(label, path)] -- explicit paths win over the name pattern."""
    if CKPTY:
        out = []
        for p in CKPTY:
            if not os.path.isfile(p):
                raise SystemExit("CKPTY: no such checkpoint: %s" % p)
            ep = torch.load(p, map_location="cpu",
                            weights_only=False).get("epoch")
            out.append(("%03d" % int(ep) if ep is not None
                        else os.path.basename(p), p))
        return out
    return [(ep, "%s/runs/%s_ep%s.pt" % (K, RUN, ep)) for ep in EPOKI]


def _abs(p, domyslna):
    """Resolve a path recorded in a checkpoint against the data root.
    """
    if not p:
        return domyslna
    return p if os.path.isabs(p) else os.path.join(K, p)


def _zeruj(b, pole: str) -> None:
    t = getattr(b, pole, None)
    if t is not None:
        try:
            setattr(b, pole, torch.zeros_like(t))
        except Exception:                      # dataclass frozen
            object.__setattr__(b, pole, torch.zeros_like(t))


def _tasuj(b, pola, rng) -> None:
    n = int(b.x.shape[0])
    perm = torch.from_numpy(rng.permutation(n))
    for pole in pola:
        t = getattr(b, pole, None)
        if t is not None and t.shape[0] == n:
            try:
                setattr(b, pole, t[perm].clone())
            except Exception:
                object.__setattr__(b, pole, t[perm].clone())


def main() -> int:
    _cele = _sciezki()
    _p0 = _cele[0][1]
    print("loading: %s" % _p0)
    ck0 = torch.load(_p0, map_location="cpu", weights_only=False)
    cfg = from_mapping(ck0["config"])
    cfg.validate()
    st = RunSettings(
        name="ablacja", device=os.environ.get("HYPPE_DEVICE", "cuda"),
        batch_size=int(os.environ.get("HYPPE_BATCH", "48")),
        sequences_path=K + "/final_datasets/sequences/sequences.tsv.gz",
        combat_path=_abs(ck0.get("combat_path"),
                         K + "/final_datasets/og_scope/anchor3_log2.tsv.gz"),
        samples_path=K + "/data/samples_metadata.tsv",
        splits_path=_abs(ck0.get("splits_path"),
                         K + "/data/splits/SPLIT_V6.tsv"),
        split_column=ck0.get("split_column") or "H_kmer90",
        flank_cache_path=K + "/cache/dna_lm_flanks/embeddings.npy")
    loop = build_loop(cfg, st)
    cor = loop.maker.corpus

    tr = cor.indices("train")
    n_val = int(cor.indices("val").size)
    sub = np.random.default_rng(11).choice(tr, size=n_val, replace=False)
    cor._by_split["trsub"] = np.sort(sub)

    _oryg = loop.maker._build
    _rng = np.random.default_rng(4242)

    def _atom(b, t):
        if t == "bez_gene":
            _zeruj(b, "gene_ok")
        elif t == "bez_flank":
            _zeruj(b, "flank_ok")
        elif t == "bez_text":
            _zeruj(b, "text_ok")
        elif t == "bez_gene_flank":          # legacy name, kept so the JSON
            _zeruj(b, "gene_ok")             # keys of earlier runs still
            _zeruj(b, "flank_ok")            # mean the same thing
        elif t == "tas_flank":
            _tasuj(b, ("flank_l", "flank_r", "flank_ok"), _rng)
        elif t == "tas_gene":
            _tasuj(b, ("gene_numeric", "gene_identity", "gene_text",
                       "gene_ok"), _rng)
        elif t == "tas_text":
            _tasuj(b, ("text_emb", "text_ok"), _rng)
        elif t == "tas_species":
            _tasuj(b, ("species_id",), _rng)
        elif t != "baza":
            raise SystemExit("unknown ablation atom: %r" % (t,))

    def _hak(*a, **kw):
        b = _oryg(*a, **kw)
        for _t in str(_TRYB["v"]).split("+"):
            _atom(b, _t)
        return b

    loop.maker._build = _hak
    results = {"kontrola_haka": None, "epoki": {}, "zrodla": {}}

    import hashlib
    for _ep, _pp in _cele:
        with open(_pp, "rb") as _fh:
            _md5 = hashlib.md5(_fh.read()).hexdigest()
        results["zrodla"][_ep] = {"path": _pp, "md5": _md5}
        print("  source %s: %s  md5 %s" % (_ep, _pp, _md5))

    for ep, _pp in _cele:
        print("  loading: %s" % _pp)
        ck = torch.load(_pp,
                        map_location="cpu", weights_only=False)
        missing, nad = loop.model.load_state_dict(ck["model"], strict=False)
        if missing or nad:
            print("STOP: %s missing %d unexpected %d"
                  % (ep, len(missing), len(nad)))
            return 3
        print("\n=== ep%s ==========================================" % ep)
        rec = {}
        baza = {}
        for tryb in TRYBY:
            _TRYB["v"] = tryb
            rec[tryb] = {}
            for split in ("val", "trsub"):
                t0 = time.time()
                m = loop.evaluate(split)
                p = float(m.get("pcc_cell_readout", float("nan")))
                rec[tryb][split] = p
                rec[tryb][split + "_pcc"] = float(m.get("pcc", float("nan")))
                _2d = getattr(loop, "_readout_2d", None)
                if _2d is not None:
                    _wr = within_row_contrast(_2d[0], _2d[1], _2d[2])
                    if (tryb == "baza" and split == "val"
                            and results.get("kontrola_within") is None):
                        _p, _v = _2d[0].astype(np.float64), _2d[2]
                        _n = _v.sum(axis=1, keepdims=True)
                        _sr = np.where(_n > 0,
                                       np.where(_v, _p, 0.0).sum(
                                           axis=1, keepdims=True)
                                       / np.maximum(_n, 1), 0.0)
                        _stala = np.repeat(_sr, _p.shape[1], axis=1)
                        _kc = within_row_contrast(_stala, _2d[1], _v)
                        results["kontrola_within"] = {
                            "pred_sd": float(_kc["within_row_pred_sd"]),
                            "within_row_pcc": float(_kc["within_row_pcc"]),
                            "prog_pred_sd": 1e-12}
                        print("  CONTROL constant-along-row: pred_sd %.3e "
                              "(must be ~0), pcc %s"
                              % (_kc["within_row_pred_sd"],
                                 _kc["within_row_pcc"]))
                        if _kc["within_row_pred_sd"] > 1e-12:
                            print("STOP: the within-row control is not zero, "
                                  "so the measure does not do what it claims")
                            return 4
                    for _k in ("within_row_pcc", "within_row_pred_sd",
                               "within_row_target_sd", "within_row_rows",
                               "within_fraction"):
                        rec[tryb]["%s_%s" % (split, _k)] = float(
                            _wr.get(_k, float("nan")))
                    if split == "val":
                        print("  %-16s %-6s within_row %.5f | pred_sd %.4f "
                              "(target_sd %.4f, %d rows)"
                              % (tryb, split, _wr["within_row_pcc"],
                                 _wr["within_row_pred_sd"],
                                 _wr["within_row_target_sd"],
                                 int(_wr["within_row_rows"])))
                if tryb == "baza":
                    baza[split] = p
                print("  %-16s %-6s pcc_cell %.5f | vs base %+.5f "
                      "(%.0f s)" % (tryb, split, p, p - baza.get(split, p),
                                    time.time() - t0))
                sys.stdout.flush()
            rec[tryb]["luka"] = rec[tryb]["trsub"] - rec[tryb]["val"]
            print("  %-16s GAP  %+.5f  (base %+.5f)"
                  % (tryb, rec[tryb]["luka"],
                     baza.get("trsub", 0) - baza.get("val", 0)))
        results["epoki"][ep] = rec
        with open(WY, "w") as f:
            json.dump(results, f, indent=1)

    print("\n<== SUMMARY (pcc_cell_readout) ===================>")
    for ep, _ in _cele:
        r = results["epoki"][ep]
        b = r["baza"]
        print(" ep%s  mode             val      d(val)    train    d(train)"
              "    GAP" % ep)
        for tryb in TRYBY:
            print("      %-16s %.5f %+.5f  %.5f %+.5f  %+.5f"
                  % (tryb, r[tryb]["val"], r[tryb]["val"] - b["val"],
                     r[tryb]["trsub"], r[tryb]["trsub"] - b["trsub"],
                     r[tryb]["luka"]))
    print("\nsaved: %s" % WY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
