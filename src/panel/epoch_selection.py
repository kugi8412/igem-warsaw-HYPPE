"""Score every archived epoch and apply `selection_rule` verbatim.

The six metrics do not share one resampling unit, and the rule asks for
orthogroups, so each error bar states its own:

    spy_scc, pcc, pair_acc, disc_auc   spy orthogroups (300)
    val_scc                            val orthogroups (299)
    rec_full                           evaluation batches, not orthogroups
    pks1_sep                           the shuffle set, not orthogroups

A pair is clustered through its promoter member, and a discriminator negative
through the promoter it was made from, recovered by recording the source row
of every negative-code call and asserting the recovery. `evaluate` builds
rec_full as a mean over per-batch scalars already rounded to 4 dp inside
`masking.py`, so no per-row spelling of it reproduces the log; batch
orthogroup purity is reported instead, as a measure of how close a batch is
to a cluster. `pks1_sep` has exactly one positive, pks1, so there is no
orthogroup to resample.

Every bootstrap draw is paired across epochs: replicate b uses the same
orthogroup, batch and shuffle draw for all 25 epochs, which is what makes the
between-epoch contrast sensitive (paired SE 0.0115-0.0169 against a per-epoch
SE of 0.022-0.025).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.environ.get("HYPPE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "src")
PARTS = os.path.join(ROOT, "src", "panel")
for _p in (PARTS, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import hyppe  # noqa: E402

if not os.path.abspath(hyppe.__file__).startswith(os.path.realpath(SRC)):
    raise SystemExit(
        "imported the WRONG hyppe package: %s (expected under %s). TWO "
        "packages named hyppe exist on this host and %s/hyppe shadows "
        "src/hyppe whenever the working directory is %s. Run with cwd=%s."
        % (hyppe.__file__, SRC, ROOT, ROOT, SRC))

import anchors as K  # noqa: E402
import model_io  # noqa: E402
import panel_spec  # noqa: E402
import repro  # noqa: E402
import scoring  # noqa: E402
import selection_rule as RULE  # noqa: E402

from hyppe.config import from_mapping  # noqa: E402
from hyppe.data.negatives import assert_shuffle_preserves, euler_shuffle  # noqa: E402
from hyppe.training import loop as loop_mod  # noqa: E402
from hyppe.training.batches import BatchMaker  # noqa: E402
from hyppe.training.loop import RunSettings, TrainLoop, build_corpus  # noqa: E402

KATEGORIA = "biotic antagonism"

#: Fixed seeds
SEED_BOOT = 20260914
SEED_SHUFFLE = 20260914

OUT_DIR = os.path.join(ROOT, "results")
NPZ = os.path.join(OUT_DIR, "wybor_epoki_wektory.npz")
JSON_STAGE1 = os.path.join(OUT_DIR, "wybor_epoki_surowe.json")
JSON_OUT = os.path.join(OUT_DIR, "epoch_selection.json")


def read_run_log(path: str) -> Dict[int, Dict[str, float]]:
    """The `val` line of every epoch in the run log, keyed by line order.

    The log is the reference this evaluation has to reproduce, so it is parsed
    rather than recalled, and the line itself carries no epoch number.
    """
    pat = re.compile(
        r"^\s+val\s+pcc\s+(-?[\d.]+)\s+scc\s+(-?[\d.]+)\s+pair_acc\s+"
        r"(-?[\d.]+)\s+auc\s+(-?[\d.]+)\s+rec_full\s+(-?[\d.]+)")
    out: Dict[int, Dict[str, float]] = {}
    if not os.path.exists(path):
        return out
    n = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            n += 1
            out[n] = {"pcc": float(m.group(1)), "scc": float(m.group(2)),
                      "pair_acc": float(m.group(3)),
                      "disc_auc": float(m.group(4)),
                      "rec_full": float(m.group(5))}
    return out


class Capture(object):
    """Per-batch record of everything `evaluate` will turn into a scalar.

    Every hook wraps a symbol `evaluate` itself calls, so nothing here is a
    second computation of a metric: it is the same tensor, copied on its way
    past. A second forward pass would give two spellings of every number and
    no way to tell which one is in the log.
    """

    def __init__(self) -> None:
        self.batches: List[dict] = []
        self.scc_vec: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.auc_vec: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.neg_src: List[int] = []
        self._installed = False

    def reset(self) -> None:
        self.batches = []
        self.scc_vec = None
        self.auc_vec = None
        self.neg_src = []

    def install_module_hooks(self) -> None:
        """Wrap `spearman` and `roc_auc` inside `hyppe.training.loop`."""
        if self._installed:
            return
        real_sp = loop_mod.spearman

        def sp(p, t):
            if self.scc_vec is None:
                self.scc_vec = (np.asarray(p).copy(), np.asarray(t).copy())
            return real_sp(p, t)

        real_auc = loop_mod.roc_auc

        def auc(s, l):
            if self.auc_vec is None:
                self.auc_vec = (np.asarray(s).copy(), np.asarray(l).copy())
            return real_auc(s, l)

        sp._hyppe_capture = True
        sp._real = real_sp
        auc._hyppe_capture = True
        auc._real = real_auc
        loop_mod.spearman = sp
        loop_mod.roc_auc = auc
        self._installed = True

    def wrap_maker(self, maker) -> None:
        """Record each batch's row ids, pair structure and negative sources.

        The source row of a negative is what lets it be clustered with the
        promoter it was made from.
        """
        for name, arg in (("negative_codes", 0), ("chimera_codes", 0),
                          ("reverse_codes", 0), ("_pool_codes", 1)):
            fn = getattr(maker, name, None)
            if fn is None:
                continue
            setattr(maker, name, self._wrap_src(fn, arg))

        original = maker.epoch

        def epoch(split, *a, **kw):
            for b in original(split, *a, **kw):
                n_pos = int(b.row_index.numel())
                size = int(len(b))
                rows = np.full(size, -1, dtype=np.int64)
                rows[:n_pos] = b.row_index.detach().cpu().numpy()
                src: List[int] = []

                for r in self.neg_src:
                    if not src or src[-1] != r:
                        src.append(int(r))

                self.neg_src = []
                self.batches.append({
                    "rows": rows, "n_pos": n_pos, "size": size,
                    "valid": b.mu_valid.detach().cpu().numpy().astype(bool),
                    "neg_src": np.asarray(src, dtype=np.int64),
                    "pair_a": b.pair_a.detach().cpu().numpy().astype(np.int64),
                    "pair_b": b.pair_b.detach().cpu().numpy().astype(np.int64),
                    "pair_sign": b.pair_sign.detach().cpu().numpy(),
                    "pair_delta": None, "rec_full": float("nan"),
                })
                yield b

        maker.epoch = epoch

    def _wrap_src(self, fn, arg_index: int):
        def wrapped(*a, **kw):
            try:
                self.neg_src.append(int(a[arg_index]))
            except (IndexError, TypeError, ValueError):
                pass
            return fn(*a, **kw)
        return wrapped

    def wrap_loop(self, loop) -> None:
        real_pair = loop._pair_output

        def pair_output(score, batch):
            po = real_pair(score, batch)
            if self.batches:
                self.batches[-1]["pair_delta"] = \
                    po.delta.detach().cpu().numpy()
            return po

        loop._pair_output = pair_output

        real_rec = loop.model.reconstruction_loss

        def reconstruction_loss(out, target_idx, *a, **kw):
            total, report = real_rec(out, target_idx, *a, **kw)
            if self.batches:
                self.batches[-1]["rec_full"] = float(
                    report.get("rec_full", float("nan")))
            return total, report

        loop.model.reconstruction_loss = reconstruction_loss

    def assemble(self, og_of_row: np.ndarray) -> dict:
        """Cluster labels aligned with each captured vector, plus diagnostics."""
        scc_og, disc_og, pair_og, pair_ok = [], [], [], []
        rec_vals, rec_ogs = [], []
        neg_recovered, neg_total = 0, 0
        for rec in self.batches:
            rows, valid, n_pos = rec["rows"], rec["valid"], rec["n_pos"]
            size = rec["size"]
            og_row = np.full(size, -1, dtype=np.int64)
            og_row[:n_pos] = og_of_row[rows[:n_pos]]
            src = rec["neg_src"]
            n_neg = size - n_pos
            neg_total += n_neg
            if src.size == n_neg and np.unique(src).size == src.size:
                og_row[n_pos:] = og_of_row[src]
                neg_recovered += n_neg
            # scc cells
            ix = og_row
            if valid.ndim == 2:
                ix = np.repeat(og_row[:, None], valid.shape[1], axis=1)
            if valid.any():
                scc_og.append(ix[valid])

            disc_og.append(og_row.copy())
            pa, pb = rec["pair_a"], rec["pair_b"]
            if rec["pair_delta"] is not None and pa.size:
                g = np.where(pa < n_pos, og_row[np.clip(pa, 0, size - 1)],
                             og_row[np.clip(pb, 0, size - 1)])
                pair_og.append(g)
                pair_ok.append((rec["pair_delta"] * rec["pair_sign"]) > 0)
            rec_vals.append(rec["rec_full"])
            rec_ogs.append(int(np.unique(og_row[:n_pos]).size))
        cat = lambda xs: (np.concatenate(xs) if xs
                          else np.zeros(0, dtype=np.int64))
        return {
            "scc_og": cat(scc_og),
            "disc_og": cat(disc_og),
            "pair_og": cat(pair_og),
            "pair_ok": (np.concatenate(pair_ok) if pair_ok
                        else np.zeros(0, dtype=bool)),
            "rec_batch": np.asarray(rec_vals, dtype=np.float64),
            "og_per_batch": np.asarray(rec_ogs, dtype=np.int64),
            "neg_recovered": int(neg_recovered), "neg_total": int(neg_total),
        }


def build_epoch_loop(path: str, splits_path: str, split_column: str,
                     device: str, corpus_cache: dict, cap: Capture):
    """One `TrainLoop` around one checkpoint, built as `build_loop` builds it."""
    model, prov = model_io.load_model(path, device=device)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_mapping(blob["config"], strict=False)
    cfg.source = path
    sp = splits_path or str(blob.get("splits_path", ""))
    col = split_column or str(blob.get("split_column", ""))
    if not sp or not col:
        raise SystemExit("no splits_path/split_column for %s" % path)
    st = RunSettings(name="epoch_selection", out_dir=OUT_DIR, device=device,
                     splits_path=sp, split_column=col,
                     negative_fraction=0.25, skip_gates=True)
    key = (cfg.data.flank_embedding_path, cfg.data.sequence_length, sp, col)
    if key not in corpus_cache:
        torch.manual_seed(int(cfg.training.seed))
        np.random.seed(int(cfg.training.seed))
        corpus_cache[key] = build_corpus(cfg, st, model.strain_names)
    corpus = corpus_cache[key]
    maker = BatchMaker(
        corpus,
        batch_size=int(st.batch_size or cfg.data.batch_size),
        negative_fraction=float(st.negative_fraction),
        seed=int(cfg.training.seed),
        per_category=bool(getattr(cfg.data, "target_per_category", False)),
        sequences_path=str(getattr(cfg.data, "sequences_path", "") or ""),
        negative_mix=(dict(getattr(cfg.heads.discriminator, "negative_mix",
                                   {}) or {}) or None))
    cap.wrap_maker(maker)
    loop = TrainLoop(cfg, model, maker, st)
    cap.wrap_loop(loop)
    return loop, cfg, corpus, prov


def score_split(loop, cap: Capture, split: str, og_of_row: np.ndarray) -> dict:
    """`TrainLoop.evaluate` on one split, plus the aligned per-item vectors."""
    cap.reset()
    met = loop.evaluate(split=split, max_steps=0)
    if cap.scc_vec is None:
        raise SystemExit("split %r produced no absolute-head cells" % split)
    p, t = cap.scc_vec
    asm = cap.assemble(og_of_row)
    if asm["scc_og"].size != p.size:
        raise SystemExit(
            "CLUSTER LABELS MISALIGNED on %r: %d labels for %d predictions. "
            "A wrong label vector makes a plausible and wrong error bar, so "
            "this is an exit code." % (split, asm["scc_og"].size, p.size))
    if cap.auc_vec is not None:
        s, l = cap.auc_vec
        if asm["disc_og"].size != s.size:
            raise SystemExit(
                "DISCRIMINATOR LABELS MISALIGNED on %r: %d labels for %d "
                "items" % (split, asm["disc_og"].size, s.size))
    if asm["pair_ok"].size != asm["pair_og"].size:
        raise SystemExit("pair vectors disagree on %r" % split)
    return {"met": {k: float(v) for k, v in met.items()},
            "pred": p, "target": t, "asm": asm,
            "disc_score": None if cap.auc_vec is None else cap.auc_vec[0],
            "disc_label": None if cap.auc_vec is None else cap.auc_vec[1]}


def make_shuffles(seq: str, n: int, shuffle_order: int,
                  verify: int = 4) -> Tuple[List[str], List[int]]:
    """`n` shuffles of one sequence, with the training negatives' shuffler.

    Same `euler_shuffle` and the same fallback ladder as
    `BatchMaker.negative_codes` (order `shuffle_order` down to 2: at order 4
    about 4.9 % of promoters admit no Eulerian path, and dropping them biases
    the class toward low homopolymer content). The order each shuffle used is
    returned, because the class is named `dinucleotide_shuffle` but at order 4
    preserves 5-mer counts, a much stronger negative than the name says.
    """
    out, orders, checked = [], [], 0
    for j in range(n):
        got = None
        for order in range(int(shuffle_order), 1, -1):
            try:
                s = euler_shuffle(seq, order=order,
                                  rng=random.Random(SEED_SHUFFLE + j))
                if checked < verify:
                    assert_shuffle_preserves(seq, s, order=order)
                    checked += 1
            except (ValueError, RuntimeError, AssertionError):
                continue
            got, ordr = s, order
            break
        if got is None:
            raise SystemExit(
                "pks1 admits no shuffle at any order from %d down to 2"
                % (shuffle_order,))
        if got == seq:
            raise SystemExit("shuffle %d returned its input" % j)
        out.append(got)
        orders.append(ordr)
    if len(set(out)) != len(out):
        raise SystemExit("the shuffle set contains duplicates; the AUC would "
                         "then be computed against fewer draws than claimed")
    return out, orders


def auc_one_vs_many(value: float, others: np.ndarray) -> float:
    """Mann-Whitney AUC with one positive: P(positive > draw) + 0.5 P(tie)."""
    o = np.asarray(others, dtype=np.float64)
    return float((np.sum(o < value) + 0.5 * np.sum(o == value)) / o.size)


def measure_pks1(model, prov, sh_pks1: Sequence[str], sh_tef1: Sequence[str],
                 anchors: K.AnchorSequences) -> dict:
    """Both anchors against their own shuffles, both heads, both constructs."""
    per: Dict[str, dict] = {}
    for construct in ("pHYPPE", "pAN7-1"):
        sc = scoring.Scorer(model, prov,
                          experiment_text=panel_spec.description(construct),
                          species="P1", constructs=(construct,),
                          anchors=anchors)
        k = sc.category_index(KATEGORIA)
        anc = K.measure_anchors(sc, construct, KATEGORIA, sequences=anchors)
        raw_a = sc.raw([anchors.pks1, anchors.tef1], construct)
        if (abs(float(raw_a["mu"][0, k]) - float(anc.mu_pks1)) > 1e-6
                or abs(float(raw_a["mu"][1, k]) - float(anc.mu_tef1)) > 1e-6):
            raise SystemExit(
                "REFUSED: `anchors.measure_anchors` and `Scorer.raw` disagree "
                "about the anchors in construct %r. They are the same forward "
                "pass; a difference means two scoring paths exist and the "
                "table would be mixing them." % (construct,))
        rp = sc.raw(list(sh_pks1), construct)
        rt = sc.raw(list(sh_tef1), construct)
        mu_shp = rp["mu"][:, k]
        mu_sht = rt["mu"][:, k]
        per[construct] = {
            "mu_pks1": float(anc.mu_pks1), "mu_tef1": float(anc.mu_tef1),
            "gap_tef1_minus_pks1": float(anc.gap),
            "ordering_ok": bool(anc.ordering_ok),
            "realness_pks1": float(raw_a["realness"][0]),
            "realness_tef1": float(raw_a["realness"][1]),
            "auc_realness_pks1": auc_one_vs_many(float(raw_a["realness"][0]),
                                                 rp["realness"]),
            "auc_realness_tef1": auc_one_vs_many(float(raw_a["realness"][1]),
                                                 rt["realness"]),
            "auc_mu_pks1": auc_one_vs_many(float(anc.mu_pks1), mu_shp),
            "auc_mu_tef1": auc_one_vs_many(float(anc.mu_tef1), mu_sht),
            "realness_shuffle_mean": float(np.mean(rp["realness"])),
            "realness_shuffle_sd": float(np.std(rp["realness"])),
            "delta_realness_vs_shuffle": float(raw_a["realness"][0]
                                               - np.mean(rp["realness"])),
            "mu_shuffle_mean": float(np.mean(mu_shp)),
            "delta_mu_vs_shuffle": float(anc.mu_pks1 - np.mean(mu_shp)),
            "context_error": float(sc.last_context_error),
            "_vec_realness_pks1_shuffles": np.asarray(rp["realness"],
                                                      dtype=np.float64),
            "_vec_mu_pks1_shuffles": np.asarray(mu_shp, dtype=np.float64),
        }
        del sc
    return per


# Bootstrap
class Draws(object):
    """One set of paired resampling draws, reused by every epoch.

    Four units, because the six metrics do not share one: spy orthogroups,
    val orthogroups, evaluation batches and the shuffle set.
    """

    def __init__(self, n_boot: int, seed: int, spy_og: np.ndarray,
                 val_og: np.ndarray, n_batches: int, n_shuffles: int) -> None:
        rng = np.random.default_rng(seed)
        self.n_boot = int(n_boot)
        self.spy_groups, self.spy_inv = np.unique(spy_og, return_inverse=True)
        self.val_groups, self.val_inv = np.unique(val_og, return_inverse=True)
        self.spy_pick = [rng.integers(0, self.spy_groups.size,
                                      size=self.spy_groups.size)
                         for _ in range(self.n_boot)]
        self.val_pick = [rng.integers(0, self.val_groups.size,
                                      size=self.val_groups.size)
                         for _ in range(self.n_boot)]
        self.batch_pick = [rng.integers(0, n_batches, size=n_batches)
                           for _ in range(self.n_boot)]
        self.shuf_pick = [rng.integers(0, n_shuffles, size=n_shuffles)
                          for _ in range(self.n_boot)]


def members_of(og: np.ndarray, groups: np.ndarray) -> List[np.ndarray]:
    _u, inv = np.unique(og, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(groups.size)]


def auc_from(scores: np.ndarray, labels: np.ndarray) -> float:
    fn = getattr(loop_mod.roc_auc, "_real", loop_mod.roc_auc)
    return float(fn(scores, labels))


def stage1(args) -> None:
    log = read_run_log(args.run_log)
    print("run log: %d val lines parsed from %s" % (len(log), args.run_log),
          flush=True)
    paths = sorted(glob.glob(os.path.join(args.runs, "%s_ep*.pt" % args.name)))
    if not paths:
        raise SystemExit("no %s_ep*.pt under %s" % (args.name, args.runs))
    todo = [(int(re.search(r"_ep(\d+)\.pt$", p).group(1)), p) for p in paths]
    if args.epochs:
        want = {int(x) for x in args.epochs.split(",")}
        todo = [(e, p) for e, p in todo if e in want]
    print("scoring %d checkpoints on device %s" % (len(todo), args.device),
          flush=True)

    anchors = K.load_anchor_sequences()
    print(anchors.note(), flush=True)
    sh_pks1, ord_p = make_shuffles(anchors.pks1, args.n_shuffles,
                                   args.shuffle_order)
    sh_tef1, ord_t = make_shuffles(anchors.tef1, args.n_shuffles,
                                   args.shuffle_order)
    from collections import Counter
    print("shuffle sets: pks1 %d seqs orders %s | tef1 %d seqs orders %s "
          "(euler_shuffle at order k preserves (k+1)-mer counts EXACTLY, so "
          "order 4 is a 5-mer-preserving negative although the training class "
          "is NAMED dinucleotide_shuffle)"
          % (len(sh_pks1), dict(Counter(ord_p)), len(sh_tef1),
             dict(Counter(ord_t))), flush=True)

    corpus_cache: dict = {}
    cap = Capture()
    cap.install_module_hooks()
    rows: List[dict] = []
    store: Dict[str, np.ndarray] = {}
    ref: Dict[str, np.ndarray] = {}
    for ep, path in todo:
        t0 = time.time()
        loop, cfg, corpus, prov = build_epoch_loop(
            path, args.splits_file, args.split_column, args.device,
            corpus_cache, cap)

        og_of_row = np.unique(np.asarray(corpus.og),
                              return_inverse=True)[1].astype(np.int64)
        rec: dict = {"epoch": ep, "file": os.path.basename(path),
                     "checkpoint_sha256": prov.checkpoint_sha256,
                     "n_missing_keys": len(prov.missing_keys),
                     "n_unexpected_keys": len(prov.unexpected_keys),
                     "ckpt_epoch": prov.epoch}
        for split in ("spy", "val"):
            r = score_split(loop, cap, split, og_of_row)
            rec[split] = r["met"]
            asm = r["asm"]
            rec[split + "_diag"] = {
                "n_cells": int(r["pred"].size),
                "n_orthogroups": int(np.unique(asm["scc_og"]).size),
                "n_pairs": int(asm["pair_ok"].size),
                "n_disc_items": int(asm["disc_og"].size),
                "n_batches": int(asm["rec_batch"].size),
                "neg_source_recovered": asm["neg_recovered"],
                "neg_total": asm["neg_total"],
                "mean_orthogroups_per_batch":
                    float(np.mean(asm["og_per_batch"])),
                "pair_acc_recomputed": (float(np.mean(asm["pair_ok"]))
                                        if asm["pair_ok"].size else
                                        float("nan")),
            }
            pre = "%s_ep%03d_" % (split, ep)
            store[pre + "pred"] = r["pred"].astype(np.float64)
            store[pre + "pair_ok"] = asm["pair_ok"]
            store[pre + "rec_batch"] = asm["rec_batch"]
            if r["disc_score"] is not None:
                store[pre + "disc_score"] = \
                    np.asarray(r["disc_score"], dtype=np.float64)

            for nm, arr in (("target", r["target"].astype(np.float64)),
                            ("scc_og", asm["scc_og"]),
                            ("disc_og", asm["disc_og"]),
                            ("pair_og", asm["pair_og"]),
                            ("disc_label", None if r["disc_label"] is None
                             else np.asarray(r["disc_label"],
                                             dtype=np.float64))):
                if arr is None:
                    continue
                key = "%s_%s" % (split, nm)
                if key not in ref:
                    ref[key] = arr
                elif not np.array_equal(ref[key], arr) and \
                        not np.allclose(ref[key], arr):
                    raise SystemExit(
                        "ep%03d saw a different %s on %s than the first "
                        "epoch did -- the paired bootstrap is refused, "
                        "because it would compare two different samples"
                        % (ep, nm, split))
        rec["pks1"] = measure_pks1(loop.model, prov, sh_pks1, sh_tef1,
                                   anchors)
        for c, dd in rec["pks1"].items():
            cc = c.replace("-", "_")
            store["pks1r_ep%03d_%s" % (ep, cc)] = \
                dd.pop("_vec_realness_pks1_shuffles")
            store["pks1m_ep%03d_%s" % (ep, cc)] = \
                dd.pop("_vec_mu_pks1_shuffles")

        rec["pks1_sep"] = float(np.mean(
            [dd["auc_realness_pks1"] for dd in rec["pks1"].values()]))
        rec["pks1_sep_mu_reading"] = float(np.mean(
            [dd["auc_mu_pks1"] for dd in rec["pks1"].values()]))
        rec["tef1_sep"] = float(np.mean(
            [dd["auc_realness_tef1"] for dd in rec["pks1"].values()]))
        rec["tef1_sep_mu_reading"] = float(np.mean(
            [dd["auc_mu_tef1"] for dd in rec["pks1"].values()]))

        if ep in log:
            rec["log_val"] = log[ep]
            rec["log_dev"] = {k: float(rec["val"].get(
                {"disc_auc": "disc_auc"}.get(k, k), float("nan")) - v)
                for k, v in log[ep].items()}
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        s, v = rec["spy"], rec["val"]
        print("  ep%03d | spy scc %.5f pair %.5f auc %.5f rec %.5f | "
              "val scc %.5f pair %.5f auc %.5f rec %.5f | pks1_sep %.4f | "
              "%.0f s"
              % (ep, s.get("scc", np.nan), s.get("pair_acc", np.nan),
                 s.get("disc_auc", np.nan), s.get("rec_full", np.nan),
                 v.get("scc", np.nan), v.get("pair_acc", np.nan),
                 v.get("disc_auc", np.nan), v.get("rec_full", np.nan),
                 rec["pks1_sep"], rec["seconds"]), flush=True)
        print("        pks1_sep realness %.4f (mu reading %.4f) | tef1 "
              "realness %.4f (mu %.4f) | gap tef1-pks1 pHYPPE %+.4f "
              "pAN7-1 %+.4f"
              % (rec["pks1_sep"], rec["pks1_sep_mu_reading"], rec["tef1_sep"],
                 rec["tef1_sep_mu_reading"],
                 rec["pks1"]["pHYPPE"]["gap_tef1_minus_pks1"],
                 rec["pks1"]["pAN7-1"]["gap_tef1_minus_pks1"]), flush=True)
        if "log_dev" in rec:
            d = rec["log_dev"]
            print("        vs run log (4 dp): d_scc %+.5f d_pcc %+.5f "
                  "d_pair %+.5f d_auc %+.5f d_rec %+.5f"
                  % (d["scc"], d["pcc"], d["pair_acc"], d["disc_auc"],
                     d["rec_full"]), flush=True)
        del loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with open(JSON_STAGE1, "w") as fh:
            json.dump({"rows": rows, "shuffle_orders_pks1": ord_p,
                       "shuffle_orders_tef1": ord_t,
                       "n_shuffles": len(sh_pks1),
                       "category": KATEGORIA,
                       "seed_boot": SEED_BOOT,
                       "seed_shuffle": SEED_SHUFFLE}, fh, indent=1)
        np.savez_compressed(NPZ, **store, **ref)
    print("stage 1 done: %s and %s" % (JSON_STAGE1, NPZ), flush=True)


def stage2(args) -> dict:
    """Bootstrap, then the committed rule, stage by stage."""
    with open(JSON_STAGE1) as fh:
        raw = json.load(fh)
    d = np.load(NPZ)
    rows = {int(r["epoch"]): r for r in raw["rows"]}
    eps = sorted(rows)
    print("loaded %d epochs: %s" % (len(eps), eps), flush=True)

    spy_og = d["spy_scc_og"]
    val_og = d["val_scc_og"]
    spy_t = d["spy_target"]
    val_t = d["val_target"]
    disc_og = d["spy_disc_og"]
    disc_lab = d["spy_disc_label"]
    pair_og = d["spy_pair_og"]
    n_batches = int(d["spy_ep%03d_rec_batch" % eps[0]].size)
    n_shuf = int(d["pks1r_ep%03d_pHYPPE" % eps[0]].size)

    draws = Draws(args.n_boot, SEED_BOOT, spy_og, val_og, n_batches, n_shuf)
    spy_members = members_of(spy_og, draws.spy_groups)
    val_members = members_of(val_og, draws.val_groups)
    disc_members = members_of(disc_og, np.unique(disc_og))
    pair_members = members_of(pair_og, np.unique(pair_og))
    disc_groups = np.unique(disc_og)
    pair_groups = np.unique(pair_og)

    if not np.array_equal(disc_groups, draws.spy_groups):
        print("  NOTE disc clusters: %d groups vs %d spy groups (negatives "
              "whose source was not recovered carry -1)"
              % (disc_groups.size, draws.spy_groups.size), flush=True)
    spearman = getattr(loop_mod.spearman, "_real", loop_mod.spearman)

    METRICS = ("spy_scc", "val_scc", "pair_acc", "disc_auc", "rec_full",
               "pks1_sep")
    point: Dict[str, Dict[int, float]] = {m: {} for m in METRICS}
    boot: Dict[str, Dict[int, np.ndarray]] = {
        m: {e: np.empty(args.n_boot) for e in eps} for m in METRICS}

    for e in eps:
        point["spy_scc"][e] = float(rows[e]["spy"]["scc"])
        point["val_scc"][e] = float(rows[e]["val"]["scc"])
        point["pair_acc"][e] = float(rows[e]["spy"]["pair_acc"])
        point["disc_auc"][e] = float(rows[e]["spy"]["disc_auc"])
        point["rec_full"][e] = float(rows[e]["spy"]["rec_full"])
        point["pks1_sep"][e] = float(rows[e]["pks1_sep"])

    gmap_d = {int(g): i for i, g in enumerate(disc_groups)}
    gmap_p = {int(g): i for i, g in enumerate(pair_groups)}
    d_route = np.array([gmap_d.get(int(g), -1) for g in draws.spy_groups])
    p_route = np.array([gmap_p.get(int(g), -1) for g in draws.spy_groups])
    print("  spy orthogroups routed to disc clusters: %d/%d, to pair "
          "clusters: %d/%d"
          % (int((d_route >= 0).sum()), d_route.size,
             int((p_route >= 0).sum()), p_route.size), flush=True)

    t0 = time.time()
    for b in range(args.n_boot):
        s_idx = np.concatenate([spy_members[i] for i in draws.spy_pick[b]])
        v_idx = np.concatenate([val_members[i] for i in draws.val_pick[b]])
        dr = d_route[draws.spy_pick[b]]
        pr = p_route[draws.spy_pick[b]]
        d_idx = np.concatenate([disc_members[i] for i in dr[dr >= 0]])
        p_idx = np.concatenate([pair_members[i] for i in pr[pr >= 0]])
        bt = draws.batch_pick[b]
        sh = draws.shuf_pick[b]
        st_ = spy_t[s_idx]
        vt_ = val_t[v_idx]
        dl_ = disc_lab[d_idx]
        for e in eps:
            boot["spy_scc"][e][b] = spearman(d["spy_ep%03d_pred" % e][s_idx],
                                             st_)
            boot["val_scc"][e][b] = spearman(d["val_ep%03d_pred" % e][v_idx],
                                             vt_)
            boot["pair_acc"][e][b] = float(
                np.mean(d["spy_ep%03d_pair_ok" % e][p_idx]))
            boot["disc_auc"][e][b] = auc_from(
                d["spy_ep%03d_disc_score" % e][d_idx], dl_)
            boot["rec_full"][e][b] = float(
                np.mean(d["spy_ep%03d_rec_batch" % e][bt]))
            a1 = auc_one_vs_many(rows[e]["pks1"]["pHYPPE"]["realness_pks1"],
                                 d["pks1r_ep%03d_pHYPPE" % e][sh])
            a2 = auc_one_vs_many(rows[e]["pks1"]["pAN7-1"]["realness_pks1"],
                                 d["pks1r_ep%03d_pAN7_1" % e][sh])
            boot["pks1_sep"][e][b] = 0.5 * (a1 + a2)
        if b and b % 200 == 0:
            print("    bootstrap %d/%d (%.0f s)"
                  % (b, args.n_boot, time.time() - t0), flush=True)

    se = {m: {e: float(np.std(boot[m][e], ddof=1)) for e in eps}
          for m in METRICS}

    admit: Dict[int, dict] = {}
    for e in eps:
        rec = float(rows[e]["spy"]["rec_full"])
        ok_rec = rec >= float(RULE.DOPUSZCZENIE["rec_full_min"])
        ord_ph = bool(rows[e]["pks1"]["pHYPPE"]["ordering_ok"])
        ord_an = bool(rows[e]["pks1"]["pAN7-1"]["ordering_ok"])
        ok_ord = (ord_ph and ord_an) or not RULE.DOPUSZCZENIE["tef1_above_pks1"]
        admit[e] = {"rec_full": rec, "rec_full_ok": ok_rec,
                    "tef1_above_pks1_pHYPPE": ord_ph,
                    "tef1_above_pks1_pAN7-1": ord_an,
                    "ordering_ok": ok_ord, "admitted": bool(ok_rec and ok_ord)}
    adm = [e for e in eps if admit[e]["admitted"]]
    if not adm:
        raise SystemExit("NO EPOCH IS ADMISSIBLE. That is a result, not a "
                         "reason to relax the floor.")

    se_metric = {m: float(np.mean([se[m][e] for e in adm])) for m in METRICS}
    mean_metric = {m: float(np.mean([point[m][e] for e in adm]))
                   for m in METRICS}
    z = {m: {e: (point[m][e] - mean_metric[m]) / se_metric[m] for e in adm}
         for m in METRICS}
    zb = {m: {e: (boot[m][e] - mean_metric[m]) / se_metric[m] for e in adm}
          for m in METRICS}

    comp = {e: float(sum(RULE.WEIGHTS[m] * z[m][e] for m in METRICS))
            for e in adm}
    comp_b = {e: sum(RULE.WEIGHTS[m] * zb[m][e] for m in METRICS) for e in adm}
    best = max(adm, key=lambda e: comp[e])
    se_comp = {e: float(np.std(comp_b[e], ddof=1)) for e in adm}
    paired_se = {e: float(np.std(comp_b[best] - comp_b[e], ddof=1))
                 for e in adm}
    wins = {e: 0 for e in adm}
    stack = np.vstack([comp_b[e] for e in adm])
    for b in range(args.n_boot):
        wins[adm[int(np.argmax(stack[:, b]))]] += 1

    band = float(RULE.PASMO_SE) * se_comp[best]
    zbior = sorted(e for e in adm if comp[e] >= comp[best] - band)
    zbior_paired = sorted(
        e for e in adm
        if comp[best] - comp[e] <= float(RULE.PASMO_SE) * paired_se[e])

    def plateau(members: Sequence[int]) -> Dict[int, Tuple[int, float]]:
        """Run length containing each member, and its distance to the centre.

        `plateau_center` is read as a maximal run of consecutive epoch numbers
        all inside the set, because the tie-break is applied within the set.
        Longer run first, then nearer the centre.
        """
        s = set(members)
        out: Dict[int, Tuple[int, float]] = {}
        for e in members:
            lo = e
            while lo - 1 in s:
                lo -= 1
            hi = e
            while hi + 1 in s:
                hi += 1
            out[e] = (hi - lo + 1, abs(e - 0.5 * (lo + hi)))
        return out

    pl = plateau(zbior)
    order = sorted(zbior, key=lambda e: (-pl[e][0], pl[e][1],
                                         -point["pks1_sep"][e], -e))
    chosen = order[0]
    why = "sole member of the set"
    if len(order) > 1:
        a, b2 = order[0], order[1]
        if pl[a][0] != pl[b2][0]:
            why = ("plateau_center: run length %d vs %d"
                   % (pl[a][0], pl[b2][0]))
        elif pl[a][1] != pl[b2][1]:
            why = ("plateau_center: distance to run centre %.1f vs %.1f"
                   % (pl[a][1], pl[b2][1]))
        elif point["pks1_sep"][a] != point["pks1_sep"][b2]:
            why = ("pks1_sep %.4f vs %.4f"
                   % (point["pks1_sep"][a], point["pks1_sep"][b2]))
        else:
            why = "later_epoch (%d over %d)" % (a, b2)

    out = {
        "rule_file": os.path.join(PARTS, "selection_rule.py"),
        "rule_sha256": model_io.sha256(os.path.join(PARTS,
                                                    "selection_rule.py")),
        "weights": dict(RULE.WEIGHTS), "source": dict(RULE.ZRODLO),
        "band_se": RULE.PASMO_SE, "tie_break": list(RULE.ROZSTRZYGNIECIE),
        "admission": dict(RULE.DOPUSZCZENIE),
        "category": KATEGORIA, "n_boot": args.n_boot,
        "seed_boot": SEED_BOOT, "seed_shuffle": SEED_SHUFFLE,
        "n_shuffles": n_shuf,
        "clusters": {
            "spy_orthogroups": int(draws.spy_groups.size),
            "val_orthogroups": int(draws.val_groups.size),
            "spy_cells": int(spy_og.size), "val_cells": int(val_og.size),
            "rec_full_unit": "evaluation batches (%d) -- NOT orthogroups"
                             % n_batches,
            "pks1_sep_unit": "the shuffle set (%d) -- NOT orthogroups"
                             % n_shuf,
        },
        "per_epoch": {}, "se_metric": se_metric, "mean_metric": mean_metric,
        "admissible": adm, "best_point": best, "band": band,
        "set": zbior, "set_paired_reading": zbior_paired,
        "chosen": chosen, "tie_break_decided_by": why,
        "argmax_fraction": {str(e): wins[e] / float(args.n_boot)
                            for e in adm},
    }
    for e in eps:
        r = rows[e]
        out["per_epoch"][str(e)] = {
            "epoch": e, "file": r["file"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "spy": r["spy"], "val": r["val"],
            "pks1": r["pks1"], "pks1_sep": r["pks1_sep"],
            "log_val": r.get("log_val"), "log_dev": r.get("log_dev"),
            "diag": {"spy": r.get("spy_diag"), "val": r.get("val_diag")},
            "se": {m: se[m][e] for m in METRICS},
            "admission": admit[e],
            "z": {m: z[m][e] for m in METRICS} if e in adm else None,
            "composite": comp.get(e), "composite_se": se_comp.get(e),
            "paired_se_vs_best": paired_se.get(e),
            "in_set": e in zbior,
        }
    with open(JSON_OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    # The report
    print("\n=== STAGE 0: does `val` reproduce the run log? ===", flush=True)
    worst = 0.0
    for e in eps:
        dv = rows[e].get("log_dev")
        if not dv:
            continue
        worst = max(worst, max(abs(x) for x in dv.values()))
    print("  worst absolute deviation over %d epochs x 5 metrics: %.2e "
          "(the log prints 4 dp, so 5e-5 is the floor of the comparison)"
          % (len(eps), worst), flush=True)

    print("\n=== STAGE 1: DOPUSZCZENIE (hard, before any score) ===",
          flush=True)
    for e in eps:
        a = admit[e]
        print("  ep%03d rec_full %.4f %s | tef1>pks1 pHYPPE %s pAN7-1 %s | %s"
              % (e, a["rec_full"], "OK" if a["rec_full_ok"] else "REFUSED",
                 a["tef1_above_pks1_pHYPPE"], a["tef1_above_pks1_pAN7-1"],
                 "ADMITTED" if a["admitted"] else "EXCLUDED"), flush=True)

    print("\n=== STAGE 2: per-epoch table (spy unless marked val) ===",
          flush=True)
    print("  ep | val_scc  spy_scc  pair_acc disc_auc rec_full pks1_sep | "
          "composite  +-SE", flush=True)
    for e in eps:
        c = comp.get(e)
        print("  %3d | %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  | %s"
              % (e, point["val_scc"][e], point["spy_scc"][e],
                 point["pair_acc"][e], point["disc_auc"][e],
                 point["rec_full"][e], point["pks1_sep"][e],
                 ("%+8.4f  %.4f" % (c, se_comp[e])) if c is not None
                 else "   EXCLUDED"), flush=True)
    print("  clustered SE used for the z-score: %s"
          % ", ".join("%s %.5f" % (m, se_metric[m]) for m in METRICS),
          flush=True)

    print("\n=== STAGE 3: the 1 SE band ===", flush=True)
    print("  best composite: ep%d at %+.4f, its bootstrap SD %.4f -> band "
          "%+.4f" % (best, comp[best], se_comp[best], comp[best] - band),
          flush=True)
    print("  SET (indistinguishable): %s" % (zbior,), flush=True)
    print("  under the PAIRED-SE reading it would be: %s" % (zbior_paired,),
          flush=True)
    print("  fraction of resamples in which each epoch has the best "
          "composite: %s"
          % ", ".join("ep%d %.3f" % (e, wins[e] / float(args.n_boot))
                      for e in adm if wins[e]), flush=True)

    print("\n=== STAGE 4: tie-break %s ===" % (RULE.ROZSTRZYGNIECIE,),
          flush=True)
    for e in order:
        print("  ep%03d run_len %d dist_to_centre %.1f pks1_sep %.4f"
              % (e, pl[e][0], pl[e][1], point["pks1_sep"][e]), flush=True)
    print("  CHOSEN: ep%d (%s)" % (chosen, why), flush=True)
    print("\nwrote %s" % JSON_OUT, flush=True)
    return out


def stage3(args, out: dict) -> None:
    """Copy the chosen checkpoint, stamp it, and move the stale one aside."""
    ep = int(out["chosen"])
    src = os.path.join(args.runs, "%s_ep%03d.pt" % (args.name, ep))
    dst = os.path.join(args.runs, "HYPPE_KONKURS.pt")
    stale = os.path.join(args.runs, "HYPPE_WYBRANY_ep002.pt")
    moved = ""
    if os.path.exists(stale):
        moved = os.path.join(
            args.runs, "STALE_POPRZEDNI_BIEG_HYPPE_WYBRANY_ep002.pt")
        shutil.move(stale, moved)
        print("moved the STALE artefact aside: %s -> %s"
              % (stale, moved), flush=True)
    shutil.copy2(src, dst)
    if model_io.sha256(src) != model_io.sha256(dst):
        raise SystemExit("REFUSED: the copy does not hash to the source")

    model, prov = model_io.load_model(dst, device="cpu")
    blob = torch.load(dst, map_location="cpu", weights_only=False)
    embedded = hashlib.sha256(
        json.dumps(blob["config"], sort_keys=True, default=str).encode()
    ).hexdigest()
    anchors = K.load_anchor_sequences()
    stamps = []
    for construct in ("pHYPPE", "pAN7-1"):
        sc = scoring.Scorer(model, prov,
                          experiment_text=panel_spec.description(construct),
                          species="P1", constructs=(construct,),
                          anchors=anchors)
        stamps.append(repro.Stamp.from_scorer(
            sc, strategy="epoch_selection",
            note="epoch %d chosen by selection_rule (sha %s)"
                 % (ep, out["rule_sha256"][:12])).as_dict())
        del sc

    run_cfg = os.path.join(ROOT, "configs", "generated", "HYPPE_F.yaml")
    payload = {
        "selected_epoch": ep,
        "selected_checkpoint": dst,
        "selected_sha256": model_io.sha256(dst),
        "source_checkpoint": src,
        "source_sha256": model_io.sha256(src),
        "rule_file": out["rule_file"], "rule_sha256": out["rule_sha256"],
        "config_source": prov.config_source,
        "embedded_config_sha256": embedded,
        "run_config_yaml": run_cfg,
        "run_config_sha256": model_io.sha256(run_cfg),
        "n_missing_keys": len(prov.missing_keys),
        "n_unexpected_keys": len(prov.unexpected_keys),
        "stale_artefact_moved_to": moved,
        "indistinguishable_set": out["set"],
        "set_paired_reading": out["set_paired_reading"],
        "tie_break": out["tie_break"],
        "tie_break_decided_by": out["tie_break_decided_by"],
        "weights": out["weights"], "source_of_each_metric": out["source"],
        "band_se": out["band_se"], "band": out["band"],
        "admission": out["admission"], "clusters": out["clusters"],
        "se_metric": out["se_metric"],
        "argmax_fraction": out["argmax_fraction"],
        "table": out["per_epoch"],
        "stamps": stamps,
    }
    path = os.path.join(args.runs, "HYPPE_KONKURS.stamp.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote %s and %s" % (dst, path), flush=True)
    for st in stamps:
        print("  stamp: ckpt %s epoch %s cfg %s | %d params"
              % (st["checkpoint_sha256"][:12], st["epoch"],
                 st["config_source"], prov.n_parameters), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HYPPE")
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--splits-file",
                    default=os.path.join(ROOT, "data/splits/SPLIT_V6.tsv"))
    ap.add_argument("--split-column", default="H_kmer90")
    ap.add_argument("--run-log",
                    default=os.path.join(ROOT, "logs/final/HYPPE.log"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-shuffles", type=int, default=256)
    ap.add_argument("--shuffle-order", type=int, default=4)
    ap.add_argument("--epochs", default="")
    ap.add_argument("--stage", default="all",
                    choices=("all", "1", "2", "3", "23"))
    args = ap.parse_args()
    torch.set_num_threads(int(args.threads))
    if args.stage in ("all", "1"):
        stage1(args)
    if args.stage in ("all", "2", "23"):
        out = stage2(args)
        if args.stage in ("all", "23"):
            stage3(args, out)
    elif args.stage == "3":
        with open(JSON_OUT) as fh:
            stage3(args, json.load(fh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
