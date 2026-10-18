"""Categoriser reading the REGRESSION module, not the description vector.

The previous head predicted the category from the description vector encoding
that category, and the tautology was measurable: an UNTRAINED model scored
0.9295 AUC against 0.8032 trained. This one reads the regression head's
representation after FiLM and its own columns `mu_k`, CENTRED over K, because
raw `mu_k` mostly carry the promoter's own strength (79-93 % of the variance)
and the profile is what remains. Only rows with `text_used == 0` train it;
elsewhere the category is in the input and the tautology just moves one module
along. Falsifier: untrained must sit at chance, trained must beat it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

__all__ = ["CategoryHead"]


class CategoryHead(nn.Module):
    """`h` [B, hidden] + `mu` [B, n_columns] -> logits [B, n_cat].

    `n_cat` is a FIXED class set, so changing it rebuilds the head: this is
    classification, not a read-out. `n_columns` is the regression column axis.
    """

    def __init__(self, in_dim: int, n_cat: int, hidden: int = 128,
                 dropout: float = 0.1, use_columns: bool = True,
                 n_columns: Optional[int] = None) -> None:
        super().__init__()
        if n_cat < 2:
            raise ValueError("n_cat must be at least 2, got %d" % n_cat)
        self.n_cat = int(n_cat)
        # Columns go where the RESOLUTION is (137-250 distinct conditioning
        # vectors measured, against 7 today), classes where the SUPPORT is
        # (median 3 samples per class at 250).
        self.n_columns = int(n_columns) if n_columns else self.n_cat
        self.use_columns = bool(use_columns)
        d_in = int(in_dim) + (self.n_columns if self.use_columns else 0)
        self.net = nn.Sequential(
            nn.Linear(d_in, int(hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Linear(int(hidden), self.n_cat))
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self._diag: Dict[str, float] = {}

    def forward(self, h: torch.Tensor,
                mu: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = h
        if self.use_columns:
            if mu is None:
                raise ValueError(
                    "use_columns=True but no mu was given; the head was "
                    "built to read the regression columns and silently "
                    "dropping them would make it a different head")
            if mu.dim() != 2 or mu.shape[-1] != self.n_columns:
                raise ValueError(
                    "mu must be [B, %d] (the REGRESSION column axis, not the "
                    "%d-class axis of this head), got %s -- the per-row "
                    "regime emits [B] and has no column profile, so the "
                    "categoriser must be given the per-column read-out "
                    "instead" % (self.n_columns, self.n_cat, tuple(mu.shape)))
            # Centring over K: see the module docstring.
            x = torch.cat([h, mu - mu.mean(dim=-1, keepdim=True)], dim=-1)
        return self.net(x)

    def loss(self, logits: torch.Tensor, target: torch.Tensor,
             text_used: Optional[torch.Tensor] = None,
             track: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy on rows WITHOUT a description and with a valid target.

        With no qualifying row it returns a zero tensor on a live graph, so the
        step survives and zero supply shows in the log as n = 0.

        `track` is accepted and deliberately ignored: callers still pass it,
        but the diagnostics below are computed on every call. See the comment
        at the computation for why skipping them was wrong.
        """
        t = target.reshape(-1).long()
        ok = (t >= 0) & (t < self.n_cat)
        if text_used is not None:
            # 1.0 = the description arrived, so the row is EXCLUDED.
            ok = ok & (text_used.reshape(-1) <= 0.5)
        n = int(ok.sum())
        # Diagnostics are computed always, not under `track`: `evaluate` calls
        # this with track=False, and returning the stale `self._diag` there put
        # a leftover from the last TRAINING step into the validation metrics.
        if n == 0:
            self._diag = {"cat_n": 0.0, "cat_acc": float("nan"),
                          "cat_floor": float("nan"),
                          "cat_gain": float("nan"), "cat_loss": 0.0}
            return logits.sum() * 0.0, dict(self._diag)
        lg = logits[ok]
        tt = t[ok]
        out = torch.nn.functional.cross_entropy(lg, tt)
        with torch.no_grad():
            acc = float((lg.argmax(dim=-1) == tt).float().mean())
            # Floor: the most frequent class IN THIS SAMPLE, without which
            # 0.45 looks like skill when it may be the class distribution.
            cnt = torch.bincount(tt, minlength=self.n_cat).float()
            floor = float(cnt.max() / cnt.sum())
        self._diag = {"cat_n": float(n), "cat_acc": acc,
                      "cat_floor": floor,
                      "cat_gain": acc - floor,
                      "cat_loss": float(out.detach())}
        return out, dict(self._diag)

    def report(self) -> Dict[str, float]:
        return dict(self._diag)
