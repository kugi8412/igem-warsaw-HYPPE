"""Decoder: sequence logits from QUANTISED CODES ONLY.

The previous track's decoder also received a continuous side channel, and its
headline reconstruction of 0.9984 turned out to be a property of that channel:
from the codes alone it was 0.29, against a composition floor of 0.2934. At
generation time no such channel exists, so this decoder takes exactly one
argument -- the list of quantised level tensors, with no `x`, no encoder
feature, no occupancy and no positional ramp. The signature is the guarantee;
`assert_batch_independence` and `assert_no_batch_statistics` keep it honest.
The position-relative-to-ATG anchor belongs to the ENCODER, folded into the
codes rather than handed to the decoder a second time.

Normalisation is GroupNorm, never BatchNorm: `LatentMask` glues six mask views
along the batch axis in one call, and running statistics over that mixture were
measured to move 12 % of predicted bases between `train` and `eval`. Upsampling
is a learned shuffle with a linear remainder, never `nearest`, which is exact
only at integer ratios -- and every variant-B step is an integer factor 2,
which is exactly why that trap would be invisible here.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Measured constants of the Trichoderma promoter set; `masking.py` imports them.

#: Accuracy of always emitting the most frequent base: not 0.25, which is the
#: uniform prior, and not the DeepSTARR composition floor of 0.2934.
RECONSTRUCTION_FLOOR = 0.2626

#: Measured unigram entropy of the promoter set, bits per base pair.
UNIGRAM_ENTROPY_BITS_PER_BP = 1.9983

#: Maximum entropy of a distribution over four bases.
MAX_ENTROPY_BITS_PER_BP = 2.0


def sequence_bits(length: int) -> float:
    """Information content of `length` bp at the measured unigram entropy.

    800 bp -> 1598.7 bit, the denominator of every latent-capacity ratio here.
    """
    return float(length) * UNIGRAM_ENTROPY_BITS_PER_BP


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= `max_groups` that divides."""
    groups = 1
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, channels)


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS, [B, 1, 1]. Per SAMPLE is a necessary condition:
    a batch-wide statistic would make the prediction a function of batch
    composition rather than of the sequence."""
    return x.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12).sqrt()


class _Upsample1d(nn.Module):
    """Integer-factor learned interleave, then an exact linear remainder.

    The same arithmetic as `ConvTranspose1d(c_in, c_out, factor,
    stride=factor)`, written out because the interleave hides phase bugs.
    """

    def __init__(self, c_in: int, c_out: int, in_len: int, out_len: int):
        super().__init__()
        if out_len < in_len:
            raise ValueError(
                "_Upsample1d only raises resolution: %d -> %d. A decoder that "
                "downsamples a code map is throwing away the only thing it is "
                "given." % (in_len, out_len))
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.factor = max(1, int(out_len) // int(in_len))
        self.c_out = int(c_out)
        self.proj = nn.Conv1d(c_in, c_out * self.factor, 1)
        #: True when the factor does not divide exactly. Reported: a config
        #: that silently interpolates no longer lines up with the ladder.
        self.needs_resample = (self.factor * self.in_len) != self.out_len

    def extra_repr(self) -> str:
        return ("in_len=%d, out_len=%d, factor=%d, remainder=%s"
                % (self.in_len, self.out_len, self.factor,
                   self.needs_resample))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        if L != self.in_len:
            raise RuntimeError(
                "_Upsample1d built for length %d got %d. Silent resizing is "
                "how a decoder stops matching the ladder it decodes."
                % (self.in_len, L))
        v = self.proj(x).view(B, self.c_out, self.factor, L)
        # Index (t * factor + m) holds v[c, m, t]: the PixelShuffle /
        # ConvTranspose(kernel == stride) convention.
        y = v.permute(0, 1, 3, 2).reshape(B, self.c_out, L * self.factor)
        if self.needs_resample:
            y = F.interpolate(y, size=self.out_len, mode="linear",
                              align_corners=False)
        return y


class _Merge(nn.Module):
    """`conv_coarse(h) + conv_level(z_k) + b`.

    Arithmetically identical to one `Conv1d` over `cat([h, z_k], 1)`, at the
    same parameter and FLOP count, and split in two so that the per-level
    contribution is an exact and free observable.
    """

    def __init__(self, c_coarse: int, c_level: int, c_out: int, kernel: int):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError("merge kernel must be odd, got %d" % (kernel,))
        pad = (kernel - 1) // 2
        self.from_coarse = nn.Conv1d(c_coarse, c_out, kernel, padding=pad,
                                     bias=True)
        self.from_level = nn.Conv1d(c_level, c_out, kernel, padding=pad,
                                    bias=False)
        self.norm = _group_norm(c_out)
        self.act = nn.SiLU()
        self._share = 0.0

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        a = self.from_coarse(h)
        b = self.from_level(z)
        with torch.no_grad():
            self._share = float(
                _rms(b).mean() / _rms(a).mean().clamp_min(1e-9))
        return self.act(self.norm(a + b))

    def level_share(self) -> float:
        """Contribution of this level relative to the coarser stream.

        Drifting to 0 means the level is dead and its bits are spent unused.
        """
        return self._share


class GlobalCodeMixing(nn.Module):
    """Content mixing across code positions, inside the decoder.

    Measured accuracy inside a masked 34 bp window, by distance from its edge,
    was 0.9583 / 0.8438 / 0.5000 / 0.3750 / ~0.33: the decoder writes the
    MARGINAL in the middle, because convolution is local and normalisation
    carries only "how much in total". Without this module the `column` and
    `degree_0` views of `LatentMask` are decoration. The bias table is indexed
    by `|i - j|`, a learned decay profile; an absolute position embedding would
    let the decoder key on the index instead of the codes. The addition is
    damped against the per-sample RMS of the map, and the damper is a CEILING
    -- renormalising to `share` would pin the post-damp ratio to the knob and
    make the module scale-invariant in `out.weight`, which weight decay then
    shrinks unopposed.
    """

    def __init__(self, channels: int, length: int, heads: int = 4,
                 head_dim: int = 16, share: float = 0.3):
        super().__init__()
        if share <= 0.0:
            raise ValueError(
                "GlobalCodeMixing with share <= 0 is a module that is built, "
                "costs parameters and does nothing -- the exact silent-no-op "
                "pattern this project keeps paying for. Do not construct it.")
        self.channels = int(channels)
        self.length = int(length)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.share = float(share)
        inner = self.heads * self.head_dim
        self.q = nn.Conv1d(self.channels, inner, 1)
        self.k = nn.Conv1d(self.channels, inner, 1)
        self.v = nn.Conv1d(self.channels, inner, 1)
        self.out = nn.Conv1d(inner, self.channels, 1)
        # Small but non-zero, and no gate: a zero output weight passes zero
        # gradient up to q/k/v, freezing the attention that feeds it.
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)
        self.distance_bias = nn.Parameter(torch.zeros(self.length))
        pos = torch.arange(self.length)
        self.register_buffer("_sep", (pos.view(-1, 1) - pos.view(1, -1)).abs(),
                             persistent=False)
        self._report: Dict[str, float] = {}

    def extra_repr(self) -> str:
        return ("channels=%d, length=%d, heads=%d, head_dim=%d, share=%.3f"
                % (self.channels, self.length, self.heads, self.head_dim,
                   self.share))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if C != self.channels or L != self.length:
            raise RuntimeError(
                "GlobalCodeMixing built for [%d, %d] got [%d, %d]; the "
                "distance-bias table is indexed by length, so a mismatch is a "
                "silent indexing bug, not a resize."
                % (self.channels, self.length, C, L))
        q = self.q(x).view(B, self.heads, self.head_dim, L)
        k = self.k(x).view(B, self.heads, self.head_dim, L)
        v = self.v(x).view(B, self.heads, self.head_dim, L)
        logits = torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.distance_bias[self._sep].unsqueeze(0).unsqueeze(0)
        w = torch.softmax(logits, dim=-1)
        mixed = torch.einsum("bhij,bhdj->bhdi", w, v).reshape(B, -1, L)
        delta = self.out(mixed)

        scale_x = _rms(x)
        scale_d = _rms(delta)
        # Ceiling, not setpoint: shrink only the samples already above it.
        factor = (self.share * scale_x / scale_d).clamp(max=1.0)
        with torch.no_grad():
            entropy = -(w.clamp_min(1e-9).log2() * w).sum(-1).mean()
            span = (w * self._sep.unsqueeze(0).unsqueeze(0).to(w.dtype)).sum(-1).mean()
            # Fraction of the addition NOT shared across the batch: the
            # previous track measured 0.0869, i.e. a near-constant modulation.
            common = delta.mean(dim=0, keepdim=True)
            dev = (delta - common).pow(2).mean().sqrt()
            whole = delta.pow(2).mean().clamp_min(1e-12).sqrt()
            self._report = {
                "mix_H": round(float(entropy), 3),
                "mix_H_max": round(math.log2(max(L, 2)), 3),
                "mix_span": round(float(span), 2),
                "mix_raw": round(float((scale_d / scale_x).mean()), 4),
                "mix_clip": round(float((factor < 1.0).to(x.dtype).mean()), 4),
                "mix_dev": round(float(dev / whole), 4),
            }
        return x + delta * factor

    def report(self) -> Dict[str, float]:
        """Falsifiers: `mix_H` at `mix_H_max` is uniform attention, routing no
        content; `mix_dev` near 0 means the addition is the same vector for
        every sequence. `mix_raw` is undamped, `mix_clip` how often the ceiling
        binds."""
        return dict(self._report)


class CodeDecoder(nn.Module):
    """Codes -> base logits. The only input is the list of level tensors.

    `level_dims` and `level_lengths` are per level, coarsest first, and come
    from the config through the ladder. `global_mixing` puts attention over
    code positions on every level no longer than `mixing_max_len`; the longer
    maps are excluded, a masked window there being already reconstructible.
    """

    def __init__(self,
                 level_dims: Sequence[int],
                 level_lengths: Sequence[int],
                 seq_len: int,
                 width: int = 128,
                 kernel: int = 3,
                 n_bases: int = 4,
                 global_mixing: bool = True,
                 mixing_max_len: int = 128,
                 mixing_heads: int = 4,
                 mixing_head_dim: int = 16,
                 mixing_share: float = 0.3):
        super().__init__()
        self.level_dims = [int(c) for c in level_dims]
        self.level_lengths = [int(n) for n in level_lengths]
        self.seq_len = int(seq_len)
        self.width = int(width)
        self.n_bases = int(n_bases)
        if len(self.level_dims) != len(self.level_lengths):
            raise ValueError("level_dims and level_lengths disagree: %r vs %r"
                             % (self.level_dims, self.level_lengths))
        if not self.level_dims:
            raise ValueError("CodeDecoder needs at least one level")
        if kernel % 2 != 1:
            raise ValueError("kernel must be odd, got %d" % (kernel,))
        for a, b in zip(self.level_lengths, self.level_lengths[1:]):
            if b < a:
                raise ValueError(
                    "level_lengths must be COARSEST FIRST and non-decreasing, "
                    "got %r. The ladder contract is level 0 = coarsest map "
                    "(25 positions of 32 bp for variant B); reversing it "
                    "would reconstruct from the finest level and make every "
                    "prefix view meaningless." % (self.level_lengths,))
        if self.level_lengths[-1] > self.seq_len:
            raise ValueError("finest level (%d) is longer than the sequence "
                             "(%d)" % (self.level_lengths[-1], self.seq_len))
        pad = (kernel - 1) // 2

        self.stem = nn.Sequential(
            nn.Conv1d(self.level_dims[0], self.width, kernel, padding=pad),
            _group_norm(self.width), nn.SiLU())

        ups: List[nn.Module] = []
        merges: List[nn.Module] = []
        for k in range(1, len(self.level_lengths)):
            ups.append(_Upsample1d(self.width, self.width,
                                   self.level_lengths[k - 1],
                                   self.level_lengths[k]))
            merges.append(_Merge(self.width, self.level_dims[k], self.width,
                                 kernel))
        self.ups = nn.ModuleList(ups)
        self.merges = nn.ModuleList(merges)

        self.to_sequence = _Upsample1d(self.width, self.width,
                                       self.level_lengths[-1], self.seq_len)
        self.head_pre = nn.Sequential(
            nn.Conv1d(self.width, self.width, kernel, padding=pad),
            _group_norm(self.width), nn.SiLU())
        self.head = nn.Conv1d(self.width, self.n_bases, kernel, padding=pad)

        # Keyed by level index: a sparse dict is the honest representation,
        # since a level either has the module or does not.
        self.mixing = nn.ModuleDict()
        if global_mixing:
            for k, length in enumerate(self.level_lengths):
                if length <= int(mixing_max_len):
                    self.mixing[str(k)] = GlobalCodeMixing(
                        self.width, length, heads=mixing_heads,
                        head_dim=mixing_head_dim, share=mixing_share)
            if not self.mixing:
                raise ValueError(
                    "global_mixing is on but no level is short enough "
                    "(mixing_max_len=%d, lengths=%r) -- the flag would be set "
                    "and inert, which is the silent-no-op pattern this "
                    "project keeps paying for. Raise mixing_max_len or turn "
                    "the flag off explicitly."
                    % (int(mixing_max_len), self.level_lengths))

        self._report: Dict[str, float] = {}
        # One BatchNorm anywhere below makes `train` and `eval` two different
        # functions, so diffusion would not run the decoder that was trained.
        self.assert_no_batch_statistics()

    def extra_repr(self) -> str:
        return ("levels=%r, lengths=%r, seq_len=%d, width=%d, mixing=%r"
                % (self.level_dims, self.level_lengths, self.seq_len,
                   self.width, sorted(int(k) for k in self.mixing)))

    def assert_no_batch_statistics(self) -> None:
        """Raise if any submodule keeps running statistics over the batch."""
        bad = [name for name, m in self.named_modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                 nn.BatchNorm3d, nn.SyncBatchNorm,
                                 nn.InstanceNorm1d))
               and getattr(m, "track_running_stats", False)]
        if bad:
            raise RuntimeError(
                "batch-coupled normalisation in the decoder: %r. Measured "
                "cost of exactly this: running_mean diverged 39.10 %%, eval "
                "logits 27.41 %% of scale, argmax agreement 0.8791 -- 12 %% of "
                "predicted bases changed. Use GroupNorm." % (bad,))

    def parameter_budget(self) -> Dict[str, int]:
        """Parameters per part, because "the decoder is small" has to be a
        number: a decoder that quietly becomes a third of the model is a
        different experiment."""
        def count(mod: nn.Module) -> int:
            return int(sum(p.numel() for p in mod.parameters()))
        out = {
            "stem": count(self.stem),
            "ups": count(self.ups) + count(self.to_sequence),
            "merges": count(self.merges),
            "head": count(self.head_pre) + count(self.head),
            "mixing": count(self.mixing),
        }
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        return out

    def information_budget(self) -> Dict[str, float]:
        """Sequence content against the output resolution, so that no caller
        hard-codes 1598.7."""
        return {
            "seq_bits": round(sequence_bits(self.seq_len), 1),
            "unigram_H": UNIGRAM_ENTROPY_BITS_PER_BP,
            "recon_floor": RECONSTRUCTION_FLOOR,
        }

    def _check(self, z_levels: Sequence[torch.Tensor]) -> None:
        if len(z_levels) != len(self.level_dims):
            raise RuntimeError(
                "CodeDecoder built for %d levels, got %d. A decoder silently "
                "dropping a level would look like a working model whose bits "
                "are unused." % (len(self.level_dims), len(z_levels)))
        batch = z_levels[0].shape[0]
        for k, z in enumerate(z_levels):
            if z.dim() != 3:
                raise RuntimeError("level %d is not [B, C, L]: %r"
                                   % (k, tuple(z.shape)))
            if z.shape[0] != batch:
                raise RuntimeError(
                    "level %d has batch %d, level 0 has %d. With masking the "
                    "batch is K*B for EVERY level; a mismatch means the views "
                    "were built per level instead of per stack."
                    % (k, z.shape[0], batch))
            if z.shape[1] != self.level_dims[k]:
                raise RuntimeError("level %d: %d channels, built for %d"
                                   % (k, z.shape[1], self.level_dims[k]))
            if z.shape[2] != self.level_lengths[k]:
                raise RuntimeError("level %d: %d positions, built for %d"
                                   % (k, z.shape[2], self.level_lengths[k]))

    def forward(self, z_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        """`z_levels` coarsest first -> logits [B, n_bases, seq_len]."""
        self._check(z_levels)
        h = self.stem(z_levels[0])
        if "0" in self.mixing:
            h = self.mixing["0"](h)
        for k in range(1, len(z_levels)):
            h = self.ups[k - 1](h)
            h = self.merges[k - 1](h, z_levels[k])
            key = str(k)
            if key in self.mixing:
                h = self.mixing[key](h)
        h = self.to_sequence(h)
        h = self.head_pre(h)
        logits = self.head(h)

        with torch.no_grad():
            rep: Dict[str, float] = {}
            for k, merge in enumerate(self.merges, start=1):
                rep["dec_level_%d" % k] = round(merge.level_share(), 4)
            probs = torch.softmax(logits.float(), dim=1)
            entropy = -(probs.clamp_min(1e-9).log2() * probs).sum(1).mean()
            rep["dec_H"] = round(float(entropy), 4)
            rep["dec_H_unigram"] = UNIGRAM_ENTROPY_BITS_PER_BP
            rep["dec_conf"] = round(float(probs.amax(dim=1).mean()), 4)
            rep["dec_floor"] = RECONSTRUCTION_FLOOR
            for key, mod in self.mixing.items():
                for name, value in mod.report().items():
                    rep["%s_%s" % (name, key)] = value
            self._report = rep
        return logits

    def report(self) -> Dict[str, float]:
        """Observables from the last forward pass.

        `dec_level_k` is level `k`'s contribution at its merge point, and 0
        means the level is dead; `dec_H` at the unigram entropy of 1.9983 on
        the FULL view means the decoder emits the MARGINAL.
        """
        return dict(self._report)

    @torch.no_grad()
    def level_ablation(self, z_levels: Sequence[torch.Tensor],
                       target_idx: Optional[torch.Tensor] = None
                       ) -> Dict[str, float]:
        """Per level: zero it, and separately SHUFFLE it across the batch.

        Both, because this project has already published a wrong conclusion
        from one. Zeroing measures the HEAD that reads the channel; shuffling
        keeps the distribution and destroys only the correspondence to this
        sequence, so a level whose shuffle delta is ~0 is decoration however
        large its zero delta.
        """
        was_training = self.training
        self.eval()
        try:
            base = self(z_levels)
            scale = base.pow(2).mean().clamp_min(1e-12).sqrt()
            out: Dict[str, float] = {}
            if target_idx is not None:
                out["abl_base_acc"] = round(
                    float((base.argmax(1) == target_idx).float().mean()), 4)
            batch = z_levels[0].shape[0]
            # Own generator: a probe that consumed the global stream would
            # change the dropout and augmentation of the run that called it.
            generator = torch.Generator(device=z_levels[0].device)
            generator.manual_seed(20260911)
            for k in range(len(z_levels)):
                # A fresh permutation per level, drawn from the same seeded
                # generator so the probe stays reproducible. One shared
                # pairing would give every level the same draw, so a pairing
                # that happens to be benign (or adversarial) would bias all
                # of them the same way.
                perm = torch.randperm(batch, device=z_levels[0].device,
                                      generator=generator)
                # Rows the permutation leaves in place are not shuffled at
                # all and would dilute the delta towards zero, which reads as
                # "this level is decoration". Roll them onto a neighbour.
                fixed = perm == torch.arange(batch, device=perm.device)
                if bool(fixed.any()) and batch > 1:
                    perm = perm.clone()
                    idx = torch.nonzero(fixed, as_tuple=False).flatten()
                    perm[idx] = perm[(idx + 1) % batch]
                for mode in ("zero", "shuffle"):
                    mod = list(z_levels)
                    mod[k] = (torch.zeros_like(z_levels[k]) if mode == "zero"
                              else z_levels[k][perm])
                    alt = self(mod)
                    out["abl_%s_%d" % (mode, k)] = round(
                        float((alt - base).pow(2).mean().sqrt() / scale), 4)
                    if target_idx is not None:
                        out["abl_%s_acc_%d" % (mode, k)] = round(
                            float((alt.argmax(1) == target_idx).float().mean()),
                            4)
        finally:
            self.train(was_training)
        return out

    @torch.no_grad()
    def assert_batch_independence(self, z_levels: Sequence[torch.Tensor],
                                  tol: Optional[float] = None) -> float:
        """Same sample, different batch -> identical logits. Returns max|d|.

        Run this in the mode the training loop uses: in `eval` a BatchNorm
        would pass, since running statistics are batch-independent. The real
        test is a CONTENT control, not a threshold -- exactly 0.0 is
        unreachable in float32, because comparing batch B against B//2 makes
        the library pick a different kernel and reduction order. Coupling,
        unlike rounding, depends on what the other samples hold.
        """
        batch = z_levels[0].shape[0]
        if batch < 2:
            raise ValueError("need at least 2 samples to test batch coupling")
        half = max(1, batch // 2)
        full = self(z_levels)
        part = self([z[:half] for z in z_levels])
        delta = float((full[:half] - part).abs().max())

        # Measured: delta unchanged / x100 / second half zeroed all give
        # 4.172e-07, digit for digit.
        probe = [z.clone() for z in z_levels]
        for z in probe:
            z[half:] = z[half:] * 100.0
        delta_probe = float(
            (self(probe)[:half] - self([z[:half] for z in probe])).abs().max())
        # Compared against the same derived tolerance as the shape test
        # below, not against exact equality. An exact-zero threshold is the
        # brittleness this routine was written to avoid: it holds here only
        # because both halves go through identical kernels, and would fire
        # on any backend where scaling the unused half perturbs a fused
        # reduction.
        probe_tol = (tol if tol is not None else
                     8.0 * float(torch.finfo(full.dtype).eps)
                     * max(1.0, float(full.abs().max())))
        if abs(delta_probe - delta) > probe_tol:
            raise RuntimeError(
                "decoder output depends on batch CONTENT: max|delta| moved "
                "%.3e -> %.3e when the other half of the batch was scaled by "
                "100, which is more than the %.3e the dtype allows. This is "
                "a coupling, not rounding: rounding does not depend on what "
                "the other samples hold."
                % (delta, delta_probe, probe_tol))

        # `tol=None` derives the threshold from the output scale and the
        # dtype's eps; see the docstring.
        if tol is None:
            eps = float(torch.finfo(full.dtype).eps)
            tol = 8.0 * eps * max(1.0, float(full.abs().max()))
        if delta > tol:
            raise RuntimeError(
                "decoder output depends on batch composition: max|delta| = "
                "%.3e (tolerance %.3e). Six mask views are glued along the "
                "batch axis, so this makes every view see the other five. "
                "The content probe PASSED, so if this fires the coupling is "
                "content-independent -- look at shape-dependent kernels "
                "before looking at the model."
                % (delta, tol))
        return delta
