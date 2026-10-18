"""Masking the codes in the latent, so the decoder trains on partial views.

The mask width is measured, not inherited from BERT. Out of sample a code
position is linearly recoverable from four code positions on each side at R^2
0.95 / 0.83 / 0.0965 for gaps of 2 / 18 / 34 bp, and at the control level from
50 bp up: narrower than ~34 bp the task is trivial and there is no gradient,
wider it is impossible and the gradient pushes the decoder towards the
marginal distribution. The width is stated in base pairs, the unit it was
measured in and the one that survives a change of strides. The same measure on
the sequence itself gives R^2 0.0014-0.0136, so the latent does not abstract,
it smears the sequence across the neighbourhood.

Masking a single level is empty -- each level is recovered from the other two
at R^2 0.82-0.95 -- so a mask must be columnar, a window of positions through
ALL levels, or prefix, dropping the tail of levels. Prefixes pass through the
decoder as real views, because the decoder is non-linear and
``sum_k g_k(prefix_k) != g(z)``.

Only the decoder's input is masked. The trunk and the heads see `z` untouched,
so the prediction path has the same input distribution in training and in
eval, and the encoder still gets gradient through the unmasked positions.

All views go through one decoder call, glued along the batch axis, which keeps
them comparable and keeps any batch-coupled normalisation to one update per
step. A view is therefore recovered with ``.chunk(n_views, dim=0)[i]`` and
never with ``[i]``; indexing the batch axis once reported ``max|delta| =
6.405`` where the truth was 0.0.

The views are one Nash player, with weights asserted to sum to 1.0 rather than
normalised silently. Views of one decoder on one sequence are collinear almost
by definition, and five collinear players instead of one drop reconstruction's
effective share from 0.667 to 0.386 while `n_clip` stays at 2.

A masked 34 bp window cannot be reconstructed "correctly", since a regulatory
sequence is not a function of its context. The falsifier is therefore that
`rec_column` EXCEEDS the composition floor 0.2626, not any high number.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import RECONSTRUCTION_FLOOR, sequence_bits

#: The optimiser builder must exempt these parameters from weight decay;
#: lambda 0.0762 over 34 epochs is a factor 1.8e-05, so the parameter
#: silently collapses to zero. Substring matching on names is a fragile
#: contract, so `decay_exempt_parameters()` exposes the list directly.
DECAY_EXEMPT_SUBSTRING = "mask"


class MaskViews(NamedTuple):
    """Return value of `LatentMask.views`.

    `views` is one tensor per level, each `[n_views * B, C_k, L_k]`, in the
    order of `names`. A list and not one glued tensor because the levels have
    different lengths, and putting them on a common grid would resample a
    MASKED latent, smearing the mask across positions. `weights` maps name to
    weight and sums to 1.0. `names` gives the view order, with `full` first by
    construction, which `full_view` and the logged `recon` both depend on.
    `masks` is `[B, 1, seq_len]` per view in base pairs, or None: the mask the
    loss uses, exact, so nothing is interpolated between the two grids.
    """

    views: List[torch.Tensor]
    weights: Dict[str, float]
    names: Tuple[str, ...]
    masks: List[Optional[torch.Tensor]]


class LatentMask(nn.Module):
    """Views of the decoder's input: full, column, degree_0, prefix_k.

    Parameters
    ----------
    level_dims, level_lengths
        Channels and positions per level, coarsest first.
    seq_len
        Sequence length in bp. Nothing is hard-coded to 800.
    level_bits
        Bits carried by one position of each level. Required, with no
        default: the observable ceiling computed from it is what
        `rec_degree_0` is read against, and a default would lose it silently.
    width_bp
        Mask block width in base pairs. 34 is the measured bottom of the leak
        cliff; wider makes the task impossible, narrower makes it trivial.
    n_blocks
        Number of disjoint blocks, and the free knob of the two: 3 blocks of
        34 bp over 800 bp masks 12.75 % of the sequence.
    """

    def __init__(self,
                 level_dims: Sequence[int],
                 level_lengths: Sequence[int],
                 seq_len: int,
                 level_bits: Sequence[float],
                 width_bp: int = 34,
                 n_blocks: int = 3,
                 weights: Optional[Dict[str, float]] = None,
                 prefixes: bool = True,
                 degree_0: bool = True,
                 seed: int = 7898):
        super().__init__()
        self.level_dims = [int(c) for c in level_dims]
        self.level_lengths = [int(n) for n in level_lengths]
        self.level_bits = [float(b) for b in level_bits]
        self.seq_len = int(seq_len)
        self.width_bp = int(width_bp)
        self.n_blocks = int(n_blocks)
        self.n_levels = len(self.level_dims)
        self.seed = int(seed)
        if not (self.n_levels == len(self.level_lengths) == len(self.level_bits)):
            raise ValueError(
                "level_dims / level_lengths / level_bits disagree: %d / %d / %d"
                % (self.n_levels, len(self.level_lengths), len(self.level_bits)))
        if self.n_levels < 1:
            raise ValueError("LatentMask needs at least one level")
        if self.width_bp < 1:
            raise ValueError("width_bp must be >= 1")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if self.width_bp * self.n_blocks >= self.seq_len:
            raise ValueError(
                "%d blocks of %d bp cover the whole %d bp sequence -- the "
                "column view would have no visible context at all and the "
                "task becomes 'sample from the prior'."
                % (self.n_blocks, self.width_bp, self.seq_len))
        if degree_0 and self.n_levels < 2:
            raise ValueError(
                "degree_0 with a single level masks nothing (there is no "
                "level above L0), so the view would be identical to `full` "
                "and would still be counted in the loss with its weight -- "
                "cost without effect, invisible in the log. Same failure "
                "family as an `intersect` mask with no position mask.")
        self.use_prefixes = bool(prefixes) and self.n_levels > 1
        self.use_degree_0 = bool(degree_0)

        # View names. `full` first is load-bearing.
        names: List[str] = ["full", "column"]
        if self.use_degree_0:
            names.append("degree_0")
        if self.use_prefixes:
            # prefix_k keeps levels 0..k and masks the tail over the whole
            # length. The last prefix would be the full view, so it is absent.
            names += ["prefix_%d" % k for k in range(self.n_levels - 1)]
        self.names: Tuple[str, ...] = tuple(names)

        self.weights = self._build_weights(weights)

        # The MASK token, one vector per level. Zero initialisation is safe
        # here, unlike for a gate: a gate MULTIPLIES, so at zero it kills the
        # gradient to its own source, whereas this vector is SUBSTITUTED for a
        # code and its gradient is proportional to the mask rather than to its
        # own value. `mask_vec_norm_k` is the falsifier: a norm stuck at 0
        # means the parameter is frozen.
        self.mask_vectors = nn.ParameterList(
            [nn.Parameter(torch.zeros(c)) for c in self.level_dims])
        # Assert the rule that actually protects these from weight decay,
        # which is `p.ndim <= 1` in the optimiser builder -- no live exemption
        # list contains "mask". Adding it to the substring list would quietly
        # exempt every future module with "mask" in its name, and substring
        # exemptions are a recorded failure of this project.
        for name, _p in self.named_parameters():
            if _p.ndim > 1:
                raise RuntimeError(
                    "mask vector %r is %d-D: the ONLY thing protecting it "
                    "from weight decay is the `p.ndim <= 1` rule, "
                    "because no live exemption list contains %r. "
                    "Either keep it 1-D, or add it EXPLICITLY to "
                    "training.no_decay_patterns."
                    % (name, _p.ndim, DECAY_EXEMPT_SUBSTRING))

        # Own random stream: drawing the mask must not consume the global RNG,
        # or a run with masking would also draw different dropout and
        # augmentation at the same seed and the comparison would measure both.
        # Measured on bit-identical weights, forward only: sharing the stream
        # gave a 22.8 % difference in eval logits and 0.9468 argmax agreement,
        # all of it from encoder dropout drawing differently. The counter is
        # persistent, so a resumed run continues the same sequence of masks.
        self.register_buffer("_draws", torch.zeros((), dtype=torch.long),
                             persistent=True)

        self._check_level_coverage()
        self._report: Dict[str, float] = {}

    # Construction helpers

    def _build_weights(self, override: Optional[Dict[str, float]]
                       ) -> Dict[str, float]:
        """Default pressure split, then the override, then assert it sums to 1.

        `full` carries the largest share because it guards the blocking
        condition of the whole run: reconstruction from codes alone must not
        fall. Prefixes get more weight on the shallower one, the measured
        broken end -- L0 alone decoded to 0.2775, below its own composition
        floor of 0.2934, where L0+L1 gave 0.3385. Flat weights would put two
        thirds of the pressure on views that are already fine.
        """
        alloc: Dict[str, float] = {"full": 0.40, "column": 0.20}
        if self.use_degree_0:
            alloc["degree_0"] = 0.10
        else:
            alloc["column"] += 0.10
        n_prefix = self.n_levels - 1 if self.use_prefixes else 0
        if n_prefix > 0:
            raw = [1.0 / (i + 1) for i in range(n_prefix)]
            total = sum(raw)
            share = [0.30 * r / total for r in raw]
            # Last one as the remainder, so the block sums to exactly 0.30
            # rather than 0.30 plus float drift.
            share[-1] = 0.30 - sum(share[:-1])
            for i in range(n_prefix):
                alloc["prefix_%d" % i] = share[i]
        else:
            alloc["full"] += 0.30

        out = dict(alloc)
        if override:
            out.update({str(k): float(v) for k, v in dict(override).items()})
        unknown = [k for k in out if k not in self.names]
        if unknown:
            raise ValueError("unknown view names in `weights`: %r "
                             "(available: %r)" % (unknown, self.names))
        missing = [n for n in self.names if n not in out]
        if missing:
            raise ValueError("no weight for views %r" % (missing,))
        total = sum(out[n] for n in self.names)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "view weights must sum to 1.0, they sum to %.6f (%r). A sum "
                "other than 1 changes the STRENGTH of the reconstruction "
                "player in the Nash bargain, and the log cannot tell that "
                "apart from anything else." % (total, out))
        return {n: float(out[n]) for n in self.names}

    def _check_level_coverage(self) -> None:
        """Refuse a configuration in which a coarse level is fully masked.

        A level whose every cell is masked in the column view becomes the
        constant MASK vector everywhere, making that view identical to a
        prefix view: two players, one task, and nothing in the log says so.
        The bound is the worst case, since a block may straddle two cells.
        """
        for k, length in enumerate(self.level_lengths):
            per_block = -(-self.width_bp * length // self.seq_len) + 1
            worst = per_block * self.n_blocks
            if worst >= length:
                raise ValueError(
                    "level %d has %d cells of %.1f bp; %d blocks of %d bp can "
                    "mask up to %d of them, i.e. the whole level. The column "
                    "view would then equal a prefix view."
                    % (k, length, self.seq_len / float(length), self.n_blocks,
                       self.width_bp, worst))

    # Introspection

    @property
    def n_views(self) -> int:
        return len(self.names)

    def extra_repr(self) -> str:
        return ("levels=%r, lengths=%r, seq_len=%d, width=%d bp, blocks=%d, "
                "views=%r, weights=%r"
                % (self.level_dims, self.level_lengths, self.seq_len,
                   self.width_bp, self.n_blocks, self.names,
                   [round(self.weights[n], 4) for n in self.names]))

    def decay_exempt_parameters(self) -> List[nn.Parameter]:
        """Parameters the optimiser must build WITHOUT weight decay."""
        return list(self.mask_vectors)

    def full_view(self, stacked: torch.Tensor) -> torch.Tensor:
        """The `full` view out of a glued `[n_views * B, ...]` tensor.

        Every reader of the logged `recon` must come through here: the glued
        tensor would give a metric averaged over views, comparable with no run
        without masking. Note the `chunk` and not `[0]` -- the glue is along
        the batch axis.
        """
        return stacked.chunk(self.n_views, dim=0)[0]

    def observable_ceiling(self) -> Dict[str, float]:
        """How much of the sequence the coarsest level CAN carry.

        Computed in code because `rec_degree_0` is unreadable without it: that
        view keeps L0 of the window and takes the finer levels away, so its
        accuracy is bounded by what L0's bits can say about the sequence.

        The window version comes out identical to the global one, which is not
        a coincidence -- the ratio is a property of the code budget and not of
        the mask width, so it does not move when `width_bp` does. Both are
        reported so that nobody rediscovers this by changing the width.
        """
        seq = sequence_bits(self.seq_len)
        l0_bits = self.level_lengths[0] * self.level_bits[0]
        cell_bp = self.seq_len / float(self.level_lengths[0])
        window_cells = self.width_bp / cell_bp
        window_bits = window_cells * self.level_bits[0]
        total_bits = float(sum(n * b for n, b in
                               zip(self.level_lengths, self.level_bits)))
        return {
            "ceil_l0": round(l0_bits / seq, 4),
            "ceil_l0_window": round(window_bits /
                                    (self.width_bp * (seq / self.seq_len)), 4),
            "l0_bits": round(l0_bits, 1),
            "seq_bits": round(seq, 1),
            "code_bits": round(total_bits, 1),
            "budget_ratio": round(total_bits / seq, 3),
            "recon_floor": RECONSTRUCTION_FLOOR,
        }

    # Mask drawing

    def _position_mask_bp(self, batch: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
        """[B, 1, seq_len] in {0,1}, 1 = masked. Drawn per sequence.

        Blocks are contiguous and disjoint. Contiguity is necessary because
        scattered single positions are recoverable from their neighbourhood at
        R^2 0.9491. Disjointness comes from splitting the sequence into
        `n_blocks` buckets and drawing one start in each; overlapping blocks
        would make the real masked fraction smaller than the requested one.
        """
        L = self.seq_len
        mask = torch.zeros(batch, 1, L, device=device, dtype=dtype)
        width = min(self.width_bp, L)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed * 1000003 + int(self._draws))
        self._draws += 1
        edges = [round(i * L / self.n_blocks) for i in range(self.n_blocks + 1)]
        index = torch.arange(L, device=device).unsqueeze(0)
        for i in range(self.n_blocks):
            lo, hi = edges[i], edges[i + 1]
            slack = max(hi - lo - width, 0)
            if slack <= 0:
                start = torch.full((batch,), lo, device=device,
                                   dtype=torch.long)
            else:
                start = lo + torch.randint(0, slack + 1, (batch,),
                                           device=device, generator=generator)
            block = ((index >= start.unsqueeze(1))
                     & (index < (start + width).unsqueeze(1)))
            mask = torch.maximum(mask, block.to(dtype).unsqueeze(1))
        return mask

    def _to_level(self, mask_bp: torch.Tensor, length: int) -> torch.Tensor:
        """Base-pair mask -> code-cell mask, any-overlap semantics.

        A cell is masked if ANY base it covers is inside the window, not if
        all are: a partially overlapping cell left visible leaks the masked
        bases straight through, which is what the 34 bp width exists to stop.
        `adaptive_max_pool1d` implements exactly this.
        """
        if length == mask_bp.shape[-1]:
            return mask_bp
        return F.adaptive_max_pool1d(mask_bp, length)

    def _substitute(self, z: torch.Tensor, level: int, level_selected: bool,
                    pos_mask: Optional[torch.Tensor],
                    intersect: bool) -> torch.Tensor:
        """Put the MASK vector where the mask is 1.

        Composed arithmetically as `z * (1 - m) + vec * m`, not with
        `torch.where`, which does not protect the gradient -- mask the
        operand, not the result. The multiplicative form also makes the
        gradient reaching `mask_vectors[level]` proportional to `m`, so it
        flows from exactly the masked positions.

        `intersect` is the whole of `degree_0`: a level mask combined with a
        position mask by union (max) gives `column` and `prefix`, by product
        gives a LOCAL prefix -- the only view that asks what L0 knows about
        its own window.
        """
        B, C, L = z.shape
        if C != self.level_dims[level]:
            raise RuntimeError(
                "level %d got %d channels, built for %d -- the mask token is "
                "per level, so a mismatch is a silent wrong-vector bug"
                % (level, C, self.level_dims[level]))
        vec = self.mask_vectors[level].to(z.dtype).view(1, C, 1)
        base_value = 1.0 if level_selected else 0.0
        if pos_mask is None and not level_selected and not intersect:
            # Nothing is masked at this level in this view, and `z * 1 +
            # vec * 0` is the exact identity, so returning `z` gives the same
            # result and the same zero gradient without allocating four
            # full-size tensors per kept level per prefix view.
            return z
        if pos_mask is None:
            if intersect:
                # An intersection with no position mask is the EMPTY mask, so
                # the view would equal `full` and still carry its weight in
                # the loss: cost without effect, invisible in the log.
                raise ValueError(
                    "intersect=True without a position mask gives an EMPTY "
                    "mask: the view would be identical to `full` and would "
                    "still carry its weight in the loss")
            m = torch.full((1, 1, 1), base_value, device=z.device,
                           dtype=z.dtype).expand(B, C, L)
        else:
            pos = pos_mask.to(z.dtype).expand(B, C, L)
            if intersect:
                m = pos * base_value
            else:
                m = torch.clamp(pos + base_value, max=1.0)
        return z * (1.0 - m) + vec * m

    # The view stack

    def views(self, z_levels: Sequence[torch.Tensor]) -> MaskViews:
        """Build every view and glue them along the batch axis.

        One decoder call on the result. Recover a view with
        `.chunk(n_views, dim=0)[i]`, never with `[i]`.
        """
        if len(z_levels) != self.n_levels:
            raise RuntimeError("LatentMask built for %d levels, got %d"
                               % (self.n_levels, len(z_levels)))
        batch = z_levels[0].shape[0]
        for k, z in enumerate(z_levels):
            if z.shape[0] != batch:
                raise RuntimeError(
                    "level %d has batch %d, level 0 has %d" % (k, z.shape[0], batch))
            if z.shape[-1] != self.level_lengths[k]:
                raise RuntimeError("level %d has %d positions, built for %d"
                                   % (k, z.shape[-1], self.level_lengths[k]))
        device, dtype = z_levels[0].device, z_levels[0].dtype
        mask_bp = self._position_mask_bp(batch, device, dtype)
        per_level_mask = [self._to_level(mask_bp, n)
                          for n in self.level_lengths]

        per_view: List[List[torch.Tensor]] = []
        bp_masks: List[Optional[torch.Tensor]] = []
        for name in self.names:
            if name == "full":
                per_view.append([z for z in z_levels])
                bp_masks.append(None)
            elif name == "column":
                # The window through ALL levels: reconstruct from the
                # neighbourhood.
                per_view.append([
                    self._substitute(z, k, False, per_level_mask[k], False)
                    for k, z in enumerate(z_levels)])
                bp_masks.append(mask_bp)
            elif name == "degree_0":
                # Local prefix: the finer levels go only inside the window,
                # the coarse code of that window stays. `rec_degree_0 -
                # rec_column` then measures how much L0 knows about its own
                # window, i.e. whether the hierarchy is positional or merely
                # structural. Read against `observable_ceiling()["ceil_l0"]`,
                # and only once `rec_full` is above 0.99.
                per_view.append([
                    self._substitute(z, k, k > 0, per_level_mask[k], True)
                    for k, z in enumerate(z_levels)])
                bp_masks.append(mask_bp)
            elif name.startswith("prefix_"):
                keep = int(name.split("_")[1])
                per_view.append([
                    self._substitute(z, k, k > keep, None, False)
                    for k, z in enumerate(z_levels)])
                bp_masks.append(None)
            else:
                raise RuntimeError("unhandled view %r" % (name,))

        stacked = [torch.cat([view[k] for view in per_view], dim=0)
                   for k in range(self.n_levels)]

        with torch.no_grad():
            report: Dict[str, float] = {
                "mask_frac_bp": round(float(mask_bp.mean()), 4),
                "mask_width_bp": float(self.width_bp),
                "mask_blocks": float(self.n_blocks),
            }
            for k, m in enumerate(per_level_mask):
                report["mask_frac_%d" % k] = round(float(m.mean()), 4)
            for k, vec in enumerate(self.mask_vectors):
                report["mask_vec_norm_%d" % k] = round(float(vec.norm()), 4)
            report.update(self.observable_ceiling())
            self._report = report

        return MaskViews(stacked, dict(self.weights), self.names, bp_masks)

    # Loss

    def reconstruction_loss(self, logits: torch.Tensor,
                            target_idx: torch.Tensor,
                            bp_masks: Sequence[Optional[torch.Tensor]],
                            per_view: Optional[Dict[str, torch.Tensor]] = None
                            ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted cross-entropy over the views: one term, one Nash player.

        `logits` is the decoder output on the glued stack,
        `[n_views * B, 4, seq_len]`, `target_idx` is `[B, seq_len]`, and
        `bp_masks` comes from the same `views()` call. It is passed explicitly
        and never cached on the module: a report computed, stored on a module
        and wiped before anything read it has cost a main observable before.

        A masked view is scored only on the masked positions. Adding the
        visible ones would dilute the gradient with the trivial part of the
        task, and they are already covered by the `full` view.
        """
        if logits.shape[0] != self.n_views * target_idx.shape[0]:
            raise RuntimeError(
                "logits batch %d is not %d views x %d sequences. Passing the "
                "glued stack to a metric that expects one view yields an "
                "average over views, a number comparable with nothing."
                % (logits.shape[0], self.n_views, target_idx.shape[0]))
        if logits.shape[-1] != target_idx.shape[-1]:
            raise RuntimeError(
                "decoder produced %d positions, target has %d. The decoder is "
                "built to output `seq_len` exactly; resizing here would hide "
                "a ladder/decoder mismatch." % (logits.shape[-1],
                                                target_idx.shape[-1]))
        if len(bp_masks) != self.n_views:
            raise RuntimeError("got %d masks for %d views"
                               % (len(bp_masks), self.n_views))
        chunks = logits.chunk(self.n_views, dim=0)
        total: Optional[torch.Tensor] = None
        report: Dict[str, float] = {}
        for i, name in enumerate(self.names):
            weight = self.weights[name]
            mask = bp_masks[i]
            if mask is None:
                ce = F.cross_entropy(chunks[i], target_idx)
                with torch.no_grad():
                    acc = float((chunks[i].argmax(1) == target_idx)
                                .to(logits.dtype).mean())
            else:
                m = mask.squeeze(1).to(logits.dtype)
                denom = m.sum().clamp_min(1.0)
                per_pos = F.cross_entropy(chunks[i], target_idx,
                                          reduction="none")
                ce = (per_pos * m).sum() / denom
                with torch.no_grad():
                    acc = float(((chunks[i].argmax(1) == target_idx)
                                 .to(logits.dtype) * m).sum() / denom)
            total = weight * ce if total is None else total + weight * ce
            report["rec_" + name] = round(acc, 4)
            if per_view is not None:
                # This view's UNWEIGHTED cross-entropy, for the controller's
                # second objective. The dict is supplied by the caller rather
                # than kept on the module, for the reason in the docstring.
                per_view[name] = ce
        if "degree_0" in self.weights:
            # The number that view exists for. Above +0.02 the hierarchy is
            # positional: L0 carries something about its own window. Around
            # 0.00 it is structural only and the direction is closed. Below
            # 0.00 is NOT an implementation bug -- more input guarantees a
            # better output only for an optimal decoder, and this one is
            # learned, so the inequality is asymptotic and must be read in an
            # epoch where `rec_full` is already above 0.99.
            report["hier_gain"] = round(report["rec_degree_0"]
                                        - report["rec_column"], 4)
        report.update(self._report)
        assert total is not None
        return total, report

    # Falsifiers

    def report(self) -> Dict[str, float]:
        """Observables from the last `views()` call.

        `mask_frac_bp` must match `n_blocks * width_bp / seq_len`; below it
        means blocks overlapped. `mask_frac_k` is the fraction of code cells
        masked at level k, growing towards the coarse end; at 1.0 the column
        view has degenerated into a prefix view. `mask_vec_norm_k` stuck at
        0.0 means the MASK token never moved, so either it is not exempt from
        weight decay or the view is not in the loss. `ceil_*` come from
        `observable_ceiling`, and `rec_*` and `hier_gain` from
        `reconstruction_loss` -- read every `rec_*` against `recon_floor` =
        0.2626, never against 0.25.
        """
        return dict(self._report)

    @torch.no_grad()
    def check_view_glue(self, z_levels: Sequence[torch.Tensor]) -> float:
        """The `full` view, recovered by chunking, must equal the input.

        Returns max|delta|, which must be exactly 0.0. This goes through
        `views()` rather than the internals on purpose, so it also checks the
        contract of the return value -- pulling a view out with `[i]`, i.e.
        indexing the batch axis, once reported 6.405 where the truth was 0.0.
        """
        draws = int(self._draws)
        out = self.views(z_levels)
        # A diagnostic must leave the run untouched, this module's own stream
        # included, or every later mask differs between a run that called the
        # check and one that did not.
        self._draws.fill_(draws)
        worst = 0.0
        for k, stacked in enumerate(out.views):
            recovered = stacked.chunk(self.n_views, dim=0)[0]
            worst = max(worst, float((recovered - z_levels[k]).abs().max()))
        if worst != 0.0:
            raise RuntimeError(
                "the `full` view is not the input: max|delta| = %.3e. Either "
                "`full` is no longer first, or the views are glued along the "
                "wrong axis." % (worst,))
        return worst

    @torch.no_grad()
    def check_view_distinctness(self, z_levels: Sequence[torch.Tensor]
                                ) -> Dict[str, float]:
        """Every view must differ from `full`, and no two views may coincide.

        A view identical to another is a second player on the same task: it
        costs its weight, changes nothing, and nothing in the log says so.
        This is the positive control the other checks need, since
        `check_view_glue` passes trivially on a module whose views are all
        the identity and this one does not.

        Returns the relative RMS distance from `full` per view, summed over
        levels. `full` is 0.0 by definition; every other entry is strictly
        positive once the MASK token has moved off zero.
        """
        draws = int(self._draws)
        out = self.views(z_levels)
        self._draws.fill_(draws)
        scale = max(float(sum(z.pow(2).mean() for z in z_levels)), 1e-12) ** 0.5
        result: Dict[str, float] = {}
        chunked = [s.chunk(self.n_views, dim=0) for s in out.views]
        for i, name in enumerate(self.names):
            diff = 0.0
            for k in range(self.n_levels):
                diff += float((chunked[k][i] - z_levels[k]).pow(2).mean())
            result["dist_" + name] = round((diff ** 0.5) / scale, 6)
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                j = self.names.index(b)
                same = all(
                    bool(torch.equal(chunked[k][i], chunked[k][j]))
                    for k in range(self.n_levels))
                if same:
                    raise RuntimeError(
                        "views %r and %r are identical -- two Nash-weighted "
                        "copies of one task" % (a, b))
        return result
