"""A prior over the ladder codes, and the schedule that unmasks them.

The encoder/ladder/decoder stack answers "which codes does THIS promoter
have"; this module answers "which code configurations are promoters at all",
which is what de-novo generation needs:

    sequence --encoder--> maps --ladder--> INDICES --decoder--> sequence
                                             ^
                                      p(indices | C), here

The prior never touches `CodeDecoder`, which keeps invariant U1 ("the decoder
sees codes only") true: the conditioning vector C -- species, experiment text,
flanks -- enters the prior, so a generated code list is still decoded by a
function of the indices alone. Putting C into the decoder would buy the same
conditioning and break `decode_indices` silently, the sequences still looking
like sequences.

Declared alphabet at `sequence_length = 800`, coarsest first:

    level  length  alphabet  bits/position       bits
      L0       25       128        7.0000      175.00   hyperbolic codebook
      L1       50       729        9.5098      475.49   tangent FSQ  3^6
      L2      100        81        6.3399      633.99   tangent FSQ  3^4
      L3      200         9        3.1699      633.99   tangent FSQ  3^2
      L4      400         9        3.1699     1267.97   tangent FSQ  3^2
                                              --------
                                               3186.43 bits, 775 tokens

against 800 bp x 1.9983 measured unigram bits = 1598.6, a reserve of 1.99x.
`ladder_bit_budget()` recomputes it from the live modules. Read the reserve
the right way round: a latent carrying twice the sequence is a redundant
re-encoding, not an abstraction, so almost every point of code space decodes
to something that is not a promoter and the prior is the only thing that knows
which points are. Achieved bits are a smaller quantity, and a prior fitted to
codes from an untrained encoder fits a degenerate corpus (`CodeCorpus
.health()`).

Masked rather than autoregressive, because this ladder needs a coarse-to-fine
order and an autoregressive model fixes one order at training time. Partial
configurations are in distribution: the null code decodes to an exactly zero
geodesic step on all four refinement levels, so "L0 and L1 decided, L2-L4 not
yet" is a legal code list. And the design loop edits as well as creates --
"keep the measured core, resample the 200 bp upstream" is masked infilling
through the same forward pass.

Codes are strongly dependent along the sequence: a position is recoverable
from its neighbours at R^2 = 0.95 one position away and 0.10 at 34 bp. That
cliff traps the evaluation rather than the architecture -- at a 15 % mask
ratio the task is nearly solved by copying the neighbour -- so every number is
reported per regime, the one that counts being the sampler's (`EvalRegime`).

Positional encoding is absolute and sinusoidal in base pairs rather than token
index, shared across levels, which lets attention align L0 cell 3 with the
sixteen L4 cells inside it. The sibling track's finding that absolute encoding
is dead weight is about enhancers, which have no positional anchor; these are
promoters, oriented 5'->3' and ending at the start codon, mirror correlation
-0.0308 against +0.9904 for a synthetic 50/50 mix. The repo's temperature law
`tau_k ~ sinh^2(r_k)` does not apply: these logits come from a `Linear` head,
are not distances, and live on no shell.

Beating uniform is free, so the falsifier is the marginal table in two forms,
one histogram per level and one per level and position; a prior that does not
beat the per-position table on held-out data is a frequency table with a
transformer bolted on, and `PriorEvaluation.verdict` says that word. The
negative control `shuffle_control=True` rescores on codes permuted across
sequences within each (level, position), destroying every dependency while
leaving the baselines unchanged, so the advantage must collapse to ~0.

Usage:

    geom  = LadderGeometry.from_model(model)       # shapes from the FACT
    sched = MaskSchedule(unmask=UnmaskSchedule.derive(geom))
    draw  = sched.sample(geom, B, dev)             # what to hide, to score
    loss, stats = prior.loss(indices, draw.mask, context, targets=draw.target)
    idx   = generate(prior, batch=8, context=C)    # coarse to fine
    seq   = model.decode_codes(idx)                # U1 intact: codes only
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.backends.mha as _mha_backend
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LadderGeometry",
    "ladder_bit_budget",
    "format_budget",
    "MaskDraw",
    "MaskSchedule",
    "UnmaskSchedule",
    "LadderCodePrior",
    "MarginalCodeTable",
    "EvalRegime",
    "PriorEvaluation",
    "evaluate_prior",
    "CodeCorpus",
    "generate",
]

_LOG2 = math.log(2.0)


def _check_generator(generator: Optional[torch.Generator],
                     device: torch.device, where: str) -> None:
    """A CPU generator cannot seed a CUDA draw, and the error is obscure.

    Raised with the caller's name, since the alternative -- falling back on
    the global stream -- makes a run depend on every other draw in the process.
    """
    if generator is None:
        return
    if generator.device.type != torch.device(device).type:
        raise ValueError(
            "%s: generator is on %s but the draw is on %s. Seed a generator "
            "on the target device (torch.Generator(device=%r)) rather than "
            "letting the draw fall back on the global stream."
            % (where, generator.device, device, str(device)))

#: Measured independence radius of the latent, in base pairs: a masked window
#: is linearly recoverable from its neighbourhood at R^2 = 0.95 (2 bp), 0.10
#: (34 bp), -0.03 (50 bp). In bp so it survives a change of strides; the
#: sampler's step count derives from it and from nothing else.
INDEPENDENCE_BP = 34.0


# Geometry and budget
@dataclass(frozen=True)
class LadderGeometry:
    """Shapes and alphabets of one ladder, coarsest level first.

    `lengths[k]` cells on level k, `alphabets[k]` symbols each, all covering
    the same `seq_len` base pairs. Everything else here derives from these
    three numbers, so a change of strides or FSQ budget propagates.
    """

    lengths: Tuple[int, ...]
    alphabets: Tuple[int, ...]
    seq_len: int

    def __post_init__(self) -> None:
        if len(self.lengths) != len(self.alphabets):
            raise ValueError("lengths and alphabets disagree: %d vs %d"
                             % (len(self.lengths), len(self.alphabets)))
        if not self.lengths:
            raise ValueError("empty ladder")
        if list(self.lengths) != sorted(self.lengths):
            raise ValueError(
                "lengths must be ordered COARSEST FIRST, got %r.  Every other "
                "module in this package uses that order and a silently "
                "reversed ladder would train and sample the wrong way round."
                % (list(self.lengths),))
        for k, a in enumerate(self.alphabets):
            if int(a) < 2:
                raise ValueError("level %d has alphabet %d" % (k, a))

    @property
    def n_levels(self) -> int:
        return len(self.lengths)

    @property
    def n_tokens(self) -> int:
        return int(sum(self.lengths))

    @property
    def offsets(self) -> Tuple[int, ...]:
        """Start of each level in the flattened token sequence."""
        out, acc = [], 0
        for n in self.lengths:
            out.append(acc)
            acc += int(n)
        return tuple(out)

    def bp_per_cell(self, level: int) -> float:
        return float(self.seq_len) / float(self.lengths[level])

    def bp_midpoints(self, level: int) -> torch.Tensor:
        """Base-pair centre of every cell of `level`, as a float tensor."""
        s = self.bp_per_cell(level)
        i = torch.arange(self.lengths[level], dtype=torch.float64)
        return (i + 0.5) * s

    def declared_bits(self) -> List[float]:
        return [float(n) * math.log2(float(a))
                for n, a in zip(self.lengths, self.alphabets)]

    @staticmethod
    def from_config(model_cfg, seq_len: int) -> "LadderGeometry":
        """From a `hyppe.config.ModelConfig`."""
        from hyppe.config import fsq_grid
        lengths = tuple(model_cfg.map_lengths(int(seq_len)))
        alphabets = [int(model_cfg.l0_codes)]
        for b in model_cfg.fsq_bits:
            _dims, cells, _real = fsq_grid(int(b))
            alphabets.append(int(cells))
        return LadderGeometry(lengths=lengths,
                              alphabets=tuple(alphabets),
                              seq_len=int(seq_len))

    @staticmethod
    def from_ladder(ladder, seq_len: int,
                    lengths: Optional[Sequence[int]] = None
                    ) -> "LadderGeometry":
        """From a live `ConeLadder`.

        Alphabets are read off the quantisers, never from a config key: a key
        is a declaration, the assembled module is the fact. `lengths` cannot
        be read off the ladder at all, the maps being a strided view the
        ladder never sees, so it comes from the caller and is only sized.
        """
        qs = [ladder.position_quantizer] + list(ladder.tangent_quantizers)
        alphabets = []
        for k, q in enumerate(qs):
            size = getattr(q, "codebook_size", None)
            if not isinstance(size, int) or size < 2:
                raise ValueError(
                    "quantiser %d (%s) has no usable `codebook_size`; a "
                    "guessed alphabet would mis-size the embedding table and "
                    "the error would only show up as an index out of range on "
                    "a rare code." % (k, type(q).__name__))
            alphabets.append(int(size))
        if lengths is None:
            raise ValueError(
                "`lengths` is required: map lengths are a function of the "
                "strides and the ladder does not know them.")
        if len(lengths) != len(alphabets):
            raise ValueError("ladder has %d levels, got %d lengths"
                             % (len(alphabets), len(lengths)))
        return LadderGeometry(lengths=tuple(int(n) for n in lengths),
                              alphabets=tuple(alphabets),
                              seq_len=int(seq_len))

    @staticmethod
    def from_model(model) -> "LadderGeometry":
        """From an assembled `HyppeModel`."""
        return LadderGeometry.from_ladder(model.ladder,
                                          seq_len=int(model.seq_len),
                                          lengths=list(model.level_lengths))


def ladder_bit_budget(geom: LadderGeometry,
                      bits_per_bp: float = 1.9983) -> Dict[str, object]:
    """Declared code bits against sequence bits.

    `bits_per_bp` defaults to the corpus's measured unigram entropy; 2.0 is
    reported too, a reserve without its denominator not being checkable.
    """
    per: List[Dict[str, object]] = []
    for k in range(geom.n_levels):
        n, a = int(geom.lengths[k]), int(geom.alphabets[k])
        bpp = math.log2(float(a))
        per.append({"level": k, "length": n, "alphabet": a,
                    "bits_per_position": bpp, "bits": n * bpp})
    total = float(sum(float(p["bits"]) for p in per))
    seq_measured = float(geom.seq_len) * float(bits_per_bp)
    seq_uniform = float(geom.seq_len) * 2.0
    return {"levels": per,
            "total_bits": total,
            "n_tokens": geom.n_tokens,
            "sequence_bits_measured": seq_measured,
            "sequence_bits_uniform": seq_uniform,
            "reserve_measured": total / seq_measured,
            "reserve_uniform": total / seq_uniform,
            "bits_per_bp": float(bits_per_bp)}


def format_budget(budget: Dict[str, object]) -> str:
    rows = ["level  length  alphabet  bits/pos      bits",
            "------------------------------------------------"]
    for p in budget["levels"]:
        rows.append("  L%-3d %7d %9d %9.4f %9.2f"
                    % (p["level"], p["length"], p["alphabet"],
                       p["bits_per_position"], p["bits"]))
    rows.append("------------------------------------------------")
    rows.append("  total %6d tokens              %9.2f bits"
                % (budget["n_tokens"], budget["total_bits"]))
    rows.append("  sequence  %.1f bits measured (%.4f b/bp)  -> reserve %.3fx"
                % (budget["sequence_bits_measured"], budget["bits_per_bp"],
                   budget["reserve_measured"]))
    rows.append("  sequence  %.1f bits uniform  (2.0000 b/bp)  -> reserve %.3fx"
                % (budget["sequence_bits_uniform"], budget["reserve_uniform"]))
    return "\n".join(rows)


# Training-time masking
def cosine_ratio(u: torch.Tensor) -> torch.Tensor:
    """MaskGIT's mask-ratio draw: r = cos(pi u / 2) for u ~ U(0, 1).

    Mass sits at high ratios (E[r] = 2/pi = 0.637), which is what this latent
    needs: at low ratios a masked code is recoverable from its neighbour at
    R^2 = 0.95 and the gradient is nearly zero.
    """
    return torch.cos(u * (math.pi / 2.0))


class MaskDraw(NamedTuple):
    """What one training draw produces.

    `mask`   -- per level [B, L_k], True where the input token becomes MASK.
    `target` -- per level [B, L_k], True where the token counts in the loss.
                A subset of `mask`: an unmasked token cannot be a target, and
                not every masked token should be one.
    """

    mask: List[torch.Tensor]
    target: List[torch.Tensor]


@dataclass
class MaskSchedule:
    """How a training batch is masked: two regimes, mixed.

    `prefix` (probability `p_prefix`) draws a frontier level, shows every
    level above it, masks everything below, and masks the frontier level at a
    ratio drawn from the sampler's own step occupancy. `scatter` applies one
    cosine-drawn ratio independently across all tokens of all levels, the
    infilling task the design loop needs. `p_prefix = 0.5` is a declared split
    between the two capabilities, not a tuned one.

    `unmask` is not optional in practice: the continuous cosine draw puts
    essentially no mass on `r = 1`, the wholly undecided frontier level, which
    is where the sampler stands at the first step of every level -- 5 of its
    55 steps, and the 5 that decide the most. With the continuous draw,
    training reported 8.49 bits/token on L1 while held-out
    `frontier_L1@1.00` was 9.8572, worse than uniform's 9.5098, the model
    having learnt to read a level off its visible neighbours. An
    `UnmaskSchedule` replaces the draw by that schedule's occupancy, giving
    `r = 1` mass `1 / T_k`.

    The loss does not cover every masked token. In the prefix regime the
    levels below the frontier are masked because they are undecided -- state,
    not questions -- and the sampler only asks for the frontier level.
    Counting them trains a question never asked with 400 of the 775 tokens:
    the training log then read 1.9763 bits on L4 against a held-out 3.3054 for
    3.1699 uniform, and 3.3149 on training rows too, so it was a state
    occurring in 0.36 % of rows rather than memorisation.
    """

    p_prefix: float = 0.5
    min_masked: int = 1
    unmask: Optional["UnmaskSchedule"] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_prefix <= 1.0:
            raise ValueError("p_prefix must be in [0, 1]")

    def _frontier_ratio(self, frontier: torch.Tensor,
                        u: torch.Tensor) -> torch.Tensor:
        """Mask ratio for the frontier level, one value per batch row.

        The geometry is not an argument: the ratio comes from the unmask
        schedule and the draw, and taking a parameter it never reads made it
        look as though the level count mattered here.
        """
        if self.unmask is None:
            return cosine_ratio(u)
        steps = torch.tensor([float(t) for t in self.unmask.steps],
                             device=u.device)
        T = steps[frontier]                      # [B]
        t = torch.floor(u * T).clamp_(max=T - 1)  # t ~ U{0..T-1}
        return torch.cos(math.pi * t / (2.0 * T))

    def sample(self, geom: LadderGeometry, batch: int,
               device: torch.device,
               generator: Optional[torch.Generator] = None
               ) -> MaskDraw:
        """-> `MaskDraw`: what is hidden on the input, and what is scored."""
        _check_generator(generator, device, "MaskSchedule.sample")
        if self.unmask is not None and len(self.unmask.steps) != geom.n_levels:
            raise ValueError(
                "the unmask schedule has %d levels, the geometry has %d: the "
                "frontier ratio would be drawn from the wrong level's step "
                "count." % (len(self.unmask.steps), geom.n_levels))

        def rnd(*shape):
            return torch.rand(*shape, device=device, generator=generator)

        use_prefix = rnd(batch) < self.p_prefix
        frontier = torch.randint(0, geom.n_levels, (batch,), device=device,
                                 generator=generator)
        ratio = torch.where(use_prefix,
                            self._frontier_ratio(frontier, rnd(batch)),
                            cosine_ratio(rnd(batch)))

        masks: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for k in range(geom.n_levels):
            n = int(geom.lengths[k])
            scatter = rnd(batch, n) < ratio.unsqueeze(1)
            # prefix: above the frontier visible, below it fully masked, the
            # frontier level itself partially masked.
            below = (frontier < k).unsqueeze(1)
            at = (frontier == k).unsqueeze(1)
            prefix = below | (at & scatter)
            use = use_prefix.unsqueeze(1)
            masks.append(torch.where(use, prefix, scatter))
            # scored: the frontier level under prefix, everything under scatter
            targets.append(torch.where(use, at & scatter, scatter))

        # A row with nothing scored contributes no loss and quietly shrinks
        # the effective batch. Force `min_masked` tokens on the finest level,
        # in the input and the target together -- an unmasked target is a
        # token the model reads off its own input.
        if self.min_masked > 0:
            total = sum(t.sum(dim=1) for t in targets)
            short = total < self.min_masked
            if bool(short.any()):
                n = int(geom.lengths[-1])
                pick = torch.randint(0, n, (batch, self.min_masked),
                                     device=device, generator=generator)
                add = torch.zeros_like(targets[-1])
                add.scatter_(1, pick, True)
                keep = short.unsqueeze(1)
                targets[-1] = torch.where(keep, targets[-1] | add, targets[-1])
                masks[-1] = torch.where(keep, masks[-1] | add, masks[-1])
        return MaskDraw(mask=masks, target=targets)


# Sampling-time schedule
@dataclass
class UnmaskSchedule:
    """Coarse-to-fine order, and the number of confidence steps per level.

    The step count is derived. A step commits tokens drawn from their
    independent marginals, so it costs the dependence among them. At
    `INDEPENDENCE_BP = 34` a promoter holds `sites = seq_len / 34 = 23.5`
    quasi-independent positions, the same on every level since all cover the
    same base pairs, and committing at most `sites` per step keeps a commit
    set's spacing at or above the dependence range. The largest cosine commit
    is the last, about `n_k * pi / 2 T_k`, so
    `T_k = ceil(pi * n_k / (2 * sites))` -- L0: 2, L1: 4, L2: 7, L3: 14,
    L4: 28, 55 forwards in all. `min_steps = 2` is a floor because at T = 1
    the level is one joint draw from independent marginals, the exact failure
    confidence-based unmasking exists to avoid.
    """

    steps: Tuple[int, ...]
    order: Tuple[int, ...]
    sites: float
    independence_bp: float

    @staticmethod
    def derive(geom: LadderGeometry,
               independence_bp: float = INDEPENDENCE_BP,
               min_steps: int = 2,
               max_steps: Optional[int] = None) -> "UnmaskSchedule":
        sites = float(geom.seq_len) / float(independence_bp)
        m = max(1.0, math.floor(sites))
        steps = []
        for n in geom.lengths:
            t = math.ceil(math.pi * float(n) / (2.0 * m))
            t = max(int(min_steps), int(t))
            t = min(t, int(n))                      # never more steps than cells
            if max_steps is not None:
                t = min(t, int(max_steps))
            steps.append(int(t))
        return UnmaskSchedule(steps=tuple(steps),
                              order=tuple(range(geom.n_levels)),
                              sites=sites,
                              independence_bp=float(independence_bp))

    @property
    def total_steps(self) -> int:
        return int(sum(self.steps))

    @staticmethod
    def keep_after(n: int, t: int, T: int) -> int:
        """Cells still masked after step t of T, cosine schedule.

        `gamma(t/T) = cos(pi t / 2T)` runs from 1 (all masked) to 0 and is
        monotone by construction, so a step can never un-commit a token.
        """
        g = math.cos(math.pi * float(t) / (2.0 * float(T)))
        return int(max(0, min(n, math.floor(n * g))))

    def commits(self, level: int, n: int) -> List[int]:
        """Tokens committed at each step of `level`.  Sums to n."""
        T = self.steps[level]
        out, prev = [], n
        for t in range(1, T + 1):
            rem = self.keep_after(n, t, T) if t < T else 0
            out.append(prev - rem)
            prev = rem
        return out

    def describe(self, geom: LadderGeometry) -> str:
        rows = ["level  cells  steps  max commit  bp per commit-gap",
                "-------------------------------------------------"]
        for k in range(geom.n_levels):
            c = self.commits(k, int(geom.lengths[k]))
            mx = max(c) if c else 0
            gap = (float(geom.seq_len) / mx) if mx else float("inf")
            rows.append("  L%-3d %6d %6d %11d %18.1f"
                        % (k, geom.lengths[k], self.steps[k], mx, gap))
        rows.append("-------------------------------------------------")
        rows.append("  sites = %d bp / %.0f bp = %.2f   total %d forwards"
                    % (geom.seq_len, self.independence_bp, self.sites,
                       self.total_steps))
        return "\n".join(rows)


# The prior
class _BpPositionEncoding(nn.Module):
    """Sinusoidal encoding of the base-pair centre of a cell.

    Shared by every level, so a coarse cell and the fine cells inside it land
    close together and cross-level attention is handed the correspondence
    rather than learning it. `n_freq` is set by the finest level: the top
    frequency must resolve two adjacent cells of the finest map.
    """

    def __init__(self, d_model: int, seq_len: int, finest_bp: float):
        super().__init__()
        need = max(1.0, float(seq_len) / max(finest_bp, 1e-9))
        self.n_freq = int(max(4, math.ceil(math.log2(need)) + 1))
        self.seq_len = int(seq_len)
        self.proj = nn.Linear(2 * self.n_freq, d_model)
        freqs = torch.tensor([2.0 ** j for j in range(self.n_freq)])
        self.register_buffer("freqs", freqs, persistent=False)

    def extra_repr(self) -> str:
        return ("n_freq=%d, top=%d cycles over %d bp (period %.2f bp)"
                % (self.n_freq, int(2 ** (self.n_freq - 1)), self.seq_len,
                   self.seq_len / float(2 ** (self.n_freq - 1))))

    def forward(self, bp: torch.Tensor) -> torch.Tensor:
        u = (bp.to(self.freqs.dtype) / float(self.seq_len)).unsqueeze(-1)
        ang = 2.0 * math.pi * u * self.freqs
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))


class _BpLocalityBias(nn.Module):
    """Additive attention bias, linear in the base-pair distance between cells.

    A flat transformer over 775 tokens starts with near-uniform attention, so
    the one token that matters -- a neighbour 2 bp away, or the parent cell
    above -- arrives divided by 780. On a synthetic corpus whose
    child-given-parent entropy is 1.5850 bits at the finest level, 200 steps
    without the bias leave that level at 3.167 bits (uniform to three
    decimals) and need ~1500 to reach 1.859; with it, 200 steps reach 2.384.
    The bias buys the rate, not the possibility.

    The scale comes from `INDEPENDENCE_BP = 34`: the bias initialises at -1
    (attention weight 1/e) at exactly that distance. It is a parameter, and
    `report()` prints the learnt range in bp so "needed" and "ignored" stay
    distinguishable. One bias is shared by all heads, per head being
    [B*heads, N, N] and 622 MB at batch 32 against [N, N] and 2.4 MB.
    """

    def __init__(self, bp: torch.Tensor, n_ctx: int,
                 scale_bp: float = INDEPENDENCE_BP):
        super().__init__()
        self.n_ctx = int(n_ctx)
        self.scale_bp = float(scale_bp)
        d = (bp.reshape(-1, 1) - bp.reshape(1, -1)).abs()
        if self.n_ctx:
            n = d.shape[0] + self.n_ctx
            full = torch.zeros(n, n, dtype=d.dtype)
            full[self.n_ctx:, self.n_ctx:] = d
            d = full
        self.register_buffer("distance", d.float(), persistent=False)
        # softplus(raw) = 1.0 at init -> bias -1 at scale_bp
        self.raw_slope = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))

    def range_bp(self) -> float:
        """Distance at which the bias reaches -1, in base pairs."""
        s = float(F.softplus(self.raw_slope.detach()))
        return self.scale_bp / max(s, 1e-6)

    def extra_repr(self) -> str:
        return "scale_bp=%.1f, init range %.1f bp, shared across heads" % (
            self.scale_bp, self.range_bp())

    def forward(self) -> torch.Tensor:
        return -F.softplus(self.raw_slope) * (self.distance / self.scale_bp)


class LadderCodePrior(nn.Module):
    """p(codes | C): a masked transformer over the flattened ladder.

    Input is the whole ladder at once, the levels not being independent: a
    level is recoverable from the others at R^2 = 0.95 (L1 from {L0, L2}) and
    0.82 (L0 from {L1, L2}), so a per-level prior would model a marginal and
    call it a joint. Each level has its own embedding table and output head,
    the alphabets meaning different things -- level 0 indexes points of a
    hyperbolic codebook, levels >= 1 cells of a scalar grid.

    The transformer size is set against the effective corpus: 238 460
    promoters x 775 tokens would allow ~9 M parameters, but orthologues are
    not independent examples (within-orthogroup identity 0.655 against 0.252
    for random pairs), so the effective n is nearer the 26 271 orthogroups and
    the defaults (~2.2 M) sit at ~9 effective tokens per parameter.
    `ctx_dropout` is the chance of replacing the whole context by the learned
    null token in training -- the branch classifier-free guidance interpolates
    from, without which `guidance != 0` extrapolates into unseen territory.
    """

    def __init__(self,
                 geom: LadderGeometry,
                 d_model: int = 192,
                 depth: int = 4,
                 heads: int = 6,
                 d_ff: int = 768,
                 dropout: float = 0.1,
                 ctx_dim: int = 0,
                 ctx_tokens: int = 4,
                 ctx_dropout: float = 0.1,
                 local_bias: bool = True) -> None:
        super().__init__()
        self.geom = geom
        self.d_model = int(d_model)
        self.ctx_dim = int(ctx_dim)
        self.ctx_tokens = int(ctx_tokens) if ctx_dim else 0
        self.ctx_dropout = float(ctx_dropout)

        # Code embeddings: one table per level, last row is MASK.
        self.embed = nn.ModuleList([
            nn.Embedding(int(a) + 1, self.d_model) for a in geom.alphabets])
        #: MASK symbol of level k: `alphabets[k]`, one past the last legal
        #: code, so an off-by-one is an index error, not a plausible symbol.
        self.mask_id: Tuple[int, ...] = tuple(int(a) for a in geom.alphabets)

        self.level_embed = nn.Parameter(
            torch.zeros(geom.n_levels, self.d_model))
        nn.init.normal_(self.level_embed, std=0.02)

        self.pos = _BpPositionEncoding(
            self.d_model, geom.seq_len,
            finest_bp=geom.bp_per_cell(geom.n_levels - 1))
        bp = torch.cat([geom.bp_midpoints(k) for k in range(geom.n_levels)])
        self.register_buffer("token_bp", bp.float(), persistent=False)
        lv = torch.cat([torch.full((int(n),), k, dtype=torch.long)
                        for k, n in enumerate(geom.lengths)])
        self.register_buffer("token_level", lv, persistent=False)

        if self.ctx_dim:
            self.ctx_proj = nn.Linear(self.ctx_dim, self.d_model)
            self.ctx_slot = nn.Parameter(
                torch.zeros(self.ctx_tokens, self.d_model))
            nn.init.normal_(self.ctx_slot, std=0.02)
            self.ctx_null = nn.Parameter(torch.zeros(1, self.d_model))
            nn.init.normal_(self.ctx_null, std=0.02)
        else:
            self.ctx_proj = None

        self.bias = (_BpLocalityBias(bp, self.ctx_tokens) if local_bias
                     else None)

        self.in_norm = nn.LayerNorm(self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(heads),
            dim_feedforward=int(d_ff), dropout=float(dropout),
            activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor=False silences the nested-tensor warning; it
        # does not disable the fused fast path -- see the guard in `forward`.
        self.trunk = nn.TransformerEncoder(layer, num_layers=int(depth),
                                           enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(self.d_model)
        self.head = nn.ModuleList([
            nn.Linear(self.d_model, int(a)) for a in geom.alphabets])

    # Reports
    def parameter_report(self) -> Dict[str, int]:
        def n(mod) -> int:
            if isinstance(mod, nn.Parameter):
                return int(mod.numel())
            return int(sum(p.numel() for p in mod.parameters()))
        out = {
            "embed_codes": int(sum(n(e) for e in self.embed)),
            "embed_level": n(self.level_embed),
            "embed_position": n(self.pos),
            "context": (n(self.ctx_proj) + n(self.ctx_slot) + n(self.ctx_null)
                        if self.ctx_proj is not None else 0),
            "trunk": n(self.trunk) + n(self.in_norm) + n(self.out_norm),
            "heads": int(sum(n(h) for h in self.head)),
            "local_bias": 0 if self.bias is None else n(self.bias),
        }
        out["total"] = int(sum(out.values()))
        counted = out["total"]
        live = int(sum(p.numel() for p in self.parameters()))
        if counted != live:
            raise RuntimeError(
                "parameter_report counts %d but the module holds %d. A report "
                "that silently misses a tensor is how a parameter budget stops "
                "being a budget." % (counted, live))
        return out

    def report(self) -> Dict[str, float]:
        """Observables that say whether the parts are doing anything."""
        out: Dict[str, float] = {}
        if self.bias is not None:
            out["local_bias_range_bp"] = self.bias.range_bp()
        if self.ctx_proj is not None:
            out["ctx_null_norm"] = float(self.ctx_null.detach().norm())
            out["ctx_slot_norm"] = float(self.ctx_slot.detach().norm())
        out["level_embed_norm"] = float(self.level_embed.detach().norm())
        return out

    def budget_note(self, corpus_sequences: int,
                    effective_sequences: Optional[int] = None) -> str:
        """Parameters against the corpus, raw and effective.

        `effective_sequences` is the orthogroup count, not the row count: the
        two differ by 9x here and the smaller binds, so printing only the
        larger is how a prior gets declared well-fed.
        """
        p = self.parameter_report()["total"]
        raw = int(corpus_sequences) * self.geom.n_tokens
        out = ("prior %d parameters | raw corpus %d x %d = %.3g tokens "
               "(%.1f per parameter)"
               % (p, corpus_sequences, self.geom.n_tokens, float(raw),
                  raw / max(p, 1)))
        if effective_sequences:
            eff = int(effective_sequences) * self.geom.n_tokens
            out += (" | EFFECTIVE %d orthogroups = %.3g tokens (%.1f per "
                    "parameter)"
                    % (effective_sequences, float(eff), eff / max(p, 1)))
        return out

    def extra_repr(self) -> str:
        return ("levels=%r alphabets=%r tokens=%d d_model=%d ctx_dim=%d"
                % (list(self.geom.lengths), list(self.geom.alphabets),
                   self.geom.n_tokens, self.d_model, self.ctx_dim))

    # Forward
    def _tokens(self, indices: Sequence[torch.Tensor],
                masks: Optional[Sequence[torch.Tensor]]) -> torch.Tensor:
        """[B, N, d] code embeddings with masked positions replaced."""
        if len(indices) != self.geom.n_levels:
            raise ValueError("expected %d index tensors, got %d"
                             % (self.geom.n_levels, len(indices)))
        parts = []
        for k, idx in enumerate(indices):
            n = int(self.geom.lengths[k])
            if idx.shape[-1] != n:
                raise ValueError("level %d: %d cells, geometry says %d"
                                 % (k, idx.shape[-1], n))
            sym = idx.long()
            if masks is not None:
                sym = torch.where(masks[k], torch.full_like(sym,
                                                            self.mask_id[k]),
                                  sym)
            parts.append(self.embed[k](sym))
        return torch.cat(parts, dim=1)

    def forward(self,
                indices: Sequence[torch.Tensor],
                masks: Optional[Sequence[torch.Tensor]] = None,
                context: Optional[torch.Tensor] = None,
                drop_context: Optional[torch.Tensor] = None
                ) -> List[torch.Tensor]:
        """-> per level, logits [B, L_k, K_k].

        `context` is [B, T, ctx_dim] (the `ConditionEncoder` tokens) or None.
        `drop_context` is an optional boolean [B]: True replaces that row's
        whole context by the learned null token. In training it is drawn from
        `ctx_dropout`; the guidance path supplies it explicitly.
        """
        h = self._tokens(indices, masks)
        h = h + self.level_embed[self.token_level].unsqueeze(0)
        h = h + self.pos(self.token_bp).unsqueeze(0)

        n_ctx = 0
        if self.ctx_proj is not None:
            batch = h.shape[0]
            if context is None:
                ctx = self.ctx_null.expand(batch, self.ctx_tokens,
                                           self.d_model)
            else:
                if context.dim() != 3 or context.shape[-1] != self.ctx_dim:
                    raise ValueError(
                        "context must be [B, T, %d], got %r"
                        % (self.ctx_dim, tuple(context.shape)))
                ctx = self.ctx_proj(context) + self.ctx_slot.unsqueeze(0)
                if drop_context is None and self.training:
                    drop_context = (torch.rand(batch, device=h.device)
                                    < self.ctx_dropout)
                if drop_context is not None:
                    null = self.ctx_null.expand(batch, ctx.shape[1],
                                                self.d_model)
                    ctx = torch.where(
                        drop_context.view(-1, 1, 1).to(torch.bool), null, ctx)
            n_ctx = ctx.shape[1]
            h = torch.cat([ctx, h], dim=1)

        attn_bias = None
        if self.bias is not None:
            attn_bias = self.bias()
            if n_ctx == 0 and attn_bias.shape[0] != h.shape[1]:
                # the bias was built with context slots; drop them
                off = attn_bias.shape[0] - h.shape[1]
                attn_bias = attn_bias[off:, off:]
            attn_bias = attn_bias.to(h.dtype)
        # The fused fast path demotes a float bias to a boolean mask, and hid
        # because the demotion only happens under no_grad -- training runs
        # with grad and takes the slow path. The bias holds 1.15 % exact
        # zeros, so a code token attended to 5 keys instead of 779, at a
        # held-out cost of 0.119 bit/token in eval and generation only. Guard
        # only for a float bias: with mask=None the paths agree to 9.5e-07.
        _fp_prev = _mha_backend.get_fastpath_enabled()
        _fp_guard = attn_bias is not None and _fp_prev
        if _fp_guard:
            _mha_backend.set_fastpath_enabled(False)
        try:
            h = self.trunk(self.in_norm(h), mask=attn_bias)
        finally:
            if _fp_guard:
                _mha_backend.set_fastpath_enabled(_fp_prev)
        h = self.out_norm(h[:, n_ctx:])

        out, off = [], 0
        for k in range(self.geom.n_levels):
            n = int(self.geom.lengths[k])
            out.append(self.head[k](h[:, off:off + n]))
            off += n
        return out

    # Loss
    def loss(self,
             indices: Sequence[torch.Tensor],
             masks: Sequence[torch.Tensor],
             context: Optional[torch.Tensor] = None,
             level_weights: Optional[Sequence[float]] = None,
             drop_context: Optional[torch.Tensor] = None,
             targets: Optional[Sequence[torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy on the scored positions.

        `targets` defaults to `masks` and must otherwise be a subset of it,
        asserted: scoring a token visible on the input is a free win. With
        `level_weights = None` this is the plain code likelihood, so the
        finest level dominates by token count (400 of 775) -- right for a
        likelihood, wrong as a summary, which is why per-level bits come back
        separately and a collapse at L0 cannot hide in a good total. Pass
        `drop_context` only to exercise both guidance branches
        deterministically; at p = 0.1 and batch 4 the draw is otherwise flaky
        in 66 % of cases.
        """
        logits = self.forward(indices, masks, context, drop_context)
        if targets is None:
            targets = masks
        else:
            for k, (t, m) in enumerate(zip(targets, masks)):
                if bool((t & ~m).any()):
                    raise ValueError(
                        "level %d scores %d positions that are NOT masked on "
                        "the input.  The model can read those off its own "
                        "input, so the loss would fall for free."
                        % (k, int((t & ~m).sum())))
        total = torch.zeros((), device=logits[0].device,
                            dtype=logits[0].dtype)
        count = 0
        stats: Dict[str, float] = {}
        for k, lg in enumerate(logits):
            m = targets[k]
            n_masked = int(m.sum())
            if n_masked == 0:
                stats["bits_L%d" % k] = float("nan")
                stats["n_L%d" % k] = 0
                continue
            tgt = indices[k].long()[m]
            ce = F.cross_entropy(lg[m], tgt, reduction="sum")
            w = 1.0 if level_weights is None else float(level_weights[k])
            total = total + w * ce
            count += n_masked
            stats["bits_L%d" % k] = float(ce.detach()) / n_masked / _LOG2
            stats["n_L%d" % k] = n_masked
        if count == 0:
            raise RuntimeError(
                "no masked token in the whole batch: the loss would be 0 and "
                "the step would be a no-op that looks like convergence.")
        loss = total / count
        stats["bits_total"] = float(loss.detach()) / _LOG2
        stats["n_masked"] = count
        return loss, stats

    # Positive tests
    @torch.no_grad()
    def check_mask_is_read(self, batch: int = 2,
                           seed: int = 0) -> float:
        """max |logits(masked) - logits(unmasked)| on the masked positions.

        A mandatory positive control, which must be > 0: every other test here
        passes trivially on a module that ignores its input, and an untrained
        model in eval often looks exactly like that.
        """
        g = torch.Generator().manual_seed(int(seed))
        dev = self._device()
        idx = self.random_indices(batch, generator=g)
        m = [torch.zeros(batch, int(n), dtype=torch.bool, device=dev)
             for n in self.geom.lengths]
        m[0][:, :1] = True
        was = self.training
        self.eval()
        a = self.forward(idx, None)[0][:, 0]
        b = self.forward(idx, m)[0][:, 0]
        self.train(was)
        return float((a - b).abs().max())

    @torch.no_grad()
    def check_context_is_read(self, batch: int = 2, seed: int = 0) -> float:
        """max |logits(C) - logits(null C)|.  Must be > 0 when ctx_dim > 0."""
        if self.ctx_proj is None:
            return float("nan")
        g = torch.Generator().manual_seed(int(seed))
        dev = self._device()
        idx = self.random_indices(batch, generator=g)
        ctx = torch.randn(batch, self.ctx_tokens, self.ctx_dim,
                          generator=g).to(dev)
        was = self.training
        self.eval()
        a = self.forward(idx, None, ctx,
                         torch.zeros(batch, dtype=torch.bool, device=dev))[0]
        b = self.forward(idx, None, ctx,
                         torch.ones(batch, dtype=torch.bool, device=dev))[0]
        self.train(was)
        return float((a - b).abs().max())

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def random_indices(self, batch: int,
                       generator: Optional[torch.Generator] = None
                       ) -> List[torch.Tensor]:
        """Uniform random codes, drawn on the CPU and moved.

        On the CPU on purpose: a CPU `Generator` cannot seed a CUDA draw, and
        dropping it would tie every self-test to the global RNG stream.
        """
        dev = self._device()
        return [torch.randint(0, int(a), (batch, int(n)),
                              generator=generator).to(dev)
                for n, a in zip(self.geom.lengths, self.geom.alphabets)]


# Baselines
class MarginalCodeTable:
    """The bar the prior has to clear: code frequencies, fitted and smoothed.

    Two tables, the weaker one being easy to beat and flattering: `global`,
    one histogram per level, and `position`, one per level and position, both
    under Krichevsky-Trofimov smoothing (alpha = 0.5, minimax-regret for a
    multinomial). Without it the per-position table gives an unseen symbol
    probability zero, L1 having 729 symbols per cell, and an infinite held-out
    cross-entropy would hand the prior an unearned win. Fit on train, score on
    held-out: a table scored on its own training rows is an upper bound on
    what a table can do, not a baseline.
    """

    def __init__(self, geom: LadderGeometry, alpha: float = 0.5):
        self.geom = geom
        self.alpha = float(alpha)
        self.counts_global = [torch.zeros(int(a), dtype=torch.float64)
                              for a in geom.alphabets]
        self.counts_pos = [torch.zeros(int(n), int(a), dtype=torch.float64)
                           for n, a in zip(geom.lengths, geom.alphabets)]
        self.n_rows = 0

    def fit(self, indices: Sequence[torch.Tensor]) -> "MarginalCodeTable":
        for k, idx in enumerate(indices):
            a = int(self.geom.alphabets[k])
            flat = idx.reshape(-1).long().cpu()
            self.counts_global[k] += torch.bincount(flat, minlength=a).double()
            for i in range(int(self.geom.lengths[k])):
                col = idx[:, i].reshape(-1).long().cpu()
                self.counts_pos[k][i] += torch.bincount(
                    col, minlength=a).double()
        self.n_rows += int(indices[0].shape[0])
        return self

    def _probs(self, counts: torch.Tensor) -> torch.Tensor:
        c = counts + self.alpha
        return c / c.sum(dim=-1, keepdim=True)

    def bits(self, indices: Sequence[torch.Tensor],
             masks: Optional[Sequence[torch.Tensor]] = None
             ) -> Dict[str, object]:
        """Held-out bits per token, per level, for both tables and uniform."""
        out: Dict[str, object] = {}
        tot = {"uniform": 0.0, "global": 0.0, "position": 0.0}
        n_tot = 0
        for k, idx in enumerate(indices):
            m = (torch.ones_like(idx, dtype=torch.bool) if masks is None
                 else masks[k])
            n = int(m.sum())
            if n == 0:
                continue
            tgt = idx.long().cpu()
            sel = m.cpu()
            pg = self._probs(self.counts_global[k])
            lg = -(pg[tgt].log2())[sel].sum()
            pp = self._probs(self.counts_pos[k])            # [n_k, K]
            # Gather on the [n_k, K] table directly: expanding to [B, n_k, K]
            # first is 300 MB of float64 on L1 alone, for B*n_k numbers.
            lp_full = -(pp.gather(1, tgt.transpose(0, 1))
                        .transpose(0, 1).log2())            # [B, n_k]
            lp = lp_full[sel].sum()
            u = float(math.log2(float(self.geom.alphabets[k]))) * n
            out["uniform_L%d" % k] = u / n
            out["global_L%d" % k] = float(lg) / n
            out["position_L%d" % k] = float(lp) / n
            tot["uniform"] += u
            tot["global"] += float(lg)
            tot["position"] += float(lp)
            n_tot += n
        for key, v in tot.items():
            out[key] = v / max(n_tot, 1)
        out["n_scored"] = n_tot
        return out


# Evaluation
@dataclass
class EvalRegime:
    """Which masking the prior is scored under. Never omit it.

    The regime is part of the number, and two distinct leaks hide in it. The
    within-level leak is a measurement artefact: a code is recoverable from a
    neighbour 2 bp away at R^2 = 0.95, dying at 34 bp, so a prior scored at a
    15 % scatter mask answers a question its neighbours already answered and
    beats every baseline knowing nothing about promoters (`is_leaky()`).
    Cross-level redundancy is not an artefact but what the sampler runs on --
    knowledge about the latent rather than about promoters -- so the two must
    not be added into one headline: read `frontier_L0` for the first and
    `frontier_L1..L4` for the second.

    `frontier` keeps levels above `level` visible, masks `level` at `ratio`
    and masks everything below -- the sampler's own state distribution, with
    `frontier(0, 1.0)` its first state. `scatter` applies a flat ratio to
    every token of every level, and `ratio = 1.0` is unconditional.
    """

    kind: str = "frontier"
    level: int = 0
    ratio: float = 1.0

    @property
    def name(self) -> str:
        if self.kind == "frontier":
            return "frontier_L%d@%.2f" % (self.level, self.ratio)
        return "scatter_%.2f" % self.ratio

    def leak_bp(self, geom: LadderGeometry) -> float:
        """Expected bp distance from a masked cell to the nearest visible one.

        A run of masked cells has mean length `ratio / (1 - ratio)`, so the
        nearest visible cell sits about `bp_per_cell / (1 - ratio)` away. The
        worst scored level wins: `scatter` scores every level at once, where
        one ratio is a harmless 89 bp gap on L0 and a 5.6 bp gap on L4.
        """
        levels = self.scored_levels(geom)
        if not levels or self.ratio >= 1.0:
            return float("inf")
        return min(geom.bp_per_cell(k) / max(1.0 - self.ratio, 1e-9)
                   for k in levels)

    def is_leaky(self, geom: LadderGeometry) -> bool:
        return self.leak_bp(geom) < INDEPENDENCE_BP

    def masks(self, geom: LadderGeometry, batch: int,
              device: torch.device,
              generator: Optional[torch.Generator] = None
              ) -> List[torch.Tensor]:
        out = []
        for k, n in enumerate(geom.lengths):
            n = int(n)
            if self.kind == "frontier":
                if k < self.level:
                    out.append(torch.zeros(batch, n, dtype=torch.bool,
                                           device=device))
                elif k > self.level:
                    out.append(torch.ones(batch, n, dtype=torch.bool,
                                          device=device))
                else:
                    r = torch.rand(batch, n, device=device,
                                   generator=generator)
                    out.append(r < self.ratio)
            elif self.kind == "scatter":
                r = torch.rand(batch, n, device=device, generator=generator)
                out.append(r < self.ratio)
            else:
                raise ValueError("unknown regime kind %r" % (self.kind,))
        return out

    def scored_levels(self, geom: LadderGeometry) -> List[int]:
        """Levels whose numbers are the point of this regime.

        Under `frontier` everything at or below it is masked, but only the
        frontier level is what the sampler queries next; the deeper levels are
        masked because they are not decided yet.
        """
        if self.kind == "frontier":
            return [self.level]
        return list(range(geom.n_levels))


@dataclass
class PriorEvaluation:
    """Bits per token for the prior and its three baselines, per level."""

    regime: str
    leaky: bool
    n_scored: int
    per_level: Dict[int, Dict[str, float]] = field(default_factory=dict)
    totals: Dict[str, float] = field(default_factory=dict)
    shuffle: Optional[Dict[str, float]] = None

    #: Below this advantage over the stronger table the prior is
    #: indistinguishable from a frequency table. 0.05 bits/token over 775
    #: tokens is ~39 bits of code, so it is worth something on the scale of
    #: the budget before it is called knowledge.
    MARGIN: float = 0.05

    def verdict(self, level: Optional[int] = None) -> str:
        row = (self.totals if level is None
               else self.per_level.get(int(level), {}))
        if not row:
            return "not scored"
        table = min(row.get("marginal_global", float("inf")),
                    row.get("marginal_position", float("inf")))
        prior = row.get("prior", float("inf"))
        uni = row.get("uniform", float("inf"))
        if prior >= uni:
            return ("NO USABLE PRIOR HERE: %.4f bits against %.4f uniform"
                    % (prior, uni))
        if prior > table - self.MARGIN:
            return ("FREQUENCY TABLE: %.4f bits against %.4f for the best "
                    "table (advantage %.4f < %.4f)"
                    % (prior, table, table - prior, self.MARGIN))
        return ("prior: %.4f bits against %.4f table / %.4f uniform "
                "(advantage %.4f bits/token)"
                % (prior, table, uni, table - prior))

    def format(self) -> str:
        rows = ["regime %s%s   scored %d tokens"
                % (self.regime,
                   "   [LEAKY: a visible cell of the same level sits inside "
                   "the 34 bp dependence range]" if self.leaky else "",
                   self.n_scored),
                "level   uniform  marg-glob  marg-pos     PRIOR   advantage",
                "----------------------------------------------------------"]
        for k in sorted(self.per_level):
            r = self.per_level[k]
            best = min(r["marginal_global"], r["marginal_position"])
            rows.append("  L%-3d %9.4f %10.4f %9.4f %9.4f %11.4f"
                        % (k, r["uniform"], r["marginal_global"],
                           r["marginal_position"], r["prior"],
                           best - r["prior"]))
        if self.totals:
            r = self.totals
            best = min(r["marginal_global"], r["marginal_position"])
            rows.append("----------------------------------------------------------")
            rows.append("  all  %9.4f %10.4f %9.4f %9.4f %11.4f"
                        % (r["uniform"], r["marginal_global"],
                           r["marginal_position"], r["prior"], best - r["prior"]))
        if self.shuffle is not None:
            rows.append("  shuffle control: prior %.4f bits, advantage %.4f "
                        "(must collapse to ~0)"
                        % (self.shuffle["prior"], self.shuffle["advantage"]))
        rows.append("  verdict: " + self.verdict())
        return "\n".join(rows)


def _shuffle_within_position(indices: Sequence[torch.Tensor],
                             generator: Optional[torch.Generator] = None
                             ) -> List[torch.Tensor]:
    """Permute rows independently at every (level, position).

    Destroys every dependency -- along the sequence, across levels, with the
    context -- while leaving each per-position histogram exactly unchanged, so
    the baselines are numerically identical on the shuffled corpus. Anything
    the prior still wins there it did not win from the corpus.
    """
    out = []
    for idx in indices:
        b = idx.shape[0]
        new = idx.clone()
        for i in range(idx.shape[1]):
            perm = torch.randperm(b, generator=generator, device=idx.device)
            new[:, i] = idx[perm, i]
        out.append(new)
    return out


@torch.no_grad()
def evaluate_prior(prior: LadderCodePrior,
                   indices: Sequence[torch.Tensor],
                   table: MarginalCodeTable,
                   regime: EvalRegime,
                   context: Optional[torch.Tensor] = None,
                   shuffle_control: bool = True,
                   seed: int = 0,
                   batch_size: int = 64) -> PriorEvaluation:
    """Bits per token of prior vs uniform vs both marginal tables.

    `indices` must be the held-out split and `table` fitted on the training
    split.
    """
    was = prior.training
    prior.eval()
    device = next(prior.parameters()).device
    geom = prior.geom
    scored = regime.scored_levels(geom)
    n_rows = int(indices[0].shape[0])

    def run(idx_all: Sequence[torch.Tensor]) -> Tuple[Dict[int, float], int]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        acc = {k: 0.0 for k in scored}
        cnt = {k: 0 for k in scored}
        for s in range(0, n_rows, batch_size):
            e = min(n_rows, s + batch_size)
            idx = [t[s:e].to(device) for t in idx_all]
            m_cpu = regime.masks(geom, e - s, torch.device("cpu"), gen)
            m = [t.to(device) for t in m_cpu]
            ctx = None if context is None else context[s:e].to(device)
            logits = prior.forward(idx, m, ctx)
            for k in scored:
                sel = m[k]
                n = int(sel.sum())
                if n == 0:
                    continue
                ce = F.cross_entropy(logits[k][sel], idx[k].long()[sel],
                                     reduction="sum")
                acc[k] += float(ce) / _LOG2
                cnt[k] += n
        return ({k: acc[k] / max(cnt[k], 1) for k in scored},
                int(sum(cnt.values())))

    prior_bits, n_scored = run(indices)

    # Baselines, on exactly the same mask draw (same seed, same generator).
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    masks = []
    for s in range(0, n_rows, batch_size):
        e = min(n_rows, s + batch_size)
        masks.append(regime.masks(geom, e - s, torch.device("cpu"), gen))
    full_masks = [torch.cat([mb[k] for mb in masks], dim=0)
                  for k in range(geom.n_levels)]
    base = table.bits([t.cpu() for t in indices],
                      [m.cpu() for m in full_masks])

    per_level: Dict[int, Dict[str, float]] = {}
    for k in scored:
        per_level[k] = {
            "uniform": float(base["uniform_L%d" % k]),
            "marginal_global": float(base["global_L%d" % k]),
            "marginal_position": float(base["position_L%d" % k]),
            "prior": float(prior_bits[k]),
        }
    # Total over the scored levels, weighted by the tokens each carried.
    weights = {k: int(full_masks[k].sum()) for k in scored}
    wsum = max(sum(weights.values()), 1)
    totals = {key: sum(per_level[k][key] * weights[k] for k in scored) / wsum
              for key in ("uniform", "marginal_global", "marginal_position",
                          "prior")}

    shuffle = None
    if shuffle_control:
        g = torch.Generator().manual_seed(int(seed) + 1)
        shuffled = _shuffle_within_position([t.cpu() for t in indices], g)
        sh_bits, _ = run([t.to(device) for t in shuffled])
        sh_total = sum(sh_bits[k] * weights[k] for k in scored) / wsum
        best = min(totals["marginal_global"], totals["marginal_position"])
        shuffle = {"prior": sh_total, "advantage": best - sh_total}

    prior.train(was)
    return PriorEvaluation(regime=regime.name, leaky=regime.is_leaky(geom),
                           n_scored=n_scored, per_level=per_level,
                           totals=totals, shuffle=shuffle)


# Corpus
@dataclass
class CodeCorpus:
    """Index tensors plus the health check that says whether they mean anything.

    A prior fitted to codes from an untrained encoder fits a degenerate
    corpus, and the failure is invisible in a loss curve: the loss falls
    beautifully, to the entropy of a near-constant. On 128 real promoters
    through a freshly initialised encoder, eval mode used 1 code of
    128 / 729 / 81 / 9 / 9 and carried 0.0 bits everywhere. `health()` names
    any level under `floor_bits` degenerate; it prevents quoting such a
    result, not producing it.
    """

    indices: List[torch.Tensor]
    geom: LadderGeometry
    source: str = "unknown"

    def __len__(self) -> int:
        return int(self.indices[0].shape[0])

    def split(self, n_train: int) -> Tuple["CodeCorpus", "CodeCorpus"]:
        a = [t[:n_train] for t in self.indices]
        b = [t[n_train:] for t in self.indices]
        return (CodeCorpus(a, self.geom, self.source + ":train"),
                CodeCorpus(b, self.geom, self.source + ":heldout"))

    def health(self, floor_bits: float = 0.5) -> Dict[str, object]:
        rows, degenerate = [], []
        achieved = 0.0
        for k, idx in enumerate(self.indices):
            a = int(self.geom.alphabets[k])
            cnt = torch.bincount(idx.reshape(-1).long().cpu(),
                                 minlength=a).double()
            p = cnt / cnt.sum()
            nz = p[p > 0]
            H = float(-(nz * nz.log2()).sum())
            used = int((cnt > 0).sum())
            achieved += H * float(self.geom.lengths[k])
            rows.append({"level": k, "used": used, "alphabet": a,
                         "bits": H, "declared_bits": math.log2(float(a))})
            if H < floor_bits:
                degenerate.append(k)
        declared = float(sum(self.geom.declared_bits()))
        return {"source": self.source, "rows": rows,
                "achieved_bits": achieved, "declared_bits": declared,
                "achieved_frac": achieved / declared if declared else 0.0,
                "degenerate_levels": degenerate}

    @staticmethod
    def format_health(h: Dict[str, object]) -> str:
        rows = ["corpus %s" % h["source"],
                "level   used/alphabet   bits/token   declared",
                "---------------------------------------------"]
        for r in h["rows"]:
            rows.append("  L%-3d %7d/%-7d %11.4f %10.4f"
                        % (r["level"], r["used"], r["alphabet"], r["bits"],
                           r["declared_bits"]))
        rows.append("---------------------------------------------")
        rows.append("  achieved %.1f of %.1f declared bits (%.1f %%)"
                    % (h["achieved_bits"], h["declared_bits"],
                       100.0 * h["achieved_frac"]))
        if h["degenerate_levels"]:
            rows.append("  DEGENERATE LEVELS %r: under 0.5 bits per token. A "
                        "prior fitted here is fitting a constant and its "
                        "advantage over the marginal will be ~0 BY "
                        "CONSTRUCTION." % (h["degenerate_levels"],))
        return "\n".join(rows)


# Sampling
@torch.no_grad()
def generate(prior: LadderCodePrior,
             batch: int,
             schedule: Optional[UnmaskSchedule] = None,
             context: Optional[torch.Tensor] = None,
             guidance: float = 0.0,
             temperature: float = 1.0,
             noise_anneal: float = 1.0,
             # "random", not "confidence": at mix_w=1.0 the per-cell law is
             # exactly the positional table, so the samples must come out
             # i.i.d. Measured (GC | pid | bits): true 0.491|0.252|2.7026,
             # random 0.490|0.252|2.6994 passes, confidence
             # 0.421|0.289|2.5253 fails. Committing on the log p of the drawn
             # symbol is selection on one's own sample, and the error grows
             # with the step count (2.5727 -> 2.4695 at x1 -> x8).
             commit_order: str = "random",
             support: Optional[Sequence[torch.Tensor]] = None,
             mix_logp: Optional[Sequence[torch.Tensor]] = None,
             mix_w: float = 0.0,
             quota: Optional[Sequence[Optional[torch.Tensor]]] = None,
             device: Optional[torch.device] = None,
             generator: Optional[torch.Generator] = None,
             trace: Optional[List[Dict[str, object]]] = None
             ) -> List[torch.Tensor]:
    """Coarse-to-fine confidence sampling. -> indices, ready for the decoder.

    Levels run coarsest first. A level not yet reached is MASK to the prior
    ("unknown") and would be the null code to the decoder ("no refinement") --
    two different objects, and conflating them is the obvious bug here, MASK
    ids indexing past the codebook and null codes telling the prior a decision
    has been taken that has not. `ConeLadder.prefix_indices` converts, and is
    what to call to decode a partial state.

    Within a level this is MaskGIT, with the number of cells kept per step
    following the cosine schedule and the step count derived in
    `UnmaskSchedule` from the measured 34 bp dependence range.
    `temperature = 1.0` because the prior is the distribution we want to draw
    from. The Gumbel noise scale `noise_anneal * (1 - t/T) * sd(log p)` is
    measured on the batch, a scale read off the quantity it competes with
    being unable to drift from it. `commit_order` decides which cells commit
    first given the same draws: `confidence` is MaskGIT as published, scoring
    the symbol a cell happened to draw, so a lucky draw from a flat
    distribution outranks the mode of a peaked one; `entropy` scores the
    distribution; `random` is the control for whether order matters.

    `support` is an optional per-level boolean `[n_cells, alphabet]` mask
    applied before temperature and softmax -- the trust region moved inside
    the loop, so a symbol outside it cannot be drawn, committed or walked
    back. The caller builds it, this module not owning a notion of which
    corpus is natural. `guidance = 0.0` is off by default: its strength needs
    calibrating against a fidelity/diversity trade-off not yet measured.
    """
    was = prior.training
    prior.eval()
    geom = prior.geom
    device = device or next(prior.parameters()).device
    _check_generator(generator, device, "generate")
    schedule = schedule or UnmaskSchedule.derive(geom)

    idx = [torch.zeros(batch, int(n), dtype=torch.long, device=device)
           for n in geom.lengths]
    mask = [torch.ones(batch, int(n), dtype=torch.bool, device=device)
            for n in geom.lengths]

    drop_all = torch.ones(batch, dtype=torch.bool, device=device)
    keep_all = torch.zeros(batch, dtype=torch.bool, device=device)

    _quota_hit = 0
    for k in schedule.order:
        n = int(geom.lengths[k])
        T = int(schedule.steps[k])
        for t in range(1, T + 1):
            logits = prior.forward(idx, mask, context, keep_all)[k]
            if guidance and prior.ctx_proj is not None and context is not None:
                uncond = prior.forward(idx, mask, context, drop_all)[k]
                logits = uncond + (1.0 + guidance) * (logits - uncond)
            if quota is not None and quota[k] is not None:
                cap = quota[k].to(logits.device).float().unsqueeze(0)
                done_ = (~mask[k]).unsqueeze(-1)
                cnt = (F.one_hot(idx[k], logits.shape[-1]) * done_).sum(1)
                bl = cnt.float() >= cap
                # A row with every symbol capped keeps none of them.
                bl = bl & ~bl.all(dim=-1, keepdim=True)
                _quota_hit += int(bl.any(dim=-1).sum())
                logits = logits.masked_fill(bl.unsqueeze(1), float("-inf"))
            if support is not None:
                m = support[k].to(logits.device)
                if bool((~m).all(dim=-1).any()):
                    raise ValueError(
                        "support mask leaves level %d with a position that "
                        "allows no symbol at all; a masked-out row would make "
                        "log_softmax return nan and multinomial raise, and a "
                        "silently renormalised row would not be the region "
                        "the caller asked for." % k)
                logits = logits.masked_fill(~m.unsqueeze(0), float("-inf"))
            logits = logits / max(float(temperature), 1e-6)
            logp = F.log_softmax(logits.float(), dim=-1)
            if mix_logp is not None and mix_w > 0.0:
                ref = mix_logp[k].to(logp.device).float().unsqueeze(0)
                logp = F.log_softmax(
                    (1.0 - float(mix_w)) * logp + float(mix_w) * ref, dim=-1)

            # sample a symbol per cell, then score the cell for ordering
            probs = logp.exp().reshape(-1, logits.shape[-1])
            draw = torch.multinomial(probs, 1, generator=generator)
            draw = draw.reshape(batch, n)
            lp_draw = logp.gather(2, draw.unsqueeze(-1)).squeeze(-1)
            if commit_order == "confidence":
                conf = lp_draw
            elif commit_order == "entropy":
                conf = (logp.exp() * logp).sum(-1)          # -H(p), nats
            elif commit_order == "random":
                conf = torch.rand(batch, n, device=device,
                                  generator=generator)
            else:
                raise ValueError(
                    "commit_order must be confidence|entropy|random, got %r"
                    % (commit_order,))

            # Gumbel noise whose scale is read off the batch, not declared.
            live = mask[k]
            if bool(live.any()):
                sd = float(conf[live].std()) if int(live.sum()) > 1 else 0.0
                scale = noise_anneal * (1.0 - (t - 1) / max(T, 1)) * sd
                if scale > 0.0:
                    u = torch.rand(conf.shape, device=device,
                                   generator=generator).clamp_(1e-9, 1 - 1e-9)
                    conf = conf + scale * (-torch.log(-torch.log(u)))

            # already-committed cells must never be reconsidered
            conf = conf.masked_fill(~live, float("inf"))
            keep_masked = UnmaskSchedule.keep_after(n, t, T) if t < T else 0
            order = conf.argsort(dim=1)                     # ascending
            rank = order.argsort(dim=1)
            still = rank < keep_masked
            commit = live & ~still
            idx[k] = torch.where(commit, draw, idx[k])
            mask[k] = live & ~commit
            if trace is not None:
                trace.append({"level": k, "step": t,
                              "committed": int(commit.sum()),
                              "remaining": int(mask[k].sum()),
                              "mean_logp": float(lp_draw[commit].mean())
                              if int(commit.sum()) else float("nan")})
        if int(mask[k].sum()) != 0:
            raise RuntimeError(
                "level %d finished with %d cells still masked: the cosine "
                "schedule must reach zero at t = T, and a level handed to the "
                "decoder with MASK ids in it would index past the codebook."
                % (k, int(mask[k].sum())))

    if trace is not None:
        trace.append({"quota_row_steps_bound": _quota_hit})
    prior.train(was)
    return idx


@torch.no_grad()
def check_bijection(model, indices: Sequence[torch.Tensor]) -> Dict[str, float]:
    """U2: does `encode_codes(decode_to_onehot(idx))` return `idx`?

    Not a correctness requirement -- `predict_from_codes` pins the code state
    rather than recomputing it, precisely because the fixed point is not the
    identity for generated codes -- but the measurement of how far a generated
    code list moves when round-tripped through the sequence, which is what a
    wet-lab construct does. The per-level agreement fraction is meaningless on
    an untrained model, so it is reported beside the corpus health.
    """
    x = model.decode_to_onehot(indices)
    again = model.encode_codes(x)
    out = {}
    for k, (a, b) in enumerate(zip(indices, again)):
        out["agree_L%d" % k] = float((a.cpu() == b.cpu()).double().mean())
    return out


# The prior that beats the tables
def parent_cell_index(geom: LadderGeometry, level: int) -> torch.Tensor:
    """Cell of `level - 1` covering cell p of `level` (same base pairs)."""
    L, Lp = int(geom.lengths[level]), int(geom.lengths[level - 1])
    p = torch.arange(L)
    return (p.double() * Lp / L).floor().long().clamp_(max=Lp - 1)


class TabledCodePrior(LadderCodePrior):
    """`LadderCodePrior` with the two baseline tables as model parameters.

    On the test orthogroups (16 441 rows, never used for selection), bits per
    scored token in the mixed regime: positional table 2.7078, parent table
    2.6970, `TabledCodePrior` 2.6838 -- +0.0240 +- 0.0004 over the positional
    table, 61 sigma paired over rows -- while the plain `LadderCodePrior`
    loses to that table by 0.051.

    That loss was structural, not a capacity story. The bar is a per
    (level, position) lookup with 53 150 fitted entries, while
    `_BpPositionEncoding` is a sinusoid through one shared `nn.Linear` and the
    head is one `nn.Linear` per level shared by every position, so the model
    had no per-position output parameter at all and `out_norm` normalises away
    the direction it would need. A size sweep at fixed data and steps shows it
    losing at every size (0.24 M by 0.079, 0.87 M by 0.106, 2.2 M by 0.051);
    with the tables the curve runs the other way, +0.038 / +0.050 / +0.057.

    `pos_bias` is logits[k][position, code] initialised at the train
    positional log-marginal, so at step 0 the model is exactly that table
    (|M - P| = 0.00e+00) and training reads as the residual over it.
    `par_bias` is logits[k][parent code, code] initialised at
    `log p(code | parent) - log p(code)`, with one extra row for a masked
    parent held at zero so the term switches itself off when the parent is
    unreadable; level 0 has no parent. Both are exempt from weight decay,
    which would pull a log-probability table toward uniform every step.

    The head is zero-initialised when either table is on, which makes the
    step-0 identity exact and leaves the trunk without gradient on step 1
    only -- measured 9.25e-04 at step 20 on the V4 arm.
    """

    def __init__(self, geom: LadderGeometry, *args,
                 pos_bias: bool = False, par_bias: bool = False,
                 **kwargs) -> None:
        super().__init__(geom, *args, **kwargs)
        self.use_pos, self.use_par = bool(pos_bias), bool(par_bias)
        if self.use_pos:
            self.pos_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(int(n), int(a)))
                for n, a in zip(geom.lengths, geom.alphabets)])
        if self.use_par:
            mods = [nn.Embedding(1, 1)]          # level 0 has no parent
            for k in range(1, geom.n_levels):
                mods.append(nn.Embedding(int(geom.alphabets[k - 1]) + 1,
                                         int(geom.alphabets[k])))
            self.par_bias = nn.ModuleList(mods)
            for k in range(1, geom.n_levels):
                self.register_buffer("par_map_%d" % k,
                                     parent_cell_index(geom, k),
                                     persistent=False)

    # Building
    @staticmethod
    def fit_tables(indices: Sequence[torch.Tensor], geom: LadderGeometry,
                   alpha: float = 0.5):
        """-> (global log p, positional log p, parent counts), on train rows.

        Krichevsky-Trofimov alpha = 0.5, the smoothing `MarginalCodeTable`
        uses, so the initialisation and the baseline it has to beat are the
        same estimator and a win cannot come from a smoothing difference.
        """
        glob, pos = [], []
        for k, a in enumerate(geom.alphabets):
            a = int(a)
            t = indices[k].long()
            c = torch.bincount(t.reshape(-1), minlength=a).double() + alpha
            glob.append((c / c.sum()).log().float())
            L = int(geom.lengths[k])
            pc = torch.zeros(L, a, dtype=torch.float64)
            for p in range(L):
                pc[p] = torch.bincount(t[:, p], minlength=a).double()
            pc += alpha
            pos.append((pc / pc.sum(1, keepdim=True)).log().float())
        pcnt = [None]
        for k in range(1, geom.n_levels):
            a, ap = int(geom.alphabets[k]), int(geom.alphabets[k - 1])
            par = parent_cell_index(geom, k)
            child = indices[k].long()
            parent = indices[k - 1].long()[:, par]
            flat = parent.reshape(-1) * a + child.reshape(-1)
            pcnt.append(torch.bincount(flat, minlength=ap * a)
                        .reshape(ap, a).double())
        return glob, pos, pcnt

    @torch.no_grad()
    def init_from_tables(self, glob, pos, pcnt) -> "TabledCodePrior":
        dev = next(self.parameters()).device
        if self.use_pos:
            for k in range(self.geom.n_levels):
                self.pos_bias[k].copy_(pos[k].to(dev))
        if self.use_par:
            for k in range(1, self.geom.n_levels):
                a = int(self.geom.alphabets[k])
                c = pcnt[k].to(dev).float() + 0.5
                lp = (c / c.sum(1, keepdim=True)).log()
                self.par_bias[k].weight[:-1].copy_(
                    lp - glob[k].to(dev).view(1, a))
                self.par_bias[k].weight[-1].zero_()
        if self.use_pos or self.use_par:
            for h in self.head:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        return self

    def table_parameters(self) -> List[nn.Parameter]:
        """The tensors that must not receive weight decay."""
        out: List[nn.Parameter] = []
        if self.use_pos:
            out += list(self.pos_bias)
        if self.use_par:
            out += [p for m in self.par_bias for p in m.parameters()]
        return out

    def network_parameters(self) -> List[nn.Parameter]:
        ids = {id(p) for p in self.table_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def param_groups(self, weight_decay: float = 0.01,
                     table_lr: Optional[float] = None) -> List[Dict]:
        """Optimiser groups with the tables exempt from decay.

        Partitioned by identity, not by name: a no-decay rule written as a
        name pattern once released `maska_wektor` and not `wektor_maski`.
        """
        tab = self.table_parameters()
        rest = self.network_parameters()
        if len(tab) + len(rest) != len(list(self.parameters())):
            raise RuntimeError("param_groups does not cover every parameter")
        g = [{"params": rest, "weight_decay": float(weight_decay)}]
        if tab:
            gt = {"params": tab, "weight_decay": 0.0}
            if table_lr is not None:
                gt["lr"] = float(table_lr)
            g.append(gt)
        return g

    # Forward
    def forward(self, indices, masks=None, context=None, drop_context=None):
        out = super().forward(indices, masks, context, drop_context)
        res = []
        for k, o in enumerate(out):
            if self.use_pos:
                o = o + self.pos_bias[k].unsqueeze(0)
            if self.use_par and k > 0:
                pm = getattr(self, "par_map_%d" % k)
                pc = indices[k - 1].long()[:, pm]
                if masks is not None:
                    vis = ~masks[k - 1][:, pm]
                    pc = torch.where(vis, pc, torch.full_like(
                        pc, int(self.geom.alphabets[k - 1])))
                o = o + self.par_bias[k](pc)
            res.append(o)
        return res

    # Loading
    @staticmethod
    def from_checkpoint(path: str, device="cpu") -> "TabledCodePrior":
        """Rebuild from a `prior_lab2.py` checkpoint, shapes and all."""
        b = torch.load(path, map_location="cpu", weights_only=False)
        g = b["geom"]
        geom = LadderGeometry(lengths=tuple(g["lengths"]),
                              alphabets=tuple(g["alphabets"]),
                              seq_len=int(g["seq_len"]))
        m = TabledCodePrior(geom, d_model=int(b["d_model"]),
                            depth=int(b["depth"]), heads=int(b["heads"]),
                            d_ff=int(b["d_ff"]), ctx_dim=int(b["ctx_dim"]),
                            ctx_tokens=int(b["ctx_tokens"]),
                            pos_bias=bool(b.get("pos_bias", False)),
                            par_bias=bool(b.get("par_bias", False)))
        m.load_state_dict(b["prior"])
        return m.to(device).eval()
