"""Motif occupancy: frozen MEME profiles plus a learned de-novo bank.

Occupancy is a deterministic function of the sequence, so ``H(occupancy|x) =
0`` and these channels add no information; they change the inductive bias.

:class:`MemeMotifBank` is frozen and known, 252 Trichoderma profiles from
``data/motifs.meme``. :class:`DeNovoMotifBank` is learned and meant to cover
what those do not, so it is kept sparse, kept orthogonal to the frozen bank,
and given a per-filter learned width whose saturation is the falsifier that
the width knob is decoration.

Both banks report at the CENTRE of a match. Reporting at the 5' end shifts
each channel by its own ``w-1``, and with dynamic widths every filter shifts
differently, so the channels stop being comparable position by position;
:meth:`MemeMotifBank.assert_centred` makes that checkable.

Both strands are scanned while the sequence itself is never canonicalised or
RC-augmented -- a promoter has a direction, but transcription-factor sites
occur on either strand, and conflating those two symmetries has cost
measurements before. The loader asserts ``ALPHABET= ACGT`` rather than
assuming it, because an A,C,T,G ordering silently maps A->G and C->T.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUCLEOTIDES = "ACGT"
# Complement as an index permutation: A<->T, C<->G under the ACGT ordering.
_COMPLEMENT_INDEX = (3, 2, 1, 0)

# Max of two independent N(mu, sd). Both strands of a profile have the same
# background moments, so these two constants carry single-strand moments
# through the `torch.maximum(f, r)` in both scanners. An idealisation, since
# the strand scores are not independent for a near-palindromic profile; what
# matters is the spread left behind, reported by `MotifOccupancy.scale_report`.
_MAX_SHIFT = 1.0 / math.sqrt(math.pi)
_MAX_SCALE = math.sqrt(1.0 - 1.0 / math.pi)
#: Floor on the per-profile sd, in log-odds units; without it a flat filter is
#: handed an unbounded gain. The smallest of the 252 frozen background sds is
#: 0.500, so this can only ever bite a collapsed de-novo filter.
_SD_FLOOR = 1e-2


def background_log_odds_moments(log_odds: torch.Tensor):
    """Mean and sd of the best-of-strands score under a uniform background.

    `log_odds` is [K, 4, W], the tensor the scanner convolves with. The
    background is uniform and i.i.d., so columns are independent and the
    single-strand moments are sums over columns; the strand maximum is carried
    through with the two Gaussian constants above.

    The result is a property of the PROFILE: nothing about the sequence being
    scanned enters it, so occupancy at position i does not depend on position
    j -- which is what a per-base encoding needs, not merely per batch.
    """
    v = log_odds.to(torch.float32)
    mu_col = v.mean(dim=1)
    var_col = (v.pow(2).mean(dim=1) - mu_col.pow(2)).clamp_min(0.0)
    mu = mu_col.sum(dim=-1)
    sd = var_col.sum(dim=-1).clamp_min(0.0).sqrt()
    return mu + _MAX_SHIFT * sd, (_MAX_SCALE * sd).clamp_min(_SD_FLOOR)


# MEME I/O
@dataclass
class Motif:
    name: str
    pwm: torch.Tensor          # [4, w] probabilities, columns sum to 1
    nsites: int = 0

    @property
    def width(self) -> int:
        return int(self.pwm.shape[1])

    def information_bits(self) -> float:
        """Total information content, sum over columns of 2 - H(column)."""
        p = self.pwm.clamp_min(1e-9)
        h = -(p * p.log2()).sum(dim=0)
        return float((2.0 - h).sum())


def read_meme(path: str) -> List[Motif]:
    """Parse a MEME v4 file. Asserts the alphabet rather than assuming it."""
    motifs: List[Motif] = []
    alphabet: Optional[str] = None
    name: Optional[str] = None
    rows: List[List[float]] = []
    nsites = 0
    expect = 0

    def flush() -> None:
        nonlocal name, rows, nsites
        if name is not None and rows:
            m = torch.tensor(rows, dtype=torch.float32).t().contiguous()
            m = m.clamp_min(0.0)
            m = m / m.sum(dim=0, keepdim=True).clamp_min(1e-9)
            motifs.append(Motif(name=name, pwm=m, nsites=nsites))
        name, rows, nsites = None, [], 0

    for line in io.open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("ALPHABET"):
            alphabet = s.split("=", 1)[1].strip()
            if alphabet != NUCLEOTIDES:
                raise ValueError(
                    "motif file declares ALPHABET=%r but this project uses %r; "
                    "a silent remap here maps A->G and C->T and every "
                    "measurement downstream becomes wrong"
                    % (alphabet, NUCLEOTIDES))
            continue
        if s.startswith("MOTIF"):
            flush()
            parts = s.split()
            name = parts[1] if len(parts) > 1 else "motif_%d" % len(motifs)
            expect = 0
            continue
        if s.startswith("letter-probability"):
            # "letter-probability matrix: alength= 4 w= 8 nsites= 20 E= 0"
            toks = s.replace("=", " = ").split()
            expect, nsites = 0, 0
            for i, t in enumerate(toks):
                if t == "=" and i >= 1 and i + 1 < len(toks):
                    key, val = toks[i - 1], toks[i + 1]
                    try:
                        if key == "w":
                            expect = int(float(val))
                        elif key == "nsites":
                            nsites = int(float(val))
                    except ValueError:
                        pass
            continue
        if name is not None and s:
            # A matrix row is four floats and nothing else.
            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue
            if len(vals) == 4:
                rows.append(vals)
                if expect and len(rows) >= expect:
                    flush()
    flush()
    if alphabet is None:
        raise ValueError("no ALPHABET line in %s -- refusing to guess" % path)
    return motifs


# Frozen scanner
class MemeMotifBank(nn.Module):
    """Frozen log-odds scan of known profiles, both strands, centred.

    The output is the per-position best-of-strands log-odds score, so a channel
    is directly comparable between filters of different widths -- the property
    that makes the de-novo bank's dynamic widths safe.
    """

    def __init__(self, motifs: Sequence[Motif], background: float = 0.25,
                 centred: bool = True, pseudocount: float = 1e-3) -> None:
        super().__init__()
        if not motifs:
            raise ValueError("empty motif bank")
        self.names = [m.name for m in motifs]
        self.widths = [m.width for m in motifs]
        self.centred = bool(centred)
        # The padded width is forced odd. With an even W the kernel's centre
        # sits at a half position while conv1d's padding must be the integer
        # W//2, so every channel comes back shifted by one base -- 12 % of a
        # median 9 bp motif, and measured as max_peak_offset_bp = 1.0.
        w_max = max(self.widths)
        if w_max % 2 == 0:
            w_max += 1
        self.w_max = int(w_max)
        self.n_even_width = int(sum(1 for x in self.widths if x % 2 == 0))
        # Pad every profile to the same width with background columns, which
        # contribute exactly zero log-odds and therefore change no score.
        fwd = torch.zeros(len(motifs), 4, w_max)
        mask = torch.zeros(len(motifs), 1, w_max)
        for i, m in enumerate(motifs):
            p = (m.pwm + pseudocount)
            p = p / p.sum(dim=0, keepdim=True)
            lo = (p / background).log()
            off = (w_max - m.width) // 2 if self.centred else 0
            fwd[i, :, off:off + m.width] = lo
            mask[i, 0, off:off + m.width] = 1.0
        self.register_buffer("weight_fwd", fwd)
        self.register_buffer("weight_rev", fwd[:, _COMPLEMENT_INDEX].flip(-1))
        self.register_buffer("support", mask)
        self.register_buffer(
            "info_bits", torch.tensor([m.information_bits() for m in motifs]))
        # Fixed per-profile scale, deliberately non-persistent: it is derived
        # from `weight_fwd`, and a persistent copy could be loaded next to a
        # different `weight_fwd` and disagree with it silently.
        _bmu, _bsd = background_log_odds_moments(fwd)
        self.register_buffer("bg_mu", _bmu, persistent=False)
        self.register_buffer("bg_sd", _bsd, persistent=False)
        # Unpadded log-odds, grouped by width. The padded form cannot be used
        # for a similarity measure: padding columns are exactly zero, so a
        # correlation over the padded width is dominated by matching zeros.
        # With that wrong, an exact copy of a frozen profile scored 0.7013
        # while a random filter scored 0.7801.
        by_width: Dict[int, List[int]] = {}
        for i, m in enumerate(motifs):
            by_width.setdefault(m.width, []).append(i)
        self._widths_present = sorted(by_width)
        for wv in self._widths_present:
            idx = by_width[wv]
            stack = torch.stack([
                ((motifs[i].pwm + pseudocount)
                 / (motifs[i].pwm + pseudocount).sum(dim=0, keepdim=True)
                 / background).log() for i in idx])
            self.register_buffer("_grp_%d" % wv, stack)
            self.register_buffer("_gidx_%d" % wv,
                                 torch.tensor(idx, dtype=torch.long))
        for p in self.parameters():
            p.requires_grad_(False)

    def background_moments(self):
        """[K], [K] -- the fixed scale of each profile, from the PWM alone."""
        return self.bg_mu, self.bg_sd

    @property
    def n_motifs(self) -> int:
        return len(self.names)

    def width_groups(self, min_information: float = 0.0
                     ) -> List[Tuple[int, torch.Tensor]]:
        """[(width, log_odds [M_w, 4, width])] -- unpadded, for similarity.

        `min_information` drops profiles below that many bits; 93 of the 252
        fall below 6 and 156 below 8. A near-uniform profile correlates
        unstably with everything, so keeping it in a novelty reference spends
        gradient telling the learned filters not to resemble noise.
        """
        out = []
        for wv in self._widths_present:
            grp = getattr(self, "_grp_%d" % wv)
            if min_information <= 0.0:
                out.append((wv, grp))
                continue
            idx = getattr(self, "_gidx_%d" % wv)
            keep = self.info_bits[idx] >= min_information
            if bool(keep.any()):
                out.append((wv, grp[keep]))
        return out

    def n_informative(self, min_information: float) -> int:
        return int((self.info_bits >= min_information).sum())

    def suspect_profiles(self, min_information: float = 8.0
                         ) -> Dict[str, List[str]]:
        """Profiles that are not usable as profiles.

        `consensus_not_pwm`: 2.000 bits per position means a consensus string
        stored in PWM form, which matches within a Hamming radius instead of
        scoring. Exactly one profile here, producing 0.317 hits per promoter.

        `low_information`: below `min_information` bits -- 93 of 252 below 6.
        Excluded from the novelty reference, since correlating against
        near-uniform is meaningless, but kept for scanning, as depletion is
        signal too.

        Not reported: which profiles can never reach a given p-value. That
        threshold is per motif, from its own null distribution, and belongs to
        a scanning layer.
        """
        w = torch.tensor([float(x) for x in self.widths])
        per_pos = self.info_bits / w.clamp_min(1.0)
        return {
            "consensus_not_pwm": [self.names[i]
                                  for i in range(self.n_motifs)
                                  if float(per_pos[i]) > 1.99],
            "low_information": [self.names[i]
                                for i in range(self.n_motifs)
                                if float(self.info_bits[i])
                                < min_information],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, L] one-hot (N as uniform 0.25) -> [B, n_motifs, L]."""
        pad = (self.w_max - 1) // 2
        f = F.conv1d(x, self.weight_fwd, padding=pad)
        r = F.conv1d(x, self.weight_rev, padding=pad)
        out = torch.maximum(f, r)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out

    def assert_centred(self, length: int = 256) -> Dict[str, float]:
        """A single planted site must score at its centre, for every width.

        An off-centre convention shifts each channel by its own ``w-1``, so
        channels stop being comparable position by position.
        """
        dev = self.weight_fwd.device
        worst = 0.0
        for i in range(min(self.n_motifs, 24)):
            w = self.widths[i]
            col = self.weight_fwd[i].argmax(dim=0)
            off = (self.w_max - w) // 2 if self.centred else 0
            x = torch.full((1, 4, length), 0.25, device=dev)
            centre = length // 2
            start = centre - w // 2
            for j in range(w):
                x[0, :, start + j] = 0.0
                x[0, col[off + j], start + j] = 1.0
            peak = int(self.forward(x)[0, i].argmax())
            worst = max(worst, abs(peak - centre))
        return {"max_peak_offset_bp": float(worst),
                "centred": bool(self.centred),
                "n_even_width": self.n_even_width,
                "padded_width": self.w_max}

    def report(self) -> Dict[str, object]:
        w = torch.tensor([float(x) for x in self.widths])
        return {
            "n_motifs": self.n_motifs,
            "width_min": float(w.min()), "width_max": float(w.max()),
            "width_mean": float(w.mean()), "width_median": float(w.median()),
            "info_bits_mean": float(self.info_bits.mean()),
            "info_bits_min": float(self.info_bits.min()),
            "n_low_information": int((self.info_bits < 6.0).sum()),
        }


# Learned de-novo bank
class DeNovoMotifBank(nn.Module):
    """Learned filters with a centred, learned width, kept sparse and novel.

    Each filter owns a scalar `half_width` in kernel positions, and the kernel
    is multiplied by a soft symmetric window around its centre, so the support
    is contiguous, centred and differentiable in its width. The penalties from
    `penalties()` enter the loss as subordinate terms with fixed weights; see
    that method for why they must not become players in a bargain.
    """

    def __init__(self, n_filters: int = 32, max_width: int = 25,
                 init_half_width: float = 4.0, min_half_width: float = 2.0,
                 window_temp: float = 0.7, background: float = 0.25,
                 target_density: float = 0.10,
                 density_frac: float = 0.5, novelty_null: float = 0.85,
                 target_bits: float = 6.0, target_width: float = 10.0,
                 novelty_min_bits: float = 8.0) -> None:
        super().__init__()
        if max_width % 2 == 0:
            raise ValueError("max_width must be odd so the centre is a "
                             "position, not a gap (got %d)" % max_width)
        self.n_filters = int(n_filters)
        self.max_width = int(max_width)
        self.min_half_width = float(min_half_width)
        self.window_temp = float(window_temp)
        self.background = float(background)
        self.target_density = float(target_density)
        # A hit is a position scoring at least `density_frac` of what THIS
        # filter can score. An absolute threshold cannot work, because the
        # readout is free to rescale a channel: a planted motif recovered at
        # accuracy 1.000 reported density 0.000 under a fixed cut of 3.0.
        self.density_frac = float(density_frac)
        self.target_bits = float(target_bits)
        # Parsimony. Widening is free and the information floor pushes width
        # up, so without this term filters saturate at 20.7-23.4 bp of a 25 bp
        # kernel. The default sits just above the frozen median width of 9 bp.
        self.target_width = float(target_width)
        # Measured, not chosen: the similarity is a maximum over 252 profiles
        # x 2 strands x every offset, so its null is an extreme value, not
        # zero -- 512 untrained filters gave mean 0.681, max 0.863, against
        # 1.0000 for an exact copy. Re-measure if the motif file changes.
        self.novelty_null = float(novelty_null)
        # Novelty is measured against the informative profiles only; 8 bits
        # keeps 96 of the 252.
        self.novelty_min_bits = float(novelty_min_bits)

        # Logits over nucleotides, turned into a PWM by softmax, so a de-novo
        # filter is directly comparable with a frozen profile -- which is what
        # makes the orthogonality measure meaningful rather than a dot product
        # between arbitrary conv weights.
        self.logits = nn.Parameter(
            torch.randn(self.n_filters, 4, self.max_width) * 0.3)
        # Softplus-parameterised so the width can never cross zero.
        inv = math.log(math.expm1(max(init_half_width - min_half_width, 1e-3)))
        self.half_width_raw = nn.Parameter(torch.full((self.n_filters,), inv))
        offs = torch.arange(self.max_width, dtype=torch.float32)
        self.register_buffer("offsets", offs - (self.max_width - 1) / 2.0)
        self.register_buffer("_last_density", torch.zeros(self.n_filters))
        self.register_buffer("_last_max_act", torch.zeros(self.n_filters))

    # Geometry
    def half_width(self) -> torch.Tensor:
        return F.softplus(self.half_width_raw) + self.min_half_width

    def window(self) -> torch.Tensor:
        """[K, 1, W] soft symmetric support, 1 at the centre."""
        hw = self.half_width().unsqueeze(-1)
        d = self.offsets.abs().unsqueeze(0)
        return torch.sigmoid((hw - d) / self.window_temp).unsqueeze(1)

    def effective_width(self) -> torch.Tensor:
        """Sum of the window -- the width the model actually chose."""
        return self.window().squeeze(1).sum(dim=-1)

    def pwm(self) -> torch.Tensor:
        """[K, 4, W] probabilities; outside the window it decays to uniform."""
        p = torch.softmax(self.logits, dim=1)
        w = self.window()
        return p * w + self.background * (1.0 - w)

    def log_odds(self) -> torch.Tensor:
        return (self.pwm().clamp_min(1e-6) / self.background).log()

    def max_score(self) -> torch.Tensor:
        """[K] the best score each filter can reach -- its own scale."""
        return self.log_odds().amax(dim=1).sum(dim=-1)

    def background_moments(self):
        """[K], [K] background mean and sd, recomputed from the live weights.

        The frozen bank can cache these; this one cannot, since its profile
        changes at every step. Deliberately not detached: attached, the
        channel is exactly invariant to the filter's amplitude, which leaves
        the scale to the three penalties meant to shape it.
        """
        return background_log_odds_moments(self.log_odds())

    def information_bits(self) -> torch.Tensor:
        """[K] information content, the same measure as the frozen bank.

        Sparsity alone is trivially satisfied by a flat filter, which fires
        nowhere and detects nothing; this is what separates a motif from a
        smoother. The 252 frozen profiles run min 1.07, median 6.70 bits.
        """
        p = self.pwm().clamp_min(1e-9)
        h = -(p * p.log2()).sum(dim=1)
        return (2.0 - h).clamp_min(0.0).sum(dim=-1)

    def _hit_threshold(self) -> torch.Tensor:
        return self.density_frac * self.max_score()

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, L] -> [B, K, L], best of both strands, centred."""
        w = self.log_odds()
        rev = w[:, _COMPLEMENT_INDEX].flip(-1)
        pad = self.max_width // 2
        f = F.conv1d(x, w, padding=pad)
        r = F.conv1d(x, rev, padding=pad)
        out = torch.maximum(f, r)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        if self.training:
            with torch.no_grad():
                thr = self._hit_threshold()
                tmp = (thr.abs() * 0.1).clamp_min(1e-3)
                soft = torch.sigmoid((out - thr.view(1, -1, 1))
                                     / tmp.view(1, -1, 1))
                self._last_density.copy_(soft.mean(dim=(0, 2)))
                # A margin relative to the filter's own cut, so "dead" means
                # "never clears its own bar" rather than "small".
                self._last_max_act.copy_((out.amax(dim=(0, 2)) - thr)
                                         / thr.abs().clamp_min(1e-6))
        return out

    # Penalties
    def sparsity_penalty(self, activation: torch.Tensor) -> torch.Tensor:
        """Penalise only excess density; being sparser than target is free.

        One-sided for a reason: a symmetric penalty would push rare, specific
        filters towards firing more, which is the opposite of the requirement.
        """
        thr = self._hit_threshold()
        tmp = (thr.abs() * 0.1).clamp_min(1e-3)
        soft = torch.sigmoid((activation - thr.view(1, -1, 1))
                             / tmp.view(1, -1, 1))
        density = soft.mean(dim=(0, 2))
        return F.relu(density - self.target_density).pow(2).mean()

    def width_penalty(self) -> torch.Tensor:
        """One-sided cost on width above `target_width`.

        Paired with the information floor the equilibrium is "the narrowest
        filter that still carries target_bits", which is what a motif is.
        Alone it would simply collapse every filter.
        """
        excess = F.relu(self.effective_width() - self.target_width)
        return (excess / max(self.target_width, 1e-6)).pow(2).mean()

    def information_penalty(self) -> torch.Tensor:
        """One-sided floor on information: be a motif, not a blur."""
        deficit = F.relu(self.target_bits - self.information_bits())
        return (deficit / max(self.target_bits, 1e-6)).pow(2).mean()

    def novelty_penalty(self, frozen: "MemeMotifBank",
                        chunk: int = 64) -> torch.Tensor:
        """Penalise similarity above the null, exactly as sparsity does.

        One-sided against `novelty_null` and not against zero: any informative
        8 bp filter has a decent best match among 252 profiles by chance, so a
        penalty on the raw similarity would be constant pressure against being
        motif-like at all rather than against being redundant.
        """
        sim = self.jaspar_similarity(frozen, chunk=chunk)
        return F.relu(sim - self.novelty_null).pow(2).mean()

    def jaspar_similarity(self, frozen: "MemeMotifBank",
                          chunk: int = 64,
                          min_information: Optional[float] = None
                          ) -> torch.Tensor:
        """[K] best similarity of each learned filter to any frozen profile.

        TOMTOM-style: Pearson correlation in log-odds space, over the overlap
        with the frozen profile, maximised over every offset and both strands.
        Three details, each of which produced a wrong number when absent:
        both operands in log-odds space (mixing probabilities against log-odds
        scored an exact copy at 0.7013 against 0.7801 for a random filter);
        normalisation over the overlap rather than the padded kernel (whole-
        kernel norms make this a maximum over ~12 600 near-random alignments,
        with a null at 0.62); and a denominator floor tied to the filter's own
        scale, which sends the 0/0 alignments outside the window to ~0.
        """
        del chunk                       # kept for call compatibility
        a = self.log_odds()                                 # [K, 4, W]
        a = a - a.mean(dim=1, keepdim=True)
        floor = a.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-6) * 0.25
        a_sq = a.pow(2).sum(dim=1, keepdim=True)            # [K, 1, W]
        best = torch.full((self.n_filters,), -1.0, device=a.device,
                          dtype=a.dtype)
        mi = (self.novelty_min_bits if min_information is None
              else min_information)
        for wv, grp in frozen.width_groups(mi):
            if wv > a.shape[-1]:
                continue                # cannot overlap a wider profile fully
            b = grp.to(a.dtype) - grp.to(a.dtype).mean(dim=1, keepdim=True)
            b_norm = b.flatten(1).norm(dim=1).clamp_min(1e-8)   # [M]
            ones = torch.ones(1, 1, wv, device=a.device, dtype=a.dtype)
            win = F.conv1d(a_sq, ones).sqrt()                   # [K, 1, O]
            den = (win * b_norm.view(1, -1, 1)).clamp_min(floor * b_norm.max())
            for bank in (b, b[:, _COMPLEMENT_INDEX].flip(-1)):
                num = F.conv1d(a, bank)                         # [K, M, O]
                best = torch.maximum(best, (num / den).amax(dim=(1, 2)))
        return best.clamp(-1.0, 1.0)

    def penalties(self, activation: torch.Tensor,
                  frozen: Optional["MemeMotifBank"] = None,
                  sparsity_weight: float = 1.0,
                  novelty_weight: float = 1.0,
                  information_weight: float = 1.0,
                  width_weight: float = 1.0
                  ) -> Dict[str, torch.Tensor]:
        """All three subordinate terms. Not players in the Nash bargain.

        A saturated task takes the largest share of a Nash bargain precisely
        because its gradient is smallest. These three sit at exactly zero once
        satisfied, which is the case a bargaining solver handles worst.
        """
        out = {"sparsity": self.sparsity_penalty(activation) * sparsity_weight}
        if information_weight > 0.0:
            out["information"] = (self.information_penalty()
                                  * information_weight)
        if width_weight > 0.0:
            out["width"] = self.width_penalty() * width_weight
        if frozen is not None and novelty_weight > 0.0:
            out["novelty"] = self.novelty_penalty(frozen) * novelty_weight
        out["total"] = sum(out.values())
        return out

    # Falsifiers
    @torch.no_grad()
    def report(self, frozen: Optional["MemeMotifBank"] = None
               ) -> Dict[str, object]:
        """Every requirement on this bank, as a number that can fail.

        width_saturated  fraction of filters pinned at the maximum width. High
                         means the learned width is decoration and this is a
                         fixed-width bank with extra parameters.
        width_collapsed  fraction pinned at the minimum, the opposite failure.
        density_*        a filter above ~0.25 is a background detector: the
                         252 frozen profiles cover 12-25 % of positions.
        dead_filters     never exceeded the threshold in the last forward.
        jaspar_sim_*     redundancy with the frozen bank.
        """
        ew = self.effective_width()
        sat = float((ew > self.max_width - 1.0).float().mean())
        col = float((ew < self.min_half_width * 2.0 + 1.5).float().mean())
        rep: Dict[str, object] = {
            "n_filters": self.n_filters,
            "width_mean": float(ew.mean()), "width_min": float(ew.min()),
            "width_max": float(ew.max()), "width_sd": float(ew.std()),
            "width_saturated": sat, "width_collapsed": col,
            "density_mean": float(self._last_density.mean()),
            "density_max": float(self._last_density.max()),
            "density_over_target": float(
                (self._last_density > self.target_density).float().mean()),
            "dead_filters": float((self._last_max_act < 0.0).float().mean()),
            "info_bits_mean": float(self.information_bits().mean()),
            "info_bits_min": float(self.information_bits().min()),
            "n_below_target_bits": int(
                (self.information_bits() < self.target_bits).sum()),
            "max_score_mean": float(self.max_score().mean()),
            "bits_per_position": float(
                (self.information_bits()
                 / self.effective_width().clamp_min(1.0)).mean()),
            "target_width": self.target_width,
        }
        if frozen is not None:
            sim = self.jaspar_similarity(frozen)
            rep.update({"jaspar_sim_mean": float(sim.mean()),
                        "jaspar_sim_max": float(sim.max()),
                        "jaspar_sim_p90": float(sim.quantile(0.9)),
                        "novelty_null": self.novelty_null,
                        "n_redundant": int((sim > self.novelty_null).sum())})
        return rep

    @torch.no_grad()
    def falsifiers(self, frozen: Optional["MemeMotifBank"] = None
                   ) -> List[Tuple[str, bool, str]]:
        """(name, passed, detail). A check that cannot fail is not a check."""
        r = self.report(frozen)
        out = [
            ("learned width is alive (not all at the maximum)",
             r["width_saturated"] < 0.5,
             "saturated=%.2f, mean width %.1f bp" % (r["width_saturated"],
                                                     r["width_mean"])),
            ("widths differ between filters (sd > 0.5 bp)",
             r["width_sd"] > 0.5, "sd=%.2f" % r["width_sd"]),
            ("width is not pinned at the kernel maximum",
             r["width_mean"] < 0.85 * self.max_width,
             "mean %.2f of max %d; without a parsimony term this measured "
             "20.7-23.4 of 25" % (r["width_mean"], self.max_width)),
            ("filters are sparse (mean density below target)",
             r["density_mean"] <= self.target_density * 1.5,
             "mean=%.3f target=%.3f" % (r["density_mean"],
                                        self.target_density)),
            ("filters are not dead (something clears their own cut)",
             r["dead_filters"] < 0.25, "dead=%.2f" % r["dead_filters"]),
            ("filters are motifs, not blurs (info >= target bits)",
             r["info_bits_mean"] >= self.target_bits,
             "mean %.2f bits, min %.2f, target %.1f (MEME library: min 1.07,"
             " median 6.70)" % (r["info_bits_mean"], r["info_bits_min"],
                                self.target_bits)),
        ]
        if frozen is not None:
            out.append(("de-novo bank is novel (few redundant with MEME)",
                        r["n_redundant"] <= max(1, self.n_filters // 8),
                        "redundant=%d/%d above null %.2f, max sim %.3f"
                        % (r["n_redundant"], self.n_filters,
                           self.novelty_null, r["jaspar_sim_max"])))
        return out


class MotifOccupancy(nn.Module):
    """Frozen bank plus de-novo bank, concatenated into encoder channels.

    The channels must be made comparable: raw per-channel RMS on real
    promoters spans 0.429 to 66.28, because a 22 bp profile carrying 38 bits
    and a 6 bp profile carrying 1.07 bits end up in one tensor.

    `scale="profile"`, the default, divides by the background moments of the
    profile itself, which brings per-channel RMS to min 0.610 / median 1.010 /
    max 1.237 on real promoters. Normalising by width instead starves the
    smallest channel to 0.068 of the median, and by information content leaves
    the channel mean at 0.897, since that is a divisor with no matching centre.

    `scale="sequence"` is the per-sequence z-score, kept for measurement. It
    equalises perfectly and makes occupancy at position i a function of every
    base: flipping one base changes all 800 positions and all 284 channels,
    including at the purely convolutional ladder levels. Outside the
    convolutional receptive field its magnitude is 2.2-2.7 % of the magnitude
    inside, against 0.01-0.04 % with it off -- a leak rather than a dominant
    path, but the largest non-convolutional one there.

    The choice changes the encoder's input distribution, so no checkpoint is
    comparable or fine-tunable across it.
    """

    #: How a channel is made comparable with the other 283.
    SCALES = ("profile", "sequence")

    def __init__(self, frozen: Optional[MemeMotifBank] = None,
                 de_novo: Optional[DeNovoMotifBank] = None,
                 normalise: bool = True,
                 scale: str = "profile") -> None:
        super().__init__()
        if frozen is None and de_novo is None:
            raise ValueError("MotifOccupancy needs at least one bank")
        if str(scale) not in self.SCALES:
            raise ValueError("scale must be one of %s, got %r"
                             % (list(self.SCALES), scale))
        self.frozen = frozen
        self.de_novo = de_novo
        self.normalise = bool(normalise)
        self.scale = str(scale)
        n = (frozen.n_motifs if frozen is not None else 0)
        n += (de_novo.n_filters if de_novo is not None else 0)
        self.n_channels = n

    def _moments(self, device, dtype):
        """[1, M, 1], [1, M, 1] -- the concatenated per-profile scale."""
        mus, sds = [], []
        for bank in (self.frozen, self.de_novo):
            if bank is None:
                continue
            mu, sd = bank.background_moments()
            mus.append(mu)
            sds.append(sd)
        mu = torch.cat(mus) if len(mus) > 1 else mus[0]
        sd = torch.cat(sds) if len(sds) > 1 else sds[0]
        return (mu.to(device=device, dtype=dtype).view(1, -1, 1),
                sd.to(device=device, dtype=dtype).view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.frozen is not None:
            parts.append(self.frozen(x))
        if self.de_novo is not None:
            parts.append(self.de_novo(x))
        occ = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if not self.normalise:
            return occ
        if self.scale == "sequence":
            # Kept bit for bit so the leak can be switched back on for a
            # measurement; see the class docstring for what it costs.
            m = occ.mean(dim=-1, keepdim=True)
            s = occ.std(dim=-1, keepdim=True).clamp_min(1e-5)
            return (occ - m) / s
        mu, sd = self._moments(occ.device, occ.dtype)
        return (occ - mu) / sd

    @torch.no_grad()
    def scale_report(self, x: torch.Tensor):
        """Per-channel RMS of the occupancy this module would emit for `x`.

        `min_over_median` is the starvation observable: a channel far below the
        median reaches the stem multiplied by a weight never trained against
        that magnitude.
        """
        occ = self.forward(x).to(torch.float32)
        rms = occ.pow(2).mean(dim=(0, 2)).sqrt()
        med = float(rms.median())
        return {
            "scale": self.scale,
            "normalise": bool(self.normalise),
            "channels": int(rms.numel()),
            "rms_min": float(rms.min()),
            "rms_p05": float(rms.quantile(0.05)),
            "rms_median": med,
            "rms_p95": float(rms.quantile(0.95)),
            "rms_max": float(rms.max()),
            "max_over_median": float(rms.max()) / max(med, 1e-9),
            "min_over_median": float(rms.min()) / max(med, 1e-9),
            "mean_abs_median": float(occ.mean(dim=(0, 2)).abs().median()),
        }
