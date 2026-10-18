"""Model layer: encoder, cone ladder, quantisers, decoder, latent masking.

    encoder.PromoterEncoder     [B, 4, 800] -> five maps, COARSEST FIRST
    quantizer.HyperbolicCodebook   level 0: quantises a POSITION on H^n
    quantizer.ScalarFSQ            levels >= 1: quantise a TANGENT REFINEMENT
    ladder.ConeLadder           composes them GEODESICALLY
    masking.LatentMask          views of the latent for the decoder
    decoder.CodeDecoder         codes ONLY -> logits [B, 4, 800]

The imports are lazy because an `__init__` that imports every sibling eagerly
makes the whole package unimportable when one of them is absent or broken.
Modules are declared in `_EXPORTS` and resolved on first attribute access
(PEP 562), so a missing name fails at the point of use naming the module it
came from. `quantizer` is eager, its symbols appearing in the others'
signatures.

`NO_DECAY_SUBSTRINGS` and `collect_manifold_state_names` are re-exported
deliberately: decay-exemption lists match a SUBSTRING of the parameter name,
so a rename such as `scale_in` -> `in_scaling` silently moves a parameter into
the decayed group, where lambda 0.0762 over 34 epochs is a multiplier of
1.8e-5.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

from .quantizer import (
    NO_DECAY_SUBSTRINGS,
    HyperbolicCodebook,
    ScalarFSQ,
    code_audit,
    collect_manifold_state_names,
    grid_bits,
    grid_size,
    levels_for_bits,
    no_codebook_update,
    reset_statistics,
)

#: Public name -> the submodule defining it: the package's declared contract,
#: which `missing_modules()` compares against what is on disk.
_EXPORTS: Dict[str, str] = {
    "HyppeModel": "assemble",
    "AssemblyNote": "assemble",
    "subcategory_names": "assemble",
    "ConeLadder": "ladder",
    "LadderOutput": "ladder",
    "PromoterEncoder": "encoder",
    "CodeDecoder": "decoder",
    "LatentMask": "masking",
    "Motif": "motifs",
    "MemeMotifBank": "motifs",
    "DeNovoMotifBank": "motifs",
    "MotifOccupancy": "motifs",
    "read_meme": "motifs",
}


def __getattr__(name: str) -> Any:
    """Resolve a declared export on first use (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    try:
        module = importlib.import_module("." + module_name, __name__)
    except ImportError as exc:
        raise AttributeError(
            "%r is defined in hyppe.model.%s, which could not be imported: %s"
            % (name, module_name, exc)
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS))


def missing_modules() -> List[Tuple[str, str]]:
    """Declared exports whose defining module is absent or broken.

    Returns ``[(export_name, reason), ...]``, empty when the package is whole.
    Use it in a smoke gate: a run started on a half-wired package fails later
    and further away than it should.
    """
    broken: List[Tuple[str, str]] = []
    for name, module_name in sorted(_EXPORTS.items()):
        try:
            module = importlib.import_module("." + module_name, __name__)
        except ImportError as exc:
            broken.append((name, "hyppe.model.%s: %s" % (module_name, exc)))
            continue
        if not hasattr(module, name):
            broken.append(
                (name, "hyppe.model.%s does not define %s" % (module_name, name))
            )
    return broken


__all__ = [
    "NO_DECAY_SUBSTRINGS",
    "HyperbolicCodebook",
    "ScalarFSQ",
    "levels_for_bits",
    "grid_bits",
    "grid_size",
    "code_audit",
    "no_codebook_update",
    "reset_statistics",
    "collect_manifold_state_names",
    "missing_modules",
] + sorted(_EXPORTS)
