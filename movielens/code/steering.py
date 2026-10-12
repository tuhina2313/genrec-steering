"""Temporary activation hooks, strength scaling, layer selection helpers, random and null controls."""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Optional

import numpy as np
import torch

from contracts import Direction, Intervention
from utils import array_hash, hash_uint


def resolve_layers(fractions: Iterable[float], num_layers: int) -> list[int]:
    """index = ceil(fraction * num_layers) - 1, deduplicated, zero-based, in depth order."""
    out = []
    for f in fractions:
        if not (0 < f <= 1):
            raise ValueError(f"layer fraction {f} outside (0, 1]")
        idx = int(math.ceil(f * num_layers)) - 1
        if idx not in out:
            out.append(idx)
    return sorted(out)


def make_intervention(direction: Direction, strength: float, kind: str = "add_block") -> Intervention:
    return Intervention(kind=kind, layer_index=direction.layer_index, direction_id=direction.direction_id,
                        vector_sha=array_hash(np.asarray(direction.vector, dtype=np.float32)),
                        strength=float(strength), radius=float(direction.radius))


def delta_vector(direction_vector: np.ndarray, intervention: Intervention) -> np.ndarray:
    """h' = h + strength * radius_l * unit_direction."""
    return (intervention.strength * intervention.radius * np.asarray(direction_vector, dtype=np.float32)).astype(np.float32)


def random_unit_vector(dim: int, seed: int, layer: int, tag: str = "random") -> np.ndarray:
    rng = np.random.default_rng(hash_uint(seed, tag, layer) % (2**63))
    v = rng.standard_normal(dim).astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


class PositionRef:
    """Mutable holder for the current microbatch's last-valid positions."""

    def __init__(self):
        self.positions: Optional[torch.Tensor] = None


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, hs):
    if isinstance(out, tuple):
        return (hs,) + tuple(out[1:])
    return hs


@contextmanager
def capture_hooks(layers_modules: dict[int, torch.nn.Module], pos_ref: PositionRef, store: dict[int, list]):
    """Store only the last valid prompt-position vector of each selected block output."""
    handles = []
    try:
        for li, mod in layers_modules.items():
            def hook(m, inp, out, li=li):
                hs = _hidden(out)
                ar = torch.arange(hs.shape[0], device=hs.device)
                store[li].append(hs[ar, pos_ref.positions].detach().float().cpu())
            handles.append(mod.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def intervention_hooks(module: Optional[torch.nn.Module], delta: Optional[np.ndarray], pos_ref: PositionRef):
    """Add `delta` to the module output at the last valid prompt position only.

    For kind=add_block the module is the decoder block (output = residual stream before the next
    block or final norm). For kind=add_postnorm it is the final normalization module. The hook is
    removed on exit, including on exceptions, so it can never accumulate across batches.
    """
    if module is None or delta is None:
        yield
        return
    d_cpu = torch.as_tensor(delta, dtype=torch.float32)
    handle = None
    try:
        def hook(m, inp, out):
            hs = _hidden(out).clone()
            ar = torch.arange(hs.shape[0], device=hs.device)
            d = d_cpu.to(device=hs.device, dtype=hs.dtype)
            hs[ar, pos_ref.positions] = hs[ar, pos_ref.positions] + d
            return _replace_hidden(out, hs)
        handle = module.register_forward_hook(hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


def count_hooks(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
