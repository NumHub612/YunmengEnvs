# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

AI adapters: bridges between backend-neutral numerics (Field/DataHub/Sample)
and torch-based AI consumers (neural operators, estimators).

This is the ONLY layer besides `Backend` implementations allowed to import
torch. Two extraction paths are provided:

  - graph-preserving (TRAIN): pulls the live tensor out of each FieldShard
    without host roundtrip, keeping autograd connections intact;
  - detached (EVAL): host aggregation + conversion, safe for logging,
    supervised-sample export, and persistence.
"""

from __future__ import annotations
import numpy as np
import torch

from yunmeng.numerics.enums import ElementType, RunMode
from yunmeng.numerics.fields.field import Field
from yunmeng.numerics.fields.datahubs import DataHub, Sample

# ---------------------------------------------------
# region Field <-> tensor
# ---------------------------------------------------


def field_to_tensor(field: "Field", mode: RunMode = RunMode.TRAIN) -> torch.Tensor:
    """Extract a single tensor from a Field.

    TRAIN: requires single-shard torch-backed field; returns the live
    tensor (graph preserved). Raises on multi-shard or numpy backend —
    silently breaking the graph is worse than failing loudly.

    EVAL: host aggregation + conversion (detached by construction).
    """
    if mode == RunMode.EVAL:
        return torch.from_numpy(np.asarray(field.to_numpy()))

    shards = field.field_shards
    if len(shards) != 1:
        raise RuntimeError(
            f"Graph-preserving extraction requires single-shard field, "
            f"got {len(shards)} shards. Use EVAL mode or gather first."
        )
    data = shards[0].data
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(
            f"Graph-preserving extraction requires torch backend, "
            f"got {type(data)}. Use EVAL mode for numpy fields."
        )
    return data


def tensor_to_field(tensor: torch.Tensor, template: "Field") -> "Field":
    """Write a tensor back into a Field sharing `template`'s layout.

    Single-shard only; preserves the autograd graph (no clone/detach).
    """
    from yunmeng.numerics.fields import Field

    shards = template.field_shards
    if len(shards) != 1:
        raise RuntimeError(
            f"tensor_to_field requires single-shard template, "
            f"got {len(shards)} shards."
        )
    out = Field.copy(template) if hasattr(Field, "copy") else template.copy()
    out.field_shards[0].data = tensor
    return out


# ---------------------------------------------------
# region DataHub tensor export
# ---------------------------------------------------


def _sample_to_tensor(sample: "Sample", mode: RunMode) -> torch.Tensor:
    return field_to_tensor(sample.data, mode)


def history_to_tensor_batch(
    hub: "DataHub",
    name: str,
    etype: ElementType,
    levels: int = None,
    mode: RunMode = None,
) -> torch.Tensor:
    """Export field history as stacked tensor (L, N, ...).

    Mode defaults to the hub's own mode: TRAIN keeps the graph
    (torch.stack of live tensors); EVAL returns a detached copy.
    """
    mode = mode or hub.mode
    key = hub._key(name, etype)
    hist = hub._history.get(key)
    if hist is None:
        return None
    samples = hist.samples(levels)
    if not samples:
        return None
    tensors = [_sample_to_tensor(s, mode) for s in samples]
    stacked = torch.stack(tensors, dim=0)
    return stacked if mode == RunMode.TRAIN else stacked.detach()


def make_training_sample(
    hub: "DataHub",
    name: str,
    etype: ElementType,
    input_levels: int = 2,
    mode: RunMode = RunMode.EVAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create (input, target) pair for supervised learning.

    Default EVAL (detached export for offline datasets). Pass TRAIN to
    keep tensors on the graph for online/end-to-end objectives.
    """
    key = hub._key(name, etype)
    hist = hub._history.get(key)
    if hist is None or len(hist) < input_levels + 1:
        return None, None

    target_t = _sample_to_tensor(hist.at(0), mode)

    past = []
    for i in range(1, input_levels + 1):
        s = hist.at(i)
        if s is None:
            return None, None
        past.append(_sample_to_tensor(s, mode))
    input_tensor = torch.stack(past, dim=0)

    if mode == RunMode.EVAL:
        input_tensor = input_tensor.detach()
        target_t = target_t.detach()

    return input_tensor, target_t
