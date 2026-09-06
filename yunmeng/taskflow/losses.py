# -*- encoding: utf-8 -*-
"""Losses (ym_losses short-name registry)."""

from __future__ import annotations

import torch


def rollout_mse(predicted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over an unrolled trajectory.

    Differentiable under TRAIN (v2.0: losses live in the estimation layer,
    never in the solver protocol).
    """
    return torch.mean((predicted - reference) ** 2)
