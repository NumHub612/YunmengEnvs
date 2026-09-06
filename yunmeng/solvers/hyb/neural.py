# -*- encoding: utf-8 -*-
"""
NeuralCorrectionOperator: AI+ form A (residual correction).

    u(t+dt) = PhysicsStep(u) + dt * N_theta(u, du/dx, du/dy)

The core is a small CNN over the (nx, ny) field. The last layer is
zero-initialized so the hybrid solver starts EXACTLY as the pure-physics
solver (safe degradation; training only adds what physics missed).

Design rulings implemented (v2.0 §4.3):
- forward(datahub, t, dt) is the ONLY evaluation entry; the network's
  forward lives in `self.core` (call op.core(x) for raw net eval);
- get/set_parameters carry theta incl. buffers; NO file IO here --
  persistence is the artifact store's job;
- set_mode switches nn.Module.train/eval.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from yunmeng.interfaces import ModelRef, OperatorResult, RunMode
from yunmeng.numerics.fields import Field, FieldMeta


class _CorrectionNet(nn.Module):
    """3-layer 3x3 CNN: in (u, dudx, dudy) -> out correction tendency."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        # zero-init the output layer: N_theta == 0 at start
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class NeuralCorrectionOperator(nn.Module):
    """Neural correction operator (nn.Module + IOperator + IParameterized
    + IModeSwitchable)."""

    kind = "nn.correction"
    differentiable = True

    def __init__(
        self,
        name: str = "correction",
        hidden: int = 16,
        model_ref: ModelRef | None = None,
    ):
        super().__init__()
        self.name = name
        self.model_ref = model_ref  # config-injected; resolved by model layer
        self.core = _CorrectionNet(hidden).to(torch.float64)
        self._grid = None
        self._mode = RunMode.EVAL

    # NOTE: nn.Module.__call__ would bypass IOperator.forward semantics;
    # the single evaluation entry is forward(datahub, t, dt) below.

    # -- IOperator ------------------------------------

    def build(self, grid, backend) -> None:
        """Static topology: record shapes; weights NOT loaded here."""
        if (
            not backend.__name__ == "torch"
            and getattr(backend, "__name__", "") != "torch"
        ):
            # backend may be the torch module itself
            if backend is not torch:
                raise RuntimeError(
                    "NeuralCorrectionOperator requires the torch backend."
                )
        self._grid = grid

    def forward(self, datahub, t: float, dt: float) -> OperatorResult:
        if self._grid is None:
            raise RuntimeError(f"Operator {self.name}: build() not called.")
        g = self._grid
        u = datahub.field("u").data  # (nx, ny) torch tensor, on graph in TRAIN

        dudx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2.0 * g.dx)
        dudy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * g.dy)

        x = torch.stack([u, dudx, dudy], dim=0).unsqueeze(0)  # (1, 3, nx, ny)
        corr = self.core(x).squeeze(0).squeeze(0)  # (nx, ny)
        field = Field(corr, FieldMeta(name="nn_correction", nx=g.nx, ny=g.ny))
        return OperatorResult(explicit=field)

    # -- IParameterized ---------------------------------

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items()}

    def set_parameters(self, params: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(
            {k: torch.as_tensor(v, dtype=torch.float64) for k, v in params.items()}
        )

    # -- IModeSwitchable ----------------------------------

    def set_mode(self, mode: RunMode) -> None:
        self._mode = mode
        self.train(mode is RunMode.TRAIN)
