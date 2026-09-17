# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Operators for the hybrid-validation demo:
  - FdmDiffusionOperator     : physics term (backward-Euler diffusion)
  - NeuralCorrectionOperator : neural term (local-stencil residual corrector)

Both satisfy IOperator. The neural operator additionally implements
IParameterized + IModeSwitchable.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from yunmeng.interfaces.solver.IOperator import OperatorKinds, OperatorResult
from yunmeng.interfaces.supports.backend import IBackend
from yunmeng.interfaces.supports.datahub import IDataHub
from yunmeng.interfaces.supports.field import DataProduct, FieldMeta, IField
from yunmeng.interfaces.supports.mesh import IMesh
from yunmeng.interfaces.types import ArrayLike, ElementType, RunMode, VariableType
from yunmeng.solvers.hyb.primitives import Field, LinearEqs
from yunmeng.interfaces.solver.IOperator import (
    IOperator,
)
from yunmeng.interfaces.capabilities import IParameterized, IModeSwitchable
import torch
import torch.nn as nn

# ---------------------------------------------------
# region Physics operator
# ---------------------------------------------------


class FdmDiffusionOperator:
    """Backward-Euler diffusion on a uniform 1D mesh:  (I - dt*nu*L) u' = rhs.

    Physics (mechanistic) term. Implicit part only; differentiable when the
    backend is (the Laplacian is a constant stencil, the solve carries AD).
    """

    def __init__(self, field: str, nu: float):
        self._field = field
        self._nu = float(nu)
        self._n = 0
        self._dx = 0.0
        self._lap: ArrayLike | None = (
            None  # (n, n) Laplacian, identity rows at BC cells
        )
        self._backend: IBackend | None = None
        self._interior: ArrayLike | None = None

    # -- metadata -----------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "FdmDiffusion"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.MATH_LAPLACIAN

    @property
    def explicit_part(self) -> bool:
        return False

    @property
    def implicit_part(self) -> bool:
        return True

    @property
    def differentiable(self) -> bool:
        return True

    @classmethod
    def produces(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return [DataProduct(name=f"laplacian:{fname}", loc=loc)]

    @classmethod
    def consumes(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return []

    @property
    def target_fields(self) -> Sequence[str]:
        return (self._field,)

    @property
    def time_order(self) -> int:
        return 1

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: assemble the Laplacian stencil once per mesh."""
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        self._dx = mesh.dx
        topo = mesh.get_topo_assistant()
        lap = np.zeros((self._n, self._n), dtype="float64")
        interior = []
        for c in range(self._n):
            nb = topo.neighbors(c, ElementType.CELL)
            if len(nb) < 2:
                continue  # boundary cell: identity row (Dirichlet by solver)
            lap[c, c] = -2.0 / self._dx**2
            for j in nb:
                lap[c, j] += 1.0 / self._dx**2
            interior.append(c)
        self._interior = np.asarray(interior, dtype="int64")
        self._lap = backend.asarray(lap)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """Runtime: A = I - dt*nu*L; rhs rows at BC cells hold the already
        scatter-written constrained values (identity rows)."""
        u = datahub.get_field(self._field).values
        xp_eye = self._backend.asarray(np.eye(self._n))
        a = xp_eye - dt * self._nu * self._lap
        rhs = self._backend.asarray(u).copy() if hasattr(u, "copy") else u.clone()
        # Interior rows: backward-Euler rhs is u_old; BC rows: identity * u_bc.
        return OperatorResult(
            explicit=None,
            implicit=LinearEqs(a, rhs, self._backend),
        )

    def __repr__(self) -> str:
        return f"FdmDiffusionOperator(field={self._field!r}, nu={self._nu})"


# ---------------------------------------------------
# region Neural operator
# ---------------------------------------------------


class _StencilNet(nn.Module):
    """MLP over the local 3-point stencil -> scalar correction tendency."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the output layer: the hybrid starts as pure physics.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


class NeuralCorrectionOperator(IParameterized, IModeSwitchable, IOperator):
    """Neural correction term (kind: nn.correction).

    Reads the target field from the DataHub, computes a per-cell correction
    tendency from the local stencil, publishes it as an explicit product and
    returns it as the explicit contribution. TRAIN contract: fresh output
    tensors every forward; parameters stay on the graph.
    """

    def __init__(self, field: str, hidden: int = 16, device: str = "cpu"):
        self._field = field
        self._mode = RunMode.EVAL
        self._n = 0
        self._dx = 0.0
        self._backend: IBackend | None = None
        self._device = torch.device(device)
        self._net = _StencilNet(hidden).to(device=self._device, dtype=torch.float64)
        self._interior_idx: "torch.Tensor | None" = None

    # -- metadata -----------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "NeuralCorrection"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.NN_CLOSURE

    @property
    def explicit_part(self) -> bool:
        return True

    @property
    def implicit_part(self) -> bool:
        return False

    @property
    def differentiable(self) -> bool:
        return True

    @classmethod
    def produces(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return [DataProduct(name=f"correction:{fname}", loc=loc)]

    @classmethod
    def consumes(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        return []

    @property
    def target_fields(self) -> Sequence[str]:
        return (self._field,)

    @property
    def time_order(self) -> int:
        return 1

    # -- capability: IParameterized -----------------

    def get_parameters(self) -> dict[str, ArrayLike]:
        return {name: p.detach().clone() for name, p in self._net.named_parameters()}

    def set_parameters(self, params: Mapping[str, ArrayLike]):
        with torch.no_grad():
            for name, p in self._net.named_parameters():
                if name not in params:
                    raise KeyError(f"missing parameter {name!r}")
                val = params[name]
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(np.asarray(val), dtype=torch.float64)
                p.copy_(val.to(device=self._device, dtype=torch.float64))

    # -- capability: IModeSwitchable ----------------

    def set_mode(self, mode: RunMode) -> None:
        self._mode = mode
        if mode == RunMode.TRAIN:
            self._net.train()
        else:
            self._net.eval()

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: record sizes, spacing and interior indices; θ untouched."""
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        self._dx = mesh.dx
        self._interior_idx = torch.arange(1, self._n - 1, device=self._device)

    def _as_torch(self, u: ArrayLike) -> "torch.Tensor":
        if isinstance(u, torch.Tensor):
            return u.to(device=self._device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(u), dtype=torch.float64, device=self._device)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """Neural closure: corr_i = kappa_i * Lap(u)_i, where the MLP
        predicts a per-cell effective-diffusivity increment kappa from the
        local stencil. Flat fields get exactly zero correction, so the
        learned closure extrapolates across decay phases."""
        u_field = datahub.get_field(self._field)
        u = self._as_torch(u_field.values)

        idx = self._interior_idx
        lap = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / self._dx**2
        stencil = torch.stack([u[idx - 1], u[idx], u[idx + 1]], dim=-1)
        kappa = self._net(stencil)
        corr_interior = kappa * lap
        corr = torch.zeros_like(u)
        corr[idx] = corr_interior

        if isinstance(u_field.values, torch.Tensor):
            out = corr  # stays on the graph (TRAIN)
        else:
            out = corr.detach().cpu().numpy()

        meta = FieldMeta(
            name=f"correction:{self._field}",
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
        )
        result_field = Field(meta, out)
        datahub.publish(f"correction:{self._field}", result_field, t)
        return OperatorResult(explicit=result_field, implicit=None)

    # -- parameter vector helpers (used by solver-level IEstimable) ---

    def torch_parameters(self) -> list["torch.nn.Parameter"]:
        return list(self._net.parameters())

    def __repr__(self) -> str:
        return (
            f"NeuralCorrectionOperator(field={self._field!r}, mode={self._mode.value})"
        )
