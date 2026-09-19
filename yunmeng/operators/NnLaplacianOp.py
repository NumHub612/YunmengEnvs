# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

NeuralLaplacianOperator: AI surrogate for the Laplacian (kind nn.surrogate).

Unlike NeuralCorrectionOperator (which modulates the PHYSICS Laplacian by
a learned diffusivity increment kappa), this operator REGRESSES the
Laplacian itself from the local stencil:

    lap_hat_i = NN( [u_{i-1}, u_i, u_{i+1}] / u_scale , dx / dx_ref )

and returns it as an explicit tendency contribution. Composed with an
implicit physics operator it acts as a residual surrogate
(du/dt = nu*L_fvm u + lap_hat); composed alone it replaces the
discretized Laplacian entirely (du/dt = lap_hat).

Design contract:
- explicit part only; publishes `laplacian:<field>` as a DataProduct.
- TRAIN contract: fresh output tensors every forward, parameters stay on
  the graph; EVAL detaches to host when the field backend is numpy.
- Zero-initialized output layer: at init the surrogate contributes
  exactly zero, so a hybrid starts as pure physics.
- implements IParameterized + IModeSwitchable (+ IDifferentiable leaves
  via torch_parameters).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from yunmeng.interfaces.capabilities import (
    ParamMeta,
    IDifferentiable,
    IModeSwitchable,
    IParameterized,
)
from yunmeng.interfaces.solver.IOperator import (
    IOperator,
    OperatorKinds,
    OperatorResult,
)
from yunmeng.interfaces.supports.backend import IBackend
from yunmeng.interfaces.supports.datahub import IDataHub
from yunmeng.interfaces.supports import DataProduct, FieldMeta
from yunmeng.interfaces.supports.mesh import IMesh
from yunmeng.interfaces.types import ArrayLike, ElementType, RunMode, VariableType
from yunmeng.numerics.fields import Field


class _LaplacianNet(nn.Module):
    """MLP over the normalized local 3-point stencil -> surrogate laplacian."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the output layer: the surrogate starts as a no-op.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


class NeuralLaplacianOperator(
    IParameterized, IModeSwitchable, IDifferentiable, IOperator
):
    """Surrogate Laplacian term (kind: nn.surrogate)."""

    def __init__(
        self,
        field: str,
        hidden: int = 32,
        device: str = "cpu",
        u_scale: float = 1.0,
        dx_ref: float = 1.0 / 32,
        out_scale: float = 1.0,
    ):
        self._field = field
        self._mode = RunMode.EVAL
        self._n = 0
        self._backend: IBackend = None
        self._device = torch.device(device)
        self._net = _LaplacianNet(hidden).to(device=self._device, dtype=torch.float64)
        # normalization constants (NOT parameters)
        self._u_scale = float(u_scale)
        self._dx_ref = float(dx_ref)
        self._out_scale = float(out_scale)
        self._interior_idx: "torch.Tensor" = None

    # -- metadata -----------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "NeuralLaplacian"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.NN_SURROGATE

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

    # -- capability: IParameterized (scalar-per-name, copy semantics) ---
    #
    # The solver-level θ channel treats every named parameter as a scalar
    # (BaseSolver zips names with values one-to-one), so each tensor
    # element is exposed as its own name: "net.0.weight[17]".

    def _scalar_names(self) -> list[str]:
        names = []
        for name, p in self._net.named_parameters():
            names.extend(f"{name}[{i}]" for i in range(p.numel()))
        return names

    def _resolve(self, name: str):
        base, _, tail = name.rpartition("[")
        index = int(tail.rstrip("]"))
        return dict(self._net.named_parameters())[base], index

    def parameter_metas(self) -> list:
        return [
            ParamMeta(name=n, description=f"{self.get_name()}.{n}", default=0.0)
            for n in self._scalar_names()
        ]

    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        names = names or self._scalar_names()
        out = np.empty(len(names), dtype="float64")
        for i, n in enumerate(names):
            p, flat = self._resolve(n)
            out[i] = float(p.detach().cpu().reshape(-1)[flat])
        return out

    def set_parameters(self, values: ArrayLike, names: list[str]):
        values = np.asarray(values, dtype="float64").reshape(-1)
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values, got {len(values)}")
        with torch.no_grad():
            for n, v in zip(names, values):
                p, flat = self._resolve(n)
                p.reshape(-1)[flat] = float(v)

    # -- capability: IModeSwitchable -----------------

    def set_mode(self, mode: RunMode) -> None:
        self._mode = mode
        if mode == RunMode.TRAIN:
            self._net.train()
        else:
            self._net.eval()

    # -- capability: IDifferentiable -----------------

    def grad_parameters(self) -> list[ArrayLike]:
        return [p for p in self._net.parameters() if p.requires_grad]

    def torch_parameters(self) -> list["torch.nn.Parameter"]:
        return list(self._net.parameters())

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: record sizes and interior indices; θ untouched."""
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        topo = mesh.get_topo_assistant()
        interior = [
            c
            for c in range(self._n)
            if all(nb >= 0 for nb in topo.neighbors(c, ElementType.CELL))
        ]
        self._interior_idx = torch.as_tensor(
            interior, dtype=torch.long, device=self._device
        )
        # actual cell spacing from geometry (first interior pair)
        geom = mesh.get_geom_assistant()
        centers = np.asarray(geom.coordinates(ElementType.CELL), dtype="float64")
        c0 = interior[0]
        c1 = next(nb for nb in topo.neighbors(c0, ElementType.CELL) if nb >= 0)
        self._dx = float(np.linalg.norm(centers[c1] - centers[c0]))

    def _as_torch(self, u: ArrayLike) -> "torch.Tensor":
        if isinstance(u, torch.Tensor):
            return u.to(device=self._device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(u), dtype=torch.float64, device=self._device)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """lap_hat over interior cells; boundary cells contribute zero
        (their values are constrained by the solver anyway)."""
        u_field = datahub.get_field(self._field)
        u = self._as_torch(u_field.values)

        idx = self._interior_idx
        stencil = torch.stack([u[idx - 1], u[idx], u[idx + 1]], dim=-1) / self._u_scale
        dx_col = torch.full(
            (idx.numel(), 1),
            self._dx / self._dx_ref,
            dtype=torch.float64,
            device=self._device,
        )
        lap_hat = self._net(torch.cat([stencil, dx_col], dim=-1)) * self._out_scale

        corr = torch.zeros_like(u)
        corr[idx] = lap_hat

        if isinstance(u_field.values, torch.Tensor):
            out = corr  # stays on the graph (TRAIN)
        else:
            out = corr.detach().cpu().numpy()

        meta = FieldMeta(
            name=f"laplacian:{self._field}",
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
        )
        result_field = Field(meta, out)
        datahub.publish(f"laplacian:{self._field}", result_field, t)
        return OperatorResult(explicit=result_field, implicit=None)

    def __repr__(self) -> str:
        return (
            f"NeuralLaplacianOperator(field={self._field!r}, "
            f"mode={self._mode.value})"
        )
