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

from typing import List
import numpy as np

from yunmeng.standard.capabilities import (
    ParamMeta,
    IDifferentiable,
    IModeSwitchable,
    IParameterized,
)
from yunmeng.standard.solver.IOperator import (
    IOperator,
    OperatorKinds,
    OperatorResult,
)
from yunmeng.standard.supports.backend import IBackend
from yunmeng.standard.supports.datahub import IDataHub
from yunmeng.standard.supports import DataProduct, FieldMeta, IGrid
from yunmeng.standard.supports.mesh import IMesh
from yunmeng.standard.types import (
    ArrayLike,
    ElementType,
    RunMode,
    VariableType,
    MeshDimension,
)
from yunmeng.numerics.fields import Field

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    _HAS_TORCH = False

# ---------------------------------------------------
# region NN model
# ---------------------------------------------------


class _LaplacianNet(nn.Module):
    """MLP over the normalized local 3-point stencil -> surrogate laplacian."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
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


# ---------------------------------------------------
# region NN operator
# ---------------------------------------------------


class NeuralDiffusionOperator(
    IOperator, IParameterized, IModeSwitchable, IDifferentiable
):
    """Surrogate Laplacian term (kind: nn.surrogate)."""

    @classmethod
    def get_name(cls) -> str:
        return "NeuralDiffusion"

    @classmethod
    def get_kind(cls) -> str:
        return OperatorKinds.NN_SURROGATE

    def __init__(
        self,
        oid: str,
        field: str,
        hidden: int = 32,
        device: str = "cpu",
        nu_hint: float = 1.0,
    ):
        self._id = oid
        self._field = field
        self._field_loc = ElementType.CELL
        self._mode = RunMode.EVAL
        self._n = 0
        self._backend: IBackend = None
        self._device = torch.device(device)
        self._net = _LaplacianNet(hidden).to(device=self._device, dtype=torch.float64)

        # adaptive normalization scales (constants, NOT parameters)
        # nu_hint: prior on the diffusion coefficient, only used to scale
        # the output to the TENDENCY magnitude (~nu*u/dx^2); the net can
        # still learn any effective nu, this just keeps Adam stable.
        self._nu_hint = float(nu_hint)
        self._dx = 0.0
        self._u_scale = 1.0
        self._out_scale = 1.0
        self._adapted = False

    # region IOperator ------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def differentiable(self) -> bool:
        return True

    @property
    def target_fields(self) -> List[str]:
        return [self._field]

    @property
    def time_order(self) -> int:
        return 1

    def produces(self) -> list[DataProduct]:
        return []

    def consumes(self) -> list[DataProduct]:
        return [
            DataProduct(
                kind=OperatorKinds.MATH_VALUE,
                field=self._field,
                loc=self._field_loc,
                time_order=0,
            )
        ]

    def set_scales(self, u_scale: float, out_scale: float):
        """Manual override of the adaptive scales; must be called before
        the first forward (i.e. right after build), otherwise ignored."""
        if self._adapted:
            raise RuntimeError(f"Op {self._id!r}: scales already adapted")
        self._u_scale = float(u_scale)
        self._out_scale = float(out_scale)
        self._adapted = True

    def build(self, mesh: IMesh, backend: IBackend):
        """Static phase: record sizes and interior indices; θ untouched."""
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        topo = mesh.get_topo_assistant()
        interior = topo.interior_elements(ElementType.CELL)
        if len(interior) == 0:
            raise RuntimeError(
                f"Op {self._id!r}: mesh has no interior cells "
                f"(n={self._n}); a surrogate stencil needs >= 3 cells"
            )

        # stencil indexing u[idx +- 1] assumes 1D C-order contiguous
        # neighbours; verify once instead of silently reading wrong cells.
        c0 = int(interior[0])
        if topo.neighbors(c0, ElementType.CELL) != [c0 - 1, c0 + 1]:
            raise RuntimeError(
                f"Op {self._id!r}: assumes contiguous [-1, +1] cell "
                "numbering; mesh topology violates this"
            )
        self._interior_idx = torch.as_tensor(
            interior, dtype=torch.long, device=self._device
        )

        # actual cell spacing, needed for the adaptive out_scale.
        geom = mesh.get_geom_assistant()
        centers = np.asarray(geom.coordinates(ElementType.CELL), dtype="float64")
        self._dx = float(np.linalg.norm(centers[c0 + 1] - centers[c0]))

    def _adapt(self, u: "torch.Tensor"):
        """One-time statistics from the first field seen after build."""
        peak = float(u.detach().abs().max())
        self._u_scale = peak if peak > 0 else 1.0
        self._out_scale = self._nu_hint * self._u_scale / self._dx**2
        self._adapted = True

    def _as_torch(self, u: ArrayLike) -> "torch.Tensor":
        if isinstance(u, torch.Tensor):
            return u.to(device=self._device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(u), dtype=torch.float64, device=self._device)

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """lap_hat over interior cells."""
        u_field = datahub.field(self._field)
        u = self._as_torch(u_field.values)
        if not self._adapted:
            self._adapt(u)

        idx = self._interior_idx
        stencil = torch.stack([u[idx - 1], u[idx], u[idx + 1]], dim=-1) / self._u_scale
        lap_hat = self._net(stencil) * self._out_scale

        corr = torch.zeros_like(u)
        corr[idx] = lap_hat

        if isinstance(u_field.values, torch.Tensor):
            out = corr  # stays on the graph (TRAIN)
        else:
            if self._mode == RunMode.TRAIN:
                raise RuntimeError(
                    f"Op {self._id!r} require torch backend in training mode"
                )
            out = corr.detach().cpu().numpy()

        meta = FieldMeta(
            name=f"laplacian:{self._field}",
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
            device=self._device,
        )
        result_field = Field(meta, out)
        return OperatorResult(explicit=result_field)

    # -- capability: IParameterized -----------------

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

    def parameter_names(self) -> list[str]:
        return self._scalar_names()

    def parameter_bounds(self, names: list[str] = None):
        return {n: None for n in (names or self._scalar_names())}

    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        names = names or self._scalar_names()
        out = np.empty(len(names), dtype="float64")
        for i, n in enumerate(names):
            p, flat = self._resolve(n)
            out[i] = float(p.detach().cpu().reshape(-1)[flat])
        return out

    def set_parameters(self, values: ArrayLike, names: list[str] = None):
        names = names or self._scalar_names()
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
