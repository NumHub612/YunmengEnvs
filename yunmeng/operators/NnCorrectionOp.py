# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from __future__ import annotations
import numpy as np

from yunmeng.interfaces.capabilities import (
    IDifferentiable,
    IModeSwitchable,
    IParameterized,
    ParamMeta,
)
from yunmeng.interfaces.solver import IOperator, OperatorKinds, OperatorResult
from yunmeng.interfaces.supports import (
    DataProduct,
    FieldMeta,
    IBackend,
    IDataHub,
    IGrid,
    TOPO_NONE,
)
from yunmeng.interfaces.types import ArrayLike, ElementType, RunMode, VariableType
from yunmeng.numerics.fields import Field
from yunmeng.numerics.algos import ym_register

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


class _StencilNet(torch.nn.Module):
    """MLP over the local 3-point stencil."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@ym_register("operator")
class NeuralCorrectionOperator(
    IParameterized, IDifferentiable, IModeSwitchable, IOperator
):
    """Local-stencil neural residual correction.

    The output layer is zero-initialized, so training starts from the pure
    physics operator. IParameterized exposes a flat, copy-semantic scalar
    vector; IDifferentiable exposes the live torch leaves to optimizers.
    """

    def __init__(self, field: str, hidden: int = 16, device: str = "cpu"):
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("NeuralCorrection requires PyTorch")

        self._field = field
        self._mode = RunMode.EVAL
        self._backend = None
        self._device = torch.device(device)
        self._net = _StencilNet(hidden).to(self._device, torch.float64)
        self._n = 0
        self._dx = 0.0
        self._interior_idx = None
        self._param_map = self._make_param_map()

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
    def target_fields(self) -> list[str]:
        return [self._field]

    @property
    def time_order(self) -> int:
        return 1

    def _make_param_map(self) -> dict:
        result = {}
        for path, param in self._net.named_parameters():
            values = param.detach().cpu().numpy()
            shape = values.shape
            for flat_index, value in enumerate(values.reshape(-1)):
                logical = np.unravel_index(flat_index, shape)
                index = ",".join(str(i) for i in logical)
                name = f"{path}[{index}]"
                result[name] = (path, flat_index, float(value))
        return result

    def parameter_metas(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=name,
                description=f"{self.get_name()}:{path}",
                default=default,
                required=True,
            )
            for name, (path, _, default) in self._param_map.items()
        ]

    def get_parameters(self, names: list[str] = None) -> ArrayLike:
        names = names or list(self._param_map)
        tensors = dict(self._net.named_parameters())
        values = []
        for name in names:
            path, flat_index, _ = self._param_map[name]
            value = tensors[path].detach().cpu().numpy().reshape(-1)[flat_index]
            values.append(float(value))
        return np.asarray(values, dtype="float64")

    def set_parameters(self, values: ArrayLike, names: list[str] = None):
        names = names or list(self._param_map)
        values = np.asarray(values, dtype="float64").reshape(-1)
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values, got {len(values)}")
        tensors = dict(self._net.named_parameters())
        with torch.no_grad():
            for name, value in zip(names, values):
                if name not in self._param_map:
                    raise KeyError(f"missing parameter {name!r}")
                path, flat_index, _ = self._param_map[name]
                tensors[path].reshape(-1)[flat_index] = float(value)

    def grad_parameters(self) -> list[ArrayLike]:
        return list(self._net.parameters())

    def set_mode(self, mode: RunMode):
        self._mode = mode
        if mode == RunMode.TRAIN:
            self._net.train()
        else:
            self._net.eval()

    def build(self, mesh: IGrid, backend: IBackend):
        self._backend = backend
        self._n = mesh.element_count(ElementType.CELL)
        self._dx = float(mesh.spacing[0])
        self._interior_idx = torch.arange(1, self._n - 1, device=self._device)

    def _as_torch(self, values: ArrayLike) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self._device, dtype=torch.float64)
        return torch.as_tensor(
            np.asarray(values), dtype=torch.float64, device=self._device
        )

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        field = datahub.get_field(self._field)
        u = self._as_torch(field.values)
        idx = self._interior_idx

        laplacian = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / self._dx**2
        stencil = torch.stack([u[idx - 1], u[idx], u[idx + 1]], dim=-1)
        kappa = self._net(stencil)
        correction = torch.zeros_like(u)
        correction[idx] = kappa * laplacian

        if isinstance(field.values, torch.Tensor):
            values = correction
        else:
            values = correction.detach().cpu().numpy()
        meta = FieldMeta(
            name=f"correction:{self._field}",
            vtype=VariableType.SCALAR,
            loc=ElementType.CELL,
            btype=self._backend.name,
            device=str(self._device),
        )
        product = Field(meta, values)
        datahub.publish(meta.name, product, t)
        return OperatorResult(explicit=product, implicit=None)
