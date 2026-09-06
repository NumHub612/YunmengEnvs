# -*- encoding: utf-8 -*-
"""
Mechanism operators for the 2D advection-diffusion equation.

    du/dt = -(vx * du/dx + vy * du/dy) + nu * lap(u)

Both operators follow the v2.0 lifecycle:
- build(grid, backend): precompute static topology (spacings, boundary
  mask); no theta, no runtime data;
- forward(datahub, t, dt): read the current field, publish the explicit
  tendency. All array math goes through the backend namespace `xp`, so the
  same code runs on numpy and torch (differentiable).

Physical coefficients (vx, vy, nu) are operator parameters theta, exposed
via IParameterized so the model layer can aggregate/route them.
"""

from __future__ import annotations

from typing import Mapping

from yunmeng.interfaces.solver import OperatorResult
from yunmeng.interfaces.types import RunMode
from yunmeng.numerics.fields import Field, FieldMeta


def _roll(xp, a, shift, dim):
    """Backend-neutral roll (numpy: axis=, torch: dims=)."""
    if xp.__name__ == "torch":
        return xp.roll(a, shift, dims=dim)
    return xp.roll(a, shift, axis=dim)


class _BaseMechanismOperator:
    """Shared plumbing for the mechanism operators below."""

    kind: str = "math.unknown"
    differentiable: bool = True

    def __init__(self, name: str):
        self.name = name
        self._grid = None
        self._xp = None
        self._mode = RunMode.EVAL

    # -- IOperator -----------------------------------

    def build(self, grid, backend) -> None:
        self._grid = grid
        self._xp = backend

    def forward(self, datahub, t: float, dt: float) -> OperatorResult:
        raise NotImplementedError()

    # -- IModeSwitchable ------------------------------

    def set_mode(self, mode: RunMode) -> None:
        self._mode = mode

    # -- helpers --------------------------------------

    def _check_built(self):
        if self._grid is None or self._xp is None:
            raise RuntimeError(f"Operator {self.name}: build() not called.")

    def _make_field(self, name: str, data) -> Field:
        g = self._grid
        return Field(data, FieldMeta(name=name, nx=g.nx, ny=g.ny))


class AdvUpwind2D(_BaseMechanismOperator):
    """First-order upwind advection tendency: -(vx dudx + vy dudy).

    Parameters (IParameterized): vx, vy -- advection velocity components.
    torch backend: velocities may carry requires_grad; `where` switching on
    their sign gives piecewise-constant (sub)gradients, which is accepted
    practice (same as JAX-Fluids; v1.0 §5).
    """

    kind = "math.div"

    def __init__(self, name: str = "advection", vx: float = 0.0, vy: float = 0.0):
        super().__init__(name)
        self._vx = vx
        self._vy = vy

    # -- IParameterized -------------------------------

    def get_parameters(self) -> dict:
        return {"vx": self._vx, "vy": self._vy}

    def set_parameters(self, params: Mapping) -> None:
        if "vx" in params:
            self._vx = params["vx"]
        if "vy" in params:
            self._vy = params["vy"]

    # -- IOperator -------------------------------------

    def forward(self, datahub, t: float, dt: float) -> OperatorResult:
        self._check_built()
        xp = self._xp
        g = self._grid
        u = datahub.field("u").data

        roll = xp.roll if hasattr(xp, "roll") else None
        if roll is None:  # torch has roll
            raise RuntimeError("backend lacks roll")

        up_xm = _roll(xp, u, 1, 0)  # u[i-1, j]
        up_xp = _roll(xp, u, -1, 0)  # u[i+1, j]
        up_ym = _roll(xp, u, 1, 1)
        up_yp = _roll(xp, u, -1, 1)

        def upwind(v, um, up_, spacing):
            back = (u - um) / spacing
            fwd = (up_ - u) / spacing
            if isinstance(v, float):  # static branch for plain scalars
                return back if v >= 0 else fwd
            return xp.where(v >= 0, back, fwd)  # tensor: subgradient where

        vx = self._vx
        vy = self._vy
        dudx = upwind(vx, up_xm, up_xp, g.dx)
        dudy = upwind(vy, up_ym, up_yp, g.dy)

        tendency = -(vx * dudx + vy * dudy)
        return OperatorResult(explicit=self._make_field("adv_tendency", tendency))


class Lap5Point2D(_BaseMechanismOperator):
    """Five-point Laplacian diffusion tendency: nu * lap(u).

    Parameters (IParameterized): nu -- diffusion coefficient.
    """

    kind = "math.laplacian"

    def __init__(self, name: str = "diffusion", nu: float = 0.0):
        super().__init__(name)
        self._nu = nu

    # -- IParameterized ---------------------------------

    def get_parameters(self) -> dict:
        return {"nu": self._nu}

    def set_parameters(self, params: Mapping) -> None:
        if "nu" in params:
            self._nu = params["nu"]

    # -- IOperator ---------------------------------------

    def forward(self, datahub, t: float, dt: float) -> OperatorResult:
        self._check_built()
        xp = self._xp
        g = self._grid
        u = datahub.field("u").data

        lap = (_roll(xp, u, 1, 0) - 2.0 * u + _roll(xp, u, -1, 0)) / (g.dx * g.dx) + (
            _roll(xp, u, 1, 1) - 2.0 * u + _roll(xp, u, -1, 1)
        ) / (g.dy * g.dy)
        return OperatorResult(
            explicit=self._make_field("diff_tendency", self._nu * lap)
        )
