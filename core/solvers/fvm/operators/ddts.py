# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time derivative operators for the finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, NodeField, Tensor, Vector, VariableType
from core.numerics.mesh import Mesh, ElementType

import numpy as np
import copy


class Ddt01(IOperator):
    """
    First order implicit Euler scheme(FOUE) for the time derivative operator.

    scheme:
        - Implicit method.
        - Upwind scheme as time interpolation profile.
        - The medium was assumed to be isotropic.
    """

    @classmethod
    def get_type(self) -> OperatorType:
        return OperatorType.DDT

    def get_name(cls) -> str:
        return "ddt01"

    def __init__(self):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._rho = None

    def prepare(self, mesh: Mesh, rho: float, **kwargs):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

        self._rho = rho

    def run(self, source: Field, time_step: float) -> Field | LinearEqs:
        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=source.dtype, variable=source.variable
        )

        for cell in self._mesh.cells:
            cidx = self._topo.cell_indices[cell.id]
            vol = self._geom.cell_volumes[cidx]
            val = source[cidx]
            coef = self._rho * vol / time_step

            ddt_eqs.matrix[cidx, cidx] += coef
            ddt_eqs.rhs[cidx] -= -coef * val

        return ddt_eqs


class Ddt02(IOperator):
    """
    Second order implicit Euler scheme(SOUE) for the time derivative operator.

    scheme:
        - implicit method.
        - upwind scheme as time interpolation profile.
        - the medium was assumed to be isotropic.
        - the time step was assumed to be constant.
    """

    @classmethod
    def get_type(self) -> OperatorType:
        return OperatorType.DDT

    def get_name(cls) -> str:
        return "ddt02"

    def __init__(self):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._pre_field = None
        self._rho = None

    def prepare(self, mesh: Mesh, rho: float, **kwargs):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

        self._pre_field = None
        self._rho = rho

    def run(self, source: Field, time_step: float) -> Field | LinearEqs:
        if self._pre_field is None:
            self._pre_field = copy.deepcopy(source)

        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count, rhs_type=source.dtype, variable=source.variable
        )

        for cell in self._mesh.cells:
            cidx = self._topo.cell_indices[cell.id]
            vol = self._geom.cell_volumes[cidx]
            cur_v = source[cidx]
            pre_v = self._pre_field[cidx]

            tmp = self._rho * vol / (2.0 * time_step)
            fluxC = 3.0 * tmp
            fluxV = 4.0 * tmp * cur_v - tmp * pre_v

            ddt_eqs.matrix[cidx, cidx] += fluxC
            ddt_eqs.rhs[cidx] += fluxV

        self._pre_field = copy.deepcopy(source)
        return ddt_eqs
