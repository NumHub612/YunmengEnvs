# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time derivative operators for the finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, DataHub
from core.numerics.mesh import Mesh

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
    def get_type(cls) -> OperatorType:
        return OperatorType.DDT

    @classmethod
    def get_name(cls) -> str:
        return "ddt01"

    def __init__(self, rho: float):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._rho = rho

    def prepare(self, mesh: Mesh, boundaries: dict):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def run(self, source: DataHub) -> Field | LinearEqs:
        source = source.fetch()
        data = source.data
        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count,
            rhs_type=data.dtype,
            variable=data.variable,
        )

        for cell in self._mesh.cells:
            cidx = self._topo.cell_indices[cell.id]
            vol = self._geom.cell_volumes[cidx]
            val = data[cidx]
            coef = self._rho * vol / source.timestep

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
    def get_type(cls) -> OperatorType:
        return OperatorType.DDT

    @classmethod
    def get_name(cls) -> str:
        return "ddt02"

    @property
    def time_order(self) -> int:
        return 2

    def __init__(self, rho: float):
        self._mesh = None
        self._topo = None
        self._geom = None

        self._rho = rho

    def prepare(self, mesh: Mesh, boundaries: dict):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def run(self, source: DataHub) -> Field | LinearEqs:
        pre_source = source.fetch(1)
        pre_data = pre_source.data
        cur_source = source.fetch(0)
        cur_data = cur_source.data

        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count,
            rhs_type=cur_data.dtype,
            variable=cur_data.variable,
        )

        for cell in self._mesh.cells:
            cidx = self._topo.cell_indices[cell.id]
            vol = self._geom.cell_volumes[cidx]
            cur_v = cur_data[cidx]
            pre_v = pre_data[cidx]

            tmp = self._rho * vol / (2.0 * cur_source.timestep)
            fluxC = 3.0 * tmp
            fluxV = 4.0 * tmp * cur_v - tmp * pre_v

            ddt_eqs.matrix[cidx, cidx] += fluxC
            ddt_eqs.rhs[cidx] += fluxV

        self._pre_field = copy.deepcopy(source)
        return ddt_eqs
