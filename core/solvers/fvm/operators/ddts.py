# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time derivative operators for the finite volume method.
"""
from core.solvers.interfaces import IOperator, OperatorType
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field, DataHub
from core.numerics.mesh import Mesh
from core.numerics.algos import MeshTopo, MeshGeom

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
        self._mesh: Mesh = None
        self._topo: MeshTopo = None
        self._geom: MeshGeom = None

        self._rho = rho
        self._var = ""

    def prepare(self, fields: list[str], mesh: Mesh, bounds: dict):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()
        self._var = fields[0]

    def run(self, sources: DataHub) -> Field | LinearEqs:
        sample = sources.field(self._var, 0)
        data = sample.data
        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count,
            rhs_type=data.dtype,
            variable=data.name,
        )

        for cid in range(self._mesh.cell_count):
            vol = self._geom.cell_volume[cid]
            val = data[cid]
            coef = self._rho * vol / sample.timestep
            ddt_eqs.matrix[cid, cid] += coef
            ddt_eqs.rhs[cid] -= -coef * val

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
        self._mesh: Mesh = None
        self._topo: MeshTopo = None
        self._geom: MeshGeom = None

        self._rho = rho
        self._var = ""

    def prepare(self, fields: list[str], mesh: Mesh, bounds: dict = None):
        self._mesh = mesh
        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

        self._var = fields[0]

    def run(self, sources: DataHub) -> Field | LinearEqs:
        pre_source = sources.field(self._var, 1)
        pre_data = pre_source.data
        cur_source = sources.field(self._var, 0)
        cur_data = cur_source.data

        ddt_eqs = LinearEqs.zeros(
            self._mesh.cell_count,
            rhs_type=cur_data.dtype,
            variable=cur_data.name,
        )

        for cid in range(self._mesh.cell_count):
            vol = self._geom.cell_volume[cid]
            cur_v = cur_data[cid]
            pre_v = pre_data[cid]

            tmp = self._rho * vol / (2.0 * cur_source.timestep)
            fluxC = 3.0 * tmp
            fluxV = 4.0 * tmp * cur_v - tmp * pre_v

            ddt_eqs.matrix[cid, cid] += fluxC
            ddt_eqs.rhs[cid] += fluxV

        self._pre_field = copy.deepcopy(sources)
        return ddt_eqs
