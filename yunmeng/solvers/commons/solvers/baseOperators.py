# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base implicit/explicit operators for solvers.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorMode,
    BoundaryType,
)
from yunmeng.numerics.mesh import Mesh
from yunmeng.numerics.fields import Field
from yunmeng.numerics.algos import MeshTopo, MeshGeom


class BaseOperator(IOperator):
    """
    Base operators for solvers.
    """

    def __init__(self, target_fields: list[str]):
        self._targets = target_fields
        self._mesh: Mesh = None
        self._topo: MeshTopo = None
        self._geom: MeshGeom = None
        self._bcs: dict[str, list[IBoundaryCondition]] = None

    @property
    def target_fields(self) -> list[str]:
        return self._targets

    def prepare(
        self,
        mesh: Mesh,
        boundaries: dict[str, list[IBoundaryCondition]],
    ):
        self._mesh = mesh
        self._bcs = boundaries

        self._topo = self._mesh.get_topo_assistant()
        self._geom = self._mesh.get_geom_assistant()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _apply_bc_directly(
        self, field: Field, bfield: str, btype: BoundaryType = BoundaryType.VALUE
    ):
        """Apply boundary conditions to the field."""
        for bc in self._bcs[bfield]:
            if btype is None or bc.get_type() == btype:
                bc.apply(field)


class BaseExplicitOperator(BaseOperator):
    """
    Base explicit operators for solvers,
    `forward()` method return `Field` object.
    """

    @classmethod
    def get_mode(cls):
        return OperatorMode.EXPLICIT


class BaseImplicitOperator(BaseOperator):
    """
    Base implicit operators for solvers,
    `forward()` method return `LinearEqs` object.
    """

    @classmethod
    def get_mode(cls):
        return OperatorMode.IMPLICIT
