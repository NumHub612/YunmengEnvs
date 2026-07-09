# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Source term operators for the finite difference method.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
)
from yunmeng.solvers.commons.solvers import BaseExplicitOperator, BaseImplicitOperator
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.fields import (
    DataHub2,
    DataProduct,
    Sample2,
    Field,
    Variable,
    VariableType,
)

from typing import Callable


class Src01(BaseExplicitOperator):
    """Explicit source term operator on structured grids."""

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.SRC

    @classmethod
    def get_name(cls) -> str:
        return "src01"

    def __init__(self, target_fields: list[str], tau: float, source_func: Callable):
        super().__init__(target_fields)

        self._var = target_fields[0]
        self._tau = tau
        self._source_func = source_func

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]],
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

    def forward(self, datahub: DataHub2, time: float) -> Field:
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Src01: no data for '{self._var}'@NODE in data_hub")
        old_field = sample.data
        new_field = Field(old_field.mesh_shards, old_field.vtype, old_field.etype)

        if len(old_field.mesh_shards) != 1:
            raise ValueError("FDM op src01 only supports cpu.")

        for nid in range(self._mesh.node_count):
            coor = self._mesh.nodes[nid].coordinate
            source_val = self._tau * self._source_func(coor, old_field[nid])
            new_field[nid] = source_val

        # publish to cache
        self._publish(
            datahub,
            self._var,
            ElementType.NODE,
            Sample2(time, new_field),
            self.get_type().value,
        )
        return new_field
