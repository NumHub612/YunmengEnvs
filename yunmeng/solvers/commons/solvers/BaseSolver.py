# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base class for all solvers.
"""
from yunmeng.solvers.interfaces import (
    IEquation,
    IOperator,
    ISolver,
    ISolverCallback,
    IInitCondition,
    IBoundaryCondition,
    OperatorType,
    SolverMeta,
    SolverStatus,
    SolverType,
)
from yunmeng.numerics.mesh import Element, ElementType, Mesh, Grid
from yunmeng.numerics.fields import Field
from yunmeng.setting import logger

from collections import defaultdict
from typing import Union


class BaseSolver(ISolver):
    """
    Basic solver.
    """

    def __init__(self, id: str, mesh: Mesh, operators: dict[str, IOperator] = None):
        """
        Basic solver.

        Args:
            id: The unique id of the solver instance.
            mesh: The mesh of the problem.
            operators: The operators used.
        """
        self._id: str = id

        if not isinstance(mesh, Mesh):
            raise ValueError(f"Invalid mesh: {mesh}")
        self._mesh: Union[Mesh, Grid] = mesh
        self._status: SolverStatus = SolverStatus()

        self._callbacks: list[ISolverCallback] = []
        self._fields: dict[str, Field] = defaultdict(dict)
        self._operators: dict[str, IOperator] = operators

        self._default_ics: IInitCondition = None
        self._ics: dict[str, IInitCondition] = defaultdict(dict)

        self._default_bcs: IBoundaryCondition = None
        self._bcs: dict[int, dict[str, IBoundaryCondition]] = defaultdict(dict)

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    def get_solution(self, field: str) -> Field:
        if field not in self._fields:
            logger.error(f"Solver {self._id} solution {field} not available.")
            return None

        return self._fields[field]

    def add_callback(self, cb: ISolverCallback):
        if not isinstance(cb, ISolverCallback):
            raise ValueError(f"Invalid callback: {cb}")

        cb.setup(self, self._mesh)
        self._callbacks.append(cb)

    def add_ic(self, field: str, ic: IInitCondition):
        if not isinstance(ic, IInitCondition):
            raise ValueError(f"Invalid initial condition: {ic}")

        if field not in self.get_meta().fields:
            raise ValueError(
                f"Solver {self._id} field {field} isn't in the available fields."
            )

        if field in self._ics:
            logger.warning(
                f"Solver {self._id} field {field} initial condition overwrited."
            )

        self._ics[field] = ic

    def add_bc(
        self,
        field: str,
        bc: IBoundaryCondition,
        eids: list[int],
        etype: ElementType,
    ):
        if not isinstance(bc, IBoundaryCondition):
            raise ValueError(f"Invalid boundary condition: {bc}")

        if etype == ElementType.CELL:
            elements = self._mesh.cells
        elif etype == ElementType.FACE:
            elements = self._mesh.faces
        elif etype == ElementType.NODE:
            elements = self._mesh.nodes
        else:
            raise ValueError(f"Invalid boundary element type: {etype}")

        for eid in eids:
            if eid < 0 or eid >= len(elements):
                raise ValueError(
                    f"Solver {self._id} boundary condition element id {eid} "
                    f"out of range for element type {etype.name}."
                )

            if field in self._bcs[eid]:
                logger.warning(
                    f"Solver {self._id} field {field} boundary condition on "
                    f"element {eid} overwrited."
                )

            self._bcs[eid][field] = bc

    def set_problems(self, equations: list[IEquation]):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def assimilate(self):
        raise NotImplementedError()

    def inference(self) -> SolverStatus:
        raise NotImplementedError()
