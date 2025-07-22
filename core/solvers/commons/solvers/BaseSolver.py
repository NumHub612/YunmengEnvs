# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base class for all solvers.
"""
from core.solvers.interfaces import (
    IEquation,
    ISolver,
    ISolverCallback,
    IInitCondition,
    IBoundaryCondition,
    SolverMeta,
    SolverStatus,
    SolverType,
)
from core.numerics.fields import Field
from core.numerics.mesh import Mesh, Node, Face, Cell
from configs.settings import logger


class BaseSolver(ISolver):
    """
    Basic solver.
    """

    def __init__(self, id: str, mesh: Mesh):
        """
        Initialize the solver.

        Args:
            id: The unique id of the solver instance.
            mesh: The mesh of the problem.
        """
        self._id = id

        if not isinstance(mesh, Mesh):
            raise ValueError(f"Invalid mesh: {mesh}")
        self._mesh = mesh

        self._callbacks = []
        self._fields = {}

        self._default_ics = None
        self._ics = {}

        self._default_bcs = None
        self._bcs = {}

        self._status = SolverStatus()

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    def get_solution(self, field_name: str) -> Field:
        if field_name not in self._fields:
            logger.warning(f"Solver {self._id} solution {field_name} isn't available.")
            return None

        return self._fields[field_name]

    def add_callback(self, callback: ISolverCallback):
        if not isinstance(callback, ISolverCallback):
            raise ValueError(f"Invalid callback: {callback}")

        callback.setup(self.get_meta(), self._mesh)
        self._callbacks.append(callback)

    def add_ic(self, var: str, ic: IInitCondition):
        if not isinstance(ic, IInitCondition):
            raise ValueError(f"Invalid initial condition: {ic}")

        if var in self._ics:
            logger.warning(f"Solver {self._id} initial condition for {var} overwrited")

        self._ics[var] = ic

    def add_bc(self, var: str, elements: list, bc: IBoundaryCondition):
        if not isinstance(bc, IBoundaryCondition):
            raise ValueError(f"Invalid boundary condition: {bc}")

        for elem in elements:
            if not isinstance(elem, (Node, Face, Cell)):
                raise ValueError(f"Invalid element: {elem}")
            if isinstance(elem, Node):
                if elem.id < 0 or elem.id > self._mesh.node_count:
                    raise ValueError(f"Invalid node id: {elem.id}")
            elif isinstance(elem, Face):
                if elem.id < 0 or elem.id > self._mesh.face_count:
                    raise ValueError(f"Invalid face id: {elem.id}")
            elif isinstance(elem, Cell):
                if elem.id < 0 or elem.id > self._mesh.cell_count:
                    raise ValueError(f"Invalid cell id: {elem.id}")

            if elem.id not in self._bcs:
                self._bcs[elem.id] = {}

            if var in self._bcs[elem.id]:
                logger.warning(
                    f"Solver {self._id} boundary condition  for {var} on {elem.id} overwrited."
                )

            self._bcs[elem.id][var] = bc

    def set_problems(self, equations: list[IEquation]):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def terminate(self):
        raise NotImplementedError()

    def assimilate(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()

    def inference(self):
        raise NotImplementedError()
