# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Base class for all solvers.
"""
from core.solvers.interfaces import (
    ISolver,
    ISolverCallback,
    IInitCondition,
    IBoundaryCondition,
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

        self._default_ic = None
        self._ics = {}

        self._default_bc = None
        self._bcs = {}

    def get_solution(self, field_name: str) -> Field:
        if field_name not in self._fields:
            logger.warning(
                f"Solver {self._id} solution for {field_name} is not available."
            )
            return None

        return self._fields[field_name]

    def set_callback(self, callback: ISolverCallback):
        if not isinstance(callback, ISolverCallback):
            raise ValueError(f"Invalid callback: {callback}")

        callback.setup(self.get_meta(), self._mesh)
        self._callbacks.append(callback)

    def set_ic(self, var: str, ic: IInitCondition):
        if not isinstance(ic, IInitCondition):
            raise ValueError(f"Invalid initial condition: {ic}")

        if var in self._ics:
            logger.warning(
                f"Solver {self._id} initial condition for {var} overwriting."
            )

        self._ics[var] = ic

    def set_bc(self, var: str, elements: list, bc: IBoundaryCondition):
        if not isinstance(bc, IBoundaryCondition):
            raise ValueError(f"Invalid boundary condition: {bc}")

        for elem in elements:
            if not isinstance(elem, (Node, Face, Cell)):
                raise ValueError(f"Invalid element: {elem}")
            if isinstance(elem, Node):
                if elem.id < 0 or elem.id >= self._mesh.node_count:
                    raise ValueError(f"Invalid node id: {elem.id}")
            elif isinstance(elem, Face):
                if elem.id < 0 or elem.id >= self._mesh.face_count:
                    raise ValueError(f"Invalid face id: {elem.id}")
            elif isinstance(elem, Cell):
                if elem.id < 0 or elem.id >= self._mesh.cell_count:
                    raise ValueError(f"Invalid cell id: {elem.id}")

            if elem.id not in self._bcs:
                self._bcs[elem.id] = {}

            if var in self._bcs[elem.id]:
                logger.warning(
                    f"Solver {self._id} boundary condition for {var} on {elem.id} overwriting."
                )

            self._bcs[elem.id][var] = bc

    def initialize(self, **kwargs):
        raise NotImplementedError()

    def reset(self, **kwargs):
        raise NotImplementedError()

    def terminate(self, **kwargs):
        raise NotImplementedError()

    def assimilate(self, **kwargs):
        raise NotImplementedError()

    def optimize(self, **kwargs):
        raise NotImplementedError()

    def inference(self, **kwargs):
        raise NotImplementedError()
