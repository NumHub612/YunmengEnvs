# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for fluid equations solvers.
"""
from core.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from core.solvers.interfaces.IInitCondition import IInitCondition
from core.solvers.interfaces.ISolverCallback import ISolverCallback
from core.numerics.fields import Field
from core.numerics.mesh import Mesh, Node, Face, Cell
from configs.settings import logger

from abc import ABC, abstractmethod


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

    @property
    @abstractmethod
    def status(self) -> dict:
        """
        The current status of the solver, not including the solution.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Unique name of this solver.
        """
        pass

    @classmethod
    @abstractmethod
    def get_meta(cls) -> dict:
        """
        The accessiable fields and other meta infomations of solver.
        """
        return {}

    @abstractmethod
    def get_solution(self, field_name: str) -> Field:
        """
        Get the solution of the solver.
        """
        pass

    @abstractmethod
    def set_callback(self, callback: ISolverCallback):
        """
        Set the callback to be called during the solving process.
        """
        pass

    @abstractmethod
    def set_ic(self, var: str, ic: IInitCondition):
        """
        Set the initial condition.
        """
        pass

    @abstractmethod
    def set_bc(self, var: str, elems: list, bc: IBoundaryCondition):
        """
        Set the boundary condition for the solver.

        Args:
            var: Name of the variable to set the boundary condition.
            elems: The list of elements to set on.
            bc: The boundary condition.
        """
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the solver.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the solver.
        """
        pass

    @abstractmethod
    def terminate(self, **kwargs):
        """
        Terminate the solver.
        """
        pass

    @abstractmethod
    def assimilate(self, **kwargs):
        """
        Assimilate the solver with extra data to improve its accuracy.
        """
        pass

    @abstractmethod
    def optimize(self, **kwargs):
        """
        Optimize the solver with data to calibrate its parameters.
        """
        pass

    @abstractmethod
    def inference(self, **kwargs) -> tuple[bool, bool, dict]:
        """
        Inference the solver to get the solutions.

        Returns:
            A tuple of results with (is_done, is_terminated, status).
        """
        pass


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
        self._mesh = mesh

        self._callbacks = []
        self._fields = {}

        self._default_init = None
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

    def set_bc(self, var: str, elems: list, bc: IBoundaryCondition):
        if not isinstance(bc, IBoundaryCondition):
            raise ValueError(f"Invalid boundary condition: {bc}")

        for elem in elems:
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
