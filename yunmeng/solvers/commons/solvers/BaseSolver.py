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
    IInitialCondition,
    IBoundaryCondition,
    OperatorType,
    SolverMeta,
    SolverStatus,
    SolverType,
    SolverConfig,
)
from yunmeng.numerics.mesh import Element, ElementType, Mesh
from yunmeng.numerics.grids import Grid
from yunmeng.numerics.fields import Field
from yunmeng.setting import logger

from collections import defaultdict
from typing import Union
import pickle


class BaseSolver(ISolver):
    """
    Basic solver.

    Example::

        solver = Solver("demo", mesh, ops, config)

        # Assembly
        solver.add_ic("u", my_ic)
        solver.add_bc("u", my_bc, [0,1], ElementType.NODE)
        cb = VtkWriter("out/")
        solver.add_callback(cb)

        # Initialize
        solver.initialize(end_time=1.0)

        # Runtime
        while not solver.status.finished:
            status = solver.forward()      # CFL adaptive
            # or: solver.forward(dt=0.001) # fixed dt
    """

    def __init__(
        self,
        id: str,
        mesh: Mesh,
        operators: list[IOperator] = None,
        config: SolverConfig = None,
    ):
        """
        Basic solver.

        Args:
            id: The unique id of the solver instance.
            mesh: The mesh of the problem.
            operators: The operators used.
            config: The solver configuration.
        """
        self._id: str = id

        if not isinstance(mesh, Mesh):
            raise ValueError(f"Invalid mesh type: {mesh}.")

        self._mesh: Union[Mesh, Grid] = mesh
        self._status: SolverStatus = SolverStatus()

        self._callbacks: list[ISolverCallback] = []
        self._fields: dict[str, Field] = {}
        self._operators: list[IOperator] = operators

        if config is None:
            config = self.__class__.get_config_class()()
        self._config: SolverConfig = config

        self._default_ic: IInitialCondition = None
        self._ics: dict[str, IInitialCondition] = {}

        self._default_bc: IBoundaryCondition = None
        self._bcs: dict[str, list[IBoundaryCondition]] = defaultdict(list)

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def config(self) -> SolverConfig:
        self._config

    def add_ic(self, field: str, ic: IInitialCondition):
        if not isinstance(ic, IInitialCondition):
            raise ValueError(f"Invalid initial condition: {ic}.")

        if ic.target_field and ic.target_field != field:
            raise ValueError(
                f"IC {ic.id} has different target field: {ic.target_field}."
            )
        else:
            ic.target_field = field

        meta = self.get_meta()
        if meta.fields is not None and field not in meta.fields:
            raise ValueError(
                f"Solver {self._id}: IC field '{field}' isn't available: "
                f"{list(meta.fields.keys())}."
            )
        if field in self._ics:
            logger.warning(
                f"Solver {self._id}: IC for field '{field}' overwritten "
                f"({self._ics[field].id} -> {ic.id})."
            )

        self._ics[field] = ic

    def clear_ics(self, field: str = None):
        if field is None:
            self._ics.clear()
        else:
            if field in self._ics:
                del self._ics[field]

    def add_bc(self, field: str, bc: IBoundaryCondition):
        if not isinstance(bc, IBoundaryCondition):
            raise ValueError(f"Invalid boundary condition: {bc}.")

        if bc.target_field and bc.target_field != field:
            raise ValueError(
                f"BC {bc.id} has different target field: {bc.target_field}."
            )
        else:
            bc.target_field = field

        new_added_ids = bc.region.get_element_ids()
        if len(new_added_ids) == 0:
            raise ValueError(f"BC {bc.id} has no elements.")

        existed_ids = []
        for bc_ in self._bcs[field]:
            if bc_.region.type != bc.region.type:
                raise ValueError(
                    f"BC {bc.id} has different region type with existing BCs: "
                    f"{bc.region.type} != {bc_.region.type}."
                )
            existed_ids.extend(bc_.region.get_element_ids())
        if len(set(new_added_ids) & set(existed_ids)) > 0:
            raise ValueError(
                f"BC {bc.id} has elements that already have BCs: "
                f"{set(new_added_ids) & set(existed_ids)}."
            )

        self._bcs[field].append(bc)

    def clear_bcs(self, field: str = None):
        if field is None:
            self._bcs.clear()
        else:
            if field in self._bcs:
                del self._bcs[field]

    def add_callback(self, cb: ISolverCallback):
        if not isinstance(cb, ISolverCallback):
            raise ValueError(f"Invalid callback: {cb}")

        cb.setup(self, self._mesh)
        self._callbacks.append(cb)

    def remove_callback(self, cb_id: str):
        for cb in self._callbacks:
            if cb.id == cb_id:
                self._callbacks.remove(cb)
                break

    def get_solution(self, field: str) -> Field:
        if field not in self._fields:
            logger.error(f"Solver {self._id} has no field {field}.")
            return None

        return self._fields[field]

    def save(self, path: str):
        snapshot = {
            "id": self._id,
            "status": self._status,
            "config": self._config,
            "fields": self._fields,
            "operators": self._operators,
            "ics": self._ics,
            "bcs": self._bcs,
            "callbacks": self._callbacks,
        }
        with open(path, "wb") as f:
            pickle.dump(snapshot, f)
        logger.info(f"Solver {self._id}: snapshot saved to {path}.")

    @classmethod
    def load(cls, path: str) -> ISolver:
        with open(path, "rb") as f:
            snapshot = pickle.load(f)

        solver = cls(
            snapshot["id"], snapshot["mesh"], snapshot["operators"], snapshot["config"]
        )
        solver._status = snapshot["status"]
        solver._fields = snapshot["fields"]
        solver._ics = snapshot["ics"]
        solver._bcs = snapshot["bcs"]
        solver._callbacks = snapshot["callbacks"]

        logger.info(f"Solver {solver.id} loaded from {path}.")
        return solver

    def set_problems(self, equations: list[IEquation]):
        pass

    def assimilate(self):
        pass

    def initialize(self):
        raise NotImplementedError()

    def forward(self) -> SolverStatus:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
