# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Surface water model.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Grid2D, ElementType, Coordinate
from core.numerics.algos import MeshFilter
from core.numerics.fields import (
    VariableType,
    Field,
    Timeseries,
    Curve,
    Pattern,
    Table,
    Var,
)
from core.solvers.interfaces import ISolver, IOperator
from core.solvers import fvm_solvers, fvm_operators
from core.solvers.commons import boundary_conditions, init_methods, callback_handlers
from core.utils.LoadData import load_data
from configs.settings import logger

from dateutil.parser import parse
from typing import Any
import datetime as dt
import pickle
import os
import json


class SurfaceWaterModel(models.BaseModel):
    """Surface water/shallow water/two-dimensional flow model,
    based on Grid2D grid and FVM algorithm Burgers equation."""

    def __init__(self, id: str, model_configs: dict, link_configs: dict):
        super().__init__(id)
        self._model_configs = model_configs
        self._link_configs = link_configs

        self._mesh: Grid2D = None
        self._solver: ISolver = None
        self._operators: dict[str, IOperator] = {}

        self._start: dt.datetime = None
        self._end: dt.datetime = None
        self._current: dt.datetime = None
        self._dt: float = None

    def initialize(self):
        self.set_status(
            models.LinkableComponentStatus.INITIALIZING,
            "SurfaceWaterModel initializing",
        )

        # time-axis
        times = self._model_configs["TEMPORAL"]
        self._start = parse(times["start_time"])
        self._end = parse(times["end_time"])
        self._dt = times["time_step"]
        self._current = self._start

        # mesh
        self._load_mesh()

        # datas
        self._load_datas()

        # operators then solver
        self._load_operators()
        self._load_solver()

        # outputs

        # inputs

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "initialized")

    def _load_mesh(self):
        """Load mesh from configuration."""
        # grid2d
        spatials = self._model_configs["SPATIAL"]
        mesh_type = spatials["type"]
        if mesh_type != "Grid2D":
            raise ValueError("SurfaceWaterModel doesn't support non-Grid2D mesh")

        low_left = spatials["params"]["lower_left"]
        up_right = spatials["params"]["upper_right"]
        nx = spatials["params"]["nx"]
        ny = spatials["params"]["ny"]
        self._mesh = Grid2D(Coordinate(*low_left), Coordinate(*up_right), nx, ny)

        # patches
        patches = spatials.get("patches", [])
        for patch in patches:
            pid = patch["id"]
            ptype = patch["etype"]
            if ptype != "face":
                raise ValueError(f"SurfaceWaterModel only supports face patches {pid}")
            ptype = ElementType.FACE
            if "from" in patch and patch["from"] is not None:
                data_file = patch["from"]
                if not os.path.exists(data_file):
                    raise ValueError(f"Patch data file {data_file} doesn't exist.")
                with open(data_file, "r") as f:
                    data = json.load(f)
                face_ids = [int(i) for i in data[pid]]
            elif "spec" in patch and patch["spec"] is not None:
                face_ids = patch["spec"]
            elif "expr" in patch and patch["expr"] is not None:
                expr = patch["expr"]
                face_ids = MeshFilter.filter_face_patch(self._mesh, expr)
            else:
                raise ValueError("Invalid patch definition.")

            self._mesh.set_group(pid, face_ids, ptype)

        # zones
        zones = spatials.get("zones", [])
        for zone in zones:
            zid = zone["id"]
            ztype = zone["etype"]
            if ztype != "cell":
                raise ValueError("SurfaceWaterModel only supports cell zones")
            ztype = ElementType.CELL
            if "from" in zone and zone["from"] is not None:
                data_file = zone["from"]
                if not os.path.exists(data_file):
                    raise ValueError(f"Zone data file {data_file} doesn't exist.")
                with open(data_file, "r") as f:
                    data = json.load(f)
                cell_ids = [int(i) for i in data[zid]]
            elif "countour" in zone and zone["countour"] is not None:
                contour = zone["countour"]
                cell_ids = MeshFilter.filter_cell_zone(self._mesh, contour=contour)
            elif "expr" in zone and zone["expr"] is not None:
                expr = zone["expr"]
                cell_ids = MeshFilter.filter_cell_zone(self._mesh, expr=expr)
            else:
                raise ValueError("Invalid zone definition.")

            self._mesh.set_group(zid, cell_ids, ztype)

    def _load_datas(self):
        """Load datas from configuration."""
        datas = self._model_configs["DATAS"]
        self._datas = {}

        # timeseries
        for ts in datas.get("timeseries", []):
            pass

        # curves
        for curve in datas.get("curves", []):
            pass

        # patterns
        for pattern in datas.get("patterns", []):
            pass

        # tables
        for table in datas.get("tables", []):
            pass

        # fields
        for field in datas.get("fields", []):
            fid = field["id"]
            var = field["var"]
            domain = field["etype"]
            dtype = field["dtype"]
            f_from = field.get("from", None)
            f_expr = field.get("expr", None)
            if f_from is not None:
                file_name, file_ext = os.path.splitext(f_from)
                if file_ext != ".csv":
                    raise ValueError(f"Unsupported data file type: {file_ext}")
                data = load_data(f_from)
                var_data = data[var].to_numpy()
                field = None  # TODO: implement Field.from_data()
            elif f_expr is not None:
                cell_count = self._mesh.cell_count
                etype = ElementType.from_str(domain)
                dtype = VariableType.from_str(dtype)
                field = Field(self._mesh.get_part_assistant(), dtype, etype)
                for expr in f_expr:
                    zone_id = expr["zone"]
                    val = expr["value"]
                    value = Var(val)
                    if zone_id is None:
                        for i in range(cell_count):
                            field[i] = value
                    else:
                        cell_ids, _ = self._mesh.get_group(zone_id)
                        for cid in cell_ids:
                            field[cid] = value
            else:
                raise ValueError("Invalid field definition.")
            self._datas[fid] = field

    def _load_operators(self):
        """Load operators from configuration."""
        operators = self._model_configs["OPERATORS"]
        for op in operators:
            params = op.get("params", {}) or {}
            method = op["method"]
            scheme, operator = method.split("::")
            if scheme != "fvm":
                raise ValueError("SurfaceWaterModel only supports fvm scheme.")

            instance = fvm_operators[operator](**params)
            op_type = instance.get_type().value
            if op_type in self._operators:
                raise ValueError(f"Duplicated operator: {op_type}, {operator}.")
            self._operators[op_type] = instance

    def _load_solver(self):
        """Load solver from configuration."""
        solvers = self._model_configs["SOLVER"]

        # fvm solver
        sid = solvers["id"]
        params = solvers.get("params", {}) or {}
        scheme, solver = solvers["type"].split("::")
        if scheme != "fvm":
            raise ValueError("SurfaceWaterModel only supports fvm scheme.")
        self._solver = fvm_solvers[solver](sid, self._mesh, self._operators)

        # initial conditions
        ics = solvers.get("ics", []) or []
        for ic in ics:
            ic_instance = self._load_ic(ic)
            if ic_instance is None:
                continue
            ic_field = ic["field"]
            self._solver.add_ic(ic_instance, ic_field)

        # boundaries
        bcs = solvers.get("bcs", []) or []
        for bc in bcs:
            bc_instance, bc_elements = self._load_bc(bc)
            if bc_instance is None:
                continue
            bc_field = bc["field"]
            self._solver.add_bc(bc_instance, bc_field, bc_elements, ElementType.FACE)

        # callbacks
        cbs = solvers.get("cbs", []) or []
        for cb in cbs:
            cb_instance = self._load_cb(cb)
            if cb_instance is None:
                continue
            self._solver.add_callback(cb_instance)

        # initialize
        self._solver.initialize(**params)

    def _load_ic(self, ic_confs: dict):
        """Load initial conditions from configuration."""
        ic_id = ic_confs["id"]
        ic_type = ic_confs["method"]
        ic_params = ic_confs.get("params", {})
        ic_from = ic_confs.get("from", None)
        if ic_type == "hotstart":
            if ic_from is None:
                raise ValueError("Hotstart initial condition requires data source.")
            init_field = self._datas.get(ic_from, None)
            if init_field is None:
                logger.warning(f"Data source {ic_from} not found, passed.")
                return None
            ic_params = {"field": init_field}

        ic_instance = init_methods[ic_type](ic_id, **ic_params)
        return ic_instance

    def _load_bc(self, bc_confs: dict):
        """Load boundary conditions from configuration."""
        bc_id = bc_confs["id"]
        bc_type = bc_confs["method"]
        bc_patches = bc_confs["patches"]
        bc_params = bc_confs.get("params", {})

        bc_elements = []
        for patch in bc_patches:
            face_ids, _ = self._mesh.get_group(patch)
            bc_elements.extend(face_ids)

        bc_instance = boundary_conditions[bc_type](bc_id, **bc_params)
        return bc_instance, bc_elements

    def _load_cb(self, cb_confs: dict):
        """Load callbacks from configuration."""
        cb_id = cb_confs["id"]
        cb_type = cb_confs["method"]
        cb_params = cb_confs.get("params", {})

        cb_instance = callback_handlers[cb_type](cb_id, **cb_params)
        return cb_instance

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "validating")
        errs = []

        if self._start >= self._end:
            errs.append("Time range error: start_time >= end_time")
        if self._dt <= 0:
            errs.append("time_step must bigger than 0")

        self.set_status(models.LinkableComponentStatus.VALID, "validated")
        return errs

    def prepare(self):
        self.set_status(models.LinkableComponentStatus.PREPARING, "preparing")

        self.set_status(models.LinkableComponentStatus.UPDATED, "prepared")

    def update(self, required_outputs: list[links.IOutput] = None):
        self.set_status(models.LinkableComponentStatus.WAITING, "waiting")

        self.set_status(models.LinkableComponentStatus.UPDATING, "updating")
        status = self._solver.inference(self._dt)
        self._current += dt.timedelta(seconds=self._dt)

        if self._current >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, "updated")
        else:
            self.set_status(models.LinkableComponentStatus.UPDATED, "done")

    def finish(self):
        self.set_status(models.LinkableComponentStatus.FINISHING, "finishing")

        # 序列化对象
        if "save_to" in self._model_configs["SPATIAL"]:
            save_to = self._model_configs["SPATIAL"]["save_to"]
            if save_to:
                mesh_file = os.path.join(save_to, f"{self._id}_mesh.pkl")
                mesh_file = os.path.abspath(mesh_file)
                # with open(mesh_file, "wb") as f:
                #     pickle.dump(self._mesh, f) # TODO: TypeError: cannot pickle 'module' object

        if "save_to" in self._model_configs["SOLVER"]:
            save_to = self._model_configs["SOLVER"]["save_to"]
            if save_to:
                solver_file = os.path.join(save_to, f"{self._id}_solver.pkl")
                solver_file = os.path.abspath(solver_file)
                with open(solver_file, "wb") as f:
                    pickle.dump(self._solver, f)

        self.set_status(models.LinkableComponentStatus.FINISHED, "finished")


class SurfaceWaterModelInput(links.BaseInput):
    def set_time(self, timestamp: float):
        time = metas.ITime(timestamp)
        self._timeset = datasets.TimeSet(None, [time])
        self._valueset = None
        self._satisfied = False
        self.notify_changed("surface water model input time reseted")


class SurfaceWaterModelOutput(links.BaseOutput):
    def add_data(self, timestamp: float, value: Any):
        time = metas.ITime(timestamp)
        self._timeset.add_time(time)
        tcount = self._timeset.size
        self._valueset.set_or_add_values((tcount,), value)
        self.notify_changed("surface water model output added data")
