# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Surface water model.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Grid2D, ElementType, Coordinate, MeshFilter, MeshChecker
from core.numerics.fields import (
    Vector,
    Field,
    DataHub,
    Timeseries,
    Curve,
    Pattern,
    Table,
)
from core.solutions.SurfaceWaterModel.IOItems import (
    SurfaceWaterModelInput,
    SurfaceWaterModelOutput,
)
from core.solvers.commons import boundaries, inits, callbacks
from core.solvers import fvm_solvers, fvm_operators
from core.solvers.commons import boundary_conditions, init_methods, callback_handlers

import datetime as dt
import numpy as np
import pickle
import os
import json


class SurfaceWaterModel(models.BaseModel):
    """Surface water/shallow water/two-dimensional flow model,
    based on Grid2D grid and FVM algorithm Burgers equation."""

    def __init__(self, model_id, model_configs: dict, io_configs: dict):
        super().__init__(model_id)
        self._model_configs = model_configs
        self._io_configs = io_configs

        self._mesh: Grid2D = None
        self._solver = None
        self._operators = {}

        self._start: dt.datetime = None
        self._end: dt.datetime = None
        self._current: dt.datetime = None
        self._dt: float = None

        self._save_path = None
        self._load_path = None

    def initialize(self):
        self.set_status(
            models.LinkableComponentStatus.INITIALIZING,
            "SurfaceWaterModel initializing",
        )

        # load- and save-path
        generals = self._model_configs["GLOBAL"]
        self._save_path = generals.get("save_path", "./")
        os.makedirs(self._save_path, exist_ok=True)
        self._load_path = generals.get("load_path", "./")

        # time-axis
        times = self._model_configs["TEMPORAL"]
        self._start = dt.datetime.strptime(times["start_time"], "%Y-%m-%d %H:%M:%S")
        self._end = dt.datetime.strptime(times["end_time"], "%Y-%m-%d %H:%M:%S")
        self._dt = times["time_step"]
        self._current = self._start

        # datas
        self._load_datas()

        # mesh
        self._load_mesh()

        # operators
        self._load_operators()

        # solver

        # initial conditions

        # boundaries

        # outputs

        # inputs

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "initialized")

    def _load_datas(self):
        """Load datas from configuration."""
        datas = self._model_configs["DATAS"]
        self._datas = {}

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
            ptype = patch["type"]
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

            if not MeshChecker.check_face_patch_connectivity(self._mesh, pid):
                raise ValueError(f"Face patch {pid} is not connected.")

            self._mesh.set_group(ptype, pid, face_ids)

        # zones
        zones = spatials.get("zones", [])
        for zone in zones:
            zid = zone["id"]
            ztype = zone["type"]
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

            self._mesh.set_group(ztype, zid, cell_ids)

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

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "validating")
        errs = []

        if self._start >= self._end:
            errs.append("Time range error: start_time >= end_time")
        if self._dt <= 0:
            errs.append("time_step must bigger than 0")

        if not os.path.exists(self._load_path):
            errs.append(f"load_path {self._load_path} doesn't exist")

        self.set_status(models.LinkableComponentStatus.VALID, "validated")
        return errs

    def prepare(self):
        self.set_status(models.LinkableComponentStatus.PREPARING, "preparing")
        # 把初始条件从第一个输入拉进来（若存在）
        # if self._inputs:
        #     ic_field = self._inputs[0].values  # 外部已挂初始场
        #     # TODO: 把 ValueSet → CellField 赋给 solver
        # else:
        #     # 默认热启动
        #     hot = inits.HotstartInitialization("ic", Vector(1, 1))
        #     self._solver.add_ic("u", hot)

        # # 预分配输出
        # for out in self._outputs:
        #     out.add_data(self._current_time.timestamp(), self._solver._fields["u"])
        self.set_status(models.LinkableComponentStatus.UPDATED, "prepared")

    def update(self, required_outputs: list[links.IOutput] = None):
        self.set_status(models.LinkableComponentStatus.WAITING, "waiting")
        # # 拉取上游最新初始场（若有）
        # if self._inputs:
        #     ic_flat = self._inputs[0].values[0, :]  # (nCells*2,)
        #     self._solver._fields["u"].data = ic_flat.reshape(-1, 2)

        # self.set_status(models.LinkableComponentStatus.UPDATING, "updating")
        # # 推进一个物理步
        # status = self._solver.inference(self._dt)
        self._current += dt.timedelta(seconds=self._dt)

        # # 写输出
        # for out in self._outputs:
        #     out.add_data(self._current_time.timestamp(), self._solver._fields["u"])

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
                mesh_file = os.path.join(
                    self._save_path, save_to, f"{self._id}_mesh.pkl"
                )
                mesh_file = os.path.abspath(mesh_file)
                with open(mesh_file, "wb") as f:
                    pickle.dump(self._mesh, f)

        self.set_status(models.LinkableComponentStatus.FINISHED, "finished")
