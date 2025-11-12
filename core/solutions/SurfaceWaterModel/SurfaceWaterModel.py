# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Surface water model.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Grid2D, Node, Coordinate
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


class SurfaceWaterModel(models.BaseModel):
    """Surface water/shallow water/two-dimensional flow model,
    based on Grid2D grid and FVM algorithm Burgers equation."""

    def __init__(self, model_configs: dict, io_configs: dict):
        super().__init__()
        self._model_configs = model_configs
        self._io_configs = io_configs
        self._params = None

        self._mesh: Grid2D = None
        self._solver = None
        self._operators = None

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
        self._load_path = generals.get("load_path", "./")

        # time-axis
        times = self._model_configs["TEMPORAL"]
        self._start = dt.datetime.strptime(times["start_time"], "%Y-%m-%d %H:%M:%S")
        self._end = dt.datetime.strptime(times["end_time"], "%Y-%m-%d %H:%M:%S")
        self._dt = times["time_step"]
        self._current = self._start

        # mesh
        spatials = self._model_configs["SPATIAL"]
        mesh_type = spatials["type"]
        if mesh_type != "Grid2D":
            raise ValueError("SurfaceWaterModel doesn't support non-Grid2D mesh")
        low_left = spatials["params"]["lower_left"]
        up_right = spatials["params"]["upper_right"]
        nx = spatials["params"]["nx"]
        ny = spatials["params"]["ny"]
        self._mesh = Grid2D(Coordinate(*low_left), Coordinate(*up_right), nx, ny)

        # patches and zones
        patches = spatials.get("patches", [])

        # operators

        # solver

        # initial conditions

        # boundaries

        # outputs

        # inputs

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "initialized")

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "validating")
        errs = []

        if self._start >= self._end:
            errs.append("Time range error: start_time >= end_time")
        if self._dt <= 0:
            errs.append("dt must bigger than 0")

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
        # 可落盘、释放显存等
        self.set_status(models.LinkableComponentStatus.FINISHED, "finished")
