# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Pipe network models.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Node, Coordinate
import datetime
import numpy as np


class PipeModel(models.BaseModel):
    """管道排水：输入上游水位，输出流速+压力"""

    def __init__(self):
        super().__init__()
        self._build_arguments()
        self._states = {"velocity": None, "pressure": None}
        self._inputs: list[PipeInput] = []
        self._outputs: list[PipeOutput] = []

    def _build_arguments(self):
        self._arguments = {
            "dia_pipe_m": metas.Argument("dia_pipe_m", float, value=1.0, readonly=True),
            "length_m": metas.Argument("length_m", float, value=100.0, readonly=True),
            "friction_factor": metas.Argument(
                "friction_factor", float, value=0.02, readonly=True
            ),
            "start_time": metas.Argument(
                "start_time", str, value="2025-01-01 00:00:00"
            ),
            "end_time": metas.Argument("end_time", str, value="2025-01-02 00:00:00"),
            "time_step": metas.Argument(
                "time_step", str, value="1:00:00", optional=True, default="1:00:00"
            ),
        }

    def setup(
        self, inputs_config=None, outputs_config=None, args_config=None, **kwargs
    ):
        if self._status != models.LinkableComponentStatus.CREATED:
            raise ValueError("模型已经初始化过了，不能再次初始化")

        if inputs_config is not None:
            h_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.METER.value,
                caption="WaterLevel",
                description="Upstream water level",
            )
            for cfg in inputs_config:
                st_name = cfg.get("station", "h_up")
                loc = cfg.get("location", (0.0, 0.0))
                elem = Node(0, Coordinate(*loc))
                elems = datasets.ElementSet(None, None, [elem.id])
                inp = PipeInput(st_name, self, h_def, elems, None, "h_up", "h_up")
                self._inputs.append(inp)

        if outputs_config is not None:
            v_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.VELOCITY.value,
                caption="Velocity",
                description="Pipe velocity",
            )
            p_def = metas.Quantity(
                0.0,
                metas.IUnit("kPa", "kilopascal", metas.IDimension(), 1000, 0),
                caption="Pressure",
                description="Pipe pressure",
            )
            loc = (0.0, 0.0)
            elem = Node(0, Coordinate(*loc))
            elems = datasets.ElementSet(None, None, [elem.id])
            for cfg in outputs_config:
                self._outputs.append(
                    PipeOutput(
                        "velocity", self, v_def, elems, None, "velocity", "velocity"
                    )
                )
                self._outputs.append(
                    PipeOutput(
                        "pressure", self, p_def, elems, None, "pressure", "pressure"
                    )
                )
                break

        if args_config:
            for cfg in args_config:
                arg_name = cfg.get("name")
                arg_value = cfg.get("value")
                self._arguments[arg_name].value = arg_value

    def initialize(self):
        self.set_status(models.LinkableComponentStatus.INITIALIZING, "管网初始化")

        self._start = datetime.datetime.strptime(
            self._arguments["start_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        self._end = datetime.datetime.strptime(
            self._arguments["end_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        self._current_time = self._start
        dt_h = int(self._arguments["time_step"].value.split(":")[0])
        self._time_step = datetime.timedelta(hours=dt_h)

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "管网初始化完成")

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "管网验证")

        errs = []
        if self._start >= self._end:
            errs.append("时间区间错误")
        if self._time_step.total_seconds() <= 0:
            errs.append("步长必须>0")

        self.set_status(models.LinkableComponentStatus.VALID, "管网验证完成")
        return errs

    def prepare(self):
        self.set_status(models.LinkableComponentStatus.PREPARING, "管网准备")

        v_def = metas.Quantity(0.0, metas.PredinedUnits.VELOCITY.value)
        p_def = metas.Quantity(
            0.0, metas.IUnit("kPa", "kilopascal", metas.IDimension(), 1000, 0)
        )
        self._states["velocity"] = datasets.ValueSet(v_def, (1, 1), np.array([[0.0]]))
        self._states["pressure"] = datasets.ValueSet(p_def, (1, 1), np.array([[0.0]]))
        for out in self._outputs:
            out.add_data(self._current_time.timestamp(), 0.0)

        self.set_status(models.LinkableComponentStatus.UPDATED, "管网准备完成")

    def update(self, required_outputs: list[links.IOutput]):
        self.set_status(models.LinkableComponentStatus.WAITING, "管网等待数据")

        h_up = 0.0
        for inp in self._inputs:
            inp.set_time(self._current_time.timestamp())
            values = inp.values
            h_up += values[0, 0]

        self.set_status(models.LinkableComponentStatus.UPDATING, "管网更新中")
        # 简单水力：v = sqrt(2gH), p = rho*g*H
        v = np.sqrt(2 * 9.81 * h_up)
        p = 1000 * 9.81 * h_up / 1000  # kPa
        self._states["velocity"][0, 0] = v
        self._states["pressure"][0, 0] = p
        self._current_time += self._time_step
        for out in self._outputs:
            value = v if out.caption == "velocity" else p
            out.add_data(self._current_time.timestamp(), value)

        self.set_status(models.LinkableComponentStatus.UPDATED, "管网更新完成")
        if self._current_time >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, "管网完成")

    def finish(self):
        self.set_status(models.LinkableComponentStatus.FINISHING, "管网结束")
        self.set_status(models.LinkableComponentStatus.FINISHED, "管网结束")


class PipeInput(links.BaseInput):
    def set_time(self, timestamp: float):
        time = metas.ITime(timestamp)
        self._timeset = datasets.TimeSet(None, [time])
        self._valueset = None
        self._satisfied = False
        self.notify_changed("pipe input time reset")


class PipeOutput(links.BaseOutput):
    def add_data(self, timestamp: float, value: float):
        time = metas.ITime(timestamp)
        self._timeset.add_time(time)
        count = self._timeset.size
        self._valueset.set_or_add_values((count,), value)
        self.notify_changed("pipe output added data")
