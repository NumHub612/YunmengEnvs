# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

River network models.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Node, Coordinate
import datetime
import numpy as np


class RiverModel(models.BaseModel):
    """河道汇流：输入上游流量，输出水位+下游流量"""

    def __init__(self):
        super().__init__()
        self._build_arguments()
        self._states = {"water_level": None, "outflow": None}
        self._inputs: list[RiverInput] = []
        self._outputs: list[RiverOutput] = []

    def _build_arguments(self):
        self._arguments = {
            "k_stage": metas.Argument("k_stage", float, value=0.5, readonly=True),
            "length_m": metas.Argument("length_m", float, value=1000.0, readonly=True),
            "bottom_width_m": metas.Argument(
                "bottom_width_m", float, value=10.0, readonly=True
            ),
            "mannning_n": metas.Argument(
                "mannning_n", float, value=0.035, readonly=True
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
            q_in_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.DISCHARGE.value,
                caption="Inflow",
                description="Upstream inflow",
            )
            for cfg in inputs_config:
                st_name = cfg.get("station", "inflow")
                loc = cfg.get("location", (0.0, 0.0))
                elem = Node(0, Coordinate(*loc))
                elems = datasets.ElementSet(None, None, [elem.id])
                inp = RiverInput(
                    st_name, self, q_in_def, elems, None, "inflow", "inflow"
                )
                self._inputs.append(inp)

        if outputs_config is not None:
            h_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.METER.value,
                caption="WaterLevel",
                description="River stage",
            )
            q_out_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.DISCHARGE.value,
                caption="Outflow",
                description="Downstream flow",
            )
            loc = (0.0, 0.0)  # 单点输出
            elem = Node(0, Coordinate(*loc))
            elems = datasets.ElementSet(None, None, [elem.id])
            for cfg in outputs_config:
                self._outputs.append(
                    RiverOutput("stage", self, h_def, elems, None, "stage", "stage")
                )
                self._outputs.append(
                    RiverOutput(
                        "outflow", self, q_out_def, elems, None, "outflow", "outflow"
                    )
                )
                break  # 只做一组

        if args_config:
            for cfg in args_config:
                arg_name = cfg.get("name")
                arg_value = cfg.get("value")
                self._arguments[arg_name].value = arg_value

    def initialize(self):
        self.set_status(models.LinkableComponentStatus.INITIALIZING, "河道初始化")

        self._start = datetime.datetime.strptime(
            self._arguments["start_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        self._end = datetime.datetime.strptime(
            self._arguments["end_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        self._current_time = self._start
        dt_h = int(self._arguments["time_step"].value.split(":")[0])
        self._time_step = datetime.timedelta(hours=dt_h)

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "河道初始化完成")

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "河道验证")

        errs = []
        if self._start >= self._end:
            errs.append("时间区间错误")
        if self._time_step.total_seconds() <= 0:
            errs.append("步长必须>0")

        self.set_status(models.LinkableComponentStatus.VALID, "河道验证完成")
        return errs

    def prepare(self):
        self.set_status(models.LinkableComponentStatus.PREPARING, "河道准备")

        # 预分配状态
        h_def = metas.Quantity(0.0, metas.PredinedUnits.METER.value)
        q_def = metas.Quantity(0.0, metas.PredinedUnits.DISCHARGE.value)
        # TODO: ValueSet的元素应该为Quantity的元素
        self._states["water_level"] = datasets.ValueSet(
            h_def, (1, 1), np.array([[0.0]])
        )
        self._states["outflow"] = datasets.ValueSet(q_def, (1, 1), np.array([[0.0]]))
        # 输出赋初值
        for out in self._outputs:
            out.add_data(self._current_time.timestamp(), 0.0)

        self.set_status(models.LinkableComponentStatus.UPDATED, "河道准备完成")

    def update(self, required_outputs: list[links.IOutput]):
        self.set_status(models.LinkableComponentStatus.WAITING, "河道等待数据")

        total_q = 0.0
        for inp in self._inputs:
            inp.set_time(self._current_time.timestamp())
            values = inp.values
            total_q += values[0, 0]

        self.set_status(models.LinkableComponentStatus.UPDATING, "河道更新中")
        # 简单水位-流量关系：H = k*Q，Q_out = Q_in
        k = self._arguments["k_stage"].value
        h = k * total_q
        self._states["water_level"][0, 0] = h
        self._states["outflow"][0, 0] = total_q

        self._current_time += self._time_step
        for out in self._outputs:
            out.add_data(
                self._current_time.timestamp(), out.caption == "stage" and h or total_q
            )

        self.set_status(models.LinkableComponentStatus.UPDATED, "河道更新完成")
        if self._current_time >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, "河道完成")

    def finish(self):
        self.set_status(models.LinkableComponentStatus.FINISHING, "河道结束")
        # 可写文件
        self.set_status(models.LinkableComponentStatus.FINISHED, "河道结束")


class RiverInput(links.BaseInput):
    def set_time(self, timestamp: float):
        time = metas.ITime(timestamp)
        self._timeset = datasets.TimeSet(None, [time])
        self._valueset = None
        self._satisfied = False
        self.notify_changed("river input time reset")


class RiverOutput(links.BaseOutput):
    def add_data(self, timestamp: float, value: float):
        time = metas.ITime(timestamp)
        self._timeset.add_time(time)
        tcount = self._timeset.size
        self._valueset.set_or_add_values((tcount,), value)
        self.notify_changed("river output added data")
