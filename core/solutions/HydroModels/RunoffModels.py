# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Hydrologic model solution.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Node, Coordinate, ElementType
import datetime
import numpy as np


class RunoffModel(models.BaseModel):
    """水文产流模型：输入降雨+用地类型，输出流域流量"""

    def __init__(
        self,
        location: tuple[float, float],
        area_km2: float,
        start_time: str,
        end_time: str,
        time_step: str,
        const_rain: float = 0.0,
        land_type: str = "farm",
        scale: float = 1.0,
    ):
        super().__init__()
        self._arguments = {
            "start_time": metas.Argument("start_time", str, value=start_time),
            "end_time": metas.Argument("end_time", str, value=end_time),
            "time_step": metas.Argument(
                "time_step", str, value=time_step, optional=True, default="1:00:00"
            ),
            "location": metas.Argument(
                "location", tuple, value=location, readonly=True
            ),
            "area": metas.Argument(
                "area", float, value=area_km2 * 1e6, readonly=True, default=0.0
            ),
            "land_type": metas.Argument(
                "land_type",
                str,
                value=land_type,
                readonly=True,
                default="farm",
                possibles=["urban", "forest", "farm", "water"],
            ),
            "scale": metas.Argument(
                "scale", float, value=scale, optional=True, default=1.0
            ),
        }
        self._states = {"rainfall": None, "runoff": None}
        self._inputs: list[RunoffInput] = []
        self._outputs: list[RunoffOutput] = []

        self._const_rain = const_rain
        self._start = None
        self._end = None
        self._current_time = None

    def setup(
        self,
        inputs_config: list = None,
        outputs_config: list = None,
        args_config: list = None,
        **kwargs,
    ):
        # 检查模型状态，不允许任意重置
        if self._status != models.LinkableComponentStatus.CREATED:
            raise ValueError("模型已经初始化过了，不能再次初始化")

        # 动态组装inputs/outputs，每个组件的input/output定义方式由文档说明
        # NOTE: 当组件setup()之后，外部通过遍历inputs/outputs属性，
        # 完成provider/consumer的绑定。
        if inputs_config is not None:
            # input的定义要精确具体对象和变量。
            rainfall_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.MILLIMETER_PER_HOUR.value,
                caption="Rainfall",
                description="Rainfall intensity",
            )
            for input_config in inputs_config:
                rain_station = input_config.get("rain_station", "none")
                location = input_config.get("location", (0.0, 0.0))

                elem = Node(0, Coordinate(*location))
                elems = datasets.ElementSet(None, None, [elem.id])

                cur_input = RunoffInput(
                    rain_station,
                    self,
                    rainfall_def,
                    elems,
                    None,
                    "rainfall",
                    "rainfall",
                    None,
                )
                self.inputs.append(cur_input)

        if outputs_config is not None:
            # output的定义只要明确变量，一次输出所有对象。
            runoff_def = metas.Quantity(
                0.0,
                metas.PredinedUnits.DISCHARGE.value,
                caption="Runoff",
                description="Runoff flow",
            )
            location = self._arguments["location"].value
            elem = Node(0, Coordinate(*location))
            elems = datasets.ElementSet(None, None, [elem.id])
            for output_config in outputs_config:
                runoff = output_config.get("runoff")

                cur_output = RunoffOutput(
                    runoff, self, runoff_def, elems, None, "runoff", "runoff"
                )
                self.outputs.append(cur_output)
                break

        # 更新配置参数
        if args_config is not None:
            for arg_config in args_config:
                arg_name = arg_config.get("name")
                arg_value = arg_config.get("value")
                self._arguments[arg_name].value = arg_value

    def initialize(self):
        self.set_status(models.LinkableComponentStatus.INITIALIZING, "初始化模型")

        self._start = datetime.datetime.strptime(
            self._arguments["start_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        self._end = datetime.datetime.strptime(
            self._arguments["end_time"].value, "%Y-%m-%d %H:%M:%S"
        )
        h, m, s = map(float, self._arguments["time_step"].value.split(":"))
        self._time_step = datetime.timedelta(hours=h, minutes=m, seconds=s)
        self._current_time = self._start

        self.set_status(models.LinkableComponentStatus.INITIALIZED, "模型初始化完成")

    def validate(self) -> list[str]:
        self.set_status(models.LinkableComponentStatus.VALIDATING, "验证模型")

        errors = []
        if self._start >= self._end:
            errors.append("起始时间必须早于结束时间")
        if self._time_step.total_seconds() <= 0:
            errors.append("时间步长必须大于0")
        if self._const_rain < 0:
            errors.append("常数降雨必须大于等于0")

        self.set_status(models.LinkableComponentStatus.VALID, "模型验证完成")
        return errors

    def prepare(self):
        self.set_status(models.LinkableComponentStatus.PREPARING, "准备模型")

        # 准备状态变量
        rainfall_def = metas.Quantity(
            0.0,
            metas.PredinedUnits.MILLIMETER_PER_HOUR.value,
            caption="Rainfall",
            description="Rainfall intensity",
        )
        rainfall = datasets.ValueSet(
            rainfall_def,
            (1, 1),
            np.array([[self._const_rain]]),
        )
        self._states["rainfall"] = rainfall

        runoff_def = metas.Quantity(
            0.0,
            metas.PredinedUnits.DISCHARGE.value,
            caption="Runoff",
            description="Runoff flow",
        )
        runoff = datasets.ValueSet(
            runoff_def,
            (1, 1),
            np.array([[0.0]]),
        )
        self._states["runoff"] = runoff

        # 激活outputs
        for output in self._outputs:
            output.add_data(self._current_time.timestamp(), 0.0)

        self.set_status(models.LinkableComponentStatus.UPDATED, "模型准备完成")

    def update(self, required_outputs: list[links.IOutput]):
        # 从inputs中取出数据
        self.set_status(models.LinkableComponentStatus.WAITING, "模型等待数据")
        total_extra_rain = 0.0
        for input in self._inputs:
            # 配置时间
            input.set_time(self._current_time.timestamp())
            # 拉取数据
            rain = input.values
            # total_extra_rain += rain[0, 0].to_si() # 当前没有严格实现ValueSet
            total_extra_rain += rain[0, 0]

        # 简单线性产流：Q = scale * alpha * P * area / 3600
        self.set_status(models.LinkableComponentStatus.UPDATING, "模型更新中")
        alpha = 0.6 if self._arguments["land_type"].value == "urban" else 0.3
        rain = self._states["rainfall"][0, 0]
        total_rain = rain + total_extra_rain
        rain_mps = total_rain
        q = (
            self._arguments["scale"].value
            * alpha
            * rain_mps
            * self._arguments["area"].value
            / 3600
        )  # m3/s

        # 更新状态变量
        self._current_time += self._time_step
        self._states["runoff"][0, 0] = q

        # 更新outputs
        for output in self._outputs:
            output.add_data(self._current_time.timestamp(), q)

        self.set_status(models.LinkableComponentStatus.UPDATED, "模型更新完成")

        # 检查是否已完成
        if self._current_time >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, "模型完成")

    def finish(self):
        self.set_status(models.LinkableComponentStatus.FINISHING, "模型结束中")

        # 输出结果
        for name, values in self._states.items():
            print(f"{name}: {values}")

        self.set_status(models.LinkableComponentStatus.FINISHED, "模型结束")


class RunoffInput(links.BaseInput):
    """降雨站输入项"""

    def set_time(self, timestamp: float):
        time = metas.ITime(timestamp)
        self._timeset = datasets.TimeSet(None, [time])
        self._valueset = None
        self._satisfied = False

        self.notify_changed("reset time_set")


class RunoffOutput(links.BaseOutput):
    """流域流量输出项"""

    def get_values(self, querier: links.IBaseExchangeItem) -> datasets.IValueSet:
        if self._in_get_values:
            return self._valueset[-1]

        self._in_get_values = True
        req_time = querier.time_set.times[0].timestamp

        while self._timeset.times[-1].timestamp < req_time:
            self._component.update([self])

        self._in_get_values = False

        req_index = -1
        for i, time in enumerate(self._timeset.times):
            if time.timestamp >= req_time:
                req_index = i
                break

        if req_index == -1:
            return self._valueset[-1]
        return self._valueset.get_values_for_time(req_index).reshape((1, -1))

    def add_data(self, timestamp: float, value: float):
        time = metas.ITime(timestamp)
        self._timeset.add_time(time)
        count = self._timeset.size
        self._valueset.set_or_add_values((count,), value)
        self.notify_changed("add_data")
