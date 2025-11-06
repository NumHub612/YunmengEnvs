import unittest
import os
import sys

from core.solutions.HydroModels import (
    RiverModel,
    RiverInput,
    RiverOutput,
    PipeModel,
    PipeInput,
    PipeOutput,
    RunoffModel,
    RunoffInput,
    RunoffOutput,
)

from core.solutions.commons import datasets, models, metas
from core.numerics.mesh import Coordinate
import numpy as np


class TestHydroModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_usage(self):
        # 1. 实例化
        rr = RunoffModel(
            location=(0, 0),
            area_km2=1.0,
            start_time="2025-01-01 00:00:00",
            end_time="2025-01-02 00:00:00",
            time_step="1:00:00",
            const_rain=15.0,
            land_type="urban",
        )

        river = RiverModel()
        pipe = PipeModel()

        # 2. 动态配置inputs/outputs
        rr.setup(
            # inputs_config=[{"rain_station": "rs1", "location": (0, 0)}],
            outputs_config=[{"runoff": "out1"}],
        )

        river.setup(
            inputs_config=[{"station": "inflow", "location": (0, 0)}],
            outputs_config=[{"stage": "stage", "outflow": "outflow"}],
            args_config=[
                {"name": "start_time", "value": "2025-01-01 00:00:00"},
                {"name": "end_time", "value": "2025-01-02 00:00:00"},
            ],
        )

        pipe.setup(
            inputs_config=[{"station": "h_up", "location": (0, 0)}],
            outputs_config=[{"velocity": "v", "pressure": "p"}],
            args_config=[
                {"name": "start_time", "value": "2025-01-01 00:00:00"},
                {"name": "end_time", "value": "2025-01-02 00:00:00"},
            ],
        )

        # 3. 拓扑连接
        river.inputs[0].provider = rr.outputs[0]  # runoff → inflow
        pipe.inputs[0].provider = river.outputs[0]  # stage → h_up

        # 4. 初始化
        rr.initialize()
        river.initialize()
        pipe.initialize()

        # 5. 检查
        rr.validate()
        river.validate()
        pipe.validate()

        # 6. 准备
        rr.prepare()
        river.prepare()
        pipe.prepare()

        # 7. 运行
        t = 0
        while (
            pipe.status != models.LinkableComponentStatus.DONE
            and pipe.status != models.LinkableComponentStatus.FAILED
        ):
            pipe.update(None)
            t += 1
            v = pipe.outputs[0].values.get_values_for_time([-1])
            p = pipe.outputs[1].values.get_values_for_time([-1])

            r = rr.outputs[0].values.get_values_for_time([-1])
            q = river.outputs[0].values.get_values_for_time([-1])

            print(
                f"time: {t}, r={r[0,0]:.2f} mm/h  q={q[0,0]:.2f} m3/s  v={v[0,0]:.2f} m/s  p={p[0,0]:.2f} kPa"
            )

        # 8. 结束
        rr.finish()
        river.finish()
        pipe.finish()


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestHydroModels("test_usage"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
