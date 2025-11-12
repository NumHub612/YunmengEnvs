# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Input and output items for surface water model.
"""
from core.solutions.commons import links, metas, datasets
from typing import Any


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
