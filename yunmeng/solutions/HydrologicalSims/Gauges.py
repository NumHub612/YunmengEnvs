# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Rainfall / evaporation station: a data source model driven by time series.
"""

from __future__ import annotations

from yunmeng.solutions.standards import ModelMeta, Quantity
from yunmeng.solutions.commons.models import BaseModel
from yunmeng.solutions.commons.datasets import (
    PointElementSet,
    Timeseries,
    Quantities,
)


class GaugeModel(BaseModel):
    """Station data source (rainfall or evaporation)."""

    def __init__(
        self,
        model_id: str,
        series: Timeseries,
        quantity: Quantity,
        port_name: str = None,
        position: tuple[float, float, float] = None,
    ):
        super().__init__(model_id, ModelMeta(name=model_id, category="gauge"))
        self._series = series
        self._cursor = 0
        elements = (
            PointElementSet(centers=[[*position]], element_ids=[model_id])
            if position is not None
            else None
        )
        port_id = f"{model_id}.{port_name or quantity.name}"
        self._out = self.create_output(quantity, elements, port_id=port_id)
        if len(series) >= 2:
            self._dt = float(series.time[1] - series.time[0])
        else:
            self._dt = 3600.0

    # -- info ---------------------------------------

    @property
    def series(self) -> Timeseries:
        return self._series

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def output_port(self):
        return self._out

    # -- lifecycle -----------------------------------------

    def _do_update(self, inquirers=None):
        if self._cursor >= len(self._series):
            self.mark_done()
            return
        self._out.add_values([self._series.value_at(self._cursor)])
        self._cursor += 1
        if self._cursor >= len(self._series):
            # publish the last frame once more, then stop
            pass

    def _do_finish(self):
        self._cursor = 0


def RainGauge(model_id: str, series: Timeseries, **kw) -> GaugeModel:
    """Rainfall station (output precipitation [mm])."""
    return GaugeModel(
        model_id,
        series,
        Quantities.PRECIPITATION,
        **kw,
    )


def EvapGauge(model_id: str, series: Timeseries, **kw) -> GaugeModel:
    """Evaporation station (output evaporation [mm])."""
    return GaugeModel(
        model_id,
        series,
        Quantities.EVAPORATION,
        **kw,
    )
