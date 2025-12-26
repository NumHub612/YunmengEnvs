# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Hydrologic model solution.
"""
from core.solutions.commons import models, datasets, links, metas
from core.numerics.mesh import Node, Coordinate, ElementType
from core.numerics.fields import (
    Timeseries,
    Curve,
    Pattern,
    Table,
)
import datetime as dt
import numpy as np
from dateutil.parser import parse


class RunoffModel(models.BaseModel):
    """RunoffModel, input: rainfall and landtype, output: runoff."""

    def __init__(self, id: str, model_configs: dict, link_configs: dict):
        super().__init__(id)
        self._arguments = [
            metas.Argument("area", float, readonly=False, default=0.0),
            metas.Argument(
                "land_type",
                str,
                readonly=False,
                default="farm",
                possibles=["urban", "forest", "farm", "water"],
            ),
            metas.Argument(
                "soil_type",
                str,
                readonly=False,
                default="clay",
                possibles=["clay", "silt", "sand"],
            ),
        ]

        self._model_configs = model_configs
        self._link_configs = link_configs
        self._location = None
        self._rainfall = None
        self._land_soil_table = None

        self._inputs: list[RunoffInput] = []
        self._outputs: list[RunoffOutput] = []

        self._start: dt.datetime = None
        self._end: dt.datetime = None
        self._current: dt.datetime = None
        self._dt: dt.timedelta = None

    def initialize(self):
        self.set_status(
            models.LinkableComponentStatus.INITIALIZING,
            f"Initializing RunoffModel: {self._id}",
        )
        predefined_units = {
            unit.value.caption: unit.value for unit in metas.PredefinedUnits
        }

        # load- and save-path
        envs = self._model_configs["ENV"]

        # temporal
        times = self._model_configs["TEMPORAL"]
        self._start = parse(times["start_time"])
        self._end = parse(times["end_time"])
        self._dt = dt.timedelta(seconds=times["time_step"])
        self._current = self._start

        # spatial
        spatials = self._model_configs["SPATIAL"]
        mesh_type = spatials["type"]
        if mesh_type != "NODE":
            raise ValueError("RunoffModel doesn't support non-NODE spatial type.")
        params = spatials["params"]
        coor = Coordinate(params["x"], params["y"])
        self._location = Node(self._id, coor)

        # datas
        self._load_datas()

        # customs
        args = self._model_configs.get("CUSTOMS", {})
        self.arguments[0].value = args["area_km2"]
        self.arguments[1].value = args["land_type"]
        self.arguments[2].value = args["soil_type"]

        # outputs
        outputs_config = self._link_configs.get("outputs", None)
        if outputs_config is not None:
            for output_config in outputs_config:
                outlet = output_config.get("id")
                # quantity
                quantity_config = output_config.get("quantity")
                var = quantity_config.get("variable")
                if var.upper() != "RUNOFF":
                    raise ValueError("RunoffModel only supports runoff output.")

                var_unit = predefined_units.get("m3/s")
                quantity = metas.Quantity(0.0, var_unit)

                # location
                elements = datasets.ElementSet(None, ElementType.NONE, [self._location])

                cur_output = RunoffOutput(outlet, self, quantity, elements, None)
                self._outputs.append(cur_output)
                break  # currently only one output

        self.set_status(
            models.LinkableComponentStatus.INITIALIZED, f"{self._id} Initialized"
        )

    def _load_datas(self):
        """Load datas from configuration."""
        datas = self._model_configs["DATAS"]

        # timeseries
        for ts in datas.get("timeseries", []):
            ts_id = ts["id"]
            xs = ts.get("xs", None)
            ys = ts.get("ys", None)
            ts_from = ts.get("from", None)
            ts_expr = ts.get("expr", None)
            if xs and ys:
                ts_obj = Timeseries(ts_id, xs, ys)
            elif ts_expr:
                ts_obj = Timeseries.from_expr(ts_id, **ts_expr)
            elif ts_from:
                pass
            else:
                raise ValueError(f"Timeseries {ts_id} lack of source.")
            self._rainfall = ts_obj
            break  # currently only one rainfall timeseries

        # tables
        for table in datas.get("tables", []):
            table_id = table["id"]
            xs = table.get("xs", None)
            ys = table.get("ys", None)
            zs = table.get("zs", None)
            vs = table.get("vs", None)
            table_from = table.get("from", None)
            table_expr = table.get("expr", None)
            if xs and ys and vs:
                table_obj = Table(table_id, np.array(vs), np.array(xs), np.array(ys))
            elif table_expr:
                table_obj = Table.from_expr(table_expr)
            elif table_from:
                pass
            else:
                raise ValueError(f"Table {table_id} lack of source.")
            self._land_soil_table = table_obj
            break  # currently only one land-soil table

    def validate(self) -> list[str]:
        self.set_status(
            models.LinkableComponentStatus.VALIDATING,
            f"Validating RunoffModel: {self._id}",
        )

        errors = []
        if self._start >= self._end:
            errors.append("Time range error: start_time >= end_time")
        if self._dt.total_seconds() <= 0:
            errors.append("time_step must bigger than 0")

        if self._location is None:
            errors.append("Spatial location is missing.")
        if self._rainfall is None:
            errors.append("Rainfall timeseries is missing.")
        if self._land_soil_table is None:
            errors.append("Land-soil table is missing.")

        self.set_status(models.LinkableComponentStatus.VALID, f"{self._id} Validated")
        return errors

    def prepare(self):
        self.set_status(
            models.LinkableComponentStatus.PREPARING,
            f"Preparing RunoffModel: {self._id}",
        )

        # Prepare outputs
        for output in self._outputs:
            output.add_data(self._current.timestamp(), 0.0)

        self.set_status(models.LinkableComponentStatus.UPDATED, f"{self._id} Prepared")

    def update(self, required_outputs: list[links.IOutput]):
        # Get required inputs
        self.set_status(
            models.LinkableComponentStatus.WAITING, f"Waiting inputs: {self._id}"
        )

        total_extra_rain = 0.0
        for input in self._inputs:
            # Set time
            input.set_time(self._current.timestamp())
            # Pull values
            total_extra_rain += input.values[0, 0].to_si()  # in m/s

        # Update runoff: Q =  alpha * rainfall * area
        self.set_status(
            models.LinkableComponentStatus.UPDATING, f"Updating RunoffModel: {self._id}"
        )

        alpha = self._land_soil_table.get_value(
            self.arguments[1].value, self.arguments[2].value
        )  # dimensionless
        rain = self._rainfall.get_value(self._current.timestamp())  # in mm/h
        total_rain = rain * 0.000277778 + total_extra_rain
        q = alpha * total_rain * self._arguments[0].value * 1e6  # in m3/s

        self._current += self._dt

        # Update outputs
        for output in self._outputs:
            output.add_data(self._current.timestamp(), q)

        self.set_status(models.LinkableComponentStatus.UPDATED, f"{self._id} Updated")

        # Check if finished
        if self._current >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, f"{self._id} Done")

    def finish(self):
        self.set_status(
            models.LinkableComponentStatus.FINISHING,
            f"Finishing RunoffModel: {self._id}",
        )

        self.set_status(models.LinkableComponentStatus.FINISHED, f"{self._id} Finished")


class RunoffInput(links.BaseInput):

    def set_time(self, timestamp: float):
        time = metas.ITime(timestamp)
        self._timeset = datasets.TimeSet(None, [time])
        self._valueset = None
        self._satisfied = False

        self.notify_changed("reset time_set")


class RunoffOutput(links.BaseOutput):

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
        return self._valueset.get_values_for_time(req_index)

    def add_data(self, timestamp: float, value: float):
        time = metas.ITime(timestamp)
        self._timeset.add_time(time)
        count = self._timeset.size
        self._valueset.set_or_add_values((count,), value)
        self.notify_changed("add_data")
