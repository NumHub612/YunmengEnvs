# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

River network models.
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


class RiverModel(models.BaseModel):
    """River network model based on Muskingum model."""

    def __init__(self, id: str, model_configs: dict, link_configs: dict):
        super().__init__(id)
        self._arguments = [
            metas.Argument("ke", float, default=3.0, readonly=False),
            metas.Argument("xe", float, default=0.3, readonly=False),
        ]
        self._model_configs = model_configs
        self._link_configs = link_configs
        self._muskingum_args = []

        self._inputs: list[RiverInput] = []
        self._inputs_map: dict = {}
        self._outputs: list[RiverOutput] = []
        self._outputs_map: dict = {}

        self._start: dt.datetime = None
        self._end: dt.datetime = None
        self._current: dt.datetime = None
        self._dt: dt.timedelta = None

        self._cross_sections: dict = None
        self._sd_curve: Curve = None
        self._up_bounds: Timeseries = None
        self._inflows: np.ndarray = None
        self._outflows: np.ndarray = None
        self._stages: np.ndarray = None

    def initialize(self):
        self.set_status(
            models.LinkableComponentStatus.INITIALIZING,
            f"Initializing RiverModel: {self._id}",
        )

        # temporal
        times = self._model_configs["TEMPORAL"]
        self._start = parse(times["start_time"])
        self._end = parse(times["end_time"])
        self._dt = dt.timedelta(seconds=times["time_step"])
        self._current = self._start

        # spatial
        self._load_rivernet()

        # datas
        self._load_datas()

        # customs
        args = self._model_configs.get("CUSTOMS", {})
        Q0 = args.get("Q0", 0.0)
        Z0 = args.get("Z0", 0.0)
        self._inflows = Q0 * np.ones(len(self._cross_sections))
        self._outflows = Q0 * np.ones(len(self._cross_sections))
        self._stages = Z0 * np.ones(len(self._cross_sections))

        ke = args.get("ke", 3.0)
        xe = args.get("xe", 0.3)
        self._arguments[0].value = ke
        self._arguments[1].value = xe

        _dt = times["time_step"] / 3600.0
        denom = 2 * ke * (1 - xe) + _dt
        C1 = (_dt - 2 * ke * xe) / denom
        C2 = (_dt + 2 * ke * xe) / denom
        C3 = (2 * ke * (1 - xe) - _dt) / denom
        self._muskingum_args = [C1, C2, C3]

        # inputs and outputs
        self._load_links()

        self.set_status(
            models.LinkableComponentStatus.INITIALIZED, f"{self._id} initialized"
        )

    def _load_rivernet(self):
        spatials = self._model_configs["SPATIAL"]
        mesh_type = spatials["type"]
        if mesh_type != "RiverNet":
            raise ValueError("RiverModel only support RiverNet spatial type.")

        params = spatials["params"]
        start_loc = params["start_loc"]
        end_loc = params["end_loc"]
        segments = params["segments"]
        dx = (end_loc[0] - start_loc[0]) / segments
        dy = (end_loc[1] - start_loc[1]) / segments

        self._cross_sections = {}
        for i in range(segments + 1):
            cross_id = f"cross_{i}"
            cross_loc = Node(
                i, Coordinate(start_loc[0] + i * dx, start_loc[1] + i * dy)
            )
            self._cross_sections[cross_id] = cross_loc

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
            self._up_bounds = ts_obj
            break  # currently only one inflow boundaries

        # curves
        for curve in datas.get("curves", []):
            curve_id = curve["id"]
            xs = curve.get("xs", None)
            ys = curve.get("ys", None)
            curve_expr = curve.get("expr", None)
            if xs and ys:
                curve_obj = Curve(curve_id, xs, ys)
            elif curve_expr:
                curve_obj = Curve.from_expr(curve_id, **curve_expr)
            else:
                raise ValueError(f"Curve {curve_id} lack of source.")
            self._sd_curve = curve_obj
            break  # currently only one stage-discharge curve

    def _load_links(self):
        """Load links from configuration."""
        predefined_units = {
            unit.value.caption: unit.value for unit in metas.PredefinedUnits
        }
        # inputs
        inputs_config = self._link_configs.get("inputs", None)
        if inputs_config is not None:
            for input_config in inputs_config:
                station = input_config.get("id")
                # quantity
                quantity_config = input_config.get("quantity")
                var = quantity_config.get("variable")
                if var.upper() != "FLOW":
                    raise ValueError("RiverModel only supports flow output.")

                var_unit = predefined_units.get("m3/s")
                quantity = metas.Quantity(0.0, var_unit)

                # location
                location_config = input_config.get("position")
                element_type = location_config.get("element")
                if element_type != "CrossSection":
                    raise ValueError(
                        f"Unsupported element type {element_type} in input."
                    )
                # TODO: support multiple elements by maping.
                element_id = location_config.get("ids")[0]
                if element_id not in self._cross_sections:
                    raise ValueError(f"Invalid element {element_id} in input.")
                element_node = self._cross_sections[element_id]
                elements = datasets.ElementSet(
                    None,
                    ElementType.NODE,
                    [element_node],
                )

                cur_input = RiverInput(station, self, quantity, elements)
                self.inputs.append(cur_input)
                self._inputs_map[len(self.inputs) - 1] = element_id

        # outputs
        outputs_config = self._link_configs.get("outputs", None)
        if outputs_config is not None:
            for output_config in outputs_config:
                outlet = output_config.get("id")
                # quantity
                quantity_config = output_config.get("quantity")
                var = quantity_config.get("variable")
                if var.upper() == "FLOW":
                    var_unit = predefined_units.get("m3/s")
                elif var.upper() == "WATER_LEVEL":
                    var_unit = predefined_units.get("m")
                else:
                    raise ValueError(f"RiverModel invalid output variable: {var}.")
                quantity = metas.Quantity(0.0, var_unit)

                # location
                location_config = input_config.get("position")
                element_type = location_config.get("element")
                if element_type != "CrossSection":
                    raise ValueError(
                        f"Unsupported element type {element_type} in output."
                    )
                # TODO: support multiple elements by maping.
                element_id = location_config.get("ids")[0]
                if element_id not in self._cross_sections:
                    raise ValueError(f"Invalid element {element_id} in output.")
                element_node = self._cross_sections[element_id]
                elements = datasets.ElementSet(
                    None,
                    ElementType.NODE,
                    [element_node],
                )

                cur_output = RiverOutput(outlet, self, quantity, elements, None)
                self.outputs.append(cur_output)
                self._outputs_map[len(self.outputs) - 1] = (element_id, var.upper())

    def validate(self) -> list[str]:
        self.set_status(
            models.LinkableComponentStatus.VALIDATING,
            f"Validating RiverModel: {self._id}",
        )

        errors = []
        if self._start >= self._end:
            errors.append("Time range error: start_time >= end_time")
        if self._dt.total_seconds() <= 0:
            errors.append("time_step must bigger than 0")

        if len(self._cross_sections) < 2:
            errors.append("RiverNet must have at least 2 cross sections.")

        if self._sd_curve is None:
            errors.append("Stage-discharge curve is missing.")

        if self._up_bounds is None:
            errors.append("Inflow boundaries are missing.")

        self.set_status(models.LinkableComponentStatus.VALID, f"{self._id} valid")
        return errors

    def prepare(self):
        self.set_status(
            models.LinkableComponentStatus.PREPARING,
            f"Preparing RiverModel: {self._id}",
        )

        # Prepare outputs
        cross_sections = list(self._cross_sections.keys())
        for i, out in enumerate(self._outputs):
            element, var = self._outputs_map[i]
            element_index = cross_sections.index(element)
            if var == "FLOW":
                init_value = self._outflows[element_index]
            else:
                init_value = self._stages[element_index]
            out.add_data(self._current.timestamp(), init_value)

        self.set_status(models.LinkableComponentStatus.UPDATED, f"{self._id} prepared")

    def update(self, required_outputs: list[links.IOutput]):
        # Get required inputs
        self.set_status(
            models.LinkableComponentStatus.WAITING, f"Waiting inputs: {self._id}"
        )

        inflows = {}
        for i, inp in enumerate(self._inputs):
            inp.set_time(self._current.timestamp())
            inflow = inp.values[0, 0].to_si()  # m3/s
            inflows[self._inputs_map[i]] = inflow

        # Update states
        self.set_status(
            models.LinkableComponentStatus.UPDATING, f"Updating RiverModel: {self._id}"
        )

        # upstream boundary
        bc_inflow = self._up_bounds.get_value(self._current.timestamp())  # m3/s
        pre_inflow = self._inflows[0]
        cur_inflow = bc_inflow + inflows.get("cross_0", 0.0)
        pre_outflow = self._outflows[0]
        cur_outflow = (
            self._muskingum_args[0] * pre_inflow
            + self._muskingum_args[1] * cur_inflow
            + self._muskingum_args[2] * pre_outflow
        )
        cur_stage = self._sd_curve.inverse(cur_inflow)
        self._stages[0] = cur_stage
        self._inflows[0] = cur_inflow
        self._outflows[0] = cur_outflow

        # river routing
        cross_sections = list(self._cross_sections.keys())
        for cid in cross_sections[1:]:
            i = cross_sections.index(cid)
            lateral_inflow = inflows.get(cid, 0.0)
            pre_inflow = self._inflows[i]
            cur_inflow = self._outflows[i - 1] + lateral_inflow
            pre_outflow = self._outflows[i]
            cur_outflow = (
                self._muskingum_args[0] * pre_inflow
                + self._muskingum_args[1] * cur_inflow
                + self._muskingum_args[2] * pre_outflow
            )
            cur_stage = self._sd_curve.inverse(cur_inflow)
            self._stages[i] = cur_stage
            self._inflows[i] = cur_inflow
            self._outflows[i] = cur_outflow

        self._current += self._dt

        # Update outputs
        for i, output in enumerate(self._outputs):
            element, var = self._outputs_map[i]
            element_index = cross_sections.index(element)
            if var == "FLOW":
                value = self._outflows[element_index]
            else:
                value = self._stages[element_index]
            output.add_data(self._current.timestamp(), value)

        self.set_status(models.LinkableComponentStatus.UPDATED, f"{self._id} Updated")

        # Check if finished
        if self._current >= self._end:
            self.set_status(models.LinkableComponentStatus.DONE, f"{self._id} Done")

    def finish(self):
        self.set_status(
            models.LinkableComponentStatus.FINISHING,
            f"Finishing RiverModel: {self._id}",
        )

        self.set_status(models.LinkableComponentStatus.FINISHED, f"{self._id} Finished")


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
