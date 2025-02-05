# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback
from core.visuals.plotter import plot_field

import os
import shutil


class RenderCallback(ISolverCallback):
    """
    Callback for real-time rendering the solutions.
    """

    @classmethod
    def get_name(cls):
        return "render"

    def __init__(self, output_dir: str, fields: dict = None):
        """Initialize the callback.

        Args:
            output_dir: The output directory for the rendered images.
            fields: The expected fields to be rendered.

        Notes:
            - `fields` is a dictionary with the field names as keys
               and the rendering options as values.
            - `fields` can be None, in which render all fields.
        """
        self._output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._fields = fields
        self._mesh = None
        self._field_dirs = {}
        self._frame = 0

    def setup(self, solver_meta: dict, mesh: object, **kwargs):
        self._mesh = mesh

        if solver_meta.get("fields"):
            self._init_output_dirs(solver_meta["fields"])

    def _init_output_dirs(self, available_fields: dict):
        # prevent multiple initializations of output directories
        if self._field_dirs:
            return

        # check outputed fields
        if self._fields is None or not self._fields:
            self._fields = {fname: {} for fname in available_fields.keys()}

        # create output directories for each field
        for fname in available_fields.keys():
            if fname not in self._fields:
                continue

            dir = os.path.join(self._output_dir, fname)
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)
            self._field_dirs[fname] = dir

    def on_task_begin(self, solver_status: dict, solver_solutions: dict, **kwargs):
        self._init_output_dirs(solver_solutions)

        # TODO: ploting the mesh only.
        self._plot_field(solver_status, solver_solutions)

    def on_step(self, solver_status: dict, solver_solutions: dict, **kwargs):
        self._plot_field(solver_status, solver_solutions)

    def _plot_field(self, solver_status: dict, solver_solutions: dict):
        solutions = {}
        for fname in self._field_dirs.keys():
            solutions[fname] = solver_solutions.get(fname)

        time = solver_status.get("current_time")
        for fname, field in solutions.items():
            code = self._frame if time is None else round(time, 6)
            title = f"{fname}-{code}"

            options = {}
            if self._fields is not None and fname in self._fields:
                options = self._fields[fname]
            options.update(
                {
                    "title": title,
                    "label": fname,
                    "save_dir": self._field_dirs[fname],
                    "show": False,
                }
            )

            plot_field(field, self._mesh, **options)
        self._frame += 1

    def on_task_end(self, solver_status: dict, solver_solutions: dict, **kwargs):
        # TODO: play the animation
        pass

    def on_step_begin(self, solver_status: dict, solver_solutions: dict, **kwargs):
        pass

    def on_step_end(self, solver_status: dict, solver_solutions: dict, **kwargs):
        pass
