# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback
from core.visuals.plotter import plot_scalar_field, plot_vector_field

import os


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

    def setup(self, solver_meta: dict, mesh: object):
        self._mesh = mesh

        for field in solver_meta["fields"]:
            fname = field["name"]
            if self._fields is not None and fname not in self._fields:
                continue

            field_dir = os.path.join(self._output_dir, fname)
            os.makedirs(field_dir, exist_ok=True)
            self._field_dirs[fname] = field_dir

    def on_task_begin(self, solutions: dict, time: float = None):
        self._plot_field(solutions, time)

    def on_step(self, solutions: dict, time: float = None):
        self._plot_field(solutions, time)

    def _plot_field(self, solutions: dict, time: float = None):
        for fname, field in solutions.items():
            if fname not in self._field_dirs:
                continue

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

            if field.dtype == "scalar":
                plot_scalar_field(
                    field,
                    self._mesh,
                    **options,
                )
            elif field.dtype == "vector":
                plot_vector_field(
                    field,
                    self._mesh,
                    **options,
                )
        self._frame += 1

    def on_step_end(self, **kwargs):
        pass

    def on_task_end(self):
        pass

    def on_step_begin(self):
        pass
