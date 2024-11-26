# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback
from core.visuals.plotter import plot_scalar_field

import os


class PlayerCallback(ISolverCallback):
    """
    Callback for real-time playing the solver solutions.
    """

    pass


class RenderCallback(ISolverCallback):
    """
    Callback for real-time rendering the solutions.
    """

    def __init__(self, output_dir: str, figsize: tuple = (10, 5)):
        self._output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._mesh = None
        self._figsize = figsize
        self._field_dirs = {}

        self._frame_count = 0

    def setup(self, solver_meta: dict, mesh: object):
        self._mesh = mesh

        for field in solver_meta["fields"]:
            field_name = field["name"]
            field_dir = os.path.join(self._output_dir, field_name)
            os.makedirs(field_dir, exist_ok=True)

            self._field_dirs[field_name] = field_dir

    def on_task_begin(self, solutions: dict):
        """
        Draw the initial scalar field solution.
        """
        for field_name, field in solutions.items():
            if field_name in self._field_dirs:
                title = f"{field_name}-{self._frame_count}"
                plot_scalar_field(
                    field,
                    self._mesh,
                    figsize=self._figsize,
                    title=title,
                    save_dir=self._field_dirs[field_name],
                    show=False,
                )
        self._frame_count += 1

    def on_task_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step(self, solutions: dict):
        """
        Draw the scalar field solution at each step.
        """
        for field_name, field in solutions.items():
            if field_name in self._field_dirs:
                title = f"{field_name}-{self._frame_count}"
                plot_scalar_field(
                    field,
                    self._mesh,
                    figsize=self._figsize,
                    title=title,
                    save_dir=self._field_dirs[field_name],
                    show=False,
                )
        self._frame_count += 1

    def on_step_end(self, **kwargs):
        """
        Callback function called at the end of each step.
        """
        pass
