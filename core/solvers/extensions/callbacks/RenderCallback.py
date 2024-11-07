# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback
from core.visuals.animator import ImageStreamPlayer


class RenderCallback(ISolverCallback):
    """
    Callback for rendering the solver scalar field solutions.
    """

    def __init__(self):
        self._animators = {}

    def setup(self, solver_meta: dict):
        """
        Sets the solver metadata.
        """
        fields = solver_meta.get("fields", [])
        for field in fields:
            if field["dtype"] == "scalar" and field["etype"] == "node":
                self._animators[field["name"]] = ImageStreamPlayer(field["name"])

    def on_task_begin(self, solutions: dict):
        """
        Draw the initial scalar field solution.
        """
        for field_name, field in solutions.items():
            if field_name in self._animators:
                self._animators[field_name].update(field)

    def on_task_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step(self, solutions: dict):
        """
        Draw the scalar field solution at each step.
        """
        for field_name, field in solutions.items():
            if field_name in self._animators:
                self._animators[field_name].update(field)

    def on_step_end(self, **kwargs):
        """
        Callback function called at the end of each step.
        """
        pass
