# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback, ISolver
from core.numerics.mesh import Mesh
from core.visuals.plotter import plot_field, plot_mesh
from core.visuals.animator import ImageSetPlayer

import os
import shutil


class ImageRender(ISolverCallback):
    """
    ImageRender  rendering the solver solutions to images while solving.
    """

    @classmethod
    def get_name(cls):
        return "render"

    @property
    def id(self) -> str:
        return self._id

    def __init__(self, id: str, output_dir: str, fields: dict = None):
        """Initialize the callback.

        Args:
            output_dir: The output directory for the rendered images.
            fields: The expected fields to be rendered.

        Notes:
            - `fields` is a dictionary with the field names as keys,
               and the rendering options as values.
            - `fields` can be None, in which render all fields.
        """
        self._id = id
        self._output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._fields = fields
        self._solver = None
        self._mesh = None
        self._frame = 0

    def setup(self, solver: ISolver, mesh: Mesh, **kwargs):
        self._solver = solver
        self._mesh = mesh

        _, available_fields = self._check_solver()
        if self._fields is None or not self._fields:
            self._fields = {f: {} for f in available_fields.keys()}

        for fname in self._fields.keys():
            dir = os.path.join(self._output_dir, fname)
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)
            self._fields[fname].update(
                {
                    "save_dir": dir,
                }
            )

    def cleanup(self):
        self._solver = None
        self._mesh = None
        self._frame = 0

    def _check_solver(self):
        """Check the solver's status and solutions."""
        if self._solver is None or not isinstance(self._solver, ISolver):
            raise RuntimeError("Solver is invalid.")

        status = self._solver.status
        meta = self._solver.get_meta()
        if meta.fields is None:
            raise ValueError(f"Solver {self._solver.id} has no fields.")

        solutions = {}
        for fname in meta.fields.keys():
            field = self._solver.get_solution(fname)
            if field is not None:
                solutions[fname] = field

        return status, solutions

    def on_task_begin(self, **kwargs):
        title = f"{self._solver.id}-mesh"
        show_edges = False
        for fname, field in self._fields.items():
            if "show_edges" in field:
                show_edges = field["show_edges"]
                break
        plot_mesh(
            self._mesh,
            title=title,
            save_dir=self._output_dir,
            show_edges=show_edges,
        )

        status, solutions = self._check_solver()
        self._plot_field(status, solutions)

    def on_step(self, **kwargs):
        self._frame += 1
        status, solutions = self._check_solver()
        self._plot_field(status, solutions)

    def _plot_field(self, solver_status, solver_solutions):
        """Plot the field solutions."""
        for fname, field in solver_solutions.items():
            if fname not in self._fields or field is None:
                continue

            title = f"{fname}-{self._frame}"

            options = self._fields[fname]
            options.update(
                {
                    "title": title,
                    "label": fname,
                    "save_dir": self._fields[fname]["save_dir"],
                    "show": False,
                }
            )

            plot_field(field, self._mesh, **options)

    def on_task_end(self, **kwargs):
        for fname, field in self._fields.items():
            img_dir = field["save_dir"]
            if not os.path.exists(img_dir):
                continue
            player = ImageSetPlayer(img_dir, pause=0.01)
            player.play(show=False, save=True)

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass
