# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Callback for rendering the solver solutions.
"""
from core.solvers.interfaces import ISolverCallback, ISolver
from core.numerics.mesh.meshes import Mesh
from core.render.plotter.FieldPlotters import plot_field
from core.render.plotter.MeshPlotters import plot_mesh
from core.render.animator import ImageSetPlayer

import os
import shutil
from typing import Dict


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

    def __init__(
        self,
        id: str,
        output_dir: str,
        fields: Dict[str, Dict] = None,
        frequency: float = None,
    ):
        """Initialize the callback.

        Args:
            output_dir: The output directory for the rendered images.
            fields: Rendering field options, e.g.,
                {"h": {"cmap": "viridis", "vmin": 0.8, "vmax": 1.2}}.
            frequency: The frequency of rendering.
        """
        self._id = id
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._fields = fields
        self._solver = None
        self._mesh = None
        self._frame = 0
        self._frequency = frequency
        self._clock = 0.0

    def setup(self, solver: ISolver, mesh: Mesh, **kwargs):
        self._solver = solver
        self._mesh = mesh

        available_fields = self._solver.get_meta().fields
        if self._fields is None:
            self._fields = {f: {} for f in available_fields.keys()}

        for fname in self._fields.keys():
            dir = os.path.join(self._output_dir, fname)
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)
            self._fields[fname].update({"save_dir": dir})

    def cleanup(self):
        self._solver = None
        self._mesh = None
        self._frame = 0

    def on_task_begin(self, **kwargs):
        if not self._if_render():
            return

        plot_mesh(
            self._mesh, title=f"{self._solver.id}-mesh", save_dir=self._output_dir
        )
        self._plot_field()

    def _if_render(self):
        if self._frequency is None:
            return True
        else:
            return self._solver.status.current_time >= self._clock

    def _get_frame_name(self):
        status = self._solver.status
        if status.current_time is not None:
            return f"t{status.current_time:.4f}"
        else:
            return f"step{self._frame:04d}"

    def _plot_field(self):
        """Plot the field solutions."""
        for fname in self._fields.keys():
            field = self._solver.get_solution(fname)
            frame = self._get_frame_name()
            title = f"{fname}-{frame}"

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
            player = ImageSetPlayer(img_dir, pause=0.01)
            player.play(show=False, save=True)

    def on_step_begin(self, **kwargs):
        pass

    def on_step(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        if not self._if_render():
            return
        self._clock += self._frequency if self._frequency is not None else 0.0
        self._frame += 1
        self._plot_field()
