# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solutions for the 2D diffusion equation using finite volume method.
"""
from core.solvers.commons import inits, boundaries, callbacks, BaseSolver
from core.numerics.algos import FieldInterpolations as fis
from core.numerics.mesh import Grid2D, MeshTopo, MeshGeom
from configs.settings import settings, logger


class Diffusion2D(BaseSolver):

    @classmethod
    def get_name(cls) -> str:
        return "diffusion2d"

    @classmethod
    def get_meta(cls) -> dict:
        metas = super().get_meta()
        metas.update(
            {
                "description": "Test solver of the 2D diffusion equation.",
                "type": "fvm",
                "equation": "Diffusion Equation",
                "equation_expr": "-div(k*grad(u)) = f",
                "dimension": "2D",
                "default_ics": "none",
                "default_bcs": "none",
            }
        )
        metas.update(
            {
                "fields": {
                    "u": {
                        "description": "scalar field",
                        "etype": "cell",
                        "dtype": "scalar",
                    },
                },
            }
        )
        return metas

    @property
    def status(self) -> dict:
        return {
            "iteration": 1,
            "elapsed_time": None,
            "current_time": None,
            "time_step": None,
            "convergence": True,
            "error": "",
        }

    def __init__(self, id: str, mesh: Grid2D):
        super().__init__(id, mesh)

    def initialize(self):
        logger.info("Initializing the 2D diffusion solver...")

    def inference(self) -> tuple[bool, bool, dict]:
        logger.info("Inference the 2D diffusion solver...")
