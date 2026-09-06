# -*- encoding: utf-8 -*-
"""Solver layer: operators and solvers."""

from .operators import AdvUpwind2D, Lap5Point2D
from .neural import NeuralCorrectionOperator
from .advdiff import AdvectionDiffusionSolver, AdvDiffConfig

__all__ = [
    "AdvUpwind2D",
    "Lap5Point2D",
    "NeuralCorrectionOperator",
    "AdvectionDiffusionSolver",
    "AdvDiffConfig",
]
