# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Enumerations for the common use in YunmengEnvs.
"""

from enum import Enum, auto

# ---------------------------------------------------
# region Geometry type
# ---------------------------------------------------


class GeometryType(Enum):
    """
    Enum for geometry type.
    """

    NONE = "none"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    CUBE = "cube"


# ---------------------------------------------------
# region Running mode
# ---------------------------------------------------


class EnvRunMode(Enum):
    """
    Enum for running mode of the environment.
    """

    # Production: Optimized for inference; saves only results, no gradients.
    PRODUCTION = auto()

    # Evaluation: For validation; computes metrics/plots, no weight updates.
    EVAL = auto()

    # Training: For learning; updates weights, saves checkpoints.
    TRAIN = auto()

    # Development: For debugging; full I/O (mesh/logs), tracks gradients.
    DEVELOP = auto()
