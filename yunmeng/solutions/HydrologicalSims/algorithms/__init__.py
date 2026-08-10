# -*- encoding: utf-8 -*-
"""Pluggable hydrological algorithms."""

from yunmeng.solutions.HydrologicalSims.algorithms.Bases import (
    HydroAlgorithm,
    RunoffGeneration,
    SurfaceRouting,
    RiverRouting,
    ReleasePolicy,
    register,
    create,
    available,
)
from yunmeng.solutions.HydrologicalSims.algorithms.XiananjiangRunoff import (
    XinanjiangRunoff,
)
from yunmeng.solutions.HydrologicalSims.algorithms.MuskingumRouting import (
    MuskingumRouting,
)
from yunmeng.solutions.HydrologicalSims.algorithms.LinearReservoirRouting import (
    ThreeSourceLinearReservoir,
)
from yunmeng.solutions.HydrologicalSims.algorithms.TargetLevelReleasing import (
    TargetLevelRelease,
)

register("runoff", "xaj", XinanjiangRunoff)
register("surface", "linear3", ThreeSourceLinearReservoir)
register("river", "muskingum", MuskingumRouting)
register("reservoir", "target_level", TargetLevelRelease)

__all__ = [
    "HydroAlgorithm",
    "RunoffGeneration",
    "SurfaceRouting",
    "RiverRouting",
    "ReleasePolicy",
    "register",
    "create",
    "available",
    "XinanjiangRunoff",
    "ThreeSourceLinearReservoir",
    "MuskingumRouting",
    "TargetLevelRelease",
]
