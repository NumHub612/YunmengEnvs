# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Common enums for solvers.
"""
from enum import Enum


class SolverType(Enum):
    """Solver type."""

    FDM = "fdm"
    FVM = "fvm"
    FEM = "fem"
    LBM = "lbm"
    AIM = "aim"
