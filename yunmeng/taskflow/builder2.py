# -*- encoding: utf-8 -*-
"""
Config-driven assembly: yaml -> model (configs -> model -> solver -> operators).

Short-name registries (v1.4 §24): yaml never references class paths;
duplicate registration raises.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from yunmeng.numerics.grids.grids2 import StructuredGrid2D
from yunmeng.solvers.hyb import (
    AdvUpwind2D,
    Lap5Point2D,
    NeuralCorrectionOperator,
)
from yunmeng.solutions.HybridModel import AdvDiffModel

ym_solvers = {}
ym_operators = {}


def _register_unique(registry: dict, name: str, obj) -> None:
    if "." in name:
        raise ValueError(f"Registry names must be short names, got '{name}'.")
    if name in registry:
        raise ValueError(f"Duplicate registration: '{name}'.")
    registry[name] = obj


_register_unique(ym_operators, "adv_upwind_2d", AdvUpwind2D)
_register_unique(ym_operators, "lap5_2d", Lap5Point2D)
_register_unique(ym_operators, "nn_correction", NeuralCorrectionOperator)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg: dict) -> AdvDiffModel:
    """Instantiate the full chain from a parsed yaml dict."""
    sp = cfg["SPATIAL"]
    p = sp["params"]
    grid = StructuredGrid2D(
        lower_left=tuple(p.get("lower_left", [0, 0])),
        upper_right=tuple(p.get("upper_right", [2, 2])),
        nx=p.get("nx", 32),
        ny=p.get("ny", 32),
    )

    solver_params = cfg.get("SOLVER", {}).get("params", {})
    operators = []
    for oc in cfg.get("OPERATORS", []):
        otype = oc["type"]
        if "." in otype:
            raise ValueError(
                f"Operator type must be a registered short name, got '{otype}'."
            )
        if otype not in ym_operators:
            raise KeyError(
                f"Unknown operator '{otype}'. Registered: {list(ym_operators)}"
            )
        op_params = dict(oc.get("params", {}))
        op_params.setdefault("name", oc.get("id", otype))
        # physical defaults may come from SOLVER.params
        if otype == "adv_upwind_2d":
            op_params.setdefault("vx", solver_params.get("vx", 0.0))
            op_params.setdefault("vy", solver_params.get("vy", 0.0))
        if otype == "lap5_2d":
            op_params.setdefault("nu", solver_params.get("nu", 0.0))
        operators.append(ym_operators[otype](**op_params))

    model_cfg = cfg.get("MODEL", {})
    model = AdvDiffModel(
        id=model_cfg.get("id", "advdiff"),
        grid=grid,
        operators=operators,
        solver_config=cfg.get("TEMPORAL", {}),
        backend=model_cfg.get("backend", "torch"),
    )
    return model
