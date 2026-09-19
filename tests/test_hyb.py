# -*- encoding: utf-8 -*-
"""Train and forecast with pure-FDM and hybrid solvers.

Pipeline:
1. Fine-grid explicit FDM generates truth observations.
2. Coarse-grid FdmSolver provides oracle and biased physics baselines.
3. Coarse-grid HybridSolver learns a neural correction by unrolled AD.
4. The trained hybrid forecasts beyond the training horizon.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from yunmeng.interfaces.supports import Region
from yunmeng.interfaces.types import ElementType
from yunmeng.numerics.fields import get_backend
from yunmeng.numerics.grids import StructuredGrid1D

from yunmeng.solvers.commons.inits import GaussianIC
from yunmeng.solvers.commons.boundaries import DirichletBC
from yunmeng.solvers.FooFdmSolver import FdmSolver, FdmSolverConfig
from yunmeng.solvers.FooHybSolver import HybridSolver, HybridSolverConfig
from yunmeng.workflow.optimizer import (
    GradientTrainer,
    TrainingMeta,
    now_iso,
    InMemoryArtifactStore,
    TrajectoryMSELoss,
    TrajectoryObservationSet,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
NU = 0.02
NU_BIAS = 0.4
LENGTH = 1.0
DT_FINE = 2.0e-4
DT = 0.01
N_FINE = 256
N_COARSE = 32
T_TRAIN = 0.5
T_FORECAST = 1.0
EPOCHS = 200
LR = 3e-3


def make_grid(n_cells: int):
    grid = StructuredGrid1D(0.0, LENGTH, n_cells)
    grid.add_region(Region("left", ElementType.CELL, np.array([0])))
    grid.add_region(Region("right", ElementType.CELL, np.array([n_cells - 1])))
    return grid


def attach_conditions(solver, grid):
    centers = grid.cell_centers()[0].reshape(-1)
    solver.add_ic(GaussianIC("ic-u", "u", centers=centers, center=0.5, width=0.07))
    solver.add_bc(DirichletBC("bc-left", "u", grid.get_region("left"), 0.0, tag="wall"))
    solver.add_bc(
        DirichletBC("bc-right", "u", grid.get_region("right"), 0.0, tag="wall")
    )
    solver.initialize()


def make_fdm(sid: str, nu: float, end_time: float):
    grid = make_grid(N_COARSE)
    solver = FdmSolver(
        sid,
        grid,
        FdmSolverConfig(dt=DT, end_time=end_time),
        get_backend("numpy"),
        nu=nu,
    )
    attach_conditions(solver, grid)
    return solver


def make_hybrid(sid: str, nu: float, end_time: float):
    grid = make_grid(N_COARSE)
    config = HybridSolverConfig(dt=DT, end_time=end_time, device="cpu")
    solver = HybridSolver(
        sid,
        grid,
        config,
        get_backend("torch", device="cpu"),
        nu=nu,
        hidden=32,
    )
    attach_conditions(solver, grid)
    return solver


def truth_trajectory():
    """Fine-grid explicit FDM truth, block-averaged to the coarse grid."""
    x = (np.arange(N_FINE) + 0.5) * LENGTH / N_FINE
    u = np.exp(-((x - 0.5) ** 2) / (2 * 0.07**2))
    dx = LENGTH / N_FINE
    stability = NU * DT_FINE / dx**2
    if stability >= 0.5:
        raise RuntimeError(f"explicit truth scheme unstable: r={stability}")

    n_total = int(T_FORECAST / DT_FINE)
    space_ratio = N_FINE // N_COARSE
    time_stride = int(DT / DT_FINE)
    snapshots = []
    for step in range(1, n_total + 1):
        laplacian = np.zeros_like(u)
        laplacian[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
        u = u + DT_FINE * NU * laplacian
        u[0] = 0.0
        u[-1] = 0.0
        if step % time_stride == 0:
            snapshots.append(u.reshape(N_COARSE, space_ratio).mean(axis=1))
    return np.stack(snapshots)


def rollout(solver, n_steps: int):
    solver.reset()
    result = solver.run(n_steps)
    if hasattr(result, "detach"):
        return result.detach().cpu().numpy()
    return np.asarray(result)


def test_hybrid():
    print("=" * 64)
    print("YunmengEnvs AI-native path validation")
    print("=" * 64)

    truth = truth_trajectory()
    n_train = int(T_TRAIN / DT)
    n_total = int(T_FORECAST / DT)
    observations = TrajectoryObservationSet(
        times=np.arange(1, n_train + 1) * DT,
        fields={"u": truth[:n_train]},
    )
    print(
        f"[truth] fine={N_FINE}, coarse={N_COARSE}, "
        f"train={n_train}, total={n_total}"
    )

    oracle = make_fdm("fdm-oracle", NU, T_FORECAST)
    oracle_trajectory = rollout(oracle, n_total)
    oracle_rmse = float(np.sqrt(np.mean((oracle_trajectory - truth) ** 2)))

    baseline = make_fdm("fdm-biased", NU * NU_BIAS, T_FORECAST)
    baseline_trajectory = rollout(baseline, n_total)
    baseline_rmse_train = float(
        np.sqrt(np.mean((baseline_trajectory[:n_train] - truth[:n_train]) ** 2))
    )
    baseline_rmse_full = float(np.sqrt(np.mean((baseline_trajectory - truth) ** 2)))
    print(f"[fdm oracle ] RMSE full={oracle_rmse:.5f}")
    print(
        f"[fdm biased ] RMSE train={baseline_rmse_train:.5f} "
        f"full={baseline_rmse_full:.5f}"
    )

    hybrid = make_hybrid("hybrid", NU * NU_BIAS, T_FORECAST)
    trainer = GradientTrainer(lr=LR, epochs=EPOCHS, n_steps=n_train)
    before = trainer.evaluate(hybrid, observations, n_steps=n_train)
    print(f"[hybrid pre ] RMSE train={before['rmse']:.5f}")

    result = trainer.fit(hybrid, observations, TrajectoryMSELoss("u"))
    losses = [item["loss"] for item in result.history if "loss" in item]
    print(
        f"[train      ] loss {losses[0]:.6e} -> {losses[-1]:.6e} "
        f"({losses[0] / max(losses[-1], 1e-30):.1f}x)"
    )

    names = hybrid.parameter_names()
    values = hybrid.get_parameters(names)
    store = InMemoryArtifactStore()
    ref = store.register(
        {"names": np.asarray(names), "values": values},
        TrainingMeta(
            estimator=GradientTrainer.get_name(),
            data_lineage="fine-grid FDM truth, u, 0..0.5s",
            metrics=result.metrics,
            created_at=now_iso(),
        ),
        model_id="hybrid-solver/nn-correction",
    )

    hybrid_trajectory = rollout(hybrid, n_total)
    hybrid_rmse_train = float(
        np.sqrt(np.mean((hybrid_trajectory[:n_train] - truth[:n_train]) ** 2))
    )
    hybrid_rmse_forecast = float(
        np.sqrt(np.mean((hybrid_trajectory[n_train:] - truth[n_train:]) ** 2))
    )
    hybrid_rmse_full = float(np.sqrt(np.mean((hybrid_trajectory - truth) ** 2)))
    print(
        f"[hybrid post] RMSE train={hybrid_rmse_train:.5f} "
        f"forecast={hybrid_rmse_forecast:.5f} full={hybrid_rmse_full:.5f}"
    )

    centers = make_grid(N_COARSE).cell_centers()[0].reshape(-1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].semilogy(losses)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("trajectory MSE")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    for ax, step, title in (
        (axes[1], n_train - 1, f"Training horizon t={T_TRAIN:.2f}s"),
        (axes[2], -1, f"Forecast horizon t={T_FORECAST:.2f}s"),
    ):
        ax.plot(centers, truth[step], "k-", lw=2, label="truth")
        ax.plot(centers, oracle_trajectory[step], "g:", lw=2, label="FDM oracle")
        ax.plot(centers, baseline_trajectory[step], "b--", label="FDM biased")
        ax.plot(centers, hybrid_trajectory[step], "r-.", label="Hybrid")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figure_path = OUT_DIR / "hybrid_validation.png"
    fig.savefig(figure_path, dpi=140)

    baseline_rmse_forecast = float(
        np.sqrt(np.mean((baseline_trajectory[n_train:] - truth[n_train:]) ** 2))
    )
    metrics = {
        "oracle_rmse_full": oracle_rmse,
        "baseline_rmse_train": baseline_rmse_train,
        "baseline_rmse_full": baseline_rmse_full,
        "baseline_rmse_forecast": baseline_rmse_forecast,
        "hybrid_rmse_train": hybrid_rmse_train,
        "hybrid_rmse_forecast": hybrid_rmse_forecast,
        "hybrid_rmse_full": hybrid_rmse_full,
        "improvement_full_x": baseline_rmse_full / hybrid_rmse_full,
        "improvement_forecast_x": baseline_rmse_forecast / hybrid_rmse_forecast,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "artifact_ref": str(ref),
    }
    metrics_path = OUT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = hybrid_rmse_full < baseline_rmse_full * 0.5
    print("=" * 64)
    print(
        f"VERDICT: {'PASS' if passed else 'FAIL'} — hybrid "
        f"{hybrid_rmse_full:.5f}, biased FDM {baseline_rmse_full:.5f}, "
        f"{metrics['improvement_full_x']:.1f}x better"
    )
    print("=" * 64)
