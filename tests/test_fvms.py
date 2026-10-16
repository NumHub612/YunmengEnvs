# -*- encoding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from yunmeng.standard.types import ArrayLike
from yunmeng.standard.solver import ISolver

from yunmeng.standard.supports import Region
from yunmeng.standard.types import ElementType
from yunmeng.numerics.fields import get_backend
from yunmeng.numerics.grids import Grid1D

from yunmeng.solvers.commons.inits import PrescribedIC
from yunmeng.solvers.commons.boundaries import DirichletBC
from yunmeng.solvers.FvmDiffusionSv import FvmDiffusionSolver, FvmDiffusionSolverConfig

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


def gaussian_ic(x: ArrayLike, center=0.5, width=0.08, amplitude=1.0) -> ArrayLike:
    """Gaussian initial condition."""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * width**2))


def make_grid(n_cells: int):
    grid = Grid1D(0.0, LENGTH, n_cells)
    grid.add_region(Region("left", ElementType.CELL, np.array([0])))
    grid.add_region(Region("right", ElementType.CELL, np.array([n_cells - 1])))
    return grid


def attach_conditions(
    solver: ISolver, grid: Grid1D, center=0.5, width=0.07, amplitude=1.0
):
    centers = grid.cell_centers()[0].reshape(-1)
    inits = gaussian_ic(centers, center=center, width=width, amplitude=amplitude)
    solver.add_ic(PrescribedIC("ic-u", "u", inits))

    solver.add_bc(DirichletBC("bc-left", "u", grid.get_region("left"), 0.0))
    solver.add_bc(DirichletBC("bc-right", "u", grid.get_region("right"), 0.0))
    solver.initialize()


def make_fvm(sid: str, nu: float, end_time: float):
    grid = make_grid(N_COARSE)
    solver = FvmDiffusionSolver(
        sid,
        grid,
        FvmDiffusionSolverConfig(dt=DT, end_time=end_time, nu=nu),
        get_backend("numpy"),
    )
    attach_conditions(solver, grid)
    return solver


def truth_trajectory(center=0.5, width=0.07):
    """Fine-grid explicit FDM truth, block-averaged to the coarse grid."""
    # explicit FDM truth trajectory
    x = (np.arange(N_FINE) + 0.5) * LENGTH / N_FINE
    u = np.exp(-((x - center) ** 2) / (2 * width**2))
    dx = LENGTH / N_FINE
    stability = NU * DT_FINE / dx**2
    if stability >= 0.5:
        raise RuntimeError(f"explicit truth scheme unstable: r={stability}")

    # compute block-averaged coarse-grid snapshots
    n_total = int(T_FORECAST / DT_FINE)
    space_ratio = N_FINE // N_COARSE  # coarse grid block size
    time_stride = int(DT / DT_FINE)
    snapshots = []
    for step in range(1, n_total + 1):
        laplacian = np.zeros_like(u)
        laplacian[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
        u = u + DT_FINE * NU * laplacian
        u[0] = 0.0
        u[-1] = 0.0
        # save snapshots at coarse time steps
        if step % time_stride == 0:
            # Spatial Downsampling via Averaging
            snapshots.append(u.reshape(N_COARSE, space_ratio).mean(axis=1))
    return np.stack(snapshots)


def rollout(solver, n_steps: int):
    solver.reset()
    result = solver.run(n_steps)
    if hasattr(result, "detach"):
        return result.detach().cpu().numpy()
    return np.asarray(result)


def plot_profiles(truth, oracle, biased, out_path: Path):
    """代表性时刻的剖面对比。"""
    x = (np.arange(N_COARSE) + 0.5) * LENGTH / N_COARSE
    times = [0.1, 0.3, T_TRAIN, T_FORECAST]

    fig, axes = plt.subplots(1, len(times), figsize=(16, 4), sharey=True)
    for ax, tt in zip(axes, times):
        k = int(round(tt / DT)) - 1  # 快照索引：第 k 步对应 t=(k+1)*DT
        ax.plot(x, truth[k], "k-", lw=2, label="Truth")
        ax.plot(x, oracle[k], "b--", lw=1.5, label="Oracle")
        ax.plot(x, biased[k], "r:", lw=1.5, label="Biased")
        ax.set_title(f"t = {tt:.1f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Solution profiles: truth vs predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def test_fvm_diffusion_solver():
    # truth trajectory at coarse grid
    truth = truth_trajectory()
    n_train = int(T_TRAIN / DT)
    n_total = int(T_FORECAST / DT)
    print(
        f"[truth] fine={N_FINE}, coarse={N_COARSE}, "
        f"train={n_train}, total={n_total}"
    )

    # coarse-grid with right diffusion coefficient,
    oracle = make_fvm("fvm-oracle", NU, T_FORECAST)
    oracle_trajectory = rollout(oracle, n_total)
    oracle_rmse = float(np.sqrt(np.mean((oracle_trajectory - truth) ** 2)))

    # coarse-grid with wrong diffusion coefficient
    baseline = make_fvm("fvm-biased", NU * NU_BIAS, T_FORECAST)
    baseline_trajectory = rollout(baseline, n_total)
    baseline_rmse_train = float(
        np.sqrt(np.mean((baseline_trajectory[:n_train] - truth[:n_train]) ** 2))
    )
    baseline_rmse_full = float(np.sqrt(np.mean((baseline_trajectory - truth) ** 2)))
    print(f"[fvm oracle ] RMSE full={oracle_rmse:.5f}")
    print(
        f"[fvm biased ] RMSE train={baseline_rmse_train:.5f} "
        f"full={baseline_rmse_full:.5f}"
    )

    # 真值 vs 预报可视化
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_profiles(
        truth,
        oracle_trajectory,
        baseline_trajectory,
        OUT_DIR / "fvm_profiles.png",
    )
    print(f"[plot] saved to {OUT_DIR}")
