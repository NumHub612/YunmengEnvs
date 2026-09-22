# -*- encoding: utf-8 -*-
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
from yunmeng.solvers.FooFvmSolver import FvmSolver, FvmSolverConfig
from yunmeng.solvers.FooHybSolver2 import HybLapSolver, HybLapSolverConfig
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
# online-learning knobs
WINDOW = 10  # steps per observation window
TAU_HIGH = 0.05  # fallback threshold (window RMSE)
TAU_LOW = 0.02  # resume threshold (shadow RMSE)
ONLINE_LR = 3e-4
ONLINE_EPOCHS = 10


def make_grid(n_cells: int):
    grid = StructuredGrid1D(0.0, LENGTH, n_cells)
    grid.add_region(Region("left", ElementType.CELL, np.array([0])))
    grid.add_region(Region("right", ElementType.CELL, np.array([n_cells - 1])))
    return grid


def attach_conditions(solver, grid, center=0.5, width=0.07):
    centers = grid.cell_centers()[0].reshape(-1)
    solver.add_ic(GaussianIC("ic-u", "u", centers=centers, center=center, width=width))
    solver.add_bc(DirichletBC("bc-left", "u", grid.get_region("left"), 0.0, tag="wall"))
    solver.add_bc(
        DirichletBC("bc-right", "u", grid.get_region("right"), 0.0, tag="wall")
    )
    solver.initialize()


def make_fvm(sid: str, nu: float, end_time: float):
    grid = make_grid(N_COARSE)
    solver = FvmSolver(
        sid,
        grid,
        FvmSolverConfig(dt=DT, end_time=end_time),
        get_backend("numpy"),
        nu=nu,
    )
    attach_conditions(solver, grid)
    return solver


def make_hyblap(sid: str, nu: float, end_time: float):
    grid = make_grid(N_COARSE)
    config = HybLapSolverConfig(dt=DT, end_time=end_time, device="cpu")
    solver = HybLapSolver(
        sid,
        grid,
        config,
        get_backend("torch", device="cpu"),
        nu=nu,
        hidden=32,
    )
    attach_conditions(solver, grid)
    return solver


def truth_trajectory(center=0.5, width=0.07):
    """Fine-grid explicit FDM truth, block-averaged to the coarse grid."""
    x = (np.arange(N_FINE) + 0.5) * LENGTH / N_FINE
    u = np.exp(-((x - center) ** 2) / (2 * width**2))
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


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_offline(solver, observations, n_train):
    trainer = GradientTrainer(lr=LR, epochs=EPOCHS, n_steps=n_train)
    before = trainer.evaluate(solver, observations, n_steps=n_train)
    result = trainer.fit(solver, observations, TrajectoryMSELoss("u"))
    losses = [item["loss"] for item in result.history if "loss" in item]
    return before, result, losses


def fine_tune_online(solver, prefix_truth: np.ndarray):
    """Expanding-horizon fine-tune on all observations so far.

    GradientTrainer always resets to the IC and unrolls from t=0, so the
    online loss covers the whole observed prefix (end-to-end)."""
    n = len(prefix_truth)
    obs = TrajectoryObservationSet(
        times=np.arange(1, n + 1) * DT,
        fields={"u": prefix_truth},
    )
    trainer = GradientTrainer(lr=ONLINE_LR, epochs=ONLINE_EPOCHS, n_steps=n)
    return trainer.fit(solver, obs, TrajectoryMSELoss("u"))


def online_phase(solver, truth: np.ndarray, n_train: int):
    """Windowed online loop with fallback / retrain / resume.

    Per window: rollout (from IC) -> window RMSE -> maybe fallback ->
    fine-tune the surrogate on the observed prefix -> shadow-validate
    -> maybe resume. Returns (online segment trajectory, events).
    """
    n_total = int(T_FORECAST / DT)
    trajectory = []
    events = []
    cursor = n_train
    while cursor < n_total:
        w = min(WINDOW, n_total - cursor)
        preds = rollout(solver, cursor + w)[cursor : cursor + w]
        obs = truth[cursor : cursor + w]
        err = rmse(preds, obs)
        trajectory.append(preds)

        was_fallback = solver.fallback_active
        if not was_fallback and err > TAU_HIGH:
            solver.fallback_to_physics()
            events.append((cursor * DT, f"fallback (rmse={err:.4f})"))
            was_fallback = True

        # fine-tune the surrogate on everything observed so far; the θ
        # channel stays bound to the surrogate even under fallback.
        solver.resume_surrogate()
        fine_tune_online(solver, truth[: cursor + w])

        if was_fallback:
            # shadow validation with the retrained surrogate
            shadow = rollout(solver, cursor + w)[cursor : cursor + w]
            if rmse(shadow, obs) < TAU_LOW:
                events.append(((cursor + w) * DT, "resume"))
            else:
                solver.fallback_to_physics()
                events.append(((cursor + w) * DT, "stay on physics"))
        cursor += w

    return np.concatenate(trajectory), events


def test_fvm_hyblap():
    print("=" * 64)
    print("YunmengEnvs surrogate-Laplacian + online learning validation")
    print("=" * 64)

    truth = truth_trajectory()
    n_train = int(T_TRAIN / DT)
    n_total = int(T_FORECAST / DT)
    observations = TrajectoryObservationSet(
        times=np.arange(1, n_train + 1) * DT,
        fields={"u": truth[:n_train]},
    )

    oracle = make_fvm("fvm-oracle", NU, T_FORECAST)
    oracle_traj = rollout(oracle, n_total)
    baseline = make_fvm("fvm-biased", NU * NU_BIAS, T_FORECAST)
    baseline_traj = rollout(baseline, n_total)
    print(f"[fvm oracle ] RMSE full={rmse(oracle_traj, truth):.5f}")
    print(f"[fvm biased ] RMSE full={rmse(baseline_traj, truth):.5f}")

    # --- offline pre-training of the surrogate ---
    sur = make_hyblap("hyblap", NU, T_FORECAST)
    before, result, losses = train_offline(sur, observations, n_train)
    print(f"[hyblap pre  ] RMSE train={before['rmse']:.5f}")
    print(f"[train        ] loss {losses[0]:.6e} -> {losses[-1]:.6e}")
    store = InMemoryArtifactStore()
    ref = store.register(
        {
            "names": np.asarray(sur.parameter_names()),
            "values": sur.get_parameters(),
        },
        TrainingMeta(
            estimator=GradientTrainer.get_name(),
            data_lineage="fine-grid FDM truth, u, 0..0.5s",
            metrics=result.metrics,
            created_at=now_iso(),
        ),
    )
    sur_traj = rollout(sur, n_total)

    # --- online phase with fallback / retrain / resume ---
    online_traj, events = online_phase(sur, truth, n_train)
    for where, what in events:
        print(f"[online t={where * DT:4.2f}s] {what}")

    metrics = {
        "fvm_oracle_rmse_full": rmse(oracle_traj, truth),
        "fvm_biased_rmse_full": rmse(baseline_traj, truth),
        "sur_rmse_train": rmse(sur_traj[:n_train], truth[:n_train]),
        "sur_rmse_forecast": rmse(sur_traj[n_train:], truth[n_train:]),
        "sur_rmse_full": rmse(sur_traj, truth),
        "online_rmse": rmse(online_traj, truth[n_train:]),
        "online_events": events,
        "artifact_ref": str(ref),
    }

    centers = make_grid(N_COARSE).cell_centers()[0].reshape(-1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].semilogy(losses)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("trajectory MSE")
    axes[0].set_title("Offline training loss")
    axes[0].grid(alpha=0.3)

    for ax, step, title in (
        (axes[1], n_train - 1, f"Training horizon t={T_TRAIN:.2f}s"),
        (axes[2], -1, f"Forecast horizon t={T_FORECAST:.2f}s"),
    ):
        ax.plot(centers, truth[step], "k-", lw=2, label="truth")
        ax.plot(centers, oracle_traj[step], "g:", lw=2, label="FVM oracle")
        ax.plot(centers, baseline_traj[step], "b--", label="FVM biased")
        ax.plot(centers, sur_traj[step], "r-.", label="Surrogate")
        if step >= n_train:
            ax.plot(centers, online_traj[step - n_train], "m-", lw=1.5, label="Online")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fvm_hyblap_validation.png", dpi=140)

    (OUT_DIR / "fvm_validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    passed = metrics["sur_rmse_full"] < metrics["fvm_biased_rmse_full"]
    print("=" * 64)
    print(
        f"VERDICT: {'PASS' if passed else 'FAIL'} — surrogate "
        f"{metrics['sur_rmse_full']:.5f} vs biased FVM "
        f"{metrics['fvm_biased_rmse_full']:.5f}; online "
        f"{metrics['online_rmse']:.5f}"
    )
    print("=" * 64)
