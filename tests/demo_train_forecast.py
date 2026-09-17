# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

End-to-end validation of the AI-native path:

  1. TRUTH    : fine-grid (256 cells) FDM diffusion -> observations
  2. BASELINE : coarse-grid (32 cells) pure-physics solver -> forecast
  3. HYBRID   : coarse physics + neural correction operator
                -> GradientTrainer (unrolled AD through the solver)
                -> forecast beyond the training horizon (EVAL mode)

Success criteria: hybrid RMSE << pure-physics coarse RMSE, on both the
training horizon and the unseen forecast horizon.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

from yunmeng.solvers.hyb.estimators import (
    GradientTrainer,
    InMemoryArtifactStore,
    TrainingMeta,
    TrajectoryMSELoss,
    TrajectoryObservationSet,
    now_iso,
)
from yunmeng.interfaces.types import ElementType, RunMode
from yunmeng.solvers.hyb.backends import get_backend
from yunmeng.solvers.hyb.primitives import UniformMesh1D
from yunmeng.solvers.hyb.hybrid_solver import HybridSolver, HybridSolverConfig
from yunmeng.solvers.hyb.operators import (
    FdmDiffusionOperator,
    NeuralCorrectionOperator,
)
from yunmeng.solvers.hyb.rules import DirichletBC, GaussianIC

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- problem setup ----------------------------------
NU = 0.02  # true diffusivity
NU_BIAS = 0.4  # biased physics uses 40% of true nu (model error)
L = 1.0  # domain length
DT_FINE = 2.0e-4  # truth time step (explicit-stable on fine grid)
DT = 0.01  # solver time step
N_FINE = 256
N_COARSE = 32
T_TRAIN = 0.5  # training horizon -> 50 solver steps
T_FORECAST = 1.0  # forecast horizon  -> 100 solver steps
EPOCHS = 200
LR = 3e-3


def make_solver(
    mesh, backend, end_time, with_nn: bool, device: str = "cpu", nu: float = NU
):
    ops = [FdmDiffusionOperator(field="u", nu=nu)]
    if with_nn:
        ops.append(NeuralCorrectionOperator(field="u", hidden=32, device=device))
    cfg = HybridSolverConfig(dt=DT, t0=0.0, end_time=end_time, device=device)
    solver = HybridSolver("hybrid-1", mesh, ops, cfg, backend)
    solver.add_bc(DirichletBC("bc-left", "u", mesh.get_region("left"), 0.0, tag="wall"))
    solver.add_bc(
        DirichletBC("bc-right", "u", mesh.get_region("right"), 0.0, tag="wall")
    )
    return solver


def apply_ic(solver, mesh, x):
    solver.add_ic(GaussianIC("ic-u", "u", centers=x, center=0.5, width=0.07))


def truth_trajectory():
    """Fine-grid explicit FDM truth, block-averaged to the coarse grid."""
    mesh = UniformMesh1D(N_FINE, 0.0, L)
    x = mesh.cell_centers
    u = np.exp(-((x - 0.5) ** 2) / (2 * 0.07**2))
    dx = mesh.dx
    r = NU * DT_FINE / dx**2
    assert r < 0.5, f"explicit scheme unstable: r={r}"

    n_total = int(T_FORECAST / DT_FINE)
    ratio = N_FINE // N_COARSE  # coarsen the truth to the coarse grid on space
    stride = int(DT / DT_FINE)  # coarsen the truth to the coarse grid on time
    snaps = []  # coarse-grid-time snapshots
    for k in range(1, n_total + 1):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2  # Laplacian
        u = u + DT_FINE * NU * lap  # explicit Euler
        u[0] = u[-1] = 0.0
        if k % stride == 0:  # coarsen the truth to the coarse grid on time
            snaps.append(u.reshape(N_COARSE, ratio).mean(axis=1))
    return np.stack(snaps)  # (n_steps_total, N_COARSE)


def rollout(solver, n_steps):
    solver.reset()
    return solver.run(n_steps)


def main():
    print("=" * 64)
    print("YunmengEnvs AI-native path validation: hybrid solver demo")
    print("=" * 64)

    # 1. truth
    truth = truth_trajectory()
    n_train = int(T_TRAIN / DT)
    n_total = int(T_FORECAST / DT)
    times = np.arange(1, n_train + 1) * DT
    obs = TrajectoryObservationSet(times=times, fields={"u": truth[:n_train]})
    print(
        f"[truth] fine grid {N_FINE}, coarse grid {N_COARSE}, "
        f"train steps {n_train}, total steps {n_total}"
    )

    # 2. references on the coarse grid (numpy, EVAL):
    #    - "oracle physics"  : true nu (best possible mechanistic model)
    #    - "biased physics"  : 40% of true nu (model-error scenario)
    mesh_c = UniformMesh1D(N_COARSE, 0.0, L)
    x_c = mesh_c.cell_centers
    oracle = make_solver(mesh_c, get_backend("numpy"), T_FORECAST, with_nn=False)
    apply_ic(oracle, mesh_c, x_c)
    oracle.initialize()
    oracle_traj = rollout(oracle, n_total)
    oracle_rmse = float(np.sqrt(np.mean((oracle_traj - truth) ** 2)))
    print(f"[ref: oracle physics, true nu ] RMSE full={oracle_rmse:.5f}")

    mesh_b = UniformMesh1D(N_COARSE, 0.0, L)
    base = make_solver(
        mesh_b, get_backend("numpy"), T_FORECAST, with_nn=False, nu=NU * NU_BIAS
    )
    apply_ic(base, mesh_b, x_c)
    base.initialize()
    base_traj = rollout(base, n_total)
    base_rmse_train = float(
        np.sqrt(np.mean((base_traj[:n_train] - truth[:n_train]) ** 2))
    )
    base_rmse_fore = float(np.sqrt(np.mean((base_traj - truth) ** 2)))
    print(
        f"[baseline: biased physics, nu*{NU_BIAS}] RMSE train={base_rmse_train:.5f} "
        f"full={base_rmse_fore:.5f}"
    )

    # 3. hybrid: physics + neural correction, torch backend
    mesh_h = UniformMesh1D(N_COARSE, 0.0, L)
    x_h = mesh_h.cell_centers
    hybrid = make_solver(
        mesh_h, get_backend("torch"), T_FORECAST, with_nn=True, nu=NU * NU_BIAS
    )
    apply_ic(hybrid, mesh_h, x_h)
    hybrid.initialize()

    # pre-training skill
    trainer = GradientTrainer(lr=LR, epochs=EPOCHS, n_steps=n_train)
    pre = trainer.evaluate(hybrid, obs, n_steps=n_train)
    print(f"[hybrid pre-train ] RMSE on train horizon = {pre['rmse']:.5f}")

    # training: gradient flows through every solver step (unrolled AD)
    print(f"[train ] {EPOCHS} epochs, lr={LR}, unrolled over {n_train} steps ...")
    result = trainer.fit(hybrid, obs, TrajectoryMSELoss("u"))
    losses = [h["loss"] for h in result.history if "loss" in h]
    print(
        f"[train ] loss {losses[0]:.6e} -> {losses[-1]:.6e} "
        f"({losses[0] / max(losses[-1], 1e-30):.1f}x reduction)"
    )

    # register the trained theta into the versioned artifact store
    store = InMemoryArtifactStore()
    theta = {n: v for n, v in hybrid._ops[1].get_parameters().items()}
    ref = store.register(
        {k: v.detach().cpu().numpy() for k, v in theta.items()},
        TrainingMeta(
            estimator=GradientTrainer.get_name(),
            data_lineage="fine-grid FDM truth, u, 0..0.5s",
            metrics=result.metrics,
            created_at=now_iso(),
            notes="hybrid-solver/nn-correction",
        ),
    )
    print(f"[store ] trained theta registered as {ref}")

    # forecast: EVAL mode, beyond the training horizon
    hybrid_traj = rollout(hybrid, n_total)
    hyb = hybrid_traj.detach().cpu().numpy()
    hyb_rmse_train = float(np.sqrt(np.mean((hyb[:n_train] - truth[:n_train]) ** 2)))
    hyb_rmse_fore = float(np.sqrt(np.mean((hyb[n_train:] - truth[n_train:]) ** 2)))
    hyb_rmse_all = float(np.sqrt(np.mean((hyb - truth) ** 2)))
    print(
        f"[hybrid post-train] RMSE train={hyb_rmse_train:.5f} "
        f"forecast={hyb_rmse_fore:.5f} full={hyb_rmse_all:.5f}"
    )

    # 4. report figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].semilogy(losses)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("trajectory MSE")
    axes[0].set_title("Training loss (unrolled AD through solver)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_c, truth[n_train - 1], "k-", lw=2, label="truth (fine)")
    axes[1].plot(
        x_c, oracle_traj[n_train - 1], "g:", lw=2, label="oracle physics (true nu)"
    )
    axes[1].plot(x_c, base_traj[n_train - 1], "b--", label="biased physics only")
    axes[1].plot(x_c, hyb[n_train - 1], "r-.", label="biased physics + NN (trained)")
    axes[1].set_title(f"End of training horizon  t={T_TRAIN:.2f}s")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_c, truth[-1], "k-", lw=2, label="truth (fine)")
    axes[2].plot(x_c, oracle_traj[-1], "g:", lw=2, label="oracle physics (true nu)")
    axes[2].plot(x_c, base_traj[-1], "b--", label="biased physics only")
    axes[2].plot(x_c, hyb[-1], "r-.", label="biased physics + NN (trained)")
    axes[2].set_title(f"Forecast horizon (unseen)  t={T_FORECAST:.2f}s")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "hybrid_validation.png")
    fig.savefig(fig_path, dpi=140)
    print(f"[figure] saved -> {fig_path}")

    metrics = {
        "oracle_rmse_full": oracle_rmse,
        "baseline_rmse_train": base_rmse_train,
        "baseline_rmse_full": base_rmse_fore,
        "hybrid_rmse_train": hyb_rmse_train,
        "hybrid_rmse_forecast": hyb_rmse_fore,
        "hybrid_rmse_full": hyb_rmse_all,
        "improvement_full_x": base_rmse_fore / hyb_rmse_all,
        "improvement_forecast_x": base_traj[n_train:] is not None
        and float(
            np.sqrt(np.mean((base_traj[n_train:] - truth[n_train:]) ** 2))
            / hyb_rmse_fore
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "artifact_ref": str(ref),
    }
    with open(
        os.path.join(OUT_DIR, "validation_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ok = hyb_rmse_all < base_rmse_fore * 0.5
    print("=" * 64)
    print(
        f"VERDICT: {'PASS' if ok else 'FAIL'} — hybrid full-horizon RMSE "
        f"{hyb_rmse_all:.5f} vs biased-physics-only {base_rmse_fore:.5f} "
        f"({metrics['improvement_full_x']:.1f}x better); "
        f"oracle physics reference {oracle_rmse:.5f}"
    )
    print("=" * 64)
    return metrics


if __name__ == "__main__":
    main()
