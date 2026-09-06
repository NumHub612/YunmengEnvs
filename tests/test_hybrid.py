# -*- encoding: utf-8 -*-
"""
Acceptance tests for the hybrid AdvDiff chain (aligned with design doc §11).

Run:  python -m pytest tests/test_all.py -q   (from the project root)
"""

from __future__ import annotations

import numpy as np
import torch

from yunmeng.taskflow.trainer import (
    GradientTrainer,
    TrajectoryDataset,
    rollout_mse,
)
from yunmeng.taskflow.losses import rollout_mse
from yunmeng.taskflow.artifacts import LocalArtifactStore
from yunmeng.taskflow.dataset import TrajectoryEpisode, TrajectoryDataset

from yunmeng.interfaces.solution import ModelRef, TrainingMeta
from yunmeng.interfaces.types import RunMode
from yunmeng.numerics.grids.grids2 import StructuredGrid2D
from yunmeng.solvers.hyb import AdvUpwind2D, Lap5Point2D, NeuralCorrectionOperator
from yunmeng.solutions.HybridModel import AdvDiffModel


def make_model(with_nn: bool = True, nx: int = 16, dt: float = 0.02):
    grid = StructuredGrid2D(nx=nx, ny=nx)
    ops = [AdvUpwind2D(vx=0.5, vy=0.3), Lap5Point2D(nu=0.01)]
    if with_nn:
        ops.append(NeuralCorrectionOperator(hidden=8))
    model = AdvDiffModel("test", grid, ops, {"time_step": dt, "end_time": 1.0})
    xx, yy = grid.coordinates
    ic = np.exp(-((xx - 0.6) ** 2 + (yy - 0.6) ** 2) / (2 * 0.15**2))
    model.set_initial_condition(ic)
    model.initialize()
    return model


def tiny_dataset(model) -> TrajectoryDataset:
    """Reference = pure-physics rollout of a NN-free twin."""
    twin = make_model(with_nn=False)
    traj = twin.run(5, dt=0.02)
    ref = traj.detach().cpu().numpy()
    ic = twin.solver.get_solution("u").values() * 0.0  # placeholder, reset below
    xx, yy = twin.grid.coordinates
    ic = np.exp(-((xx - 0.6) ** 2 + (yy - 0.6) ** 2) / (2 * 0.15**2))
    return TrajectoryDataset([TrajectoryEpisode(ic=ic, reference=ref, dt=0.02)])


# 1. graph connectivity + gradient correctness (design §11.6 / v1.3 §18)
def test_gradient_vs_finite_difference():
    model = make_model()
    model.train()
    model.reset_run()
    traj = model.run(5, dt=0.02)
    loss = (traj**2).mean()
    loss.backward()

    # NN grads flow
    nn_op = [
        op for op in model.solver.operators if isinstance(op, NeuralCorrectionOperator)
    ][0]
    grads = [p.grad for p in nn_op.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)

    # AD vs FD on phys.nu (needs nu to be a grad-carrying leaf: wrap it)
    model.eval()
    model.set_parameters(
        {"phys.nu": torch.tensor(0.01, dtype=torch.float64, requires_grad=True)}
    )
    model.train()
    model.reset_run()
    loss = (model.run(5, dt=0.02) ** 2).mean()
    nu_tensor = model.get_parameters()["phys.nu"]
    (ad_grad,) = torch.autograd.grad(loss, nu_tensor)

    eps = 1e-4
    with torch.no_grad():
        vals = []
        for d in (eps, -eps):
            model.set_parameters(
                {"phys.nu": torch.tensor(0.01 + d, dtype=torch.float64)}
            )
            model.reset_run()
            vals.append(float((model.run(5, dt=0.02) ** 2).mean()))
    fd_grad = (vals[0] - vals[1]) / (2 * eps)
    rel = abs(float(ad_grad) - fd_grad) / max(abs(fd_grad), 1e-12)
    assert rel < 0.01, f"AD/FD mismatch: ad={float(ad_grad)}, fd={fd_grad}, rel={rel}"


# 2. mode propagation (design §11.5)
def test_mode_propagation():
    model = make_model()
    assert model.mode is RunMode.EVAL
    model.train()
    assert model.solver.mode is RunMode.TRAIN
    assert model.solver.datahub.mode is RunMode.TRAIN
    nn_op = [
        op for op in model.solver.operators if isinstance(op, NeuralCorrectionOperator)
    ][0]
    assert nn_op.training is True
    model.eval()
    assert nn_op.training is False


# 3. runtime purity: same operator instance serves two runs (design §11.3)
def test_operator_reuse_across_runs():
    model = make_model()
    r1 = model.run(3, dt=0.02).detach().cpu().numpy()
    model.reset_run()
    r2 = model.run(3, dt=0.02).detach().cpu().numpy()
    np.testing.assert_allclose(r1, r2, rtol=0, atol=0)


# 4. artifact store roundtrip (design §11.4 partial)
def test_artifact_roundtrip(tmp_path):
    store = LocalArtifactStore(tmp_path)
    op = NeuralCorrectionOperator(hidden=8)
    params = {k: v + 0.1 for k, v in op.get_parameters().items()}
    ref = store.register(params, TrainingMeta(estimator="t"), model_id="correction")
    assert isinstance(ref, ModelRef)
    loaded = store.resolve(ref)
    op2 = NeuralCorrectionOperator(hidden=8)
    op2.set_parameters(loaded)
    for k, v in op2.get_parameters().items():
        assert torch.allclose(v, params[k].to(torch.float64))
    assert ref.version in store.list_versions("correction")


# 5. training reduces loss + zero-init degrades to pure physics
def test_training_reduces_loss():
    data = tiny_dataset(None)
    model = make_model()

    # zero-init: hybrid == pure physics at start
    model.eval()
    start = model.run(5, dt=0.02).detach().cpu().numpy()
    np.testing.assert_allclose(start, data.episodes[0].reference, atol=1e-12)

    # give the student a wrong diffusion coefficient so the correction has
    # something to learn (reference was generated with nu=0.01)
    model.set_parameters({"phys.nu": torch.tensor(0.005, dtype=torch.float64)})

    trainer = GradientTrainer(unroll_steps=5, epochs=30, lr=5e-3)
    result = trainer.fit(model, data, rollout_mse)
    assert result.history[-1]["loss"] < 0.2 * result.history[0]["loss"]
    assert model.mode is RunMode.EVAL  # restored after fit
