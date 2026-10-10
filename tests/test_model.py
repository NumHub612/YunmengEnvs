# -*- encoding: utf-8 -*-

import argparse
import os

import numpy as np
from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.supports import IField
from yunmeng.interfaces.solver import IInitialCondition
from yunmeng.numerics.algos import ym_register

from yunmeng import solutions, solvers  # noqa: F401 — trigger registration
from yunmeng.workflow.builder import EstimatorBuilder, SchedulerBuilder
from yunmeng.workflow.parser import Orchestrator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "foos"
HERE = os.path.join(ROOT, "benchmarks", CASE)


@ym_register("init")
class GaussianIC(IInitialCondition):
    """Gaussian initial pulse on cell centers."""

    def __init__(
        self,
        ic_id: str,
        target_field: str,
        centers: ArrayLike,
        center: float = 0.5,
        width: float = 0.08,
        amplitude: float = 1.0,
    ):
        if width <= 0:
            raise ValueError("GaussianIC width must be positive")
        self._id = ic_id
        self._field = target_field
        self._centers = np.asarray(centers, dtype="float64")
        self._center = float(center)
        self._width = float(width)
        self._amplitude = float(amplitude)

    @classmethod
    def get_name(cls) -> str:
        return "GaussianIC"

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_field(self) -> str:
        return self._field

    def get(self, **kwargs) -> ArrayLike:
        distance = self._centers - self._center
        return self._amplitude * np.exp(-(distance**2) / (2.0 * self._width**2))

    def apply(self, field: IField):
        field.values = self.get()


def run_config(name: str) -> Orchestrator:
    return Orchestrator(argparse.Namespace(config=os.path.join(HERE, name)))


def test_simulation():
    print("=" * 60, "\n[1] simulation task: single hybrid model\n", "=" * 60, sep="")
    sched = SchedulerBuilder().build(run_config("sim_hyb.yml"))
    sched.initialize()
    print("execution order:", sched.execution_order)
    sched.run()
    m = sched.models[0]
    print("hyb.u head:", m.get_output("hyb.u").get_values()[:4].round(5))


def test_coupling():
    print(
        "=" * 60,
        "\n[2] coupling task: truth -> hyb left boundary (PULL + adapter)\n",
        "=" * 60,
        sep="",
    )
    sched = SchedulerBuilder().build(run_config("links_coupled.yml"))
    sched.initialize()
    print("execution order:", sched.execution_order)
    sched.run()
    for m in sched.models:
        print(f"  {m.id}: {m.get_last_error() or 'ok'}")


def test_estimation():
    print(
        "=" * 60,
        "\n[3] estimation task: train the neural correction operator\n",
        "=" * 60,
        sep="",
    )
    estimator, target, data, loss, names = EstimatorBuilder().build(
        run_config("estimation.yml")
    )
    pre = estimator.evaluate(target, data, n_steps=50)
    result = estimator.fit(target, data, loss)
    post = estimator.evaluate(target, data, n_steps=50)
    print(
        f"  rmse pre={pre['rmse']:.5f} -> post={post['rmse']:.5f} "
        f"({pre['rmse'] / post['rmse']:.1f}x better)"
    )
    print(" ", result.message)
