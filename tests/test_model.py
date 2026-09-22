# -*- encoding: utf-8 -*-

import argparse
import os
from yunmeng import solutions, solvers  # noqa: F401 — trigger registration
from yunmeng.workflow.builder import EstimatorBuilder, SchedulerBuilder
from yunmeng.workflow.parser import Orchestrator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "foos"
HERE = os.path.join(ROOT, "benchmarks", CASE)


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
