# -*- encoding: utf-8 -*-

import argparse
import os
from yunmeng import solutions, solvers  # noqa: F401 — trigger registration
from yunmeng.workflow.builder import EstimatorBuilder, SchedulerBuilder
from yunmeng.workflow.parser import Orchestrator

HERE = os.path.dirname(os.path.abspath(__file__))


def run_config(name: str) -> Orchestrator:
    return Orchestrator(argparse.Namespace(config=os.path.join(HERE, name)))
