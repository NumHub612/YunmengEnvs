import unittest
import os
import sys

from core.solutions.commons import links
from core.solutions import ym_models
from core.solutions.commons import datasets, models, metas
from core.numerics.mesh import Coordinate
import numpy as np
import argparse


class TestHydroModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n---------- Testing < {cls.__name__} > \n")

    @classmethod
    def tearDownClass(cls):
        print("\n---------- Done \n")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_links(self):
        """Test links"""
        self._parser = argparse.ArgumentParser(description="YunmengEnvs")
        self._parser.add_argument(
            "--config",
            default=r".\benchmarks\hydro_links\links.yml",
            type=str,
            help="config yaml",
        )
        self._parser.add_argument(
            "--log-level",
            type=str,
            default="WARNING",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="log level",
        )
        self._parser.add_argument(
            "--gpus",
            nargs="+",
            type=int,
            help="GPU ids to use",
        )
        self._parser.add_argument(
            "--cpu",
            action="store_true",
            help="always use CPU",
        )

        self._configer = links.Orchestrator(self._parser)
        self._configer.activate()
        self._scheduler = links.Scheduler(ym_models)
        self._scheduler.setup(self._configer)
        self._scheduler.initialize()
        errors = self._scheduler.validate()
        self._scheduler.prepare()
        self._scheduler.run()
        self._scheduler.finish()

    def test_tasks(self):
        """Test tasks"""
        self._parser = argparse.ArgumentParser(description="YunmengEnvs")
        self._parser.add_argument(
            "--config",
            default=r".\benchmarks\burgers_grid2d\links.yml",
            type=str,
            help="config yaml",
        )
        self._parser.add_argument(
            "--log-level",
            type=str,
            default="WARNING",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="log level",
        )
        self._parser.add_argument(
            "--gpus",
            nargs="+",
            type=int,
            help="GPU ids to use",
        )
        self._parser.add_argument(
            "--cpu",
            action="store_true",
            help="always use CPU",
        )

        self._configer = links.Orchestrator(self._parser)
        self._configer.activate()
        self._scheduler = links.Scheduler(ym_models)
        self._scheduler.setup(self._configer)
        self._scheduler.initialize()
        errors = self._scheduler.validate()
        self._scheduler.prepare()
        self._scheduler.run()
        self._scheduler.finish()


if __name__ == "__main__":
    with open("./tests/reports/report.txt", "w", encoding="utf8") as reporter:
        suit = unittest.TestSuite()
        suit.addTest(TestHydroModels("test_links"))
        suit.addTest(TestHydroModels("test_tasks"))

        runner = unittest.TextTestRunner(stream=reporter, verbosity=2)
        runner.run(suit)
