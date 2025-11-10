# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

YunmengEnvs entrence.
"""
from configs.orchestrator import Orchestrator
from configs.settings import LOGO, logger

from core.solutions.commons import Scheduler
from core.solutions import ym_models
from core.solvers import ym_solvers, ym_operators
from core.solvers.commons import boundary_conditions, init_methods, callback_handlers

import argparse
import os


class YunmengEnvsApp:
    """YunmengEnvs application."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="YunmengEnvs")
        self.parser.add_argument("config", type=str, help="config yaml")
        self.parser.add_argument(
            "--log-level",
            type=str,
            default="WARNING",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="log level",
        )
        self.parser.add_argument(
            "--gpus",
            nargs="+",
            type=int,
            help="GPU ids to use",
        )
        self.parser.add_argument(
            "--cpu",
            action="store_true",
            default=False,
            help="always use CPU",
        )

        self.args = self.parser.parse_args()
        self.orchestrator = Orchestrator(self.args)
        self.scheduler = Scheduler()
        # self.scheduler.build()

    def prologue(self):
        version = self._get_version()
        print(LOGO % version)
        print(
            "Copyright (C) 2025, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!"
        )
        print("https://github.com/NumHub612/YunmengEnvs\n")

    def _get_version(self):
        version_file = os.path.join(os.path.dirname(__file__), "VERSION")
        with open(version_file, "r") as f:
            version = f.read().strip()
        return version

    def run(self):
        try:
            # self.orchestrator.run()
            ...
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            ...
            # self.orchestrator.exit()

    def epilogue(self):
        print("\nYunmengEnvs exited. Thanks for supporting YunmengEnvs!")


if __name__ == "__main__":
    app = YunmengEnvsApp()
    logger.info("YunmengEnvs starting.")
    app.prologue()
    app.run()
    app.epilogue()
    logger.info("YunmengEnvs finished.")
