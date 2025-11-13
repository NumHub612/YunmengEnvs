# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

YunmengEnvs entrence.
"""
from configs.orchestrator import Orchestrator
from configs.settings import LOGO, logger

from core.solutions.commons import Scheduler
from core.solutions import ym_models

import argparse
import os


class YunmengEnvsApp:
    """YunmengEnvs application."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(description="YunmengEnvs")
        self._parser.add_argument("config", type=str, help="config yaml")
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

    def prologue(self):
        version = self._get_version()
        print(LOGO % version)
        print(
            "Copyright (C) 2025, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!"
        )
        print("https://github.com/NumHub612/YunmengEnvs\n")

        try:
            self._configer = Orchestrator(self._parser)
            self._configer.activate()

            self._scheduler = Scheduler(ym_models)
            self._scheduler.setup(self._configer)
            self._scheduler.initialize()

            errors = self._scheduler.validate()
            if errors:
                raise ValueError(f"Scheduler validation failed: {errors}")
        except Exception as e:
            logger.exception(e)
            raise e

    def _get_version(self):
        version_file = os.path.join(os.path.dirname(__file__), "VERSION")
        with open(version_file, "r") as f:
            version = f.read().strip()
        return version

    def run(self):
        try:
            self._scheduler.prepare()
            self._scheduler.run()
            self._scheduler.finish()
        except Exception as e:
            logger.exception(e)
            raise e

    def epilogue(self):
        print("\nYunmengEnvs exited. Thanks for supporting YunmengEnvs!")


if __name__ == "__main__":
    app = YunmengEnvsApp()
    logger.info("YunmengEnvs starting.")
    app.prologue()
    app.run()
    app.epilogue()
    logger.info("YunmengEnvs finished.")
