# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

YunmengEnvs entrence.
"""
from core.solutions.commons.links import Scheduler
from core.solutions import ym_models
from configs.orchestrator import Orchestrator
from configs.settings import LOGO, logger

import datetime
import argparse
import os
import sys


def _no_tb_hook(etype, val, _tb):
    sys.stderr.write(f"{etype.__name__}: {val}\n")


sys.excepthook = _no_tb_hook


class YunmengEnvsApp:
    """YunmengEnvs application."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(description="YunmengEnvs")
        self._parser.add_argument("config", type=str, help="config yaml")

    def prologue(self):
        version_file = os.path.join(os.path.dirname(__file__), "VERSION")
        with open(version_file, "r") as f:
            version = f.read().strip()
        year = datetime.datetime.now().year
        print(LOGO % version)
        print(
            f"Copyright (C) {year}, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!"
        )
        print("https://github.com/NumHub612/YunmengEnvs\n")

        try:
            self._configer = Orchestrator(self._parser.parse_args())
            self._scheduler = Scheduler(ym_models)
            self._scheduler.setup(self._configer)
            self._scheduler.initialize()
            errors = self._scheduler.validate()
            if errors:
                raise ValueError(f"Scheduler validation failed: {errors}")
        except Exception as e:
            logger.exception(e)
            raise e

    def run(self):
        try:
            self._scheduler.prepare()
            self._scheduler.run()
            self._scheduler.finish()
        except Exception as e:
            logger.exception(e)
            raise e

    def epilogue(self):
        print("\nYunmengEnvs completed. Thanks for supporting, enjoy your journey!")


if __name__ == "__main__":
    app = YunmengEnvsApp()
    logger.info("YunmengEnvs starting.")
    app.prologue()
    app.run()
    app.epilogue()
    logger.info("YunmengEnvs finished.")
