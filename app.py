# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

YunmengEnvs entrence.
"""

import yunmeng as ym
from yunmeng.taskflow.parser import Orchestrator
from yunmeng.taskflow.builder import SchedulerBuilder
from yunmeng.setting import print_logo
from yunmeng.solutions import ym_models
from yunmeng.setting import logger

import datetime
import argparse


class YunmengEnvsApp:
    """YunmengEnvs application."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(description="YunmengEnvs")
        self._parser.add_argument("config", type=str, help="config yaml")

    def prologue(self):
        year = datetime.datetime.now().year
        version = ym.__version__
        print_logo(version, year)

        try:
            self._configer = Orchestrator(self._parser.parse_args())
            builder = SchedulerBuilder(ym_models)
            self._scheduler = builder.build(self._configer)
            self._scheduler.initialize()
        except Exception as e:
            logger.exception(e)
            raise e

    def run(self):
        try:
            max_steps = self._configer.schedules.get("max_steps")
            self._scheduler.run(max_steps=max_steps)
        except Exception as e:
            logger.exception(e)
            raise e

    def epilogue(self):
        print("\nYunmengEnvs completed. Thanks for supporting, enjoy your journey!")


if __name__ == "__main__":
    app = YunmengEnvsApp()
    logger.info("--------- YunmengEnvs starting ---------")
    app.prologue()
    app.run()
    app.epilogue()
    logger.info("--------- YunmengEnvs finished ---------")
