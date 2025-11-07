# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!

YunmengEnvs entrence.
"""
from configs.orchestrator import Orchestrator
from configs.settings import logger

from core.solutions import ym_models
from core.solvers import ym_solvers

import argparse
import os


class YunmengEnvsApp:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="YunmengEnvs")
        self.parser.add_argument(
            "-c", "--config", type=str, help="config file path", required=True
        )
        self.args = self.parser.parse_args()
        self.orchestrator = Orchestrator(self.args.config)

    def run(self):
        logger.info("YunmengEnvs started.")
        self.orchestrator.run()
        logger.info("YunmengEnvs finished.")


if __name__ == "__main__":
    app = YunmengEnvsApp()
    app.run()
