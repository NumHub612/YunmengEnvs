# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import json
import torch
import GPUtil


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._configs = {}
        return cls._instance

    def load_configs(self, config_file="configs.json"):
        with open(config_file, "r") as f:
            self._configs = json.load(f)

    def save_configs(self, config_file="configs.json"):
        with open(config_file, "w") as f:
            json.dump(self._configs, f, indent=4)

    def __getattr__(self, name):
        if name in self._configs:
            return self._configs[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._configs[name] = value

    @property
    def NUMERIC_TOLERANCE(self) -> float:
        """Numeric tolerance for floating point comparison."""
        return self._configs.get("NUMERIC_TOLERANCE", 1e-6)

    @NUMERIC_TOLERANCE.setter
    def NUMERIC_TOLERANCE(self, value: float):
        self._configs["NUMERIC_TOLERANCE"] = abs(value)

    @property
    def DEVICE(self) -> torch.device:
        """Device for PyTorch."""
        return self._configs.get(
            "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @DEVICE.setter
    def DEVICE(self, value: torch.device):
        self._configs["DEVICE"] = value

    @property
    def GPUs(self) -> list:
        """List of GPUs available."""
        return self._configs.get("GPUs", GPUtil.getGPUs())

    @GPUs.setter
    def GPUs(self, value: list):
        self._configs["GPUs"] = value


# global settings object
settings = Settings()


# set up logging system
logger = logging.getLogger("yunmengenvs")
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s][%(process)d][%(thread)d]: %(message)s"
)

log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
file_handler = logging.handlers.RotatingFileHandler(log_file, "a", 1024 * 1024 * 10, 10)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
