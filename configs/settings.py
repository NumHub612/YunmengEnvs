# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import numpy as np
import json
import threading

LOGO = """
 __   __                                                   
 \ \ / /  _   _   _ __    _ __ ___     ___   _ __     __ _ 
  \ V /  | | | | | '_ \  | '_ ` _ \   / _ \ | '_ \   / _` |
   | |   | |_| | | | | | | | | | | | |  __/ | | | | | (_| |   
   |_|    \__,_| |_| |_| |_| |_| |_|  \___| |_| |_|  \__, |
                                                     |___/ 
  _____          ><(((('>                                  
 | ____|  _ __   __   __  ___        ><(((('>                      
 |  _|   | '_ \  \ \ / / / __|                          ><(((('>         
 | |___  | | | |  \ V /  \__ \                           
 |_____| |_| |_|   \_/   |___/       -- v%s                                                                                                      
"""  # noqa


# set up logging system
logger = logging.getLogger("yunmeng")
formatter = logging.Formatter(
    "[%(asctime)s][%(process)d][%(thread)d][%(name)s][%(levelname)s]:%(message)s"
)
log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
file_handler = logging.handlers.RotatingFileHandler(log_file, "a", 1024 * 1024 * 10, 10)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


# set up random seed
SEED = 1234
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

# set up gpu device
GPUS = None
try:
    import torch

    GPUS = list(range(torch.cuda.device_count()))
except ImportError:
    pass


# set up global settings object
class YunmengSettings:
    """Project settings。

    NOTE: Sugguest to use `settings` object to access and modify settings,
    instead of directly cachine each attribute.
    """

    _lock = threading.RLock()
    _instance = None
    _configs = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YunmengSettings, cls).__new__(cls)
        return cls._instance

    def load(self, configs: dict):
        self._configs.update(configs)
        logger.info(f"Load configs from {configs}.")

    def save(self, out_file: str):
        with open(out_file, "w") as f:
            json.dump(self._configs, f)
        logger.info(f"Saved configs to {out_file}.")

    def has_attr(self, name: str) -> bool:
        return name in self._configs

    def __getattr__(self, name: str):
        if name == "_configs":
            return self._configs
        if name in self._configs:
            return self._configs[name]
        else:
            raise AttributeError(f"Has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name == "_configs":
            raise AttributeError("Can't reset built-in attribute '_configs'")
        self._configs[name] = value

    @property
    def log_level(self) -> int:
        """日志级别。"""
        return self._configs.get("log_level", logging.INFO)

    @log_level.setter
    def log_level(self, value: int):
        if value not in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]:
            raise ValueError(f"Invalid log level: {value}.")

        self._configs["log_level"] = value
        logger.setLevel(value)

    @property
    def device(self) -> str:
        """GPU or CPU device."""
        return self._configs.get("device", "cuda" if GPUS else "cpu")

    @device.setter
    def device(self, value: str):
        if value not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {value}.")
        if not GPUS and value == "cuda":
            raise ValueError("No GPUs available.")

        self._configs["device"] = value
        if value == "cpu":
            self._configs["gpus"] = []

    @property
    def gpus(self) -> list:
        return self._configs.get("gpus", GPUS)

    @gpus.setter
    def gpus(self, value: list[int]):
        if self.device == "cpu":
            raise ValueError("Cannot set GPUs when device is CPU.")
        if not GPUS and len(value) > 0:
            raise ValueError("No GPUs available.")
        if GPUS and max(value) > GPUS[-1]:
            raise ValueError(f"Invalid GPU index: {max(value)}")

        self._configs["gpus"] = value


# global settings object
settings = YunmengSettings()
