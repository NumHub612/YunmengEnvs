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
import torch


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
file_size = 1024 * 1024 * 10  # 10MB
file_handler = logging.handlers.RotatingFileHandler(log_file, "a", file_size, 9)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


# set up random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# set up gpu device
GPUS = list(range(torch.cuda.device_count()))


# set up global settings object
class YunmengSettings:
    """Project settings。

    NOTE: Not to use `settings` object to access and modify settings,
    instead of directly cachine each attribute.
    """

    _lock = threading.RLock()
    _instance = None
    _configs = {}

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YunmengSettings, cls).__new__(cls)
        return cls._instance

    def load(self, configs: dict):
        with self._lock:
            for k, v in configs.items():
                self.__setattr__(k, v)
        logger.info(f"Load configs from {configs}.")

    def save(self, out_file: str):
        with self._lock:
            configs_copy = self._configs.copy()

        with open(out_file, "w") as f:
            json.dump(configs_copy, f)
        logger.info(f"Saved configs to {out_file}.")

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._configs

    def __getattr__(self, name: str):
        with self._lock:
            if name in self._configs:
                return self._configs[name]
            else:
                raise AttributeError(f"Settings has no attribute {name}.")

    def __setattr__(self, name: str, value):
        with self._lock:
            if name == "log_level":
                logger.setLevel(value)
            if name == "device":
                value = value.lower()
                assert value in ["cuda", "cpu"]
            self._configs[name] = value

    @property
    def log_level(self) -> int:
        return self._configs.get("log_level", logging.INFO)

    @property
    def device(self) -> str:
        """[cuda, cpu]"""
        return self._configs.get("device", "cuda" if GPUS else "cpu")

    @property
    def gpus(self) -> list[int]:
        return self._configs.get("gpus", GPUS)


# global settings object
settings = YunmengSettings()
