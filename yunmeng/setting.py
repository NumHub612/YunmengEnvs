# -*- encoding: utf-8 -*-
"""
Project configurations.
"""

import logging
import sys

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


def print_logo(version: str, year: int):
    print(LOGO % version)
    print(
        f"Copyright (C) {year}, The YunMengEnvs Project Contributors. Welcome aboard YunmengEnvs!"
    )
    print("https://github.com/NumHub612/YunmengEnvs\n")


# set up logging system
def _make_logger() -> logging.Logger:
    logger = logging.getLogger("yunmeng")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


logger = _make_logger()


# set up random seed
class _Settings:
    """Global settings activated from the ENVS section of a config file."""

    def __init__(self) -> None:
        self._store: dict = {"log_level": "INFO", "device": "cpu"}

    def load(self, configs: dict) -> None:
        if not configs:
            return
        self._store.update(configs)
        level = str(self._store.get("log_level", "INFO")).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

    def get(self, key: str, default: object = None):
        return self._store.get(key, default)

    def __getitem__(self, key: str):
        return self._store[key]


settings = _Settings()
