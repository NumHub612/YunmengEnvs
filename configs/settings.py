# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import json

# set up logging
logger = logging.getLogger("yunmengenvs")
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s][%(process)d][%(thread)d]: %(message)s"
)
logger.setLevel(logging.INFO)

log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
file_handler = logging.handlers.RotatingFileHandler(log_file, "a", 1024 * 1024 * 10, 10)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)


# constants
NUMERIC_TOLERANCE = 1e-6


# confirm config file exists
cnf_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "configures.json"))

configs = {}


def load_configs(config_file: str):
    global configs
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configures file {config_file} not found.")

    configs = json.load(open(config_file, "r", encoding="utf8"))


# get logger
def get_logger():
    global logger, configs
    log_level = configs.get("log_level", "INFO")
    logger.setLevel(log_level)
    return logger


# get full configs
def full_configs():
    global configs
    return configs


# get global configs
def global_configs():
    global configs
    usr_global_configs = configs.get("global", {})
    return usr_global_configs
