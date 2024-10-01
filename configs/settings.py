# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import json


# confirm config file exists
cnf_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "configures.json"))

configs = {}
if os.path.exists(cnf_file):
    configs = json.load(open(cnf_file, "r", encoding="utf8"))
else:
    raise FileNotFoundError(f"Configures file {cnf_file} not found.")


# set up logging
logger = logging.getLogger("yunmengenvs")
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s][%(process)d][%(thread)d]: %(message)s"
)

log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=1024 * 1024 * 10, backupCount=10
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
