# -*- encoding: utf-8 -*-
"""
Project configurations.
"""
import os
import logging
import logging.handlers
import json
import torch


# set up logging system
def create_logger(name: str = "yunmengenvs", level=logging.INFO):
    """Create a logger with the given name."""
    if name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s][%(process)d][%(thread)d]: %(message)s"
    )

    log_file = os.path.abspath(os.path.join("./", "yunmeng.log"))
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, "a", 1024 * 1024 * 10, 10
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    logger.setLevel(level)
    return logger


logger = create_logger()

# set up random seed
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# set up global settings object
class Settings:

    default_configs = {
        "DTYPE": torch.float64,
        "NUMERIC_TOLERANCE": 1e-6,
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "GPUs": (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance.__dict__["_configs"] = cls.default_configs.copy()
        return cls._instance

    def load_configs(self, config_file="configs.json"):
        with open(config_file, "r") as f:
            self._configs = json.load(f)

        for key, value in self.default_configs.items():
            if key not in self._configs:
                self._configs[key] = value

        logger.info(f"Loaded configurations from {config_file}")

    def save_configs(self, config_file="configs.json"):
        with open(config_file, "w") as f:
            json.dump(self._configs, f, indent=4)
        logger.info(f"Saved configurations to {config_file}")

    def has_attr(self, name: str) -> bool:
        return name in self._configs

    def __getattr__(self, name):
        if name in self.__dict__["_configs"]:
            return self.__dict__["_configs"][name]
        else:
            raise AttributeError(f"Has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__dict__["_configs"]:
            self.__dict__["_configs"][name] = value
        else:
            raise AttributeError(f"Has no attribute '{name}'")

    @property
    def DTYPE(self) -> torch.dtype:
        """Data type for PyTorch."""
        return self._configs.get("DTYPE", torch.float64)

    @DTYPE.setter
    def DTYPE(self, value: torch.dtype):
        self._configs["DTYPE"] = value
        logger.info(f"Set DTYPE to {value}")

    @property
    def NUMERIC_TOLERANCE(self) -> float:
        """Tolerance for float point comparison."""
        return self._configs.get("NUMERIC_TOLERANCE", 1e-6)

    @NUMERIC_TOLERANCE.setter
    def NUMERIC_TOLERANCE(self, value: float):
        self._configs["NUMERIC_TOLERANCE"] = abs(value)
        logger.info(f"Set NUMERIC_TOLERANCE to {value}")

    @property
    def DEVICE(self) -> torch.device:
        """Device for PyTorch."""
        return self._configs.get(
            "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @DEVICE.setter
    def DEVICE(self, value: torch.device):
        self._configs["DEVICE"] = value
        logger.info(f"Set DEVICE to {value}")

    @property
    def GPUs(self) -> list:
        """List of GPUs available."""
        if torch.cuda.is_available():
            gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            gpus = []
        return self._configs.get("GPUs", gpus)

    @GPUs.setter
    def GPUs(self, value: list[int]):
        if max(value) >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU index: {max(value)}")
        self._configs["GPUs"] = [f"cuda:{i}" for i in value]
        if len(value) == 0:
            self._configs["DEVICE"] = torch.device("cpu")
        else:
            self._configs["DEVICE"] = torch.device(self._configs["GPUs"][0])
        logger.info(f"Set GPUs to {value}")


# global settings object
settings = Settings()
