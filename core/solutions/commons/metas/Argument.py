# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Component argument.
"""
from core.solutions.standards import IArgument
from dataclasses import dataclass
from typing import Any


@dataclass
class Argument(IArgument):
    """Argument for component meta configs."""

    def __init__(
        self,
        name: str,
        value_type: type,
        value: Any = None,
        optional: bool = False,
        readonly: bool = False,
        default: Any = None,
        possibles: list = None,
    ):
        super().__init__(name)
        self.value_type = value_type
        self.optional = optional
        self.readonly = readonly
        self.default = default
        self.possibles = possibles
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any):
        if self.readonly:
            pass
        if self.possibles is not None and value not in self.possibles:
            raise ValueError(f"Invalid value: {value}.")
        if value is not None and not isinstance(value, self.value_type):
            raise TypeError(f"Invalid type: {type(value)}.")
        if value is not None:
            self._value = self.value_type(value)
