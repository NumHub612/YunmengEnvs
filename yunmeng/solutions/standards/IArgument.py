# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for arguments.
"""
from yunmeng.solutions.standards.IIdentifiable import IIdentifiable

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class IArgument(IIdentifiable):
    """Class for providing arguments."""

    # The type of the value of the argument.
    value_type: type = None

    # Whether the argument is optional.
    optional: bool = False

    # Whether the value property can be edited.
    readonly: bool = False

    # The default value of the argument.
    default: Any = None

    # Possible allowed values.
    possibles: list = None

    @property
    @abstractmethod
    def value(self) -> Any:
        """The current value of the argument."""
        pass

    @value.setter
    @abstractmethod
    def value(self, value: Any) -> None:
        """Sets the value of the argument."""
        pass
