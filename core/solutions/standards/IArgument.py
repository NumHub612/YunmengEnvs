# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for arguments.
"""
from abc import abstractmethod
from typing import Any, List

from core.solutions.standards.IIdentifiable import IIdentifiable


class IArgument(IIdentifiable):
    """Class for providing arguments."""

    @property
    @abstractmethod
    def value_type(self) -> type:
        """The type of the value of the argument."""
        pass

    @property
    @abstractmethod
    def optional(self) -> bool:
        """Whether the argument is optional."""
        pass

    @property
    @abstractmethod
    def readonly(self) -> bool:
        """Whether the value property can be edited."""
        pass

    @property
    @abstractmethod
    def value(self) -> Any:
        """The current value of the argument."""
        pass

    @value.setter
    @abstractmethod
    def value(self, value: Any):
        """Sets the argument value, if settable."""
        pass

    @property
    @abstractmethod
    def default_value(self) -> Any:
        """The default value of the argument."""
        pass

    @property
    @abstractmethod
    def possible_values(self) -> List[Any]:
        """Possible allowed values for the argument."""
        pass
