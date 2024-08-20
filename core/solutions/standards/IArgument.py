# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for arguments.
"""
from abc import abstractmethod
from typing import Any, List

from .IIdentifiable import IIdentifiable


class IArgument(IIdentifiable):
    """Class for providing arguments for an `ILinkableComponent`."""

    @abstractmethod
    def get_value_type(self) -> type:
        """Gets the type of the value of the argument."""
        pass

    @abstractmethod
    def is_optional(self) -> bool:
        """Specifies whether the argument is optional."""
        pass

    @abstractmethod
    def is_read_only(self) -> bool:
        """Defines whether the value property can be edited."""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """Gets the current value of the argument."""
        pass

    @abstractmethod
    def set_value(self, value: Any):
        """Sets the argument value, if settable."""
        pass

    @abstractmethod
    def get_default_value(self) -> Any:
        """The default value of the argument."""
        pass

    @abstractmethod
    def get_possible_values(self) -> List[Any]:
        """List of possible allowed values for the argument."""
        pass
