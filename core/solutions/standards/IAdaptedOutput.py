# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for an output item that can be adapted with data operations.
"""
from abc import abstractmethod
from typing import List, Optional

from .IOutput import IOutput
from .IArgument import IArgument


class IAdaptedOutput(IOutput):
    """
    Extension class for adding data operations on top of an output item.

    The `IAdaptedOutput` interface extends an `IOutput` item with functionality, such as
    spatial interpolation, temporal interpolation, unit conversion etc.
    """

    @abstractmethod
    def get_arguments(self) -> List[Optional[IArgument]]:
        """Arguments needed to let the adapted output do its work."""
        pass

    @abstractmethod
    def initialize(self):
        """Lets the adapted output initialize itself based on the current values
        specified by the arguments.
        """
        pass

    @abstractmethod
    def get_adaptee(self) -> Optional[IOutput]:
        """Returns the output item."""
        pass

    @abstractmethod
    def set_adaptee(self, adaptee: IOutput):
        """Sets an output that necessitates the specified adpative data operations."""
        pass

    @abstractmethod
    def refresh(self):
        """Requests the adapted output item to refresh itself."""
        pass
