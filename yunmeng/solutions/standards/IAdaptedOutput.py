# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for adapted output items.
"""
from yunmeng.solutions.standards.IArgument import IArgument
from yunmeng.solutions.standards.IOutput import IOutput

from abc import abstractmethod


class IAdaptedOutput(IOutput):
    """
    Extension class for adding data operations on top of an output item.

    The `IAdaptedOutput` interface extends an `IOutput` item with functionality, such as
    spatial interpolation, temporal interpolation, unit conversion etc.
    """

    @abstractmethod
    def initialize(self):
        """Initializes the adapter."""
        pass

    @abstractmethod
    def refresh(self):
        """Refreshes the adapter."""
        pass

    @property
    @abstractmethod
    def arguments(self) -> list[IArgument]:
        """Arguments of the adapter."""
        pass

    @property
    @abstractmethod
    def adaptee(self) -> IOutput:
        """The adapted output item."""
        pass

    @adaptee.setter
    @abstractmethod
    def adaptee(self, adaptee: IOutput):
        """Sets an output item."""
        pass
