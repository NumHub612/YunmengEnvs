# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for creating instances of `IAdaptedOutput` items.
"""
from yunmeng.solutions.standards.IAdaptedOutput import IAdaptedOutput
from yunmeng.solutions.standards.IIdentifiable import IIdentifiable
from yunmeng.solutions.standards.IOutput import IOutput
from yunmeng.solutions.standards.IInput import IInput

from abc import abstractmethod
from typing import Optional


class IAdaptedOutputFactory(IIdentifiable):
    """
    Factory class for creating instances of the `IAdaptedOutput` item.
    """

    @abstractmethod
    def get_available_adapter_ids(
        self, adaptee: IOutput, target: IInput
    ) -> list[IIdentifiable]:
        """Gets a list of identifier of the available `IAdaptedOutput`
        that can make the adaptee match the target."""
        pass

    @abstractmethod
    def create_adapter(
        self,
        adapter_id: IIdentifiable,
        adaptee: IOutput,
        target: IInput,
    ) -> Optional[IAdaptedOutput]:
        """Creates an adapter adapting the adaptee to target."""
        pass
