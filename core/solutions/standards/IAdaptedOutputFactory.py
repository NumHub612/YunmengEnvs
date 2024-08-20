# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for creating instances of `IAdaptedOutput` items.
"""
from abc import abstractmethod
from typing import List, Optional

from .IIdentifiable import IIdentifiable
from .IAdaptedOutput import IAdaptedOutput
from .IOutput import IOutput
from .IInput import IInput


class IAdaptedOutputFactory(IIdentifiable):
    """
    Factory class for creating instances of `IAdaptedOutput` items.
    """

    @abstractmethod
    def get_available_adapter_ids(
        self, output: IOutput, target: Optional[IInput]
    ) -> List[IIdentifiable]:
        """Gets a list of identifier of the available `IAdaptedOutput` that can
        make the adaptee match the target."""
        pass

    @abstractmethod
    def create_adapter(
        self,
        adapter_id: IIdentifiable,
        adaptee: IOutput,
        target: Optional[IInput],
    ) -> IAdaptedOutput:
        """Creates an instance of `IAdaptedOutput` that can adapt the adaptee."""
        pass
