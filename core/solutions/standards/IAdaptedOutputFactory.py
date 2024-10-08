# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for creating instances of `IAdaptedOutput` items.
"""
from abc import abstractmethod
from typing import List, Optional

from core.solutions.standards.IIdentifiable import IIdentifiable
from core.solutions.standards.IAdaptedOutput import IAdaptedOutput
from core.solutions.standards.IOutput import IOutput
from core.solutions.standards.IInput import IInput


class IAdaptedOutputFactory(IIdentifiable):
    """
    Factory class for creating instances of the `IAdaptedOutput` item.
    """

    @abstractmethod
    def get_available_adapter_ids(
        self, adaptee: IOutput, target: IInput
    ) -> List[IIdentifiable]:
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
        """Creates an adapter instance adapting the adaptee to target."""
        pass
