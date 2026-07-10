# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Adapters and factories for different types of links.
"""
from yunmeng.solutions.standards import (
    IAdaptedOutputFactory,
    IAdaptedOutput,
    IOutput,
    IInput,
    IArgument,
    IValueSet,
)


class AdapterFactory(IAdaptedOutputFactory):
    """Factory class for creating instances of the `IAdaptedOutput` item."""

    def __init__(
        self,
        id: str,
        adapters: list[IAdaptedOutput] = None,
        caption: str = "",
        description: str = "",
    ):
        super().__init__(caption, description, id)
        self._adapters = adapters or []

    def get_available_adapter_ids(
        self, adaptee: IOutput, target: IInput
    ) -> list[IAdaptedOutputFactory]:
        pass

    def create_adapter(
        self,
        adapter_id: IAdaptedOutputFactory,
        adaptee: IOutput,
        target: IInput,
    ) -> IAdaptedOutput | None:
        pass


class OutputAdapter(IAdaptedOutput):
    """Adapter class for the `IOutput` item."""

    def __init__(
        self,
        id: str,
        adaptee: IOutput,
        caption: str = "",
        description: str = "",
    ):
        super().__init__(caption, description, id)
        self._adaptee = adaptee
        self._arguments = []
        self._consumers = []
        self._adapters = []

    @property
    def arguments(self) -> list[IArgument]:
        return self._arguments

    @property
    def adaptee(self) -> IOutput:
        return self._adaptee

    @adaptee.setter
    def adaptee(self, adaptee: IOutput):
        self._adaptee = adaptee

    def get_values(self, querier: IInput) -> IValueSet:
        pass

    def initialize(self):
        pass

    def refresh(self):
        pass
