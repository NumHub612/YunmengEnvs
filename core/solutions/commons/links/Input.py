# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Input items.
"""
from core.solutions.standards import (
    ILinkableComponent,
    IOutput,
    IInput,
    ISpatialDefinition,
    IElementSet,
    ITimeSet,
    IValueDefinition,
    IValueSet,
    ExchangeItemChangeEventArgs,
)
from core.solutions.commons import events
from configs.settings import logger

from typing import Optional


class Input(IInput):
    """Input item which provides values for an `ILinkableComponent`."""

    def __init__(
        self,
        id: str,
        component: ILinkableComponent,
        value_definition: IValueDefinition,
        elementset: IElementSet = None,
        timeset: ITimeSet = None,
        caption: str = "",
        description: str = "",
        provider: IOutput = None,
    ):
        super().__init__(caption, description, id)
        self._component = component
        self._value_definition = value_definition
        self._elementset = elementset
        self._timeset = timeset
        self._provider = provider
        self._valuset = None
        self._satisfied = False
        self._event_manager = events.EventManager()

    @property
    def spatial_definition(self) -> Optional[ISpatialDefinition]:
        return self._elementset

    @property
    def time_set(self) -> Optional[ITimeSet]:
        return self._timeset

    @time_set.setter
    def time_set(self, timeset: ITimeSet):
        """Resets the time set of the input."""
        self._timeset = timeset
        self._valuset = None
        self._satisfied = False
        self.notify_changed("Time set changed.")

    @property
    def value_definition(self) -> IValueDefinition:
        return self._value_definition

    @property
    def provider(self) -> Optional[IOutput]:
        return self._provider

    @provider.setter
    def provider(self, provider: IOutput):
        if self._provider is not None:
            self._provider.remove_consumer(self)

        self._provider = provider
        if self._provider is not None:
            self._provider.add_consumer(self)

        self._valuset = None
        self._satisfied = False

        self.notify_changed("Provider changed.")

    @property
    def component(self) -> ILinkableComponent:
        return self._component

    @property
    def event_manager(self) -> events.EventManager:
        return self._event_manager

    @property
    def values(self) -> IValueSet:
        if self._provider is None:
            raise ValueError("Input has no provider.")

        if self._valuset is None or not self._satisfied:
            self._valuset = self._provider.get_values(self)
            # TODO: do validation here.
            self._satisfied = True
            self.notify_changed("Values updated.")
        return self._valuset

    def notify_changed(self, message: str):
        """Broadcasts a change event."""
        logger.info(f"Input '{self.id}' changed: {message}")
        if self._event_manager is None:
            return

        event_args = ExchangeItemChangeEventArgs(self, message)
        self._event_manager.invoke(event_args)
