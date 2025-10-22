# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Output items.
"""
from core.solutions.standards import (
    ILinkableComponent,
    IBaseExchangeItem,
    IOutput,
    IAdaptedOutput,
    IInput,
    ISpatialDefinition,
    IElementSet,
    ITimeSet,
    ITime,
    IValueDefinition,
    IValueSet,
    ExchangeItemChangeEventArgs,
)
from core.solutions.commons import events, datasets
from configs.settings import logger

from typing import Optional
import numpy as np


class BaseOutput(IOutput):
    """Base output item for all outputs.

    When the output `get_values` method is invoked, it'll call the `update`
    method of the owning component B to generate data and
    obtain the data required.

    The output item and the component are tightly coupled, which means the
    former depends on the specific implementation of the latter.
    """

    def __init__(
        self,
        id: str,
        component: ILinkableComponent,
        value_definition: IValueDefinition,
        elementset: IElementSet,
        timeset: ITimeSet = None,
        caption: str = "",
        description: str = "",
    ):
        super().__init__(caption, description, id)
        self._component = component
        self._value_definition = value_definition
        self._elementset = elementset
        self._timeset = timeset
        if self._timeset is None:
            self._timeset = datasets.TimeSet(None, [datasets.ITime])
        self._valueset: IValueSet = datasets.ValueSet(
            value_definition,
            (self._timeset.size, elementset.element_count),
        )

        self._consumers: list[IInput] = []
        self._adapters: list[IAdaptedOutput] = []
        self._in_get_values: bool = False
        self._event_manager = events.EventManager()

    def __del__(self):
        if not self._consumers:
            for consumer in self._consumers:
                consumer.provider = None
        if not self._adapters:
            for adapter in self._adapters:
                adapter.adaptee = None

    @property
    def spatial_definition(self) -> Optional[ISpatialDefinition]:
        return self._elementset

    @property
    def time_set(self) -> Optional[ITimeSet]:
        return self._timeset

    @property
    def value_definition(self) -> IValueDefinition:
        return self._value_definition

    @property
    def values(self) -> IValueSet:
        return self._valueset

    @property
    def consumers(self) -> Optional[list[IInput]]:
        return self._consumers

    @property
    def adapters(self) -> Optional[list[IOutput]]:
        return self._adapters

    @property
    def component(self) -> ILinkableComponent:
        return self._component

    @property
    def event_manager(self) -> events.EventManager:
        return self._event_manager

    def add_consumer(self, consumer: IInput):
        # TODO: add check for consumer's spatial definition
        if consumer not in self._consumers:
            self._consumers.append(consumer)

            self.notify_changed("Consumer added.")

    def add_adapter(self, adapter: IAdaptedOutput):
        if adapter not in self._adapters:
            self._adapters.append(adapter)
            adapter.adaptee = self

            self.notify_changed("Adapter added.")

    def remove_consumer(self, consumer: IInput):
        if consumer in self._consumers:
            self._consumers.remove(consumer)

            self.notify_changed("Consumer removed.")

    def remove_adapter(self, adapter: IAdaptedOutput):
        if adapter in self._adapters:
            self._adapters.remove(adapter)
            adapter.adaptee = None

            self.notify_changed("Adapter removed.")

    def get_values(self, querier: IBaseExchangeItem) -> IValueSet:
        # NOTE: This is a simplified pull-based implementation.
        # The implementation class may override this method for more complex logic.

        # Prevent re-entrance deadlock
        if self._in_get_values:
            return self._guess_extrapolate(querier)

        self._in_get_values = True
        try:
            # retrieve the value for only one moment at a time
            req_time = querier.time_set.times[0].timestamp
            cur_time = self._current_time()

            # 1. time: future → push component until that time
            if req_time > cur_time:
                while self._current_time() < req_time:
                    self._pull_component()
                cur_time = self._current_time()

            # 2. time: past → check cache / interpolate
            outputs = self._get_cache(querier)
            if outputs is None:
                outputs = self._interpolate(querier)

            # shrink according to consumer's request
            self._shrink(querier)

            return outputs

        finally:
            self._in_get_values = False

    def _current_time(self) -> float:
        """Gets the current time in Modified Julian Day (MJD)."""
        if self._timeset is None or len(self._timeset.times) == 0:
            return 0.0
        return self._timeset.times[-1].timestamp

    def _get_cache(self, querier: IBaseExchangeItem) -> IValueSet:
        """Gets cached value set for given time, if any."""
        cached = None

        # match exact time
        req_time = querier.time_set.times[0].timestamp
        tolerance = 1e-6  # tolerance for time matching
        for i, t in enumerate(self._timeset.times):
            if abs(t.timestamp - req_time) <= tolerance:
                cached = np.array(self._valueset[i])
                break
        if cached is None:
            return None

        # match spatial definition
        element_indices = []
        for i, elem in enumerate(querier.spatial_definition.elements):
            idx = self._elementset.get_element_index(elem.id)
            if idx is not None:
                element_indices.append(idx)
        if len(element_indices) == 0:
            return None
        cached = cached[element_indices]

        # return valueset
        return datasets.ValueSet(
            self._value_definition,
            (1, len(element_indices)),
            cached,
        )

    def _guess_extrapolate(self, query: IBaseExchangeItem) -> IValueSet:
        """When re-entrance deadlock occurs, extrapolate values."""
        if not self._valueset or self._valueset.shape[0] == 0:
            # no data yet, return missing
            n_elem = query.spatial_definition.element_count
            return datasets.ValueSet(
                self._value_definition,
                (1, n_elem),
            )

        # directly return last value, the implementation class
        # may override this method.
        return self._valueset[-1]

    def _interpolate(self, querier: IBaseExchangeItem) -> IValueSet:
        """Interpolates values for the querier."""
        req_time = querier.time_set.times[0].timestamp
        times = self._timeset.times
        for i in range(1, len(times)):
            time_l = times[i - 1].timestamp
            time_r = times[i].timestamp
            # valueset interpolation
            if time_l <= req_time <= time_r:
                ratio = (req_time - time_l) / (time_r - time_l)
                val_l, val_r = self._valueset[i - 1 : i + 1]
                results = [None] * val_l.shape[1]
                for j in range(val_l.shape[1]):
                    v1, v2 = val_l[j], val_r[j]
                    results[j] = v1 * (1 - ratio) + v2 * ratio
                return datasets.ValueSet(
                    self._value_definition,
                    (1, len(results)),
                    results,
                )
        return None

    def _pull_component(self):
        """Pulls the component to update itself to get time and values."""
        self._component.update([self])

        # NOTE: just for easier implementation, use implicit presumed
        # attributes and states here.

        # Use "current_time" as attribute name to get current time.
        cur_time = self._component.attributes.get("current_time", None)
        if cur_time is not None:
            raise ValueError("Component no 'current_time' attribute.")

        # Use `caption` as state name to get current values.
        values = self._component.states.get(self.caption, None)
        if values is None:
            raise ValueError(f"Component no '{self.caption}' state.")

        # Update time set and value set
        if self._timeset is not None:
            new_time = ITime(timestamp=cur_time)
            self._timeset.times.append(new_time)
        if self._valueset is None:
            self._valueset = datasets.ValueSet(
                self._value_definition,
                (0, values.shape[1]),
                values,
            )

        self.notify_changed("Component pulled.")

    def _shrink(self, querier: IBaseExchangeItem):
        """Shrinks the datas according to consumer's request."""
        # search the earliest time index
        req_time = querier.time_set.times[0].timestamp
        for consumer in self._consumers:
            t = consumer.time_set.times[0].timestamp
            if t < req_time:
                req_time = t

        remove_indices = []
        for i, t in enumerate(self._timeset.times):
            if t.timestamp < req_time:
                remove_indices.append(i)

        # shrink data
        if remove_indices:
            for idx in reversed(remove_indices):
                self._timeset.remove_time(idx)
                self._valueset.remove_values(idx)

        self.notify_changed("Data shrunk.")

    def notify_changed(self, message: str):
        """Notifies all consumers that the output item has changed."""
        logger.info(f"Output item {self.id} has changed: {message}")
        event_args = ExchangeItemChangeEventArgs(self, message)
        self._event_manager.invoke(event_args)
