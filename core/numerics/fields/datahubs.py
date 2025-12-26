# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""
from core.numerics.fields import Field
from typing import NamedTuple


class Sample(NamedTuple):
    """A sample of the field at a certain time step."""

    timestamp: float
    timestep: float
    data: Field


class DataHub:
    """Datahub for managing the fields and its history.

    NOTE: Not modify the sample data in place.
    """

    def __init__(self, fields: list[str], levels: int):
        self._bufs = {f: RingBuffer(levels) for f in fields}
        self._size = levels

    @property
    def levels(self) -> int:
        """Return the number of levels."""
        return self._size

    @property
    def fields(self) -> list[str]:
        """Return the list of fields."""
        return list(self._bufs.keys())

    def clear(self):
        """Clear the datahub."""
        for buf in self._bufs.values():
            buf.clear()

    def update(self, **kwargs):
        """Update the datahub with the new fields.

        NOTE: It's possible to update regisitered fields separately,
        which may cause misalignment.
        """
        for k, v in kwargs.items():
            if k not in self._bufs:
                continue
            self._bufs[k].push(v)

    def fetch(self, level: int = 0, name: str = None) -> Sample:
        """Fetch the field at the given level.

        NOTE: level=0 present the latest data,
        level=1 present the previous data,
        and so on.
        """
        if name is None:
            name = list(self._bufs.keys())[0]
        if name not in self._bufs:
            raise ValueError(f"Field {name} not found.")
        return self._bufs[name][level]


class RingBuffer:
    """RingBuffer for storing the fields and its history."""

    def __init__(self, size: int):
        self._i = 0
        self._size = size
        self._data = [None] * size

    def clear(self):
        self._data = [None] * self._size
        self._i = 0

    def push(self, obj: Sample):
        if not isinstance(obj, Sample):
            raise TypeError(f"Invalid buffer type {type(obj)}.")
        self._data[self._i] = obj
        self._i = (self._i + 1) % self._size

    def __getitem__(self, level: int):
        if level >= self._size or level < 0:
            raise IndexError(f"Index {level} out of range.")
        i = (self._i - 1 - level) % self._size
        return self._data[i]

    def __len__(self):
        return self._size
