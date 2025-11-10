# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""
from core.numerics.fields import Field
from typing import NamedTuple


class BufferedField(NamedTuple):
    timestamp: float
    timestep: float
    data: Field


class DataHub:
    """Datahub for managing the fields and its history."""

    def __init__(self, fields: list[str], levels: int):
        self._bufs = {f: RingBuffer(levels) for f in fields}

    def update(self, **kwargs):
        """Update the datahub with the new fields."""
        for k, v in kwargs.items():
            if k not in self._bufs:
                continue
            self._bufs[k].push(v)

    def fetch(self, name: str, level: int = 0) -> BufferedField:
        """Fetch the field at the given level."""
        if name not in self._bufs:
            raise ValueError(f"Field {name} not found.")
        return self._bufs[name][level]


class RingBuffer:
    """RingBuffer for storing the fields and its history."""

    def __init__(self, size: int):
        self._data = [None] * size
        self._size = size
        self._i = 0

    def push(self, obj: BufferedField):
        if not isinstance(obj, BufferedField):
            raise TypeError(f"Invalid type {type(obj)}.")
        self._data[self._i] = obj
        self._i = (self._i + 1) % self._size

    def __getitem__(self, level: int):
        i = (self._i - 1 - level) % self._size
        return self._data[i]
