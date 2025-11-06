# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""
from core.numerics.fields import Field
from typing import NamedTuple


class TimestampedField(NamedTuple):
    data: Field
    timestamp: float


class RingBuffer:
    """RingBuffer for storing the fields and its history."""

    def __init__(self, size: int):
        self._data = [None] * size
        self._cursor = 0

    def push(self, obj: TimestampedField):
        self._data[self._cursor] = obj
        self._cursor = (self._cursor + 1) % len(self._data)

    def __getitem__(self, level: int):
        i = (self._cursor - 1 - level) % len(self._data)
        return self._data[i]


class DataHub:
    """Datahub for managing the fields and its history."""

    def __init__(self, fields: list[str], levels: int):
        self._bufs = {f: RingBuffer(levels) for f in fields}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._bufs[k].push(v)

    def fetch(self, name: str, level: int = 0):
        return self._bufs[name][level]
