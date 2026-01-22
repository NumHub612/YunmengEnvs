# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Datahubs for managing of the fields and its history.
"""
from core.numerics.fields import Field
from dataclasses import dataclass


@dataclass(slots=True, frozen=True, order=False)
class Sample:
    """A sample of the field at a certain time step."""

    timestamp: float
    timestep: float
    data: Field


class DataHub:
    """Datahub for managing the fields and its history.

    NOTE:
    - Not modify the sample data in place.
    - Not update regisitered fields separately,
    which may cause misalignment.
    """

    def __init__(self, fields: list[str], levels: int):
        """Initialize the datahub with the given fields and levels."""
        self._buffs = {f: RingBuffer(levels) for f in fields}
        self._grads = {f: RingBuffer(levels) for f in fields}
        self._size = levels

    def field(self, name: str, level: int = 0) -> Sample:
        """Fetch the specified field at the given level.

        NOTE: level=0 present the latest data,
        level=1 present the previous data,
        and so on.
        """
        return self._buffs[name][level]

    def grad(self, name: str, level: int = 0) -> Sample:
        """Fetch the specified field's gradient at the given level.

        NOTE: level=0 present the latest gradient,
        level=1 present the previous gradient,
        and so on.
        """
        return self._grads[name][level]

    def has(self, name: str, level: int = 0) -> bool:
        """Check if the datahub has the field and the given level."""
        if name not in self._buffs:
            return False
        if level >= self._size or level < 0:
            return False
        return True

    def clear(self):
        """Clear the datahub."""
        for buff in self._buffs.values():
            buff.clear()
        for grad in self._grads.values():
            grad.clear()

    def push_field(self, name: str, sample: Sample):
        """Push origin field."""
        self._buffs[name].push(sample)

    def push_grad(self, name: str, sample: Sample):
        """Push gradient."""
        self._grads[name].push(sample)

    def push(self, name: str, field: Sample, grad: Sample):
        """Push both field and gradient."""
        self.push_field(name, field)
        self.push_grad(name, grad)


class RingBuffer:
    """Class RingBuffer for managing the history of a field."""

    def __init__(self, size: int = 1):
        """Initialize the ring buffer."""
        if size < 1:
            raise ValueError("Size must be greater than 0.")
        self._i = 0
        self._size = size
        self._data = [None] * size

    def clear(self):
        self._data = [None] * self._size
        self._i = 0

    def push(self, obj: Sample):
        if not isinstance(obj, Sample):
            raise TypeError(f"Invalid buffer {type(obj)}.")
        self._data[self._i] = obj
        self._i = (self._i + 1) % self._size

    def __getitem__(self, level: int):
        if level >= self._size or level < 0:
            raise IndexError(f"Level {level} out of range.")
        i = (self._i - 1 - level) % self._size
        return self._data[i]

    def __len__(self):
        return self._size
