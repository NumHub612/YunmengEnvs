# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initial condition protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from yunmeng.interfaces.support.field import IField
from yunmeng.interfaces.types import Variable


class IInitialCondition(ABC):
    """Field initializer. Unchanged semantics; Field type protocolized."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of the initial condition."""
        ...

    @property
    @abstractmethod
    def target_field(self) -> str:
        """The name of field to which this initial condition applies."""
        ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def get(self, **kwargs) -> Variable: ...

    @abstractmethod
    def apply(self, field: IField):
        """Initializes the target field."""
        ...
