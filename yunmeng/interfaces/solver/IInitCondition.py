# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Initial condition protocols.

Initial conditions are the solver layer's FIELD-SEEDING interface:
rules that write the starting state of a field. They are applied
at exactly two lifecycle points — initialize() and reset().

One IC covers the WHOLE field, a field's start state is defined by
exactly ONE IC per-field. Composing several partial rules into one
start field is the CALLER's job, done by building the combined
value function upfront — not the framework's, not at apply() time.

Design invariants:

1. Whole-field coverage. apply() is a full-field assignment.

2. Idempotence. apply() must be safe to run repeatedly (initialize,
   then reset() per estimation trial) and must depend only on
   its construction parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from yunmeng.interfaces.supports import IField


class IInitialCondition(ABC):
    """Field initializer."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of the initial condition."""
        ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def target_field(self) -> str:
        """The name of field to be initialized here."""
        ...

    @abstractmethod
    def apply(self, field: IField):
        """Initializes the target field."""
        ...
