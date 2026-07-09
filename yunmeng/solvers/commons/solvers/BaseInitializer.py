# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Baseic initialization condition class.
"""

from yunmeng.solvers.interfaces import IInitialCondition
from yunmeng.numerics.fields import Variable, Field

import numpy as np


class BaseInitializer(IInitialCondition):
    """
    Abstract base class for initailization conditions.
    """

    def __init__(self, id: str, target_field: str):
        """
        Args:
            id: Unique identifier.
            target_field: Name of the field to be initalized (e.g., "u").
        """
        self._id = id
        self._target_field = target_field

    @property
    def target_field(self) -> str:
        return self._target_field

    @property
    def id(self) -> str:
        return self._id
