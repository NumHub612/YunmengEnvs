# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors.  Join us !
License: Apache License 2.0

Interface for time-space components.
"""
from abc import ABC, abstractmethod
from typing import Optional

from core.solutions.standards.ITimeSet import ITimeSet
from core.solutions.standards.ITime import ITime


class ITimeExtension(ABC):
    """Methods that are specific for a time-space component."""

    @property
    @abstractmethod
    def time_extent(self) -> Optional[ITimeSet]:
        """
        The property describes in what timespan the component can operate.
        This can be used to support the user when creating a composition.
        """
        pass

    @property
    @abstractmethod
    def curr_time(self) -> Optional[ITime]:
        """The property describes what timestamp the component is at."""
        pass
