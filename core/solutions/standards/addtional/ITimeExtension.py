# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors.  Join us !
License: Apache License 2.0

Interface for time-space components.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ..ITimeSet import ITimeSet
from ..ITime import ITime


class ITimeExtension(ABC):
    """Methods that are specific for a time-space component."""

    @abstractmethod
    def get_time_extent(self) -> Optional[ITimeSet]:
        """
        The property describes in what timespan the component can operate.
        This can be used to support the user when creating a composition.
        """
        pass

    @abstractmethod
    def get_curr_time(self) -> Optional[ITime]:
        """The property describes what timestamp the component is at."""
        pass
