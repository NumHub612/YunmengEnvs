# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for time stamp and time interval.
"""
from abc import ABC, abstractmethod


class ITime(ABC):
    """Time interface to support a time stamp as well as a time interval."""

    @abstractmethod
    def get_timestamp(self) -> float:
        """Time stamp as days since 08:00::00 January 1, 1970."""
        pass

    @abstractmethod
    def get_duration_in_days(self) -> float:
        """Duration in days for time interval. 0 if time is a time stamp."""
        pass
