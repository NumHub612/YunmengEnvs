# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for time stamp and time interval.
"""
from abc import ABC, abstractmethod


class ITime(ABC):
    """Time interface to support a time stamp as well as a time interval."""

    @property
    @abstractmethod
    def timestamp(self) -> float:
        """Time stamp."""
        pass

    @property
    @abstractmethod
    def duration_in_days(self) -> float:
        """Duration in days for time interval."""
        pass
