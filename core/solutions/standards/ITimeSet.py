# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for time set.
"""
from core.solutions.standards.ITime import ITime

from abc import ABC, abstractmethod
from typing import List


class ITimeSet(ABC):
    @abstractmethod
    def get_times(self) -> List[ITime]:
        """
        Gets the list of `ITime` elements in this timeset.
        """
        pass

    @abstractmethod
    def remove_time(self, index: int) -> None:
        """
        Removes the `ITime` element with specified index.
        """
        pass

    @abstractmethod
    def add_time(self, time: ITime) -> None:
        """
        Adds a `ITime` to this timeset.
        """
        pass

    @abstractmethod
    def has_durations(self) -> bool:
        """Whether each `ITime` have duration.

        In this case, duration value greater than zero is expected for each
        `ITime` in the `get_times()` list.
        """
        pass

    @abstractmethod
    def get_offset_from_utc_in_hours(self) -> float:
        """
        Time zone offset from UTC, expressed in the number of hours.
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> ITime:
        """
        Gets the time horizon of this timeset.
        """
        pass
