# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

A set of time stamps or time intervals, used to indicate when an output item
has values and can provide values, and when an input item
does or may require values.
"""
from yunmeng.solutions.standards.ITime import ITime

from abc import ABC, abstractmethod


class ITimeSet(ABC):
    """Set of time stamps or time intervals used for data exchange."""

    @property
    @abstractmethod
    def times(self) -> list[ITime]:
        """
        Time stamps or spans as available in of an output item,
        or as required by an input item.

        Specifically, if times size is 0, it means:
        + in case of input, currently no values required.
        + in case of output, no values available or required yet.
        """
        pass

    @abstractmethod
    def remove_time(self, index: int) -> None:
        """
        Removes an `ITime` with given index and updates the duration.
        """
        pass

    @abstractmethod
    def add_time(self, time: ITime) -> None:
        """
        Adds a `ITime` to this timeset and updates the duration.
        """
        pass

    @abstractmethod
    def has_duration(self) -> bool:
        """Whether each `ITime` have duration."""
        pass

    @abstractmethod
    def get_offset_from_utc_in_hours(self) -> float:
        """
        Time zone offset from UTC.
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> ITime:
        """
        Gets the time horizon of this timeset, independent to 'times'.

        For an input item, it will never go backfurther in time than
        the time horizon's begin time: time_horizon.timestamp.
        Also, it will never go further ahead than the time horizon's
        end time:
        time_horizon.timestamp + time_horizon.duration_in_days.

        For an output item, thus for an adapted output, time horizon
        indicates in what time span the item can provide values.
        Specifically, if the time horizon's begin time is -Infinity,
        that means far back in time;
        if this time horizon's end time is +Infinity, that means far
        in the future.
        """
        pass
