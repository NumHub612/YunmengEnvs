# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

TimeSet used for time-related data.
"""
from core.solutions.standards import ITimeSet, ITime


class TimeSet(ITimeSet):
    """TimeSet used for time-related data."""

    def __init__(
        self, time_horizon: ITime, times: list[ITime] = None, time_offset: float = 0.0
    ):
        """TimeSet.

        Args:
            time_horizon: The time horizon of the time set.
            times: The list of times. Defaults to None.
            time_offset: The offset from UTC in hours.
        """
        self._time_horizon = time_horizon
        self._times = times or []
        self._time_offset = time_offset

    @property
    def times(self) -> list[ITime]:
        return self._times

    @property
    def size(self) -> int:
        """The size of the time set."""
        return len(self._times)

    def is_within_time_horizon(self, time: ITime) -> bool:
        """Check if the given time is within the time horizon."""
        if time.timestamp < self._time_horizon.timestamp:
            return False
        end_time1 = time.timestamp + time.duration_in_hours
        end_time2 = self._time_horizon.timestamp + self._time_horizon.duration_in_hours
        if end_time1 > end_time2:
            return False
        return True

    def is_within_time_set(self, time: ITime) -> bool:
        """Check if the given time is within the time set."""
        start_time = self._times[0].timestamp
        end_time = self._times[-1].timestamp + self._times[-1].duration_in_hours

        if time.timestamp < start_time:
            return False
        if time.timestamp + time.duration_in_hours > end_time:
            return False
        return True

    def remove_time(self, index: int) -> None:
        self._times.pop(index)

    def add_time(self, time: ITime) -> None:
        # TODO: check if the time is within the time horizon.
        # TODO: check the order of the times.
        # TODO: check the time offset.
        self._times.append(time)

    def has_duration(self) -> bool:
        return all(time.duration_in_hours != 0.0 for time in self._times)

    def get_offset_from_utc_in_hours(self) -> float:
        return self._time_offset

    def get_time_horizon(self) -> ITime:
        return self._time_horizon
