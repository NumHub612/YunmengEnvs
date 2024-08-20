# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface provides for object comparison.
"""
from abc import ABC, abstractmethod


class IComparable(ABC):
    """
    Interface for object comparison.
    """

    @abstractmethod
    def compare_to(self, obj: "IComparable") -> int:
        """
        Compare the two comparable objects.

        If this > obj, return +1;
        If this = obj, return 0;
        If this < obj, return -1;

        Parameters:
            obj: A comparable object.

        Returns:
            The comparison result.

        Raises:
            ValueError: If the two objects were not comparable.
        """
        pass
