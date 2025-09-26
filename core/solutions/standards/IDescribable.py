# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for entities that can be described.
"""
from abc import ABC, abstractmethod


class IDescribable(ABC):
    """To provide descriptive information on an OpenOasis entity."""

    @property
    @abstractmethod
    def caption(self) -> str:
        """Caption string."""
        pass

    @caption.setter
    @abstractmethod
    def caption(self, value: str):
        """Sets the caption string."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Additional descriptive."""
        pass

    @description.setter
    @abstractmethod
    def description(self, value: str):
        """Sets the description."""
        pass
