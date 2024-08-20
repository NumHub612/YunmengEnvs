# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for entities that can be described.
"""
from abc import ABC, abstractmethod


class IDescribable(ABC):
    """To provide descriptive information on an OpenOasis entity."""

    @abstractmethod
    def get_caption(self) -> str:
        """Caption string."""
        pass

    @abstractmethod
    def set_caption(self, value: str) -> None:
        """Set the caption string."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Additional descriptive information about the entity."""
        pass

    @abstractmethod
    def set_description(self, value: str) -> None:
        """Set the description string."""
        pass
