# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for managing state of linkable components.
"""
from abc import ABC, abstractmethod
from typing import Optional

from core.solutions.standards.IIdentifiable import IIdentifiable


class IManageState(ABC):
    """
    Provides additional methods for handling component state
    so it can be saved, restored and cleared.
    """

    @abstractmethod
    def keep_current_state(self) -> Optional[IIdentifiable]:
        """
        Stores the linkable component's current state.
        """
        pass

    @abstractmethod
    def restore_state(self, state_id: Optional[IIdentifiable]):
        """
        Restores the state identified by the state_id.
        """
        pass

    @abstractmethod
    def clear_state(self, state_id: Optional[IIdentifiable]):
        """
        Clears a specified state from the linkable component.
        """
        pass
