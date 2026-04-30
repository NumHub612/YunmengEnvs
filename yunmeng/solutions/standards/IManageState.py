# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for managing state of linkable components.
"""
from yunmeng.solutions.standards.IIdentifiable import IIdentifiable

from abc import ABC, abstractmethod
from typing import Optional


class IManageState(ABC):
    """
    Provides additional methods for handling component state
    so it can be saved, restored and cleared.
    """

    @abstractmethod
    def keep_current_state(self) -> Optional[IIdentifiable]:
        """
        Stores the linkable component's current state.

        The model state is identified by a unique identifier
        and can be restored later.
        """
        pass

    @abstractmethod
    def restore_state(self, state_id: IIdentifiable):
        """
        Restores the state identified by the state_id.

        After calling this method, the linkable component
        must be in the same state as when it was saved,
        and turned to `Updated` status.
        """
        pass

    @abstractmethod
    def clear_state(self, state_id: IIdentifiable):
        """
        Clears specified state from linkable component.
        """
        pass

    @abstractmethod
    def save_state(self, state_id: IIdentifiable, path: str):
        """
        Persists the specified state of linkable component.
        """
        pass

    @abstractmethod
    def load_state(self, path: str) -> IIdentifiable:
        """
        Loads the components state from persistent storage.
        """
        pass
