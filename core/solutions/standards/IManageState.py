# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for managing state of linkable components.
"""
from core.solutions.standards.IIdentifiable import IIdentifiable

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
        """
        pass

    @abstractmethod
    def restore_state(self, state_id: IIdentifiable):
        """
        Restores the state identified by the state_id.
        """
        pass

    @abstractmethod
    def clear_state(self, state_id: IIdentifiable):
        """
        Clears a specified state from the linkable component.
        """
        pass
