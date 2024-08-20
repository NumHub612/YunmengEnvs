# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for managing state of linkable components.
"""
from abc import ABC, abstractmethod
from typing import Optional

from .IIdentifiable import IIdentifiable


class IManageState(ABC):
    """
    It provides additional methods for handling component state so it can be
    saved, restored and cleared.

    An optional interface to be implemented by components in addition to the
    `ILinkableComponent` interface. It can be left completely to the component to handle
    persistence of state or it can also implement `IByteStateConverter` and provide
    ways for state to be converted to and from an array of bytes.
    """

    @abstractmethod
    def keep_current_state(self) -> Optional[IIdentifiable]:
        """
        Stores the linkable component's current state to a snapshot.
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
        Clears a state from the linkable component.
        """
        pass
