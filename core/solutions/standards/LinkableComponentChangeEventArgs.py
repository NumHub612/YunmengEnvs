# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
 
Interface for arguments of linkable component status change event.
"""
from abc import ABC, abstractmethod

from core.solutions.standards.ILinkableComponent import ILinkableComponent
from core.solutions.standards.LinkableComponentStatus import LinkableComponentStatus


class LinkableComponentStatusChangeEventArgs(ABC):
    """Interface class for arguments of linkable component status change event."""

    @property
    @abstractmethod
    def linkable_component(self) -> ILinkableComponent:
        """The linkable component."""
        pass

    @linkable_component.setter
    @abstractmethod
    def linkable_component(self, obj: ILinkableComponent):
        """Sets the linkable component."""
        pass

    @property
    @abstractmethod
    def message(self) -> str:
        """The message for the event."""
        pass

    @message.setter
    @abstractmethod
    def message(self, msg: str) -> None:
        """Sets the message."""
        pass

    @property
    @abstractmethod
    def old_status(self) -> LinkableComponentStatus:
        """The old status of the linkable component."""
        pass

    @old_status.setter
    @abstractmethod
    def old_status(self, value: LinkableComponentStatus):
        """Sets the old status."""
        pass

    @property
    @abstractmethod
    def new_status(self) -> LinkableComponentStatus:
        """The new status of the linkable component."""
        pass

    @new_status.setter
    @abstractmethod
    def new_status(self, value: LinkableComponentStatus):
        """Sets the new status."""
        pass
