# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  
 
Interface class for linkable modules connection and data transfer.
"""
from core.solutions.standards.LinkableComponentStatus import LinkableComponentStatus
from core.solutions.standards.IAdaptedOutputFactory import IAdaptedOutputFactory
from core.solutions.standards.IIdentifiable import IIdentifiable
from core.solutions.standards.IArgument import IArgument
from core.solutions.standards.IInput import IInput
from core.solutions.standards.IOutput import IOutput

from abc import abstractmethod
from typing import List


class ILinkableComponent(IIdentifiable):
    """class for linkable modules connection and data transfer."""

    @abstractmethod
    def has_attribute(self, attribute_name: str) -> bool:
        """Checks if the component has the specified attribute.

        Args:
            attribute_name: The name of the attribute to check.

        Returns:
            bool: True if the component has the attribute, False otherwise.
        """
        pass

    @abstractmethod
    def get_attribute(self, attribute_name: str) -> object:
        """Gets the value of the specified attribute.

        Args:
            attribute_name: The name of the attribute to get.

        Returns:
            object: The value of the attribute.
        """
        pass

    @abstractmethod
    def set_attribute(self, attribute_name: str, attribute_value: object):
        """Sets the value of the specified attribute.

        Args:
            attribute_name: The name of the attribute to set.
            attribute_value: The value of the attribute to set.
        """
        pass

    @abstractmethod
    def has_state(self, state_name: str, **kwargs) -> bool:
        """Checks if the component has the specified state.

        Args:
            state_name: The name of the state to check.

        Returns:
            bool: True if the component has the state, False otherwise.
        """
        pass

    @abstractmethod
    def get_state(self, state_name: str, **kwargs) -> object:
        """Gets the value of the specified state.

        Args:
            state_name: The name of the state to get.

        Returns:
            object: The value of the state.
        """
        pass

    @abstractmethod
    def set_state(self, state_name: str, state_value: object, **kwargs):
        """Sets the value of the specified state.

        Args:
            state_name: The name of the state to set.
            state_value: The value of the state to set.
        """
        pass

    @abstractmethod
    def has_data(self, data_name: str, **kwargs) -> bool:
        """Checks if the component has the specified data.

        Args:
            data_name: The name of the data to check.

        Returns:
            bool: True if the component has the data, False otherwise.
        """
        pass

    @abstractmethod
    def get_data(self, data_name: str, **kwargs) -> object:
        """Gets the value of the specified data.

        Args:
            data_name: The name of the data to get.

        Returns:
            object: The value of the data.
        """
        pass

    @abstractmethod
    def has_element(self, element_name: str, **kwargs) -> bool:
        """Checks if the component has the specified element.

        Args:
            element_name: The name of the element to check.

        Returns:
            bool: True if the component has the element, False otherwise.
        """
        pass

    @abstractmethod
    def initialize(self):
        """Initializes the component.

        The `Initialize()` will and must be invoked before any other methods
        in the `ILinkableComponent` interface is invoked
        or accessed, except for the `GetArguments`.

        Immediatly after the method is been invoked, it changes the linkable
        component's status to `Initializing`. If component initializes
        succesfully, the status is changed to `Initialized`.
        """
        pass

    @abstractmethod
    def validate(self) -> List[str]:
        """Validates the populated instance of the component.

        The method will must be invoked after various provider/consumer relations
        between this component's exchange items and the exchange items of
        other components present in the composition.

        Immediatly after this method is invoked, it changes the component's
        status to `Validating`. When the method has finished, the status
        of the component has changed to either `Valid` or `Invalid`.

        If there are messages while components status is `Valid`, the messages
        are purely informative. If there're messages while components
        status is `Invalid`, at least one of the messages
        indicates a fatal error.
        """
        pass

    @abstractmethod
    def prepare(self):
        """Prepares the component for calls to the `Update()`.

        Before `Prepare()` is called, the component are not required to honor any
        type of action that retrieves values from the component.
        After `Prepare()` is called, the component must be ready for
        providing values.

        Immediatly after the method is invoked, it changes the component's
        status to `Preparing`. When the method has finished, the status
        of the component has changed to either `Updated` or `Failed`.
        """
        pass

    @abstractmethod
    def update(self):
        """Updates the linkable component itself, thus reaching its next state.

        Immediately after `Update()` is invoked, it changes the component's
        status to `Updating`. If the method is performed succesfully,
        the component sets its status to `Updated`, unless
        after this update action is at the end of its computation,
        in which case it will set its status to `Done`.

        According to the 'pull-driven' approach, linkable components can be
        connected in a chain, where invoking `Update()` method
        on the last component in the chain trigger
        the entire stack of data exchange.

        The type of actions a component takes during the `Update()` method
        depends on the type of component.
        A numerical model that progresses in time will typically compute a time
        step. A database would typically look at the consumers of its
        output items, and perform one or more queries to be able
        to provide the values that the consumers required.
        """
        pass

    @abstractmethod
    def finish(self):
        """Finishes the component computation, and then restart it if needed.

        This method is and must be invoked as the last of any methods in the
        `ILinkableComponent` interfaces.

        Immediatly after this method is invoked, it changes the component's
        status to `Finishing`. Once the finishing is completed,
        the component changes status to `Finished` if it can't be restarted;
        `Created`, otherwise.
        """
        pass

    @property
    @abstractmethod
    def arguments(self) -> List[IArgument]:
        """Arguments of the component."""
        pass

    @property
    @abstractmethod
    def status(self) -> LinkableComponentStatus:
        """The status of the component."""
        pass

    @property
    @abstractmethod
    def input_items(self) -> List[IInput]:
        """The input items."""
        pass

    @property
    @abstractmethod
    def output_items(self) -> List[IOutput]:
        """The output items."""
        pass

    @property
    @abstractmethod
    def adapter_factories(self) -> List[IAdaptedOutputFactory]:
        """The adapted output factories."""
        pass
