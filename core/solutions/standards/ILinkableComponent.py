# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaced class for linkable modules connection and data transfer.
"""
from abc import abstractmethod
from typing import List, Callable, Optional

from IIdentifiable import IIdentifiable
from IArgument import IArgument
from IInput import IInput
from IOutput import IOutput
from IAdaptedOutputFactory import IAdaptedOutputFactory
from LinkableComponentStatus import LinkableComponentStatus


class ILinkableComponent(IIdentifiable):
    """class for linkable modules connection and data transfer."""

    @abstractmethod
    def get_arguments(self) -> List[Optional[IArgument]]:
        """Gets arguments of the component."""
        pass

    @abstractmethod
    def get_status(self) -> LinkableComponentStatus:
        """Gets the status."""
        pass

    @abstractmethod
    def get_inputs(self) -> List[Optional[IInput]]:
        """Gets the input items."""
        pass

    @abstractmethod
    def get_outputs(self) -> List[Optional[IOutput]]:
        """Gets the output items."""
        pass

    @abstractmethod
    def get_adapter_factories(self) -> List[Optional[IAdaptedOutputFactory]]:
        """Gets the adapted output factories."""
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

    @abstractmethod
    def remove_listener(self, func: Callable):
        """Removes a listener from the component."""
        pass

    @abstractmethod
    def add_listener(self, func: Callable):
        """Adds a listener to the component."""
        pass
