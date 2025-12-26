# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface class for linkable modules connection and data transfer.
"""
from core.solutions.standards.LinkableComponentStatus import LinkableComponentStatus
from core.solutions.standards.IIdentifiable import IIdentifiable
from core.solutions.standards.IArgument import IArgument
from core.solutions.standards.IInput import IInput
from core.solutions.standards.IOutput import IOutput

from abc import abstractmethod
from typing import Any


class ILinkableComponent(IIdentifiable):
    """class for linkable modules connection and data transfer."""

    @property
    @abstractmethod
    def arguments(self) -> list[IArgument]:
        """Arguments of the component."""
        pass

    @property
    @abstractmethod
    def status(self) -> LinkableComponentStatus:
        """The status of the component."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> list[IInput]:
        """The input items."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> list[IOutput]:
        """The output items."""
        pass

    @property
    @abstractmethod
    def CascadingUpdate(self) -> bool:
        """The flag to disable cascading update calls."""
        pass

    @CascadingUpdate.setter
    @abstractmethod
    def CascadingUpdate(self, value: bool):
        pass

    @abstractmethod
    def initialize(self):
        """Initializes the component.

        The `Initialize()` will and must be invoked before any other
        methods in the `ILinkableComponent` interface is invoked
        or accessed.

        Immediatly after the method is been invoked, it changes the
        linkable component's status to `INITIALIZING`. If component
        initializes succesfully, it changed to `INITIALIZED`.
        """
        pass

    @abstractmethod
    def validate(self) -> list[str]:
        """Validates the populated instance of the component.

        The method will be invoked after various provider-consumer
        relations between these components' exchange items.

        Immediatly after this method is invoked, it changes
        the component's status to `VALIDATING`. When the method has
        finished, the status of the component has changed to either
        `VALID` or `INVALID`.

        If there are any issues while validating the component,
        the method returns list of messages describing these issues.
        """
        pass

    @abstractmethod
    def prepare(self):
        """Prepares the component for calls to the `update()`.

        Before `prepare()` is called, the component aren't required
        to honor any type of action that retrieves values from
        the component. After `prepare()` is called,
        the component must be ready for providing values.

        Immediatly after the method invoked, it changes
        the component's status to `PREPARING`. When the method has
        finished, the status of the component has changed to
        either `UPDATED` or `FAILED`.
        """
        pass

    @abstractmethod
    def update(self, required_outputs: list[IOutput]):
        """Updates the component to next status.

        Immediately after the method is invoked, it changes the
        component's status to `UPDATING`. If the method's performed
        succesfully, the component'll set its status to `UPDATED`,
        unless after this update action is at the end of its
        computation, in which case it'll set its status to `DONE`.

        According to the 'pull-driven' approach,
        linkable components can be connected in a chain,
        where invoking `update()` method on the last component in
        the chain trigger the entire stack of data exchange.

        The type of actions a component takes
        during `update()` method depends on the type of component.
        A numerical model that progresses in time will typically
        compute a timestep. A database will typically look at the
        consumers of its outputs, and perform one or more queries
        to be able to provide the values that are required.
        """
        pass

    @abstractmethod
    def finish(self):
        """Finishes the component computation, restart it if needed.

        This method is and must be invoked as the last of any
        methods in the `ILinkableComponent` interfaces.

        Immediatly after this method is invoked, it changes the
        component status to `FINISHING`.
        Once the finishing is completed, the component changes
        status to `FINISHED` if can't be restarted, or to `CREATED`.
        """
        pass
