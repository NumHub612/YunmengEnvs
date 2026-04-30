# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base model for all linkable components.
"""
from yunmeng.solutions.standards import (
    ILinkableComponent,
    IArgument,
    IInput,
    IOutput,
    IIdentifiable,
    IManageState,
    LinkableComponentStatus,
    LinkableComponentStatusChangeEventArgs,
)
from yunmeng.solutions.commons import events
from yunmeng.solutions.commons.enums import EnvRunMode
from typing import Any


class BaseModel(ILinkableComponent, IManageState):
    """Base model for all linkable components.

    In a typical pull-driven scenario, the component A `update` method would
    call the `values` property of its input items, which in turn calls
    the `get_values` method of the bound output item. This method then calls
    the `update` method of the owner, component B, to update the data,
    and after retrieving the data, it propagates back along this calls chain.

    While in Loop-driven scenario, there is a bidirectional data requirement
    between coupled components, and data exchange would occur back and forth
    multiple times within the same step (external loop or iterative loop)
    until the next time step is reached or the convergence is achieved.
    """

    def __init__(self, id: str):
        self._id = id
        self._arguments: list[IArgument] = []
        self._inputs: list[IInput] = []
        self._outputs: list[IOutput] = []

        self._run_mode = EnvRunMode.DEVELOP
        self._cascading = False
        self._status = LinkableComponentStatus.CREATED
        self._event_manager = events.EventManager()

    @property
    def status(self) -> LinkableComponentStatus:
        return self._status

    @property
    def arguments(self) -> list[IArgument]:
        return self._arguments

    @property
    def outputs(self) -> list[IOutput]:
        return self._outputs

    @property
    def inputs(self) -> list[IInput]:
        return self._inputs

    @property
    def CascadingUpdate(self) -> bool:
        return self._cascading

    @CascadingUpdate.setter
    def CascadingUpdate(self, cascading: bool):
        self._cascading = cascading

    def initialize(self):
        raise NotImplementedError()

    def validate(self) -> list[str]:
        raise NotImplementedError()

    def prepare(self):
        raise NotImplementedError()

    def update(self, required_outputs: list[IOutput]):
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()

    def set_status(self, status: LinkableComponentStatus, message: str):
        event_args = LinkableComponentStatusChangeEventArgs(
            self, message, self._status, status
        )
        self._status = status
        self._event_manager.invoke(event_args)

    def load_state(self, path: str) -> IIdentifiable:
        raise NotImplementedError()

    def keep_current_state(self) -> IIdentifiable:
        raise NotImplementedError()

    def restore_state(self, state_id: IIdentifiable):
        raise NotImplementedError()

    def clear_state(self, state_id: IIdentifiable):
        raise NotImplementedError()

    def save_state(self, state_id: IIdentifiable, path: str):
        raise NotImplementedError()
