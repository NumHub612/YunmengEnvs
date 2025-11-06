# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base model for all linkable components.
"""
from core.solutions.standards import (
    ILinkableComponent,
    IArgument,
    IInput,
    IOutput,
    IUnit,
    IIdentifiable,
    IManageState,
    LinkableComponentStatus,
    LinkableComponentStatusChangeEventArgs,
)
from core.solutions.commons import events
from configs.settings import logger


class BaseModel(ILinkableComponent, IManageState):
    """Base model for all linkable components.

    In a typical pull-driven scenario, the component A `update` method would
    call the `values` property of its input items, which in turn calls
    the `get_values` method of the bound output item. This method then calls
    the `update` method of the owner, component B, to update the data,
    and after retrieving the data, it propagates back along this calls chain.

    After the component is instantiated, `setup` method can provide dynamic
    generation of inputs and outputs, but it requires freezing the link framework
    after the `initialize` method is called.

    Typically, the external scheduler should traverse the inputs/outputs
    properties after the component `setup` to complete the consumer/provider
    relationship binding.
    """

    def __init__(self):
        self._arguments: dict[str, IArgument] = {}
        self._inputs: list[IInput] = []
        self._outputs: list[IOutput] = []
        self._states: dict[str, any] = {}
        self._CascadingDisabled = False
        self._status = LinkableComponentStatus.CREATED
        self._event_manager = events.EventManager()

    @property
    def arguments(self) -> dict[str, IArgument]:
        return self._arguments

    @property
    def status(self) -> LinkableComponentStatus:
        return self._status

    @property
    def inputs(self) -> list[IInput]:
        return self._inputs

    @property
    def outputs(self) -> list[IOutput]:
        return self._outputs

    @property
    def states(self) -> dict[str, any]:
        return self._states

    @property
    def CascadingUpdateCallsDisabled(self) -> bool:
        return self._CascadingDisabled

    def setup(
        self,
        inputs_config: list = None,
        outputs_config: list = None,
        args_config: list = None,
        **kwargs,
    ):
        raise NotImplementedError()

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

    def keep_current_state(self) -> IIdentifiable:
        raise NotImplementedError()

    def restore_state(self, state_id: IIdentifiable):
        raise NotImplementedError()

    def clear_state(self, state_id: IIdentifiable):
        raise NotImplementedError()

    def save_state(self, state_id: IIdentifiable, path: str):
        raise NotImplementedError()

    def load_state(self, path: str) -> IIdentifiable:
        raise NotImplementedError()

    def set_status(self, status: LinkableComponentStatus, message: str):
        """设置状态"""
        old_status = self._status
        self._status = status
        self.notify_status_changed(old_status, status, message)

    def notify_status_changed(
        self,
        old_status: LinkableComponentStatus,
        new_status: LinkableComponentStatus,
        message: str,
    ):
        """Notifies subscribers that the status has changed."""
        logger.info(
            f"Component {self} changed from {old_status} to {new_status}: {message}"
        )
        event_args = LinkableComponentStatusChangeEventArgs(
            self, message, old_status, new_status
        )
        self._event_manager.invoke(event_args)
