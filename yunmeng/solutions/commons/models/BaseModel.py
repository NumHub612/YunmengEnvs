# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight base implementation of ``ILinkableModel``.
"""

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    ICallback,
    CallbackEvent,
    ModelStatus,
    ModelMeta,
    Quantity,
    IElementSet,
)
from yunmeng.solutions.commons.datasets import ScalarElementSet
from yunmeng.solutions.commons.models.Input import BaseInput
from yunmeng.solutions.commons.models.Output import BaseOutput


class BaseModel(ILinkableModel):
    """Concrete-ish base class for a linkable model."""

    def __init__(self, model_id: str, meta: ModelMeta = None):
        self._id = model_id
        self._meta = meta or ModelMeta(name=model_id)
        self._status = ModelStatus.CREATED
        self._inputs: list[IInput] = []
        self._outputs: list[IOutput] = []
        self._callbacks: list[ICallback] = []

    # -- class-level metadata -----------------------

    @classmethod
    def get_meta(cls) -> ModelMeta:
        return ModelMeta(name=cls.__name__)

    # -- instance properties ------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def callbacks(self) -> list[ICallback]:
        return self._callbacks

    @property
    def inputs(self) -> list[IInput]:
        return self._inputs

    @property
    def outputs(self) -> list[IOutput]:
        return self._outputs

    # -- assemble -----------------------------------

    def add_callback(self, callback: ICallback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ICallback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, event: str, **context):
        for cb in list(self._callbacks):
            cb.on_event(event, self, context)

    def get_output(self, port_id: str) -> IOutput:
        for p in self._outputs:
            if p.id == port_id:
                return p
        return None

    def add_output(self, item: IOutput):
        self._outputs.append(item)

    def add_input(self, item: IInput):
        self._inputs.append(item)

    def get_input(self, port_id: str) -> IInput:
        for p in self._inputs:
            if p.id == port_id:
                return p
        return None

    def create_input(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
    ) -> IInput:
        port_id = port_id or f"{self._id}.{quantity.name}"
        if self.get_port(port_id) is not None:
            raise ValueError(f"{self._id}: port '{port_id}' already exists.")
        port = BaseInput(
            port_id,
            quantity,
            elements or ScalarElementSet(self._id),
        )
        self.add_input(port)
        return port

    def create_output(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
    ) -> IOutput:
        port_id = port_id or f"{self._id}.{quantity.name}"
        if self.get_port(port_id) is not None:
            raise ValueError(f"{self._id}: port '{port_id}' already exists.")
        port = BaseOutput(
            port_id,
            quantity,
            elements or ScalarElementSet(self._id),
            None,
            self,
        )
        self.add_output(port)
        return port

    def remove_port(self, port_id: str) -> bool:
        inp = self.get_input(port_id)
        if inp is not None:
            inp.provider = None
            self._inputs.remove(inp)
            return True
        out = self.get_output(port_id)
        if out is not None:
            for consumer in list(out.consumers):
                consumer.provider = None
            self._outputs.remove(out)
            return True
        return False

    # -- lifecycle ----------------------------------

    def initialize(self):
        self._fire(CallbackEvent.BEFORE_INITIALIZE)
        self._do_initialize()
        self._status = ModelStatus.READY
        self._fire(CallbackEvent.AFTER_INITIALIZE)

    def _do_initialize(self):
        """Subclass hook: build internal structures.  Runs between the
        BEFORE/AFTER_INITIALIZE events."""
        pass

    def validate(self) -> list[str]:
        return []

    def prepare(self):
        self._fire(CallbackEvent.ON_PREPARE)

    def update(self, required_outputs: list[IOutput] = None) -> ModelStatus:
        if self._status in (ModelStatus.DONE, ModelStatus.FAILED):
            return self._status
        self._status = ModelStatus.RUNNING
        self._fire(CallbackEvent.BEFORE_UPDATE)
        self._do_update(required_outputs)

        if self._status == ModelStatus.RUNNING:
            self._status = ModelStatus.READY
        if self._status != ModelStatus.FAILED:
            self._fire(CallbackEvent.AFTER_UPDATE)
        return self._status

    def _do_update(self, required_outputs: list[IOutput] = None):
        """Subclass hook: advance one step."""
        pass

    def finish(self):
        self._do_finish()
        self._fire(CallbackEvent.ON_FINISH)
        self._status = ModelStatus.CREATED

    def _do_finish(self):
        """Subclass hook: release resources, flush outputs."""
        pass

    def mark_done(self):
        self._status = ModelStatus.DONE

    def mark_failed(self):
        self._status = ModelStatus.FAILED
