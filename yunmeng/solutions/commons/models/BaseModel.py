# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight base implementation of ``ILinkableModel``.
"""

from __future__ import annotations

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
        self._last_error: str = ""

    # -- class-level metadata -----------------------

    @classmethod
    def get_meta(cls) -> ModelMeta:
        return ModelMeta(name=cls.__name__)

    # -- instance properties ------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def callbacks(self) -> list[ICallback]:
        return self._callbacks

    @property
    def inputs(self) -> list[IInput]:
        return self._inputs

    @property
    def outputs(self) -> list[IOutput]:
        return self._outputs

    @property
    def status(self) -> ModelStatus:
        return self._status

    # -- callbacks ----------------------------------

    def add_callback(self, callback: ICallback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ICallback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, event: str, **context):
        for cb in list(self._callbacks):
            cb.on_event(event, self, context)

    # -- ports --------------------------------------

    def get_output(self, port_id: str) -> IOutput:
        for p in self._outputs:
            if p.id == port_id:
                return p
        return None

    def get_input(self, port_id: str) -> IInput:
        for p in self._inputs:
            if p.id == port_id:
                return p
        return None

    def get_port(self, port_id: str):
        """Look up any port by id."""
        return self.get_input(port_id) or self.get_output(port_id)

    def add_input(self, item: IInput):
        self._inputs.append(item)

    def add_output(self, item: IOutput):
        self._outputs.append(item)

    def _new_port_id(self, port_id: str, suffix: str) -> str:
        pid = port_id or f"{self._id}.{suffix}"
        if self.get_port(pid) is not None:
            raise ValueError(f"{self._id}: port '{pid}' exists.")
        return pid

    def create_input(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
        required: bool = True,
    ) -> IInput:
        pid = self._new_port_id(port_id, quantity.name)
        port = BaseInput(
            pid,
            quantity,
            elements or ScalarElementSet(self._id),
            owner=self,
            required=required,
        )
        self.add_input(port)
        return port

    def create_output(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
    ) -> IOutput:
        pid = self._new_port_id(port_id, quantity.name)
        port = BaseOutput(
            pid,
            quantity,
            elements or ScalarElementSet(self._id),
            owner=self,
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
            out.clear_adapters()
            out.clear_consumers()
            self._outputs.remove(out)
            return True
        return False

    # -- lifecycle ----------------------------------

    def initialize(self):
        if self._status == ModelStatus.FAILED:
            raise RuntimeError(
                f"{self._id}: model is FAILED ({self._last_error}); "
                f"call finish() before re-initializing."
            )
        self._fire(CallbackEvent.BEFORE_INITIALIZE)
        try:
            self._do_initialize()
        except Exception as e:
            self._fail(e)
            raise
        self._status = ModelStatus.READY
        self._fire(CallbackEvent.AFTER_INITIALIZE)

    def _do_initialize(self):
        """Subclass hook: build internal structures."""

    def validate(self) -> list[str]:
        return []

    def prepare(self):
        self._fire(CallbackEvent.ON_PREPARE)

    def update(self, inquirers: list[IOutput] = None) -> ModelStatus:
        if self._status in (ModelStatus.DONE, ModelStatus.FAILED):
            return self._status
        self._status = ModelStatus.RUNNING
        self._fire(CallbackEvent.BEFORE_UPDATE)
        try:
            self._do_update(inquirers)
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return self._status
        if self._status == ModelStatus.RUNNING:
            self._status = ModelStatus.READY
        self._fire(CallbackEvent.AFTER_UPDATE)
        return self._status

    def _do_update(self, inquirers: list[IOutput] = None):
        """Subclass hook: advance one step."""

    def finish(self):
        try:
            self._do_finish()
        finally:
            self._fire(CallbackEvent.ON_FINISH)
            self._status = ModelStatus.CREATED
            self._last_error = ""

    def _do_finish(self):
        """Subclass hook: release resources, flush outputs."""

    # -- failure handling ---------------------------

    def _fail(self, exc: Exception):
        self._status = ModelStatus.FAILED
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._fire(
            CallbackEvent.ON_ERROR,
            error=self._last_error,
        )

    def get_last_error(self) -> str:
        return self._last_error

    def mark_done(self):
        self._status = ModelStatus.DONE

    def mark_failed(self, message: str = ""):
        self._status = ModelStatus.FAILED
        if message:
            self._last_error = message
