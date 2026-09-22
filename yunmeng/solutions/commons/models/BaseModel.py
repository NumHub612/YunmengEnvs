# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight base implementation of ``ILinkableModel``.
"""

from yunmeng.interfaces.solution import ModelEvent, ModelMeta, ModelStatus, IExchange
from yunmeng.interfaces.solution.IDataset import IElementSet, Quantity
from yunmeng.solutions.commons.dataset import ScalarElementSet
from yunmeng.solutions.commons.models.BaseInput import IInput, BaseInput
from yunmeng.solutions.commons.models.BaseOutput import IOutput, BaseOutput


class BaseModel:
    """Concrete-ish base class for a linkable model."""

    def __init__(self, model_id: str, meta: ModelMeta = None):
        self._id = model_id
        self._meta = meta or ModelMeta(name=model_id)
        self._status = ModelStatus.CREATED
        self._inputs = []
        self._outputs = []
        self._callbacks = []
        self._last_error = ""

    @classmethod
    def get_meta(cls) -> ModelMeta:
        return ModelMeta(name=cls.__name__)

    # -- instance properties ------------------------

    @property
    def id(self):
        return self._id

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def status(self):
        return self._status

    # -- callbacks ----------------------------------

    def add_callback(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _fire(self, event, **context):
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

    def get_port(self, port_id: str) -> IExchange:
        return self.get_input(port_id) or self.get_output(port_id)

    def add_input(self, item):
        self._inputs.append(item)

    def add_output(self, item):
        self._outputs.append(item)

    def _new_port_id(self, port_id, suffix):
        pid = port_id or f"{self._id}.{suffix}"
        if self.get_port(pid) is not None:
            raise ValueError(f"{self._id}: port '{pid}' exists.")
        return pid

    def create_input(
        self, quantity, elements=None, port_id=None, required=True, time_span=None
    ) -> IInput:
        pid = self._new_port_id(port_id, quantity.name)
        port = BaseInput(
            pid,
            quantity,
            elements or ScalarElementSet(self._id),
            time_span=time_span,
            owner=self,
            required=required,
        )
        self.add_input(port)
        return port

    def create_output(
        self, quantity: Quantity, elements=None, port_id=None, time_span=None
    ) -> IOutput:
        pid = self._new_port_id(port_id, quantity.name)
        port = BaseOutput(
            pid,
            quantity,
            elements or ScalarElementSet(self._id),
            time_span=time_span,
            owner=self,
        )
        self.add_output(port)
        return port

    def remove_port(self, port_id: str):
        inp = self.get_input(port_id)
        if inp is not None:
            provider = inp.provider
            if provider is not None:
                provider.remove_consumer(inp)
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
        self._fire(ModelEvent.BEFORE_INITIALIZE)
        try:
            self._do_initialize()
        except Exception as e:
            self._fail(e)
            raise
        self._status = ModelStatus.READY
        self._fire(ModelEvent.AFTER_INITIALIZE)

    def _do_initialize(self):
        """Subclass hook: build internal structures."""

    def validate(self):
        return []

    def prepare(self):
        self._fire(ModelEvent.ON_PREPARE)

    def update(self, inquirers=None):
        if self._status in (ModelStatus.DONE, ModelStatus.FAILED):
            return self._status
        self._status = ModelStatus.RUNNING
        self._fire(ModelEvent.BEFORE_UPDATE)
        try:
            self._do_update(inquirers)
        except Exception as e:
            self._fail(e)
            return self._status
        if self._status == ModelStatus.RUNNING:
            self._status = ModelStatus.READY
        self._fire(ModelEvent.AFTER_UPDATE)
        return self._status

    def _do_update(self, inquirers=None):
        """Subclass hook: advance one step."""

    def finish(self):
        try:
            self._do_finish()
        finally:
            self._fire(ModelEvent.ON_FINISH)
            self._status = ModelStatus.CREATED
            self._last_error = ""

    def _do_finish(self):
        """Subclass hook: release resources, flush outputs."""

    # -- failure handling ---------------------------

    def _fail(self, exc):
        self._status = ModelStatus.FAILED
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._fire(ModelEvent.ON_ERROR, error=self._last_error)

    def get_last_error(self):
        return self._last_error

    def mark_done(self):
        self._status = ModelStatus.DONE

    def mark_failed(self, message=""):
        self._status = ModelStatus.FAILED
        if message:
            self._last_error = message
