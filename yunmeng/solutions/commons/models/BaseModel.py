# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight base implementation of ``ILinkableModel``.
"""

from __future__ import annotations
from typing import Any

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    ModelStatus,
    ModelMeta,
)


class BaseModel(ILinkableModel):
    """Concrete-ish base class for a linkable model.

    Subclasses override ``_do_initialize``, ``_do_validate``, ``_do_prepare``,
    ``_do_update`` and ``_do_finish``.
    """

    def __init__(self, model_id: str, meta: ModelMeta = None):
        self._id = model_id
        self._meta = meta or ModelMeta(name=model_id)
        self._status = ModelStatus.CREATED
        self._inputs: list[IInput] = []
        self._outputs: list[IOutput] = []

    @classmethod
    def get_meta(cls) -> ModelMeta:
        return ModelMeta(name=cls.__name__)

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def inputs(self) -> list[IInput]:
        return self._inputs

    @property
    def outputs(self) -> list[IOutput]:
        return self._outputs

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def _set_status(self, status: ModelStatus):
        self._status = status

    def add_input(self, item: IInput):
        self._inputs.append(item)

    def add_output(self, item: IOutput):
        self._outputs.append(item)
        if hasattr(item, "_component"):
            item._component = self

    def initialize(self):
        self._set_status(ModelStatus.READY)

    def validate(self) -> list[str]:
        return []

    def prepare(self):
        pass

    def update(self, required_outputs: list[IOutput] = None) -> ModelStatus:
        if self._status in (ModelStatus.DONE, ModelStatus.FAILED):
            return self._status
        self._set_status(ModelStatus.RUNNING)
        self._do_update(required_outputs)
        if self._status == ModelStatus.RUNNING:
            self._set_status(ModelStatus.READY)
        return self._status

    def _do_update(self, required_outputs: list[IOutput] = None):
        """Override in subclasses."""
        pass

    def finish(self):
        self._set_status(ModelStatus.CREATED)

    def mark_done(self):
        self._set_status(ModelStatus.DONE)

    def mark_failed(self):
        self._set_status(ModelStatus.FAILED)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._id}, {self._status.value})"
