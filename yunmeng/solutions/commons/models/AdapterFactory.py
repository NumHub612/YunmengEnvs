# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight adapter registry for exchange items.
"""

from __future__ import annotations
from typing import Optional

from yunmeng.solutions.standards import (
    IExchangeAdapter,
    IInput,
    IOutput,
)


class AdapterFactory:
    """Finds and creates exchange adapters for a given source/target pair."""

    def __init__(self, adapters: list[IExchangeAdapter] = None):
        self._adapters = list(adapters or [])

    def register(self, adapter: IExchangeAdapter):
        self._adapters.append(adapter)

    def find_adapter(self, source: IOutput, target: IInput) -> Optional[IExchangeAdapter]:
        for adapter in self._adapters:
            if adapter.can_adapt(source, target):
                return adapter
        return None

    def create_chain(self, source: IOutput, target: IInput) -> list[IExchangeAdapter]:
        chain = []
        for adapter in self._adapters:
            if adapter.can_adapt(source, target):
                chain.append(adapter)
        return chain

    def __repr__(self) -> str:
        return f"AdapterFactory(adapters={len(self._adapters)})"
