# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Lightweight coupling strategy executor.
"""

from __future__ import annotations

from yunmeng.solutions.standards import (
    ILinkableModel,
    ICouplingStrategy,
    CouplingConfig,
    IterationResult,
)


class LoopController:
    """Holds a coupling strategy and delegates execution."""

    def __init__(self, strategy: ICouplingStrategy = None):
        self._strategy = strategy

    @property
    def strategy(self) -> ICouplingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ICouplingStrategy):
        self._strategy = strategy

    def execute(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig = None,
    ) -> IterationResult:
        if self._strategy is None:
            return IterationResult(
                converged=True, iterations=0, residual=0.0, message="no strategy"
            )
        config = config or CouplingConfig()
        return self._strategy.execute(source, target, config)

    def __call__(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig = None,
    ) -> IterationResult:
        return self.execute(source, target, config)
