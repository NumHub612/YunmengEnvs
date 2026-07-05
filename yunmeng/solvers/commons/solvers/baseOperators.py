# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Base implicit/explicit operators for solvers.
"""

from yunmeng.solvers.interfaces import IOperator


class BaseImplicitOperator(IOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BaseExplicitOperator(IOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
