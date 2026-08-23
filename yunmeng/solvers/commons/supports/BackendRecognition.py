# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Recognize the backend type.
"""

from yunmeng.numerics.fields import Field, Backend, get_backend


def backend_of_field(field: Field) -> Backend:
    return get_backend(field.meta.btype)
