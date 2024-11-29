# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Addtional field manipulation functions.
"""
from core.numerics.fields import Field, Vector, Scalar, Tensor


def split_field(field: Field) -> tuple:
    """
    Split the field into its components.

    Notes:
        - Scalar field would just be returned;
        - Vector field would be split into 3 scalar fields;
        - Tensor field would be split into 3 vector fields.
    """
    if field.dtype == "scalar":
        return field

    if field.dtype == "vector":
        values = field.to_np()
        return (
            Field.from_np(values[:, 0], field.etype),
            Field.from_np(values[:, 1], field.etype),
            Field.from_np(values[:, 2], field.etype),
        )

    if field.dtype == "tensor":
        values = field.to_np()
        return (
            Field.from_np(values[:, 0:3], field.etype),
            Field.from_np(values[:, 3:6], field.etype),
            Field.from_np(values[:, 6:9], field.etype),
        )
