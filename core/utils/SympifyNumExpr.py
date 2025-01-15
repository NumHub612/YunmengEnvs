# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Tool to convert a string expression to a sympy function.
"""
import sympy as sp
from sympy.utilities.lambdify import lambdify
from typing import Callable


def lambdify_numexpr(expr: str, symbols: list) -> Callable:
    """
    Convert a string expression to a lambda function.

    Args:
        expr: A string expression.
        symbols: List of symbols used in the `expr`.

    Returns:
        A lambda function.
    """
    return lambdify(
        symbols, sp.sympify(expr, locals=dict(zip(symbols, sp.symbols(symbols))))
    )
