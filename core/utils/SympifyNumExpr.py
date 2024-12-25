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


if __name__ == "__main__":
    expr = "x**2 + y**2"
    symbols = ["x", "y"]
    f = lambdify_numexpr(expr, symbols)
    print(f(2, 3))  # Output: 13.0

    """\begin{eqnarray}
    u &=& -\frac{2 \nu}{\phi} \frac{\partial \phi}{\partial x} + 4 \

    \phi &=& \exp \bigg(\frac{-(x-4t)^2}{4 \nu (t+1)} \bigg) + \exp \bigg(\frac{-(x-4t -2 \pi)^2}{4 \nu(t+1)} \bigg)
    \end{eqnarray}"""

    phi = (
        "exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1))) + exp(-(-4*t + x)**2/(4*nu*(t + 1)))"
    )
    phiprime = "-(-8*t + 2*x)*exp(-(-4*t + x)**2/(4*nu*(t + 1)))/(4*nu*(t + 1)) - (-8*t + 2*x - 4*pi)*exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1)))/(4*nu*(t + 1))"
    expr = f"-2 * nu * (({phiprime}) / ({phi})) + 4"
    symbols = ["t", "x", "nu"]
    f = lambdify_numexpr(expr, symbols)
    print(f(1, 4, 3))  # Output: 3.49170664206445
