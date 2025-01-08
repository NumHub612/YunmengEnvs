# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Solver for user customized pde equations.
"""
from core.solvers.interfaces import IEquation, IOperator
from core.solvers.fdm.operators import fdm_operators

from core.solvers.commons import inits, boundaries, callbacks, BaseSolver
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import Variable, Scalar, Vector, Tensor
from core.numerics.fields import Field, NodeField
from core.numerics.matrix import LinearEqs, Matrix

import re


class CustomSolver(BaseSolver):
    def __init__(self, id: str, mesh: Mesh):
        super().__init__(id, mesh)
        self._equations = []

    def set_problems(self, equations: list[IEquation]):
        self._equations = equations

    def initialize(self, **kwargs):
        pass

    def inference(self, **kwargs):
        pass


class BaseEquation(IEquation):
    """Base class for user customized pde equations.

    Notes:
        - Don't support nested operators; if necessary, use intermediate variables.
        - Don't use float numbers directly, define coefficients for them.
        - Mark elementary function under `func` flag.
    """

    def __init__(self, name: str, operators: dict[str, IOperator]):
        self._name = name
        self._mesh = None
        self._operators = operators
        self._coefficients = {}
        self._fields = {}

        self._equations = []
        self._symbols = {}

        self._defualt_ops = {}

    def set_equations(self, equations: list, symbols: dict):
        self._equations = equations
        self._symbols.update(symbols)

    def set_coefficients(self, coefficients: dict):
        self._coefficients.update(coefficients)

    def set_fields(self, fields: dict):
        self._fields.update(fields)

    def set_mesh(self, mesh):
        self._mesh = mesh

    def get_variables(self) -> dict:
        vars = {}
        for key, value in self._symbols.items():
            if not value.get("coefficient", False):
                vars[key] = value
        return vars

    def discretize(self) -> LinearEqs:
        raise NotImplementedError

    def parse_equation(self, equation: str) -> list[str]:
        """
        Parse the equation string into terms with sign.

        Args:
            equation: The equation string.

        Returns:
            A list of terms with sign.
        """
        terms = []

        in_brackets, in_right = False, False
        term = ""
        for char in equation.replace(" ", "") + "&":
            if char == "(":
                if in_brackets:
                    raise ValueError("Don't support nested brackets yet")
                in_brackets = True
            if char == ")":
                in_brackets = False

            if char in ["+", "-", "=", "&"] and not in_brackets:
                if char == "=" and in_right:
                    continue

                if in_right and term:
                    term = self._to_negative(term)
                if term:
                    terms.append(term)
                if char == "=":
                    in_right = True

                term = char if char == "-" else ""
                continue

            term += char
        return terms

    def parse_term(self, token: str) -> list[dict]:
        """
        Parse the equation term.

        Args:
            token: The equation term.

        Returns:
            A list of dictionaries describing the operators and variables.
        """
        parts = []

        in_brackets = False
        part = ""
        for char in token + "&":
            if char == "(":
                in_brackets = True
            if char == ")":
                in_brackets = False

            if char in ["*", "/", "+", "-", "&"] and not in_brackets:
                if part:
                    parts.append(part)
                if char != "&":
                    parts.append(char)

                part = ""
                continue

            part += char

        terms = []
        for it in parts:
            if self.is_variable(it):
                var = self.parse_variable(it)
                terms.append(var)
            elif self.is_operator(it):
                op = self.parse_operator(it)
                terms.append(op)
            elif self.is_coefficient(it):
                coe = self.parse_coefficient(it)
                terms.append(coe)
            else:
                op = {"type": None, "content": it}
                terms.append(op)

        return terms

    def _to_negative(self, term: str) -> str:
        """Convert a term into negative."""
        if not term or term is None:
            return ""
        if term[0] == "-":
            return term[1:]
        if term[0] == "+":
            return f"-{term[1:]}"
        return f"-{term}"

    def is_operator(self, token: str) -> bool:
        """
        Check if the token is an operator.
        """
        parts = token.strip().split("(")
        if len(parts) == 1:
            return False

        ops = parts[0].split("::")
        return ops[0].upper() in self.SUPPORTED_OPS

    def parse_operator(self, token: str) -> dict:
        """
        Parse the operator name.

        Notes:
            - The operator should be in the format of `op_type::op_name`.
            - The operator arguments expression can't contain `()`.
        """
        operator, operand = token.split("(")
        operand = operand.replace(")", "")

        ops = operator.split("::")
        if len(ops) == 1:
            op_type = ops[0].upper()
            op_name = self._defualt_ops.get(op_type)
        else:
            op_type = ops[0].upper()
            op_name = ops[1]

        if op_type != "FUNC" and op_name not in self._operators:
            raise ValueError(f"Unsupported operator: {token}.")

        args = self.parse_term(operand)
        return {
            "type": "operator",
            "operator_type": op_type,
            "operator_name": op_name,
            "args": args,
        }

    def is_variable(self, token: str) -> bool:
        """
        Check if the token is a variable.
        """
        return token in self.get_variables()

    def parse_variable(self, token: str) -> dict:
        """
        Parse the variable name.
        """
        if token not in self._fields:
            raise ValueError(f"Un-initialized variable: {token}.")
        return {
            "type": "variable",
            "variable_name": token,
            "variable_field": self._fields.get(token),
        }

    def is_coefficient(self, token: str) -> bool:
        """
        Check if the token is a coefficient.
        """
        return token in self._coefficients

    def parse_coefficient(self, token: str) -> dict:
        """
        Parse the coefficient value.
        """
        coeff_value = self._coefficients.get(token)
        if coeff_value is None:
            raise ValueError(f"Un-setted coefficient: {token}.")
        return {
            "type": "coefficient",
            "coefficient_name": token,
            "coefficient_value": coeff_value,
        }


class SimpleEquation(BaseEquation):
    def __init__(self, name: str, operators: dict[str, IOperator]):
        super().__init__(name, operators)
        self._defualt_ops = {
            "GRAD": "grad01",
            "LAPLACIAN": "lap02",
            "DDT": "ddt01",
            "D2DT2": "d2dt2",
            "FUNC": "func",
            "DIV": "div03",
            "CURL": "curl04",
        }
        self._operators = {
            "grad01": Operator("grad01", {"u": "vector"}),
            "lap02": Operator("lap02", {"u": "vector"}),
            "ddt01": Operator("ddt01", {"u": "vector"}),
            "d2dt2": Operator("d2dt2", {"u": "vector"}),
            "func": Operator("func", {"u": "vector"}),
            "div03": Operator("div03", {"u": "vector"}),
            "curl04": Operator("curl04", {"u": "vector"}),
        }

    def discretize(self, **kwargs) -> list[LinearEqs]:
        pass


class Operator(IOperator):
    def __init__(self, name: str, variables: dict):
        self._name = name
        self._variables = variables

    @property
    def type(self) -> str:
        return "custom"

    @property
    def scheme(self) -> str:
        return "custom"

    def prepare(self, mesh: Mesh, coefficents: dict):
        pass

    def run(self, element: int, neighbors: list[int], **kwargs) -> Variable | LinearEqs:
        pass


if __name__ == "__main__":
    foo = SimpleEquation("foo", fdm_operators)
    foo.set_equations(
        [
            "k*u*grad(rho*u+p) - ddt::ddt01(u) + nu == -nu*laplacian(u)*div(u) + curl(p) + k"
        ],
        {
            "k": {
                "description": "diffusion coefficient",
                "type": "vector",
                "coefficient": True,
                "bounds": (0, None),
            },
            "u": {
                "description": "velocity",
                "type": "vector",
                "coefficient": False,
                "bounds": (None, None),
            },
            "rho": {
                "description": "density",
                "type": "scalar",
                "coefficient": True,
                "bounds": (0, None),
            },
            "p": {
                "description": "pressure",
                "type": "vector",
                "coefficient": False,
                "bounds": (0, None),
            },
            "nu": {
                "description": "viscosity",
                "type": "scalar",
                "coefficient": True,
                "bounds": (0, None),
            },
        },
    )
    foo.set_coefficients(
        {
            "k": Vector(1.0, 1.0),
            "nu": Scalar(0.1),
            "rho": Scalar(1.0),
        }
    )
    foo.set_fields(
        {
            "u": NodeField(100, "vector"),
            "p": NodeField(100, "vector"),
        }
    )

    result = foo.parse_equation(
        "k*u*grad(rho*u+p) - ddt::ddt01(u) + nu == -nu*laplacian(u)*div(u) + curl(p) + k"
    )
    print(result)

    vars = foo.get_variables()
    print(vars)

    for term in result:
        print(f"\nTerm: {term}, splited: \n{foo.parse_term(term)}")

    equation_expr = "ddt::default(u) + u*grad(u) == nu*laplacian(u)"
    variables = {
        "u": {"description": "velocity", "type": "vector", "bounds": (None, None)},
        "nu": {"description": "viscosity", "type": "scalar", "bounds": (0, None)},
    }
    nu = 0.1
    equation = BaseEquation("burgers2d")
