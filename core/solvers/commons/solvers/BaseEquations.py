# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Basic and simple equation class for user customized pde equations.
"""
from core.solvers.interfaces import IEquation, IOperator
from core.numerics.fields import Variable, Field
from core.numerics.matrix import LinearEqs
from core.utils.SympifyNumExpr import lambdify_numexpr

import numpy as np


class BaseEquation(IEquation):
    """Base class for user customized pde equations.

    Notes:
        - Don't support nested brackets.
        - Don't support nested operators; if necessary, use intermediate variables.
        - Don't use float directly(except `0`), define coefficient for them.
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
                if term == "0":
                    continue

                if in_right and term:
                    if term[0] == "-":
                        term = term[1:]
                    elif term[0] == "+":
                        term = f"-{term[1:]}"
                    else:
                        term = f"-{term}"

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
                op = {"type": None, "name": it}
                terms.append(op)

        return terms

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
        if token not in self._symbols:
            raise ValueError(f"Un-defined variable: {token}.")
        return {
            "type": "variable",
            "name": token,
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
        if token not in self._symbols:
            raise ValueError(f"Un-defined coefficient: {token}.")
        return {
            "type": "coefficient",
            "name": token,
        }


class SimpleEquation(BaseEquation):
    """Simple single pde equation class.

    Notes:
        - Only support one variable and one equation.
        - Only support fixed grid mesh.
    """

    def __init__(self, name: str, operators: dict[str, IOperator]):
        super().__init__(name, operators)
        self._defualt_ops = {
            "GRAD": "Grad01",
            "LAPLACIAN": "Lap01",
            "DDT": "Ddt01",
            "D2DT2": "D2dt01",
            "SRC": "Src01",
            "DIV": "Div01",
            "CURL": "Curl01",
        }

        self._op_terms = None
        self._configs = {}

    def _get_equation_info(self):
        """
        Get the equation information, variable name and type,
        and total number of equations.
        """
        var_name = list(self.get_variables().keys())[0]
        var = self._symbols.get(var_name)
        var_type = var.get("type")
        eq_num = self._mesh.node_count
        return var_name, var_type, eq_num

    def discretize(self, dt: float) -> LinearEqs:
        # update configurations
        self._configs["dt"] = dt

        # parse the equation
        if self._op_terms is None:
            eq_terms = self.parse_equation(self._equations[0])
            self._op_terms = []
            for it in eq_terms:
                self._op_terms.append(self.parse_term(it))

        # discretization
        var_name, var_type, eq_num = self._get_equation_info()
        final_eq = LinearEqs.zeros(var_name, eq_num, var_type)
        for terms in self._op_terms:
            op, term_result = "+", None
            for it in terms:
                curr = None
                if it["type"] is None:
                    op = it["name"]
                    continue
                elif it["type"] == "operator":
                    if it["operator_type"] != "FUNC":
                        curr = self.run_operator(it["operator_name"], it["args"])
                    else:
                        curr = self.run_func(it["operator_name"], it["args"])
                elif it["type"] == "coefficient":
                    curr = self._coefficients.get(it["name"])
                elif it["type"] == "variable":
                    curr = self._fields.get(it["name"])
                else:
                    raise ValueError(f"Unsupported term: {it}.")

                if op and term_result:
                    term_result = self.operate(op, term_result, curr)
                elif op:
                    term_result = self.operate(op, None, curr)
                else:
                    term_result = curr
                op = ""
            if term_result:
                final_eq += term_result
        return final_eq

    def operate(self, op: str, left, right):
        """Run the operator on the left and right operands."""
        if left is None:
            return -right if op == "-" else right

        if op == "*":
            if isinstance(left, Field) and isinstance(right, LinearEqs):
                rhs = [left[i] * right.rhs[i] for i in range(left.size)]
                rhs = np.array(rhs)
                return LinearEqs(right.variable, right.matrix, rhs)
            elif isinstance(right, Field) and isinstance(left, LinearEqs):
                rhs = [left.rhs[i] * right[i] for i in range(right.size)]
                rhs = np.array(rhs)
                return LinearEqs(left.variable, left.matrix, rhs)
            elif isinstance(left, Variable) and isinstance(right, LinearEqs):
                rhs = [left * right.rhs[i] for i in range(right.size)]
                rhs = np.array(rhs)
                return LinearEqs(right.variable, right.matrix, rhs)
            else:
                return left * right
        elif op == "/":
            return left / right
        elif op == "+":
            return left + right
        elif op == "-":
            return left - right
        else:
            raise ValueError(f"Unsupported operator: {op}.")

    def run_operator(self, op_name: str, op_args: list):
        """Run the operator with the given arguments."""
        # prepare the target variable
        if len(op_args) == 1:
            field = self._fields.get(op_args[0]["name"])
        else:
            field = self.run_func("", op_args)

        # perpare the operator
        op = self._operators.get(op_name)()
        op.prepare(
            field,
            self._mesh.get_topo_assistant(),
            self._mesh.get_geom_assistant(),
            **self._configs,
        )

        # run the operator
        var_name, var_type, eq_num = self._get_equation_info()
        final_eq = LinearEqs.zeros(var_name, eq_num, var_type)
        for node in self._mesh.nodes:
            curr = op.run(node.id)
            if isinstance(curr, LinearEqs):
                final_eq += curr
            else:
                final_eq.rhs[node.id] = -curr

        return final_eq

    def run_func(self, func_name: str, func_args: list):
        """Run the elementary function with the given arguments."""
        args_expr = "".join([f"{arg['name']}" for arg in func_args])
        func_expr = f"{func_name}({args_expr})"
        symbols = [arg["name"] for arg in func_args if arg["type"]]
        func = lambdify_numexpr(func_expr, symbols)

        inputs = []
        for arg in func_args:
            if arg["type"] == "coefficient":
                inputs.append(self._coefficients.get(arg["name"]))
            elif arg["type"] == "variable":
                inputs.append(self._fields.get(arg["name"]))
            else:
                raise ValueError(f"Unsupported argument: {arg}.")

        result = func(*inputs)
        return result
