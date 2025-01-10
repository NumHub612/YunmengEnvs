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
from core.utils.SympifyNumExpr import lambdify_numexpr


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
        if token not in self._fields:
            raise ValueError(f"Un-initialized variable: {token}.")
        return {
            "type": "variable",
            "name": token,
            "value": self._fields.get(token),
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
            "name": token,
            "value": coeff_value,
        }


class SimpleEquation(BaseEquation):
    """Simple single pde equation class."""

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
                    curr = it["value"]
                elif it["type"] == "variable":
                    curr = it["value"]
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
            field = op_args[0]["value"]
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
                final_eq.rhs[node.id] = curr

        return final_eq

    def run_func(self, func_name: str, func_args: list):
        """Run the elementary function with the given arguments."""
        args_expr = "".join([f"{arg['name']}" for arg in func_args])
        func_expr = f"{func_name}({args_expr})"
        symbols = [arg["name"] for arg in func_args if arg["type"]]
        func = lambdify_numexpr(func_expr, symbols)

        inputs = [arg["value"] for arg in func_args if arg["type"]]
        result = func(*inputs)
        return result


if __name__ == "__main__":
    from core.numerics.mesh import Grid2D, Coordinate
    from core.solvers.commons import inits, boundaries, callbacks
    from core.visuals.animator import ImageSetPlayer
    from core.visuals.plotter import plot_vector_field
    import numpy as np
    import os
    import shutil

    # set mesh
    low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
    nx, ny = 11, 11
    grid = Grid2D(low_left, upper_right, nx, ny)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    start_x, end_x = int(0.5 / dx), int(1.0 / dx) + 1
    start_y, end_y = int(0.5 / dy), int(1.0 / dy) + 1
    init_groups = []
    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            index = i * ny + j
            init_groups.append(index)

    topo = MeshTopo(grid)
    bc_groups = []
    for i in topo.boundary_nodes_indexes:
        bc_groups.append(grid.nodes[i])

    # set initial condition
    node_num = grid.node_count
    init_field = NodeField(node_num, "vector", Vector(1, 1), "u")
    for i in init_groups:
        init_field[i] = Vector(2, 2)

    ic = inits.HotstartInitialization("ic1", init_field)

    # set boundary condition
    bc_value = Vector(1, 1)
    bc = boundaries.ConstantBoundary("bc1", bc_value, None)

    # set callback
    output_dir = "./tests/results"
    results_dir = f"{output_dir}/u"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    confs = {"u": {"style": "cloudmap", "dimension": "x"}}
    cb = callbacks.RenderCallback(output_dir, confs)

    # set equations
    # equation_expr = "ddt::Ddt01(u) + u*grad::Grad01(u) == nu*laplacian::Lap01(u)"
    equation_expr = "ddt::Ddt01(u) + u*grad::Grad01(u) - nu*laplacian::Lap01(u) == 0"
    symbols = {
        "u": {
            "description": "velocity",
            "coefficient": False,
            "type": "vector",
            "bounds": (None, None),
        },
        "nu": {
            "description": "viscosity",
            "coefficient": True,
            "type": "scalar",
            "bounds": (0, None),
        },
    }
    coefficients = {"nu": Scalar(0.1)}
    variables = {"u": init_field}

    problem = SimpleEquation("burgers2d", fdm_operators)
    problem.set_equations([equation_expr], symbols)
    problem.set_coefficients(coefficients)
    problem.set_fields(variables)
    problem.set_mesh(grid)

    frame = 0
    plot_vector_field(
        init_field,
        grid,
        title=f"u-{frame}",
        save_dir=results_dir,
        style="cloudmap",
        dimension="x",
    )

    def print_matrix(mat: Matrix, num_rows=10):
        print(f"shape: {mat.shape}")
        for i in range(mat.shape[0]):
            if i >= num_rows:
                break

            if len(mat.shape) == 2:
                print(f"{i}-{mat[(i, j)]}", end=", ")
            else:
                print(f"{i}-{mat[i]}", end=", ")
        print()

    # discretize and solve
    sigma = 0.2
    dt = sigma * dx
    steps = 20
    while steps > 0:
        # solve
        eqs = problem.discretize(dt)
        solution = eqs.solve()

        # plot solution
        var_field = NodeField.from_np(solution, "node", "u")
        frame += 1
        plot_vector_field(
            var_field,
            grid,
            title=f"u-{frame}",
            save_dir=results_dir,
            style="cloudmap",
            dimension="x",
        )

        # update boundary condition
        for node in grid.get_topo_assistant().boundary_nodes_indexes:
            _, val = bc.evaluate(None, None)
            var_field[node] = val

        # update initial condition
        problem.set_fields({"u": var_field})
        steps -= 1

    # results player
    player = ImageSetPlayer(results_dir)
    player.play()
