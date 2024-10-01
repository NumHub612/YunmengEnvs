# -*- encoding: utf-8 -*-
"""
Solve 1d Burgers equation by uniform pde steps.
"""
import numpy as np
from numba import njit
from jax import grad, jit, vmap
import jax.numpy as jnp
import sympy as sp

from sympy.utilities.lambdify import lambdify
from matplotlib import pyplot as plt
from typing import Callable, Tuple, Any
import yaml


class TaskConfigurer:
    """
    Configure the simulation task.
    """

    def __init__(self, configs: str | dict):
        with open(configs, "r", encoding="utf-8") as f:
            self.task_config = yaml.safe_load(f)

        self.validate()

    def __getitem__(self, key: str) -> Any:
        return self.task_config[key]

    def __setitem__(self, key: str, value: Any):
        self.task_config[key] = value

    def validate(self):
        pass

    @property
    def configs(self) -> dict:
        return self.task_config


task_file = "./scripts/pdes/task.yml"
configs = TaskConfigurer(task_file)
print(configs["TITLE"])
# print(configs.configs)


class Node:
    """
    Node in the mesh.
    """

    node_index: int
    x: float
    y: float
    z: float

    faces: list
    cells: list

    def __init__(self, index: int, x: float, y: float = 0, z: float = 0):
        self.node_index = index
        self.x = x
        self.y = y
        self.z = z
        self.faces = []
        self.cells = []


class Face:
    """
    Face in the mesh.
    """

    face_index: int
    nodes: list
    cells: list

    center: np.ndarray
    normal: np.ndarray
    area: float

    def __init__(self, index: int, nodes: list):
        self.face_index = index
        self.nodes = nodes
        self.cells = []


class Cell:
    """
    Cell in the mesh.
    """

    cell_index: int
    faces: list
    nodes: list

    center: np.ndarray
    volume: float

    def __init__(self, index: int, faces: list):
        self.cell_index = index
        self.faces = faces
        self.nodes = []


class Mesh:
    """
    Mesh descretizating the domain, consisting of nodes, faces and cells.
    """

    def __init__(self, mesh_type: str):
        self.mesh_type = mesh_type
        self.nodes = np.empty(0, dtype=object)
        self.faces = np.empty(0, dtype=object)
        self.cells = np.empty(0, dtype=object)

    def initialize(self, **kwargs):
        nx = kwargs.get("nx")
        x_extent = kwargs.get("mesh_extent", [0, 2 * np.pi])
        x = np.linspace(0, x_extent[1], nx)

        self.nodes.resize(nx)
        for i in range(nx):
            self.nodes[i] = Node(i, x[i])

    def get_inner_node_indexs(self) -> np.ndarray:
        return np.arange(1, self.num_nodes - 1)

    def get_boundary_node_indexs(self) -> np.ndarray:
        return np.array([0, self.num_nodes - 1])

    def get_node(self, index: int) -> Node:
        return self.nodes[index]

    def get_inner_face_indexs(self) -> np.ndarray:
        pass

    def get_boundary_face_indexs(self) -> np.ndarray:
        pass

    def get_face(self, index: int) -> Face:
        pass

    def get_inner_cell_indexs(self) -> np.ndarray:
        pass

    def get_boundary_cell_indexs(self) -> np.ndarray:
        pass

    def get_cell(self, index: int) -> Cell:
        pass

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_faces(self) -> int:
        pass

    @property
    def num_cells(self) -> int:
        pass

    @property
    def extents(self) -> dict:
        x_extent = (self.nodes[0].x, self.nodes[-1].x)
        return {"x": x_extent}


mesh = Mesh("uniform")
mesh.initialize(**configs["SPATIAL"]["mesh_configs"])
print(mesh.num_nodes)


class Vector:
    """
    Vector variable.
    """

    def __init__(self, name: str, components: list):
        self.name = name
        self.components = components

    def subs(self, substitutions: dict) -> "Vector":
        return Vector(self.name, [c.subs(substitutions) for c in self.components])

    def lambdify(self, args: list) -> Callable:
        return lambdify(args, self.components)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.components)

    def dot(self, other: "Vector") -> sp.Expr:
        return sp.Mul(*[c1 * c2 for c1, c2 in zip(self.components, other.components)])

    def cross(self, other: "Vector") -> "Vector":
        pass


class Tensor:
    """
    Tensor variable.
    """

    def __init__(self, name: str, components: list):
        self.name = name
        self.components = components

    def subs(self, substitutions: dict) -> "Tensor":
        return Tensor(
            self.name,
            [
                [c1.subs(substitutions), c2.subs(substitutions)]
                for c1, c2 in self.components
            ],
        )

    def lambdify(self, args: list) -> Callable:
        return lambdify(args, [c1.components for c1 in self.components])

    def to_numpy(self) -> np.ndarray:
        return np.array(self.components)

    def dot(self, other: "Tensor") -> sp.Expr:
        pass

    def trace(self) -> sp.Expr:
        pass

    def inverse(self) -> "Tensor":
        pass


class Field:
    """
    Variables and their values on the mesh.
    """

    def __init__(self):
        pass

    def initialize(self, **kwargs):
        pass


class Equation:
    """
    PDE equation to be solved.
    """

    def __init__(self, expr: str):
        self.expr = expr

    def items(self):
        pass


class Solver:
    """
    Solver to solve the problem.
    """

    def __init__(self):
        pass


class Problem:
    """
    Problem to be solved, including mesh, field and equation.
    """

    def __init__(self):
        pass


class Viewer:
    """
    Viewer to visualize the results.
    """

    def __init__(self):
        pass

    def __del__(self):
        pass
