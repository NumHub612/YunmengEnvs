# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Common handy tools for topology and geometry.
"""

from yunmeng.numerics.fields import Variable
from yunmeng.numerics.mesh.elements import Element, Face, Coordinate

from typing import List
import numpy as np
import math

# -----------------------------------------------
# region topo methods
# -----------------------------------------------


def extract_coordinates(elements: list[Element]) -> np.ndarray:
    """Extract the coordinates of each element."""
    coords_list = [e.coordinate.to_numpy() for e in elements]
    return np.asarray(coords_list, dtype=np.float64)


def calculate_center(points: list[Element]) -> Coordinate:
    """Calculate the center of the given points."""
    coords = extract_coordinates(points)
    return Coordinate.from_numpy(np.mean(coords, axis=0))


def check_projection_axis(points: list[Element]) -> str:
    """Check the projection axis (x, y, z)."""
    coords = extract_coordinates(points)
    x_var = np.var(coords[:, 0])
    y_var = np.var(coords[:, 1])
    z_var = np.var(coords[:, 2])
    vars = [x_var, y_var, z_var]
    axis = np.argsort(vars)[0]  # smallest variance axis
    axis = ["x", "y", "z"][axis]
    return axis


def sort_anticlockwise(
    points: list[Element], indexes: List[int] = None
) -> tuple[list[Element], List[int]]:
    """Sort points in anticlockwise order."""
    if indexes is None:
        indexes = list(range(len(points)))
    coord_map = {idx: p for idx, p in zip(indexes, points)}
    coord_lst = [p.coordinate.to_numpy() for p in points]
    center = np.mean(coord_lst, axis=0, dtype=np.float64)

    axis = check_projection_axis(points)
    if axis.lower() == "z":
        sorted_points = sorted(
            coord_map.items(),
            key=lambda x: math.atan2(
                x[1].coordinate.y - center[1], x[1].coordinate.x - center[0]
            ),
        )
    elif axis.lower() == "y":
        sorted_points = sorted(
            coord_map.items(),
            key=lambda x: math.atan2(
                x[1].coordinate.z - center[2], x[1].coordinate.x - center[0]
            ),
        )
    elif axis.lower() == "x":
        sorted_points = sorted(
            coord_map.items(),
            key=lambda x: math.atan2(
                x[1].coordinate.y - center[1], x[1].coordinate.z - center[2]
            ),
        )
    else:
        raise ValueError(f"Invalid projection axis: {axis}")

    indexes, elements = zip(*sorted_points)
    return list(elements), list(indexes)


# -----------------------------------------------
# region geom methods
# -----------------------------------------------


def calculate_distance(
    point1: Coordinate | Element, point2: Coordinate | Element
) -> float:
    """Calculate the distance between two coordinates."""
    if isinstance(point1, Element):
        point1 = point1.coordinate
    if isinstance(point2, Element):
        point2 = point2.coordinate
    return np.linalg.norm(point1.to_numpy() - point2.to_numpy())


def generate_projection(
    coordinate: Coordinate,
    face: Face,
    normal: Variable,
) -> Coordinate:
    """Generate the projection on the given face."""
    vec_np = (coordinate - face.coordinate).to_numpy()
    proj_np = np.dot(vec_np, normal.to_numpy()) * normal.to_numpy()
    proj_np = proj_np + face.coordinate.to_numpy()
    proj_coord = Coordinate.from_numpy(proj_np)
    return proj_coord
