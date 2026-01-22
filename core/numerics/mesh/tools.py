# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Auxiliary functions for mesh processing.
"""
from core.numerics.mesh.elements import Coordinate, Element, Cell, Face, ElementType
from core.numerics.fields import Vector
from scipy.spatial import cKDTree
import numpy as np
import math
import copy


# -----------------------------------------------
# region topo methods
# -----------------------------------------------


def check_projection_axis(points: list) -> str:
    """Check the projection axis (x, y, z)."""
    coords = extract_coordinates(points)
    x_var = np.var([c.x for c in coords])
    y_var = np.var([c.y for c in coords])
    z_var = np.var([c.z for c in coords])
    vars = [x_var, y_var, z_var]
    axis = np.argsort(vars)[0]  # Axis with the smallest variance
    axis = ["x", "y", "z"][axis]
    return axis


def sort_anticlockwise(points: list) -> list:
    """Sort points in anticlockwise order."""
    coords = {}
    for i, point in enumerate(points):
        if isinstance(point, Element):
            coords[i] = point.coordinate
        else:
            coords[i] = point
    center = calculate_center(list(coords.values()))
    axis = check_projection_axis(points)
    if axis.lower() == "z":
        sorted_coords = sorted(
            coords.items(),
            key=lambda x: math.atan2(x[1].y - center.y, x[1].x - center.x),
        )
    elif axis.lower() == "y":
        sorted_coords = sorted(
            coords.items(),
            key=lambda x: math.atan2(x[1].z - center.z, x[1].x - center.x),
        )
    elif axis.lower() == "x":
        sorted_coords = sorted(
            coords.items(),
            key=lambda x: math.atan2(x[1].y - center.y, x[1].z - center.z),
        )
    return [points[i] for i, _ in sorted_coords]


def extract_coordinates(elements: list) -> list:
    """Extract the coordinates of each element."""
    coords = copy.deepcopy(elements)
    for i, element in enumerate(elements):
        if isinstance(element, Element):
            coords[i] = element.coordinate
        elif isinstance(element, Coordinate):
            continue
        else:
            raise ValueError(f"Invalid element type: {type(element)}.")
    return coords


def search_nearest_elements(
    elements: list[Element],
    coordinate: Coordinate,
    etype: ElementType,
    top_k: int = 1,
    max_dist: float = np.inf,
) -> list[int]:
    """Search the k nearest elements to the given coordinate."""
    points = np.array([e.coordinate.to_np() for e in elements])
    indexes = [e.id for e in elements]
    tree = cKDTree(points)

    dists, idx = tree.query(
        coordinate.to_np(),
        k=top_k,
        distance_upper_bound=max_dist,
    )
    valid = np.isfinite(dists)
    results = indexes[idx[valid]]
    return results


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
    return np.linalg.norm(point1.to_np() - point2.to_np())


def calculate_center(points: list) -> Coordinate:
    """Calculate the center of the given coordinates."""
    coords = extract_coordinates(points)
    return Coordinate.from_np(
        np.mean([coord.to_np() for coord in coords], axis=0),
    )


def calculate_area(points: list) -> float:
    """Calculate the area of the given coordinates."""
    if len(points) < 3:
        return 0.0
    coords = extract_coordinates(points)
    coords = sort_anticlockwise(coords)
    center = calculate_center(coords)
    # Calculate the area using the shoelace formula
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += (coords[i].x - center.x) * (coords[j].y - center.y)
        area -= (coords[i].y - center.y) * (coords[j].x - center.x)
    area /= 2.0
    return abs(area)


def extract_coordinates_separated(
    elements: list[Element],
    dims: str = "xyz",
) -> dict:
    """Extract the coordinates of each element separatedly."""
    dims = dims.lower()
    if dims not in ["xyz", "xy", "xz", "yz", "x", "y", "z"]:
        raise ValueError(f"Invalid dimension: {dims}.")
    xs = np.array([e.coordinate.x for e in elements])
    ys = np.array([e.coordinate.y for e in elements])
    zs = np.array([e.coordinate.z for e in elements])
    coordinate_map = {"x": xs, "y": ys, "z": zs}
    coordinates = {d: coordinate_map.get(d) for d in dims}
    return coordinates


def generate_projection(
    coordinate: Coordinate,
    face: Face,
    normal: Vector,
) -> Coordinate:
    """Generate the projection on the given face from the given coordinate."""
    vec_np = (coordinate - face.coordinate).to_np()
    proj_np = np.dot(vec_np, normal.to_np()) * normal.to_np()
    proj_np = proj_np + face.coordinate.to_np()
    proj_coord = Coordinate.from_np(proj_np)
    return proj_coord
