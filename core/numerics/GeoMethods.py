# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Geometric methods for mesh processing.
"""
from core.numerics.mesh import Coordinate
import math


def calculate_center(coordinates: list) -> Coordinate:
    """
    Calculate the center of a list of coordinates.

    Args:
        coordinates: List of coordinates.

    Returns:
        The center coordinate.
    """
    if len(coordinates) == 0:
        return Coordinate(0, 0, 0)
    if len(coordinates) == 1:
        return coordinates[0]

    x_sum = 0
    y_sum = 0
    z_sum = 0
    for coord in coordinates:
        x_sum += coord.x
        y_sum += coord.y
        z_sum += coord.z

    x = x_sum / len(coordinates)
    y = y_sum / len(coordinates)
    z = z_sum / len(coordinates)
    return Coordinate(x, y, z)


def sort_coordinates_anticlockwise(coordinates: dict, ignored_axis: str = "z") -> list:
    """
    Sort plane coordinates in anticlockwise order.

    Args:
        coordinates: Coodinates with id as key.
        ignored_axis: The axis to be folded.

    Returns:
        Ids of sorted coordinates.
    """
    # Calculate the center of the coordinates
    center = calculate_center(coordinates.values())

    # Sort the coordinates by their angle with the center
    ignored_axis = ignored_axis.lower()
    if ignored_axis == "z":
        sorted_coordinates = sorted(
            coordinates.items(),
            key=lambda x: math.atan2(x[1].y - center.y, x[1].x - center.x),
        )
    elif ignored_axis == "y":
        sorted_coordinates = sorted(
            coordinates.items(),
            key=lambda x: math.atan2(x[1].z - center.z, x[1].x - center.x),
        )
    elif ignored_axis == "x":
        sorted_coordinates = sorted(
            coordinates.items(),
            key=lambda x: math.atan2(x[1].y - center.y, x[1].z - center.z),
        )
    else:
        raise ValueError("Invalid ignored_axis: {}".format(ignored_axis))

    # Return the ids of the sorted coordinates
    return [coord_id for coord_id, _ in sorted_coordinates]
