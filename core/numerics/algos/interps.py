# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Provides numerical algorithms for field interpolation.
"""
from core.numerics.fields import Field
from core.numerics.mesh import Mesh
import numpy as np


class FieldInterpolations:
    """
    Provides numerical algorithms for field interpolation.
    """

    @staticmethod
    def interp_from_node_to_face(
        field: Field, mesh: Mesh, method: str = "linear"
    ) -> Field:
        """
        Interpolates a field from nodes to faces.

        Args:
            field: The field to be interpolated.
            mesh: The mesh to be used for interpolation.
            method: The interpolation method.

        Returns:
            The interpolated field.
        """
        pass


class ArrayInterpolations:
    """
    Provides numerical algorithms for array interpolation.
    """

    @staticmethod
    def interp_linearly(arr: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Interpolates an array using linear interpolation.

        Args:
            arr: The array to be interpolated.
            xi: The coordinates of the interpolation points.

        Returns:
            The interpolated array.
        """
        pass

    @staticmethod
    def interp_nearest(arr: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Interpolates an array using nearest-neighbor interpolation.

        Args:
            arr: The array to be interpolated.
            xi: The coordinates of the interpolation points.

        Returns:
            The interpolated array.
        """
        pass
