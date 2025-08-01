# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Provides numerical algorithms for field interpolation.
"""
from core.numerics.fields import Field, NodeField, CellField, FaceField
from core.numerics.mesh import Mesh
import numpy as np


class FieldInterpolators:
    """
    Provides numerical algorithms for field interpolation.
    """

    @staticmethod
    def interp_cell_to_face(field: Field, mesh: Mesh, method: str = "linear") -> Field:
        """
        Interpolates a field from cells to faces.

        Args:
            field: The field to be interpolated.
            mesh: The mesh to be used for interpolation.
            method: The interpolation method.

        Returns:
            The interpolated field.
        """
        result = FaceField(mesh.face_count, field.dtype)

        topo = mesh.get_topo_assistant()
        geom = mesh.get_geom_assistant()
        for face in mesh.faces:
            fid = topo.face_indices[face.id]
            cids = topo.face_cells[face.id]
            if len(cids) == 1:
                # boundary face
                continue
            c1, c2 = cids
            cid1 = topo.cell_indices[c1]
            cid2 = topo.cell_indices[c2]

            dist1 = geom.cell2face_distances[cid1][fid]
            dist2 = geom.cell2face_distances[cid2][fid]
            ratio = dist1 / (dist1 + dist2)

            if method == "linear":
                values = [field[cid1], field[cid2]]
                result[fid] = (1 - ratio) * values[0] + ratio * values[1]
            else:
                raise NotImplementedError()
        return result

    @staticmethod
    def interp_node_to_face(field: Field, mesh: Mesh, method: str = "linear") -> Field:
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


class ArrayInterpolators:
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
