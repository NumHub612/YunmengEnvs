# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Provides numerical algorithms for field interpolation.
"""
from __future__ import annotations
from core.numerics.fields import Field
from core.numerics.mesh import Mesh, ElementType


class FieldInterp:
    """
    Provides numerical algorithms for field interpolation.
    """

    @staticmethod
    def cell_to_face(field: Field, mesh: "Mesh", method: str = "linear") -> Field:
        """
        Interpolates a field from cells to faces.

        Args:
            field: The field to be interpolated.
            mesh: The mesh to be used for interpolation.
            method: The interpolation method.

        Returns:
            The interpolated field.
        """
        topo = mesh.get_topo_assistant()
        geom = mesh.get_geom_assistant()
        part = mesh.get_part_assistant()

        result = Field(
            part,
            field.dtype,
            ElementType.FACE,
            requires_grad=field._meta.requires_grad,
        )
        for fid in mesh.faces:
            cids = topo.face_cells[fid]
            if len(cids) == 1:
                # boundary face
                continue
            c1, c2 = cids
            dist1 = geom.cell2face_distance[c1][fid]
            dist2 = geom.cell2face_distance[c2][fid]
            ratio = dist1 / (dist1 + dist2)

            if method == "linear":
                values = [field[c1], field[c2]]
                result[fid] = (1 - ratio) * values[0] + ratio * values[1]
            else:
                raise NotImplementedError()
        return result

    @staticmethod
    def node_to_face(field: Field, mesh: "Mesh", method: str = "linear") -> Field:
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

    @staticmethod
    def face_to_cell(field: Field, mesh: "Mesh", method: str = "linear") -> Field:
        """
        Interpolates a field from faces to cells.

        Args:
            field: The field to be interpolated.
            mesh: The mesh to be used for interpolation.
            method: The interpolation method.

        Returns:
            The interpolated field.
        """
        pass
