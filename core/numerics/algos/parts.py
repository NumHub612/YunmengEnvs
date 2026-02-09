# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

"Mesh" filtering and partitioning methods.
"""
from __future__ import annotations
from core.numerics.mesh import Face, Cell, Node
from shapely.geometry import box, Polygon


class MeshFilter:
    """
    To provide mesh element filtering methods.
    """

    @staticmethod
    def filter_face_patch(mesh: "Mesh", expr: str):
        """
        Filter mesh faces by given expression.

        Args:
            mesh: The mesh instance.
            expr: The filtering expression.

        Returns:
            list: The filtered face IDs.
        """
        filter_func = eval(expr)

        patch = []
        for face in mesh.faces:
            if filter_func(face.coordinate.x, face.coordinate.y, face.coordinate.z):
                patch.append(face.id)
        return patch

    @staticmethod
    def filter_cell_zone(mesh: "Mesh", **conditions):
        """
        Filter mesh cells by given conditions.

        Args:
            mesh: The mesh instance.
            **conditions: The filtering conditions, following:
              + expr: The filtering expression.
              + countour: The contour of the zone.

        Returns:
            list: The filtered cell IDs.
        """
        zone = []

        if "expr" in conditions:
            expr = conditions["expr"]
            filter_func = eval(expr)
            for cell in mesh.cells:
                if filter_func(cell.coordinate.x, cell.coordinate.y, cell.coordinate.z):
                    zone.append(cell.id)
        elif "countour" in conditions:
            contour = conditions["countour"]
            coors = [mesh.nodes[c].coordinate.to_np() for c in contour]
            profiler = Polygon(coors)
            for cell in mesh.cells:
                coors = [mesh.faces[f].coordinate.to_np() for f in cell.faces]
                minx = min([c[0] for c in coors])
                maxx = max([c[0] for c in coors])
                miny = min([c[1] for c in coors])
                maxy = max([c[1] for c in coors])
                extent = box(minx, miny, maxx, maxy)
                if profiler.intersects(extent):
                    zone.append(cell.id)
        return zone
