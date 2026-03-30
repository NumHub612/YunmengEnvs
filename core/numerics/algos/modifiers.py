# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh modifiers.
"""
from core.numerics.mesh.spatials import Mesh, MeshModifier, MeshModifyMode
from core.numerics.algos.topos import extract_coordinates
from core.numerics.enums import ElementType
import numpy as np


class ElevationModifier(MeshModifier):
    """Mesh elelevation modifier."""

    @property
    def mode(self) -> MeshModifyMode:
        return MeshModifyMode.GEOMETRY

    def validate(
        self,
        mesh: Mesh,
        elevations: np.ndarray,
        indexies: list[int] = None,
        etype: ElementType = ElementType.NODE,
        max_dz: float = None,
        **kwargs,
    ) -> bool:
        """Validate if the modification can be applied.

        Args:
            mesh: Mesh object.
            elevations: Elevations to modify.
            indexies: Indexies of elements to modify.
            etype: Element type.
            max_dz: Maximum elevation change.
        """
        elems = mesh.get_elements(etype)

        if indexies is None and len(elevations) != len(elems):
            return False
        if indexies is not None:
            if len(indexies) != len(elevations):
                return False
            min_idx = min(indexies)
            max_idx = max(indexies)
            if min_idx < 0 or max_idx >= len(elems):
                return False

        if max_dz is not None:
            if indexies is None:
                indexies = np.arange(len(elems))
            coordinates = extract_coordinates(elems[indexies])
            dz = np.abs(elevations - coordinates[:, 2])
            if np.max(dz) > max_dz:
                return False

        return True

    def modify(
        self,
        mesh: Mesh,
        elevations: np.ndarray,
        indexies: list[int] = None,
        etype: ElementType = ElementType.NODE,
        **kwargs,
    ):
        """Modify mesh elevation at specified elements.

        Args:
            mesh: Mesh object.
            elevations: Elevations to modify.
            indexies: Indexies of elements to modify.
            etype: Element type.
        """
        if indexies is None:
            indexies = np.arange(len(mesh.get_elements(etype)))
        else:
            indexies = np.array(indexies)

        elems = mesh.get_elements(etype)
        for idx, elev in zip(indexies, elevations):
            elems[idx].coordinate.z = elev
