# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Spatial domain classes and methods for the cfd.
"""

from yunmeng.numerics.mesh import Mesh, ElementType, MeshDimension

# -----------------------------------------------
# region Grid
# -----------------------------------------------


class Grid(Mesh):
    """Abstract class for orthogonal structured grids."""

    def __init__(self):
        super().__init__()
        self._orthogonal = True
        self._uniform = False
        self._nx = None
        self._ny = None
        self._nz = None
        self._lx = None
        self._ly = None
        self._lz = None

    # -----------------------------------------------
    # properties
    # -----------------------------------------------

    @property
    def nx(self) -> int:
        """Discretization size in the x-direction."""
        return self._nx

    @property
    def ny(self) -> int:
        """Discretization size in the y-direction."""
        return self._ny

    @property
    def nz(self) -> int:
        """Discretization size in the z-direction."""
        return self._nz

    @property
    def uniform(self) -> bool:
        """Return if the grid is uniform."""
        return self._uniform

    @property
    def lx(self) -> float:
        """Length of the grid in the x-direction."""
        return self._lx

    @property
    def ly(self) -> float:
        """Length of the grid in the y-direction."""
        return self._ly

    @property
    def lz(self) -> float:
        """Length of the grid in the z-direction."""
        return self._lz

    # -----------------------------------------------
    # methods
    # -----------------------------------------------

    def match_node(self, i: int, j: int, k: int) -> int:
        """Match node with the local indices."""
        raise NotImplementedError()

    def match_cell(self, i: int, j: int, k: int) -> int:
        """Match cell with the local indices."""
        raise NotImplementedError()

    def get_node_neighbours(self, id: int) -> list[int]:
        """Get the neighbours node indices, sorted in:
        [east, west, north, south, top, bottom]
        """
        raise NotImplementedError()

    def get_cell_neighbours(self, id: int) -> list[int]:
        """Get the neighbours cell indices, sorted in:
        [east, west, north, south, top, bottom]
        """
        raise NotImplementedError()
