# -*- encoding: utf-8 -*-
"""
StructuredGrid2D: uniform Cartesian 2D grid (node-centered).

Static-topology provider for operators' build() phase:
geometry (dx/dy), canonical boundary node numbering, coordinates.
"""

from __future__ import annotations

import numpy as np


class StructuredGrid2D:
    """Uniform structured 2D grid on [x0, x1] x [y0, y1]."""

    def __init__(
        self,
        lower_left: tuple[float, float] = (0.0, 0.0),
        upper_right: tuple[float, float] = (2.0, 2.0),
        nx: int = 32,
        ny: int = 32,
    ):
        if nx < 3 or ny < 3:
            raise ValueError("nx/ny must be >= 3.")
        self._x0, self._y0 = map(float, lower_left)
        self._x1, self._y1 = map(float, upper_right)
        self._nx = int(nx)
        self._ny = int(ny)
        self._lx = self._x1 - self._x0
        self._ly = self._y1 - self._y0
        self._dx = self._lx / (self._nx - 1)
        self._dy = self._ly / (self._ny - 1)

        xs = self._x0 + self._dx * np.arange(self._nx)
        ys = self._y0 + self._dy * np.arange(self._ny)
        self._xx, self._yy = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

    # -- geometry -----------------------------------

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dy(self) -> float:
        return self._dy

    @property
    def lx(self) -> float:
        return self._lx

    @property
    def ly(self) -> float:
        return self._ly

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """(xx, yy) node coordinate arrays of shape (nx, ny)."""
        return self._xx, self._yy

    # -- topology ------------------------------------

    def boundary_node_indices(self) -> np.ndarray:
        """Canonical flat indices (C-order over (nx, ny)) of boundary nodes."""
        ii, jj = np.meshgrid(np.arange(self._nx), np.arange(self._ny), indexing="ij")
        mask = (ii == 0) | (ii == self._nx - 1) | (jj == 0) | (jj == self._ny - 1)
        return np.flatnonzero(mask.ravel())

    def interior_node_indices(self) -> np.ndarray:
        ii, jj = np.meshgrid(np.arange(self._nx), np.arange(self._ny), indexing="ij")
        mask = (ii == 0) | (ii == self._nx - 1) | (jj == 0) | (jj == self._ny - 1)
        return np.flatnonzero((~mask).ravel())
