# -*- encoding: utf-8 -*-
"""
Unittests for the fdms solvers.
"""
import pytest
import numpy as np

from core.numerics.algos.modifiers import ElevationModifier
from core.numerics.algos.parts import MeshShard
from core.numerics.mesh.grids import Grid2D, Coordinate
from core.numerics.fields.fields import Field, Variable
from core.numerics.enums import VariableType, ElementType
from core.solvers.commons.inits import HotstartInitialization
from core.render.plotter.MeshPlotters import plot_mesh
from core.render.plotter.FieldPlotters import plot_field


# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def grid_41x41():
    """Create a flat 2D grid for testing."""
    ll, ur = Coordinate(0, 0), Coordinate(10, 10)
    grid_41x41 = Grid2D.by_uniform(ll, ur, 41, 41)

    # Add elevation modifier
    modifier = ElevationModifier()
    beds = np.ones(grid_41x41.node_count)
    grid_41x41.modify(modifier, elevations=beds, etype=ElementType.NODE, max_dz=2.0)
    return grid_41x41


@pytest.fixture
def H0(grid_41x41: Grid2D) -> Field:
    """Initial condition for the water depth field."""
    init_val = 1.0
    H0 = Field.from_size(
        grid_41x41.node_count, VariableType.SCALAR, ElementType.NODE, init_val
    )

    # Gaussian perturbation
    Lx = grid_41x41.lx
    Ly = grid_41x41.ly
    Nx = grid_41x41.nx
    Ny = grid_41x41.ny
    xc = Lx / 2
    yc = Ly / 2
    for i in range(Nx):
        for j in range(Ny):
            idx = grid_41x41.match_node(i, j)
            X = grid_41x41.nodes[idx].coordinate.x
            Y = grid_41x41.nodes[idx].coordinate.y
            dz = 0.5 * np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * 0.5**2))
            H0[idx] += dz
    return H0


@pytest.fixture
def U0(grid_41x41: Grid2D) -> Field:
    """Initial condition for the velocity field."""
    init_val = Variable.vector(0.0, 0.0, 0.0)
    U0 = Field.from_size(
        grid_41x41.node_count, VariableType.VECTOR, ElementType.NODE, init_val
    )
    return U0


# ============================================
# region Swe2D Tests
# ============================================


class TestSwe2D:
    """Test suite for SWE2D solver."""

    def test_swe2d_grid(self, grid_41x41: Grid2D, H0: Field, U0: Field):
        """Test swe2d on grid2d."""
        # visualize the grid and initial conditions
        # plot_mesh(
        #     grid_41x41, title="grid_41x41", save_dir="tests/results/", show_edges=True
        # )
        # plot_field(
        #     H0, grid_41x41, title="H0", save_dir="tests/results/", show_edges=True
        # )
        # plot_field(
        #     U0, grid_41x41, title="U0", save_dir="tests/results/", show_edges=True
        # )

        # initial condition
        u_init = HotstartInitialization("U0", U0)
        h_init = HotstartInitialization("H0", H0)
