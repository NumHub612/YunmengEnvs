# -*- encoding: utf-8 -*-
"""
Unittests for the fdms solvers.
"""
import pytest
import numpy as np

from yunmeng.numerics.algos.modifiers import ElevationModifier
from yunmeng.numerics.algos.parts import MeshShard
from yunmeng.numerics.mesh.grids import Grid2D, Coordinate
from yunmeng.numerics.fields.fields import Field, Variable
from yunmeng.numerics.enums import VariableType, ElementType
from yunmeng.solvers.commons.inits import *
from yunmeng.solvers.commons.boundaries import *
from yunmeng.solvers.commons.callbacks import ImageRender
from yunmeng.solvers.fdm.BurgersSolver import *
from yunmeng.solvers.fdm.operators import *
from yunmeng.render.plotter.MeshPlotters import plot_mesh
from yunmeng.render.plotter.FieldPlotters import plot_field


# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def grid_101x101() -> Grid2D:
    """Create a flat 2D grid for testing."""
    ll, ur = Coordinate(0, 0), Coordinate(2.0, 2.0)
    grid_101x101 = Grid2D.by_uniform(ll, ur, 101, 101)

    # Add elevation modifier
    modifier = ElevationModifier()
    beds = np.ones(grid_101x101.node_count)
    grid_101x101.modify(modifier, elevations=beds, etype=ElementType.NODE, max_dz=2.0)
    return grid_101x101


@pytest.fixture
def H0(grid_101x101: Grid2D) -> Field:
    """Initial condition for the water depth field."""
    init_val = 1.0
    H0 = Field.from_size(
        grid_101x101.node_count, VariableType.SCALAR, ElementType.NODE, init_val
    )

    # Gaussian perturbation
    Lx = grid_101x101.lx
    Ly = grid_101x101.ly
    Nx = grid_101x101.nx
    Ny = grid_101x101.ny
    xc = Lx / 2
    yc = Ly / 2
    for i in range(Nx):
        for j in range(Ny):
            idx = grid_101x101.match_node(i, j)
            X = grid_101x101.nodes[idx].coordinate.x
            Y = grid_101x101.nodes[idx].coordinate.y
            dz = 0.5 * np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * 0.5**2))
            H0[idx] += dz
    return H0


@pytest.fixture
def U0(grid_101x101: Grid2D) -> Field:
    """Initial condition for the velocity field."""
    init_val = Variable.vector(1.0, 1.0, 0.0)
    U0 = Field.from_size(
        grid_101x101.node_count, VariableType.VECTOR, ElementType.NODE, init_val
    )

    nx, ny = grid_101x101.nx, grid_101x101.ny
    dx = grid_101x101.lx / (nx - 1)
    dy = grid_101x101.ly / (ny - 1)

    y_start = int(0.5 / dy)
    y_end = int(1 / dy + 1)
    x_start = int(0.5 / dx)
    x_end = int(1 / dx + 1)

    for i in range(nx):
        for j in range(ny):
            idx = grid_101x101.match_node(i, j)
            if y_start <= j <= y_end and x_start <= i <= x_end:
                U0[idx] = Variable.vector(2.0, 2.0, 0.0)

    return U0


# ============================================
# region Burgers2D Tests
# ============================================


class TestBurgers2D:
    """Test suite for Burgers2D solver."""

    def test_2d_grid(self, grid_101x101: Grid2D, H0: Field, U0: Field):
        """Test burgers2d on grid2d."""
        # visualize the grid and initial conditions
        plot_mesh(
            grid_101x101,
            title="grid_101x101",
            save_dir="tests/results/",
            show_edges=True,
        )
        plot_field(
            U0, grid_101x101, title="U0", save_dir="tests/results/", show_edges=True
        )

        # initial condition
        u_init = HotstartInitialization("U0", U0)

        # boundary condition
        value_bc = ValueBoundary("bc", Var([1.0, 1.0, 0.0]))
        bc_nodes = grid_101x101.get_topo_assistant().boundary_nodes

        # callbacks
        cb = ImageRender("render", "tests/results/", frequency=0.05)

        # operators
        operators = {
            "div": Div01(),
            "lap": Lap01(diffusivity=0.01),
        }

        # solver
        solver = BurgersExplicitSolver("solver", grid_101x101, operators)
        solver.add_ic("u", u_init)
        solver.add_bc("u", value_bc, bc_nodes, ElementType.NODE)
        solver.add_callback(cb)

        # initialize
        total_time = 1.5
        solver.initialize(total_time, time_step=0.008, cfl=0.5)

        # run the simulation
        t, dt = 0.0, total_time / 10
        while not solver.status.finished:
            status = solver.inference()
            if status.current_time >= t:
                t += dt
                print(
                    f"Time: {status.current_time:.4f} / {status.end_time:.4f}, "
                    f"Time step: {status.time_step:.4f}, "
                    f"Step time: {status.step_time:.4f}"
                )

        # visualize results
        u_end = solver.get_solution("u")
        plot_field(
            u_end,
            grid_101x101,
            title="u_end",
            save_dir="tests/results/",
            show_edges=True,
        )
