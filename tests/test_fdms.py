# -*- encoding: utf-8 -*-
"""
Unittests for the fdms solvers.
"""

import pytest
import numpy as np

from yunmeng.numerics.algos import ElevationModifier, MeshShard
from yunmeng.numerics.grids import Grid2D, Coordinate
from yunmeng.numerics.fields import Field, Variable
from yunmeng.numerics.enums import VariableType, ElementType
from yunmeng.render.plotter import plot_mesh_ids, plot_mesh, plot_field

from yunmeng.solvers.commons.inits import *
from yunmeng.solvers.commons.boundaries import *
from yunmeng.solvers.commons.callbacks import ImageRender

from yunmeng.solvers.fdm.BurgersSolver import *
from yunmeng.solvers.fdm.NavierStokesSolver import *
from yunmeng.solvers.fdm.operators import *

# ============================================
# region Fixtures
# ============================================


@pytest.fixture
def grid_41x41() -> Grid2D:
    """Create a flat 2D grid for testing."""
    ll, ur = Coordinate(0, 0), Coordinate(2.0, 2.0)
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
    init_val = Variable.vector(1.0, 1.0, 0.0)
    U0 = Field.from_size(
        grid_41x41.node_count, VariableType.VECTOR, ElementType.NODE, init_val
    )

    nx, ny = grid_41x41.nx, grid_41x41.ny
    dx = grid_41x41.lx / (nx - 1)
    dy = grid_41x41.ly / (ny - 1)

    y_start = int(0.5 / dy)
    y_end = int(1 / dy + 1)
    x_start = int(0.5 / dx)
    x_end = int(1 / dx + 1)

    for i in range(nx):
        for j in range(ny):
            idx = grid_41x41.match_node(i, j)
            if y_start <= j <= y_end and x_start <= i <= x_end:
                U0[idx] = Variable.vector(2.0, 2.0, 0.0)

    return U0


# ============================================
# region Burgers2D Tests
# ============================================


class TestBurgers2D:
    """Test suite for Burgers2D solver."""

    def test_2d_grid(self, grid_41x41: Grid2D, H0: Field, U0: Field):
        """Test burgers2d on grid2d."""
        # visualize the grid and initial conditions
        plot_mesh(
            grid_41x41,
            title="grid_41x41",
            save_dir="tests/results/bg",
            show_edges=True,
        )
        plot_field(
            U0, grid_41x41, title="U0", save_dir="tests/results/bg", show_edges=True
        )

        # initial condition
        u_init = HotstartInitialization("U0", U0)

        # boundary condition
        value_bc = ValueBoundary("bc", Var([1.0, 1.0, 0.0]))
        bc_nodes = grid_41x41.get_topo_assistant().boundary_nodes

        # callbacks
        cb = ImageRender("render", "tests/results/bg", frequency=0.05)

        # operators
        def source_func(loc: Coordinate, u: Variable) -> float:
            # zero source
            forcing = Variable.vector(0.0, 0.0, 0.0)
            return forcing

        operators = [
            Grad01(["u"]),
            Lap01(["u"], diffusivity=0.01),
            Src01(["u"], tau=1.0, source_func=source_func),
        ]

        # solver
        solver = BurgersExplicitSolver("solver", grid_41x41, operators)
        solver.add_ic("u", u_init)
        solver.add_bc("u", value_bc, bc_nodes, ElementType.NODE)
        solver.add_callback(cb)

        # initialize
        total_time = 1.0
        solver.initialize(total_time, time_step=0.002, cfl=0.5)

        # run the simulation
        t, dt = 0.0, total_time / 10
        while not solver.status.finished:
            status = solver.inference()
            if status.current_time >= t or status.finished:
                t += dt
                print(
                    f"Time: {status.current_time:.4f} / {status.end_time:.4f}, "
                    f"Time step: {status.time_step:.4f}, "
                    f"Step time: {status.step_time:.4f}"
                )

        # visualize the last frame result
        u_end = solver.get_solution("u")
        plot_field(
            u_end,
            grid_41x41,
            title="u_end",
            save_dir="tests/results/bg",
            show_edges=True,
        )


# ============================================
# region Navier-Stokes Tests
# ============================================


class TestNavierStokes2D:
    """Test suite for NavierStokes2D solver."""

    def test_driven_cavity_flow(self, grid_41x41: Grid2D):
        """Test 2d Navier-Stokes solver on driven cavity flow."""
        ll, ur = Coordinate(0, 0), Coordinate(2.0, 2.0)
        grid_41x41 = Grid2D.by_uniform(ll, ur, 41, 41)
        plot_mesh_ids(grid_41x41, title="grid_41X41", save_dir="tests/results/ns")

        # Initial Conditions
        u_init_val = Variable.vector(0.0, 0.0, 0.0)
        u_field = Field.from_size(
            grid_41x41.node_count, VariableType.VECTOR, ElementType.NODE, u_init_val
        )
        u_init = HotstartInitialization("U0", u_field)

        p_init_val = 0.0
        p_field = Field.from_size(
            grid_41x41.node_count, VariableType.SCALAR, ElementType.NODE, p_init_val
        )
        p_init = HotstartInitialization("P0", p_field)

        # Boundary
        topo = grid_41x41.get_topo_assistant()
        all_bc_nodes = topo.boundary_nodes
        north_nodes, ohter_nodes = [], []
        for nid in all_bc_nodes:
            node = grid_41x41.nodes[nid]
            if node.coordinate.y == 2.0:
                north_nodes.append(nid)
            else:
                ohter_nodes.append(nid)

        # Boundary Conditions
        north_v_bc = ValueBoundary("v_bc1", [1.0, 0.0, 0.0])
        other_v_bc = ValueBoundary("v_bc2", [0.0, 0.0, 0.0])

        north_p_bc = ValueBoundary("p_bc1", 0.0)
        other_p_bc = FluxBoundary("p_bc2", [0.0, 0.0, 0.0])

        # cfd operators
        operators = [
            Grad01(["u"]),
            Grad02(["p"]),
            Lap01(["u"], diffusivity=0.01),
            Lap02(["p"]),
            Div01(["u"]),
        ]

        # callbacks
        cb = ImageRender(
            "render",
            "tests/results/ns",
            frequency=0.5,
            fields={
                "u": {"style": "streamplot"},
                "p": {"style": "cloudmap"},
            },
        )

        # solver
        solver = NavierStokesSolver("solver", grid_41x41, operators)
        solver.add_ic("u", u_init)
        solver.add_ic("p", p_init)
        solver.add_bc("u", north_v_bc, north_nodes, ElementType.NODE)
        solver.add_bc("u", other_v_bc, ohter_nodes, ElementType.NODE)
        solver.add_bc("p", north_p_bc, north_nodes, ElementType.NODE)
        solver.add_bc("p", other_p_bc, ohter_nodes, ElementType.NODE)
        solver.add_callback(cb)

        # initialize
        total_time = 5.0
        solver.initialize(total_time, time_step=0.01, cfl=0.5)

        # run the simulation
        t, dt = 0.0, total_time / 10
        while not solver.status.finished:
            status = solver.inference()
            if status.current_time >= t or status.finished:
                t += dt
                print(
                    f"Time: {status.current_time:.4f} / {status.end_time:.4f}, "
                    f"Time step: {status.time_step:.4f}, "
                    f"Step time: {status.step_time:.4f}"
                )

        u_end = solver.get_solution("u")
        plot_field(
            u_end,
            grid_41x41,
            title="u_end",
            save_dir="tests/results/ns",
            show_edges=True,
        )

        p_end = solver.get_solution("p")
        plot_field(
            p_end,
            grid_41x41,
            title="p_end",
            save_dir="tests/results/ns",
            show_edges=True,
        )
