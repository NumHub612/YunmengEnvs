# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Simulation solution for 1d river network problem.
"""
from core.solutions.standards import ILinkableComponent, IOutput, IInput
from core.numerics.mesh import Grid1D, Coordinate, Node, Mesh
from core.numerics.fields import NodeField, Scalar
from core.solvers.commons import boundaries, inits, callbacks
from core.solvers import fdm
from core.solvers import (
    solver_routines,
    init_methods,
    boundary_conditions,
    callback_handlers,
)
from core.utils.SympifyNumExpr import lambdify_numexpr
from configs.settings import logger


class RiverNetSimulation(ILinkableComponent):
    """
    Simulation solution for 1d river network problem.
    """

    def __init__(self, name: str, input_component: IInput, output_component: IOutput):
        super().__init__(name, input_component, output_component)
        self.name = name
        self.input_component = input_component
        self.output_component = output_component
        self.logger = logger


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # set mesh
    mesh_confs = {
        "type": "grid1d",
        "args": {"x0": 0, "x1": 2 * np.pi, "nx": 401},
        "groups": {"gp1": {"domain": "node", "indexes": [0, 400]}},
    }
    start, end = Coordinate(0), Coordinate(2 * np.pi)
    grid = Grid1D(start, end, 401)

    xs = np.linspace(0, 2 * np.pi, 401)
    group1 = [grid.nodes[0], grid.nodes[400]]

    # set initial condition
    phi = (
        "exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1))) + exp(-(-4*t + x)**2/(4*nu*(t + 1)))"
    )
    phiprime = "-(-8*t + 2*x)*exp(-(-4*t + x)**2/(4*nu*(t + 1)))/(4*nu*(t + 1)) - (-8*t + 2*x - 4*pi)*exp(-(-4*t + x - 2*pi)**2/(4*nu*(t + 1)))/(4*nu*(t + 1))"
    expr = f"-2 * nu * (({phiprime}) / ({phi})) + 4"
    symbols = ["t", "x", "nu"]

    init_confs = {
        "ic1": {
            "var": "u",
            "type": "custom",
            "domain": "node",
            "args": {"func": expr, "symbols": symbols},
        }
    }
    func = lambdify_numexpr(expr, symbols)
    nu = 0.07

    def ic_func(mesh: Mesh):
        field = NodeField(mesh.node_count, "scalar")
        for i in range(mesh.node_count):
            x = mesh.nodes[i].coordinate.x
            value = Scalar(func(0.0, x, nu))
            field[i] = value
        return field

    ic = inits.CustomInitialization("ic1", grid, ic_func)

    # set boundary condition
    bc_confs = {
        "bc1": {
            "var": "u",
            "type": "dirichlet",
            "domain": "node",
            "args": {"func": "0"},
        }
    }

    def bc_func(t, node: Node):
        value = Scalar(func(t, node.coordinate.x, nu))
        return None, value

    bc = boundaries.CustomBoundary("bc1", bc_func)

    # set callback
    output_dir = "./tests/results"
    cb = callbacks.RenderCallback(output_dir)

    # set solver
    solver = fdm.Burgers1D("solver1", grid)
    solver.set_callback(cb)
    solver.set_ic("u", ic)
    solver.set_bc("u", group1, bc)

    nb_steps = 100
    solver.initialize(nb_steps)

    # analytical solution
    t = 0
    u = np.asarray([func(t, x0, nu) for x0 in xs])
    nx = 401
    dx = 2 * np.pi / (nx - 1)
    nu = 0.07
    dt = dx * nu

    def analytical_solution(u):
        un = u.copy()
        # update all inner points at once
        for i in range(1, nx - 1):
            u[i] = (
                un[i]
                - un[i] * dt / dx * (un[i] - un[i - 1])
                + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
            )

        # set boundary conditions
        u[-1] = (
            un[-1]
            - un[-1] * dt / dx * (un[-1] - un[-2])
            + nu * dt / dx**2 * (un[1] - 2 * un[-1] + un[-2])
        )
        u[0] = u[-1]
        return u

    # run solver
    while solver.current_time < solver.total_time:
        solver.inference(dt)

    # get solution
    u_simu = solver.get_solution("u")
    u_simu = np.asarray([var.value for var in u_simu])
    u_real = np.asarray([func(solver.total_time, x, nu) for x in xs])

    # results player
    from core.visuals.animator import ImageSetPlayer

    results_dir = "./tests/results/u"
    player = ImageSetPlayer(results_dir)
    player.play()

    # plot results
    # plt.figure(figsize=(11, 7), dpi=100)
    # plt.plot(xs, u_simu, color="blue", linewidth=2, marker="o", label="Computational")
    # plt.plot(xs, u_real, color="red", linewidth=2, label="Analytical")
    # plt.xlabel("x", fontsize=16)
    # plt.ylabel("u", fontsize=16)
    # plt.title("Comparison of computational and analytical results", fontsize=18)

    # plt.grid(True)
    # plt.legend(loc="upper right")
    # plt.xlim([0, 2 * np.pi])
    # plt.ylim([0, 10])

    # plt.xticks(np.arange(0, 2 * np.pi, 1))
    # plt.yticks(np.arange(0, 10, 1))
    # plt.show()
