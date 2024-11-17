# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Simulation solution for surface water modeling.
"""
from core.solutions.standards import ILinkableComponent, IOutput, IInput
from core.numerics.mesh import Grid2D, Coordinate
from core.numerics.fields import NodeField, Vector
from core.solvers import (
    solver_routines,
    init_methods,
    boundary_conditions,
    callback_handlers,
)
from configs.settings import logger


class SurfaceWaterSimulation(ILinkableComponent):
    """
    Surface water simulation component.
    """

    def __init__(self, name: str, input_component: IInput, output_component: IOutput):
        super().__init__(name, input_component, output_component)
        self.name = name
        self.input_component = input_component
        self.output_component = output_component


if __name__ == "__main__":
    from core.numerics.mesh import MeshTopo
    import numpy as np
    import matplotlib.pyplot as plt

    # set mesh
    low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
    nx, ny = 41, 41
    grid = Grid2D(low_left, upper_right, nx, ny)

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    start_x, end_x = int(0.5 / dx), int(1.0 / dx) + 1
    start_y, end_y = int(0.5 / dy), int(1.0 / dy) + 1
    init_groups = []
    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            index = i * ny + j
            init_groups.append(index)

    topo = MeshTopo(grid)
    bc_groups = []
    for i in topo.boundary_nodes_indexes:
        bc_groups.append(grid.nodes[i])

    # set initial condition
    node_num = grid.node_count
    init_field = NodeField(node_num, Vector(1, 1))
    for i in init_groups:
        init_field[i] = Vector(2, 2)

    ic = init_methods["hotstart"]("ic1", init_field)

    # set boundary condition
    bc_value = Vector(1, 1)
    bc = boundary_conditions["constant"]("bc1", bc_value, None)

    # set callback
    cbs = [callback_handlers["RenderCallback"]()]

    # set solver
    solver = solver_routines["fdm"]["burgers2d"]("solver1", grid, cbs)
    solver.set_ic("vel", ic)
    solver.set_bc("vel", bc_groups, bc)

    sigma = 0.2
    dt = sigma * dx
    nb_steps = 12
    solver.initialize(nb_steps, sigma=sigma)

    # run solver
    is_done = False
    while not is_done:
        is_done, _, _ = solver.inference(dt)

    # get solution
    u_simu = solver.get_solution("vel")
