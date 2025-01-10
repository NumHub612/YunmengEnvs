# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Simulation solution for 3d fluid dynamics.
"""
from core.solutions.standards import ILinkableComponent, IOutput, IInput
from core.numerics.mesh import Grid3D, Coordinate
from core.numerics.fields import NodeField, Vector
from core.solvers import (
    solver_routines,
    init_methods,
    boundary_conditions,
    callback_handlers,
)
from configs.settings import logger


class Fluid3DSimulation(ILinkableComponent):
    """
    The 3D fluid dynamics simulation.
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
    low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
    nx, ny, nz = 11, 11, 11
    grid = Grid3D(low_left, upper_right, nx, ny, nz)
    topo = MeshTopo(grid)

    node_index_groups = []
    for i in range(9, 11):
        for j in range(2):
            for k in range(2):
                node_index_groups.append(grid.match_node(i, j, k))

    bc_node_groups = []
    for b in topo.boundary_nodes_indexes:
        bc_node_groups.append(grid.nodes[b])

    # set initial condition
    node_num = grid.node_count
    init_field = NodeField(node_num, "vector", Vector(0, 0, 0))
    for i in node_index_groups:
        init_field[i] = Vector(-2, 2, 2)

    ic = init_methods["hotstart"]("ic1", init_field)

    # set boundary condition
    bc_value = Vector(0, 0, 0)
    bc = boundary_conditions["constant"]("bc1", bc_value, None)

    # set callback
    output_dir = "./tests/results"
    confs = {"vel": {"style": "scatter", "dimension": "x"}}
    # confs = {"vel": {"style": "scatter", "dimension": "z"}}
    cb = callback_handlers["render"](output_dir, confs)

    # set solver
    solver = solver_routines["fdm"]["burgers3d"]("solver1", grid)
    solver.add_callback(cb)
    solver.add_ic("vel", ic)
    solver.add_bc("vel", bc_node_groups, bc)

    sigma = 0.2
    nu = 0.02
    dx = 2 / grid.nx
    dt = sigma * dx
    nb_steps = 60
    solver.initialize(nb_steps, nu)

    # run solver
    is_done = False
    while not is_done:
        is_done, _, status = solver.inference(dt)
        print(f"Step {status['current_time']}")

    # get solution
    u_simu = solver.get_solution("vel")

    # results player
    from core.visuals.animator import ImageSetPlayer

    results_dir = "./tests/results/vel"
    player = ImageSetPlayer(results_dir, figure_size=(15, 10), pause=30)
    player.play()
