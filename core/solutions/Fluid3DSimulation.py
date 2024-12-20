# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

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
    pass
