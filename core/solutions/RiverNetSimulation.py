# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Simulation solution for 1d river network problem.
"""
from core.solutions.standards import ILinkableComponent, IOutput, IInput
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
