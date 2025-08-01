# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Network structures mesh.
"""
from core.numerics.mesh.meshes import Mesh
from abc import abstractmethod


class Network:
    """
    Network class.
    """

    @abstractmethod
    def to_mesh(self) -> Mesh:
        """
        Convert network to mesh.
        """
        pass


class RiverNet(Network):
    """
    River network class.
    """

    pass


class PipeNet(Network):
    """
    Pipe network class.
    """

    pass
