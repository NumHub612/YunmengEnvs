# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Network structures mesh.
"""

from yunmeng.numerics.mesh import Mesh

# -----------------------------------------------
# region Network
# -----------------------------------------------


class Network:
    """
    Abstract network class for topological connectivity.
    """

    def to_mesh(self) -> Mesh:
        """
        Convert network to mesh.
        """
        raise NotImplementedError()
