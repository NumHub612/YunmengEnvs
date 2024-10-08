# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

1d/2d/2d structured grids.
"""
from core.numerics.mesh import Mesh


class Grid1D(Mesh):
    """1D structured grid."""

    def __init__(self, start, end, num):
        super().__init__()
        self.start = start
        self.end = end
        self.num = num
        self.delta = (end - start) / (num - 1)
        self.nodes = [start + i * self.delta for i in range(num)]
        self.elements = [(i, i + 1) for i in range(num - 1)]
        self.boundaries = []


class Grid2D(Mesh):
    """2D structured grid."""

    def __init__(self, start, end, num):
        super().__init__()
        self.start = start
        self.end = end
        self.num = num
        self.delta = (end - start) / (num - 1)
        self.nodes = [
            (start[0] + i * self.delta[0], start[1] + j * self.delta[1])
            for i in range(num)
            for j in range(num)
        ]
        self.elements = [
            (i, i + 1, i + num, i + num + 1) for i in range(0, num**2 - num, num)
        ]


class Grid3D(Mesh):
    """3D structured grid."""

    def __init__(self, start, end, num):
        super().__init__()
        self.start = start
        self.end = end
        self.num = num
        self.delta = (end - start) / (num - 1)
        self.nodes = [
            (
                start[0] + i * self.delta[0],
                start[1] + j * self.delta[1],
                start[2] + k * self.delta[2],
            )
            for i in range(num)
            for j in range(num)
            for k in range(num)
        ]
        self.elements = [
            (
                i,
                i + 1,
                i + num,
                i + num + 1,
                i + num**2,
                i + num**2 + 1,
                i + num**2 + num,
                i + num**2 + num + 1,
            )
            for i in range(0, num**3 - num**2, num**2)
        ]
