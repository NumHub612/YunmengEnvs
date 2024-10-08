# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Fields definition.
"""


class Vector:
    """
    A 3D vector field with x, y, and z components.
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Scalar:
    """
    A scalar field with a single value.
    """

    def __init__(self, value):
        self.value = value


class Tensor:
    """
    A 3x3 tensor field with xx, xy, xz, yy, yz, and zz components.
    """

    def __init__(self, xx, xy, xz, yy, yz, zz):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yy = yy
        self.yz = yz
        self.zz = zz
