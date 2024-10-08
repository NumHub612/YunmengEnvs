# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Variables definition.
"""
from abc import ABCMeta, abstractmethod


class Variable(metaclass=ABCMeta):
    """
    Abstract variable class.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value


class CellVariable(Variable):
    """
    Cell variable which represents statues of cells.

    Default at each cell center.
    """

    pass


class FaceVariable(Variable):
    """
    Face variable which represents statues of faces.

    Default at each face center.
    """

    pass


class NodeVariable(Variable):
    """
    Node variable which represents statues of nodes.
    """

    pass
