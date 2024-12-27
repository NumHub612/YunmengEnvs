# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Interface for describing and discretizing pde equations.
"""
from abc import ABC, abstractmethod


class IEquation(ABC):
    """
    Interface for describing and discretizing pde equations.
    """

    pass


class IOperator(ABC):
    """
    Interface for describing and discretizing pde operators.
    """

    pass
