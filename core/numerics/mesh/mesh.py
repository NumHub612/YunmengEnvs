# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Abstract mesh class for describing the geometry and topology.
"""
from abc import ABC, abstractmethod


class Mesh(ABC):
    """Abstract mesh class for describing the geometry and topology."""

    @abstractmethod
    def __init__(self):
        pass
