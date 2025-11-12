# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Times-, curve- and patter-series data structures.
"""
import numpy as np


class Timeseries:
    """
    A timeseries is a sequence of data points indexed by time.
    """

    def __init__(self, time: np.ndarray, data: np.ndarray):
        self._data = data
        self._time = time
        self._cursor = 0


class Curve:
    """
    A curve is a function that maps two float dataseries.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self._y = y
        self._cursor = 0

        self._segments = []
        self._index = 0


class Pattern:
    """
    A pattern is a recurring structure in a timeseries.
    """

    def __init__(self):
        pass


class Table:
    """
    A table is a multidimensional data structure.
    """

    def __init__(self):
        pass
