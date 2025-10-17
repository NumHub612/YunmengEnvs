# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for time stamp and time interval.
"""
from dataclasses import dataclass


@dataclass
class ITime:
    """Time interface to support a time stamp as well as a time interval."""

    timestamp: float = 0.0
    duration_in_days: float = 0.0
