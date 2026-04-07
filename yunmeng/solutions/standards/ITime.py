# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for time stamp and time interval.
"""
from dataclasses import dataclass


@dataclass
class ITime:
    """Time interface to support a time stamp as well as a time interval."""

    # Time stamp in seconds since epoch (Jan 1, 1970)
    timestamp: float = 0.0

    # Time interval in hours
    duration_in_hours: float = 0.0
