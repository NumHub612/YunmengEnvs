# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for entities that can be described.
"""
from dataclasses import dataclass


@dataclass
class IDescribable:
    """To provide descriptive information on an entity."""

    caption: str = ""
    description: str = ""
