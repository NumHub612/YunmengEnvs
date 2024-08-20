# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for extra space information.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ..ISpatialDefinition import ISpatialDefinition


class ISpaceExtension(ABC):
    @abstractmethod
    def get_spatial_definition(self) -> Optional[ISpatialDefinition]:
        """Spatial information."""
        pass
