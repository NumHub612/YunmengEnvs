# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interface for extra space information.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ..ISpatialDefinition import ISpatialDefinition


class ISpaceExtension(ABC):

    @property
    @abstractmethod
    def spatial_definition(self) -> Optional[ISpatialDefinition]:
        """Spatial information."""
        pass
