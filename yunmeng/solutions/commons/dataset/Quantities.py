# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Canonical quantity definitions.

Keeping these in one module guarantees that exchange ports declared by
different components always speak the same (name, unit) vocabulary —
adapters and the coupling framework matchports by quantity,
so a single source of truth matters.
"""

from yunmeng.interfaces.solution import Quantity

PRECIPITATION = Quantity("precipitation", "Rainfall depth per time step", "mm")
EVAPORATION = Quantity("evaporation", "Potential evaporation depth", "mm")
RUNOFF = Quantity("runoff", "Runoff depth", "mm")
SOIL_MOISTURE = Quantity("soil_moisture", "Tension water storage", "mm")
DISCHARGE = Quantity("discharge", "Flow rate", "m3/s")
WATER_LEVEL = Quantity("water_level", "Water surface elevation", "m")
STORAGE = Quantity("storage", "Stored water volume", "m3")
PRESSURE = Quantity("pressure", "Atmospheric pressure", "Pa")
TEMPERATURE = Quantity("temperature", "Air temperature", "K")
VELOCITY = Quantity("velocity", "Flow velocity", "m/s")
SCALAR_FIELD = Quantity("scalar_field", "Generic scalar field", "")

CANONICAL = {
    q.name: q
    for q in (
        PRECIPITATION,
        EVAPORATION,
        RUNOFF,
        SOIL_MOISTURE,
        DISCHARGE,
        WATER_LEVEL,
        STORAGE,
        PRESSURE,
        TEMPERATURE,
        VELOCITY,
        SCALAR_FIELD,
    )
}


def quantity_of(name, unit=""):
    """Resolve a canonical quantity by name (case-insensitive), else create."""
    q = CANONICAL.get(str(name).lower())
    return q if q is not None else Quantity(str(name).lower(), unit=unit)
