# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Canonical quantity definitions.

Keeping these in one module guarantees that exchange ports declared by
different components always speak the same (name, unit) vocabulary —
adapters and the coupling framework matchports by quantity,
so a single source of truth matters.
"""

from yunmeng.solutions.standards import Quantity

PRECIPITATION = Quantity(
    name="precipitation",
    description="Rainfall depth per time step",
    unit="mm",
)

EVAPORATION = Quantity(
    name="evaporation",
    description="Potential evaporation depth per time step",
    unit="mm",
)

RUNOFF = Quantity(
    name="runoff",
    description="Runoff depth",
    unit="mm",
)

SOIL_MOISTURE = Quantity(
    name="soil_moisture",
    description="Tension water storage",
    unit="mm",
)

DISCHARGE = Quantity(
    name="discharge",
    description="Flow rate",
    unit="m3/s",
)

WATER_LEVEL = Quantity(
    name="water_level",
    description="Water surface elevation",
    unit="m",
)

STORAGE = Quantity(
    name="storage",
    description="Stored water volume",
    unit="m3",
)

PRESSURE = Quantity(name="pressure", description="Atmospheric pressure", unit="Pa")
TEMPERATURE = Quantity(name="temperature", description="Air temperature", unit="K")
VELOCITY = Quantity(name="velocity", description="Flow velocity", unit="m/s")

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
    )
}
