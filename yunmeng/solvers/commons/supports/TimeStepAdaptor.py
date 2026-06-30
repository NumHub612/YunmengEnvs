# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time step limiters for the solvers.
"""

from yunmeng.numerics.mesh.spatials import Mesh, Grid
from yunmeng.numerics.fields.fields import Field
from yunmeng.numerics.consts import G
import numpy as np


def cfl_timestep(
    mesh: Mesh,
    velocity: Field,
    cfl_nb: float,
    diffusivity: float = None,
    water_depth: Field = None,
    min_dt: float = 1e-6,
) -> float:
    """
    Compute the time step based on the CFL condition.

    Args:
        mesh: The mesh.
        velocity: The velocity field.
        cfl_nb: The CFL number.
        diffusivity: The diffusivity coefficient.
        water_depth: The water depth field.
        min_dt: The minimum time step.

    Returns:
        The time step.

    TODO: # Add source term effect on time step.
    """
    if mesh.dimension.value != "2d":
        raise NotImplementedError(f"Not support for {mesh.dimension}.")

    # Compute the wave speed
    wave = 0.0
    if water_depth is not None:
        gh = G * water_depth.gather_to_host()
        wave = np.sqrt(gh)  # wave speed

    # Compute min distance
    geom = mesh.get_geom_assistant()
    etype = velocity.meta.etype.value
    if isinstance(mesh, Grid):
        lx, nx = mesh.lx, mesh.nx
        ly, ny = mesh.ly, mesh.ny
        min_dist = np.min([lx / nx, ly / ny])
    elif etype == "cell":
        min_dist, _, _ = geom.cell2cell_distance_stats
    elif etype == "node":
        min_dist = np.min(geom.face_perimeter)
    else:
        raise NotImplementedError(f"Not support for {etype}.")

    # Convective time step
    speed = np.abs(velocity.gather_to_host()) + wave
    max_speed = np.max(speed)
    dt = cfl_nb * min_dist / (max_speed + 1e-8)

    # Diffusive time step
    if diffusivity is not None:
        dt_diff = cfl_nb * min_dist**2 / (2 * diffusivity)
        dt = min(dt, dt_diff)

    dt = max(dt, min_dt)
    return dt
