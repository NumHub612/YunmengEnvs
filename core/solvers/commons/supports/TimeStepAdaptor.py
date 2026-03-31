# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Time step limiters for the solvers.
"""
from core.numerics.mesh.spatials import Mesh, Grid
from core.numerics.fields.fields import Field
import numpy as np


def cfl_time_step(
    mesh: Mesh, water_depth: Field, velocity: Field, cfl_nb: float
) -> float:
    """
    Compute the time step based on the CFL condition.
    """
    if mesh.dimension.value != "2d":
        raise NotImplementedError(f"Not support for {mesh.dimension}.")

    # Compute the wave speed
    g = 9.81
    gh = g * water_depth.gather_to_host()
    c = np.sqrt(gh)  # wave speed
    speed = np.abs(velocity.gather_to_host()) + c
    max_speed = np.max(speed)
    if max_speed < 1e-8:
        return 1e-3

    # Compute the time step
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

    dt = cfl_nb * min_dist / max_speed
    return dt
