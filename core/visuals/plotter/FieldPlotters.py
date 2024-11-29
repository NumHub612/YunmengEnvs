# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Plotters for visualizing the fluid fields.
"""
from core.visuals.plotter import PlotKits
from core.numerics.fields import Field
from core.numerics.mesh import Mesh, MeshTopo
from core.numerics import GeoMethods

import numpy as np


def plot_scalar_field(
    field: Field,
    mesh: Mesh,
    *,
    title: str = "Scalar Field",
    label: str = "value",
    figsize: tuple = (10, 6),
    style: str = "cloudmap",
    save_dir: str = None,
    show: bool = False,
    **kwargs,
):
    """
    Plot the scalar field.

    Args:
        field: The scalar field to be plotted.
        mesh: The mesh of the field.
        title: The title of the plot, default is "Scalar Field".
        label: The label of field.
        figsize: The size of the figure.
        style: The style of the plot, can be "cloudmap" or "scatter".
        save_dir: The directory to save the plot.
        show: Whether to show the plot.
        kwargs: Other arguments for rendering.

    Notes:
        - 3d mesh is always a scatter plot;1d is always a line chart.
        - For line chart, `kwargs` can be used to specify
          the line style, e.g. 'color', 'marker', etc.
        - For cloudmap, `kwargs` can be used to specify
          the colormap, e.g. "cmap", "show_edges", etc.
        - For streamplot, `kwargs` can be used to specify
          the vector, e.g. "color", "mag", etc.
    """
    # plot 1d mesh special case
    if mesh.domain == "1d":
        x = GeoMethods.extract_coordinates_separated(mesh, field.etype, dim="x")
        y = {label: {"values": field.to_np()}.update(kwargs)}

        PlotKits.plot_data_series(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # perpare the geometric data
    points = GeoMethods.extract_coordinates(mesh, field.etype)

    # plot 3d mesh special case
    if mesh.domain == "3d":
        PlotKits.plot_mesh_scatters(
            points,
            field.to_np(),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
        )
        return

    # plot 2d mesh
    topo = MeshTopo(mesh)
    polygons = np.array([topo.collect_cell_nodes(i) for i in range(mesh.cell_count)])

    style = style.lower()
    if style == "cloudmap":
        PlotKits.plot_mesh_cloudmap(
            points,
            polygons,
            field.to_np(),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            **kwargs,
        )
    elif style == "scatter":
        PlotKits.plot_mesh_scatters(
            points,
            field.to_np(),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
        )
    else:
        raise ValueError(f"Unsupported style: {style}")


def plot_vector_field(
    field: Field,
    mesh: Mesh,
    *,
    title: str = "Vector Field",
    label: str = "value",
    figsize: tuple = (10, 6),
    style: str = "streamplot",
    save_dir: str = None,
    show: bool = False,
    dimension: str = None,
    **kwargs,
):
    """
    Plot the vector field on the given mesh.

    Args:
        field: The vector field to be plotted.
        mesh: The mesh of the field.
        title: The title of the plot.
        label: The label of field.
        figsize: The size of the figure.
        style: The plot style, can be "streamplot" or "cloudmap", "scatter".
        save_dir: The directory to save the plot.
        show: Whether to show the plot.
        dimension: Dimension to plot, can be "x", "y", or "z".
        kwargs: Other arguments for rendering.

    Notes:
        - 3d mesh is always a scatter plot;1d is always a line chart.
        - If `style` isn't "streamplot", user need to specify `dimension` to plot.
        - For line chart, `kwargs` can be used to specify
          the line style, e.g. 'color', 'marker', etc.
        - For cloudmap, `kwargs` can be used to specify
          the colormap, e.g. "cmap", "show_edges", etc.
        - For streamplot, `kwargs` can be used to specify
          the vector, e.g. "color", "mag", etc.
    """
    # split the vector field into components
    values = field.to_np()
    us = values[:, 0]
    vs = values[:, 1]
    ws = values[:, 2]

    data_map = {"x": us, "y": vs, "z": ws}

    # plot 1d mesh special case
    if mesh.domain == "1d":
        x = GeoMethods.extract_coordinates_separated(mesh, field.etype, dim="x")
        y = {f"{label}_x": {"values": data_map.get(dimension)}.update(kwargs)}

        PlotKits.plot_data_series(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # perpare the geometric data
    points = GeoMethods.extract_coordinates(mesh, field.etype)

    # plot 3d mesh special case
    if mesh.domain == "3d":
        PlotKits.plot_mesh_scatters(
            points,
            data_map.get(dimension),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            **kwargs,
        )
        return

    # plot 2d mesh
    topo = MeshTopo(mesh)
    polygons = np.array([topo.collect_cell_nodes(i) for i in range(mesh.cell_count)])

    style = style.lower()
    if style == "cloudmap":
        PlotKits.plot_mesh_cloudmap(
            points,
            polygons,
            data_map.get(dimension),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            **kwargs,
        )
    elif style == "scatter":
        PlotKits.plot_mesh_scatters(
            points,
            data_map.get(dimension),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            **kwargs,
        )
    elif style == "streamplot":
        PlotKits.plot_mesh_streamplot(
            points,
            polygons,
            values,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported style: {style}")
