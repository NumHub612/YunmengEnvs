# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Plotters for visualizing the fluid fields.
"""

from yunmeng.numerics.mesh import Mesh, MeshDimension, ElementType
from yunmeng.numerics.fields import Field, VariableType
from yunmeng.render.plotter.PlotKits import (
    _extract_field_data,
    _extract_mesh_data,
    plot_lines,
    plot_mesh_cloudmap,
    plot_mesh_scatters,
    plot_mesh_streamplot,
)
import numpy as np


def plot_field(
    field: Field,
    mesh: Mesh,
    *,
    title: str = "FieldPlot",
    label: str = "value",
    figsize: tuple = (10, 6),
    style: str = "cloudmap",
    save_dir: str = None,
    show: bool = False,
    **kwargs,
):
    """
    Plot the field on the given mesh.

    Args:
        field: The field to be plotted.
        mesh: The mesh of the field.
        title: The title of the plot.
        label: The label of field.
        figsize: The size of the figure.
        style: The plot style, can be "streamplot" or "cloudmap", "scatter".
        save_dir: The directory to save the plot.
        show: Whether to show the plot.

        kwargs: Other arguments for rendering.

    Notes:
        - dimension: Dimension to plot, can be "x", "y", or "z".
        - slice_set: Slice style and configs.
        - 3d mesh is always a scatter plot;1d is always a line chart.
        - If `style` isn't "streamplot", user need to specify `dimension` to plot.
        - For line chart, `kwargs` can be used to specify
          the line style, e.g. 'color', 'marker', etc.
        - For cloudmap, `kwargs` can be used to specify
          the colormap, e.g. "cmap", "show_edges", etc.
        - For streamplot, `kwargs` can be used to specify
          the vector, e.g. "color", "mag", etc.
    """
    # extract the mesh data
    cells, points, points_splited = _extract_mesh_data(mesh)
    mesh_domain = "point" if field.etype == ElementType.NODE else "cell"
    mesh_type = mesh.dimension.value

    # extract the field data
    data, data_map = _extract_field_data(field)
    dimension = kwargs.pop("dimension", "x")

    # plot net
    if mesh.dimension == MeshDimension.D1:
        x = points_splited.get(dimension)
        y = {
            f"{label}_{dimension}": {
                "values": data_map.get(dimension),
                **kwargs,
            }
        }

        plot_lines(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # plot mesh
    slice_set = kwargs.pop("slice_set", None)
    style = style.lower()
    if style == "cloudmap":
        plot_mesh_cloudmap(
            points,
            cells,
            mesh_type,
            data_map.get(dimension),
            mesh_domain,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            slice_set=slice_set,
            **kwargs,
        )
    elif style == "scatter":
        plot_mesh_scatters(
            points,
            data_map.get(dimension),
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            label=label,
            **kwargs,
        )
    elif style == "streamplot" and field.vtype == VariableType.VECTOR:
        plot_mesh_streamplot(
            points,
            cells,
            mesh_type,
            data,
            mesh_domain,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            slice_set=slice_set,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported style: {style} with field type: {field.vtype}")
