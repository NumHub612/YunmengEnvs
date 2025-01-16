# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Plotters for visualizing the fluid fields.
"""
from core.numerics.mesh import Mesh, MeshTopo, MeshGeom
from core.numerics.fields import Field
from core.visuals.plotter import PlotKits

import numpy as np


def plot_field(
    field: Field,
    mesh: Mesh,
    *,
    title: str = "FieldPlot",
    label: str = "value",
    figsize: tuple = (10, 6),
    style: str = "streamplot",
    save_dir: str = None,
    show: bool = False,
    dimension: str = "x",
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
    # extract the mesh data
    cells, points, points_splited = _extract_mesh_data(mesh)
    mesh_domain = "point" if field.etype == "node" else "cell"
    mesh_type = mesh.domain

    # extract the field data
    data, data_map = _extract_field_data(field)

    # plot net
    if mesh.domain == "1d":
        x = points_splited.get(dimension)
        y = {
            f"{label}_{dimension}": {
                "values": data_map.get(dimension),
                **kwargs,
            }
        }

        PlotKits.plot_data_series(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # plot mesh
    style = style.lower()
    if style == "cloudmap":
        PlotKits.plot_mesh_cloudmap(
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
    elif style == "streamplot" and field.dtype == "vector":
        PlotKits.plot_mesh_streamplot(
            points,
            cells,
            mesh.domain,
            data,
            mesh_domain,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported style: {style}")


def _extract_mesh_data(mesh: Mesh):
    """
    Extract the data of the given mesh.
    """
    topo, geom = MeshTopo(mesh), MeshGeom(mesh)

    # extract the points
    points = geom.extract_coordinates("node")
    points_splited = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

    cells = []
    if mesh.domain == "2d":
        for cell in mesh.cells:
            nodes = topo.collect_cell_nodes(cell.id)
            coors = [mesh.nodes[i].coordinate for i in nodes]
            indexes = geom.sort_anticlockwise(dict(zip(nodes, coors)))
            cells.append([len(indexes)] + indexes)
    else:
        for cell in mesh.cells:
            nodes1 = mesh.faces[cell.faces[-1]].nodes
            coors1 = [mesh.nodes[i].coordinate for i in nodes1]
            nodes2 = mesh.faces[cell.faces[-2]].nodes
            coors2 = [mesh.nodes[i].coordinate for i in nodes2]

            # Points need to be sorted.
            points1 = geom.sort_anticlockwise(dict(zip(nodes1, coors1)))
            points2 = geom.sort_anticlockwise(dict(zip(nodes2, coors2)))
            cell = points1 + points2
            cells.append([len(cell)] + cell)

    return cells, points, points_splited


def _extract_field_data(field: Field):
    """
    Extract the data of the given field.
    """
    values = field.to_np()

    if field.dtype == "scalar":
        return values, {"x": values, "y": values, "z": values}
    elif field.dtype == "vector":
        us = values[:, 0]
        vs = values[:, 1]
        ws = values[:, 2]
        return values, {"x": us, "y": vs, "z": ws}
    else:
        raise ValueError(f"Unsupported field type: {field.dtype}")
