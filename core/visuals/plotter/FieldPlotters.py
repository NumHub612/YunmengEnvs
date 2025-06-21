# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Plotters for visualizing the fluid fields.
"""
from core.numerics.mesh import Mesh, MeshTopo, MeshGeom, MeshDim
from core.numerics.fields import Field, VariableType
from core.visuals.plotter import PlotKits
import numpy as np


def extract_coordinates(mesh, element_type: str) -> np.ndarray:
    """Extract the coordinates of all elements."""
    etype = element_type.lower()
    if etype not in ["node", "cell", "face"]:
        raise ValueError(f"Invalid element type: {etype}.")
    elements_map = {
        "node": mesh.nodes,
        "cell": mesh.cells,
        "face": mesh.faces,
    }
    elements = elements_map.get(etype)
    coordinates = np.array([e.coordinate.to_np() for e in elements])
    return coordinates


def plot_mesh(
    mesh: Mesh,
    *,
    title: str = "MeshPlot",
    save_dir: str = None,
    show: bool = False,
    show_edges: bool = False,
    slice_set: dict = None,
    **kwargs,
):
    """
    Plot the mesh.

    Args:
        mesh: The mesh to be plotted.
        title: The title of the plot.
        save_dir: The directory to save the plot.
        show: Whether to show the plot.
        show_edges: Whether to show the edges.
        slice_set: Slice style and configs.
    """
    cells, points, _ = _extract_mesh_data(mesh)
    mesh_type = mesh.dimension.value

    PlotKits.plot_mesh_geometry(
        points,
        cells,
        mesh_type,
        save_dir=save_dir,
        show=show,
        title=title,
        show_edges=show_edges,
        slice_set=slice_set,
        **kwargs,
    )


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
    dimension: str = "x",
    slice_set: dict = None,
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
        slice_set: Slice style and configs.
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
    mesh_domain = "point" if field.etype.value == "node" else "cell"
    mesh_type = mesh.dimension.value

    # extract the field data
    data, data_map = _extract_field_data(field)

    # plot net
    if mesh.dimension == MeshDim.DIM1:
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
            slice_set=slice_set,
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
    elif style == "streamplot" and field.dtype.value == "vector":
        PlotKits.plot_mesh_streamplot(
            points,
            cells,
            mesh.dimension.value,
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
        raise ValueError(f"Unsupported style: {style}")


def _extract_mesh_data(mesh: Mesh):
    """
    Extract the data of the given mesh.
    """
    topo, geom = MeshTopo(mesh), MeshGeom(mesh)

    # extract the points
    points = extract_coordinates(mesh, "node")
    points_splited = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

    cells = []
    if mesh.dimension.value == "2d":
        for cell in mesh.cells:
            node_ids = topo.cell_nodes[cell.id]
            nodes = mesh.get_nodes(node_ids)
            coors = [n.coordinate for n in nodes]
            indices = geom.sort_anticlockwise(dict(zip(node_ids, coors)))
            cells.append([len(indices)] + indices)
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

    if field.dtype == VariableType.SCALAR:
        return values, {"x": values, "y": values, "z": values}
    elif field.dtype == VariableType.VECTOR:
        us = values[:, 0]
        vs = values[:, 1]
        ws = values[:, 2]
        return values, {"x": us, "y": vs, "z": ws}
    else:
        raise ValueError(f"Unsupported field type: {field.dtype}")
