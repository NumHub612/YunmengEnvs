# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Basic plot kits for visualizing the data.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pyvista as pv
import vtk
import seaborn as sns
import numpy as np
import os
import copy

from yunmeng.numerics.mesh.meshes import Mesh, MeshDimension
from yunmeng.numerics.fields import Field, VariableType
from yunmeng.numerics.algos import (
    sort_anticlockwise,
    extract_coordinates,
    MeshTopo,
    MeshGeom,
)

# ---------------------------------------------------
# region matplotlib 2D
# ---------------------------------------------------

installed_fonts = [f.name for f in fm.fontManager.ttflist]
if "SimHei" in installed_fonts:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
elif "Microsoft YaHei" in installed_fonts:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
else:
    plt.rcParams["font.sans-serif"] = ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_lines(
    x: list | np.ndarray,
    ys: dict,
    title: str = "data series",
    save_dir: str = None,
    *,
    figsize: tuple = (8, 6),
    show: bool = True,
    style: str = "merged",
    xlabel: str = "x",
    ylabel: str = "y",
    grid: bool = True,
):
    """
    Plot serial datas as a line chart.

    Args:
        x: List of 1d x-axis values.
        ys: Dictionary of y-axis values and redering theme descriptions.
        save_dir: Directory to save the plot.
        title: Title of the plot.
        figsize: Figure size.
        show: Whether to show the plot.
        style: Style of the plot, options: "merged", "separated".
        xlabel: Label of x-axis.
        ylabel: Label of y-axis.
        grid: Whether to show grid.

    Example:
    ````
        >>> x = [1, 2, 3, 4, 5]
        >>> ys = {
                "Simulation": {
                    "values": [1, 2, 3, 4, 5],
                    "color": "blue",
                    "marker": "o"
                },
                "Real": {
                    "values": [2, 4, 6, 8, 9],
                }
            }
        >>> plot_lines(x, ys)
    ````
    """
    fig = plt.figure(figsize=figsize)
    if style == "merged":
        ax = fig.add_subplot(111)
        for label, y in ys.items():
            values = y["values"]
            styles = {k: v for k, v in y.items() if k not in ["values"]}

            if len(values) != len(x):
                print(values.shape, x.shape)
                raise ValueError(f"The length of {label} values must be equal to x.")

            ax.plot(x, values, label=label, **styles)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(grid)
    elif style == "separated":
        n = len(ys)
        for i, (label, y) in enumerate(ys.items()):
            values = y["values"]
            styles = {k: v for k, v in y.items() if k not in ["values"]}

            if len(values) != len(x):
                raise ValueError(f"The length of {label} values must be equal to x.")

            ax = fig.add_subplot(n, 1, i + 1)
            ax.plot(x, values, label=label, **styles)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} - {label}")
            ax.grid(grid)
    else:
        raise ValueError(f"Unsupported style: {style}")

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    title: str = "Heatmap",
    save_dir: str = None,
    *,
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
    show: bool = True,
):
    """
    Plot a heatmap.

    Args:
        matrix: The matrix to be plotted.
        title: The title.
        save_dir: Folder to save.
        cmap: The color map.
        figsize: The figure size.
        show: Whether to show.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, cmap=cmap)
    plt.title(title)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_scatter(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    title: str = "Scatter Plot",
    save_dir: str = None,
    *,
    figsize: tuple = (8, 6),
    show: bool = True,
    xlabel: str = "x",
    ylabel: str = "y",
    color: str = "blue",
    marker: str = "o",
    alpha: float = 0.5,
    scale: float = 100.0,
    grid: bool = True,
):
    """
    Plot scatter chart.

    Args:
        rows: 1d array of x-axis values.
        cols: 1d array of y-axis values.
        data: 2d array of z-axis values.
        title: Title of the plot.
        save_dir: Directory to save the plot.
        figsize: Figure size.
        show: Whether to show the plot.
        xlabel: Label of x-axis.
        ylabel: Label of y-axis.
        color: Color of the markers.
        marker: Marker style.
        alpha: Transparency of the markers.
        scale: Scale of the markers.
        grid: Whether to show grid.
    """
    plt.figure(figsize=figsize)
    plt.scatter(
        rows.flatten(),
        cols.flatten(),
        s=data.flatten() * scale,
        cmap="viridis",
        c=color,
        marker=marker,
        alpha=alpha,
    )

    plt.colorbar(label=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if grid:
        plt.grid()
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------
# region pyvista 3D
# ---------------------------------------------------


def plot_mesh_cloudmap(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: int,
    scalars: np.ndarray,
    domain: str,
    title: str = "Cloudmap",
    save_dir: str = None,
    *,
    label: str = "value",
    figsize: tuple = (8, 6),
    show: bool = True,
    cmap: str = "coolwarm",
    show_edges: bool = False,
    show_scalars: bool = False,
    slice_set: dict = None,
):
    """
    Plot cloudmap with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, e.g., 1, 2, 3.
        scalars: Scalar values.
        domain: Domain of the values, options: "point", "cell".
        title: Title of the plot.
        save_dir: Directory to save the plot.
        label: Label of the values.
        figsize: Figure size.
        show: Whether to show the plot.
        cmap: Colormap of the plot.
        show_edges: Whether to show edges.
        show_scalars: Whether to show scalars.
        slice_set: Choose slice style.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type == 2 else vtk.VTK_HEXAHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = scalars
    else:
        mesh.cell_data[label] = scalars
    mesh = _mesh_slice_set(mesh, slice_set)

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, scalars=label, cmap=cmap, show_edges=show_edges)

    # show scalar values
    if show_scalars:
        scalars = np.around(scalars, decimals=3)
        if domain == "point":
            plotter.add_point_labels(points, scalars, name=label)
        else:
            centroids = mesh.cell_centers().points
            plotter.add_point_labels(centroids, scalars, name=label)

    # Set title and save plot
    _save_plot(plotter, figsize, save_dir, title, show)
    plotter.close()


def plot_mesh_streamplot(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: int,
    vectors: np.ndarray,
    domain: str,
    title: str = "Streamplot",
    save_dir: str = None,
    *,
    label: str = "value",
    figsize: tuple = (8, 6),
    show: bool = True,
    color: str = "red",
    mag: float = 0.1,
    show_edges: bool = False,
    slice_set: dict = None,
):
    """
    Plot streamplot with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, e.g., 1, 2, 3.
        vectors: Vector values.
        domain: Domain of the values, e.g. "point", "cell".
        title: Title of the plot.
        save_dir: Directory to save the plot.
        label: Label of the values.
        figsize: Figure size.
        show: Whether to show the plot.
        color: Color of the arrows.
        mag: Magnitude of the arrows.
        show_edges: Whether to show edges.
        slice_set: Choose slice style.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type == 2 else vtk.VTK_HEXAHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = vectors
    else:
        mesh.cell_data[label] = vectors
    mesh = _mesh_slice_set(mesh, slice_set)

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, show_edges=show_edges)

    cents = mesh.points if domain == "point" else mesh.cell_centers().points
    plotter.add_arrows(cents, vectors, mag=mag, color=color)

    # Set title and save plot
    _save_plot(plotter, figsize, save_dir, title, show)
    plotter.close()


def plot_mesh_scatters(
    points_coordinates: np.ndarray,
    scalars: np.ndarray,
    title: str = "Scatters",
    save_dir: str = None,
    *,
    label: str = "value",
    figsize: tuple = (8, 6),
    show: bool = True,
    cmap: str = "viridis",
    show_edges: bool = False,
):
    """
    Plot contour with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        scalars: Scalar field values.
        title: Title of the plot.
        save_dir: Directory to save the plot.
        label: Label of the values.
        figsize: Figure size.
        show: Whether to show.
        cmap: Colormap of the plot.
        show_edges: Whether to show edges.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mesh = pv.PolyData(points)
    mesh.point_data[label] = scalars

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, scalars=label, cmap=cmap, show_edges=show_edges)

    # Set title and save plot
    _save_plot(plotter, figsize, save_dir, title, show)
    plotter.close()


def plot_mesh_geometry(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: int,
    title: str = "Mesh",
    save_dir: str = None,
    *,
    figsize: tuple = (8, 6),
    show: bool = True,
    show_edges: bool = False,
    slice_set: dict = None,
    cmap: str = "viridis",
):
    """
    Plot mesh geometry.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, e.g., 1, 2, 3.
        domain: Domain of the values, e.g. "point", "cell".
        title: Title of the plot.
        save_dir: Directory to save the plot.
        figsize: Figure size.
        show: Whether to show the plot.
        show_edges: Whether to show edges.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type == 2 else vtk.VTK_HEXAHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Set values to the mesh
    elevations = points[:, 2]
    label = "elevation"
    mesh.point_data[label] = elevations
    mesh = _mesh_slice_set(mesh, slice_set)

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, cmap=cmap, show_edges=show_edges)

    # Set title and save plot
    _save_plot(plotter, figsize, save_dir, title, show)
    plotter.close()


# ---------------------------------------------------
# region utils
# ---------------------------------------------------


def _mesh_slice_set(mesh: pv.UnstructuredGrid, slice_set: dict):
    """
    Set the slice set for the mesh.
    """
    if slice_set and "style" in slice_set:
        configs = copy.deepcopy(slice_set)
        style = configs.pop("style")
        if style == "slice_along_axis":
            mesh = mesh.slice_along_axis(**configs)
        elif style == "slice_orthogonal":
            mesh = mesh.slice_orthogonal(**configs)
        elif style == "slice":
            mesh = mesh.slice(**configs)
    return mesh


def _save_plot(
    plotter: pv.Plotter, figsize: tuple, save_dir: str, title: str, show: bool
):
    """
    Save the plot.
    """
    # set plotting style
    plotter.add_text(title)
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    # save ans show
    if save_dir and not show:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")

        winsize = (figsize[0] * 100, figsize[1] * 100)
        plotter.screenshot(save_path, window_size=winsize)
    if show:
        plotter.show()


def _extract_mesh_data(mesh: Mesh):
    """
    Extract the data of the given mesh.
    """
    topo, geom = MeshTopo(mesh), MeshGeom(mesh)

    # extract the points
    points = extract_coordinates(mesh.nodes)
    points_splited = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

    cells = []
    if mesh.dimension == MeshDimension.D2:
        for i, cell in enumerate(mesh.cells):
            node_ids = topo.cell_nodes[i]
            nodes = mesh.get_nodes(node_ids)
            _, indexes = sort_anticlockwise(nodes, node_ids)
            cells.append([len(indexes)] + indexes)
    else:
        for cid in range(mesh.cell_count):
            cell_faces = topo.cell_faces[cid]
            node_ids1 = topo.face_nodes[cell_faces[-1]]
            node_ids2 = topo.face_nodes[cell_faces[-2]]
            cell = node_ids1 + node_ids2
            cells.append([len(cell)] + cell)

    return cells, points, points_splited


def _extract_field_data(field: Field):
    """
    Extract the data of the given field.
    """
    values = field.gather_to_host()

    if field.vtype == VariableType.SCALAR:
        return values, {"x": values}
    elif field.vtype == VariableType.VECTOR:
        us = values[:, 0]
        vs = values[:, 1]
        ws = values[:, 2]
        return values, {"x": us, "y": vs, "z": ws}
    else:
        raise ValueError(f"Unsupported field type: {field.vtype}")
