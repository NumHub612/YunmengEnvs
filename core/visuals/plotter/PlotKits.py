# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Basic plot kits for visualizing the data.
"""
import matplotlib.pyplot as plt
import pyvista as pv
import vtk
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_data_series(
    x: list | np.ndarray,
    ys: dict,
    *,
    title: str = "data series",
    figsize: tuple = (8, 6),
    save_dir: str = None,
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
        title: Title of the plot.
        figsize: Figure size.
        save_dir: Directory to save the plot.
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
        >>> plot_data_series(x, ys)
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


def plot_mesh(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: str,
    *,
    title: str = "Mesh",
    figsize: tuple = (8, 6),
    save_dir: str = None,
    show: bool = True,
    show_edges: bool = False,
):
    """
    Plot 2d mesh with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, options: "2d", "3d".
        title: Title of the plot.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        show_edges: Whether to show edges.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type.lower() == "2d" else vtk.VTK_POLYHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, show_edges=show_edges)
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    # Set title and save plot
    if save_dir and not show:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")

        winsize = (figsize[0] * 100, figsize[1] * 100)
        plotter.screenshot(save_path, window_size=winsize)
    if show:
        plotter.show()
    plotter.close()


def plot_mesh_cloudmap(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: str,
    scalars: np.ndarray,
    domain: str,
    *,
    title: str = "Cloudmap",
    label: str = "value",
    figsize: tuple = (8, 6),
    save_dir: str = None,
    show: bool = True,
    cmap: str = "coolwarm",
    show_edges: bool = False,
):
    """
    Plot cloudmap with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, options: "2d", "3d".
        scalars: Scalar values.
        domain: Domain of the values bounded, options: "point", "cell".
        title: Title of the plot.
        label: Label of the values.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        cmap: Colormap of the plot.
        show_edges: Whether to show edges.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type.lower() == "2d" else vtk.VTK_HEXAHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = scalars
    else:
        mesh.cell_data[label] = scalars

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, scalars=label, cmap=cmap, show_edges=show_edges)
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    # Set title and save plot
    if save_dir and not show:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")

        winsize = (figsize[0] * 100, figsize[1] * 100)
        plotter.screenshot(save_path, window_size=winsize)
    if show:
        plotter.show()
    plotter.close()


def plot_mesh_streamplot(
    points_coordinates: np.ndarray,
    cells: np.ndarray,
    mesh_type: str,
    vectors: np.ndarray,
    domain: str,
    *,
    title: str = "Streamplot",
    label: str = "value",
    figsize: tuple = (8, 6),
    save_dir: str = None,
    show: bool = True,
    color: str = "red",
    mag: float = 0.1,
    show_edges: bool = False,
):
    """
    Plot streamplot with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        cells: Polygons or polyhedrons of the mesh.
        mesh_type: Type of the mesh, options: "2d", "3d".
        vectors: Vector values.
        domain: Domain of the values bounded, options: "point", "cell".
        title: Title of the plot.
        label: Label of the values.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        color: Color of the arrows.
        mag: Magnitude of the arrows.
        show_edges: Whether to show edges.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mtype = vtk.VTK_POLYGON if mesh_type.lower() == "2d" else vtk.VTK_HEXAHEDRON
    types = np.array([mtype] * len(cells))
    cells = np.concatenate(cells)

    mesh = pv.UnstructuredGrid(cells, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = vectors
    else:
        mesh.cell_data[label] = vectors

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, show_edges=show_edges)

    cents = mesh.points if domain == "point" else mesh.cell_centers().points
    plotter.add_arrows(cents, vectors, mag=mag, color=color)

    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    # Set title and save plot
    if save_dir and not show:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")

        winsize = (figsize[0] * 100, figsize[1] * 100)
        plotter.screenshot(save_path, window_size=winsize)
    if show:
        plotter.show()
    plotter.close()


def plot_mesh_scatters(
    points_coordinates: np.ndarray,
    scalars: np.ndarray,
    *,
    title: str = "Scatters",
    label: str = "value",
    figsize: tuple = (8, 6),
    save_dir: str = None,
    show: bool = True,
    cmap: str = "viridis",
    show_edges: bool = False,
):
    """
    Plot contour with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        scalars: Scalar values.
        title: Title of the plot.
        label: Label of the values.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        cmap: Colormap of the plot.
        show_edges: Whether to show edges.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mesh = pv.PolyData(points)
    mesh.point_data[label] = scalars

    # Create a plotter object
    plotter = pv.Plotter(off_screen=not show, title=title)
    plotter.add_mesh(mesh, scalars=label, cmap=cmap, show_edges=show_edges)

    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    # Set title and save plot
    if save_dir and not show:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")

        winsize = (figsize[0] * 100, figsize[1] * 100)
        plotter.screenshot(save_path, window_size=winsize)
    if show:
        plotter.show()
    plotter.close()
