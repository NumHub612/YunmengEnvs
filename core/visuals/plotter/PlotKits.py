# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

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
    polygons: np.ndarray,
    *,
    title: str = "Mesh",
    figsize: tuple = (8, 6),
    save_dir: str = None,
    show: bool = True,
    show_edges: bool = False,
    slice: str | tuple = None,
    origin: tuple = None,
):
    """
    Plot 2d mesh with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        polygons: Polygons of the mesh.
        title: Title of the plot.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        show_edges: Whether to show edges.
        slice: Slice of the mesh, options: "x", "y", "z".
        origin: Origin of the slice.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)

    faces, types = [], []
    for polygon in polygons:
        faces.append([len(polygon)] + polygon)
        types.append(vtk.VTK_POLYGON)
    faces = np.concatenate(faces)
    types = np.array(types)

    mesh = pv.UnstructuredGrid(faces, types, points)
    if slice and origin:
        mesh = mesh.slice(normal=slice, origin=origin)

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
    polygons: np.ndarray,
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
    slice: str | tuple = None,
    origin: tuple = None,
):
    """
    Plot cloudmap with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        polygons: Polygons of the mesh.
        scalars: Scalar values.
        domain: Domain of the values bounded, options: "point", "polygon".
        title: Title of the plot.
        label: Label of the values.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        cmap: Colormap of the plot.
        show_edges: Whether to show edges.
        slice: Slice of the mesh, options: "x", "y", "z" or vector.
        origin: Origin of the slice.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)

    faces, types = [], []
    for polygon in polygons:
        faces.append([len(polygon)] + polygon)
        types.append(vtk.VTK_POLYGON)
    faces = np.concatenate(faces)
    types = np.array(types)

    mesh = pv.UnstructuredGrid(faces, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = scalars
    else:
        mesh.cell_data[label] = scalars

    if slice and origin:
        mesh = mesh.slice(normal=slice, origin=origin)

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
    polygons: np.ndarray,
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
    slice: str | tuple = None,
    origin: tuple = None,
):
    """
    Plot streamplot with unstructured mesh.

    Args:
        points_coordinates: List of coordinates of points.
        polygons: Polygons of the mesh.
        vectors: Vector values.
        domain: Domain of the values bounded, options: "point", "polygon".
        title: Title of the plot.
        label: Label of the values.
        figsize: Figure size.
        save_dir: Directory to save the plot.
        show: Whether to show the plot.
        color: Color of the arrows.
        mag: Magnitude of the arrows.
        show_edges: Whether to show edges.
        slice: Slice of the mesh, options: "x", "y", "z" or vector.
        origin: Origin of the slice.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)

    faces, types = [], []
    for polygon in polygons:
        faces.append([len(polygon)] + polygon)
        types.append(vtk.VTK_POLYGON)
    faces = np.concatenate(faces)
    types = np.array(types)

    mesh = pv.UnstructuredGrid(faces, types, points)

    # Set values to the mesh
    domain = domain.lower()
    if domain == "point":
        mesh.point_data[label] = vectors
    else:
        mesh.cell_data[label] = vectors

    if slice and origin:
        mesh = mesh.slice(normal=slice, origin=origin)

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
    slice: str | tuple = None,
    origin: tuple = None,
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
        slice: Slice of the mesh.
        origin: Origin of the slice.

    Notes:
        - `show` and `save_dir` are mutually exclusive.
    """
    # Create a pyvista mesh object
    points = points_coordinates.astype(np.float32)
    mesh = pv.PolyData(points)
    mesh.point_data[label] = scalars

    if slice and origin:
        mesh = mesh.slice(normal=slice, origin=origin)

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


if __name__ == "__main__":
    from core.numerics.mesh import Coordinate, Grid2D, MeshTopo
    import numpy as np
    import matplotlib.pyplot as plt

    # set mesh
    low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
    nx, ny = 41, 41
    grid = Grid2D(low_left, upper_right, nx, ny)
    topo = MeshTopo(grid)

    scalar_field_n = np.array([np.random.rand(1) for i in range(grid.node_count)])
    vector_field_n = np.random.rand(grid.node_count, 3)
    vector_field_n[:, 2] = 0.0

    scalar_field_c = np.array([np.random.rand(1) for i in range(grid.cell_count)])
    vector_field_c = np.random.rand(grid.cell_count, 3)
    vector_field_c[:, 2] = 0.0

    # get points
    points = []
    for n in grid.nodes:
        arr = n.coordinate.to_np().tolist()
        arr[2] = np.random.rand(1)[0] * 0.1
        points.append(arr)
    points = np.array(points).astype(float)

    # get faces
    cells = []
    for i in range(grid.cell_count):
        nodes = topo.collect_cell_nodes(i)
        cells.append(nodes)

    plot_mesh_cloudmap(
        points,
        cells,
        scalar_field_n,
        "point",
        save_dir=r"D:\3_codes\1_AIs\YunmengEnvs\tests\results",
        title="2D Cloudmap",
        # show=False,
        show=True,
        slice="x",
        origin=(0.5, 0.5, 0.0),
    )

    plot_mesh_cloudmap(
        points,
        cells,
        scalar_field_c,
        "cell",
        title="2D Cloudmap",
        show=True,
    )

    plot_mesh_streamplot(
        points,
        cells,
        vector_field_n,
        "point",
        title="2D Streamplot",
        show=True,
    )

    plot_mesh_streamplot(
        points,
        cells,
        vector_field_c,
        "cell",
        title="2D Streamplot",
        show=True,
    )

    plot_mesh_scatters(
        points,
        scalar_field_n,
        title="2D Scatters",
        show=True,
    )
