# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Plotters for visualizing the fluid fields.
"""
from core.numerics.mesh import Mesh, MeshTopo, MeshGeom
from core.numerics.fields import Field
from core.visuals.plotter import PlotKits

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
        - 1d is always a line chart.
        - For line chart, `kwargs` can be used to specify
          the line style, e.g. 'color', 'marker', etc.
        - For cloudmap, `kwargs` can be used to specify
          the colormap, e.g. "cmap", "show_edges", etc.
    """
    if field.dtype != "scalar":
        raise ValueError("Field must be a scalar field.")

    geom = MeshGeom(mesh)
    points = geom.extract_coordinates("node")

    # plot 1d mesh special case
    if mesh.domain == "1d":
        x = geom.extract_coordinates_separated(field.etype, dims="x")[0]
        y = {label: {"values": field.to_np(), **kwargs}}

        PlotKits.plot_data_series(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # plot 2d/3d mesh
    topo = MeshTopo(mesh)
    domain = "point" if field.etype == "node" else "cell"

    cells = []
    if mesh.domain == "2d":
        for cell in mesh.cells:
            nodes = topo.collect_cell_nodes(cell.id)
            coors = [mesh.nodes[i].coordinate for i in nodes]
            indexes = geom.sort_anticlockwise(dict(zip(nodes, coors)))
            cells.append([len(indexes)] + indexes)
    else:
        for cell in mesh.cells:
            # Here we assume that the cell faces sorted as [north, south, east, west, bottom, top]
            nodes1 = mesh.faces[cell.faces[-1]].nodes
            coors1 = [mesh.nodes[i].coordinate for i in nodes1]
            nodes2 = mesh.faces[cell.faces[-2]].nodes
            coors2 = [mesh.nodes[i].coordinate for i in nodes2]

            # Points need to be sorted.
            points1 = geom.sort_anticlockwise(dict(zip(nodes1, coors1)))
            points2 = geom.sort_anticlockwise(dict(zip(nodes2, coors2)))
            cell = points1 + points2
            cells.append([len(cell)] + cell)

    style = style.lower()
    if style == "cloudmap":
        PlotKits.plot_mesh_cloudmap(
            points,
            cells,
            mesh.domain,
            field.to_np(),
            domain,
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
            **kwargs,
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
    if field.dtype != "vector":
        raise ValueError("Field must be a vector field.")

    geom = MeshGeom(mesh)
    points = geom.extract_coordinates("node")

    # split the vector field into components
    values = field.to_np()
    us = values[:, 0]
    vs = values[:, 1]
    ws = values[:, 2]

    data_map = {"x": us, "y": vs, "z": ws}

    # plot 1d mesh special case
    if mesh.domain == "1d":
        x = geom.extract_coordinates_separated(field.etype, dims="x")[0]
        y = {f"{label}_x": {"values": data_map.get(dimension), **kwargs}}

        PlotKits.plot_data_series(
            x,
            y,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
        )
        return

    # plot 2d mesh
    topo = MeshTopo(mesh)
    domain = "point" if field.etype == "node" else "cell"

    cells = []
    if mesh.domain == "2d":
        for cell in mesh.cells:
            nodes = topo.collect_cell_nodes(cell.id)
            coors = [mesh.nodes[i].coordinate for i in nodes]
            indexes = geom.sort_anticlockwise(dict(zip(nodes, coors)))
            cells.append([len(indexes)] + indexes)
    else:
        for cell in mesh.cells:
            # Here we assume that the cell faces sorted as [north, south, east, west, bottom, top]
            nodes1 = mesh.faces[cell.faces[-1]].nodes
            coors1 = [mesh.nodes[i].coordinate for i in nodes1]
            nodes2 = mesh.faces[cell.faces[-2]].nodes
            coors2 = [mesh.nodes[i].coordinate for i in nodes2]

            # Points need to be sorted.
            points1 = geom.sort_anticlockwise(dict(zip(nodes1, coors1)))
            points2 = geom.sort_anticlockwise(dict(zip(nodes2, coors2)))
            cell = points1 + points2
            cells.append([len(cell)] + cell)

    style = style.lower()
    if style == "cloudmap":
        PlotKits.plot_mesh_cloudmap(
            points,
            cells,
            mesh.domain,
            data_map.get(dimension),
            domain,
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
            cells,
            mesh.domain,
            values,
            domain,
            save_dir=save_dir,
            show=show,
            title=title,
            figsize=figsize,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported style: {style}")


if __name__ == "__main__":
    from core.numerics.mesh import Coordinate, Grid3D, Grid2D, Grid1D

    # set 3d mesh
    low_left, upper_right = Coordinate(0, 0, 0), Coordinate(2, 2, 2)
    nx, ny, nz = 3, 4, 5
    grid = Grid3D(low_left, upper_right, nx, ny, nz)

    scalar_values_n = np.array([[i % 100] for i in range(grid.node_count)])
    scalar_field = Field.from_np(scalar_values_n, "node")

    plot_scalar_field(
        scalar_field,
        grid,
        title="3D Scalar Field",
        label="value",
        save_dir=None,
        show=True,
    )

    plot_scalar_field(
        scalar_field,
        grid,
        title="3D Scalar Field",
        label="value",
        style="scatter",
        save_dir=None,
        show=True,
    )

    vector_values_n = np.random.rand(grid.node_count, 3)
    vector_field = Field.from_np(vector_values_n, "node")

    plot_vector_field(
        vector_field,
        grid,
        title="3D Vector Field",
        label="u",
        style="cloudmap",
        save_dir=None,
        show=True,
        dimension="x",
    )

    # set 2d mesh
    low_left, upper_right = Coordinate(0, 0), Coordinate(2, 2)
    nx, ny = 41, 41
    grid = Grid2D(low_left, upper_right, nx, ny)

    scalar_values_n = np.array([np.random.rand(1) for i in range(grid.node_count)])
    vector_values_n = np.random.rand(grid.node_count, 3)
    vector_values_n[:, 2] = 0.0

    scalar_values_c = np.array([np.random.rand(1) for i in range(grid.cell_count)])
    vector_values_c = np.random.rand(grid.cell_count, 3)
    vector_values_c[:, 2] = 0.0

    # plot scalar field
    scaler_field_n = Field.from_np(scalar_values_n, "node")

    plot_scalar_field(
        scaler_field_n,
        grid,
        title="Node Scalar Field1",
        label="value",
        style="cloudmap",
        save_dir=None,
        show=True,
    )

    plot_scalar_field(
        scaler_field_n,
        grid,
        title="Node Scalar Field2",
        label="value",
        style="scatter",
        save_dir=None,
        show=True,
    )

    scaler_field_c = Field.from_np(scalar_values_c, "cell")

    plot_scalar_field(
        scaler_field_c,
        grid,
        title="Cell Scalar Field",
        label="value",
        style="cloudmap",
        save_dir=None,
        show=True,
    )

    # plot vector field
    vector_field_n = Field.from_np(vector_values_n, "node")

    plot_vector_field(
        vector_field_n,
        grid,
        title="Node Vector Field1",
        label="u",
        style="cloudmap",
        save_dir=None,
        show=True,
        dimension="x",
    )

    plot_vector_field(
        vector_field_n,
        grid,
        title="Node Vector Field2",
        label="v",
        style="scatter",
        save_dir=None,
        show=True,
        dimension="y",
    )

    plot_vector_field(
        vector_field_n,
        grid,
        title="Node Vector Field3",
        label="value",
        style="streamplot",
        save_dir=None,
        show=True,
        color="b",
        mag=0.1,
    )

    vector_field_c = Field.from_np(vector_values_c, "cell")

    plot_vector_field(
        vector_field_c,
        grid,
        title="Cell Vector Field",
        label="u",
        style="cloudmap",
        save_dir=None,
        show=True,
        dimension="x",
    )

    # set 1d mesh
    start, end = Coordinate(0), Coordinate(2 * np.pi)
    grid = Grid1D(start, end, 401)

    scalar_values_n = np.array([np.random.rand(1) for i in range(grid.node_count)])
    scalar_field = Field.from_np(scalar_values_n, "node")

    plot_scalar_field(
        scalar_field,
        grid,
        title="1D Scalar Field",
        label="value",
        save_dir=None,
        show=True,
        color="r",
        marker="o",
    )

    vector_values_n = np.random.rand(grid.node_count, 3)
    vector_field = Field.from_np(vector_values_n, "node")

    plot_vector_field(
        vector_field,
        grid,
        title="1D Vector Field",
        label="u",
        save_dir=None,
        show=True,
        dimension="x",
        color="r",
        marker="o",
    )
