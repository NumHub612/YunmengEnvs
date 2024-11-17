# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Plotter for vector fields.
"""
from core.numerics.fields import Field
from core.numerics.mesh import Mesh

import matplotlib.pyplot as plt


def plot_vector_field(
    mesh: Mesh,
    field: Field,
    title: str = "Vector Field",
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    xlim: tuple = None,
    ylim: tuple = None,
    zlim: tuple = None,
    grid: bool = True,
    colorbar: bool = True,
    cmap: str = "viridis",
):
    """
    Plot a vector field.

    Args:
        mesh: The mesh of the vector field.
        field: The vector field to plot.
        title: The title of the plot.
        xlabel: The label of the x-axis.
        ylabel: The label of the y-axis.
        zlabel: The label of the z-axis.
        xlim: The range of the x-axis.
        ylim: The range of the y-axis.
        zlim: The range of the z-axis.
        grid: Whether to show the grid.
        colorbar: Whether to show the colorbar.
        cmap: The colormap of the plot.
    """
    if field.dtype != "vector":
        raise ValueError("The field must be a vector field.")

    # Get the spacial coordinates of the elements.
    elements = None
    if field.etype == "node":
        elements = [node.coord for node in mesh.nodes]
    elif field.etype == "face":
        elements = [face.center for face in mesh.faces]
    elif field.etype == "cell":
        elements = [cell.center for cell in mesh.cells]
    else:
        raise ValueError("Invalid element type.")

    X = [e.x for e in elements]
    Y = [e.y for e in elements]

    # Get the vector values at the elements.
    values = field.to_np()
    U = [v.x for v in values]
    V = [v.y for v in values]
    W = [v.z for v in values]

    # Plot the vector field.
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(X, Y, U, V, W, color="r", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    if grid:
        ax.grid()

    if colorbar:
        plt.colorbar(ax.get_children()[0], shrink=0.5)

    plt.show()
