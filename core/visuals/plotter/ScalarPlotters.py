# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Plotter for displaying scalar values.
"""
from core.numerics.fields import Field, Scalar
from core.numerics.mesh import Mesh

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_data_series(
    xs: list | np.ndarray,
    ys: dict,
    *,
    title="plot",
    xlabel="x",
    ylabel="y",
    grid=True,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    show=True,
    save_dir=None,
):
    """
    Plot serial datas as a line chart.

    Args:
        xs: List of 1d x-axis values.
        ys: Dictionary of y-axis values descriptions.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        grid: Whether to show the grid.
        xlim: Limits of the x-axis.
        ylim: Limits of the y-axis.
        xticks: List of x ticks.
        yticks: List of y ticks.
        show: Whether to show the plot.
        save_dir: Directory to save the plot.

    Example:
    ````
        >>> xs = [1, 2, 3, 4, 5]
        >>> ys = {
                    "Simulation": {
                        "values": [Scalar(1), Scalar(2), Scalar(3), Scalar(4), Scalar(5)],
                        "color": "blue",
                        "marker": "o"
                    },
                    "Real": {
                        "values": [2, 4, 6, 8, 10],
                    }
            }
        >>> plot_series(xs, ys, title="Comparison")
    ````
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot data series.
    xs = np.array(xs).reshape(-1, 1)
    for label, y in ys.items():
        values = y.get("values")

        if len(values) != len(xs):
            raise ValueError(f"The length of {label} values must be equal to xs.")

        # support Scalar object.
        if isinstance(values[0], Scalar):
            values = [e.value for e in values]

        color = y.get("color", None)
        marker = y.get("marker", None)
        ax.plot(xs, values, color=color, marker=marker, label=label)

    # set plot properties.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(grid)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # set ticks.
    if xticks is not None:
        xl = xlim[0] if xlim is not None else xticks[0]
        xr = xlim[-1] if xlim is not None else xticks[-1]
        xticks = [x for x in xticks if xl <= x <= xr]
        ax.set_xticks(xticks)

    if yticks is not None:
        yl = ylim[0] if ylim is not None else yticks[0]
        yr = ylim[-1] if ylim is not None else yticks[-1]
        yticks = [y for y in yticks if yl <= y <= yr]
        ax.set_yticks(yticks)

    # save and show.
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_scalar_field(
    data: Field,
    mesh: Mesh,
    *,
    figsize=None,
    title="plot",
    axis="auto",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    grid=True,
    show=True,
    save_dir=None,
):
    """
    Plot scalar field with geometric information.

    Args:
        data: Scalar field to be plotted.
        mesh: Mesh of the scalar field.
        title: Title of the plot.
        axis: Axis to plot, options: "auto", "xy", "yz", "xz", "x", "y", "z".
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        zlabel: Label of the z-axis.
        grid: Whether to show the grid.
        show: Whether to show the plot.
        save_dir: Directory to save.

    Notes:
        + If `axis` is "auto", the function will automatically choose a plane to plot.
        + If `axis` is "xy", plot the scalar field in the xy-plane, etc.

    """
    # extract values from field.
    if not isinstance(data, Field):
        raise TypeError("data must be a Field object.")
    if data.dtype != "scalar":
        raise TypeError("data must be a scalar field.")

    values = [e.value for e in data]

    # extract geometric coordinates.
    if data.etype == "node":
        coords = [node.coordinate for node in mesh.nodes]
    elif data.etype == "cell":
        coords = [cell.coordinate for cell in mesh.cells]
    elif data.etype == "face":
        coords = [face.coordinate for face in mesh.faces]
    else:
        raise ValueError("Unsupported element type.")

    xs = [coord.x for coord in coords]
    ys = [coord.y for coord in coords]
    zs = [coord.z for coord in coords]

    # plot scalar field.
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    if axis == "auto":
        if mesh.domain == "1d":
            ax = fig.add_subplot(111)
            ax.plot(xs, values)
        elif mesh.domain == "2d":
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(xs, ys, values, cmap=plt.cm.jet)
        elif mesh.domain == "3d":
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(xs, ys, zs, values, cmap=plt.cm.jet)
    elif axis == "x":
        ax = fig.add_subplot(111)
        ax.plot(ys, values)
    elif axis == "y":
        ax = fig.add_subplot(111)
        ax.plot(xs, values)
    elif axis == "z":
        ax = fig.add_subplot(111)
        ax.plot(xs, values)
    elif axis == "xy":
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, values)
    elif axis == "yz":
        ax = fig.add_subplot(111)
        ax.plot(ys, zs, values)
    elif axis == "xz":
        ax = fig.add_subplot(111)
        ax.plot(xs, zs, values)
    else:
        raise ValueError(f"Unsupported axis choice: {axis}.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_label(zlabel)
    ax.set_title(title)
    ax.grid(grid)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()
