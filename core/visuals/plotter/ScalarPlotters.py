# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Plotter for displaying scalar values.
"""
from core.numerics.fields import Field

import matplotlib.pyplot as plt


def plot_scalar_field(
    data: Field,
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
    save_path=None,
):
    """
    Plot scalar field.

    Args:
        data: Scalar field to be plotted.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        grid: Whether to show the grid.
        xlim: Limits of the x-axis.
        ylim: Limits of the y-axis.
        xticks: List of ticks of the x-axis.
        yticks: List of ticks of the y-axis.
        show: Whether to show the plot.
        save_path: Path to save the plot.

    Example:
    ````
        >>> from core.numerics.fields import Scalar
        >>> import numpy as np
        >>> data = Field.from_np(np.array([Scalar(i) for i in range(100)]))
        >>> plot_scalar_field(data,
                title="Scalar Field", xlabel="x", ylabel="y",
                grid=True, xlim=[10, 80], ylim=[20, 60],
                xticks=[i for i in range(10, 81, 5)],
                yticks=None,
                show=True, save_path="./scalar_field.png")
    ````
    """
    # 检查数据类型
    if not isinstance(data, Field):
        raise TypeError("data must be a Field object.")
    if data.dtype != "scalar":
        raise TypeError("data must be a scalar field.")

    values = [e.value for e in data]

    # 绘图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure()
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(grid)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if xticks is not None:
        xl = xlim[0] if xlim is not None else xticks[0]
        xr = xlim[1] if len(xlim) == 2 else xticks[-1]
        xticks = [x for x in xticks if xl <= x <= xr]
        plt.xticks(xticks)

    if yticks is not None:
        yl = ylim[0] if ylim is not None else yticks[0]
        yr = ylim[1] if len(ylim) == 2 else yticks[-1]
        yticks = [y for y in yticks if yl <= y <= yr]
        plt.yticks(yticks)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()


def plot_series(
    xs: list,
    ys: list,
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
    save_path=None,
):
    """
    Plot serial datas.

    Args:
        xs: List of x-axis values.
        ys: List of y-axis values descriptions.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        grid: Whether to show the grid.
        xlim: Limits of the x-axis.
        ylim: Limits of the y-axis.
        xticks: List of ticks of the x-axis.
        yticks: List of ticks of the y-axis.
        show: Whether to show the plot.
        save_path: Path to save the plot.

    Example:
    ````
        >>> xs = [1, 2, 3, 4, 5]
        >>> ys = [{
                    "values": [1, 2, 3, 4, 5],  # float list or Scalar list.
                    "label": "Simulation",
                    "color": "blue",
                    "marker": "o"
                    },
                    {
                    "values": [2, 4, 6, 8, 10],
                    "label": "Real",
                    "color": "red",
                    }
                ]
        >>> plot_series(xs, ys, title="Comparison")
    ````
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 检查数据类型
    plt.figure()
    for y in ys:
        values = y.get("values")
        label = y.get("label")

        if len(values) != len(xs):
            raise ValueError(f"The length of {label} values must be equal to xs.")
        if isinstance(values[0], Scalar):
            values = [e.value for e in values]

        color = y.get("color", None)
        marker = y.get("marker", None)
        plt.plot(xs, values, color=color, marker=marker, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(grid)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if xticks is not None:
        xl = xlim[0] if xlim is not None else xticks[0]
        xr = xlim[1] if len(xlim) == 2 else xticks[-1]
        xticks = [x for x in xticks if xl <= x <= xr]
        plt.xticks(xticks)

    if yticks is not None:
        yl = ylim[0] if ylim is not None else yticks[0]
        yr = ylim[1] if len(ylim) == 2 else yticks[-1]
        yticks = [y for y in yticks if yl <= y <= yr]
        plt.yticks(yticks)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()
