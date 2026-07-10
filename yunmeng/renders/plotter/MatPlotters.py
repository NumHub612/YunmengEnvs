# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix plotter for matrix visualization.
"""

from yunmeng.numerics.linalgs import Matrix, LinearEqs
from yunmeng.renders.plotter import PlotKits

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def show_matrix_heatmap(
    matrix: Matrix,
    title: str = "Matrix",
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
    show: bool = True,
    save_dir: str = None,
):
    """
    Plot a heatmap of a matrix.

    Args:
        matrix: The matrix to be plotted.
        title: The title of the plot.
        cmap: The color map of the heatmap.
        figsize: The size of the figure.
        show: Whether to show the plot.
        save_dir: The directory to save.
    """
    mat_np = matrix.to_dense()

    PlotKits.plot_heatmap(
        mat_np, title=title, cmap=cmap, figsize=figsize, show=show, save_dir=save_dir
    )


def show_matrix_scatter(
    matrix: Matrix,
    title: str = "Matrix",
    figsize: tuple = (8, 6),
    show: bool = True,
    save_dir: str = None,
):
    """
    Plot a scatter plot of a matrix.

    Args:
        matrix: The 2d matrix to be plotted.
        title: The title of the plot.
        figsize: The size of the figure.
        show: Whether to show the plot.
        save_dir: The directory to save.
    """
    mat_np = matrix.to_dense()
    rows, cols = np.indices(mat_np.shape)

    PlotKits.plot_scatter(
        rows,
        cols,
        mat_np,
        title=title,
        figsize=figsize,
        show=show,
        save_dir=save_dir,
    )


def show_lineareqs_heatmap(
    eqs: LinearEqs,
    title: str = "Linear Equation",
    cmap: str = "viridis",
    figsize: tuple = (10, 6),
    show: bool = True,
    save_dir: str = None,
):
    """
    Plot a heatmap of a linear equation.

    Args:
        eqs: The linear equation to be plotted.
        title: The title of the plot.
        cmap: The color map of the heatmap.
        figsize: The size of the figure.
        show: Whether to show the plot.
        save_dir: The directory to save.
    """
    matrix = eqs.matrix
    mat_np = matrix.to_dense()
    rhs_np = eqs.rhs.gather_to_host()

    fig = plt.figure(figsize=figsize)
    grid_spec = fig.add_gridspec(1, 2, width_ratios=[len(rhs_np), rhs_np.shape[1]])

    ax0 = fig.add_subplot(grid_spec[0, 0])
    sns.heatmap(mat_np, annot=True, cmap=cmap, cbar=False, ax=ax0)
    ax0.set_title(f"{title}_A")

    ax1 = fig.add_subplot(grid_spec[0, 1])
    sns.heatmap(rhs_np, annot=True, cmap=cmap, cbar=False, ax=ax1)
    ax1.set_title(f"{title}_b")

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
