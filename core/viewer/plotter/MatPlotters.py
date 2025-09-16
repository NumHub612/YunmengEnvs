# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix plotter for matrix visualization.
"""
from core.numerics.mats import Matrix
from core.viewer.plotter import PlotKits

import numpy as np


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
    matrix = matrix.scalarize()[0]
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
    matrix = matrix.scalarize()[0]
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
