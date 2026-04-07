# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh visualization utilities for 2D meshs.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from yunmeng.numerics.mesh.spatials import MeshDimension, Mesh
from yunmeng.render.plotter.PlotKits import _extract_mesh_data, plot_mesh_geometry


def plot_mesh_ids(
    mesh: Mesh,
    show_nodes: bool = True,
    show_faces: bool = True,
    show_cells: bool = True,
    node_color: str = "blue",
    face_color: str = "green",
    cell_color: str = "red",
    figsize: tuple = (12, 10),
    dpi: int = 100,
    title: str = None,
    save_path: str = None,
):
    """
    Plot a 2D mesh with node, face and cell IDs labeled.
    Supports structured grids and unstructured meshes.

    Args:
        mesh: The mesh object to visualize
        show_nodes: Whether to show node IDs
        show_faces: Whether to show face IDs
        show_cells: Whether to show cell IDs
        node_color: Color for node labels
        face_color: Color for face labels
        cell_color: Color for cell labels
        figsize: Figure size (width, height)
        dpi: DPI for the figure
        title: Title for the plot
        save_path: If provided, save the figure to this path
    """
    # Check if mesh is 2D
    if mesh.dimension != MeshDimension.D2:
        raise ValueError("This function only supports 2D meshes")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Extract node coordinates
    node_coords = np.array(
        [[node.coordinate.x, node.coordinate.y] for node in mesh.nodes]
    )

    # Calculate bounds with some padding
    x_min, x_max = node_coords[:, 0].min(), node_coords[:, 0].max()
    y_min, y_max = node_coords[:, 1].min(), node_coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Draw faces (edges) - this works for both structured and unstructured meshes
    for face in mesh.faces:
        face_coords = node_coords[face.nodes]
        ax.plot(
            face_coords[:, 0],
            face_coords[:, 1],
            "k-",
            linewidth=0.5,
            alpha=0.3,
        )

    # Plot cells with IDs
    if show_cells:
        for cell_idx in range(mesh.cell_count):
            cell = mesh.cells[cell_idx]
            # Get faces of this cell
            faces = mesh.get_faces(cell.faces)

            # Get unique nodes from all faces
            node_ids = set()
            for face in faces:
                node_ids.update(face.nodes)
            node_ids = sorted(list(node_ids))

            # Get coordinates of cell nodes
            cell_coords = node_coords[node_ids]

            # Draw cell boundary
            # Sort nodes anticlockwise to form proper polygon
            from shapely.geometry import Polygon

            poly = Polygon(
                [
                    (cell_coords[i, 0], cell_coords[i, 1])
                    for i in range(len(cell_coords))
                ]
            )
            if not poly.is_valid:
                # If polygon is not valid, try to fix it
                poly = poly.buffer(0)
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                color=cell_color,
                linewidth=1.0,
                alpha=0.5,
            )

            # Draw cell center and ID
            center_x = cell.coordinate.x
            center_y = cell.coordinate.y

            # Draw a small marker at cell center
            ax.plot(
                center_x,
                center_y,
                "s",
                color=cell_color,
                markersize=8,
                alpha=0.3,
            )

            # Add cell ID label
            ax.text(
                center_x,
                center_y,
                str(cell_idx),
                color=cell_color,
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
            )

    # Plot faces with IDs
    if show_faces:
        for face_idx in range(mesh.face_count):
            face = mesh.faces[face_idx]
            # Get coordinates of face nodes
            face_coords = node_coords[face.nodes]

            # Calculate face center
            center_x = np.mean(face_coords[:, 0])
            center_y = np.mean(face_coords[:, 1])

            # Draw a small marker at face center
            ax.plot(center_x, center_y, "o", color=face_color, markersize=6, alpha=0.5)

            # Add face ID label
            ax.text(
                center_x,
                center_y,
                str(face_idx),
                color=face_color,
                fontsize=8,
                ha="center",
                va="center",
            )

    # Plot nodes with IDs
    if show_nodes:
        for node_idx in range(mesh.node_count):
            coord = node_coords[node_idx]
            # Draw node marker
            ax.plot(coord[0], coord[1], "o", color=node_color, markersize=4)
            # Add node ID label with offset
            ax.text(
                coord[0],
                coord[1],
                str(node_idx),
                color=node_color,
                fontsize=9,
                ha="left",
                va="bottom",
                fontweight="bold",
            )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"2D Mesh Visualization\n"
            f"Nodes: {mesh.node_count}, Faces: {mesh.face_count}, Cells: {mesh.cell_count}"
        )

    # Add legend
    legend_elements = []
    if show_nodes:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Nodes",
                markerfacecolor=node_color,
                markersize=8,
            )
        )
    if show_faces:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Faces",
                markerfacecolor=face_color,
                markersize=6,
            )
        )
    if show_cells:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Cells",
                markerfacecolor=cell_color,
                markersize=8,
            )
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, ax


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

    plot_mesh_geometry(
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
