# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh visualization utilities for 2D meshs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from shapely.geometry import Polygon

from yunmeng.numerics.mesh import MeshDimension, Mesh
from yunmeng.renders.plotter.PlotKits import _extract_mesh_data, plot_mesh_geometry


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
    save_dir: str = None,
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
        save_dir: If provided, save the figure to this path
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
        if not hasattr(face, "nodes") or face.nodes is None:
            continue
        face_node_ids = np.asarray(face.nodes)
        if len(face_node_ids) < 2:
            continue
        face_coords = node_coords[face_node_ids]
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
            # Get faces of this cell (already sorted anticlockwise by mesh generator)
            faces = mesh.get_faces(cell.faces)

            # Build cell boundary by traversing faces in order
            # For each face, get its two endpoints and build a closed loop
            boundary_coords = _build_cell_boundary(faces, node_coords)

            if boundary_coords is not None and len(boundary_coords) >= 3:
                # Close the polygon
                boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
                ax.plot(
                    boundary_coords[:, 0],
                    boundary_coords[:, 1],
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
            if not hasattr(face, "nodes") or face.nodes is None:
                continue
            face_node_ids = np.asarray(face.nodes)
            if len(face_node_ids) < 2:
                continue

            # Get coordinates of face nodes
            face_coords = node_coords[face_node_ids]

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
    if save_dir:
        # Sanitize title for filename
        safe_title = str(title) if title else "mesh_visualization"
        safe_title = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in safe_title
        )
        save_path = os.path.join(save_dir, f"{safe_title}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def _build_cell_boundary(faces, node_coords):
    """
    Build a closed polygon boundary from a list of faces.
    Assumes faces are ordered and form a closed loop.

    For a quadrilateral cell with 4 faces, traces the outer boundary
    by connecting face endpoints in sequence.
    """
    if faces is None or len(faces) < 3:
        return None

    # Build edge connectivity: map each node to its neighbors in the face loop
    from collections import defaultdict

    node_neighbors = defaultdict(set)

    for face in faces:
        if not hasattr(face, "nodes") or face.nodes is None:
            continue
        node_ids = list(face.nodes)
        for i in range(len(node_ids)):
            a, b = node_ids[i], node_ids[(i + 1) % len(node_ids)]
            node_neighbors[a].add(b)
            node_neighbors[b].add(a)

    # Find a starting node (any node with neighbors)
    start_node = None
    for node, neighbors in node_neighbors.items():
        if len(neighbors) == 2:  # Corner node has exactly 2 neighbors in boundary
            start_node = node
            break

    if start_node is None:
        # Fallback: use any node
        start_node = next(iter(node_neighbors))

    # Trace the boundary
    boundary = [start_node]
    prev_node = None
    current = start_node

    while True:
        neighbors = list(node_neighbors[current])
        # Pick next node that is not the one we came from
        next_node = None
        for n in neighbors:
            if n != prev_node:
                next_node = n
                break

        if next_node is None or next_node == start_node:
            break

        boundary.append(next_node)
        prev_node = current
        current = next_node

        if len(boundary) > len(node_neighbors) * 2:  # Safety break
            break

    # Convert to coordinates
    boundary_coords = np.array([node_coords[n] for n in boundary])
    return boundary_coords


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
