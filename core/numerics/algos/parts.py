# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh partitioning methods.
"""
from core.numerics.mesh import Mesh
import numpy as np
import pymetis
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class PartitionInfo:
    """Partition data for a mesh partition."""

    partition_id: int

    # Local entities (global indices)
    local_cells: np.ndarray  # cells owned by this partition
    local_faces: np.ndarray  # faces owned by this partition (including ghost)
    local_nodes: np.ndarray  # nodes owned by this partition

    # Element mapping: global -> local
    cell_global_to_local: Dict[int, int]
    face_global_to_local: Dict[int, int]
    node_global_to_local: Dict[int, int]

    # Partition boundary communication data
    send_faces: Dict[
        int, List[Tuple[int, int]]
    ]  # {target_part: [(local_face_idx, target_cell_global_id), ...]}
    recv_faces: Dict[int, List[int]]  # {source_part: [local_face_idx, ...]}

    # Node sharing info
    shared_nodes: Dict[int, List[int]]  # {other_part: [local_node_idx, ..]}


class MeshPart:
    """Mesh partition assitant."""

    def __init__(self, global_mesh: Mesh):
        self.mesh = global_mesh
        self.topo = global_mesh.get_topo_assistant()
        self.geom = global_mesh.get_geom_assistant()

        self.cell_parts: Optional[np.ndarray] = None
        self.cut_edges: int = None
        self.partitions: List[PartitionInfo] = []

    @property
    def num_parts(self) -> int:
        """Number of partitions."""
        return len(self.partitions)

    def reset(self, mesh: Mesh):
        """Reset mesh."""
        self.__init__(mesh)

    def partition(self, num_parts: int) -> List[PartitionInfo]:
        """Partition the mesh."""
        # Partition cells
        self._partition_cells(num_parts)

        # Build parts
        for part_id in range(num_parts):
            part = self._build_parts(part_id)
            self.partitions.append(part)

        # Build halo info
        self._build_halos()
        return self.partitions

    def get_partition(self, part_id: int) -> PartitionInfo:
        """Get partition data."""
        return self.partitions[part_id]

    def _partition_cells(self, num_parts):
        """Partition cells."""
        # Setup partition
        options = pymetis.Options()
        options.minconn = True
        options.contig = True

        # Partition, cell in which partition
        cut_edges, membership = pymetis.part_graph(
            num_parts,
            adjacency=self.topo.cell_neighbours,
            recursive=(num_parts <= 8),
            options=options,
        )

        self.cell_parts = np.array(membership)
        self.cut_edges = cut_edges

    def _build_parts(self, part_id):
        """Build partition data."""
        # Get local cells
        local_cells = np.where(self.cell_parts == part_id)[0]
        local_cells = np.unique(local_cells)

        # Get local faces
        local_faces = []
        for fid, (c_l, c_r) in enumerate(self.topo.face_cells):
            l_in = c_l in local_cells if c_l is not None else False
            r_in = c_r in local_cells if c_r is not None else False
            if l_in or r_in:
                local_faces.append(fid)
        local_faces = np.array(local_faces)

        # Get local nodes
        local_nodes = set()
        for cell_idx in local_cells:
            for node_idx in self.topo.cell_nodes[cell_idx]:
                local_nodes.add(node_idx)
        for face_idx in local_faces:
            for node_idx in self.topo.face_nodes[face_idx]:
                local_nodes.add(node_idx)
        local_nodes = np.array(sorted(local_nodes))

        # Build global -> local maps
        cell_g2l = {g: l for l, g in enumerate(local_cells)}
        face_g2l = {g: l for l, g in enumerate(local_faces)}
        node_g2l = {g: l for l, g in enumerate(local_nodes)}

        # Build partition data
        return PartitionInfo(
            partition_id=part_id,
            local_cells=local_cells,
            local_faces=local_faces,
            local_nodes=local_nodes,
            cell_global_to_local=cell_g2l,
            face_global_to_local=face_g2l,
            node_global_to_local=node_g2l,
            send_faces={},
            recv_faces={},
            shared_nodes={},
        )

    def _build_halos(self):
        """Build halo info."""
        # Build interface faces
        interface_faces = defaultdict(list)
        for face_idx, (c_l, c_r) in enumerate(self.topo.face_cells):
            if c_l is None or c_r is None:
                continue

            p_l = self.cell_parts[c_l]
            p_r = self.cell_parts[c_r]
            if p_l != p_r:
                p1, p2 = sorted((p_l, p_r))
                interface_faces[(p1, p2)].append(face_idx)

        # Build send/recv faces
        for (p_a, p_b), faces in interface_faces.items():
            for face_idx in faces:
                c_l, c_r = self.topo.face_cells[face_idx]
                p_l, p_r = (
                    self.cell_parts[c_l],
                    self.cell_parts[c_r],
                )

                sender, receiver = (p_l, p_r) if p_l != p_r else (p_r, p_l)
                if sender == receiver:
                    continue

                send_part = self.partitions[sender]
                recv_part = self.partitions[receiver]

                local_idx = send_part.face_global_to_local[face_idx]
                target_cell_gid = c_r if sender == p_l else c_l

                if receiver not in send_part.send_faces:
                    send_part.send_faces[receiver] = []
                send_part.send_faces[receiver].append((local_idx, target_cell_gid))

                local_idx = recv_part.face_global_to_local[face_idx]
                if sender not in recv_part.recv_faces:
                    recv_part.recv_faces[sender] = []
                recv_part.recv_faces[sender].append(local_idx)

        self._sharing_nodes()

    def _sharing_nodes(self):
        """Build shared nodes."""
        node_partitions = defaultdict(set)
        for part in self.partitions:
            for idx in part.local_nodes:
                node_partitions[idx].add(part.partition_id)

        for node_idx, parts in node_partitions.items():
            if len(parts) <= 1:
                continue
            for part_id in parts:
                part = self.partitions[part_id]
                local_idx = part.node_global_to_local[node_idx]
                for other_part in parts:
                    if other_part == part_id:
                        continue
                    if other_part not in part.shared_nodes:
                        part.shared_nodes[other_part] = []
                    part.shared_nodes[other_part].append(local_idx)
