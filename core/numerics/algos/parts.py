# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh partitioning methods.
"""
from core.numerics.mesh import Mesh, ElementType
from core.utils.ParseGpu import parse_gpu
from configs.settings import settings
import numpy as np
import pymetis
import torch

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass(slots=True)
class SharedInfo:
    """Halo communication information."""

    neighbours: List[int] = field(default_factory=list)  # Neighbor shard IDs
    send_map: Dict[int, List[Tuple[int, int]]] = field(
        default_factory=dict
    )  # [target_shard, (local_idx, target_global_idx)]
    recv_map: Dict[int, List[int]] = field(
        default_factory=dict
    )  # [source_shard, local_ghost_idxs], for unpack
    shared_map: Dict[int, List[int]] = field(
        default_factory=dict
    )  # [neighbour_part, shared_local_idxs], for sync


@dataclass(slots=True)
class MeshShard:
    """Mesh shard for distributed computation."""

    shard_id: int
    gpu: torch.device

    # Local entities (global indices)
    cells: np.ndarray  # ghost cells at the end
    faces: np.ndarray
    nodes: np.ndarray

    # Entity mapping:global -> local
    cell_g2l: Dict  # no ghost cells
    halo_g2l: Dict  # ghost cells
    face_g2l: Dict
    node_g2l: Dict

    # 3-levels halo information
    cell_halo: SharedInfo
    face_halo: SharedInfo
    node_halo: SharedInfo


class MeshPart:
    """Mesh partition assitant."""

    def __init__(self, global_mesh: Mesh):
        self._mesh = global_mesh
        self._topo = global_mesh.get_topo_assistant() if global_mesh else None
        self._geom = global_mesh.get_geom_assistant() if global_mesh else None

        self._cell_parts: np.ndarray = None
        self._shards: List[MeshShard] = None

    def reset(self, mesh: Mesh):
        """Reset mesh."""
        self.__init__(mesh)

    @staticmethod
    def MiniPart(element_size: int, device: str = settings.device) -> "MeshShard":
        """Return a mini partition without partition."""
        shard = MeshShard(
            shard_id=0,
            gpu=torch.device(device),
            cells=np.arange(element_size),
            faces=np.arange(element_size),
            nodes=np.arange(element_size),
            cell_g2l={i: i for i in range(element_size)},
            halo_g2l={},
            face_g2l={i: i for i in range(element_size)},
            node_g2l={i: i for i in range(element_size)},
            cell_halo=SharedInfo(),
            face_halo=SharedInfo(),
            node_halo=SharedInfo(),
        )
        part = MeshPart(None)
        part._shards = [shard]
        return part

    @property
    def num_shards(self) -> int:
        """Return number of shards."""
        if self._shards is None:
            self.partition(max(len(settings.gpus), 1))
        return len(self._shards)

    @property
    def cell_parts(self) -> np.ndarray:
        """Return cell partitions."""
        if self._cell_parts is None:
            self.partition(max(len(settings.gpus), 1))
        return self._cell_parts

    @property
    def shards(self) -> List[MeshShard]:
        """Return all shards."""
        if self._shards is None:
            self.partition(max(len(settings.gpus), 1))
        return self._shards

    def get_size(self, etype: ElementType) -> int:
        """Return the size of a given entity type."""
        if self._mesh is None:
            if self._shards is None:
                return 0
            else:
                return len(self._shards[0].cells)

        if etype == ElementType.CELL:
            return self._mesh.cell_count
        elif etype == ElementType.FACE:
            return self._mesh.face_count
        elif etype == ElementType.NODE:
            return self._mesh.node_count
        else:
            raise ValueError(f"Unknown entity type {etype}")

    def partition(
        self,
        num_shards: int,
        device: str = settings.device,
        gpus: List[int | str] = settings.gpus,
    ) -> List[MeshShard]:
        """Run partitioning with devices."""
        if self._mesh is None:
            return self._shards or None

        # Check devices
        if device == "cuda" and len(gpus) < num_shards:
            raise ValueError(f"GPU count {len(gpus)} less than shard num {num_shards}.")
        if device == "cpu":
            gpus = None

        # Metis partitioning
        self._cell_parts = self._part_cells(num_shards)
        self._shards = []

        # Build shards
        for sid in range(num_shards):
            shard = self._build_shard(sid, gpus)
            self._shards.append(shard)

        # Build halos
        self._build_halos()

        return self._shards

    def _part_cells(self, num_shards: int) -> np.ndarray:
        """Partition cells."""
        options = pymetis.Options()
        options.minconn = True
        options.contig = True

        adjacency = self._topo.cell_neighbours
        recursive = num_shards <= 8
        _, membership = pymetis.part_graph(
            num_shards,
            adjacency=adjacency,
            recursive=recursive,  # Small-scale partition
            options=options,
        )
        return np.array(membership)

    def _build_shard(self, sid: int, gpus: List[int | str]) -> MeshShard:
        """Build a shard."""
        # Get local cells
        local_cells = np.where(self.cell_parts == sid)[0]
        local_cells = np.unique(local_cells)

        # Get local faces
        local_faces = []
        for fid, (cl, cr) in enumerate(self._topo.face_cells):
            li = cl is not None and self._cell_parts[cl] == sid
            ri = cr is not None and self._cell_parts[cr] == sid
            if li or ri:
                local_faces.append(fid)
        local_faces = np.array(local_faces)

        # Get local nodes
        local_nodes = set()
        for cid in local_cells:
            local_nodes.update(self._topo.cell_nodes[cid])
        for fid in local_faces:
            local_nodes.update(self._topo.face_nodes[fid])
        local_nodes = np.array(sorted(local_nodes))

        # Build global -> local maps
        cell_g2l = {g: l for l, g in enumerate(local_cells)}
        face_g2l = {g: l for l, g in enumerate(local_faces)}
        node_g2l = {g: l for l, g in enumerate(local_nodes)}

        # Set GPU device
        gpu = gpus[sid] if gpus is not None else None
        gpu = parse_gpu(gpu)

        return MeshShard(
            shard_id=sid,
            gpu=gpu,
            cells=local_cells,
            faces=local_faces,
            nodes=local_nodes,
            cell_g2l=cell_g2l,
            face_g2l=face_g2l,
            node_g2l=node_g2l,
            halo_g2l={},
            cell_halo=SharedInfo(),
            face_halo=SharedInfo(),
            node_halo=SharedInfo(),
        )

    def _build_halos(self):
        """Build halo info."""
        # Get interfaces
        interfaces = defaultdict(list)
        for fid, (cl, cr) in enumerate(self._topo.face_cells):
            if cr is None:  # Boundary face
                continue
            pl = self._cell_parts[cl]
            pr = self._cell_parts[cr]
            if pl != pr:  # Interface face
                ps = tuple(sorted((pl, pr)))
                interfaces[ps].append((fid, cl, cr, pl, pr))

        # Cell/Face Halo
        for (pa, pb), items in interfaces.items():
            sa, sb = self._shards[pa], self._shards[pb]
            for fid, cl, cr, pl, pr in items:
                # pa own the cl, pb own the cr
                if pl == pa:
                    self._add_cell_halo(sa, sb, fid, cl)
                    self._add_cell_halo(sb, sa, fid, cr)
                    self._add_face_halo(sa, sb, fid)
                else:
                    self._add_cell_halo(sb, sa, fid, cl)
                    self._add_cell_halo(sa, sb, fid, cr)
                    self._add_face_halo(sb, sa, fid)

        # Node Halo
        self._add_node_halo()

    def _add_cell_halo(
        self, sender: MeshShard, receiver: MeshShard, fid: int, ghost_cell: int
    ):
        """Add cell halo entry."""
        # Allocate ghost cell in receiver
        ghost_idx = len(receiver.cells) + len(receiver.halo_g2l)
        receiver.halo_g2l[ghost_cell] = ghost_idx
        receiver.cells = np.append(receiver.cells, ghost_cell)

        # Recorder halo mapping
        sender.cell_halo.send_map.setdefault(receiver.shard_id, []).append(
            (sender.cell_g2l[ghost_cell], ghost_cell)
        )
        receiver.cell_halo.recv_map.setdefault(
            sender.shard_id,
            [],
        ).append(ghost_idx)

        # Update neighbor lists
        if receiver.shard_id not in sender.cell_halo.neighbours:
            sender.cell_halo.neighbours.append(receiver.shard_id)
        if sender.shard_id not in receiver.cell_halo.neighbours:
            receiver.cell_halo.neighbours.append(sender.shard_id)

    def _add_face_halo(self, s1: MeshShard, s2: MeshShard, fid: int):
        """Add face halo entry."""
        for a, b in [(s1, s2), (s2, s1)]:
            a.face_halo.shared_map.setdefault(b.shard_id, []).append(a.face_g2l[fid])

    def _add_node_halo(self):
        """Add node halo entry."""
        if self.num_shards <= 1:
            return

        node_parts = defaultdict(set[int])
        for s in self._shards:
            for nid in s.nodes:
                node_parts[nid].add(s.shard_id)

        for nid, parts in node_parts.items():
            for sid in parts:
                s = self._shards[sid]
                li = s.node_g2l[nid]
                for other in parts:
                    if other != sid:
                        s.node_halo.shared_map.setdefault(
                            other,
                            [],
                        ).append(li)
