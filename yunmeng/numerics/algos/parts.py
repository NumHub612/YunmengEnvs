# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Mesh partitioning methods.
"""

from yunmeng.numerics.fields import MeshShard, SharedInfo
from yunmeng.numerics.mesh import Mesh, ElementType
from yunmeng.setting import settings
import numpy as np
import pymetis

from typing import List
from collections import defaultdict
import torch


def parse_gpu(gpu: str | int | None, device: str = "cuda") -> torch.device:
    """Parse single GPU device strings into torch.device object.

    Args:
        gpu (str | int | None): GPU device string or index.
        device (str, optional): Device type. Defaults to "cuda".

    Returns:
        torch.device: Parsed GPU device.
    """
    if device.lower() == "cpu":
        return torch.device("cpu")
    if gpu is None:
        return torch.device("cpu")

    if isinstance(gpu, int):
        gpu = f"cuda:{gpu}"
        return torch.device(gpu)
    elif isinstance(gpu, str):
        gpu = gpu.lower()
        return torch.device(gpu)
    else:
        raise ValueError(f"Invalid GPU specification: {gpu}")


class MeshPart:
    """Mesh partition assitant."""

    def __init__(self, global_mesh: Mesh):
        self._mesh = global_mesh
        self._topo = global_mesh.get_topo_assistant()
        self._geom = global_mesh.get_geom_assistant()

        self._cell_parts: np.ndarray = None
        self._shards: List[MeshShard] = None

    def reset(self, mesh: Mesh):
        """Reset mesh."""
        self.__init__(mesh)

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
        if etype == ElementType.CELL:
            return self._mesh.cell_count
        elif etype == ElementType.FACE:
            return self._mesh.face_count
        elif etype == ElementType.NODE:
            return self._mesh.node_count
        else:
            return None

    def partition(
        self,
        num_shards: int,
        device: str = settings.device,
        gpus: List[int] = settings.gpus,
    ) -> List[MeshShard]:
        """Run partitioning with specified devices."""
        # Check devices
        if device == "cuda" and len(gpus) < num_shards:
            raise ValueError(f"GPU count {len(gpus)} less than shard num {num_shards}")
        if device == "cpu":
            gpus = None

        # Metis partitioning
        self._cell_parts = self._part_cells(num_shards)
        self._shards = []

        # Build shards
        for sid in range(num_shards):
            shard = self._build_shard_core(sid, gpus)
            self._shards.append(shard)

        # Build halos
        self._build_halos_consistent()

        return self._shards

    def _part_cells(self, num_shards: int):
        """Partition cells."""
        options = pymetis.Options()
        options.minconn = True
        options.contig = True

        _, membership = pymetis.part_graph(
            num_shards,
            adjacency=self._topo.cell_neighbours,
            recursive=num_shards <= 8,  # Small-scale
            options=options,
        )
        return np.array(membership)

    def _build_shard_core(self, sid: int, gpus):
        """Build a shard."""
        # Get local cells
        local_cells = np.where(self.cell_parts == sid)[0]

        # Get local faces
        local_faces = []
        for fid, (cl, cr) in enumerate(self._topo.face_cells):
            # the face's owner cell is local
            if self._cell_parts[cl] == sid:
                local_faces.append(fid)
        local_faces = np.array(local_faces)

        # Get local nodes
        local_nodes = set()
        for fid in local_faces:
            local_nodes.update(self._topo.face_nodes[fid])
        local_nodes = np.array(list(local_nodes))

        # Build global -> local maps
        cell_g2l = {g: l for l, g in enumerate(local_cells)}
        face_g2l = {g: l for l, g in enumerate(local_faces)}
        node_g2l = {g: l for l, g in enumerate(local_nodes)}

        # Build shard with core info
        return MeshShard(
            shard_id=sid,
            device=parse_gpu(gpus[sid] if gpus else None),
            cells=local_cells,  # Will append ghosts later
            faces=local_faces,  # Will append ghosts later
            nodes=local_nodes,  # Will append ghosts later
            cell_g2l_core=cell_g2l,
            face_g2l_core=face_g2l,
            node_g2l_core=node_g2l,
            cell_g2l_halo={},
            face_g2l_halo={},
            node_g2l_halo={},
            cell_halo=SharedInfo(),
            face_halo=SharedInfo(),
            node_halo=SharedInfo(),
            n_core_cells=len(local_cells),
            n_core_faces=len(local_faces),
            n_core_nodes=len(local_nodes),
        )

    def _build_halos_consistent(self):
        """Build halo info consistently for Cells, Faces, and Nodes."""
        # --- Cell Halos ---
        interfaces = defaultdict(list)
        for fid, (cl, cr) in enumerate(self._topo.face_cells):
            if cr is None:
                continue
            pl, pr = self._cell_parts[cl], self._cell_parts[cr]
            if pl != pr:
                interfaces[tuple(sorted((pl, pr)))].append((fid, cl, cr))

        for (pl, pr), items in interfaces.items():
            sl, sr = self._shards[pl], self._shards[pr]
            for fid, cl, cr in items:
                # sender (owner of cell) and receiver (needs ghost)
                self._add_halo_entity(sl, sr, "cell", cl)
                self._add_halo_entity(sr, sl, "cell", cr)
                # Left cell owns the face as sender
                self._add_halo_entity(sl, sr, "face", fid)

        # --- Node Halos ---
        node_to_shards = defaultdict(set)
        for s in self._shards:
            for cid in s.cells:
                for nid in self._topo.cell_nodes[cid]:
                    node_to_shards[nid].add(s.shard_id)

        for nid, shard_ids in node_to_shards.items():
            if len(shard_ids) == 1:  # Internal node
                continue

            # Owner: Min shard ID
            owner_sid = min(shard_ids)
            for sid in shard_ids:
                s = self._shards[sid]
                if sid == owner_sid:  # Owner: Needs to be Core
                    pass
                else:  # Non-owner: Needs to be Ghost
                    if nid in s.node_g2l_core:
                        pass
                    self._add_halo_node(s, self._shards[owner_sid], "node", nid)

    def _add_halo_entity(
        self, sender: MeshShard, receiver: MeshShard, etype: str, global_id: int
    ):
        """
        Establish a Ghost relationship: Sender (Owner) -> Receiver (Ghost).
        """
        # Get references based on type
        if etype == "cell":
            r_arr = receiver.cells
            r_g2l_core = receiver.cell_g2l_core
            r_g2l_halo = receiver.cell_g2l_halo
            r_halo_info = receiver.cell_halo

            s_g2l_core = sender.cell_g2l_core
            s_halo_info = sender.cell_halo
        elif etype == "face":
            r_arr = receiver.faces
            r_g2l_core = receiver.face_g2l_core
            r_g2l_halo = receiver.face_g2l_halo
            r_halo_info = receiver.face_halo

            s_g2l_core = sender.face_g2l_core
            s_halo_info = sender.face_halo
        elif etype == "node":
            r_arr = receiver.nodes
            r_g2l_core = receiver.node_g2l_core
            r_g2l_halo = receiver.node_g2l_halo
            r_halo_info = receiver.node_halo

            s_g2l_core = sender.node_g2l_core
            s_halo_info = sender.node_halo
        else:
            raise ValueError(f"Unknown entity type: {etype}")

        # Skip if already present (either as core or ghost)
        if global_id in r_g2l_core or global_id in r_g2l_halo:
            return

        # Calculate new local index for the ghost entity
        ghost_idx = len(r_arr)

        # Extend Receiver's physical array
        if etype == "cell":
            receiver.cells = np.append(receiver.cells, global_id)
        elif etype == "face":
            receiver.faces = np.append(receiver.faces, global_id)
        elif etype == "node":
            receiver.nodes = np.append(receiver.nodes, global_id)

        # Update Receiver's Halo Map
        r_g2l_halo[global_id] = ghost_idx

        # Sender must own this entity (should be in its Core map)
        if global_id not in s_g2l_core:
            raise RuntimeError(
                f"Logic Error: Sender Shard {sender.shard_id} does not own {etype} {global_id}. "
                f"Cannot establish halo from non-owner."
            )

        # Update Sender's SEND map in ordered
        s_local_idx = s_g2l_core[global_id]
        s_halo_info.send_map.setdefault(receiver.shard_id, []).append(
            (s_local_idx, global_id)
        )

        # Update Receiver's RECV map
        r_halo_info.recv_map.setdefault(sender.shard_id, []).append(ghost_idx)

        # Update Neighbour lists
        if receiver.shard_id not in s_halo_info.neighbours:
            s_halo_info.neighbours.append(receiver.shard_id)
        if sender.shard_id not in r_halo_info.neighbours:
            r_halo_info.neighbours.append(sender.shard_id)

    def _add_halo_node(
        self, receiver: MeshShard, sender: MeshShard, etype: str, global_id: int
    ):
        """Establish a Ghost relationship for nodes."""
        if etype == "node":
            if global_id in receiver.node_g2l_core:
                pass

        # Just record the dependency in shared_map for sync
        li = receiver.node_g2l_core.get(global_id)
        if li is None:
            li = receiver.node_g2l_halo.get(global_id)
        if li is not None:
            receiver.node_halo.shared_map.setdefault(sender.shard_id, []).append(li)
