# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""

from yunmeng.numerics.enums import ElementType, BackendType
from yunmeng.numerics.mesh import Mesh
from yunmeng.numerics.fields.variables import Variable, VariableType
from yunmeng.numerics.fields.backends import get_backend, ArrayLike, DeviceLike
from yunmeng.setting import settings

import numpy as np
import torch
from typing import Callable, Union, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy

# ---------------------------------------------------
# region Mesh Partition
# ---------------------------------------------------


@dataclass(slots=True)
class SharedInfo:
    """Halo communication information."""

    # Neighbor shard IDs
    neighbours: List[int] = field(default_factory=list)

    # Sender pack data: "Send my local_idx data to target_global_idx on neighbor"
    send_map: Dict[int, List[Tuple[int, int]]] = field(
        default_factory=dict
    )  # [target_shard, (local_idx, target_global_idx)]

    # Receiver unpack data: "Put data from source into my local_ghost_idxs"
    recv_map: Dict[int, List[int]] = field(
        default_factory=dict
    )  # [source_shard, local_ghost_idxs]

    # Synchronous operations where both sides own the entity (e.g., node, face)
    shared_map: Dict[int, List[int]] = field(
        default_factory=dict
    )  # [neighbour_part, shared_local_idxs]


@dataclass(slots=True)
class MeshShard:
    """Mesh shard for distributed computation."""

    shard_id: int
    device: DeviceLike

    # Local entities (global indices): [Core..., Ghost...]
    # Entity index tables are constant indices (not on the autograd graph);
    # numpy is acceptable here regardless of compute backend.
    cells: np.ndarray
    faces: np.ndarray
    nodes: np.ndarray

    # Entity indices mapping: global -> local (Owner only)
    cell_g2l_core: Dict[int, int]
    face_g2l_core: Dict[int, int]
    node_g2l_core: Dict[int, int]

    # Entity indices mapping: global -> local (Ghost only)
    cell_g2l_halo: Dict[int, int]
    face_g2l_halo: Dict[int, int]
    node_g2l_halo: Dict[int, int]

    # Halo communication info
    cell_halo: SharedInfo
    face_halo: SharedInfo
    node_halo: SharedInfo

    # Metadata: Count of core vs ghost
    n_core_cells: int
    n_core_faces: int
    n_core_nodes: int

    @property
    def n_ghost_cells(self) -> int:
        return len(self.cell_g2l_halo)

    @property
    def n_ghost_faces(self) -> int:
        return len(self.face_g2l_halo)

    @property
    def n_ghost_nodes(self) -> int:
        return len(self.node_g2l_halo)

    def get_halo_info(self, etype: ElementType) -> SharedInfo:
        if etype == ElementType.CELL:
            return self.cell_halo
        elif etype == ElementType.FACE:
            return self.face_halo
        elif etype == ElementType.NODE:
            return self.node_halo
        else:
            raise ValueError("Unsupport ElementType!")

    def get_entities(self, etype: ElementType) -> np.ndarray:
        if etype == ElementType.CELL:
            return self.cells
        elif etype == ElementType.FACE:
            return self.faces
        elif etype == ElementType.NODE:
            return self.nodes
        else:
            raise ValueError("Unsupport ElementType!")

    def get_g2l_maps(self, etype: ElementType) -> tuple:
        if etype == ElementType.CELL:
            return self.cell_g2l_core, self.cell_g2l_halo
        elif etype == ElementType.FACE:
            return self.face_g2l_core, self.face_g2l_halo
        elif etype == ElementType.NODE:
            return self.node_g2l_core, self.node_g2l_halo
        else:
            raise ValueError("Unsupport ElementType!")

    def get_sizes(self, etype: ElementType) -> tuple:
        if etype == ElementType.CELL:
            return len(self.cells), self.n_core_cells, self.n_ghost_cells
        elif etype == ElementType.FACE:
            return len(self.faces), self.n_core_faces, self.n_ghost_faces
        elif etype == ElementType.NODE:
            return len(self.nodes), self.n_core_nodes, self.n_ghost_nodes
        else:
            raise ValueError("Unsupport ElementType!")

    @staticmethod
    def from_size(
        element_size: int,
        etype: ElementType = ElementType.CELL,
        device: str = settings.device,
    ) -> "MeshShard":
        """Single shard."""
        ids = np.arange(element_size, dtype=np.int64)
        g2l = {i: i for i in range(element_size)}

        cells, faces, nodes = [], [], []
        cell_g2l, face_g2l, node_g2l = {}, {}, {}

        if etype == ElementType.CELL:
            cells = ids.copy()
            cell_g2l = g2l.copy()
        elif etype == ElementType.FACE:
            faces = ids.copy()
            face_g2l = g2l.copy()
        elif etype == ElementType.NODE:
            nodes = ids.copy()
            node_g2l = g2l.copy()
        else:
            raise ValueError("Unsupport ElementType!")

        return MeshShard(
            shard_id=0,
            device=torch.device(device),
            cells=cells,
            faces=faces,
            nodes=nodes,
            cell_g2l_core=cell_g2l,
            face_g2l_core=face_g2l,
            node_g2l_core=node_g2l,
            cell_g2l_halo={},
            face_g2l_halo={},
            node_g2l_halo={},
            cell_halo=SharedInfo(),
            face_halo=SharedInfo(),
            node_halo=SharedInfo(),
            n_core_cells=len(cells),
            n_core_faces=len(faces),
            n_core_nodes=len(nodes),
        )


# ---------------------------------------------------
# region Field Infrastruct
# ---------------------------------------------------

DataIndex = Union[int, slice, List[int], np.ndarray]
DataItem = Union[float, np.ndarray, torch.Tensor, Variable]


class HaloMode(Enum):
    """Halo synchronization operations."""

    SUM = auto()
    MAX = auto()
    MIN = auto()
    OVERWRITE = auto()


@dataclass(slots=True)
class FieldMeta:
    """Field metadata."""

    name: str = None
    version: int = 0
    size: int = None
    etype: ElementType = ElementType.CELL
    vtype: VariableType = VariableType.scalar()
    btype: BackendType = BackendType.NUMPY
    requires_grad: bool = False


@dataclass
class FieldShard:
    """Field shard for distributed computation."""

    shard_id: int
    device: DeviceLike
    data: ArrayLike  # [Core..., Ghost...]
    n_core: int

    @property
    def n_ghost(self) -> int:
        """Ghost data size"""
        return self.data.shape[0] - self.n_core

    @property
    def minmax(self) -> Tuple[float, float]:
        """Min/max values of this shard."""
        local_view = self.local_view()
        if isinstance(self.data, torch.Tensor):
            return (float(local_view.min()), float(local_view.max()))
        else:
            return (float(local_view.min()), float(local_view.max()))

    def local_view(self) -> ArrayLike:
        """Local data view"""
        return self.data[: self.n_core]

    def ghost_view(self) -> ArrayLike:
        """Ghost data view"""
        return self.data[self.n_core :]

    def zero_grad(self):
        """Clear gradients"""
        if self.data.grad is not None:
            self.data.grad.zero_()

    def to_host(self) -> ArrayLike:
        """Sync data to host."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu()
        else:
            return self.data


# ---------------------------------------------------
# region Field
# ---------------------------------------------------


class Field:
    """Distributed physical field based on mesh partition."""

    def __init__(
        self,
        mesh_shards: list[MeshShard],
        vtype: VariableType,
        etype: ElementType,
        name: str = None,
        init_val: DataItem = None,
        requires_grad: bool = False,
    ):
        # Init metadata
        self._backend = get_backend()
        self._mesh_shards = mesh_shards
        self._meta = FieldMeta(
            name=name,
            version=0,
            size=self._get_total_size(mesh_shards, etype),
            etype=etype,
            vtype=vtype,
            btype=self._backend.type,
            requires_grad=requires_grad,
        )

        # Init shards for each mesh shard
        self._shards: List[FieldShard] = []
        self._init_shards(init_val, requires_grad)

        # Init halo communication buffers
        self._halo_buffers: Dict[int, Dict] = {}
        self._init_halo_buffers()

        # Build global-shard index maps
        self._global_in_shard = None

        # Comms group (NCCL/NCCL-like)
        self._comm_group = None

        # Dirty flags for shard halo sync
        self._dirty_flags = {s.shard_id: False for s in mesh_shards}

    # --------------------------------------------------
    # region preProcessing
    # --------------------------------------------------

    def _get_total_size(self, mesh_shards, etype):
        """Get total size of field data."""
        if etype == ElementType.CELL:
            return sum([s.n_core_cells for s in mesh_shards])
        elif etype == ElementType.NODE:
            return sum([s.n_core_nodes for s in mesh_shards])
        elif etype == ElementType.FACE:
            return sum([s.n_core_faces for s in mesh_shards])
        else:
            raise ValueError(f"Invalid element type: {etype}")

    def _init_shards(self, init_val, requires_grad=True):
        """Init field shards."""
        canonical_init = self._get_init_val(init_val)
        for shard in self._mesh_shards:
            sid = shard.shard_id
            n_data, n_core, _ = shard.get_sizes(self._meta.etype)
            n_comp = self._meta.vtype.shape

            fill_shape = (n_data, *n_comp)
            data = self._backend.full(
                fill_shape,
                canonical_init,
                self._backend.float64,
                shard.device,
                requires_grad,
            )

            self._shards.append(
                FieldShard(
                    shard_id=sid,
                    device=shard.device,
                    data=data,
                    n_core=n_core,
                )
            )

    def _get_init_val(self, init_val) -> Variable:
        """Get initial value for this shard."""
        if init_val is None:
            return Variable.zeros(self._meta.vtype).data
        if isinstance(init_val, Variable):
            return init_val.data
        if isinstance(init_val, (float, np.ndarray, torch.Tensor)):
            return init_val
        return None

    def _init_halo_buffers(self):
        """Pre-allocate halo communication buffers."""
        if len(self._shards) < 2:
            return

        for shard in self._mesh_shards:
            sid = shard.shard_id
            halo_info = shard.get_halo_info(self._meta.etype)
            n_comp = self._meta.vtype.shape

            # Send/recv buffers for each neighbor
            buffers = {}
            for neighbor_id in halo_info.neighbours:
                # Send
                send_list = halo_info.send_map[neighbor_id]
                n_send = len(send_list)

                # Recv
                recv_list = halo_info.recv_map[neighbor_id]
                n_recv = len(recv_list)

                if n_send == 0 and n_recv == 0:
                    continue

                # Buffer
                buffers[neighbor_id] = {
                    "send_buf": self._backend.empty(
                        (n_send, *n_comp), device=shard.device
                    ),
                    "recv_buf": self._backend.empty(
                        (n_recv, *n_comp), device=shard.device
                    ),
                    "send_indices": [l_idx for l_idx, _ in send_list],
                    "recv_indices": recv_list,  # ghost
                }
            self._halo_buffers[sid] = buffers

    def _build_global_index_maps(self):
        """Build global index -> (shard_id, local_index) mapping."""
        self._global_in_shard = np.empty(self._meta.size, dtype=object)
        for shard in self._mesh_shards:
            sid = shard.shard_id
            g2ls = shard.cell_g2l_core
            if self._meta.etype == ElementType.NODE:
                g2ls = shard.node_g2l_core
            elif self._meta.etype == ElementType.FACE:
                g2ls = shard.face_g2l_core
            for g, l in g2ls.items():
                self._global_in_shard[g] = (sid, l)

    def _ensure_global_index_maps(self):
        """Lazy initialization of global index maps."""
        if self._global_in_shard is None:
            self._build_global_index_maps()

    # --------------------------------------------------
    # region Construction
    # --------------------------------------------------

    @staticmethod
    def from_array(
        data: ArrayLike,
        mesh_shards: list[MeshShard],
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a field from a global array.

        Uses batch copy via entity indices for reliable data mapping,
        replacing the legacy per-element loop.
        """
        mesh_size = sum([s.get_sizes(etype)[1] for s in mesh_shards])
        assert (
            data.shape[0] == mesh_size
        ), f"Data size {data.shape[0]} != mesh size {mesh_size}"

        # Canonicalize scalar field storage from legacy (N, 1) to (N,).
        if vtype.is_scalar and data.ndim == 2 and data.shape[-1] == 1:
            data = data.reshape(data.shape[0])

        field = Field(
            mesh_shards,
            vtype,
            etype,
            requires_grad=requires_grad,
        )

        # Optimized: batch copy via entity indices (like scatter_from_host)
        for sid, shard in enumerate(field._shards):
            mesh_shard = field._mesh_shards[sid]
            indices = mesh_shard.get_entities(etype)[: shard.n_core]

            local_data = field._backend.array(data[indices])
            if field._backend.is_torch:
                local_data = local_data.to(shard.device)
            shape = vtype.shape
            shard.data[: shard.n_core] = local_data.reshape((-1, *shape))

        return field

    @staticmethod
    def from_shard(
        shards: list[FieldShard], mesh_shards: list[MeshShard], meta: FieldMeta
    ) -> "Field":
        """Create a field from pre-initialized shards.

        TODO: refactor.
        """
        total_size = sum([shard.n_core for shard in shards])
        assert total_size == meta.size, "Shard size != field size"

        field = Field(
            mesh_shards,
            meta.vtype,
            meta.etype,
            requires_grad=meta.requires_grad,
        )
        field._shards = shards
        return field

    @staticmethod
    def from_size(
        size: int,
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        init_val: Variable = None,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a continuous field with specified size."""
        return Field(
            [MeshShard.from_size(size, etype)],
            vtype,
            etype,
            init_val=init_val,
            requires_grad=requires_grad,
        )

    @staticmethod
    def zeros(
        size: int,
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a zero-initialized field with specified size."""
        return Field.from_size(
            size, vtype, etype, init_val=0.0, requires_grad=requires_grad
        )

    @staticmethod
    def ones(
        size: int,
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a one-initialized field with specified size."""
        return Field.from_size(
            size, vtype, etype, init_val=1.0, requires_grad=requires_grad
        )

    @staticmethod
    def full(
        size: int,
        fill_value: float,
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a field filled with a constant value."""
        return Field.from_size(
            size, vtype, etype, init_val=fill_value, requires_grad=requires_grad
        )

    @staticmethod
    def from_grid(
        mesh: Mesh,
        vtype: VariableType = VariableType.scalar(),
        etype: ElementType = ElementType.CELL,
        init_val: DataItem = 0.0,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a Field directly from a Grid or Mesh instance."""
        size = mesh.get_element_count(etype)
        return Field.from_size(
            size, vtype, etype, init_val=init_val, requires_grad=requires_grad
        )

    def copy(self) -> "Field":
        """Copy a field by copying another field."""
        return Field.from_shard(
            [deepcopy(s) for s in self._shards],
            self._mesh_shards,
            self._meta,
        )

    def apply(self, func: Callable) -> "Field":
        """
        Apply a in-place function to each element of the field.
        (Local operation, non-communication)
        """
        # Apply func to each shard's data in-place
        for shard in self._shards:
            func(shard.data)

        # Mark all shards as dirty for halo sync
        self._mark_dirty()
        return self

    def reset_name(self, name: str):
        self._meta.name = name

    # --------------------------------------------------
    # region Properties
    # --------------------------------------------------

    @property
    def values(self) -> "ArrayLike":
        """Direct access to underlying data.

        For single-shard fields, returns the shard data directly (zero-copy).
        For multi-shard fields, returns gathered host data.
        """
        if len(self._shards) == 1:
            return self._shards[0].data
        return self.gather_to_host()

    @property
    def meta(self) -> FieldMeta:
        """Field metadata."""
        return self._meta

    @property
    def field_shards(self) -> List[FieldShard]:
        """Field shards."""
        return self._shards

    @property
    def mesh_shards(self) -> List[MeshShard]:
        """Mesh shards."""
        return self._mesh_shards

    @property
    def etype(self) -> ElementType:
        """Element type."""
        return self._meta.etype

    @property
    def vtype(self) -> VariableType:
        """Variable type."""
        return self._meta.vtype

    @property
    def shape(self) -> Tuple:
        """Field shape."""
        return (self._meta.size, *self._meta.vtype.shape)

    @property
    def size(self) -> int:
        """Field size."""
        return self._meta.size

    @property
    def minmax(self) -> Tuple[float, float]:
        ls, us = zip(*[sd.minmax for sd in self._shards])
        return (min(ls), max(us))

    def requires_grad(self, requires_grad: bool = True):
        if self._backend.type == BackendType.TORCH:
            for shard in self._shards:
                shard.data.requires_grad_(requires_grad)
            self._meta.requires_grad = requires_grad

    def gradient(self):
        """Get gradient (only for torch backend).

        NOTE: this function not work steady, need to be fixed.
        """
        if self._backend.type != BackendType.TORCH:
            raise RuntimeError("gradient() only available for torch backend")
        if not self._meta.requires_grad:
            raise RuntimeError("requires_grad=False")

        # Collect all gradients from all shards
        grads = torch.empty(
            (self._meta.size, *self._meta.vtype.shape),
            dtype=torch.float64,
        )
        self._ensure_global_index_maps()

        # Fast path: single shard, direct copy
        if len(self._shards) == 1:
            g = self._shards[0].data.grad
            if g is not None:
                grads[:] = g[: self._meta.size]
            return grads

        # Multi-shard: use index maps
        for i in range(self._meta.size):
            sid, l = self._global_in_shard[i]
            if self._shards[sid].data.grad is None:
                continue
            grads[i] = self._shards[sid].data.grad[l]
        return grads

    # --------------------------------------------------
    # region Indexing
    # --------------------------------------------------

    def __getitem__(self, indices: DataIndex) -> ArrayLike:
        # Fast path: single shard, direct access bypassing global mapping
        if len(self._shards) == 1:
            if isinstance(indices, (int, np.integer)):
                return self._shards[0].data[int(indices)]
            elif isinstance(indices, slice):
                return self._shards[0].data[indices]
            elif isinstance(indices, (list, np.ndarray)):
                return self._shards[0].data[indices]
            else:
                raise TypeError("Invalid index type.")

        # Multi-shard: use global-to-local mapping
        self._ensure_global_index_maps()
        shard_indices = self._get_shard_indices(indices)
        values = [self._shards[sid].data[l] for sid, l in shard_indices]
        if len(values) == 1:
            return values[0]
        else:
            return self._backend.stack(values)

    def __setitem__(self, indices: DataIndex, value):
        # Fast path: single shard, direct assignment
        if len(self._shards) == 1:
            self._setitem_single_shard(indices, value)
            self._mark_dirty()
            return

        # Multi-shard: use global-to-local mapping
        self._ensure_global_index_maps()
        shard_indices = self._get_shard_indices(indices)
        if isinstance(value, (float, Variable, ArrayLike)):
            value = [value]
        for (sid, l), val in zip(shard_indices, value):
            if isinstance(val, Variable):
                val = val.data
            data = self._backend.array(
                val,
                dtype=self._backend.float64,
                device=self._shards[sid].device,
            )
            # Guard against legacy (1,) scalar values being assigned to scalar slots.
            if (
                self._meta.vtype.is_scalar
                and hasattr(data, "shape")
                and data.shape == (1,)
            ):
                data = data.reshape(())
            self._shards[sid].data[l] = data
        self._mark_dirty()

    def _setitem_single_shard(self, indices: DataIndex, value):
        """Optimized setitem for single-shard fields."""
        shard = self._shards[0]
        if isinstance(value, Variable):
            value = value.data

        if isinstance(indices, (int, np.integer)):
            # Single element: wrap scalar values properly
            if self._meta.vtype.is_scalar and np.isscalar(value):
                shard.data[int(indices)] = value
                return
            data = self._backend.array(
                value, dtype=self._backend.float64, device=shard.device
            )
            if (
                self._meta.vtype.is_scalar
                and hasattr(data, "shape")
                and data.shape == (1,)
            ):
                data = data.reshape(())
            shard.data[int(indices)] = data
        elif isinstance(indices, slice):
            # Slice assignment: batch
            if indices == slice(None):
                # Full assignment: field[:] = data
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    shard.data[:] = value
                elif np.isscalar(value):
                    shard.data[:] = value
                else:
                    raise TypeError(
                        f"Unsupported value type for slice assignment: {type(value)}"
                    )
            else:
                shard.data[indices] = value
        elif isinstance(indices, (list, np.ndarray)):
            # Array indexing: batch assignment
            indices = np.asarray(indices)
            if isinstance(value, (np.ndarray, torch.Tensor)):
                shard.data[indices] = value
            elif np.isscalar(value):
                shard.data[indices] = value
            else:
                # Per-element assignment
                for idx, val in zip(indices, value):
                    self._setitem_single_shard(int(idx), val)
        else:
            raise TypeError("Invalid index type.")

    def _get_shard_indices(self, indices: DataIndex):
        # Get the global indices for this slice
        self._ensure_global_index_maps()
        if isinstance(indices, slice):
            g_indices = np.arange(
                indices.start,
                indices.stop,
                indices.step,
            )
        elif isinstance(indices, (int, np.integer)):
            g_indices = np.array([indices])
        elif isinstance(indices, list):
            g_indices = np.array(indices)
        elif isinstance(indices, np.ndarray):
            g_indices = indices
        else:
            raise TypeError("Invalid index type.")

        # Get the local indices for this slice
        l_indices = []
        for g in g_indices:
            sid, l = self._global_in_shard[g]
            l_indices.append((sid, l))
        return l_indices

    # --------------------------------------------------
    # region Operations
    # --------------------------------------------------

    def _binary_op(self, other: "Field", op: Callable) -> "Field":
        """Unified binary operation, auto-align shards."""
        if isinstance(other, Field):
            if self._mesh_shards is not other._mesh_shards:
                raise ValueError("Meshshards mismatch")
            if self._meta.vtype != other._meta.vtype:
                raise TypeError(
                    f"Binary op requires same vtype, got {self._meta.vtype} and {other._meta.vtype}"
                )
            new_shards = []
            for sid, f1 in enumerate(self._shards):
                f2 = other._shards[sid]
                new_f = FieldShard(
                    shard_id=sid,
                    device=f1.device,
                    data=op(f1.data, f2.data),
                    n_core=f1.n_core,
                )
                new_shards.append(new_f)
            return Field.from_shard(new_shards, self._mesh_shards, self._meta)
        return NotImplemented

    def __add__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a - b)

    def __iadd__(self, other: "Field") -> "Field":
        """In-place addition: self += other. Avoids creating new Field objects."""
        if isinstance(other, Field):
            if self._mesh_shards is not other._mesh_shards:
                raise ValueError("Meshshards mismatch")
            if self._meta.vtype != other._meta.vtype:
                raise TypeError(
                    f"Binary op requires same vtype, got {self._meta.vtype} and {other._meta.vtype}"
                )
            for s1, s2 in zip(self._shards, other._shards):
                s1.data += s2.data
            self._mark_dirty()
            return self
        return NotImplemented

    def __isub__(self, other: "Field") -> "Field":
        """In-place subtraction: self -= other."""
        if isinstance(other, Field):
            if self._mesh_shards is not other._mesh_shards:
                raise ValueError("Meshshards mismatch")
            if self._meta.vtype != other._meta.vtype:
                raise TypeError(
                    f"Binary op requires same vtype, got {self._meta.vtype} and {other._meta.vtype}"
                )
            for s1, s2 in zip(self._shards, other._shards):
                s1.data -= s2.data
            self._mark_dirty()
            return self
        return NotImplemented

    def _field_op(self, other: "Field") -> "Field":
        """Unified field operation with type inference, auto-align shards."""
        assert self._mesh_shards is other._mesh_shards, "Meshshards mismatch"
        xp = self._backend.xp

        result_datas, result_vtype = [], None
        for i in range(len(self._shards)):
            a = self._shards[i].data
            vtype_a = self._meta.vtype
            b = other._shards[i].data
            vtype_b = other._meta.vtype

            # --- case 1: Scalar * Any (Broadcasting) ---
            if vtype_a.is_scalar:
                shard_data = xp.multiply(a, b)
                result_vtype = vtype_b
                result_datas.append(shard_data)

            # --- case 2: Any * Scalar (Broadcasting) ---
            elif vtype_b.is_scalar:
                shard_data = xp.multiply(a, b)
                result_vtype = vtype_a
                result_datas.append(shard_data)

            # --- case 3: Vector * Vector (Dot Product) ---
            elif vtype_a.is_vector and vtype_b.is_vector:
                # dot product: (N, dim) -> (N,)
                shard_data = xp.einsum("ni,ni->n", a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.scalar()

            # --- case 4: Vector * Tensor ---
            # (N, dim) * (N, dim, dim) -> (N, dim)
            elif vtype_a.is_vector and vtype_b.is_tensor:
                shard_data = xp.einsum("ni,nij->nj", a, b)
                result_datas.append(shard_data)
                result_vtype = vtype_a

            # --- case 5: Tensor * Tensor (Element-wise) ---
            elif vtype_a.is_tensor and vtype_b.is_tensor:
                shard_data = xp.multiply(a, b)
                result_datas.append(shard_data)
                result_vtype = vtype_a

            else:
                raise ValueError(
                    f"Unsupported operation between {vtype_a} and {vtype_b}"
                )

        return self._build_field(result_datas, result_vtype)

    def _build_field(self, shard_data_list, result_vtype):
        new_shards = []
        for sid, shard in enumerate(self._shards):
            new_shards.append(
                FieldShard(
                    shard_id=sid,
                    device=shard.device,
                    data=shard_data_list[sid],
                    n_core=shard.n_core,
                )
            )
        new_meta = FieldMeta(
            version=0,
            size=self._meta.size,
            etype=self._meta.etype,
            vtype=result_vtype,
            btype=self._backend.type,
            requires_grad=self._meta.requires_grad,
        )
        return Field.from_shard(new_shards, self._mesh_shards, new_meta)

    def __mul__(self, other: Union[float, "Field"]) -> "Field":
        if isinstance(other, Field):
            return self._field_op(other)
        elif isinstance(other, (float, Variable)) or np.isscalar(other):
            if isinstance(other, Variable):
                other = other.data
            new_shards = [s.data * other for s in self._shards]
            return self._build_field(new_shards, self._meta.vtype)

    __rmul__ = __mul__

    def __imul__(self, other: Union[float, Variable]) -> "Field":
        """In-place multiplication: self *= scalar. Avoids creating new Field objects."""
        if isinstance(other, (float, Variable)) or np.isscalar(other):
            val = other.data if isinstance(other, Variable) else other
            for s in self._shards:
                s.data *= val
            self._mark_dirty()
            return self
        return NotImplemented

    def __truediv__(self, scalar: float) -> "Field":
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("Division by zero")
        return self * (1.0 / scalar)

    def __itruediv__(self, scalar: float) -> "Field":
        """In-place division: self /= scalar."""
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("Division by zero")
        return self.__imul__(1.0 / scalar)

    def __neg__(self) -> "Field":
        return self * -1.0

    def __matmul__(self, other: "Field") -> "Field":
        """Execute matrix multiplication between two Fields: A @ B
        - Tensor @ Tensor -> Tensor (std matmul)
        - Tensor @ Vector -> Vector (dot product)
        - otherwise, raise error
        """
        assert self._mesh_shards is other._mesh_shards, "Meshshards mismatch"
        xp = self._backend.xp

        result_datas, result_vtype = [], None
        for i in range(len(self._shards)):
            a = self._shards[i].data
            vtype_a = self._meta.vtype
            b = other._shards[i].data
            vtype_b = other._meta.vtype

            # Tensor @ Tensor: [N, dim, dim] @ [N, dim, dim] -> [N, dim, dim]
            if vtype_a.is_tensor and vtype_b.is_tensor:
                # (N, dim, dim) @ (N, dim, dim) -> (N, dim, dim)
                shard_data = xp.matmul(a, b)
                result_datas.append(shard_data)
                result_vtype = vtype_a

            # Vector @ Tensor: [N, dim] @ [N, dim, dim] -> [N, dim]
            elif vtype_a.is_vector and vtype_b.is_tensor:
                # need to expand vector to (N, 1, dim) for matmul:
                # (N, dim) -> (N, 1, dim)
                # (N, 1, dim) @ (N, dim, dim) -> (N, 1, dim) -> (N, dim)
                a_exp = xp.expand_dims(a, axis=1)  # (N, 1, dim)
                shard_data = xp.matmul(a_exp, b)  # (N, 1, dim)
                shard_data = xp.squeeze(shard_data, axis=1)  # (N, dim)
                result_datas.append(shard_data)
                result_vtype = vtype_a

            # Tensor @ Vector: [N, dim, dim] @ [N, dim] -> [N, dim]
            elif vtype_a.is_tensor and vtype_b.is_vector:
                # (N, dim, dim) @ (N, dim, 1) -> (N, dim, 1) -> (N, dim)
                b_exp = xp.expand_dims(b, axis=-1)  # (N, dim, 1)
                shard_data = xp.matmul(a, b_exp)  # (N, dim, 1)
                shard_data = xp.squeeze(shard_data, axis=-1)  # (N, dim)
                result_datas.append(shard_data)
                result_vtype = vtype_b

            else:
                raise TypeError(f"Invalid matmul between {vtype_a} and {vtype_b}")

        return self._build_field(result_datas, result_vtype)

    def __rmatmul__(self, other: "Field") -> "Field":
        if isinstance(other, Field):
            return other.__matmul__(self)
        return NotImplemented

    def __xor__(self, other: "Field") -> "Field":
        """Execute outer product between two Fields: A ^ B
        - Vector ^ Vector -> Tensor (P_ij = A_i * B_j)
        - otherwise, raise error
        """
        assert self._mesh_shards is other._mesh_shards, "Meshshards mismatch"
        xp = self._backend.xp

        result_datas, result_vtype = [], None
        for i in range(len(self._shards)):
            # --- Einsum notation:
            # 'n' for batch dimension, 'i', 'j', 'k' for component dimensions

            a = self._shards[i].data
            vtype_a = self._meta.vtype
            b = other._shards[i].data
            vtype_b = other._meta.vtype

            # Vector * Vector -> Tensor
            if vtype_a.is_vector and vtype_b.is_vector:
                # (N, dim) & (N, dim) -> (N, dim, dim)
                shard_data = xp.einsum("ni,nj->nij", a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.tensor(vtype_a.shape[0])
            else:
                raise TypeError(
                    f"Invalid outer product between {vtype_a} and {vtype_b}"
                )

        return self._build_field(result_datas, result_vtype)

    def __iter__(self):
        for shard in self._shards:
            for data in shard.data:
                yield data

    # --------------------------------------------------
    # region Halo sync
    # --------------------------------------------------

    def sync_halos(self, op: HaloMode = HaloMode.OVERWRITE):
        """Synchronize halo regions with neighbors."""
        # Check if need to sync
        if len(self._shards) < 2:
            return self
        if not any(self._dirty_flags.values()):
            return self

        # Sender pack data to buffer
        self._sender_pack()

        # Communicate
        self._exchange()

        # Receiver unpack data
        self._receiver_unpack(op)

        # Clear dirty flags
        self._clear_dirty()
        self._meta.version += 1
        return self

    def _sender_pack(self):
        """Pack data in shard.data to send buffers."""
        for sid, shard in enumerate(self._shards):
            if not self._dirty_flags[sid]:
                continue

            # Pack send data (local -> send buffer)
            buffers = self._halo_buffers[sid]
            for _, buf in buffers.items():
                send_idxs = buf["send_indices"]
                if self._backend.type == BackendType.TORCH:
                    send_idxs = self._backend.array(
                        send_idxs, dtype=torch.int64, device=shard.device
                    )
                    buf["send_buf"].copy_(shard.data[send_idxs])
                else:
                    buf["send_buf"] = shard.data[send_idxs]

    def _exchange(self):
        """Send/recv halo data."""
        for sid, _ in enumerate(self._shards):
            if not self._dirty_flags[sid]:
                continue

            buffers = self._halo_buffers[sid]
            for nbr_id, buf in buffers.items():
                send_buf = buf["send_buf"]

                # Find the corresponding recv buffer in the neighbor shard
                nbr_buffers = self._halo_buffers.get(nbr_id, {})
                target_buf = nbr_buffers.get(sid)
                recv_buf = target_buf["recv_buf"]

                # Direct Copy
                if isinstance(send_buf, torch.Tensor):
                    recv_buf.copy_(send_buf)
                else:
                    recv_buf[:] = send_buf

    def _receiver_unpack(self, op):
        """Unpack data from recv buffers to ghost positions."""
        for sid, shard in enumerate(self._shards):
            if not self._dirty_flags[sid]:
                continue

            buffers = self._halo_buffers[sid]
            for _, buf in buffers.items():
                recv_idxs = buf["recv_indices"]

                # Unpack recv data to ghost positions
                if self._backend.type == BackendType.TORCH:
                    recv_idxs = self._backend.array(
                        recv_idxs, dtype=torch.int64, device=shard.device
                    )
                recv_data = buf["recv_buf"]

                if op == HaloMode.SUM:
                    shard.data[recv_idxs] += recv_data
                elif op == HaloMode.MAX:
                    shard.data[recv_idxs] = self._minmax_op(op)(
                        shard.data[recv_idxs], recv_data
                    )
                elif op == HaloMode.MIN:
                    shard.data[recv_idxs] = self._minmax_op(op)(
                        shard.data[recv_idxs], recv_data
                    )
                else:
                    shard.data[recv_idxs] = recv_data

    def _minmax_op(self, op: HaloMode):
        if op == HaloMode.MAX:
            return self._backend.maximum
        elif op == HaloMode.MIN:
            return self._backend.minimum
        raise RuntimeError("Invalid halo op")

    def _mark_dirty(self):
        """Flag all shards as dirty for halo sync."""
        for sid in self._dirty_flags:
            self._dirty_flags[sid] = True

    def _clear_dirty(self):
        """Clear dirty flags after halo sync."""
        for sid in self._dirty_flags:
            self._dirty_flags[sid] = False

    # --------------------------------------------------
    # region IO operations
    # --------------------------------------------------

    def gather_to_host(self) -> np.ndarray:
        """Collect all partition data to the host global array."""
        global_size = self._meta.size
        n_comp = self._meta.vtype.shape
        etype = self._meta.etype
        global_arr = np.empty((global_size, *n_comp), dtype=np.float64)

        for sid, shard in enumerate(self._shards):
            mesh_shard = self._mesh_shards[sid]
            local_data = shard.local_view()
            local_data = self._backend.to_numpy(local_data)

            # Return the local data to the global array
            indices = mesh_shard.get_entities(etype)[: shard.n_core]
            global_arr[indices] = local_data

        return global_arr

    def scatter_from_host(self, data: np.ndarray):
        """Distribute from the host global array to each shard."""
        etype = self._meta.etype
        for sid, shard in enumerate(self._shards):
            mesh_shard = self._mesh_shards[sid]
            indices = mesh_shard.get_entities(etype)[: shard.n_core]

            local_data = self._backend.array(data[indices])
            if self._backend.is_torch:
                local_data = local_data.to(shard.device)
            shape = self._meta.vtype.shape
            shard.data[: shard.n_core] = local_data.reshape((-1, *shape))

        self._mark_dirty()

    def to_numpy(self) -> np.ndarray:
        """Convert field to a numpy array.

        For single-shard fields on CPU, returns a view (zero-copy when possible).
        For multi-shard or GPU fields, gathers and converts data.
        """
        if len(self._shards) == 1:
            data = self._shards[0].data
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return np.asarray(data)
        return self.gather_to_host()

    def to_tensor(self, **kwargs) -> torch.Tensor:
        """Convert field to a PyTorch tensor (detached, host-aggregated).

        WARNING: this path gathers to host numpy first and therefore BREAKS
        the autograd graph. For graph-preserving extraction (training),
        use `yunmeng.ai.adapters.field_to_tensor` instead.
        """
        return torch.tensor(self.gather_to_host())

    def scalarize(self) -> list["Field"]:
        """Convert the field to a list of scalar fields."""
        if self._meta.vtype.is_tensor:
            raise ValueError("Cannot scalarize a tensor field.")
        if self._meta.vtype.is_scalar:
            return [self]

        data = self.gather_to_host()
        scalar_fields = []
        for i in range(data.shape[1]):
            field = Field.from_array(
                data[:, i],
                self._mesh_shards,
                VariableType.scalar(),
                self._meta.etype,
                self._meta.requires_grad,
            )
            scalar_fields.append(field)

        return scalar_fields

    @staticmethod
    def merge(f1: "Field", f2: "Field", f3: "Field" = None) -> "Field":
        """Merge two or three scalar fields into a vector field."""
        if f1.mesh_shards != f2.mesh_shards:
            raise ValueError("Field mesh shards mismatch.")
        if f3 is not None and f3.mesh_shards != f1.mesh_shards:
            raise ValueError("Field mesh shards mismatch.")
        mesh_shards = f1.mesh_shards
        fields = [f1, f2]
        if f3 is not None:
            fields.append(f3)

        for f in fields:
            if not f._meta.vtype.is_scalar:
                raise ValueError("Only scalar fields can be merged.")
        if not all(f._meta.size == fields[0]._meta.size for f in fields):
            raise ValueError("Field size mismatch.")

        data = [f.gather_to_host() for f in fields]
        merged_data = np.stack(data, axis=-1)
        dim = merged_data.shape[-1]
        return Field.from_array(
            merged_data,
            mesh_shards,
            VariableType.vector(dim),
            f1.meta.etype,
            f1.meta.requires_grad,
        )
