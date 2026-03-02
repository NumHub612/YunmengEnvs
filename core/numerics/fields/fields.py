# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""
from core.numerics.fields.variables import Variable, VariableType
from core.numerics.fields.backends import get_backend
from core.numerics.algos.parts import MeshPart
from core.numerics.enums import ElementType

import numpy as np
import torch
from typing import Callable, Union, List, Dict
from dataclasses import dataclass
from enum import Enum, auto


# --------------------------------------------------
# region Field Infrastruct
# --------------------------------------------------

DataArray = Union[np.ndarray, torch.Tensor]
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

    version: int
    size: int
    etype: ElementType
    dtype: VariableType
    backend_name: str
    requires_grad: bool = False
    unit: str = None


@dataclass
class FieldShard:
    """Field shard for distributed computation."""

    shard_id: int
    gpu: torch.device
    data: DataArray  # n_local + n_ghost
    n_local: int

    @property
    def n_ghost(self) -> int:
        """Ghost data size"""
        return self.data.shape[0] - self.n_local

    def local_view(self) -> DataArray:
        """Local data view"""
        return self.data[: self.n_local]

    def ghost_view(self) -> DataArray:
        """Ghost data view"""
        return self.data[self.n_local :]

    def zero_grad(self):
        """Clear gradients"""
        if self.data.grad is not None:
            self.data.grad.zero_()

    def to_host(self) -> DataArray:
        """Sync data to host."""
        if isinstance(self.data, torch.Tensor):
            return self.data.cpu()
        else:
            return self.data


# --------------------------------------------------
# region Field
# --------------------------------------------------


class Field:
    """Distributed physical field."""

    __slots__ = (
        "_meta",
        "_shards",
        "_backend",
        "_mesh_part",
        "_halo_buffers",
        "_global_in_shard",
        "_comm_group",
        "_dirty_flags",
    )

    def __init__(
        self,
        mesh_part: MeshPart,
        dtype: VariableType,
        etype: ElementType,
        init_val: DataItem = None,
        requires_grad: bool = False,
    ):
        # Init metadata
        self._backend = get_backend()
        self._mesh_part = mesh_part
        self._meta = FieldMeta(
            version=0,
            size=mesh_part.get_size(etype),
            etype=etype,
            dtype=dtype,
            backend_name=self._backend.name,
            requires_grad=requires_grad,
        )

        # Init shards for each mesh shard
        self._shards: Dict[int, FieldShard] = {}
        self._init_shards(init_val, requires_grad)

        # Init halo communication buffers
        self._halo_buffers: Dict[int, Dict] = {}
        self._init_halo_buffers()

        # Build global-shard index maps
        self._global_in_shard: List = []
        self._build_global_index_maps()

        # Comms group (NCCL/NCCL-like)
        self._comm_group = None

        # Dirty flags for shard halo sync
        self._dirty_flags: Dict[int, bool] = {
            sid: False for sid in range(mesh_part.num_shards)
        }

    def _init_shards(self, init_val, requires_grad: bool = True):
        """Init field shards."""
        for mesh_shard in self._mesh_part.shards:
            sid = mesh_shard.shard_id
            n_data, n_comp, n_ghost = self._get_shard_shape(sid)
            init_val = self._get_init_val(init_val)

            if self._backend.name == "torch":
                with torch.cuda.device(mesh_shard.gpu):
                    data = torch.full(
                        (n_data, *n_comp),
                        init_val,
                        dtype=torch.float64,
                        device=mesh_shard.gpu,
                        requires_grad=requires_grad,
                    )
            else:  # numpy
                data = np.full(
                    (n_data, *n_comp),
                    init_val,
                    dtype=np.float64,
                )

            self._shards[sid] = FieldShard(
                shard_id=sid,
                gpu=mesh_shard.gpu,
                data=data,
                n_local=n_data - n_ghost,
            )

    def _get_init_val(self, init_val) -> Variable:
        """Get initial value for this shard."""
        if init_val is None:
            return Variable.zero(self._meta.dtype, self._meta.requires_grad).data
        if isinstance(init_val, Variable):
            return init_val.data
        if isinstance(init_val, (float, np.ndarray, torch.Tensor)):
            return init_val

    def _get_shard_shape(self, sid) -> tuple:
        """Get the shape of mesh shard data."""
        shard = self._mesh_part.shards[sid]

        # element count depends on field type
        n_ghost = 0
        if self._meta.etype == ElementType.CELL:
            n_elements = len(shard.cells)  # local + ghost
            n_ghost = len(shard.halo_g2l)
        elif self._meta.etype == ElementType.FACE:
            n_elements = len(shard.faces)
        elif self._meta.etype == ElementType.NODE:
            n_elements = len(shard.nodes)
        else:
            n_elements = 0

        # component count depends on variable type
        n_components = self._meta.dtype.value

        # return the full shape for this shard
        return (n_elements, n_components, n_ghost)

    def _init_halo_buffers(self):
        """Pre-allocate halo communication buffers."""
        for mesh_shard in self._mesh_part.shards:
            sid = mesh_shard.shard_id
            cell_halo = mesh_shard.cell_halo
            n_components = self._meta.dtype.value

            # Pre-allocate send/recv buffers for each neighbor
            buffers = {}
            for neighbor_id in cell_halo.neighbours:
                # Send
                send_map = cell_halo.send_map[neighbor_id]
                n_send = len(send_map)

                # Recv
                recv_map = cell_halo.recv_map[neighbor_id]
                n_recv = len(recv_map)

                buffers[neighbor_id] = {
                    "send_buf": self._make_buffer(
                        n_send,
                        n_components,
                        mesh_shard.gpu,
                    ),
                    "recv_buf": self._make_buffer(
                        n_recv,
                        n_components,
                        mesh_shard.gpu,
                    ),
                    "send_indices": [local_idx for local_idx, _ in send_map],
                    "recv_indices": recv_map,  # ghost cell indices
                }
            self._halo_buffers[sid] = buffers

    def _make_buffer(self, size, shape, device):
        if self._backend.name == "torch":
            return torch.empty(
                size,
                shape,
                device=device,
                dtype=torch.float64,
            )
        else:
            return np.empty((size, shape), dtype=np.float64)

    def _build_global_index_maps(self):
        """Build global index -> (shard_id, local_index) mapping."""
        self._global_in_shard = np.empty(self._meta.size, dtype=object)
        for shard in self._mesh_part.shards:
            sid = shard.shard_id
            for g, l in shard.cell_g2l.items():
                self._global_in_shard[g] = (sid, l)

    def _get_shard_indices(self, indices: DataIndex):
        # Get the global indices for this slice
        if isinstance(indices, slice):
            g_indices = np.arange(indices.start, indices.stop, indices.step)
        elif isinstance(indices, int):
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

    def _to_data(self, value, sid):
        if isinstance(value, Variable):
            value = value.data
        if self._backend.name == "torch":
            return torch.as_tensor(
                value, dtype=torch.float64, device=self._shards[sid].gpu
            )
        else:
            return np.array(value, dtype=np.float64)

    # --------------------------------------------------
    # Property accessors
    # --------------------------------------------------

    @property
    def meta(self) -> FieldMeta:
        return self._meta

    @property
    def shards(self) -> List[FieldShard]:
        return self._shards.values()

    @property
    def etype(self) -> ElementType:
        return self._meta.etype

    @property
    def dtype(self) -> VariableType:
        return self._meta.dtype

    @property
    def size(self) -> int:
        return self._meta.size

    @property
    def mesh_part(self) -> MeshPart:
        return self._mesh_part

    # --------------------------------------------------
    # Core field operations
    # --------------------------------------------------

    @staticmethod
    def from_array(data: DataArray, mesh_part: MeshPart, meta: FieldMeta) -> "Field":
        """Create a field from a global array."""
        assert data.shape[0] == meta.size, "Data size must match field size"
        field = Field(
            mesh_part,
            meta.dtype,
            meta.etype,
            requires_grad=meta.requires_grad,
        )

        for g in range(meta.size):
            sid, l = field._global_in_shard[g]
            field._shards[sid].data[l] = data[g]
        return field

    @staticmethod
    def from_shard(
        shards: Dict[int, FieldShard], mesh_part: MeshPart, meta: FieldMeta
    ) -> "Field":
        """Create a field from pre-initialized shards."""
        total_size = sum([shard.n_local for shard in shards.values()])
        assert total_size == meta.size, "Shard sizes must sum to field size"

        field = Field(
            mesh_part,
            meta.dtype,
            meta.etype,
            requires_grad=meta.requires_grad,
        )
        field._shards = shards
        return field

    def apply(self, func: Callable) -> "Field":
        """
        Apply a function to each element of the field.
        (Local operation, non-communication)
        """
        # Apply func to each shard's data in-place
        for shard in self._shards.values():
            shard.data = func(shard.data)

        # Mark all shards as dirty for halo sync
        self._mark_dirty()
        return self

    def __getitem__(self, indices: DataIndex) -> DataArray:
        shard_indices = self._get_shard_indices(indices)
        results = [self._shards[sid].data[l] for sid, l in shard_indices]
        if len(results) == 1:
            return results[0]
        elif self._backend.name == "torch":
            return torch.stack(results)
        else:
            return np.array(results)

    def __setitem__(self, indices: DataIndex, value):
        shard_indices = self._get_shard_indices(indices)
        if isinstance(value, Variable):
            value = [value]
        for (sid, l), val in zip(shard_indices, value):
            self._shards[sid].data[l] = self._to_data(val, sid)

    def _binary_op(self, other: "Field", op: Callable) -> "Field":
        """Unified binary operation, auto-align shards."""
        if isinstance(other, Field):
            assert self._mesh_part is other._mesh_part, "MeshPart mismatch"
            # align_shards
            new_shards = {}
            for sid in self._shards.keys():
                f1 = self._shards[sid]
                f2 = other._shards[sid]
                new_f = FieldShard(sid, f1.gpu, op(f1.data, f2.data), f1.n_local)
                new_shards[sid] = new_f
            return Field.from_shard(new_shards, self._mesh_part, self._meta)

    def __add__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, scalar: float) -> "Field":
        new_shards = {}
        for sid, shard in self._shards.items():
            new_shards[sid] = FieldShard(
                sid, shard.gpu, shard.data * scalar, shard.n_local
            )
        return Field.from_shard(new_shards, self._mesh_part, self._meta)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Field":
        scalar = 1.0 / scalar if abs(scalar) > 1e-12 else 0.0
        return self * scalar

    def __neg__(self) -> "Field":
        return self * -1

    def __iter__(self):
        for sid in self._shards:
            for data in self._shards[sid].data:
                yield data

    def requires_grad(self, requires_grad: bool = True):
        if self._backend.name == "torch":
            for shard in self._shards.values():
                shard.data.requires_grad_(requires_grad)
            self._meta.requires_grad = requires_grad

    def gradient(self):
        """Get gradient (only for torch backend)."""
        if self._backend.name != "torch":
            raise RuntimeError("backward() only available for torch backend")
        if not self._meta.requires_grad:
            raise RuntimeError("requires_grad=False")

        # Collect all gradients from all shards
        grads = torch.empty(
            (self._meta.size, self._meta.dtype.value), dtype=torch.float64
        )
        for i in range(self._meta.size):
            sid, l = self._global_in_shard[i]
            grads[i] = self._shards[sid].data.grad[l]
        return grads

    # --------------------------------------------------
    # Core Halo sync
    # --------------------------------------------------

    def sync_halos(self, op: HaloMode = HaloMode.OVERWRITE):
        """
        Synchronize halo regions with neighbors.

        NOTE:
        1. Overlap computation with communication using CUDA streams
        2. Merge small messages to avoid launch overhead
        3. Lazy execution: only sync when needed
        """
        if not any(self._dirty_flags.values()):
            return self

        for sid, shard in self._shards.items():
            if not self._dirty_flags[sid]:
                continue

            buffers = self._halo_buffers[sid]
            for neighbor_id, buf in buffers.items():
                # 1. Pack send data (local cells -> send buffer)
                send_indices = torch.tensor(buf["send_indices"], device=shard.device)
                buf["send_buf"].copy_(shard.data[send_indices])

                # 2. Non-blocking send/recv
                self._send_receive(
                    send_buf=buf["send_buf"],
                    recv_buf=buf["recv_buf"],
                    dst=neighbor_id,
                    src=neighbor_id,
                )

                # 3. Unpack recv data to ghost positions
                recv_indices = buf["recv_indices"]
                shard.data[recv_indices] = buf["recv_buf"]

                # 4. Apply aggregation operation
                if op == HaloMode.SUM:
                    shard.data[recv_indices] += buf["recv_buf"]
                elif op == HaloMode.MAX:
                    shard.data[recv_indices] = torch.max(
                        shard.data[recv_indices], buf["recv_buf"]
                    )
                elif op == HaloMode.MIN:
                    shard.data[recv_indices] = torch.min(
                        shard.data[recv_indices], buf["recv_buf"]
                    )

        self._clear_dirty()
        self._version += 1
        return self

    def _send_receive(self, send_buf, recv_buf, dst, src):
        """
        Perform non-blocking send and receive using torch.distributed.
        Assumes that the current process corresponds to the shard's rank.
        """
        import torch.distributed as dist

        # Ensure tensors are on GPU and contiguous
        send_buf = send_buf.contiguous()
        recv_buf = recv_buf.contiguous()

        # Non-blocking send and receive
        send_handle = dist.isend(tensor=send_buf, dst=dst)
        recv_handle = dist.irecv(tensor=recv_buf, src=src)

        # Wait for both operations to complete
        send_handle.wait()
        recv_handle.wait()

    def _mark_dirty(self):
        """Flag all shards as dirty for halo sync."""
        for sid in self._dirty_flags:
            self._dirty_flags[sid] = True

    def _clear_dirty(self):
        """Clear dirty flags after halo sync."""
        for sid in self._dirty_flags:
            self._dirty_flags[sid] = False

    # --------------------------------------------------
    # IO operations
    # --------------------------------------------------

    def gather_to_host(self) -> np.ndarray:
        """Collect all partition data to the host global array."""
        global_size = self._meta.size
        n_comp = self._meta.dtype.value
        global_arr = np.empty((global_size, *n_comp), dtype=np.float64)

        for sid, shard in self._shards.items():
            mesh_shard = self._mesh_part.shards[sid]
            local_data = shard.local_view()
            if self._backend.name == "torch":
                local_data = local_data.cpu().numpy()

            # Return by global index
            global_indices = mesh_shard.cells[: shard.n_local]
            global_arr[global_indices] = local_data

        return global_arr

    def scatter_from_host(self, global_arr: np.ndarray):
        """Distribute from the host global array to each shard."""
        for sid, shard in self._shards.items():
            mesh_shard = self._mesh_part.shards[sid]
            local_indices = mesh_shard.cells[: shard.n_local]

            # Extract the local part and upload
            local_data = torch.from_numpy(global_arr[local_indices]).to(shard.gpu)
            shard.data[: shard.n_local] = local_data

        self._mark_dirty()

    def scalarize(self) -> list["Field"]:
        """
        Convert the field to a list of scalar fields.
        """
        if self._meta.dtype == VariableType.TENSOR:
            raise ValueError("Cannot scalarize a tensor field.")

        data = self.gather_to_host()
        scalar_fields = []
        for i in range(data.shape[1]):
            field = Field.from_array(data[:, i], self._mesh_part, self._meta)
            scalar_fields.append(field)

        return scalar_fields
