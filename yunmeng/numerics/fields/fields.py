# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""

from yunmeng.numerics.fields.variables import Variable, VariableType
from yunmeng.numerics.fields.backends import get_backend
from yunmeng.numerics.algos.parts import MeshShard
from yunmeng.numerics.enums import ElementType, BackendType

import numpy as np
import torch
from typing import Callable, Union, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from copy import deepcopy

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

    version: int = 0
    size: int = None
    etype: ElementType = ElementType.CELL
    vtype: VariableType = VariableType.SCALAR
    btype: BackendType = BackendType.NUMPY
    requires_grad: bool = False
    field: str = None


@dataclass
class FieldShard:
    """Field shard for distributed computation."""

    shard_id: int
    gpu: torch.device
    data: DataArray  # [Core..., Ghost...]
    n_core: int

    @property
    def n_ghost(self) -> int:
        """Ghost data size"""
        return self.data.shape[0] - self.n_core

    @property
    def minmax(self) -> Tuple[Variable, Variable]:
        """Min/max values of this shard."""
        local_view = self.local_view()
        if isinstance(self.data, torch.Tensor):
            return (
                local_view.min().item(),
                local_view.max().item(),
            )
        else:
            return (
                local_view.min(),
                local_view.max(),
            )

    def local_view(self) -> DataArray:
        """Local data view"""
        return self.data[: self.n_core]

    def ghost_view(self) -> DataArray:
        """Ghost data view"""
        return self.data[self.n_core :]

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
    """Distributed physical field based on mesh partition."""

    def __init__(
        self,
        mesh_shards: list[MeshShard],
        vtype: VariableType,
        etype: ElementType,
        init_val: DataItem = None,
        requires_grad: bool = False,
    ):
        # Init metadata
        self._backend = get_backend()
        self._mesh_shards = mesh_shards
        self._meta = FieldMeta(
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
        self._global_in_shard = []
        self._build_global_index_maps()

        # Comms group (NCCL/NCCL-like)
        self._comm_group = None

        # Dirty flags for shard halo sync
        self._dirty_flags = {s.shard_id: False for s in mesh_shards}

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
        for shard in self._mesh_shards:
            sid = shard.shard_id
            n_data, n_core, _ = shard.get_sizes(self._meta.etype)
            n_comp = self._meta.vtype.value

            fill_shape = (n_data, *n_comp)
            init_val = self._get_init_val(init_val)
            data = self._backend.full(
                fill_shape, init_val, self._backend.float64, shard.gpu, requires_grad
            )

            self._shards.append(
                FieldShard(
                    shard_id=sid,
                    gpu=shard.gpu,
                    data=data,
                    n_core=n_core,
                )
            )

    def _get_init_val(self, init_val) -> Variable:
        """Get initial value for this shard."""
        if init_val is None:
            return Variable.zero(self._meta.vtype, self._meta.requires_grad).data
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
            n_comp = self._meta.vtype.value

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
                        (n_send, *n_comp), device=shard.gpu
                    ),
                    "recv_buf": self._backend.empty(
                        (n_recv, *n_comp), device=shard.gpu
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

    # --------------------------------------------------
    # region Properties
    # --------------------------------------------------

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
        return (self._meta.size, *self._meta.vtype.value)

    @property
    def size(self) -> int:
        """Field size."""
        return self._meta.size

    @property
    def minmax(self) -> Tuple[Variable, Variable]:
        """Field min/max values."""
        ls, us = zip(*[sd.minmax for sd in self._shards])
        return min(ls), max(us)

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
            (self._meta.size, *self._meta.vtype.value),
            dtype=torch.float64,
        )
        for i in range(self._meta.size):
            sid, l = self._global_in_shard[i]
            if self._shards[sid].data.grad is None:
                continue
            grads[i] = self._shards[sid].data.grad[l]
        return grads

    # --------------------------------------------------
    # region Construction
    # --------------------------------------------------

    @staticmethod
    def from_array(
        data: DataArray,
        mesh_shards: list[MeshShard],
        vtype: VariableType = VariableType.SCALAR,
        etype: ElementType = ElementType.CELL,
        requires_grad: bool = False,
    ) -> "Field":
        """Create a field from a global array."""
        mesh_size = sum([s.get_sizes(etype)[1] for s in mesh_shards])
        assert data.shape[0] == mesh_size, "Data size != mesh size"
        field = Field(
            mesh_shards,
            vtype,
            etype,
            requires_grad=requires_grad,
        )

        for g in range(mesh_size):
            sid, l = field._global_in_shard[g]
            field._shards[sid].data[l] = data[g]
        return field

    @staticmethod
    def from_shard(
        shards: list[FieldShard], mesh_shards: list[MeshShard], meta: FieldMeta
    ) -> "Field":
        """Create a field from pre-initialized shards."""
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
        vtype: VariableType = VariableType.SCALAR,
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

    def __getitem__(self, indices: DataIndex) -> DataArray:
        shard_indices = self._get_shard_indices(indices)
        values = [self._shards[sid].data[l] for sid, l in shard_indices]
        if len(values) == 1:
            return values[0]
        else:
            return self._backend.stack(values)

    def __setitem__(self, indices: DataIndex, value):
        shard_indices = self._get_shard_indices(indices)
        if isinstance(value, (float, Variable)):
            value = [value]
        for (sid, l), val in zip(shard_indices, value):
            if isinstance(val, Variable):
                val = val.data
            data = self._backend.data(
                val,
                dtype=self._backend.float64,
                gpu=self._shards[sid].gpu,
            )
            self._shards[sid].data[l] = data

    def _get_shard_indices(self, indices: DataIndex):
        # Get the global indices for this slice
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
            assert self._mesh_shards is other._mesh_shards, "Meshshards mismatch"
            # align_shards
            new_shards = []
            for sid, f1 in enumerate(self._shards):
                f2 = other._shards[sid]
                new_f = FieldShard(
                    shard_id=sid,
                    gpu=f1.gpu,
                    data=op(f1.data, f2.data),
                    n_core=f1.n_core,
                )
                new_shards.append(new_f)
            return Field.from_shard(new_shards, self._mesh_shards, self._meta)

    def __add__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a - b)

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
            if vtype_a == VariableType.SCALAR:
                shard_data = xp.multiply(a, b)
                result_vtype = vtype_b
                result_datas.append(shard_data)

            # --- case 2: Any * Scalar (Broadcasting) ---
            elif vtype_b == VariableType.SCALAR:
                shard_data = xp.multiply(a, b)
                result_vtype = vtype_a
                result_datas.append(shard_data)

            # --- case 3: Vector * Vector (Dot Product) ---
            elif vtype_a == VariableType.VECTOR and vtype_b == VariableType.VECTOR:
                # dot product: (N, 3) -> (N, 1)
                shard_data = xp.einsum("ni,ni->n", a, b)
                # make it (N, 1) for consistent storage
                shard_data = shard_data[..., np.newaxis]
                result_datas.append(shard_data)
                result_vtype = VariableType.SCALAR

            # --- case 4: Vector * Tensor ---
            # (N, 3) * (N, 3, 3) -> (N, 3)
            elif vtype_a == VariableType.VECTOR and vtype_b == VariableType.TENSOR:
                shard_data = xp.einsum("ni,nij->nj", a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.VECTOR

            # --- case 5: Tensor * Tensor (Element-wise) ---
            elif vtype_a == VariableType.TENSOR and vtype_b == VariableType.TENSOR:
                shard_data = xp.multiply(a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.TENSOR

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
                    gpu=shard.gpu,
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

            # Tensor @ Tensor: [N, 3, 3] @ [N, 3, 3] -> [N, 3, 3]
            if vtype_a == VariableType.TENSOR and vtype_b == VariableType.TENSOR:
                # (N, 3, 3) @ (N, 3, 3) -> (N, 3, 3)
                shard_data = xp.matmul(a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.TENSOR

            # Vector @ Tensor: [N, 3] @ [N, 3, 3] -> [N, 3]
            elif vtype_a == VariableType.VECTOR and vtype_b == VariableType.TENSOR:
                # need to expand vector to (N, 1, 3) for matmul:
                # (N, 3) -> (N, 1, 3)
                # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3) -> (N, 3)
                a_exp = xp.expand_dims(a, axis=1)  # (N, 1, 3)
                shard_data = xp.matmul(a_exp, b)  # (N, 1, 3)
                shard_data = xp.squeeze(shard_data, axis=1)  # (N, 3)
                result_datas.append(shard_data)
                result_vtype = VariableType.VECTOR

            # Tensor @ Vector: [N, 3, 3] @ [N, 3] -> [N, 3]
            elif vtype_a == VariableType.TENSOR and vtype_b == VariableType.VECTOR:
                # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
                b_exp = xp.expand_dims(b, axis=-1)  # (N, 3, 1)
                shard_data = xp.matmul(a, b_exp)  # (N, 3, 1)
                shard_data = xp.squeeze(shard_data, axis=-1)  # (N, 3)
                result_datas.append(shard_data)
                result_vtype = VariableType.VECTOR

            else:
                raise TypeError(f"Invalid matmul between {vtype_a} and {vtype_b}")

        return self._build_field(result_datas, result_vtype)

    __rmatmul__ = __matmul__

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
            if vtype_a == VariableType.VECTOR and vtype_b == VariableType.VECTOR:
                # (N, 3) & (N, 3) -> (N, 3, 3)
                shard_data = xp.einsum("ni,nj->nij", a, b)
                result_datas.append(shard_data)
                result_vtype = VariableType.TENSOR
            else:
                raise TypeError(
                    f"Invalid outer product between {vtype_a} and {vtype_b}"
                )

        return self._build_field(result_datas, result_vtype)

    def __truediv__(self, scalar: float) -> "Field":
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("Division by zero")
        return self * (1.0 / scalar)

    def __neg__(self) -> "Field":
        return self * -1.0

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
                        send_idxs, dtype=torch.int64, gpu=shard.gpu
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
                        recv_idxs, dtype=torch.int64, gpu=shard.gpu
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
        if self._backend.type == BackendType.TORCH:
            if op == HaloMode.MAX:
                return torch.max
            elif op == HaloMode.MIN:
                return torch.min
        else:
            if op == HaloMode.MAX:
                return np.maximum
            elif op == HaloMode.MIN:
                return np.minimum
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
        n_comp = self._meta.vtype.value
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

            # Extract the local part and upload
            local_data = torch.from_numpy(data[indices]).to(shard.gpu)
            shape = self._meta.vtype.value
            shard.data[: shard.n_core] = local_data.view((-1, *shape))

        self._mark_dirty()

    def scalarize(self) -> list["Field"]:
        """Convert the field to a list of scalar fields."""
        if self._meta.vtype == VariableType.TENSOR:
            raise ValueError("Cannot scalarize a tensor field.")

        data = self.gather_to_host()
        scalar_fields = []
        for i in range(data.shape[1]):
            field = Field.from_array(
                data[:, i],
                self._mesh_shards,
                VariableType.SCALAR,
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
            if f._meta.vtype != VariableType.SCALAR:
                raise ValueError("Only scalar fields can be merged.")
        if not all(f._meta.size == fields[0]._meta.size for f in fields):
            raise ValueError("Field size mismatch.")

        data = [f.gather_to_host() for f in fields]
        merged_data = np.stack(data, axis=-1)
        return Field.from_array(
            merged_data,
            mesh_shards,
            VariableType.VECTOR,
            f1.meta.etype,
            f1.meta.requires_grad,
        )
