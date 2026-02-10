# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""
from core.numerics.fields.variables import Variable, VariableType, Backend
from core.numerics.fields.backends import get_backend, use_numpy, use_torch
from core.numerics.enums import ElementType
from configs.settings import settings

import numpy as np
import torch
from typing import Optional, Callable, Union, List, Dict
from dataclasses import dataclass


# --------------------------------------------------
# region Parallel Strategy
# --------------------------------------------------


@dataclass(slots=True)
class ShardInfo:
    """Record shard information, including ghost cell exchange."""

    global_size: int  # total number of cells on all GPUs.
    global_offset: int  # global offset in global array.
    local_size: int  # local number of cells.
    ghost_size: int  # number of ghost cells on the left.
    halo_sends: Dict[int, List[int]]  # send map.
    halo_recvs: Dict[int, List[int]]  # receive map.
    neighbours: List[int]  # neighbor shards.
    device: torch.device  # device of this shard storaged.

    @property
    def local_slice(self) -> slice:
        """Slice of local cells in local array."""
        return slice(
            self.ghost_size,
            self.ghost_size + self.local_size,
        )

    @property
    def global_slice(self) -> slice:
        """Slice of local cells in global array."""
        return slice(
            self.global_offset,
            self.global_offset + self.local_size,
        )

    def to_local(self, global_idx: int) -> int:
        """Convert global index to local index."""
        if global_idx < self.global_offset:
            return None
        local_idx = global_idx - self.global_offset + self.ghost_size
        if local_idx >= self.local_size:
            return None
        return local_idx

    def to_global(self, local_idx: int) -> int:
        """Convert local index to global index."""
        if local_idx >= self.ghost_size + self.local_size:
            return None
        if local_idx < self.ghost_size:
            return None
        global_idx = local_idx - self.ghost_size + self.global_offset
        return global_idx


# --------------------------------------------------
# region Field Infrastruct
# --------------------------------------------------


@dataclass(slots=True)
class FieldDesc:
    """Field metadata."""

    name: str
    etype: ElementType
    dtype: VariableType
    backend: str
    unit: str
    bc: dict


DataArray = Union[np.ndarray, torch.Tensor, List[torch.Tensor]]


class FieldData:
    """Field data container."""

    __slots__ = ("_chunks", "_shards", "_back", "_dtype", "_shape")

    def __init__(
        self,
        data: DataArray,
        dtype: VariableType,
        backend: Backend = None,
        device: str = None,
        gpus: List[torch.device] = None,
        ghost: int = 0,
    ):
        """Field data container.

        Args:
            data: Data array.
            dtype: Data type.
            device: Device to store data, [cpu, cuda].
            gpus: List of GPUs to store data.
            ghost: Number of ghost cells.
        """
        self._dtype = dtype
        self._shards: List[ShardInfo] = []

        if backend is None:
            backend = get_backend()
        if device is None:
            device = settings.device
        if gpus is None:
            gpus = settings.gpus
        self._back = backend

        if self._back.name == "numpy":
            self._init_numpy(data)
        else:
            self._init_torch(data, device, gpus, ghost)

    def _init_numpy(self, data):
        """Numpy backend: single contiguous memory."""
        # data from chunks
        if isinstance(data, list):
            data = torch.cat(
                [torch.from_numpy(d) if isinstance(d, np.ndarray) else d for d in data]
            ).numpy()

        # data from np
        if isinstance(data, np.ndarray):
            if data.shape[1:] != self._dtype.value:
                data = data.reshape(-1, *self._dtype.value)
            self._chunks = [data]
            self._shape = data.shape
            self._shards = [
                ShardInfo(
                    self._shape[0],
                    0,
                    self._shape[0],
                    0,
                    0,
                    None,
                    None,
                    torch.device("cpu"),
                )
            ]
        else:
            raise TypeError(f"numpy backend got {type(data)}")

    def _init_torch(
        self,
        data,
        device: str,
        gpus: List[torch.device],
        ghost: int,
    ):
        """Torch backend: support multi-GPU sharding."""
        # Convert to tensor
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, list):
            # validate chunks
            self._chunks = self._validate_chunks(data, gpus)
            total_size = sum(c.shape[0] for c in self._chunks)
            self._shape = (total_size, *self._dtype.value)
            return
        else:
            raise TypeError(f"Unsupport dtype: {type(data)}")

        # Reshape
        if tensor.shape[1:] != self._dtype.value:
            tensor = tensor.view(-1, *self._dtype.value)
        self._shape = tensor.shape

        # Slice to shards
        if device == "cpu" or gpus is None:  # cpu
            self._chunks = [tensor]
            self._shards = [
                ShardInfo(
                    self._shape[0],
                    0,
                    self._shape[0],
                    None,
                    None,
                    None,
                    torch.device("cpu"),
                    ghost,
                )
            ]
        elif len(gpus) == 1:  # single GPU
            self._chunks = [tensor.to(gpus[0])]
            self._shards = [
                ShardInfo(
                    self._shape[0],
                    0,
                    self._shape[0],
                    gpus[0],
                    None,
                    None,
                    None,
                    ghost,
                )
            ]
        else:  # multi-GPU
            total_size = tensor.shape[0]
            num_gpus = len(gpus)
            base_size = total_size // num_gpus
            remainder = total_size % num_gpus

            chunks = []
            offset = 0
            for i, dev in enumerate(gpus):
                # load balance
                local_size = base_size + (1 if i < remainder else 0)

                # slice chunk
                chunk = tensor[offset : offset + local_size].to(dev)
                chunks.append(chunk)
                self._shards.append(
                    ShardInfo(
                        total_size,
                        offset,
                        local_size,
                        None,
                        None,
                        None,
                        dev,
                        ghost,
                    )
                )
                offset += local_size
            self._chunks = chunks

    def _validate_chunks(self, chunks, gpus):
        """Validate chunks and gpus."""
        if len(chunks) != len(gpus):
            raise ValueError(f"Chunk count {len(chunks)} != gpu count {len(gpus)}")

        validated = []
        for ck, dev in zip(chunks, gpus):
            if isinstance(ck, np.ndarray):
                ck = torch.from_numpy(ck)
            if not isinstance(ck, torch.Tensor):
                raise TypeError(f"Chunk must be tensor, got {type(ck)}")

            if ck.shape[1:] != self._dtype.value:
                ck = ck.view(-1, *self._dtype.value)
            validated.append(ck.to(dev))
        return validated

    @property
    def backend(self) -> Backend:
        return self._back

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> VariableType:
        return self._dtype

    @property
    def chunks(self) -> List[Union[np.ndarray, torch.Tensor]]:
        return self._chunks

    @property
    def shards(self) -> List[ShardInfo]:
        return self._shards

    def to(self, device: str) -> "FieldData":
        if self._back.name == "numpy":
            if isinstance(device, str) and device.startswith("cuda"):
                # numpy -> cuda
                back = use_torch()
                _data = torch.from_numpy(self._chunks[0]).to(device)
                return FieldData(
                    _data,
                    self._dtype,
                    back,
                    "cuda",
                    [device],
                )
            return self  # cpu

        # torch backend
        if isinstance(device, str):
            device = torch.device(device)
        new_chunks = [c.to(device) for c in self._chunks]
        return FieldData(
            new_chunks,
            self._dtype,
            self._back,
            "cuda",
            [device] * len(new_chunks),
        )

    def as_numpy(self) -> np.ndarray:
        """Transfer to numpy (aggregate all shards)."""
        if self._back.name == "numpy":
            return self._chunks[0]

        # torch -> numpy
        cpu_chunks = [c.cpu() for c in self._chunks]
        return torch.cat(cpu_chunks, dim=0).numpy()

    def requires_grad(self, requires_grad: bool = True):
        """Switch on/off gradient calculation."""
        if self._back.name == "torch":
            for chunk in self._chunks:
                chunk.requires_grad_(requires_grad)
        return self

    def save(self, path: str):
        if self._back.name == "numpy":
            np.save(path, self._chunks[0])
        else:
            meta = {
                "shape": self._shape,
                "dtype": self._dtype.name,
                "backend": self._back.name,
                "shards": [
                    (s.global_size, s.global_offset, s.local_size) for s in self._shards
                ],
            }
            torch.save({"meta": meta, "chunks": self._chunks}, path)

    @staticmethod
    def load(path: str, gpus: Optional[List[torch.device]] = None) -> "FieldData":
        try:
            # try numpy format
            data = np.load(path)
            return FieldData(
                data,
                VariableType.from_shape(data.shape[1:]),
                use_numpy(),
                "cpu",
            )
        except:
            # torch format
            checkpoint = torch.load(path, map_location="cpu")
            meta = checkpoint["meta"]
            chunks = checkpoint["chunks"]
            dtype = VariableType[meta["dtype"]]

            if gpus:  # reshard
                full = torch.cat(chunks, dim=0)
                return FieldData(
                    full,
                    dtype,
                    use_torch(),
                    "cuda",
                    gpus,
                )
            return FieldData(chunks, dtype)


# --------------------------------------------------
# region Field
# --------------------------------------------------


class Field:
    """Field supporting different backends and parallel strategies."""

    __slots__ = ("_desc", "_data")

    def __init__(
        self,
        data: FieldData,
        etype: ElementType,
        name: str = "",
        unit: str = "",
        bc: dict = None,
    ):
        self._data = data
        self._desc = FieldDesc(
            name,
            etype,
            data.dtype,
            data.backend.name,
            unit,
            bc,
        )

    @staticmethod
    def zeros(
        size: int,
        dtype: VariableType,
        etype: ElementType,
        name: str = "",
        unit: str = "",
        device: str = None,
        gpus: List[Union[str, int]] = None,
    ) -> "Field":
        """Create a zero field with given size and data type."""
        # Choose backend
        if device and device.startswith("cuda"):
            backend = use_torch(device)
            gpus = [
                torch.device(f"cuda:{g}") if isinstance(g, int) else torch.device(g)
                for g in (gpus or [device])
            ]
        else:
            backend = use_numpy()
            gpus = None

        # Create data
        shape = (size, *dtype.value)
        if backend.name == "numpy":
            data = np.zeros(shape, dtype=np.float64)
        else:
            data = torch.zeros(
                shape,
                dtype=torch.float64,
                device=gpus[0],
            )
        field_data = FieldData(data, dtype, backend, device, gpus)
        return Field(field_data, etype, name, unit)

    @staticmethod
    def from_variable(
        var: Variable,
        size: int,
        etype: ElementType,
        name: str = "",
        unit: str = "",
        device: str = None,
        gpus: List[Union[str, int]] = None,
    ) -> "Field":
        """Create a field from a single variable."""
        if gpus or (device and device.startswith("cuda")):
            backend = use_torch()
            gpus = [
                torch.device(f"cuda:{g}") if isinstance(g, int) else torch.device(g)
                for g in (gpus or [device])
            ]

            if isinstance(var.data, torch.Tensor):
                base = var.data.clone().detach()
            else:
                base = torch.from_numpy(np.asarray(var.data))

            tensor = base.to(dtype=torch.float64, device=gpus[0])
            tensor = tensor.expand(size, *var.shape)
            data = FieldData(
                tensor,
                var.type,
                backend,
                "cuda",
                gpus,
            )
        else:
            backend = use_numpy()
            arr = np.broadcast_to(
                np.asarray(var.data),
                (size, *var.shape),
            ).copy()
            data = FieldData(arr, var.type, backend, "cpu")
        return Field(data, etype, name, unit)

    @property
    def desc(self) -> FieldDesc:
        return self._desc

    @property
    def name(self) -> str:
        return self._desc.name

    @property
    def etype(self) -> ElementType:
        return self._desc.etype

    @property
    def dtype(self) -> VariableType:
        return self._data.dtype

    @property
    def data(self) -> FieldData:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> Variable:
        if self._desc.backend == "numpy":
            return Variable.from_numpy(self._data.chunks[0][idx])

        for ck, info in zip(self._data.chunks, self._data.shards):
            local_idx = info.to_local(idx)
            if local_idx is not None:
                return Variable.from_numpy(
                    ck[local_idx].cpu().numpy(),
                )
        raise IndexError(f"Index {idx} out of range")

    def __setitem__(self, idx: int, value: Variable):
        if value.type != self.dtype:
            raise TypeError(f"Type mismatch: {value.type} vs {self.dtype}")

        if self._desc.backend == "numpy":
            self._data.chunks[0][idx] = value.data
        else:
            for chunk, info in zip(self._data.chunks, self._data.shards):
                local_idx = info.to_local(idx)
                if local_idx is not None:
                    with torch.no_grad():
                        chunk[local_idx] = torch.tensor(
                            value.data,
                            device=info.device,
                        )
                    return
            raise IndexError(f"Index {idx} out of range")

    def _binary_op(self, other: "Field", op: Callable) -> "Field":
        """Unified binary operation, auto-align shards."""
        if isinstance(other, Field):
            if self.dtype != other.dtype:
                raise TypeError(f"Type mismatch: {self.dtype} vs {other.dtype}")

            # align_shards
            new_chunks = []
            for c1, c2 in zip(self._data.chunks, other._data.chunks):
                new_chunks.append(op(c1, c2))

            new_data = FieldData(
                new_chunks,
                self.dtype,
                self._data.backend,
                [s.device for s in self._data.shards],
            )
            return Field(new_data, self.etype, self.name)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: "Field") -> "Field":
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, scalar: float) -> "Field":
        new_chunks = [c * scalar for c in self._data.chunks]
        new_data = FieldData(
            new_chunks,
            self.dtype,
            self._data.backend,
            [s.device for s in self._data.shards],
        )
        return Field(new_data, self.etype, self.name)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Field":
        return self * (1.0 / scalar)

    def __neg__(self) -> "Field":
        return self * -1

    def sum(self):
        """Global summation."""
        if self._desc.backend == "numpy":
            return float(self._data.chunks[0].sum())

        # torch: keep tensor，support backward
        local_sums = [c.sum() for c in self._data.chunks]
        # Sum on host, keep computation graph
        total = sum(s for s in local_sums)  # tensor
        return total

    def mean(self):
        """Global mean."""
        if self._desc.backend == "numpy":
            return float(self._data.chunks[0].mean())
        return self.sum() / self.size

    def magnitude(self) -> "Field":
        """Magnitude of a vector field."""
        if self.dtype == VariableType.SCALAR:
            return self

        if self._desc.backend == "numpy":
            new_data = np.linalg.norm(self._data.chunks[0], axis=-1, keepdims=True)
        else:
            new_chunks = [
                torch.norm(c, dim=-1, keepdim=True) for c in self._data.chunks
            ]
            new_data = FieldData(
                new_chunks,
                VariableType.SCALAR,
                self._data.backend,
                [s.device for s in self._data.shards],
            )
            return Field(new_data, self.etype, self.name)

        return Field(
            FieldData(new_data, VariableType.SCALAR),
            self.etype,
            self.name,
        )

    def requires_grad_(self, requires_grad: bool = True) -> "Field":
        self._data.requires_grad(requires_grad)
        return self

    def backward(self):
        """Backward propagation (only for torch backend)."""
        if self._desc.backend != "torch":
            raise RuntimeError("backward() only available for torch backend")

        # Collect all gradients from all shards
        grads = []
        for chunk in self._data.chunks:
            if chunk.grad is None:
                raise RuntimeError("No gradient computed.")
            grads.append(chunk.grad)
        return grads

    def to(self, device: Union[str, torch.device]) -> "Field":
        new_data = self._data.to(device)
        return Field(new_data, self.etype, self.name)

    def cpu(self) -> "Field":
        return self.to("cpu")

    def cuda(self, device: int = 0) -> "Field":
        dev = f"cuda:{device}" if device is not None else "cuda"
        return self.to(dev)

    def scalarize(self) -> list["Field"]:
        """
        Convert the field to a list of scalar fields.
        """
        if self.dtype == VariableType.SCALAR:
            return [self]

        if self.dtype == VariableType.VECTOR:
            raw_data = self._data.as_numpy()
            scalar_fields = []
            for i in range(3):
                _data = FieldData(
                    raw_data[:, i],
                    VariableType.SCALAR,
                    self._data.backend,
                    self._data.shards,
                )
                scalar_fields.append(
                    Field(
                        _data,
                        self.etype,
                        f"{self.name}_{i}",
                        self._desc.unit,
                        self._desc.bc,
                    )
                )
            return scalar_fields

        raise ValueError(f"Unsupported field type: {self.dtype}")

    def save(self, path: str):
        self._data.save(path)

    @staticmethod
    def load(path: str, gpus: List[torch.device] = None):
        data = FieldData.load(path, gpus)
        return Field(data)

    def __repr__(self) -> str:
        return f"Field({self.name}, {self.dtype.name}, {self.desc.backend}, shape={self.shape})"


# --------------------------------------------------
# region preDefines
# --------------------------------------------------


class NodeField(Field):
    """Node field."""

    def __init__(
        self,
        size: int,
        dtype: VariableType,
        name: str = "",
        bc: dict = None,
    ):
        data = FieldData(np.zeros((size, *dtype.value)), dtype, get_backend())
        super().__init__(data, ElementType.NODE, name, "", bc)


class CellField(Field):
    """Cell field."""

    def __init__(
        self,
        size: int,
        dtype: VariableType,
        name: str = "",
        bc: dict = None,
    ):
        data = FieldData(np.zeros((size, *dtype.value)), dtype, get_backend())
        super().__init__(data, ElementType.CELL, name, "", bc)


class FaceField(Field):
    """Face field."""

    def __init__(
        self,
        size: int,
        dtype: VariableType,
        name: str = "",
        bc: dict = None,
    ):
        data = FieldData(np.zeros((size, *dtype.value)), dtype, get_backend())
        super().__init__(data, ElementType.FACE, name, "", bc)


class ScalarField(Field):
    """Scalar field."""

    def __init__(
        self,
        size: int,
        etype: ElementType,
        name: str = "",
        bc: dict = None,
    ):
        data = FieldData(np.zeros((size, 1)), VariableType.SCALAR, get_backend())
        super().__init__(data, etype, name, "", bc)


class VectorField(Field):
    """Vector field."""

    def __init__(
        self,
        size: int,
        etype: ElementType,
        name: str = "",
        bc: dict = None,
    ):
        data = FieldData(np.zeros((size, 3)), VariableType.VECTOR, get_backend())
        super().__init__(data, etype, name, "", bc)
