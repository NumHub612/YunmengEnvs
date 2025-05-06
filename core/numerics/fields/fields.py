# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""
from core.numerics.fields.variables import (
    Variable,
    VariableType,
    Scalar,
    Vector,
    Tensor,
)
from configs.settings import settings

import numpy as np
import torch
import enum


class ElementType(enum.Enum):
    CELL = "cell"
    FACE = "face"
    NODE = "node"
    NONE = "none"


class Field:
    """
    Abstract field class, support multi-GPU data storage.
    """

    def __init__(
        self,
        size: int,
        element_type: ElementType,
        data_type: VariableType,
        data: Variable | np.ndarray | torch.Tensor | list[torch.Tensor] = None,
        variable: str = "none",
        device: torch.device = None,
        gpus: list[str | torch.device] = None,
    ):
        """
        Initialize the field.

        Args:
            size: The number of variables in the field.
            element_type: The element type.
            data_type: The data type.
            data: The initial data of the field.
            variable: The variable name for the field.
            device: The device to store the data.
            gpus: A list of devices, e.g., ["cuda:0",].
        """
        assert element_type in ElementType, f"Invalid element type: {element_type}"
        assert data_type in VariableType, f"Invalid data type: {data_type}"

        self._variable = variable
        self._etype = element_type
        self._dtype = data_type
        self._size = size
        self._device = device or settings.DEVICE

        if gpus is not None:
            # Check if all devices are valid
            for dev in gpus:
                if isinstance(dev, str):
                    assert dev.startswith("cuda:"), f"Invalid device: {dev}"
                    assert (
                        int(dev.split(":")[1]) < torch.cuda.device_count()
                    ), f"Device {dev} not available"
                elif isinstance(dev, torch.device):
                    assert dev.type == "cuda", f"Only CUDA devices are supported: {dev}"
            self._gpus = [torch.device(dev) for dev in gpus]
        else:
            # Use the default GPU devices
            if self._device.type == "cuda":
                gpus = [f"cuda:{i}" for i in settings.GPUs]
            else:
                gpus = []
            self._gpus = gpus

        if data is None:
            type_map = {
                VariableType.SCALAR: Scalar.zero().data,
                VariableType.VECTOR: Vector.zero().data,
                VariableType.TENSOR: Tensor.zero().data,
            }
            default = type_map[data_type]
            values = torch.full((size, *default.shape), 0.0, dtype=settings.DTYPE)
            values[:] = default.data
        else:
            if isinstance(data, Variable):
                if data.type != data_type:
                    raise ValueError(f"Invalid data type: {data.type}")
                values = torch.full((size, *data.shape), 0.0, dtype=settings.DTYPE)
                values[:] = data.data
            elif isinstance(data, np.ndarray):
                if data.shape[0] != size:
                    raise ValueError(f"Invalid data shape: {data.shape}")
                values = torch.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                if data.shape[0] != size:
                    raise ValueError(f"Invalid data shape: {data.shape}")
                values = data
            elif isinstance(data, list):
                if len(data) > 1 and len(data) != len(gpus):
                    raise ValueError(f"Invalid number of GPUs: {len(gpus)}")
                num = sum(d.shape[0] for d in data)
                if num != size:
                    raise ValueError(f"Invalid multi-data size: {num}")
                values = data
            else:
                raise TypeError(f"Invalid data type: {type(data)}")

        if isinstance(values, list):
            # Field tensor data checked above
            self._values = values
        elif self._device.type == "cuda" and len(self._gpus) > 1:
            # Split the data across GPUs
            self._values = torch.chunk(values, len(self._gpus), dim=0)
            self._values = [v.to(dev) for v, dev in zip(self._values, self._gpus)]
        else:
            # Single GPU or CPU
            self._values = [values.to(device)]

    # -----------------------------------------------
    # --- Properties ---
    # -----------------------------------------------

    @property
    def data(self) -> list[torch.Tensor]:
        """
        Get the raw data of the field.
        """
        return self._values

    @property
    def variable(self) -> str:
        """
        Get the variable name of the field.
        """
        return self._variable

    @variable.setter
    def variable(self, value: str):
        """
        Set the variable name of the field.
        """
        self._variable = value

    @property
    def size(self) -> int:
        """
        Get the field total size.
        """
        return self._size

    @property
    def chunks(self) -> int:
        """
        Get the number of chunks.
        """
        return len(self._values)

    @property
    def dtype(self) -> VariableType:
        """
        Get the field data type.
        """
        return self._dtype

    @property
    def etype(self) -> ElementType:
        """
        Get the field element type.
        """
        return self._etype

    # -----------------------------------------------
    # --- auxiliary methods ---
    # -----------------------------------------------

    @classmethod
    def from_torch(
        cls,
        values: torch.Tensor | list[torch.Tensor],
        element_type: ElementType = ElementType.NONE,
        variable: str = "none",
        device: torch.device = None,
        gpus: list[str | torch.device] = None,
    ) -> "Field":
        values0 = values[0] if isinstance(values, list) else values
        dtype = None
        if values0.ndim == 1:
            dtype = VariableType.SCALAR
        elif values0.ndim == 2:
            if values0.shape[1] == 1:
                dtype = VariableType.SCALAR
            if values0.shape[1] == 3:
                dtype = VariableType.VECTOR
        elif values0.ndim == 3:
            if values0.shape[1] == 3 and values0.shape[2] == 3:
                dtype = VariableType.TENSOR

        if dtype is None:
            raise ValueError(f"Invalid shape: {values0.shape}")

        if isinstance(values, list):
            size = sum(v.shape[0] for v in values)
        else:
            size = values.shape[0]
        args = {
            "size": size,
            "element_type": element_type,
            "data_type": dtype,
            "data": values,
            "variable": variable,
            "device": device,
            "gpus": gpus,
        }
        return cls(**args)

    def to_np(self) -> np.ndarray:
        """
        Convert the field to a numpy array.
        """
        return torch.cat([v.cpu() for v in self._values], dim=0).numpy()

    def assign(self, other):
        """
        Assign the field values with another field or a variable.
        """
        if isinstance(other, Field):
            try:
                self._check_fields_compatible(other)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot assign fields: {e}")

            self._values = other._values
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(
                    f"Invalid value type: \
                        {other.type} (expected {self.dtype})"
                )

            for i in range(self.size):
                self._values[i][:] = other.data
        else:
            raise TypeError(f"Can't assign with {type(other)}")

    def scalarize(self) -> list["Field"]:
        """
        Convert the field to a list of scalar fields.
        """
        if self.dtype == VariableType.SCALAR:
            return [self]

        shape = 3 if self.dtype == VariableType.VECTOR else 9
        values = torch.cat(self._values, dim=0)
        data = values.view(-1, shape)
        scalar_fields = []

        for i, d in enumerate(torch.unbind(data, dim=1)):
            scalar_fields.append(
                Field.from_torch(
                    d,
                    self.etype,
                    f"{self.variable}_{i}",
                    self._device,
                    self._gpus,
                )
            )
        return scalar_fields

    # -----------------------------------------------
    # --- reload query methods ---
    # -----------------------------------------------

    def _get_local_indices(self, global_indices: int) -> tuple:
        """Get the local indices of the global index."""
        chunks_size = [v.shape[0] for v in self._values]
        cur = 0
        dev_index, local_index = None, None
        for i, size in enumerate(chunks_size):
            if global_indices < cur + size:
                dev_index = i
                local_index = global_indices - cur
                break
            cur += size
        return dev_index, local_index

    def __getitem__(self, index: int | slice) -> Variable | torch.Tensor:
        if isinstance(index, int):
            if index < 0 or index >= self.size:
                raise IndexError(f"Index out of range: {index}")

            dev, idx = self._get_local_indices(index)
            return self._values[dev][idx]
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            global_indices = range(start, stop, step)
            local_indices = [self._get_local_indices(i) for i in global_indices]

            result = []
            for dev, idx in local_indices:
                result.append(self._values[dev][idx])
            return torch.stack(result, dim=0)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, index: int | slice, value: Variable):
        if value.type != self.dtype:
            raise TypeError(f"Invalid type: {value.type}")
        if isinstance(index, int):
            if index < 0 or index >= self.size:
                raise IndexError(f"Index out of range: {index}")

            local_indices = [self._get_local_indices(index)]
        else:
            start, stop, step = index.indices(self.size)
            global_indices = range(start, stop, step)
            local_indices = [self._get_local_indices(i) for i in global_indices]

        for dev, idx in local_indices:
            self._values[dev][idx] = value.data

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        for i in range(self.size):
            dev, idx = self._get_local_indices(i)
            yield self._values[dev][idx]

    # -----------------------------------------------
    # --- override arithmetic operations ---
    # -----------------------------------------------

    def _check_fields_compatible(self, other: "Field"):
        if self.size != other.size:
            raise ValueError("Fields must have the same number of variables")

        if self.dtype != other.dtype:
            raise TypeError(
                f"Fields must have the same data type: \
                    {self.dtype} vs {other.dtype}"
            )
        if self.etype != other.etype and ElementType.NONE not in [
            self.etype,
            other.etype,
        ]:
            raise TypeError(
                f"Fields must have the same element type:\
                      {self.etype} vs {other.etype}"
            )

        if self._gpus != other._gpus:
            raise ValueError(
                f"Fields must have the same GPU devices: \
                    {self._gpus} vs {other._gpus}"
            )

    def __add__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = [v1 + v2 for v1, v2 in zip(self.data, other.data)]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = [v + other.data for v in self.data]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __radd__(self, other) -> "Field":
        return self.__add__(other)

    def __iadd__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            for i in range(self.chunks):
                self._values[i] += other.data[i]
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            for i in range(self.chunks):
                self._values[i] += other.data
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __sub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = [v1 - v2 for v1, v2 in zip(self.data, other.data)]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = [v - other.data for v in self.data]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __rsub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = [v2 - v1 for v1, v2 in zip(self.data, other.data)]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = [other.data - v for v in self.data]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __isub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            for i in range(self.chunks):
                self._values[i] -= other.data[i]
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            for i in range(self.chunks):
                self._values[i] -= other.data
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __mul__(self, other) -> "Field":
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value

            data = [v * other for v in self.data]
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
                self._gpus,
            )
        elif isinstance(other, Field):
            if other.size != self.size:
                raise TypeError(
                    f"Cannot multiply fields of different sizes: \
                        {self.size} and {other.size}"
                )

            data = [v1 * v2 for v1, v2 in zip(self.data, other.data)]
            return Field.from_torch(
                data,
                self.etype,
                "none",
                self._device,
                self._gpus,
            )
        else:
            raise TypeError(f"Cannot multiply field by {type(other)}")

    def __rmul__(self, other) -> "Field":
        return self.__mul__(other)

    def __imul__(self, other) -> "Field":
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value

            for i in range(self.chunks):
                self._values[i] *= other
        elif isinstance(other, Field):
            if other.size != self.size:
                raise TypeError(
                    f"Cannot multiply fields of different sizes: \
                        {self.size} and {other.size}"
                )

            for i in range(self.chunks):
                self._values[i] *= other.data
        else:
            raise TypeError(f"Cannot multiply field by {type(other)}")

    def __truediv__(self, other) -> "Field":
        if not isinstance(other, (Scalar, int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")
        if isinstance(other, Scalar):
            other = other.value

        data = [v / other for v in self.data]
        return Field.from_torch(
            data,
            self.etype,
            self.variable,
            self._device,
            self._gpus,
        )

    def __itruediv__(self, other) -> "Field":
        if not isinstance(other, (Scalar, int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")
        if isinstance(other, Scalar):
            other = other.value

        for i in range(self.chunks):
            self._values[i] /= other

    def __neg__(self) -> "Field":
        return self * -1

    def __abs__(self) -> "Field":
        data = [v.abs() for v in self.data]
        result = Field(
            self.size,
            self.etype,
            self.dtype,
            data,
            self.variable,
            self._device,
            self._gpus,
        )
        return result


class CellField(Field):
    """
    Cell field which represents statues of cells.

    Default at each cell center.
    """

    def __init__(
        self,
        size: int,
        data_type: VariableType,
        data: Variable | np.ndarray | torch.Tensor = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the cell field.

        Args:
            size: The number of all cells in the field.
            data_type: The data type.
            default: The default value of each cell.
            variable: The variable name.
        """
        super().__init__(
            size,
            ElementType.CELL,
            data_type,
            data,
            variable,
        )


class FaceField(Field):
    """
    Face field which represents statues of faces.

    Default at each face center.
    """

    def __init__(
        self,
        size: int,
        data_type: VariableType,
        data: Variable | np.ndarray | torch.Tensor = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the face field.

        Args:
            size: The number of all faces in the field.
            data_type: The data type.
            default: The default value of each face.
            variable: The variable name.
        """
        super().__init__(
            size,
            ElementType.FACE,
            data_type,
            data,
            variable,
        )


class NodeField(Field):
    """
    Node field which represents statues of nodes.
    """

    def __init__(
        self,
        size: int,
        data_type: VariableType,
        data: Variable | np.ndarray | torch.Tensor = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the node field.

        Args:
            size: The number of all nodes in the field.
            data_type: The data type.
            default: The default value of each node.
            variable: The variable name.
        """
        super().__init__(
            size,
            ElementType.NODE,
            data_type,
            data,
            variable,
        )
