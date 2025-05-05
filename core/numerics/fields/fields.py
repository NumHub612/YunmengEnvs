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
    Abstract field class.
    """

    def __init__(
        self,
        size: int,
        element_type: ElementType,
        data_type: VariableType,
        data: Variable | np.ndarray | torch.Tensor = None,
        variable: str = "none",
        device: torch.device = settings.DEVICE,
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
        """
        assert element_type in ElementType, f"Invalid element type: {element_type}"
        assert data_type in VariableType, f"Invalid data type: {data_type}"

        self._etype = element_type
        self._dtype = data_type
        self._device = device

        if data is None:
            type_map = {
                VariableType.SCALAR: Scalar.zero().data,
                VariableType.VECTOR: Vector.zero().data,
                VariableType.TENSOR: Tensor.zero().data,
            }
            default = type_map[data_type]
            values = torch.full(
                (size, *default.shape), 0.0, dtype=settings.DTYPE, device=device
            )
            values[:] = default.data
        else:
            if isinstance(data, Variable):
                if data.type != data_type:
                    raise ValueError(f"Invalid data type: {data.type}")
                values = torch.full(
                    (size, *data.shape), 0.0, dtype=settings.DTYPE, device=device
                )
                values[:] = data.data
            elif isinstance(data, np.ndarray):
                if data.shape[0] != size:
                    raise ValueError(f"Invalid data shape: {data.shape}")
                values = torch.from_numpy(data).to(device)
            elif isinstance(data, torch.Tensor):
                if data.shape[0] != size:
                    raise ValueError(f"Invalid data shape: {data.shape}")
                values = data.to(device)
            else:
                raise TypeError(f"Invalid data type: {type(data)}")

        self._values = values
        self._variable = variable

    # -----------------------------------------------
    # --- Properties ---
    # -----------------------------------------------

    @property
    def data(self) -> torch.Tensor:
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
        Get the field size.
        """
        return self._values.size(0)

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
        values: torch.Tensor,
        element_type: ElementType = ElementType.NONE,
        variable: str = "none",
        device: torch.device = settings.DEVICE,
    ) -> "Field":
        """
        Create a field from a torch tensor.
        """
        if values.ndim == 1:
            dtype = VariableType.SCALAR
        elif values.ndim == 2:
            if values.shape[1] == 1:
                dtype = VariableType.SCALAR
                values = values.flatten()
            elif values.shape[1] == 3:
                dtype = VariableType.VECTOR
        elif values.ndim == 3:
            if values.shape[1] == 3 and values.shape[2] == 3:
                dtype = VariableType.TENSOR
        else:
            raise ValueError(f"Invalid shape: {values.shape}")

        args = {
            "size": values.shape[0],
            "element_type": element_type,
            "data_type": dtype,
            "data": values,
            "variable": variable,
            "device": device,
        }
        return cls(**args)

    @classmethod
    def from_np(
        cls,
        values: np.ndarray,
        element_type: ElementType = ElementType.NONE,
        variable: str = "none",
        device: torch.device = settings.DEVICE,
    ) -> "Field":
        """
        Create a field from a numpy array.
        """
        if values.ndim == 1:
            dtype = VariableType.SCALAR
        elif values.ndim == 2:
            if values.shape[1] == 1:
                dtype = VariableType.SCALAR
                values = values.flatten()
            elif values.shape[1] == 3:
                dtype = VariableType.VECTOR
        elif values.ndim == 3:
            if values.shape[1] == 3 and values.shape[2] == 3:
                dtype = VariableType.TENSOR
        else:
            raise ValueError(f"Invalid shape: {values.shape}")

        args = {
            "size": values.shape[0],
            "element_type": element_type,
            "data_type": dtype,
            "data": values,
            "variable": variable,
            "device": device,
        }
        return cls(**args)

    def to_np(self) -> np.ndarray:
        """
        Convert the field to a numpy array.
        """
        return self._values.cpu().numpy()

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
                        {other.dtype} (expected {self.dtype})"
                )

            self._values[:] = other.data
        else:
            raise TypeError(f"Can't assign with {type(other)}")

    def scalarize(self) -> list["Field"]:
        """
        Convert the field to a list of scalar fields.
        """
        if self.dtype == VariableType.SCALAR:
            return [self]

        size = 3 if self.dtype == VariableType.VECTOR else 9
        data = self._values.view(-1, size)
        scalar_fields = list(torch.unbind(data, dim=1))
        return scalar_fields

    # -----------------------------------------------
    # --- reload query methods ---
    # -----------------------------------------------

    def __getitem__(self, index: int) -> Variable:
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        return self._values[index]

    def __setitem__(self, index: int, value: Variable):
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        if value.type != self.dtype:
            raise TypeError(f"Invalid value type: {value.type}")

        self._values[index] = value.data

    def __len__(self) -> int:
        return self._values.size(0)

    def __iter__(self):
        for i in range(self.size):
            yield self._values[i]

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

    def __add__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = self.data + other.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = self.data + other.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __radd__(self, other) -> "Field":
        return self.__add__(other)

    def __iadd__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            self._values += other.data
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            self._values += other.data
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __sub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = self.data - other.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = self.data - other.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __rsub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = other.data - self.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            data = other.data - self.data
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __isub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            self._values -= other.data
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(f"Invalid value type: {other.type}")

            self._values -= other.data
        else:
            raise TypeError(f"Cannot subtract {type(other)} from")

    def __mul__(self, other) -> "Field":
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value

            data = self.data * other
            return Field.from_torch(
                data,
                self.etype,
                self.variable,
                self._device,
            )
        elif isinstance(other, Field):
            if other.size != self.size:
                raise TypeError(
                    f"Cannot multiply fields of different sizes: \
                        {self.size} and {other.size}"
                )

            data = self.data * other.data
            return Field.from_torch(data, self.etype, "none", self._device)
        else:
            raise TypeError(f"Cannot multiply field by {type(other)}")

    def __rmul__(self, other) -> "Field":
        return self.__mul__(other)

    def __imul__(self, other) -> "Field":
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value

            self._values *= other
        elif isinstance(other, Field):
            if other.size != self.size:
                raise TypeError(
                    f"Cannot multiply fields of different sizes: \
                        {self.size} and {other.size}"
                )

            self._values *= other.data
        else:
            raise TypeError(f"Cannot multiply field by {type(other)}")

    def __truediv__(self, other) -> "Field":
        if not isinstance(other, (Scalar, int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")

        data = self.data / other
        return Field.from_torch(
            data,
            self.etype,
            self.variable,
            self._device,
        )

    def __itruediv__(self, other) -> "Field":
        if not isinstance(other, (Scalar, int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")

        self._values /= other

    def __neg__(self) -> "Field":
        result = Field(
            self.size,
            self.etype,
            self.dtype,
            -self.data,
            self.variable,
            self._device,
        )
        return result

    def __abs__(self) -> "Field":
        result = Field(
            self.size,
            self.etype,
            self.dtype,
            torch.abs(self._values),
            self.variable,
            self._device,
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
