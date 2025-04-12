# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Fields definition.
"""
from core.numerics.fields.variables import Variable, Scalar, Vector, Tensor

from typing import Callable
import numpy as np


class Field:
    """
    Abstract field class.
    """

    def __init__(
        self,
        size: int,
        element_type: str,
        data_type: str,
        data: Variable | np.ndarray = None,
        variable: str = "none",
    ):
        """
        Initialize the field.

        Args:
            size: The number of variables in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            element_type: The element type, e.g. "cell", "face", "node".
            data: The initial data of the field.
            variable: The variable name for the field.
        """
        if element_type not in ["cell", "face", "node", "none"]:
            raise ValueError(f"Invalid element type: {element_type} for Field")
        self._etype = element_type

        if data_type not in ["float", "scalar", "vector", "tensor"]:
            raise ValueError(f"Invalid data type: {data_type}")
        self._dtype = data_type

        if data is None:
            type_map = {
                "float": 0.0,
                "scalar": Scalar.zero(),
                "vector": Vector.zero(),
                "tensor": Tensor.zero(),
            }
            default = type_map[data_type]
            data = np.full(size, default)
        else:
            if isinstance(data, Variable):
                if data.type != data_type:
                    raise ValueError(f"Invalid data type: {data.type}")
                data = np.full(size, data)
            elif isinstance(data, np.ndarray):
                if data.shape != (size,):
                    raise ValueError(f"Invalid data shape: {data.shape}")
            else:
                raise TypeError(f"Invalid data type: {type(data)}")

        self._values = data
        self._variable = variable

    # -----------------------------------------------
    # --- Properties ---
    # -----------------------------------------------

    @property
    def data(self) -> np.ndarray:
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
        return self._values.size

    @property
    def dtype(self) -> str:
        """
        Get the data type of the field, e.g. "scalar", "vector", "tensor".
        """
        return self._dtype

    @property
    def etype(self) -> str:
        """
        Get the element type of the field, e.g. "cell", "face", "node".
        """
        return self._etype

    # -----------------------------------------------
    # --- auxiliary methods ---
    # -----------------------------------------------

    @classmethod
    def from_np(
        cls, values: np.ndarray, element_type: str = "none", variable: str = "none"
    ) -> "Field":
        """
        Create a field from a numpy array.

        Notes:
            - If values.shape[1] == 1, the field is a scalar field.
            - If values.shape[1] == 3, the field is a vector field.
            - If values.shape[1] == 9, the field is a tensor field
        """
        etype = element_type if element_type else "none"

        if values.ndim == 1 or values.dtype != object:
            dtype = "float"
            data = values
        elif values.dtype != object and values.ndim == 2:
            if values.shape[1] == 1:
                data = np.array([Scalar.from_np(v) for v in values])
                dtype = "scalar"
            elif values.shape[1] == 3:
                data = np.array([Vector.from_np(v) for v in values])
                dtype = "vector"
            elif values.shape[1] == 9:
                data = np.array([Tensor.from_np(v) for v in values])
                dtype = "tensor"
        elif values.dtype == object and values.ndim == 2:
            if isinstance(values[0][0], Variable):
                data = values
                dtype = values[0][0].type
        else:
            raise ValueError(f"Invalid data shape: {values.shape}")

        args = {
            "size": values.shape[0],
            "element_type": etype,
            "data_type": dtype,
            "data": data,
            "variable": variable,
        }
        return cls(**args)

    def to_np(self) -> np.ndarray:
        """
        Convert the field to a numpy float array.
        """
        if self.dtype == "float":
            return self._values
        else:
            return np.array([v.to_np() for v in self._values])

    def scalarize(self) -> list["Field"]:
        """
        Convert the field to a list of scalar fields.
        """
        np_values = self.to_np()
        scalar_fields = []
        for i in range(self.size):
            field = Field.from_np(np_values[:, i], self.etype, self.variable)
            scalar_fields.append(field)
        return scalar_fields

    def filter(self, func: Callable) -> list[int]:
        """
        Filter the field by a given function.

        Args:
            func: Function taking a variable as input and returning a boolean value.

        Returns:
            The filtered variable indices.
        """
        if not callable(func):
            raise TypeError(f"Invalid function type: {type(func)}")

        # vectorize the function to apply it to each variable
        vectorized_func = np.vectorize(func)
        mask = vectorized_func(self._values)

        # get the variables that satisfy the condition
        indices = np.where(mask)[0]
        return indices.tolist()

    def for_each(self, func: Callable):
        """
        Apply a function to each variable of the field.

        Args:
            func: Function taking a variable as input.
        """
        if not callable(func):
            raise TypeError(f"Invalid function type: {type(func)}")

        for i in range(self.size):
            res = func(self._values[i])
            if isinstance(res, Variable) and res.dtype == self.dtype:
                self._values[i] = res
            else:
                raise TypeError(f"Invalid return type: {res}")

    def at(self, indexes: list[int], func: Callable):
        """
        Apply a function to a variable of the field at given position.

        Args:
            indexes: The index of the variable to apply the function to.
            func: Function taking a variable as input.
        """
        if not callable(func):
            raise TypeError(f"Invalid function type: {type(func)}")

        min_index = min(indexes)
        max_index = max(indexes)
        if min_index < 0 or max_index >= self.size:
            raise IndexError(f"Index out of range: {min_index}/{max_index}")

        for i in indexes:
            res = func(self._values[i])
            if isinstance(res, Variable) and res.dtype == self.dtype:
                self._values[i] = res
            else:
                raise TypeError(f"Invalid return type: {res}")

    def assign(self, other: "Field | Variable"):
        """
        Assign the values of another field or a variable to the current field.

        Args:
            other: The other field or variable to assign.
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
                    f"Invalid value type: {other.dtype} (expected {self.dtype})"
                )

            self._values = np.full(self.size, other)
        else:
            raise TypeError(f"Can't assign {type(other)} to field")

    def resize(self, size: int):
        """Resize the field to a new size inplace."""
        if size < 1:
            raise ValueError(f"Invalid size: {size}")

        if size == self.size:
            return

        if size > self.size:
            default = {
                "float": 0.0,
                "scalar": Scalar.zero(),
                "vector": Vector.zero(),
                "tensor": Tensor.zero(),
            }
            data = np.concatenate(
                [self._values, np.full(size - self.size, default[self.dtype])]
            )
        else:
            data = self._values[:size]
        self._values = data

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
            raise TypeError(f"Invalid value type: {value.type} (expected {self.dtype})")

        self._values[index] = value

    def __len__(self) -> int:
        return self._values.size

    def __iter__(self):
        for i in range(self.size):
            yield self._values[i]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def _check_fields_compatible(self, other: "Field"):
        if self.size != other.size:
            raise ValueError("Fields must have the same number of variables")

        if self.dtype != other.dtype:
            raise TypeError("Fields must have the same data type")

        if self.etype != other.etype:
            raise TypeError("Fields must have the same element type")

    def __add__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = self.data + other.data
            return Field.from_np(data, self.etype, self.variable)
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self.dtype})"
                )

            data = self.data + other  # element-wise addition
            return Field.from_np(data, self.etype, self.variable)
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __radd__(self, other) -> "Field":
        return self.__add__(other)

    def __sub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = self.data - other.data
            return Field.from_np(data, self.etype, self.variable)
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self.dtype})"
                )

            data = self.data - other
            return Field.from_np(data, self.etype, self.variable)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from field")

    def __rsub__(self, other) -> "Field":
        if isinstance(other, Field):
            self._check_fields_compatible(other)

            data = other.data - self.data
            return Field.from_np(data, self.etype, self.variable)
        elif isinstance(other, Variable):
            if other.type != self.dtype:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self.dtype})"
                )

            data = other - self.data
            return Field.from_np(data, self.etype, self.variable)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from field")

    def __mul__(self, other) -> "Field":
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value

            data = self.data * other
            return Field.from_np(data, self.etype, self.variable)
        elif isinstance(other, Field):
            if other.size != self.size:
                raise TypeError(
                    f"Cannot multiply fields of different sizes: {self.size} and {other.size}"
                )

            data = self.data * other.data
            return Field.from_np(data, self.etype, "none")
        else:
            raise TypeError(f"Cannot multiply field by {type(other)}")

    def __rmul__(self, other) -> "Field":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Field":
        if not isinstance(other, (Scalar, int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")

        data = self.data / other
        return Field.from_np(data, self.etype, self.variable)

    def __neg__(self) -> "Field":
        result = Field(self.size, self.etype, self.dtype, -self.data, self.variable)
        return result

    def __abs__(self) -> "Field":
        result = Field(
            self.size, self.etype, self.dtype, np.abs(self._values), self.variable
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
        data_type: str,
        data: Variable | np.ndarray = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the cell field.

        Args:
            size: The number of cells in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each cell.
            variable: The variable name.
        """
        etype = kwargs.get("element_type", "none")
        if etype != "none" and etype != "cell":
            raise ValueError(f"Invalid element type: {etype} for CellField")

        super().__init__(
            size,
            "cell",
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
        data_type: str,
        data: Variable | np.ndarray = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the face field.

        Args:
            size: The number of faces in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each face.
            variable: The variable name.
        """
        etype = kwargs.get("element_type", "none")
        if etype != "none" and etype != "face":
            raise ValueError(f"Invalid element type: {etype} for FaceField")

        super().__init__(
            size,
            "face",
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
        data_type: str,
        data: Variable | np.ndarray = None,
        variable: str = "none",
        **kwargs,
    ):
        """
        Initialize the node field.

        Args:
            size: The number of nodes in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each node.
            variable: The variable name.
        """
        etype = kwargs.get("element_type", "none")
        if etype != "none" and etype != "node":
            raise ValueError(f"Invalid element type: {etype} for NodeField")

        super().__init__(
            size,
            "node",
            data_type,
            data,
            variable,
        )
