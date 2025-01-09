# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

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
        variable_num: int,
        element_type: str,
        data_type: str,
        default: Scalar | Vector | Tensor = None,
        variable: str = "none",
    ):
        """
        Initialize the field with a given number of variables and a default value.

        Args:
            variable_num: The number of variables in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            element_type: The element type, e.g. "cell", "face", "node".
            default: The default value of the field.
            variable: The variable name.
        """
        element_type = element_type.lower()
        if element_type not in ["cell", "face", "node"]:
            raise ValueError(f"Invalid element type: {element_type}")
        self._etype = element_type

        data_type = data_type.lower()
        if data_type not in ["scalar", "vector", "tensor"]:
            raise ValueError(f"Invalid data type: {data_type}")
        self._dtype = data_type

        if default and default.type != self._dtype:
            raise ValueError(f"Conflicting default value type: {default.type}")
        if default is None:
            type_map = {"scalar": Scalar, "vector": Vector, "tensor": Tensor}
            default = type_map[data_type]()
        self._default = default
        self._values = np.full(variable_num, default)

        self._variable = variable

    # -----------------------------------------------
    # --- Properties and auxiliary methods ---
    # -----------------------------------------------

    @property
    def variable(self) -> str:
        """
        Get the variable name of the field.
        """
        return self._variable

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

    @classmethod
    def from_np(
        cls,
        values: np.ndarray,
        element_type: str,
        variable: str = "none",
    ) -> "Field":
        """
        Create a field from a numpy array.

        Notes:
            - If values.shape[1] == 1, the field is a scalar field.
            - If values.shape[1] == 3, the field is a vector field.
            - If values.shape[1] == 9, the field is a tensor field
        """
        if values.ndim != 2:
            raise ValueError(f"Invalid data shape: {values.shape}")

        element_type = element_type.lower()
        if element_type not in ["cell", "face", "node"]:
            raise ValueError(f"Invalid element type: {element_type}")

        if values.shape[1] == 1:
            data = np.array([Scalar.from_np(v) for v in values])
            dtype = "scalar"
        elif values.shape[1] == 3:
            data = np.array([Vector.from_np(v) for v in values])
            dtype = "vector"
        elif values.shape[1] == 9:
            data = np.array([Tensor.from_np(v.reshape(3, 3)) for v in values])
            dtype = "tensor"
        else:
            raise ValueError(f"Invalid data shape: {values.shape}")

        field = cls(values.size, dtype, None, variable)
        field._values = data
        return field

    def to_np(self) -> np.ndarray:
        """
        Convert the field to a numpy array.
        """
        return np.array([v.to_np().flatten() for v in self._values])

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
            if other.dtype != self.dtype:
                raise TypeError(
                    f"Invalid value type: {other.dtype} (expected {self.dtype})"
                )

            self._values = np.full(self.size, other)
        else:
            raise TypeError(f"Can't assign {type(other)} to field")

    # -----------------------------------------------
    # --- reload query methods ---
    # -----------------------------------------------

    def __getitem__(self, index: int) -> Variable:
        """
        Get the value of the field at a given position.
        """
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        return self._values[index]

    def __setitem__(self, index: int, value: Variable):
        """
        Set the value of the field at a given position.
        """
        if index < 0 or index >= self.size:
            raise IndexError(f"Index out of range: {index}")

        if value.type != self._default.type:
            raise TypeError(
                f"Invalid value type: {value.type} (expected {self._default.type})"
            )

        self._values[index] = value

    def __len__(self) -> int:
        """
        Get the number of variables in the field.
        """
        return self._values.size

    def __iter__(self):
        """
        Iterate over the variables in the field.
        """
        for i in range(self.size):
            yield self._values[i]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def _check_fields_compatible(self, other: "Field"):
        """
        Check if two fields are compatible for arithmetic operations.
        """
        if self.size != other.size:
            raise ValueError("Fields must have the same number of variables")

        if self.dtype != other.dtype:
            raise TypeError("Fields must have the same data type")

        if self.etype != other.etype:
            raise TypeError("Fields must have the same element type")

        # if self.variable != other.variable:
        #     raise TypeError("Fields must have the same variable")

    def __add__(self, other) -> "Field":
        """
        Add two fields or a field and a constant element-wise.
        """
        if isinstance(other, Field):
            try:
                self._check_fields_compatible(other)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot add fields: {e}")

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values += other._values
            return result
        elif isinstance(other, Variable):
            if other.type != self._default.type:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self._default.type})"
                )

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values += other
            return result
        else:
            raise TypeError(f"Cannot add {type(other)} to field")

    def __radd__(self, other) -> "Field":
        """
        Add two fields or a field and a constant element-wise.
        """
        return self.__add__(other)

    def __sub__(self, other) -> "Field":
        """
        Subtract two fields or a field and a constant element-wise.
        """
        if isinstance(other, Field):
            try:
                self._check_fields_compatible(other)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot subtract fields: {e}")

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values -= other._values
            return result
        elif isinstance(other, Variable):
            if other.type != self._default.type:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self._default.type})"
                )

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values -= other
            return result
        else:
            raise TypeError(f"Cannot subtract {type(other)} from field")

    def __rsub__(self, other) -> "Field":
        """
        Subtract two fields or a field and a constant element-wise.
        """
        if isinstance(other, Field):
            try:
                self._check_fields_compatible(other)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot subtract fields: {e}")

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values = other._values - self._values
            return result
        elif isinstance(other, Variable):
            if other.type != self._default.type:
                raise TypeError(
                    f"Invalid value type: {other.type} (expected {self._default.type})"
                )

            result = Field(
                self.size, self.etype, self.dtype, self._default, self.variable
            )
            result._values = other - self._values
            return result
        else:
            raise TypeError(f"Cannot subtract {type(other)} from field")

    def __mul__(self, other: Scalar | float | int) -> "Field":
        """
        Multiply the field by a scalar element-wise.
        """
        if not isinstance(other, Scalar) and not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply field by {type(other)}")

        result = Field(self.size, self.etype, self.dtype, self._default, self.variable)
        result._values = self._values * other
        return result

    def __rmul__(self, other: Scalar | float | int) -> "Field":
        """
        Multiply the field by a scalar element-wise.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Scalar | float | int) -> "Field":
        """
        Divide the field by a scalar element-wise.
        """
        if not isinstance(other, Scalar) and not isinstance(other, (int, float)):
            raise TypeError(f"Cannot divide field by {type(other)}")

        result = Field(self.size, self.etype, self.dtype, self._default, self.variable)
        result._values = self._values / other
        return result

    def __neg__(self) -> "Field":
        """
        Negate the field element-wise.
        """
        result = Field(self.size, self.etype, self.dtype, self._default, self.variable)
        result._values = -self._values
        return result

    def __abs__(self) -> "Field":
        """
        Take the absolute value of the field element-wise.
        """
        result = Field(self.size, self.etype, self.dtype, self._default, self.variable)
        result._values = np.abs(self._values)
        return result


class CellField(Field):
    """
    Cell field which represents statues of cells.

    Default at each cell center.
    """

    def __init__(
        self,
        num_cells: int,
        data_type: str,
        default: Variable = None,
        varialbe: str = "none",
    ):
        """
        Initialize the cell field.

        Args:
            num_cells: The number of cells in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each cell.
        """
        super().__init__(
            num_cells,
            "cell",
            data_type,
            default,
            varialbe,
        )


class FaceField(Field):
    """
    Face field which represents statues of faces.

    Default at each face center.
    """

    def __init__(
        self,
        num_faces: int,
        data_type: str,
        default: Variable = None,
        varialbe: str = "none",
    ):
        """
        Initialize the face field.

        Args:
            num_faces: The number of faces in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each face.
        """
        super().__init__(
            num_faces,
            "face",
            data_type,
            default,
            varialbe,
        )


class NodeField(Field):
    """
    Node field which represents statues of nodes.
    """

    def __init__(
        self,
        num_nodes: int,
        data_type: str,
        default: Variable = None,
        varialbe: str = "none",
    ):
        """
        Initialize the node field.

        Args:
            num_nodes: The number of nodes in the field.
            data_type: The data type, e.g. "scalar", "vector", "tensor".
            default: The default value of each node.
        """
        super().__init__(
            num_nodes,
            "node",
            data_type,
            default,
            varialbe,
        )
