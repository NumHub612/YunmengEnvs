# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix.
"""
from core.numerics.fields import Variable, Scalar, Vector, Tensor
from scipy.sparse import dok_matrix
import torch
import numpy as np
import copy
from abc import abstractmethod


class Matrix:
    """
    Abstract matrix class.
    """

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    @abstractmethod
    def identity(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        """Create a Identity Matrix."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def ones(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        """Create a matrix with all elements set to one."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def zeros(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        """Create a matrix with all elements set to zero."""
        raise NotImplementedError()

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The matrix shape, e.g. (rows, cols)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> str:
        """The matrix data type, e.g. float, scalar, vector, tensor."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def diag(self) -> np.ndarray:
        """The diagonal elements of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def nnz(self) -> int:
        """The number of all non-zero elements."""
        raise NotImplementedError()

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    @abstractmethod
    def __getitem__(self, index: tuple):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, index: tuple, value: float | Variable):
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other: "Matrix"):
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, other: "Matrix"):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __truediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __rtruediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __rmul__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError()

    @abstractmethod
    def __abs__(self):
        raise NotImplementedError()

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        """Convert the matrix to a dense numpy array."""
        raise NotImplementedError()

    @abstractmethod
    def transpose(self) -> "Matrix":
        """Transpose the matrix."""
        raise NotImplementedError()

    @abstractmethod
    def reshape(self, shape: tuple) -> "Matrix":
        """Reshape the matrix."""
        raise NotImplementedError()

    @abstractmethod
    def scalarize(self) -> list["Matrix"]:
        """Scalarize the matrix."""
        raise NotImplementedError()


class DenseMatrix(Matrix):
    """
    Dense matrix class supporting float, Variable data types.
    """

    def __init__(self, shape: tuple, data_type: str = "float", data: np.ndarray = None):
        """Matrix class.

        Args:
            shape: The shape of the matrix.
            data_type: The matrix data type, e.g. float, scalar, vector, tensor.
            data: The data of the matrix.
        """
        assert data_type in ["float", "scalar", "vector", "tensor"]
        self._dtype = data_type
        self._shape = shape
        self._data = None

        if data is None:
            data_map = {
                "float": 0.0,
                "scalar": Scalar().zero(),
                "vector": Vector().zero(),
                "tensor": Tensor().zero(),
            }
            self._data = np.full(shape, data_map[data_type])
        else:
            self._check_shape_compatible(data)
            self._check_type_compatible(data)
            self._data = data

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def zeros(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        data_map = {
            "float": 0.0,
            "scalar": Scalar().zero(),
            "vector": Vector().zero(),
            "tensor": Tensor().zero(),
        }
        zero = data_map[data_type]
        return DenseMatrix(shape, data_type, np.full(shape, zero))

    @classmethod
    def ones(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        data_map = {
            "float": 1.0,
            "scalar": Scalar().unit(),
            "vector": Vector().unit(),
            "tensor": Tensor().unit(),
        }
        one = data_map[data_type]
        return DenseMatrix(shape, data_type, np.full(shape, one))

    @classmethod
    def identity(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix must be squared.")

        data_map = {
            "float": 1.0,
            "scalar": Scalar().unit(),
            "vector": Vector().unit(),
            "tensor": Tensor().unit(),
        }
        data = np.identity(shape[0]) * data_map[data_type]
        return DenseMatrix(shape, data_type, data)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def diag(self) -> np.ndarray:
        return np.diag(self._data)

    @property
    def type(self) -> str:
        return self._dtype

    @property
    def nnz(self) -> int:
        return self._data.size

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: Variable):
        self._data[index] = value

    def _check_type_compatible(self, other):
        if isinstance(other, np.ndarray):
            if other.dtype == object:
                check = np.all([v.type == self.type for v in other.flatten()])
            else:
                check = self.type == "float"
            if not check:
                raise ValueError(f"Matrix types don't match: {self.type}.")
        elif isinstance(other, Matrix):
            if other.type != self.type:
                raise ValueError(f"Matrix types don't match: {self.type}.")
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def _check_shape_compatible(self, other, mode: str = "add"):
        if mode == "add":
            if other.shape != self.shape:
                raise ValueError(f"Matrix shapes don't match: {self.shape}.")
        elif mode == "mul":
            if other.shape[1] != self.shape[0]:
                raise ValueError(f"Matrix shapes don't match: {self.shape}.")
        else:
            raise ValueError(f"Invalid mode {mode}.")

    def __add__(self, other):
        self._check_type_compatible(other)
        self._check_shape_compatible(other)
        if isinstance(other, DenseMatrix):
            return DenseMatrix(self.shape, self._dtype, self._data + other._data)
        else:
            data = copy.deepcopy(self._data)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    data[i, j] += other[i, j]
            return DenseMatrix(self.shape, self._dtype, data)

    def __sub__(self, other):
        self._check_type_compatible(other)
        self._check_shape_compatible(other)
        if isinstance(other, DenseMatrix):
            return DenseMatrix(self.shape, self._dtype, self._data - other._data)
        else:
            data = copy.deepcopy(self._data)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    data[i, j] -= other[i, j]
            return DenseMatrix(self.shape, self._dtype, data)

    def __mul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return DenseMatrix(self.shape, self._dtype, self._data * other)
        elif isinstance(other, Matrix):
            self._check_type_compatible(other)
            self._check_shape_compatible(other, mode="mul")

            shape = (self.shape[0], other.shape[1])
            if isinstance(other, DenseMatrix):
                return DenseMatrix(shape, self._dtype, self._data @ other._data)
            else:
                data = np.zeros(shape)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(self.shape[1]):
                            data[i, j] += self._data[i, k] * other[k, j]
                return DenseMatrix(shape, self._dtype, data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return DenseMatrix(self.shape, self._dtype, self._data / other)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return DenseMatrix(self.shape, self._dtype, other / self._data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return DenseMatrix(self.shape, self._dtype, other * self._data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __neg__(self):
        return DenseMatrix(self.shape, self._dtype, -self._data)

    def __abs__(self):
        return DenseMatrix(self.shape, self._dtype, np.abs(self._data))

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def to_dense(self) -> np.ndarray:
        return self._data

    def transpose(self) -> "Matrix":
        return DenseMatrix(self.shape[::-1], self._dtype, self._data.T)

    def reshape(self, shape: tuple) -> "Matrix":
        return DenseMatrix(shape, self._dtype, self._data.reshape(shape))

    def scalarize(self) -> list["Matrix"]:
        if self.type == "float":
            return [self]

        type_dims = {"scalar": 1, "vector": 3, "tensor": 9}
        dim = type_dims[self.type]

        mats = [self.zeros(self.shape) for _ in range(dim)]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                arr = self._data[i, j].to_np().flatten()
                for k in range(dim):
                    mats[k][(i, j)] = arr[k]
        return mats


class SciMatrix(Matrix):
    """
    SciPy sparse matrix only supports float data type.
    """

    def __init__(self, shape: tuple, data: dok_matrix = None):
        """Complex matrix class.

        Args:
            shape: The matrix shape.
            data: The matrix data.
        """
        self._shape = shape
        self._data = None

        if data is not None:
            if data.shape != shape:
                raise ValueError(f"Input data doesn't match shape: {shape}.")
            self._data = data
        else:
            self._data = dok_matrix(shape)

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def zeros(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        if data_type != "float":
            raise ValueError("Only float type is supported")
        return SciMatrix(shape)

    @classmethod
    def ones(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        if data_type != "float":
            raise ValueError("Only float type is supported")
        data = np.full(shape, 1.0)
        return SciMatrix(shape, dok_matrix(data))

    @classmethod
    def identity(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix must be squared.")
        if data_type != "float":
            raise ValueError("Only float type is supported")

        data = dok_matrix(shape)
        for i in range(shape[0]):
            data[i, i] = 1.0
        return SciMatrix(shape, data)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def diag(self) -> np.ndarray:
        return self._data.diagonal()

    @property
    def type(self) -> str:
        return "float"

    @property
    def nnz(self) -> int:
        return self._data.nnz

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value

    def _check_type_compatible(self, other: "Matrix"):
        if not isinstance(other, Matrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        if other.type not in ["float"]:
            raise ValueError(f"Matrix types don't match: {self.type} vs {other.type}.")

    def _check_shape_compatible(self, other: "Matrix", mode: str = "add"):
        if mode == "add" and self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )
        if mode == "mul" and self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

    def __add__(self, other: "Matrix"):
        self._check_type_compatible(other)
        self._check_shape_compatible(other)
        if isinstance(other, SciMatrix):
            return SciMatrix(self.shape, self._data + other._data)

        data = dok_matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data[i, j] = self._data[i, j] + other[i, j]
        return SciMatrix(self.shape, data)

    def __sub__(self, other):
        self._check_type_compatible(other)
        self._check_shape_compatible(other)
        if isinstance(other, SciMatrix):
            return SciMatrix(self.shape, self._data - other._data)

        data = dok_matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data[i, j] = self._data[i, j] - other[i, j]
        return SciMatrix(self.shape, data)

    def __mul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix(self.shape, self._data * other)
        elif isinstance(other, Matrix):
            self._check_type_compatible(other)
            self._check_shape_compatible(other, mode="mul")

            shape = (self.shape[0], other.shape[1])
            if isinstance(other, SciMatrix):
                return SciMatrix(shape, self._data * other._data)
            else:
                data = np.zeros(shape)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(self.shape[1]):
                            data[i, j] += self._data[i, k] * other[k, j]
                return SciMatrix(shape, dok_matrix(data))
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix(self.shape, other * self._data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix(self.shape, self._data / other)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix(self, other / self._data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __neg__(self):
        return SciMatrix(self.shape, -self._data)

    def __abs__(self):
        return SciMatrix(self.shape, abs(self._data))

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def to_dense(self) -> np.ndarray:
        return self._data.toarray()

    def transpose(self) -> "Matrix":
        return SciMatrix(self.shape[::-1], self._data.transpose())

    def reshape(self, shape: tuple) -> "Matrix":
        if shape == self.shape:
            return self
        return SciMatrix(shape, self._data.reshape(shape))

    def scalarize(self) -> list["Matrix"]:
        return [self]


class SparseMatrix(Matrix):
    """
    Sparse matrix class supporting float, Variable data types.
    """

    def __init__(
        self,
        shape: tuple,
        data_type: str = "float",
        data: dict = None,
        default: float | Variable = None,
    ):
        """Complex sparse matrix class.

        Args:
            shape: The matrix shape.
            data_type: The matrix data type, e.g. float, scalar, vector, tensor.
            data: The matrix data.
            default: The default value.
        """
        assert data_type in ["float", "scalar", "vector", "tensor"]

        self._dtype = data_type
        self._rows, self._cols = shape

        if data is not None:
            max_idx = max(data.keys())
            if max_idx >= self._rows * self._cols:
                raise ValueError(f"Input data doesn't match shape: {shape}.")
            self._matrix = data
        else:
            self._matrix = {}

        if data_type == "float" and default is None:
            self._default = 0.0
        elif default is None:
            type_map = {"scalar": Scalar, "vector": Vector, "tensor": Tensor}
            self._default = type_map[data_type].zero()
        else:
            self._default = default

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def zeros(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        return SparseMatrix(shape, data_type)

    @classmethod
    def ones(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        type_map = {
            "float": 1.0,
            "scalar": Scalar.unit(),
            "vector": Vector.unit(),
            "tensor": Tensor.unit(),
        }

        return SparseMatrix(shape, data_type, default=type_map[data_type])

    @classmethod
    def identity(cls, shape: tuple, data_type: str = "float") -> "Matrix":
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix must be squared.")

        type_map = {
            "float": 1.0,
            "scalar": Scalar.unit(),
            "vector": Vector.unit(),
            "tensor": Tensor.unit(),
        }
        one = type_map[data_type]
        data = {}
        for i in range(shape[0]):
            data[i * shape[1] + i] = one
        return SparseMatrix(shape, data_type, data)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------
    @property
    def shape(self) -> tuple:
        """The matrix shape, e.g. (rows, cols)."""
        return self._rows, self._cols

    @property
    def type(self) -> str:
        """The matrix data type, e.g. float, scalar, vector, tensor."""
        return self._dtype

    @property
    def diag(self) -> np.ndarray:
        """The diagonal elements of the matrix."""
        if len(self._matrix) == 0:
            return None
        if self._rows != self._cols:
            return None

        diag = []
        for i in range(self._rows):
            idx = i * self._rows + i
            if idx in self._matrix:
                diag.append(self._matrix[idx])
            else:
                diag.append(self._default)
        return np.array(diag)

    @property
    def nnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return len(self._matrix)

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        if index[0] >= self._rows or index[1] >= self._cols:
            raise IndexError(f"Index out of range: {index}")

        idx = index[0] * self._cols + index[1]
        if idx in self._matrix:
            return self._matrix[idx]
        else:
            return self._default

    def __setitem__(self, index: tuple, value: float | Variable):
        if index[0] >= self._rows or index[1] >= self._cols:
            raise IndexError(f"Index out of range: {index}")

        if isinstance(value, object) and value.type != self._dtype:
            raise ValueError(f"Invalid data type {value.type} for matrix {self.type}.")

        idx = index[0] * self._cols + index[1]
        self._matrix[idx] = value

    def _check_compatible(self, other: "Matrix"):
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

        if self.type != other.type:
            raise ValueError(f"Matrix types don't match: {self.type} vs {other.type}.")

    def __add__(self, other):
        if not isinstance(other, SparseMatrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        self._check_compatible(other)

        data = copy.deepcopy(self._matrix)
        for idx, val in other._matrix.items():
            if idx in data:
                data[idx] = val + data[idx]
            else:
                data[idx] = val

        return SparseMatrix(self.shape, self.type, data, self._default)

    def __sub__(self, other):
        if not isinstance(other, SparseMatrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        self._check_compatible(other)

        data = copy.deepcopy(self._matrix)
        for idx, val in other._matrix.items():
            if idx in data:
                data[idx] = data[idx] - val
            else:
                data[idx] = -val

        return SparseMatrix(self.shape, self.type, data, self._default)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            data = {}
            for idx, val in self._matrix.items():
                data[idx] = val * other
            return SparseMatrix(self.shape, self.type, data, self._default)
        elif isinstance(other, Variable):
            data = {}
            for idx, val in self._matrix.items():
                data[idx] = val * other
            return SparseMatrix(self.shape, self.type, data, self._default)
        elif isinstance(other, SparseMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Matrix shapes don't match: {self.shape} vs {other.shape}."
                )

            default = self._default * other._default
            if isinstance(self._default, (int, float)):
                dtype = "float"
            else:
                dtype = self._default.type
            data = {}
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    val = self._matrix.get(i * self._cols + j, None)
                    for k in range(other.shape[0]):
                        idx2 = k * other._cols + j
                        val2 = other._matrix.get(idx2, None)
                        new_idx = i * other.shape[1] + k
                        if val and val2:
                            data[new_idx] = val * val2
                        elif val2:
                            data[new_idx] = other._default * val2
                        elif val:
                            data[new_idx] = val * other._default
            return SparseMatrix((self.shape[0], other.shape[1]), dtype, data, default)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        elif isinstance(other, Variable):
            data = {}
            for idx, val in self._matrix.items():
                data[idx] = other * val
            return SparseMatrix(self.shape, self.type, data, self._default)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value
            data = {}
            for idx, val in self._matrix.items():
                data[idx] = val / other
            return SparseMatrix(self.shape, self.type, data, self._default)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.value
            data = {}
            for idx, val in self._matrix.items():
                data[idx] = other / val
            return SparseMatrix(self.shape, self.type, data, self._default)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __neg__(self):
        data = copy.deepcopy(self._matrix)
        for idx, val in data.items():
            data[idx] = -val
        return SparseMatrix(self.shape, self.type, data, self._default)

    def __abs__(self):
        data = copy.deepcopy(self._matrix)
        for idx, val in data.items():
            data[idx] = abs(val)
        return SparseMatrix(self.shape, self.type, data, self._default)

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def to_dense(self) -> np.ndarray:
        data = np.zeros(self.shape)
        for idx, val in self._matrix.items():
            i, j = idx // self._cols, idx % self._cols
            data[i, j] = val
        return data

    def transpose(self) -> "Matrix":
        cols, rows = self.shape
        data = {}
        for idx, val in self._matrix.items():
            i, j = idx // rows, idx % rows
            new_idx = j * cols + i
            data[new_idx] = val
        return SparseMatrix((rows, cols), self.type, data, self._default)

    def reshape(self, shape: tuple) -> "Matrix":
        if shape == self.shape:
            return self

        rows, cols = shape
        if rows * cols > self._rows * self._cols:
            raise ValueError(f"Invalid shape: {shape}.")

        data = copy.deepcopy(self._matrix)
        for idx in data.keys():
            if idx >= rows * cols:
                del data[idx]

        return SparseMatrix(shape, self.type, data, self._default)

    def scalarize(self) -> list["Matrix"]:
        if self.type == "float":
            return [self]

        type_dims = {"scalar": 1, "vector": 3, "tensor": 9}
        dim = type_dims[self.type]

        mats = [self.zeros(self.shape) for _ in range(dim)]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                arr = self._data[(i, j)].to_np().flatten()
                for k in range(dim):
                    mats[k][(i, j)] = arr[k]
        return mats


class TorchMatrix(Matrix):
    """
    Torch matrix class only supporting float data types.
    """

    def __init__(
        self, shape: tuple, data: torch.Tensor = None, device: str = "default"
    ):
        """Torch matrix class.

        Args:
            shape: The matrix shape.
            data: The matrix data.
            device: The device to store the matrix.
        """
        self._shape = shape
        self._data = None

        if device == "default":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        if data is not None:
            if data.shape != shape:
                raise ValueError(f"Input data doesn't match shape: {shape}.")
            self._data = data.to_sparse_coo().to(self._device)
        else:
            self._data = torch.sparse_coo_tensor(size=shape, device=self._device)

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------
    @classmethod
    def identity(cls, shape: tuple, data_type: str = "float") -> "TorchMatrix":
        if data_type != "float":
            raise ValueError("Only float type is supported")
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix must be square")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = shape[0]

        # Create row_indices, col_indices, and values for the identity matrix
        row_indices = torch.arange(size, device=device)
        col_indices = torch.arange(size, device=device)
        values = torch.ones(size, device=device)

        indices = torch.stack([row_indices, col_indices])
        data = torch.sparse_coo_tensor(indices, values, shape)
        return cls(shape, data, device)

    @classmethod
    def ones(cls, shape: tuple, data_type: str = "float") -> "TorchMatrix":
        if data_type != "float":
            raise ValueError("Only float type is supported")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        row_indices, col_indices = torch.meshgrid(
            torch.arange(shape[0], device=device), torch.arange(shape[1], device=device)
        )
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        values = torch.ones(shape[0] * shape[1], device=device, dtype=torch.float32)

        indices = torch.stack([row_indices, col_indices])
        data = torch.sparse_coo_tensor(indices, values, shape)
        return cls(shape, data, device)

    @classmethod
    def zeros(cls, shape: tuple, data_type: str = "float") -> "TorchMatrix":
        if data_type != "float":
            raise ValueError("Only float type is supported")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.sparse_coo_tensor(size=shape, device=device)
        return cls(shape, data, device)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def type(self) -> str:
        return "float"

    @property
    def diag(self) -> np.ndarray:
        if self._shape[0] != self._shape[1]:
            raise ValueError("Diagonal is only defined for square matrices")

        row_indices = self._data.indices()[0]
        col_indices = self._data.indices()[1]

        diag_mask = row_indices == col_indices
        diag_values = self._data.values()[diag_mask]
        return diag_values.cpu().numpy()

    @property
    def nnz(self) -> int:
        return self._data._nnz()

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        row, col = index
        mask = (self._data.indices()[0] == row) & (self._data.indices()[1] == col)
        if mask.any():
            return self._data.values()[mask].item()
        else:
            return 0.0

    def __setitem__(self, index: tuple, value: float):
        if isinstance(value, Variable):
            raise ValueError("Only float values are supported")
        row, col = index
        mask = (self._data.indices()[0] == row) & (self._data.indices()[1] == col)
        if mask.any():
            self._data.values()[mask] = value
        else:
            new_indices = torch.cat(
                (
                    self._data.indices(),
                    torch.tensor([[row], [col]], device=self._device),
                ),
                dim=1,
            )
            new_values = torch.cat(
                (self._data.values(), torch.tensor([value], device=self._device))
            )
            self._data = torch.sparse_coo_tensor(
                new_indices,
                new_values,
                self._shape,
            )

    def _check_compatible(self, other: "Matrix"):
        if not isinstance(other, Matrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

        if self.type != other.type:
            raise ValueError(
                f"Matrix types don't match: \
                             {self.type} vs {other.type}."
            )

    def __add__(self, other: "Matrix"):
        self._check_compatible(other)
        if isinstance(other, TorchMatrix):
            return TorchMatrix(self._shape, self._data + other._data, self._device)
        else:
            data = self._data.clone()
            for i in range(self._shape[0]):
                for j in range(self._shape[1]):
                    data[i, j] += other[i, j]
            return TorchMatrix(self._shape, data, self._device)

    def __sub__(self, other: "Matrix"):
        self._check_compatible(other)
        if isinstance(other, TorchMatrix):
            return TorchMatrix(self._shape, self._data - other._data, self._device)
        else:
            data = self._data.clone()
            for i in range(self._shape[0]):
                for j in range(self._shape[1]):
                    data[i, j] -= other[i, j]
            return TorchMatrix(self._shape, data, self._device)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TorchMatrix(self._shape, self._data / other, self._device)
        else:
            raise ValueError("Unsupported operand type for /")

    def __rtruediv__(self, other):
        raise ValueError("Division by a sparse matrix is not supported")

    def __mul__(self, other):
        if isinstance(other, TorchMatrix):
            return TorchMatrix(
                self._shape, torch.sparse.mm(self._data, other._data), self._device
            )
        elif isinstance(other, (int, float)):
            return TorchMatrix(self._shape, self._data * other, self._device)
        else:
            raise ValueError("Unsupported operand type for *")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return TorchMatrix(self._shape, -self._data, self._device)

    def __abs__(self):
        return TorchMatrix(self._shape, torch.abs(self._data), self._device)

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def to_dense(self) -> np.ndarray:
        return self._data.to_dense().cpu().numpy()

    def transpose(self) -> "Matrix":
        return TorchMatrix(self._shape[::-1], self._data.t(), self._device)

    def reshape(self, shape: tuple) -> "Matrix":
        if self._shape != shape:
            raise ValueError("Reshape is not supported for sparse matrices")
        return self

    def scalarize(self) -> list["Matrix"]:
        return [self]
