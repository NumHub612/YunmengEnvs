# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix.
"""
from core.numerics.fields import Variable, VariableType, Scalar
from configs.settings import settings, logger
import scipy.sparse as sp
from scipy.sparse import dok_matrix
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
import torch
import numpy as np
from typing import Any, Tuple, List
import os
import pickle
from abc import abstractmethod

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*")


class Matrix:
    """
    Abstract matrix class.
    """

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(
        cls,
        shape: tuple,
        indices: torch.Tensor | np.ndarray = None,
        values: torch.Tensor | np.ndarray = None,
        device: torch.device = None,
    ) -> "Matrix":
        """Create a matrix from data."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def identity(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        """Create a Identity Matrix."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def zeros(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        """Create a matrix with all elements set to zero."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, file_path: str):
        """Save the matrix to a file."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "Matrix":
        """Load a matrix from a file."""
        raise NotImplementedError()

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def data(self) -> Any:
        """The raw data of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The matrix shape, e.g. (rows, cols)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self) -> VariableType:
        """The matrix data type."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def diag(self) -> list:
        """The diagonal elements of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def nnz(self) -> list[int]:
        """The number of all non-zero elements."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def T(self) -> "Matrix":
        """The transpose of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def inv(self) -> "Matrix":
        """The inverse of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def det(self) -> list[float]:
        """The determinant of the matrix."""
        raise NotImplementedError()

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    @abstractmethod
    def scalarize(self) -> list["Matrix"]:
        """Scalarize the matrix."""
        raise NotImplementedError()

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        """Convert the matrix to a dense format."""
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
    def __add__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __radd__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __iadd__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __rsub__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __isub__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __rmul__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __imul__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __truediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __rtruediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __itruediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError()

    @abstractmethod
    def __abs__(self):
        raise NotImplementedError()


class TorchMatrix(Matrix):
    """
    Torch matrix class implemented using torch.sparse_coo_tensor.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        indices: torch.Tensor | np.ndarray = None,
        values: torch.Tensor | np.ndarray = None,
        device: torch.device = None,
    ):
        """
        Initialize the sparse matrix.

        Args:
            shape: The shape of the matrix (rows, cols).
            indices: The indices of the non-zero elements.
            values: The values of the non-zero elements.
            device: The device to store the matrix.
        """
        self._device = device or settings.DEVICE
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._shape = shape

        self._fptype = torch.float64 if settings.FPTYPE == "fp64" else torch.float32
        if settings.FPTYPE == "fp16":
            self._fptype = torch.float16

        if indices is None or values is None:
            self._values = torch.sparse_coo_tensor(
                size=shape,
                dtype=self._fptype,
                device=self._device,
            ).coalesce()
            self._indices = self._values.indices()
            return

        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=self._fptype, device=self._device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long, device=self._device)

        self._values = torch.sparse_coo_tensor(
            indices,
            values,
            shape,
            dtype=self._fptype,
            device=self._device,
        ).coalesce()

        self._indices = self._values.indices()

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def from_data(
        cls,
        shape: Tuple[int, int],
        indices: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
        device: torch.device = None,
    ) -> "TorchMatrix":
        return cls(shape, indices, values, device)

    @classmethod
    def identity(
        cls,
        shape: Tuple[int, int],
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "TorchMatrix":
        assert shape[0] == shape[1], "Identity matrix must be square."
        indices = torch.tensor(
            [[i, i] for i in range(shape[0])],
            dtype=torch.long,
            device=device,
        ).t()

        fptype = torch.float64 if settings.FPTYPE == "fp64" else torch.float32
        if settings.FPTYPE == "fp16":
            fptype = torch.float16

        values = torch.ones(
            shape[0],
            dtype=fptype,
            device=device,
        )
        return cls(shape, indices, values, device)

    @classmethod
    def zeros(
        cls,
        shape: Tuple[int, int],
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "TorchMatrix":
        return cls(shape, device=device)

    def save(self, file_path: str):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        torch.save(self._values, file_path)

    @classmethod
    def load(cls, file_path: str) -> "TorchMatrix":
        values = torch.load(file_path)
        shape = values.shape
        indices = values.indices()
        return cls(shape, indices, values=values)

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def scalarize(self) -> List["TorchMatrix"]:
        return [self]

    def to_dense(self) -> np.ndarray:
        return self._values.to_dense().cpu().numpy()

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def data(self) -> torch.Tensor:
        return self._values

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> VariableType:
        return VariableType.SCALAR

    @property
    def diag(self) -> List[np.ndarray]:
        diag_indices = self._indices[:, self._indices[0] == self._indices[1]]
        diag_values = self._values.values()[diag_indices[0]]
        return [diag_values.cpu().numpy()]

    @property
    def nnz(self) -> List[int]:
        return [self._values._nnz()]

    @property
    def T(self) -> "TorchMatrix":
        rows, cols = self.shape
        transposed_indices = torch.stack(
            [self._indices[1], self._indices[0]],
            dim=0,
        )
        return TorchMatrix(
            (cols, rows),
            transposed_indices,
            self._values.values(),
            self._device,
        )

    @property
    def inv(self) -> "TorchMatrix":
        dense_matrix = self._values.to_dense()
        inv_dense_matrix = torch.inverse(dense_matrix)
        inv_indices = inv_dense_matrix.nonzero().t()
        inv_values = inv_dense_matrix[inv_indices[0], inv_indices[1]]
        return TorchMatrix(
            self._shape,
            inv_indices,
            inv_values,
            self._device,
        )

    @property
    def det(self) -> List[float]:
        dense_matrix = self._values.to_dense()
        return [torch.det(dense_matrix).item()]

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: Tuple[int, int]):
        value_index = (self._indices[0] == index[0]) & (self._indices[1] == index[1])
        if value_index.any():
            self._values = self._values.coalesce()
            return self._values.values()[value_index].item()
        else:
            return 0.0

    def __setitem__(self, index: Tuple[int, int], value: float):
        value_index = (self._indices[0] == index[0]) & (self._indices[1] == index[1])
        if value_index.any():
            self._values.values()[value_index] = value
        else:
            new_indices = torch.tensor(
                [[index[0]], [index[1]]],
                dtype=torch.long,
                device=self._device,
            )
            new_values = torch.tensor([value], dtype=self._fptype, device=self._device)
            self._indices = torch.cat((self._indices, new_indices), dim=1)
            self._values = torch.cat((self._values.values(), new_values))

            self._values = torch.sparse_coo_tensor(
                self._indices,
                self._values,
                self._shape,
            ).coalesce()
            self._indices = self._values.indices()

    def _check_compatible(self, other: "Matrix", is_mul: bool = False):
        if not isinstance(other, Matrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

        if self.dtype != other.dtype:
            raise ValueError(
                f"Matrix types don't match: \
                             {self.dtype} vs {other.dtype}."
            )
        if is_mul and self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrix shapes don't match for multiplication: \
                             {self.shape} vs {other.shape}."
            )

    def __add__(self, other: "Matrix"):
        self._check_compatible(other)

        if isinstance(other, TorchMatrix):
            combined = (self._values + other._values).coalesce()
            return TorchMatrix(
                self._shape,
                combined.indices(),
                combined.values(),
                self._device,
            )
        else:
            combined = (self._values.to_dense() + other.to_dense()).coalesce()
            return TorchMatrix(
                self._shape,
                combined.indices(),
                combined.values(),
                self._device,
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self._check_compatible(other)

        if isinstance(other, TorchMatrix):
            self._values = (self._values + other._values).coalesce()
            self._indices = self._values.indices()
        else:
            self._values = (self._values.to_dense() + other.to_dense()).coalesce()
            self._indices = self._values.indices()
        return self

    def __sub__(self, other):
        self._check_compatible(other)
        if isinstance(other, TorchMatrix):
            values = (self._values - other._values).coalesce()
            return TorchMatrix(
                self._shape,
                values.indices(),
                values.values(),
                self._device,
            )
        else:
            substracted = (self._values.to_dense() - other.to_dense()).coalesce()
            return TorchMatrix(
                self._shape,
                substracted.indices(),
                substracted.values(),
                self._device,
            )

    def __rsub__(self, other):
        return other.__sub__(self)

    def __isub__(self, other):
        self._check_compatible(other)

        if isinstance(other, TorchMatrix):
            self._values = (self._values - other._values).coalesce()
            self._indices = self._values.indices()
        else:
            negated_values = -other.to_dense()
            other_dense = other.to_dense()
            negated_values = -other_dense
            self._values = torch.cat((self._values, negated_values)).coalesce()
            self._indices = self._values.indices()
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TorchMatrix(
                self._shape,
                self._indices,
                self._values.values() * other,
                self._device,
            )

        self._check_compatible(other, is_mul=True)
        if isinstance(other, TorchMatrix):
            result_data = torch.sparse.mm(self._values, other._values)
            result_indices = result_data.indices()
            result_values = result_data.values()
            shape = (self._shape[0], other._shape[1])
            return TorchMatrix(
                shape,
                result_indices,
                result_values,
                self._device,
            )
        else:
            result_data = torch.sparse.mm(self._values, other.to_dense())
            result_indices = result_data.indices()
            result_values = result_data.values()
            shape = (self._shape[0], other.shape[1])
            return TorchMatrix(
                shape,
                result_indices,
                result_values,
                self._device,
            )

    def __rmul__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._values = self._values * other
            return self

        self._check_compatible(other, is_mul=True)
        if isinstance(other, TorchMatrix):
            self._values = torch.sparse.mm(self._values, other._values)
            self._indices = self._values.indices()
        else:
            self._values = torch.sparse.mm(
                self._values.to_dense(),
                other.to_dense(),
            )
            self._indices = self._values.indices()
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TorchMatrix(
                self._shape,
                self._indices,
                self._values.values() / other,
                self._device,
            )
        else:
            raise ValueError("Unsupported operand type for /")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return TorchMatrix(
                self._shape,
                self._indices,
                other / self._values,
                self._device,
            )
        else:
            raise ValueError("Unsupported operand type for /")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._values = self._values / other
            return self
        else:
            raise ValueError("Unsupported operand type for /")

    def __neg__(self):
        negated_values = -self._values.values()
        return TorchMatrix(
            self._shape,
            self._indices,
            negated_values,
            self._device,
        )

    def __abs__(self):
        abs_values = torch.abs(self._values.values())
        return TorchMatrix(
            self._shape,
            self._indices,
            abs_values,
            self._device,
        )


class CupyMatrix(Matrix):
    """
    Cupy matrix class supported Scalar type matrix only.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        row_indices: torch.Tensor | np.ndarray | cp.ndarray = None,
        col_indices: torch.Tensor | np.ndarray | cp.ndarray = None,
        values: torch.Tensor | np.ndarray | cp.ndarray = None,
    ):
        """
        Initialize the sparse matrix.

        Args:
            shape: The shape of the matrix (rows, cols).
            row_indices: The row indices of the non-zero elements.
            col_indices: The column indices of the non-zero elements.
            values: The values of the non-zero elements.
        """
        fptype = cp.float64 if settings.FPTYPE == "fp64" else cp.float32
        if settings.FPTYPE == "fp16":
            fptype = cp.float16

        self._shape = shape
        if row_indices is None or col_indices is None or values is None:
            self._data = coo_matrix(shape, dtype=fptype)
            return

        if not isinstance(row_indices, cp.ndarray):
            row_indices = cp.asarray(row_indices, dtype=cp.int32)
        if not isinstance(col_indices, cp.ndarray):
            col_indices = cp.asarray(col_indices, dtype=cp.int32)
        if not isinstance(values, cp.ndarray):
            values = cp.asarray(values, dtype=fptype)

        self._data = coo_matrix(
            (
                values,
                (
                    row_indices,
                    col_indices,
                ),
            ),
            shape=shape,
            dtype=fptype,
        )

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def from_matrix(cls, mat: coo_matrix) -> "CupyMatrix":
        mat = mat.tocoo()
        if not isinstance(mat, coo_matrix):
            raise TypeError("Input must be a coo_matrix")
        mat.eliminate_zeros()
        return cls(mat.shape, mat.row, mat.col, mat.data)

    @classmethod
    def from_data(
        cls,
        shape: tuple,
        indices: torch.Tensor | np.ndarray | cp.ndarray = None,
        values: torch.Tensor | np.ndarray | cp.ndarray = None,
        device: str = None,
    ) -> "CupyMatrix":
        row_indices = indices[0]
        col_indices = indices[1]
        return cls(shape, row_indices, col_indices, values)

    @classmethod
    def identity(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: str = None,
    ) -> "CupyMatrix":
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Identity matrix must be square")

        rows = cp.arange(shape[0], dtype=cp.int32)
        cols = cp.arange(shape[1], dtype=cp.int32)
        data = cp.ones(shape[0], dtype=cp.float32)
        return cls(shape, rows, cols, data)

    @classmethod
    def zeros(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: str = None,
    ) -> "CupyMatrix":
        return cls(shape)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def data(self) -> coo_matrix:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> VariableType:
        return VariableType.SCALAR

    @property
    def diag(self) -> List:
        return [self._data.diagonal()]

    @property
    def nnz(self) -> List[int]:
        return [self._data.nnz]

    @property
    def T(self) -> "CupyMatrix":
        return CupyMatrix.from_matrix(self._data.transpose())

    @property
    def inv(self) -> "CupyMatrix":
        dense_inv = cp.linalg.inv(self._data.todense())
        mat_arr = cp.asarray(dense_inv)
        mat = coo_matrix(mat_arr)
        return CupyMatrix.from_matrix(mat)

    @property
    def det(self) -> List[float]:
        return [cp.linalg.det(self._data.todense())]

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def scalarize(self) -> List["CupyMatrix"]:
        return [self]

    def to_dense(self) -> np.ndarray:
        dense_mat = self._data.todense()
        return cp.asnumpy(dense_mat)

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value

    def _check_compatible(self, other: "Matrix", is_mul: bool = False):
        if not isinstance(other, Matrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

        if self.dtype != other.dtype:
            raise ValueError(
                f"Matrix types don't match: \
                             {self.dtype} vs {other.dtype}."
            )
        if is_mul and self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrix shapes don't match for multiplication: \
                             {self.shape} vs {other.shape}."
            )

    def __add__(self, other):
        self._check_compatible(other)

        if isinstance(other, CupyMatrix):
            return CupyMatrix.from_matrix(self._data + other.data)
        elif isinstance(other, Matrix):
            res_mat = self.to_dense() + other.to_dense()
            res_mat.eliminate_zeros()
            indices = cp.nonzero(res_mat)
            return CupyMatrix.from_data(self.shape, indices, res_mat)
        else:
            raise TypeError("Unsupported operand type for +")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self._check_compatible(other)

        if isinstance(other, CupyMatrix):
            self._data += other._data
            self._data.eliminate_zeros()
            return self
        else:
            raise TypeError("Unsupported operand type for +=")

    def __sub__(self, other):
        self._check_compatible(other)

        if isinstance(other, CupyMatrix):
            return CupyMatrix.from_matrix(self._data - other.data)
        elif isinstance(other, Matrix):
            res_mat = self.to_dense() - other.to_dense()
            res_mat.eliminate_zeros()
            indices = cp.nonzero(res_mat)
            return CupyMatrix.from_data(self.shape, indices, res_mat)
        else:
            raise TypeError("Unsupported operand type for -")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        self._check_compatible(other)
        if isinstance(other, CupyMatrix):
            self._data -= other._data
            self._data.eliminate_zeros()
            return self
        else:
            raise TypeError("Unsupported operand type for -=")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return CupyMatrix.from_matrix(self._data * other)

        self._check_compatible(other, is_mul=True)
        if isinstance(other, CupyMatrix):
            return CupyMatrix.from_matrix(self._data @ other.data)
        elif isinstance(other, Matrix):
            res_mat = self.to_dense() @ other.to_dense()
            res_mat.eliminate_zeros()
            indices = cp.nonzero(res_mat)
            return CupyMatrix.from_data(self.shape, indices, res_mat)
        else:
            raise TypeError("Unsupported operand type for *")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = Scalar.value
            return self.__mul__(other)

        raise TypeError("Unsupported operand type for *")

    def __imul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = Scalar.value
            self._data *= other
            self._data.eliminate_zeros()
            return self

        self._check_compatible(other, is_mul=True)
        if isinstance(other, CupyMatrix):
            self._data = self._data @ other._data
            self._data.eliminate_zeros()
            return self
        else:
            raise TypeError("Unsupported operand type for *=")

    def __truediv__(self, other):
        if not isinstance(other, (int, float, Scalar)):
            raise ValueError("Unsupported operand type for /")
        if isinstance(other, Scalar):
            other = other.value

        return CupyMatrix.from_matrix(self._data / other)

    def __rtruediv__(self, other):
        raise NotImplementedError("Not supported.")

    def __itruediv__(self, other):
        if not isinstance(other, (int, float, Scalar)):
            raise ValueError("Unsupported operand type for /=")
        if isinstance(other, Scalar):
            other = other.value

        self._data /= other
        return self

    def __neg__(self):
        return CupyMatrix.from_matrix(-self._data)

    def __abs__(self):
        return CupyMatrix.from_matrix(cp.abs(self._data))


class SciMatrix(Matrix):
    """
    SciPy sparse matrix only supports float data type.
    """

    def __init__(
        self,
        shape: tuple,
        row_indices: torch.Tensor | np.ndarray = None,
        col_indices: torch.Tensor | np.ndarray = None,
        values: torch.Tensor | np.ndarray = None,
    ):
        """Complex matrix class.

        Args:
            shape: The matrix shape.
            row_indices: The row indices of the non-zero elements.
            col_indices: The column indices of the non-zero elements.
            values: The values of the non-zero elements.
        """
        self._shape = shape
        self._data = None

        fptype = np.float64 if settings.FPTYPE == "fp64" else np.float32
        if settings.FPTYPE == "fp16":
            fptype = np.float16

        if row_indices is None or col_indices is None or values is None:
            self._data = dok_matrix(shape, dtype=fptype)
            return

        if isinstance(row_indices, torch.Tensor):
            row_indices = row_indices.cpu().numpy()
        if isinstance(col_indices, torch.Tensor):
            col_indices = col_indices.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()

        coo = sp.coo_matrix(
            (values, (row_indices, col_indices)), shape=shape, dtype=fptype
        )
        self._data = coo.todok()

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def from_matrix(cls, mat: dok_matrix) -> "CupyMatrix":
        mat = mat.todok()
        if not isinstance(mat, dok_matrix):
            raise TypeError("Input must be a dok_matrix")

        indices = mat.nonzero()
        values = np.array(list(mat.values()))
        return cls(mat.shape, indices[0], indices[1], values)

    @classmethod
    def from_data(
        cls,
        shape: tuple,
        indices: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
        device: torch.device = None,
    ) -> "Matrix":
        return cls(shape, indices[0], indices[1], values)

    @classmethod
    def zeros(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        return SciMatrix(shape)

    @classmethod
    def identity(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix must be squared.")

        rows = np.arange(shape[0])
        cols = np.arange(shape[1])
        data = np.ones(shape[0])
        return SciMatrix(shape, rows, cols, data)

    def save(self, file_path: str):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "SciMatrix":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def data(self) -> dok_matrix:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def diag(self) -> list[np.ndarray]:
        return [self._data.diagonal()]

    @property
    def dtype(self) -> VariableType:
        return VariableType.SCALAR

    @property
    def nnz(self) -> list[int]:
        return [self._data.nnz]

    @property
    def inv(self) -> "Matrix":
        dense_inv = np.linalg.inv(self._data.todense())
        new_mat = dok_matrix(dense_inv)
        return SciMatrix.from_matrix(new_mat)

    @property
    def T(self) -> "Matrix":
        return SciMatrix.from_matrix(self._data.transpose())

    @property
    def det(self) -> list[float]:
        return [np.linalg.det(self._data.todense())]

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def _check_compatible(self, other: "Matrix", is_mul: bool = False):
        if not isinstance(other, Matrix):
            raise ValueError(f"Invalid operand type {type(other)}.")
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shapes don't match: {self.shape} vs {other.shape}."
            )

        if self.dtype != other.dtype:
            raise ValueError(
                f"Matrix types don't match: \
                             {self.dtype} vs {other.dtype}."
            )
        if is_mul and self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrix shapes don't match for multiplication: \
                             {self.shape} vs {other.shape}."
            )

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value

    def __add__(self, other: "Matrix"):
        self._check_compatible(other)

        if isinstance(other, SciMatrix):
            return SciMatrix.from_matrix(self._data + other._data)
        elif isinstance(other, Matrix):
            data = dok_matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    data[i, j] = self._data[i, j] + other[i, j]
            return SciMatrix.from_matrix(data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self._check_compatible(other)
        if isinstance(other, SciMatrix):
            self._data += other._data
            self._data = self._data.todok()
            return self

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self._data[i, j] += other[i, j]
        self._data = self._data.todok()
        return self

    def __sub__(self, other):
        self._check_compatible(other)
        if isinstance(other, SciMatrix):
            return SciMatrix.from_matrix(self._data - other.data)

        data = dok_matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data[i, j] = self._data[i, j] - other[i, j]
        return SciMatrix.from_matrix(data)

    def __rsub__(self, other):
        self._check_compatible(other)
        if isinstance(other, SciMatrix):
            return SciMatrix.from_matrix(other.data - self._data)

        data = dok_matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data[i, j] = other[i, j] - self._data[i, j]
        return SciMatrix.from_matrix(data)

    def __isub__(self, other):
        self._check_compatible(other)
        if isinstance(other, SciMatrix):
            self._data -= other._data
            self._data = self._data.todok()
            return self

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self._data[i, j] -= other[i, j]
        self._data = self._data.todok()
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix.from_matrix(self._data * other)

        self._check_compatible(other, is_mul=True)
        if isinstance(other, SciMatrix):
            return SciMatrix.from_matrix(self._data @ other.data)

        elif isinstance(other, Matrix):
            shape = (self.shape[0], other.shape[1])
            data = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(self.shape[1]):
                        data[i, j] += self._data[i, k] * other[k, j]
            return SciMatrix.from_matrix(dok_matrix(data))
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix.from_matrix(other * self._data)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __imul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            self._data *= other
            self._data = self._data.todok()
            return self
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            return SciMatrix.from_matrix(self._data / other)
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __itruediv__(self, other):
        if isinstance(other, (int, float, Scalar)):
            if isinstance(other, Scalar):
                other = other.data
            self._data /= other
            self._data = self._data.todok()
            return self
        else:
            raise ValueError(f"Invalid operand type {type(other)}.")

    def __neg__(self):
        return SciMatrix.from_matrix(-self._data)

    def __abs__(self):
        return SciMatrix.from_matrix(abs(self._data))

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def to_dense(self) -> np.ndarray:
        return self._data.toarray()

    def scalarize(self) -> list["SciMatrix"]:
        return [self]


class SparseMatrix(Matrix):
    """
    Sparse matrix class implemented

    TODO: support Variable elements.
    """

    SIZE_MAP = {
        VariableType.SCALAR: 1,
        VariableType.VECTOR: 3,
        VariableType.TENSOR: 9,
    }

    def __init__(
        self,
        shape: Tuple[int, int],
        indices: torch.Tensor | np.ndarray = None,
        values: torch.Tensor | np.ndarray = None,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
        backend: str = "torch",
    ):
        """
        Initialize the sparse matrix.

        Args:
            shape: The shape of the matrix (rows, cols).
            indices: The indices of the non-zero elements.
            values: The values of the non-zero elements.
            data_type: The data type of the matrix.
            device: The device to store the matrix.
            backend: The backend to use for the matrix.
        """
        self._device = device or settings.DEVICE
        self._shape = shape
        self._dtype = data_type
        self._values = []

        if settings.DEVICE != "cuda" and backend == "cupy":
            backend = "torch"
            logger.warning("Using torch backend instead of cupy.")

        size = self.SIZE_MAP[data_type]
        if backend == "cupy":
            self._values = self._create_cupy_matrix(shape, indices, values, size)
        elif backend == "torch":
            self._values = self._create_torch_matrix(shape, indices, values, size)
        elif backend == "scipy":
            self._values = self._create_scipy_matrix(shape, indices, values, size)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self._backend = backend

    def _create_cupy_matrix(self, shape, indices, values, size):
        mats = []
        if values is None or indices is None:
            for i in range(size):
                mats.append(CupyMatrix(shape))
        else:
            values = values.reshape(-1, size)
            rows = indices[0]
            cols = indices[1]
            for i in range(size):
                value = values[:, i]
                mats.append(CupyMatrix.from_data(shape, [rows, cols], value))
        return mats

    def _create_torch_matrix(self, shape, indices, values, size):
        mats = []
        if values is None or indices is None:
            for i in range(size):
                mats.append(TorchMatrix(shape, device=self._device))
        else:
            values = values.reshape(-1, size)
            for i in range(size):
                value = values[:, i]
                mats.append(
                    TorchMatrix(
                        shape,
                        indices,
                        value,
                        device=self._device,
                    )
                )
        return mats

    def _create_scipy_matrix(self, shape, indices, values, size):
        mats = []
        if values is None or indices is None:
            for i in range(size):
                mats.append(SciMatrix(shape))
        else:
            values = values.reshape(-1, size)
            rows = indices[0]
            cols = indices[1]
            for i in range(size):
                value = values[:, i]
                mats.append(SciMatrix(shape, rows, cols, value))
        return mats

    # -----------------------------------------------
    # --- class methods ---
    # -----------------------------------------------

    @classmethod
    def from_data(
        cls,
        shape: tuple,
        indices: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
        device: torch.device = None,
    ) -> "Matrix":
        dtype = None
        if values.ndim == 1:
            dtype = VariableType.SCALAR
        elif values.ndim == 2:
            if values.shape[1] == 1:
                dtype = VariableType.SCALAR
            elif values.shape[1] == 3:
                dtype = VariableType.VECTOR
            elif values.shape[1] == 9:
                dtype = VariableType.TENSOR
        elif values.ndim == 3:
            if values.shape[1] == 3 and values.shape[2] == 3:
                dtype = VariableType.TENSOR
                values = values.reshape(-1, 9)
            elif values.shape[1] == 3 and values.shape[2] == 1:
                dtype = VariableType.VECTOR
                values = values.reshape(-1, 3)
            elif values.shape[1] == 1 and values.shape[2] == 1:
                dtype = VariableType.SCALAR
                values = values.reshape(-1, 1)
        if dtype is None:
            raise ValueError(f"Invalid shape: {values.shape}")

        return cls(shape, indices, values, dtype, device)

    @classmethod
    def identity(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        assert shape[0] == shape[1], "Identity matrix must be square."
        indices = torch.tensor(
            [[i, i] for i in range(shape[0])],
            dtype=torch.long,
            device=device,
        ).t()

        fptype = torch.float64 if settings.FPTYPE == "fp64" else torch.float32
        if settings.FPTYPE == "fp16":
            fptype = torch.float16

        size = cls.SIZE_MAP[data_type]
        values = torch.ones(
            shape[0],
            size,
            dtype=fptype,
            device=device,
        )
        return cls(shape, indices, values, data_type, device)

    @classmethod
    def zeros(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        return cls(shape, device=device)

    def save(self, file_path: str):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "SparseMatrix":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def backend(self) -> str:
        """The backend used for the matrix."""
        return self._backend

    @property
    def data(self) -> List[Matrix]:
        return self._values

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> VariableType:
        return self._dtype

    @property
    def diag(self) -> list[np.ndarray]:
        digs = [mat.diag[0] for mat in self._values]
        return digs

    @property
    def nnz(self) -> list[int]:
        nnz = [mat.nnz[0] for mat in self._values]
        return nnz

    @property
    def T(self) -> "SparseMatrix":
        new_values = []
        for mat in self._values:
            new_values.append(mat.T)
        new_mat = SparseMatrix(
            self._shape[::-1],
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    @property
    def inv(self) -> "SparseMatrix":
        new_values = []
        for mat in self._values:
            new_values.append(mat.inv)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    @property
    def det(self) -> list[float]:
        dets = [mat.det[0] for mat in self._values]
        return dets

    # -----------------------------------------------
    # --- matrix methods ---
    # -----------------------------------------------

    def scalarize(self) -> list[Matrix]:
        return self._values

    def to_dense(self) -> np.ndarray:
        dense_mats = []
        for mat in self._values:
            dense_mats.append(mat.to_dense())
        return np.stack(dense_mats, axis=2)

    # -----------------------------------------------
    # --- overload methods ---
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        item = [v[index] for v in self._values]
        return item

    def __setitem__(self, index: tuple, value: float | Variable):
        for v in self._values:
            v[index] = value

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Unsupported operand type for +")

        new_values = []
        if isinstance(other, SparseMatrix):
            for i in range(len(self._values)):
                mat = self._values[i] + other._values[i]
                new_values.append(mat)
        else:
            for mat in self._values:
                mat = mat + other
                new_values.append(mat)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Unsupported operand type for +")

        if isinstance(other, SparseMatrix):
            for i in range(len(self._values)):
                self._values[i] += other._values[i]
                self._values[i] = self._values[i]
        else:
            for mat in self._values:
                mat += other
        return self

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Unsupported operand type for -")
        new_values = []
        if isinstance(other, SparseMatrix):
            for i in range(len(self._values)):
                mat = self._values[i] - other._values[i]
                new_values.append(mat)
        else:
            for mat in self._values:
                mat = mat - other
                new_values.append(mat)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __rsub__(self, other):
        return other.__sub__(self)

    def __isub__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Unsupported operand type for -")

        if isinstance(other, SparseMatrix):
            for i in range(len(self._values)):
                self._values[i] -= other._values[i]
                self._values[i] = self._values[i]
        else:
            for mat in self._values:
                mat -= other
        return self

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for *")

        new_values = []
        for mat in self._values:
            new_values.append(mat * other)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for *")

        return self.__mul__(other)

    def __imul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for *")

        for mat in self._values:
            mat *= other
        return self

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for /")

        new_values = []
        for mat in self._values:
            new_values.append(mat / other)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __rtruediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for /")

        new_values = []
        for mat in self._values:
            new_values.append(other / mat)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __itruediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Unsupported operand type for /")

        for mat in self._values:
            mat /= other
        return self

    def __neg__(self):
        new_values = []
        for mat in self._values:
            new_values.append(-mat)
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat

    def __abs__(self):
        new_values = []
        for mat in self._values:
            new_values.append(abs(mat))
        new_mat = SparseMatrix(
            self._shape,
            data_type=self._dtype,
            device=self._device,
        )
        new_mat._values = new_values
        return new_mat
