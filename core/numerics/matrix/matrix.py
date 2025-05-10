# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix.
"""
from core.numerics.fields import Variable, VariableType, Scalar, Vector, Tensor
from configs.settings import settings, logger

import torch
import numpy as np
from typing import Any, Tuple, List, Union
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
        indices: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
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
    def ones(
        cls,
        shape: tuple,
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "Matrix":
        """Create a matrix with all elements set to one."""
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
    def diag(self) -> list[np.ndarray]:
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
        self._shape = shape

        if indices is None or values is None:
            self._values = torch.sparse_coo_tensor(
                size=shape,
                dtype=settings.DTYPE,
                device=self._device,
            ).coalesce()
            self._indices = self._values.indices()
            return

        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=settings.DTYPE, device=self._device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long, device=self._device)

        self._values = torch.sparse_coo_tensor(
            indices,
            values,
            shape,
            dtype=settings.DTYPE,
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
        values = torch.ones(
            shape[0],
            dtype=settings.DTYPE,
            device=device,
        )
        return cls(shape, indices, values, device)

    @classmethod
    def ones(
        cls,
        shape: Tuple[int, int],
        data_type: VariableType = VariableType.SCALAR,
        device: torch.device = None,
    ) -> "TorchMatrix":
        indices = torch.tensor(
            [[i, j] for i in range(shape[0]) for j in range(shape[1])],
            dtype=torch.long,
            device=device,
        ).t()
        values = torch.ones(
            shape[0] * shape[1],
            dtype=settings.DTYPE,
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
    def data(self) -> torch.sparse_coo_tensor:
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
            new_values = torch.tensor(
                [value], dtype=settings.DTYPE, device=self._device
            )
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
            negated_values = -other.to_dense()
            combined_indices = torch.cat(
                (self._indices, other._indices),
                dim=1,
            )
            combined_values = torch.cat((self._values, negated_values))
            return TorchMatrix(
                self._shape,
                combined_indices,
                combined_values,
                self._device,
            )

    def __rsub__(self, other):
        return other.__sub__(self)

    def __isub__(self, other):
        self._check_compatible(other)

        if isinstance(other, TorchMatrix):
            negated_indices = other._indices
            negated_values = -other._values
            self._indices = torch.cat(
                (self._indices, negated_indices),
                dim=1,
            )
            self._values = torch.cat((self._values, negated_values))
        else:
            other_dense = other.to_dense()
            negated_values = -other_dense
            self._indices = torch.cat(
                (self._indices, other._indices),
                dim=1,
            )
            self._values = torch.cat((self._values, negated_values))

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


class SparseMatrix(Matrix):
    """
    Sparse matrix class implemented
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
        self._shape = shape

        if indices is None or values is None:
            self._values = torch.sparse_coo_tensor(
                size=shape,
                dtype=settings.DTYPE,
                device=self._device,
            ).coalesce()
            self._indices = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=self._device,
            )
            return

        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=settings.DTYPE)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long)

        self._values = torch.sparse_coo_tensor(
            indices,
            values,
            shape,
            dtype=settings.DTYPE,
            device=self._device,
        ).coalesce()
        self._indices = self._values.indices()
