# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Sparse matrixs.
"""
# -*- encoding: utf-8 -*-
"""
PyTorch Implementation of the Matrix Interface.
Optimized for GPU-accelerated CFD solvers and AutoDiff.
"""
from yunmeng.numerics.mats.matrix import Matrix, TensorLike
import scipy.sparse as sp
import torch
import numpy as np
from typing import Union, Tuple, List


# -----------------------------------------------
# region TorchMatrix
# -----------------------------------------------


class TorchMatrix(Matrix):
    """
    Concrete implementation using PyTorch Sparse Tensors.
    Supports CUDA, AutoDiff, and mixed precision.
    """

    def __init__(self, tensor: torch.Tensor):
        self._data = tensor

    # -----------------------------------------------
    # Factory Methods
    # -----------------------------------------------

    @classmethod
    def from_data(
        cls,
        values: TensorLike,
        indices: TensorLike = None,
        shape: Tuple[int, int] = None,
        device: torch.device = None,
    ) -> "TorchMatrix":
        values = torch.as_tensor(values)
        if indices is not None:
            # Sparse tensor
            if isinstance(indices, tuple):
                row_idx = torch.as_tensor(indices[0])
                col_idx = torch.as_tensor(indices[1])
                indices = torch.stack([row_idx, col_idx], dim=0)
            else:
                indices = torch.as_tensor(indices)

            device = device or values.device
            indices = indices.to(device)
            values = values.to(device)
            if shape is None:
                rows = int(indices[0].max()) + 1
                cols = int(indices[1].max()) + 1
                shape = (rows, cols)

            # Create COO sparse tensor
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, device=device
            )
            sparse_tensor = sparse_tensor.coalesce()

            # Convert to CSR tensor
            try:
                if device.type == "cuda":
                    csr_tensor = sparse_tensor.to_sparse_csr()
                    return cls(csr_tensor)
            except Exception:
                pass
            return cls(sparse_tensor)
        else:
            # Dense tensor
            if shape is not None:
                values = values.reshape(shape)
            if device is not None:
                values = values.to(device)
            return cls(values)

    @classmethod
    def from_coo(
        cls,
        shape: Tuple[int, int],
        values: TensorLike,
        rows: TensorLike,
        cols: TensorLike,
        device: torch.device = None,
    ) -> "TorchMatrix":
        return cls.from_data(values, indices=(rows, cols), shape=shape, device=device)

    @classmethod
    def from_csr(
        cls,
        shape: Tuple[int, int],
        values: TensorLike,
        ptrs: TensorLike,
        idxs: TensorLike,
        device: torch.device = None,
    ) -> "TorchMatrix":
        values = torch.as_tensor(values)
        ptrs = torch.as_tensor(ptrs)
        idxs = torch.as_tensor(idxs)

        device = device or values.device
        values = values.to(device)
        ptrs = ptrs.to(device)
        idxs = idxs.to(device)

        csr_tensor = torch.sparse_csr_tensor(
            ptrs, idxs, values, size=shape, device=device
        )
        return cls(csr_tensor)

    @classmethod
    def zeros(
        cls,
        shape: Tuple[int, int],
        device: torch.device = None,
    ) -> "TorchMatrix":
        # Creating an empty COO tensor
        device = device or torch.device("cpu")
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty(0, dtype=torch.float64, device=device)
        sparse_zero = torch.sparse_coo_tensor(
            indices, values, size=shape, device=device
        )
        return cls(sparse_zero)

    @classmethod
    def identity(
        cls,
        size: int,
        device: torch.device = None,
    ) -> "TorchMatrix":
        # Create diagonal indices
        device = device or torch.device("cpu")
        idx = torch.arange(size, device=device)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.full((size,), 1.0, dtype=torch.float64, device=device)
        sparse_eye = torch.sparse_coo_tensor(
            indices, values, size=(size, size), device=device
        ).coalesce()
        return cls(sparse_eye)

    # -----------------------------------------------
    # Properties
    # -----------------------------------------------

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def nnz(self) -> int:
        if self._data.is_sparse:
            # Coalesce the tensor before getting values
            coalesced = self._data.coalesce()
            return coalesced.values().shape[0]
        else:
            return torch.count_nonzero(self._data).item()

    @property
    def device(self) -> torch.device:
        return self._data.device

    @property
    def T(self) -> "TorchMatrix":
        transposed = self._data.transpose(0, 1)
        # Coalesce the transposed tensor if it's sparse
        if transposed.is_sparse:
            transposed = transposed.coalesce()
        return TorchMatrix(transposed)

    @property
    def diags(self) -> List[torch.Tensor]:
        if self._data.is_sparse:
            coo = self._data.coalesce()
            rows = coo.indices()[0]
            cols = coo.indices()[1]
            vals = coo.values()
            mask = rows == cols
            return [vals[mask]]
        else:
            return [torch.diag(self._data)]

    # -----------------------------------------------
    # Utils
    # -----------------------------------------------

    def to(self, device: torch.device) -> "TorchMatrix":
        return TorchMatrix(self._data.to(device))

    def convert(self, format_str: str) -> "TorchMatrix":
        if format_str == "csr":
            if self._data.layout == torch.sparse_csr:
                return self
            return TorchMatrix(self._data.to_sparse_csr())
        elif format_str == "coo":
            if self._data.layout == torch.sparse_coo:
                return self
            return TorchMatrix(self._data.to_sparse_coo())
        else:
            raise RuntimeError(f"Format {format_str} not supported.")

    def to_numpy(self) -> "NumpyMatrix":
        """Convert back to NumpyMatrix."""
        self_cpu = self._data.cpu()
        if self_cpu.is_sparse:
            coo = self_cpu.coalesce()
            indices = coo.indices().numpy()
            values = coo.values().numpy()
            return NumpyMatrix.from_data(
                values,
                indices=(indices[0], indices[1]),
                shape=self.shape,
            )
        else:
            return NumpyMatrix.from_data(self_cpu.numpy())

    def to_torch(self, device: torch.device = None):
        return super().to_torch(device)

    def diagonal(self, offset: int = 0) -> torch.Tensor:
        if self._data.is_sparse:
            coo = self._data.coalesce()
            rows = coo.indices()[0]
            cols = coo.indices()[1]
            vals = coo.values()
            if offset == 0:
                mask = rows == cols
            elif offset > 0:
                mask = (cols - rows) == offset
            else:
                mask = (rows - cols) == -offset
            return vals[mask]
        else:
            return torch.diag(self._data, offset=offset)

    # -----------------------------------------------
    # Operators
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value

    def __matmul__(
        self, other: Union["TorchMatrix", torch.Tensor]
    ) -> Union["TorchMatrix", torch.Tensor]:
        if isinstance(other, TorchMatrix):
            res = self._data @ other._data
            return TorchMatrix(res)
        elif isinstance(other, torch.Tensor):
            # Matrix-Vector (Sparse @ Dense) -> Dense
            if other.dim() == 1:
                other = other.unsqueeze(-1)  # to column vector
                res = self._data @ other
                return res.squeeze(-1)
            elif other.dim() == 2:
                return self._data @ other
            else:
                raise ValueError("Invalid tensor dimension")
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    def __add__(self, other: "TorchMatrix") -> "TorchMatrix":
        if not isinstance(other, TorchMatrix):
            raise TypeError("Addition requires TorchMatrix")
        return TorchMatrix(self._data + other._data)

    def __sub__(self, other: "TorchMatrix") -> "TorchMatrix":
        if not isinstance(other, TorchMatrix):
            raise TypeError("Subtraction requires TorchMatrix")
        return TorchMatrix(self._data - other._data)

    def __mul__(self, scalar: float) -> "TorchMatrix":
        return TorchMatrix(self._data * scalar)

    def __rmul__(self, scalar: float) -> "TorchMatrix":
        return self.__mul__(scalar)


# -----------------------------------------------
# region NumpyMatrix
# -----------------------------------------------


class NumpyMatrix(Matrix):
    """
    NumpyMatrix class optimized for CFD applications.
    """

    def __init__(self, sparse_matrix: sp.spmatrix):
        if not sp.issparse(sparse_matrix):
            raise TypeError("Input must be a scipy sparse matrix")
        self._data = sparse_matrix

    # -----------------------------------------------
    # Factory Methods
    # -----------------------------------------------

    @classmethod
    def from_data(
        cls,
        values: np.ndarray,
        indices: Tuple[np.ndarray, np.ndarray] = None,
        shape: Tuple[int, int] = None,
        device: torch.device = None,
    ) -> "NumpyMatrix":
        if device is not None and device != "cpu":
            raise ValueError("Only supports CPU ('cpu' or None).")

        if indices is None:
            # Dense matrix
            mat = sp.csr_matrix(values)
        else:
            # Sparse matrix (COO style)
            row_idx, col_idx = indices
            if shape is None:
                rows = int(np.max(row_idx)) + 1
                cols = int(np.max(col_idx)) + 1
                shape = (rows, cols)
            mat = sp.coo_matrix(
                (values, (row_idx, col_idx)),
                shape=shape,
            ).tocsr()
        return cls(mat)

    @classmethod
    def zeros(
        cls, shape: Tuple[int, int], device: torch.device = None
    ) -> "NumpyMatrix":
        if device is not None and device != "cpu":
            raise ValueError("NumpyMatrix only supports CPU.")
        # Create an empty sparse matrix
        mat = sp.csr_matrix(shape, dtype=np.float64)
        return cls(mat)

    @classmethod
    def identity(cls, size: int, device: torch.device = None) -> "NumpyMatrix":
        if device is not None and device != "cpu":
            raise ValueError("NumpyMatrix only supports CPU.")
        mat = sp.eye(size, dtype=np.float64)
        return cls(mat)

    # -----------------------------------------------
    # Properties
    # -----------------------------------------------

    @property
    def data(self) -> sp.spmatrix:
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def nnz(self) -> int:
        return self._data.nnz

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def T(self) -> "NumpyMatrix":
        return NumpyMatrix(self._data.transpose())

    @property
    def diags(self) -> List[np.ndarray]:
        # Returns the main diagonal (offset 0)
        return [self._data.diagonal(k=0)]

    # -----------------------------------------------
    # Operators
    # -----------------------------------------------

    def __getitem__(self, index: tuple):
        return self._data[index]

    def __setitem__(self, index: tuple, value: float):
        self._data[index] = value

    def __matmul__(
        self, other: Union["NumpyMatrix", torch.Tensor]
    ) -> Union["NumpyMatrix", torch.Tensor]:
        if isinstance(other, NumpyMatrix):
            return NumpyMatrix(self._data @ other._data)
        elif isinstance(other, np.ndarray):
            # Sparse matrix-vector multiplication
            if other.ndim == 1:
                return self._data @ other
            elif other.ndim == 2:
                return self._data @ other
            else:
                raise ValueError("Vector must be 1D or 2D numpy array")
        else:
            raise TypeError(f"Unsupported type for matmul: {type(other)}")

    def __add__(self, other: "NumpyMatrix") -> "NumpyMatrix":
        if not isinstance(other, NumpyMatrix):
            raise TypeError("Can only add NumpyMatrix to NumpyMatrix")
        return NumpyMatrix(self._data + other._data)

    def __sub__(self, other: "NumpyMatrix") -> "NumpyMatrix":
        if not isinstance(other, NumpyMatrix):
            raise TypeError("Can only subtract NumpyMatrix from NumpyMatrix")
        return NumpyMatrix(self._data - other._data)

    def __mul__(self, scalar: float) -> "NumpyMatrix":
        return NumpyMatrix(self._data * scalar)

    def __rmul__(self, scalar: float) -> "NumpyMatrix":
        return self.__mul__(scalar)

    # -----------------------------------------------
    # Utilities
    # -----------------------------------------------

    def to(self, device: str) -> "NumpyMatrix":
        if device != "cpu":
            raise ValueError("NumpyMatrix cannot move to GPU.")
        return self

    def to_numpy(self) -> "NumpyMatrix":
        return self

    def to_torch(self, device: torch.device = None) -> "TorchMatrix":
        self_coo = self._data.tocoo()
        rows = torch.from_numpy(self_coo.row)
        cols = torch.from_numpy(self_coo.col)
        vals = torch.from_numpy(self_coo.data)
        if device:
            rows, cols, vals = rows.to(device), cols.to(device), vals.to(device)
        return TorchMatrix.from_coo(
            shape=self.shape, values=vals, rows=rows, cols=cols, device=device
        )

    def convert(self, format_str: str) -> "NumpyMatrix":
        return NumpyMatrix(self._data.asformat(format_str))

    def diagonal(self, offset: int = 0) -> np.ndarray:
        return self._data.diagonal(k=offset)
