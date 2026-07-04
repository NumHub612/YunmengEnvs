# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Matrix interface.
"""

from yunmeng.numerics.fields import ArrayLike

import torch
import numpy as np
from abc import abstractmethod
from typing import Union, Tuple, List


class Matrix:
    """Abstract matrix class optimized for CFD applications."""

    # -----------------------------------------------
    # region Constructions
    # -----------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(
        cls,
        values: ArrayLike,
        indices: ArrayLike = None,
        shape: Tuple[int, int] = None,
        device: torch.device = None,
    ) -> "Matrix":
        """Create from dense data."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_coo(
        cls,
        shape: Tuple[int, int],
        values: ArrayLike,
        rows: ArrayLike,
        cols: ArrayLike,
        device: torch.device = None,
    ) -> "Matrix":
        """Create from Coo-format."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_csr(
        cls,
        shape: Tuple[int, int],
        values: ArrayLike,
        ptrs: ArrayLike,
        idxs: ArrayLike,
        device: torch.device = None,
    ) -> "Matrix":
        """Create from CSR-format."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def identity(
        cls,
        size: int,
        device: torch.device = None,
    ) -> "Matrix":
        """Create identity matrix."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def zeros(
        cls,
        shape: Tuple[int, int],
        device: torch.device = None,
    ) -> "Matrix":
        """Create an empty matrix."""
        raise NotImplementedError()

    # -----------------------------------------------
    # region Properties
    # -----------------------------------------------

    @property
    @abstractmethod
    def data(self):
        """The raw data of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Matrix shape."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def nnz(self) -> int:
        """Number of non-zero elements."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def T(self) -> "Matrix":
        """Transpose of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def diags(self) -> List[ArrayLike]:
        """Diagonals of the matrix."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the matrix resides."""
        raise NotImplementedError()

    # -----------------------------------------------
    # region utils
    # -----------------------------------------------

    def to(self, device: torch.device) -> "Matrix":
        """Move matrix to a specific device."""
        raise NotImplementedError()

    def to_dense(self) -> ArrayLike:
        """Convert to dense matrix."""
        raise NotImplementedError()

    def to_numpy(self) -> "Matrix":
        """Convert to numpy matrix."""
        raise NotImplementedError()

    def to_torch(self, device: torch.device = None) -> "Matrix":
        """Convert to torch matrix."""
        raise NotImplementedError()

    def convert(self, format: str) -> "Matrix":
        """Convert between sparse formats."""
        raise NotImplementedError()

    def diagonal(self, offset: int = 0) -> ArrayLike:
        """Extract a specific diagonal.
        Offset > 0 for upper, < 0 for lower.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # region Operators
    # -----------------------------------------------

    @abstractmethod
    def __matmul__(
        self, other: Union["Matrix", ArrayLike]
    ) -> Union["Matrix", ArrayLike]:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other: "Matrix"):
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, other: "Matrix"):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, scalar: float):
        raise NotImplementedError()

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError()
        scalar = 1.0 / scalar
        return self.__mul__(scalar)
