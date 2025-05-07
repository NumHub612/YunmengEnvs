# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Variables definition.
"""
from abc import abstractmethod
import numpy as np
import torch
import enum

from configs.settings import settings


class VariableType(enum.Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


class Variable:
    """
    Abstract variable class.
    """

    # -----------------------------------------------
    # --- abstract methods ---
    # -----------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, data: torch.Tensor | np.ndarray) -> "Variable":
        """
        Set the variable by a torch tensor or numpy array.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_np(self) -> np.ndarray:
        """
        Get a numpy array representation of the variable.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_var(cls, var: "Variable") -> "Variable":
        """
        Set a variable by another variable.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def unit(self) -> "Variable":
        """
        Get the unit of the variable.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def zero(self) -> "Variable":
        """
        Get the zero value of the variable.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        Get the shape of the variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def type(self) -> VariableType:
        """
        Get the type of the variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def magnitude(self) -> float:
        """
        Get the magnitude of the variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def data(self) -> torch.Tensor:
        """
        Get the data of the variable.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    @abstractmethod
    def __str__(self):
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
    def __itruediv__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError()

    @abstractmethod
    def __abs__(self):
        raise NotImplementedError()

    # -----------------------------------------------
    # --- reload comparison operations ---
    # -----------------------------------------------

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __ne__(self, other) -> bool:
        raise NotImplementedError()


class Vector(Variable):
    """
    A 3d vector variable with x, y, and z components.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._value = torch.tensor(
            [x, y, z], dtype=settings.DTYPE, device=settings.DEVICE
        )
        self._magtitude = torch.linalg.vector_norm(self._value).item()

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_data(cls, data: torch.Tensor | np.ndarray) -> "Vector":
        if data.ndim != 1 or data.shape[0] != 3:
            raise ValueError(f"Invalid data shape for Vector: {data.shape}.")

        return Vector(*data.tolist())

    def to_np(self) -> np.ndarray:
        return self._value.cpu().numpy()

    @classmethod
    def from_var(cls, var: "Vector") -> "Vector":
        if not isinstance(var, Vector):
            raise TypeError("Invalid variable Vector.")

        return Vector(var.x, var.y, var.z)

    def __str__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    @classmethod
    def unit(self) -> "Vector":
        return Vector(1.0, 1.0, 1.0)

    @classmethod
    def zero(self) -> "Vector":
        return Vector(0.0, 0.0, 0.0)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return (3,)

    @property
    def type(self) -> VariableType:
        return VariableType.VECTOR

    @property
    def magnitude(self) -> float:
        return self._magtitude

    @property
    def data(self) -> torch.Tensor:
        return self._value

    @property
    def x(self) -> float:
        return self._value[0]

    @property
    def y(self) -> float:
        return self._value[1]

    @property
    def z(self) -> float:
        return self._value[2]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector.from_data(self._value + other.data)
        else:
            raise TypeError(f"Invalid Vector add(): {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self._value += other.data
        else:
            raise TypeError(f"Invalid Vector iadd(): {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector.from_data(self._value - other.data)
        else:
            raise TypeError(f"Invalid Vector sub(): {type(other)}.")

    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector.from_data(other.data - self._value)
        else:
            raise TypeError(f"Invalid Vector rsub(): {type(other)}.")

    def __isub__(self, other):
        if isinstance(other, Vector):
            self._value -= other.data
        else:
            raise TypeError(f"Invalid Vector isub(): {type(other)}.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Variable):
            raise TypeError(f"Invalid Vector mul(): {type(other)}.")

        if isinstance(other, Scalar):
            return Vector.from_data(self._value * other.value)
        elif isinstance(other, Vector):
            return Scalar(torch.dot(self._value, other.data).item())
        elif isinstance(other, Tensor):
            result = torch.matmul(self._value, other.data)
            return Vector.from_data(result)
        else:
            raise TypeError(f"Invalid Vector mul(): {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._value *= other
        elif isinstance(other, Scalar):
            self._value *= other.value
        else:
            raise TypeError(f"Invalid Vector imul(): {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_data(self._value / other.value)
        else:
            raise TypeError(f"Invalid Vector div(): {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.magnitude < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_data(other.value / self._value)
        else:
            raise TypeError(f"Invalid Vector rdiv(): {type(other)}.")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other.value
        else:
            raise TypeError(f"Invalid Vector idiv(): {type(other)}.")

    def __neg__(self):
        return Vector.from_data(-self._value)

    def __abs__(self):
        return Vector.from_data(torch.abs(self._value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return False

        return torch.allclose(self._value, other.data, atol=settings.NUMERIC_TOLERANCE)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Scalar(Variable):
    """
    A scalar variable with a single value.
    """

    def __init__(self, value: float = 0.0):
        self._value = torch.tensor(
            [value], dtype=settings.DTYPE, device=settings.DEVICE
        )
        self._magtitude = abs(value)

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_data(cls, data: torch.Tensor | np.ndarray) -> "Scalar":
        if data.ndim != 1 or data.shape[0] != 1:
            raise ValueError("Invalid data shape for Scalar.")

        if isinstance(data, torch.Tensor):
            return Scalar(data.item())
        else:
            return Scalar(data[0])

    def to_np(self) -> np.ndarray:
        return np.array([self._value])

    @classmethod
    def from_var(cls, var: "Scalar") -> "Scalar":
        if not isinstance(var, Scalar):
            raise TypeError("Invalid variable Scalar.")

        return Scalar(var.value)

    @classmethod
    def unit(self) -> "Scalar":
        return Scalar(1.0)

    @classmethod
    def zero(self) -> "Scalar":
        return Scalar(0.0)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return (1,)

    @property
    def type(self) -> VariableType:
        return VariableType.SCALAR

    @property
    def magnitude(self) -> float:
        return self._magtitude

    @property
    def data(self) -> torch.Tensor:
        return self._value

    @property
    def value(self) -> float:
        return self._value.item()

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Scalar(self.value + other)
        elif isinstance(other, Scalar):
            return Scalar.from_data(self._value + other.data)
        else:
            raise TypeError(f"Invalid Scalar add(): {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self._value += other
        elif isinstance(other, Scalar):
            self._value += other.data
        else:
            raise TypeError(f"Invalid Scalar iadd(): {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Scalar(self.value - other)
        elif isinstance(other, Scalar):
            return Scalar.from_data(self._value - other.data)
        else:
            raise TypeError(f"Invalid Scalar sub(): {type(other)}.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Scalar(other - self.value)
        elif isinstance(other, Scalar):
            return Scalar.from_data(other.data - self._value)
        else:
            raise TypeError(f"Invalid Scalar rsub(): {type(other)}.")

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self._value -= other
        elif isinstance(other, Scalar):
            self._value -= other.data
        else:
            raise TypeError(f"Invalid Scalar isub(): {type(other)}.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar.from_data(self._value * other.data)
        elif isinstance(other, Vector):
            return Vector.from_data(self._value * other.data)
        elif isinstance(other, Tensor):
            return Tensor.from_data(self._value * other.data)
        else:
            raise TypeError(f"Invalid Scalar mul(): {type(other)}.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._value *= other
        elif isinstance(other, Scalar):
            self._value *= other.data
        else:
            raise TypeError(f"Invalid Scalar imul(): {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(self.value / other.value)
        else:
            raise TypeError(f"Invalid Scalar div(): {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(other.value / self.value)
        else:
            raise TypeError(f"Invalid Scalar rdiv(): {type(other)}.")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._value /= other
        elif isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other.data
        else:
            raise TypeError(f"Invalid Scalar idiv(): {type(other)}.")

    def __neg__(self):
        return Scalar(-self.value)

    def __abs__(self):
        return Scalar(abs(self.value))

    def __eq__(self, other) -> bool:
        if isinstance(other, Scalar):
            return abs(self.value - other.value) < settings.NUMERIC_TOLERANCE
        elif isinstance(other, (int, float)):
            return abs(self.value - other) < settings.NUMERIC_TOLERANCE
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __str__(self):
        return f"Scalar({self.value})"


class Tensor(Variable):
    """
    A 2x2 or 3x3 tensor variable.
    """

    def __init__(
        self,
        xx: float = 0.0,
        xy: float = 0.0,
        xz: float = 0.0,
        yx: float = 0.0,
        yy: float = 0.0,
        yz: float = 0.0,
        zx: float = 0.0,
        zy: float = 0.0,
        zz: float = 0.0,
    ):
        self._value = torch.tensor(
            [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]],
            dtype=settings.DTYPE,
            device=settings.DEVICE,
        )
        self._magtitude = np.linalg.norm(self.to_np())

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_data(cls, data: torch.Tensor | np.ndarray) -> "Tensor":
        if (
            len(data.shape) != 2
            or data.shape[0] not in [2, 3]
            or data.shape[1] not in [2, 3]
        ):
            raise ValueError("Invalid data shape for Tensor.")

        if data.shape[0] == 2 and data.shape[1] == 2:
            return Tensor(
                data[0, 0],
                data[0, 1],
                0.0,
                data[1, 0],
                data[1, 1],
                0.0,
            )
        else:
            return Tensor(*data.reshape(9).tolist())

    def to_np(self) -> np.ndarray:
        return self._value

    @classmethod
    def from_var(cls, var: "Tensor") -> "Tensor":
        if not isinstance(var, Tensor):
            raise TypeError("Invalid variable Tensor.")

        tensor = Tensor()
        tensor._value = var.data.copy()
        return tensor

    def __str__(self):
        return f"Tensor({self._value})"

    @classmethod
    def unit(self) -> "Tensor":
        return Tensor(1, 0, 0, 0, 1, 0, 0, 0, 1)

    @classmethod
    def zero(self) -> "Tensor":
        return Tensor(0, 0, 0, 0, 0, 0, 0, 0, 0)

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def shape(self) -> tuple:
        return (3, 3)

    @property
    def type(self) -> VariableType:
        return VariableType.TENSOR

    @property
    def magnitude(self) -> float:
        return self._magtitude

    @property
    def data(self) -> torch.Tensor:
        return self._value

    @property
    def xx(self) -> float:
        return self._value[0][0]

    @property
    def xy(self) -> float:
        return self._value[0][1]

    @property
    def xz(self) -> float:
        return self._value[0][2]

    @property
    def yx(self) -> float:
        return self._value[1][0]

    @property
    def yy(self) -> float:
        return self._value[1][1]

    @property
    def yz(self) -> float:
        return self._value[1][2]

    @property
    def zx(self) -> float:
        return self._value[2][0]

    @property
    def zy(self) -> float:
        return self._value[2][1]

    @property
    def zz(self) -> float:
        return self._value[2][2]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor.from_data(self._value + other.data)
        else:
            raise TypeError(f"Invalid Tensor add(): {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Tensor):
            self._value += other.data
        else:
            raise TypeError(f"Invalid Tensor iadd(): {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor.from_data(self._value - other.data)
        else:
            raise TypeError(f"Invalid Tensor sub(): {type(other)}.")

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor.from_data(other.data - self._value)
        else:
            raise TypeError(f"Invalid Tensor rsub(): {type(other)}.")

    def __isub__(self, other):
        if isinstance(other, Tensor):
            self._value -= other.data
        else:
            raise TypeError(f"Invalid Tensor isub(): {type(other)}.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor.from_data(self._value * other)
        elif isinstance(other, Scalar):
            return Tensor.from_data(self._value * other.data)
        elif isinstance(other, Vector):
            return Vector.from_data(self._value @ other.data)
        elif isinstance(other, Tensor):
            return Tensor.from_data(self._value @ other.data)
        else:
            raise TypeError(f"Invalid Tensor mul(): {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._value *= other
        elif isinstance(other, Scalar):
            self._value *= other.data
        elif isinstance(other, Tensor):
            self._value = self._value @ other.data
        else:
            raise TypeError(f"Invalid Tensor imul(): {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_data(self._value / other.data)
        else:
            raise TypeError(f"Invalid Tensor div(): {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.magnitude < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_data(other.data / self._value)
        else:
            raise TypeError(f"Invalid Tensor rdiv(): {type(other)}.")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._value /= other
        elif isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other.data
        else:
            raise TypeError(f"Invalid Tensor idiv(): {type(other)}.")

    def __abs__(self):
        return Tensor.from_data(torch.abs(self._value))

    def __neg__(self):
        return Tensor.from_data(-self._value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            return False

        return torch.allclose(
            self._value,
            other.data,
            atol=settings.NUMERIC_TOLERANCE,
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
