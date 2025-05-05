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
    def from_torch(cls, torch_tensor: torch.Tensor) -> "Variable":
        """
        Set a variable by a torch tensor.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_np(cls, np_arr: np.ndarray) -> "Variable":
        """
        Set a variable by a numpy array.
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
            [x, y, z], dtype=torch.float64, device=settings.DEVICE
        )
        self._magtitude = torch.linalg.vector_norm(self._value).item()

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_torch(cls, torch_tensor: torch.Tensor) -> "Vector":
        if len(torch_tensor.shape) != 1 or torch_tensor.shape[0] != 3:
            raise ValueError("Invalid torch tensor shape for Vector.")

        return Vector(*torch_tensor.tolist())

    @classmethod
    def from_np(cls, np_arr: np.ndarray) -> "Vector":
        if len(np_arr.shape) != 1 or np_arr.shape[0] != 3:
            raise ValueError("Invalid numpy array shape for Vector.")

        return Vector(*np_arr.tolist())

    def to_np(self) -> np.ndarray:
        return self._value.cpu().numpy()

    @classmethod
    def from_var(cls, var: "Vector") -> "Vector":
        if not isinstance(var, Vector):
            raise TypeError("Invalid variable type for Vector.")

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
            return Vector.from_torch(self._value + other._value)
        else:
            raise TypeError(f"Invalid type for Vector add(): {type(other)}.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self._value += other._value
        else:
            raise TypeError(f"Invalid type for Vector iadd(): {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector.from_torch(self._value - other._value)
        else:
            raise TypeError(f"Invalid type for Vector sub(): {type(other)}.")

    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector.from_torch(other._value - self._value)
        else:
            raise TypeError(f"Invalid type for Vector rsub(): {type(other)}.")

    def __isub__(self, other):
        if isinstance(other, Vector):
            self._value -= other._value
        else:
            raise TypeError(f"Invalid type for Vector isub(): {type(other)}.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Variable):
            raise TypeError(f"Invalid type for Vector mul() with {type(other)}.")

        if isinstance(other, Scalar):
            return Vector.from_torch(self._value * other.value)
        elif isinstance(other, Vector):
            return Scalar(torch.dot(self._value, other._value).item())
        elif isinstance(other, Tensor):
            rhs = other._value.reshape(3, 3)
            result = torch.matmul(self._value, rhs)
            return Vector.from_torch(result)
        else:
            raise TypeError(f"Invalid type for Vector mul() with {type(other)}.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)
        if not isinstance(other, Variable):
            raise TypeError(f"Invalid type for Vector imul() with {type(other)}.")

        if isinstance(other, Scalar):
            self._value *= other.value
        else:
            raise TypeError(f"Invalid type for Vector imul() with {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_torch(self._value / other.value)
        else:
            raise TypeError(f"Invalid type for Vector div(): {type(other)}.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.magnitude < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_torch(other.value / self._value)
        else:
            raise TypeError(f"Invalid type for Vector rdiv(): {type(other)}.")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other.value
        else:
            raise TypeError(f"Invalid type for Vector idiv(): {type(other)}.")

    def __neg__(self):
        return Vector.from_torch(-self._value)

    def __abs__(self):
        return Vector.from_torch(torch.abs(self._value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return False

        return torch.allclose(
            self._value, other._value, atol=settings.NUMERIC_TOLERANCE
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Scalar(Variable):
    """
    A scalar variable with a single value.
    """

    def __init__(self, value: float = 0.0):
        self._value = torch.tensor([value], dtype=torch.float64, device=settings.DEVICE)
        self._magtitude = abs(value)

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_torch(cls, torch_tensor: torch.Tensor) -> "Scalar":
        if len(torch_tensor.shape) != 1 or torch_tensor.shape[0] != 1:
            raise ValueError("Invalid torch tensor shape for Scalar.")

        return Scalar(torch_tensor.item())

    @classmethod
    def from_np(cls, np_arr: np.ndarray) -> "Scalar":
        if len(np_arr.shape) != 1 or np_arr.shape[0] != 1:
            raise ValueError("Invalid numpy array shape for Scalar.")

        return Scalar(np_arr[0])

    def to_np(self) -> np.ndarray:
        return np.array([self._value])

    @classmethod
    def from_var(cls, var: "Scalar") -> "Scalar":
        if not isinstance(var, Scalar):
            raise TypeError("Invalid variable type for Scalar.")

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
            return Scalar.from_torch(self._value + other._value)
        else:
            raise TypeError("Invalid type for Scalar add().")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self._value += other
        elif isinstance(other, Scalar):
            self._value += other._value
        else:
            raise TypeError("Invalid type for Scalar iadd().")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Scalar(self.value - other)
        elif isinstance(other, Scalar):
            return Scalar.from_torch(self._value - other._value)
        else:
            raise TypeError("Invalid type for Scalar sub().")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Scalar(other - self.value)
        elif isinstance(other, Scalar):
            return Scalar.from_torch(other._value - self._value)
        else:
            raise TypeError("Invalid type for Scalar rsub().")

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self._value -= other
        elif isinstance(other, Scalar):
            self._value -= other._value
        else:
            raise TypeError("Invalid type for Scalar isub().")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar.from_torch(self._value * other._value)
        elif isinstance(other, Vector):
            return Vector.from_torch(self._value * other._value)
        elif isinstance(other, Tensor):
            return Tensor.from_torch(self._value * other._value)
        else:
            raise TypeError(f"Invalid type for Scalar mul() with {type(other)}.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._value *= other
        elif isinstance(other, Scalar):
            self._value *= other._value
        else:
            raise TypeError(f"Invalid type for Scalar imul() with {type(other)}.")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(self.value / other.value)
        else:
            raise TypeError("Invalid type for Scalar div().")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(other.value / self.value)
        else:
            raise TypeError("Invalid type for Scalar rdiv().")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._value /= other
        elif isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other._value
        else:
            raise TypeError("Invalid type for Scalar idiv().")

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
            dtype=torch.float64,
            device=settings.DEVICE,
        )
        self._magtitude = np.linalg.norm(self.to_np())

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_torch(cls, torch_tensor: torch.Tensor) -> "Tensor":
        if (
            len(torch_tensor.shape) != 2
            or torch_tensor.shape[0] not in [2, 3]
            or torch_tensor.shape[1] not in [2, 3]
        ):
            raise ValueError("Invalid torch tensor shape for Tensor.")

        if torch_tensor.shape[0] == 2 and torch_tensor.shape[1] == 2:
            return Tensor(
                torch_tensor[0, 0],
                torch_tensor[0, 1],
                0.0,
                torch_tensor[1, 0],
                torch_tensor[1, 1],
                0.0,
            )
        else:
            return Tensor(*torch_tensor.reshape(9).tolist())

    @classmethod
    def from_np(cls, np_array: np.ndarray) -> "Tensor":
        if np_array.ndim == 1 and np_array.shape[0] == 9:
            return Tensor(*np_array)
        elif np_array.ndim == 2:
            if np_array.shape[0] == 3 and np_array.shape[1] == 3:
                return Tensor(*np_array.reshape(9))
            if np_array.shape[0] == 2 and np_array.shape[1] == 2:
                return Tensor(
                    np_array[0, 0],
                    np_array[0, 1],
                    0.0,
                    np_array[1, 0],
                    np_array[1, 1],
                    0.0,
                )
        else:
            raise ValueError("Invalid numpy array shape for Tensor.")

    def to_np(self) -> np.ndarray:
        return self._value

    @classmethod
    def from_var(cls, var: "Tensor") -> "Tensor":
        if not isinstance(var, Tensor):
            raise TypeError("Invalid variable type for Tensor.")

        tensor = Tensor()
        tensor._value = var._value.copy()
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
            return Tensor.from_torch(self._value + other._value)
        else:
            raise TypeError("Invalid type for Tensor add().")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Tensor):
            self._value += other._value
        else:
            raise TypeError("Invalid type for Tensor iadd().")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor.from_torch(self._value - other._value)
        else:
            raise TypeError("Invalid type for Tensor sub().")

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor.from_torch(other._value - self._value)
        else:
            raise TypeError("Invalid type for Tensor rsub().")

    def __isub__(self, other):
        if isinstance(other, Tensor):
            self._value -= other._value
        else:
            raise TypeError("Invalid type for Tensor isub().")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor.from_torch(self._value * other)
        elif isinstance(other, Scalar):
            return Tensor.from_torch(self._value * other._value)
        elif isinstance(other, Vector):
            return Vector.from_torch(self._value @ other._value)
        elif isinstance(other, Tensor):
            return Tensor.from_torch(self._value @ other._value)
        else:
            raise TypeError("Invalid type for Tensor mul().")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._value *= other
        elif isinstance(other, Scalar):
            self._value *= other._value
        elif isinstance(other, Tensor):
            self._value = self._value @ other._value
        else:
            raise TypeError("Invalid type for Tensor imul().")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_torch(self._value / other._value)
        else:
            raise TypeError("Invalid type for Tensor div().")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.magnitude < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_torch(other._value / self._value)
        else:
            raise TypeError("Invalid type for Tensor rdiv().")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._value /= other
        elif isinstance(other, Scalar):
            if other.value < settings.NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            self._value /= other._value
        else:
            raise TypeError("Invalid type for Tensor idiv().")

    def __abs__(self):
        return Tensor.from_torch(torch.abs(self._value))

    def __neg__(self):
        return Tensor.from_torch(-self._value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            return False

        return torch.allclose(
            self._value, other._value, atol=settings.NUMERIC_TOLERANCE
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
