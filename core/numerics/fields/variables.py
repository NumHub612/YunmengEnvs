# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Variables definition.
"""
from abc import abstractmethod
from typing import Union
import numpy as np

from configs.settings import global_configs


class Variable:
    """
    Abstract variable class.
    """

    # -----------------------------------------------
    # --- abstract methods ---
    # -----------------------------------------------

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

    @abstractmethod
    def from_var(cls, var: "Variable") -> "Variable":
        """
        Set a variable by another variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def rank(self) -> str:
        """
        Get the type of the variable, e.g. scalar, vector, tensor.
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
    def unit(self) -> "Variable":
        """
        Get the unit of the variable.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def zero(self) -> "Variable":
        """
        Get the zero value of the variable.
        """
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        """
        Get the string representation of the variable.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    @abstractmethod
    def __add__(self, other: Union["Variable", float]) -> "Variable":
        """
        Add two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __radd__(self, other: Union["Variable", float]) -> "Variable":
        """
        Add two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, other: Union["Variable", float]) -> "Variable":
        """
        Subtract two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __rsub__(self, other: Union["Variable", float]) -> "Variable":
        """
        Subtract two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other: Union["Variable", float]) -> "Variable":
        """
        Multiply two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __rmul__(self, other: Union["Variable", float]) -> "Variable":
        """
        Multiply two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __truediv__(self, other: Union[float]) -> "Variable":
        """
        Divide two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __neg__(self) -> "Variable":
        """
        Negate a variable.
        """
        raise NotImplementedError()

    @abstractmethod
    def __abs__(self) -> "Variable":
        """
        Get the absolute value of a variable.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # --- reload comparison operations ---
    # -----------------------------------------------

    @abstractmethod
    def __eq__(self, other: Union["Variable", float]) -> bool:
        """
        Compare two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __ne__(self, other: Union["Variable", float]) -> bool:
        """
        Compare two variables.
        """
        raise NotImplementedError()


class Vector(Variable):
    """
    A 3d vector variable with x, y, and z components.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._value = np.array([x, y, z])
        self._tol = global_configs().get("numeric_tolerance", 1e-12)

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_np(cls, np_arr: np.ndarray) -> "Vector":
        if len(np_arr.shape) != 1 or np_arr.shape[0] != 3:
            raise ValueError("Invalid numpy array shape for Vector.")

        return Vector(np_arr[0], np_arr[1], np_arr[2])

    def to_np(self) -> np.ndarray:
        return self._value

    @classmethod
    def from_var(cls, var: "Vector") -> "Vector":
        if not isinstance(var, Vector):
            raise TypeError("Invalid variable type for Vector.")

        return Vector(var.x, var.y, var.z)

    def __str__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def rank(self) -> str:
        return "vector"

    @property
    def magnitude(self) -> float:
        length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return length

    @property
    def unit(self) -> "Vector":
        length = self.magnitude
        if length < self._tol:
            return Vector(1.0, 1.0, 1.0)
        else:
            return self / length

    @property
    def zero(self) -> "Vector":
        return Vector(0.0, 0.0, 0.0)

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

    def __add__(self, other: Union["Variable", float]) -> "Vector":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Vector):
            return Vector.from_np(self.to_np() + other.to_np())
        elif isinstance(other, Scalar):
            return Vector.from_np(self.to_np() + other.value)
        else:
            raise TypeError("Invalid type for Vector add().")

    def __radd__(self, other: Union["Variable", float]) -> "Vector":
        return self.__add__(other)

    def __sub__(self, other: Union["Variable", float]) -> "Vector":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Vector):
            return Vector.from_np(self.to_np() - other.to_np())
        elif isinstance(other, Scalar):
            return Vector.from_np(self.to_np() - other.value)
        else:
            raise TypeError("Invalid type for Vector sub().")

    def __rsub__(self, other: Union["Variable", float]) -> "Vector":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Vector):
            return Vector.from_np(other.to_np() - self.to_np())
        elif isinstance(other, Scalar):
            return Vector.from_np(other.value - self.to_np())
        else:
            raise TypeError("Invalid type for Vector sub().")

    def __mul__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Vector):
            return Scalar.from_np(np.dot(self.to_np(), other.to_np()))
        elif isinstance(other, Scalar):
            return Vector.from_np(self.to_np() * other.value)
        else:
            raise TypeError("Invalid type for Vector mul().")

    def __rmul__(self, other: Union["Variable", float]) -> "Variable":
        return self.__mul__(other)

    def __truediv__(self, other: Union[float]) -> "Vector":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < self._tol:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_np(self.to_np() / other.value)
        else:
            raise TypeError("Invalid type for Vector div().")

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    def __abs__(self) -> "Vector":
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def __eq__(self, other: Union["Vector"]) -> bool:
        if not isinstance(other, Vector):
            return False

        return (
            abs(self.x - other.x) < self._tol
            and abs(self.y - other.y) < self._tol
            and abs(self.z - other.z) < self._tol
        )

    def __ne__(self, other: Union["Variable", float]) -> bool:
        return not self.__eq__(other)


class Scalar(Variable):
    """
    A scalar variable with a single value.
    """

    def __init__(self, value: float = 0.0):
        self._value = value
        self._tol = global_configs().get("numeric_tolerance", 1e-12)

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

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

    def __str__(self):
        return f"Scalar({self.value})"

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def rank(self) -> str:
        return "scalar"

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def unit(self) -> "Scalar":
        return Scalar(1.0)

    @property
    def zero(self) -> "Scalar":
        return Scalar(0.0)

    @property
    def value(self) -> float:
        return self._value

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value + other.value)
        elif isinstance(other, Vector):
            return Vector.from_np(self.value + other.to_np())
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.value + other.to_np())
        else:
            raise TypeError("Invalid type for Scalar add().")

    def __radd__(self, other: Union["Variable", float]) -> "Variable":
        return self.__add__(other)

    def __sub__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value - other.value)
        elif isinstance(other, Vector):
            return Vector.from_np(self.value - other.to_np())
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.value - other.to_np())
        else:
            raise TypeError("Invalid type for Scalar sub().")

    def __rsub__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(other.value - self.value)
        elif isinstance(other, Vector):
            return Vector.from_np(other.to_np() - self.value)
        elif isinstance(other, Tensor):
            return Tensor.from_np(other.to_np() - self.value)
        else:
            raise TypeError("Invalid type for Scalar sub().")

    def __mul__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value * other.value)
        elif isinstance(other, Vector):
            return Vector.from_np(self.value * other.to_np())
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.value * other.to_np())
        else:
            raise TypeError("Invalid type for Scalar mul().")

    def __rmul__(self, other: Union["Variable", float]) -> "Variable":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Variable", float]) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < self._tol:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(self.value / other.value)
        else:
            raise TypeError("Invalid type for Scalar div().")

    def __neg__(self) -> "Scalar":
        return Scalar(-self.value)

    def __abs__(self) -> "Scalar":
        return Scalar(abs(self.value))

    def __eq__(self, other: Union["Scalar", float]) -> bool:
        if isinstance(other, Scalar):
            return abs(self.value - other.value) < self._tol
        elif isinstance(other, (int, float)):
            return abs(self.value - other) < self._tol
        else:
            return False

    def __ne__(self, other: Union["Variable", float]) -> bool:
        return not self.__eq__(other)


class Tensor(Variable):
    """
    A 3x3 tensor variable with xx, xy, xz, yy, yz, and zz components.
    """

    def __init__(
        self,
        xx: float = 0.0,
        xy: float = 0.0,
        xz: float = 0.0,
        yy: float = 0.0,
        yz: float = 0.0,
        zz: float = 0.0,
    ):
        self._value = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

    @classmethod
    def from_np(cls, np_array: np.ndarray) -> "Tensor":
        if len(np_array.shape) != 2 or np_array.shape[0] != 3 or np_array.shape[1] != 3:
            raise ValueError("Invalid numpy array shape for Tensor.")

        return Tensor(
            np_array[0, 0],
            np_array[0, 1],
            np_array[0, 2],
            np_array[1, 1],
            np_array[1, 2],
            np_array[2, 2],
        )

    def to_np(self) -> np.ndarray:
        return self._value

    @classmethod
    def from_var(cls, var: "Tensor") -> "Tensor":
        if not isinstance(var, Tensor):
            raise TypeError("Invalid variable type for Tensor.")

        return Tensor(var.xx, var.xy, var.xz, var.yy, var.yz, var.zz)

    def __str__(self):
        return f"Tensor([{self.xx}, {self.xy}, {self.xz}], [{self.xy}, {self.yy}, {self.yz}], [{self.xz}, {self.yz}, {self.zz}])"

    # -----------------------------------------------
    # --- properties ---
    # -----------------------------------------------

    @property
    def rank(self) -> str:
        return "tensor"

    @property
    def magnitude(self) -> float:
        length = np.sqrt(
            self.xx**2 + self.xy**2 + self.xz**2 + self.yy**2 + self.yz**2 + self.zz**2
        )
        return length

    @property
    def unit(self) -> "Tensor":
        length = self.magnitude
        if length < self._tol:
            return Tensor(1, 0, 0, 0, 1, 0, 0, 0, 1)
        else:
            return self / length

    @property
    def zero(self) -> "Tensor":
        return Tensor(0, 0, 0, 0, 0, 0, 0, 0, 0)

    @property
    def xx(self) -> float:
        return self._value[0, 0]

    @property
    def xy(self) -> float:
        return self._value[0, 1]

    @property
    def xz(self) -> float:
        return self._value[0, 2]

    @property
    def yy(self) -> float:
        return self._value[1, 1]

    @property
    def yz(self) -> float:
        return self._value[1, 2]

    @property
    def zz(self) -> float:
        return self._value[2, 2]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other: Union["Variable", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Tensor.from_np(self.to_np() + other.value)
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.to_np() + other.to_np())
        else:
            raise TypeError("Invalid type for Tensor add().")

    def __radd__(self, other: Union["Variable", float]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union["Variable", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Tensor.from_np(self.to_np() - other.value)
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.to_np() - other.to_np())
        else:
            raise TypeError("Invalid type for Tensor sub().")

    def __rsub__(self, other: Union["Variable", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Tensor.from_np(other.value - self.to_np())
        elif isinstance(other, Tensor):
            return Tensor.from_np(other.to_np() - self.to_np())
        else:
            raise TypeError("Invalid type for Tensor sub().")

    def __mul__(self, other: Union["Variable", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Tensor.from_np(self.to_np() * other.value)
        else:
            raise TypeError("Invalid type for Tensor mul().")

    def __rmul__(self, other: Union["Variable", float]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Variable", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < self._tol:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_np(self.to_np() / other.value)
        else:
            raise TypeError("Invalid type for Tensor div().")

    def __neg__(self) -> "Tensor":
        return Tensor(-self.xx, -self.xy, -self.xz, -self.yy, -self.yz, -self.zz)

    def __abs__(self) -> "Tensor":
        return Tensor(
            abs(self.xx),
            abs(self.xy),
            abs(self.xz),
            abs(self.yy),
            abs(self.yz),
            abs(self.zz),
        )
