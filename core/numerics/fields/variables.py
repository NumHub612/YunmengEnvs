# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Variables definition.
"""
from abc import abstractmethod
import numpy as np

from configs.settings import NUMERIC_TOLERANCE


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
    def type(self) -> str:
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
    def data(self) -> np.ndarray:
        """
        Get the data of the variable.
        """
        raise NotImplementedError()

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    @abstractmethod
    def __str__(self):
        """
        Get the string representation of the variable.
        """
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other) -> "Variable":
        """
        Add two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __radd__(self, other) -> "Variable":
        """
        Add two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, other) -> "Variable":
        """
        Subtract two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __rsub__(self, other) -> "Variable":
        """
        Subtract two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other) -> "Variable":
        """
        Multiply two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __rmul__(self, other) -> "Variable":
        """
        Multiply two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __truediv__(self, other) -> "Variable":
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
    def __eq__(self, other) -> bool:
        """
        Compare two variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def __ne__(self, other) -> bool:
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
    def type(self) -> str:
        return "vector"

    @property
    def magnitude(self) -> float:
        length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return length

    @property
    def data(self) -> np.ndarray:
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

    def __add__(self, other) -> "Vector":
        if isinstance(other, Vector):
            return Vector.from_np(self.to_np() + other.to_np())
        else:
            raise TypeError(f"Invalid type for Vector add(): {type(other)}.")

    def __sub__(self, other) -> "Vector":
        if isinstance(other, Vector):
            return Vector.from_np(self.to_np() - other.to_np())
        else:
            raise TypeError(f"Invalid type for Vector sub(): {type(other)}.")

    def __mul__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Vector):
            return Scalar.from_np(np.dot(self.to_np(), other.to_np()))
        elif isinstance(other, Scalar):
            return Vector.from_np(self.to_np() * other.value)
        elif isinstance(other, Tensor):
            lf = self.to_np()
            rhs = other.to_np()
            result = [np.dot(lf, rhs[:, i]) for i in range(3)]
            return Vector.from_np(np.array(result))
        else:
            raise TypeError(f"Invalid type for Vector mul() with {type(other)}.")

    def __rmul__(self, other) -> "Variable":
        if isinstance(other, (int, float, Scalar)):
            return self.__mul__(other)

    def __truediv__(self, other) -> "Vector":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Vector.from_np(self.to_np() / other.value)
        else:
            raise TypeError(f"Invalid type for Vector div(): {type(other)}.")

    def __neg__(self) -> "Vector":
        return Vector.from_np(-self._value)

    def __abs__(self) -> "Vector":
        return Vector.from_np(np.abs(self._value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return False

        is_equal = np.allclose(self.to_np(), other.to_np(), atol=NUMERIC_TOLERANCE)
        return is_equal

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Scalar(Variable):
    """
    A scalar variable with a single value.
    """

    def __init__(self, value: float = 0.0):
        self._value = value

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
    def type(self) -> str:
        return "scalar"

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def data(self) -> np.ndarray:
        return np.array([self._value])

    @property
    def value(self) -> float:
        return self._value

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value + other.value)
        else:
            raise TypeError("Invalid type for Scalar add().")

    def __radd__(self, other) -> "Variable":
        return self.__add__(other)

    def __sub__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value - other.value)
        else:
            raise TypeError("Invalid type for Scalar sub().")

    def __rsub__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(other.value - self.value)
        else:
            raise TypeError("Invalid type for Scalar rsub().")

    def __mul__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Scalar(self.value * other.value)
        elif isinstance(other, Vector):
            return Vector.from_np(self.value * other.to_np())
        elif isinstance(other, Tensor):
            return Tensor.from_np(self.value * other.to_np())
        else:
            raise TypeError(f"Invalid type for Scalar mul() with {type(other)}.")

    def __rmul__(self, other) -> "Variable":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(self.value / other.value)
        else:
            raise TypeError("Invalid type for Scalar div().")

    def __rtruediv__(self, other) -> "Variable":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if self.value < NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Scalar(other.value / self.value)
        else:
            raise TypeError("Invalid type for Scalar rdiv().")

    def __neg__(self) -> "Scalar":
        return Scalar(-self.value)

    def __abs__(self) -> "Scalar":
        return Scalar(abs(self.value))

    def __eq__(self, other) -> bool:
        if isinstance(other, Scalar):
            return abs(self.value - other.value) < NUMERIC_TOLERANCE
        elif isinstance(other, (int, float)):
            return abs(self.value - other) < NUMERIC_TOLERANCE
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


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
        self._value = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

    # -----------------------------------------------
    # --- override methods ---
    # -----------------------------------------------

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
    def type(self) -> str:
        return "tensor"

    @property
    def magnitude(self) -> float:
        length = np.linalg.norm(self.to_np())
        return length

    @property
    def data(self) -> np.ndarray:
        return self._value

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
    def yx(self) -> float:
        return self._value[1, 0]

    @property
    def yy(self) -> float:
        return self._value[1, 1]

    @property
    def yz(self) -> float:
        return self._value[1, 2]

    @property
    def zx(self) -> float:
        return self._value[2, 0]

    @property
    def zy(self) -> float:
        return self._value[2, 1]

    @property
    def zz(self) -> float:
        return self._value[2, 2]

    # -----------------------------------------------
    # --- reload arithmetic operations ---
    # -----------------------------------------------

    def __add__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return Tensor.from_np(self.to_np() + other.to_np())
        else:
            raise TypeError("Invalid type for Tensor add().")

    def __sub__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return Tensor.from_np(self.to_np() - other.to_np())
        else:
            raise TypeError("Invalid type for Tensor sub().")

    def __mul__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            return Tensor.from_np(self.to_np() * other.value)
        elif isinstance(other, Vector):
            rh = other.to_np()
            result = [np.dot(self._value[i], rh) for i in range(3)]
            return Vector.from_np(np.array(result))
        elif isinstance(other, Tensor):
            results = []
            for i in range(3):
                row = []
                for j in range(3):
                    row.append(np.dot(self._value[i], other._value[:, j]))
                results.append(row)
            return Tensor.from_np(np.array(results))
        else:
            raise TypeError("Invalid type for Tensor mul().")

    def __rmul__(self, other):
        if isinstance(other, (int, float, Scalar)):
            return self.__rmul__(other)

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Scalar(other)

        if isinstance(other, Scalar):
            if other.value < NUMERIC_TOLERANCE:
                raise ZeroDivisionError("Division by zero.")
            return Tensor.from_np(self.to_np() / other.value)
        else:
            raise TypeError("Invalid type for Tensor div().")

    def __neg__(self) -> "Tensor":
        return Tensor.from_np(-self.to_np())

    def __abs__(self) -> "Tensor":
        return Tensor.from_np(np.abs(self._value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            return False

        is_equal = np.allclose(self.to_np(), other.to_np(), atol=NUMERIC_TOLERANCE)
        return is_equal

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
