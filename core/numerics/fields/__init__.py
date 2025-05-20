from core.numerics.fields.variables import *

DTYPE_MAP = {
    VariableType.SCALAR: Scalar,
    VariableType.VECTOR: Vector,
    VariableType.TENSOR: Tensor,
}
from core.numerics.fields.fields import *
