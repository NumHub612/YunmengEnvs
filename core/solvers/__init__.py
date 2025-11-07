from core.solvers.commons import boundaries
from core.solvers.commons import inits
from core.solvers.commons import callbacks

from core.solvers.commons.boundaries import boundary_conditions
from core.solvers.commons.inits import init_methods
from core.solvers.commons.callbacks import callback_handlers

from core.solvers.fdm import fdm_solvers, fdm_operators
from core.solvers.fvm import fvm_solvers, fvm_operators

ym_solvers = {
    "fdm": {"solvers": fdm_solvers, "operators": fdm_operators},
    "fvm": {"solvers": fvm_solvers, "operators": fvm_operators},
}
