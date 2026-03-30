from core.solvers.interfaces import SolverType
from core.solvers.commons import boundaries
from core.solvers.commons import inits
from core.solvers.commons import callbacks
from core.solvers.commons.enums import *

from core.solvers.commons.boundaries import boundary_conditions
from core.solvers.commons.inits import init_methods
from core.solvers.commons.callbacks import callback_handlers

from core.solvers.fdm import fdm_solvers, fdm_operators
from core.solvers.fvm import fvm_solvers, fvm_operators

ym_solvers = {
    SolverType.FDM: fdm_solvers,
    SolverType.FVM: fvm_solvers,
    SolverType.FEM: None,
    SolverType.LBM: None,
    SolverType.AIM: None,
}

ym_operators = {
    SolverType.FDM: fdm_operators,
    SolverType.FVM: fvm_operators,
    SolverType.FEM: None,
    SolverType.LBM: None,
    SolverType.AIM: None,
}
