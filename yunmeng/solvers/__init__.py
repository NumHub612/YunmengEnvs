from yunmeng.solvers.interfaces import SolverType
from yunmeng.solvers.commons import boundaries
from yunmeng.solvers.commons import inits
from yunmeng.solvers.commons import callbacks
from yunmeng.solvers.commons.enums import *

from yunmeng.solvers.commons.boundaries import boundary_conditions
from yunmeng.solvers.commons.inits import init_methods
from yunmeng.solvers.commons.callbacks import callback_handlers

from yunmeng.solvers.fdm import fdm_solvers, fdm_operators
from yunmeng.solvers.fvm import fvm_solvers, fvm_operators

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
