from core.solvers.commons import boundaries
from core.solvers.commons import inits
from core.solvers.commons import callbacks

from core.solvers.commons.boundaries import boundary_conditions
from core.solvers.commons.inits import init_methods
from core.solvers.commons.callbacks import callback_handlers

from core.solvers.fdm import fdm_solvers

# register all the solvers here.
solver_routines = {
    "fdm": fdm_solvers,
}
