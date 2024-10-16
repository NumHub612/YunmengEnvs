from core.solvers.extensions import boundaries
from core.solvers.extensions import inits
from core.solvers.extensions import callbacks

from core.solvers.extensions.boundaries import boundary_conditions
from core.solvers.extensions.inits import init_methods
from core.solvers.extensions.callbacks import callback_handlers

from core.solvers.fdm import fdm_solvers

# register all the solvers here.
solver_routines = {
    "fdm": fdm_solvers,
}
