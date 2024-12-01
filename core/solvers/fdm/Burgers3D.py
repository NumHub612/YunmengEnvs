# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Solving the 3D Burgers equation using finite difference method.
"""
from core.solvers.interfaces import BaseSolver
from core.numerics.mesh import Mesh, MeshGeom, MeshTopo
from core.numerics.fields import NodeField, Vector
from core.numerics.mesh import Grid3D
from configs.settings import logger

import copy
