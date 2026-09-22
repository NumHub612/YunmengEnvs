# -*- encoding: utf-8 -*-
"""ComputationalModel — solution-layer wrapper around any ISolver.

One class serves FooFdmSolver / FooFvmSolver / FooHybSolver alike: it
assembles mesh -> backend -> operators -> solver from the model config,
exposes solver fields as output ports, turns coupled input ports into
boundary conditions, and forwards the IEstimable contract when the
underlying solver supports it.
"""

from dataclasses import fields as dc_fields
import numpy as np

from yunmeng.interfaces.capabilities import IEstimable, ISnapshottable
from yunmeng.interfaces.solution import ModelMeta, TimeSpan
from yunmeng.interfaces.supports import Region, IGrid
from yunmeng.interfaces.types import ArrayLike, ElementType, MeshDimension

from yunmeng.numerics.algos import get_class, kind_of, ym_register
from yunmeng.numerics.fields import get_backend
from yunmeng.numerics.grids import ym_meshes
from yunmeng.setting import logger, settings

from yunmeng.solutions.commons.models import BaseInput, BaseModel
from yunmeng.solutions.commons.dataset import (
    MeshCellElementSet,
    Timeseries,
    quantity_of,
)


@ym_register("model")
class ComputationalModel(BaseModel, IEstimable, ISnapshottable):
    """Linkable model wrapping a mesh-based ISolver (fdm/fvm/hyb ...)."""

    @classmethod
    def get_name(cls) -> str:
        return "ComputationalModel"

    def __init__(self, config: dict) -> None:
        self._cfg = config
        model_id = config.get("PROJECT") or config.get("id")
        super().__init__(model_id, ModelMeta(name=model_id, category="computational"))

        temporal = config.get("TEMPORAL") or {}
        self._dt = float(temporal.get("time_step", 1.0))
        self._start = 0.0
        self._end = self._time_length(temporal)
        self._cursor = 0

        self._mesh: IGrid = None
        self._backend = None
        self._solver = None
        self._operators: list = []
        self._series: dict = {}
        self._expose_ports: dict = {}
        self._coupled_ports: dict = {}
        # Ports must exist before linking, and links are wired before
        # Scheduler.initialize() -> assemble eagerly at construction.
        self._assemble()

    def _assemble(self) -> None:
        self._load_datas()
        self._load_mesh()
        self._backend = self._load_backend()
        self._operators = self._load_operators()
        self._load_solver()
        self._create_ports()
        self._publish_exposed()

    def _do_initialize(self) -> None:
        # structure built eagerly; reset the solver to a clean initial state
        self._solver.reset()
        self._cursor = 0
        self._publish_exposed()

    @staticmethod
    def _parse_time(t: "str | dt.datetime"):
        import datetime as dt

        if isinstance(t, dt.datetime):
            return t
        return dt.datetime.fromisoformat(str(t))

    def _time_length(self, temporal: dict) -> float:
        start, end = temporal.get("start_time"), temporal.get("end_time")
        if start is None or end is None:
            return float(temporal.get("horizon", 0.0))
        return (self._parse_time(end) - self._parse_time(start)).total_seconds()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def current_time(self) -> float:
        return self._start + self._cursor * self._dt

    @property
    def solver(self):
        return self._solver

    @property
    def mesh(self) -> IGrid:
        return self._mesh

    # -- assembly --------------------------------------

    def _load_datas(self) -> None:
        datas = self._cfg.get("DATAS") or {}
        for ts in datas.get("timeseries") or []:
            self._series[ts["id"]] = Timeseries(ts["id"], ts["xs"], ts["ys"])

    def _load_mesh(self) -> None:
        spatial = self._cfg.get("SPATIAL") or {}
        mtype = spatial.get("type", "uniform1d")
        cls = ym_meshes.get(str(mtype).lower())
        if cls is None:
            raise ValueError(
                f"{self._id}: unknown mesh type '{mtype}' "
                f"(registered: {sorted(ym_meshes)})."
            )
        self._mesh = cls(**(spatial.get("params") or {}))
        self._register_regions(spatial)

    # -- regions (patches / zones) ----------------------

    def _register_regions(self, spatial: dict) -> None:
        """Restore the boundary-region registration step of the legacy design.

        1D grids get built-in boundary regions (left/right faces, all cells)
        unless explicitly re-declared in the config; then SPATIAL.patches
        (face regions) and SPATIAL.zones (cell regions) are registered.
        Each entry supports `ids: [...]` or `spec: "i:j"` slicing.
        """
        patches = spatial.get("patches") or []
        zones = spatial.get("zones") or []
        declared = {p["id"] for p in patches} | {z["id"] for z in zones}

        if self._mesh.dimension == MeshDimension.D1:
            nx = self._mesh.cell_count
            builtins = [
                ("left", ElementType.FACE, [0]),
                ("right", ElementType.FACE, [nx]),
                ("all", ElementType.CELL, list(range(nx))),
            ]
            for rid, etype, ids in builtins:
                if rid not in declared:
                    self._add_region(rid, etype, ids)

        for patch in patches:
            etype = ElementType[str(patch.get("etype", "face")).upper()]
            self._add_region(patch["id"], etype, self._region_ids(patch))
        for zone in zones:
            etype = ElementType[str(zone.get("etype", "cell")).upper()]
            self._add_region(zone["id"], etype, self._region_ids(zone))

    def _region_ids(self, entry: dict) -> list:
        if entry.get("ids") is not None:
            return [int(i) for i in entry["ids"]]
        if entry.get("spec") is not None:
            parts = str(entry["spec"]).split(":")
            lo = int(parts[0]) if parts[0] else 0
            hi = int(parts[1]) if len(parts) > 1 and parts[1] else None
            return list(range(lo, hi))
        raise ValueError(
            f"{self._id}: region '{entry.get('id')}' needs `ids` or `spec`."
        )

    def _add_region(self, rid: str, etype: ElementType, ids: list) -> None:
        region = Region(name=rid, loc=etype, element_ids=np.asarray(ids, dtype="int64"))
        self._mesh.add_region(region)
        logger.debug(f"{self._id}: region '{rid}' registered ({len(ids)} elements).")

    def _load_backend(self):
        sol = self._cfg.get("SOLVER") or {}
        params = sol.get("params") or {}
        device = params.get("device", settings.get("device", "cpu"))
        return get_backend(params.get("backend", "numpy"), device)

    def _component(self, name: str, kind: str) -> type:
        """Resolve a registered component class with kind checking."""
        cls = get_class(name)  # KeyError lists available names
        actual = kind_of(name)
        if actual != kind:
            raise ValueError(f"{self._id}: '{name}' is a {actual}, not a {kind}.")
        return cls

    def _load_operators(self) -> list:
        ops = []
        for op_cfg in self._cfg.get("OPERATORS") or []:
            cls = self._component(op_cfg["method"], "operator")
            ops.append(cls(**(op_cfg.get("params") or {})))
        return ops

    def _solver_params(self) -> dict:
        sol = self._cfg.get("SOLVER") or {}
        params = dict(sol.get("params") or {})
        params.pop("backend", None)
        params.pop("device", None)
        return params

    def _load_solver(self) -> None:
        sol = self._cfg.get("SOLVER") or {}
        stype = sol.get("type")
        cls = self._component(stype, "solver")

        cfg_cls = cls.get_config_class()
        allowed = {f.name for f in dc_fields(cfg_cls)}
        params = {k: v for k, v in self._solver_params().items() if k in allowed}
        params.setdefault("dt", self._dt)
        params.setdefault("t0", self._start)
        params.setdefault("end_time", self._end)
        config = cfg_cls(**params)

        solver_kwargs = {
            k: v for k, v in self._solver_params().items() if k not in allowed
        }
        self._solver = cls(
            sol.get("id", f"{self._id}.solver"),
            self._mesh,
            self._operators,
            config,
            self._backend,
            **solver_kwargs,
        )

        for ic in sol.get("ics") or []:
            self._solver.add_ic(self._build_ic(ic))
        for bc in sol.get("bcs") or []:
            self._build_bc(bc)
        self._solver.initialize()

    def _build_ic(self, ic: dict):
        cls = self._component(ic["method"], "init")
        params = dict(ic.get("params") or {})
        params.setdefault("centers", self._mesh.cell_centers()[0])
        return cls(ic["id"], ic["field"], **params)

    def _region_of(self, name: str) -> Region:
        try:
            return self._mesh.get_region(name)
        except KeyError:
            raise ValueError(
                f"{self._id}: unknown region '{name}' "
                f"(mesh provides {[r.name for r in self._mesh.regions()]})."
            ) from None

    def _coupled_port_for(self, bc: dict, patch: str) -> BaseInput:
        """Create (or reuse) the coupled input port ``<model>.<bc id>``."""
        port = self._coupled_ports.get(bc["id"])
        if port is not None:
            return port
        region = self._region_of(patch)
        ids = np.asarray(region.element_ids, dtype="int64")
        loc = getattr(region, "loc", None)
        n = self._mesh.cell_count
        if loc == ElementType.FACE:
            # 1D face coordinates: face i sits at origin + i * dx
            xs = self._mesh.origin[0] + ids * self._mesh.spacing[0]
        else:
            xs = self._mesh.cell_centers()[0][np.clip(ids, 0, n - 1)]
        coords = np.column_stack([xs, np.zeros(len(ids)), np.zeros(len(ids))])
        port = self.create_input(
            quantity_of(bc["field"]),
            MeshCellElementSet(coords),
            port_id=f"{self._id}.{bc['id']}",
            time_span=TimeSpan(start=self._start, step=self._dt),
            required=bc.get("required", True),
        )
        self._coupled_ports[bc["id"]] = port
        return port

    def _build_bc(self, bc: dict) -> None:
        cls = self._component(bc["method"], "boundary")
        patches = bc.get("patches") or [bc.get("region", "all")]
        coupled = bc["method"] == "CoupledBC"
        for patch in patches:
            params = dict(bc.get("params") or {})
            value = params.pop("value", None)
            series_id = params.pop("series", None)
            if series_id is not None:
                ts = self._series.get(series_id)
                if ts is None:
                    raise ValueError(f"{self._id}: unknown timeseries '{series_id}'.")
                value = (ts.xs, ts.ys)
            if value is not None or series_id is not None:
                key = "value" if bc["method"] == "DirichletBC" else "flux"
                params[key] = value
            if coupled:
                params["port"] = self._coupled_port_for(bc, patch)
            inst = cls(bc["id"], bc["field"], self._region_of(patch), **params)
            self._solver.add_bc(inst)

    # -- ports ----------------------------------------

    def _create_ports(self) -> None:
        sol = self._cfg.get("SOLVER") or {}
        ts = TimeSpan(start=self._start, step=self._dt)
        centers = self._mesh.cell_centers()[0]
        n = self._mesh.cell_count
        coords = np.column_stack([centers, np.zeros(n), np.zeros(n)])

        for exp in sol.get("expose") or []:
            field = exp["field"]
            elements = MeshCellElementSet(coords)
            port = self.create_output(
                quantity_of(field),
                elements,
                port_id=f"{self._id}.{field}",
                time_span=ts,
            )
            self._expose_ports[field] = port

    def _publish_exposed(self) -> None:
        for field, port in self._expose_ports.items():
            values = self._backend.to_host(self._solver.get_solution(field).values)
            port.set_values(values)

    # -- stepping --------------------------------------

    def validate(self) -> list:
        errs = []
        if self._dt <= 0:
            errs.append(f"{self._id}: time_step must be > 0.")
        if self._end <= self._start:
            errs.append(f"{self._id}: end_time must be after start_time.")
        for port in self._coupled_ports.values():
            if port.required and not port.is_connected:
                errs.append(f"{self._id}: required input '{port.id}' is not connected.")
        return errs

    def _do_update(self, inquirers: list | None = None) -> None:
        status = self._solver.step(self._dt)
        self._cursor += 1
        self._publish_exposed()
        if status.finished or self.current_time >= self._end - 1e-12:
            self.mark_done()

    def _do_finish(self) -> None:
        self._cursor = 0

    # -- ISnapshottable ---------------------------------

    def snapshot(self) -> dict:
        return {
            "cursor": self._cursor,
            "solver": self._solver.snapshot(),
            "ports": {p.id: p._state() for p in self._outputs},
        }

    def restore(self, snapshot: dict) -> None:
        self._cursor = int(snapshot["cursor"])
        self._solver.restore(snapshot["solver"])
        for p in self._outputs:
            state = snapshot["ports"].get(p.id)
            if state is not None:
                p._set_state(state)

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError(
            "model load requires config and registry; rebuild from the "
            "orchestrator config, then restore() the snapshot payload"
        )

    # -- IEstimable passthrough --------------------------

    def _estimable(self):
        for meth in (
            "parameter_metas",
            "get_parameters",
            "set_parameters",
            "run",
            "supports_gradients",
        ):
            if not callable(getattr(self._solver, meth, None)):
                raise TypeError(
                    f"{self._id}: solver '{type(self._solver).__name__}' "
                    f"is not estimable."
                )
        return self._solver

    def parameter_metas(self) -> list:
        return self._estimable().parameter_metas()

    def get_parameters(self, names: list | None = None):
        return self._estimable().get_parameters(names)

    def set_parameters(self, values: ArrayLike, names: list | None = None) -> None:
        self._estimable().set_parameters(values, names)

    # Legacy aliases kept for EstimatorBuilder call sites.
    def param_names(self) -> list:
        return self._estimable().parameter_names()

    def reset_run(self) -> None:
        self._solver.reset()
        self._cursor = 0
        self._publish_exposed()

    def run(self, n_steps: int, **kwargs):
        return self._estimable().run(n_steps, **kwargs)

    def supports_gradients(self) -> bool:
        try:
            return self._estimable().supports_gradients()
        except TypeError:
            return False

    def train(self) -> None:
        self._solver.train()

    def eval(self) -> None:
        self._solver.eval()

    # -- diagnostics --------------------------------

    def describe(self) -> str:
        lines = [
            f"ComputationalModel '{self._id}' "
            f"(solver={type(self._solver).__name__}, dt={self._dt}s)",
            f"  inputs:  {[p.id for p in self._inputs]}",
            f"  outputs: {[p.id for p in self._outputs]}",
        ]
        return "\n".join(lines)
