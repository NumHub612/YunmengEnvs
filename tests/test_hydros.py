# -*- encoding: utf-8 -*-
"""
YunmengEnvs solution-layer refactor test suite.
"""

import numpy as np
import pytest

from yunmeng.solutions.commons.datasets import Timeseries, Quantities
from yunmeng.solutions.HydrologicalSims.algorithms import (
    available,
    create,
    MuskingumRouting,
    ThreeSourceLinearReservoir,
    XinanjiangRunoff,
    TargetLevelRelease,
)
from yunmeng.solutions.HydrologicalSims.HydroNodes import (
    ReservoirNode,
    SubBasinNode,
    LinearStorageCurve,
    TableStorageCurve,
)
from yunmeng.solutions.HydrologicalSims.HydroModel import HydrologyModel
from yunmeng.solutions.HydrologicalSims.Reservoir import ReservoirModel
from yunmeng.solutions.HydrologicalSims.Gauges import GaugeSet
from yunmeng.solutions.standards import ModelStatus
from yunmeng.taskflow.scheduler import Scheduler

DT = 3600.0
STEPS = 24


# ---------------------------------------------------------------
# region fixtures
# ---------------------------------------------------------------


def const_series(gid, value, n=STEPS, dt=DT):
    return Timeseries(gid, np.arange(n) * dt, np.full(n, float(value)))


def xaj_cfg(**over):
    cfg = {"algo": "xaj", "params": {}}
    cfg["params"].update(over)
    return cfg


def subbasin_cfg(name="sub1", area=100.0, forcing=None):
    return {
        "name": name,
        "type": "subbasin",
        "area": area,
        "forcing": forcing or {},
        "runoff": xaj_cfg(),
        "surface": {"algo": "linear3", "params": {}},
    }


def reservoir_cfg(name="rsv1", target=5.0e7):
    return {
        "name": name,
        "type": "reservoir",
        "storage": {
            "curve": {
                "method": "linear",
                "params": {"z0": 100.0, "s0": 5.0e7, "area_km2": 12.0},
            },
            "s_init": 5.0e7,
        },
        "reservoir": {
            "algo": "target_level",
            "params": {"target_storage": target, "q_max": 1e6},
        },
    }


def basin_model():
    return HydrologyModel(
        {
            "id": "basin",
            "dt": DT,
            "steps": STEPS,
            "gauges": {
                "g1": const_series("g1", 10.0),
                "g2": const_series("g2", 20.0),
            },
            "nodes": [
                subbasin_cfg(forcing={"P": {"g1": 0.5, "g2": 0.5}, "E": "g1"}),
                reservoir_cfg(),
            ],
            "edges": [["sub1", "rsv1"]],
            "expose": ["sub1.Q", "rsv1.Q", "rsv1.Z", "rsv1.S"],
        }
    )


# ---------------------------------------------------------------
# region registry
# ---------------------------------------------------------------


class TestRegistry:
    def test_all_kinds_registered(self):
        reg = available()
        assert reg["runoff"] == ["xaj"]
        assert reg["surface"] == ["linear3"]
        assert reg["river"] == ["muskingum"]
        assert reg["reservoir"] == ["target_level"]

    def test_create_release_kind_works(self):
        # regression: registry kind used to be "reservoir" while callers
        # asked for "release" -> guaranteed KeyError
        algo = create("reservoir", "target_level", target_storage=1.0)
        assert isinstance(algo, TargetLevelRelease)

    def test_unknown_param_rejected(self):
        with pytest.raises(ValueError, match="unknown parameters"):
            create("river", "muskingum", NOT_A_PARAM=1.0)

    def test_bounds_enforced(self):
        with pytest.raises(ValueError, match="outside bounds"):
            create("river", "muskingum", X=0.9)

    def test_unknown_algorithm_message(self):
        with pytest.raises(KeyError, match="available"):
            create("river", "nope")


# ---------------------------------------------------------------
# region algorithms
# ---------------------------------------------------------------


class TestMuskingum:
    def test_steady_state_passthrough(self):
        m = MuskingumRouting(K=1.5 * 3600.0, X=0.2)
        for _ in range(200):
            q = m.route(100.0, DT)
        assert q == pytest.approx(100.0, rel=1e-6)

    def test_negative_outflow_audited(self):
        # strong wave + high K relative to dt can undershoot below zero;
        # clipped volume must be tracked, not silently lost
        m = MuskingumRouting(K=24 * 3600.0, X=0.0)
        m.route(10000.0, DT)
        m.route(0.0, DT)
        assert m.negative_volume >= 0.0
        assert "NEG_VOLUME" in m.state()

    def test_snapshot_restore(self):
        m = MuskingumRouting(K=1.5 * 3600.0, X=0.2)
        m.route(50.0, DT)
        snap = m.snapshot()
        q1 = m.route(80.0, DT)
        m.restore(snap)
        q2 = m.route(80.0, DT)
        assert q1 == q2


class TestLinearReservoir:
    def test_mm_km2_conversion(self):
        # 1 mm over 1 km2 in 1 s = 1000 m3/s inflow; with KS=dt the first
        # step output is (1 - e^-1) * I
        lr = ThreeSourceLinearReservoir(KS=DT, KI=DT, KG=86400.0)
        q = lr.route(1.0, 0.0, 0.0, area_km2=1.0, dt=DT)
        assert q == pytest.approx((1 - np.exp(-1)) * 1000.0 / DT, rel=1e-9)

    def test_long_run_mass_approaches_input(self):
        lr = ThreeSourceLinearReservoir(KS=DT, KI=2 * DT, KG=86400.0)
        qs = [lr.route(1.0, 1.0, 1.0, 100.0, DT) for _ in range(3000)]
        assert qs[-1] == pytest.approx(3.0 * 100.0 * 1000.0 / DT, rel=1e-3)


class TestXinanjiang:
    def test_tension_water_bounded(self):
        x = XinanjiangRunoff()
        rng = np.random.default_rng(42)
        for _ in range(365):
            p = float(rng.gamma(2.0, 5.0))
            e = float(rng.uniform(0, 8))
            x.produce(p, e, DT)
            assert 0.0 <= x.tension_water <= x.p("WM") + 1e-9
            assert x.free_water >= 0.0

    def test_param_consistency_checks(self):
        with pytest.raises(ValueError, match="WUM"):
            XinanjiangRunoff(WM=50.0, WUM=30.0, WLM=30.0)
        with pytest.raises(ValueError, match="KI"):
            XinanjiangRunoff(KI=0.6, KG=0.6)

    def test_no_rain_recession_only(self):
        x = XinanjiangRunoff(S0=20.0)
        rs, ri, rg = x.produce(0.0, 0.0, DT)
        assert rs == 0.0
        assert ri > 0.0 and rg > 0.0
        assert x.free_water == pytest.approx(20.0 - ri - rg)


class TestTargetLevelRelease:
    def test_returns_to_target(self):
        r = TargetLevelRelease(target_storage=5.0e7, q_max=1e12)
        # inflow raises storage above target -> release the excess
        q = r.release(5.0e7, 1000.0, 0.0, DT)
        assert q == pytest.approx(1000.0)

    def test_dead_storage_blocks_release(self):
        r = TargetLevelRelease(target_storage=0.0, dead_storage=100.0, q_max=1e12)
        assert r.release(50.0, 0.0, 0.0, DT) == 0.0

    def test_plan_does_not_mutate_inputs(self):
        # regression: plan() used to pop(0) the caller's list
        r = TargetLevelRelease(target_storage=5.0e7, q_max=1e12)
        inflows = [100.0] * 10
        stamps = list(np.arange(10) * DT)
        out = r.plan(5.0e7, inflows, stamps, DT, context={})
        assert len(inflows) == 10 and len(stamps) == 10
        assert len(out) == 10
        assert out[0] == pytest.approx(100.0)

    def test_negative_inflow_honoured(self):
        # pumping: release target falls because inflow is negative
        r = TargetLevelRelease(target_storage=5.0e7, q_max=1e12)
        q = r.release(5.0e7, -500.0, 0.0, DT)
        assert q == 0.0  # storage below target -> nothing to release


# ---------------------------------------------------------------
# region nodes
# ---------------------------------------------------------------


class TestReservoirNode:
    def test_water_balance(self):
        node = ReservoirNode("r", reservoir_cfg("r"))
        node.add_internal_inflow("up")
        s0 = node.storage
        node.set_inflow("in.up", 1000.0)
        node.step(DT, 0.0, {})
        expected = max(s0 + (1000.0 - node.current("Q")) * DT, 0.0)
        assert node.storage == pytest.approx(expected)
        assert node.current("Z") == pytest.approx(
            node.storage_curve.level(node.storage)
        )

    def test_table_curve(self):
        curve = TableStorageCurve(storages=[0, 1e6, 2e6], levels=[90, 95, 100])
        assert curve.level(1e6) == pytest.approx(95.0)
        assert curve.storage(97.5) == pytest.approx(1.5e6)

    def test_external_inflow_normalization(self):
        # both dict and plain-string entries must work
        n1 = SubBasinNode("a", subbasin_cfg("a") | {"external_inflows": ["div"]})
        n2 = SubBasinNode(
            "b",
            subbasin_cfg("b")
            | {"external_inflows": [{"name": "div", "required": False}]},
        )
        assert n1.external_slots() == ["P", "E", "in.div"]
        assert n2.external_slots() == ["P", "E", "in.div"]
        assert n1.slot_required("in.div") is True
        assert n2.slot_required("in.div") is False


# ---------------------------------------------------------------
# region HydroModels
# ---------------------------------------------------------------


class TestHydroModel:
    def test_gauge_weighting_and_run(self):
        m = basin_model()
        m.initialize()
        # P/E fed internally -> only the reservoir Q default output exists,
        # no P/E input ports were created
        port_ids = [p.id for p in m.inputs]
        assert not any(pid.endswith(".P") or pid.endswith(".E") for pid in port_ids)

        for _ in range(STEPS):
            status = m.update()
        assert status == ModelStatus.DONE

        sub_q = np.asarray(m.get_output("basin.sub1.Q").get_values()).flat[0]
        rsv_q = np.asarray(m.get_output("basin.rsv1.Q").get_values()).flat[0]
        assert sub_q > 0.0
        assert rsv_q >= 0.0
        # P = 0.5*10 + 0.5*20 = 15 mm must have produced runoff
        r = m.node("sub1").current("R")
        assert r > 0.0

    def test_internal_feed_reaches_reservoir(self):
        m = basin_model()
        m.initialize()
        m.update()
        # sub1.Q must have been injected into rsv1 as internal inflow
        assert m.node("rsv1").total_inflow() >= 0.0
        assert m.node("rsv1").current("QI") == pytest.approx(
            m.node("rsv1").total_inflow()
        )

    def test_snapshot_restore_deterministic(self):
        m = basin_model()
        m.initialize()
        for _ in range(3):
            m.update()
        snap = m.snapshot()
        q1 = [np.asarray(m.get_output("basin.rsv1.Q").get_values()).flat[0]]
        for _ in range(3):
            m.update()
            q1.append(np.asarray(m.get_output("basin.rsv1.Q").get_values()).flat[0])
        m.restore(snap)
        q2 = [np.asarray(m.get_output("basin.rsv1.Q").get_values()).flat[0]]
        for _ in range(3):
            m.update()
            q2.append(np.asarray(m.get_output("basin.rsv1.Q").get_values()).flat[0])
        assert q1 == pytest.approx(q2)

    def test_param_vector_roundtrip(self):
        m = basin_model()
        m.initialize()
        names = m.param_names()
        assert any("sub1.runoff.WM" == n for n in names)
        assert any("rsv1.release.q_max" == n for n in names)
        vec = m.get_param_vector()
        m.set_param_vector(vec * 1.0)
        assert np.allclose(m.get_param_vector(), vec)

    def test_reset_run(self):
        m = basin_model()
        m.initialize()
        for _ in range(5):
            m.update()
        m.reset_run()
        assert m.node("rsv1").storage == pytest.approx(5.0e7)


class TestGaugeSet:
    def test_weight_validation(self):
        g = GaugeSet({"a": const_series("a", 1.0), "b": const_series("b", 2.0)})
        with pytest.raises(ValueError, match="sum to 1"):
            g.bind("P", {"a": 0.3, "b": 0.3})
        with pytest.raises(KeyError, match="unknown gauge"):
            g.bind("P", {"missing": 1.0})

    def test_weighted_value_and_clamping(self):
        g = GaugeSet({"a": const_series("a", 10.0), "b": const_series("b", 30.0)})
        g.bind("P", {"a": 0.25, "b": 0.75})
        assert g.value("P", 0) == pytest.approx(25.0)
        assert g.value("P", 10_000) == pytest.approx(25.0)  # clamps at end


# ---------------------------------------------------------------
# region ReservoirModel
# ---------------------------------------------------------------


class TestReservoirModel:
    def test_runs_with_unconnected_optional_inflow(self):
        r = ReservoirModel(
            {
                "id": "rsv",
                "dt": DT,
                "steps": 4,
                "storage": reservoir_cfg("x")["storage"],
                "reservoir": reservoir_cfg("x")["reservoir"],
                "inflows": [{"name": "up1", "required": False}],
            }
        )
        r.initialize()
        for _ in range(4):
            r.update()
        assert r.status == ModelStatus.DONE
        assert r.node.storage <= 5.0e7  # no inflow -> releases to target

    def test_required_inflow_must_be_connected(self):
        r = ReservoirModel(
            {
                "id": "rsv",
                "dt": DT,
                "steps": 4,
                "storage": reservoir_cfg("x")["storage"],
                "reservoir": reservoir_cfg("x")["reservoir"],
                "inflows": ["up1"],
            }
        )
        r.initialize()
        status = r.update()
        assert status == ModelStatus.FAILED
        assert "not connected" in r.get_last_error()


# ---------------------------------------------------------------
# region Scheduler
# ---------------------------------------------------------------


class TestScheduler:
    def test_topological_order_and_run(self):
        m = basin_model()
        sched = Scheduler()
        sched.add(m)
        sched.initialize()
        assert sched.execution_order == ["basin"]
        sched.run(max_steps=STEPS)
        assert m.status == ModelStatus.DONE

    def test_single_provider_enforced(self):
        m1 = basin_model()
        m2 = basin_model()
        sched = Scheduler()
        out = m1.get_output("basin.rsv1.Q")
        # give m2 a required external inflow port by direct construction
        inp_owner = ReservoirModel(
            {
                "id": "down",
                "dt": DT,
                "steps": 2,
                "storage": reservoir_cfg("x")["storage"],
                "reservoir": reservoir_cfg("x")["reservoir"],
                "inflows": ["up1"],
            }
        )
        inp = inp_owner.get_input("down.in.up1")
        sched.link(out, inp)
        with pytest.raises(ValueError, match="exactly one provider"):
            sched.link(m2.get_output("basin.rsv1.Q"), inp)

    def test_quantity_mismatch_rejected(self):
        m = basin_model()
        r = ReservoirModel(
            {
                "id": "down",
                "dt": DT,
                "steps": 2,
                "storage": reservoir_cfg("x")["storage"],
                "reservoir": reservoir_cfg("x")["reservoir"],
                "inflows": ["up1"],
            }
        )
        sched = Scheduler()
        # water level into a discharge port must be refused
        level_port = m.get_output("basin.rsv1.Z")
        with pytest.raises(ValueError, match="Quantity mismatch"):
            sched.link(level_port, r.get_input("down.in.up1"))

    def test_pull_chain_executes_upstream_first(self):
        up = basin_model()
        down = ReservoirModel(
            {
                "id": "down",
                "dt": DT,
                "steps": STEPS,
                "storage": reservoir_cfg("x")["storage"],
                "reservoir": reservoir_cfg("x")["reservoir"],
                "inflows": ["up1"],
            }
        )
        sched = Scheduler()
        sched.link(up.get_output("basin.rsv1.Q"), down.get_input("down.in.up1"))
        sched.initialize()
        assert sched.execution_order == ["basin", "down"]
        sched.run(max_steps=STEPS)
        assert down.node.current("QI") > 0.0
