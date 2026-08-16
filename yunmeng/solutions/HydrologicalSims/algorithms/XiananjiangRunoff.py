# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Xinanjiang (XAJ) three-source runoff generation algorithem.

Structure:
  * three-layer evapotranspiration (upper/lower/deep tension water)
  * tension-water capacity distribution curve (B) -> total runoff R
  * free-water storage with distribution curve (EX) -> separation into
    surface runoff RS, interflow RI and groundwater RG

Implementation note: the free-water storage S is tracked as the
*basin-mean* depth [mm] (equivalent reformulation of the classic
contributing-area formulation; mass-conserving and simpler to keep
consistent under snapshot/restore).

Recession coefficients KI/KG are per-step fractions (0<KI+KG<1).
"""

from __future__ import annotations

from yunmeng.solutions.standards import ParamMeta
from yunmeng.solutions.HydrologicalSims.algorithms.Bases import (
    RunoffGeneration,
    register,
)


class XinanjiangRunoff(RunoffGeneration):
    """Xinanjiang (XAJ) three-source runoff generation."""

    algo_name = "xaj"

    @classmethod
    def inits_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "W0",
                "initial tension water ratio (W/WM)",
                bounds=(0.0, 1.0),
                default=0.5,
            ),
            ParamMeta(
                "S0",
                "initial free water storage [mm]",
                bounds=(0.0, 60.0),
                default=10.0,
            ),
        ]

    @classmethod
    def param_spec(cls) -> list[ParamMeta]:
        return [
            ParamMeta(
                "WM",
                "areal mean tension water capacity [mm]",
                bounds=(10.0, 200.0),
                default=120.0,
            ),
            ParamMeta(
                "WUM",
                "upper layer tension water capacity [mm]",
                bounds=(5.0, 50.0),
                default=20.0,
            ),
            ParamMeta(
                "WLM",
                "lower layer tension water capacity [mm]",
                bounds=(10.0, 90.0),
                default=60.0,
            ),
            ParamMeta(
                "B",
                "tension water capacity curve exponent",
                bounds=(0.05, 2.0),
                default=0.3,
            ),
            ParamMeta(
                "C",
                "deep evaporation coefficient",
                bounds=(0.05, 0.4),
                default=0.15,
            ),
            ParamMeta(
                "SM",
                "areal mean free water capacity [mm]",
                bounds=(5.0, 60.0),
                default=30.0,
            ),
            ParamMeta(
                "EX",
                "free water capacity curve exponent",
                bounds=(0.5, 2.0),
                default=1.0,
            ),
            ParamMeta(
                "KI",
                "interflow recession",
                bounds=(0.0, 0.7),
                default=0.3,
            ),
            ParamMeta(
                "KG",
                "groundwater recession",
                bounds=(0.0, 0.7),
                default=0.2,
            ),
        ]

    def __init__(self, **params):
        super().__init__(**params)
        if self.p("WUM") + self.p("WLM") >= self.p("WM"):
            raise ValueError(
                "XinanjiangRunoff: WUM + WLM must be < WM "
                f"({self.p('WUM')} + {self.p('WLM')} >= {self.p('WM')})."
            )
        if self.p("KI") + self.p("KG") >= 1.0:
            raise ValueError("XinanjiangRunoff: KI + KG must be < 1.")
        self._init_states()

    def _init_states(self):
        wm = self.p("WM")
        w = self.p("W0") * wm
        # distribute initial tension water top-down
        self._wu = min(w, self.p("WUM"))
        self._wl = min(max(w - self._wu, 0.0), self.p("WLM"))
        self._wd = max(w - self._wu - self._wl, 0.0)
        self._s = self.p("S0")
        self._fr = 0.5  # contributing-area fraction memory

    # -- state --------------------------------------

    def state(self) -> dict:
        return {
            "WU": self._wu,
            "WL": self._wl,
            "WD": self._wd,
            "S": self._s,
            "FR": self._fr,
        }

    def set_state(self, state: dict):
        self._wu = float(state["WU"])
        self._wl = float(state["WL"])
        self._wd = float(state["WD"])
        self._s = float(state["S"])
        self._fr = float(state["FR"])

    @property
    def tension_water(self) -> float:
        return self._wu + self._wl + self._wd

    @property
    def free_water(self) -> float:
        return self._s

    # -- one step -----------------------------------

    def produce(
        self, rain: float, evap: float, dt: float
    ) -> tuple[float, float, float]:
        p_in = max(float(rain), 0.0)
        ep = max(float(evap), 0.0)

        wm, wum, wlm = self.p("WM"), self.p("WUM"), self.p("WLM")
        b, c = self.p("B"), self.p("C")
        sm, ex = self.p("SM"), self.p("EX")
        ki, kg = self.p("KI"), self.p("KG")
        wdm = wm - wum - wlm

        # ---- 1. three-layer evapotranspiration
        if self._wu + p_in >= ep:
            eu, el, ed = ep, 0.0, 0.0
        else:
            eu = self._wu + p_in
            rest = ep - eu
            if self._wl >= c * wlm:
                el, ed = rest * self._wl / wlm, 0.0
            elif self._wl >= c * rest:
                el, ed = c * rest, 0.0
            else:
                el = self._wl
                ed = min(max(c * rest - el, 0.0), self._wd)

        pe = p_in - (eu + el + ed)

        # ---- 2. runoff generation (tension water curve)
        r = 0.0
        if pe > 0.0:
            w = self.tension_water
            wmm = wm * (1.0 + b)
            a = wmm * (1.0 - (1.0 - w / wm) ** (1.0 / (1.0 + b)))
            if pe + a < wmm:
                r = pe - (wm - w) + wm * (1.0 - (pe + a) / wmm) ** (1.0 + b)
            else:
                r = pe - (wm - w)
            r = max(r, 0.0)

            # tension water balance: W += PE - R, filled top-down
            w_new = min(w + pe - r, wm)
            self._wu = min(w_new, wum)
            self._wl = min(max(w_new - self._wu, 0.0), wlm)
            self._wd = max(w_new - self._wu - self._wl, 0.0)
        else:
            self._wu = max(self._wu + p_in - eu, 0.0)
            self._wl = max(self._wl - el, 0.0)
            self._wd = max(self._wd - ed, 0.0)

        # ---- 3. source separation (free water storage)
        if r > 0.0 and pe > 0.0:
            fr = min(max(r / pe, 0.0), 1.0)
            frs = 0.5 * (self._fr + fr)
            self._fr = fr
        else:
            frs = self._fr

        rs = ri = rg = 0.0
        if r > 0.0 and frs > 1e-6:
            s_loc = self._s / frs  # storage over contributing area
            s_loc = min(s_loc, sm * 0.999999)
            smm = sm * (1.0 + ex)
            au = smm * (1.0 - (1.0 - s_loc / sm) ** (1.0 / (1.0 + ex)))
            if pe + au < smm:
                rs = frs * (
                    pe - sm + s_loc + sm * (1.0 - (pe + au) / smm) ** (1.0 + ex)
                )
            else:
                rs = frs * (pe + s_loc - sm)
            rs = min(max(rs, 0.0), r)

        # interflow / groundwater recession continues even without rain
        ri = ki * self._s
        rg = kg * self._s
        self._s = max(self._s + r - rs - ri - rg, 0.0)

        return rs, ri, rg


register("runoff", XinanjiangRunoff.algo_name, XinanjiangRunoff)
