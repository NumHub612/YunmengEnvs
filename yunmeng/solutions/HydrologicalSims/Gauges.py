# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Gauge forcing — component of a hydrological model.
"""

from __future__ import annotations

from yunmeng.solutions.commons.datasets import Timeseries


def _as_series(spec) -> Timeseries:
    """Accept a Timeseries or a {'time': [...], 'data': [...]} dict."""
    if isinstance(spec, Timeseries):
        return spec
    if isinstance(spec, dict) and "time" in spec and "data" in spec:
        return Timeseries(spec.get("id", "gauge"), spec["time"], spec["data"])
    raise ValueError(
        f"Gauge spec must be a Timeseries or a dict with time/data, got {spec!r}."
    )


class GaugeSet:
    """Named collection of station series owned by a model.

    Usage (inside HydrologyModel):
        gauges = GaugeSet(cfg.get("gauges", {}))
        gauges.bind("P", {"g1": 0.6, "g2": 0.4})   # weighted areal mean
        gauges.bind("E", "g3")                     # single station
        p = gauges.value("P", step_idx)
    """

    def __init__(self, specs: dict):
        self._series = {gid: _as_series(s) for gid, s in specs.items()}
        self._bindings: dict[str, list[tuple[str, float]]] = {}

    def __contains__(self, gid: str) -> bool:
        return gid in self._series

    def __len__(self) -> int:
        return len(self._series)

    @property
    def gauge_ids(self) -> list[str]:
        return list(self._series)

    def bind(self, slot: str, binding):
        """Bind a forcing slot to one gauge id or a {gauge: weight} map."""
        if isinstance(binding, str):
            binding = {binding: 1.0}
        pairs = []
        total = 0.0
        for gid, w in binding.items():
            if gid not in self._series:
                raise KeyError(
                    f"Forcing slot '{slot}' references unknown gauge '{gid}' "
                    f"(available: {sorted(self._series)})."
                )
            w = float(w)
            if w <= 0:
                raise ValueError(f"Gauge weight for '{gid}' must be > 0.")
            pairs.append((gid, w))
            total += w
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Gauge weights for slot '{slot}' must sum to 1 (got {total:.4f})."
            )
        self._bindings[slot] = pairs

    def has_binding(self, slot: str) -> bool:
        return slot in self._bindings

    def value(self, slot: str, step_idx: int) -> float:
        """Weighted value of a bound slot at *step_idx* (clamped at end)."""
        try:
            pairs = self._bindings[slot]
        except KeyError:
            raise KeyError(
                f"Forcing slot '{slot}' is not bound to any gauge."
            ) from None
        total = 0.0
        for gid, w in pairs:
            series = self._series[gid]
            idx = min(step_idx, len(series) - 1)
            total += w * series.value_at(idx)
        return total
