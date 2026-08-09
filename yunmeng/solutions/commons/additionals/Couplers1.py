# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Concrete coupling execution layer.
"""

from __future__ import annotations
import numpy as np

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    IStateful,
    ICouplingStrategy,
    IIterativeCoupler,
    CouplingMode,
    CouplingConfig,
    IterationResult,
    DivergenceAction,
)
from yunmeng.setting import logger

# ---------------------------------------------------
# region PullCoupler
# ---------------------------------------------------


class PullCoupler(ICouplingStrategy):
    """One-way PULL coupling: data transfer happens implicitly through
    the target's input ports when it updates.  The strategy therefore
    only needs to advance the target."""

    @property
    def mode(self) -> CouplingMode:
        return CouplingMode.PULL

    def execute(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
        target.update()
        return IterationResult(
            converged=True, iterations=1, residual=0.0, message="pull transfer"
        )


# ---------------------------------------------------
# region FixedPointCoupler
# ---------------------------------------------------


class FixedPointCoupler(IIterativeCoupler):
    """Fixed-point iterative coupler for LOOP-linked component pairs.

    Both components must implement ``IStateful`` — re-advancing the
    same step is only possible via snapshot/restore.

    Snapshot contract: a snapshot captures the component's complete
    observable state INCLUDING the frames published on its output
    ports.  Each iteration therefore re-injects the latest relaxed
    iterate into the exchanged output caches *after* restoring, which
    keeps the fixed-point sequence intact across restores.
    """

    def __init__(self, use_relative: bool = False):
        self._use_relative = use_relative

    @property
    def mode(self) -> CouplingMode:
        return CouplingMode.LOOP

    def execute(self, source, target, config) -> IterationResult:
        return self.iterate(source, target, config)

    # -- helpers ------------------------------------------

    @staticmethod
    def _require_stateful(comp: ILinkableModel):
        if not isinstance(comp, IStateful):
            raise TypeError(
                f"LOOP coupling requires IStateful components; "
                f"'{comp.id}' does not implement snapshot/restore."
            )

    @staticmethod
    def _exchanged_ports(comp_a, comp_b) -> list[tuple[IOutput, IInput]]:
        """All (output, input) pairs linking the two components."""
        pairs = []
        for consumer, provider_owner in ((comp_a, comp_b), (comp_b, comp_a)):
            for inp in consumer.inputs:
                for out in inp.providers:
                    if getattr(out, "owner", None) is provider_owner:
                        pairs.append((out, inp))
        return pairs

    def _extract_vector(self, pairs, config):
        parts, slices, pos = [], [], 0
        for out, _ in pairs:
            short = out.id.split(".")[-1]
            if config.convergence_vars and short not in config.convergence_vars:
                continue
            v = np.atleast_1d(np.asarray(out.get_values(), dtype=float))
            parts.append(v)
            slices.append((out, pos, pos + v.size))
            pos += v.size
        if not parts:
            return np.zeros(0), []
        return np.concatenate(parts), slices

    @staticmethod
    def _write_vector(slices, vector: np.ndarray):
        for out, start, end in slices:
            out.set_values(vector[start:end])

    def _vector_residual(self, previous, current) -> float:
        diff = np.abs(current - previous)
        if self._use_relative:
            diff = diff / (0.5 * (np.abs(previous) + np.abs(current)) + 1e-12)
        return float(np.max(diff))

    # -- main loop ------------------------------------------

    def iterate(self, comp_a, comp_b, config) -> IterationResult:
        self._require_stateful(comp_a)
        self._require_stateful(comp_b)

        pairs = self._exchanged_ports(comp_a, comp_b)
        if not pairs:
            return IterationResult(
                converged=True,
                iterations=0,
                residual=0.0,
                message="no exchanged ports between components",
            )

        snap_a = comp_a.snapshot()
        snap_b = comp_b.snapshot()

        history: list[float] = []
        converged = False
        residual = np.inf
        k = 0

        omega = config.relaxation
        u_pp = None  # iterate k-2 (relaxed)
        u_p = None  # iterate k-1 (relaxed)

        for k in range(1, config.max_iterations + 1):
            # Re-advance the SAME step from the pre-step state.
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)
            # Re-inject the relaxed iterate so this round's update()
            # pulls the *latest* boundary conditions, not the
            # pre-step ones wiped out by restore().
            if u_p is not None:
                # slices are recomputed below but geometry is stable
                self._write_vector(self._last_slices, u_p)

            comp_a.update()
            comp_b.update()

            u_c, slices = self._extract_vector(pairs, config)
            self._last_slices = slices
            if u_p is None:
                u_p = u_c
                continue

            residual = self._vector_residual(u_p, u_c)
            history.append(residual)
            if residual < config.tolerance:
                converged = True
                break

            if u_pp is not None:
                d1 = u_p - u_pp
                d2 = u_c - u_p
                dd = d2 - d1
                denom = float(np.dot(dd, dd))
                if denom > 1e-30:
                    omega = float(np.clip(-omega * np.dot(d1, dd) / denom, 0.05, 2.0))
            u_rel = u_p + omega * (u_c - u_p)
            self._write_vector(slices, u_rel)
            u_pp, u_p = u_p, u_rel

        if not converged:
            self._on_divergence(comp_a, comp_b, snap_a, snap_b, config)
            return IterationResult(
                converged=False,
                iterations=k,
                residual=float(residual),
                residual_history=history,
                message=f"diverged after {k} iterations; "
                f"action={config.divergence_action.name}",
            )

        return IterationResult(
            converged=True,
            iterations=k,
            residual=float(residual),
            residual_history=history,
            message=f"converged in {k} iterations",
        )

    def converge(self, previous, current, config) -> tuple[bool, float]:
        residual = 0.0
        for name, cur in current.items():
            prev = previous.get(name)
            if prev is None or prev.shape != cur.shape:
                continue
            diff = np.abs(cur - prev)
            if self._use_relative:
                diff = diff / (0.5 * (np.abs(prev) + np.abs(cur)) + 1e-12)
            residual = max(residual, float(np.max(diff)))
        return residual < config.tolerance, residual

    # -- divergence handling ---------------------------------

    @staticmethod
    def _on_divergence(comp_a, comp_b, snap_a, snap_b, config):
        action = config.divergence_action
        if not isinstance(action, DivergenceAction):
            # tolerate the enum *name* or *value* being passed
            try:
                action = DivergenceAction[action]
            except (KeyError, TypeError):
                try:
                    action = DivergenceAction(action)
                except ValueError:
                    raise ValueError(f"Unknown divergence_action '{action}'.") from None

        if action is DivergenceAction.ROLLBACK:
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)
            comp_a.mark_failed("LOOP coupling diverged (rollback).")
            comp_b.mark_failed("LOOP coupling diverged (rollback).")
        elif action is DivergenceAction.FREEZE:
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)
        elif action is DivergenceAction.CONTINUE:
            logger.warning.warn(
                f"LOOP coupling {comp_a.id}<->{comp_b.id} diverged; "
                f"keeping the best approximation.",
                stacklevel=2,
            )
