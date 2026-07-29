# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Concrete coupling execution layer.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from yunmeng.solutions.standards import (
    ILinkableModel,
    IInput,
    IOutput,
    ICouplingStrategy,
    IIterativeCoupler,
    CouplingMode,
    CouplingConfig,
    IterationResult,
)

# ---------------------------------------------------
# region PullCoupler
# ---------------------------------------------------


class PullCoupler(ICouplingStrategy):
    """One-way PULL coupling: data transfer happens implicitly through
    the target's input ports when it updates (`BaseInput.pull` reads
    the provider's cache, applying any adapter chain).  The strategy
    therefore only needs to advance the target."""

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
            converged=True,
            iterations=1,
            residual=0.0,
            message="pull transfer",
        )


# ---------------------------------------------------
# region FixedPointCoupler
# ---------------------------------------------------


class FixedPointCoupler(IIterativeCoupler):
    """Fixed-point iterative coupler for LOOP-linked component pairs.

    Both components must implement `IStateful` (snapshot / restore);
    that is what makes re-advancing the same step possible.

    A LOOP pair advances one time step as follows:

    1. Snapshot both components (pre-step state).
    2. Iteration k:
         a. Restore both to the pre-step snapshot, so update()
            re-advances the *same* step (no over-accumulation).
         b. update(A) — A pulls B's latest exchanged outputs (cached
            from the previous iteration) as boundary conditions.
         c. update(B) — B pulls A's fresh outputs.
         d. Compare exchanged variables against iteration k-1.
         e. If omega < 1, write relaxed values into exchanged output
            caches, which is what the next iteration will pull.
    3. On convergence: the pair has advanced one step with mutually
       consistent boundary conditions.
    4. On divergence: apply CouplingConfig.divergence_action.
    """

    def __init__(self, use_relative: bool = False):
        """Initialize with convergence criteria.

        Args:
            use_relative: when True, convergence is judged on the
                relative change instead of the abs. difference.
        """
        self._use_relative = use_relative

    @property
    def mode(self) -> CouplingMode:
        return CouplingMode.LOOP

    def execute(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
        return self.iterate(source, target, config)

    def iterate(
        self,
        comp_a: ILinkableModel,
        comp_b: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
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

        # Fixed-point iterate over the stacked vector of exchanged
        # values.  Relaxation factor omega starts at config.relaxation
        # and is then adapted per iteration by Aitken's delta-squared
        # acceleration, which suppresses the oscillatory divergence
        # typical of strongly coupled pairs.
        omega = config.relaxation
        u_pp: np.ndarray = None  # iterate k-2 (relaxed)
        u_p: np.ndarray = None  # iterate k-1 (relaxed)

        for k in range(1, config.max_iterations + 1):
            # restore pre-step state: the step is re-advanced, never
            # accumulated, so source terms are applied exactly once
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)

            comp_a.update()
            comp_b.update()

            u_c, slices = self._extract_vector(pairs, config)
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
                f"action='{config.divergence_action}'",
            )

        return IterationResult(
            converged=True,
            iterations=k,
            residual=float(residual),
            residual_history=history,
            message=f"converged in {k} iterations",
        )

    def converge(
        self,
        previous: dict[str, np.ndarray],
        current: dict[str, np.ndarray],
        config: CouplingConfig,
    ) -> tuple[bool, float]:
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

    @staticmethod
    def _exchanged_ports(
        comp_a: ILinkableModel, comp_b: ILinkableModel
    ) -> list[tuple[IOutput, IInput]]:
        """All (output, input) pairs linking the two components, both ways."""
        pairs = []
        for consumer, provider_owner in ((comp_a, comp_b), (comp_b, comp_a)):
            for inp in consumer.inputs:
                if not inp.is_connected:
                    continue
                out = inp.provider
                if getattr(out, "model", None) is provider_owner:
                    pairs.append((out, inp))
        return pairs

    @staticmethod
    def _extract_vector(
        pairs: list[tuple[IOutput, IInput]], config: CouplingConfig
    ) -> tuple[np.ndarray, list[tuple]]:
        """Stack the exchanged output values into one vector.

        Returns the vector and a slice list [(output, start, end)] used
        to write relaxed values back into the output caches."""
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
    def _write_vector(slices: list[tuple], vector: np.ndarray):
        """Write relaxed values into exchanged output caches; the next
        iteration's pull() will read them as boundary conditions."""
        for out, start, end in slices:
            if hasattr(out, "set_values"):
                out.set_values(vector[start:end])

    def _vector_residual(self, previous: np.ndarray, current: np.ndarray) -> float:
        diff = np.abs(current - previous)
        if self._use_relative:
            diff = diff / (0.5 * (np.abs(previous) + np.abs(current)) + 1e-12)
        return float(np.max(diff))

    @staticmethod
    def _on_divergence(
        comp_a: ILinkableModel,
        comp_b: ILinkableModel,
        snap_a: Any,
        snap_b: Any,
        config: CouplingConfig,
    ):
        action = config.divergence_action
        if action == "rollback":
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)
            comp_a.mark_failed()
            comp_b.mark_failed()
        elif action == "freeze":
            comp_a.restore(snap_a)
            comp_b.restore(snap_b)
        elif action == "continue":
            pass  # keep the best approximation reached so far
        else:
            raise ValueError(f"Unknown divergence_action '{action}'.")
