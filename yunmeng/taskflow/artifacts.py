# -*- encoding: utf-8 -*-
"""
LocalArtifactStore: theta persistence via a local directory convention
(v1.0 §7.3 first edition: models/<model_id>/<version>/weights.pt + meta.json).

Operators never touch files; model layer resolves ModelRef -> set_parameters.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import torch

from yunmeng.interfaces.solution import ModelRef, TrainingMeta


class LocalArtifactStore:
    """Directory-convention artifact store."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # -- IModelArtifactStore ---------------------------

    def resolve(self, ref: ModelRef) -> dict[str, torch.Tensor]:
        d = self._dir_of(ref)
        wpath = d / "weights.pt"
        if not wpath.exists():
            raise FileNotFoundError(
                f"Missing model artifact {ref} (expected at {wpath}). "
                f"Available versions of '{ref.model_id}': {self.list_versions(ref.model_id)}"
            )
        state = torch.load(wpath, map_location="cpu", weights_only=True)
        return {k: v.to(torch.float64) for k, v in state.items()}

    def register(
        self, params: Mapping, meta: TrainingMeta, model_id: str = "model"
    ) -> ModelRef:
        version = self._next_version(model_id)
        ref = ModelRef(model_id=model_id, version=version)
        d = self._dir_of(ref)
        d.mkdir(parents=True, exist_ok=False)

        state = {
            k: (
                v.detach().cpu().to(torch.float64)
                if isinstance(v, torch.Tensor)
                else torch.as_tensor(v, dtype=torch.float64)
            )
            for k, v in params.items()
        }
        torch.save(state, d / "weights.pt")

        m = {
            "model_id": ref.model_id,
            "version": ref.version,
            "estimator": meta.estimator,
            "data_lineage": meta.data_lineage,
            "metrics": meta.metrics,
            "created_at": meta.created_at,
            "notes": meta.notes,
        }
        (d / "meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
        return ref

    def meta(self, ref: ModelRef) -> TrainingMeta:
        d = self._dir_of(ref)
        mpath = d / "meta.json"
        if not mpath.exists():
            raise FileNotFoundError(f"Missing meta for {ref}.")
        m = json.loads(mpath.read_text())
        return TrainingMeta(
            estimator=m.get("estimator", ""),
            data_lineage=m.get("data_lineage", ""),
            metrics=m.get("metrics", {}),
            created_at=m.get("created_at", ""),
            notes=m.get("notes", ""),
        )

    def list_versions(self, model_id: str) -> list[str]:
        d = self._root / model_id
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    # -- internals ---------------------------------------

    def _dir_of(self, ref: ModelRef) -> Path:
        return self._root / ref.model_id / ref.version

    def _next_version(self, model_id: str) -> str:
        best = (0, 0, 0)
        for v in self.list_versions(model_id):
            m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
            if m:
                best = max(best, tuple(map(int, m.groups())))
        return "%d.%d.%d" % (best[0], best[1], best[2] + 1)
