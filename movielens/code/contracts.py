"""Typed query, candidate, pair, direction, intervention, and result schemas.

These contracts are dataset-agnostic. Dataset-specific fields live in `metadata`
dicts populated by data.py adapters and rendered by prompts.py via configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

SCHEMA_VERSION = 1

# Column contracts for persisted tables (Parquet). Checked on write and read.
LABEL_COLUMNS = [
    "query_id", "item_id", "rating", "satisfaction_label", "graded_gain", "target_attribute",
    "observed_timestamp", "label_observed",
]
SCORE_COLUMNS = [
    "pair_id", "query_id", "item_id", "prompt_hash", "baseline_score", "logit_pos", "logit_neg",
    "verbalizer_ids", "model_revision", "scorer_version",
]
STATE_KEY_COLUMNS = ["pair_id", "layer_index", "hook_location", "prediction_position"]


def make_pair_id(query_id: str, item_id: Any) -> str:
    return f"{query_id}#{item_id}"


def split_pair_id(pair_id: str) -> tuple[str, str]:
    q, _, i = pair_id.rpartition("#")
    return q, i


@dataclass(frozen=True)
class HistoryEvent:
    item_id: str
    timestamp: int
    feedback: dict[str, Any]          # already-observed feedback, e.g. {"rating": 4}
    metadata: dict[str, Any]          # item metadata available for rendering
    metadata_missing: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """What the prompt renderer may see. Never carries the current-query outcome."""

    item_id: str
    display_text: str
    metadata: dict[str, Any]
    metadata_missing: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        forbidden = {"rating", "satisfaction_label", "graded_gain", "observed_timestamp", "label"}
        leaked = forbidden & set(self.metadata)
        if leaked:
            raise ValueError(f"candidate {self.item_id} carries outcome fields {sorted(leaked)}")


@dataclass(frozen=True)
class Query:
    query_id: str
    user_id: str
    dataset: str
    split: str                        # fit | validation | test
    cutoff_timestamp: int
    history: tuple[HistoryEvent, ...]
    candidate_ids: tuple[str, ...]
    candidate_source: str
    protocol_id: str
    user_group: str                   # inner_fit | probe_val | unseen_by_direction

    @staticmethod
    def from_record(r: dict) -> "Query":
        return Query(
            query_id=r["query_id"], user_id=str(r["user_id"]), dataset=r["dataset"], split=r["split"],
            cutoff_timestamp=int(r["cutoff_timestamp"]),
            history=tuple(HistoryEvent(**h) for h in r["history"]),
            candidate_ids=tuple(str(c) for c in r["candidate_ids"]),
            candidate_source=r["candidate_source"], protocol_id=r["protocol_id"], user_group=r["user_group"],
        )

    def to_record(self) -> dict:
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        d["candidate_ids"] = list(self.candidate_ids)
        return d


@dataclass(frozen=True)
class Direction:
    direction_id: str
    estimator: str                    # mean_diff | probe | random | shuffled_mean_diff
    objective: Optional[str]
    layer_index: int
    hook_location: str
    seed: int
    vector: np.ndarray                # float32 unit vector, raw residual coordinates
    radius: float                     # median raw activation norm on inner-fit pairs
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.vector, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if not np.all(np.isfinite(v)) or n == 0.0:
            raise ValueError(f"direction {self.direction_id}: zero or non-finite vector (failed fit)")
        if abs(n - 1.0) > 1e-4:
            raise ValueError(f"direction {self.direction_id}: norm {n} != 1")
        if not (np.isfinite(self.radius) and self.radius > 0):
            raise ValueError(f"direction {self.direction_id}: bad radius {self.radius}")


@dataclass(frozen=True)
class Intervention:
    """One additive edit at one layer at the last valid prompt position.

    kind: none | add_block (block output, before next block/final norm)
          | add_postnorm (after final normalization; null control)
    delta = strength * radius * vector
    """

    kind: str = "none"
    layer_index: Optional[int] = None
    direction_id: Optional[str] = None
    vector_sha: Optional[str] = None
    strength: float = 0.0
    radius: float = 0.0

    def identity(self) -> dict:
        return {
            "kind": self.kind, "layer_index": self.layer_index, "direction_id": self.direction_id,
            "vector_sha": self.vector_sha, "strength": round(float(self.strength), 10),
            "radius": round(float(self.radius), 8),
        }


NO_INTERVENTION = Intervention()


@dataclass
class MetricRow:
    """One row of the final comparison table. Undefined values are None with a reason."""

    objective: str
    method: str
    split: str
    subgroup: str
    metric: str
    vector_seed: Optional[int]
    layer: Optional[int]
    strength_or_lambda: Optional[float]
    guardrail_budget: Optional[float]
    baseline_value: Optional[float]
    method_value: Optional[float]
    abs_delta: Optional[float]
    rel_delta: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    n_queries: int
    n_users: int
    n_candidates: int
    undefined_reason: Optional[str] = None
    selection_id: Optional[str] = None
