"""ReplayManifestV0 schema — benchmark execution plan derived from canonical traces.

One trace can produce events in many manifests. The manifest defines arrival timing,
session interleaving, and cache expectations for a specific benchmark mode.

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from workloads.session_trace import CacheExpectation


class ReplayInputMessage(BaseModel):
    role: str
    content_blocks: list[dict[str, Any]]


class ReplayEvent(BaseModel):
    """A single request in a replay benchmark run."""

    event_id: str
    trace_id: str
    session_id: str
    turn_id: int
    arrival_time_offset_ms: int = Field(ge=0)
    input_messages: list[ReplayInputMessage]
    target_output_tokens: int = Field(ge=1)
    cache_expectation: CacheExpectation
    prefix_group_id: str
    prefix_overlap_ratio: float = Field(ge=0.0, le=1.0)
    reuse_distance_turns: int = Field(ge=0)
    reuse_distance_ms: int = Field(ge=0)
    estimated_kv_bytes: int = Field(ge=0)
    workload_family: str


class ManifestMetadata(BaseModel):
    concurrency_profile: str
    duration_sec: int = Field(ge=1)
    notes: str = ""


class ReplayManifest(BaseModel):
    """Benchmark execution plan — events + timing + concurrency profile."""

    manifest_version: str = "0.1.0"
    name: str
    events: list[ReplayEvent]
    metadata: ManifestMetadata

    def total_events(self) -> int:
        return len(self.events)

    def total_input_tokens(self) -> int:
        return sum(
            sum(len(str(m.content_blocks)) for m in e.input_messages)
            for e in self.events
        )
