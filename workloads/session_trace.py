"""Canonical SessionTraceV0 schema — the source of truth for synthetic benchmark traces.

This is the richest representation. Flat rows, replay events, and InferenceX adapters
are derived from this. Do NOT use flat per-turn rows as primary storage.

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool", "retrieval", "execution"]
BlockType = Literal["text", "image", "log", "code", "document", "table"]
WorkloadFamily = Literal[
    "short_chat", "long_chat", "rag", "coding", "agent", "multimodal", "cache_stress"
]
CacheExpectation = Literal["cold", "warm", "reactivated"]
ArrivalPattern = Literal[
    "poisson_bursty", "steady_interactive", "reactivation_heavy", "swarm"
]
SourceType = Literal["real", "synthetic", "hybrid"]
KVTier = Literal["HBM", "CPU", "NVMe", "remote"]


class ContentBlock(BaseModel):
    type: BlockType
    text: Optional[str] = None
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceTurn(BaseModel):
    turn_id: int
    parent_turn_id: Optional[int]
    role: Role
    content_blocks: list[ContentBlock]
    timestamp: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cumulative_tokens: int = Field(ge=0)


class ReplayInfo(BaseModel):
    prefix_group_id: str
    prefix_overlap_ratio: float = Field(ge=0.0, le=1.0)
    reuse_distance_turns: int = Field(ge=0)
    reuse_distance_seconds: int = Field(ge=0)
    estimated_kv_bytes_peak: int = Field(ge=0)
    branch_count: int = Field(ge=0)
    idle_windows_ms: list[int]
    arrival_pattern: ArrivalPattern


class KVPressure(BaseModel):
    cold_start: bool
    warm_cache_hit_rate: float = Field(ge=0.0, le=1.0)
    estimated_offload_bytes: int = Field(ge=0)
    estimated_reload_bytes: int = Field(ge=0)
    tier: KVTier
    cache_expectation: CacheExpectation
    eviction_risk_score: float = Field(ge=0.0, le=1.0)


class SessionMeta(BaseModel):
    session_id: str
    goal: str
    contains_tools: bool
    contains_rag: bool
    contains_branching: bool
    contains_multimodal: bool


class Provenance(BaseModel):
    generator_model: str
    generator_prompt_version: str
    pipeline_version: str


class TokenizerInfo(BaseModel):
    name: str
    version: str


class SessionTrace(BaseModel):
    """Canonical trace — the single source of truth for one benchmark session."""

    trace_id: str
    dataset_version: str = "0.1.0"
    source_type: SourceType
    workload_family: WorkloadFamily
    language: str = "en"
    tokenizer: TokenizerInfo
    session: SessionMeta
    turns: list[TraceTurn]
    replay: ReplayInfo
    kv_pressure: KVPressure
    provenance: Provenance

    def total_tokens(self) -> int:
        """Total cumulative tokens (last turn's cumulative value)."""
        if not self.turns:
            return 0
        return self.turns[-1].cumulative_tokens

    def turn_count(self) -> int:
        return len(self.turns)

    def tool_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role in ("tool", "execution"))

    def retrieval_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "retrieval")
