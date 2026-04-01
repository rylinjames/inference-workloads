"""Materialize canonical traces into replay manifest events.

One trace can appear in multiple manifests. Each manifest defines a different
benchmark mode (steady state, shared prefix, reactivation, offload cliff).

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0, Section 4.2 + 6
"""

from __future__ import annotations

from uuid import uuid4

from workloads.session_trace import SessionTrace
from workloads.replay_manifest import (
    ManifestMetadata,
    ReplayEvent,
    ReplayInputMessage,
    ReplayManifest,
)


def materialize_trace_to_events(trace: SessionTrace) -> list[ReplayEvent]:
    """Convert a canonical trace into replay events (one per user turn)."""
    events: list[ReplayEvent] = []

    for turn in trace.turns:
        if turn.role != "user":
            continue

        # Build input_messages: full history up to this turn
        history: list[ReplayInputMessage] = []
        for past in trace.turns:
            if past.turn_id > turn.turn_id:
                break
            history.append(ReplayInputMessage(
                role=past.role,
                content_blocks=[b.model_dump() for b in past.content_blocks],
            ))

        # Pick idle window
        idle_idx = min(
            len(trace.replay.idle_windows_ms) - 1,
            max(0, turn.turn_id - 1),
        )
        arrival_offset = trace.replay.idle_windows_ms[idle_idx]

        events.append(ReplayEvent(
            event_id=f"{trace.trace_id}:{turn.turn_id}",
            trace_id=trace.trace_id,
            session_id=trace.session.session_id,
            turn_id=turn.turn_id,
            arrival_time_offset_ms=arrival_offset,
            input_messages=history,
            target_output_tokens=max(64, turn.output_tokens or 256),
            cache_expectation=trace.kv_pressure.cache_expectation,
            prefix_group_id=trace.replay.prefix_group_id,
            prefix_overlap_ratio=trace.replay.prefix_overlap_ratio,
            reuse_distance_turns=trace.replay.reuse_distance_turns,
            reuse_distance_ms=trace.replay.reuse_distance_seconds * 1000,
            estimated_kv_bytes=trace.replay.estimated_kv_bytes_peak,
            workload_family=trace.workload_family,
        ))

    return events


def build_manifest(
    name: str,
    traces: list[SessionTrace],
    concurrency_profile: str,
    duration_sec: int,
    notes: str = "",
) -> ReplayManifest:
    """Build a replay manifest from multiple traces."""
    all_events: list[ReplayEvent] = []
    for trace in traces:
        all_events.extend(materialize_trace_to_events(trace))

    return ReplayManifest(
        name=name,
        events=all_events,
        metadata=ManifestMetadata(
            concurrency_profile=concurrency_profile,
            duration_sec=duration_sec,
            notes=notes,
        ),
    )


def build_v0_manifests(traces: list[SessionTrace]) -> list[ReplayManifest]:
    """Build the 4 v0 manifests from the 24 canonical traces.

    Manifests:
    1. interactive_steady_state_v0 — all traces, natural timing
    2. shared_prefix_swarm_v0 — traces with high prefix overlap
    3. reactivation_stress_v0 — traces with reactivation cache expectation
    4. offload_cliff_v0 — largest context traces (128k+)
    """
    manifests = []

    # 1. Interactive steady state — all traces
    manifests.append(build_manifest(
        name="interactive_steady_state_v0",
        traces=traces,
        concurrency_profile="steady_interactive",
        duration_sec=300,
        notes="All 24 traces with natural think time and session interleaving.",
    ))

    # 2. Shared prefix swarm — traces with high overlap ratio
    high_overlap = [t for t in traces if t.replay.prefix_overlap_ratio >= 0.80]
    if high_overlap:
        manifests.append(build_manifest(
            name="shared_prefix_swarm_v0",
            traces=high_overlap,
            concurrency_profile="swarm",
            duration_sec=180,
            notes="Traces with >=80% prefix overlap. Tests prefix caching vs offload.",
        ))

    # 3. Reactivation stress — traces with reactivation expectation
    reactivation = [t for t in traces if t.kv_pressure.cache_expectation == "reactivated"]
    if reactivation:
        manifests.append(build_manifest(
            name="reactivation_stress_v0",
            traces=reactivation,
            concurrency_profile="reactivation_heavy",
            duration_sec=600,
            notes="Traces with idle windows and reactivation. Tests reload cost and resumed-session TTFT.",
        ))

    # 4. Offload cliff — largest context traces
    large_ctx = [t for t in traces if t.total_tokens() >= 96000]
    if not large_ctx:
        large_ctx = sorted(traces, key=lambda t: t.total_tokens(), reverse=True)[:6]
    manifests.append(build_manifest(
        name="offload_cliff_v0",
        traces=large_ctx,
        concurrency_profile="poisson_bursty",
        duration_sec=300,
        notes="Largest context traces. Push working set beyond easy HBM fit.",
    ))

    return manifests
