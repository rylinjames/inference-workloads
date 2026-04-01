"""Evaluator stage — validates generated traces for benchmark usefulness.

Cross-model evaluation: Claude evaluates Codex's traces, Codex evaluates Claude's.
Also runs automated structural/schema validation.

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0, Section 8.4 + 9
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from workloads.session_trace import SessionTrace
from workloads.synthetic_planner import TracePlan


# ---------------------------------------------------------------------------
# Automated validation (runs on 100% of traces)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    stage: str  # "schema" | "structural" | "kv_sanity" | "workload_heuristics"
    errors: list[str]
    warnings: list[str]


def validate_schema(trace: SessionTrace) -> ValidationResult:
    """Stage A — Schema validation. Pydantic handles this at parse time."""
    # If we got here, the trace parsed successfully
    return ValidationResult(passed=True, stage="schema", errors=[], warnings=[])


def validate_structural(trace: SessionTrace) -> ValidationResult:
    """Stage B — Structural validation."""
    errors = []
    warnings = []

    # Timestamps should be non-decreasing (lexicographic for ISO strings)
    prev_ts = ""
    for turn in trace.turns:
        if turn.timestamp < prev_ts:
            errors.append(f"Turn {turn.turn_id}: timestamp {turn.timestamp} < previous {prev_ts}")
        prev_ts = turn.timestamp

    # turn_id unique and ordered
    ids = [t.turn_id for t in trace.turns]
    if ids != sorted(set(ids)):
        errors.append("turn_ids not unique and ordered")

    # parent_turn_id points backward or null
    for turn in trace.turns:
        if turn.parent_turn_id is not None and turn.parent_turn_id >= turn.turn_id:
            errors.append(f"Turn {turn.turn_id}: parent_turn_id {turn.parent_turn_id} not backward")

    # cumulative_tokens non-decreasing
    prev_cum = -1
    for turn in trace.turns:
        if turn.cumulative_tokens < prev_cum:
            errors.append(f"Turn {turn.turn_id}: cumulative_tokens regressed ({turn.cumulative_tokens} < {prev_cum})")
        prev_cum = turn.cumulative_tokens

    # At least one content block per turn
    for turn in trace.turns:
        if not turn.content_blocks:
            errors.append(f"Turn {turn.turn_id}: no content blocks")

    return ValidationResult(
        passed=len(errors) == 0,
        stage="structural",
        errors=errors,
        warnings=warnings,
    )


def validate_kv_sanity(trace: SessionTrace) -> ValidationResult:
    """Stage C — Tokenization / KV sanity checks."""
    errors = []
    warnings = []

    # Verify estimated_kv_bytes_peak is reasonable
    total_tokens = trace.total_tokens()
    # Rough estimate: 2 bytes per token per layer for KV cache
    # Very approximate — just sanity check, not exact
    if trace.replay.estimated_kv_bytes_peak == 0 and total_tokens > 1000:
        warnings.append("estimated_kv_bytes_peak is 0 for a non-trivial trace")

    if trace.kv_pressure.eviction_risk_score > 0.8 and trace.kv_pressure.tier == "HBM":
        warnings.append("High eviction risk but tier is HBM — may need CPU/NVMe tier")

    return ValidationResult(
        passed=len(errors) == 0,
        stage="kv_sanity",
        errors=errors,
        warnings=warnings,
    )


# Token bin ranges
BIN_RANGES = {
    "8k_16k": (8000, 16000),
    "32k_64k": (32000, 64000),
    "96k_128k": (96000, 128000),
    "16k_32k": (16000, 32000),
    "64k_96k": (64000, 96000),
    "128k_plus": (128000, 2_000_000),
}


def validate_workload_heuristics(trace: SessionTrace, plan: TracePlan) -> ValidationResult:
    """Stage D — Workload-specific heuristics."""
    errors = []
    warnings = []

    # Check context bin
    total = trace.total_tokens()
    if plan.target_bin in BIN_RANGES:
        lo, hi = BIN_RANGES[plan.target_bin]
        if total < lo * 0.5:  # Allow 50% tolerance for v0
            errors.append(f"Total tokens {total} far below target bin {plan.target_bin} ({lo}-{hi})")
        elif total < lo:
            warnings.append(f"Total tokens {total} below target bin floor {lo}")

    # Long chat traces: at least 10 turns
    if plan.family == "long_chat" and trace.turn_count() < 10:
        warnings.append(f"Long chat trace has only {trace.turn_count()} turns (want >= 10)")

    # Coding traces: must include tool/execution/retrieval turns
    if plan.family == "coding":
        if trace.tool_turn_count() + trace.retrieval_turn_count() < 2:
            warnings.append("Coding trace has fewer than 2 tool/execution/retrieval turns")

    # Check for idle windows in reactivation traces
    if plan.cache_expectation == "reactivated":
        max_idle = max(plan.idle_windows_ms) if plan.idle_windows_ms else 0
        if max_idle < 10000:
            warnings.append("Reactivation trace has no idle window > 10s")

    return ValidationResult(
        passed=len(errors) == 0,
        stage="workload_heuristics",
        errors=errors,
        warnings=warnings,
    )


def run_all_validations(trace: SessionTrace, plan: TracePlan) -> list[ValidationResult]:
    """Run all 4 validation stages."""
    return [
        validate_schema(trace),
        validate_structural(trace),
        validate_kv_sanity(trace),
        validate_workload_heuristics(trace, plan),
    ]


# ---------------------------------------------------------------------------
# LLM-based evaluation (cross-model)
# ---------------------------------------------------------------------------

EVALUATOR_PROMPT = """\
You are reviewing a candidate SessionTraceV0 object for benchmark usefulness.

Decide PASS or FAIL and explain why.

Check the following strictly:
1. Coherence: do turns logically follow from earlier turns?
2. Realism: does this feel like a real user / coding / tool workflow?
3. Workload fidelity: would this actually stress long-context serving, prefix reuse, and KV offload?
4. Schema fidelity: are replay and kv_pressure fields plausible given the trace?
5. Context bin fidelity: does the trace realistically reach its intended cumulative token target?

Reject traces that are:
- too generic
- too clean
- too short-horizon
- not actually tool/agent heavy
- missing meaningful long-context growth
- missing meaningful cache-relevant structure

The trace's target context bin is: {target_bin}
The trace's workload family is: {family}

Here is the trace (truncated to first 20 turns for review):

{trace_json}

Return JSON only:
{{"decision": "PASS|FAIL", "scores": {{"coherence": 0.0, "realism": 0.0, "workload_fidelity": 0.0, "schema_fidelity": 0.0}}, "reasons": ["..."], "required_fixes": ["..."]}}
"""


@dataclass
class EvalResult:
    decision: str  # "PASS" | "FAIL"
    scores: dict[str, float]
    reasons: list[str]
    required_fixes: list[str]
    evaluator_model: str


def evaluate_with_claude(trace: SessionTrace, plan: TracePlan, timeout: int = 120) -> EvalResult:
    """Evaluate a trace using Claude Code CLI."""
    return _evaluate_with_model(trace, plan, "claude", timeout)


def evaluate_with_codex(trace: SessionTrace, plan: TracePlan, timeout: int = 120) -> EvalResult:
    """Evaluate a trace using Codex CLI."""
    return _evaluate_with_model(trace, plan, "codex", timeout)


def _evaluate_with_model(
    trace: SessionTrace, plan: TracePlan, model: str, timeout: int
) -> EvalResult:
    """Run LLM evaluation using the specified model CLI."""
    # Truncate trace to first 20 turns for review
    trace_dict = trace.model_dump()
    trace_dict["turns"] = trace_dict["turns"][:20]
    trace_json = json.dumps(trace_dict, indent=2)[:50000]  # Cap at 50k chars

    prompt = EVALUATOR_PROMPT.format(
        target_bin=plan.target_bin,
        family=plan.family,
        trace_json=trace_json,
    )

    if model == "claude":
        result = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", "haiku"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
    else:
        result = subprocess.run(
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "-"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )

    try:
        from workloads.synthetic_generator import _extract_json_from_output
        raw = _extract_json_from_output(result.stdout, model)
        return EvalResult(
            decision=raw.get("decision", "FAIL"),
            scores=raw.get("scores", {}),
            reasons=raw.get("reasons", []),
            required_fixes=raw.get("required_fixes", []),
            evaluator_model=model,
        )
    except (ValueError, KeyError) as e:
        return EvalResult(
            decision="FAIL",
            scores={},
            reasons=[f"Evaluator parse error: {e}"],
            required_fixes=["Re-evaluate with different model"],
            evaluator_model=model,
        )


def cross_evaluate(trace: SessionTrace, plan: TracePlan) -> EvalResult:
    """Cross-model evaluation: Claude evaluates Codex's traces and vice versa."""
    if plan.generator_model == "codex":
        return evaluate_with_claude(trace, plan)
    else:
        return evaluate_with_codex(trace, plan)
