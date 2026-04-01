"""Generator stage — produces canonical SessionTraceV0 from planner blueprints.

Uses Claude Code CLI or Codex CLI as the LLM backend (both free via subscriptions).
Cross-model generation: Claude generates some traces, Codex generates others.

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0, Section 8.2-8.3
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from workloads.synthetic_planner import TracePlan
from workloads.session_trace import SessionTrace


GENERATION_PROMPT_TEMPLATE = """\
You are generating a single canonical SessionTraceV0 object for inference benchmarking.

Goal:
Create a synthetic but realistic long-context trace that is useful for benchmarking \
KV cache reuse, offloading, long-context growth, multi-turn reasoning, and tool-driven \
agent workflows.

Hard requirements:
- Output valid JSON only.
- Must conform to SessionTraceV0 schema exactly.
- Must include replay and kv_pressure fields.
- Must preserve realistic turn structure.
- Must not produce generic textbook chat.
- Must produce a trace that would stress a real inference system.

Workload family: {family}
Target context bin: {target_bin}
Planned goal: {goal}
Number of turns: {num_turns}
Tool pattern: {tool_pattern}
Branch pattern: {branch_pattern}
Idle windows ms: {idle_windows_ms}
Expected prefix overlap ratio: {expected_prefix_overlap_ratio}
Expected cache state: {cache_expectation}
Contains RAG/retrieval: {contains_rag}

Specific instructions:
- Strongly emphasize agents, tool use, coding, and reasoning.
- Include messy but plausible user behavior.
- Include realistic tool outputs, logs, diffs, or retrieval chunks where appropriate.
- Include prior-turn references after turn 5.
- Ensure cumulative context grows naturally toward the target bin.
- Do not create idealized or perfectly clean conversations.
- Do not use ShareGPT-style short, generic Q&A structure.
- Do not emit training-only assistant polish.
- Do not emit refusal-style safety boilerplate.

The SessionTraceV0 schema requires these top-level fields:
- trace_id (string)
- dataset_version (string, use "0.1.0")
- source_type (string, use "synthetic")
- workload_family (string: "long_chat" or "coding")
- language (string, use "en")
- tokenizer (object: {{"name": "cl100k_base", "version": "1.0"}})
- session (object: session_id, goal, contains_tools, contains_rag, contains_branching, contains_multimodal)
- turns (array of turn objects, each with: turn_id, parent_turn_id, role, content_blocks, timestamp, input_tokens, output_tokens, cumulative_tokens)
- replay (object: prefix_group_id, prefix_overlap_ratio, reuse_distance_turns, reuse_distance_seconds, estimated_kv_bytes_peak, branch_count, idle_windows_ms, arrival_pattern)
- kv_pressure (object: cold_start, warm_cache_hit_rate, estimated_offload_bytes, estimated_reload_bytes, tier, cache_expectation, eviction_risk_score)
- provenance (object: generator_model, generator_prompt_version, pipeline_version)

Each turn's content_blocks is an array of objects with: type ("text"|"code"|"log"|"document"), text, and optional uri/mime_type/metadata.

Return one JSON object only. No markdown fencing. No explanation.
"""


def _build_prompt(plan: TracePlan) -> str:
    """Build the generation prompt from a plan."""
    return GENERATION_PROMPT_TEMPLATE.format(
        family=plan.family,
        target_bin=plan.target_bin,
        goal=plan.goal,
        num_turns=plan.num_turns,
        tool_pattern=json.dumps(plan.tool_pattern),
        branch_pattern=plan.branch_pattern,
        idle_windows_ms=json.dumps(plan.idle_windows_ms),
        expected_prefix_overlap_ratio=plan.expected_prefix_overlap_ratio,
        cache_expectation=plan.cache_expectation,
        contains_rag=plan.contains_rag,
    )


def generate_with_claude(plan: TracePlan, timeout: int = 1200) -> dict:
    """Generate a trace using Claude Code CLI (Max subscription)."""
    prompt = _build_prompt(plan)

    # Pipe prompt via stdin to avoid shell argument length limits
    result = subprocess.run(
        [
            "claude", "-p",
            "--output-format", "json",
            "--max-budget-usd", "1.00",
            "--model", "opus",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return _extract_json_from_output(result.stdout, "claude")


def generate_with_codex(plan: TracePlan, timeout: int = 1200) -> dict:
    """Generate a trace using Codex CLI."""
    prompt = _build_prompt(plan)

    # Pipe prompt via stdin
    result = subprocess.run(
        [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "-",  # read from stdin
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return _extract_json_from_output(result.stdout, "codex")


def _extract_json_from_output(output: str, model: str) -> dict:
    """Extract a JSON object from CLI output, handling various formats."""
    # Try direct parse
    try:
        data = json.loads(output)
        if isinstance(data, list):
            # Claude returns a list of messages — find the assistant message with the trace
            for msg in data:
                if msg.get("type") == "result":
                    text = msg.get("result", "")
                    return _parse_trace_json(text)
                if msg.get("type") == "assistant":
                    content = msg.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            return _parse_trace_json(block.get("text", ""))
            raise ValueError(f"No trace JSON found in {model} output")
        if isinstance(data, dict) and "trace_id" in data:
            return data
        raise ValueError(f"Unexpected JSON structure from {model}")
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in raw text
    return _parse_trace_json(output)


def _parse_trace_json(text: str) -> dict:
    """Extract a JSON object from text that may contain markdown fencing or extra content."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in text: {text[:200]}")

    # Find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {e}") from e

    raise ValueError(f"Unbalanced braces in JSON: {text[:200]}")


def _normalize_raw_trace(raw: dict) -> dict:
    """Fix common LLM output quirks before Pydantic validation."""
    import re

    # Fix turn_id / parent_turn_id: "t01" -> 1, "t12" -> 12
    for turn in raw.get("turns", []):
        for key in ("turn_id", "parent_turn_id"):
            val = turn.get(key)
            if isinstance(val, str):
                nums = re.findall(r"\d+", val)
                turn[key] = int(nums[0]) if nums else None
            if key == "parent_turn_id" and turn.get(key) in (0, "null", "none", ""):
                turn[key] = None

    # Fix arrival_pattern variants
    arrival_map = {
        "bursty": "poisson_bursty",
        "poisson": "poisson_bursty",
        "interactive": "steady_interactive",
        "steady": "steady_interactive",
        "reactivation": "reactivation_heavy",
    }
    replay = raw.get("replay", {})
    ap = replay.get("arrival_pattern", "")
    if ap in arrival_map:
        replay["arrival_pattern"] = arrival_map[ap]

    # Fix KV tier variants
    tier_map = {
        "gpu_hbm": "HBM",
        "gpu": "HBM",
        "hbm": "HBM",
        "cpu_dram": "CPU",
        "cpu": "CPU",
        "dram": "CPU",
        "nvme": "NVMe",
        "ssd": "NVMe",
        "s3": "remote",
    }
    kv = raw.get("kv_pressure", {})
    tier = kv.get("tier", "")
    if tier.lower() in tier_map:
        kv["tier"] = tier_map[tier.lower()]

    # Fix cache_expectation variants
    ce_map = {"hot": "warm", "cold_start": "cold", "resumed": "reactivated"}
    ce = kv.get("cache_expectation", "")
    if ce in ce_map:
        kv["cache_expectation"] = ce_map[ce]

    # Ensure tokenizer is an object
    if "tokenizer" not in raw or not isinstance(raw["tokenizer"], dict):
        raw["tokenizer"] = {"name": "cl100k_base", "version": "1.0"}

    # Ensure dataset_version
    if "dataset_version" not in raw:
        raw["dataset_version"] = "0.1.0"

    # Ensure source_type
    if "source_type" not in raw:
        raw["source_type"] = "synthetic"

    # Ensure language
    if "language" not in raw:
        raw["language"] = "en"

    return raw


def generate_trace(plan: TracePlan) -> SessionTrace:
    """Generate a canonical trace from a plan using the assigned model."""
    if plan.generator_model == "codex":
        raw = generate_with_codex(plan)
    else:
        raw = generate_with_claude(plan)

    # Normalize common LLM output quirks
    raw = _normalize_raw_trace(raw)

    # Override provenance with actual model used
    if "provenance" not in raw:
        raw["provenance"] = {}
    raw["provenance"]["generator_model"] = plan.generator_model
    raw["provenance"]["generator_prompt_version"] = "v0.1.0"
    raw["provenance"]["pipeline_version"] = "0.1.0"

    # Ensure trace_id matches plan
    raw["trace_id"] = plan.trace_id

    return SessionTrace.model_validate(raw)
