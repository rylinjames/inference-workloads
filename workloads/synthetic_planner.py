"""Planner stage — produces scenario blueprints for the 24-trace v0 matrix.

The planner does NOT generate traces. It produces structured plans that the
generator consumes. This separation ensures intentional coverage of stress surfaces.

Reference: Production-Grade Synthetic Dataset Design Spec v0.1.0, Section 4 + 8.2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class TracePlan:
    """Blueprint for one canonical trace."""

    trace_id: str
    family: str  # "long_chat" | "coding"
    goal: str
    target_bin: str  # e.g., "8k_16k", "32k_64k", "96k_128k", "16k_32k", "64k_96k", "128k_plus"
    num_turns: int
    prefix_style: str  # e.g., "conversational", "rag_retrieval", "large_repo_prefix"
    tool_pattern: list[str]  # e.g., ["read", "grep", "bash", "edit", "execution"]
    branch_pattern: str  # e.g., "none", "one_failed_fix_then_retry", "two_alternatives"
    idle_windows_ms: list[int]
    expected_prefix_overlap_ratio: float
    cache_expectation: str  # "cold" | "warm" | "reactivated"
    contains_rag: bool = False
    generator_model: str = ""  # "claude" | "codex" — assigned by the runner

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "family": self.family,
            "goal": self.goal,
            "target_bin": self.target_bin,
            "num_turns": self.num_turns,
            "prefix_style": self.prefix_style,
            "tool_pattern": self.tool_pattern,
            "branch_pattern": self.branch_pattern,
            "idle_windows_ms": self.idle_windows_ms,
            "expected_prefix_overlap_ratio": self.expected_prefix_overlap_ratio,
            "cache_expectation": self.cache_expectation,
            "contains_rag": self.contains_rag,
            "generator_model": self.generator_model,
        }


def build_v0_plan_matrix() -> list[TracePlan]:
    """Build the 24-trace v0 planning matrix.

    A. Long-context multi-turn chat / reasoning — 12 traces
        LC1: 4 traces at 8k-16k (ordinary multi-turn)
        LC2: 4 traces at 32k-64k (long-context follow-up)
        LC3: 4 traces at 96k-128k (deep context, near-offload)
        Of these 12, 4 should be retrieval-heavy (RAG-like).

    B. Coding / agent / tool traces — 12 traces
        CA1: 4 traces at 16k-32k (repo review, simple tools)
        CA2: 4 traces at 64k-96k (larger repo, retries)
        CA3: 4 traces at 128k+ (largest prefixes, offload-relevant)
        Of these 12, 4 should include retrieved docs/logs/codebase context.
    """
    plans: list[TracePlan] = []

    # --- A. Long-context chat/reasoning (12 traces) ---

    # LC1: 8k-16k — ordinary multi-turn reasoning
    plans.append(TracePlan(
        trace_id=f"lc1_{uuid4().hex[:8]}",
        family="long_chat", goal="Debug a subtle Python type error through multi-turn reasoning",
        target_bin="8k_16k", num_turns=12, prefix_style="conversational",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 2000, 5000, 3000],
        expected_prefix_overlap_ratio=0.85, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"lc1_{uuid4().hex[:8]}",
        family="long_chat", goal="Plan a database migration strategy with tradeoff analysis",
        target_bin="8k_16k", num_turns=14, prefix_style="conversational",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 1000, 8000, 2000],
        expected_prefix_overlap_ratio=0.82, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"lc1_{uuid4().hex[:8]}",
        family="long_chat", goal="Analyze a research paper with iterative clarification questions",
        target_bin="8k_16k", num_turns=10, prefix_style="rag_retrieval",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 3000, 6000, 4000],
        expected_prefix_overlap_ratio=0.78, cache_expectation="warm",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"lc1_{uuid4().hex[:8]}",
        family="long_chat", goal="Design a REST API with evolving requirements across turns",
        target_bin="8k_16k", num_turns=11, prefix_style="conversational",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 2000, 4000, 1000],
        expected_prefix_overlap_ratio=0.88, cache_expectation="warm",
    ))

    # LC2: 32k-64k — long-context follow-up chains
    plans.append(TracePlan(
        trace_id=f"lc2_{uuid4().hex[:8]}",
        family="long_chat", goal="Extended code review of a 2000-line PR with multi-file discussion",
        target_bin="32k_64k", num_turns=20, prefix_style="large_repo_prefix",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 5000, 15000, 8000, 30000],
        expected_prefix_overlap_ratio=0.90, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"lc2_{uuid4().hex[:8]}",
        family="long_chat", goal="Troubleshoot production incident with log analysis across multiple turns",
        target_bin="32k_64k", num_turns=18, prefix_style="rag_retrieval",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 3000, 10000, 60000, 5000],
        expected_prefix_overlap_ratio=0.85, cache_expectation="reactivated",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"lc2_{uuid4().hex[:8]}",
        family="long_chat", goal="Write a technical RFC with iterative feedback and revisions",
        target_bin="32k_64k", num_turns=22, prefix_style="conversational",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 4000, 8000, 120000, 3000],
        expected_prefix_overlap_ratio=0.88, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"lc2_{uuid4().hex[:8]}",
        family="long_chat", goal="Deep-dive into Kubernetes networking with reference docs and follow-ups",
        target_bin="32k_64k", num_turns=16, prefix_style="rag_retrieval",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 6000, 12000, 5000],
        expected_prefix_overlap_ratio=0.82, cache_expectation="warm",
        contains_rag=True,
    ))

    # LC3: 96k-128k — deep context, memory-heavy, near-offload
    plans.append(TracePlan(
        trace_id=f"lc3_{uuid4().hex[:8]}",
        family="long_chat", goal="Analyze and compare three competing system architectures with full specs",
        target_bin="96k_128k", num_turns=28, prefix_style="large_repo_prefix",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 8000, 20000, 180000, 10000, 5000],
        expected_prefix_overlap_ratio=0.92, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"lc3_{uuid4().hex[:8]}",
        family="long_chat", goal="Multi-session legal document review with clause-by-clause analysis",
        target_bin="96k_128k", num_turns=30, prefix_style="rag_retrieval",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 5000, 15000, 300000, 8000],
        expected_prefix_overlap_ratio=0.94, cache_expectation="reactivated",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"lc3_{uuid4().hex[:8]}",
        family="long_chat", goal="Extended debugging session across a large codebase with stack traces",
        target_bin="96k_128k", num_turns=25, prefix_style="large_repo_prefix",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 10000, 30000, 60000, 15000],
        expected_prefix_overlap_ratio=0.91, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"lc3_{uuid4().hex[:8]}",
        family="long_chat", goal="Full-day pair programming session with context accumulation",
        target_bin="96k_128k", num_turns=32, prefix_style="conversational",
        tool_pattern=[], branch_pattern="none",
        idle_windows_ms=[0, 3000, 8000, 600000, 5000, 120000],
        expected_prefix_overlap_ratio=0.95, cache_expectation="reactivated",
    ))

    # --- B. Coding / agent / tool traces (12 traces) ---

    # CA1: 16k-32k — repo review, simple tools
    plans.append(TracePlan(
        trace_id=f"ca1_{uuid4().hex[:8]}",
        family="coding", goal="Fix a failing unit test in a Flask app",
        target_bin="16k_32k", num_turns=12, prefix_style="large_repo_prefix",
        tool_pattern=["read", "grep", "edit", "bash"],
        branch_pattern="none",
        idle_windows_ms=[0, 2000, 4000, 3000],
        expected_prefix_overlap_ratio=0.70, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"ca1_{uuid4().hex[:8]}",
        family="coding", goal="Add pagination to a Django REST endpoint",
        target_bin="16k_32k", num_turns=14, prefix_style="large_repo_prefix",
        tool_pattern=["read", "edit", "bash"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 3000, 5000, 2000],
        expected_prefix_overlap_ratio=0.68, cache_expectation="warm",
    ))
    plans.append(TracePlan(
        trace_id=f"ca1_{uuid4().hex[:8]}",
        family="coding", goal="Review and fix security vulnerabilities flagged by Bandit",
        target_bin="16k_32k", num_turns=10, prefix_style="large_repo_prefix",
        tool_pattern=["read", "grep", "edit"],
        branch_pattern="none",
        idle_windows_ms=[0, 2000, 6000, 4000],
        expected_prefix_overlap_ratio=0.72, cache_expectation="warm",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"ca1_{uuid4().hex[:8]}",
        family="coding", goal="Refactor a monolithic function into smaller units with tests",
        target_bin="16k_32k", num_turns=16, prefix_style="large_repo_prefix",
        tool_pattern=["read", "edit", "bash", "execution"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 1000, 3000, 8000],
        expected_prefix_overlap_ratio=0.75, cache_expectation="warm",
    ))

    # CA2: 64k-96k — larger repo, tests, logs, retries
    plans.append(TracePlan(
        trace_id=f"ca2_{uuid4().hex[:8]}",
        family="coding", goal="Debug flaky CI test in a Django-like repo with log analysis",
        target_bin="64k_96k", num_turns=22, prefix_style="large_repo_prefix",
        tool_pattern=["read", "grep", "bash", "edit", "execution"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 4000, 15000, 120000, 5000],
        expected_prefix_overlap_ratio=0.72, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"ca2_{uuid4().hex[:8]}",
        family="coding", goal="Implement a new API endpoint with database migration and tests",
        target_bin="64k_96k", num_turns=20, prefix_style="large_repo_prefix",
        tool_pattern=["read", "edit", "bash", "execution"],
        branch_pattern="two_alternatives",
        idle_windows_ms=[0, 3000, 8000, 60000, 4000],
        expected_prefix_overlap_ratio=0.68, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"ca2_{uuid4().hex[:8]}",
        family="coding", goal="Investigate and fix a memory leak using profiling output",
        target_bin="64k_96k", num_turns=18, prefix_style="large_repo_prefix",
        tool_pattern=["read", "bash", "grep", "edit"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 5000, 20000, 10000],
        expected_prefix_overlap_ratio=0.75, cache_expectation="warm",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"ca2_{uuid4().hex[:8]}",
        family="coding", goal="Port a Python 2 codebase to Python 3 with incremental test verification",
        target_bin="64k_96k", num_turns=24, prefix_style="large_repo_prefix",
        tool_pattern=["read", "grep", "edit", "bash", "execution"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 2000, 10000, 5000, 30000],
        expected_prefix_overlap_ratio=0.80, cache_expectation="warm",
        contains_rag=True,
    ))

    # CA3: 128k+ — largest prefixes, long logs/diffs, offload-relevant
    plans.append(TracePlan(
        trace_id=f"ca3_{uuid4().hex[:8]}",
        family="coding", goal="Full repo audit with security review, performance profiling, and fixes",
        target_bin="128k_plus", num_turns=30, prefix_style="large_repo_prefix",
        tool_pattern=["read", "grep", "bash", "edit", "execution"],
        branch_pattern="two_alternatives",
        idle_windows_ms=[0, 5000, 30000, 300000, 10000, 60000],
        expected_prefix_overlap_ratio=0.65, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"ca3_{uuid4().hex[:8]}",
        family="coding", goal="Migrate a microservice from REST to gRPC with full test coverage",
        target_bin="128k_plus", num_turns=28, prefix_style="large_repo_prefix",
        tool_pattern=["read", "edit", "bash", "execution", "grep"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 8000, 20000, 180000, 15000],
        expected_prefix_overlap_ratio=0.70, cache_expectation="reactivated",
    ))
    plans.append(TracePlan(
        trace_id=f"ca3_{uuid4().hex[:8]}",
        family="coding", goal="Build a complete CLI tool from scratch with tests, CI, and docs",
        target_bin="128k_plus", num_turns=35, prefix_style="large_repo_prefix",
        tool_pattern=["read", "edit", "bash", "execution"],
        branch_pattern="two_alternatives",
        idle_windows_ms=[0, 3000, 15000, 600000, 8000, 120000],
        expected_prefix_overlap_ratio=0.60, cache_expectation="reactivated",
        contains_rag=True,
    ))
    plans.append(TracePlan(
        trace_id=f"ca3_{uuid4().hex[:8]}",
        family="coding", goal="Debug and optimize a data pipeline processing 1M+ records with OOM errors",
        target_bin="128k_plus", num_turns=26, prefix_style="large_repo_prefix",
        tool_pattern=["read", "bash", "grep", "edit", "execution"],
        branch_pattern="one_failed_fix_then_retry",
        idle_windows_ms=[0, 10000, 45000, 240000, 20000],
        expected_prefix_overlap_ratio=0.72, cache_expectation="reactivated",
    ))

    # Assign models: alternate Claude and Codex
    for i, plan in enumerate(plans):
        plan.generator_model = "claude" if i % 2 == 0 else "codex"

    return plans


def save_plan_matrix(plans: list[TracePlan], output_path: str) -> None:
    """Save the plan matrix as JSON for review before generation."""
    import json
    with open(output_path, "w") as f:
        json.dump([p.to_dict() for p in plans], f, indent=2)
