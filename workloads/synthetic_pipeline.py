"""Main orchestrator — Planner → Generator → Evaluator → Materialize.

Runs the full 24-trace v0 pipeline with dual-model generation (Claude + Codex)
and cross-model evaluation. Resumable — skips already-generated traces.

Usage:
    cd products/isb1
    python -m workloads.synthetic_pipeline plan       # Review the 24-trace matrix
    python -m workloads.synthetic_pipeline generate    # Generate all traces (resumable)
    python -m workloads.synthetic_pipeline validate    # Validate all traces
    python -m workloads.synthetic_pipeline evaluate    # Cross-model evaluation
    python -m workloads.synthetic_pipeline materialize # Build 4 replay manifests
    python -m workloads.synthetic_pipeline all         # Full pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from workloads.synthetic_planner import TracePlan, build_v0_plan_matrix, save_plan_matrix
from workloads.synthetic_generator import generate_trace
from workloads.synthetic_evaluator import (
    cross_evaluate,
    run_all_validations,
    EvalResult,
    ValidationResult,
)
from workloads.materialize_session import build_v0_manifests
from workloads.session_trace import SessionTrace


OUTPUT_DIR = Path("workloads/synthetic_v0")
TRACES_DIR = OUTPUT_DIR / "traces"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"
PLAN_FILE = OUTPUT_DIR / "plan_matrix.json"
EVAL_FILE = OUTPUT_DIR / "evaluations.jsonl"


def ensure_dirs() -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_trace_ids() -> set[str]:
    """Find already-generated trace IDs for resume support."""
    existing = set()
    if TRACES_DIR.exists():
        for f in TRACES_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                existing.add(data.get("trace_id", ""))
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


def load_traces() -> list[SessionTrace]:
    """Load all generated traces."""
    traces = []
    if TRACES_DIR.exists():
        for f in sorted(TRACES_DIR.glob("*.json")):
            try:
                traces.append(SessionTrace.model_validate_json(f.read_text()))
            except Exception as e:
                print(f"  WARNING: Failed to load {f.name}: {e}")
    return traces


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def cmd_plan() -> None:
    """Stage 1: Generate and display the 24-trace planning matrix."""
    ensure_dirs()
    plans = build_v0_plan_matrix()
    save_plan_matrix(plans, str(PLAN_FILE))

    print(f"Plan matrix: {len(plans)} traces")
    print(f"Saved to: {PLAN_FILE}")
    print()

    claude_count = sum(1 for p in plans if p.generator_model == "claude")
    codex_count = sum(1 for p in plans if p.generator_model == "codex")
    print(f"  Claude: {claude_count} traces")
    print(f"  Codex:  {codex_count} traces")
    print()

    for p in plans:
        print(f"  {p.trace_id}: {p.family}/{p.target_bin} [{p.generator_model}] — {p.goal[:60]}")


def cmd_generate() -> None:
    """Stage 2: Generate traces using Claude + Codex (resumable)."""
    ensure_dirs()
    plans = build_v0_plan_matrix()
    existing = load_existing_trace_ids()

    # Filter out already generated
    remaining = [p for p in plans if p.trace_id not in existing]
    if len(remaining) < len(plans):
        print(f"Resuming: {len(plans) - len(remaining)} already done, {len(remaining)} remaining")

    if not remaining:
        print("All traces already generated.")
        return

    print(f"Generating {len(remaining)} traces...\n")

    completed = 0
    failed = 0

    for i, plan in enumerate(remaining):
        print(f"[{i + 1}/{len(remaining)}] {plan.trace_id} ({plan.generator_model})...", end=" ", flush=True)
        try:
            trace = generate_trace(plan)

            # Save trace
            trace_file = TRACES_DIR / f"{plan.trace_id}.json"
            trace_file.write_text(trace.model_dump_json(indent=2))

            print(
                f"OK | {trace.turn_count()} turns | "
                f"{trace.total_tokens():,} tokens | "
                f"{trace.tool_turn_count()} tool turns"
            )
            completed += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {completed} generated, {failed} failed")
    print(f"Total traces: {len(load_existing_trace_ids())}/{len(plans)}")


def cmd_validate() -> None:
    """Stage 3: Run automated validation on all traces."""
    plans = build_v0_plan_matrix()
    plan_map = {p.trace_id: p for p in plans}
    traces = load_traces()

    if not traces:
        print("No traces found. Run 'generate' first.")
        return

    print(f"Validating {len(traces)} traces...\n")

    all_passed = 0
    all_warned = 0
    all_failed = 0

    for trace in traces:
        plan = plan_map.get(trace.trace_id)
        if not plan:
            print(f"  {trace.trace_id}: SKIP (no matching plan)")
            continue

        results = run_all_validations(trace, plan)
        errors = [r for r in results if not r.passed]
        warnings = [w for r in results for w in r.warnings]

        if errors:
            print(f"  {trace.trace_id}: FAIL")
            for r in errors:
                for e in r.errors:
                    print(f"    [{r.stage}] {e}")
            all_failed += 1
        elif warnings:
            print(f"  {trace.trace_id}: WARN")
            for w in warnings:
                print(f"    {w}")
            all_warned += 1
        else:
            print(f"  {trace.trace_id}: PASS")
            all_passed += 1

    print(f"\nResults: {all_passed} passed, {all_warned} warnings, {all_failed} failed")


def cmd_evaluate() -> None:
    """Stage 4: Cross-model LLM evaluation."""
    plans = build_v0_plan_matrix()
    plan_map = {p.trace_id: p for p in plans}
    traces = load_traces()

    if not traces:
        print("No traces found. Run 'generate' first.")
        return

    print(f"Cross-evaluating {len(traces)} traces...\n")

    with open(EVAL_FILE, "w") as f:
        passed = 0
        failed = 0

        for trace in traces:
            plan = plan_map.get(trace.trace_id)
            if not plan:
                continue

            evaluator = "codex" if plan.generator_model == "claude" else "claude"
            print(f"  {trace.trace_id} (gen={plan.generator_model}, eval={evaluator})...", end=" ", flush=True)

            try:
                result = cross_evaluate(trace, plan)
                print(f"{result.decision} | scores={result.scores}")

                f.write(json.dumps({
                    "trace_id": trace.trace_id,
                    "decision": result.decision,
                    "scores": result.scores,
                    "reasons": result.reasons,
                    "required_fixes": result.required_fixes,
                    "evaluator_model": result.evaluator_model,
                }) + "\n")

                if result.decision == "PASS":
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"ERROR: {e}")
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    print(f"Evaluations: {EVAL_FILE}")


def cmd_materialize() -> None:
    """Stage 5: Build 4 replay manifests from traces."""
    ensure_dirs()
    traces = load_traces()

    if not traces:
        print("No traces found. Run 'generate' first.")
        return

    print(f"Materializing manifests from {len(traces)} traces...\n")

    manifests = build_v0_manifests(traces)

    for manifest in manifests:
        path = MANIFESTS_DIR / f"{manifest.name}.json"
        path.write_text(manifest.model_dump_json(indent=2))
        print(f"  {manifest.name}: {manifest.total_events()} events")

    print(f"\nManifests saved to: {MANIFESTS_DIR}")


def cmd_all() -> None:
    """Run the full pipeline: plan → generate → validate → evaluate → materialize."""
    print("=" * 60)
    print("STAGE 1: PLAN")
    print("=" * 60)
    cmd_plan()

    print("\n" + "=" * 60)
    print("STAGE 2: GENERATE (Claude + Codex)")
    print("=" * 60)
    cmd_generate()

    print("\n" + "=" * 60)
    print("STAGE 3: VALIDATE")
    print("=" * 60)
    cmd_validate()

    print("\n" + "=" * 60)
    print("STAGE 4: CROSS-MODEL EVALUATE")
    print("=" * 60)
    cmd_evaluate()

    print("\n" + "=" * 60)
    print("STAGE 5: MATERIALIZE MANIFESTS")
    print("=" * 60)
    cmd_materialize()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    traces = load_traces()
    total_tokens = sum(t.total_tokens() for t in traces)
    print(f"  Traces: {len(traces)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Manifests: {len(list(MANIFESTS_DIR.glob('*.json')))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic dataset pipeline: 24 canonical traces + 4 replay manifests"
    )
    parser.add_argument(
        "command",
        choices=["plan", "generate", "validate", "evaluate", "materialize", "all"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    commands = {
        "plan": cmd_plan,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "evaluate": cmd_evaluate,
        "materialize": cmd_materialize,
        "all": cmd_all,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
