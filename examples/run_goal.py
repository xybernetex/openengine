"""
examples/run_goal.py — OpenEngine SDK usage examples.

Prerequisites
-------------
  pip install -r requirements.txt
  cp .env.example .env          # fill in LLM_PROVIDER + credentials
  # Place base_policy.pt in models/ (see models/README.md)

Run
---
  python examples/run_goal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk import Engine, GoalResult


def example_research_and_email() -> None:
    """
    Example 1 — Pure research goal.
    The agent plans, searches the web, writes a report, and emails it.
    """
    print("=" * 60)
    print("Example 1: Research and email")
    print("=" * 60)

    engine = Engine()
    result: GoalResult = engine.run(
        "Research the top 5 lithium battery suppliers for EVs, "
        "summarise key risks and costs, and email the report to me@example.com"
    )

    print(f"\nStatus  : {result.status}")
    print(f"Success : {result.success}")
    print(f"Steps   : {result.steps}")
    print(f"Reward  : {result.reward:.1f}")

    print(f"\nAspects ({len(result.aspects)}):")
    for aspect in result.aspects:
        marker = "✓" if aspect.complete else "○"
        print(f"  {marker}  {aspect.name}")

    print(f"\nArtifacts ({len(result.artifacts)}):")
    for art in result.artifacts:
        print(f"  [{art.type}] {art.title}")

    doc = result.document
    if doc:
        print(f"\nDocument preview:\n{doc[:400]}")


def example_code_generation() -> None:
    """
    Example 2 — Code generation goal.
    The agent writes Python, executes it, and returns the output.
    """
    print("\n" + "=" * 60)
    print("Example 2: Code generation")
    print("=" * 60)

    engine = Engine()
    result: GoalResult = engine.run(
        "Write a Python script that generates a multiplication table "
        "for numbers 1-10 and prints it as a formatted grid."
    )

    print(f"\nStatus : {result.status}")
    print(f"Steps  : {result.steps}")

    code = result.code
    if code:
        print(f"\nGenerated script ({len(code)} chars):")
        print("-" * 40)
        print(code[:600])
        print("-" * 40)


def example_workflow_mode() -> None:
    """
    Example 3 — Workflow mode.
    Caller pre-defines the aspects; PLAN is skipped entirely.
    The agent executes exactly the structure you specify.
    """
    print("\n" + "=" * 60)
    print("Example 3: Workflow mode (pre-defined aspects)")
    print("=" * 60)

    engine = Engine()
    result: GoalResult = engine.run(
        goal=(
            "Research current solar panel manufacturing costs by country "
            "and email a comparison report to analyst@example.com"
        ),
        aspects=[
            "solar_manufacturing_cost_research",
            "country_cost_comparison",
            "email_transmission_confirmation",
        ],
    )

    print(f"\nStatus : {result.status}")
    print(f"Steps  : {result.steps}")
    print(f"Reward : {result.reward:.1f}")

    print(f"\nAspects:")
    for aspect in result.aspects:
        marker = "✓" if aspect.complete else "○"
        print(f"  {marker}  {aspect.name}")

    narrative = result.narrative
    if narrative:
        print(f"\nResearch summary preview:\n{narrative[:400]}")


def example_programmatic_result() -> None:
    """
    Example 4 — Working with results in code.
    Shows how to extract and use artifacts programmatically.
    """
    print("\n" + "=" * 60)
    print("Example 4: Programmatic result handling")
    print("=" * 60)

    engine = Engine()
    result: GoalResult = engine.run(
        "Write a Python function that calculates compound interest "
        "given principal, rate, and years."
    )

    # Access typed artifacts
    code_artifact = result.first("CODE")
    if code_artifact:
        print(f"\nCode artifact: {code_artifact.title}")
        # Execute the generated code in an isolated namespace
        namespace: dict = {}
        try:
            exec(compile(code_artifact.content, "<agent-code>", "exec"), namespace)
            print("Code executed successfully.")
            # If it defined a function, call it
            if "compound_interest" in namespace:
                value = namespace["compound_interest"](1000, 0.05, 10)
                print(f"compound_interest(1000, 0.05, 10) = {value:.2f}")
        except Exception as exc:
            print(f"Execution error: {exc}")

    # Inspect all aspects
    complete = [a for a in result.aspects if a.complete]
    pending  = [a for a in result.aspects if not a.complete]
    print(f"\nComplete aspects : {[a.name for a in complete]}")
    if pending:
        print(f"Incomplete aspects: {[a.name for a in pending]}")

    # The raw dict is always available as an escape hatch
    print(f"\nRaw keys: {list(result.raw.keys())}")


if __name__ == "__main__":
    # Run all examples — comment out any you don't want to execute.
    # Note: examples 1 and 3 send real emails if send_email tool is configured.
    example_code_generation()
    # example_research_and_email()
    # example_workflow_mode()
    # example_programmatic_result()
