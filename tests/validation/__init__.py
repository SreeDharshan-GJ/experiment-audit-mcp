"""Scientific Validation suite for the Experiment Audit reasoning engine.

This package does not modify, wrap, or monkeypatch any implementation
module. Every test here calls the frozen `ScientificRule` subclasses
and `ScientificReasoningPipeline` exactly as a real caller would,
against realistic ML-experiment scenarios (W&B-style runs, claims a
researcher might actually assert about them), and checks the resulting
`RuleResult` / `ScientificReport` against a scientifically justified
expected outcome.
"""
