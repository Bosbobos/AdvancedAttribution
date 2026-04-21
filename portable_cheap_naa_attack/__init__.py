"""Portable cheap-IG based NAA attack package."""

from .attack_core import AttackRunResult, run_attack
from .attribution_methods import (
    AttributionBenchmarkConfig,
    AttributionComparisonResult,
    AttributionMethodResult,
    benchmark_classifier_attribution_runtimes,
    plot_classifier_runtime_barplot,
    plot_visual_examples_table,
    run_classifier_attribution_comparison,
)
from .comparison import plot_attack_comparison, plot_importance_map_comparison
from .config import AttackConfig
from .naa_reference import run_naa_attack

__all__ = [
    "AttackConfig",
    "AttackRunResult",
    "AttributionBenchmarkConfig",
    "AttributionComparisonResult",
    "AttributionMethodResult",
    "benchmark_classifier_attribution_runtimes",
    "plot_classifier_runtime_barplot",
    "plot_visual_examples_table",
    "run_classifier_attribution_comparison",
    "run_attack",
    "run_naa_attack",
    "plot_attack_comparison",
    "plot_importance_map_comparison",
]
