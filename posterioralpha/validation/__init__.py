"""Stage 4 — validation: performance metrics, robustness checks, dashboards."""
from posterioralpha.validation.metrics import compute_metrics
from posterioralpha.validation.plots import (
    plot_lambda_heatmap,
    plot_multi_seed,
    plot_single_run,
)
from posterioralpha.validation.robustness import (
    dsr_report,
    rebalance_offset_dispersion,
)

__all__ = [
    "compute_metrics",
    "plot_single_run",
    "plot_multi_seed",
    "plot_lambda_heatmap",
    "rebalance_offset_dispersion",
    "dsr_report",
]
