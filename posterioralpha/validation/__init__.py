"""Stage 4 — validation: performance metrics and plotting dashboards."""
from posterioralpha.validation.metrics import compute_metrics
from posterioralpha.validation.plots import (
    plot_lambda_heatmap,
    plot_multi_seed,
    plot_single_run,
)

__all__ = [
    "compute_metrics",
    "plot_single_run",
    "plot_multi_seed",
    "plot_lambda_heatmap",
]
