"""
PosteriorAlpha — a Bayesian adaptive portfolio research framework.

The package is organised as a four-stage research pipeline. Each stage is a
subpackage with a single responsibility, so a strategy can be traced cleanly
from idea to validated result:

    1. data        ── posterioralpha.data
       Market-data access. Live download (yfinance / S&P 500 universe),
       synthetic factor-model universe expansion, and bundled-dataset loaders.

    2. research    ── posterioralpha.research
       Strategy definitions: the models and signals. Bayesian shrinkage
       machinery + optimisers (bayesian), AMR optimisers + λ calibration +
       vol-targeting overlay (amr), and regime detectors (regimes: RegimeHMM,
       BOCPD, HMM3). Pure, time-step-agnostic building blocks.

    3. backtest    ── posterioralpha.backtest
       Time-stepping engines that walk research primitives through history
       with no lookahead: bayesian (monthly) and amr (weekly), each with a
       result container and an "all strategies" runner.

    4. validation  ── posterioralpha.validation
       Turning return streams into evidence: performance/risk metrics and
       the plotting dashboards.

Reproducible experiments that wire these stages together live in the top-level
``experiments/`` directory.
"""

__version__ = "1.0.0"

__all__ = ["data", "research", "backtest", "validation"]
