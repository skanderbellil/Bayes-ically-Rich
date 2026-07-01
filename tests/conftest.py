"""Shared test configuration: force a headless matplotlib backend and expose
the repository root so tests never depend on the CWD or a display server."""
import os
from pathlib import Path

# Must be set before matplotlib is imported anywhere in the package.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
