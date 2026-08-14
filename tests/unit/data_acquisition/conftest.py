"""Make src/data_acquisition importable for the gefs_fetcher unit tests."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src" / "data_acquisition"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
