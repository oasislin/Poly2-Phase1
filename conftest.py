"""Root conftest: make project root importable for pytest.

All modules use `src.`-prefixed imports (e.g. `from src.data_acquisition...`),
so the project root must be on sys.path. This also makes plain `pytest` work
(no implicit dependency on `python -m pytest` adding the CWD).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
