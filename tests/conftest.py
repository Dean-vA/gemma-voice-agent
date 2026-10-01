"""Make the repo root importable so tests can `import app` under plain `pytest`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
