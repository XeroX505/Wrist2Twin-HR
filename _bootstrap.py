"""Keep legacy import names stable without installing workstation code."""
import sys
from pathlib import Path


def activate():
    path = str(Path(__file__).resolve().parent / "legacy")
    if path not in sys.path:
        sys.path.insert(0, path)
