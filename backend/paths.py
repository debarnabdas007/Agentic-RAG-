"""Repository root resolution (avoid cwd-dependent bugs in pipeline scripts)"""
from pathlib import Path


def project_root() -> Path:
    """Root of the repo (parent of the `backend` package directory)"""
    return Path(__file__).resolve().parent.parent
