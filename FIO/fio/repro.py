"""Reproducibility helpers: one seed for every RNG, and a record of each run.

    from fio.repro import set_seed, run_info, write_json
    seed = set_seed()                      # SEED env var, default 1234
    write_json(out / "run_config.json", {"seed": seed, **my_settings, "env": run_info()})
"""

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SEED = 1234
FIO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FIO_ROOT.parent

# Packages whose version changes results or on-disk formats.
_TRACKED_PACKAGES = ("torch", "transformers", "sentence-transformers", "chromadb",
                     "numpy", "tokenizers", "accelerate", "openai")
_TRACKED_MODELS = ("Qwen/Qwen2.5-7B-Instruct", "sentence-transformers/all-MiniLM-L6-v2")


def get_seed() -> int:
    return int(os.environ.get("SEED", DEFAULT_SEED))


def set_seed(seed: Optional[int] = None) -> int:
    """Seed Python, NumPy and torch (CPU + CUDA). Returns the seed used."""
    seed = get_seed() if seed is None else int(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def hf_snapshot(repo_id: str) -> Optional[str]:
    """Commit hash of the locally cached Hugging Face snapshot for repo_id, if any."""
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    ref = hub / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text().strip()
    except OSError:
        return None


def run_info() -> Dict[str, Any]:
    """Where and with what this run executed: git state, packages, models, host."""
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "packages": versions,
        "hf_snapshots": {m: hf_snapshot(m) for m in _TRACKED_MODELS},
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


def write_json(path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
