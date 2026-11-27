"""Portable configuration, provenance, and synchronized experiment timing."""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def read_config(name: str) -> dict:
    """Return a bundled JSON experiment configuration by filename."""
    return json.loads((CONFIG_DIR / name).read_text())


def write_json(path: Path, value: object) -> None:
    """Atomically save a JSON-serializable result, rejecting nonfinite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    """Seed CPU/CUDA generators and preserve the paper's FP32 math settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def token_fingerprint(batches: list[dict]) -> str:
    """Hash ordered token IDs and shifted labels independently of batch size."""
    ids = torch.cat([b["input_ids"] for b in batches]).cpu().contiguous()
    labels = torch.cat([b["labels"] for b in batches]).cpu().contiguous()
    return hashlib.sha256(ids.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()


def environment(device: torch.device) -> dict:
    """Return hardware and dependency version metadata."""
    versions = {}
    for package in ("torch", "transformers", "datasets", "numpy", "scipy", "accelerate", "hqq", "pymoo", "pySOT"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(), "versions": versions}


class Timer:
    """Measure named stages with CUDA synchronization and an explicit selection scope."""

    def __init__(self, device: torch.device):
        """Start a timer on device after caller-controlled setup and imports."""
        self.device = device
        self.seconds = {}
        self.selection_stages = []
        self.started = self.now()

    def now(self) -> float:
        """Return a timestamp after queued work on this CUDA device completes."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @contextmanager
    def stage(self, name: str, *, selection: bool = True):
        """Time a stage, optionally including it in the selection total."""
        start = self.now()
        yield
        self.seconds[name] = self.seconds.get(name, 0.0) + self.now() - start
        if selection and name not in self.selection_stages:
            self.selection_stages.append(name)

    def report(self) -> dict:
        """Return stage times and totals; cache flags belong in the experiment metadata."""
        return {"stages": dict(self.seconds), "selection_stages": list(self.selection_stages),
                "selection_seconds": sum(self.seconds[name] for name in self.selection_stages),
                "wall_seconds": self.now() - self.started}


def prepare_output(path: str) -> Path:
    """Create an empty output directory without replacing a prior experiment."""
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; use a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    return output
