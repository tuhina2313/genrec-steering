"""Hashes, seeds, atomic writes, manifests, cache validation, environment capture."""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PrerequisiteMissing(RuntimeError):
    """A stage cannot run because a named artifact or resource is unavailable."""

    def __init__(self, missing: str, detail: str = ""):
        self.missing = missing
        self.detail = detail
        super().__init__(f"missing prerequisite: {missing}" + (f" ({detail})" if detail else ""))


class NotEstimable(RuntimeError):
    """The planned protocol cannot be estimated on the realized data."""


class WorkCapExceeded(RuntimeError):
    """A configured forward-pass cap was reached; the stage is partial."""


# ---------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(obj: Any) -> str:
    """Hex sha256 of a canonical JSON encoding."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def short_hash(obj: Any, n: int = 16) -> str:
    return stable_hash(obj)[:n]


def hash_uint(*parts: Any) -> int:
    """Deterministic 64-bit integer from parts; independent of PYTHONHASHSEED."""
    s = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def hash_unit(*parts: Any) -> float:
    return hash_uint(*parts) / float(1 << 64)


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return sha256_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


# ---------------------------------------------------------------- json helpers

def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas scalars; non-finite floats become None (never NaN in JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------- atomic writes

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_bytes(path: str | Path, data: bytes) -> None:
    _atomic_write_bytes(Path(path), data)


def write_text(path: str | Path, text: str) -> None:
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    text = json.dumps(to_jsonable(obj), indent=2, sort_keys=False, allow_nan=False)
    write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(path: str | Path, df) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _atomic_write_bytes(Path(path), buf.getvalue())


def write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    _atomic_write_bytes(Path(path), buf.getvalue())


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    lines = [json.dumps(to_jsonable(r), sort_keys=True, allow_nan=False) for r in records]
    write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_yaml(path: str | Path, obj: Any) -> None:
    import yaml

    write_text(path, yaml.safe_dump(to_jsonable(obj), sort_keys=False))


# ---------------------------------------------------------------- seeds & config

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def utc_iso(ts: float | int) -> str:
    return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).isoformat()


def now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------- uniqueness / joins

def assert_unique(df, cols: list[str], what: str) -> None:
    dup = df.duplicated(subset=cols, keep=False)
    if dup.any():
        raise AssertionError(f"{what}: {int(dup.sum())} rows with duplicate key {cols}")


def safe_merge(left, right, on: list[str], how: str = "inner", what: str = "join", validate: str = "one_to_one"):
    """Join only after asserting key uniqueness; never by row position."""
    if validate in ("one_to_one", "many_to_one"):
        assert_unique(right, on, f"{what} (right)")
    if validate in ("one_to_one", "one_to_many"):
        assert_unique(left, on, f"{what} (left)")
    return left.merge(right, on=on, how=how, validate=validate)


# ---------------------------------------------------------------- environment

def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def git_commit(root: Path = PROJECT_ROOT) -> dict:
    def run(args):
        try:
            return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(run(["git", "status", "--porcelain"]))}


def capture_environment(extra: dict | None = None) -> dict:
    env: dict[str, Any] = {
        "captured_at": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p: _pkg_version(p)
            for p in ["torch", "transformers", "tokenizers", "numpy", "pandas", "pyarrow", "scikit-learn",
                      "matplotlib", "pyyaml", "huggingface-hub", "safetensors", "accelerate"]
        },
        "source": git_commit(),
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        env["cuda_available"] = False
    if extra:
        env.update(extra)
    return env
