from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from .config import SUPPORTED_MODEL_EXTENSIONS


class DependencyError(RuntimeError):
    pass


def validate_model_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_EXTENSIONS))
        raise ValueError(f"Unsupported extension: {path.suffix}. Allowed: {allowed}")


def scan_models(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise DependencyError("PyTorch is required to load .ckpt/.bin files and merge tensors.")
    import torch

    return torch


def require_safetensors() -> tuple[Any, Any]:
    if importlib.util.find_spec("safetensors") is None:
        raise DependencyError("safetensors is required to load or save .safetensors files.")
    from safetensors.torch import load_file, save_file

    return load_file, save_file


def normalize_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return obj
    raise ValueError("Loaded object is not a state dict.")


def load_state_dict(path: Path, device: str) -> dict[str, Any]:
    validate_model_path(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        load_file, _ = require_safetensors()
        return dict(load_file(str(path), device=device))

    torch = require_torch()
    try:
        loaded = torch.load(str(path), map_location=device)
    except Exception:
        loaded = torch.load(str(path), map_location=device, weights_only=False)
    return normalize_state_dict(loaded)


def list_state_dict_layers(path: Path, device: str = "cpu") -> list[tuple[str, str]]:
    validate_model_path(path)
    if path.suffix.lower() == ".safetensors":
        if importlib.util.find_spec("safetensors") is None:
            raise DependencyError("safetensors is required to inspect .safetensors files.")
        from safetensors import safe_open

        rows: list[tuple[str, str]] = []
        with safe_open(str(path), framework="pt", device=device) as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                rows.append((key, "x".join(str(part) for part in shape)))
        return rows

    state_dict = load_state_dict(path, device)
    rows = []
    for key, value in state_dict.items():
        shape = getattr(value, "shape", None)
        shape_text = "x".join(str(part) for part in shape) if shape is not None else "-"
        rows.append((key, shape_text))
    return rows


def save_state_dict(path: Path, state_dict: dict[str, Any], metadata: dict[str, str] | None = None) -> None:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        _, save_file = require_safetensors()
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in state_dict.items()
            if hasattr(value, "detach")
        }
        save_file(tensors, str(path), metadata=metadata or {})
        return

    torch = require_torch()
    payload = {"state_dict": state_dict, "metadata": metadata or {}}
    torch.save(payload, str(path))
