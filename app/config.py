from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".bin"}


@dataclass(frozen=True)
class AppPaths:
    root: Path
    app: Path
    checkpoints: Path
    lora: Path
    log_analysis: Path
    configs: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "AppPaths":
        base = (root or Path(__file__).resolve().parents[1]).resolve()
        return cls(
            root=base,
            app=base / "app",
            checkpoints=base / "checkpoints",
            lora=base / "lora",
            log_analysis=base / "log" / "log_analysis",
            configs=base / "configs",
        )

    def ensure(self) -> None:
        for directory in (
            self.app,
            self.checkpoints,
            self.lora,
            self.log_analysis,
            self.configs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class MergeOptions:
    alpha: float = 0.5
    alpha_input: float = 1.0
    alpha_middle: float = 1.0
    alpha_output: float = 1.0
    alpha_attention: float = 1.0
    alpha_mlp: float = 1.0
    alpha_norm: float = 1.0
    alpha_resnet: float = 1.0
    alpha_timestep: float = 1.0
    alpha_other: float = 1.0
    layer_display_mode: str = "Matrix"
    parameter_scales: dict[str, float] = field(default_factory=dict)
    layer_overrides: dict[str, float] = field(default_factory=dict)
    cosine_threshold: float = 0.4
    auto_correction: bool = True
    freeze_bias_input: bool = False
    freeze_bias_middle: bool = False
    freeze_bias_output: bool = False
    dry_run: bool = True
    output_name: str = "merged_model.safetensors"
    output_key_format: str = "anima-base-v1.0"
