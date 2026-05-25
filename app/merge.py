from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import MergeOptions
from .model_io import load_state_dict, save_state_dict, sha256_file, validate_model_path


ProgressCallback = Callable[[str], None]


EXCLUDED_COMPONENT_MARKERS = (
    "clip",
    "text_encoder",
    "conditioner",
    "vae",
    "first_stage_model",
)


KEY_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.model.",
    "model.",
    "module.",
    "state_dict.",
    "net.",
)


@dataclass
class MergeReport:
    output_path: Path
    total_tensors: int = 0
    merged_tensors: int = 0
    skipped_tensors: int = 0
    auto_corrected_tensors: int = 0
    warnings: list[str] = field(default_factory=list)


def is_merge_target(name: str) -> bool:
    lowered = name.lower()
    return not any(marker in lowered for marker in EXCLUDED_COMPONENT_MARKERS)


def block_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("input", "down", "in_blocks", "input_blocks")):
        return "input"
    if any(token in lowered for token in ("middle", "mid_block", "mid.", "middle_block")):
        return "middle"
    if any(token in lowered for token in ("output", "up", "out_blocks", "output_blocks")):
        return "output"
    match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", lowered)
    if match:
        index = int(match.group(1))
        if index <= 8:
            return "input"
        if index <= 18:
            return "middle"
        return "output"
    return "other"


def component_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("attn", "attention", "to_q", "to_k", "to_v", "to_out", "self_attn", "cross_attn")):
        return "attention"
    if any(token in lowered for token in ("mlp", "ff", "feed_forward", "ffn", "proj_in", "proj_out")):
        return "mlp"
    if any(token in lowered for token in ("norm", "ln_", "layernorm", "groupnorm")):
        return "norm"
    if any(token in lowered for token in ("resnet", "resblock", "resnets", "skip_connection")):
        return "resnet"
    if any(token in lowered for token in ("time_embed", "timestep", "temb", "time_embedding")):
        return "timestep"
    return "other"


def transformer_group(name: str) -> str:
    lowered = name.lower()
    patterns = (
        r"(single_transformer_blocks\.\d+)",
        r"(transformer_blocks\.\d+)",
        r"(input_blocks\.\d+)",
        r"(output_blocks\.\d+)",
        r"(in_blocks\.\d+)",
        r"(out_blocks\.\d+)",
        r"(down_blocks\.\d+)",
        r"(up_blocks\.\d+)",
        r"(blocks\.\d+)",
        r"(layers\.\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1)
    if "middle_block" in lowered or "mid_block" in lowered:
        return "middle_block"
    return block_category(name)


def adjustment_group(name: str, mode: str) -> str:
    block = {
        "input": "Input",
        "middle": "Middle",
        "output": "Output",
        "other": "Other",
    }[block_category(name)]
    component = {
        "attention": "Attention",
        "mlp": "MLP",
        "norm": "Norm",
        "resnet": "ResNet",
        "timestep": "Timestep",
        "other": "Other",
    }[component_category(name)]
    normalized_mode = mode.lower()
    if normalized_mode == "matrix":
        return f"{block}_{component}"
    if normalized_mode == "transformer":
        return transformer_group(name)
    if normalized_mode == "component":
        if component == "Attention":
            return f"{transformer_group(name)}_Attention"
        return component
    return f"{block}_{component}"


def canonical_key(name: str) -> str:
    lowered = name
    changed = True
    while changed:
        changed = False
        for prefix in KEY_PREFIXES:
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :]
                changed = True
    return lowered


def canonical_state_map(state_dict: dict[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in state_dict:
        normalized = canonical_key(key)
        mapping.setdefault(normalized, key)
    return mapping


def canonical_lora_key(name: str) -> str:
    normalized = canonical_key(name)
    normalized = re.sub(r"^diffusion_model\.", "", normalized)
    normalized = re.sub(r"^lora_(?:unet|te|te1|te2)_", "", normalized)
    normalized = normalized.replace(".processor.", ".")
    normalized = normalized.replace(".lora_A.", ".lora_down.")
    normalized = normalized.replace(".lora_B.", ".lora_up.")
    normalized = normalized.replace(".lora_down.default.", ".lora_down.")
    normalized = normalized.replace(".lora_up.default.", ".lora_up.")
    normalized = normalized.replace(".lora_down.weight", ".lora_down.weight")
    normalized = normalized.replace(".lora_up.weight", ".lora_up.weight")
    return normalized.lower()


def canonical_lora_state_map(state_dict: dict[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in state_dict:
        mapping.setdefault(canonical_lora_key(key), key)
    return mapping


def anima_v1_key(name: str) -> str:
    normalized = canonical_key(name)
    if normalized.startswith("net."):
        return normalized
    return f"net.{normalized}"


def output_key_name(name: str, options: MergeOptions) -> str:
    if options.output_key_format == "anima-base-v1.0":
        return anima_v1_key(name)
    return name


def output_lora_key_name(key: str, options: MergeOptions) -> str:
    """LoRAキーを output_key_format 形式に正規化する。

    anima-base-v1.0 形式:
      canonical_lora_key でプレフィックス・サフィックスを正規化し、
      lora_unet_blocks_N_xxx.{lora_up|lora_down|alpha} 形式に統一する。
      net_ などの不正なプレフィックスも除去する。
    """
    import re as _re
    if options.output_key_format != "anima-base-v1.0":
        return key
    suffix = ""
    for s in (
        ".lora_up.weight", ".lora_down.weight",
        ".lora_A.weight", ".lora_B.weight",
        ".lora_up.default.weight", ".lora_down.default.weight",
        ".alpha",
    ):
        if key.endswith(s):
            suffix = s
            break
    ck = canonical_lora_key(key)
    ck = _re.sub(r"\.(lora_up|lora_down|lora_A|lora_B)(\.default)?\.weight$", "", ck)
    ck = _re.sub(r"\.alpha$", "", ck)
    ck = _re.sub(r"^net[._]", "", ck)
    root = f"lora_unet_{ck.replace('.', '_')}"
    return f"{root}{suffix}"



def lora_base_name(name: str) -> str:
    base = name
    for suffix in (
        ".lora_up.default.weight",
        ".lora_down.default.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".alpha",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for prefix in ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    return base


def lora_target_candidates(name: str) -> list[str]:
    base = lora_base_name(name)
    base = re.sub(r"\.(?:processor\.)?lora_(?:up|down|A|B)$", "", base)
    match = re.match(r"blocks_(\d+)_(.+)$", base)
    if match:
        block = f"blocks.{match.group(1)}"
        tail = match.group(2)
    else:
        block = ""
        tail = base

    replacements = (
        ("cross_attn_output_proj", "cross_attn.output_proj"),
        ("cross_attn_k_proj", "cross_attn.k_proj"),
        ("cross_attn_q_proj", "cross_attn.q_proj"),
        ("cross_attn_v_proj", "cross_attn.v_proj"),
        ("self_attn_output_proj", "self_attn.output_proj"),
        ("self_attn_k_proj", "self_attn.k_proj"),
        ("self_attn_q_proj", "self_attn.q_proj"),
        ("self_attn_v_proj", "self_attn.v_proj"),
        ("mlp_layer1", "mlp.layer1"),
        ("mlp_layer2", "mlp.layer2"),
    )
    converted_tail = tail
    for source, target in replacements:
        if tail == source:
            converted_tail = target
            break

    converted = f"{block}.{converted_tail}.weight" if block else f"{converted_tail}.weight"
    fallback = f"{base.replace('_', '.')}.weight"
    dotted = base.replace("_", ".")
    candidates = [
        f"net.{converted}",
        converted,
        fallback,
        f"model.diffusion_model.{converted}",
        f"diffusion_model.{converted}",
        f"{dotted}.weight",
    ]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def lora_alpha_scale(lora: dict[str, object], up_key: str, rank: int) -> float:
    alpha_candidates = (
        up_key.replace(".lora_up.weight", ".alpha"),
        up_key.replace(".lora_B.weight", ".alpha"),
        up_key.replace(".lora_up.default.weight", ".alpha"),
        up_key.replace(".lora_B.default.weight", ".alpha"),
    )
    alpha = next((lora.get(key) for key in alpha_candidates if key in lora), None)
    if alpha is None or not hasattr(alpha, "detach") or rank <= 0:
        return 1.0
    try:
        return float(alpha.detach().float().reshape(-1)[0]) / float(rank)
    except Exception:
        return 1.0


def lora_down_key_for(up_key: str) -> str:
    if "lora_up" in up_key:
        return up_key.replace("lora_up", "lora_down")
    if "lora_B" in up_key:
        return up_key.replace("lora_B", "lora_A")
    return up_key


def layer_alpha(name: str, options: MergeOptions) -> float:
    if name in options.layer_overrides:
        return max(0.0, min(1.0, options.alpha * options.layer_overrides[name]))
    group = adjustment_group(name, options.layer_display_mode)
    scale = options.parameter_scales.get(group)
    if scale is not None:
        return max(0.0, min(1.0, options.alpha * scale))
    block = block_category(name)
    component = component_category(name)
    block_scale = {
        "input": options.alpha_input,
        "middle": options.alpha_middle,
        "output": options.alpha_output,
        "other": options.alpha_other,
    }[block]
    component_scale = {
        "attention": options.alpha_attention,
        "mlp": options.alpha_mlp,
        "norm": options.alpha_norm,
        "resnet": options.alpha_resnet,
        "timestep": options.alpha_timestep,
        "other": 1.0,
    }[component]
    return max(0.0, min(1.0, options.alpha * block_scale * component_scale))


def should_freeze_bias(name: str, options: MergeOptions) -> bool:
    lowered = name.lower()
    if not (lowered.endswith(".bias") or ".bias." in lowered or lowered.endswith("bias")):
        return False
    category = block_category(name)
    return (
        (category == "input" and options.freeze_bias_input)
        or (category == "middle" and options.freeze_bias_middle)
        or (category == "output" and options.freeze_bias_output)
    )


def cosine_similarity(torch: object, a: object, b: object) -> float:
    av = a.detach().float().flatten()
    bv = b.detach().float().flatten()
    denom = torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv)
    if float(denom) == 0.0:
        return 1.0
    return float(torch.dot(av, bv) / denom)


def corrected_alpha(torch: object, name: str, a: object, b: object, options: MergeOptions) -> tuple[float, bool]:
    alpha = layer_alpha(name, options)
    if not options.auto_correction:
        return alpha, False
    similarity = cosine_similarity(torch, a, b)
    if similarity >= options.cosine_threshold:
        return alpha, False
    scale = max(0.0, similarity) / max(options.cosine_threshold, 1e-6)
    return alpha * scale, True


def validate_compatible(base: dict[str, object], other: dict[str, object], other_map: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for key, base_tensor in base.items():
        other_key = other_map.get(canonical_key(key))
        if other_key is None:
            warnings.append(f"Missing in secondary model: {key}")
            continue
        other_tensor = other[other_key]
        if getattr(base_tensor, "shape", None) != getattr(other_tensor, "shape", None):
            warnings.append(f"Shape mismatch: {key} <-> {other_key}")
    return warnings


def dry_run_check(torch: object, state_dict: dict[str, object]) -> None:
    checked = 0
    for key, tensor in state_dict.items():
        if checked >= 32:
            break
        if not hasattr(tensor, "detach"):
            continue
        sample = tensor.detach().float()
        if sample.numel() > 4096:
            sample = sample.flatten()[:4096]
        if not bool(torch.isfinite(sample).all()):
            raise ValueError(f"Dry-run failed: non-finite tensor detected at {key}")
        checked += 1


def merge_loras(
    base_lora_path: Path,
    secondary_lora_path: Path,
    output_path: Path,
    options: MergeOptions,
    device: str = "cpu",
    progress: ProgressCallback | None = None,
) -> MergeReport:
    from .model_io import require_torch

    torch = require_torch()
    log = progress or (lambda _message: None)
    if device.startswith("cuda") and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
        log("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    validate_model_path(base_lora_path)
    validate_model_path(secondary_lora_path)
    log(f"Loading base LoRA: {base_lora_path.name}")
    base = load_state_dict(base_lora_path, device)
    log(f"Loading secondary LoRA: {secondary_lora_path.name}")
    other = load_state_dict(secondary_lora_path, device)
    other_map = canonical_lora_state_map(other)

    report = MergeReport(output_path=output_path)
    remapped_count = sum(1 for key in base if key not in other and canonical_lora_key(key) in other_map)
    if remapped_count:
        log(f"LoRA key remap enabled: {remapped_count} tensor key(s)")

    # .alpha はスキップ対象のため実マージ対象数を事前カウント
    _lora_merge_targets = [
        key for key, base_tensor in base.items()
        if not key.endswith(".alpha")
        and hasattr(base_tensor, "is_floating_point")
        and base_tensor.is_floating_point()
    ]
    _total_lora_merge = len(_lora_merge_targets)
    log(f"LoRA merge target layers: {_total_lora_merge} / total keys: {len(base)}")

    merged: dict[str, object] = {}
    _lora_merge_index = 0
    _lora_key_corrected_count = 0
    for key, base_tensor in base.items():
        report.total_tensors += 1
        out_key = output_lora_key_name(key, options)
        if out_key != key:
            _lora_key_corrected_count += 1
        other_key = key if key in other else other_map.get(canonical_lora_key(key))
        other_tensor = other.get(other_key) if other_key is not None else None
        target_name = lora_target_candidates(key)[0]
        if (
            other_tensor is None
            or should_freeze_bias(target_name, options)
            or not is_merge_target(target_name)
            or getattr(base_tensor, "shape", None) != getattr(other_tensor, "shape", None)
            or not hasattr(base_tensor, "detach")
            or not hasattr(base_tensor, "is_floating_point")
            or not base_tensor.is_floating_point()
            or not hasattr(other_tensor, "is_floating_point")
            or not other_tensor.is_floating_point()
        ):
            merged[out_key] = base_tensor.detach().to("cpu") if hasattr(base_tensor, "detach") else base_tensor
            report.skipped_tensors += 1
            continue

        _lora_merge_index += 1
        alpha, corrected = corrected_alpha(torch, target_name, base_tensor, other_tensor, options)
        base_d = base_tensor.detach().to(device)
        other_d = other_tensor.detach().to(device)
        merged[out_key] = (base_d * (1.0 - alpha) + other_d * alpha).to(dtype=base_d.dtype).cpu()
        report.merged_tensors += 1
        if corrected:
            report.auto_corrected_tensors += 1
        if _lora_merge_index % 100 == 0:
            log(f"Merged LoRA tensors: {_lora_merge_index}/{_total_lora_merge}")
    if _lora_key_corrected_count:
        log(f"Key normalization applied (anima-base-v1.0): {_lora_key_corrected_count} key(s) renamed")

    base_canonical_keys = {canonical_lora_key(base_key) for base_key in base}
    extra_count = sum(1 for key in other if canonical_lora_key(key) not in base_canonical_keys)
    if extra_count:
        report.warnings.append(f"Secondary-only LoRA tensors skipped: {extra_count}")
    if report.merged_tensors == 0:
        raise ValueError(
            "No compatible LoRA tensors were merged. "
            "Check that both LoRAs use the same target architecture and tensor shapes."
        )

    if options.dry_run:
        log("Running dry-run tensor validation")
        dry_run_check(torch, merged)

    metadata = {
        "anima_model_editor": "2.0-tab1",
        "merge_type": "lora_to_lora",
        "base_lora_sha256": sha256_file(base_lora_path),
        "secondary_lora_sha256": sha256_file(secondary_lora_path),
        "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving merged LoRA: {output_path}")
    save_state_dict(output_path, merged, metadata)
    del base, other, merged
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return report


def extract_lora_difference(
    base_path: Path,
    target_path: Path,
    output_path: Path,
    options: MergeOptions,
    rank: int = 16,
    device: str = "cpu",
    progress: ProgressCallback | None = None,
) -> MergeReport:
    from .model_io import require_torch

    torch = require_torch()
    log = progress or (lambda _message: None)
    if device.startswith("cuda") and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
        log("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    validate_model_path(base_path)
    validate_model_path(target_path)
    rank = max(1, int(rank))
    log(f"Loading base model: {base_path.name}")
    base = load_state_dict(base_path, device)
    log(f"Loading target model: {target_path.name}")
    target = load_state_dict(target_path, device)
    target_map = canonical_state_map(target)

    report = MergeReport(output_path=output_path)
    extracted: dict[str, object] = {}
    _extract_key_corrected_count = 0
    # 実抽出対象数を事前カウント（2D浮動小数点テンソルのみ）
    _extract_targets = [
        key for key, t in base.items()
        if is_merge_target(key)
        and not should_freeze_bias(key, options)
        and hasattr(t, "detach")
        and hasattr(t, "is_floating_point")
        and t.is_floating_point()
        and len(tuple(t.shape)) == 2
    ]
    _extract_total = len(_extract_targets)
    log(f"LoRA extraction candidate layers (2D): {_extract_total} / total keys: {len(base)}")
    _extract_index = 0
    for index, (key, base_tensor) in enumerate(base.items(), start=1):
        target_key = key if key in target else target_map.get(canonical_key(key))
        target_tensor = target.get(target_key) if target_key is not None else None
        report.total_tensors += 1
        if (
            target_tensor is None
            or should_freeze_bias(key, options)
            or not is_merge_target(key)
            or getattr(base_tensor, "shape", None) != getattr(target_tensor, "shape", None)
            or not hasattr(base_tensor, "detach")
            or not hasattr(target_tensor, "detach")
            or not hasattr(base_tensor, "is_floating_point")
            or not base_tensor.is_floating_point()
            or len(tuple(base_tensor.shape)) != 2
        ):
            report.skipped_tensors += 1
            continue

        scale = layer_alpha(key, options)
        if scale <= 0.0:
            report.skipped_tensors += 1
            continue
        delta = (target_tensor.detach().float() - base_tensor.detach().float()) * scale
        if not bool(torch.any(delta)):
            report.skipped_tensors += 1
            continue
        effective_rank = min(rank, int(delta.shape[0]), int(delta.shape[1]))
        try:
            u, s, vh = torch.linalg.svd(delta.to(device), full_matrices=False)
        except Exception as exc:
            report.warnings.append(f"LoRA extraction SVD failed: {key} ({exc})")
            report.skipped_tensors += 1
            continue

        # canonical_key でプレフィックス除去し lora_unet_ 形式に統一
        # output_lora_key_name と同一ロジックで余分なプレフィックスを排除する
        _base_layer = canonical_key(key).removesuffix(".weight")
        root = f"lora_unet_{_base_layer.replace('.', '_')}"
        sqrt_s = torch.sqrt(s[:effective_rank].clamp_min(0.0))
        up = (u[:, :effective_rank] * sqrt_s.unsqueeze(0)).to("cpu")
        down = (sqrt_s.unsqueeze(1) * vh[:effective_rank, :]).to("cpu")
        _raw_up_key = f"{root}.lora_up.weight"
        _raw_down_key = f"{root}.lora_down.weight"
        _raw_alpha_key = f"{root}.alpha"
        _norm_up_key = output_lora_key_name(_raw_up_key, options)
        _norm_down_key = output_lora_key_name(_raw_down_key, options)
        _norm_alpha_key = output_lora_key_name(_raw_alpha_key, options)
        if _norm_up_key != _raw_up_key:
            _extract_key_corrected_count += 1
        extracted[_norm_up_key] = up.to(dtype=base_tensor.dtype)
        extracted[_norm_down_key] = down.to(dtype=base_tensor.dtype)
        extracted[_norm_alpha_key] = torch.tensor(float(effective_rank))
        report.merged_tensors += 1
        _extract_index += 1
        if _extract_index % 100 == 0:
            log(f"Extracted LoRA layers: {_extract_index}/{_extract_total}")
    if _extract_key_corrected_count:
        log(f"Key normalization applied (anima-base-v1.0): {_extract_key_corrected_count} LoRA key group(s) renamed")

    if report.merged_tensors == 0:
        raise ValueError(
            "No compatible 2D model-difference tensors were extracted. "
            "Check that both models use the same architecture and tensor shapes."
        )

    if options.dry_run:
        log("Running dry-run tensor validation")
        dry_run_check(torch, extracted)

    metadata = {
        "anima_model_editor": "2.0-tab1",
        "merge_type": "model_difference_to_lora",
        "base_sha256": sha256_file(base_path),
        "target_sha256": sha256_file(target_path),
        "rank": str(rank),
        "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving extracted LoRA: {output_path}")
    save_state_dict(output_path, extracted, metadata)
    del base, target, extracted
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return report


def merge_models(
    base_path: Path,
    secondary_path: Path,
    output_path: Path,
    options: MergeOptions,
    device: str = "cpu",
    progress: ProgressCallback | None = None,
) -> MergeReport:
    from .model_io import require_torch

    torch = require_torch()
    log = progress or (lambda _message: None)
    if device.startswith("cuda") and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
        log("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    validate_model_path(base_path)
    validate_model_path(secondary_path)
    log(f"Loading base model: {base_path.name}")
    base = load_state_dict(base_path, device)
    log(f"Loading secondary model: {secondary_path.name}")
    other = load_state_dict(secondary_path, device)
    other_map = canonical_state_map(other)

    report = MergeReport(output_path=output_path)
    compatibility_warnings = validate_compatible(base, other, other_map)
    report.warnings.extend(compatibility_warnings[:100])
    remapped_count = sum(1 for key in base if key not in other and canonical_key(key) in other_map)
    if remapped_count:
        log(f"Key prefix remap enabled: {remapped_count} tensor key(s)")

    # 実マージ対象数を事前カウント（ログ分母を実態に合わせる）
    _merge_targets = [
        key for key, base_tensor in base.items()
        if is_merge_target(key)
        and not should_freeze_bias(key, options)
        and hasattr(base_tensor, "detach")
        and hasattr(base_tensor, "is_floating_point")
        and base_tensor.is_floating_point()
    ]
    _total_merge = len(_merge_targets)
    log(f"Merge target layers: {_total_merge} / total keys: {len(base)}")

    merged: dict[str, object] = {}
    _merge_index = 0
    _key_corrected_count = 0
    for key, base_tensor in base.items():
        report.total_tensors += 1
        out_key = output_key_name(key, options)
        if out_key != key:
            _key_corrected_count += 1
        other_key = key if key in other else other_map.get(canonical_key(key))
        other_tensor = other.get(other_key) if other_key is not None else None
        if (
            other_tensor is None
            or not is_merge_target(key)
            or should_freeze_bias(key, options)
            or getattr(base_tensor, "shape", None) != getattr(other_tensor, "shape", None)
            or not hasattr(base_tensor, "detach")
            or not hasattr(base_tensor, "is_floating_point")
            or not base_tensor.is_floating_point()
            or not hasattr(other_tensor, "is_floating_point")
            or not other_tensor.is_floating_point()
        ):
            merged[out_key] = base_tensor.detach().to("cpu") if hasattr(base_tensor, "detach") else base_tensor
            report.skipped_tensors += 1
            continue

        _merge_index += 1
        alpha, corrected = corrected_alpha(torch, key, base_tensor, other_tensor, options)
        base_d = base_tensor.detach().to(device)
        other_d = other_tensor.detach().to(device)
        merged_tensor = base_d * (1.0 - alpha) + other_d * alpha
        merged[out_key] = merged_tensor.to(dtype=base_d.dtype).cpu()
        report.merged_tensors += 1
        if corrected:
            report.auto_corrected_tensors += 1
        if _merge_index % 100 == 0:
            log(f"Merged tensors: {_merge_index}/{_total_merge}")
    if _key_corrected_count:
        log(f"Key normalization applied (anima-base-v1.0): {_key_corrected_count} key(s) renamed")

    if report.merged_tensors == 0:
        raise ValueError(
            "No compatible merge-target tensors were merged. "
            "Check that both models use the same architecture and tensor shapes."
        )

    if options.dry_run:
        log("Running dry-run tensor validation")
        dry_run_check(torch, merged)

    metadata = {
        "anima_model_editor": "2.0-tab1",
        "merge_type": "model_to_model",
        "base_sha256": sha256_file(base_path),
        "secondary_sha256": sha256_file(secondary_path),
        "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving merged model: {output_path}")
    save_state_dict(output_path, merged, metadata)
    del base, other, merged
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return report


def fuse_lora_into_model(
    base_path: Path,
    lora_path: Path,
    output_path: Path,
    options: MergeOptions,
    device: str = "cpu",
    progress: ProgressCallback | None = None,
) -> MergeReport:
    from .model_io import require_torch

    torch = require_torch()
    log = progress or (lambda _message: None)
    if device.startswith("cuda") and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
        log("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    validate_model_path(base_path)
    validate_model_path(lora_path)
    log(f"Loading base model: {base_path.name}")
    base = load_state_dict(base_path, device)
    log(f"Loading LoRA: {lora_path.name}")
    lora = load_state_dict(lora_path, device)

    report = MergeReport(output_path=output_path)
    _fuse_key_corrected = sum(
        1 for key in base if output_key_name(key, options) != key
    )
    merged = {
        output_key_name(key, options): value.detach().to("cpu") if hasattr(value, "detach") else value
        for key, value in base.items()
    }
    if _fuse_key_corrected:
        log(f"Key normalization applied to base model (anima-base-v1.0): {_fuse_key_corrected} key(s) renamed")
    base_lookup = {
        canonical_key(key): output_key_name(key, options)
        for key in base
    }
    used: set[str] = set()
    _fuse_index = 0
    _fuse_total = sum(
        1 for key in lora
        if ("lora_up" in key or "lora_B" in key) and hasattr(lora[key], "detach")
    )
    log(f"LoRA fuse target pairs: {_fuse_total}")

    for key, up in lora.items():
        if key in used or not ("lora_up" in key or "lora_B" in key) or not hasattr(up, "detach"):
            continue
        down_key = lora_down_key_for(key)
        down = lora.get(down_key)
        if down is None or not hasattr(down, "detach"):
            continue

        candidates = lora_target_candidates(key)
        base_key = candidates[0]
        target_key = None
        for candidate in candidates:
            target_key = base_lookup.get(canonical_key(candidate))
            if target_key is not None:
                break
        if target_key is None:
            target_key = output_key_name(base_key, options)
        target = merged.get(target_key)
        if target is None or not hasattr(target, "detach"):
            report.warnings.append(f"Target not found for LoRA pair: {key} -> {base_key}")
            continue

        try:
            rank = int(down.shape[0])
            delta = torch.mm(up.detach().float(), down.detach().float()) * lora_alpha_scale(lora, key, rank)
            delta = delta.reshape(target.shape).to("cpu")
        except Exception as exc:
            report.warnings.append(f"LoRA shape mismatch: {key} ({exc})")
            continue

        if (
            should_freeze_bias(base_key, options)
            or not is_merge_target(base_key)
            or not hasattr(target, "is_floating_point")
            or not target.is_floating_point()
        ):
            report.skipped_tensors += 1
            continue
        merged[target_key] = (target.detach().to("cpu") + delta * layer_alpha(target_key, options)).to(dtype=target.dtype)
        used.add(key)
        used.add(down_key)
        report.total_tensors += 1
        report.merged_tensors += 1
        _fuse_index += 1
        if _fuse_index % 100 == 0:
            log(f"Fused LoRA pairs: {_fuse_index}/{_fuse_total}")

    if report.merged_tensors == 0:
        raise ValueError(
            "No compatible LoRA tensors were fused. "
            "Check that the LoRA targets the selected model architecture."
        )

    if options.dry_run:
        log("Running dry-run tensor validation")
        dry_run_check(torch, merged)

    metadata = {
        "anima_model_editor": "2.0-tab1",
        "merge_type": "lora_to_model",
        "base_sha256": sha256_file(base_path),
        "lora_sha256": sha256_file(lora_path),
        "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving fused model: {output_path}")
    save_state_dict(output_path, merged, metadata)
    del base, lora, merged
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return report
