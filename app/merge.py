from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import MergeOptions
from .model_io import (
    DependencyError,
    load_state_dict,
    read_model_metadata,
    save_state_dict,
    sha256_file,
    validate_model_path,
)


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


# ──────────────────────────────────────────────────────────────────────
# Anima DiT ブロック構成・バージョン検出(本体マージのAnima 3.8B対応)
#
# AnimaBase v1.0(28block)とAnima 3.8B(52block、v1.0/v1.1でblock構成は
# 共通)の対応関係は、実チェックポイントdiff実測により「block 0-27は
# indexそのまま一致・block 28-51はBase側に対応物なしの単純追加」と
# 判明済み(LLaMA-Pro方式の交互挿入ではない)。このため既存の汎用マージ
# ロジック(canonical_key一致・shape一致、不一致/欠落キーはBase側の値を
# 保持)は、追加のblock remap実装なしでこの2アーキテクチャ間のマージ
# 要件を満たす。ここでは(1)Input/Middle/Output区分をAnima実構成に正しく
# 合わせるための検出、(2)未検証のblock数の組み合わせでのマージ実行を
# 防ぐための検証、(3)Base側metadataの保持・マージ履歴の追記、を提供する。
# ──────────────────────────────────────────────────────────────────────

_ANIMA_BLOCK_KEY_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
_ANIMA_CONNECTOR_V2_KEY_MARKER = "anima_v2_connector."
_ANIMA_CONNECTOR_V2_METADATA_KEY = "anima_v2_adapter_architecture"
_ANIMA_CONNECTOR_V2_METADATA_VALUE = "anima_qwen35_quality_anchored_semantic_connector_v2"
_KNOWN_ANIMA_DIT_BLOCK_COUNTS = (28, 52)

# 実測・確認済みのAnimaモデルバリアントの組み合わせのみマージを許可する。
# それ以外(未検証のblock数の組み合わせ)はUnverifiedModelPairErrorで停止する。
_VERIFIED_ANIMA_VARIANT_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("anima-base-v1.0", "anima-base-v1.0"),
        ("anima-base-v1.0", "anima-3.8b-v1.0"),
        ("anima-base-v1.0", "anima-3.8b-v1.1"),
        ("anima-3.8b-v1.0", "anima-3.8b-v1.0"),
        ("anima-3.8b-v1.1", "anima-3.8b-v1.1"),
        ("anima-3.8b-v1.0", "anima-3.8b-v1.1"),
    )
)


class UnverifiedModelPairError(ValueError):
    """未検証のAnimaモデルバリアントの組み合わせでマージが要求された場合に送出する。"""


def count_dit_blocks_from_keys(keys: Iterable[str]) -> Optional[int]:
    """テンソルキー群から`blocks.N.`パターンの最大indexを検出し、ブロック総数を返す。

    テンソル本体には触れず、キー名文字列のみを走査する。state_dictのキー、
    safetensorsヘッダのキー一覧など、文字列を列挙できるものであれば
    入力形式を問わない。

    Args:
        keys: 判定対象のテンソルキー文字列を列挙するイテラブル。

    Returns:
        検出したブロック総数(最大index+1)。`blocks.N.`パターンが
        1件も無ければNone。
    """
    max_index: Optional[int] = None
    for key in keys:
        match = _ANIMA_BLOCK_KEY_PATTERN.search(canonical_key(key))
        if match:
            index = int(match.group(1))
            max_index = index if max_index is None else max(max_index, index)
    return None if max_index is None else max_index + 1


def detect_connector_v2_from_keys_and_metadata(
    keys: Iterable[str], metadata: dict[str, str] | None
) -> bool:
    """テンソルキー群とmetadataから、Anima 3.8B v1.1(Semantic Connector v2内蔵)か判定する。

    判定優先順位: (1) metadataの`anima_v2_adapter_architecture`、
    (2) テンソルキー名前空間(`anima_v2_connector.`の有無)。
    lora_train.pyの`_read_safetensors_is_semantic_connector_v2`と
    同一ロジック(モジュール間でimportし合わない既存構成のため、
    ここでも独立実装として複製している)。

    Args:
        keys: 判定対象のテンソルキー文字列を列挙するイテラブル。
        metadata: safetensorsのmetadata辞書。無ければNone。

    Returns:
        Semantic Connector v2が内蔵されていると判定できればTrue。
    """
    keys = list(keys)
    if metadata:
        architecture = metadata.get(_ANIMA_CONNECTOR_V2_METADATA_KEY)
        if architecture is not None:
            return architecture == _ANIMA_CONNECTOR_V2_METADATA_VALUE
    return any(_ANIMA_CONNECTOR_V2_KEY_MARKER in canonical_key(key) for key in keys)


def classify_anima_model_variant(num_blocks: Optional[int], is_connector_v2: bool) -> str:
    """検出済みのブロック総数とconnector v2有無から、Animaモデルのバリアントを分類する。

    Args:
        num_blocks: count_dit_blocks_from_keys()で検出したブロック総数。
        is_connector_v2: detect_connector_v2_from_keys_and_metadata()の判定結果。

    Returns:
        "anima-base-v1.0"(28block) / "anima-3.8b-v1.1"(52block、connector
        内蔵) / "anima-3.8b-v1.0"(52block、connector無し。外付け
        Progressive Cross Adapter想定だが、当該外部ファイルは本体マージの
        対象外) / "unknown"(28/52以外、またはblocks.N.パターン非検出で
        Anima系と判定できない)のいずれか。
    """
    if num_blocks == 28:
        return "anima-base-v1.0"
    if num_blocks == 52:
        return "anima-3.8b-v1.1" if is_connector_v2 else "anima-3.8b-v1.0"
    return "unknown"


def verify_anima_variant_pair_if_applicable(base_variant: str, secondary_variant: str) -> None:
    """Base/Secondaryの組み合わせが検証済みか確認し、未検証なら例外で処理を停止する。

    両方が"unknown"(=どちらもAnima系と判定できない、Anima以外の汎用モデルの
    マージと推定される)場合は検証をスキップし、既存の汎用マージ動作を
    そのまま許可する。片方のみAnima系と判定された場合、または既知でない
    組み合わせの場合は、推測でのマージを避けるためUnverifiedModelPairErrorを
    送出して停止する。

    Args:
        base_variant: classify_anima_model_variant()の戻り値(Base側)。
        secondary_variant: 同(Secondary側)。

    Raises:
        UnverifiedModelPairError: 未検証の組み合わせの場合。
    """
    if base_variant == "unknown" and secondary_variant == "unknown":
        return
    if frozenset({base_variant, secondary_variant}) not in _VERIFIED_ANIMA_VARIANT_PAIRS:
        raise UnverifiedModelPairError(
            "未検証のモデル組み合わせのためマージを中断しました "
            f"(base={base_variant}, secondary={secondary_variant})。"
            "検証済みの組み合わせは AnimaBase v1.0(28block) と "
            "Anima 3.8B v1.0/v1.1(52block) の間のみです。"
        )


def anima_block_category_layout(num_blocks: int) -> list[str]:
    """ブロック総数からInput/Middle/Outputのカテゴリ列を生成する。

    lora_train.pyの`_anima_block_categories()`と同一規則(28ブロックは
    オリジナルの9/10/9分割と完全一致、それ以外は均等3分割で近似)。
    姉妹コード間の一貫性のため同一ロジックを複製している。

    Args:
        num_blocks: DiTのブロック総数。

    Returns:
        index=0..num_blocks-1に対応する"input"/"middle"/"output"のリスト。
    """
    if num_blocks == 28:
        return ["input"] * 9 + ["middle"] * 10 + ["output"] * 9
    third = num_blocks // 3
    remainder = num_blocks - third * 3
    return ["input"] * third + ["middle"] * (third + remainder) + ["output"] * third


def is_known_anima_architecture_gap(key: str, secondary_num_blocks: Optional[int]) -> bool:
    """キーがSecondary側の(より小さい)アーキテクチャゆえに存在しないと想定できるか判定する。

    52block→28blockマージ時のblock28-51や、3.8B v1.0→v1.1マージ時の
    anima_v2_connector.*のように、「Secondary側に存在しないことが設計上
    想定済み」のキーを識別する。validate_compatible()の警告を、意図的な
    除外(想定内)と本当に予期しない不整合とで区別するために使う。

    Args:
        key: Base側のテンソルキー。
        secondary_num_blocks: Secondary側で検出されたブロック総数(未検出ならNone)。

    Returns:
        設計上想定済みの欠落と判定できればTrue。
    """
    canonical = canonical_key(key)
    if secondary_num_blocks is not None:
        match = _ANIMA_BLOCK_KEY_PATTERN.search(canonical)
        if match and int(match.group(1)) >= secondary_num_blocks:
            return True
    return _ANIMA_CONNECTOR_V2_KEY_MARKER in canonical


_MERGE_HISTORY_METADATA_KEY = "anima_model_editor_merge_history"


def compose_output_metadata(
    base_metadata: dict[str, str], overrides: dict[str, str]
) -> dict[str, str]:
    """Base側の元のmetadataを保持しつつマージ由来の項目で上書きし、マージ履歴を追記する。

    Args:
        base_metadata: マージ元(Base)モデルが元々持っていたsafetensors metadata。
        overrides: 今回のマージ処理自身が付与するメタデータ(merge_type,
            base_sha256等)。base_metadataの同名キーはこちらで上書きされる。

    Returns:
        base_metadataをベースにoverridesで上書きした辞書。既存の
        `anima_model_editor_merge_history`(JSON配列文字列)があれば1件
        追記し、無ければ新規に1件分の配列として作成する。
    """
    composed = dict(base_metadata)
    composed.update(overrides)

    history_entry = {
        key: overrides[key]
        for key in (
            "merge_type",
            "base_sha256",
            "secondary_sha256",
            "lora_sha256",
            "target_sha256",
            "rank",
        )
        if key in overrides
    }
    try:
        existing_history = json.loads(base_metadata.get(_MERGE_HISTORY_METADATA_KEY, "[]"))
        if not isinstance(existing_history, list):
            existing_history = []
    except (ValueError, TypeError):
        existing_history = []
    existing_history.append(history_entry)
    composed[_MERGE_HISTORY_METADATA_KEY] = json.dumps(existing_history, ensure_ascii=False)
    return composed


def describe_anima_model_variant_from_path(path: Path) -> tuple[str, Optional[int], bool]:
    """モデルファイルのヘッダのみを読み取り、Animaバリアント分類に必要な情報を返す。

    テンソル本体は読み込まない(safetensorsのヘッダ情報のみ走査)。GUI側の
    「モデルを検出」ボタンなど、フルロード前の軽量な事前確認に用いる。
    ckpt/bin形式はヘッダのみでの判定に対応していないため、この関数は
    safetensors形式のみを対象とする。

    Args:
        path: モデルファイルへのパス。

    Returns:
        (variant, num_blocks, is_connector_v2) のタプル。safetensors以外の
        拡張子、または`blocks.N.`パターンを検出できない場合は
        variant="unknown", num_blocks=None, is_connector_v2=False。

    Raises:
        FileNotFoundError: pathが存在しない場合。
        DependencyError: safetensorsパッケージが無い場合。
    """
    validate_model_path(path)
    if path.suffix.lower() != ".safetensors":
        return "unknown", None, False

    import importlib.util

    if importlib.util.find_spec("safetensors") is None:
        raise DependencyError("safetensors is required to inspect .safetensors files.")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})

    num_blocks = count_dit_blocks_from_keys(keys)
    is_connector_v2 = detect_connector_v2_from_keys_and_metadata(keys, metadata)
    variant = classify_anima_model_variant(num_blocks, is_connector_v2)
    return variant, num_blocks, is_connector_v2


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


def block_category(name: str, num_blocks: Optional[int] = None) -> str:
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
        if num_blocks in _KNOWN_ANIMA_DIT_BLOCK_COUNTS and 0 <= index < num_blocks:
            return anima_block_category_layout(num_blocks)[index]
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


def adjustment_group(name: str, mode: str, num_blocks: Optional[int] = None) -> str:
    block = {
        "input": "Input",
        "middle": "Middle",
        "output": "Output",
        "other": "Other",
    }[block_category(name, num_blocks)]
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


def layer_alpha(name: str, options: MergeOptions, num_blocks: Optional[int] = None) -> float:
    if name in options.layer_overrides:
        return max(0.0, min(1.0, options.alpha * options.layer_overrides[name]))
    group = adjustment_group(name, options.layer_display_mode, num_blocks)
    scale = options.parameter_scales.get(group)
    if scale is not None:
        return max(0.0, min(1.0, options.alpha * scale))
    block = block_category(name, num_blocks)
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


def should_freeze_bias(name: str, options: MergeOptions, num_blocks: Optional[int] = None) -> bool:
    lowered = name.lower()
    if not (lowered.endswith(".bias") or ".bias." in lowered or lowered.endswith("bias")):
        return False
    category = block_category(name, num_blocks)
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


def corrected_alpha(
    torch: object,
    name: str,
    a: object,
    b: object,
    options: MergeOptions,
    num_blocks: Optional[int] = None,
) -> tuple[float, bool]:
    alpha = layer_alpha(name, options, num_blocks)
    if not options.auto_correction:
        return alpha, False
    similarity = cosine_similarity(torch, a, b)
    if similarity >= options.cosine_threshold:
        return alpha, False
    scale = max(0.0, similarity) / max(options.cosine_threshold, 1e-6)
    return alpha * scale, True


def validate_compatible(
    base: dict[str, object],
    other: dict[str, object],
    other_map: dict[str, str],
    is_expected_gap: Callable[[str], bool] | None = None,
) -> list[str]:
    """Base/Secondary間のキー・shape不整合を検出し、警告メッセージ一覧を返す。

    Args:
        base: Baseモデルのstate_dict。
        other: Secondaryモデルのstate_dict。
        other_map: canonical_state_map(other)の結果。
        is_expected_gap: Secondary側に存在しないことが設計上想定済みの
            キー(例: 52block→28blockマージ時のblock28-51・connector関連
            キー)かどうかを判定する関数。Trueを返すキーは個別警告に
            含めず、件数のみ末尾に1行でまとめて報告する(想定内の欠落で
            本来の警告が埋没するのを防ぐため)。Noneの場合は全件を
            個別警告として報告する(従来動作)。

    Returns:
        警告メッセージのリスト。
    """
    warnings: list[str] = []
    expected_gap_count = 0
    for key, base_tensor in base.items():
        other_key = other_map.get(canonical_key(key))
        if other_key is None:
            if is_expected_gap is not None and is_expected_gap(key):
                expected_gap_count += 1
                continue
            warnings.append(f"Missing in secondary model: {key}")
            continue
        other_tensor = other[other_key]
        if getattr(base_tensor, "shape", None) != getattr(other_tensor, "shape", None):
            warnings.append(f"Shape mismatch: {key} <-> {other_key}")
    if expected_gap_count:
        warnings.append(
            "Secondary model architecture is smaller by design: "
            f"{expected_gap_count} tensor(s) kept from base only (expected, not an error)."
        )
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
    base_num_blocks = count_dit_blocks_from_keys(base.keys())
    base_metadata = read_model_metadata(base_path)

    report = MergeReport(output_path=output_path)
    extracted: dict[str, object] = {}
    _extract_key_corrected_count = 0
    # 実抽出対象数を事前カウント（2D浮動小数点テンソルのみ）
    _extract_targets = [
        key for key, t in base.items()
        if is_merge_target(key)
        and not should_freeze_bias(key, options, base_num_blocks)
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
            or should_freeze_bias(key, options, base_num_blocks)
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

        scale = layer_alpha(key, options, base_num_blocks)
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

    metadata = compose_output_metadata(
        base_metadata,
        {
            "anima_model_editor": "2.0-tab1",
            "merge_type": "model_difference_to_lora",
            "base_sha256": sha256_file(base_path),
            "target_sha256": sha256_file(target_path),
            "rank": str(rank),
            "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
        },
    )
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

    base_num_blocks = count_dit_blocks_from_keys(base.keys())
    secondary_num_blocks = count_dit_blocks_from_keys(other.keys())
    base_metadata = read_model_metadata(base_path)
    base_is_connector_v2 = detect_connector_v2_from_keys_and_metadata(base.keys(), base_metadata)
    secondary_metadata = read_model_metadata(secondary_path)
    secondary_is_connector_v2 = detect_connector_v2_from_keys_and_metadata(other.keys(), secondary_metadata)
    base_variant = classify_anima_model_variant(base_num_blocks, base_is_connector_v2)
    secondary_variant = classify_anima_model_variant(secondary_num_blocks, secondary_is_connector_v2)
    verify_anima_variant_pair_if_applicable(base_variant, secondary_variant)
    if base_variant != "unknown" or secondary_variant != "unknown":
        log(f"Detected variants: base={base_variant}, secondary={secondary_variant}")

    report = MergeReport(output_path=output_path)
    compatibility_warnings = validate_compatible(
        base,
        other,
        other_map,
        is_expected_gap=lambda key: is_known_anima_architecture_gap(key, secondary_num_blocks),
    )
    report.warnings.extend(compatibility_warnings[:100])
    remapped_count = sum(1 for key in base if key not in other and canonical_key(key) in other_map)
    if remapped_count:
        log(f"Key prefix remap enabled: {remapped_count} tensor key(s)")

    # 実マージ対象数を事前カウント（ログ分母を実態に合わせる）
    _merge_targets = [
        key for key, base_tensor in base.items()
        if is_merge_target(key)
        and not should_freeze_bias(key, options, base_num_blocks)
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
            or should_freeze_bias(key, options, base_num_blocks)
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
        alpha, corrected = corrected_alpha(torch, key, base_tensor, other_tensor, options, base_num_blocks)
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

    metadata = compose_output_metadata(
        base_metadata,
        {
            "anima_model_editor": "2.0-tab1",
            "merge_type": "model_to_model",
            "base_sha256": sha256_file(base_path),
            "secondary_sha256": sha256_file(secondary_path),
            "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
        },
    )
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
    base_num_blocks = count_dit_blocks_from_keys(base.keys())
    base_metadata = read_model_metadata(base_path)

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
            should_freeze_bias(base_key, options, base_num_blocks)
            or not is_merge_target(base_key)
            or not hasattr(target, "is_floating_point")
            or not target.is_floating_point()
        ):
            report.skipped_tensors += 1
            continue
        merged[target_key] = (
            target.detach().to("cpu") + delta * layer_alpha(target_key, options, base_num_blocks)
        ).to(dtype=target.dtype)
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

    metadata = compose_output_metadata(
        base_metadata,
        {
            "anima_model_editor": "2.0-tab1",
            "merge_type": "lora_to_model",
            "base_sha256": sha256_file(base_path),
            "lora_sha256": sha256_file(lora_path),
            "license_guardrail": "NVIDIA Open Model License may apply to Cosmos-Predict2 derivatives.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving fused model: {output_path}")
    save_state_dict(output_path, merged, metadata)
    del base, lora, merged
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()
    return report
