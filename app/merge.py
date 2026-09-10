from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import MergeOptions
from .anima_block_manifests import (
    block_correspondence_map,
    inserted_block_positions,
    is_expansion_pair_official,
)
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


# ──────────────────────────────────────────────────────────────────────
# Layer 2: Architecture-aware block mapping
#
# AnimaBase v1.0(28block)とAnima 3.8B(52block)は、28->40(Anima 2.9B)->52の
# 2段階のLLaMA-Pro方式block挿入で拡張されている。挿入位置の前後でblock
# indexがズレるため、「base.blocks.N と secondary.blocks.N は同一index」
# という前提(canonical_key一致・shape一致のみによる対応付け)は誤りであり、
# 意味的に無関係なblock同士をmergeしてモデルを破損させる恐れがある。
# ここでは expand_manifest(anima_block_manifests.py、一次資料から検証済み)
# に基づき、継承block同士を正しく対応付けるためのmapping層を提供する。
# ──────────────────────────────────────────────────────────────────────

_BLOCK_KEY_REWRITE_PATTERN = re.compile(r"(\.blocks\.)(\d+)(\.)")


@dataclass(frozen=True)
class BlockMapping:
    """1つのoutput block位置に対する、Base/Secondary側の対応block index。

    kindは以下のいずれか:
        "native": Base/Secondary双方に、expand_manifestのbase_to_target対応に
            基づく継承blockが存在する。
        "inserted_no_counterpart": より小さいblock数側のモデルには存在しない、
            アーキテクチャ拡張で新規挿入されたblock位置。
    """

    output_block: int
    base_block: Optional[int]
    secondary_block: Optional[int]
    kind: str


def extract_block_index_from_key(key: str) -> Optional[int]:
    """テンソルキーから`.blocks.N.`のNを抽出する。

    llm_adapter配下のキーは、将来的にAnima派生モデルが独自のblock構造
    (メインのnet.blocksとは別物)を持つケースに備えた防御的除外として
    対象外にする(anima_common.pyの`_is_llm_adapter_key`と同じ方針)。
    `.blocks.N.`パターンを含まないキー(connector, embedder等)もNoneを返す。

    Args:
        key: テンソルキー文字列。

    Returns:
        block index。抽出できなければNone。
    """
    if "llm_adapter" in key.lower():
        return None
    match = _BLOCK_KEY_REWRITE_PATTERN.search(key)
    if not match:
        return None
    return int(match.group(2))


def rewrite_block_index_in_key(key: str, new_index: int) -> Optional[str]:
    """テンソルキー内の`.blocks.N.`のNをnew_indexに書き換えたキーを返す。

    extract_block_index_from_key()と同じ除外規則に従う(llm_adapter配下、
    `.blocks.N.`パターン非該当はNone)。

    Args:
        key: 元のテンソルキー文字列。
        new_index: 書き換え後のblock index。

    Returns:
        書き換え後のキー文字列。書き換え対象でなければNone。
    """
    if "llm_adapter" in key.lower():
        return None
    match = _BLOCK_KEY_REWRITE_PATTERN.search(key)
    if not match:
        return None
    return key[: match.start()] + match.group(1) + str(new_index) + match.group(3) + key[match.end():]


def build_block_mappings(base_num_blocks: int, secondary_num_blocks: int) -> tuple[list[BlockMapping], int]:
    """Base/Secondaryのブロック総数から、architecture-awareなBlockMapping一覧を構築する。

    出力architectureは常により大きいblock数側に合わせる(28+52なら52、
    52+28なら52。同数の場合はこの関数を呼ぶ必要が無い、呼び出し側で
    同一architecture用の既存経路を使うこと)。expand_manifestに存在しない
    未知の組み合わせの場合は、推測でマージせずUnverifiedModelPairErrorを
    送出して停止する。

    Args:
        base_num_blocks: Base側のブロック総数。
        secondary_num_blocks: Secondary側のブロック総数(base_num_blocksと
            異なる前提)。

    Returns:
        (mappings, output_num_blocks) のタプル。mappingsはoutput_block
        昇順に output_num_blocks 件並ぶ。

    Raises:
        UnverifiedModelPairError: 対応するexpand_manifestが存在しない場合。
    """
    smaller_num_blocks = min(base_num_blocks, secondary_num_blocks)
    larger_num_blocks = max(base_num_blocks, secondary_num_blocks)
    correspondence = block_correspondence_map(smaller_num_blocks, larger_num_blocks)
    inserted = inserted_block_positions(smaller_num_blocks, larger_num_blocks)
    if correspondence is None or inserted is None:
        raise UnverifiedModelPairError(
            "未検証のblock数の組み合わせのためマージを中断しました "
            f"({smaller_num_blocks}block <-> {larger_num_blocks}block)。"
            "既知のexpand_manifest(28<->40<->52)が存在する組み合わせのみ"
            "サポートします。"
        )

    base_is_larger = base_num_blocks == larger_num_blocks
    inverse_correspondence = {large_idx: small_idx for small_idx, large_idx in correspondence.items()}

    mappings: list[BlockMapping] = []
    for output_idx in range(larger_num_blocks):
        if output_idx in inserted:
            mappings.append(
                BlockMapping(
                    output_block=output_idx,
                    base_block=output_idx if base_is_larger else None,
                    secondary_block=None if base_is_larger else output_idx,
                    kind="inserted_no_counterpart",
                )
            )
        else:
            small_idx = inverse_correspondence[output_idx]
            mappings.append(
                BlockMapping(
                    output_block=output_idx,
                    base_block=output_idx if base_is_larger else small_idx,
                    secondary_block=small_idx if base_is_larger else output_idx,
                    kind="native",
                )
            )
    return mappings, larger_num_blocks


def build_cross_architecture_merge_plan(
    base: dict[str, object],
    secondary: dict[str, object],
    base_num_blocks: int,
    secondary_num_blocks: int,
) -> tuple[list[tuple[str, object, object, str]], list[BlockMapping], int]:
    """異なるblock数のBase/Secondary間で、architecture-awareなマージ計画を構築する。

    block構造を持つキー(`.blocks.N.`)はBlockMapping(expand_manifest由来)に
    従って対応付ける。block構造を持たないキー(final_layer/t_embedder/
    x_embedder/llm_adapter/anima_v2_connector等)は、世代拡張の対象外の
    共通コンポーネントであるため、shapeやblock indexとは無関係に
    canonical_keyの一致でのみ対応付ける(従来通りの挙動)。

    出力のキー集合は、より大きいblock数側(output architecture側)の
    キー集合と一致する(小さい側だけが持つキーは出力に含まれない)。

    Args:
        base: Baseモデルのstate_dict。
        secondary: Secondaryモデルのstate_dict。
        base_num_blocks: Base側のブロック総数。
        secondary_num_blocks: Secondary側のブロック総数(base_num_blocksと
            異なる前提)。

    Returns:
        (plan, mappings, output_num_blocks) のタプル。
        planは(output_key_source, base_side_tensor_or_None,
        secondary_side_tensor_or_None, kind)のリストで、出力architecture側の
        全キーを1件ずつ含む。output_key_sourceはoutput_key_name()適用前の
        生キー。kindは "native"(block対応あり、通常のalphaブレンド対象) /
        "native_missing_in_secondary"(block位置は対応するはずだが実際の
        キーが見つからない) / "inserted_no_counterpart"(拡張で新規挿入
        されたblock、片側の値をそのまま使う) / "non_block_matched"
        (block構造を持たないキーで両側に対応物あり、通常のalphaブレンド
        対象) / "non_block_unmatched"(block構造を持たないキーで片側にしか
        存在しない、そのまま保持)のいずれか。

    Raises:
        UnverifiedModelPairError: 対応するexpand_manifestが存在しない場合。
    """
    mappings, output_num_blocks = build_block_mappings(base_num_blocks, secondary_num_blocks)
    base_is_larger = base_num_blocks == output_num_blocks
    larger = base if base_is_larger else secondary
    smaller = secondary if base_is_larger else base

    output_to_smaller_block = {
        m.output_block: (m.secondary_block if base_is_larger else m.base_block)
        for m in mappings
        if m.kind == "native"
    }
    inserted_output_blocks = {m.output_block for m in mappings if m.kind == "inserted_no_counterpart"}
    smaller_canonical_map = canonical_state_map(smaller)

    plan: list[tuple[str, object, object, str]] = []
    for larger_key, larger_tensor in larger.items():
        block_idx = extract_block_index_from_key(larger_key)
        if block_idx is None:
            smaller_key = larger_key if larger_key in smaller else smaller_canonical_map.get(canonical_key(larger_key))
            smaller_tensor = smaller.get(smaller_key) if smaller_key is not None else None
            kind = "non_block_matched" if smaller_tensor is not None else "non_block_unmatched"
        elif block_idx in inserted_output_blocks:
            smaller_tensor = None
            kind = "inserted_no_counterpart"
        else:
            smaller_block_idx = output_to_smaller_block.get(block_idx)
            smaller_key = None
            if smaller_block_idx is not None:
                smaller_key = rewrite_block_index_in_key(larger_key, smaller_block_idx)
                if smaller_key is not None and smaller_key not in smaller:
                    smaller_key = smaller_canonical_map.get(canonical_key(smaller_key))
            smaller_tensor = smaller.get(smaller_key) if smaller_key is not None else None
            kind = "native" if smaller_tensor is not None else "native_missing_in_secondary"

        if base_is_larger:
            plan.append((larger_key, larger_tensor, smaller_tensor, kind))
        else:
            plan.append((larger_key, smaller_tensor, larger_tensor, kind))

    return plan, mappings, output_num_blocks


def format_block_mapping_log(mappings: list[BlockMapping]) -> list[str]:
    """BlockMapping一覧を、ユーザー向けの人間可読なログ行(1block=1行)に整形する。

    出力例:
        output  0 <- base  0 / secondary  0
        output  2 <- inserted (no counterpart)
        output  3 <- base  3 / secondary  1

    Args:
        mappings: build_block_mappings()の戻り値。

    Returns:
        ログ行のリスト。
    """
    lines: list[str] = []
    for mapping in mappings:
        if mapping.kind == "inserted_no_counterpart":
            lines.append(f"  output {mapping.output_block:2d} <- inserted (no counterpart)")
        else:
            lines.append(
                f"  output {mapping.output_block:2d} <- base {mapping.base_block:2d} "
                f"/ secondary {mapping.secondary_block:2d}"
            )
    return lines


def validate_merged_architecture(merged: dict[str, object], expected_num_blocks: Optional[int]) -> None:
    """マージ後の出力state_dictが期待するblock構成になっているか、保存前に検証する。

    Args:
        merged: マージ後のstate_dict。
        expected_num_blocks: 期待されるブロック総数。Noneなら検証をスキップする
            (Anima系と無関係な汎用マージの場合)。

    Raises:
        ValueError: 実際に検出されたブロック総数が期待値と一致しない場合。
    """
    if expected_num_blocks is None:
        return
    actual = count_dit_blocks_from_keys(merged.keys())
    if actual != expected_num_blocks:
        raise ValueError(
            "マージ後のモデルのブロック総数が期待値と一致しないため、保存を中止しました "
            f"(期待={expected_num_blocks}, 実際={actual})。"
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


# ──────────────────────────────────────────────────────────────────────
# LoRAのblock index remapping(本体マージと同じexpand_manifestを再利用)
#
# LoRAキー(`blocks_N_`形式)が参照するblock indexは、そのLoRAが学習された
# 時点のモデルのブロック総数に基づく。適用先モデルのブロック総数が異なる
# 場合、merge_models()と同じexpand_manifest対応表を使って正しいblockへ
# 変換する必要がある(単純な同一index対応では意味的に異なるblockへ
# LoRAを適用してしまう)。
# ──────────────────────────────────────────────────────────────────────

_LORA_BLOCK_KEY_PATTERN = re.compile(r"(blocks_)(\d+)(_)")


def count_lora_authoring_block_count(lora: dict[str, object]) -> Optional[int]:
    """LoRAのstate_dictから、学習時点のモデルのブロック総数を推定する。

    LoRAキー自体は`blocks_N_`(アンダースコア区切り)形式のため、
    lora_target_candidates()が生成するdot区切り形式の代表キー
    (`net.blocks.N.xxx.weight`)からblock indexを抽出する
    (extract_block_index_from_key()を再利用)。

    Args:
        lora: LoRAのstate_dict。

    Returns:
        検出したブロック総数(最大index+1)。block構造を持つキーが
        1件も無ければNone。
    """
    max_index: Optional[int] = None
    for key in lora:
        candidate = lora_target_candidates(key)[0]
        index = extract_block_index_from_key(candidate)
        if index is not None:
            max_index = index if max_index is None else max(max_index, index)
    return None if max_index is None else max_index + 1


def remap_lora_block_index(
    lora_own_index: int, lora_num_blocks: int, target_num_blocks: int
) -> Optional[int]:
    """LoRAが学習された時点のblock indexを、適用先のblock indexへ変換する。

    lora_num_blocks < target_num_blocks(小さいblock数のモデルで学習した
    LoRAをより大きいモデルへ適用する場合)は、expand_manifestの
    base_to_target対応をそのまま使う。lora_num_blocks > target_num_blocks
    (逆方向、より大きいモデルで学習したLoRAをより小さいモデルへ適用する
    場合)は、対応表を逆引きする。本体マージ(merge_models)が双方向対応
    である一貫性のため、LoRA側もこの逆方向を明示的にサポートする。

    Args:
        lora_own_index: LoRAのキーに書かれているblock index(学習時の
            block空間でのindex)。
        lora_num_blocks: LoRAが学習された時点のモデルのブロック総数。
        target_num_blocks: 適用先モデルのブロック総数。

    Returns:
        適用先モデルでのblock index。適用先に対応物が無い場合
        (逆方向remapで、lora_own_indexが拡張により新規挿入された
        block由来の場合)はNone。

    Raises:
        UnverifiedModelPairError: lora_num_blocksとtarget_num_blocksの
            組み合わせに対応するexpand_manifestが存在しない場合。
    """
    if lora_num_blocks == target_num_blocks:
        return lora_own_index

    smaller_n, larger_n = sorted((lora_num_blocks, target_num_blocks))
    correspondence = block_correspondence_map(smaller_n, larger_n)
    inserted = inserted_block_positions(smaller_n, larger_n)
    if correspondence is None or inserted is None:
        raise UnverifiedModelPairError(
            "未検証のblock数の組み合わせのためLoRAのblock remapを中断しました "
            f"(LoRA学習時={lora_num_blocks}block, 適用先={target_num_blocks}block)。"
        )

    if lora_num_blocks < target_num_blocks:
        # 小(LoRA学習時) -> 大(適用先): 順方向の対応表をそのまま使う。
        return correspondence.get(lora_own_index)

    # 大(LoRA学習時) -> 小(適用先): 対応表を逆引きする。lora_own_indexが
    # 挿入block(小さい側に対応物が無い)ならNone。
    if lora_own_index in inserted:
        return None
    inverse = {large_idx: small_idx for small_idx, large_idx in correspondence.items()}
    return inverse.get(lora_own_index)


def describe_lora_block_count_from_path(path: Path) -> Optional[int]:
    """LoRAファイルのヘッダのみを読み取り、学習時点のブロック総数を推定する。

    テンソル本体は読み込まない(safetensorsのヘッダのキー一覧のみ走査)。
    GUI側の「モデルを検出」ボタンなど、フルロード前の軽量な事前確認に
    用いる。ckpt/bin形式はヘッダのみでの判定に対応していないため、
    この関数はsafetensors形式のみを対象とする。

    LoRAファイルは本体モデルと異なり"Anima 3.8B v1.0/v1.1"のような
    バリアント区分を持たない(内蔵adapterの有無等はLoRA自体には無関係)
    ため、describe_anima_model_variant_from_path()とは違い、返すのは
    ブロック総数のみ。

    Args:
        path: LoRAファイルへのパス。

    Returns:
        検出したブロック総数(最大index+1)。safetensors以外の拡張子、
        またはblock構造を持つキーを検出できない場合はNone。

    Raises:
        FileNotFoundError: pathが存在しない場合。
        DependencyError: safetensorsパッケージが無い場合。
    """
    validate_model_path(path)
    if path.suffix.lower() != ".safetensors":
        return None

    import importlib.util

    if importlib.util.find_spec("safetensors") is None:
        raise DependencyError("safetensors is required to inspect .safetensors files.")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())

    return count_lora_authoring_block_count(keys)


def remap_lora_key_block_index(
    key: str, lora_num_blocks: Optional[int], target_num_blocks: Optional[int]
) -> Optional[str]:
    """LoRAキー内のblock indexを、学習時->適用先のblock空間へ変換する。

    lora_num_blocks/target_num_blocksのいずれかが不明、または両者が
    一致する場合はkeyをそのまま返す(remap不要)。block構造を持たない
    キー(将来的な非block系LoRA、または解析できないキー)もそのまま返す。

    Args:
        key: 元のLoRAキー(`blocks_N_`形式のアンダースコア区切りを想定)。
        lora_num_blocks: LoRAが学習された時点のブロック総数。
        target_num_blocks: 適用先のブロック総数。

    Returns:
        block indexを書き換えた後のキー。remap不要ならkeyそのまま。
        適用先に対応物が無ければNone(呼び出し側はこのLoRA layerの
        適用をスキップすること)。

    Raises:
        UnverifiedModelPairError: 未検証のblock数の組み合わせの場合。
    """
    if lora_num_blocks is None or target_num_blocks is None or lora_num_blocks == target_num_blocks:
        return key
    match = _LORA_BLOCK_KEY_PATTERN.search(key)
    if not match:
        return key
    lora_own_index = int(match.group(2))
    new_index = remap_lora_block_index(lora_own_index, lora_num_blocks, target_num_blocks)
    if new_index is None:
        return None
    return key[: match.start()] + match.group(1) + str(new_index) + match.group(3) + key[match.end():]


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

    report = MergeReport(output_path=output_path)

    base_lora_num_blocks = count_lora_authoring_block_count(base)
    secondary_lora_num_blocks = count_lora_authoring_block_count(other)
    if (
        base_lora_num_blocks is not None
        and secondary_lora_num_blocks is not None
        and base_lora_num_blocks != secondary_lora_num_blocks
    ):
        output_lora_num_blocks = max(base_lora_num_blocks, secondary_lora_num_blocks)
        base_is_output_source = base_lora_num_blocks == output_lora_num_blocks
        log(
            "Cross block-count LoRA merge detected: "
            f"base={base_lora_num_blocks}block, secondary={secondary_lora_num_blocks}block, "
            f"output={output_lora_num_blocks}block"
        )
        _lora_no_counterpart = 0
        if base_is_output_source:
            remapped_other: dict[str, object] = {}
            for key, tensor in other.items():
                remapped_key = remap_lora_key_block_index(key, secondary_lora_num_blocks, base_lora_num_blocks)
                if remapped_key is None:
                    _lora_no_counterpart += 1
                    continue
                remapped_other[remapped_key] = tensor
            other = remapped_other
        else:
            remapped_base: dict[str, object] = {}
            for key, tensor in base.items():
                remapped_key = remap_lora_key_block_index(key, base_lora_num_blocks, secondary_lora_num_blocks)
                if remapped_key is None:
                    _lora_no_counterpart += 1
                    continue
                remapped_base[remapped_key] = tensor
            base = remapped_base
        if _lora_no_counterpart:
            report.warnings.append(
                f"{_lora_no_counterpart} LoRA tensor(s) skipped: no counterpart at the "
                "output architecture (block introduced by architecture expansion)."
            )

    other_map = canonical_lora_state_map(other)
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

    # block数が一致しない場合(かつ双方がAnima系と判定できる場合)のみ、
    # architecture-aware なblock mapping経路へ分岐する。片方でもblock数が
    # 未検出(非Anima系汎用モデル)、または双方のblock数が一致する場合は、
    # 既存のcanonical_key直接比較経路(変更なし)を使う。
    same_architecture = (
        base_num_blocks is None
        or secondary_num_blocks is None
        or base_num_blocks == secondary_num_blocks
    )

    if same_architecture:
        # ── 既存の同一architecture間マージ経路(ロジック変更なし) ──
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

        output_num_blocks_for_validation = base_num_blocks
        metadata_source = base_metadata
    else:
        # ── 新規: architecture-aware block mapping による cross-architecture 経路 ──
        log(
            f"Cross-architecture merge detected: base={base_num_blocks}block, "
            f"secondary={secondary_num_blocks}block"
        )
        plan, block_mappings, output_num_blocks = build_cross_architecture_merge_plan(
            base, other, base_num_blocks, secondary_num_blocks
        )
        base_is_output_source = base_num_blocks == output_num_blocks
        log(
            f"Output architecture: {output_num_blocks}block "
            f"({'base' if base_is_output_source else 'secondary'} side)"
        )
        smaller_n, larger_n = sorted((base_num_blocks, secondary_num_blocks))
        if is_expansion_pair_official(smaller_n, larger_n) is False:
            log(
                "WARNING: the block-count expansion mapping used for this merge "
                f"({smaller_n}->{larger_n}) is UNOFFICIAL (reconstructed via "
                "cosine-similarity comparison, not a published manifest). "
                "Review the block mapping log below carefully."
            )
        log("Block mapping:")
        for line in format_block_mapping_log(block_mappings):
            log(line)

        native_count = sum(1 for m in block_mappings if m.kind == "native")
        inserted_count = sum(1 for m in block_mappings if m.kind == "inserted_no_counterpart")
        non_block_matched_count = sum(1 for _k, _b, _s, kind in plan if kind == "non_block_matched")
        non_block_unmatched_count = sum(1 for _k, _b, _s, kind in plan if kind == "non_block_unmatched")
        native_missing_count = sum(1 for _k, _b, _s, kind in plan if kind == "native_missing_in_secondary")
        log(
            "Architecture-aware key correspondence: "
            f"native block pairs={native_count}, inserted (no counterpart)={inserted_count}, "
            f"non-block matched={non_block_matched_count}, "
            f"non-block unmatched={non_block_unmatched_count}, "
            f"native block pairs missing expected tensor={native_missing_count}"
        )
        if native_missing_count:
            report.warnings.append(
                f"{native_missing_count} tensor(s) expected at a mapped native block position "
                "were not found in the smaller-architecture model (kept from the larger side only)."
            )

        _total_merge = sum(
            1
            for out_key, base_t, other_t, _kind in plan
            if base_t is not None
            and other_t is not None
            and is_merge_target(out_key)
            and not should_freeze_bias(out_key, options, output_num_blocks)
            and hasattr(base_t, "detach")
            and hasattr(base_t, "is_floating_point")
            and base_t.is_floating_point()
        )
        log(f"Merge target layers: {_total_merge} / total keys: {len(plan)}")

        merged = {}
        _merge_index = 0
        _key_corrected_count = 0
        for out_key_raw, base_tensor, other_tensor, _plan_kind in plan:
            report.total_tensors += 1
            out_key = output_key_name(out_key_raw, options)
            if out_key != out_key_raw:
                _key_corrected_count += 1

            if base_tensor is None and other_tensor is None:
                report.skipped_tensors += 1
                continue
            if base_tensor is None:
                merged[out_key] = (
                    other_tensor.detach().to("cpu") if hasattr(other_tensor, "detach") else other_tensor
                )
                report.skipped_tensors += 1
                continue

            eligible = (
                other_tensor is not None
                and is_merge_target(out_key_raw)
                and not should_freeze_bias(out_key_raw, options, output_num_blocks)
                and getattr(base_tensor, "shape", None) == getattr(other_tensor, "shape", None)
                and hasattr(base_tensor, "detach")
                and hasattr(base_tensor, "is_floating_point")
                and base_tensor.is_floating_point()
                and hasattr(other_tensor, "is_floating_point")
                and other_tensor.is_floating_point()
            )
            if not eligible:
                merged[out_key] = base_tensor.detach().to("cpu") if hasattr(base_tensor, "detach") else base_tensor
                report.skipped_tensors += 1
                continue

            _merge_index += 1
            alpha, corrected = corrected_alpha(
                torch, out_key_raw, base_tensor, other_tensor, options, output_num_blocks
            )
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

        output_num_blocks_for_validation = output_num_blocks
        metadata_source = base_metadata if base_is_output_source else secondary_metadata

    if report.merged_tensors == 0:
        raise ValueError(
            "No compatible merge-target tensors were merged. "
            "Check that both models use the same architecture and tensor shapes."
        )

    if options.dry_run:
        log("Running dry-run tensor validation")
        dry_run_check(torch, merged)

    validate_merged_architecture(merged, output_num_blocks_for_validation)

    metadata = compose_output_metadata(
        metadata_source,
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
    lora_num_blocks = count_lora_authoring_block_count(lora)
    if (
        lora_num_blocks is not None
        and base_num_blocks is not None
        and lora_num_blocks != base_num_blocks
    ):
        log(
            f"LoRA block remap enabled: LoRA authored for {lora_num_blocks}block, "
            f"target model is {base_num_blocks}block"
        )
    used: set[str] = set()
    _fuse_index = 0
    _fuse_no_counterpart = 0
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

        remapped_key = remap_lora_key_block_index(key, lora_num_blocks, base_num_blocks)
        if remapped_key is None:
            used.add(key)
            used.add(down_key)
            _fuse_no_counterpart += 1
            report.skipped_tensors += 1
            continue

        candidates = lora_target_candidates(remapped_key)
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

    if _fuse_no_counterpart:
        report.warnings.append(
            f"{_fuse_no_counterpart} LoRA layer(s) skipped: no counterpart at the target "
            "architecture (block introduced by architecture expansion)."
        )

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
