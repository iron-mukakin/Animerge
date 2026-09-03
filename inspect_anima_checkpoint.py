"""inspect_anima_checkpoint.py — Anima 3.8B移行調査用チェックポイント検査スクリプト

目的:
    以下3種のファイルからテンソルを実体化せずヘッダ情報(key/shape/dtype)のみを読み取り、
    Anima 3.8B対応実装に必要な数値(num_blocks, semantic_source_dim, native/trained境界等)を
    確定させるためのレポートを出力する。

    1. DiT本体チェックポイント        (例: Anima-3.8B.safetensors)
    2. Progressive Cross Adapterチェックポイント (例: Anima-3.8B-expanded_adapter.safetensors)
    3. Qwen3.5 4B テキストエンコーダ  (ディレクトリ or 単一safetensorsファイル)

使い方:
    python inspect_anima_checkpoint.py \
        --dit ./path/to/Anima-3.8B.safetensors \
        --adapter ./path/to/Anima-3.8B-expanded_adapter.safetensors \
        --qwen35 ./path/to/qwen35_4b_dir_or_file

    3引数はいずれも省略可(指定した対象のみ検査する)。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from safetensors import safe_open

_BLOCK_INDEX_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_LLM_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_KNOWN_DIT_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.model.",
    "model.",
    "module.",
    "state_dict.",
    "net.",
)


@dataclass
class TensorHeaderEntry:
    """safetensorsヘッダ1エントリ分のkey/shape/dtype。"""

    key: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class CheckpointReport:
    """1ファイル分の検査結果をまとめる集計コンテナ。"""

    source_path: str
    metadata: dict[str, str] = field(default_factory=dict)
    findings: dict[str, object] = field(default_factory=dict)
    entry_count: int = 0


def strip_known_dit_prefix(key: str) -> str:
    """DiTチェックポイントで観測される既知プレフィックス(net.等)を1段だけ除去する。

    Args:
        key: safetensors内の生key文字列。

    Returns:
        既知プレフィックスを取り除いたkey。該当プレフィックスがなければ元のまま返す。
    """
    for prefix in _KNOWN_DIT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def read_safetensors_header(path: Path) -> tuple[list[TensorHeaderEntry], dict[str, str]]:
    """safetensorsファイルのヘッダのみを読み取り、テンソル本体は読み込まない。

    Args:
        path: safetensorsファイルへのパス。

    Returns:
        (エントリ一覧, メタデータ辞書) のタプル。

    Raises:
        FileNotFoundError: pathが存在しない場合。
    """
    if not path.is_file():
        raise FileNotFoundError(f"safetensorsファイルが見つかりません: {path}")

    entries: list[TensorHeaderEntry] = []
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            slice_obj = handle.get_slice(key)
            entries.append(
                TensorHeaderEntry(key=key, shape=tuple(slice_obj.get_shape()), dtype=str(slice_obj.get_dtype()))
            )
    return entries, dict(metadata)


def extract_max_block_index(entries: list[TensorHeaderEntry], key_transform=strip_known_dit_prefix) -> Optional[int]:
    """`blocks.N.`パターンから最大ブロックindex(=ブロック総数-1)を抽出する。

    Args:
        entries: read_safetensors_headerで得たエントリ一覧。
        key_transform: keyの前処理関数(プレフィックス除去等)。

    Returns:
        最大ブロックindex。`blocks.N.`パターンが1件もなければNone。
    """
    max_index: Optional[int] = None
    for entry in entries:
        transformed_key = key_transform(entry.key)
        match = _BLOCK_INDEX_PATTERN.search(transformed_key)
        if match:
            index = int(match.group(1))
            max_index = index if max_index is None else max(max_index, index)
    return max_index


def summarize_dit_checkpoint(entries: list[TensorHeaderEntry]) -> dict[str, object]:
    """DiT本体チェックポイントから軸A(ブロック構成)に必要な数値を抽出する。

    Args:
        entries: read_safetensors_headerで得たエントリ一覧。

    Returns:
        num_blocks / model_channels / has_llm_adapter 等を含む辞書。
    """
    findings: dict[str, object] = {}

    max_block_index = extract_max_block_index(entries)
    findings["num_blocks"] = None if max_block_index is None else max_block_index + 1

    x_embedder_entries = [e for e in entries if "x_embedder" in strip_known_dit_prefix(e.key) and "proj" in e.key.lower()]
    if x_embedder_entries:
        findings["x_embedder_shape_example"] = {x_embedder_entries[0].key: x_embedder_entries[0].shape}

    findings["has_embedded_llm_adapter"] = any("llm_adapter" in strip_known_dit_prefix(e.key) for e in entries)

    cross_attn_q_proj = [
        e for e in entries if re.search(r"blocks\.0\.cross_attn\.q_proj\.weight$", strip_known_dit_prefix(e.key))
    ]
    if cross_attn_q_proj:
        findings["block0_cross_attn_q_proj_shape"] = cross_attn_q_proj[0].shape

    findings["total_entries"] = len(entries)
    return findings


def summarize_adapter_checkpoint(entries: list[TensorHeaderEntry], metadata: dict[str, str]) -> dict[str, object]:
    """Progressive Cross Adapterチェックポイントから軸B(Qwen3.5連携)に必要な数値を抽出する。

    Args:
        entries: read_safetensors_headerで得たエントリ一覧。
        metadata: safetensorsのmetadataブロック(architecture識別子等)。

    Returns:
        semantic_source_dim / num_native_adapter_blocks / num_semantic_layers 等を含む辞書。
    """
    findings: dict[str, object] = {}
    findings["architecture_tag"] = metadata.get("architecture")

    by_key = {e.key: e for e in entries}

    mix_logits_entry = next((e for e in entries if e.key.endswith("layer_mix_logits")), None)
    if mix_logits_entry is not None and len(mix_logits_entry.shape) == 2:
        findings["num_native_adapter_blocks"] = mix_logits_entry.shape[0]
        findings["num_semantic_layers"] = mix_logits_entry.shape[1]

    k_proj_entry = next(
        (e for e in entries if re.search(r"semantic_attentions\.0\.k_proj\.weight$", e.key)), None
    )
    if k_proj_entry is not None:
        # nn.Linear(context_dim, inner_dim).weight の shape は (inner_dim, context_dim)
        findings["semantic_source_dim"] = k_proj_entry.shape[1]
        findings["semantic_attn_inner_dim"] = k_proj_entry.shape[0]

    q_proj_entry = next(
        (e for e in entries if re.search(r"semantic_attentions\.0\.q_proj\.weight$", e.key)), None
    )
    if q_proj_entry is not None:
        findings["semantic_attn_query_dim"] = q_proj_entry.shape[1]

    native_prefixed_keys = [k for k in by_key if k.startswith("native_adapter.")]
    findings["adapter_checkpoint_includes_native_weights"] = len(native_prefixed_keys) > 0
    findings["total_entries"] = len(entries)
    return findings


def read_qwen35_config_from_directory(directory: Path) -> dict[str, object]:
    """HuggingFace形式ディレクトリのconfig.jsonからQwen3.5 4Bの構成値を読み取る。

    Args:
        directory: config.jsonを含むモデルディレクトリ。

    Returns:
        hidden_size / num_hidden_layers 等を含む辞書。
    """
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.jsonが見つかりません: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "hidden_size": raw_config.get("hidden_size"),
        "num_hidden_layers": raw_config.get("num_hidden_layers"),
        "num_attention_heads": raw_config.get("num_attention_heads"),
        "model_type": raw_config.get("model_type"),
    }


def infer_qwen35_config_from_safetensors(entries: list[TensorHeaderEntry]) -> dict[str, object]:
    """単一safetensorsファイルのみが与えられた場合にkey shapeからQwen3.5構成を推定する。

    Args:
        entries: read_safetensors_headerで得たエントリ一覧。

    Returns:
        hidden_size(推定) / num_hidden_layers(推定) を含む辞書。
    """
    findings: dict[str, object] = {}

    embed_entry = next(
        (e for e in entries if e.key.endswith("embed_tokens.weight")), None
    )
    if embed_entry is not None and len(embed_entry.shape) == 2:
        findings["hidden_size_estimated"] = embed_entry.shape[1]
        findings["vocab_size_estimated"] = embed_entry.shape[0]

    max_layer_index: Optional[int] = None
    for entry in entries:
        match = _LLM_LAYER_INDEX_PATTERN.search(entry.key)
        if match:
            index = int(match.group(1))
            max_layer_index = index if max_layer_index is None else max(max_layer_index, index)
    findings["num_hidden_layers_estimated"] = None if max_layer_index is None else max_layer_index + 1

    return findings


def report_dit_checkpoint(path: Path) -> CheckpointReport:
    """DiTチェックポイント1件を検査してCheckpointReportを構築する。"""
    entries, metadata = read_safetensors_header(path)
    return CheckpointReport(
        source_path=str(path),
        metadata=metadata,
        findings=summarize_dit_checkpoint(entries),
        entry_count=len(entries),
    )


def report_adapter_checkpoint(path: Path) -> CheckpointReport:
    """Progressive Cross Adapterチェックポイント1件を検査してCheckpointReportを構築する。"""
    entries, metadata = read_safetensors_header(path)
    return CheckpointReport(
        source_path=str(path),
        metadata=metadata,
        findings=summarize_adapter_checkpoint(entries, metadata),
        entry_count=len(entries),
    )


def report_qwen35_source(path: Path) -> CheckpointReport:
    """Qwen3.5 4Bのディレクトリまたは単一safetensorsファイルを検査してCheckpointReportを構築する。"""
    if path.is_dir():
        findings = read_qwen35_config_from_directory(path)
        return CheckpointReport(source_path=str(path), findings=findings)

    entries, metadata = read_safetensors_header(path)
    findings = infer_qwen35_config_from_safetensors(entries)
    return CheckpointReport(source_path=str(path), metadata=metadata, findings=findings, entry_count=len(entries))


def format_report_as_text(title: str, report: CheckpointReport) -> str:
    """CheckpointReportを人間可読なテキストブロックに整形する。"""
    lines = [f"### {title}", f"path: {report.source_path}"]
    if report.entry_count:
        lines.append(f"total_entries: {report.entry_count}")
    if report.metadata:
        lines.append(f"metadata: {report.metadata}")
    for key, value in report.findings.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def parse_cli_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Anima 3.8B移行調査用チェックポイント検査スクリプト")
    parser.add_argument("--dit", type=str, default=None, help="DiT本体チェックポイント(safetensors)へのパス")
    parser.add_argument("--adapter", type=str, default=None, help="Progressive Cross Adapterチェックポイント(safetensors)へのパス")
    parser.add_argument("--qwen35", type=str, default=None, help="Qwen3.5 4Bのディレクトリまたはsafetensorsファイルへのパス")
    return parser.parse_args()


def report_checkpoint_summary() -> None:
    """CLIエントリポイント: 指定された各対象を検査し、結果をまとめて標準出力に表示する。"""
    args = parse_cli_arguments()

    if args.dit is None and args.adapter is None and args.qwen35 is None:
        print("少なくとも1つの引数(--dit / --adapter / --qwen35)を指定してください。")
        return

    if args.dit is not None:
        try:
            print(format_report_as_text("DiT本体 (軸A: ブロック構成)", report_dit_checkpoint(Path(args.dit))))
        except Exception as error:  # noqa: BLE001 - 検査ツールのため各対象の失敗を個別に報告する
            print(f"### DiT本体の検査に失敗しました: {error}")
        print()

    if args.adapter is not None:
        try:
            print(
                format_report_as_text(
                    "Progressive Cross Adapter (軸B: Qwen3.5連携)", report_adapter_checkpoint(Path(args.adapter))
                )
            )
        except Exception as error:  # noqa: BLE001
            print(f"### Adapterの検査に失敗しました: {error}")
        print()

    if args.qwen35 is not None:
        try:
            print(format_report_as_text("Qwen3.5 4B テキストエンコーダ", report_qwen35_source(Path(args.qwen35))))
        except Exception as error:  # noqa: BLE001
            print(f"### Qwen3.5の検査に失敗しました: {error}")
        print()


if __name__ == "__main__":
    report_checkpoint_summary()
