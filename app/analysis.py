"""
analysis.py  –  Anima Model Editor v2.0
Layer Analysis backend (Tab 3)

分析対象: 本体モデル（checkpoints/）・LoRA（lora/）両方に対応
分析手法: Feature Map / Statistical / SVD Rank / Attention Map
表示レイヤー: Matrix / Transformer / Component  （merge.py の正規化ロジックと共有）

ログ保存形式:
  log/log_analysis/{モデル名}_{model|lora}_{手法}_{レイヤー}.txt

ログ先頭には後続の Preview / Compare 機能が読み取れる
メタデータブロックを必須フィールドとして付与する。
"""

from __future__ import annotations

import datetime
import gc
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

# merge.py から正規化ロジックを再利用
from .merge import (
    adjustment_group,
    anima_v1_key,
    block_category,
    canonical_key,
    canonical_lora_key,
    canonical_state_map,
    component_category,
    is_merge_target,
    lora_target_candidates,
    transformer_group,
)
from .model_io import (
    DependencyError,
    load_state_dict,
    sha256_file,
    validate_model_path,
)
from .i18n import gettext as _


# ─── 定数 ────────────────────────────────────────────────────────────────────

ANALYSIS_METHODS = ("Feature Map", "Statistical", "SVD Rank", "Attention Map")
LAYER_DISPLAY_MODES = ("Matrix", "Transformer", "Component")

# LoRA キーに含まれる典型的なサフィックス（分析対象の判別に使用）
_LORA_KEY_HINTS = ("lora_up", "lora_down", "lora_A", "lora_B", ".alpha")

# 統計的分析での異常値判定（L2ノルムのIQR倍率）
_OUTLIER_IQR_FACTOR = 1.5

# SVDランク分析: 累積寄与率の閾値
_SVD_CUMVAR_THRESHOLD = 0.99


# ─── データクラス ─────────────────────────────────────────────────────────────

@dataclass
class LayerRecord:
    """1レイヤー分の分析結果を保持する。"""
    key: str                         # 正規化済みキー
    original_key: str               # モデル内の元キー
    group: str                       # 表示レイヤーグループ名
    shape: tuple[int, ...]          # テンソル形状
    # --- Feature Map ---
    feat_mean: float = 0.0
    feat_var: float = 0.0
    feat_complexity: float = 0.0     # 空間的複雑度（std / (|mean| + ε)）
    # --- Statistical ---
    stat_l2: float = 0.0
    stat_mean: float = 0.0
    stat_var: float = 0.0
    # --- SVD Rank ---
    svd_effective_rank: int = 0
    svd_decay_rate: float = 0.0      # 最大特異値に対する有効ランク末尾の比率
    svd_cumvar_at_threshold: float = 0.0  # _SVD_CUMVAR_THRESHOLD 到達までの累積寄与率
    svd_singular_values: list[float] = field(default_factory=list)
    # --- Attention Map ---
    attn_mean_weight: float = 0.0    # アテンションウェイトの平均絶対値
    attn_head_variance: float = 0.0  # マルチヘッド間の分散（推定値）
    is_attention_layer: bool = False


@dataclass
class AnalysisReport:
    """run_analysis() の戻り値。"""
    model_name: str
    model_type: str          # "model" or "lora"
    method: str
    layer_mode: str
    device: str
    sha256: str
    timestamp: str
    records: list[LayerRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log_path: Path | None = None
    # 自動テキストレポート（ルールベース）
    auto_report_lines: list[str] = field(default_factory=list)
    # 重要層 / 非重要層キー（GUI 色分け用）
    important_layer_keys: list[str] = field(default_factory=list)
    unimportant_layer_keys: list[str] = field(default_factory=list)


# ─── キー正規化ユーティリティ ──────────────────────────────────────────────────

def _is_lora_state_dict(state_dict: dict[str, Any]) -> bool:
    """LoRAファイルかどうかをキー名から推定する。"""
    for key in list(state_dict.keys())[:20]:
        if any(hint in key for hint in _LORA_KEY_HINTS):
            return True
    return False


def _lora_base_key(key: str, key_correction: bool = False) -> str:
    """LoRAキーからベース層名を推定する（正規化）。

    key_correction=True の場合は lora_target_candidates を使って
    anima-base-v1.0 形式 (net.blocks.N.xxx.weight) に変換する。
    False の場合は従来の簡易変換を行う。
    """
    if key_correction:
        candidates = lora_target_candidates(key)
        # net. プレフィックスが付いた候補を優先して返す
        for c in candidates:
            if c.startswith("net."):
                return c
        return candidates[0] if candidates else canonical_lora_key(key)

    normalized = canonical_key(key)
    normalized = re.sub(r"\.(lora_up|lora_down|lora_A|lora_B)(\.default)?\.weight$", "", normalized)
    normalized = re.sub(r"^lora_(?:unet|te|te1|te2)_", "", normalized)
    normalized = re.sub(r"_(\d+)_", r".\1.", normalized)
    return normalized


def _normalize_analysis_key(
    key: str, is_lora: bool, key_correction: bool = False
) -> str:
    """分析用の正規化キーを返す。

    key_correction=True の場合:
      - LoRA: lora_target_candidates 経由で anima-base-v1.0 形式に変換
      - ベースモデル: anima_v1_key() で net. プレフィックスを付与
    False の場合: 従来動作（canonical_key ベース）
    """
    if is_lora:
        return _lora_base_key(key, key_correction=key_correction)
    if key_correction:
        return anima_v1_key(key)
    return canonical_key(key)


def _group_label(norm_key: str, mode: str) -> str:
    """正規化キーから表示グループ名を返す（adjustment_group と同ロジック）。"""
    try:
        return adjustment_group(norm_key, mode)
    except Exception:
        return "Other"


# ─── 段階的テンソルイテレータ ─────────────────────────────────────────────────

def _iter_analysis_tensors(
    state_dict: dict[str, Any],
    is_lora: bool,
    layer_mode: str,
    device: str,
    progress: Callable[[str], None],
    torch: Any,
    key_correction: bool = False,
) -> Iterator[tuple[str, str, str, Any]]:
    """
    state_dict を 1テンソルずつ yield する段階的イテレータ。
    Yield: (original_key, norm_key, group_label, tensor_on_cpu)

    - マージ対象外キー（CLIP, VAE 等）はスキップ
    - LoRA の場合は .alpha をスキップ（数値スカラーのみ）
    - 各テンソルをCPUに退避後 VRAM キャッシュを解放
    - total はスキップ対象を除いた実処理数を事前計算して使用
    """
    # スキップ対象を除いた実処理数を事前計算（進捗表示をrecord_countと一致させる）
    def _is_target(k: str, t: object) -> bool:
        if is_lora and k.endswith(".alpha"):
            return False
        norm = _normalize_analysis_key(k, is_lora, key_correction)
        if not is_merge_target(norm):
            return False
        if not hasattr(t, "detach"):
            return False
        return True

    total = sum(1 for k, t in state_dict.items() if _is_target(k, t))

    processed = 0
    for orig_key, tensor in state_dict.items():
        if is_lora and orig_key.endswith(".alpha"):
            continue

        norm_key = _normalize_analysis_key(orig_key, is_lora, key_correction)

        if not is_merge_target(norm_key):
            continue

        if not hasattr(tensor, "detach"):
            continue

        try:
            calc_tensor = tensor.detach().to(device).float()
        except Exception as exc:
            progress(_("analysis_log_tensor_fail", key=orig_key, error=exc))
            continue

        group = _group_label(norm_key, layer_mode)
        processed += 1

        if processed % 50 == 0 or processed == total:
            try:
                if device.startswith("cuda") and hasattr(torch, "cuda"):
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    progress(_("analysis_log_analyzing_vram", done=processed, total=total, vram=allocated))
                else:
                    progress(_("analysis_log_analyzing", done=processed, total=total))
            except Exception:
                progress(f"分析中 {processed}/{total} 層")

        yield orig_key, norm_key, group, calc_tensor

        del calc_tensor
        if device.startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        gc.collect()


# ─── 各分析手法の演算 ────────────────────────────────────────────────────────

def _analyze_feature_map(torch: Any, t: Any) -> dict[str, float]:
    flat = t.flatten()
    mean = float(flat.mean())
    var = float(flat.var())
    std = float(flat.std())
    complexity = std / (abs(mean) + 1e-8)
    return {"feat_mean": mean, "feat_var": var, "feat_complexity": complexity}


def _analyze_statistical(torch: Any, t: Any) -> dict[str, float]:
    flat = t.flatten()
    l2 = float(torch.linalg.vector_norm(flat))
    mean = float(flat.mean())
    var = float(flat.var())
    return {"stat_l2": l2, "stat_mean": mean, "stat_var": var}


def _analyze_svd_rank(torch: Any, t: Any) -> dict[str, Any]:
    # 2D に変形（非2Dは最初の次元×残り）
    if t.ndim < 2:
        return {
            "svd_effective_rank": 1,
            "svd_decay_rate": 1.0,
            "svd_cumvar_at_threshold": 1.0,
            "svd_singular_values": [],
        }
    mat = t.reshape(t.shape[0], -1)
    try:
        _, s, _ = torch.linalg.svd(mat, full_matrices=False)
    except Exception:
        return {
            "svd_effective_rank": 0,
            "svd_decay_rate": 0.0,
            "svd_cumvar_at_threshold": 0.0,
            "svd_singular_values": [],
        }
    s_vals = s.tolist()
    total_energy = float(s.pow(2).sum())
    if total_energy == 0.0:
        return {
            "svd_effective_rank": 0,
            "svd_decay_rate": 0.0,
            "svd_cumvar_at_threshold": 0.0,
            "svd_singular_values": s_vals[:64],
        }
    cumvar = torch.cumsum(s.pow(2), dim=0) / total_energy
    effective_rank = int((cumvar < _SVD_CUMVAR_THRESHOLD).sum().item()) + 1
    effective_rank = min(effective_rank, len(s_vals))
    decay_rate = float(s[effective_rank - 1]) / (float(s[0]) + 1e-12)
    cumvar_at = float(cumvar[effective_rank - 1])
    return {
        "svd_effective_rank": effective_rank,
        "svd_decay_rate": decay_rate,
        "svd_cumvar_at_threshold": cumvar_at,
        "svd_singular_values": s_vals[:64],  # 最大64値を保存
    }


def _analyze_attention_map(torch: Any, t: Any, norm_key: str) -> dict[str, Any]:
    is_attn = component_category(norm_key) == "attention"
    if not is_attn:
        return {
            "attn_mean_weight": 0.0,
            "attn_head_variance": 0.0,
            "is_attention_layer": False,
        }
    flat = t.flatten().abs()
    mean_w = float(flat.mean())
    # マルチヘッド分散の推定: テンソルを行方向にスライスして行間分散を計算
    if t.ndim >= 2:
        row_means = t.reshape(t.shape[0], -1).mean(dim=1)
        head_var = float(row_means.var())
    else:
        head_var = 0.0
    return {
        "attn_mean_weight": mean_w,
        "attn_head_variance": head_var,
        "is_attention_layer": True,
    }


def _run_method(
    torch: Any,
    method: str,
    t: Any,
    norm_key: str,
) -> dict[str, Any]:
    if method == "Feature Map":
        return _analyze_feature_map(torch, t)
    if method == "Statistical":
        return _analyze_statistical(torch, t)
    if method == "SVD Rank":
        return _analyze_svd_rank(torch, t)
    if method == "Attention Map":
        return _analyze_attention_map(torch, t, norm_key)
    return {}


# ─── グループ集計 ─────────────────────────────────────────────────────────────

def _aggregate_groups(
    records: list[LayerRecord],
    method: str,
    torch: Any,
) -> dict[str, dict[str, float]]:
    """
    レイヤーレコードをグループ単位で集計し、
    {group_label: {metric: value}} を返す。
    ログのサマリーセクションに使用。
    """
    from collections import defaultdict
    groups: dict[str, list[LayerRecord]] = defaultdict(list)
    for r in records:
        groups[r.group].append(r)

    result: dict[str, dict[str, float]] = {}
    for group, recs in groups.items():
        n = len(recs)
        if method == "Feature Map":
            result[group] = {
                "mean_feat_mean": sum(r.feat_mean for r in recs) / n,
                "mean_feat_var": sum(r.feat_var for r in recs) / n,
                "mean_feat_complexity": sum(r.feat_complexity for r in recs) / n,
                "layer_count": n,
            }
        elif method == "Statistical":
            l2s = [r.stat_l2 for r in recs]
            result[group] = {
                "mean_l2": sum(l2s) / n,
                "max_l2": max(l2s),
                "min_l2": min(l2s),
                "mean_mean": sum(r.stat_mean for r in recs) / n,
                "mean_var": sum(r.stat_var for r in recs) / n,
                "layer_count": n,
            }
        elif method == "SVD Rank":
            result[group] = {
                "mean_effective_rank": sum(r.svd_effective_rank for r in recs) / n,
                "mean_decay_rate": sum(r.svd_decay_rate for r in recs) / n,
                "mean_cumvar": sum(r.svd_cumvar_at_threshold for r in recs) / n,
                "layer_count": n,
            }
        elif method == "Attention Map":
            attn_recs = [r for r in recs if r.is_attention_layer]
            if attn_recs:
                na = len(attn_recs)
                result[group] = {
                    "mean_attn_weight": sum(r.attn_mean_weight for r in attn_recs) / na,
                    "mean_head_variance": sum(r.attn_head_variance for r in attn_recs) / na,
                    "attention_layer_count": na,
                    "layer_count": n,
                }
            else:
                result[group] = {"attention_layer_count": 0, "layer_count": n}
    return result


# ─── ルールベース自動レポート ─────────────────────────────────────────────────

def _generate_auto_report(
    records: list[LayerRecord],
    method: str,
    aggregated: dict[str, dict[str, float]],
) -> list[str]:
    """
    ルールベース判定を実行しテキスト行を返す。
    Statistical / SVD Rank は判定パターンA（学習の偏り）を適用。
    モデル特徴・マージ用ヒント情報を含む拡張レポートを生成。
    """
    lines: list[str] = []

    # ── 判定パターンA: Middle Block の偏り検出 ──────────────────────────
    if method in ("Statistical", "SVD Rank"):
        # グループを block ベースにまとめる
        block_scores: dict[str, list[float]] = {"Input": [], "Middle": [], "Output": []}
        for group, agg in aggregated.items():
            block_token = None
            for b in ("Input", "Middle", "Output"):
                if b in group:
                    block_token = b
                    break
            if block_token is None:
                continue
            if method == "Statistical":
                score = agg.get("mean_l2", 0.0)
            else:  # SVD Rank
                score = agg.get("mean_effective_rank", 0.0)
            block_scores[block_token].append(score)

        block_means = {b: (sum(v) / len(v)) if v else 0.0 for b, v in block_scores.items()}
        all_means = [v for v in block_means.values() if v > 0]
        if all_means and block_means["Middle"] > 0:
            global_mean = sum(all_means) / len(all_means)
            if block_means["Middle"] > global_mean * 1.3:
                lines.append(_("analysis_report_pattern_a"))

    # ── 統計: 異常値レイヤー検出 ────────────────────────────────────────
    if method == "Statistical":
        l2s = [r.stat_l2 for r in records if r.stat_l2 > 0]
        if len(l2s) >= 4:
            sorted_l2 = sorted(l2s)
            q1 = sorted_l2[len(sorted_l2) // 4]
            q3 = sorted_l2[(len(sorted_l2) * 3) // 4]
            iqr = q3 - q1
            threshold = q3 + _OUTLIER_IQR_FACTOR * iqr
            outliers = [r for r in records if r.stat_l2 > threshold]
            if outliers:
                keys_str = ", ".join(r.key[:60] for r in outliers[:5])
                lines.append(
                    _("analysis_report_outlier", threshold=threshold, count=len(outliers),
                      keys=keys_str + ("..." if len(outliers) > 5 else ""))
                )

    # ── SVD: 高冗長レイヤー検出 ─────────────────────────────────────────
    if method == "SVD Rank":
        high_decay = [r for r in records if r.svd_decay_rate < 0.05 and r.svd_effective_rank > 0]
        if high_decay:
            lines.append(
                _("analysis_report_redundant", count=len(high_decay))
            )

    if not lines:
        lines.append(_("analysis_report_no_issue"))

    # ── モデル特徴サマリー（マージ・学習共通ヒント） ──────────────
    lines.append("")
    lines.append(_("analysis_report_model_feature"))

    if records and aggregated:
        total = len(records)

        # --- グループサマリーベース: マージ優先(group) / 破棄候補(group) ---
        priority_groups, discard_groups = _rank_groups_by_score(aggregated, method)
        top_g = max(1, len(aggregated) // 3)

        lines.append(_("analysis_report_merge_priority_group"))
        for g in priority_groups[:top_g]:
            lines.append(f"  {g}")

        lines.append(_("analysis_report_discard_group"))
        for g in discard_groups[:top_g]:
            lines.append(f"  {g}")

        # --- レイヤーレベル: マージ優先(full) / 破棄候補(full) ---
        lines.append("")
        lines.append(_("analysis_report_merge_priority_full"))
        merge_priority = _pick_merge_priority(records, method)
        for r in merge_priority:
            lines.append(f"  {r.key}  ({r.group})")

        lines.append(_("analysis_report_discard_full"))
        discard_cands = _pick_discard_candidates(records, method)
        for r in discard_cands:
            lines.append(f"  {r.key}  ({r.group})")

        # --- LoRA ランク推奨 (SVD のみ) ---
        if method == "SVD Rank":
            median_rank = sorted(r.svd_effective_rank for r in records)[total // 2]
            lines.append("")
            lines.append(_("analysis_report_lora_rank", rank=median_rank))
            lines.append(_("analysis_report_lora_rank_rec", rank=_recommend_lora_rank(median_rank)))

    return lines


# ─── マージヒントユーティリティ ──────────────────────────────────────────────

def _rank_groups_by_score(
    aggregated: dict[str, dict[str, float]], method: str
) -> tuple[list[str], list[str]]:
    """
    aggregated グループを主スコアでソートし、
    (優先順グループ名リスト, 破棄候補順グループ名リスト) を返す。
    グループ名は表示レイヤーそのまま (例: Input_Attention, Transformer_0 等)。
    """
    def _score(agg: dict) -> float:
        if method == "Statistical":
            return agg.get("mean_l2", 0.0)
        if method == "SVD Rank":
            return agg.get("mean_effective_rank", 0.0)
        if method == "Feature Map":
            return agg.get("mean_feat_complexity", 0.0)
        if method == "Attention Map":
            return agg.get("mean_attn_weight", 0.0)
        return 0.0

    scored = sorted(aggregated.items(), key=lambda kv: _score(kv[1]), reverse=True)
    priority = [g for g, _ in scored]
    discard = list(reversed(priority))
    return priority, discard


def _pick_merge_priority(records: list[LayerRecord], method: str) -> list[LayerRecord]:
    """マージ優先(full): 情報密度が高く、モデル表現への寄与が大きい層を返す。"""
    n = max(1, len(records) // 10)
    if method == "Statistical":
        return sorted(records, key=lambda r: r.stat_l2, reverse=True)[:n]
    if method == "SVD Rank":
        return sorted(records, key=lambda r: r.svd_effective_rank, reverse=True)[:n]
    if method == "Feature Map":
        return sorted(records, key=lambda r: r.feat_complexity, reverse=True)[:n]
    if method == "Attention Map":
        attn = [r for r in records if r.is_attention_layer]
        return sorted(attn, key=lambda r: r.attn_mean_weight, reverse=True)[:n]
    return []


def _pick_discard_candidates(records: list[LayerRecord], method: str) -> list[LayerRecord]:
    """切り捨て候補層: 情報密度が低く、マージ時にBase側で上書き可能な層を返す。"""
    n = max(1, len(records) // 10)
    if method == "Statistical":
        return sorted(records, key=lambda r: r.stat_l2)[:n]
    if method == "SVD Rank":
        # 有効ランクが低く減衰率も低い → 高冗長
        return sorted(records, key=lambda r: (r.svd_effective_rank, r.svd_decay_rate))[:n]
    if method == "Feature Map":
        return sorted(records, key=lambda r: r.feat_complexity)[:n]
    if method == "Attention Map":
        non_attn = [r for r in records if not r.is_attention_layer]
        return non_attn[:n]
    return []


def _recommend_lora_rank(median_rank: int) -> int:
    """中央有効ランクから推奨 LoRA ランクを返す。"""
    for rank in (4, 8, 16, 32, 64, 128):
        if median_rank <= rank * 2:
            return rank
    return 128


def _collect_important_keys(
    records: list[LayerRecord], method: str
) -> tuple[list[str], list[str]]:
    """重要層キー・非重要層キーを (important, unimportant) として返す。"""
    priority = _pick_merge_priority(records, method)
    discard = _pick_discard_candidates(records, method)
    p_keys = [r.key for r in priority]
    d_keys = [r.key for r in discard if r.key not in p_keys]
    return p_keys, d_keys


# ─── ログ保存 ─────────────────────────────────────────────────────────────────

_LOG_FORMAT_VERSION = "2.0"

_METADATA_SEPARATOR = "=" * 72


def _build_log_text(report: AnalysisReport, aggregated: dict[str, dict[str, float]]) -> str:
    """
    ログファイルのテキストを生成する。

    構造:
      [METADATA]  ← Preview / Compare が必ず読み取るヘッダーブロック
      [AUTO_REPORT]
      [GROUP_SUMMARY]
      [LAYER_RECORDS]
    """
    lines: list[str] = []

    # ── METADATA ────────────────────────────────────────────────────────
    meta = {
        "format_version": _LOG_FORMAT_VERSION,
        "model_name": report.model_name,
        "model_type": report.model_type,
        "method": report.method,
        "layer_mode": report.layer_mode,
        "device": report.device,
        "sha256": report.sha256,
        "timestamp": report.timestamp,
        "record_count": len(report.records),
        "warnings_count": len(report.warnings),
    }
    lines.append("[METADATA]")
    lines.append(_METADATA_SEPARATOR)
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append(_METADATA_SEPARATOR)
    lines.append("")

    # ── AUTO_REPORT ──────────────────────────────────────────────────────
    lines.append("[AUTO_REPORT]")
    lines.append(_METADATA_SEPARATOR)
    for line in report.auto_report_lines:
        lines.append(line)
    lines.append(_METADATA_SEPARATOR)
    lines.append("")

    # ── GROUP_SUMMARY ────────────────────────────────────────────────────
    lines.append("[GROUP_SUMMARY]")
    lines.append(_METADATA_SEPARATOR)
    for group, agg in sorted(aggregated.items()):
        agg_json = json.dumps(agg, ensure_ascii=False)
        lines.append(f"{group}\t{agg_json}")
    lines.append(_METADATA_SEPARATOR)
    lines.append("")

    # ── LAYER_RECORDS ─────────────────────────────────────────────────────
    lines.append("[LAYER_RECORDS]")
    lines.append(_METADATA_SEPARATOR)
    # ヘッダー行
    if report.method == "Feature Map":
        lines.append("key\toriginal_key\tgroup\tshape\tfeat_mean\tfeat_var\tfeat_complexity")
        for r in report.records:
            shape_str = "x".join(str(d) for d in r.shape)
            lines.append(
                f"{r.key}\t{r.original_key}\t{r.group}\t{shape_str}"
                f"\t{r.feat_mean:.6f}\t{r.feat_var:.6f}\t{r.feat_complexity:.6f}"
            )
    elif report.method == "Statistical":
        lines.append("key\toriginal_key\tgroup\tshape\tl2\tmean\tvar")
        for r in report.records:
            shape_str = "x".join(str(d) for d in r.shape)
            lines.append(
                f"{r.key}\t{r.original_key}\t{r.group}\t{shape_str}"
                f"\t{r.stat_l2:.6f}\t{r.stat_mean:.6f}\t{r.stat_var:.6f}"
            )
    elif report.method == "SVD Rank":
        lines.append("key\toriginal_key\tgroup\tshape\teffective_rank\tdecay_rate\tcumvar\tsingular_values_head")
        for r in report.records:
            shape_str = "x".join(str(d) for d in r.shape)
            sv_str = ",".join(f"{v:.4f}" for v in r.svd_singular_values[:16])
            lines.append(
                f"{r.key}\t{r.original_key}\t{r.group}\t{shape_str}"
                f"\t{r.svd_effective_rank}\t{r.svd_decay_rate:.6f}"
                f"\t{r.svd_cumvar_at_threshold:.6f}\t{sv_str}"
            )
    elif report.method == "Attention Map":
        lines.append("key\toriginal_key\tgroup\tshape\tis_attention\tattn_mean_weight\thead_variance")
        for r in report.records:
            shape_str = "x".join(str(d) for d in r.shape)
            lines.append(
                f"{r.key}\t{r.original_key}\t{r.group}\t{shape_str}"
                f"\t{int(r.is_attention_layer)}\t{r.attn_mean_weight:.6f}\t{r.attn_head_variance:.6f}"
            )
    lines.append(_METADATA_SEPARATOR)

    # ── MERGE_HINTS ──────────────────────────────────────────────────────
    if report.important_layer_keys or report.unimportant_layer_keys:
        lines.append("[MERGE_HINTS]")
        lines.append(_METADATA_SEPARATOR)
        lines.append(_("analysis_report_merge_hint_priority"))
        for k in report.important_layer_keys:
            lines.append(f"PRIORITY(full): {k}")
        lines.append(_("analysis_report_merge_hint_discard"))
        for k in report.unimportant_layer_keys:
            lines.append(f"DISCARD(full):  {k}")
        lines.append(_METADATA_SEPARATOR)
        lines.append("")

    # ── WARNINGS ─────────────────────────────────────────────────────────
    if report.warnings:
        lines.append("")
        lines.append("[WARNINGS]")
        lines.append(_METADATA_SEPARATOR)
        for w in report.warnings:
            lines.append(w)
        lines.append(_METADATA_SEPARATOR)

    return "\n".join(lines)


def save_analysis_log(report: AnalysisReport, log_dir: Path, aggregated: dict[str, dict[str, float]]) -> Path:
    """
    ログを {model_name}_{model|lora}_{method}_{layer_mode}.txt として保存する。
    ファイル名に使えない文字をアンダースコアに置換する。
    戻り値: 保存したファイルのパス
    """
    def sanitize(s: str) -> str:
        return re.sub(r"[^\w\-]", "_", s)

    method_tag = sanitize(report.method.replace(" ", "_"))
    layer_tag = sanitize(report.layer_mode)
    filename = f"{sanitize(report.model_name)}_{report.model_type}_{method_tag}_{layer_tag}.txt"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    log_path.write_text(_build_log_text(report, aggregated), encoding="utf-8")
    return log_path


# ─── ログ読み込み（Preview / Compare 向け） ───────────────────────────────────

def load_analysis_log(log_path: Path) -> dict[str, Any]:
    """
    save_analysis_log() が出力したテキストを読み込み、
    構造化辞書として返す。

    戻り値:
      {
        "metadata": dict,
        "auto_report": list[str],
        "group_summary": dict[str, dict],
        "layer_records": list[dict],
        "warnings": list[str],
      }

    フォーマット不一致の場合は ValueError を送出。
    """
    text = log_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _extract_section(section_tag: str) -> list[str]:
        start = None
        content: list[str] = []
        in_section = False
        sep_count = 0
        for line in lines:
            if line == f"[{section_tag}]":
                in_section = True
                sep_count = 0
                continue
            if in_section:
                if line == _METADATA_SEPARATOR:
                    sep_count += 1
                    if sep_count == 2:
                        break
                    continue
                content.append(line)
        return content

    # METADATA の存在確認（整合性検証）
    if "[METADATA]" not in text:
        raise ValueError(_("log_format_err", path=log_path))

    # METADATA パース
    meta_lines = _extract_section("METADATA")
    if not meta_lines:
        raise ValueError(_("log_metadata_empty", path=log_path))
    metadata: dict[str, Any] = {}
    for line in meta_lines:
        if ": " in line:
            k, _, v = line.partition(": ")
            metadata[k.strip()] = v.strip()

    if metadata.get("format_version") != _LOG_FORMAT_VERSION:
        raise ValueError(
            _("log_version_mismatch",
              expected=_LOG_FORMAT_VERSION,
              got=metadata.get('format_version'),
              path=log_path)
        )

    # AUTO_REPORT
    auto_report = _extract_section("AUTO_REPORT")

    # GROUP_SUMMARY
    group_lines = _extract_section("GROUP_SUMMARY")
    group_summary: dict[str, dict] = {}
    for line in group_lines:
        if "\t" in line:
            group_name, _, json_str = line.partition("\t")
            try:
                group_summary[group_name] = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # LAYER_RECORDS
    record_lines = _extract_section("LAYER_RECORDS")
    layer_records: list[dict] = []
    header: list[str] = []
    for line in record_lines:
        if not header:
            header = line.split("\t")
            continue
        cols = line.split("\t")
        if len(cols) == len(header):
            layer_records.append(dict(zip(header, cols)))

    # WARNINGS
    warnings: list[str] = []
    if "[WARNINGS]" in text:
        warnings = _extract_section("WARNINGS")

    return {
        "metadata": metadata,
        "auto_report": auto_report,
        "group_summary": group_summary,
        "layer_records": layer_records,
        "warnings": warnings,
    }


# ─── メイン分析エントリポイント ───────────────────────────────────────────────

def run_analysis(
    model_path: Path,
    method: str,
    layer_mode: str,
    log_dir: Path,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
    key_correction: bool = False,
) -> AnalysisReport:
    """
    レイヤー分析を実行し AnalysisReport を返す。
    同時に log_dir へログファイルを保存する。

    Parameters
    ----------
    model_path : Path
        分析対象のモデル/LoRAファイルパス
    method : str
        ANALYSIS_METHODS のいずれか
    layer_mode : str
        LAYER_DISPLAY_MODES のいずれか
    log_dir : Path
        ログ保存先ディレクトリ（AppPaths.log_analysis）
    device : str
        "cpu" or "cuda"
    progress : Callable[[str], None] | None
        進捗メッセージを受け取るコールバック（GUIのlog_queue.put 等）
    key_correction : bool
        True の場合、分析前にキーを anima-base-v1.0 形式に正規化する。
        LoRA: lora_target_candidates() で net.blocks.N.xxx.weight 形式に変換。
        ベースモデル: anima_v1_key() で net. プレフィックスを付与。
    """
    from .model_io import require_torch

    if method not in ANALYSIS_METHODS:
        raise ValueError(_("analysis_warn_bad_method", method=method, choices=ANALYSIS_METHODS))
    if layer_mode not in LAYER_DISPLAY_MODES:
        raise ValueError(_("analysis_warn_bad_mode", mode=layer_mode, choices=LAYER_DISPLAY_MODES))

    log = progress or (lambda _: None)
    torch = require_torch()

    # CUDA フォールバック
    if device.startswith("cuda") and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
        log(_("analysis_warn_cuda_fallback"))
        device = "cpu"

    validate_model_path(model_path)
    log(_("analysis_log_loading", name=model_path.name))
    state_dict = load_state_dict(model_path, device)

    is_lora = _is_lora_state_dict(state_dict)
    model_type = "lora" if is_lora else "model"
    model_name = model_path.stem
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    log(_("analysis_log_sha256"))
    sha = sha256_file(model_path)

    report = AnalysisReport(
        model_name=model_name,
        model_type=model_type,
        method=method,
        layer_mode=layer_mode,
        device=device,
        sha256=sha,
        timestamp=timestamp,
    )

    log(_("analysis_log_start", method=method, mode=layer_mode, type=model_type, name=model_name))

    if key_correction:
        log(_("analysis_log_key_correct"))

    # 段階的レイヤー分析
    for orig_key, norm_key, group, cpu_tensor in _iter_analysis_tensors(
        state_dict, is_lora, layer_mode, device, log, torch,
        key_correction=key_correction,
    ):
        shape = tuple(cpu_tensor.shape)

        rec = LayerRecord(
            key=norm_key,
            original_key=orig_key,
            group=group,
            shape=shape,
        )

        try:
            result = _run_method(torch, method, cpu_tensor, norm_key)
        except Exception as exc:
            report.warnings.append(_("analysis_log_fail", key=orig_key, error=exc))
            continue

        # 結果をレコードに書き込む
        for attr, val in result.items():
            if hasattr(rec, attr):
                setattr(rec, attr, val)

        report.records.append(rec)

    del state_dict
    gc.collect()
    if device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()

    if not report.records:
        raise ValueError(_("analysis_warn_no_tensors"))

    log(_("analysis_log_aggregate", count=len(report.records)))
    aggregated = _aggregate_groups(report.records, method, torch)

    log(_("analysis_log_report"))
    report.auto_report_lines = _generate_auto_report(report.records, method, aggregated)
    imp_keys, unimp_keys = _collect_important_keys(report.records, method)
    report.important_layer_keys = imp_keys
    report.unimportant_layer_keys = unimp_keys

    log(_("analysis_log_saving"))
    log_path = save_analysis_log(report, log_dir, aggregated)
    report.log_path = log_path
    log(_("analysis_log_saved", name=log_path.name))

    return report
