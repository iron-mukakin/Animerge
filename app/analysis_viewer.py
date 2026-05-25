"""
analysis_viewer.py  –  Anima Model Editor v2.0
Tab 4: 詳細分析ビュアー (Analysis Viewer)

内タブ構成:
  4-1 分析プレビュー   : log_analysis/ 内のログをプルダウン選択 → グラフ/ヒートマップ表示
  4-2 レポート比較     : 同一手法ログ2本を読み込み → 相性スコア・差分チャート表示

依存:
  - analysis.py の load_analysis_log() でログ読み込み
  - tkinter + tkinter.ttk (標準)
  - tkinter.Canvas ベースのグラフ描画 (matplotlib不要)

gui.py 側での組み込み方法:
  from .analysis_viewer import build_viewer_tab
  viewer_frame = ttk.Frame(self.main_notebook, padding=4)
  self.main_notebook.add(viewer_frame, text="  詳細分析  ")
  build_viewer_tab(viewer_frame, paths, log_fn)
"""

from __future__ import annotations

import math
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

# ─── 定数 ─────────────────────────────────────────────────────────────────────

_LOG_SUFFIX = ".txt"
_GRAPH_BG = "#1E293B"
_GRAPH_FG = "#F1F5F9"
_COLOR_HIGH = "#F97316"   # 高値/重要 (オレンジ)
_COLOR_MID  = "#38BDF8"   # 中間 (水色)
_COLOR_LOW  = "#22D3EE"   # 低値 (シアン)
_COLOR_WARN = "#EF4444"   # 警告/乖離 (赤)
_COLOR_OK   = "#4ADE80"   # 正常 (緑)
_COLOR_LINE_A = "#818CF8"  # 比較ライン A (紫)
_COLOR_LINE_B = "#FB923C"  # 比較ライン B (オレンジ)

_METHOD_METRICS: dict[str, str] = {
    "Feature_Map": "mean_feat_complexity",
    "Statistical":  "mean_l2",
    "SVD_Rank":     "mean_effective_rank",
    "Attention_Map": "mean_attn_weight",
}

# ─── エントリポイント ──────────────────────────────────────────────────────────

def build_viewer_tab(
    parent: ttk.Frame,
    log_dir: Path,
    log_fn: Callable[[str], None],
) -> None:
    """
    gui.py から呼び出されるエントリポイント。
    parent フレーム内に内タブを構築する。

    Parameters
    ----------
    parent  : タブ4のベースフレーム
    log_dir : AppPaths.log_analysis (= log/log_analysis/)
    log_fn  : gui.py の self.log 相当
    """
    sub = ttk.Notebook(parent, style="SubTab.TNotebook")
    sub.pack(fill=tk.BOTH, expand=True)

    preview_frame = ttk.Frame(sub, padding=4)
    compare_frame = ttk.Frame(sub, padding=4)
    sub.add(preview_frame, text="  4-1 分析プレビュー  ")
    sub.add(compare_frame, text="  4-2 レポート比較  ")

    _build_preview_tab(preview_frame, log_dir, log_fn)
    _build_compare_tab(compare_frame, log_dir, log_fn)


# ─── ユーティリティ ───────────────────────────────────────────────────────────

def _scan_logs(log_dir: Path) -> list[str]:
    """log_dir 内の .txt ファイル一覧を返す（新しい順）。"""
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob(f"*{_LOG_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


def _safe_load(log_dir: Path, filename: str, log_fn: Callable) -> dict[str, Any] | None:
    if not filename:
        return None
    path = log_dir / filename
    try:
        from .analysis import load_analysis_log
        return load_analysis_log(path)
    except ImportError:
        # スタンドアロンテスト用フォールバック: ダミーデータ
        return _dummy_data(filename)
    except Exception as exc:
        log_fn(f"[Viewer] ログ読み込み失敗: {exc}")
        messagebox.showerror("読み込みエラー", str(exc))
        return None


def _dummy_data(filename: str) -> dict[str, Any]:
    """テスト用ダミーデータ（analysis.py が利用不可の環境向け）。"""
    import random
    rng = random.Random(hash(filename) & 0xFFFF)
    method = "Statistical"
    for tag in ("Feature_Map", "Statistical", "SVD_Rank", "Attention_Map"):
        if tag in filename:
            method = tag.replace("_", " ")
            break

    groups = ["Input_Attention", "Input_MLP", "Middle_Attention", "Middle_MLP",
              "Output_Attention", "Output_MLP", "Other"]
    group_summary: dict[str, dict] = {}
    for g in groups:
        group_summary[g] = {
            "mean_l2": rng.uniform(0.5, 8.0),
            "mean_feat_complexity": rng.uniform(0.1, 3.0),
            "mean_effective_rank": rng.randint(4, 64),
            "mean_attn_weight": rng.uniform(0.0, 0.5),
            "layer_count": rng.randint(4, 20),
        }

    records = []
    for g in groups:
        for i in range(rng.randint(3, 8)):
            records.append({
                "key": f"{g.lower()}.layer_{i}.weight",
                "group": g,
                "stat_l2": str(rng.uniform(0.3, 10.0)),
                "feat_complexity": str(rng.uniform(0.05, 3.5)),
                "svd_effective_rank": str(rng.randint(2, 64)),
                "attn_mean_weight": str(rng.uniform(0.0, 0.6)),
                "is_attention_layer": str("Attention" in g),
            })

    return {
        "metadata": {
            "model_name": filename.replace(_LOG_SUFFIX, ""),
            "model_type": "model",
            "method": method,
            "layer_mode": "Matrix",
            "timestamp": "2025-01-01T00:00:00",
            "sha256": "dummy",
            "format_version": "1.0",
        },
        "auto_report": [
            "[自動レポート] ダミーデータによるプレビュー表示です。",
            "実際のモデルファイルをレイヤー分析タブで分析してください。",
        ],
        "group_summary": group_summary,
        "layer_records": records,
        "warnings": [],
    }


def _method_tag(data: dict) -> str:
    """メタデータからメソッドタグ（アンダースコア区切り）を返す。"""
    return data["metadata"].get("method", "Statistical").replace(" ", "_")


# ─── Canvas ベース グラフ描画ユーティリティ ───────────────────────────────────

class _BarChart:
    """シンプルな横棒グラフを Canvas に描画する。"""

    def __init__(self, canvas: tk.Canvas, labels: list[str], values: list[float],
                 colors: list[str] | None = None, title: str = "") -> None:
        self.canvas = canvas
        self.labels = labels
        self.values = values
        self.colors = colors or [_COLOR_MID] * len(values)
        self.title = title

    def draw(self, w: int, h: int) -> None:
        c = self.canvas
        c.delete("all")
        if not self.values:
            c.create_text(w // 2, h // 2, text="データなし", fill=_GRAPH_FG)
            return

        pad_l, pad_r, pad_t, pad_b = 130, 20, 30, 20
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        # タイトル
        if self.title:
            c.create_text(pad_l + chart_w // 2, 12, text=self.title,
                          fill=_GRAPH_FG, font=("TkDefaultFont", 9, "bold"))

        max_val = max(abs(v) for v in self.values) or 1.0
        bar_h = max(6, chart_h // len(self.values) - 4)

        for i, (label, val, col) in enumerate(zip(self.labels, self.values, self.colors)):
            y_center = pad_t + i * (chart_h / len(self.values)) + bar_h // 2
            bar_w = int(chart_w * abs(val) / max_val)
            x0 = pad_l
            y0 = y_center - bar_h // 2
            y1 = y_center + bar_h // 2

            c.create_rectangle(x0, y0, x0 + bar_w, y1, fill=col, outline="")
            # ラベル
            short = label if len(label) <= 18 else label[:15] + "..."
            c.create_text(pad_l - 4, y_center, text=short, fill=_GRAPH_FG,
                          anchor=tk.E, font=("TkFixedFont", 8))
            # 値
            c.create_text(x0 + bar_w + 4, y_center,
                          text=f"{val:.3f}", fill=_GRAPH_FG,
                          anchor=tk.W, font=("TkFixedFont", 8))


class _LineChart:
    """2系列の折れ線グラフを Canvas に描画する。"""

    def __init__(self, canvas: tk.Canvas,
                 values_a: list[float], values_b: list[float] | None,
                 labels: list[str] | None = None,
                 highlight_indices: list[int] | None = None,
                 title: str = "",
                 label_a: str = "A", label_b: str = "B") -> None:
        self.canvas = canvas
        self.values_a = values_a
        self.values_b = values_b
        self.labels = labels
        self.highlight = set(highlight_indices or [])
        self.title = title
        self.label_a = label_a
        self.label_b = label_b

    def draw(self, w: int, h: int) -> None:
        c = self.canvas
        c.delete("all")
        if not self.values_a:
            c.create_text(w // 2, h // 2, text="データなし", fill=_GRAPH_FG)
            return

        pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 40
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        all_vals = list(self.values_a)
        if self.values_b:
            all_vals += self.values_b
        min_v = min(all_vals)
        max_v = max(all_vals)
        span = max_v - min_v or 1.0

        n = len(self.values_a)

        def _px(i: int, v: float):
            x = pad_l + i * chart_w / max(n - 1, 1)
            y = pad_t + chart_h - (v - min_v) / span * chart_h
            return x, y

        # タイトル
        if self.title:
            c.create_text(pad_l + chart_w // 2, 14, text=self.title,
                          fill=_GRAPH_FG, font=("TkDefaultFont", 9, "bold"))

        # 軸
        c.create_line(pad_l, pad_t, pad_l, pad_t + chart_h, fill="#475569")
        c.create_line(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h, fill="#475569")

        # ハイライト背景
        for idx in self.highlight:
            if 0 <= idx < n:
                x = pad_l + idx * chart_w / max(n - 1, 1)
                c.create_rectangle(x - 2, pad_t, x + 2, pad_t + chart_h,
                                   fill=_COLOR_WARN, outline="", stipple="gray25")

        # 系列 A
        pts_a = [_px(i, v) for i, v in enumerate(self.values_a)]
        if len(pts_a) >= 2:
            flat = [coord for pt in pts_a for coord in pt]
            c.create_line(*flat, fill=_COLOR_LINE_A, width=2)
        for x, y in pts_a:
            c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=_COLOR_LINE_A, outline="")

        # 系列 B
        if self.values_b and len(self.values_b) == n:
            pts_b = [_px(i, v) for i, v in enumerate(self.values_b)]
            if len(pts_b) >= 2:
                flat = [coord for pt in pts_b for coord in pt]
                c.create_line(*flat, fill=_COLOR_LINE_B, width=2, dash=(4, 2))
            for x, y in pts_b:
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=_COLOR_LINE_B, outline="")

        # 凡例
        leg_x = pad_l + chart_w - 100
        c.create_line(leg_x, pad_t + 8, leg_x + 16, pad_t + 8, fill=_COLOR_LINE_A, width=2)
        c.create_text(leg_x + 20, pad_t + 8, text=self.label_a, fill=_COLOR_LINE_A,
                      anchor=tk.W, font=("TkFixedFont", 8))
        if self.values_b:
            c.create_line(leg_x, pad_t + 22, leg_x + 16, pad_t + 22, fill=_COLOR_LINE_B, width=2, dash=(4, 2))
            c.create_text(leg_x + 20, pad_t + 22, text=self.label_b, fill=_COLOR_LINE_B,
                          anchor=tk.W, font=("TkFixedFont", 8))

        # Y軸ラベル（最小・最大）
        c.create_text(pad_l - 4, pad_t, text=f"{max_v:.2f}", fill="#94A3B8",
                      anchor=tk.E, font=("TkFixedFont", 8))
        c.create_text(pad_l - 4, pad_t + chart_h, text=f"{min_v:.2f}", fill="#94A3B8",
                      anchor=tk.E, font=("TkFixedFont", 8))


class _HeatmapChart:
    """2D ヒートマップを Canvas に描画する。"""

    def __init__(self, canvas: tk.Canvas,
                 row_labels: list[str], col_labels: list[str],
                 matrix: list[list[float]],
                 title: str = "") -> None:
        self.canvas = canvas
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.matrix = matrix
        self.title = title

    def draw(self, w: int, h: int) -> None:
        c = self.canvas
        c.delete("all")
        rows = len(self.row_labels)
        cols = len(self.col_labels)
        if rows == 0 or cols == 0:
            c.create_text(w // 2, h // 2, text="データなし", fill=_GRAPH_FG)
            return

        all_vals = [v for row in self.matrix for v in row]
        min_v = min(all_vals) if all_vals else 0.0
        max_v = max(all_vals) if all_vals else 1.0
        span = max_v - min_v or 1.0

        pad_l = max(80, 10 * max(len(r) for r in self.row_labels))
        pad_t = 50
        pad_r, pad_b = 20, 40

        if self.title:
            c.create_text(pad_l + (w - pad_l - pad_r) // 2, 14, text=self.title,
                          fill=_GRAPH_FG, font=("TkDefaultFont", 12, "bold"))

        cell_w = (w - pad_l - pad_r) / cols
        cell_h = (h - pad_t - pad_b) / rows

        # 列ラベル
        for j, cl in enumerate(self.col_labels):
            cx = pad_l + (j + 0.5) * cell_w
            c.create_text(cx, pad_t - 8, text=cl, fill="#94A3B8",
                          font=("TkFixedFont", 12), angle=0 if len(cl) > 5 else 0)

        for i, (row_label, row_vals) in enumerate(zip(self.row_labels, self.matrix)):
            cy = pad_t + (i + 0.5) * cell_h
            c.create_text(pad_l - 4, cy, text=row_label, fill=_GRAPH_FG,
                          anchor=tk.E, font=("TkFixedFont", 12))
            for j, v in enumerate(row_vals):
                x0 = pad_l + j * cell_w
                y0 = pad_t + i * cell_h
                norm = (v - min_v) / span
                color = _interp_color(norm)
                c.create_rectangle(x0 + 1, y0 + 1, x0 + cell_w - 1, y0 + cell_h - 1,
                                   fill=color, outline="")
                if cell_w > 30 and cell_h > 14:
                    c.create_text(x0 + cell_w / 2, y0 + cell_h / 2,
                                  text=f"{v:.2f}", fill="white",
                                  font=("TkFixedFont", 12))

        # カラースケール
        scale_x = w - pad_r - 12
        for k in range(h - pad_t - pad_b):
            norm = 1.0 - k / (h - pad_t - pad_b)
            c.create_line(scale_x, pad_t + k, scale_x + 10, pad_t + k,
                          fill=_interp_color(norm))
        c.create_text(scale_x + 5, pad_t - 6, text=f"{max_v:.1f}", fill="#94A3B8", font=("TkFixedFont", 12))
        c.create_text(scale_x + 5, h - pad_b + 6, text=f"{min_v:.1f}", fill="#94A3B8", font=("TkFixedFont", 12))


class _RadarChart:
    """レーダーチャート（クモの巣グラフ）を Canvas に描画する。"""

    def __init__(self, canvas: tk.Canvas,
                 labels: list[str], values: list[float],
                 title: str = "") -> None:
        self.canvas = canvas
        self.labels = labels
        self.values = values
        self.title = title

    def draw(self, w: int, h: int) -> None:
        c = self.canvas
        c.delete("all")
        n = len(self.labels)
        if n < 3:
            c.create_text(w // 2, h // 2, text="データ不足 (最低3項目)", fill=_GRAPH_FG)
            return

        if self.title:
            c.create_text(w // 2, 14, text=self.title,
                          fill=_GRAPH_FG, font=("TkDefaultFont", 9, "bold"))

        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 40

        max_v = max(self.values) or 1.0
        angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]

        def _pt(angle: float, val: float):
            ratio = val / max_v
            return cx + r * ratio * math.cos(angle), cy - r * ratio * math.sin(angle)

        # グリッド（4段）
        for ring in range(1, 5):
            pts = []
            for a in angles:
                x = cx + r * ring / 4 * math.cos(a)
                y = cy - r * ring / 4 * math.sin(a)
                pts.extend([x, y])
            pts.extend(pts[:2])
            c.create_line(*pts, fill="#334155", smooth=False)

        # 軸
        for a in angles:
            c.create_line(cx, cy, cx + r * math.cos(a), cy - r * math.sin(a), fill="#475569")

        # データポリゴン
        pts = [_pt(a, v) for a, v in zip(angles, self.values)]
        flat = [coord for pt in pts for coord in pt] + list(pts[0])
        c.create_polygon(*flat[:-2], fill=_COLOR_MID, outline=_COLOR_MID,
                         stipple="gray50", width=2)
        c.create_line(*flat, fill=_COLOR_MID, width=2)

        # 頂点マーカー + ラベル
        for (x, y), lbl, val in zip(pts, self.labels, self.values):
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=_COLOR_HIGH, outline="")
            # ラベルを外側に配置
            dx = x - cx
            dy = y - cy
            dist = math.hypot(dx, dy) or 1
            lx = x + dx / dist * 14
            ly = y + dy / dist * 14
            c.create_text(lx, ly, text=f"{lbl}\n{val:.2f}", fill=_GRAPH_FG,
                          font=("TkFixedFont", 12), justify=tk.CENTER)


def _interp_color(t: float) -> str:
    """0.0→青, 0.5→黄, 1.0→赤 のカラーグラデーション。"""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        r = int(0 + t * 2 * 255)
        g = int(t * 2 * 220)
        b = int(255 - t * 2 * 255)
    else:
        r = 255
        g = int(220 - (t - 0.5) * 2 * 220)
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _bind_resize(canvas: tk.Canvas, draw_fn: Callable[[int, int], None]) -> None:
    """Canvas のリサイズ時に再描画するバインド。"""
    def _on_resize(event):
        draw_fn(event.width, event.height)
    canvas.bind("<Configure>", _on_resize)


# ─── タブ 4-1: 分析プレビュー ─────────────────────────────────────────────────

def _build_preview_tab(parent: ttk.Frame, log_dir: Path, log_fn: Callable) -> None:

    # ── コントロール行 ──────────────────────────────────────────────────────
    ctrl = ttk.Frame(parent)
    ctrl.pack(fill=tk.X, pady=(0, 6))

    ttk.Label(ctrl, text="ログファイル:").pack(side=tk.LEFT, padx=(0, 4))
    log_var = tk.StringVar()
    log_combo = ttk.Combobox(ctrl, textvariable=log_var, state="readonly", width=55)
    log_combo.pack(side=tk.LEFT, padx=(0, 6))

    def _refresh_combo():
        logs = _scan_logs(log_dir)
        log_combo["values"] = logs
        if logs and not log_var.get():
            log_var.set(logs[0])

    ttk.Button(ctrl, text="再スキャン", command=_refresh_combo).pack(side=tk.LEFT, padx=(0, 6))

    meta_var = tk.StringVar(value="")
    ttk.Label(ctrl, textvariable=meta_var, foreground="#64748B").pack(side=tk.LEFT, padx=8)

    # ── ノートブック（分析種別ごとのタブ） ──────────────────────────────────
    view_nb = ttk.Notebook(parent, style="SubTab.TNotebook")
    view_nb.pack(fill=tk.BOTH, expand=True)

    # サマリータブ
    summary_tab = ttk.Frame(view_nb, padding=4)
    view_nb.add(summary_tab, text=" サマリー ")

    # グラフタブ
    graph_tab = ttk.Frame(view_nb, padding=4)
    view_nb.add(graph_tab, text=" グラフ ")

    # ヒートマップタブ
    heat_tab = ttk.Frame(view_nb, padding=4)
    view_nb.add(heat_tab, text=" ヒートマップ ")

    # レーダータブ
    radar_tab = ttk.Frame(view_nb, padding=4)
    view_nb.add(radar_tab, text=" レーダーチャート ")

    # 自動レポートタブ
    report_tab = ttk.Frame(view_nb, padding=4)
    view_nb.add(report_tab, text=" 自動レポート ")

    # ── サマリーツリービュー ───────────────────────────────────────────────
    sum_tree = ttk.Treeview(summary_tab, show="headings", selectmode="browse")
    sum_sy = ttk.Scrollbar(summary_tab, orient=tk.VERTICAL, command=sum_tree.yview)
    sum_sx = ttk.Scrollbar(summary_tab, orient=tk.HORIZONTAL, command=sum_tree.xview)
    sum_tree.configure(yscrollcommand=sum_sy.set, xscrollcommand=sum_sx.set)
    sum_sx.pack(side=tk.BOTTOM, fill=tk.X)
    sum_sy.pack(side=tk.RIGHT, fill=tk.Y)
    sum_tree.pack(fill=tk.BOTH, expand=True)

    # ── グラフ Canvas ──────────────────────────────────────────────────────
    graph_canvas = tk.Canvas(graph_tab, bg=_GRAPH_BG, highlightthickness=0)
    graph_canvas.pack(fill=tk.BOTH, expand=True)

    # ── ヒートマップ Canvas ────────────────────────────────────────────────
    heat_canvas = tk.Canvas(heat_tab, bg=_GRAPH_BG, highlightthickness=0)
    heat_canvas.pack(fill=tk.BOTH, expand=True)

    # ── レーダー Canvas ────────────────────────────────────────────────────
    radar_canvas = tk.Canvas(radar_tab, bg=_GRAPH_BG, highlightthickness=0)
    radar_canvas.pack(fill=tk.BOTH, expand=True)

    # ── 自動レポートテキスト ───────────────────────────────────────────────
    report_text = tk.Text(report_tab, wrap=tk.WORD, font=("TkFixedFont", 15),
                          bg="#0F172A", fg=_GRAPH_FG, relief=tk.FLAT)
    rep_sy = ttk.Scrollbar(report_tab, orient=tk.VERTICAL, command=report_text.yview)
    report_text.configure(yscrollcommand=rep_sy.set)
    rep_sy.pack(side=tk.RIGHT, fill=tk.Y)
    report_text.pack(fill=tk.BOTH, expand=True)

    # ── 描画関数群 ─────────────────────────────────────────────────────────

    _current: dict[str, Any] = {"data": None, "graph": None, "heat": None, "radar": None}

    def _draw_graph(w: int, h: int):
        obj = _current["graph"]
        if obj:
            obj.draw(w, h)

    def _draw_heat(w: int, h: int):
        obj = _current["heat"]
        if obj:
            obj.draw(w, h)

    def _draw_radar(w: int, h: int):
        obj = _current["radar"]
        if obj:
            obj.draw(w, h)

    _bind_resize(graph_canvas, _draw_graph)
    _bind_resize(heat_canvas, _draw_heat)
    _bind_resize(radar_canvas, _draw_radar)

    def _load_and_render(*_):
        fname = log_var.get()
        if not fname:
            return
        data = _safe_load(log_dir, fname, log_fn)
        if data is None:
            return
        _current["data"] = data
        meta = data["metadata"]
        method_tag = _method_tag(data)
        meta_var.set(
            f"モデル: {meta.get('model_name','')}  /  "
            f"手法: {meta.get('method','')}  /  "
            f"種別: {meta.get('model_type','')}  /  "
            f"{meta.get('timestamp','')}"
        )

        gs = data.get("group_summary", {})
        groups = list(gs.keys())

        # ── サマリーツリー更新 ─────────────────────────────────────────
        for col in sum_tree["columns"]:
            sum_tree.heading(col, text="")
        sum_tree.delete(*sum_tree.get_children())

        cols, hdrs = _summary_cols(method_tag)
        sum_tree["columns"] = cols
        for cid, hdr in zip(cols, hdrs):
            sum_tree.heading(cid, text=hdr)
            sum_tree.column(cid, width=110, anchor=tk.CENTER)
        sum_tree.column(cols[0], width=160)

        for grp in groups:
            agg = gs[grp]
            row = _make_summary_row(grp, agg, method_tag)
            sum_tree.insert("", tk.END, values=row)

        # ── グラフ（折れ線 + 棒）─────────────────────────────────────
        metric = _method_metric(method_tag)
        g_vals = [gs[g].get(metric, 0.0) for g in groups]
        # 異常値インデックス（平均+1.5σ超）
        if g_vals:
            mu = sum(g_vals) / len(g_vals)
            sigma = math.sqrt(sum((v - mu) ** 2 for v in g_vals) / len(g_vals)) if len(g_vals) > 1 else 0.0
            hi_idx = [i for i, v in enumerate(g_vals) if v > mu + 1.5 * sigma]
        else:
            hi_idx = []

        graph_colors = [_COLOR_WARN if i in hi_idx else _COLOR_MID for i in range(len(groups))]
        bar = _BarChart(graph_canvas, groups, g_vals, graph_colors,
                        title=f"グループ別 {metric.replace('_',' ')}")
        _current["graph"] = bar
        graph_canvas.update_idletasks()
        w = graph_canvas.winfo_width() or 600
        h = graph_canvas.winfo_height() or 300
        bar.draw(w, h)

        # ── ヒートマップ ───────────────────────────────────────────────
        block_keys = ["Input", "Middle", "Output"]
        comp_keys = _heat_col_keys(method_tag)
        heat_mat: list[list[float]] = []
        heat_row_labels: list[str] = []
        for bk in block_keys:
            sub_groups = [g for g in groups if bk in g]
            if not sub_groups:
                continue
            heat_row_labels.append(bk)
            row_vals = []
            for ck in comp_keys:
                vals = [gs[g].get(ck, 0.0) for g in sub_groups if ck in gs[g]]
                row_vals.append(sum(vals) / len(vals) if vals else 0.0)
            heat_mat.append(row_vals)

        heat = _HeatmapChart(
            heat_canvas, heat_row_labels,
            [k.replace("mean_", "").replace("_", " ") for k in comp_keys],
            heat_mat,
            title="ブロック × メトリクス ヒートマップ",
        )
        _current["heat"] = heat
        heat_canvas.update_idletasks()
        wh = heat_canvas.winfo_width() or 600
        hh = heat_canvas.winfo_height() or 300
        heat.draw(wh, hh)

        # ── レーダーチャート ───────────────────────────────────────────
        radar_labels, radar_vals = _radar_data(method_tag, gs)
        radar = _RadarChart(radar_canvas, radar_labels, radar_vals,
                            title="モデル特徴レーダー")
        _current["radar"] = radar
        radar_canvas.update_idletasks()
        wr = radar_canvas.winfo_width() or 400
        hr = radar_canvas.winfo_height() or 300
        radar.draw(wr, hr)

        # ── 自動レポート ───────────────────────────────────────────────
        report_text.config(state=tk.NORMAL)
        report_text.delete("1.0", tk.END)
        auto = data.get("auto_report", [])
        report_text.insert(tk.END, "\n".join(auto))
        report_text.config(state=tk.DISABLED)

        log_fn(f"[Viewer] プレビュー読み込み完了: {fname}")

    log_combo.bind("<<ComboboxSelected>>", _load_and_render)
    _refresh_combo()


# ─── タブ 4-2: レポート比較 ───────────────────────────────────────────────────

def _build_compare_tab(parent: ttk.Frame, log_dir: Path, log_fn: Callable) -> None:

    # ── コントロール ────────────────────────────────────────────────────────
    ctrl = ttk.LabelFrame(parent, text="ログ選択（同一分析手法のログを2本選択）")
    ctrl.pack(fill=tk.X, pady=(0, 6))

    ttk.Label(ctrl, text="ログ A:").grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
    log_a_var = tk.StringVar()
    log_a_combo = ttk.Combobox(ctrl, textvariable=log_a_var, state="readonly", width=52)
    log_a_combo.grid(row=0, column=1, sticky=tk.EW, padx=4)

    ttk.Label(ctrl, text="ログ B:").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
    log_b_var = tk.StringVar()
    log_b_combo = ttk.Combobox(ctrl, textvariable=log_b_var, state="readonly", width=52)
    log_b_combo.grid(row=1, column=1, sticky=tk.EW, padx=4)

    # 種別ラベル（本体モデル vs LoRA 対比 OK）
    type_info_var = tk.StringVar(value="")
    ttk.Label(ctrl, textvariable=type_info_var, foreground="#64748B").grid(
        row=0, column=2, rowspan=2, padx=8
    )
    ctrl.columnconfigure(1, weight=1)

    def _refresh_combos():
        logs = _scan_logs(log_dir)
        log_a_combo["values"] = logs
        log_b_combo["values"] = logs
        if len(logs) >= 1 and not log_a_var.get():
            log_a_var.set(logs[0])
        if len(logs) >= 2 and not log_b_var.get():
            log_b_var.set(logs[1])

    ttk.Button(ctrl, text="再スキャン", command=_refresh_combos).grid(
        row=0, column=3, rowspan=2, padx=8, pady=6
    )
    ttk.Button(ctrl, text="▶  比較実行", style="Run.TButton",
               command=lambda: _run_compare()).grid(
        row=0, column=4, rowspan=2, padx=8, pady=6
    )

    # ── 結果エリア ─────────────────────────────────────────────────────────
    result_nb = ttk.Notebook(parent, style="SubTab.TNotebook")
    result_nb.pack(fill=tk.BOTH, expand=True)

    diff_tab = ttk.Frame(result_nb, padding=4)
    score_tab = ttk.Frame(result_nb, padding=4)
    compat_tab = ttk.Frame(result_nb, padding=4)
    result_nb.add(diff_tab,   text=" 差分チャート ")
    result_nb.add(score_tab,  text=" 近似度スコア ")
    result_nb.add(compat_tab, text=" 相性レポート ")

    # 差分折れ線グラフ
    diff_canvas = tk.Canvas(diff_tab, bg=_GRAPH_BG, highlightthickness=0)
    diff_canvas.pack(fill=tk.BOTH, expand=True)

    # 近似度スコアツリー
    score_frame = ttk.Frame(score_tab)
    score_frame.pack(fill=tk.BOTH, expand=True)
    score_tree = ttk.Treeview(score_frame, columns=("item", "score_a", "score_b", "sim", "diff"),
                               show="headings", selectmode="browse")
    score_tree.heading("item",    text="グループ")
    score_tree.heading("score_a", text="値 A")
    score_tree.heading("score_b", text="値 B")
    score_tree.heading("sim",     text="コサイン類似度")
    score_tree.heading("diff",    text="差分比率")
    for col in ("item", "score_a", "score_b", "sim", "diff"):
        score_tree.column(col, width=120, anchor=tk.CENTER)
    score_tree.column("item", width=180)
    sc_sy = ttk.Scrollbar(score_frame, orient=tk.VERTICAL, command=score_tree.yview)
    score_tree.configure(yscrollcommand=sc_sy.set)
    sc_sy.pack(side=tk.RIGHT, fill=tk.Y)
    score_tree.pack(fill=tk.BOTH, expand=True)

    # 相性レポートテキスト
    compat_text = tk.Text(compat_tab, wrap=tk.WORD, font=("TkFixedFont", 15),
                          bg="#0F172A", fg=_GRAPH_FG, relief=tk.FLAT)
    cp_sy = ttk.Scrollbar(compat_tab, orient=tk.VERTICAL, command=compat_text.yview)
    compat_text.configure(yscrollcommand=cp_sy.set)
    cp_sy.pack(side=tk.RIGHT, fill=tk.Y)
    compat_text.pack(fill=tk.BOTH, expand=True)

    _cmp_state: dict[str, Any] = {"line": None}

    def _draw_diff(w: int, h: int):
        obj = _cmp_state.get("line")
        if obj:
            obj.draw(w, h)

    _bind_resize(diff_canvas, _draw_diff)

    def _run_compare():
        fa = log_a_var.get()
        fb = log_b_var.get()
        if not fa or not fb:
            messagebox.showerror("比較エラー", "ログ A と ログ B を両方選択してください。")
            return

        da = _safe_load(log_dir, fa, log_fn)
        db = _safe_load(log_dir, fb, log_fn)
        if da is None or db is None:
            return

        meta_a = da["metadata"]
        meta_b = db["metadata"]
        method_a = _method_tag(da)
        method_b = _method_tag(db)

        # 異種手法の場合は警告
        if method_a != method_b:
            messagebox.showwarning(
                "手法不一致",
                f"ログ A の手法 ({method_a}) とログ B の手法 ({method_b}) が異なります。\n"
                "比較精度が低下する可能性があります。続行します。"
            )

        type_a = meta_a.get("model_type", "model")
        type_b = meta_b.get("model_type", "model")
        type_info_var.set(f"A: {type_a}  /  B: {type_b}")

        gs_a = da.get("group_summary", {})
        gs_b = db.get("group_summary", {})
        metric = _method_metric(method_a)

        groups_a = set(gs_a.keys())
        groups_b = set(gs_b.keys())
        common = sorted(groups_a & groups_b)

        vals_a = [gs_a[g].get(metric, 0.0) for g in common]
        vals_b = [gs_b[g].get(metric, 0.0) for g in common]

        # 乖離度（ユークリッド距離）が大きい層を赤ハイライト
        diffs = [abs(a - b) for a, b in zip(vals_a, vals_b)]
        if diffs:
            mu_d = sum(diffs) / len(diffs)
            hi_idx = [i for i, d in enumerate(diffs) if d > mu_d * 1.5]
        else:
            hi_idx = []

        name_a = meta_a.get("model_name", fa)
        name_b = meta_b.get("model_name", fb)

        line = _LineChart(
            diff_canvas, vals_a, vals_b,
            labels=common, highlight_indices=hi_idx,
            title=f"{metric.replace('_',' ')} 比較  赤=乖離大",
            label_a=name_a[:20], label_b=name_b[:20],
        )
        _cmp_state["line"] = line
        diff_canvas.update_idletasks()
        w = diff_canvas.winfo_width() or 700
        h = diff_canvas.winfo_height() or 300
        line.draw(w, h)

        # ── 近似度スコアテーブル ───────────────────────────────────────
        score_tree.delete(*score_tree.get_children())
        overall_sims: list[float] = []
        for i, grp in enumerate(common):
            va = vals_a[i]
            vb = vals_b[i]
            # コサイン類似度（スカラーの場合は符号一致率で代用）
            denom = (abs(va) + abs(vb)) or 1.0
            sim = 1.0 - abs(va - vb) / denom
            diff_ratio = abs(va - vb) / (abs(va) + 1e-8)
            overall_sims.append(sim)
            tag = "warn" if i in hi_idx else ""
            score_tree.insert("", tk.END,
                              values=(grp, f"{va:.4f}", f"{vb:.4f}",
                                      f"{sim:.3f}", f"{diff_ratio:.3f}"),
                              tags=(tag,))
        score_tree.tag_configure("warn", foreground=_COLOR_WARN)

        overall_sim = sum(overall_sims) / len(overall_sims) if overall_sims else 0.0

        # ── 相性レポート生成 ────────────────────────────────────────────
        report_lines = _generate_compat_report(
            meta_a, meta_b, method_a,
            gs_a, gs_b, common, vals_a, vals_b,
            hi_idx, overall_sim, diffs,
        )

        compat_text.config(state=tk.NORMAL)
        compat_text.delete("1.0", tk.END)
        compat_text.insert(tk.END, "\n".join(report_lines))
        compat_text.config(state=tk.DISABLED)

        result_nb.select(compat_tab)
        log_fn(f"[Viewer] 比較完了: {fa} vs {fb}  総合近似度={overall_sim:.1%}")

    _refresh_combos()


# ─── 相性レポート生成 ─────────────────────────────────────────────────────────

def _generate_compat_report(
    meta_a: dict, meta_b: dict, method_tag: str,
    gs_a: dict, gs_b: dict,
    common: list[str],
    vals_a: list[float], vals_b: list[float],
    hi_idx: list[int],
    overall_sim: float,
    diffs: list[float],
) -> list[str]:
    """
    2つのログから相性・マージ適性レポートを生成する。
    本体モデル同士・LoRA同士・本体×LoRA の全組み合わせに対応。
    """
    lines: list[str] = []
    name_a = meta_a.get("model_name", "A")
    name_b = meta_b.get("model_name", "B")
    type_a = meta_a.get("model_type", "model")
    type_b = meta_b.get("model_type", "model")
    method = meta_a.get("method", method_tag.replace("_", " "))

    sep = "=" * 60
    lines += [
        sep,
        "  相性・マージ適性 レポート",
        sep,
        f"  ログ A : {name_a}  [{type_a}]",
        f"  ログ B : {name_b}  [{type_b}]",
        f"  分析手法: {method}",
        f"  共通グループ数: {len(common)}",
        "",
    ]

    # ── 総合近似度スコア ────────────────────────────────────────────────
    sim_pct = overall_sim * 100
    lines.append(f"【総合近似度スコア】  {sim_pct:.1f} %")
    if sim_pct >= 80:
        lines.append("  → 高い構造的類似性。マージ時の崩壊リスクは低いと推定されます。")
    elif sim_pct >= 55:
        lines.append("  → 中程度の類似性。Alpha値を0.3〜0.5に設定し、Cosine補正を有効にすることを推奨します。")
    else:
        lines.append("  → 構造的乖離が大きいです。Alpha≤0.3 + Cosine自動減衰を強く推奨します。")
        lines.append("  　 特定層のみ選択的マージ（Module Filtering）の検討を推奨します。")
    lines.append("")

    # ── 乖離層の詳細 ────────────────────────────────────────────────────
    if hi_idx:
        lines.append(f"【高乖離グループ ({len(hi_idx)} 件) — 要注意】")
        for i in hi_idx:
            if i < len(common):
                grp = common[i]
                lines.append(f"  {grp}  |  A={vals_a[i]:.4f}  B={vals_b[i]:.4f}  diff={diffs[i]:.4f}")
        lines.append("")
        lines.append("  → 上記グループのマージ比率を 0.3 以下に下げるか、")
        lines.append("     Cosine Similarity Threshold の自動減衰機能を有効にしてください。")
    else:
        lines.append("【高乖離グループ】 — 検出されませんでした。")
    lines.append("")

    # ── ブロック別評価 ──────────────────────────────────────────────────
    lines.append("【ブロック別 相性評価】")
    metric = _method_metric(method_tag)
    for block in ("Input", "Middle", "Output"):
        sub_a = {g: gs_a[g].get(metric, 0.0) for g in gs_a if block in g}
        sub_b = {g: gs_b[g].get(metric, 0.0) for g in gs_b if block in g}
        common_sub = set(sub_a) & set(sub_b)
        if not common_sub:
            lines.append(f"  {block}: データなし")
            continue
        va_mean = sum(sub_a[g] for g in common_sub) / len(common_sub)
        vb_mean = sum(sub_b[g] for g in common_sub) / len(common_sub)
        ratio = abs(va_mean - vb_mean) / (max(va_mean, vb_mean) + 1e-8)
        compat = "◎ 相性良好" if ratio < 0.2 else ("△ 注意" if ratio < 0.5 else "✕ 高リスク")
        lines.append(f"  {block:<10}: {compat}  (乖離率={ratio:.1%}  A={va_mean:.3f}  B={vb_mean:.3f})")
    lines.append("")

    # ── マージ組み合わせ別アドバイス ────────────────────────────────────
    lines.append("【マージ手法アドバイス】")
    if type_a == "model" and type_b == "model":
        lines.append("  ● 本体モデル同士のマージ（Model-to-Model Merge）")
        lines.append("    → Block Merge で Input/Middle/Output を個別に比率調整してください。")
        lines.append("    → 類似度が低い層は Residual Connection Scaling で影響を緩和できます。")
    elif (type_a == "lora" and type_b == "lora"):
        lines.append("  ● LoRA 同士のマージ（LoRA-to-LoRA Merge）")
        lines.append("    → Alpha Scaling + Module Filtering で影響層を限定してください。")
        lines.append("    → 高乖離層がある場合は SVD Rank 分析で有効ランクを確認し、")
        lines.append("       Rank-Based Merging で再計算することを推奨します。")
    elif "lora" in (type_a, type_b):
        lines.append("  ● 本体モデル × LoRA の相性チェック（LoRA-to-Model Merge 前の事前確認）")
        lines.append("    → 本体の高エネルギー層に LoRA が強い影響を持つ場合、")
        lines.append("       Alpha を下げて Fuse することで崩壊リスクを低減できます。")
        lines.append("    → Module Filtering で LoRA 適用層を高乖離グループから除外することを検討してください。")
    lines.append("")

    # ── 仕様書 判定パターン B チェック（Attention層のコサイン類似度） ──
    attn_groups = [g for g in common if "Attention" in g]
    if attn_groups and method_tag in ("Statistical", "SVD_Rank"):
        attn_sims = []
        for g in attn_groups:
            va = gs_a.get(g, {}).get(metric, 0.0)
            vb = gs_b.get(g, {}).get(metric, 0.0)
            denom = (abs(va) + abs(vb)) or 1.0
            attn_sims.append(1.0 - abs(va - vb) / denom)
        mean_attn_sim = sum(attn_sims) / len(attn_sims)
        lines.append("【判定パターンB: Attention層コサイン類似度チェック】")
        lines.append(f"  平均 Attention コサイン類似度: {mean_attn_sim:.3f}  (閾値: 0.4)")
        if mean_attn_sim < 0.4:
            lines += [
                "  ⚠ 対象モデル間で特定のAttention層の方向性が大きく乖離しています。",
                "    そのままマージすると構造の破綻（モデル崩壊）を招く高リスクがあります。",
                "    『Cosine Similarity Threshold』の自動減衰機能を有効にするか、",
                "    該当層のマージ比率を0.3以下に下げることを推奨します。",
            ]
        else:
            lines.append("  ✓ Attention層の方向性は許容範囲内です。")
        lines.append("")

    lines += [
        sep,
        "※ 本レポートはログの数値傾向に基づくルールベース推定です。",
        "   実際のマージ前に Dry-Run 検証を必ず実施してください。",
        sep,
    ]
    return lines


# ─── グラフデータ構築ユーティリティ ──────────────────────────────────────────

def _method_metric(method_tag: str) -> str:
    for tag, metric in _METHOD_METRICS.items():
        if tag in method_tag:
            return metric
    return "mean_l2"


def _summary_cols(method_tag: str) -> tuple[list[str], list[str]]:
    base_ids = ["group", "layers"]
    base_hdr = ["グループ", "層数"]
    if "Feature_Map" in method_tag:
        return (base_ids + ["mean", "var", "complexity"],
                base_hdr + ["平均値", "分散", "空間複雑度"])
    if "Statistical" in method_tag:
        return (base_ids + ["l2", "max_l2", "mean_val"],
                base_hdr + ["L2ノルム(avg)", "L2(max)", "平均値"])
    if "SVD_Rank" in method_tag:
        return (base_ids + ["eff_rank", "decay", "cumvar"],
                base_hdr + ["有効ランク", "減衰率", "累積寄与率"])
    if "Attention_Map" in method_tag:
        return (base_ids + ["attn_w", "head_var"],
                base_hdr + ["Attnウェイト", "ヘッド分散"])
    return (base_ids + ["value"], base_hdr + ["値"])


def _make_summary_row(grp: str, agg: dict, method_tag: str) -> tuple:
    n = int(agg.get("layer_count", agg.get("attention_layer_count", 0)))
    if "Feature_Map" in method_tag:
        return (grp, n,
                f"{agg.get('mean_feat_mean', 0):.4f}",
                f"{agg.get('mean_feat_var', 0):.4f}",
                f"{agg.get('mean_feat_complexity', 0):.4f}")
    if "Statistical" in method_tag:
        return (grp, n,
                f"{agg.get('mean_l2', 0):.4f}",
                f"{agg.get('max_l2', 0):.4f}",
                f"{agg.get('mean_mean', 0):.4f}")
    if "SVD_Rank" in method_tag:
        return (grp, n,
                f"{agg.get('mean_effective_rank', 0):.1f}",
                f"{agg.get('mean_decay_rate', 0):.4f}",
                f"{agg.get('mean_cumvar', 0):.4f}")
    if "Attention_Map" in method_tag:
        return (grp, n,
                f"{agg.get('mean_attn_weight', 0):.4f}",
                f"{agg.get('mean_head_variance', 0):.4f}")
    return (grp, n, "—")


def _heat_col_keys(method_tag: str) -> list[str]:
    if "Feature_Map" in method_tag:
        return ["mean_feat_mean", "mean_feat_var", "mean_feat_complexity"]
    if "Statistical" in method_tag:
        return ["mean_l2", "max_l2", "mean_mean", "mean_var"]
    if "SVD_Rank" in method_tag:
        return ["mean_effective_rank", "mean_decay_rate", "mean_cumvar"]
    if "Attention_Map" in method_tag:
        return ["mean_attn_weight", "mean_head_variance"]
    return ["mean_l2"]


def _radar_data(method_tag: str, gs: dict) -> tuple[list[str], list[float]]:
    """
    仕様書 5.1「テクスチャ・構図寄与度グラフ」相当のレーダーデータを生成。
    Input / Middle / Output の各ブロックの主メトリクスを軸に使用。
    """
    metric = _method_metric(method_tag)
    labels: list[str] = []
    vals: list[float] = []

    block_comp = [
        ("Input",  "Attention"),
        ("Input",  "MLP"),
        ("Middle", "Attention"),
        ("Middle", "MLP"),
        ("Output", "Attention"),
        ("Output", "MLP"),
    ]
    for block, comp in block_comp:
        sub = [gs[g].get(metric, 0.0) for g in gs if block in g and comp in g]
        if sub:
            labels.append(f"{block}\n{comp}")
            vals.append(sum(sub) / len(sub))

    if not labels:
        # フォールバック: 全グループをそのまま使用
        for g, agg in list(gs.items())[:6]:
            labels.append(g[:12])
            vals.append(agg.get(metric, 0.0))

    return labels, vals
