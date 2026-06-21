"""app/monitor_graph_addift.py — ADDifT学習リアルタイムモニタリングウィジェット

_build_monitor_tab(parent, state) から呼び出される AddifTMonitorGraph クラスを提供する。

monitor_graph_leco.py (LecoMonitorGraph) からの移植であり、ロジックは同一:
  - Val Loss / ΔLoss を監視しない（ADDifTもLECOと同様にvalidation lossの概念を持たない）
  - EarlyStopping は Train Loss の指定step数連続上昇を監視（leco_train.py と同機能）
  - state._monitor_queue を参照（addift_train._AddifTTrainState 互換）
  - ステップ管理（epochではなくstep単位）に合わせた表示
  - ログ参照元: <project_root>/log/addift_train/*.txt
    （ただし本ウィジェット自体はファイルを直接読まず、addift_train._start_training の
      ワーカースレッドが書き込みと同時に state._monitor_queue へ流すラインを購読する）
"""
from __future__ import annotations

import datetime
import math
import queue as _queue_mod
import re
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Optional

try:
    from .i18n import gettext, load_language
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _app_dir = _Path(__file__).resolve().parent
    if str(_app_dir) not in _sys.path:
        _sys.path.insert(0, str(_app_dir))
    from i18n import gettext, load_language  # type: ignore[no-redef]

if TYPE_CHECKING:
    from .addift_train import _AddifTTrainState

# ──────────────────────────────────────────────────────────────────────────────
# 色定数
# ──────────────────────────────────────────────────────────────────────────────
COLOR_NORMAL   = "#16A34A"
COLOR_CAUTION  = "#CA8A04"
COLOR_WARNING  = "#EA580C"
COLOR_DANGER   = "#DC2626"
COLOR_INFO     = "#E2E8F0"
COLOR_TRAIN_LOSS = "#38BDF8"
COLOR_WIN_LOSS    = "#22C55E"
COLOR_LOSE_LOSS   = "#F472B6"

# ──────────────────────────────────────────────────────────────────────────────
# ログパースパターン
# ──────────────────────────────────────────────────────────────────────────────
# tqdm postfix: "loss=0.0044" （ADDifT も anima_train_addift.py 内で同形式で出力される）
_RE_LOSS_TQDM    = re.compile(r",\s*loss=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
# DPOモード専用: "win_loss=-0.0050" / "lose_loss=0.0080" （符号付き）
_RE_WIN_LOSS_TQDM  = re.compile(r",?\s*win_loss=([+-]?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_LOSE_LOSS_TQDM = re.compile(r",?\s*lose_loss=([+-]?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
# tqdm 時間フィールド: "[00:03<26:53,  3.23s/it"
_RE_TQDM_TIME    = re.compile(r"\[(\d+:\d+)<((?:\d+:)?\d+:\d+),\s*([0-9.]+)s/it")
_RE_STEP_POSTFIX = re.compile(r"(\d+)/(\d+)\s*\[")   # tqdm: "120/500 ["


def _is_dpo_mode_active(state: "_AddifTTrainState") -> bool:
    """state がDPOモード有効状態かどうかを判定する。"""
    try:
        return bool(state.addift_mode_enabled.get()) and state.addift_mode_name.get() == "dpo"
    except Exception:
        return False


class AddifTMonitorGraph:
    """ADDifT学習リアルタイムモニタリングウィジェット。

    Parameters
    ----------
    parent : ttk.Frame
        モニターグラフタブの親フレーム
    state : _AddifTTrainState
        addift_train.py の状態オブジェクト。_monitor_queue を参照する。
    """

    MAX_REPORT_LINES = 200

    def __init__(self, parent: ttk.Frame, state: "_AddifTTrainState") -> None:
        self._state  = state
        self._parent = parent

        # _build_ui() 内で _refresh_dpo_visibility() 経由で参照されるため、
        # _init_matplotlib() 実行前に先行初期化しておく。
        self._mpl_ok = False
        self._ax_loss = None
        self._ax_lr = None
        self._ax_win_loss = None
        self._ax_lose_loss = None

        # ── データ系列 ──────────────────────────────────────────────
        self._steps:      list[int]   = []
        self._train_loss: list[float] = []
        self._lr_vals:    list[float] = []
        self._win_loss:   list[float] = []
        self._lose_loss:  list[float] = []

        # ── 状態変数 ──────────────────────────────────────────────
        self._global_step  = 0
        self._step_in_run  = 0
        self._total_steps  = 0
        self._last_lr      = 0.0
        self._last_train   = float("nan")
        self._last_win_loss  = float("nan")
        self._last_lose_loss = float("nan")

        # EarlyStopping 状態
        self._es_rise_count = 0
        self._es_prev_loss  = float("nan")
        self._es_warned     = False
        self._es_stopped    = False

        # EarlyStoppingDPO 状態
        self._es_dpo_count        = 0
        self._es_dpo_prev_win     = float("nan")
        self._es_dpo_prev_lose    = float("nan")
        self._es_dpo_warned       = False
        self._es_dpo_stopped      = False

        # グラフレイアウト・パネル表示の状態追跡 (不要な再構築/再パックを防止)
        self._graph_layout_is_dpo: Optional[bool] = None
        self._es_dpo_frame_visible = False

        # 時間計測
        self._start_time:   float | None = None
        self._step_times:   list[float]  = []
        self._last_step_ts: float | None = None
        self._tqdm_eta_sec: float | None = None
        # ── UI構築 ──────────────────────────────────────────────────
        self._build_ui(parent)
        self._init_matplotlib()
        parent.after(300, self._poll)

    # ─────────────────────────────────────────────────────────────────
    # UI構築
    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=7)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(0, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        self._graph_frame = left

        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky=tk.NSEW)
        right.rowconfigure(1, weight=1)

        self._build_param_panel(right)
        self._build_report_panel(right)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Button(btn_row, text=gettext("lora_monitor_btn_reset"), command=self._reset).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(btn_row, text=gettext("lora_monitor_btn_stop"), command=self._stop_training).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0)
        )

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=gettext("lora_monitor_param_title"))
        lf.pack(fill=tk.X, padx=4, pady=(4, 4))

        self._param_vars: dict[str, tk.StringVar] = {}
        rows = [
            (gettext("lora_monitor_param_step"),       "step"),
            (gettext("lora_monitor_param_lr"),         "lr"),
            (gettext("lora_monitor_param_train_loss"), "train_loss"),
        ]
        self._param_rows: dict[str, tuple[ttk.Label, ttk.Label]] = {}
        for i, (label, key) in enumerate(rows):
            label_w = ttk.Label(lf, text=label + ":", width=12, anchor=tk.W)
            label_w.grid(row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2)
            v = tk.StringVar(value="—")
            self._param_vars[key] = v
            value_w = ttk.Label(lf, textvariable=v, anchor=tk.W, font=("TkFixedFont", 9))
            value_w.grid(row=i, column=1, sticky=tk.EW, padx=(0, 6), pady=2)
            self._param_rows[key] = (label_w, value_w)
        lf.columnconfigure(1, weight=1)

        # DPOモード専用行: Win Loss / Lose Loss (デフォルト非表示)
        dpo_rows = [
            (gettext("addift_monitor_param_win_loss"),  "win_loss"),
            (gettext("addift_monitor_param_lose_loss"), "lose_loss"),
        ]
        for offset, (label, key) in enumerate(dpo_rows):
            row_index = len(rows) + offset
            label_w = ttk.Label(lf, text=label + ":", width=12, anchor=tk.W)
            v = tk.StringVar(value="—")
            self._param_vars[key] = v
            value_w = ttk.Label(lf, textvariable=v, anchor=tk.W, font=("TkFixedFont", 9))
            self._param_rows[key] = (label_w, value_w)
            self._dpo_param_row_index = {**getattr(self, "_dpo_param_row_index", {}), key: row_index}

        # EarlyStopping パネル
        es_lf = ttk.LabelFrame(parent, text=gettext("lora_monitor_es_title"))
        es_lf.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._es_status_var = tk.StringVar(value=gettext("leco_monitor_es_disabled"))
        ttk.Label(es_lf, textvariable=self._es_status_var,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W, padx=6, pady=(2, 0))
        self._es_progress = ttk.Progressbar(es_lf, maximum=100, value=0, length=180)
        self._es_progress.pack(fill=tk.X, padx=6, pady=(2, 4))

        # EarlyStoppingDPO パネル (DPOモード時のみ表示。常にes_lfの直後に固定配置する)
        self._es_dpo_anchor = es_lf
        self._es_dpo_frame = ttk.LabelFrame(parent, text=gettext("addift_monitor_es_dpo_title"))
        self._es_dpo_status_var = tk.StringVar(value=gettext("leco_monitor_es_disabled"))
        ttk.Label(self._es_dpo_frame, textvariable=self._es_dpo_status_var,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W, padx=6, pady=(2, 0))
        self._es_dpo_progress = ttk.Progressbar(self._es_dpo_frame, maximum=100, value=0, length=180)
        self._es_dpo_progress.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._es_dpo_frame_visible = False

        self._refresh_dpo_visibility()

        # 時間情報
        time_lf = ttk.LabelFrame(parent, text=gettext("lora_monitor_time_title"))
        time_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        time_rows = [
            (gettext("lora_monitor_time_start"),     "start_time"),
            (gettext("lora_monitor_time_eta_clock"), "eta_clock"),
            (gettext("lora_monitor_time_eta_remain"), "eta_remain"),
            (gettext("lora_monitor_time_elapsed"),   "elapsed"),
        ]
        self._time_vars: dict[str, tk.StringVar] = {}
        for i, (label, key) in enumerate(time_rows):
            ttk.Label(time_lf, text=label + ":", width=10, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2
            )
            v = tk.StringVar(value="—")
            self._time_vars[key] = v
            ttk.Label(time_lf, textvariable=v, anchor=tk.W,
                      font=("TkFixedFont", 9)).grid(
                row=i, column=1, sticky=tk.EW, padx=(0, 6), pady=2
            )
        time_lf.columnconfigure(1, weight=1)

    def _build_report_panel(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=gettext("lora_monitor_report_title"))
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self._report_text = tk.Text(
            lf,
            height=12,
            wrap=tk.WORD,
            font=("TkFixedFont", 10),
            state=tk.DISABLED,
            bg="#1E293B",
            fg="#CBD5E1",
        )
        scroll = ttk.Scrollbar(lf, orient=tk.VERTICAL,
                               command=self._report_text.yview)
        self._report_text.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        self._report_text.grid(row=0, column=0, sticky=tk.NSEW)

        self._report_text.tag_configure("normal",      foreground=COLOR_NORMAL)
        self._report_text.tag_configure("caution",     foreground=COLOR_CAUTION)
        self._report_text.tag_configure("warning",     foreground=COLOR_WARNING)
        self._report_text.tag_configure("danger",      foreground=COLOR_DANGER)
        self._report_text.tag_configure("info",        foreground=COLOR_INFO)
        self._report_text.tag_configure("diag",        foreground="#7DD3FC")
        self._report_text.tag_configure("sample_info", foreground="#A78BFA")
        self._report_text.tag_configure("es_warn",     foreground=COLOR_WARNING)
        self._report_text.tag_configure("es_stop",     foreground=COLOR_DANGER)
    # ─────────────────────────────────────────────────────────────────
    # matplotlib 初期化
    # ─────────────────────────────────────────────────────────────────
    def _init_matplotlib(self) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            matplotlib.rcParams.update({
                "figure.facecolor":  "#0F172A",
                "axes.facecolor":    "#1E293B",
                "axes.edgecolor":    "#475569",
                "axes.labelcolor":   "#CBD5E1",
                "xtick.color":       "#94A3B8",
                "ytick.color":       "#94A3B8",
                "text.color":        "#CBD5E1",
                "grid.color":        "#334155",
                "grid.linestyle":    "--",
                "grid.alpha":        0.5,
                "legend.facecolor":  "#1E293B",
                "legend.edgecolor":  "#475569",
            })

            self._fig = Figure(figsize=(8, 8))
            self._canvas = FigureCanvasTkAgg(self._fig, master=self._graph_frame)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_ok = True
            self._rebuild_graph_layout(is_dpo=False)

        except Exception as exc:
            self._mpl_ok = False
            ttk.Label(
                self._graph_frame,
                text=gettext("lora_monitor_mpl_error", error=exc),
                foreground="#EF4444",
                justify=tk.LEFT,
            ).pack(padx=16, pady=16, anchor=tk.NW)

    def _refresh_dpo_visibility(self) -> None:
        """DPOモードの有効状態に応じてWin/Lose Loss行・EarlyStoppingDPO枠・グラフ段数を切り替える。"""
        is_dpo = _is_dpo_mode_active(self._state)

        for key in ("win_loss", "lose_loss"):
            label_w, value_w = self._param_rows[key]
            row_index = self._dpo_param_row_index[key]
            if is_dpo:
                label_w.grid(row=row_index, column=0, sticky=tk.W, padx=(6, 2), pady=2)
                value_w.grid(row=row_index, column=1, sticky=tk.EW, padx=(0, 6), pady=2)
            else:
                label_w.grid_remove()
                value_w.grid_remove()

        # EarlyStoppingDPO枠: 状態が変化した時のみpack/pack_forgetを行い、
        # es_dpo_anchor(EarlyStopping枠)の直後に固定する。
        # 毎ポーリングで無条件にpack()し直すと、後から配置された他パネル
        # (自動レポート等)より後方へ再スタックされてしまうため。
        if is_dpo != self._es_dpo_frame_visible:
            if is_dpo:
                self._es_dpo_frame.pack(fill=tk.X, padx=4, pady=(0, 4), after=self._es_dpo_anchor)
            else:
                self._es_dpo_frame.pack_forget()
            self._es_dpo_frame_visible = is_dpo

        # グラフ段数の切替 (1行2列 ⇔ 2行2列) は状態変化時のみ再構築する。
        if self._mpl_ok and is_dpo != self._graph_layout_is_dpo:
            self._rebuild_graph_layout(is_dpo)

    def _rebuild_graph_layout(self, is_dpo: bool) -> None:
        """グラフ段数を切り替えて再構築する。

        通常時: 1行2列 (Train Loss / Learning Rate)
        DPOモード時: 2行2列 (Train Loss / Learning Rate / Win Loss / Lose Loss)

        Args:
            is_dpo: DPOモードが有効かどうか。

        Returns:
            None
        """
        self._fig.clf()

        if is_dpo:
            gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 1])
            self._fig.subplots_adjust(
                left=0.10, right=0.96, top=0.95, bottom=0.07, hspace=0.45, wspace=0.30
            )
            self._ax_loss      = self._fig.add_subplot(gs[0, 0])
            self._ax_lr        = self._fig.add_subplot(gs[0, 1])
            self._ax_win_loss  = self._fig.add_subplot(gs[1, 0])
            self._ax_lose_loss = self._fig.add_subplot(gs[1, 1])
            axes = (self._ax_loss, self._ax_lr, self._ax_win_loss, self._ax_lose_loss)
        else:
            gs = self._fig.add_gridspec(2, 1)
            self._fig.subplots_adjust(
                left=0.10, right=0.96, top=0.95, bottom=0.08, hspace=0.40
            )
            self._ax_loss = self._fig.add_subplot(gs[0, 0])
            self._ax_lr   = self._fig.add_subplot(gs[1, 0])
            self._ax_win_loss  = None
            self._ax_lose_loss = None
            axes = (self._ax_loss, self._ax_lr)

        self._ax_loss.set_title("Train Loss", fontsize=10)
        self._ax_lr.set_title("Learning Rate", fontsize=9)
        if is_dpo:
            self._ax_win_loss.set_title("Win Loss", fontsize=9)
            self._ax_lose_loss.set_title("Lose Loss", fontsize=9)

        for ax in axes:
            ax.grid(True)
            ax.set_xlabel("step", fontsize=8)

        self._graph_layout_is_dpo = is_dpo
        self._canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    # ポーリング & ログパース
    # ─────────────────────────────────────────────────────────────────
    def _poll(self) -> None:
        updated = False
        try:
            for _ in range(500):
                try:
                    line = self._state._monitor_queue.get_nowait()
                    self._parse_line(line)
                    updated = True
                except _queue_mod.Empty:
                    break
        except Exception:
            pass

        if updated:
            self._update_params()
            self._update_time()
            self._update_graph()
        self._refresh_dpo_visibility()

        self._parent.after(300, self._poll)

    def _parse_line(self, line: str) -> None:
        # ── step 進捗 (tqdm: "N/total [") ────────────────────────────
        m = _RE_STEP_POSTFIX.search(line)
        if m:
            self._step_in_run = int(m.group(1))
            self._total_steps = int(m.group(2))

        # ── Train Loss (tqdm postfix: ", loss=0.0044") ────────────────
        m = _RE_LOSS_TQDM.search(line)
        if m:
            val = float(m.group(1))
            self._global_step += 1
            self._steps.append(self._global_step)
            self._train_loss.append(val)
            self._last_train = val

            now = time.time()
            if self._last_step_ts is not None:
                self._step_times.append(now - self._last_step_ts)
                if len(self._step_times) > 50:
                    self._step_times.pop(0)
            self._last_step_ts = now
            if self._start_time is None:
                self._start_time = now

            # LR は学習ログに出現しないため state.lr から直接読む
            try:
                lr_val = float(self._state.lr.get())
                if math.isfinite(lr_val) and lr_val > 0.0:
                    self._last_lr = lr_val
                    self._lr_vals.append(lr_val)
            except Exception:
                pass

            # EarlyStopping チェック
            self._check_es(val)

        # ── DPOモード: Win Loss / Lose Loss (tqdm postfix) ──────────────
        m_win = _RE_WIN_LOSS_TQDM.search(line)
        m_lose = _RE_LOSE_LOSS_TQDM.search(line)
        if m_win and m_lose:
            win_val = float(m_win.group(1))
            lose_val = float(m_lose.group(1))
            self._win_loss.append(win_val)
            self._lose_loss.append(lose_val)
            self._last_win_loss = win_val
            self._last_lose_loss = lose_val
            self._check_es_dpo(win_val, lose_val)

        # ── tqdm 時間フィールドから eta を取得 ────────────────────────
        m = _RE_TQDM_TIME.search(line)
        if m:
            try:
                eta_str = m.group(2)
                parts = eta_str.split(":")
                if len(parts) == 2:
                    eta_sec = int(parts[0]) * 60 + int(parts[1])
                else:
                    eta_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                self._tqdm_eta_sec = float(eta_sec)

                elapsed_str = m.group(1)
                ep = elapsed_str.split(":")
                elapsed_sec = int(ep[0]) * 60 + int(ep[1])
                if self._start_time is None:
                    self._start_time = time.time() - elapsed_sec
            except Exception:
                pass

        # ── サンプル生成ログ ─────────────────────────────────────────
        if "Generating sample images at step" in line:
            step_info = ""
            ms = _RE_STEP_POSTFIX.search(line)
            if ms:
                step_info = gettext("leco_monitor_sample_step_info", step=ms.group(1))
            self._append_report(
                gettext("leco_monitor_sample_start", info=step_info), "sample_info"
            )
        elif "sample images saved" in line.lower() or "sample saved" in line.lower():
            self._append_report(
                gettext("leco_monitor_sample_done", line=line.strip()), "sample_info"
            )
        elif "  prompt:" in line and ", size:" in line:
            self._append_report(
                gettext("leco_monitor_sample", line=line.strip()), "sample_info"
            )

        # ── 診断 ─────────────────────────────────────────────────────
        self._auto_diagnose(line)

    def _auto_diagnose(self, line: str) -> None:
        # Train Loss NaN/Inf
        m = _RE_LOSS_TQDM.search(line)
        if m:
            try:
                v = float(m.group(1))
                if math.isnan(v) or math.isinf(v):
                    self._append_report(
                        gettext("leco_monitor_loss_nan"),
                        "danger",
                    )
            except ValueError:
                pass

    def _check_es(self, current_loss: float) -> None:
        """Train Loss の連続上昇を監視して警告/緊急停止を行う。"""
        try:
            enabled  = bool(self._state.es_enabled.get())
            patience = int(self._state.es_patience.get())
        except Exception:
            self._es_status_var.set(gettext("leco_monitor_es_disabled"))
            self._es_progress["value"] = 0
            return

        if not enabled or patience <= 0:
            self._es_status_var.set(gettext("leco_monitor_es_disabled"))
            self._es_progress["value"] = 0
            return

        if not math.isnan(self._es_prev_loss):
            if current_loss > self._es_prev_loss:
                self._es_rise_count += 1
            else:
                if self._es_rise_count > 0:
                    self._append_report(
                        gettext("leco_monitor_es_reset", old=self._es_rise_count),
                        "normal",
                    )
                self._es_rise_count = 0
                self._es_warned     = False
        self._es_prev_loss = current_loss

        pct = int(self._es_rise_count / patience * 100)
        self._es_progress["value"] = min(pct, 100)
        self._es_status_var.set(
            gettext("leco_monitor_es_rise", count=self._es_rise_count, patience=patience)
        )

        warn_threshold = max(1, patience // 2)
        if self._es_rise_count >= warn_threshold and not self._es_warned:
            self._es_warned = True
            self._append_report(
                gettext(
                    "leco_monitor_es_warn",
                    warn=warn_threshold,
                    count=self._es_rise_count,
                    patience=patience,
                ),
                "es_warn",
            )

        if self._es_rise_count >= patience and not self._es_stopped:
            self._es_stopped = True
            self._append_report(
                gettext("leco_monitor_es_stop", patience=patience),
                "es_stop",
            )
            self._stop_training()

    def _check_es_dpo(self, win_loss: float, lose_loss: float) -> None:
        """Win Loss低下・Lose Loss増加の正しい推移を監視して警告/緊急停止を行う。

        受け取るwin_loss/lose_lossは、referenceモデルのwin/lose各々に対する
        ベースライン誤差(ref_loss_win, ref_loss_lose)で正規化済みの相対指標
        ((loss_win-ref_loss_win)/ref_loss_win 等)。referenceモデルがwin画像を
        もともと得意としている(ref_loss_winが小さい)場合でも、絶対量ではなく
        相対変化で判定するため公平に推移を検出できる。

        毎stepの判定:
            Win Lossが前stepより低下 かつ Lose Lossが前stepより増加
                → カウントを0にリセット (正常な推移)
            いずれかを満たさない
                → カウントを+1
        """
        try:
            enabled  = bool(self._state.es_dpo_enabled.get())
            patience = int(self._state.es_dpo_patience.get())
        except Exception:
            self._es_dpo_status_var.set(gettext("leco_monitor_es_disabled"))
            self._es_dpo_progress["value"] = 0
            return

        if not enabled or patience <= 0:
            self._es_dpo_status_var.set(gettext("leco_monitor_es_disabled"))
            self._es_dpo_progress["value"] = 0
            return

        if not math.isnan(self._es_dpo_prev_win) and not math.isnan(self._es_dpo_prev_lose):
            is_healthy = win_loss < self._es_dpo_prev_win and lose_loss > self._es_dpo_prev_lose
            if is_healthy:
                if self._es_dpo_count > 0:
                    self._append_report(
                        gettext("addift_monitor_es_dpo_reset", old=self._es_dpo_count),
                        "normal",
                    )
                self._es_dpo_count  = 0
                self._es_dpo_warned = False
            else:
                self._es_dpo_count += 1
        self._es_dpo_prev_win  = win_loss
        self._es_dpo_prev_lose = lose_loss

        pct = int(self._es_dpo_count / patience * 100)
        self._es_dpo_progress["value"] = min(pct, 100)
        self._es_dpo_status_var.set(
            gettext("addift_monitor_es_dpo_progress", count=self._es_dpo_count, patience=patience)
        )

        warn_threshold = max(1, patience // 2)
        if self._es_dpo_count >= warn_threshold and not self._es_dpo_warned:
            self._es_dpo_warned = True
            self._append_report(
                gettext(
                    "addift_monitor_es_dpo_warn",
                    warn=warn_threshold,
                    count=self._es_dpo_count,
                    patience=patience,
                ),
                "es_warn",
            )

        if self._es_dpo_count >= patience and not self._es_dpo_stopped:
            self._es_dpo_stopped = True
            self._append_report(
                gettext("addift_monitor_es_dpo_stop", patience=patience),
                "es_stop",
            )
            self._stop_training()

    # ─────────────────────────────────────────────────────────────────
    # レポート書き込み
    # ─────────────────────────────────────────────────────────────────
    def _append_report(self, msg: str, tag: str) -> None:
        valid_tags = {"normal", "caution", "warning", "danger", "info",
                      "diag", "sample_info", "es_warn", "es_stop"}
        if tag not in valid_tags:
            tag = "info"

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}\n"

        self._report_text.config(state=tk.NORMAL)
        self._report_text.insert(tk.END, text, tag)

        lines = int(self._report_text.index(tk.END).split(".")[0])
        if lines > self.MAX_REPORT_LINES:
            self._report_text.delete("1.0", f"{lines - self.MAX_REPORT_LINES}.0")

        self._report_text.see(tk.END)
        self._report_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────
    # パラメーター表示更新
    # ─────────────────────────────────────────────────────────────────
    def _update_params(self) -> None:
        if self._total_steps:
            st_str = f"{self._step_in_run}/{self._total_steps}"
        else:
            st_str = str(self._global_step)

        self._param_vars["step"].set(st_str)
        self._param_vars["lr"].set(_fmt(self._last_lr, ".3e"))
        self._param_vars["train_loss"].set(_fmt(self._last_train, ".6f"))
        self._param_vars["win_loss"].set(_fmt(self._last_win_loss, "+.6f"))
        self._param_vars["lose_loss"].set(_fmt(self._last_lose_loss, "+.6f"))

    def _update_time(self) -> None:
        if self._start_time is None:
            return

        now = time.time()
        elapsed = now - self._start_time
        self._time_vars["start_time"].set(
            datetime.datetime.fromtimestamp(self._start_time).strftime("%H:%M:%S")
        )
        self._time_vars["elapsed"].set(_fmt_duration(elapsed))

        # tqdm から直接 eta が得られている場合はそれを優先
        if self._tqdm_eta_sec is not None:
            eta_sec = self._tqdm_eta_sec
            eta_dt  = datetime.datetime.fromtimestamp(now + eta_sec)
            self._time_vars["eta_remain"].set(_fmt_duration(eta_sec))
            self._time_vars["eta_clock"].set(eta_dt.strftime("%H:%M:%S"))
        elif self._step_times and self._total_steps > 0:
            avg_step_sec = sum(self._step_times) / len(self._step_times)
            remaining = max(0, self._total_steps - self._step_in_run)
            eta_sec = avg_step_sec * remaining
            eta_dt  = datetime.datetime.fromtimestamp(now + eta_sec)
            self._time_vars["eta_remain"].set(_fmt_duration(eta_sec))
            self._time_vars["eta_clock"].set(eta_dt.strftime("%H:%M:%S"))
        else:
            self._time_vars["eta_remain"].set(gettext("lora_monitor_time_calculating"))
            self._time_vars["eta_clock"].set("—")

    # ─────────────────────────────────────────────────────────────────
    # グラフ更新
    # ─────────────────────────────────────────────────────────────────
    def _update_graph(self) -> None:
        if not self._mpl_ok or not self._steps:
            return

        try:
            # Train Loss (上段左)
            ax = self._ax_loss
            ax.cla()
            if self._train_loss:
                ax.plot(
                    self._steps[:len(self._train_loss)],
                    self._train_loss,
                    color=COLOR_TRAIN_LOSS, linewidth=1.2, label="Train Loss", alpha=0.9,
                )
            ax.set_title("Train Loss", fontsize=10, color="#CBD5E1")
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("loss", fontsize=8)
            ax.grid(True)
            ax.legend(fontsize=7, loc="upper right")

            # LR (上段右)
            ax2 = self._ax_lr
            ax2.cla()
            if self._lr_vals:
                ax2.plot(
                    self._steps[:len(self._lr_vals)],
                    self._lr_vals,
                    color="#A3E635", linewidth=1.0,
                )
                ax2.set_yscale("log")
            ax2.set_title("Learning Rate", fontsize=9, color="#CBD5E1")
            ax2.set_xlabel("step", fontsize=8)
            ax2.grid(True)

            # Win Loss / Lose Loss (下段、DPOモード時のみ軸が存在する)
            is_dpo = _is_dpo_mode_active(self._state)
            if is_dpo and self._ax_win_loss is not None and self._ax_lose_loss is not None:
                ax3 = self._ax_win_loss
                ax4 = self._ax_lose_loss
                ax3.cla()
                ax4.cla()
                if self._win_loss:
                    ax3.plot(
                        self._steps[:len(self._win_loss)],
                        self._win_loss,
                        color=COLOR_WIN_LOSS, linewidth=1.2, label="Win Loss", alpha=0.9,
                    )
                ax3.set_title("Win Loss", fontsize=9, color="#CBD5E1")
                ax3.set_xlabel("step", fontsize=8)
                ax3.set_ylabel("(loss_win - ref_loss_win) / ref_loss_win", fontsize=7)
                ax3.grid(True)

                if self._lose_loss:
                    ax4.plot(
                        self._steps[:len(self._lose_loss)],
                        self._lose_loss,
                        color=COLOR_LOSE_LOSS, linewidth=1.2, label="Lose Loss", alpha=0.9,
                    )
                ax4.set_title("Lose Loss", fontsize=9, color="#CBD5E1")
                ax4.set_xlabel("step", fontsize=8)
                ax4.set_ylabel("(loss_lose - ref_loss_lose) / ref_loss_lose", fontsize=7)
                ax4.grid(True)

            self._canvas.draw_idle()

        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # リセット / 停止
    # ─────────────────────────────────────────────────────────────────
    def _reset(self) -> None:
        self._steps.clear()
        self._train_loss.clear()
        self._lr_vals.clear()
        self._win_loss.clear()
        self._lose_loss.clear()
        self._global_step    = 0
        self._step_in_run    = 0
        self._total_steps    = 0
        self._last_lr        = 0.0
        self._last_train     = float("nan")
        self._last_win_loss  = float("nan")
        self._last_lose_loss = float("nan")
        self._es_rise_count  = 0
        self._es_prev_loss   = float("nan")
        self._es_warned      = False
        self._es_stopped     = False
        self._es_dpo_count     = 0
        self._es_dpo_prev_win  = float("nan")
        self._es_dpo_prev_lose = float("nan")
        self._es_dpo_warned    = False
        self._es_dpo_stopped   = False
        self._tqdm_eta_sec   = None
        self._start_time     = None
        self._last_step_ts   = None
        self._step_times.clear()
        self._tqdm_eta_sec  = None

        for v in self._param_vars.values():
            v.set("—")
        for v in self._time_vars.values():
            v.set("—")

        self._report_text.config(state=tk.NORMAL)
        self._report_text.delete("1.0", tk.END)
        self._report_text.config(state=tk.DISABLED)

        if self._mpl_ok:
            for ax in (self._ax_loss, self._ax_lr, self._ax_win_loss, self._ax_lose_loss):
                if ax is not None:
                    ax.cla()
                    ax.grid(True)
            self._canvas.draw_idle()
        self._es_status_var.set(gettext("leco_monitor_es_disabled"))
        self._es_progress["value"] = 0
        self._es_dpo_status_var.set(gettext("leco_monitor_es_disabled"))
        self._es_dpo_progress["value"] = 0
        self._refresh_dpo_visibility()

    def _stop_training(self) -> None:
        proc = getattr(self._state, "_proc", None)
        if proc is None or proc.poll() is not None:
            self._state.log_fn(gettext("leco_monitor_no_proc"))
            return

        import os
        import signal

        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        except Exception:
            proc.terminate()
        self._state.status_var.set(gettext("status_stop_requested"))
        self._state.log_fn(gettext("leco_monitor_stop_sent"))


# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(v: float, fmt: str) -> str:
    try:
        if math.isnan(v):
            return "—"
        return format(v, fmt)
    except Exception:
        return "—"


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return gettext("lora_monitor_duration_hms", h=h, m=f"{m:02d}", s=f"{s:02d}")
    return gettext("lora_monitor_duration_ms", m=m, s=f"{s:02d}")
