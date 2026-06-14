"""app/monitor_graph_leco.py — LECO学習リアルタイムモニタリングウィジェット

_build_monitor_tab(parent, state) から呼び出される LecoMonitorGraph クラスを提供する。

lora_train の MonitorGraph との主な差異:
  - Val Loss / ΔLoss を監視しない
  - EarlyStopping パネルを省略
  - state._monitor_queue を参照（leco_train._LecoTrainState 互換）
  - ステップ管理（epochではなくstep単位）に合わせた表示
"""
from __future__ import annotations

import datetime
import math
import queue as _queue_mod
import re
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

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
    from .leco_train import _LecoTrainState

# ──────────────────────────────────────────────────────────────────────────────
# 色定数
# ──────────────────────────────────────────────────────────────────────────────
COLOR_NORMAL   = "#16A34A"
COLOR_CAUTION  = "#CA8A04"
COLOR_WARNING  = "#EA580C"
COLOR_DANGER   = "#DC2626"
COLOR_INFO     = "#E2E8F0"

# ──────────────────────────────────────────────────────────────────────────────
# ログパースパターン
# ──────────────────────────────────────────────────────────────────────────────
# tqdm postfix: "loss=0.0044" （LECO はこの形式で出力される）
_RE_LOSS_TQDM    = re.compile(r",\s*loss=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
# tqdm 時間フィールド: "[00:03<26:53,  3.23s/it"
_RE_TQDM_TIME    = re.compile(r"\[(\d+:\d+)<((?:\d+:)?\d+:\d+),\s*([0-9.]+)s/it")
_RE_STEP_POSTFIX = re.compile(r"(\d+)/(\d+)\s*\[")   # tqdm: "120/500 ["


class LecoMonitorGraph:
    """LECO学習リアルタイムモニタリングウィジェット。

    Parameters
    ----------
    parent : ttk.Frame
        モニターグラフタブの親フレーム
    state : _LecoTrainState
        leco_train.py の状態オブジェクト。_monitor_queue を参照する。
    """

    MAX_REPORT_LINES = 200

    def __init__(self, parent: ttk.Frame, state: "_LecoTrainState") -> None:
        self._state  = state
        self._parent = parent

        # ── データ系列 ──────────────────────────────────────────────
        self._steps:      list[int]   = []
        self._train_loss: list[float] = []
        self._lr_vals:    list[float] = []

        # ── 状態変数 ──────────────────────────────────────────────
        self._global_step  = 0
        self._step_in_run  = 0
        self._total_steps  = 0
        self._last_lr      = 0.0
        self._last_train   = float("nan")

        # EarlyStopping 状態
        self._es_rise_count = 0
        self._es_prev_loss  = float("nan")
        self._es_warned     = False
        self._es_stopped    = False

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
        for i, (label, key) in enumerate(rows):
            ttk.Label(lf, text=label + ":", width=12, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2
            )
            v = tk.StringVar(value="—")
            self._param_vars[key] = v
            ttk.Label(lf, textvariable=v, anchor=tk.W,
                      font=("TkFixedFont", 9)).grid(
                row=i, column=1, sticky=tk.EW, padx=(0, 6), pady=2
            )
        lf.columnconfigure(1, weight=1)

        # EarlyStopping パネル
        es_lf = ttk.LabelFrame(parent, text=gettext("lora_monitor_es_title"))
        es_lf.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._es_status_var = tk.StringVar(value=gettext("leco_monitor_es_disabled"))
        ttk.Label(es_lf, textvariable=self._es_status_var,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W, padx=6, pady=(2, 0))
        self._es_progress = ttk.Progressbar(es_lf, maximum=100, value=0, length=180)
        self._es_progress.pack(fill=tk.X, padx=6, pady=(2, 4))

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

            # サブプロット: Loss(上大) / LR(下)
            self._fig = Figure(figsize=(7, 8))
            gs = self._fig.add_gridspec(2, 1, height_ratios=[3, 2])
            self._fig.subplots_adjust(
                left=0.10, right=0.95, top=0.95, bottom=0.07, hspace=0.45
            )

            self._ax_loss = self._fig.add_subplot(gs[0])
            self._ax_lr   = self._fig.add_subplot(gs[1])

            self._ax_loss.set_title("Train Loss", fontsize=10)
            self._ax_lr.set_title("Learning Rate", fontsize=9)

            for ax in (self._ax_loss, self._ax_lr):
                ax.grid(True)
                ax.set_xlabel("step", fontsize=8)
            self._canvas = FigureCanvasTkAgg(self._fig,
                                             master=self._graph_frame)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_ok = True

        except Exception as exc:
            self._mpl_ok = False
            ttk.Label(
                self._graph_frame,
                text=gettext("lora_monitor_mpl_error", error=exc),
                foreground="#EF4444",
                justify=tk.LEFT,
            ).pack(padx=16, pady=16, anchor=tk.NW)

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
            # Train Loss
            ax = self._ax_loss
            ax.cla()
            if self._train_loss:
                ax.plot(
                    self._steps[:len(self._train_loss)],
                    self._train_loss,
                    color="#38BDF8", linewidth=1.2, label="Train Loss", alpha=0.9,
                )
            ax.set_title("Train Loss", fontsize=10, color="#CBD5E1")
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("loss", fontsize=8)
            ax.grid(True)
            ax.legend(fontsize=7, loc="upper right")

            # LR
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
        self._global_step    = 0
        self._step_in_run    = 0
        self._total_steps    = 0
        self._last_lr        = 0.0
        self._last_train     = float("nan")
        self._es_rise_count  = 0
        self._es_prev_loss   = float("nan")
        self._es_warned      = False
        self._es_stopped     = False
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
            for ax in (self._ax_loss, self._ax_lr):
                ax.cla()
                ax.grid(True)
            self._canvas.draw_idle()
        self._es_status_var.set(gettext("leco_monitor_es_disabled"))
        self._es_progress["value"] = 0

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
