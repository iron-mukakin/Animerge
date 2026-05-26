# 変更日: 2026-05-26  モニターグラフ実装 (monitor_graph.py 新規作成)
"""app/monitor_graph.py — LoRA学習リアルタイムモニタリングウィジェット

_build_monitor_tab(parent, state) から呼び出されるMonitorGraphクラスを提供する。
学習プロセスの _log_queue から流れるログ行をパースして以下を表示する:

グラフ (matplotlib FigureCanvasTkAgg):
    - Train Loss (avr_loss=)
    - Val Loss   (val_epoch_avg_loss= / val_avg_loss=)
    - LR         (lr=)
    - grad_norm  (grad_norm=)
    - loss_scale (loss_scale=)
    - ΔLoss      (Val - Train の差分、独立サブプロット)

パラメーターパネル (右側):
    - epoch / step / LR / Train Loss / Val Loss / ΔLoss / grad_norm / loss_scale
    - EarlyStopping カウンタ進捗バー
    - 推定残り時間 / 開始時刻 / 完了予測時刻

自動レポートパネル:
    - EarlyStoppingメッセージを色分けで表示
    - 学習状態の自動診断メッセージ
"""
from __future__ import annotations

import re
import time
import datetime
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lora_train import _TrainState

# ──────────────────────────────────────────────────────────────────────────────
# 色定数
# ──────────────────────────────────────────────────────────────────────────────
COLOR_NORMAL   = "#16A34A"   # 緑
COLOR_CAUTION  = "#CA8A04"   # 黄
COLOR_WARNING  = "#EA580C"   # 橙
COLOR_DANGER   = "#DC2626"   # 赤
COLOR_INFO     = "#E2E8F0"   # 明るいグレー（通常ログ）

# ──────────────────────────────────────────────────────────────────────────────
# ログパースパターン
# ──────────────────────────────────────────────────────────────────────────────
_RE_AVR_LOSS     = re.compile(r"avr_loss=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_VAL_EPOCH    = re.compile(r"val_epoch_avg_loss=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_VAL_STEP     = re.compile(r"val_avg_loss=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_LR           = re.compile(r"\blr=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_EPOCH        = re.compile(r"epoch\s+([0-9]+)/([0-9]+)")
_RE_GRAD_NORM    = re.compile(r"grad_norm=([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_LOSS_SCALE   = re.compile(r"loss_scale=([0-9]+(?:\.[0-9]+)?)")
_RE_STEP_POSTFIX = re.compile(r"(\d+)/(\d+)\s*\[")   # tqdm step: "120/400 ["

# EarlyStopping メッセージキーワード → (色, 重大度)
_ES_KEYWORDS = [
    ("[緊急停止]",   COLOR_DANGER,  4),
    ("[警告]",       COLOR_WARNING, 3),
    ("[注意]",       COLOR_CAUTION, 2),
    ("[監視正常化]", COLOR_NORMAL,  1),
    ("[正常]",       COLOR_NORMAL,  1),
    ("[EarlyStopping]", COLOR_INFO, 0),
]


def _classify_es_message(msg: str) -> tuple[str, int]:
    """EarlyStoppingメッセージを色と重大度に分類する。"""
    for kw, color, severity in _ES_KEYWORDS:
        if kw in msg:
            return color, severity
    return COLOR_INFO, 0


# ──────────────────────────────────────────────────────────────────────────────
# MonitorGraph クラス
# ──────────────────────────────────────────────────────────────────────────────
class MonitorGraph:
    """LoRA学習リアルタイムモニタリングウィジェット。

    Parameters
    ----------
    parent : ttk.Frame
        モニターグラフタブの親フレーム
    state : _TrainState
        lora_train.py の状態オブジェクト。_log_queue を参照する。
    """

    MAX_REPORT_LINES = 200   # 自動レポート最大保持行数

    def __init__(self, parent: ttk.Frame, state: "_TrainState") -> None:
        self._state = state
        self._parent = parent

        # ── データ系列 ──────────────────────────────────────────────
        self._steps:      list[int]   = []   # global step インデックス
        self._train_loss: list[float] = []
        self._val_loss:   list[float] = []   # epochまたはstep validation
        self._lr_vals:    list[float] = []
        self._grad_norms: list[float] = []
        self._loss_scales:list[float] = []
        self._delta_loss: list[float] = []   # val - train (epochごと)

        # x軸: step系とepoch系を分けて管理
        self._val_steps:  list[int]   = []   # val_lossに対応するstep
        self._delta_steps:list[int]   = []

        # ── 状態変数 ──────────────────────────────────────────────
        self._global_step  = 0
        self._current_epoch = 0
        self._total_epochs  = 0
        self._step_in_epoch = 0
        self._total_steps   = 0
        self._last_lr       = 0.0
        self._last_train    = float("nan")
        self._last_val      = float("nan")
        self._last_grad     = float("nan")
        self._last_scale    = float("nan")
        self._es_patience   = 3
        self._es_count      = 0

        # 時間計測
        self._start_time:  float | None = None
        self._step_times:  list[float]  = []   # 直近N stepの所要時間 (秒)
        self._last_step_ts: float | None = None

        # ── UI構築 ──────────────────────────────────────────────────
        self._build_ui(parent)

        # ── matplotlib 初期化 ───────────────────────────────────────
        self._init_matplotlib()

        # ── ポーリング開始 ──────────────────────────────────────────
        parent.after(300, self._poll)

    # ─────────────────────────────────────────────────────────────────
    # UI構築
    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self, parent: ttk.Frame) -> None:
        """左: グラフエリア(70%) / 右: パラメーター+レポート(30%)"""
        parent.columnconfigure(0, weight=7)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(0, weight=1)

        # ── 左ペイン: グラフ ────────────────────────────────────────
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        self._graph_frame = left

        # ── 右ペイン: パラメーター + レポート ──────────────────────
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky=tk.NSEW)
        right.rowconfigure(1, weight=1)

        self._build_param_panel(right)
        self._build_report_panel(right)

        # リセットボタン
        ttk.Button(right, text="グラフリセット", command=self._reset).pack(
            fill=tk.X, padx=4, pady=(4, 0)
        )

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text="学習パラメーター")
        lf.pack(fill=tk.X, padx=4, pady=(4, 4))

        self._param_vars: dict[str, tk.StringVar] = {}

        rows = [
            ("epoch",      "epoch"),
            ("step",       "step"),
            ("LR",         "lr"),
            ("Train Loss", "train_loss"),
            ("Val Loss",   "val_loss"),
            ("ΔLoss",      "delta_loss"),
            ("grad norm",  "grad_norm"),
            ("loss scale", "loss_scale"),
        ]
        for i, (label, key) in enumerate(rows):
            ttk.Label(lf, text=label + ":", width=12, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=2
            )
            v = tk.StringVar(value="—")
            self._param_vars[key] = v
            ttk.Label(lf, textvariable=v, anchor=tk.W, font=("TkFixedFont", 9)).grid(
                row=i, column=1, sticky=tk.EW, padx=(0, 6), pady=2
            )
        lf.columnconfigure(1, weight=1)

        # EarlyStopping 進捗バー
        es_lf = ttk.LabelFrame(parent, text="EarlyStopping")
        es_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._es_var = tk.StringVar(value="無効 / 非設定")
        ttk.Label(es_lf, textvariable=self._es_var, font=("TkFixedFont", 9)).pack(
            anchor=tk.W, padx=6, pady=(2, 0)
        )
        self._es_progress = ttk.Progressbar(es_lf, maximum=100, value=0, length=180)
        self._es_progress.pack(fill=tk.X, padx=6, pady=(2, 4))

        # 時間情報
        time_lf = ttk.LabelFrame(parent, text="時間")
        time_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        time_rows = [
            ("開始",       "start_time"),
            ("完了予測",   "eta_clock"),
            ("残り時間",   "eta_remain"),
            ("経過",       "elapsed"),
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
        lf = ttk.LabelFrame(parent, text="自動レポート")
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self._report_text = tk.Text(
            lf,
            height=12,
            wrap=tk.WORD,
            font=("TkFixedFont", 8),
            state=tk.DISABLED,
            bg="#1E293B",
            fg="#CBD5E1",
        )
        scroll = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self._report_text.yview)
        self._report_text.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        self._report_text.grid(row=0, column=0, sticky=tk.NSEW)

        # タグ定義（色分け）
        self._report_text.tag_configure("normal",   foreground=COLOR_NORMAL)
        self._report_text.tag_configure("caution",  foreground=COLOR_CAUTION)
        self._report_text.tag_configure("warning",  foreground=COLOR_WARNING)
        self._report_text.tag_configure("danger",   foreground=COLOR_DANGER)
        self._report_text.tag_configure("info",     foreground=COLOR_INFO)
        self._report_text.tag_configure("diag",     foreground="#7DD3FC")   # 診断メッセージ: 水色

    # ─────────────────────────────────────────────────────────────────
    # matplotlib 初期化
    # ─────────────────────────────────────────────────────────────────
    def _init_matplotlib(self) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # ダークスタイル適用
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

            # サブプロット構成: Loss(上大), LR(中), grad/scale(下左右)
            self._fig = Figure(figsize=(7, 8), tight_layout=True)
            gs = self._fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5], hspace=0.4, wspace=0.35)

            self._ax_loss  = self._fig.add_subplot(gs[0, :])   # 上段全幅: Loss + ΔLoss
            self._ax_lr    = self._fig.add_subplot(gs[1, :])   # 中段全幅: LR
            self._ax_grad  = self._fig.add_subplot(gs[2, 0])   # 下段左: grad_norm
            self._ax_scale = self._fig.add_subplot(gs[2, 1])   # 下段右: loss_scale

            self._ax_loss.set_title("Loss", fontsize=10)
            self._ax_lr.set_title("Learning Rate", fontsize=9)
            self._ax_grad.set_title("grad norm", fontsize=9)
            self._ax_scale.set_title("loss scale", fontsize=9)

            for ax in (self._ax_loss, self._ax_lr, self._ax_grad, self._ax_scale):
                ax.grid(True)
                ax.set_xlabel("step", fontsize=8)

            # ΔLoss 用右軸
            self._ax_delta = self._ax_loss.twinx()
            self._ax_delta.set_ylabel("ΔLoss (val-train)", color="#F472B6", fontsize=8)
            self._ax_delta.tick_params(axis="y", colors="#F472B6")

            self._canvas = FigureCanvasTkAgg(self._fig, master=self._graph_frame)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_ok = True

        except Exception as exc:
            self._mpl_ok = False
            ttk.Label(
                self._graph_frame,
                text=f"matplotlib を読み込めませんでした。\n{exc}\n\npip install matplotlib で解決します。",
                foreground="#EF4444",
                justify=tk.LEFT,
            ).pack(padx=16, pady=16, anchor=tk.NW)

    # ─────────────────────────────────────────────────────────────────
    # ポーリング & ログパース
    # ─────────────────────────────────────────────────────────────────
    def _poll(self) -> None:
        """300ms ごとに _monitor_queue を消費してデータを更新する。
        _monitor_queue は _log_queue とは独立しており、
        _worker が両方に同じ行を投入する。
        """
        import queue as _queue
        updated = False
        try:
            for _ in range(500):   # 一度に最大500行を処理（バースト対応）
                try:
                    line = self._state._monitor_queue.get_nowait()
                    self._parse_line(line)
                    updated = True
                except _queue.Empty:
                    break
        except Exception:
            pass

        if updated:
            self._update_params()
            self._update_time()
            self._update_graph()

        self._parent.after(300, self._poll)

    def _parse_line(self, line: str) -> None:
        """ログ1行をパースしてデータ系列に追加する。"""
        # epoch 進捗
        m = _RE_EPOCH.search(line)
        if m:
            self._current_epoch = int(m.group(1))
            self._total_epochs  = int(m.group(2))
            if self._start_time is None:
                self._start_time = time.time()

        # step 進捗 (tqdm形式 "120/400 [")
        m = _RE_STEP_POSTFIX.search(line)
        if m:
            self._step_in_epoch = int(m.group(1))
            self._total_steps   = int(m.group(2))

        # Train Loss (avr_loss=)
        m = _RE_AVR_LOSS.search(line)
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

        # LR
        m = _RE_LR.search(line)
        if m:
            self._last_lr = float(m.group(1))
            self._lr_vals.append(self._last_lr)

        # grad_norm
        m = _RE_GRAD_NORM.search(line)
        if m:
            v = float(m.group(1))
            self._last_grad = v
            self._grad_norms.append(v)

        # loss_scale
        m = _RE_LOSS_SCALE.search(line)
        if m:
            v = float(m.group(1))
            self._last_scale = v
            self._loss_scales.append(v)

        # Val Loss (epoch)
        m = _RE_VAL_EPOCH.search(line)
        if m:
            v = float(m.group(1))
            self._last_val = v
            self._val_loss.append(v)
            self._val_steps.append(self._global_step)
            # ΔLoss
            if not _isnan(self._last_train):
                delta = v - self._last_train
                self._delta_loss.append(delta)
                self._delta_steps.append(self._global_step)

        # Val Loss (step)
        m = _RE_VAL_STEP.search(line)
        if m and not _RE_VAL_EPOCH.search(line):
            self._last_val = float(m.group(1))

        # EarlyStopping メッセージ
        for kw, _, _ in _ES_KEYWORDS:
            if kw in line:
                self._append_report(line, kw)
                self._update_es_counter(line)
                break

        # 診断ログ
        self._auto_diagnose(line)

    def _update_es_counter(self, line: str) -> None:
        """EarlyStopping カウンタを行から読み取る。例: (2/3)"""
        m = re.search(r"\((\d+)/(\d+)\)", line)
        if m:
            self._es_count    = int(m.group(1))
            self._es_patience = int(m.group(2))
        elif "[監視正常化]" in line or "[正常]" in line:
            self._es_count = 0

    def _auto_diagnose(self, line: str) -> None:
        """train_loss / grad_norm の異常を自動検出してレポートに追記する。"""
        # grad_norm 急増検知 (直近平均の3倍超)
        if len(self._grad_norms) >= 5:
            recent_avg = sum(self._grad_norms[-5:-1]) / 4
            latest = self._grad_norms[-1]
            if recent_avg > 0 and latest > recent_avg * 3.0:
                msg = (
                    f"[診断] grad_norm が急増しています "
                    f"({latest:.3f} / 直近平均 {recent_avg:.3f})。"
                    "LR低下またはmax_grad_norm調整を検討してください。"
                )
                self._append_report(msg, "diag")

        # loss_scale 低下検知 (65536 → 1 付近)
        if len(self._loss_scales) >= 2:
            prev  = self._loss_scales[-2]
            latest = self._loss_scales[-1]
            if prev > 64 and latest <= 1:
                msg = (
                    "[診断] loss_scale が最小値付近まで低下しました。"
                    "NaN発生 / 学習崩壊の可能性があります。"
                )
                self._append_report(msg, "diag")

        # Train Loss NaN / Inf 検知
        m = _RE_AVR_LOSS.search(line)
        if m:
            try:
                v = float(m.group(1))
                import math
                if math.isnan(v) or math.isinf(v):
                    self._append_report(
                        "[診断] Train Loss が NaN/Inf になっています。学習を停止することを推奨します。",
                        "danger",
                    )
            except ValueError:
                pass

    # ─────────────────────────────────────────────────────────────────
    # レポート書き込み
    # ─────────────────────────────────────────────────────────────────
    def _append_report(self, msg: str, keyword_or_tag: str) -> None:
        """自動レポートテキストに色付きで1行追記する。"""
        # タグ決定
        tag = "info"
        if keyword_or_tag in ("diag", "danger"):
            tag = keyword_or_tag if keyword_or_tag in (
                "normal", "caution", "warning", "danger", "info", "diag") else "info"
        else:
            for kw, color, severity in _ES_KEYWORDS:
                if kw in keyword_or_tag or keyword_or_tag == kw:
                    if severity >= 4:
                        tag = "danger"
                    elif severity == 3:
                        tag = "warning"
                    elif severity == 2:
                        tag = "caution"
                    elif severity >= 1:
                        tag = "normal"
                    else:
                        tag = "info"
                    break

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}\n"

        self._report_text.config(state=tk.NORMAL)
        self._report_text.insert(tk.END, text, tag)

        # 行数制限
        lines = int(self._report_text.index(tk.END).split(".")[0])
        if lines > self.MAX_REPORT_LINES:
            self._report_text.delete("1.0", f"{lines - self.MAX_REPORT_LINES}.0")

        self._report_text.see(tk.END)
        self._report_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────
    # パラメーター表示更新
    # ─────────────────────────────────────────────────────────────────
    def _update_params(self) -> None:
        ep_str = f"{self._current_epoch}/{self._total_epochs}" if self._total_epochs else "—"
        st_str = f"{self._step_in_epoch}/{self._total_steps}" if self._total_steps else str(self._global_step)

        self._param_vars["epoch"].set(ep_str)
        self._param_vars["step"].set(st_str)
        self._param_vars["lr"].set(_fmt(self._last_lr, ".3e"))
        self._param_vars["train_loss"].set(_fmt(self._last_train, ".6f"))
        self._param_vars["val_loss"].set(_fmt(self._last_val, ".6f"))

        if not _isnan(self._last_train) and not _isnan(self._last_val):
            delta = self._last_val - self._last_train
            sign = "+" if delta >= 0 else ""
            self._param_vars["delta_loss"].set(f"{sign}{delta:.6f}")
        else:
            self._param_vars["delta_loss"].set("—")

        self._param_vars["grad_norm"].set(_fmt(self._last_grad, ".4f"))
        self._param_vars["loss_scale"].set(_fmt(self._last_scale, ".0f"))

        # EarlyStopping プログレス
        if self._es_patience > 0:
            pct = int(self._es_count / self._es_patience * 100)
            self._es_var.set(f"{self._es_count} / {self._es_patience} 回悪化")
            self._es_progress["value"] = pct
            # バー色 (ttk は直接変更不可なのでスタイルで近似)
        else:
            self._es_var.set("無効 / 非設定")
            self._es_progress["value"] = 0

    def _update_time(self) -> None:
        if self._start_time is None:
            return

        now = time.time()
        elapsed = now - self._start_time
        self._time_vars["start_time"].set(
            datetime.datetime.fromtimestamp(self._start_time).strftime("%H:%M:%S")
        )
        self._time_vars["elapsed"].set(_fmt_duration(elapsed))

        # 残り時間推定: 直近step時間 × 残りstep数
        if self._step_times and self._total_steps > 0 and self._total_epochs > 0:
            avg_step_sec = sum(self._step_times) / len(self._step_times)
            # 残りstep概算: (total_epochs - current_epoch) * steps_per_epoch + (total - current in epoch)
            remaining_epochs = max(0, self._total_epochs - self._current_epoch)
            steps_per_epoch  = self._total_steps
            current_in_epoch = self._step_in_epoch
            remaining_steps  = remaining_epochs * steps_per_epoch + (steps_per_epoch - current_in_epoch)
            eta_sec = avg_step_sec * remaining_steps
            eta_dt  = datetime.datetime.fromtimestamp(now + eta_sec)
            self._time_vars["eta_remain"].set(_fmt_duration(eta_sec))
            self._time_vars["eta_clock"].set(eta_dt.strftime("%H:%M:%S"))
        else:
            self._time_vars["eta_remain"].set("計算中...")
            self._time_vars["eta_clock"].set("—")

    # ─────────────────────────────────────────────────────────────────
    # グラフ更新
    # ─────────────────────────────────────────────────────────────────
    def _update_graph(self) -> None:
        if not self._mpl_ok:
            return
        if not self._steps:
            return

        try:
            # ── Loss subplot ──────────────────────────────────────────
            ax = self._ax_loss
            ax.cla()
            self._ax_delta.cla()

            if self._train_loss:
                ax.plot(
                    self._steps[:len(self._train_loss)],
                    self._train_loss,
                    color="#38BDF8", linewidth=1.2, label="Train Loss", alpha=0.9,
                )
            if self._val_loss:
                ax.plot(
                    self._val_steps[:len(self._val_loss)],
                    self._val_loss,
                    color="#FB923C", linewidth=1.5, linestyle="--",
                    label="Val Loss", marker="o", markersize=4,
                )
            if self._delta_loss:
                self._ax_delta.plot(
                    self._delta_steps[:len(self._delta_loss)],
                    self._delta_loss,
                    color="#F472B6", linewidth=0.9, linestyle=":",
                    label="ΔLoss", alpha=0.8,
                )
                self._ax_delta.set_ylabel("ΔLoss", color="#F472B6", fontsize=8)
                self._ax_delta.tick_params(axis="y", colors="#F472B6", labelsize=7)
                self._ax_delta.axhline(0, color="#F472B6", linewidth=0.5, alpha=0.4)

            ax.set_title("Loss", fontsize=10, color="#CBD5E1")
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("loss", fontsize=8)
            ax.grid(True)
            # 凡例統合
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = self._ax_delta.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=7, loc="upper right")

            # ── LR subplot ────────────────────────────────────────────
            ax2 = self._ax_lr
            ax2.cla()
            if self._lr_vals:
                lr_steps = self._steps[:len(self._lr_vals)]
                ax2.plot(lr_steps, self._lr_vals, color="#A3E635", linewidth=1.0)
                ax2.set_yscale("log")
            ax2.set_title("Learning Rate", fontsize=9, color="#CBD5E1")
            ax2.set_xlabel("step", fontsize=8)
            ax2.grid(True)

            # ── grad_norm subplot ─────────────────────────────────────
            ax3 = self._ax_grad
            ax3.cla()
            if self._grad_norms:
                gn_steps = self._steps[:len(self._grad_norms)]
                ax3.plot(gn_steps, self._grad_norms, color="#C084FC", linewidth=0.9)
                # 急増ライン (平均 × 3)
                if len(self._grad_norms) >= 5:
                    avg = sum(self._grad_norms) / len(self._grad_norms)
                    ax3.axhline(avg * 3, color="#EF4444", linewidth=0.7,
                                linestyle="--", alpha=0.6, label="×3 avg")
                    ax3.legend(fontsize=7)
            ax3.set_title("grad norm", fontsize=9, color="#CBD5E1")
            ax3.set_xlabel("step", fontsize=8)
            ax3.grid(True)

            # ── loss_scale subplot ────────────────────────────────────
            ax4 = self._ax_scale
            ax4.cla()
            if self._loss_scales:
                ls_steps = self._steps[:len(self._loss_scales)]
                ax4.plot(ls_steps, self._loss_scales, color="#FCD34D", linewidth=0.9)
                ax4.axhline(1, color="#EF4444", linewidth=0.7, linestyle="--",
                            alpha=0.5, label="scale=1")
                ax4.legend(fontsize=7)
                ax4.set_yscale("log")
            ax4.set_title("loss scale", fontsize=9, color="#CBD5E1")
            ax4.set_xlabel("step", fontsize=8)
            ax4.grid(True)

            self._canvas.draw_idle()

        except Exception:
            pass   # グラフ更新失敗は無視して学習継続を妨げない

    # ─────────────────────────────────────────────────────────────────
    # リセット
    # ─────────────────────────────────────────────────────────────────
    def _reset(self) -> None:
        self._steps.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._val_steps.clear()
        self._lr_vals.clear()
        self._grad_norms.clear()
        self._loss_scales.clear()
        self._delta_loss.clear()
        self._delta_steps.clear()
        self._global_step   = 0
        self._current_epoch = 0
        self._total_epochs  = 0
        self._step_in_epoch = 0
        self._total_steps   = 0
        self._last_lr       = 0.0
        self._last_train    = float("nan")
        self._last_val      = float("nan")
        self._last_grad     = float("nan")
        self._last_scale    = float("nan")
        self._es_count      = 0
        self._start_time    = None
        self._last_step_ts  = None
        self._step_times.clear()

        for v in self._param_vars.values():
            v.set("—")
        for v in self._time_vars.values():
            v.set("—")
        self._es_var.set("無効 / 非設定")
        self._es_progress["value"] = 0

        self._report_text.config(state=tk.NORMAL)
        self._report_text.delete("1.0", tk.END)
        self._report_text.config(state=tk.DISABLED)

        if self._mpl_ok:
            for ax in (self._ax_loss, self._ax_lr, self._ax_grad, self._ax_scale):
                ax.cla()
                ax.grid(True)
            self._ax_delta.cla()
            self._canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────
def _isnan(v: float) -> bool:
    import math
    try:
        return math.isnan(v)
    except Exception:
        return True


def _fmt(v: float, fmt: str) -> str:
    try:
        import math
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
        return f"{h}時間 {m:02d}分 {s:02d}秒"
    return f"{m}分 {s:02d}秒"
