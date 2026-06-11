"""apply_fix_leco_es.py
以下のパッチを適用する:

[leco_train.py]
  1. _LecoTrainState に EarlyStopping 設定変数を追加
  2. _build_adv_tab に EarlyStopping 設定 UI を追加

[monitor_graph_leco.py]
  1. grad_norm 項目を全箇所から削除
  2. EarlyStopping 実装（状態変数・UIパネル・_check_es メソッド）
  3. _parse_line のサンプル生成検出を強化
  4. _auto_diagnose から grad_norm 急増診断を削除

使い方:
    python apply_fix_leco_es.py [app ディレクトリのパス]
app ディレクトリ省略時はスクリプトと同ディレクトリを対象にする。
"""
from __future__ import annotations
import sys
from pathlib import Path


def _a(t: str) -> str:
    return t.replace("\r\n", "\n").replace("\r", "\n")


def _apply_patches(path: Path, patches: list[tuple[str, str, str]]) -> None:
    src = _a(path.read_text(encoding="utf-8"))
    for name, old, new in patches:
        old_a, new_a = _a(old), _a(new)
        if old_a not in src:
            print(f"  [ERROR] '{name}' が見つかりません。スキップします。")
            sys.exit(1)
        src = src.replace(old_a, new_a, 1)
        print(f"  [OK] {name}")
    path.write_text(src, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# leco_train.py パッチ
# ══════════════════════════════════════════════════════════════════════════════

LECO_PATCHES: list[tuple[str, str, str]] = [

    # 1. _LecoTrainState に ES 変数追加
    (
        "_LecoTrainState ES 変数",
        '''\
        # ステータス
        self.status_var       = tk.StringVar(value="待機中")''',
        '''\
        # ── EarlyStopping ────────────────────────────────────────────
        self.es_enabled   = tk.BooleanVar(value=False)
        self.es_patience  = tk.IntVar(value=5)   # 連続上昇で警告/停止する step 数

        # ステータス
        self.status_var       = tk.StringVar(value="待機中")''',
    ),

    # 2. _build_adv_tab 末尾に ES 設定 UI を追加
    (
        "_build_adv_tab ES UI",
        '''\
    lf2 = ttk.LabelFrame(parent, text="オフロード")
    lf2.pack(fill=tk.X)
    ttk.Checkbutton(lf2, text="split_attn",
                    variable=s.split_attn).grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="unsloth_offload_checkpointing",
                    variable=s.unsloth_offload_checkpointing).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cpu_offload_checkpointing",
                    variable=s.cpu_offload_checkpointing).grid(
        row=0, column=2, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="vae_disable_cache",
                    variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)''',
        '''\
    lf2 = ttk.LabelFrame(parent, text="オフロード")
    lf2.pack(fill=tk.X)
    ttk.Checkbutton(lf2, text="split_attn",
                    variable=s.split_attn).grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="unsloth_offload_checkpointing",
                    variable=s.unsloth_offload_checkpointing).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cpu_offload_checkpointing",
                    variable=s.cpu_offload_checkpointing).grid(
        row=0, column=2, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="vae_disable_cache",
                    variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)

    lf3 = ttk.LabelFrame(parent, text="EarlyStopping（モニター）")
    lf3.pack(fill=tk.X, pady=(8, 0))
    lf3.columnconfigure(3, weight=1)
    ttk.Checkbutton(lf3, text="Train Loss 連続上昇で警告/停止を有効にする",
                    variable=s.es_enabled).grid(
        row=0, column=0, columnspan=4, sticky=tk.W, padx=8, pady=3)
    ttk.Label(lf3, text="監視 step 数（連続上昇でカウント）:", anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(8, 2), pady=3)
    ttk.Spinbox(lf3, from_=2, to=500, textvariable=s.es_patience, width=6).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(lf3,
              text="50% 到達→警告  100% 到達→緊急停止",
              foreground="#64748B").grid(
        row=1, column=2, columnspan=2, sticky=tk.W, padx=(8, 4), pady=3)''',
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# monitor_graph_leco.py パッチ
# ══════════════════════════════════════════════════════════════════════════════

MONITOR_PATCHES: list[tuple[str, str, str]] = [

    # 1. __init__ データ系列から grad_norms 削除、ES 状態変数追加
    (
        "__init__ grad_norms 削除 / ES 変数追加",
        '''\
        # ── データ系列 ──────────────────────────────────────────────
        self._steps:      list[int]   = []
        self._train_loss: list[float] = []
        self._lr_vals:    list[float] = []
        self._grad_norms: list[float] = []

        # ── 状態変数 ──────────────────────────────────────────────
        self._global_step  = 0
        self._step_in_run  = 0
        self._total_steps  = 0
        self._last_lr      = 0.0
        self._last_train   = float("nan")
        self._last_grad    = float("nan")

        # 時間計測
        self._start_time:   float | None = None
        self._step_times:   list[float]  = []
        self._last_step_ts: float | None = None''',
        '''\
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
        self._es_rise_count  = 0     # 連続上昇カウント
        self._es_prev_loss   = float("nan")   # 前ステップの loss
        self._es_warned      = False  # 50% 警告済みフラグ
        self._es_stopped     = False  # 緊急停止済みフラグ

        # 時間計測
        self._start_time:   float | None = None
        self._step_times:   list[float]  = []
        self._last_step_ts: float | None = None''',
    ),

    # 2. _build_param_panel の rows から grad_norm 削除
    (
        "_build_param_panel grad_norm 削除",
        '''\
        rows = [
            ("step",       "step"),
            ("LR",         "lr"),
            ("Train Loss", "train_loss"),
            ("grad norm",  "grad_norm"),
        ]''',
        '''\
        rows = [
            ("step",       "step"),
            ("LR",         "lr"),
            ("Train Loss", "train_loss"),
        ]''',
    ),

    # 3. _build_param_panel 後に EarlyStopping パネルを追加
    (
        "_build_param_panel ES パネル追加",
        '''\
        lf.columnconfigure(1, weight=1)

        # 時間情報
        time_lf = ttk.LabelFrame(parent, text="時間")''',
        '''\
        lf.columnconfigure(1, weight=1)

        # EarlyStopping パネル
        es_lf = ttk.LabelFrame(parent, text="EarlyStopping")
        es_lf.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._es_status_var = tk.StringVar(value="無効")
        ttk.Label(es_lf, textvariable=self._es_status_var,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W, padx=6, pady=(2, 0))
        self._es_progress = ttk.Progressbar(es_lf, maximum=100, value=0, length=180)
        self._es_progress.pack(fill=tk.X, padx=6, pady=(2, 4))

        # 時間情報
        time_lf = ttk.LabelFrame(parent, text="時間")''',
    ),

    # 4. _build_report_panel タグに es_warn / es_stop 追加
    (
        "_build_report_panel タグ追加",
        '''\
        self._report_text.tag_configure("normal",  foreground=COLOR_NORMAL)
        self._report_text.tag_configure("caution", foreground=COLOR_CAUTION)
        self._report_text.tag_configure("warning", foreground=COLOR_WARNING)
        self._report_text.tag_configure("danger",  foreground=COLOR_DANGER)
        self._report_text.tag_configure("info",    foreground=COLOR_INFO)
        self._report_text.tag_configure("diag",    foreground="#7DD3FC")
        self._report_text.tag_configure("sample_info", foreground="#A78BFA")''',
        '''\
        self._report_text.tag_configure("normal",      foreground=COLOR_NORMAL)
        self._report_text.tag_configure("caution",     foreground=COLOR_CAUTION)
        self._report_text.tag_configure("warning",     foreground=COLOR_WARNING)
        self._report_text.tag_configure("danger",      foreground=COLOR_DANGER)
        self._report_text.tag_configure("info",        foreground=COLOR_INFO)
        self._report_text.tag_configure("diag",        foreground="#7DD3FC")
        self._report_text.tag_configure("sample_info", foreground="#A78BFA")
        self._report_text.tag_configure("es_warn",     foreground=COLOR_WARNING)
        self._report_text.tag_configure("es_stop",     foreground=COLOR_DANGER)''',
    ),

    # 5. _init_matplotlib: 3サブプロット→2サブプロット（grad_norm 削除）
    (
        "_init_matplotlib 3→2サブプロット",
        '''\
            # サブプロット: Loss(上大) / LR(中) / grad_norm(下)
            self._fig = Figure(figsize=(7, 8))
            gs = self._fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1.5])
            self._fig.subplots_adjust(
                left=0.10, right=0.95, top=0.95, bottom=0.07, hspace=0.50
            )

            self._ax_loss = self._fig.add_subplot(gs[0])
            self._ax_lr   = self._fig.add_subplot(gs[1])
            self._ax_grad = self._fig.add_subplot(gs[2])

            self._ax_loss.set_title("Train Loss", fontsize=10)
            self._ax_lr.set_title("Learning Rate", fontsize=9)
            self._ax_grad.set_title("grad norm", fontsize=9)

            for ax in (self._ax_loss, self._ax_lr, self._ax_grad):
                ax.grid(True)
                ax.set_xlabel("step", fontsize=8)''',
        '''\
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
                ax.set_xlabel("step", fontsize=8)''',
    ),

    # 6. _parse_line: grad_norm 削除・サンプル生成検出強化・ES チェック呼び出し追加
    (
        "_parse_line 全体差し替え",
        '''\
    def _parse_line(self, line: str) -> None:
        # step 進捗 (tqdm形式)
        m = _RE_STEP_POSTFIX.search(line)
        if m:
            self._step_in_run  = int(m.group(1))
            self._total_steps  = int(m.group(2))

        # Train Loss
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

        # サンプル生成ログ
        if "Generating sample images at step" in line:
            self._append_report(f"[サンプル生成] {line.strip()}", "sample_info")
        elif "  prompt:" in line and ", size:" in line:
            self._append_report(f"[サンプル生成] {line.strip()}", "sample_info")

        # 診断
        self._auto_diagnose(line)''',
        '''\
    def _parse_line(self, line: str) -> None:
        # ── step 進捗 (tqdm: "N/total [") ────────────────────────────
        m = _RE_STEP_POSTFIX.search(line)
        if m:
            self._step_in_run = int(m.group(1))
            self._total_steps = int(m.group(2))

        # ── Train Loss (tqdm postfix: ", loss=N") ─────────────────────
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
                step_info = f" (step {ms.group(1)})"
            self._append_report(f"[サンプル生成開始]{step_info}", "sample_info")
        elif "sample images saved" in line.lower() or "sample saved" in line.lower():
            self._append_report(f"[サンプル生成完了] {line.strip()}", "sample_info")
        elif "  prompt:" in line and ", size:" in line:
            self._append_report(f"[サンプル生成] {line.strip()}", "sample_info")

        # ── 診断 ─────────────────────────────────────────────────────
        self._auto_diagnose(line)''',
    ),

    # 7. _auto_diagnose: grad_norm 急増削除・参照を _RE_LOSS_TQDM に修正
    (
        "_auto_diagnose grad_norm 削除 / 参照修正",
        '''\
    def _auto_diagnose(self, line: str) -> None:
        # grad_norm 急増
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

        # Train Loss NaN/Inf
        m = _RE_AVR_LOSS.search(line)
        if m:
            try:
                v = float(m.group(1))
                if math.isnan(v) or math.isinf(v):
                    self._append_report(
                        "[診断] Train Loss が NaN/Inf になっています。"
                        "学習を停止することを推奨します。",
                        "danger",
                    )
            except ValueError:
                pass''',
        '''\
    def _auto_diagnose(self, line: str) -> None:
        # Train Loss NaN/Inf
        m = _RE_LOSS_TQDM.search(line)
        if m:
            try:
                v = float(m.group(1))
                if math.isnan(v) or math.isinf(v):
                    self._append_report(
                        "[診断] Train Loss が NaN/Inf になっています。"
                        "学習を停止することを推奨します。",
                        "danger",
                    )
            except ValueError:
                pass''',
    ),

    # 8. _check_es メソッドを _auto_diagnose の直後に追加
    (
        "_check_es メソッド追加",
        '''\
    # ─────────────────────────────────────────────────────────────────
    # レポート書き込み
    # ─────────────────────────────────────────────────────────────────
    def _append_report(self, msg: str, tag: str) -> None:''',
        '''\
    def _check_es(self, current_loss: float) -> None:
        """Train Loss の連続上昇を監視して警告/緊急停止を行う。"""
        # ES が無効または設定なしの場合はパネルを更新して戻る
        try:
            enabled  = bool(self._state.es_enabled.get())
            patience = int(self._state.es_patience.get())
        except Exception:
            self._es_status_var.set("無効")
            self._es_progress["value"] = 0
            return

        if not enabled or patience <= 0:
            self._es_status_var.set("無効")
            self._es_progress["value"] = 0
            return

        # 前ステップと比較して上昇カウント
        if not math.isnan(self._es_prev_loss):
            if current_loss > self._es_prev_loss:
                self._es_rise_count += 1
            else:
                # loss が改善 → カウントリセット、フラグ解除
                if self._es_rise_count > 0:
                    self._append_report(
                        f"[ES] Loss 改善を確認。カウントリセット "
                        f"({self._es_rise_count} → 0)",
                        "normal",
                    )
                self._es_rise_count = 0
                self._es_warned     = False
        self._es_prev_loss = current_loss

        pct = int(self._es_rise_count / patience * 100)
        self._es_progress["value"] = min(pct, 100)
        self._es_status_var.set(
            f"連続上昇 {self._es_rise_count} / {patience} step"
        )

        # 50% 警告
        warn_threshold = max(1, patience // 2)
        if self._es_rise_count >= warn_threshold and not self._es_warned:
            self._es_warned = True
            self._append_report(
                f"[ES 警告] Train Loss が {warn_threshold} step 連続上昇しています "
                f"({self._es_rise_count}/{patience})。過学習の可能性があります。",
                "es_warn",
            )

        # 100% 緊急停止
        if self._es_rise_count >= patience and not self._es_stopped:
            self._es_stopped = True
            self._append_report(
                f"[ES 緊急停止] Train Loss が {patience} step 連続上昇しました。"
                "学習を停止します。",
                "es_stop",
            )
            self._stop_training()

    # ─────────────────────────────────────────────────────────────────
    # レポート書き込み
    # ─────────────────────────────────────────────────────────────────
    def _append_report(self, msg: str, tag: str) -> None:''',
    ),

    # 9. _append_report の valid_tags に es_warn / es_stop 追加
    (
        "_append_report valid_tags 追加",
        '''\
        valid_tags = {"normal", "caution", "warning", "danger", "info",
                      "diag", "sample_info"}''',
        '''\
        valid_tags = {"normal", "caution", "warning", "danger", "info",
                      "diag", "sample_info", "es_warn", "es_stop"}''',
    ),

    # 10. _update_params から grad_norm 削除
    (
        "_update_params grad_norm 削除",
        '''\
        self._param_vars["step"].set(st_str)
        self._param_vars["lr"].set(_fmt(self._last_lr, ".3e"))
        self._param_vars["train_loss"].set(_fmt(self._last_train, ".6f"))
        self._param_vars["grad_norm"].set(_fmt(self._last_grad, ".4f"))''',
        '''\
        self._param_vars["step"].set(st_str)
        self._param_vars["lr"].set(_fmt(self._last_lr, ".3e"))
        self._param_vars["train_loss"].set(_fmt(self._last_train, ".6f"))''',
    ),

    # 11. _update_time に tqdm eta 優先ブランチ追加
    (
        "_update_time tqdm eta 優先",
        '''\
        if self._step_times and self._total_steps > 0:
            avg_step_sec = sum(self._step_times) / len(self._step_times)
            remaining = max(0, self._total_steps - self._step_in_run)
            eta_sec = avg_step_sec * remaining
            eta_dt  = datetime.datetime.fromtimestamp(now + eta_sec)
            self._time_vars["eta_remain"].set(_fmt_duration(eta_sec))
            self._time_vars["eta_clock"].set(eta_dt.strftime("%H:%M:%S"))
        else:
            self._time_vars["eta_remain"].set("計算中...")
            self._time_vars["eta_clock"].set("—")''',
        '''\
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
            self._time_vars["eta_remain"].set("計算中...")
            self._time_vars["eta_clock"].set("—")''',
    ),

    # 12. _update_graph から grad_norm サブプロット削除
    (
        "_update_graph grad_norm 削除",
        '''\
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

            # grad_norm
            ax3 = self._ax_grad
            ax3.cla()
            if self._grad_norms:
                ax3.plot(
                    self._steps[:len(self._grad_norms)],
                    self._grad_norms,
                    color="#C084FC", linewidth=0.9,
                )
                if len(self._grad_norms) >= 5:
                    avg = sum(self._grad_norms) / len(self._grad_norms)
                    ax3.axhline(avg * 3, color="#EF4444", linewidth=0.7,
                                linestyle="--", alpha=0.6, label="×3 avg")
                    ax3.legend(fontsize=7)
            ax3.set_title("grad norm", fontsize=9, color="#CBD5E1")
            ax3.set_xlabel("step", fontsize=8)
            ax3.grid(True)

            self._canvas.draw_idle()''',
        '''\
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

            self._canvas.draw_idle()''',
    ),

    # 13. _reset から grad_norms / _last_grad / ES 状態 クリア
    (
        "_reset grad_norms / ES クリア",
        '''\
        self._steps.clear()
        self._train_loss.clear()
        self._lr_vals.clear()
        self._grad_norms.clear()
        self._global_step   = 0
        self._step_in_run   = 0
        self._total_steps   = 0
        self._last_lr       = 0.0
        self._last_train    = float("nan")
        self._last_grad     = float("nan")
        self._start_time    = None
        self._last_step_ts  = None
        self._step_times.clear()''',
        '''\
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
        self._step_times.clear()''',
    ),

    # 14. _reset の ax クリアから ax_grad 削除
    (
        "_reset ax_grad 削除",
        '''\
        if self._mpl_ok:
            for ax in (self._ax_loss, self._ax_lr, self._ax_grad):
                ax.cla()
                ax.grid(True)
            self._canvas.draw_idle()''',
        '''\
        if self._mpl_ok:
            for ax in (self._ax_loss, self._ax_lr):
                ax.cla()
                ax.grid(True)
            self._canvas.draw_idle()
        self._es_status_var.set("無効")
        self._es_progress["value"] = 0''',
    ),

    # 15. 正規表現ブロックを tqdm 形式に差し替え
    (
        "_RE ブロック tqdm 対応",
        '''\
_RE_AVR_LOSS     = re.compile(r"avr_loss=([0-9]+\\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_LR           = re.compile(r"\\blr=([0-9]+\\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_GRAD_NORM    = re.compile(r"grad_norm=([0-9]+\\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RE_STEP_POSTFIX = re.compile(r"(\\d+)/(\\d+)\\s*\\[")   # tqdm: "120/500 ["
_RE_STEP_LOG     = re.compile(r"step[s]?\\s*=?\\s*(\\d+)", re.IGNORECASE)''',
        '''\
# tqdm postfix: ", loss=0.0044"（LECO はこの形式で出力される）
_RE_LOSS_TQDM    = re.compile(r",\\s*loss=([0-9]+\\.[0-9]+(?:[eE][+-]?[0-9]+)?)")
# tqdm 時間フィールド: "[00:03<26:53,  3.23s/it"
_RE_TQDM_TIME    = re.compile(r"\\[(\\d+:\\d+)<((?:\\d+:)?\\d+:\\d+),\\s*([0-9.]+)s/it")
_RE_STEP_POSTFIX = re.compile(r"(\\d+)/(\\d+)\\s*\\[")   # tqdm: "120/500 ["''',
    ),

    # 16. __init__ に _tqdm_eta_sec 追加（時間計測ブロック末尾）
    (
        "__init__ _tqdm_eta_sec 追加",
        '''\
        # 時間計測
        self._start_time:   float | None = None
        self._step_times:   list[float]  = []
        self._last_step_ts: float | None = None''',
        '''\
        # 時間計測
        self._start_time:   float | None = None
        self._step_times:   list[float]  = []
        self._last_step_ts: float | None = None
        self._tqdm_eta_sec: float | None = None''',
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# エントリポイント
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(__file__).resolve().parent

    leco  = app_dir / "leco_train.py"
    mgleo = app_dir / "monitor_graph_leco.py"

    for p in (leco, mgleo):
        if not p.exists():
            print(f"[ERROR] ファイルが見つかりません: {p}")
            sys.exit(1)

    print(f"=== leco_train.py ({leco}) ===")
    _apply_patches(leco, LECO_PATCHES)

    print(f"=== monitor_graph_leco.py ({mgleo}) ===")
    _apply_patches(mgleo, MONITOR_PATCHES)

    print("\n[完了] 全パッチ適用済み")


if __name__ == "__main__":
    main()
