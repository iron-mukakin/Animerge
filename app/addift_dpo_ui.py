"""app/addift_dpo_ui.py — ADDifT学習タブ用 DPOモード拡張UI

addift_train.py から呼び出されるフック関数群をまとめたモジュール。
addift_train.py 本体への侵襲を最小化しつつ、以下の機能を提供する:

  - DPOモード有効化チェックボックス + モード選択プルダウン (DPO/SDPO/MaPO)
  - DPOモード専用パラメータ (preference_beta, win_aux_weight) の表示/非表示制御
  - データセットタブの画像A/B ラベルを win/lose 表記へ動的切替
  - accelerate launch コマンドへの --addift_mode / --preference_beta 付与
  - プリセット (JSON) への保存・復元

呼び出し元: addift_train.py
  - _AddifTTrainState.__init__()  -> attach_dpo_mode_vars(self)
  - _build_train_tab()            -> build_dpo_mode_controls(parent, s)
  - _build_dataset_tab()          -> attach_dataset_label_refresh(s, label_a, label_b)
  - _build_command()              -> append_dpo_command_args(s, cmd)
  - _validate()                   -> validate_dpo_mode(s)
  - プリセット _collect()/_apply() -> collect_dpo_mode_preset(s) / apply_dpo_mode_preset(s, data, set_var_fn)

未実装モードについて:
  SDPO / MaPO はプルダウンの選択肢としてのみ用意し、選択された場合は
  validate_dpo_mode() がエラー文字列を返すことで学習開始を阻止する。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from .i18n import gettext
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _app_dir = _Path(__file__).resolve().parent
    if str(_app_dir) not in _sys.path:
        _sys.path.insert(0, str(_app_dir))
    from i18n import gettext  # type: ignore[no-redef]

# ── サポートするモード ────────────────────────────────────────────────────
ADDIFT_MODE_DPO = "dpo"
ADDIFT_MODE_SDPO = "sdpo"
ADDIFT_MODE_MAPO = "mapo"
ADDIFT_MODES_UNIMPLEMENTED: frozenset[str] = frozenset()

_MODE_LABEL_KEYS: dict[str, str] = {
    ADDIFT_MODE_DPO:  "addift_mode_dpo",
    ADDIFT_MODE_SDPO: "addift_mode_sdpo",
    ADDIFT_MODE_MAPO: "addift_mode_mapo",
}

_DEFAULT_PREFERENCE_BETA = 5.0
_DEFAULT_WIN_AUX_WEIGHT = 0.1
_DEFAULT_SDPO_MU = 0.6  # SDPO安全マージン係数 (論文推奨レンジ: 0.2〜0.9)
_DEFAULT_MAPO_BETA = 5.0  # MaPO温度係数 (論文に推奨レンジ記載なし、経験的に調整)
_DEFAULT_MAPO_BG_WEIGHT = 0.0  # MaPO専用マスク外正則化重み(正解ノイズ基準)
_DEFAULT_ES_DPO_PATIENCE = 10
_DEFAULT_ES_DPO_WARMUP_RATIO = 0.1  # 全stepに対する判定スキップ割合(序盤の不安定期間を除外)


def attach_dpo_mode_vars(state: object) -> None:
    """DPOモード関連の tk 変数を学習タブ状態オブジェクトへ追加する。

    Args:
        state: addift_train._AddifTTrainState のインスタンス
            (循環import回避のため型ヒントは object とする)。

    Returns:
        None
    """
    state.addift_mode_enabled = tk.BooleanVar(value=False)
    state.addift_mode_name = tk.StringVar(value=ADDIFT_MODE_DPO)
    state.preference_beta = tk.DoubleVar(value=_DEFAULT_PREFERENCE_BETA)
    state.win_aux_weight_enabled = tk.BooleanVar(value=False)
    state.win_aux_weight = tk.DoubleVar(value=_DEFAULT_WIN_AUX_WEIGHT)
    state.sdpo_mu = tk.DoubleVar(value=_DEFAULT_SDPO_MU)
    state.mapo_beta = tk.DoubleVar(value=_DEFAULT_MAPO_BETA)
    state.mapo_bg_weight_enabled = tk.BooleanVar(value=False)
    state.mapo_bg_weight = tk.DoubleVar(value=_DEFAULT_MAPO_BG_WEIGHT)
    state.es_dpo_enabled = tk.BooleanVar(value=False)
    state.es_dpo_patience = tk.IntVar(value=_DEFAULT_ES_DPO_PATIENCE)
    state.es_dpo_warmup_ratio = tk.DoubleVar(value=_DEFAULT_ES_DPO_WARMUP_RATIO)
    state._dpo_dataset_label_widgets: dict[str, ttk.Label] = {}
    state._dpo_mode_widgets: list[tk.Widget] = []
    state._dpo_adv_widgets: list[tk.Widget] = []


def build_dpo_mode_controls(parent: ttk.Frame, state: object) -> ttk.LabelFrame:
    """DPOモード切替UI (チェックボックス + モード選択 + モード別パラメータ) を構築する。

    ADDifTパラメータ枠 (lf2) の直前に配置することで、要件である
    「ADDifTパラメータの先頭」相当の視覚的位置を確保する。
    選択中モード(DPO/SDPO/MaPO)で使用しないオプション群は、非活性化(disabled)
    ではなく grid_remove() により画面から完全に取り除く。表示対象のセクションは
    行1から詰めて再配置するため、余白(空行)は生じない。

    Args:
        parent: 学習設定タブのコンテナ。
        state: _AddifTTrainState インスタンス。

    Returns:
        ttk.LabelFrame: 構築した枠。
    """
    frame = ttk.LabelFrame(parent, text=gettext("addift_dpo_mode_label"))
    frame.pack(fill=tk.X, pady=(0, 8))
    frame.columnconfigure(3, weight=1)

    mode_label_by_key = {key: gettext(label_key) for key, label_key in _MODE_LABEL_KEYS.items()}
    mode_key_by_label = {v: k for k, v in mode_label_by_key.items()}
    mode_display = tk.StringVar(value=mode_label_by_key[state.addift_mode_name.get()])

    beta_label = ttk.Label(frame, text=gettext("addift_preference_beta_label"), width=22, anchor=tk.W)
    beta_entry = ttk.Entry(frame, textvariable=state.preference_beta, width=10)
    beta_note = ttk.Label(frame, text=gettext("addift_preference_beta_note"), foreground="#64748B")
    mode_combobox = ttk.Combobox(
        frame, textvariable=mode_display, values=list(mode_label_by_key.values()),
        state="readonly", width=20,
    )
    win_aux_check = ttk.Checkbutton(frame, text=gettext("addift_win_aux_weight_enable"),
                                     variable=state.win_aux_weight_enabled)
    win_aux_entry = ttk.Entry(frame, textvariable=state.win_aux_weight, width=10)
    win_aux_note = ttk.Label(frame, text=gettext("addift_win_aux_weight_note"), foreground="#64748B")
    mu_label = ttk.Label(frame, text=gettext("addift_sdpo_mu_label"), width=22, anchor=tk.W)
    mu_entry = ttk.Entry(frame, textvariable=state.sdpo_mu, width=10)
    mu_note = ttk.Label(frame, text=gettext("addift_sdpo_mu_note"), foreground="#64748B")
    mapo_beta_label = ttk.Label(frame, text=gettext("addift_mapo_beta_label"), width=22, anchor=tk.W)
    mapo_beta_entry = ttk.Entry(frame, textvariable=state.mapo_beta, width=10)
    mapo_beta_note = ttk.Label(frame, text=gettext("addift_mapo_beta_note"), foreground="#64748B")
    mapo_bg_check = ttk.Checkbutton(frame, text=gettext("addift_mapo_bg_weight_enable"),
                                    variable=state.mapo_bg_weight_enabled)
    mapo_bg_entry = ttk.Entry(frame, textvariable=state.mapo_bg_weight, width=10)
    mapo_bg_note = ttk.Label(frame, text=gettext("addift_mapo_bg_weight_note"), foreground="#64748B")

    beta_section: list[list[tuple[tk.Widget, dict]]] = [
        [
            (beta_label, {"column": 0, "sticky": tk.W, "padx": (4, 2), "pady": 3}),
            (beta_entry, {"column": 1, "sticky": tk.W, "padx": (0, 12), "pady": 3}),
        ],
        [(beta_note, {"column": 0, "columnspan": 4, "sticky": tk.W, "padx": 4, "pady": (0, 4)})],
    ]
    win_aux_section: list[list[tuple[tk.Widget, dict]]] = [
        [
            (win_aux_check, {"column": 0, "columnspan": 2, "sticky": tk.W, "padx": 4, "pady": 3}),
            (win_aux_entry, {"column": 2, "sticky": tk.W, "padx": (0, 4), "pady": 3}),
        ],
        [(win_aux_note, {"column": 0, "columnspan": 4, "sticky": tk.W, "padx": 4, "pady": (0, 4)})],
    ]
    mu_section: list[list[tuple[tk.Widget, dict]]] = [
        [
            (mu_label, {"column": 0, "sticky": tk.W, "padx": (4, 2), "pady": 3}),
            (mu_entry, {"column": 1, "sticky": tk.W, "padx": (0, 12), "pady": 3}),
        ],
        [(mu_note, {"column": 0, "columnspan": 4, "sticky": tk.W, "padx": 4, "pady": (0, 4)})],
    ]
    mapo_beta_section: list[list[tuple[tk.Widget, dict]]] = [
        [
            (mapo_beta_label, {"column": 0, "sticky": tk.W, "padx": (4, 2), "pady": 3}),
            (mapo_beta_entry, {"column": 1, "sticky": tk.W, "padx": (0, 12), "pady": 3}),
        ],
        [(mapo_beta_note, {"column": 0, "columnspan": 4, "sticky": tk.W, "padx": 4, "pady": (0, 4)})],
    ]
    mapo_bg_section: list[list[tuple[tk.Widget, dict]]] = [
        [
            (mapo_bg_check, {"column": 0, "columnspan": 2, "sticky": tk.W, "padx": 4, "pady": 3}),
            (mapo_bg_entry, {"column": 2, "sticky": tk.W, "padx": (0, 4), "pady": 3}),
        ],
        [(mapo_bg_note, {"column": 0, "columnspan": 4, "sticky": tk.W, "padx": 4, "pady": (0, 4)})],
    ]
    all_mode_widgets = (
        beta_label, beta_entry, beta_note,
        win_aux_check, win_aux_entry, win_aux_note,
        mu_label, mu_entry, mu_note,
        mapo_beta_label, mapo_beta_entry, mapo_beta_note,
        mapo_bg_check, mapo_bg_entry, mapo_bg_note,
    )

    def _apply_visible_sections(sections: list[list[tuple[tk.Widget, dict]]]) -> None:
        """指定セクションのみを行1から詰めて表示し、それ以外は grid_remove() で非表示にする。

        Args:
            sections: 表示する行のリスト。各行は同一tkinter行に配置する
                (ウィジェット, gridキーワード引数) の組の集合。

        Returns:
            None
        """
        visible_widgets = {widget for row in sections for widget, _ in row}
        for widget in all_mode_widgets:
            if widget not in visible_widgets:
                widget.grid_remove()
        for row_index, row in enumerate(sections, start=1):
            for widget, grid_kwargs in row:
                widget.grid(row=row_index, **grid_kwargs)

    def _render_dpo_mode_widgets() -> None:
        """現在の state.addift_mode_enabled / state.addift_mode_name に基づき、
        コンボボックスの活性状態・モード別オプションの表示/非表示・データセット
        ラベル・EarlyStoppingDPOパネルの可視性をまとめて描画する。

        表示(mode_display)とstate(addift_mode_name)の同期はこの関数の責務外。
        呼び出し元(_reflect_dpo_mode_state / _sync_mode_controls_from_state)が
        同期方向をそれぞれ確定させてから呼び出すこと。

        Returns:
            None
        """
        enabled = state.addift_mode_enabled.get()
        mode_combobox.configure(state="readonly" if enabled else tk.DISABLED)

        is_dpo_active = enabled and state.addift_mode_name.get() == ADDIFT_MODE_DPO
        is_sdpo_active = enabled and state.addift_mode_name.get() == ADDIFT_MODE_SDPO
        is_mapo_active = enabled and state.addift_mode_name.get() == ADDIFT_MODE_MAPO

        win_aux_entry.configure(state=tk.NORMAL if state.win_aux_weight_enabled.get() else tk.DISABLED)
        mapo_bg_entry.configure(state=tk.NORMAL if state.mapo_bg_weight_enabled.get() else tk.DISABLED)

        if is_dpo_active:
            _apply_visible_sections(beta_section + win_aux_section)
        elif is_sdpo_active:
            _apply_visible_sections(beta_section + win_aux_section + mu_section)
        elif is_mapo_active:
            _apply_visible_sections(mapo_beta_section + mapo_bg_section)
        else:
            _apply_visible_sections([])

        _refresh_dataset_labels(state)
        getattr(state, "_dpo_adv_visibility_cb", lambda: None)()

    def _reflect_dpo_mode_state(_event: object = None) -> None:
        """ウィジェット操作(コンボボックス選択・チェック操作)を受けて、
        表示中の値をstateへ反映してから描画する (表示→state方向)。
        """
        selected_key = mode_key_by_label.get(mode_display.get())
        if selected_key is not None:
            state.addift_mode_name.set(selected_key)
        _render_dpo_mode_widgets()

    def _sync_mode_controls_from_state() -> None:
        """プリセット読込など、外部からstateが変更された場合に表示側を同期する
        (state→表示方向)。_reflect_dpo_mode_state()とは同期方向が逆である点に注意。
        """
        current_key = state.addift_mode_name.get()
        mode_display.set(mode_label_by_key.get(current_key, mode_label_by_key[ADDIFT_MODE_DPO]))
        _render_dpo_mode_widgets()

    state._dpo_mode_sync_cb = _sync_mode_controls_from_state

    win_aux_check.configure(command=_reflect_dpo_mode_state)
    mapo_bg_check.configure(command=_reflect_dpo_mode_state)

    ttk.Checkbutton(
        frame, text=gettext("addift_dpo_mode_enable"),
        variable=state.addift_mode_enabled, command=_reflect_dpo_mode_state,
    ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    ttk.Label(frame, text=gettext("addift_mode_select_label"), anchor=tk.E).grid(
        row=0, column=2, sticky=tk.E, padx=(0, 2), pady=3)
    mode_combobox.grid(row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)
    mode_combobox.bind("<<ComboboxSelected>>", _reflect_dpo_mode_state)

    state._dpo_mode_widgets = [
        mode_combobox, beta_entry, win_aux_check, win_aux_entry, mu_entry,
        mapo_beta_entry, mapo_bg_check, mapo_bg_entry,
    ]
    _reflect_dpo_mode_state()
    return frame


def build_es_dpo_controls(parent: ttk.Frame, state: object) -> ttk.LabelFrame:
    """EarlyStoppingDPO設定枠 (詳細タブ・既存EarlyStopping下段) を構築する。

    DPOモード有効時のみ表示される。

    Args:
        parent: 詳細タブのコンテナ。
        state: _AddifTTrainState インスタンス。

    Returns:
        ttk.LabelFrame: 構築した枠 (pack/pack_forgetで表示制御される)。
    """
    frame = ttk.LabelFrame(parent, text=gettext("addift_es_dpo_label"))

    ttk.Checkbutton(frame, text=gettext("addift_es_dpo_enable"),
                     variable=state.es_dpo_enabled).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=4)
    ttk.Label(frame, text=gettext("leco_es_watch_steps"), anchor=tk.W).grid(
        row=0, column=1, sticky=tk.W, padx=(12, 2), pady=4)
    ttk.Spinbox(frame, from_=2, to=500, textvariable=state.es_dpo_patience, width=6).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 8), pady=4)
    ttk.Label(frame, text=gettext("addift_es_dpo_note"), foreground="#64748B").grid(
        row=0, column=3, sticky=tk.W, padx=(0, 8), pady=4)

    ttk.Label(frame, text=gettext("addift_es_dpo_warmup_label"), anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=(0, 4))
    ttk.Spinbox(
        frame, from_=0.0, to=0.9, increment=0.05, textvariable=state.es_dpo_warmup_ratio,
        width=6, format="%.2f",
    ).grid(row=1, column=1, sticky=tk.W, padx=(0, 2), pady=(0, 4))
    ttk.Label(frame, text=gettext("addift_es_dpo_warmup_note"), foreground="#64748B").grid(
        row=1, column=2, columnspan=2, sticky=tk.W, padx=(0, 8), pady=(0, 4))

    def _refresh_visibility() -> None:
        is_dpo_family_active = (
            state.addift_mode_enabled.get()
            and state.addift_mode_name.get() in (ADDIFT_MODE_DPO, ADDIFT_MODE_SDPO, ADDIFT_MODE_MAPO)
        )
        if is_dpo_family_active:
            frame.pack(fill=tk.X, pady=(8, 0))
        else:
            frame.pack_forget()

    state._dpo_adv_visibility_cb = _refresh_visibility
    _refresh_visibility()
    return frame


def attach_dataset_label_refresh(
    state: object, label_image_a: ttk.Label, label_image_b: ttk.Label,
) -> None:
    """データセットタブの画像A/Bラベルを DPOモードに応じて win/lose 表記へ切替可能にする。

    Args:
        state: _AddifTTrainState インスタンス。
        label_image_a: 画像A(変換前)用のラベルウィジェット。
        label_image_b: 画像B(変換後)用のラベルウィジェット。

    Returns:
        None
    """
    state._dpo_dataset_label_widgets = {"a": label_image_a, "b": label_image_b}
    _refresh_dataset_labels(state)


def _refresh_dataset_labels(state: object) -> None:
    """現在のモードに応じてデータセットタブのラベル文言を更新する内部関数。"""
    widgets: dict[str, ttk.Label] = getattr(state, "_dpo_dataset_label_widgets", {})
    if not widgets:
        return

    is_dpo_family_active = (
        state.addift_mode_enabled.get()
        and state.addift_mode_name.get() in (ADDIFT_MODE_DPO, ADDIFT_MODE_SDPO, ADDIFT_MODE_MAPO)
    )
    if is_dpo_family_active:
        widgets["a"].configure(text=gettext("addift_image_lose_label"))
        widgets["b"].configure(text=gettext("addift_image_win_label"))
    else:
        widgets["a"].configure(text=gettext("addift_image_a_label"))
        widgets["b"].configure(text=gettext("addift_image_b_label"))


def validate_dpo_mode(state: object) -> str | None:
    """DPOモード関連のバリデーション。問題があればエラーメッセージを返す。

    Args:
        state: _AddifTTrainState インスタンス。

    Returns:
        str | None: エラーメッセージ。問題なければ None。
    """
    if not state.addift_mode_enabled.get():
        return None
    if state.addift_mode_name.get() in ADDIFT_MODES_UNIMPLEMENTED:
        return gettext("addift_validate_mapo_unimplemented")
    if state.addift_mode_name.get() == ADDIFT_MODE_SDPO and not (0.0 <= state.sdpo_mu.get() <= 1.0):
        return gettext("addift_validate_sdpo_mu_range")
    if state.addift_mode_name.get() == ADDIFT_MODE_MAPO and state.mapo_beta.get() <= 0.0:
        return gettext("addift_validate_mapo_beta_range")
    if state.mapo_bg_weight.get() < 0.0:
        return gettext("addift_validate_mapo_bg_weight_range")
    return None


def append_dpo_command_args(state: object, cmd: list[str]) -> None:
    """accelerate launch コマンド配列へ --addift_mode / --preference_beta / --win_aux_weight を追加する(破壊的)。

    Args:
        state: _AddifTTrainState インスタンス。
        cmd: _build_command() で組み立て中のコマンド配列。

    Returns:
        None
    """
    mode_name = state.addift_mode_name.get()
    is_dpo_active = state.addift_mode_enabled.get() and mode_name == ADDIFT_MODE_DPO
    is_sdpo_active = state.addift_mode_enabled.get() and mode_name == ADDIFT_MODE_SDPO
    is_mapo_active = state.addift_mode_enabled.get() and mode_name == ADDIFT_MODE_MAPO

    if is_dpo_active or is_sdpo_active:
        cmd += ["--addift_mode", mode_name, "--preference_beta", str(state.preference_beta.get())]
        if state.win_aux_weight_enabled.get():
            cmd += ["--win_aux_weight", str(state.win_aux_weight.get())]
        if is_sdpo_active:
            cmd += ["--sdpo_mu", str(state.sdpo_mu.get())]
    elif is_mapo_active:
        # MaPOはpreference_beta/win_aux_weightを使用しない(reference非依存のため)。
        cmd += ["--addift_mode", mode_name, "--mapo_beta", str(state.mapo_beta.get())]
        if state.mapo_bg_weight_enabled.get():
            cmd += ["--mapo_bg_weight", str(state.mapo_bg_weight.get())]
    else:
        cmd += ["--addift_mode", "none"]


def collect_dpo_mode_preset(state: object) -> dict:
    """プリセット保存用に DPOモード設定を dict 化する。"""
    return {
        "addift_mode_enabled":     bool(state.addift_mode_enabled.get()),
        "addift_mode_name":        state.addift_mode_name.get(),
        "preference_beta":         float(state.preference_beta.get()),
        "win_aux_weight_enabled":  bool(state.win_aux_weight_enabled.get()),
        "win_aux_weight":          float(state.win_aux_weight.get()),
        "sdpo_mu":                 float(state.sdpo_mu.get()),
        "mapo_beta":               float(state.mapo_beta.get()),
        "mapo_bg_weight_enabled":  bool(state.mapo_bg_weight_enabled.get()),
        "mapo_bg_weight":          float(state.mapo_bg_weight.get()),
        "es_dpo_enabled":          bool(state.es_dpo_enabled.get()),
        "es_dpo_patience":         int(state.es_dpo_patience.get()),
        "es_dpo_warmup_ratio":     float(state.es_dpo_warmup_ratio.get()),
    }


def apply_dpo_mode_preset(state: object, data: dict) -> None:
    """プリセットJSONから DPOモード設定を復元する。

    Args:
        state: _AddifTTrainState インスタンス。
        data: プリセットJSONをロードした dict。

    Returns:
        None
    """
    def _restore(var: tk.Variable, key: str, default: object) -> None:
        if key not in data:
            return
        try:
            var.set(data[key])
        except (tk.TclError, ValueError):
            var.set(default)

    _restore(state.addift_mode_enabled,    "addift_mode_enabled",    False)
    _restore(state.addift_mode_name,       "addift_mode_name",       ADDIFT_MODE_DPO)
    _restore(state.preference_beta,        "preference_beta",        _DEFAULT_PREFERENCE_BETA)
    _restore(state.win_aux_weight_enabled, "win_aux_weight_enabled", False)
    _restore(state.win_aux_weight,         "win_aux_weight",         _DEFAULT_WIN_AUX_WEIGHT)
    _restore(state.sdpo_mu,                "sdpo_mu",                _DEFAULT_SDPO_MU)
    _restore(state.mapo_beta,              "mapo_beta",              _DEFAULT_MAPO_BETA)
    _restore(state.mapo_bg_weight_enabled, "mapo_bg_weight_enabled", False)
    _restore(state.mapo_bg_weight,         "mapo_bg_weight",         _DEFAULT_MAPO_BG_WEIGHT)
    _restore(state.es_dpo_enabled,         "es_dpo_enabled",         False)
    _restore(state.es_dpo_patience,        "es_dpo_patience",        _DEFAULT_ES_DPO_PATIENCE)
    _restore(state.es_dpo_warmup_ratio,    "es_dpo_warmup_ratio",    _DEFAULT_ES_DPO_WARMUP_RATIO)
    # コンボボックス表示・モード別オプションの表示/非表示をstateへ同期する。
    # (_refresh_dataset_labels/_dpo_adv_visibility_cbは_sync_mode_controls_from_state
    #  内の_render_dpo_mode_widgets()が呼び出すため、個別の呼び出しは不要になった)
    getattr(state, "_dpo_mode_sync_cb", lambda: None)()
