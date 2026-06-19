#!/usr/bin/env python3
"""ADDifT学習機能フェーズ2 §4「timesteps プリセット選択方式」適用パッチ。

引継ぎ仕様書（handover_session12.md）セクション4に対応する変更を
以下の2ファイルへ適用する。

対象ファイル（このスクリプトの実行時カレントディレクトリ、または
--root で指定したディレクトリを起点に相対パスで解決する）:
    app/addift_train.py
    app/texts_ja.json

実行方法:
    python apply_fix_addift_timesteps_preset.py
    python apply_fix_addift_timesteps_preset.py --root /path/to/project

特記事項:
    - Python標準ライブラリのみで動作する（`patch` コマンド・外部ライブラリ
      いずれにも依存しない）。
    - パッチの差分文字列は LF で記述しているが、対象ファイルが CRLF / LF
      いずれの改行コードであっても _adapt() が自動的に揃えるため、
      実行環境やファイルの改行コードを意識する必要はない。
    - 冪等（idempotent）に設計されている。既に適用済みのパッチは自動検出して
      スキップするため、再実行しても安全。
    - 適用前に `<元ファイル名>.bak` を同階層へ作成する（既に存在する場合は
      初回適用時のバックアップを保持するため上書きしない）。
    - 1件でも old/new のどちらにも一致しないパッチがあれば、ファイルへの
      書き込みは行わず処理を中断する（対象ファイルが引継ぎ書時点の前提と
      異なる状態に変更されている可能性があるため、安全側に倒す）。
"""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _TextPatch:
    """単一ファイルに対する1件の文字列置換パッチ。

    Attributes:
        description: ログ表示用の説明文。
        old: 置換対象の文字列（LF改行で記述する）。
        new: 置換後の文字列（LF改行で記述する）。
    """

    description: str
    old: str
    new: str


def _detect_newline(raw_bytes: bytes) -> str:
    """ファイル内で優勢な改行コードを判定する。

    Args:
        raw_bytes: 対象ファイルの生バイト列。

    Returns:
        "\\r\\n"（CRLFを含む場合）または "\\n"（それ以外）。
    """
    return "\r\n" if b"\r\n" in raw_bytes else "\n"


def _adapt(text: str, newline: str) -> str:
    """LFで記述された差分文字列を、対象ファイルの改行コードへ変換する。

    Args:
        text: LF（"\\n"）で改行されたパッチ文字列。
        newline: 変換先の改行コード（"\\r\\n" または "\\n"）。

    Returns:
        newline へ変換した文字列。
    """
    normalized = text.replace("\r\n", "\n")
    if newline == "\n":
        return normalized
    return normalized.replace("\n", newline)


def _apply_text_patches(target: Path, patches: list[_TextPatch]) -> None:
    """target に対し patches を順に適用し、結果をファイルへ書き戻す。

    Args:
        target: 修正対象ファイルのパス。
        patches: 適用するパッチの並び（適用順を保つ）。

    Returns:
        None

    Raises:
        FileNotFoundError: target が存在しない場合。
        ValueError: old/new のいずれにも一致しないパッチが見つかった場合。
    """
    if not target.exists():
        raise FileNotFoundError(f"対象ファイルが見つかりません: {target}")

    raw = target.read_bytes()
    newline = _detect_newline(raw)
    text = raw.decode("utf-8")

    applied_count = 0
    skipped_count = 0
    for patch in patches:
        adapted_old = _adapt(patch.old, newline)
        adapted_new = _adapt(patch.new, newline)

        if adapted_new in text:
            print(f"  [skip] 既に適用済み: {patch.description}")
            skipped_count += 1
            continue

        match_count = text.count(adapted_old)
        if match_count != 1:
            raise ValueError(
                f"パッチ適用に失敗しました（一致数={match_count}、期待値=1）: "
                f"{patch.description}\n"
                f"  対象ファイル: {target}\n"
                f"  ファイルの内容が引継ぎ書時点の前提（フェーズ1完成版）と"
                f"異なっている可能性があります。処理を中断します。"
            )

        text = text.replace(adapted_old, adapted_new, 1)
        print(f"  [ok]   適用: {patch.description}")
        applied_count += 1

    if applied_count == 0:
        print(f"  -> 変更なし（{skipped_count}件すべて適用済み）: {target}")
        return

    backup_path = target.with_name(target.name + ".bak")
    if not backup_path.exists():
        backup_path.write_bytes(raw)
        print(f"  -> バックアップ作成: {backup_path}")

    target.write_bytes(text.encode("utf-8"))
    print(f"  -> 書き込み完了（適用 {applied_count}件 / スキップ {skipped_count}件）: {target}")


def _addift_train_patches() -> list[_TextPatch]:
    """app/addift_train.py に対するパッチ一覧を返す。

    Returns:
        引継ぎ書 §4 に対応する7件のパッチ（適用順）。
    """
    return [
        _TextPatch(
            description="TIMESTEPS_PRESETS 定数の追加",
            old='''LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"]

# 階層学習定数（lora_train.py / leco_train.py と共通。フェーズ2で使用予定）''',
            new='''LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"]

# timesteps プリセット（フェーズ2 §4: train_min/max_timesteps + network_strength を一括設定）
TIMESTEPS_PRESETS: dict[str, dict] = {
    "local":       {"min": 100, "max": 300,  "strength": 5.0, "label_key": "addift_timesteps_preset_local"},
    "style":       {"min": 200, "max": 400,  "strength": 5.0, "label_key": "addift_timesteps_preset_style"},
    "composition": {"min": 500, "max": 1000, "strength": 1.0, "label_key": "addift_timesteps_preset_composition"},
}

# 階層学習定数（lora_train.py / leco_train.py と共通。フェーズ2で使用予定）''',
        ),
        _TextPatch(
            description="_AddifTTrainState への timesteps_preset_* 変数追加",
            old='''        self.train_loss_function = tk.StringVar(value="MSE")
        self.train_snr_gamma  = tk.DoubleVar(value=0.0)

        # ── 詳細（Anima固有） ────────────────────────────────────''',
            new='''        self.train_loss_function = tk.StringVar(value="MSE")
        self.train_snr_gamma  = tk.DoubleVar(value=0.0)

        # ── timesteps プリセット選択（フェーズ2 §4） ────────────────
        self.timesteps_preset_enabled = tk.BooleanVar(value=False)
        self.timesteps_preset_name    = tk.StringVar(value="local")
        self.timesteps_preset_display: "tk.StringVar | None" = None
        self.timesteps_preset_widgets: list[tk.Widget] = []

        # ── 詳細（Anima固有） ────────────────────────────────────''',
        ),
        _TextPatch(
            description="_build_train_tab: プリセット選択UI追加 + 既存行のシフト",
            old='''    # ── ADDifT固有パラメータ ─────────────────────────────────────────
    lf2 = ttk.LabelFrame(parent, text=gettext("addift_params_label"))
    lf2.pack(fill=tk.X, pady=(0, 8))
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    ttk.Label(lf2, text="train_min_timesteps", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf2, from_=0, to=999, textvariable=s.train_min_timesteps, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_max_timesteps", width=22, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=1000, textvariable=s.train_max_timesteps, width=8).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf2, text="network_strength", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.network_strength, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="diff_alt_ratio", width=22, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.diff_alt_ratio, width=10).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf2, text="train_loss_function", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.train_loss_function, values=LOSS_FUNCTIONS,
                 state="readonly", width=12).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_snr_gamma", width=22, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.train_snr_gamma, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Checkbutton(lf2, text="train_fixed_timesteps_in_batch",
                    variable=s.train_fixed_timesteps_in_batch).grid(
        row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)''',
            new='''    # ── ADDifT固有パラメータ ─────────────────────────────────────────
    lf2 = ttk.LabelFrame(parent, text=gettext("addift_params_label"))
    lf2.pack(fill=tk.X, pady=(0, 8))
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    # row 0: timesteps プリセット選択（フェーズ2 §4）
    preset_label_by_key_addift = {
        key: gettext(spec["label_key"]) for key, spec in TIMESTEPS_PRESETS.items()
    }
    preset_key_by_label_addift = {v: k for k, v in preset_label_by_key_addift.items()}
    s.timesteps_preset_display = tk.StringVar(
        value=preset_label_by_key_addift.get(s.timesteps_preset_name.get(), "")
    )

    def _reflect_timesteps_preset_combobox_addift(_event: object = None) -> None:
        selected_key = preset_key_by_label_addift.get(s.timesteps_preset_display.get())
        if selected_key is not None:
            s.timesteps_preset_name.set(selected_key)
        _reflect_timesteps_preset_state_addift(s)

    ttk.Checkbutton(
        lf2, text=gettext("addift_timesteps_preset_enable"),
        variable=s.timesteps_preset_enabled,
        command=lambda: _reflect_timesteps_preset_state_addift(s),
    ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    ttk.Label(lf2, text=gettext("addift_timesteps_preset_label"), anchor=tk.E).grid(
        row=0, column=2, sticky=tk.E, padx=(0, 2), pady=3)

    timesteps_preset_combobox = ttk.Combobox(
        lf2, textvariable=s.timesteps_preset_display,
        values=list(preset_label_by_key_addift.values()),
        state="readonly", width=26,
    )
    timesteps_preset_combobox.grid(row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)
    timesteps_preset_combobox.bind(
        "<<ComboboxSelected>>", _reflect_timesteps_preset_combobox_addift
    )

    # row 1: train_min_timesteps / train_max_timesteps
    ttk.Label(lf2, text="train_min_timesteps", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    spin_min_timesteps_addift = ttk.Spinbox(
        lf2, from_=0, to=999, textvariable=s.train_min_timesteps, width=8
    )
    spin_min_timesteps_addift.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_max_timesteps", width=22, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    spin_max_timesteps_addift = ttk.Spinbox(
        lf2, from_=1, to=1000, textvariable=s.train_max_timesteps, width=8
    )
    spin_max_timesteps_addift.grid(row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: network_strength / diff_alt_ratio
    ttk.Label(lf2, text="network_strength", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    entry_network_strength_addift = ttk.Entry(
        lf2, textvariable=s.network_strength, width=10
    )
    entry_network_strength_addift.grid(row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="diff_alt_ratio", width=22, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.diff_alt_ratio, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 3: train_loss_function / train_snr_gamma
    ttk.Label(lf2, text="train_loss_function", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.train_loss_function, values=LOSS_FUNCTIONS,
                 state="readonly", width=12).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_snr_gamma", width=22, anchor=tk.W).grid(
        row=3, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.train_snr_gamma, width=10).grid(
        row=3, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 4: train_fixed_timesteps_in_batch
    ttk.Checkbutton(lf2, text="train_fixed_timesteps_in_batch",
                    variable=s.train_fixed_timesteps_in_batch).grid(
        row=4, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    s.timesteps_preset_widgets = [
        spin_min_timesteps_addift, spin_max_timesteps_addift, entry_network_strength_addift,
    ]
    _reflect_timesteps_preset_state_addift(s)''',
        ),
        _TextPatch(
            description="_reflect_timesteps_preset_state_addift 関数の新規追加",
            old='''    lf3 = ttk.LabelFrame(parent, text=gettext("lora_memory_opt"))
    lf3.pack(fill=tk.X)
    ttk.Checkbutton(lf3, text="gradient_checkpointing",
                    variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ5: 詳細（Anima固有）
# ──────────────────────────────────────────────────────────────────────────────
def _build_adv_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:''',
            new='''    lf3 = ttk.LabelFrame(parent, text=gettext("lora_memory_opt"))
    lf3.pack(fill=tk.X)
    ttk.Checkbutton(lf3, text="gradient_checkpointing",
                    variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)


def _reflect_timesteps_preset_state_addift(s: _AddifTTrainState) -> None:
    """timesteps_preset_enabled / timesteps_preset_name の現在値をUIへ反映する。

    プリセットチェックボックスのON/OFF切替時・プリセットComboboxの選択時・
    プリセットファイルのロード後（preset_apply実行後）のいずれからも呼び出され、
    以下を一括で同期する:
        - timesteps_preset_display（Combobox表示文字列）
        - timesteps_preset_widgets（train_min/max_timesteps用Spinbox,
          network_strength用Entry）の有効/無効状態
        - 有効時のみ train_min_timesteps / train_max_timesteps / network_strength
          へプリセット値を反映

    Args:
        s: ADDifT学習タブの状態オブジェクト。

    Returns:
        None
    """
    preset_key: str = s.timesteps_preset_name.get()
    preset: dict | None = TIMESTEPS_PRESETS.get(preset_key)
    if preset is None:
        preset_key = next(iter(TIMESTEPS_PRESETS))
        preset = TIMESTEPS_PRESETS[preset_key]
        s.timesteps_preset_name.set(preset_key)

    if s.timesteps_preset_display is not None:
        s.timesteps_preset_display.set(gettext(preset["label_key"]))

    widget_state = tk.DISABLED if s.timesteps_preset_enabled.get() else tk.NORMAL
    for widget in s.timesteps_preset_widgets:
        widget.configure(state=widget_state)

    if s.timesteps_preset_enabled.get():
        s.train_min_timesteps.set(preset["min"])
        s.train_max_timesteps.set(preset["max"])
        s.network_strength.set(preset["strength"])


# ──────────────────────────────────────────────────────────────────────────────
# タブ5: 詳細（Anima固有）
# ──────────────────────────────────────────────────────────────────────────────
def _build_adv_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:''',
        ),
        _TextPatch(
            description="プリセット保存(_collect)へ timesteps_preset_* を追加",
            old='''            "train_loss_function": s.train_loss_function.get(),
            "train_snr_gamma":    float(s.train_snr_gamma.get()),
            # 詳細''',
            new='''            "train_loss_function": s.train_loss_function.get(),
            "train_snr_gamma":    float(s.train_snr_gamma.get()),
            "timesteps_preset_enabled": bool(s.timesteps_preset_enabled.get()),
            "timesteps_preset_name":    s.timesteps_preset_name.get(),
            # 詳細''',
        ),
        _TextPatch(
            description="プリセット復元(_apply)へ timesteps_preset_* を追加",
            old='''        _s(s.train_loss_function, "train_loss_function", "MSE")
        _s(s.train_snr_gamma,    "train_snr_gamma",    0.0)
        _s(s.attn_mode,         "attn_mode",          "torch")''',
            new='''        _s(s.train_loss_function, "train_loss_function", "MSE")
        _s(s.train_snr_gamma,    "train_snr_gamma",    0.0)
        _s(s.timesteps_preset_enabled, "timesteps_preset_enabled", False)
        _s(s.timesteps_preset_name,    "timesteps_preset_name",    "local")
        _s(s.attn_mode,         "attn_mode",          "torch")''',
        ),
        _TextPatch(
            description="プリセットロード(_load)後にUI同期処理を追加",
            old='''        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls_addift(s, s.layer_canvas, s.layer_inner)
        _apply(data)
        s.log_fn(gettext("lora_preset_log_loaded", name=src.name))''',
            new='''        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls_addift(s, s.layer_canvas, s.layer_inner)
        _apply(data)
        _reflect_timesteps_preset_state_addift(s)
        s.log_fn(gettext("lora_preset_log_loaded", name=src.name))''',
        ),
    ]


def _texts_ja_patches() -> list[_TextPatch]:
    """app/texts_ja.json に対するパッチ一覧を返す。

    Returns:
        引継ぎ書 §4.4 で定義された5キーを追加する1件のパッチ。
    """
    return [
        _TextPatch(
            description="timesteps プリセット用 翻訳キー5件の追加",
            old='''  "addift_params_label": "ADDifTパラメータ",
  "addift_phase2_layer_placeholder":''',
            new='''  "addift_params_label": "ADDifTパラメータ",
  "addift_timesteps_preset_enable": "プリセットから選択する",
  "addift_timesteps_preset_label": "プリセット:",
  "addift_timesteps_preset_local": "局所（瞳・小物などの色/質感変化）",
  "addift_timesteps_preset_style": "画風（全体の色調/画風転換）",
  "addift_timesteps_preset_composition": "構図（ポーズ/構造変化）",
  "addift_phase2_layer_placeholder":''',
        ),
    ]


def main() -> int:
    """エントリポイント。

    Returns:
        終了コード（0: 成功, 1: 失敗）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="プロジェクトルート（既定: 現在のディレクトリ）。"
             "app/addift_train.py, app/texts_ja.json をこの基点から解決する。",
    )
    args = parser.parse_args()

    targets: list[tuple[Path, list[_TextPatch]]] = [
        (args.root / "app" / "addift_train.py", _addift_train_patches()),
        (args.root / "app" / "texts_ja.json", _texts_ja_patches()),
    ]

    for target, patches in targets:
        print(f"\n=== {target} ===")
        try:
            _apply_text_patches(target, patches)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [error] {exc}", file=sys.stderr)
            return 1

    py_target = args.root / "app" / "addift_train.py"
    py_compile.compile(str(py_target), doraise=True)
    print(f"\n構文チェック OK: {py_target}")

    json_target = args.root / "app" / "texts_ja.json"
    json.loads(json_target.read_text(encoding="utf-8"))
    print(f"JSON妥当性チェック OK: {json_target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
