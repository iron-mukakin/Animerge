"""apply_fix_addift_train.py — addift_train.py へ DPOモードのフック呼び出しを追加するパッチ。

純Python実装(`patch`コマンド不要)。CRLF/LF混在ファイルにも `_adapt()` で対応する。

使い方:
    python apply_fix_addift_train.py <addift_train.py のパス>
    (省略時はカレントディレクトリの addift_train.py を対象とする)
"""
from __future__ import annotations

import sys
from pathlib import Path


def _adapt(reference_text: str, snippet: str) -> str:
    """snippet の改行コードを reference_text の支配的な改行コードへ合わせる。"""
    uses_crlf = reference_text.count("\r\n") >= reference_text.count("\n") - reference_text.count("\r\n")
    normalized = snippet.replace("\r\n", "\n")
    return normalized.replace("\n", "\r\n") if uses_crlf else normalized


def _apply_one_fix(text: str, old: str, new: str, description: str) -> str:
    """old(改行コード非依存)を new へ1箇所だけ置換する。見つからない/複数一致時は例外送出。"""
    old_adapted = _adapt(text, old)
    new_adapted = _adapt(text, new)
    count = text.count(old_adapted)
    if count != 1:
        raise ValueError(f"[{description}] 一致数 {count} (期待値 1) — ファイルが既に変更済みか想定外の内容です。")
    return text.replace(old_adapted, new_adapted, 1)


FIXES: list[tuple[str, str, str]] = [
    (
        "import文へ addift_dpo_ui を追加",
        "try:\n    from .i18n import gettext, load_language\n"
        "except ImportError:\n    import sys as _sys\n"
        "    from pathlib import Path as _Path\n"
        "    _app_dir = _Path(__file__).resolve().parent\n"
        "    if str(_app_dir) not in _sys.path:\n"
        "        _sys.path.insert(0, str(_app_dir))\n"
        "    from i18n import gettext, load_language  # type: ignore[no-redef]\n",
        "try:\n    from .i18n import gettext, load_language\n    from . import addift_dpo_ui\n"
        "except ImportError:\n    import sys as _sys\n"
        "    from pathlib import Path as _Path\n"
        "    _app_dir = _Path(__file__).resolve().parent\n"
        "    if str(_app_dir) not in _sys.path:\n"
        "        _sys.path.insert(0, str(_app_dir))\n"
        "    from i18n import gettext, load_language  # type: ignore[no-redef]\n"
        "    import addift_dpo_ui  # type: ignore[no-redef]\n",
    ),
    (
        "_AddifTTrainStateにDPOモード変数を追加",
        '        # ── ADDifT固有パラメータ ──────────────────────────────────\n'
        '        self.train_min_timesteps = tk.IntVar(value=200)',
        '        # ── ADDifT固有パラメータ ──────────────────────────────────\n'
        '        addift_dpo_ui.attach_dpo_mode_vars(self)\n'
        '        self.train_min_timesteps = tk.IntVar(value=200)',
    ),
    (
        "_build_dataset_tabのラベル変数化とDPOフック登録",
        '    ttk.Label(lf, text=gettext("addift_image_a_label"), foreground="#1D4ED8").grid(\n'
        '        row=0, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(3, 0))\n'
        '    _image_preview_row(lf, 1, gettext("addift_image_a_path"), s.image_a_path, preview_a)\n'
        '\n'
        '    ttk.Label(lf, text=gettext("addift_image_b_label"), foreground="#1D4ED8").grid(\n'
        '        row=2, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(6, 0))\n'
        '    _image_preview_row(lf, 3, gettext("addift_image_b_path"), s.image_b_path, preview_b)\n',
        '    label_image_a = ttk.Label(lf, text=gettext("addift_image_a_label"), foreground="#1D4ED8")\n'
        '    label_image_a.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(3, 0))\n'
        '    _image_preview_row(lf, 1, gettext("addift_image_a_path"), s.image_a_path, preview_a)\n'
        '\n'
        '    label_image_b = ttk.Label(lf, text=gettext("addift_image_b_label"), foreground="#1D4ED8")\n'
        '    label_image_b.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(6, 0))\n'
        '    _image_preview_row(lf, 3, gettext("addift_image_b_path"), s.image_b_path, preview_b)\n'
        '    addift_dpo_ui.attach_dataset_label_refresh(s, label_image_a, label_image_b)\n',
    ),
    (
        "_build_train_tabへDPOモードUI挿入(ADDifTパラメータ枠の直前)",
        '    # ── ADDifT固有パラメータ ─────────────────────────────────────────\n'
        '    lf2 = ttk.LabelFrame(parent, text=gettext("addift_params_label"))',
        '    # ── DPOモード（ADDifT固有パラメータの先頭に相当）─────────────────\n'
        '    addift_dpo_ui.build_dpo_mode_controls(parent, s)\n'
        '\n'
        '    # ── ADDifT固有パラメータ ─────────────────────────────────────────\n'
        '    lf2 = ttk.LabelFrame(parent, text=gettext("addift_params_label"))',
    ),
    (
        "_build_commandへDPO関連引数を追加",
        '            cmd += ["--network_args", f"anima_block_lr_weight={_weight_str}"]\n'
        '\n'
        '    return cmd',
        '            cmd += ["--network_args", f"anima_block_lr_weight={_weight_str}"]\n'
        '\n'
        '    addift_dpo_ui.append_dpo_command_args(s, cmd)\n'
        '\n'
        '    return cmd',
    ),
    (
        "_validateへDPOバリデーションを追加",
        'def _validate(s: _AddifTTrainState) -> str | None:\n'
        '    if not s.model_path.get():',
        'def _validate(s: _AddifTTrainState) -> str | None:\n'
        '    _dpo_error = addift_dpo_ui.validate_dpo_mode(s)\n'
        '    if _dpo_error is not None:\n'
        '        return _dpo_error\n'
        '    if not s.model_path.get():',
    ),
    (
        "プリセット_collect()へDPO設定を追加",
        '            "sample_b_negative_prompt":  s.sample_b_negative_prompt.get(),\n'
        '        }\n'
        '\n'
        '    def _apply(data: dict) -> None:',
        '            "sample_b_negative_prompt":  s.sample_b_negative_prompt.get(),\n'
        '            **addift_dpo_ui.collect_dpo_mode_preset(s),\n'
        '        }\n'
        '\n'
        '    def _apply(data: dict) -> None:',
    ),
    (
        "プリセット_load()へDPO設定復元を追加",
        '        _apply(data)\n'
        '        _reflect_timesteps_preset_state_addift(s)\n'
        '        s.log_fn(gettext("lora_preset_log_loaded", name=src.name))',
        '        _apply(data)\n'
        '        addift_dpo_ui.apply_dpo_mode_preset(s, data)\n'
        '        _reflect_timesteps_preset_state_addift(s)\n'
        '        s.log_fn(gettext("lora_preset_log_loaded", name=src.name))',
    ),
]


def _verify_run() -> None:
    target_arg = sys.argv[1] if len(sys.argv) > 1 else "addift_train.py"
    target_path = Path(target_arg)
    if not target_path.is_file():
        raise FileNotFoundError(f"対象ファイルが見つかりません: {target_path}")

    text = target_path.read_text(encoding="utf-8")
    for description, old, new in FIXES:
        text = _apply_one_fix(text, old, new, description)
        print(f"[OK] {description}")

    target_path.write_text(text, encoding="utf-8")
    print(f"パッチ適用完了: {target_path}")


if __name__ == "__main__":
    _verify_run()
