"""
apply_fix_sample_preview.py
サンプルプレビューが表示されない問題を2ファイルへのパッチで修正する。

修正1: lora_train.py
  _build_sample_tab_common の is_leco=True 時のglobパターン誤り
  *_a_*.png / *_b_*.png → *_00_*.png / *_01_*.png

修正2: leco_train.py
  _build_leco_sample_tab の相対インポート失敗を絶対インポートに変更し
  例外の握りつぶしを除去する。

使い方:
    python apply_fix_sample_preview.py [app_dir]
    app_dir 省略時はスクリプトと同じディレクトリを対象にする。
"""
from __future__ import annotations
import sys
from pathlib import Path


def _adapt(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def apply_patch(target: Path, old_raw: str, new_raw: str, label: str) -> bool:
    old = _adapt(old_raw)
    new = _adapt(new_raw)
    src = _adapt(target.read_text(encoding="utf-8"))
    count = src.count(old)
    if count == 0:
        print(f"[SKIP] {label}: 対象文字列が見つかりません（既適用か不一致）")
        return False
    if count > 1:
        print(f"[ERROR] {label}: 対象文字列が複数箇所にマッチします。中断します。")
        return False
    result = src.replace(old, new, 1)
    original = target.read_text(encoding="utf-8")
    if "\r\n" in original:
        result = result.replace("\n", "\r\n")
    target.write_text(result, encoding="utf-8")
    print(f"[OK]   {label}: 適用完了")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 修正1: lora_train.py — is_leco=True 時のglobパターン誤り
#
# 実際の保存ファイル名: leco_output_000002_00_YYYYMMDDHHMMSS_42.png
# 誤: *_a_*.png / *_b_*.png  (マッチしない)
# 正: *_00_*.png / *_01_*.png
# ──────────────────────────────────────────────────────────────────────────────
OLD_LORA = """\
    sample_dir = _sample_dir(s) if not is_leco else (s.paths.root / "log" / "sample_gen")
    pat_a = "*_a_*.png" if is_leco else "*_e*_00_*.png"
    pat_b = "*_b_*.png" if is_leco else "*_e*_01_*.png\""""

NEW_LORA = """\
    sample_dir = _sample_dir(s) if not is_leco else (s.paths.root / "log" / "sample_gen")
    pat_a = "*_00_*.png" if is_leco else "*_e*_00_*.png"
    pat_b = "*_01_*.png" if is_leco else "*_e*_01_*.png\""""

# ──────────────────────────────────────────────────────────────────────────────
# 修正2: leco_train.py — 相対インポート失敗と例外握りつぶしの除去
#
# from . import lora_train はスクリプト実行時に必ず ImportError になる。
# importlib を使った絶対インポートに変更し、失敗時は詳細を表示する。
# ──────────────────────────────────────────────────────────────────────────────
OLD_LECO = """\
def _build_leco_sample_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    \"\"\"LECO サンプル生成タブ。lora_train._build_sample_tab_common を流用。\"\"\"
    # lora_train モジュールが同一パッケージにある前提で動的インポート
    try:
        from . import lora_train as _lt
        _lt._build_sample_tab_common(parent, s, is_leco=True)
    except Exception:
        # フォールバック: 直接インポートが使えない場合は簡易UI
        _build_leco_sample_tab_inline(parent, s)"""

NEW_LECO = """\
def _build_leco_sample_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    \"\"\"LECO サンプル生成タブ。lora_train._build_sample_tab_common を流用。\"\"\"
    import importlib as _il
    import importlib.util as _ilu
    import logging as _lg
    _log = _lg.getLogger(__name__)
    try:
        # 同ディレクトリの lora_train.py を __file__ 基準で絶対パス解決
        _here = Path(__file__).resolve().parent
        _spec = _ilu.spec_from_file_location("lora_train", _here / "lora_train.py")
        if _spec is None or _spec.loader is None:
            raise ImportError(f"spec_from_file_location failed: {_here / 'lora_train.py'}")
        _lt = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_lt)
        _lt._build_sample_tab_common(parent, s, is_leco=True)
    except Exception as _e:
        _log.error("[_build_leco_sample_tab] lora_train ロード失敗、フォールバックへ: %s", _e)
        _build_leco_sample_tab_inline(parent, s)"""


def main() -> None:
    if len(sys.argv) >= 2:
        app_dir = Path(sys.argv[1])
    else:
        app_dir = Path(__file__).parent

    lora_train = app_dir / "lora_train.py"
    leco_train = app_dir / "leco_train.py"

    for f in (lora_train, leco_train):
        if not f.exists():
            print(f"[ERROR] ファイルが見つかりません: {f}")
            sys.exit(1)

    print(f"対象ディレクトリ: {app_dir}")

    # バックアップ
    for f in (lora_train, leco_train):
        bak = f.with_suffix(".py.bak")
        bak.write_bytes(f.read_bytes())
        print(f"バックアップ: {bak}")

    ok1 = apply_patch(lora_train, OLD_LORA, NEW_LORA,
                      "lora_train.py — is_leco globパターン修正")
    ok2 = apply_patch(leco_train,  OLD_LECO, NEW_LECO,
                      "leco_train.py — 相対インポート→絶対インポート + 例外可視化")

    applied = sum([ok1, ok2])
    print(f"\n{applied}/2 件の修正を適用しました。")
    if applied == 0:
        print("変更なし。バックアップを復元します。")
        for f in (lora_train, leco_train):
            bak = f.with_suffix(".py.bak")
            f.write_bytes(bak.read_bytes())


if __name__ == "__main__":
    main()
