"""
apply_fix_leco_wrapper.py
leco_train.py の _build_command 内で生成する _gui_leco_wrapper.py の
ハードコードパスを __file__ ベースの動的解決に書き換える。

使い方:
    python apply_fix_leco_wrapper.py [leco_train.py のパス]
    省略時は同ディレクトリの leco_train.py を対象にする。
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
# 修正: _gui_leco_wrapper.py 生成コードのハードコードパス除去
#
# 変更前: f文字列で sd_scripts_root / train_script の絶対パスを直接展開
# 変更後: ラッパー内で Path(__file__).resolve().parent から動的に解決
#
# ラッパーは sd-scripts/ 直下に置かれるため:
#   __file__ の親 = sd_scripts_root
#   sd_scripts_root / "anima_train_leco.py" = train_script
# ──────────────────────────────────────────────────────────────────────────────
OLD = """\
    wrapper = sd_scripts_root / "_gui_leco_wrapper.py"
    wrapper.write_text(
        "import sys, os\\n"
        f"sys.path.insert(0, '{sd_scripts_root.as_posix()}')\\n"
        f"os.chdir('{sd_scripts_root.as_posix()}')\\n"
        f"with open('{train_script.as_posix()}', encoding='utf-8') as _f:\\n"
        "    _code = compile(_f.read(), _f.name, 'exec')\\n"
        f"exec(_code, {{'__name__': '__main__', '__file__': '{train_script.as_posix()}'}})\\n",
        encoding="utf-8",
    )"""

NEW = """\
    wrapper = sd_scripts_root / "_gui_leco_wrapper.py"
    wrapper.write_text(
        "import sys, os\\n"
        "from pathlib import Path\\n"
        "_root = Path(__file__).resolve().parent\\n"
        "sys.path.insert(0, str(_root))\\n"
        "os.chdir(str(_root))\\n"
        "_train = _root / 'anima_train_leco.py'\\n"
        "with open(_train, encoding='utf-8') as _f:\\n"
        "    _code = compile(_f.read(), str(_train), 'exec')\\n"
        "exec(_code, {'__name__': '__main__', '__file__': str(_train)})\\n",
        encoding="utf-8",
    )"""


def main() -> None:
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        target = Path(__file__).parent / "leco_train.py"

    if not target.exists():
        print(f"[ERROR] ファイルが見つかりません: {target}")
        sys.exit(1)

    print(f"対象ファイル: {target}")

    backup = target.with_suffix(".py.bak")
    backup.write_bytes(target.read_bytes())
    print(f"バックアップ: {backup}")

    ok = apply_patch(target, OLD, NEW, "wrapper ハードコードパス除去")

    if not ok:
        print("\n変更なし。バックアップを復元します。")
        target.write_bytes(backup.read_bytes())
    else:
        print("\n1/1 件の修正を適用しました。")
        print("次回 GUI から学習を開始すると _gui_leco_wrapper.py が動的パスで再生成されます。")


if __name__ == "__main__":
    main()
