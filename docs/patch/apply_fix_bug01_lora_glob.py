"""apply_fix_bug01_lora_glob.py
BUG-01 修正: lora_train.py のサンプルギャラリーglobパターンを実際のファイル名に合わせる。

実際のsd-scripts出力ファイル名:
  {output_name}_e{epoch:06d}_{idx:02d}_{timestamp}_{seed}.png
  例: lora_output_e000001_00_20260603124939_42.png
       lora_output_e000001_01_20260603124939_42.png

変更前: "*_00.png" / "*_01.png"
変更後: "*_e*_00_*.png" / "*_e*_01_*.png"

対象ファイル: app/lora_train.py
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime


def _adapt(s: str) -> str:
    """CRLF/LF 両対応: 比較・置換とも LF に統一する。"""
    return s.replace("\r\n", "\n")


TARGET_FILE = Path("app/lora_train.py")

OLD = '    pat_a = "*_a_*.png" if is_leco else "*_00.png"\n    pat_b = "*_b_*.png" if is_leco else "*_01.png"'
NEW = '    pat_a = "*_a_*.png" if is_leco else "*_e*_00_*.png"\n    pat_b = "*_b_*.png" if is_leco else "*_e*_01_*.png"'


def apply():
    if not TARGET_FILE.exists():
        print(f"[ERROR] 対象ファイルが見つかりません: {TARGET_FILE}")
        sys.exit(1)

    raw = TARGET_FILE.read_bytes()
    text = _adapt(raw.decode("utf-8"))

    old_norm = _adapt(OLD)
    new_norm = _adapt(NEW)

    if old_norm not in text:
        print("[ERROR] 置換対象の文字列が見つかりません。既に適用済みか、ファイルが変更されています。")
        print("  探している文字列:")
        print(repr(old_norm))
        sys.exit(1)

    count = text.count(old_norm)
    if count != 1:
        print(f"[ERROR] 置換対象が {count} 箇所見つかりました。一意でないため中断します。")
        sys.exit(1)

    # バックアップ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET_FILE.with_suffix(f".bak_{ts}")
    shutil.copy2(TARGET_FILE, bak)
    print(f"[INFO] バックアップ作成: {bak}")

    new_text = text.replace(old_norm, new_norm, 1)

    # 元のファイルが CRLF だった場合は CRLF で書き戻す
    if b"\r\n" in raw:
        new_text = new_text.replace("\n", "\r\n")

    TARGET_FILE.write_text(new_text, encoding="utf-8", newline="")
    print(f"[OK] BUG-01 修正を適用しました: {TARGET_FILE}")
    print("  変更前: *_00.png / *_01.png")
    print("  変更後: *_e*_00_*.png / *_e*_01_*.png")


if __name__ == "__main__":
    # プロジェクトルートから実行すること: python apply_fix_bug01_lora_glob.py
    apply()
