"""apply_fix_bug04b_leco_seed_order.py
BUG-04 hotfix: leco_train.py の起動不能エラーを修正する。

原因:
  apply_fix_bug04_sample_seed.py 適用後、
  _leco_build_prompt_line のシグネチャが
    seed: int = SAMPLE_FIXED_SEED
  となったが、leco_train.py では SAMPLE_FIXED_SEED の定義 (L1519) が
  _leco_build_prompt_line の定義 (L1502) より後にある。
  Pythonはデフォルト引数を関数定義時に評価するため NameError が発生する。

修正:
  シグネチャのデフォルト値を定数リテラル 42 に変更する。
  lora_train.py は SAMPLE_FIXED_SEED が L42 で定義済みのため影響なし。

対象ファイル: app/leco_train.py のみ
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET_FILE = Path("app/leco_train.py")

OLD = (
    'def _leco_build_prompt_line(\n'
    '    prompt: str, neg: str, s: "_LecoTrainState", seed: int = SAMPLE_FIXED_SEED\n'
    ') -> str:'
)

NEW = (
    'def _leco_build_prompt_line(\n'
    '    prompt: str, neg: str, s: "_LecoTrainState", seed: int = 42\n'
    ') -> str:'
)


def apply():
    if not TARGET_FILE.exists():
        print(f"[ERROR] 対象ファイルが見つかりません: {TARGET_FILE}")
        sys.exit(1)

    raw = TARGET_FILE.read_bytes()
    text = _adapt(raw.decode("utf-8"))
    old_n = _adapt(OLD)
    new_n = _adapt(NEW)

    if old_n not in text:
        print("[ERROR] 置換対象が見つかりません。BUG-04パッチが未適用の可能性があります。")
        print(f"  探している文字列: {repr(old_n)}")
        sys.exit(1)

    if text.count(old_n) != 1:
        print(f"[ERROR] 置換対象が複数箇所存在します。中断します。")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET_FILE.with_suffix(f".bak_{ts}")
    shutil.copy2(TARGET_FILE, bak)
    print(f"[INFO] バックアップ: {bak}")

    new_text = text.replace(old_n, new_n, 1)
    if b"\r\n" in raw:
        new_text = new_text.replace("\n", "\r\n")
    TARGET_FILE.write_text(new_text, encoding="utf-8", newline="")
    print(f"[OK] leco_train.py のシグネチャを修正しました")
    print(f"  seed: int = SAMPLE_FIXED_SEED  →  seed: int = 42")


if __name__ == "__main__":
    # プロジェクトルートから実行: python apply_fix_bug04b_leco_seed_order.py
    apply()
