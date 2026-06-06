"""apply_fix_bug03_sample_prompt_parse.py
BUG-03 修正: anima_train_leco.py の _parse_sample_prompt_line を改修する。

問題点:
1. neg_pat の正規表現 `[ \t]--n[ \t]+...(?=[ \t]*$)` が
   行末に続くスペースなし・フラグなしのケースでマッチしない。
2. prompt_end の検索で ` {flag} ` (両端スペース) を使っており、
   フラグが行末に来るとプロンプト本体が正しく切り出されない。
3. `__import__("re")` を使っておりモジュールレベルの import re と不整合。

対象ファイル: sd-scripts/anima_train_leco.py
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET_FILE = Path("sd-scripts/anima_train_leco.py")

# ファイル内の実際の文字列と完全一致させる（エスケープに注意）
OLD = (
    'def _parse_sample_prompt_line(line: str) -> tuple[str, dict]:\n'
    '    """サンプルプロンプト行をパースして (prompt_text, kwargs) を返す。\n'
    '\n'
    '    書式: <prompt> [--n <neg>] [--w W] [--h H] [--s S] [--l L] [--fs FS] [--d D]\n'
    '    """\n'
    '    import shlex\n'
    '\n'
    '    result: dict = {}\n'
    '    # --n は残りすべてを取るため最初に抽出\n'
    '    neg = ""\n'
    '    neg_pat = __import__("re").search(r"[ \t]--n[ \t]+(.+?)(?=[ \t]+--[a-z]|[ \t]*$)", line)\n'
    '    if neg_pat:\n'
    '        neg = neg_pat.group(1).strip()\n'
    '        line = line[:neg_pat.start()] + line[neg_pat.end():]\n'
    '    result["n"] = neg\n'
    '\n'
    '    flag_map = {\n'
    '        "--w": ("w", int),\n'
    '        "--h": ("h", int),\n'
    '        "--s": ("s", int),\n'
    '        "--l": ("l", float),\n'
    '        "--fs": ("fs", float),\n'
    '        "--d": ("d", int),\n'
    '    }\n'
    '    prompt_end = len(line)\n'
    '    for flag in flag_map:\n'
    '        idx = line.find(f" {flag} ")\n'
    '        if idx != -1:\n'
    '            prompt_end = min(prompt_end, idx)\n'
    '    prompt_text = line[:prompt_end].strip()\n'
    '\n'
    '    remainder = line[prompt_end:]\n'
    '    for flag, (key, cast) in flag_map.items():\n'
    '        pat = re.search(rf"{re.escape(flag)}[ \t]+([^ \t]+)", remainder)\n'
    '        if pat:\n'
    '            try:\n'
    '                result[key] = cast(pat.group(1))\n'
    '            except (ValueError, TypeError):\n'
    '                pass\n'
    '\n'
    '    return prompt_text, result'
)

NEW = (
    'def _parse_sample_prompt_line(line: str) -> tuple[str, dict]:\n'
    '    """サンプルプロンプト行をパースして (prompt_text, kwargs) を返す。\n'
    '\n'
    '    書式: <prompt> [--w W] [--h H] [--s S] [--l L] [--fs FS] [--d D] [--n <neg>]\n'
    '    フラグの順序は問わない。--n はスペースを含む値をとりうるため専用処理する。\n'
    '    """\n'
    '    result: dict = {}\n'
    '\n'
    '    # --n を最初に除去（値にスペースが含まれうるため専用処理）\n'
    '    # \\s+--n\\s+ の後ろ: 次のフラグ (\\s+--[a-zA-Z]) または行末 (\\s*$) まで\n'
    '    neg = ""\n'
    '    neg_pat = re.search(r"\\s+--n\\s+(.+?)(?=\\s+--[a-zA-Z]|\\s*$)", line)\n'
    '    if neg_pat:\n'
    '        neg = neg_pat.group(1).strip()\n'
    '        line = line[:neg_pat.start()] + line[neg_pat.end():]\n'
    '    result["n"] = neg\n'
    '\n'
    '    flag_map = {\n'
    '        "--w":  ("w",  int),\n'
    '        "--h":  ("h",  int),\n'
    '        "--s":  ("s",  int),\n'
    '        "--l":  ("l",  float),\n'
    '        "--fs": ("fs", float),\n'
    '        "--d":  ("d",  int),\n'
    '    }\n'
    '\n'
    '    # プロンプト本体の終端 = 最初の " --flag" の出現位置\n'
    '    # フラグが行末に来る場合も考慮し、空白+フラグ で検索する\n'
    '    prompt_end = len(line)\n'
    '    for flag in flag_map:\n'
    '        m = re.search(re.escape(f" {flag}") + r"(?=\\s|$)", line)\n'
    '        if m:\n'
    '            prompt_end = min(prompt_end, m.start())\n'
    '    prompt_text = line[:prompt_end].strip()\n'
    '\n'
    '    remainder = line[prompt_end:]\n'
    '    for flag, (key, cast) in flag_map.items():\n'
    '        pat = re.search(re.escape(flag) + r"\\s+(\\S+)", remainder)\n'
    '        if pat:\n'
    '            try:\n'
    '                result[key] = cast(pat.group(1))\n'
    '            except (ValueError, TypeError):\n'
    '                pass\n'
    '\n'
    '    return prompt_text, result'
)


def apply():
    if not TARGET_FILE.exists():
        print(f"[ERROR] 対象ファイルが見つかりません: {TARGET_FILE}")
        sys.exit(1)

    raw = TARGET_FILE.read_bytes()
    text = _adapt(raw.decode("utf-8"))

    old_norm = _adapt(OLD)
    new_norm = _adapt(NEW)

    if old_norm not in text:
        print("[ERROR] 置換対象の文字列が見つかりません。")
        print("  先頭部分:")
        print(repr(old_norm[:300]))
        sys.exit(1)

    count = text.count(old_norm)
    if count != 1:
        print(f"[ERROR] 置換対象が {count} 箇所見つかりました。中断します。")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET_FILE.with_suffix(f".bak_{ts}")
    shutil.copy2(TARGET_FILE, bak)
    print(f"[INFO] バックアップ作成: {bak}")

    new_text = text.replace(old_norm, new_norm, 1)

    if b"\r\n" in raw:
        new_text = new_text.replace("\n", "\r\n")

    TARGET_FILE.write_text(new_text, encoding="utf-8", newline="")
    print(f"[OK] BUG-03 修正を適用しました: {TARGET_FILE}")
    print("  _parse_sample_prompt_line の正規表現とプロンプト本体抽出ロジックを修正しました。")


if __name__ == "__main__":
    # プロジェクトルートから実行すること (python apply_fix_*.py)
    apply()
