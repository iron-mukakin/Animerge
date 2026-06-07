"""apply_fix_leco_gallery_glob.py

app/leco_train.py のギャラリー表示を修正する。

修正1: glob パターン
  変更前: f"*_{label.lower()}_*.png"  → *_a_*.png / *_b_*.png
  変更後: "*_00_*.png" / "*_01_*.png"
  理由: anima_sample_gen.py が生成するファイル名は
        {output_name}_{step:06d}_{ab_index:02d}_{timestamp}_{seed}.png
        形式であり _a_ / _b_ を含まない。

修正2: ステップ番号抽出 regex
  変更前: r"step([0-9]+)"  → ファイル名に "step" 文字列がないためマッチしない
  変更後: r"_([0-9]{6})_"  → 6桁のグローバルステップ番号を抽出
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("app/leco_train.py")

# ── 修正1: glob パターン ─────────────────────────────────────────────────────

OLD_GLOB = '        _glob_pat = f"*_{label.lower()}_*.png"'
NEW_GLOB = '        _glob_pat = "*_00_*.png" if label == "A" else "*_01_*.png"'

# ── 修正2: ステップ番号抽出 regex ────────────────────────────────────────────

OLD_REGEX = '                m = _re_search(r"step([0-9]+)", p.stem)\n                el.configure(text=f"step {m.group(1)}" if m else p.name)'
NEW_REGEX = '                m = _re_search(r"_([0-9]{6})_", p.stem)\n                el.configure(text=f"step {int(m.group(1))}" if m else p.name)'


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    is_crlf = b"\r\n" in raw
    content = _adapt(raw.decode("utf-8"))

    modified = False

    # 修正1
    if OLD_GLOB in content:
        content = content.replace(OLD_GLOB, NEW_GLOB, 1)
        print("[OK] 修正1: glob パターンを変更しました。")
        modified = True
    elif NEW_GLOB in content:
        print("[SKIP] 修正1: 既に適用済みです。")
    else:
        print("[ERROR] 修正1: 対象文字列が見つかりません。", file=sys.stderr)
        sys.exit(1)

    # 修正2
    if OLD_REGEX in content:
        content = content.replace(OLD_REGEX, NEW_REGEX, 1)
        print("[OK] 修正2: ステップ番号抽出 regex を変更しました。")
        modified = True
    elif NEW_REGEX in content:
        print("[SKIP] 修正2: 既に適用済みです。")
    else:
        print("[ERROR] 修正2: 対象文字列が見つかりません。", file=sys.stderr)
        sys.exit(1)

    if not modified:
        return

    if is_crlf:
        content = content.replace("\n", "\r\n")
    path.write_text(content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
