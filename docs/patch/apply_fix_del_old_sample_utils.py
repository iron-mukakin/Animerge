"""apply_fix_del_old_sample_utils.py

sd-scripts/library/anima_train_utils.py から旧サンプル生成コードを削除する。

削除対象:
  - do_sample
  - sample_images
  - _sample_image_inference
  および直前の「# Sampling」セクションコメントから以降のファイル末尾まで
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    """CRLF / LF どちらのファイルでも一致するよう LF に統一する。"""
    return s.replace("\r\n", "\n")


TARGET = Path("sd-scripts/library/anima_train_utils.py")

OLD = _adapt("""\


# Sampling (Euler discrete for rectified flow)
def do_sample(""")


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    content = _adapt(raw.decode("utf-8"))

    idx = content.find(OLD)
    if idx == -1:
        # 既に削除済みか確認
        if "def do_sample" not in content and "def sample_images" not in content:
            print("[SKIP] 既に適用済みです（削除対象関数が存在しない）。")
            return
        print("[ERROR] 削除対象の文字列が見つかりません。", file=sys.stderr)
        print("  探索文字列:", repr(OLD[:80]))
        sys.exit(1)

    # "\n\n\n# Sampling..." の先頭 2文字 "\n\n" は前ブロック末尾なので残す
    keep_end = idx + 2
    new_content = content[:keep_end]

    if is_crlf := b"\r\n" in raw:
        new_content = new_content.replace("\n", "\r\n")
    path.write_text(new_content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")
    print("     削除: do_sample / sample_images / _sample_image_inference")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
