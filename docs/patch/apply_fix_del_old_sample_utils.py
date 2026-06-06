"""apply_fix_del_old_sample_utils.py

anima_train_utils.py から旧サンプル生成コードを削除する。

削除対象:
  - do_sample (L310)
  - sample_images (L420)
  - _sample_image_inference (L515)

および直前のコメント行・セクション区切り。
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    """CRLF / LF どちらのファイルでも一致するよう LF に統一する。"""
    return s.replace("\r\n", "\n")


TARGET = Path("sd-scripts/anima_train_utils.py")

# 削除する範囲: 「# Sampling」コメントから始まりファイル末尾まで
# （save_anima_model_on_epoch_end_or_stepwise の closing ")" の直後の空行まで残す）

OLD = _adapt("""\


# Sampling (Euler discrete for rectified flow)
def do_sample(""")

NEW = ""  # このセクション以降を全て削除するので NEW は空


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    # LF 正規化して処理
    content = _adapt(raw.decode("utf-8"))

    idx = content.find(OLD)
    if idx == -1:
        print("[ERROR] 削除対象の文字列が見つかりません。", file=sys.stderr)
        print("  探索文字列:", repr(OLD[:80]))
        sys.exit(1)

    # OLD が見つかった位置以降を全て削除（"\n\n\n# Sampling..." 以降）
    # ただし OLD の先頭 2 文字（\n\n）は前のブロックの末尾なので残す
    keep_end = idx + 2  # "\n\n" の後ろ（コメントと関数定義の手前）
    new_content = content[:keep_end]

    if _adapt(path.read_bytes().decode("utf-8")) == new_content:
        print("[SKIP] 既に適用済みです。")
        return

    # 書き戻し（元のファイルの改行コードに合わせる）
    if b"\r\n" in raw:
        new_content = new_content.replace("\n", "\r\n")
    path.write_text(new_content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")
    print(f"     削除: do_sample / sample_images / _sample_image_inference")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
