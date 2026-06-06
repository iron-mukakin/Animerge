"""apply_fix_leco_argdup.py
anima_train_leco.py の argparse 引数重複エラー修正パッチ

問題:
  train_util.add_training_arguments() が --sample_every_n_steps /
  --sample_prompts / --sample_save_dir をすでに登録しているため、
  前パッチで追加した同名の add_argument() が重複エラーを起こす。

修正内容:
  重複する 3 引数の add_argument() ブロックを削除し、
  新規追加が必要な --sample_keep_vae のみ残す。

使い方:
    python apply_fix_leco_argdup.py [anima_train_leco.py のパス]
    省略時は ./sd-scripts/anima_train_leco.py を対象とする。
"""
from __future__ import annotations
import sys
from pathlib import Path


def _adapt(src: str, ref: str) -> str:
    if "\r\n" in ref:
        return src.replace("\n", "\r\n")
    return src.replace("\r\n", "\n")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    old_a = _adapt(old, text)
    new_a = _adapt(new, text)
    count = text.count(old_a)
    if count == 0:
        raise RuntimeError(
            f"[{label}] 差分文字列が見つかりません。\n---\n{old[:120]}\n---"
        )
    if count > 1:
        raise RuntimeError(f"[{label}] 差分文字列が複数マッチしました ({count}箇所)。")
    return text.replace(old_a, new_a, 1)


# ── 重複する 3 引数を削除し、keep_vae のみ残す ──────────────────────────────
P_OLD = """\
    # Sample generation
    parser.add_argument(
        "--sample_every_n_steps", type=int, default=None,
        help="Generate sample images every N training steps",
    )
    parser.add_argument(
        "--sample_prompts", type=str, default=None,
        help="Path to sample prompt file (one prompt-line per sample)",
    )
    parser.add_argument(
        "--sample_save_dir", type=str, default=None,
        help="Directory to save sample images",
    )
    parser.add_argument(
        "--sample_keep_vae", action="store_true",
        help=(
            "Keep VAE loaded in VRAM throughout training for sample generation. "
            "Default: reload VAE each time samples are generated, then unload."
        ),
    )"""

P_NEW = """\
    # Sample generation
    # --sample_every_n_steps / --sample_prompts / --sample_save_dir は
    # train_util.add_training_arguments() で登録済みのため追加不要。
    parser.add_argument(
        "--sample_keep_vae", action="store_true",
        help=(
            "Keep VAE loaded in VRAM throughout training for sample generation. "
            "Default: reload VAE each time samples are generated, then unload."
        ),
    )"""


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("sd-scripts") / "anima_train_leco.py"

    if not target.exists():
        print(f"[ERROR] ファイルが見つかりません: {target}")
        sys.exit(1)

    text = target.read_text(encoding="utf-8")
    original = text

    text = _replace_once(text, P_OLD, P_NEW, "重複引数削除")
    print("  OK: 重複引数削除 (--sample_every_n_steps / --sample_prompts / --sample_save_dir)")

    import ast
    try:
        ast.parse(text)
        print("  構文チェック: OK")
    except SyntaxError as e:
        print(f"  [ERROR] 構文エラー: {e}")
        sys.exit(1)

    bak = target.with_suffix(".py.bak_argdup")
    bak.write_text(original, encoding="utf-8")
    print(f"  バックアップ: {bak.name}")

    target.write_text(text, encoding="utf-8")
    print(f"  書き込み完了: {target.name}")


if __name__ == "__main__":
    main()
