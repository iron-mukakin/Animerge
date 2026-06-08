"""
apply_fix_step_epoch_label.py
------------------------------
修正内容:
  lora_train.py : サンプルプレビューのラベルを
                  LoRA学習では「epoch」、LECO学習では「step」と表示する

  適用対象（未適用の2箇所のみ）:
    1. 初期ラベルテキスト "step -"  → is_leco で分岐
    2. 実ファイル時のラベル f"step ..." → is_leco で分岐

使用方法:
  python apply_fix_step_epoch_label.py
  ※ lora_train.py と同じディレクトリで実行すること
"""

from __future__ import annotations
import pathlib
import sys


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def apply_patch(filepath: pathlib.Path, old: str, new: str, label: str) -> bool:
    raw = filepath.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    norm_text = _adapt(text)
    norm_old  = _adapt(old)
    norm_new  = _adapt(new)

    if norm_old not in norm_text:
        print(f"[SKIP] {label}: 対象文字列が見つかりません ({filepath.name})")
        return False

    patched_norm = norm_text.replace(norm_old, norm_new, 1)

    if b"\r\n" in raw:
        patched = patched_norm.replace("\n", "\r\n")
    else:
        patched = patched_norm

    filepath.write_bytes(patched.encode("utf-8"))
    print(f"[OK]   {label}: 適用完了 ({filepath.name})")
    return True


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    target = here / "lora_train.py"

    if not target.exists():
        print(f"[ERROR] ファイルが見つかりません: {target}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 修正1: 初期ラベルテキスト "step -" を is_leco で分岐
    # ------------------------------------------------------------------
    apply_patch(
        target,
        old='        ep_lbl = ttk.Label(cell, text="step -", anchor=tk.CENTER)',
        new='        ep_lbl = ttk.Label(cell, text="step -" if is_leco else "epoch -", anchor=tk.CENTER)',
        label="修正1: 初期ラベル分岐",
    )

    # ------------------------------------------------------------------
    # 修正2: 実ファイル時のラベル f"step ..." を is_leco で分岐
    # ------------------------------------------------------------------
    apply_patch(
        target,
        old='            el.configure(text=f"step {_extract_sample_epoch(p)}")',
        new='            _lbl_prefix = "step" if is_leco else "epoch"\n            el.configure(text=f"{_lbl_prefix} {_extract_sample_epoch(p)}")',
        label="修正2: 実ファイルラベル分岐",
    )

    print("\n完了")


if __name__ == "__main__":
    main()
