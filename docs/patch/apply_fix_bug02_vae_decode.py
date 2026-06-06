"""apply_fix_bug02_vae_decode.py
BUG-02 修正: anima_train_leco.py の vae.decode() 戻り値を
dict / DecoderOutput オブジェクト両対応に変更する。

qwen_image_autoencoder_kl の decode() は diffusers の AutoencoderKL と異なり
dict を返す場合がある。

変更前:
    decoded = vae.decode(latents_in).sample

変更後:
    _dec = vae.decode(latents_in)
    if isinstance(_dec, dict):
        decoded = _dec.get("sample", _dec.get("frames", next(iter(_dec.values()))))
    else:
        decoded = _dec.sample if hasattr(_dec, "sample") else _dec

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

OLD = (
    "                latents_in = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]\n"
    "                decoded = vae.decode(latents_in).sample"
)
NEW = (
    "                latents_in = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]\n"
    "                _dec = vae.decode(latents_in)\n"
    "                if isinstance(_dec, dict):\n"
    "                    decoded = _dec.get(\"sample\", _dec.get(\"frames\", next(iter(_dec.values()))))\n"
    "                else:\n"
    "                    decoded = _dec.sample if hasattr(_dec, \"sample\") else _dec"
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
        print("  探している文字列:")
        for line in old_norm.splitlines():
            print(f"    {repr(line)}")
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
    print(f"[OK] BUG-02 修正を適用しました: {TARGET_FILE}")
    print("  vae.decode() の戻り値を dict / DecoderOutput 両対応に変更しました。")


if __name__ == "__main__":
    # プロジェクトルートから実行すること: python apply_fix_bug02_vae_decode.py
    apply()
