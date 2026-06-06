"""apply_fix_del_old_sample_leco.py

anima_train_leco.py から旧サンプル生成コードを削除する。

削除対象:
  - _generate_samples_leco (学習ループから呼ばれていない・デッドコード)
  - _parse_sample_prompt_line (_generate_samples_leco 内からのみ参照)
  - 両関数を囲む「# Sample generation helper」セクションコメント

削除しない:
  - encode_prompt_anima   (学習ループ L635 で使用中)
  - _anima_forward        (学習ループで使用中)
  - diffusion_anima       (学習ループで使用中)
  - predict_noise_anima   (学習ループで使用中)
  - concat_embeds_anima   (学習ループで使用中)
  - repeat_embeds_anima   (学習ループで使用中)
  - get_initial_latents_anima (学習ループで使用中)
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("sd-scripts/anima_train_leco.py")

# 削除対象: セクションコメント + _generate_samples_leco + _parse_sample_prompt_line
# 終端は「# ---------------------------------------------------------------------------\n# Main」の手前まで

OLD = _adapt("""\
# ---------------------------------------------------------------------------
# Sample generation helper
# ---------------------------------------------------------------------------

def _generate_samples_leco(""")

END_MARKER = _adapt("""\
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------""")


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    content = _adapt(raw.decode("utf-8"))

    start_idx = content.find(OLD)
    if start_idx == -1:
        print("[ERROR] 削除開始マーカーが見つかりません。", file=sys.stderr)
        print("  探索文字列:", repr(OLD[:80]))
        sys.exit(1)

    end_idx = content.find(END_MARKER, start_idx)
    if end_idx == -1:
        print("[ERROR] 削除終了マーカー（# Main）が見つかりません。", file=sys.stderr)
        sys.exit(1)

    # start_idx の手前に残っている空行を1つ保持して削除
    # start_idx 直前の "\n\n" を "\n\n" のまま残し、OLD の1文字目 '#' から end_idx まで削除
    new_content = content[:start_idx] + content[end_idx:]

    if content == new_content:
        print("[SKIP] 既に適用済みです。")
        return

    if b"\r\n" in raw:
        new_content = new_content.replace("\n", "\r\n")
    path.write_text(new_content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")
    print(f"     削除: _generate_samples_leco / _parse_sample_prompt_line / Sample generation helperセクション")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
