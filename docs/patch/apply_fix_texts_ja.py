"""apply_fix_texts_ja.py - texts_ja.json に ADDifT学習タブ用の翻訳キーを追加するパッチ。

末尾の "}" の直前に新規キー群を挿入する(末尾キーのみ前後をtrailing-comma対応)。

使い方:
    python3 apply_fix_texts_ja.py /path/to/texts_ja.json
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

NEW_KEYS: "OrderedDict[str, str]" = OrderedDict([
    # 主タブ
    ("main_tab_addift_train", "ADDifT学習"),

    # 内タブ
    ("addift_tab_dataset", "データセット"),

    # データセットタブ
    ("addift_dataset_label", "画像ペア (ADDifT)"),
    ("addift_image_a_label", "画像A（変換前）"),
    ("addift_image_a_path", "image_a"),
    ("addift_image_b_label", "画像B（変換後）"),
    ("addift_image_b_path", "image_b"),
    ("addift_caption_label", "caption（共通）"),
    ("addift_caption_note", "画像A・B共通のキャプションを指定します。空欄でも学習可能です。"),
    ("addift_diff_mask_label", "差分マスク（任意）"),
    ("addift_diff_use_diff_mask", "差分マスクを使用する (diff_use_diff_mask)"),
    ("addift_diff_mask_path", "diff_mask_path"),
    ("addift_diff_mask_note", "白=学習対象領域、黒=対象外領域として扱われます。"),

    # ネットワークタブ
    ("addift_network_unet_only_note", "ADDifT は DiT (UNet相当) のみを学習します。"),

    # 学習設定タブ
    ("addift_params_label", "ADDifTパラメータ"),

    # フェーズ2 仮設置タブ
    ("addift_phase2_layer_placeholder",
     "階層学習タブは仮設置です。フェーズ2でlora_train/leco_train相当の階層別スケール設定を実装予定です。"),
    ("addift_phase2_monitor_placeholder",
     "モニターグラフタブは仮設置です。フェーズ2でlossカーブ等のグラフ描画を実装予定です。"),
    ("addift_phase2_monitor_layer_placeholder",
     "モニター階層タブは仮設置です。フェーズ2で階層別スケールのグラフ表示を実装予定です。"),
    ("addift_phase2_sample_placeholder",
     "サンプル生成タブは仮設置です。フェーズ2でA/Bギャラリー表示を実装予定です。"),

    # バリデーション
    ("addift_validate_no_image_a", "画像A (image_a) を指定してください。"),
    ("addift_validate_image_a_missing", "画像Aが見つかりません: {path}"),
    ("addift_validate_no_image_b", "画像B (image_b) を指定してください。"),
    ("addift_validate_image_b_missing", "画像Bが見つかりません: {path}"),
    ("addift_validate_no_diff_mask", "差分マスクのパスを指定してください。"),
    ("addift_validate_diff_mask_missing", "差分マスク画像が見つかりません: {path}"),
    ("addift_validate_iterations", "train_iterations は1以上を指定してください。"),
    ("addift_validate_timesteps_range", "train_min_timesteps は train_max_timesteps 未満を指定してください。"),

    # 実行ログ
    ("addift_start_log", "[ADDifT Train] 学習を開始します..."),
    ("addift_log_path", "[ADDifT Train] ログ: {path}"),
    ("addift_done", "[ADDifT Train] 終了 (code={rc})"),
    ("addift_start_error", "[ADDifT Train] 起動エラー: {error}"),
    ("addift_stop_no_proc", "[ADDifT Train] 実行中のプロセスはありません。"),
    ("addift_stop_sent", "[ADDifT Train] 停止信号を送信しました。"),

    # アプリ終了時クリーンアップ
    ("unload_addift_proc", "[Unload] ADDifT学習プロセスを終了しました。"),
])


def apply_fix(target_path: Path) -> None:
    with open(target_path, encoding="utf-8") as f:
        data: "OrderedDict[str, object]" = json.load(f, object_pairs_hook=OrderedDict)

    added = 0
    skipped = 0
    for key, value in NEW_KEYS.items():
        if key in data:
            skipped += 1
            continue
        data[key] = value
        added += 1

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"適用完了: {added} 件追加 / {skipped} 件は既存のためスキップ -> {target_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 apply_fix_texts_ja.py /path/to/texts_ja.json", file=sys.stderr)
        raise SystemExit(1)
    apply_fix(Path(sys.argv[1]))
