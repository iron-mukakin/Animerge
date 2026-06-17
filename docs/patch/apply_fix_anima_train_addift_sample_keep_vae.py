"""anima_train_addift.py の `setup_parser()` に、欠落している
`--sample_keep_vae` 引数登録を追加するパッチ。

不具合の実体:
    GUI(addift_train.py)はチェックボックスがONのとき `--sample_keep_vae` を
    コマンドラインに追加するが、`anima_train_addift.py` の `setup_parser()` は
    この引数を一度も `parser.add_argument()` で登録していない
    (447行目で `getattr(args, "sample_keep_vae", False)` として参照のみ行っている)。
    そのため argparse が `unrecognized arguments: --sample_keep_vae` で停止する。
    `anima_train_leco.py` には同名の登録が既に存在しており(304行目付近)、本パッチは
    その内容をそのまま `anima_train_addift.py` 側にも追加するもの。
    `--qwen_image_vae_2d` とは無関係の既存バグ。

実行例:
    python apply_fix_anima_train_addift_sample_keep_vae.py anima_train_addift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _patch_lib import apply_substitutions, parse_target_path_argument, report_failure_and_exit

DEFAULT_RELATIVE_PATH: str = "anima_train_addift.py"

SEARCH_TEXT: str = '''    # train_util.verify_training_args が要求するダミー引数
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    return parser
'''

REPLACEMENT_TEXT: str = '''    # train_util.verify_training_args が要求するダミー引数
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    # サンプル生成
    # --sample_every_n_steps / --sample_prompts / --sample_save_dir は
    # train_util.add_training_arguments() で登録済みのため追加不要。
    parser.add_argument(
        "--sample_keep_vae", action="store_true",
        help="Keep VAE loaded in VRAM throughout training for sample generation. Default: reload VAE each time samples are generated, then unload."
        + " / サンプル生成のためVAEを学習中VRAMに保持し続けます。デフォルトではサンプル生成ごとにVAEをロード/解放します。",
    )

    return parser
'''


def main() -> None:
    target_path = parse_target_path_argument(DEFAULT_RELATIVE_PATH)
    try:
        apply_substitutions(target_path, [(SEARCH_TEXT, REPLACEMENT_TEXT)])
    except (FileNotFoundError, ValueError) as error:
        report_failure_and_exit(error)


if __name__ == "__main__":
    main()
