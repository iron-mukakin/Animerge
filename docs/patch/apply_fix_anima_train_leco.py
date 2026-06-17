"""anima_train_leco.py のVAEロード箇所を `anima_train_utils.load_qwen_image_vae`
ディスパッチャ経由の呼び出しに変更するパッチ。これにより `--qwen_image_vae_2d` が
leco学習でも有効になる。

実行例:
    python apply_fix_anima_train_leco.py anima_train_leco.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _patch_lib import apply_substitutions, parse_target_path_argument, report_failure_and_exit

DEFAULT_RELATIVE_PATH: str = "anima_train_leco.py"

SEARCH_TEXT: str = '''    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True,
        spatial_chunk_size=getattr(args, "vae_chunk_size", None),
        disable_cache=getattr(args, "vae_disable_cache", False),
    )
'''

REPLACEMENT_TEXT: str = '''    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
'''


def main() -> None:
    target_path = parse_target_path_argument(DEFAULT_RELATIVE_PATH)
    try:
        apply_substitutions(target_path, [(SEARCH_TEXT, REPLACEMENT_TEXT)])
    except (FileNotFoundError, ValueError) as error:
        report_failure_and_exit(error)


if __name__ == "__main__":
    main()
