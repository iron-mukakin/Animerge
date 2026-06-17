"""library/anima_train_utils.py に `--qwen_image_vae_2d` 引数と
`load_qwen_image_vae` ディスパッチャ関数を追加するパッチ。

実行例:
    python apply_fix_anima_train_utils.py library/anima_train_utils.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _patch_lib import apply_substitutions, parse_target_path_argument, report_failure_and_exit

DEFAULT_RELATIVE_PATH: str = "library/anima_train_utils.py"

SEARCH_TEXT: str = '''    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )


# Loss weighting
'''

REPLACEMENT_TEXT: str = '''    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )
    parser.add_argument(
        "--qwen_image_vae_2d",
        action="store_true",
        help="Use the image-only 2D Qwen-Image VAE implementation. Official Qwen-Image VAE weights are converted to 2D convolutions on load; numerically equivalent to the 3D VAE for single images. --vae_disable_cache becomes a no-op (the 2D VAE has no temporal cache)."
        + " / 画像専用の2D Qwen-Image VAE実装を使用します。公式の重みはロード時に2D畳み込みへ変換され、単一画像であれば3D VAEと数値的に等価です。--vae_disable_cacheはno-opになります（2D VAEは時間方向キャッシュを持たないため）。",
    )


def load_qwen_image_vae(args: argparse.Namespace, device: str = "cpu", disable_mmap: bool = True) -> torch.nn.Module:
    """Qwen-Image VAEをロードする。`--qwen_image_vae_2d` 指定時は画像専用2D実装にディスパッチする。

    Args:
        args: パース済み学習引数。`vae` / `vae_chunk_size` / `vae_disable_cache` /
            `qwen_image_vae_2d` を含むこと(`add_anima_training_arguments` 参照)。
        device: VAEをロードするデバイス。
        disable_mmap: safetensors読み込み時にmmapを無効化するかどうか。

    Returns:
        ロード済みのVAEモデル(3D版`AutoencoderKLQwenImage`または2D版`AutoencoderKLQwenImage2D`)。
    """
    if getattr(args, "qwen_image_vae_2d", False):
        from library import qwen_image_autoencoder_kl_2d

        logger.info("Using image-only Qwen-Image 2D VAE")
        return qwen_image_autoencoder_kl_2d.load_vae(
            args.vae,
            device=device,
            disable_mmap=disable_mmap,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )

    return qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device=device,
        disable_mmap=disable_mmap,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )


# Loss weighting
'''


def main() -> None:
    target_path = parse_target_path_argument(DEFAULT_RELATIVE_PATH)
    try:
        apply_substitutions(target_path, [(SEARCH_TEXT, REPLACEMENT_TEXT)])
    except (FileNotFoundError, ValueError) as error:
        report_failure_and_exit(error)


if __name__ == "__main__":
    main()
