"""anima_train_addift.py - ADDifT (Additive Difference Training) for Anima (Cosmos-based DiT) models.

ADDifT は2枚の画像ペア(image_a / image_b)の差分を LoRA に学習させる手法。
TrainTrain の train_diff2() (ADDifT mode) を Rectified Flow / Anima DiT 向けに移植したもの。

Ported from anima_train_leco.py by replacing:
  - LECO prompt-based denoising loop -> 2画像ペアの交互ノイズ予測比較ループ (train_diff2 相当)
  - diffusion_anima() による複数ステップデノイズ -> 不要 (1step ノイズ予測の直接比較)
  - PromptSettings (複数プロンプト) -> 単一 caption (画像A/B共通)

Usage:
    accelerate launch --mixed_precision bf16 anima_train_addift.py \\
        --pretrained_model_name_or_path anima_model.safetensors \\
        --vae vae.safetensors \\
        --qwen3 /path/to/qwen3 \\
        --image_a before.png \\
        --image_b after.png \\
        --caption "a photo" \\
        --output_dir output \\
        --output_name my_addift \\
        --network_dim 8 \\
        --network_alpha 4 \\
        --learning_rate 5e-5 \\
        --optimizer_type AdamW8bit \\
        --train_iterations 50 \\
        --train_min_timesteps 200 \\
        --train_max_timesteps 400 \\
        --mixed_precision bf16 \\
        --gradient_checkpointing
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    custom_train_functions,
    qwen_image_autoencoder_kl,
    strategy_anima,
    train_util,
)
import anima_sample_gen
from library.leco_train_util import build_network_kwargs, get_save_extension
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


LOSS_FUNCTIONS = ("MSE", "L1", "Smooth-L1")


# ---------------------------------------------------------------------------
# Image -> latent
# ---------------------------------------------------------------------------

def load_image_as_tensor(image_path: str, weight_dtype: torch.dtype, device) -> torch.Tensor:
    """画像ファイルを読み込み、VAE入力用の4DテンソルへPILで変換する。

    Returns:
        torch.Tensor: shape [1, 3, H, W], 値域 [-1, 1]
    """
    with Image.open(image_path) as img:
        rgb_image = img.convert("RGB")
    array_image = np.asarray(rgb_image, dtype=np.float32) / 255.0
    array_image = np.transpose(array_image, (2, 0, 1))  # HWC -> CHW
    tensor_image = torch.from_numpy(array_image).unsqueeze(0)  # [1, 3, H, W]
    tensor_image = tensor_image * 2.0 - 1.0
    return tensor_image.to(device, dtype=weight_dtype)


def encode_image_to_latent(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    image_path: str,
    weight_dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """画像をVAEでエンコードし、latent (4D: [1, C, H, W]) を返す。"""
    image_tensor = load_image_as_tensor(image_path, weight_dtype, device)
    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(image_tensor)
    if latent.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
        latent = latent.squeeze(2)
    return latent


def load_diff_mask(mask_path: str, latent_shape: torch.Size, device, dtype) -> torch.Tensor:
    """差分マスク画像を latent と同じ空間サイズへリサイズし、[1,1,H,W] のテンソルとして返す。"""
    with Image.open(mask_path) as img:
        gray_image = img.convert("L")
        h_lat, w_lat = latent_shape[-2], latent_shape[-1]
        gray_image = gray_image.resize((w_lat, h_lat), Image.BILINEAR)
    array_mask = np.asarray(gray_image, dtype=np.float32) / 255.0
    tensor_mask = torch.from_numpy(array_mask).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor_mask.to(device, dtype=dtype)


# ---------------------------------------------------------------------------
# Anima forward / noise utilities
# ---------------------------------------------------------------------------

def predict_noise_anima(
    model: anima_models.Anima,
    timesteps_normalized: torch.Tensor,
    noisy_latents: torch.Tensor,
    embeds_tuple,
    weight_dtype,
    device,
) -> torch.Tensor:
    """Anima DiT で1stepのノイズ予測を行う。

    Args:
        timesteps_normalized: [0, 1] に正規化済みのタイムステップ (shape: [B])
        noisy_latents: ノイズ付加済みlatent (4D: [B, C, H, W])
        embeds_tuple: (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)
    """
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = embeds_tuple

    bs = noisy_latents.shape[0]
    h_lat = noisy_latents.shape[-2]
    w_lat = noisy_latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_lat, w_lat, dtype=weight_dtype, device=device)

    inp = noisy_latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
    pred = model(
        inp,
        timesteps_normalized,
        prompt_embeds.to(device, dtype=weight_dtype),
        padding_mask=padding_mask,
        target_input_ids=t5_input_ids.to(device, dtype=torch.long),
        target_attention_mask=t5_attn_mask.to(device),
        source_attention_mask=attn_mask.to(device),
    )
    return pred.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]


def add_rectified_flow_noise(
    latent: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Rectified Flow のノイズ付加: noisy = sigma * noise + (1 - sigma) * latent

    Args:
        latent: [B, C, H, W]
        noise: [B, C, H, W]
        timesteps: [B], 値域 [0, 1000]
    """
    sigmas = (timesteps / 1000.0).view(-1, 1, 1, 1).to(latent.dtype)
    return sigmas * noise + (1.0 - sigmas) * latent


def sample_curriculum_timesteps(
    step_index: int,
    batch_size: int,
    train_min_timesteps: int,
    train_max_timesteps: int,
    fixed_timesteps_in_batch: bool,
    device,
) -> torch.Tensor:
    """train_diff2 のカリキュラムウィンドウに基づきタイムステップをサンプリングする。

    [train_min_timesteps, train_max_timesteps] の範囲を10分割し、
    step_index に応じてウィンドウを移動させながらサンプリングする。
    """
    span = (train_max_timesteps - train_min_timesteps) / 10.0
    window_index = step_index % 10 + 1
    time_min = train_min_timesteps + span * window_index
    time_max = train_min_timesteps + span * (window_index + 1)
    if time_min >= time_max:
        time_max = time_min + 1.0

    low = int(min(max(time_min, 0), 999))
    high = int(min(max(time_max, low + 1), 1000))

    sample_count = 1 if fixed_timesteps_in_batch else batch_size
    timesteps = torch.randint(low, high, (sample_count,), device=device)
    if fixed_timesteps_in_batch:
        timesteps = timesteps.repeat(batch_size)
    return timesteps.long()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_addift_loss(
    args: argparse.Namespace,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    timesteps: torch.Tensor,
    diff_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """ADDifT の損失: prediction (network有効側予測) と reference (network無効側予測) の差。

    train_snr_gamma > 0 の場合、Rectified Flow の SNR = (1 - sigma)^2 / sigma^2 を用いて
    Min-SNR-gamma 重み付けを行う(sigma = timesteps / 1000)。
    """
    if diff_mask is not None:
        prediction = prediction * diff_mask
        reference = reference * diff_mask

    if args.train_loss_function == "MSE":
        loss = torch.nn.functional.mse_loss(prediction.float(), reference.float(), reduction="none")
    elif args.train_loss_function == "L1":
        loss = torch.nn.functional.l1_loss(prediction.float(), reference.float(), reduction="none")
    else:  # Smooth-L1
        loss = torch.nn.functional.smooth_l1_loss(prediction.float(), reference.float(), reduction="none")

    loss = loss.mean(dim=(1, 2, 3))

    if args.train_snr_gamma > 0:
        sigmas = (timesteps.float() / 1000.0).clamp(min=1e-6, max=1.0 - 1e-6)
        snr = ((1.0 - sigmas) ** 2) / (sigmas**2)
        gamma_over_snr = args.train_snr_gamma / snr
        snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr))
        loss = loss * snr_weight

    return loss.mean()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_addift_weights(
    accelerator,
    network,
    args: argparse.Namespace,
    save_dtype,
    global_step: int,
    last: bool = False,
) -> None:
    """ADDifT用メタデータ付きでネットワーク重みを保存する。"""
    os.makedirs(args.output_dir, exist_ok=True)
    ext = get_save_extension(args)
    ckpt_name = (
        train_util.get_last_ckpt_name(args, ext)
        if last
        else train_util.get_step_ckpt_name(args, ext, global_step)
    )
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    metadata = None
    if not args.no_metadata:
        metadata = {
            "ss_network_module": args.network_module,
            "ss_network_dim": str(args.network_dim),
            "ss_network_alpha": str(args.network_alpha),
            "ss_addift_image_a": os.path.basename(args.image_a),
            "ss_addift_image_b": os.path.basename(args.image_b),
            "ss_addift_caption": args.caption,
            "ss_addift_train_min_timesteps": str(args.train_min_timesteps),
            "ss_addift_train_max_timesteps": str(args.train_max_timesteps),
            "ss_addift_network_strength": str(args.network_strength),
            "ss_addift_diff_alt_ratio": str(args.diff_alt_ratio),
        }
        if args.training_comment:
            metadata["ss_training_comment"] = args.training_comment

    unwrapped = accelerator.unwrap_model(network)
    unwrapped.save_weights(ckpt_file, save_dtype, metadata)
    logger.info(f"saved model to: {ckpt_file}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADDifT training for Anima (Cosmos DiT) models")

    train_util.add_sd_models_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    train_util.add_training_arguments(parser, support_dreambooth=False)
    custom_train_functions.add_custom_train_arguments(parser, support_weighted_captions=False)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    add_logging_arguments(parser)

    # ── ADDifT固有: 画像ペア ────────────────────────────────────────────
    parser.add_argument(
        "--image_a", type=str, required=True,
        help="変換前(差分の起点)の画像パス / Source image path",
    )
    parser.add_argument(
        "--image_b", type=str, required=True,
        help="変換後(差分の目標)の画像パス / Target image path",
    )
    parser.add_argument(
        "--caption", type=str, default="",
        help="画像A/B共通のキャプション / Shared caption for image_a and image_b",
    )

    # ── ADDifT固有: タイムステップ ──────────────────────────────────────
    parser.add_argument(
        "--train_min_timesteps", type=int, default=200,
        help="学習対象タイムステップの下限 (0-1000スケール) / Minimum training timestep",
    )
    parser.add_argument(
        "--train_max_timesteps", type=int, default=400,
        help="学習対象タイムステップの上限 (0-1000スケール) / Maximum training timestep",
    )
    parser.add_argument(
        "--train_fixed_timesteps_in_batch", action="store_true",
        help="バッチ内で同一タイムステップを使用する / Use the same timestep across the batch",
    )

    # ── ADDifT固有: 学習動作 ────────────────────────────────────────────
    parser.add_argument(
        "--train_iterations", type=int, default=50,
        help="学習イテレーション数 / Number of training iterations",
    )
    parser.add_argument(
        "--diff_alt_ratio", type=float, default=1.0,
        help="逆方向ターン時のmultiplier倍率 (|diff_alt_ratio|) / Multiplier ratio for the alternate turn",
    )
    parser.add_argument(
        "--network_strength", type=float, default=5.0,
        help="LoRA適用時のmultiplierスケール (base=0.25) / Network multiplier scale (base unit = 0.25)",
    )
    parser.add_argument(
        "--diff_use_diff_mask", action="store_true",
        help="差分マスクで損失計算範囲を限定する / Restrict loss computation using a difference mask",
    )
    parser.add_argument(
        "--diff_mask_path", type=str, default=None,
        help="差分マスク画像のパス (--diff_use_diff_mask 指定時に使用) / Path to the difference mask image",
    )
    parser.add_argument(
        "--train_loss_function", type=str, default="MSE", choices=LOSS_FUNCTIONS,
        help="損失関数 / Loss function",
    )
    parser.add_argument(
        "--train_snr_gamma", type=float, default=0.0,
        help="Min-SNR-gamma (0で無効) / Min-SNR-gamma weighting (0 disables)",
    )

    # ── ネットワーク ─────────────────────────────────────────────────────
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--network_dim", type=int, default=8)
    parser.add_argument("--network_alpha", type=float, default=4.0)
    parser.add_argument("--network_dropout", type=float, default=None)
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*",
        help="追加ネットワーク引数 e.g. anima_block_lr_weight=... / anima_matrix_scales=...",
    )
    parser.add_argument("--network_train_unet_only", action="store_true")
    parser.add_argument("--training_comment", type=str, default=None)
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--unet_lr", type=float, default=None)

    # ── 保存 ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_model_as", type=str, default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
    )
    parser.add_argument("--no_metadata", action="store_true")

    # train_util.verify_training_args が要求するダミー引数
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train_util.verify_training_args(args)

    if args.output_dir is None:
        raise ValueError("--output_dir is required")
    if not Path(args.image_a).is_file():
        raise FileNotFoundError(f"--image_a not found: {args.image_a}")
    if not Path(args.image_b).is_file():
        raise FileNotFoundError(f"--image_b not found: {args.image_b}")
    if args.diff_use_diff_mask and not args.diff_mask_path:
        raise ValueError("--diff_use_diff_mask requires --diff_mask_path")
    if args.train_min_timesteps >= args.train_max_timesteps:
        raise ValueError("--train_min_timesteps must be less than --train_max_timesteps")

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    set_seed(args.seed)

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    device = accelerator.device

    # ── Load models ─────────────────────────────────────────────────────
    logger.info("Loading Qwen3 text encoder...")
    qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    qwen3_text_encoder.eval()
    qwen3_text_encoder.requires_grad_(False)

    logger.info("Loading Anima VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True,
        spatial_chunk_size=getattr(args, "vae_chunk_size", None),
        disable_cache=getattr(args, "vae_disable_cache", False),
    )
    vae.to(device, dtype=weight_dtype)
    vae.eval()

    # ── 画像ペアをlatent化 (学習中は固定値として保持) ───────────────────
    logger.info(f"Encoding image_a: {args.image_a}")
    latent_a = encode_image_to_latent(vae, args.image_a, weight_dtype, device).detach()
    logger.info(f"Encoding image_b: {args.image_b}")
    latent_b = encode_image_to_latent(vae, args.image_b, weight_dtype, device).detach()

    if latent_a.shape[-2:] != latent_b.shape[-2:]:
        raise ValueError(
            f"image_a と image_b の latent サイズが一致しません: {tuple(latent_a.shape[-2:])} vs {tuple(latent_b.shape[-2:])}"
        )

    diff_mask = None
    if args.diff_use_diff_mask:
        diff_mask = load_diff_mask(args.diff_mask_path, latent_a.shape, device, weight_dtype)

    _keep_vae = getattr(args, "sample_keep_vae", False) and getattr(args, "sample_every_n_steps", None)
    if _keep_vae:
        logger.info("sample_keep_vae=True: VAE をVRAMに保持します。")
        _vae_for_sample = vae
    else:
        vae.to("cpu")
        del vae
        _vae_for_sample = None
    clean_memory_on_device(device)

    attn_mode = getattr(args, "attn_mode", "torch") or "torch"
    if getattr(args, "xformers", False):
        attn_mode = "xformers"

    logger.info(f"Loading Anima DiT model (attn_mode={attn_mode})...")
    dit: anima_models.Anima = anima_utils.load_anima_model(
        device,
        args.pretrained_model_name_or_path,
        attn_mode,
        getattr(args, "split_attn", False),
        "cpu",
        weight_dtype,
        False,  # fp8_scaled not supported for ADDifT
    )
    dit.requires_grad_(False)
    dit.to(device, dtype=weight_dtype)
    dit.train()

    # ── Text encoding (caption 1回のみ) ────────────────────────────────
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None) or None,
        qwen3_max_length=getattr(args, "qwen3_max_token_length", 512),
        t5_max_length=getattr(args, "t5_max_token_length", 512),
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    qwen3_text_encoder.to(device, dtype=weight_dtype)
    with torch.no_grad():
        tokens = tokenize_strategy.tokenize(args.caption)
        embeds_tuple = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen3_text_encoder], tokens)
    qwen3_text_encoder.to("cpu")
    clean_memory_on_device(device)

    # ── Network ──────────────────────────────────────────────────────────
    network_module = importlib.import_module(args.network_module)
    net_kwargs = build_network_kwargs(args)

    if args.dim_from_weights:
        if args.network_weights is None:
            raise ValueError("--dim_from_weights requires --network_weights")
        network, _ = network_module.create_network_from_weights(
            1.0, args.network_weights, None, qwen3_text_encoder, dit, **net_kwargs
        )
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            qwen3_text_encoder,
            dit,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )

    network.apply_to(qwen3_text_encoder, dit, apply_text_encoder=False, apply_unet=True)
    network.set_multiplier(0.0)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        logger.info(f"Loaded network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()

    # ── Optimizer / scheduler ──────────────────────────────────────────────
    unet_lr = args.unet_lr if args.unet_lr is not None else args.learning_rate
    trainable_params, _ = network.prepare_optimizer_params(None, unet_lr, args.learning_rate)
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)
    accelerator.unwrap_model(network).prepare_grad_etc(qwen3_text_encoder, dit)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    optimizer_train_fn, _ = train_util.get_optimizer_train_eval_fn(optimizer, args)
    optimizer_train_fn()
    train_util.init_trackers(accelerator, args, "anima_addift_train")

    # ── Training loop ────────────────────────────────────────────────────
    progress_bar = tqdm(
        total=args.train_iterations,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
    global_step = 0
    base_multiplier_unit = 0.25 * args.network_strength

    while global_step < args.train_iterations:
        with accelerator.accumulate(network):
            optimizer.zero_grad(set_to_none=True)

            batch_size = latent_a.shape[0]
            noise = torch.randn_like(latent_a)

            turn = global_step % 2 == 0

            timesteps = sample_curriculum_timesteps(
                step_index=global_step,
                batch_size=batch_size,
                train_min_timesteps=args.train_min_timesteps,
                train_max_timesteps=args.train_max_timesteps,
                fixed_timesteps_in_batch=args.train_fixed_timesteps_in_batch,
                device=device,
            )
            timesteps_normalized = (timesteps.float() / 1000.0).to(device, dtype=weight_dtype)

            # turn=True:  A=無効側(reference) / B=有効側(prediction)
            # turn=False: B=無効側(reference) / A=有効側(prediction), multiplierは逆方向
            reference_latent = latent_a if turn else latent_b
            prediction_latent = latent_b if turn else latent_a

            noisy_reference = add_rectified_flow_noise(reference_latent, noise, timesteps)
            noisy_prediction = add_rectified_flow_noise(prediction_latent, noise, timesteps)

            net_unwrapped = accelerator.unwrap_model(network)

            # ── 無効側: network OFF (no_grad) ──────────────────────────
            net_unwrapped.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                reference_pred = predict_noise_anima(
                    dit, timesteps_normalized, noisy_reference, embeds_tuple, weight_dtype, device,
                ).detach()

            # ── 有効側: network ON ───────────────────────────────────────
            multiplier = base_multiplier_unit if turn else -base_multiplier_unit * abs(args.diff_alt_ratio)
            net_unwrapped.set_multiplier(multiplier)
            with accelerator.autocast():
                prediction_pred = predict_noise_anima(
                    dit, timesteps_normalized, noisy_prediction, embeds_tuple, weight_dtype, device,
                )
            net_unwrapped.set_multiplier(0.0)

            loss = compute_addift_loss(
                args, prediction_pred, reference_pred, timesteps, diff_mask,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

        if accelerator.sync_gradients:
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "network_multiplier": multiplier,
                "turn": int(turn),
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(loss=f"{logs['loss']:.4f}", refresh=False)
            progress_bar.update(1)

            if (
                args.save_every_n_steps
                and global_step % args.save_every_n_steps == 0
                and global_step < args.train_iterations
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_addift_weights(accelerator, network, args, save_dtype, global_step, last=False)

            # ── サンプル生成 ──────────────────────────────────────────────
            if (
                getattr(args, "sample_every_n_steps", None)
                and global_step % args.sample_every_n_steps == 0
                and args.sample_prompts
                and args.sample_save_dir
                and accelerator.is_main_process
            ):
                net_unwrapped = accelerator.unwrap_model(network)
                net_unwrapped.set_multiplier(1.0)
                net_unwrapped.eval()
                dit_unwrapped = accelerator.unwrap_model(dit)
                try:
                    anima_sample_gen.sample_images_from_prompts(
                        args=args,
                        dit=dit_unwrapped,
                        vae_for_sample=_vae_for_sample,
                        text_encoder=qwen3_text_encoder,
                        tokenize_strategy=tokenize_strategy,
                        text_encoding_strategy=text_encoding_strategy,
                        accelerator=accelerator,
                        epoch=None,
                        global_step=global_step,
                    )
                finally:
                    net_unwrapped.train()
                    net_unwrapped.set_multiplier(0.0)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_addift_weights(accelerator, network, args, save_dtype, global_step, last=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
