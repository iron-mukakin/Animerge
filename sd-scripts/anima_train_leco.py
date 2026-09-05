"""anima_train_leco.py - LECO training script for Anima (Cosmos-based DiT) models.

Ported from train_leco.py (SD1.x/SDXL) by replacing:
  - DDPMScheduler          -> FlowMatchEulerDiscreteScheduler (Rectified Flow)
  - encode_prompt_sd       -> AnimaTokenizeStrategy + AnimaTextEncodingStrategy
  - UNet forward signature -> Anima DiT forward signature
  - network creation       -> Anima-compatible network (supports anima_block_lr_weight / anima_matrix_scales)

Usage:
    accelerate launch --mixed_precision bf16 anima_train_leco.py \
        --pretrained_model_name_or_path anima_model.safetensors \
        --vae vae.safetensors \
        --qwen3 /path/to/qwen3 \
        --prompts_file prompts.toml \
        --output_dir output \
        --output_name my_leco \
        --network_dim 8 \
        --network_alpha 4 \
        --learning_rate 1e-4 \
        --optimizer_type AdamW8bit \
        --max_train_steps 500 \
        --max_denoising_steps 40 \
        --mixed_precision bf16 \
        --gradient_checkpointing \
        --save_every_n_steps 100
"""
from __future__ import annotations

import argparse
import importlib
import math
import random
import re
from pathlib import Path
from typing import Optional

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    compile_utils,
    custom_train_functions,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    train_util,
)
import anima_sample_gen
from library.custom_train_functions import apply_snr_weight
from library.leco_train_util import (
    PromptEmbedsCache,
    apply_noise_offset,
    build_network_kwargs,
    get_random_resolution,
    get_save_extension,
    load_prompt_settings,
    save_weights,
)
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anima-specific helpers replacing leco_train_util SD/SDXL equivalents
# ---------------------------------------------------------------------------

def encode_prompt_anima(
    tokenize_strategy: strategy_anima.AnimaTokenizeStrategy,
    text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy,
    text_encoder,
    prompt: str,
    semantic_text_encoder=None,
    layer_indices=None,
):
    """Encode a single prompt using Anima's tokenize + text encoding pipeline.

    apply_fix_025: Anima 3.8B (semantic branch) 対応。
    semantic_text_encoder / layer_indices が両方揃っている場合のみ
    Qwen3.5を含めた2モデル構成でencode_tokens()を呼ぶ。
    戻り値は常に6要素のタプルとし、semantic branch非使用時は
    末尾2要素をNoneにする(呼び出し元の分岐を単純化するため)。
    """
    use_semantic_branch = semantic_text_encoder is not None and layer_indices is not None
    models = [text_encoder, semantic_text_encoder] if use_semantic_branch else [text_encoder]

    tokens = tokenize_strategy.tokenize(prompt)
    with torch.no_grad():
        outputs = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)

    if use_semantic_branch:
        # outputs: (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
        #           semantic_hidden_states, semantic_attn_mask)
        return tuple(outputs)
    # outputs: (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = outputs
    return (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, None, None)


def _anima_forward(
    model: anima_models.Anima,
    noisy_latents: torch.Tensor,
    timesteps_normalized: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    t5_input_ids: torch.Tensor,
    t5_attn_mask: torch.Tensor,
    weight_dtype,
    device,
    semantic_hidden_states=None,
    semantic_attention_mask=None,
) -> torch.Tensor:
    """Single forward pass through the Anima DiT model."""
    bs = noisy_latents.shape[0]
    h_lat = noisy_latents.shape[-2]
    w_lat = noisy_latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_lat, w_lat, dtype=weight_dtype, device=device)

    # Anima expects 5D input: [B, C, 1, H, W]
    inp = noisy_latents.unsqueeze(2)

    # apply_fix_025: Anima 3.8B (semantic branch) 対応。
    # Anima.forward() は semantic_hidden_states を Sequence[Tensor] (層ごとの
    # リスト、各shape (B, L, D)) として受け取るため、(num_layers, B, L, D) の
    # スタック済みテンソルをここで初めてunbindする。
    semantic_kwargs = {}
    if semantic_hidden_states is not None:
        semantic_kwargs["semantic_hidden_states"] = list(
            semantic_hidden_states.to(device, dtype=weight_dtype).unbind(dim=0)
        )
        semantic_kwargs["semantic_attention_mask"] = semantic_attention_mask.to(device)

    pred = model(
        inp,
        timesteps_normalized,
        prompt_embeds.to(device, dtype=weight_dtype),
        padding_mask=padding_mask,
        target_input_ids=t5_input_ids.to(device, dtype=torch.long),
        target_attention_mask=t5_attn_mask.to(device),
        source_attention_mask=attn_mask.to(device),
        **semantic_kwargs,
    )
    # Back to 4D: [B, C, H, W]
    return pred.squeeze(2)


def diffusion_anima(
    model: anima_models.Anima,
    noise_scheduler,
    latents: torch.Tensor,
    embeds_tuple,          # (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask) batched
    total_timesteps: int,
    guidance_scale: float,
    weight_dtype,
    device,
) -> torch.Tensor:
    """Partial denoising pass to generate intermediate latents (LECO forward phase).

    Replicates leco_train_util.diffusion() but uses Anima's DiT forward signature
    and Rectified Flow timestep convention (normalized to [0, 1]).
    Classifier-free guidance is applied when guidance_scale > 1.0.
    """
    # apply_fix_025: Anima 3.8B (semantic branch) 対応。6要素タプルに拡張。
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask = embeds_tuple

    for i, t in enumerate(noise_scheduler.timesteps[:total_timesteps]):
        # Rectified Flow: timestep normalized to [0, 1]
        t_norm = (t.float() / 1000.0).to(device, dtype=weight_dtype)
        t_batch = t_norm.expand(latents.shape[0])

        if guidance_scale > 1.0:
            # CFG: concatenate unconditional (first half) and conditional (second half)
            latents_input = torch.cat([latents] * 2)
            t_input = t_batch.repeat(2)
            noise_pred = _anima_forward(
                model, latents_input, t_input,
                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                weight_dtype, device,
                semantic_hidden_states=semantic_hidden_states,
                semantic_attention_mask=semantic_attn_mask,
            )
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = _anima_forward(
                model, latents, t_batch,
                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                weight_dtype, device,
                semantic_hidden_states=semantic_hidden_states,
                semantic_attention_mask=semantic_attn_mask,
            )

        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def predict_noise_anima(
    model: anima_models.Anima,
    noise_scheduler,
    timestep,
    latents: torch.Tensor,
    embeds_tuple,
    weight_dtype,
    device,
) -> torch.Tensor:
    """Single-step noise prediction (LECO loss phase).

    Replicates leco_train_util.predict_noise() for Anima.
    guidance_scale is always 1.0 for this phase (no CFG).
    """
    # apply_fix_025: Anima 3.8B (semantic branch) 対応。6要素タプルに拡張。
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask = embeds_tuple
    t_norm = (timestep.float() / 1000.0).to(device, dtype=weight_dtype)
    t_batch = t_norm.expand(latents.shape[0])

    return _anima_forward(
        model, latents, t_batch,
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
        weight_dtype, device,
        semantic_hidden_states=semantic_hidden_states,
        semantic_attention_mask=semantic_attn_mask,
    )


def concat_embeds_anima(uncond_embeds, cond_embeds, batch_size: int):
    """Concatenate unconditional and conditional embeddings for CFG (Anima variant).

    Returns a tuple of (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
    semantic_hidden_states, semantic_attn_mask) where unconditional is first
    and conditional is second along batch dim.

    apply_fix_025: Anima 3.8B (semantic branch) 対応。
    semantic_hidden_states は (num_layers, B, L, D) でバッチ軸がdim=1のため、
    他の4テンソル(dim=0がバッチ)とは異なる軸でrepeat/catする。
    """
    pe_u, am_u, t5_u, t5am_u, sh_u, sam_u = uncond_embeds
    pe_c, am_c, t5_c, t5am_c, sh_c, sam_c = cond_embeds

    def _repeat(x):
        return x.repeat(batch_size, *([1] * (x.dim() - 1))) if x.shape[0] == 1 else x

    result = (
        torch.cat([_repeat(pe_u), _repeat(pe_c)], dim=0),
        torch.cat([_repeat(am_u), _repeat(am_c)], dim=0),
        torch.cat([_repeat(t5_u), _repeat(t5_c)], dim=0),
        torch.cat([_repeat(t5am_u), _repeat(t5am_c)], dim=0),
    )

    if sh_u is None or sh_c is None:
        return result + (None, None)

    def _repeat_semantic(x):
        # (num_layers, B, L, D): バッチ軸はdim=1
        return x.repeat(1, batch_size, *([1] * (x.dim() - 2))) if x.shape[1] == 1 else x

    semantic_hidden_states = torch.cat([_repeat_semantic(sh_u), _repeat_semantic(sh_c)], dim=1)
    semantic_attn_mask = torch.cat([_repeat(sam_u), _repeat(sam_c)], dim=0)
    return result + (semantic_hidden_states, semantic_attn_mask)


def repeat_embeds_anima(embeds, batch_size: int):
    """Repeat single-prompt embeddings to batch_size (for non-CFG prediction).

    apply_fix_025: Anima 3.8B (semantic branch) 対応。
    semantic_hidden_states は (num_layers, B, L, D) でバッチ軸がdim=1のため、
    他の4テンソル(dim=0がバッチ)とは異なる軸でrepeatする。
    """
    pe, am, t5, t5am, sh, sam = embeds

    def _repeat(x):
        return x.repeat(batch_size, *([1] * (x.dim() - 1))) if x.shape[0] == 1 else x

    result = (_repeat(pe), _repeat(am), _repeat(t5), _repeat(t5am))

    if sh is None:
        return result + (None, None)

    def _repeat_semantic(x):
        return x.repeat(1, batch_size, *([1] * (x.dim() - 2))) if x.shape[1] == 1 else x

    return result + (_repeat_semantic(sh), _repeat(sam))


def get_initial_latents_anima(
    batch_size: int,
    height: int,
    width: int,
    latent_channels: int = 16,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Generate initial random latents for Anima (FlowMatch / Rectified Flow).

    FlowMatchEulerDiscreteScheduler does not have init_noise_sigma,
    so no scaling is applied. The latent shape follows Anima's VAE:
      (batch_size, latent_channels, height // vae_scale_factor, width // vae_scale_factor)
    """
    h_lat = height // vae_scale_factor
    w_lat = width // vae_scale_factor
    return torch.randn(batch_size, latent_channels, h_lat, w_lat)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LECO training for Anima (Cosmos DiT) models")

    # Reuse Anima training arguments
    train_util.add_sd_models_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    train_util.add_training_arguments(parser, support_dreambooth=False)
    custom_train_functions.add_custom_train_arguments(parser, support_weighted_captions=False)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    add_logging_arguments(parser)

    # LECO-specific
    parser.add_argument(
        "--prompts_file", type=str, required=True,
        help="Path to LECO prompt TOML file / LECO用プロンプトTOMLファイルのパス",
    )
    parser.add_argument(
        "--max_denoising_steps", type=int, default=40,
        help="Partial denoising steps per iteration (default: 40) / 各イテレーションの部分デノイズステップ数",
    )
    parser.add_argument(
        "--leco_denoise_guidance_scale", type=float, default=3.0,
        help="CFG scale for partial denoising pass (default: 3.0) / 部分デノイズ時のCFGスケール",
    )

    # Network
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--network_dim", type=int, default=4)
    parser.add_argument("--network_alpha", type=float, default=1.0)
    parser.add_argument("--network_dropout", type=float, default=None)
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="Extra network args e.g. anima_block_lr_weight=... / anima_matrix_scales=...")
    parser.add_argument("--network_train_unet_only", action="store_true")
    parser.add_argument("--network_train_text_encoder_only", action="store_true")
    parser.add_argument("--training_comment", type=str, default=None)
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--unet_lr", type=float, default=None)

    # Save
    parser.add_argument(
        "--save_model_as", type=str, default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
    )
    parser.add_argument("--no_metadata", action="store_true")

    # Dummy args required by train_util.verify_training_args
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    # Sample generation
    # --sample_every_n_steps / --sample_prompts / --sample_save_dir は
    # train_util.add_training_arguments() で登録済みのため追加不要。
    parser.add_argument(
        "--sample_keep_vae", action="store_true",
        help=(
            "Keep VAE loaded in VRAM throughout training for sample generation. "
            "Default: reload VAE each time samples are generated, then unload."
        ),
    )

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
    if args.network_train_text_encoder_only:
        raise ValueError("LECO does not support text encoder LoRA training")
    if args.compile:
        if args.torch_compile:
            raise ValueError("--compile and --torch_compile cannot be used together")
        if args.compile_fullgraph and args.split_attn:
            raise ValueError("--compile_fullgraph cannot be used with --split_attn")

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32 - 1)
    set_seed(args.seed)

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    device = accelerator.device

    # ── Prompt settings ────────────────────────────────────────────────────
    prompt_settings = load_prompt_settings(args.prompts_file)
    logger.info(f"Loaded {len(prompt_settings)} LECO prompt settings from {args.prompts_file}")

    # ── Load models ────────────────────────────────────────────────────────
    logger.info("Loading Qwen3 text encoder...")
    qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=weight_dtype, device="cpu"
    )
    qwen3_text_encoder.eval()
    qwen3_text_encoder.requires_grad_(False)

    # apply_fix_025: Anima 3.8B (semantic branch) 対応。
    # apply_fix_039: Anima 3.8B v1.1(Semantic Connector v2内蔵)対応
    # (anima_train_network.pyのapply_fix_031と同一仕様)。
    # --progressive_adapter_path が指定されている場合、またはDiT checkpoint自体が
    # v1.1(Semantic Connector v2内蔵)の場合にQwen3.5をロードし、layer_indicesを検出する。
    # どちらでもない場合は従来通りsemantic branch無効。
    qwen35_text_encoder = None
    layer_indices = None
    progressive_adapter_path = getattr(args, "progressive_adapter_path", None)
    is_semantic_connector_v2 = anima_utils.detect_semantic_connector_v2_architecture(
        args.pretrained_model_name_or_path
    )
    if (progressive_adapter_path or is_semantic_connector_v2) and not getattr(args, "qwen35", None):
        raise ValueError(
            "--progressive_adapter_path、またはv1.1(Semantic Connector v2内蔵)の"
            "DiT checkpointを指定する場合は --qwen35 も指定してください。"
            " / --qwen35 is required when --progressive_adapter_path is set, "
            "or when the DiT checkpoint bundles Semantic Connector v2 (Anima 3.8B v1.1)."
        )
    if progressive_adapter_path or is_semantic_connector_v2:
        logger.info("Loading Qwen3.5 text encoder...")
        qwen35_text_encoder, _ = anima_utils.load_qwen35_text_encoder(
            args.qwen35, dtype=weight_dtype, device="cpu"
        )
        qwen35_text_encoder.eval()
        qwen35_text_encoder.requires_grad_(False)
        if progressive_adapter_path:
            # v1.0: 外部Progressive Cross Adapter checkpointのmetadataから検出。
            # (v1.1との併用は禁止されており、両方指定時はload_anima_model内で
            # RuntimeErrorとなる。)
            _, layer_indices = anima_utils.detect_progressive_adapter_architecture(progressive_adapter_path)
        else:
            # v1.1: DiT checkpoint自体にSemantic Connector v2が内蔵されているため、
            # そちらのmetadataから検出する。
            connector_config = anima_utils.detect_anima_v2_connector_config(
                args.pretrained_model_name_or_path
            )
            layer_indices = connector_config["layer_indices"]

    logger.info("Loading Anima VAE...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(weight_dtype)
    vae.eval()

    # VAE 保持オプション: keep_vae=True のときはVRAMに残す
    _keep_vae = getattr(args, "sample_keep_vae", False) and getattr(args, "sample_every_n_steps", None)
    if _keep_vae:
        logger.info("sample_keep_vae=True: VAE をVRAMに保持します。")
        _vae_for_sample = vae
    else:
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
        False,  # fp8_scaled not supported for LECO
        # apply_fix_025: Anima 3.8B (semantic branch) 対応。
        progressive_adapter_path=progressive_adapter_path,
    )
    dit.requires_grad_(False)
    dit.to(device, dtype=weight_dtype)
    dit.train()

    # ── Text encoding ──────────────────────────────────────────────────────
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None) or None,
        qwen3_max_length=getattr(args, "qwen3_max_token_length", 512),
        t5_max_length=getattr(args, "t5_max_token_length", 512),
        # apply_fix_025: Anima 3.8B (semantic branch) 対応。
        qwen35_path=getattr(args, "qwen35", None),
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy(layer_indices=layer_indices)

    qwen3_text_encoder.to(device, dtype=weight_dtype)
    # apply_fix_025: Anima 3.8B (semantic branch) 対応。
    # Qwen3.5もプロンプト事前計算の間だけGPUへ載せ、完了後CPUへ退避する
    # (LECOの既存設計:起動時に一度だけロード→事前計算→CPUへ退避、を踏襲)。
    if qwen35_text_encoder is not None:
        qwen35_text_encoder.to(device, dtype=weight_dtype)
    prompt_cache = PromptEmbedsCache()
    unique_prompts = sorted({
        prompt
        for setting in prompt_settings
        for prompt in (setting.target, setting.positive, setting.unconditional, setting.neutral)
    })
    with torch.no_grad():
        for prompt in unique_prompts:
            prompt_cache[prompt] = encode_prompt_anima(
                tokenize_strategy, text_encoding_strategy, qwen3_text_encoder, prompt,
                semantic_text_encoder=qwen35_text_encoder,
                layer_indices=layer_indices,
            )
    qwen3_text_encoder.to("cpu")
    if qwen35_text_encoder is not None:
        qwen35_text_encoder.to("cpu")
    clean_memory_on_device(device)

    # ── Noise scheduler (Rectified Flow) ───────────────────────────────────
    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=getattr(args, "discrete_flow_shift", 1.0),
    )

    # ── Network ────────────────────────────────────────────────────────────
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

    compile_utils.apply_cuda_optimizations(args)
    if args.compile:
        compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=False)

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
    train_util.init_trackers(accelerator, args, "anima_leco_train")

    # ── Training loop ──────────────────────────────────────────────────────
    progress_bar = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
    global_step = 0

    while global_step < args.max_train_steps:
        with accelerator.accumulate(network):
            optimizer.zero_grad(set_to_none=True)

            setting = prompt_settings[torch.randint(0, len(prompt_settings), (1,)).item()]
            noise_scheduler.set_timesteps(args.max_denoising_steps, device=device)

            timesteps_to = torch.randint(
                1, args.max_denoising_steps, (1,), device=device
            ).item()
            height, width = get_random_resolution(setting)

            latents = get_initial_latents_anima(
                setting.batch_size, height, width,
                latent_channels=16, vae_scale_factor=8,
            ).to(device, dtype=weight_dtype)
            latents = apply_noise_offset(latents, getattr(args, "noise_offset", None))

            net_unwrapped = accelerator.unwrap_model(network)

            # ── Phase 1: partial denoising with LoRA active ───────────────
            # torch.no_grad() で包み 40step 分の計算グラフ蓄積を防止する。
            # grad が必要なのは Phase 3 の 1forward のみ。
            net_unwrapped.set_multiplier(setting.multiplier)
            with torch.no_grad(), accelerator.autocast():
                denoised_latents = diffusion_anima(
                    dit,
                    noise_scheduler,
                    latents,
                    concat_embeds_anima(
                        prompt_cache[setting.unconditional],
                        prompt_cache[setting.target],
                        setting.batch_size,
                    ),
                    total_timesteps=timesteps_to,
                    guidance_scale=args.leco_denoise_guidance_scale,
                    weight_dtype=weight_dtype,
                    device=device,
                ).detach()  # 計算グラフを明示的に切断

            # Determine evaluation timestep
            noise_scheduler.set_timesteps(1000, device=device)
            current_timestep_index = int(timesteps_to * 1000 / args.max_denoising_steps)
            current_timestep = noise_scheduler.timesteps[current_timestep_index]

            # ── Phase 2: noise predictions (LoRA OFF for reference passes) ─
            net_unwrapped.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                positive_latents = predict_noise_anima(
                    dit, noise_scheduler, current_timestep, denoised_latents,
                    repeat_embeds_anima(prompt_cache[setting.positive], setting.batch_size),
                    weight_dtype, device,
                )
                neutral_latents = predict_noise_anima(
                    dit, noise_scheduler, current_timestep, denoised_latents,
                    repeat_embeds_anima(prompt_cache[setting.neutral], setting.batch_size),
                    weight_dtype, device,
                )
                unconditional_latents = predict_noise_anima(
                    dit, noise_scheduler, current_timestep, denoised_latents,
                    repeat_embeds_anima(prompt_cache[setting.unconditional], setting.batch_size),
                    weight_dtype, device,
                )

            # ── Phase 3: LoRA ON prediction + loss ────────────────────────
            net_unwrapped.set_multiplier(setting.multiplier)
            with accelerator.autocast():
                target_latents = predict_noise_anima(
                    dit, noise_scheduler, current_timestep, denoised_latents,
                    repeat_embeds_anima(prompt_cache[setting.target], setting.batch_size),
                    weight_dtype, device,
                )

                target = setting.build_target(positive_latents, neutral_latents, unconditional_latents)
                loss = torch.nn.functional.mse_loss(
                    target_latents.float(), target.float(), reduction="none"
                )
                loss = loss.mean(dim=(1, 2, 3))
                if args.min_snr_gamma is not None and args.min_snr_gamma > 0:
                    timesteps_tensor = torch.full(
                        (loss.shape[0],), current_timestep_index,
                        device=loss.device, dtype=torch.long,
                    )
                    loss = apply_snr_weight(
                        loss, timesteps_tensor, noise_scheduler,
                        args.min_snr_gamma, args.v_parameterization,
                    )
                loss = loss.mean() * setting.weight

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

        if accelerator.sync_gradients:
            global_step += 1
            net_unwrapped = accelerator.unwrap_model(network)
            net_unwrapped.set_multiplier(0.0)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "guidance_scale": setting.guidance_scale,
                "network_multiplier": setting.multiplier,
            }
            accelerator.log(logs, step=global_step)
            # set_postfix(refresh=False) で描画を抑制し update(1) の1回で出力する
            # （別々に呼ぶと tqdm が2行出力しログに重複が生じるため）
            progress_bar.set_postfix(loss=f"{logs['loss']:.4f}", refresh=False)
            progress_bar.update(1)

            if (
                args.save_every_n_steps
                and global_step % args.save_every_n_steps == 0
                and global_step < args.max_train_steps
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_weights(
                        accelerator, network, args, save_dtype,
                        prompt_settings, global_step, last=False,
                    )

            # ── サンプル生成 ──────────────────────────────────────────────
            if (
                getattr(args, "sample_every_n_steps", None)
                and global_step % args.sample_every_n_steps == 0
                and args.sample_prompts
                and args.sample_save_dir
                and accelerator.is_main_process
            ):
                # LECO: LoRA を推論モードへ切り替えてサンプル生成
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
                        # apply_fix_025: Anima 3.8B (semantic branch) 対応。
                        semantic_text_encoder=qwen35_text_encoder,
                        layer_indices=layer_indices,
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
        save_weights(
            accelerator, network, args, save_dtype,
            prompt_settings, global_step, last=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
