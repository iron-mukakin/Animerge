"""anima_sample_gen.py - Anima (Cosmos系) モデル用サンプル生成モジュール

学習中に学習済みLoRAを適用した状態でサンプル画像を生成する。
参照実装: anima_minimal_inference.py の prepare_text_inputs / generate_body

推論経路（anima_minimal_inference.py と同一）:
  tokenize → encode_tokens
  → _preprocess_text_embeds(source, target_input_ids, target_attn_mask, source_attn_mask)
  → crossattn_emb[~t5_attn_mask.bool()] = 0   # T5マスクでゼロ埋め
  → embed[0] = crossattn_emb                   # 前処理済みに上書き
  → anima(latents, t, embed[0], padding_mask)  # target_input_ids=None → adapter不通過
  → vae.decode_to_pixels → PIL.Image

LoRA ネットワークは呼び出し元で既にDiTにフックされた状態で渡す。
このモジュールは set_multiplier / eval / train を呼ばない
（LoRA側は train_network.py のフック機構、LECO側は呼び出し元で制御する）。
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from library import anima_models, qwen_image_autoencoder_kl, train_util
from library.device_utils import clean_memory_on_device, synchronize_device

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# テキストエンコード + _preprocess_text_embeds
# ---------------------------------------------------------------------------

def encode_prompt_for_sample(
    prompt: str,
    tokenize_strategy,
    text_encoding_strategy,
    text_encoder: torch.nn.Module,
    dit: anima_models.Anima,
    device: torch.device,
    dtype: torch.dtype,
):
    """プロンプトをエンコードし _preprocess_text_embeds を適用して返す。

    Returns
    -------
    crossattn_emb : torch.Tensor  shape (1, N, D)  前処理済みテンソル
    """
    tokens = tokenize_strategy.tokenize(prompt)
    # encode_tokens returns [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
    with torch.no_grad():
        embed = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = embed

    # numpy -> tensor (キャッシュから来た場合)
    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds)
    if isinstance(attn_mask, np.ndarray):
        attn_mask = torch.from_numpy(attn_mask)
    if isinstance(t5_input_ids, np.ndarray):
        t5_input_ids = torch.from_numpy(t5_input_ids)
    if isinstance(t5_attn_mask, np.ndarray):
        t5_attn_mask = torch.from_numpy(t5_attn_mask)

    # batch dim 付与（tokenize が単文字列の場合バッチ次元がない場合がある）
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
        attn_mask     = attn_mask.unsqueeze(0)
        t5_input_ids  = t5_input_ids.unsqueeze(0)
        t5_attn_mask  = t5_attn_mask.unsqueeze(0)

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    attn_mask     = attn_mask.to(device)
    t5_input_ids  = t5_input_ids.to(device)
    t5_attn_mask  = t5_attn_mask.to(device)

    # --- 診断ログ2: adapter 通過前 Qwen3 raw output ---
    qwen3_norm     = float(prompt_embeds.norm().item())
    qwen3_mean     = float(prompt_embeds.abs().mean().item())
    qwen3_nonzero  = int(attn_mask.sum().item())
    t5_nonzero_pre = int(t5_attn_mask.sum().item())
    t5_ids_head    = t5_input_ids[0, :t5_nonzero_pre].tolist()
    prompt_short   = prompt[:40].replace("\n", " ")
    logger.info(
        f"[DIAG2] PRE-adapter prompt='{prompt_short}' "
        f"qwen3_nonzero={qwen3_nonzero}/{attn_mask.numel()} "
        f"qwen3_norm={qwen3_norm:.4f} qwen3_mean={qwen3_mean:.6f} "
        f"t5_nonzero={t5_nonzero_pre} t5_ids={t5_ids_head}"
    )
    # --- 診断ログ2ここまで ---

    # _preprocess_text_embeds: LLM adapter を通して T5 空間へ変換
    with torch.no_grad():
        crossattn_emb = dit._preprocess_text_embeds(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        # T5 attn mask でパディング位置をゼロ埋め（anima_minimal_inference.py と同一処理）
        crossattn_emb[~t5_attn_mask.bool()] = 0

    # --- 診断ログ ---
    nonzero_t5 = int(t5_attn_mask.sum().item())
    emb_norm = float(crossattn_emb.norm().item())
    emb_mean = float(crossattn_emb.abs().mean().item())
    # adapter 通過後の有効行（t5非ゼロ行）のみの norm
    valid_rows = crossattn_emb[t5_attn_mask.bool()]  # shape (nonzero_t5, D)
    valid_norm = float(valid_rows.norm().item()) if valid_rows.numel() > 0 else 0.0
    logger.info(
        f"[DIAG] encode_prompt: t5_nonzero={nonzero_t5}/{t5_attn_mask.numel()}, "
        f"crossattn_emb norm={emb_norm:.4f} valid_rows_norm={valid_norm:.4f} "
        f"mean_abs={emb_mean:.6f} shape={tuple(crossattn_emb.shape)}"
    )
    # --- 診断ログここまで ---

    return crossattn_emb  # (1, N, D)


# ---------------------------------------------------------------------------
# デノイズループ（anima_minimal_inference.generate_body と同一経路）
# ---------------------------------------------------------------------------

def _denoise(
    dit: anima_models.Anima,
    crossattn_emb: torch.Tensor,
    neg_crossattn_emb: Optional[torch.Tensor],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Euler ステップによるデノイズ。anima_minimal_inference.generate_body の経路に準拠。

    Returns
    -------
    latents : torch.Tensor  shape (1, 16, 1, H/8, W/8)
    """
    # 初期ノイズ生成
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)
    shape = (1, anima_models.Anima.LATENT_CHANNELS, 1, height // 8, width // 8)
    latents = torch.randn(shape, generator=seed_g, device="cpu", dtype=torch.float32)
    latents = latents.to(device, dtype=dtype)

    # パディングマスク
    padding_mask = torch.zeros(1, 1, height // 8, width // 8, dtype=dtype, device=device)

    # シグマスケジュール（do_sample と同一式）
    # sigmas: steps+1 個の値 (1.0 → 0.0)、flow_shift を適用
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    flow_shift_f = float(flow_shift)
    if flow_shift_f != 1.0:
        sigmas = (sigmas * flow_shift_f) / (1.0 + (flow_shift_f - 1.0) * sigmas)

    do_cfg = guidance_scale != 1.0 and neg_crossattn_emb is not None

    logger.info(
        f"[DIAG] _denoise start: latents norm={float(latents.norm().item()):.4f} "
        f"shape={tuple(latents.shape)} sigmas[0]={float(sigmas[0]):.4f} sigmas[-1]={float(sigmas[-1]):.4f} "
        f"do_cfg={do_cfg} guidance_scale={guidance_scale} flow_shift={flow_shift}"
    )
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Sampling", leave=False):
            # t = sigma 値（0〜1）をそのまま渡す（do_sample と同一）
            t = sigmas[i].unsqueeze(0).to(dtype)

            noise_pred = dit(latents, t, crossattn_emb, padding_mask=padding_mask)
            # target_input_ids を渡さない → forward 内の _preprocess_text_embeds は素通り

            if do_cfg:
                uncond_pred = dit(latents, t, neg_crossattn_emb, padding_mask=padding_mask)
                # 診断ログ（最初と最後のステップのみ）: CFG前の cond/uncond を記録
                if i == 0 or i == steps - 1:
                    cond_norm_pre  = float(noise_pred.norm().item())
                    uncond_norm_pre = float(uncond_pred.norm().item())
                    cfg_diff = float((noise_pred - uncond_pred).norm().item())
                    logger.info(
                        f"[DIAG2] step {i}: t={float(t.item()):.4f} "
                        f"cond_norm={cond_norm_pre:.4f} uncond_norm={uncond_norm_pre:.4f} "
                        f"cond-uncond_norm={cfg_diff:.4f} (CFG前)"
                    )
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            noise_pred = noise_pred.float()

            # 診断ログ（最初と最後のステップのみ）
            if i == 0 or i == steps - 1:
                logger.info(
                    f"[DIAG] step {i}: t={float(t.item()):.4f} "
                    f"noise_pred norm={float(noise_pred.norm().item()):.4f} "
                    f"latents norm={float(latents.norm().item()):.4f}"
                )

            # Euler ステップ: x = x + (sigma_{i+1} - sigma_i) * noise_pred
            # sigma は単調減少なので dt < 0 → x = x - (sigma_i - sigma_{i+1}) * noise_pred
            dt = sigmas[i + 1] - sigmas[i]
            latents = (latents.float() + noise_pred * dt).to(dtype)

    logger.info(f"[DIAG] _denoise end: latents norm={float(latents.norm().item()):.4f}")
    return latents


# ---------------------------------------------------------------------------
# VAE デコード
# ---------------------------------------------------------------------------

def _decode_latents(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latents: torch.Tensor,
    device: torch.device,
) -> Image.Image:
    """latents (1,16,1,H/8,W/8) → PIL Image（anima_train_utils と同一処理）"""
    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.to(device, dtype=vae.dtype))

    # pixels: (B, C, H, W) または (B, C, F, H, W)
    image = pixels.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    # temporal dim が残っている場合: (C, F, H, W) → (C, H, W)
    if image.ndim == 4:
        image = image[:, 0, :, :]
    img_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# 単一プロンプトのサンプル生成
# ---------------------------------------------------------------------------

def generate_sample(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
    dit: anima_models.Anima,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    text_encoder: torch.nn.Module,
    tokenize_strategy,
    text_encoding_strategy,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    """1プロンプトのサンプル画像を生成して返す。

    dit は呼び出し元で LoRA が適用された状態（または未適用）で渡す。
    このメソッド内で set_multiplier / eval / train は呼ばない。
    """
    height = max(64, height - height % 16)
    width  = max(64, width  - width  % 16)

    logger.info(
        f"[SampleGen] prompt='{prompt}', neg='{negative_prompt}', "
        f"{width}x{height}, steps={steps}, scale={guidance_scale}, "
        f"flow_shift={flow_shift}, seed={seed}"
    )

    # テキストエンコード（TEをデバイスへ）
    org_te_device = text_encoder.device
    text_encoder.to(device, dtype=dtype)
    try:
        crossattn_emb = encode_prompt_for_sample(
            prompt, tokenize_strategy, text_encoding_strategy,
            text_encoder, dit, device, dtype,
        )
        neg_crossattn_emb = None
        if guidance_scale != 1.0 and negative_prompt:
            neg_crossattn_emb = encode_prompt_for_sample(
                negative_prompt, tokenize_strategy, text_encoding_strategy,
                text_encoder, dit, device, dtype,
            )
    finally:
        text_encoder.to(org_te_device)
    clean_memory_on_device(device)

    # デノイズ
    latents = _denoise(
        dit=dit,
        crossattn_emb=crossattn_emb,
        neg_crossattn_emb=neg_crossattn_emb,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    # VAE デコード（VAEをデバイスへ、デコード後CPUへ戻す）
    org_vae_device = vae.device
    try:
        pil_img = _decode_latents(vae, latents, device)
    finally:
        vae.to(org_vae_device)
    clean_memory_on_device(device)

    return pil_img


# ---------------------------------------------------------------------------
# プロンプトファイルのパース
# ---------------------------------------------------------------------------

def parse_sample_prompt_file(prompt_file: str) -> list[dict]:
    """_sample_prompt.txt を読み込み prompt_dict のリストを返す。

    書式（train_util.line_to_prompt_dict と同一）:
      <prompt> --w W --h H --s S --l L --fs FS --d D [--n <neg>]

    Returns
    -------
    list of dict with keys:
        prompt, negative_prompt, width, height, sample_steps,
        scale, flow_shift, seed, enum
    """
    import re

    path = Path(prompt_file)
    if not path.is_file():
        logger.error(f"[SampleGen] プロンプトファイルが見つかりません: {prompt_file}")
        return []

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    results = []
    for idx, line in enumerate(lines):
        d: dict = {"enum": idx}

        # --n を最初に除去（スペースを含む値の可能性があるため専用処理）
        neg_m = re.search(r"\s+--n\s+(.+?)(?=\s+--[a-zA-Z]|\s*$)", line)
        neg = ""
        if neg_m:
            neg = neg_m.group(1).strip()
            line = line[:neg_m.start()] + line[neg_m.end():]
        d["negative_prompt"] = neg

        # プロンプト本体の終端（最初の --flag の手前）
        flag_names = ["--w", "--h", "--s", "--l", "--fs", "--d"]
        prompt_end = len(line)
        for flag in flag_names:
            m = re.search(re.escape(f" {flag}") + r"(?=\s|$)", line)
            if m:
                prompt_end = min(prompt_end, m.start())
        d["prompt"] = line[:prompt_end].strip()

        # 各フラグの値を抽出
        remainder = line[prompt_end:]
        flag_map = {
            "--w":  ("width",        int,   512),
            "--h":  ("height",       int,   512),
            "--s":  ("sample_steps", int,   20),
            "--l":  ("scale",        float, 7.5),
            "--fs": ("flow_shift",   float, 3.0),
            "--d":  ("seed",         int,   42),
        }
        for flag, (key, cast, default) in flag_map.items():
            m = re.search(re.escape(flag) + r"\s+(\S+)", remainder)
            if m:
                try:
                    d[key] = cast(m.group(1))
                except (ValueError, TypeError):
                    d[key] = default
            else:
                d[key] = default

        results.append(d)

    return results


# ---------------------------------------------------------------------------
# 学習ループからの呼び出しエントリポイント（LoRA / LECO 共通）
# ---------------------------------------------------------------------------

def sample_images_from_prompts(
    *,
    args,
    dit: anima_models.Anima,
    vae: Optional[qwen_image_autoencoder_kl.AutoencoderKLQwenImage] = None,
    text_encoder: torch.nn.Module,
    tokenize_strategy,
    text_encoding_strategy,
    accelerator,
    epoch: Optional[int],
    global_step: int,
    vae_for_sample: Optional[qwen_image_autoencoder_kl.AutoencoderKLQwenImage] = None,
) -> None:
    """学習中のサンプル生成エントリポイント。

    LoRA側（anima_train_network.py）:
        network は accelerator でフック済みのため、unwrap した dit を渡せばLoRA適用状態。

    LECO側（anima_train_leco.py）:
        呼び出し前に net_unwrapped.set_multiplier(1.0) / eval() を実施し、
        呼び出し後に net_unwrapped.train() / set_multiplier(0.0) を戻すこと。
        sample_keep_vae=False の場合は vae=None / vae_for_sample=None で呼ぶ。
        その場合、args.vae パスから内部で都度ロード・アンロードする。

    Parameters
    ----------
    args : argparse.Namespace
        sample_prompts, sample_save_dir, output_name, vae を参照する。
    dit : Anima
        unwrap 済みまたは LoRA フック済みモデル。
    vae : AutoencoderKLQwenImage or None
        LoRA 側では常駐 VAE を渡す。None の場合は args.vae から都度ロードする。
    text_encoder : torch.nn.Module
        Qwen3 テキストエンコーダ（CPU上でよい）。
    tokenize_strategy / text_encoding_strategy
        AnimaTokenizeStrategy / AnimaTextEncodingStrategy インスタンス。
    accelerator : Accelerator
    epoch : int or None
        エポック番号（ファイル名に使用）。None の場合はステップ番号を使用。
    global_step : int
        グローバルステップ数（ファイル名・条件判定に使用）。
    vae_for_sample : AutoencoderKLQwenImage or None
        LECO keep_vae=True 用の再利用 VAE。指定時は vae より優先する。
    """
    if not getattr(args, "sample_prompts", None):
        return
    if not getattr(args, "sample_save_dir", None):
        return

    # ── 間引き条件判定（train_util.sample_images と同一ロジック）──────────────
    # global_step=0 かつ sample_at_first=False なら学習前サンプルをスキップ
    if global_step == 0:
        if not getattr(args, "sample_at_first", False):
            return
    else:
        sample_every_n_epochs = getattr(args, "sample_every_n_epochs", None)
        sample_every_n_steps  = getattr(args, "sample_every_n_steps", None)
        if sample_every_n_epochs is None and sample_every_n_steps is None:
            return
        if sample_every_n_epochs is not None:
            # epoch ベース: epoch=None（ステップ呼び出し）は対象外
            if epoch is None or epoch % sample_every_n_epochs != 0:
                return
        else:
            # step ベース: epoch が指定されている（エポック末尾呼び出し）は対象外
            if global_step % sample_every_n_steps != 0 or epoch is not None:
                return
    # ───────────────────────────────────────────────────────────────────────────

    prompt_dicts = parse_sample_prompt_file(args.sample_prompts)
    if not prompt_dicts:
        logger.warning("[SampleGen] 有効なプロンプト行がありません。スキップします。")
        return

    save_dir = Path(args.sample_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = accelerator.device
    dtype  = dit.dtype

    # VAE の選択: vae_for_sample > vae > args.vae からロード
    _vae_loaded_here = False
    if vae_for_sample is not None:
        active_vae = vae_for_sample
    elif vae is not None:
        active_vae = vae
    else:
        # vae=None かつ vae_for_sample=None → args.vae パスから都度ロード
        vae_path = getattr(args, "vae", None)
        if not vae_path:
            logger.error("[SampleGen] VAE が指定されていません（args.vae が空）。スキップします。")
            return
        logger.info(f"[SampleGen] VAE を一時ロードします: {vae_path}")
        active_vae = qwen_image_autoencoder_kl.load_vae(
            vae_path,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
        )
        active_vae.to(dtype)
        active_vae.eval()
        _vae_loaded_here = True

    # dit を推論モードへ（LoRA フックは維持）
    dit_was_training = dit.training
    dit.eval()
    if hasattr(dit, "switch_block_swap_for_inference"):
        dit.switch_block_swap_for_inference()

    # RNG 保存
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
    except Exception:
        pass

    try:
        with accelerator.autocast():
            for pd in prompt_dicts:
                try:
                    if hasattr(dit, "prepare_block_swap_before_forward"):
                        dit.prepare_block_swap_before_forward()

                    pil_img = generate_sample(
                        prompt=pd["prompt"],
                        negative_prompt=pd["negative_prompt"],
                        width=pd["width"],
                        height=pd["height"],
                        steps=pd["sample_steps"],
                        guidance_scale=pd["scale"],
                        flow_shift=pd["flow_shift"],
                        seed=pd["seed"],
                        dit=dit,
                        vae=active_vae,
                        text_encoder=text_encoder,
                        tokenize_strategy=tokenize_strategy,
                        text_encoding_strategy=text_encoding_strategy,
                        device=device,
                        dtype=dtype,
                    )

                    # ファイル名生成
                    ts_str    = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
                    prefix     = (args.output_name + "_") if getattr(args, "output_name", None) else ""
                    seed_suffix = f"_{pd['seed']}"
                    fname = f"{prefix}{num_suffix}_{pd['enum']:02d}_{ts_str}{seed_suffix}.png"
                    pil_img.save(str(save_dir / fname))
                    logger.info(f"[SampleGen] saved: {save_dir / fname}")

                except Exception as exc:
                    logger.warning(f"[SampleGen] プロンプト {pd['enum']} 生成失敗: {exc}", exc_info=True)

    finally:
        # RNG 復元
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            try:
                torch.cuda.set_rng_state(cuda_rng_state)
            except Exception:
                pass

        # dit を学習モードへ戻す
        if dit_was_training:
            dit.train()
        if hasattr(dit, "switch_block_swap_for_training"):
            dit.switch_block_swap_for_training()

        # 一時ロードした VAE をアンロード
        if _vae_loaded_here:
            active_vae.to("cpu")
            del active_vae
            logger.info("[SampleGen] 一時ロードした VAE をアンロードしました。")

    clean_memory_on_device(device)
