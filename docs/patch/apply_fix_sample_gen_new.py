"""
apply_fix_sample_gen_new.py
---------------------------
anima_sample_gen.py を sd-scripts/ に配置する。

実行場所: プロジェクトルート
  python apply_fix_sample_gen_new.py
"""

import shutil
import sys
from pathlib import Path


TARGET = Path("sd-scripts") / "anima_sample_gen.py"
SOURCE = Path("anima_sample_gen.py")   # このスクリプトと同階層に置くか、フルパスを指定


CONTENT = '''\
"""anima_sample_gen.py - Anima (Cosmos\u7cfb) \u30e2\u30c7\u30eb\u7528\u30b5\u30f3\u30d7\u30eb\u751f\u6210\u30e2\u30b8\u30e5\u30fc\u30eb

\u5b66\u7fd2\u4e2d\u306b\u5b66\u7fd2\u6e08\u307fLoRA\u3092\u9069\u7528\u3057\u305f\u72b6\u614b\u3067\u30b5\u30f3\u30d7\u30eb\u753b\u50cf\u3092\u751f\u6210\u3059\u308b\u3002
\u53c2\u7167\u5b9f\u88c5: anima_minimal_inference.py \u306e prepare_text_inputs / generate_body

\u63a8\u8ad6\u7d4c\u8def\uff08anima_minimal_inference.py \u3068\u540c\u4e00\uff09:
  tokenize \u2192 encode_tokens
  \u2192 _preprocess_text_embeds(source, target_input_ids, target_attn_mask, source_attn_mask)
  \u2192 crossattn_emb[~t5_attn_mask.bool()] = 0   # T5\u30de\u30b9\u30af\u3067\u30bc\u30ed\u57cb\u3081
  \u2192 embed[0] = crossattn_emb                   # \u524d\u51e6\u7406\u6e08\u307f\u306b\u4e0a\u66f8\u304d
  \u2192 anima(latents, t, embed[0], padding_mask)  # target_input_ids=None \u2192 adapter\u4e0d\u901a\u904e
  \u2192 vae.decode_to_pixels \u2192 PIL.Image

LoRA \u30cd\u30c3\u30c8\u30ef\u30fc\u30af\u306f\u547c\u3073\u51fa\u3057\u5143\u3067\u65e2\u306bDiT\u306b\u30d5\u30c3\u30af\u3055\u308c\u305f\u72b6\u614b\u3067\u6e21\u3059\u3002
\u3053\u306e\u30e2\u30b8\u30e5\u30fc\u30eb\u306f set_multiplier / eval / train \u3092\u547c\u3070\u306a\u3044
\uff08LoRA\u5074\u306f train_network.py \u306e\u30d5\u30c3\u30af\u6a5f\u69cb\u3001LECO\u5074\u306f\u547c\u3073\u51fa\u3057\u5143\u3067\u5236\u5fa1\u3059\u308b\uff09\u3002
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

from library import anima_models, hunyuan_image_utils, qwen_image_autoencoder_kl, train_util
from library.device_utils import clean_memory_on_device, synchronize_device

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# \u30c6\u30ad\u30b9\u30c8\u30a8\u30f3\u30b3\u30fc\u30c9 + _preprocess_text_embeds
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
    """prompt \u3092\u30a8\u30f3\u30b3\u30fc\u30c9\u3057 _preprocess_text_embeds \u3092\u9069\u7528\u3057\u3066\u8fd4\u3059\u3002

    Returns
    -------
    crossattn_emb : torch.Tensor  shape (1, N, D)  \u524d\u51e6\u7406\u6e08\u307f\u30c6\u30f3\u30bd\u30eb
    """
    tokens = tokenize_strategy.tokenize(prompt)
    with torch.no_grad():
        embed = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = embed

    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds)
    if isinstance(attn_mask, np.ndarray):
        attn_mask = torch.from_numpy(attn_mask)
    if isinstance(t5_input_ids, np.ndarray):
        t5_input_ids = torch.from_numpy(t5_input_ids)
    if isinstance(t5_attn_mask, np.ndarray):
        t5_attn_mask = torch.from_numpy(t5_attn_mask)

    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
        attn_mask     = attn_mask.unsqueeze(0)
        t5_input_ids  = t5_input_ids.unsqueeze(0)
        t5_attn_mask  = t5_attn_mask.unsqueeze(0)

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    attn_mask     = attn_mask.to(device)
    t5_input_ids  = t5_input_ids.to(device)
    t5_attn_mask  = t5_attn_mask.to(device)

    with torch.no_grad():
        crossattn_emb = dit._preprocess_text_embeds(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0

    return crossattn_emb


# ---------------------------------------------------------------------------
# \u30c7\u30ce\u30a4\u30ba\u30eb\u30fc\u30d7
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
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)
    shape = (1, anima_models.Anima.LATENT_CHANNELS, 1, height // 8, width // 8)
    latents = torch.randn(shape, generator=seed_g, device="cpu", dtype=torch.float32)
    latents = latents.to(device, dtype=dtype)

    padding_mask = torch.zeros(1, 1, height // 8, width // 8, dtype=dtype, device=device)

    timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(steps, flow_shift, device)
    timesteps = timesteps.to(device, dtype=dtype)
    sigmas = sigmas.to(device)

    do_cfg = guidance_scale != 1.0 and neg_crossattn_emb is not None

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_expand = t.expand(latents.shape[0])
            noise_pred = dit(latents, t_expand, crossattn_emb, padding_mask=padding_mask)
            if do_cfg:
                uncond_pred = dit(latents, t_expand, neg_crossattn_emb, padding_mask=padding_mask)
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(dtype)

    return latents


# ---------------------------------------------------------------------------
# VAE \u30c7\u30b3\u30fc\u30c9
# ---------------------------------------------------------------------------

def _decode_latents(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latents: torch.Tensor,
    device: torch.device,
) -> Image.Image:
    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.to(device, dtype=vae.dtype))
    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)
    img = pixels[0].float().clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# \u5358\u4e00\u30d7\u30ed\u30f3\u30d7\u30c8\u306e\u30b5\u30f3\u30d7\u30eb\u751f\u6210
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
    height = max(64, height - height % 16)
    width  = max(64, width  - width  % 16)

    logger.info(
        f"[SampleGen] prompt=\\'{prompt}\\', neg=\\'{negative_prompt}\\', "
        f"{width}x{height}, steps={steps}, scale={guidance_scale}, "
        f"flow_shift={flow_shift}, seed={seed}"
    )

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

    org_vae_device = vae.device
    try:
        pil_img = _decode_latents(vae, latents, device)
    finally:
        vae.to(org_vae_device)
    clean_memory_on_device(device)

    return pil_img


# ---------------------------------------------------------------------------
# \u30d7\u30ed\u30f3\u30d7\u30c8\u30d5\u30a1\u30a4\u30eb\u306e\u30d1\u30fc\u30b9
# ---------------------------------------------------------------------------

def parse_sample_prompt_file(prompt_file: str) -> list:
    import re

    path = Path(prompt_file)
    if not path.is_file():
        logger.error(f"[SampleGen] \u30d7\u30ed\u30f3\u30d7\u30c8\u30d5\u30a1\u30a4\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093: {prompt_file}")
        return []

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    results = []
    for idx, line in enumerate(lines):
        d: dict = {"enum": idx}

        neg_m = re.search(r"\\s+--n\\s+(.+?)(?=\\s+--[a-zA-Z]|\\s*$)", line)
        neg = ""
        if neg_m:
            neg = neg_m.group(1).strip()
            line = line[:neg_m.start()] + line[neg_m.end():]
        d["negative_prompt"] = neg

        flag_names = ["--w", "--h", "--s", "--l", "--fs", "--d"]
        prompt_end = len(line)
        for flag in flag_names:
            m = re.search(re.escape(f" {flag}") + r"(?=\\s|$)", line)
            if m:
                prompt_end = min(prompt_end, m.start())
        d["prompt"] = line[:prompt_end].strip()

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
            m = re.search(re.escape(flag) + r"\\s+(\\S+)", remainder)
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
# \u5b66\u7fd2\u30eb\u30fc\u30d7\u304b\u3089\u306e\u547c\u3073\u51fa\u3057\u30a8\u30f3\u30c8\u30ea\u30dd\u30a4\u30f3\u30c8
# ---------------------------------------------------------------------------

def sample_images_from_prompts(
    *,
    args,
    dit: anima_models.Anima,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    text_encoder: torch.nn.Module,
    tokenize_strategy,
    text_encoding_strategy,
    accelerator,
    epoch: Optional[int],
    global_step: int,
    vae_for_sample=None,
) -> None:
    if not getattr(args, "sample_prompts", None):
        return
    if not getattr(args, "sample_save_dir", None):
        return

    prompt_dicts = parse_sample_prompt_file(args.sample_prompts)
    if not prompt_dicts:
        logger.warning("[SampleGen] \u6709\u52b9\u306a\u30d7\u30ed\u30f3\u30d7\u30c8\u884c\u304c\u3042\u308a\u307e\u305b\u3093\u3002\u30b9\u30ad\u30c3\u30d7\u3057\u307e\u3059\u3002")
        return

    save_dir = Path(args.sample_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = accelerator.device
    dtype  = dit.dtype

    active_vae = vae_for_sample if vae_for_sample is not None else vae

    dit_was_training = dit.training
    dit.eval()
    if hasattr(dit, "switch_block_swap_for_inference"):
        dit.switch_block_swap_for_inference()

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

                    ts_str    = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
                    prefix     = (args.output_name + "_") if getattr(args, "output_name", None) else ""
                    seed_suffix = f"_{pd[\'seed\']}"
                    fname = f"{prefix}{num_suffix}_{pd[\'enum\']:02d}_{ts_str}{seed_suffix}.png"
                    pil_img.save(str(save_dir / fname))
                    logger.info(f"[SampleGen] saved: {save_dir / fname}")

                except Exception as exc:
                    logger.warning(f"[SampleGen] \u30d7\u30ed\u30f3\u30d7\u30c8 {pd[\'enum\']} \u751f\u6210\u5931\u6557: {exc}", exc_info=True)

    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            try:
                torch.cuda.set_rng_state(cuda_rng_state)
            except Exception:
                pass

        if dit_was_training:
            dit.train()
        if hasattr(dit, "switch_block_swap_for_training"):
            dit.switch_block_swap_for_training()

    clean_memory_on_device(device)
'''


def apply():
    print(f"[apply_fix_sample_gen_new] 配置先: {TARGET}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        import time as _time
        bak = TARGET.with_suffix(f".bak_{int(_time.time())}")
        shutil.copy2(TARGET, bak)
        print(f"  バックアップ: {bak}")

    TARGET.write_text(CONTENT, encoding="utf-8", newline="\n")
    print(f"  書き込み完了: {TARGET}")
    print("[apply_fix_sample_gen_new] 完了")


if __name__ == "__main__":
    apply()
