"""apply_fix_bug05_do_sample_args.py
BUG-05 修正: anima_train_utils.py の do_sample に
attn_mask / t5_input_ids / t5_attn_mask 引数を追加し、
dit() 呼び出し時にこれらを渡す。

問題:
  学習時の dit() 呼び出し (anima_train_network.py L308-316) では
    target_input_ids / target_attention_mask / source_attention_mask
  の3引数をすべて渡しているが、do_sample 内の dit() 呼び出しでは
  これらが欠落していた。
  その結果サンプル生成時にプロンプトがモデルに到達せず、
  どのプロンプトを入力しても同一構図の画像が出力されていた。

修正箇所:
  1. do_sample シグネチャに3引数を追加
  2. do_sample 内 dit() 呼び出し (CFG有/無 両方) に3引数を追加
  3. _sample_image_inference から do_sample を呼ぶ箇所に
     attn_mask / t5_input_ids / t5_attn_mask を渡すよう変更
     (neg側も同様)

対象ファイル: sd-scripts/library/anima_train_utils.py
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _adapt(s: str) -> str:
    """CRLF/LF 両対応"""
    return s.replace("\r\n", "\n")


TARGET_FILE = Path("sd-scripts/library/anima_train_utils.py")

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1: do_sample シグネチャ + dit() 呼び出し
# ──────────────────────────────────────────────────────────────────────────────
OLD1 = (
    'def do_sample(\n'
    '    height: int,\n'
    '    width: int,\n'
    '    seed: Optional[int],\n'
    '    dit: anima_models.Anima,\n'
    '    crossattn_emb: torch.Tensor,\n'
    '    steps: int,\n'
    '    dtype: torch.dtype,\n'
    '    device: torch.device,\n'
    '    guidance_scale: float = 1.0,\n'
    '    flow_shift: float = 3.0,\n'
    '    neg_crossattn_emb: Optional[torch.Tensor] = None,\n'
    ') -> torch.Tensor:\n'
    '    """Generate a sample using Euler discrete sampling for rectified flow.\n'
    '\n'
    '    Args:\n'
    '        height, width: Output image dimensions\n'
    '        seed: Random seed (None for random)\n'
    '        dit: Anima model\n'
    '        crossattn_emb: Cross-attention embeddings (B, N, D)\n'
    '        steps: Number of sampling steps\n'
    '        dtype: Compute dtype\n'
    '        device: Compute device\n'
    '        guidance_scale: CFG scale (1.0 = no guidance)\n'
    '        flow_shift: Flow shift parameter for rectified flow\n'
    '        neg_crossattn_emb: Negative cross-attention embeddings for CFG\n'
    '\n'
    '    Returns:\n'
    '        Denoised latents\n'
    '    """\n'
    '    # Latent shape: (1, 16, 1, H/8, W/8) for single image\n'
    '    latent_h = height // 8\n'
    '    latent_w = width // 8\n'
    '    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)\n'
    '\n'
    '    # Generate noise\n'
    '    if seed is not None:\n'
    '        generator = torch.manual_seed(seed)\n'
    '    else:\n'
    '        generator = None\n'
    '    noise = torch.randn(latent.size(), dtype=torch.float32, generator=generator, device="cpu").to(dtype).to(device)\n'
    '\n'
    '    # Timestep schedule: linear from 1.0 to 0.0\n'
    '    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)\n'
    '    flow_shift = float(flow_shift)\n'
    '    if flow_shift != 1.0:\n'
    '        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)\n'
    '\n'
    '    # Start from pure noise\n'
    '    x = noise.clone()\n'
    '\n'
    '    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims\n'
    '    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)\n'
    '\n'
    '    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None\n'
    '\n'
    '    for i in tqdm(range(steps), desc="Sampling"):\n'
    '        sigma = sigmas[i]\n'
    '        t = sigma.unsqueeze(0)  # (1,)\n'
    '\n'
    '        if use_cfg:\n'
    '            # CFG: two separate passes to reduce memory usage\n'
    '            pos_out = dit(x, t, crossattn_emb, padding_mask=padding_mask)\n'
    '            pos_out = pos_out.float()\n'
    '            neg_out = dit(x, t, neg_crossattn_emb, padding_mask=padding_mask)\n'
    '            neg_out = neg_out.float()\n'
    '\n'
    '            model_output = neg_out + guidance_scale * (pos_out - neg_out)\n'
    '        else:\n'
    '            model_output = dit(x, t, crossattn_emb, padding_mask=padding_mask)\n'
    '            model_output = model_output.float()\n'
    '\n'
    '        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * model_output\n'
    '        dt = sigmas[i + 1] - sigma\n'
    '        x = x + model_output * dt\n'
    '        x = x.to(dtype)\n'
    '\n'
    '    return x\n'
)

NEW1 = (
    'def do_sample(\n'
    '    height: int,\n'
    '    width: int,\n'
    '    seed: Optional[int],\n'
    '    dit: anima_models.Anima,\n'
    '    crossattn_emb: torch.Tensor,\n'
    '    steps: int,\n'
    '    dtype: torch.dtype,\n'
    '    device: torch.device,\n'
    '    guidance_scale: float = 1.0,\n'
    '    flow_shift: float = 3.0,\n'
    '    neg_crossattn_emb: Optional[torch.Tensor] = None,\n'
    '    attn_mask: Optional[torch.Tensor] = None,\n'
    '    t5_input_ids: Optional[torch.Tensor] = None,\n'
    '    t5_attn_mask: Optional[torch.Tensor] = None,\n'
    '    neg_attn_mask: Optional[torch.Tensor] = None,\n'
    '    neg_t5_input_ids: Optional[torch.Tensor] = None,\n'
    '    neg_t5_attn_mask: Optional[torch.Tensor] = None,\n'
    ') -> torch.Tensor:\n'
    '    """Generate a sample using Euler discrete sampling for rectified flow.\n'
    '\n'
    '    Args:\n'
    '        height, width: Output image dimensions\n'
    '        seed: Random seed (None for random)\n'
    '        dit: Anima model\n'
    '        crossattn_emb: Cross-attention embeddings (B, N, D)\n'
    '        steps: Number of sampling steps\n'
    '        dtype: Compute dtype\n'
    '        device: Compute device\n'
    '        guidance_scale: CFG scale (1.0 = no guidance)\n'
    '        flow_shift: Flow shift parameter for rectified flow\n'
    '        neg_crossattn_emb: Negative cross-attention embeddings for CFG\n'
    '        attn_mask: Qwen3 attention mask for positive prompt\n'
    '        t5_input_ids: T5 input IDs for positive prompt\n'
    '        t5_attn_mask: T5 attention mask for positive prompt\n'
    '        neg_attn_mask: Qwen3 attention mask for negative prompt\n'
    '        neg_t5_input_ids: T5 input IDs for negative prompt\n'
    '        neg_t5_attn_mask: T5 attention mask for negative prompt\n'
    '\n'
    '    Returns:\n'
    '        Denoised latents\n'
    '    """\n'
    '    # Latent shape: (1, 16, 1, H/8, W/8) for single image\n'
    '    latent_h = height // 8\n'
    '    latent_w = width // 8\n'
    '    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)\n'
    '\n'
    '    # Generate noise\n'
    '    if seed is not None:\n'
    '        generator = torch.manual_seed(seed)\n'
    '    else:\n'
    '        generator = None\n'
    '    noise = torch.randn(latent.size(), dtype=torch.float32, generator=generator, device="cpu").to(dtype).to(device)\n'
    '\n'
    '    # Timestep schedule: linear from 1.0 to 0.0\n'
    '    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)\n'
    '    flow_shift = float(flow_shift)\n'
    '    if flow_shift != 1.0:\n'
    '        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)\n'
    '\n'
    '    # Start from pure noise\n'
    '    x = noise.clone()\n'
    '\n'
    '    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims\n'
    '    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)\n'
    '\n'
    '    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None\n'
    '\n'
    '    for i in tqdm(range(steps), desc="Sampling"):\n'
    '        sigma = sigmas[i]\n'
    '        t = sigma.unsqueeze(0)  # (1,)\n'
    '\n'
    '        if use_cfg:\n'
    '            # CFG: two separate passes to reduce memory usage\n'
    '            pos_out = dit(\n'
    '                x, t, crossattn_emb,\n'
    '                padding_mask=padding_mask,\n'
    '                target_input_ids=t5_input_ids,\n'
    '                target_attention_mask=t5_attn_mask,\n'
    '                source_attention_mask=attn_mask,\n'
    '            )\n'
    '            pos_out = pos_out.float()\n'
    '            neg_out = dit(\n'
    '                x, t, neg_crossattn_emb,\n'
    '                padding_mask=padding_mask,\n'
    '                target_input_ids=neg_t5_input_ids,\n'
    '                target_attention_mask=neg_t5_attn_mask,\n'
    '                source_attention_mask=neg_attn_mask,\n'
    '            )\n'
    '            neg_out = neg_out.float()\n'
    '\n'
    '            model_output = neg_out + guidance_scale * (pos_out - neg_out)\n'
    '        else:\n'
    '            model_output = dit(\n'
    '                x, t, crossattn_emb,\n'
    '                padding_mask=padding_mask,\n'
    '                target_input_ids=t5_input_ids,\n'
    '                target_attention_mask=t5_attn_mask,\n'
    '                source_attention_mask=attn_mask,\n'
    '            )\n'
    '            model_output = model_output.float()\n'
    '\n'
    '        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * model_output\n'
    '        dt = sigmas[i + 1] - sigma\n'
    '        x = x + model_output * dt\n'
    '        x = x.to(dtype)\n'
    '\n'
    '    return x\n'
)

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2: _sample_image_inference から do_sample を呼ぶ箇所
# ──────────────────────────────────────────────────────────────────────────────
OLD2 = (
    '    # Generate sample\n'
    '    clean_memory_on_device(accelerator.device)\n'
    '    latents = do_sample(\n'
    '        height, width, seed, dit, crossattn_emb, sample_steps, dit.dtype, accelerator.device, scale, flow_shift, neg_crossattn_emb\n'
    '    )\n'
)

NEW2 = (
    '    # Generate sample\n'
    '    clean_memory_on_device(accelerator.device)\n'
    '    latents = do_sample(\n'
    '        height, width, seed, dit, crossattn_emb, sample_steps, dit.dtype, accelerator.device,\n'
    '        scale, flow_shift, neg_crossattn_emb,\n'
    '        attn_mask=attn_mask,\n'
    '        t5_input_ids=t5_input_ids,\n'
    '        t5_attn_mask=t5_attn_mask,\n'
    '        neg_attn_mask=neg_am if neg_crossattn_emb is not None else None,\n'
    '        neg_t5_input_ids=neg_t5_ids if neg_crossattn_emb is not None else None,\n'
    '        neg_t5_attn_mask=neg_t5_am if neg_crossattn_emb is not None else None,\n'
    '    )\n'
)


def _patch(text: str, old: str, new: str, label: str) -> str:
    old_n = _adapt(old)
    new_n = _adapt(new)

    if old_n not in text:
        print(f"[ERROR] {label}: 置換対象の文字列が見つかりません。")
        print("  先頭部分:")
        print(repr(old_n[:200]))
        sys.exit(1)

    count = text.count(old_n)
    if count != 1:
        print(f"[ERROR] {label}: 置換対象が {count} 箇所見つかりました。中断します。")
        sys.exit(1)

    print(f"[OK] {label}: 置換対象を確認しました。")
    return text.replace(old_n, new_n, 1)


def apply():
    if not TARGET_FILE.exists():
        print(f"[ERROR] 対象ファイルが見つかりません: {TARGET_FILE}")
        sys.exit(1)

    raw = TARGET_FILE.read_bytes()
    text = _adapt(raw.decode("utf-8"))

    text = _patch(text, OLD1, NEW1, "PATCH-1: do_sample シグネチャ + dit() 呼び出し")
    text = _patch(text, OLD2, NEW2, "PATCH-2: _sample_image_inference do_sample 呼び出し")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET_FILE.with_suffix(f".bak_{ts}")
    shutil.copy2(TARGET_FILE, bak)
    print(f"[INFO] バックアップ作成: {bak}")

    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")

    TARGET_FILE.write_text(text, encoding="utf-8", newline="")
    print(f"[OK] BUG-05 修正を適用しました: {TARGET_FILE}")
    print("  do_sample に attn_mask / t5_input_ids / t5_attn_mask を追加しました。")
    print("  _sample_image_inference からの呼び出し引数を更新しました。")


if __name__ == "__main__":
    # プロジェクトルートから実行すること (python apply_fix_bug05_do_sample_args.py)
    apply()
