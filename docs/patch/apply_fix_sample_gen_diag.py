"""apply_fix_sample_gen_diag.py

anima_sample_gen.py に診断ログを追加する。

追加する情報:
  - encode_prompt_for_sample: crossattn_emb の norm / 非ゼロ要素数
  - _denoise: 各ステップの noise_pred.norm() (最初と最後のステップのみ)
  - _denoise: latents の norm（初期・最終）

問題確認後にこのパッチは削除可能。
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("sd-scripts/anima_sample_gen.py")


# ---------- パッチ1: encode_prompt_for_sample の診断ログ ----------
OLD1 = _adapt("""\
        # T5 attn mask でパディング位置をゼロ埋め（anima_minimal_inference.py と同一処理）
        crossattn_emb[~t5_attn_mask.bool()] = 0

    return crossattn_emb  # (1, N, D)""")

NEW1 = _adapt("""\
        # T5 attn mask でパディング位置をゼロ埋め（anima_minimal_inference.py と同一処理）
        crossattn_emb[~t5_attn_mask.bool()] = 0

    # --- 診断ログ ---
    nonzero_t5 = int(t5_attn_mask.sum().item())
    emb_norm = float(crossattn_emb.norm().item())
    emb_mean = float(crossattn_emb.abs().mean().item())
    logger.info(
        f"[DIAG] encode_prompt: t5_nonzero={nonzero_t5}/{t5_attn_mask.numel()}, "
        f"crossattn_emb norm={emb_norm:.4f} mean_abs={emb_mean:.6f} shape={tuple(crossattn_emb.shape)}"
    )
    # --- 診断ログここまで ---

    return crossattn_emb  # (1, N, D)""")


# ---------- パッチ2: _denoise の診断ログ ----------
OLD2 = _adapt("""\
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Sampling", leave=False):
            # t = sigma 値（0〜1）をそのまま渡す（do_sample と同一）
            t = sigmas[i].unsqueeze(0).to(dtype)

            noise_pred = dit(latents, t, crossattn_emb, padding_mask=padding_mask)
            # target_input_ids を渡さない → forward 内の _preprocess_text_embeds は素通り

            if do_cfg:
                uncond_pred = dit(latents, t, neg_crossattn_emb, padding_mask=padding_mask)
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            noise_pred = noise_pred.float()

            # Euler ステップ: x = x + (sigma_{i+1} - sigma_i) * noise_pred
            # sigma は単調減少なので dt < 0 → x = x - (sigma_i - sigma_{i+1}) * noise_pred
            dt = sigmas[i + 1] - sigmas[i]
            latents = (latents.float() + noise_pred * dt).to(dtype)

    return latents""")

NEW2 = _adapt("""\
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
    return latents""")


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    content = _adapt(raw.decode("utf-8"))

    # パッチ1
    if _adapt(NEW1) in content:
        print("[SKIP1] パッチ1は既に適用済みです。")
        p1_done = True
    elif OLD1 in content:
        content = content.replace(OLD1, NEW1, 1)
        print("[OK1] encode_prompt_for_sample に診断ログを追加しました。")
        p1_done = True
    else:
        print("[ERROR1] パッチ1の対象文字列が見つかりません。", file=sys.stderr)
        print("  探索:", repr(OLD1[:60]))
        p1_done = False

    # パッチ2
    if _adapt(NEW2) in content:
        print("[SKIP2] パッチ2は既に適用済みです。")
        p2_done = True
    elif OLD2 in content:
        content = content.replace(OLD2, NEW2, 1)
        print("[OK2] _denoise に診断ログを追加しました。")
        p2_done = True
    else:
        print("[ERROR2] パッチ2の対象文字列が見つかりません。", file=sys.stderr)
        print("  探索:", repr(OLD2[:60]))
        p2_done = False

    if p1_done and p2_done:
        if b"\r\n" in raw:
            content = content.replace("\n", "\r\n")
        path.write_text(content, encoding="utf-8", newline="")
        print(f"[OK] {path} を更新しました。")
    else:
        sys.exit(1)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
