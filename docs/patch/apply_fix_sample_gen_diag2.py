"""apply_fix_sample_gen_diag2.py

anima_sample_gen.py の既存診断ログを拡張する。

追加する診断情報:
  [DIAG2] Qwen3 raw output (adapter通過前) の norm / mean_abs / 非ゼロ行数
  [DIAG2] T5 input_ids の先頭15トークンID
  [DIAG2] 1girl vs 1boy など複数プロンプト間の差を確認できるよう
          prompt 文字列を短縮して併記

適用対象: sd-scripts/anima_sample_gen.py
  ※ apply_fix_sample_gen_diag.py 適用済みの状態を前提とする
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("sd-scripts/anima_sample_gen.py")

# ---------- パッチ1: adapter 通過前の Qwen3 raw output と T5 token IDs を記録 ----------
# 既存ログの直前（_preprocess_text_embeds 呼び出し前）に挿入する

OLD1 = _adapt("""\
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
    logger.info(
        f"[DIAG] encode_prompt: t5_nonzero={nonzero_t5}/{t5_attn_mask.numel()}, "
        f"crossattn_emb norm={emb_norm:.4f} mean_abs={emb_mean:.6f} shape={tuple(crossattn_emb.shape)}"
    )
    # --- 診断ログここまで ---

    return crossattn_emb  # (1, N, D)""")

NEW1 = _adapt("""\
    # --- 診断ログ2: adapter 通過前 Qwen3 raw output ---
    qwen3_norm     = float(prompt_embeds.norm().item())
    qwen3_mean     = float(prompt_embeds.abs().mean().item())
    qwen3_nonzero  = int(attn_mask.sum().item())
    t5_nonzero_pre = int(t5_attn_mask.sum().item())
    t5_ids_head    = t5_input_ids[0, :t5_nonzero_pre].tolist()
    prompt_short   = prompt[:40].replace("\\n", " ")
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

    return crossattn_emb  # (1, N, D)""")


# ---------- パッチ2: _denoise の CFG前後の noise_pred を分離して記録 ----------
OLD2 = _adapt("""\
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
                )""")

NEW2 = _adapt("""\
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
                )""")


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    content = _adapt(raw.decode("utf-8"))

    results = []

    # パッチ1
    if NEW1 in content:
        results.append("[SKIP1] パッチ1は既に適用済みです。")
    elif OLD1 in content:
        content = content.replace(OLD1, NEW1, 1)
        results.append("[OK1] Qwen3 raw output / T5 token IDs の診断ログを追加しました。")
    else:
        print("[ERROR1] パッチ1の対象文字列が見つかりません。", file=sys.stderr)
        # デバッグ用: 前後20文字を探して差異を特定
        key = "# _preprocess_text_embeds: LLM adapter"
        idx = content.find(_adapt(key))
        if idx != -1:
            print(f"  参照位置(idx={idx}): {repr(content[idx:idx+80])}", file=sys.stderr)
        sys.exit(1)

    # パッチ2
    if NEW2 in content:
        results.append("[SKIP2] パッチ2は既に適用済みです。")
    elif OLD2 in content:
        content = content.replace(OLD2, NEW2, 1)
        results.append("[OK2] CFG前後の noise_pred 診断ログを追加しました。")
    else:
        print("[ERROR2] パッチ2の対象文字列が見つかりません。", file=sys.stderr)
        key = "noise_pred = dit(latents, t, crossattn_emb"
        idx = content.find(_adapt(key))
        if idx != -1:
            print(f"  参照位置(idx={idx}): {repr(content[idx:idx+80])}", file=sys.stderr)
        sys.exit(1)

    for r in results:
        print(r)

    if b"\r\n" in raw:
        content = content.replace("\n", "\r\n")
    path.write_text(content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
