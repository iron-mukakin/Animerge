"""apply_fix_bug05b_preprocess_text_embeds.py
BUG-05 再修正: anima_train_utils.py の _sample_image_inference における
プロンプト前処理を anima_minimal_inference.py と同一の経路に統一する。

【前回 apply_fix_bug05_do_sample_args.py の問題点】
  do_sample 内の dit() 呼び出しに target_input_ids 等を追加したが、
  Anima.forward() は target_input_ids が渡されると内部で
  _preprocess_text_embeds() → llm_adapter を呼ぶ。
  _sample_image_inference では既に dit.llm_adapter を手動呼び出し
  していたため、LLM adapter が 2 重適用されていた。

【正しい経路 (anima_minimal_inference.py 準拠)】
  エンコード直後に anima._preprocess_text_embeds() を呼んで
  crossattn_emb を確定させ、dit() には target_input_ids を渡さない。
  → dit.forward() 内では target_input_ids=None のため
    _preprocess_text_embeds は source_hidden_states をそのまま返す。

【このパッチが行う変更】
  _sample_image_inference 内の以下ブロックを置換:
    - LLM adapter 手動呼び出し (pos/neg 両方) を
      dit._preprocess_text_embeds() 呼び出しに変更

  ※ do_sample のシグネチャ・本体は変更しない
  ※ apply_fix_bug05_do_sample_args.py が適用済みの場合も
     apply_fix_bug05b_preprocess_text_embeds.py は
     _sample_image_inference 部分のみを対象とするため競合しない。
     ただし bug05 パッチは不要であり、将来的に差し戻しを推奨する。

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
# PATCH: pos/neg プロンプト前処理ブロックを _preprocess_text_embeds に統一
#
# OLD: dit.llm_adapter を手動呼び出し (use_llm_adapter 分岐あり)
# NEW: dit._preprocess_text_embeds() に統一
#      (内部で use_llm_adapter を判定し adapter 有無を吸収する)
# ──────────────────────────────────────────────────────────────────────────────

OLD = (
    '    # Process through LLM adapter if available\n'
    '    if dit.use_llm_adapter:\n'
    '        crossattn_emb = dit.llm_adapter(\n'
    '            source_hidden_states=prompt_embeds,\n'
    '            target_input_ids=t5_input_ids,\n'
    '            target_attention_mask=t5_attn_mask,\n'
    '            source_attention_mask=attn_mask,\n'
    '        )\n'
    '        crossattn_emb[~t5_attn_mask.bool()] = 0\n'
    '    else:\n'
    '        crossattn_emb = prompt_embeds\n'
    '\n'
    '    # Encode negative prompt for CFG\n'
    '    neg_crossattn_emb = None\n'
    '    if scale > 1.0 and negative_prompt is not None:\n'
    '        neg_encoded = encode_prompt(negative_prompt)\n'
    '        if neg_encoded is not None:\n'
    '            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded\n'
    '            if isinstance(neg_pe, np.ndarray):\n'
    '                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)\n'
    '                neg_am = torch.from_numpy(neg_am).unsqueeze(0)\n'
    '                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)\n'
    '                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)\n'
    '\n'
    '            neg_pe = neg_pe.to(accelerator.device, dtype=dit.dtype)\n'
    '            neg_am = neg_am.to(accelerator.device)\n'
    '            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)\n'
    '            neg_t5_am = neg_t5_am.to(accelerator.device)\n'
    '\n'
    '            if dit.use_llm_adapter:\n'
    '                neg_crossattn_emb = dit.llm_adapter(\n'
    '                    source_hidden_states=neg_pe,\n'
    '                    target_input_ids=neg_t5_ids,\n'
    '                    target_attention_mask=neg_t5_am,\n'
    '                    source_attention_mask=neg_am,\n'
    '                )\n'
    '                neg_crossattn_emb[~neg_t5_am.bool()] = 0\n'
    '            else:\n'
    '                neg_crossattn_emb = neg_pe\n'
)

NEW = (
    '    # Preprocess positive prompt embeddings via _preprocess_text_embeds\n'
    '    # (handles llm_adapter internally; same path as anima_minimal_inference.py)\n'
    '    crossattn_emb = dit._preprocess_text_embeds(\n'
    '        source_hidden_states=prompt_embeds,\n'
    '        target_input_ids=t5_input_ids,\n'
    '        target_attention_mask=t5_attn_mask,\n'
    '        source_attention_mask=attn_mask,\n'
    '    )\n'
    '\n'
    '    # Encode negative prompt for CFG\n'
    '    neg_crossattn_emb = None\n'
    '    if scale > 1.0 and negative_prompt is not None:\n'
    '        neg_encoded = encode_prompt(negative_prompt)\n'
    '        if neg_encoded is not None:\n'
    '            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded\n'
    '            if isinstance(neg_pe, np.ndarray):\n'
    '                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)\n'
    '                neg_am = torch.from_numpy(neg_am).unsqueeze(0)\n'
    '                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)\n'
    '                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)\n'
    '\n'
    '            neg_pe = neg_pe.to(accelerator.device, dtype=dit.dtype)\n'
    '            neg_am = neg_am.to(accelerator.device)\n'
    '            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)\n'
    '            neg_t5_am = neg_t5_am.to(accelerator.device)\n'
    '\n'
    '            neg_crossattn_emb = dit._preprocess_text_embeds(\n'
    '                source_hidden_states=neg_pe,\n'
    '                target_input_ids=neg_t5_ids,\n'
    '                target_attention_mask=neg_t5_am,\n'
    '                source_attention_mask=neg_am,\n'
    '            )\n'
)


def apply():
    if not TARGET_FILE.exists():
        print(f"[ERROR] 対象ファイルが見つかりません: {TARGET_FILE}")
        sys.exit(1)

    raw = TARGET_FILE.read_bytes()
    text = _adapt(raw.decode("utf-8"))

    old_n = _adapt(OLD)
    new_n = _adapt(NEW)

    if old_n not in text:
        print("[ERROR] 置換対象の文字列が見つかりません。")
        print("  先頭部分:")
        print(repr(old_n[:200]))
        sys.exit(1)

    count = text.count(old_n)
    if count != 1:
        print(f"[ERROR] 置換対象が {count} 箇所見つかりました。中断します。")
        sys.exit(1)

    print("[OK] 置換対象を確認しました。")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET_FILE.with_suffix(f".bak_{ts}")
    shutil.copy2(TARGET_FILE, bak)
    print(f"[INFO] バックアップ作成: {bak}")

    new_text = text.replace(old_n, new_n, 1)

    if b"\r\n" in raw:
        new_text = new_text.replace("\n", "\r\n")

    TARGET_FILE.write_text(new_text, encoding="utf-8", newline="")
    print(f"[OK] BUG-05b 修正を適用しました: {TARGET_FILE}")
    print("  _sample_image_inference のプロンプト前処理を")
    print("  dit._preprocess_text_embeds() に統一しました。")


if __name__ == "__main__":
    # プロジェクトルートから実行すること (python apply_fix_bug05b_preprocess_text_embeds.py)
    apply()
