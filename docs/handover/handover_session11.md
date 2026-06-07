# anima(cosmos) サンプル生成 改修引き継ぎ仕様書

作成日: 2026-06-07
対象: セッション11

---

## 1. セッション10 の成果と解決済み問題

### 1-1. 旧サンプル生成コード削除（完了）

以下のパッチを作成・検証済み（次セッション冒頭で適用すること）。

| パッチファイル | 対象 | 内容 |
|--------------|------|------|
| `apply_fix_del_old_sample_utils.py` | `sd-scripts/anima_train_utils.py` | `do_sample` / `sample_images` / `_sample_image_inference` を L310〜末尾ごと削除 |
| `apply_fix_del_old_sample_leco.py` | `sd-scripts/anima_train_leco.py` | `_generate_samples_leco` / `_parse_sample_prompt_line` / Sample generation helper セクションを削除 |

**削除しなかった関数（学習ループで現役使用中）:**
`_anima_forward` / `diffusion_anima` / `predict_noise_anima` / `concat_embeds_anima` / `repeat_embeds_anima` / `get_initial_latents_anima` / `encode_prompt_anima`

### 1-2. プロンプト非反映問題の根本原因特定と解決（完了）

**原因:** `configs/t5_old/` の T5 tokenizer ファイルが破損または不整合だった。全単語が `<unk>`（ID=2）に落ち、LLM Adapter の query が全プロンプトで同一になっていた。

**解決:** `sd-scripts` にバンドルされた正規の configs ファイルを使用することで解消。

**解決後の DIAG 値（正常）:**

| 項目 | 修正前（破損tokenizer） | 修正後（正規tokenizer） |
|------|----------------------|----------------------|
| `t5_ids` | `[3,2,3,2,3,2,3,2,1]`（全単語 unk） | `[209, 18722, 6, 6729, 6, 423, 643, 1]`（正常） |
| `1girl` qwen3_norm | 267.5838 | 267.5838（変化なし） |
| `1girl` crossattn_emb norm | 13.49〜13.58 | 適切な値 |
| do_cfg | False（neg 非空でも） | True（neg 非空のとき） |

**正常動作確認済みログ（2026-06-06）:**
- LoRA: `lora_output_e000001_00_20260606140502_42.png` 正常生成
- LECO: `leco_output_000004_00_20260606140901_42.png` 正常生成

### 1-3. 診断ログパッチ（今後の必要性なし）

以下のパッチは診断目的のみ。本番運用では除去してよい（または残存しても動作に影響なし）。

- `apply_fix_sample_gen_diag.py`
- `apply_fix_sample_gen_diag2.py`

---

## 2. 現在のファイル状態

| ファイル | 状態 |
|----------|------|
| `sd-scripts/anima_train_utils.py` | 旧サンプル生成コード残存（パッチ未適用） |
| `sd-scripts/anima_train_leco.py` | 旧サンプル生成コード残存（パッチ未適用）、`sample_images_from_prompts` 呼び出し適用済み |
| `sd-scripts/anima_train_network.py` | `sample_images_from_prompts` 呼び出し適用済み |
| `sd-scripts/anima_sample_gen.py` | 正常動作確認済み、診断ログ追加済み |
| `app/lora_train.py` | セッション1〜9パッチ適用済み |
| `app/leco_train.py` | セッション1〜9パッチ適用済み |

---

## 3. セッション11 の作業内容

### 作業A: LECO サンプル画像がプレビューに表示されない問題

#### 症状

`app/leco_train.py` のギャラリー更新処理が glob パターン `*_e*_00_*.png` / `*_e*_01_*.png` でサンプル画像を検索している。

LECO のサンプルファイル名はステップベース（`epoch=None`）のため `e` プレフィックスがなく、glob にヒットしない。

**実際のファイル名例:**
```
leco_output_000002_00_20260607000900_42.png   ← ステップベース、e プレフィックスなし
leco_output_000002_01_20260607000900_43.png
```

**LoRA のファイル名例（glob ヒットする）:**
```
lora_output_e000001_00_20260606140502_42.png  ← epoch ベース、e プレフィックスあり
```

#### 修正方針

`app/leco_train.py` のギャラリー glob パターンを変更する。

```python
# 変更前
"*_e*_00_*.png"
"*_e*_01_*.png"

# 変更後（epoch/step どちらにも対応）
"*_00_*.png"
"*_01_*.png"
```

**注意:** `*_00_*.png` は既存の他ファイルと誤マッチする可能性があるため、`app/leco_train.py` 内の glob 使用箇所全体を確認してから適用すること。

**作成するパッチ:** `apply_fix_leco_gallery_glob.py`

---

### 作業B: LECO プリセット保存にサンプル生成設定を反映

#### 現状

LoRA 学習（`app/lora_train.py`）ではプリセットにサンプル生成タブの入力値（プロンプト・ネガティブプロンプト・解像度・ステップ数・scale・flow_shift・seed など）が含まれる。

LECO 学習（`app/leco_train.py`）ではプリセット保存・読み込みにサンプル生成設定が含まれていない。

#### 対象となるサンプル生成設定項目

`app/lora_train.py` のサンプル生成タブを参照して、同等の項目を LECO プリセットに追加する。

具体的な項目は `app/lora_train.py` と `app/leco_train.py` の双方を次セッションで読み込んで差分を確認すること。

**作成するパッチ:** `apply_fix_leco_preset_sample.py`

---

### 作業C: LECO サンプル生成タブをプリセットタブの左へ移動

#### 現状

`app/leco_train.py` のタブ構成（現在の順序）:

```
[基本設定] [詳細設定] [プリセット] ... [サンプル生成]
```

LoRA 学習（`app/lora_train.py`）のタブ構成:

```
[基本設定] [詳細設定] ... [サンプル生成] [プリセット]
```

#### 修正方針

`app/leco_train.py` のタブ定義順序を変更し、サンプル生成タブをプリセットタブの左（直前）に移動する。

Gradio の `gr.Tab` / `gr.TabItem` の定義順がそのまま表示順になるため、コードブロックの順序を入れ替える。

**注意:** タブ内コンポーネントの変数参照（イベントハンドラの `inputs`/`outputs`）は順序変更の影響を受けないため、タブコンテナブロックの移動のみで対応できる見込み。移動前に変数参照の依存関係を確認すること。

**作成するパッチ:** `apply_fix_leco_tab_order.py`

---

## 4. セッション11 の作業順序

```
STEP 1: 旧サンプル生成コード削除パッチを適用
        apply_fix_del_old_sample_utils.py
        apply_fix_del_old_sample_leco.py

STEP 2: app/leco_train.py / app/lora_train.py を読み込み
        - glob パターン使用箇所を全確認（作業A用）
        - プリセット保存・読み込み処理を確認（作業B用）
        - タブ定義順序とコンポーネント変数参照を確認（作業C用）

STEP 3: 作業A パッチ作成・適用
        apply_fix_leco_gallery_glob.py

STEP 4: 作業B パッチ作成・適用
        apply_fix_leco_preset_sample.py

STEP 5: 作業C パッチ作成・適用
        apply_fix_leco_tab_order.py

STEP 6: 動作確認
        - LECO 学習実行→サンプル生成→プレビュー表示確認
        - プリセット保存→再読み込み→サンプル生成設定が復元されることを確認
        - タブ表示順序の確認
```

---

## 5. 適用済みパッチ一覧（セッション1〜10）

```
1.  apply_fix_leco_train2.py                フェーズ2: 階層学習・プリセットタブ追加
2.  apply_fix_sample_ab.py                  サンプルA/B変数・UI追加
3.  apply_fix_leco_argdup.py                argparse重複引数削除
4.  apply_fix_leco_funcorder.py             関数定義順序修正（NameError対策）
5.  apply_fix_sample_filenames.py           ファイル名ベース判別・サブディレクトリ廃止
6.  apply_fix_bug01_lora_glob.py            BUG-01: globパターン修正（LoRA側）
7.  apply_fix_bug02_vae_decode.py           BUG-02: vae.decode dict/object両対応
8.  apply_fix_bug03_sample_prompt_parse.py  BUG-03: _parse_sample_prompt_line修正
9.  apply_fix_bug04_sample_seed.py          BUG-04: A/B seed分離（A=42, B=43）
10. apply_fix_bug04b_leco_seed_order.py     BUG-04b hotfix: LECO NameError修正
11. apply_fix_lora_sample_replace.py        anima_train_network.py sample_images差し替え
12. apply_fix_leco_sample_replace.py        anima_train_leco.py sample_images差し替え
13. apply_fix_sample_gen_diag.py            診断ログ追加（診断目的のみ・除去可）
14. apply_fix_sample_gen_diag2.py           診断ログ拡張（診断目的のみ・除去可）
```

未適用（セッション11冒頭で適用）:
```
15. apply_fix_del_old_sample_utils.py       anima_train_utils.py 旧コード削除
16. apply_fix_del_old_sample_leco.py        anima_train_leco.py 旧コード削除
```

---

## 6. 重要な技術情報

### T5 tokenizer の正しい配置

`sd-scripts/configs/` にバンドルされている T5 tokenizer ファイルを使用すること。独自に用意した `configs/t5_old/` は語彙が不整合で全単語が `<unk>` に落ちるため使用禁止。

### anima_sample_gen.py の正しい推論経路

```
tokenize(prompt)
→ encode_tokens(qwen3_te) → [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]
→ dit._preprocess_text_embeds(
      source_hidden_states=prompt_embeds,
      target_input_ids=t5_input_ids,       # LLM Adapter の query
      target_attention_mask=t5_attn_mask,
      source_attention_mask=qwen3_attn_mask
  )
→ crossattn_emb[~t5_attn_mask.bool()] = 0  # T5マスクでゼロ埋め（参照実装と同一）
→ dit(latents, t, crossattn_emb, padding_mask=padding_mask)
  # target_input_ids=None → Anima.forward 内の _preprocess_text_embeds は素通り
→ vae.decode_to_pixels → PIL
```

### LECO サンプルファイル名の命名規則

```
{output_name}_{global_step:06d}_{ab_index:02d}_{timestamp}_{seed}.png

例: leco_output_000002_00_20260607000900_42.png
    leco_output_000002_01_20260607000900_43.png
```

epoch=None（ステップベース）のため `e` プレフィックスは付かない。

### LoRA サンプルファイル名の命名規則

```
{output_name}_e{epoch:06d}_{ab_index:02d}_{timestamp}_{seed}.png

例: lora_output_e000001_00_20260606140502_42.png
```

---

## 7. 次セッションで必要なファイル

| ファイル | 用途 |
|----------|------|
| `app/leco_train.py` | 作業A・B・C の対象ファイル |
| `app/lora_train.py` | 作業B・C の参照ファイル（プリセット・タブ構成の比較用） |
