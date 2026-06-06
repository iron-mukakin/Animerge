# anima(cosmos) サンプル生成 改修引き継ぎ仕様書

作成日: 2026-06-05
対象: セッション10

---

## 1. 現状の問題サマリー

### 症状
- LoRA・LECO 両学習でサンプル画像は生成される
- 解像度・ステップ数・flow_shift などのパラメータは正常に反映される
- **プロンプトの内容が画像に一切反映されない**（空プロンプトと同等の出力）
- LECO のサンプル画像が GUI のギャラリーに表示されない

### 根本原因（確定済み）

#### 主原因: `do_sample` 内での LLM adapter 二重適用

`anima_train_utils._sample_image_inference`（L587〜592）が `_preprocess_text_embeds` を呼んで adapter を通した `crossattn_emb` を生成する。その後 `do_sample`（L620〜628）を呼ぶ際に `target_input_ids=t5_input_ids` を渡す。`do_sample` 内の `dit(crossattn_emb, target_input_ids=t5_input_ids)` → `Anima.forward` 内で `_preprocess_text_embeds` が**再度実行される（adapter 二重適用）**。adapter を2回通すと出力がプロンプトと無関係な空間に写像される。

```
_sample_image_inference:
  crossattn_emb = dit._preprocess_text_embeds(prompt_embeds, t5_input_ids)  # 1回目
  ↓
do_sample → dit(crossattn_emb, target_input_ids=t5_input_ids)
  → Anima.forward → _preprocess_text_embeds(crossattn_emb, t5_input_ids)   # 2回目 ← バグ
```

#### 副原因: LoRA側パッチ未適用

`apply_fix_lora_sample_replace.py` の OLD 文字列が実ファイルと不一致のためパッチが適用されず、`anima_train_network.py` の `sample_images` メソッドが未改修のまま旧実装（`anima_train_utils.sample_images`）を呼び続けている。

---

## 2. セッション10 の作業方針

### 基本方針

> **旧サンプル生成コードを完全削除してから `anima_sample_gen.py` に一本化する。**

バグを含む旧コードが残存することで混乱が継続している。次セッションは削除から開始する。

---

## 3. 削除対象

### 3-1. `sd-scripts/anima_train_utils.py`

以下の3関数をまとめて削除する（L310〜ファイル末尾 L665 がほぼ全て対象）。

| 関数名 | 開始行 | 備考 |
|--------|--------|------|
| `do_sample` | L310 | adapter 二重適用バグの温床 |
| `sample_images` | L420 | `_sample_image_inference` を呼ぶラッパー |
| `_sample_image_inference` | L515 | `do_sample` を呼ぶ本体 |

**注意:** `encode_prompt_anima` は `anima_train_utils.py` には存在しない。`anima_train_leco.py` の関数。

削除後の `anima_train_utils.py` はサンプル生成関数を持たない（他の学習ユーティリティ関数 L1〜L309 は残す）。

### 3-2. `sd-scripts/anima_train_leco.py`

以下の5関数を削除する。

| 関数名 | 開始行 | 削除可否 | 備考 |
|--------|--------|---------|------|
| `encode_prompt_anima` | L77 | **条件付き削除** | L634〜635で学習ループの**プロンプトキャッシュに使用中**。削除不可。代わりに `anima_sample_gen.encode_prompt_for_sample` への移行は不要（学習ループ用エンコードはそのまま使う） |
| `_anima_forward` | L91 | 削除可 | 学習ループ内 `_anima_forward` 呼び出しを確認してから削除 |
| `diffusion_anima` | L124 | 削除可 | `_generate_samples_leco` からのみ参照 |
| `concat_embeds_anima` | L195 | 削除可 | 同上 |
| `get_initial_latents_anima` | L225 | 削除可 | `_generate_samples_leco` と L715 から参照（L715 も `_generate_samples_leco` 内） |
| `_generate_samples_leco` | L317 | 削除可 | サンプル生成呼び出しブロック（L835〜L848）も同時削除 |

**重要確認事項:**
- `encode_prompt_anima`（L77）は L634 で学習ループのプロンプトキャッシュに使用中 → **削除禁止**
- `_anima_forward`（L91）は学習ループの forward pass に使用中かどうか要確認 → 削除前に `grep _anima_forward anima_train_leco.py` で呼び出し箇所を全確認すること

---

## 4. 適用すべきパッチ（セッション9末時点での最新版）

削除後に以下を適用する。

### 4-1. `anima_sample_gen.py`（新規ファイル）

`sd-scripts/anima_sample_gen.py` として配置する。出力済みファイルをそのまま使う。

**推論経路（`anima_minimal_inference.py` と同一）:**
```
tokenize → encode_tokens
→ _preprocess_text_embeds(source=qwen3_embeds, target_input_ids=t5_ids)
→ crossattn_emb[~t5_attn_mask.bool()] = 0   # T5マスクでゼロ埋め
→ dit(latents, t, crossattn_emb, padding_mask)  # target_input_ids=None → adapter不通過
→ vae.decode_to_pixels → PIL
```

**デノイズループ（`anima_train_utils.do_sample` と同一式）:**
```python
sigmas = linspace(1.0, 0.0, steps+1)
if flow_shift != 1.0:
    sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)
for i in range(steps):
    t = sigmas[i].unsqueeze(0)
    noise_pred = dit(latents, t, crossattn_emb, padding_mask=padding_mask)
    # CFG: uncond_pred + scale * (noise_pred - uncond_pred)
    dt = sigmas[i+1] - sigmas[i]      # 負値
    latents = latents + noise_pred * dt
```

### 4-2. `apply_fix_lora_sample_replace.py`

`sd-scripts/anima_train_network.py` への変更:
1. `import anima_sample_gen` を追加
2. `cache_text_encoder_outputs_if_needed` 内の `sample_prompts_te_outputs` キャッシュブロックを削除
3. `sample_images` メソッドを `anima_sample_gen.sample_images_from_prompts` 呼び出しに差し替え
4. `cache_text_encoder_outputs=True` 時に `qwen3_te=None` になる問題を回避（`text_encoders[0]` を直接使用）

**OLD 文字列は実ファイルから直接スライスして作成済み（照合確認済み）。**

### 4-3. `apply_fix_leco_sample_replace.py`

`sd-scripts/anima_train_leco.py` への変更:
1. `import anima_sample_gen` を追加
2. サンプル生成呼び出しブロック（L827〜L848）を `anima_sample_gen.sample_images_from_prompts` に差し替え
3. LECO 固有の LoRA 制御（`net_unwrapped.set_multiplier(1.0) / eval()` → 生成後 `train() / set_multiplier(0.0)`）は維持

**`vae=None` のとき `anima_sample_gen` が `args.vae` から都度ロードする機構を実装済み。**

---

## 5. `anima_sample_gen.py` の間引き条件判定

`sample_images_from_prompts` 冒頭に以下の条件判定を実装済み（`train_util.sample_images` と同一ロジック）:

```python
if global_step == 0:
    if not getattr(args, "sample_at_first", False):
        return
else:
    sample_every_n_epochs = getattr(args, "sample_every_n_epochs", None)
    sample_every_n_steps  = getattr(args, "sample_every_n_steps", None)
    if sample_every_n_epochs is None and sample_every_n_steps is None:
        return
    if sample_every_n_epochs is not None:
        if epoch is None or epoch % sample_every_n_epochs != 0:
            return
    else:
        if global_step % sample_every_n_steps != 0 or epoch is not None:
            return
```

---

## 6. LECO サンプルが GUI に表示されない問題

### 原因

GUI（`app/leco_train.py`）のギャラリー更新処理が glob パターン `*_e*_00_*.png` / `*_e*_01_*.png` でサンプル画像を検索している（BUG-01 修正後のパターン）。

`anima_sample_gen` が出力するファイル名は epoch ベースのとき `e{epoch:06d}` プレフィックスが付く。しかし LECO はステップベース（`epoch=None`）のため `{global_step:06d}` 形式となり `e` プレフィックスがない。

**例:**
- LoRA出力: `lora_output_e000001_00_20260605123456_42.png` → glob ヒット ✓
- LECO出力: `leco_output_000002_00_20260605123456_42.png` → glob ミス ✗

### 修正方針

`app/leco_train.py` のギャラリー glob パターンを変更する。

```python
# 変更前（BUG-01適用後）
"*_e*_00_*.png"   # epoch 形式のみ
"*_e*_01_*.png"

# 変更後
"*_00_*.png"      # epoch/step どちらにも対応
"*_01_*.png"
```

または `anima_sample_gen` 側でファイル名形式を統一する（epoch=None のときも `e` プレフィックスなしの数字6桁形式を維持するか、別形式を採用するかは GUI 側の要件による）。

---

## 7. 適用済みパッチ一覧（セッション1〜9）

```
1. apply_fix_leco_train2.py           フェーズ2: 階層学習・プリセットタブ追加
2. apply_fix_sample_ab.py             サンプルA/B変数・UI追加
3. apply_fix_leco_argdup.py           argparse重複引数削除
4. apply_fix_leco_funcorder.py        関数定義順序修正（NameError対策）
5. apply_fix_sample_filenames.py      ファイル名ベース判別・サブディレクトリ廃止
6. apply_fix_bug01_lora_glob.py       BUG-01: globパターン修正
7. apply_fix_bug02_vae_decode.py      BUG-02: vae.decode dict/object両対応
8. apply_fix_bug03_sample_prompt_parse.py  BUG-03: _parse_sample_prompt_line修正
9. apply_fix_bug04_sample_seed.py     BUG-04: A/B seed分離（A=42, B=43）
10. apply_fix_bug04b_leco_seed_order.py    BUG-04b hotfix: LECO NameError修正
```

セッション9で作成・出力したがまだ正常に適用されていないパッチ:
```
11. apply_fix_lora_sample_replace.py  ← 次セッションで削除後に再適用
12. apply_fix_leco_sample_replace.py  ← 次セッションで削除後に再適用
```

新規ファイル（`sd-scripts/` に直接コピー）:
```
anima_sample_gen.py  ← sd-scripts/ に手動コピー（apply_fix_sample_gen_new.py は廃止）
```

---

## 8. ファイルの現在状態

| ファイル | 状態 |
|----------|------|
| `app/lora_train.py` | BUG-01・BUG-04 適用済み |
| `app/leco_train.py` | BUG-04・BUG-04b 適用済み、モニタータブ空 |
| `sd-scripts/anima_train_leco.py` | BUG-02・BUG-03 適用済み、旧サンプル生成コード残存 |
| `sd-scripts/anima_train_network.py` | パッチ未適用、旧 sample_images メソッド使用中 |
| `sd-scripts/anima_train_utils.py` | 旧 do_sample / sample_images / _sample_image_inference 残存（バグあり） |
| `sd-scripts/anima_sample_gen.py` | 出力済み、sd-scripts/ への配置待ち |

---

## 9. 次セッションの作業順序

```
STEP 1: anima_train_utils.py から do_sample / sample_images / _sample_image_inference を削除
        apply_fix_del_old_sample_utils.py を作成・適用

STEP 2: anima_train_leco.py から以下を削除
        - _generate_samples_leco（+ 呼び出しブロック L827〜L848）
        - diffusion_anima / concat_embeds_anima / get_initial_latents_anima
        ※ encode_prompt_anima（L77）と _anima_forward（L91）は削除前に使用箇所を再確認
        apply_fix_del_old_sample_leco.py を作成・適用

STEP 3: anima_sample_gen.py を sd-scripts/ に配置
        （手動コピー。apply_fix_sample_gen_new.py は使用しない）

STEP 4: apply_fix_lora_sample_replace.py を適用
        （照合確認済み。適用失敗時は出力ログを確認）

STEP 5: apply_fix_leco_sample_replace.py を適用

STEP 6: 動作確認
        - LoRA 学習でサンプル生成が正しいエポック間隔で実行されるか
        - プロンプトが画像に反映されているか
        - LECO 学習でサンプル生成が正しいステップ間隔で実行されるか
        - LECO サンプルの GUI 表示（glob パターン修正が必要なら STEP 7）

STEP 7: （必要なら）app/leco_train.py の glob パターン修正
        apply_fix_leco_gallery_glob.py を作成・適用
```

---

## 10. 参照実装

`sd-scripts/anima_minimal_inference.py` が正しい推論経路の参照実装。

**重要:**
- `_preprocess_text_embeds` は推論前に1回だけ呼ぶ（`prepare_text_inputs` 内）
- `dit(latents, t, crossattn_emb, padding_mask=padding_mask)` のみ。`target_input_ids` は渡さない
- `t` は sigma 値（0〜1）をそのまま渡す（×1000 しない）
- `sigmas = linspace(1, 0, steps+1)` に flow_shift を適用してスケジュールを生成

```python
# anima_minimal_inference.py L551〜572 （正しい実装）
timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
timesteps /= 1000  # 0〜1 に戻す
for i, t in enumerate(timesteps):
    t_expand = t.expand(latents.shape[0])
    noise_pred = anima(latents, t_expand, embed, padding_mask=padding_mask)
    # target_input_ids を渡していない
    latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)
```

---

## 11. 提供済みファイル（次セッションで使用）

| ファイル | 場所 | 説明 |
|----------|------|------|
| `anima_sample_gen.py` | 出力フォルダ | 新規サンプル生成モジュール本体 |
| `apply_fix_lora_sample_replace.py` | 出力フォルダ | anima_train_network.py 差し替えパッチ（照合確認済み） |
| `apply_fix_leco_sample_replace.py` | 出力フォルダ | anima_train_leco.py 差し替えパッチ（照合確認済み） |

次セッションで必要な追加ファイル（現状未アップロード）:
- `sd-scripts/library/anima_utils.py`（`load_vae` の正確なシグネチャ確認用）
- 削除後のパッチ適用エラーログ（発生した場合）
