# LoRA Train 作業引き継ぎメモ
作成日: 2026-05-23

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10, 動作確認済み)

## 現在の状態

### 解決済み問題
1. `library\` 欠落スタブ — 以下を `E:\Animerge\sd-scripts\library\` に配置済み
   - `model_util.py`
   - `strategy_sd.py`
   - `huggingface_util.py`
   - `custom_train_functions.py` (add_custom_train_arguments の引数登録実装済み)
   - `sd3_utils.py` (ModelSamplingDiscreteFlow クラス含む)
   - `sd3_models.py` (MMDiT, SDVAE クラス)
   - `flux_models.py` (Flux, AutoEncoder, ControlNetFlux クラス)
   - `flux_utils.py` (prepare_img_ids, unpack_latents)
   - `original_unet.py` (UNet2DConditionModel クラス)
   - `lpw_stable_diffusion.py` (StableDiffusionLongPromptWeightingPipeline クラス)
   - `sdxl_lpw_stable_diffusion.py` (SdxlStableDiffusionLongPromptWeightingPipeline クラス)

2. `voluptuous` パッケージ — pip install 済み

3. `networks\lora.py` の修正済み内容
   - `UNET_TARGET_REPLACE_MODULE = ["Block", "LLMAdapterTransformerBlock"]` (Anima DiT用)
   - `UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = []`
   - `is_sdxl = False` 固定 (create_network, create_network_from_weights 両方)

4. `configs\` ディレクトリ
   - `E:\Animerge\sd-scripts\configs\qwen3_06b\` — HuggingFace から取得済み
   - `E:\Animerge\sd-scripts\configs\t5_old\` — HuggingFace から取得済み

5. `app\lora_train.py` — ログファイル出力機能を追加済み

6. `networks\network_base.py` — 手動追加済み
7. `library\sdxl_original_unet.py` — 手動追加済み

### 現在の未解決問題
**`create LoRA for U-Net: 0 modules` → `optimizer got an empty parameter list`**

- `UNET_TARGET_REPLACE_MODULE = ["Block", "LLMAdapterTransformerBlock"]` に変更したが
  直前のログでまだ 0 modules になっていた
- PowerShell で直接書き換えを行い、変更は確認済み (`Select-String` で確認)
- **次のセッションで学習を再実行してログを確認する必要がある**

もし依然 0 modules の場合、確認すべき点:
1. AnimaのDiTモデル (`anima_models.Anima`) の `named_modules()` で `Block` クラスが
   実際に列挙されるか確認する
2. `Block` クラスが `anima_models.py` で定義されているが、
   `lora.py` の `create_modules` は `module.__class__.__name__` で比較するため
   継承やラッパーがあると一致しない可能性がある
3. 確認コマンド:
   ```powershell
   & 'E:\Animerge\.venv\Scripts\python.exe' -c "
   import sys; sys.path.insert(0, r'E:\Animerge\sd-scripts')
   from library import anima_models
   import torch
   # Block クラスのnamed_modulesを確認するためダミーモデルは作れないが
   # クラス名を確認
   print(anima_models.Block.__name__)
   print(anima_models.LLMAdapterTransformerBlock.__name__)
   "
   ```

## 文字化けについて
ログの日本語が文字化けする問題は未解決。
原因: Windows の subprocess stdout エンコーディングの問題。
`lora_train.py` の Popen に `encoding='utf-8'` は指定済みだが、
sd-scripts 側の logger が cp932 で出力している可能性がある。
対処が必要な場合は `env["PYTHONUTF8"] = "1"` を Popen の env に追加する。

## GUI のVAE/Qwen3パス入れ違い注意
- `--vae` → `qwen_image_vae.safetensors`
- `--qwen3` → `qwen_3_06b_base.safetensors`
GUIのフィールドラベルと実際の引数が混同しやすい。

## 次のアクション
1. GUIから学習実行 → ログ確認
2. 0 modules が続く場合は上記の `Block` クラス確認を実施
3. LoRAモジュール数が0でなくなれば学習ループに入るはず
