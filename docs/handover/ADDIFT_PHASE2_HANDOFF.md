# ADDifT学習機能 引継ぎ仕様書（フェーズ2）

対象: `app/addift_train.py`, `sd-scripts/anima_train_addift.py`

本書はフェーズ1（最小実装・実機検証済み）からフェーズ2（段階実装）への引継ぎ資料。
フェーズ1の成果物（コア学習ロジック）は**動作確認済み**であり、本フェーズでは
GUI内タブの仮設置部分の実装と、timesteps指定方式の改良を行う。


## 1. フェーズ1完了状況（実機検証済み）

### 1.1 動作確認結果
- モデルロード（Anima DiT / VAE / Qwen3）、LoRA構築、optimizer設定まで正常動作。
- 学習ループ（forward/backward/optimizer.step）が50step完走。
- LoRA適用後、出力画像に実際の差分（瞳の色変化）が観測され、
  ADDifTの交互ターン・multiplier切替・損失計算ロジックが機能していることを実証済み。
- **コア学習機構の実装は成功と判断**。残課題はチューニング（マスク・caption・
  timesteps・階層スケール）であり、これがフェーズ2のスコープ。

### 1.2 修正済みの既知バグ
`anima_train_addift.py` 566行目付近、`predict_noise_anima` 呼び出し前の
timesteps正規化処理で、dtypeキャストが欠落していた。

```python
# 修正前（float32のままDiTへ渡りLoRA(bf16)とdtype不一致でエラー）
timesteps_normalized = timesteps.float() / 1000.0

# 修正後（anima_train_leco.pyと同様にweight_dtypeへキャスト）
timesteps_normalized = (timesteps.float() / 1000.0).to(device, dtype=weight_dtype)
```

このバグは解消済み。フェーズ2作業時、同様のdtype不一致パターン（float32が
そのままAnima.forwardやLoRA層に渡る経路）に注意すること。


## 2. 実機検証から得られた知見（チューニング指針）

フェーズ2のUI実装（特にtimestepsプリセット）はこの知見に基づく。

### 2.1 Rectified Flow timestepsの意味
`sigma = timesteps / 1000`、`noisy = sigma*noise + (1-sigma)*latent`

| timesteps範囲 | sigma | 支配的成分 | 影響範囲 |
|---|---|---|---|
| 低 (0付近) | 小 | latent (元画像) | 細部・質感・色・局所変化 |
| 高 (1000付近) | 大 | noise | 大域的な構図・形状・ポーズ |

### 2.2 TrainTrain準拠プリセット値（アルゴリズム.txt由来）
| プリセット | train_min/max_timesteps | network_strength | 用途 |
|---|---|---|---|
| Style（画風/質感/色/局所） | 200-400 | 5.0 | 局所的な色・質感変化（例: 瞳の色） |
| Action（構図/ポーズ/構造） | 500-1000 | 1.0 | 構図・ポーズなど大域的変化 |

### 2.3 「画像全体が変わる」問題への対処
- 原因: `network_train_unet_only`でDiT全508モジュールにLoRAを適用しているため、
  局所的な変化を狙ってもmultiplierを上げると他領域にも変化が漏れる。
- 第一の対処: `diff_use_diff_mask` + `diff_mask_path`（実装済み・フェーズ1で提供）。
  マスクは「黒背景+変化させたい部位のみ白」の画像。`load_diff_mask`が
  `convert("L")`でグレースケール化し、latentサイズへリサイズして
  `compute_addift_loss`内で`prediction`/`reference`双方に乗算する。
  解像度はlatentサイズ（VAEダウンサンプル比）まで縮小されるため、
  対象部位が小さい場合はマスクを広めに塗ることを推奨する旨をUI上に注記済み
  （`addift_diff_mask_note`）。
- 第二の対処（フェーズ2の階層学習）: 顔・目周辺に対応する層のみ高スケール、
  他層を低スケールにする階層別重み付け。


## 3. フェーズ2 実装スコープ

以下4タブを仮設置（プレースホルダ）からフル実装へ移行する。
すべて `app/addift_train.py` 内に実装し、`leco_train.py` / `lora_train.py` の
既存実装パターンを最大限流用すること（過去セッションで構造確認済み）。

### 3.1 階層学習タブ（`_build_layer_train_tab_stub` → フル実装）
- 参照実装: `leco_train.py` の `_build_layer_train_tab`,
  `_layer_group_names`, `_refresh_layer_controls`, `_load_layer_preset`,
  `_layer_scales_to_block_weights`。
- 定数は `addift_train.py` に既に定義済み:
  `LAYER_TRAIN_MODES = ("Matrix", "Transformer", "Component")`,
  `MATRIX_BLOCKS = ("Input", "Middle", "Output")`,
  `MATRIX_COMPONENTS = ("Attention", "MLP", "Norm", "ResNet", "Timestep")`,
  `COMPONENT_GROUPS`, `LAYER_COLUMNS`。
- `_AddifTTrainState` に既存の `layer_train_enabled`, `layer_display_mode`,
  `layer_parameter_vars`, `layer_canvas`, `layer_inner` を使用。
- `_build_command`内に既存の「階層学習」ブロックがあるが、現状`Matrix`モードの
  `anima_matrix_scales`のみ対応。`Transformer`/`Component`モード時の
  `anima_block_lr_weight`生成ロジック（`_layer_scales_to_block_weights`相当）を
  `leco_train.py`から移植して追加すること。
- プリセット（`_build_addift_preset_tab`）の`_collect`/`_apply`は
  既に`layer_train_enabled`/`layer_display_mode`/`layer_parameter_vars`を
  保存・復元する形になっているが、`_load`時に`layer_canvas`/`layer_inner`への
  再描画呼び出し（`leco_train.py`の`_load`内処理を参照）が未実装のため追加する。

### 3.2 モニターグラフタブ（`_build_monitor_tab_stub` → フル実装）
- 参照実装: `leco_train.py` の `_build_monitor_tab`（`monitor_graph_leco.py`の
  `LecoMonitorGraph`を動的importして組み込む形）。
- ADDifT用に `monitor_graph_addift.py`（新規）を作成し、`LecoMonitorGraph`相当の
  クラスを実装。学習ログ（`s._monitor_queue`、実装済み）からloss/lrをパースして
  グラフ描画する。
- **注意**: 現在の`_build_monitor_tab_stub`はgrid専用で実装済み
  （`pack`との混在エラーを修正済み）。グラフ領域は`row=0`、ログ欄は`row=1`に
  配置する既存のgrid構成を維持すること。

### 3.3 モニター階層タブ（`_build_monitor_layer_tab_stub` → フル実装）
- 参照実装: `leco_train.py` の `_build_monitor_layer_tab`,
  `_group_names_for_mode_leco`（`monitor_layer.py`の`MonitorLayerGraph`を
  動的importして組み込む）。
- ADDifT用に `_group_names_for_mode_addift` を実装し、3.1で実装する
  `layer_display_mode`（Matrix/Transformer/Component）に対応するグループ名
  リストを返す（`leco_train.py`の`_group_names_for_mode_leco`と同一仕様）。
- `s._monitor_layer_queue`（実装済み）を入力ソースとする。

### 3.4 サンプル生成タブ（`_build_sample_tab_stub` → フル実装）
- 参照実装: `leco_train.py` の `_build_leco_sample_tab`,
  `_build_leco_sample_tab_inline`, `_leco_write_sample_prompt_file`,
  `_leco_build_prompt_line`。
- `_AddifTTrainState`には既にサンプル関連変数が定義済み
  （`sample_every_n_steps`, `sample_width/height/steps/scale/flow_shift`,
  `sample_keep_vae`, `sample_enabled/prompt/negative_prompt`,
  `sample_b_enabled/prompt/negative_prompt`）。
- **ADDifT特有の検討事項**: LECOはプロンプトA/Bの2系統サンプルだったが、
  ADDifTは画像A/B＋共通captionの構造。サンプル生成の意味付け
  （例: 「A: captionのみ」「B: caption+LoRA適用後」の比較表示にするか、
  従来通りA/B独立プロンプトにするか）は次セクション開始時に方針を確認すること。
- `anima_train_addift.py`側は`anima_sample_gen.sample_images_from_prompts`を
  既に呼び出す実装が入っている（`--sample_every_n_steps`, `--sample_prompts`,
  `--sample_save_dir`, `--sample_keep_vae`に対応済み）。`_build_command`への
  これらの引数追加もフェーズ2で行う（現状未追加）。


## 4. timesteps プリセット選択方式（新規要件）

### 4.1 要件
学習設定タブの `train_min_timesteps` / `train_max_timesteps` に加え、
オプションのチェックボックスを新設する。

- チェックボックス有効時: プルダウンで以下3プリセットから選択。
  選択時、`train_min_timesteps`/`train_max_timesteps`（および推奨
  `network_strength`）を自動設定する。
- チェックボックス無効時: 既存の手動スピンボックス入力を維持。

### 4.2 プリセット値（セクション2.2の知見に基づく）

| プリセット名（UI表示） | キー | min | max | network_strength（推奨値） | 用途 |
|---|---|---|---|---|---|
| 局所 | `local` | 100 | 300 | 5.0 | 瞳の色・小物・局所的な質感変化 |
| 画風 | `style` | 200 | 400 | 5.0 | 全体的な色調・画風転換 |
| 構図 | `composition` | 500 | 1000 | 1.0 | ポーズ・構図・構造的変化 |

> 「局所」は本セッションの実機検証（瞳の白目化）から導出した新規プリセットで、
> TrainTrainの`Style`プリセット(200-400)よりさらに低いtimesteps域
> (100-300)を割り当てる。「画風」はTrainTrain `Style`相当(200-400)、
> 「構図」はTrainTrain `Action`相当(500-1000)に対応させる。

### 4.3 実装方針
- `_AddifTTrainState`に追加する変数:
  ```python
  self.timesteps_preset_enabled = tk.BooleanVar(value=False)
  self.timesteps_preset_name    = tk.StringVar(value="local")  # local/style/composition
  ```
- プリセット定義は定数辞書として`addift_train.py`上部に追加:
  ```python
  TIMESTEPS_PRESETS: dict[str, dict] = {
      "local":       {"min": 100, "max": 300,  "strength": 5.0, "label_key": "addift_timesteps_preset_local"},
      "style":       {"min": 200, "max": 400,  "strength": 5.0, "label_key": "addift_timesteps_preset_style"},
      "composition": {"min": 500, "max": 1000, "strength": 1.0, "label_key": "addift_timesteps_preset_composition"},
  }
  ```
- `_build_train_tab`内のADDifTパラメータ部（`train_min_timesteps`/
  `train_max_timesteps`/`network_strength`のSpinbox/Entry群）の上に
  チェックボックス＋Comboboxを追加。
  - チェック有効時: 既存のSpinbox/Entryを`state=tk.DISABLED`にし、
    プリセット選択時に`train_min_timesteps.set(...)`等で値を反映。
  - チェック無効時: 既存Spinbox/Entryを`state=tk.NORMAL`に戻す
    （手動編集可能、直前のプリセット値が初期値として残る）。
- `_build_command`は最終的に`s.train_min_timesteps`/`s.train_max_timesteps`/
  `s.network_strength`の値（プリセット選択時はプリセット値が反映済み）を
  そのまま使うため、コマンド生成側の変更は不要。
- プリセット（`_collect`/`_apply`）に`timesteps_preset_enabled`/
  `timesteps_preset_name`を追加保存。

### 4.4 必要な翻訳キー追加（`texts_ja.json`）
```
addift_timesteps_preset_enable      : "プリセットから選択する"
addift_timesteps_preset_label       : "プリセット:"
addift_timesteps_preset_local       : "局所（瞳・小物などの色/質感変化）"
addift_timesteps_preset_style       : "画風（全体の色調/画風転換）"
addift_timesteps_preset_composition : "構図（ポーズ/構造変化）"
```


## 5. 既知の未検証事項（フェーズ1から継続）

優先度は低いが、フェーズ2作業中に問題が出た場合の調査起点として記録する。

1. `qwen_image_autoencoder_kl.AutoencoderKLQwenImage.encode_pixels_to_latents`の
   入力テンソル形状仮定（[-1,1]・NCHW）— 実機では正常動作確認済みのため
   現状問題なしと判断。
2. `networks.lora`の`set_multiplier`の負値対応 — 実機で`-0.25*diff_alt_ratio`
   相当の負multiplierが使用される`turn=False`時の動作は、50step学習が
   エラーなく完走しているため問題ないと判断できる。


## 6. ファイル構成サマリ（フェーズ2開始時点）

| ファイル | 状態 | フェーズ2での扱い |
|---|---|---|
| `app/addift_train.py` | フェーズ1完成・dtype修正反映済み | 3.1〜3.4, 4節を追記実装 |
| `sd-scripts/anima_train_addift.py` | フェーズ1完成・dtype修正済み・実機検証済み | 3.4のサンプル生成引数を`_build_command`から渡す前提で、
スクリプト側は追加修正不要（既に対応済み） |
| `app/gui.py` | パッチ適用済み（ADDifTタブ登録・終了時クリーンアップ） | 変更不要 |
| `app/texts_ja.json` | フェーズ1キー追加済み | 4.4のキーおよび3節の各タブ用キーを追加 |
| `monitor_graph_addift.py` | 未作成 | 3.2で新規作成（`monitor_graph_leco.py`参照） |


## 7. 次セクション開始時の確認事項

1. 3.4（サンプル生成）のA/B表示意味付けの方針確認。
2. 階層学習プリセット（`_load_layer_preset`相当）をADDifT用に新規作成するか、
   LECO用プリセットを共有するかの確認。
3. `monitor_graph_addift.py` / モニター階層用グループ名関数の新規作成可否
   （`leco_train.py`依存モジュールをそのままimportできるかの環境確認）。
