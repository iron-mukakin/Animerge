# LoRA Train / GUI 作業引き継ぎメモ（セッション4）
作成日: 2026-05-24

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10)

---

## セッション4開始時点の完了済み作業

### 完了済み一覧

| 項目 | ファイル | 内容 |
|------|----------|------|
| 停止ボタン修正 | `lora_train.py` | `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` |
| VRAMアンロード完全対応 | `gui.py` + `lora_train.py` | `build_lora_train_tab` 戻り値を `self._lora_train_state` に代入、`return state` 追加 |
| マージ系プリセット保存先変更 | `gui.py` | `preset/merge/` に統一、tab_type 分離廃止 |
| キー名称正規化 | `anima_utils.py` | `canonical_dit_key()` + `_make_logging_rename_hook()` を追加、`net.` のみ除去から全プレフィックス正規化へ変更 |

### キー名称正規化の動作確認済み内容（ログ: 20260524_111733.txt）
- チェックポイント `cottonanima_base1.safetensors` は `model.diffusion_model.` プレフィックス付き形式
- 685キーが正規化され `load_state_dict` の missing/unexpected ともに 0 で正常ロード確認済み

---

## 未対処の課題

### 備考
- `strategy_base.py:96` の SyntaxWarning（`\(` エスケープ）は動作に影響なし、後回し可
- ログの細かい改行はGUIの `Text` ウィジェット幅折り返しによるもの（コード側の問題ではない）→ UI改修時に対応予定
- 学習の正常動作確認済み（`create LoRA for U-Net: 508 modules`、epoch/step/loss推移正常）

---

## 次のセッションで実施予定の機能

### 優先度高

#### A. 階層学習 GUI実装（Phase 1）— 未着手
**対象ファイル:** `lora_train.py`

**実装内容:**
- `build_lora_train_tab` 内のタブ構成に「階層学習」タブを追加
- ON/OFFチェックボックス（デフォルト: OFF）
- Matrix / Transformer / Component の3モード切り替え（`gui.py` の `rebuild_parameter_controls` と同構造）
- 各グループのスライダー＋エントリUI
- プリセット読み込み：`preset/merge/*.json` の `parameter_scales` + `layer_display_mode` を共用
- 実行時ログに `[LayerLR]` プレフィックスで設定値を出力（実行ログで確認できるようにする）

**GUI定数（`gui.py` から流用）:**
```python
MATRIX_BLOCKS     = ("Input", "Middle", "Output")
MATRIX_COMPONENTS = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS  = ("MLP", "Norm", "ResNet", "Timestep", "Other")
```

**グループ名生成ロジック:** `merge.py` の `adjustment_group()` と同一。`lora_train.py` 内では `merge.adjustment_group` を直接 import して使用する。

**プリセット共用仕様:**
- 保存先: `E:\Animerge\preset\merge\*.json`（マージ側と同一ディレクトリ）
- JSON内の `parameter_scales` と `layer_display_mode` のみ読み込む
- LoRA学習固有の設定（alpha等）はプリセットに含まれていても無視
- 保存はしない（読み込みのみ）。LoRA学習用プリセット保存は優先度中のCで対応

**`_TrainState` への追加変数:**
```python
self.layer_train_enabled = tk.BooleanVar(value=False)  # 階層学習ON/OFF
self.layer_display_mode  = tk.StringVar(value="Matrix")
self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}  # グループ名 -> 値
```

**`_build_command` への追加（Phase 1では `--network_args` として渡す準備のみ、実際の学習側適用はPhase 2）:**
```python
# 階層学習が有効な場合、ログ出力のみ行う（Phase 1）
if s.layer_train_enabled.get():
    scales = {k: v.get() for k, v in s.layer_parameter_vars.items()}
    # [LayerLR] プレフィックスでログ出力
    log_fn(f"[LayerLR] mode={s.layer_display_mode.get()}, scales={scales}")
```

---

#### B. 階層学習 学習側実装（Phase 2）— 設計済み・未着手
**対象ファイル:** `lora.py`

**背景・制約:**
`lora.py` の既存 `get_block_index()` は `up_blocks_N` / `down_blocks_N` / `mid_block_` という SD1.x/SDXL 形式のキー名を前提としている（`RE_UPDOWN` パターン）。Anima のキーは `blocks.0` ～ `blocks.27`（28ブロック）形式のため `block_idx = -1` になり、既存の `down_lr_weight` / `up_lr_weight` / `mid_lr_weight` 機構はそのままでは使えない。

**実装内容:**
1. `lora.py` に `get_block_index_anima(lora_name)` を追加
   - `lora_unet_blocks_N_` パターンを検出して `blocks.N` の N を返す
   - Anima の28ブロック（blocks.0 ～ blocks.27）に対応
2. `create_network` の `kwargs` に `anima_block_lr_weight` キーを追加
   - `network_args "anima_block_lr_weight=0,1,1,...,1"` の形式で渡す
3. `set_block_lr_weight` の前段で Anima 形式を検出し既存パスと分岐

**`network_args` の渡し方（`_build_command` 追加分）:**
```python
# Phase 2 実装時に追加
if s.layer_train_enabled.get():
    weight_str = ",".join(str(v.get()) for v in s.layer_parameter_vars.values())
    cmd += ["--network_args", f"anima_block_lr_weight={weight_str}"]
```

**必要な確認:** `anima_train_network.py` が `train_network.setup_parser()` を継承しており、`--network_args` は `train_network` 側で定義されているはず。Phase 2着手前に `train_network.py` を確認すること。

---

### 優先度中（予定）

#### C. LoRA学習プリセット機能 — 未着手
- 保存先: `E:\Animerge\preset\lora_train\`
- `_TrainState` の全 `tk.Variable` を JSON に保存・復元
- `lora_train.py` の詳細タブ内に新タブ「プリセット」として追加
- 階層学習タブのスライダー値も保存対象に含める

#### D. Validation Loss・早期停止オプション — 未着手
- `train_network.py` 側に validation dataset サポートが必要（現状プレースホルダ）
- GUIにオプション追加 + `train_network.py` 改修が両方必要

#### E. UI改修・リアルタイム表示 — 未着手
- loss / step / epoch / 予測終了時間を色付き個別窓で表示
- `_drain()` の200ms ポーリングを利用してログから正規表現でパース
- グラフタブ追加: `matplotlib` の `FigureCanvasTkAgg` を詳細タブ内の新タブに埋め込み
- タブ構成: 詳細 → 右へ「レイヤー学習」「モニターグラフ」「プリセット」を追加
- ログ細かい改行問題もこのタイミングで対応（ウィジェット幅調整）

---

## 関連ファイルと役割

| ファイル | パス | 役割 |
|----------|------|------|
| `gui.py` | `app/gui.py` | メインGUI。レイヤーUI実装の参照元 |
| `lora_train.py` | `app/lora_train.py` | LoRA学習タブ本体。Phase 1の主な編集対象 |
| `merge.py` | `app/merge.py` | `adjustment_group()` 等のレイヤー分類ロジック定義元 |
| `config.py` | `app/config.py` | `MergeOptions` / `AppPaths` 定義 |
| `anima_utils.py` | `sd-scripts/library/anima_utils.py` | DiTモデルロード・キー正規化 |
| `lora.py` | `sd-scripts/networks/lora.py` | LoRAネットワーク本体。Phase 2の編集対象 |
| `anima_train_network.py` | `sd-scripts/anima_train_network.py` | 学習スクリプト本体 |
| `train_network.py` | `sd-scripts/train_network.py` | `--network_args` 定義元（Phase 2着手前に確認必要） |

---

## パッチ適用ルール（セッション共通）

- 修正は `apply_fix.py`（Python純正・`patch` コマンド不要）で適用
- Apache-2.0 ライセンス準拠のため、変更ファイル冒頭に以下形式のコメントを付与:
  ```python
  # 変更日: YYYY-MM-DD  変更内容の簡潔な説明
  ```
- パッチ内の差分文字列は CRLF/LF 両対応（`_adapt()` 関数で吸収）
