# LoRA Train / GUI 作業引き継ぎメモ（セッション6）
作成日: 2026-05-26

---

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10)

---

## セッション5からの引き継ぎ（完了済み・再掲）

| 項目 | ファイル | 内容 |
|------|----------|------|
| 階層学習 GUI実装 Phase1 | `app/lora_train.py` | 階層学習タブ追加 |
| 階層学習 学習側実装 Phase2 | `sd-scripts/networks/lora.py` | anima_block_lr_weight / anima_matrix_scales |
| Phase1/2 接続 | `app/lora_train.py` | `_build_command` 拡張 |
| Component警告・プリセットコンバート | `app/lora_train.py` | 6方向コンバート実装 |
| Matrix精度損失解消 | `lora.py` / `lora_train.py` | anima_matrix_scales 完全実装 |

---

## セッション6完了済み作業

### A. LoRA学習プリセット機能 — **完了**
**パッチファイル:** `lora_train_fix5.py` / `apply_fix_phase5.py`
**対象ファイル:** `app/lora_train.py`

**内タブ最終構成:**

| # | タブ名 | 状態 |
|---|--------|------|
| 1 | モデル | 既存 |
| 2 | データセット | 既存 |
| 3 | ネットワーク | 既存 |
| 4 | 学習設定 | 既存 |
| 5 | 詳細 | 既存 |
| 6 | 階層学習 | セッション5実装済み |
| 7 | モニターグラフ | PH（空フレーム）|
| 8 | プリセット | 実装済み |

**プリセット機能仕様:**
- 保存先: `preset/lora_train/*.json`
- 操作: 保存 / 読み込み / 削除 / エクスポート / インポート / 一覧更新
- 保存対象: `_TrainState` の全 `tk.Variable`（階層学習スケール含む）

---

### B. プリセット読み込み時の階層学習先行設定 — **完了**
**パッチファイル:** `lora_train_fix6.py` / `apply_fix_phase6.py`
**対象ファイル:** `app/lora_train.py`

**実装内容（3パッチ）:**
- `_TrainState` に `layer_canvas: tk.Canvas | None` / `layer_inner: ttk.Frame | None` を追加
- `_build_layer_train_tab` で上記フィールドへ参照を格納
- `_load` 内で `_apply` 前に `layer_train_enabled` → `layer_display_mode` → `_refresh_layer_controls` を順に実行し `layer_parameter_vars` を生成してからスケール値を反映

---

### C. SyntaxWarning 修正 — **完了**
**パッチファイル:** `strategy_base_fix.py` / `apply_fix_strategy_base.py`
**対象ファイル:** `sd-scripts/library/strategy_base.py`

**修正内容:**
- docstring 内の `\(` `\[` `\)` `\]` `\\` を `\\(` `\\[` `\\)` `\\]` `\\\\` に変更
- Python 3.12 の SyntaxWarning（line 96）を解消

---

### D. Validation Loss / EarlyStopping 実装 — **完了**

#### D-1. GUI側
**パッチファイル:** `lora_train_fix7.py` / `apply_fix_phase7.py`（初回）、`lora_train_fix8.py` / `apply_fix_phase8.py`（プリセット追加）、`lora_train_fix9.py` / `apply_fix_phase9.py`（ラベル日本語化）
**対象ファイル:** `app/lora_train.py`

**追加変数（`_TrainState`）:**
```python
self.validation_split          = tk.StringVar(value="0.0")
self.early_stopping            = tk.BooleanVar(value=False)
self.early_stopping_mode       = tk.StringVar(value="epoch")
self.early_stopping_patience   = tk.IntVar(value=3)
self.early_stopping_threshold  = tk.DoubleVar(value=0.01)
```

**GUI項目（`_build_adv_tab` 末尾 LabelFrame「Validation / Early Stopping」）:**

| 項目 | 内容 |
|------|------|
| 検証データ分割比率 | `validation_split`（0.0=無効、0.1=10%分割） |
| Early Stopping を有効にする | チェックボックス（デフォルト: OFF） |
| 判定タイミング | epoch / step 選択 Combobox |
| 連続悪化の許容回数 | Spinbox（patience） |
| 悪化判定しきい値 | Entry（threshold） |

**`_build_command` 追加 args:**
```
--validation_split <value>
--early_stopping
--early_stopping_mode epoch|step
--early_stopping_patience <int>
--early_stopping_threshold <float>
```

**プリセット保存対象に追加済み（fix8）:**
`validation_split` / `early_stopping` / `early_stopping_mode` / `early_stopping_patience` / `early_stopping_threshold`

#### D-2. 学習側
**パッチファイル:** `train_network_fix.py` / `apply_fix_phase7.py`（EarlyStopper本体）
**対象ファイル:** `sd-scripts/train_network.py`

**`EarlyStopper` クラス仕様（`NetworkTrainer` 直前に定義）:**

- 判定基準: Val Loss が `prev_val_loss + threshold` を超えて上昇 → 悪化カウント +1
- Train Loss: 停止判定に使用しない。警告メッセージの文脈情報として使用
- リセット: Val Loss が前回以下に戻った場合にカウントリセット

**警告段階:**

| カウント | Train Loss | ログレベル | メッセージ |
|---------|-----------|-----------|-----------|
| 初回計測 | — | INFO | `[EarlyStopping] 初回計測  val_loss=X  train_loss=Y  監視を開始します。` |
| 正常継続 | — | INFO | `[正常] Val Loss は良好です  val_loss: 前→今  train_loss: 今` |
| 1/N | 下降中 | WARNING | `[注意] … Train Loss は下降中のため過学習の初期兆候の可能性があります。` |
| 1/N | 上昇/横ばい | WARNING | `[注意] … 引き続き監視します。` |
| 中間(2〜N-1) | 下降中 | WARNING | `[警告] … 過学習の可能性が高まっています。` |
| 中間(2〜N-1) | 上昇/横ばい | WARNING | `[警告] … 学習の発散またはデータ不足の可能性があります。` |
| N以上 | 下降中 | WARNING | `[緊急停止] … 過学習と判断し学習を停止します。` |
| N以上 | 上昇/横ばい | WARNING | `[緊急停止] … 学習の収束失敗と判断し学習を停止します。` |
| カウントリセット | — | WARNING | `[監視正常化] … 通常監視へ移行します。` |

**epoch内集計変数（累積平均問題の解消）:**
```python
# epoch先頭でリセット
_epoch_train_loss_sum   = 0.0
_epoch_train_loss_count = 0
# epoch validation開始前にリセット
_epoch_val_loss_sum   = 0.0
_epoch_val_loss_count = 0
# EarlyStopper に渡す値
_es_val   = _epoch_val_loss_sum   / max(_epoch_val_loss_count,   1)
_es_train = _epoch_train_loss_sum / max(_epoch_train_loss_count, 1)
```

**`setup_parser` 追加引数:**
```
--early_stopping           (store_true)
--early_stopping_mode      epoch|step (default: epoch)
--early_stopping_patience  int (default: 3)
--early_stopping_threshold float (default: 0.01)
```

---

### E. バグ修正・品質改善 — **完了**

#### E-1. `validation_steps` UnboundLocalError
**パッチファイル:** `train_network_fix2.py` / `apply_fix_tn2.py`
- EarlyStopper インスタンス生成ブロックから `validation_steps == 0` チェックを分離
- `validation_steps` 定義直後に移動

#### E-2. ANSIエスケープ除去 / EarlyStopping改行修正 / epoch内Loss集計
**パッチファイル:** `fix_log_display.py` / `apply_fix_log_display.py`
- `lora_train.py`: `proc.stdout` 読み取り時に `re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)` を適用（`[A[A` 等を除去）
- `train_network.py`: `accelerator.print` 前に `"\n"` を挿入（プログレスバー末尾への連結を防止）
- `train_network.py`: epoch内 train/val loss 集計変数を追加（累積平均問題を解消）

#### E-3. epoch ES の `_es_val` 未変更バグ
**パッチファイル:** `fix_es_val.py` / `apply_fix_es_val.py`
- `_es_val = val_epoch_loss_recorder.moving_average`（累積）→ `_epoch_val_loss_sum / max(_epoch_val_loss_count, 1)`（epoch内平均）に修正

#### E-4. epoch validation 二重表示解消 / 正常動作通知追加
**パッチファイル:** `fix_val_display.py` / `apply_fix_val_display.py`
- epoch validation ループ開始前に `val_epoch_loss_recorder = train_util.LossRecorder()` を再生成
- EarlyStopper 正常時（`_count==0` かつ `val_improved`）に `[正常]` メッセージを追加

#### E-5. EarlyStopping ログレベル修正
**パッチファイル:** `fix_es_loglevel.py` / `apply_fix_es_loglevel.py`
- `[正常]` `[EarlyStopping]` → `logger.info`
- `[注意]` `[警告]` `[緊急停止]` `[監視正常化]` → `logger.warning`（従来通り）

---

## パッチ適用順序（セッション6の全パッチ）

```
# app/lora_train.py 系
python apply_fix_phase5.py          # プリセットタブ + モニターグラフPH
python apply_fix_phase6.py          # プリセット読み込み時の階層学習先行設定
python apply_fix_phase7.py          # EarlyStopping GUI + train_network.py EarlyStopper本体
python apply_fix_phase8.py          # プリセット保存対象にEarlyStopping追加
python apply_fix_phase9.py          # GUI ラベル日本語化 / EarlyStopper両値表示

# sd-scripts/train_network.py 系（apply_fix_phase7.py も含む）
python apply_fix_tn2.py             # validation_steps UnboundLocalError 修正
python apply_fix_log_display.py     # ANSIエスケープ除去 / 改行 / epoch内集計
python apply_fix_es_val.py          # epoch ES _es_val epoch内平均に修正
python apply_fix_val_display.py     # 二重表示解消 / [正常]通知追加
python apply_fix_es_loglevel.py     # ログレベル INFO/WARNING 分岐

# sd-scripts/library/strategy_base.py
python apply_fix_strategy_base.py   # SyntaxWarning 修正
```

---

## 残存する既知の制約

| 項目 | 内容 | 解消方法 |
|------|------|----------|
| Component モードのブロック別精度 | 構造上の限界。GUI に赤字警告表示済み | 解消不可 |
| ログの折り返し | GUIの `Text` ウィジェット幅による | UI改修時（次セクション）に対応 |
| `forrtl: error (200)` | ユーザーの停止ボタン操作による正常停止。Fortranランタイムのクラッシュ出力 | 動作上問題なし |

---

## 次のセッションで実施予定の機能

### 優先度高

#### F. モニターグラフ実装 — 未着手
- 現在はプレースホルダ（空フレーム + テキスト）
- `_drain()` の 200ms ポーリングを利用してログから正規表現でパース
- 表示対象:
  - Train Loss（`avr_loss=X` をパース）
  - Val Loss（`val_epoch_avg_loss=X` をパース）
  - EarlyStopping メッセージ（`[正常]` `[注意]` `[警告]` `[緊急停止]` をレポート欄に表示）
  - step / epoch / 予測終了時間
- グラフ描画: `matplotlib` の `FigureCanvasTkAgg` をモニターグラフタブに埋め込み
- X軸: epoch または step、Y軸: loss値、Train/Val を色分けして同一グラフに表示

#### G. UI視認性調整 — 未着手
- ログウィジェットの折り返し改善（幅調整またはワードラップ設定）
- EarlyStopping メッセージの色分け表示（`[注意]`=黄、`[警告]`=橙、`[緊急停止]`=赤、`[正常]`=緑）
- その他レイアウト・フォント等の視認性改善

### 優先度中

#### H. Validation Loss 早期停止オプション — 完了済み（セッション6）

#### I. UI改修・リアルタイム表示（モニターグラフと統合）
- タブ構成: 詳細 → 階層学習 → **モニターグラフ** → プリセット

---

## 関連ファイルと役割

| ファイル | パス | 役割 |
|----------|------|------|
| `gui.py` | `app/gui.py` | メインGUI |
| `lora_train.py` | `app/lora_train.py` | LoRA学習タブ本体。セッション6で大幅修正 |
| `train_network.py` | `sd-scripts/train_network.py` | 学習スクリプト本体。セッション6で大幅修正 |
| `strategy_base.py` | `sd-scripts/library/strategy_base.py` | SyntaxWarning修正済み |
| `lora.py` | `sd-scripts/networks/lora.py` | LoRAネットワーク本体 |
| `anima_train_network.py` | `sd-scripts/anima_train_network.py` | 学習スクリプトエントリポイント |
| `merge.py` | `app/merge.py` | レイヤー分類ロジック |
| `config.py` | `app/config.py` | `MergeOptions` / `AppPaths` 定義 |
| `anima_utils.py` | `sd-scripts/library/anima_utils.py` | DiTモデルロード・キー正規化 |

---

## パッチ適用ルール（セッション共通）

- 修正は `apply_fix*.py`（Python純正・`patch` コマンド不要）で適用
- Apache-2.0 ライセンス準拠のため、変更ファイル冒頭に以下形式のコメントを付与:
  ```python
  # 変更日: YYYY-MM-DD  変更内容の簡潔な説明
  ```
- パッチ内の差分文字列は CRLF/LF 両対応（`_adapt()` 関数で吸収）
- `lora.py` / `strategy_base.py` は UTF-8 BOM 付きのため `read_text(encoding="utf-8-sig")` で読み込み、`write_text(encoding="utf-8")` で書き戻す
- 絶対パス禁止・相対パス使用。Windowsユーザー名がパスに入り込まないよう注意
