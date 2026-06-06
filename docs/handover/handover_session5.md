# LoRA Train / GUI 作業引き継ぎメモ（セッション5）
作成日: 2026-05-25

---

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10)

---

## セッション4からの引き継ぎ（完了済み）

### セッション4完了済み（再掲）

| 項目 | ファイル | 内容 |
|------|----------|------|
| 停止ボタン修正 | `lora_train.py` | `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` |
| VRAMアンロード完全対応 | `gui.py` + `lora_train.py` | `build_lora_train_tab` 戻り値を `self._lora_train_state` に代入、`return state` 追加 |
| マージ系プリセット保存先変更 | `gui.py` | `preset/merge/` に統一、tab_type 分離廃止 |
| キー名称正規化 | `anima_utils.py` | `canonical_dit_key()` + `_make_logging_rename_hook()` を追加 |

---

## セッション5完了済み作業

### A. 階層学習 GUI実装（Phase 1）— **完了**
**パッチファイル:** `lora_train_fix.py` / `apply_fix.py`
**対象ファイル:** `app/lora_train.py`

**実装内容:**
- `build_lora_train_tab` に「階層学習」タブを追加（6番目）
- ON/OFF チェックボックス（デフォルト: OFF）
- Matrix / Transformer / Component の3モード切り替え（Combobox）
- 各グループのスライダー＋エントリ UI（スクロール対応）
- プリセット読み込みボタン（`preset/merge/*.json` の `parameter_scales` + `layer_display_mode` を読み込み）
- `_TrainState` に追加した変数:
  ```python
  self.layer_train_enabled = tk.BooleanVar(value=False)
  self.layer_display_mode  = tk.StringVar(value="Matrix")
  self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
  ```

**追加定数（`lora_train.py` モジュールレベル）:**
```python
LAYER_TRAIN_MODES   = ("Matrix", "Transformer", "Component")
MATRIX_BLOCKS       = ("Input", "Middle", "Output")
MATRIX_COMPONENTS   = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS    = ("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")  # Attention含む
LAYER_COLUMNS       = 3
```

---

### B. 階層学習 学習側実装（Phase 2）— **完了**
**パッチファイル:** `lora_fix.py` / `apply_fix_lora.py`（基本実装）
**対象ファイル:** `sd-scripts/networks/lora.py`

**実装内容（lora_fix.py、3パッチ）:**
- `RE_ANIMA_BLOCK` / `get_block_index_anima()` / `parse_anima_block_lr_weight()` を追加
- `parse_block_lr_kwargs()` に `anima_block_lr_weight` 分岐を追加
- `prepare_optimizer_params()` の `block_lr` 分岐に Anima パスを追加

**追加修正（lora_fix2.py / apply_fix_lora2.py）:**
- `create_network()` で `anima_block_lr_weight` 指定時に `block_dims` 生成フロー（SD1.x前提・25要素）をバイパスする修正
- 判定: `len(block_lr_weight) == 29` → `_is_anima_block_lr = True` → `block_dims=None` のまま `LoRANetwork` を構築

---

### C. Phase 1/2 接続（`_build_command` 拡張）— **完了**
**パッチファイル:** `lora_train_phase2_fix.py` / `apply_fix_phase2.py`
**対象ファイル:** `app/lora_train.py`

**実装内容:**
- `_BLOCK_CAT` 定数（`["Input"]*9 + ["Middle"]*10 + ["Output"]*9`）をモジュールレベルに追加
- `_layer_scales_to_block_weights(mode, scales)` 関数を追加
  - Transformer: `blocks.N` → インデックス N へ 1:1 マッピング
  - Matrix: Block カテゴリ内全 Component の平均（後に精度損失として判明・後続修正で解消）
  - Component: 全コンポーネント平均を 28 ブロック全体に適用
- `_build_command` に階層学習ブロックを追加（当初はログ出力のみ → Phase 2 で `--network_args` 出力に更新）

---

### D. Component Attention 欠落修正 / Component 警告表示 / プリセットコンバート — **完了**
**パッチファイル:** `lora_train_fix3.py` / `apply_fix_phase3.py`
**対象ファイル:** `app/lora_train.py`

**修正内容（5パッチ）:**

1. **`COMPONENT_GROUPS` に `"Attention"` を追加**
   - 修正前: `("MLP", "Norm", "ResNet", "Timestep", "Other")`
   - 修正後: `("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")`
   - 効果: Component モードの GUI スライダーに Attention が表示される

2. **Component モード時に赤字警告ラベルを表示**（`_refresh_layer_controls` に追加）
   ```
   ⚠ Component モードはブロック情報をLoRAキーから分解できないため、
     全ブロック共通の平均スケールとして適用されます。
     ブロック別精度が必要な場合は Transformer または Matrix モードを使用してください。
   ```
   ステータスバーにも `[警告] ブロック別精度低下あり・構造限界` を表示

3. **`_convert_preset_scales()` 関数を新規追加**（6方向コンバート）
   - Component → Matrix: `blocks.N_Attention` を Block カテゴリ別に平均 → `Input/Middle/Output_Attention` に変換
   - Component → Transformer: Attention 含む全コンポーネント平均を 28 ブロックに展開
   - Matrix → Component: Block 次元を平均して集約
   - Matrix → Transformer: Block カテゴリ内の全 Component 平均を各ブロックに展開
   - Transformer → Matrix: Block カテゴリ平均を Block_Component 全てに適用
   - Transformer → Component: 28 ブロック全平均を全コンポーネントに適用

4. **`_load_layer_preset` にコンバート処理を追加**
   - プリセット読み込み後、`_convert_preset_scales(scales, preset_mode, gui_mode)` を通してからスライダーに反映

5. **`_layer_scales_to_block_weights` Component 分岐に Attention を含む**（COMPONENT_GROUPS 変更により自動対応）

---

### E. Matrix モード精度損失解消（anima_matrix_scales 実装）— **完了**
**パッチファイル:** `lora_fix3.py` / `apply_fix_lora3.py`（lora.py）、`lora_train_fix4.py` / `apply_fix_phase4.py`（lora_train.py）

#### lora.py 変更内容（6パッチ）

**追加定数・関数（モジュールレベル）:**
```python
_ANIMA_BLOCK_CAT: List[str] = ["Input"] * 9 + ["Middle"] * 10 + ["Output"] * 9

def _component_from_lora_name(lora_name: str) -> str:
    # merge.py の component_category() と同等を lora.py 内に inline 実装
    # Attention / MLP / Norm / ResNet / Timestep / Other を返す
```

**`parse_block_lr_kwargs()` 変更:**
- `anima_matrix_scales` が `kwargs` にある場合、長さ 29 の番兵リスト（全 1.0）を返して `is_anima` 判定を通す
- LR 計算は `get_lr_weight_for_lora()` が担うため `block_lr_weight` の値は使われない

**`create_network()` 変更:**
- `anima_matrix_scales` を JSON パースして `network.set_anima_matrix_scales(scales)` を呼び出す

**`LoRANetwork` 追加フィールド・メソッド:**
```python
self.anima_matrix_scales: Optional[dict] = None

def set_anima_matrix_scales(self, scales: dict) -> None: ...

def get_lr_weight_for_lora(self, lora_name: str) -> float:
    # anima_matrix_scales が設定されている場合:
    #   (Block カテゴリ, Component) のキーでスケールを参照
    # 未設定の場合: get_lr_weight(get_block_index_anima(lora_name)) に委譲
```

**`prepare_optimizer_params()` Anima パス変更:**
- `anima_matrix_scales` 有無で分岐
  - あり: `(block_idx, comp)` でグループ化（最大 28×6=168 グループ）→ `get_lr_weight_for_lora()` で LR 取得
  - なし: `block_idx` 単位グループ化（従来動作・Transformer/Component モード）

#### lora_train.py 変更内容（1パッチ）

**`_build_command` の階層学習ブロックをモード別に分岐:**
```python
if _mode == "Matrix":
    # anima_matrix_scales={"Input_Attention":0.9,...} として JSON で渡す
    cmd += ["--network_args", f"anima_matrix_scales={_scales_json}"]
else:
    # Transformer / Component: 28要素の anima_block_lr_weight として渡す
    cmd += ["--network_args", f"anima_block_lr_weight={weight_str}"]
```

**精度改善の確認済み結果:**
- 旧実装（Block 内 Component 平均）: `Input_Attention=0.9`, `Input_MLP=0.3` → 全モジュール `0.70`（希釈）
- 新実装: `attn_to_q` → `0.90`、`ff_net` → `0.30` と GUI 設定値を直接参照

---

## パッチ適用順序（セッション5の全パッチ）

```
# lora_train.py 系（app/lora_train.py が対象）
python apply_fix.py              # Phase 1: 階層学習GUIタブ
python apply_fix_phase2.py       # Phase 2接続: _build_command拡張
python apply_fix_phase3.py       # Attention修正 / Component警告 / プリセットコンバート
python apply_fix_phase4.py       # Matrix anima_matrix_scales 接続

# lora.py 系（sd-scripts/networks/lora.py が対象）
python apply_fix_lora.py         # Anima block_lr_weight 基本実装
python apply_fix_lora2.py        # block_dims バイパス修正
python apply_fix_lora3.py        # anima_matrix_scales 完全実装
```

---

## 残存する既知の制約

| 項目 | 内容 | 解消方法 |
|------|------|----------|
| Component モードのブロック別精度 | LoRA キー名からブロック情報を分解できないため全ブロック共通スケール | 構造上の限界。GUI に赤字警告を表示済み。解消不可 |
| `strategy_base.py:96` の SyntaxWarning | `\(` エスケープ警告。動作に影響なし | 後回し可 |
| ログの細かい改行 | GUI の `Text` ウィジェット幅折り返しによるもの | UI 改修時（優先度中 E）に対応予定 |

---

## 次のセッションで実施予定の機能

### 優先度中（予定）

#### C. LoRA学習プリセット機能 — 未着手
- 保存先: `E:\Animerge\preset\lora_train\`
- `_TrainState` の全 `tk.Variable` を JSON に保存・復元
- `lora_train.py` の詳細タブ内に新タブ「プリセット」として追加
- 階層学習タブのスライダー値（`layer_parameter_vars`）も保存対象に含める

#### D. Validation Loss・早期停止オプション — 未着手
- `train_network.py` 側に validation dataset サポートが必要（現状プレースホルダ）
- GUI にオプション追加 + `train_network.py` 改修が両方必要

#### E. UI改修・リアルタイム表示 — 未着手
- loss / step / epoch / 予測終了時間を色付き個別窓で表示
- `_drain()` の 200ms ポーリングを利用してログから正規表現でパース
- グラフタブ追加: `matplotlib` の `FigureCanvasTkAgg` を詳細タブ内の新タブに埋め込み
- タブ構成: 詳細 → 右へ「レイヤー学習」「モニターグラフ」「プリセット」を追加
- ログ細かい改行問題もこのタイミングで対応（ウィジェット幅調整）

---

## 関連ファイルと役割

| ファイル | パス | 役割 |
|----------|------|------|
| `gui.py` | `app/gui.py` | メインGUI。レイヤーUI実装の参照元 |
| `lora_train.py` | `app/lora_train.py` | LoRA学習タブ本体。セッション5で大幅修正 |
| `merge.py` | `app/merge.py` | `adjustment_group()` 等のレイヤー分類ロジック定義元 |
| `config.py` | `app/config.py` | `MergeOptions` / `AppPaths` 定義 |
| `anima_utils.py` | `sd-scripts/library/anima_utils.py` | DiTモデルロード・キー正規化 |
| `lora.py` | `sd-scripts/networks/lora.py` | LoRAネットワーク本体。セッション5で大幅修正 |
| `anima_train_network.py` | `sd-scripts/anima_train_network.py` | 学習スクリプト本体 |
| `train_network.py` | `sd-scripts/train_network.py` | `--network_args` 定義元 |

---

## パッチ適用ルール（セッション共通）

- 修正は `apply_fix*.py`（Python純正・`patch` コマンド不要）で適用
- Apache-2.0 ライセンス準拠のため、変更ファイル冒頭に以下形式のコメントを付与:
  ```python
  # 変更日: YYYY-MM-DD  変更内容の簡潔な説明
  ```
- パッチ内の差分文字列は CRLF/LF 両対応（`_adapt()` 関数で吸収）
- `lora.py` は UTF-8 BOM 付きのため `read_text(encoding="utf-8-sig")` で読み込み、`write_text(encoding="utf-8")` で書き戻す
