# anima(cosmos) モデル編集ツール 引き継ぎ手順書

作成日: 2026-06-03  
対象ブランチ: 現行開発中  

---

## 1. 未解決バグ一覧と修正方針

### BUG-01: LoRAサンプルギャラリーにA/Bが表示されない

**原因**  
sd-scriptsの実際のファイル名規則が想定と異なる。

| 想定 | 実際 |
|------|------|
| `{name}_e{epoch:06d}_{idx:02d}.png` | `{name}_e{epoch:06d}_{idx:02d}_{timestamp}_{seed}.png` |

例: `lora_output_e000001_01_20260603124939_42.png`

現行のglobパターン `*_00.png` / `*_01.png` はタイムスタンプとseedが後置されるためマッチしない。

**修正内容: `lora_train.py`**  
`_build_sample_tab_common` 内のglobパターンを変更する。

```python
# 変更前
pat_a = "*_a_*.png" if is_leco else "*_00.png"
pat_b = "*_b_*.png" if is_leco else "*_01.png"

# 変更後
pat_a = "*_a_*.png" if is_leco else "*_e*_00_*.png"
pat_b = "*_b_*.png" if is_leco else "*_e*_01_*.png"
```

**対象ファイル:** `app/lora_train.py`  
**対象パッチ:** `apply_fix_sample_filenames.py` の P7 を上記に修正して再適用

---

### BUG-02: LECOサンプル生成が失敗する (`'dict' object has no attribute 'sample'`)

**原因**  
`qwen_image_autoencoder_kl` の `vae.decode()` は `diffusers` の `AutoencoderKL` と異なり、戻り値が `DecoderOutput` オブジェクトではなく **辞書 (`dict`)** を返す。  
`vae.decode(latents_in).sample` の `.sample` アクセスが失敗する。

実際の戻り値形式は以下のいずれか（要実機確認）:
- `{"sample": tensor}` → `result["sample"]`
- `{"frames": tensor}` → `result["frames"]`
- テンソル直返し → そのまま使用

**修正内容: `anima_train_leco.py`**  
784行目を辞書・テンソル両対応に変更する。

```python
# 変更前
decoded = vae.decode(latents_in).sample

# 変更後
_dec = vae.decode(latents_in)
if isinstance(_dec, dict):
    decoded = _dec.get("sample", _dec.get("frames", next(iter(_dec.values()))))
else:
    decoded = _dec.sample if hasattr(_dec, "sample") else _dec
```

**対象ファイル:** `sd-scripts/anima_train_leco.py`  
**確認手順:** 修正後に実行してデコードが成功するか確認。  
失敗する場合は `qwen_image_autoencoder_kl.py` の `decode()` 定義を直接確認して戻り値のキー名を特定すること。

---

### BUG-03: LECOサンプル生成でプロンプトが反映されない

**原因**  
`_parse_sample_prompt_line` の `--n` 抽出正規表現が `[ \t]--n[ \t]+` になっており、`--n` の前後スペースが必須。  
`_leco_build_prompt_line` が生成する行は `... --d {seed} --n {neg}` の形式で末尾に `--n` が来るが、その後にフラグが続かないため `(?=[ \t]+--[a-z]|[ \t]*$)` の終端マッチが正しく動作していない可能性がある。

また `--l` (scale) と `--fs` (flow_shift) の値を `_parse_sample_prompt_line` が正しく取れているか未検証。

**調査手順**  
`anima_train_leco.py` の `_generate_samples_leco` 内に以下のデバッグログを一時追加して確認する。

```python
prompt_text, gen_kwargs = _parse_sample_prompt_line(line)
logger.info(f"[SampleGen] parsed: prompt={prompt_text!r} kwargs={gen_kwargs}")
```

**修正内容: `anima_train_leco.py`**  
`_parse_sample_prompt_line` の `neg_pat` を以下に変更する。

```python
# 変更前
neg_pat = __import__("re").search(r"[ \t]--n[ \t]+(.+?)(?=[ \t]+--[a-z]|[ \t]*$)", line)

# 変更後
neg_pat = re.search(r"\s--n\s+(.+?)(?:\s+--[a-zA-Z]|\s*$)", line)
```

**対象ファイル:** `sd-scripts/anima_train_leco.py`

---

## 2. 未実装機能: モニターグラフ移植

### 概要

`lora_train.py` に実装済みのモニターグラフ機能を `leco_train.py` に移植する。  
現在 `leco_train.py` の「モニターグラフ」「モニター階層」タブは空プレースホルダー。

### 移植元ファイル

| ファイル | 役割 |
|----------|------|
| `app/monitor_graph.py` | Loss/LRグラフウィジェット |
| `app/monitor_layer.py` | 階層別実効LRウィジェット |
| `app/lora_train.py` | `_build_monitor_tab` / `_build_monitor_layer_tab` の実装 |

### 移植手順

#### STEP 1: lora_train.py から対象関数を確認

```python
# lora_train.py 内の以下を確認
def _build_monitor_tab(parent: ttk.Frame, s: _TrainState) -> None: ...
def _build_monitor_layer_tab(parent: ttk.Frame, s: _TrainState) -> None: ...
```

`_TrainState` から参照している変数のうち `_LecoTrainState` に不足しているものを洗い出す。

#### STEP 2: _LecoTrainState への変数追加確認

フェーズ2パッチ (`apply_fix_leco_train2.py`) で以下は追加済み:

```python
self._monitor_queue:       queue.Queue[str] = queue.Queue()
self._monitor_layer_queue: queue.Queue[str] = queue.Queue()
self.layer_train_enabled   = tk.BooleanVar(value=False)
self.layer_display_mode    = tk.StringVar(value="Matrix")
self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
```

不足している変数があれば `_LecoTrainState.__init__` に追記する。

#### STEP 3: leco_train.py の空タブ実装を置換

```python
# 変更前 (leco_train.py)
def _build_monitor_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    ttk.Label(parent, text="モニターグラフ（実装予定）...").pack(expand=True)

def _build_monitor_layer_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    ttk.Label(parent, text="モニター階層（実装予定）...").pack(expand=True)
```

```python
# 変更後: monitor_graph.py / monitor_layer.py のウィジェットを組み込む
def _build_monitor_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    from .monitor_graph import MonitorGraph
    MonitorGraph(parent, s)

def _build_monitor_layer_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    from .monitor_layer import MonitorLayerGraph
    MonitorLayerGraph(
        parent, s,
        group_names_for_mode=_layer_group_names,
    )
```

#### STEP 4: _worker のキュー送信確認

フェーズ2パッチで以下は適用済み。追加不要。

```python
s._monitor_queue.put(line)
s._monitor_layer_queue.put(line)
```

#### STEP 5: monitor_graph.py が要求するインターフェース確認

`monitor_graph.py` の `MonitorGraph.__init__` が `state` から参照するプロパティを確認し、`_LecoTrainState` との互換性を検証する。  
`lora_train.py` の `_TrainState` と `_LecoTrainState` で異なるプロパティがあれば duck typing で吸収するか、`MonitorGraph` 側にデフォルト値を追加する。

#### STEP 6: パッチ作成

移植内容が確定したら以下のパッチスクリプトを作成する。

```
apply_fix_leco_monitor.py
  - leco_train.py の _build_monitor_tab を MonitorGraph に置換
  - leco_train.py の _build_monitor_layer_tab を MonitorLayerGraph に置換
  - 不足変数があれば _LecoTrainState に追加
```

---

## 3. 適用済みパッチの順序

現在までに作成・適用が必要なパッチの正しい適用順序。

```
1. apply_fix_leco_train2.py          # フェーズ2: 階層学習・プリセットタブ追加
2. apply_fix_sample_ab.py            # サンプルA/B変数・UI追加
3. apply_fix_leco_argdup.py          # argparse重複引数削除
4. apply_fix_leco_funcorder.py       # 関数定義順序修正 (NameError対策)
5. apply_fix_sample_filenames.py     # ファイル名ベース判別・サブディレクトリ廃止
6. apply_fix_sample_filenames2.py    # (作成予定) BUG-01〜03の修正
7. apply_fix_leco_monitor.py         # (作成予定) モニターグラフ移植
```

各パッチはバックアップ (`.bak_*`) を自動生成する。  
適用前に対象ファイルのコミットまたはバックアップを取ること。

---

## 4. 次回セッションで要求するファイル

### BUG-02 修正に必要
- `sd-scripts/library/qwen_image_autoencoder_kl.py`  
  → `decode()` メソッドの戻り値型を確認するため

### モニターグラフ移植に必要
- `app/monitor_graph.py` (アップロード済み)
- `app/monitor_layer.py` (アップロード済み)
- `app/lora_train.py` (最新パッチ適用後のもの)
- `app/leco_train.py` (最新パッチ適用後のもの)

---

## 5. 現在の各ファイルの状態

| ファイル | 状態 |
|----------|------|
| `sd-scripts/anima_train_leco.py` | BUG-02/03未修正 |
| `app/lora_train.py` | BUG-01未修正（globパターン） |
| `app/leco_train.py` | モニタータブ空 |
| `app/monitor_graph.py` | 移植待ち |
| `app/monitor_layer.py` | 移植待ち |
