# anima(cosmos) モデル編集ツール 引き継ぎ手順書

作成日: 2026-06-03
対象ブランチ: 現行開発中

---

## 1. プロジェクト構成

```
プロジェクトルート/
├── app/
│   ├── gui.py
│   ├── lora_train.py          # LoRA学習GUI・コマンド生成
│   ├── leco_train.py          # LECO学習GUI・コマンド生成
│   ├── monitor_graph.py       # Loss/LRグラフウィジェット
│   └── monitor_layer.py       # 階層別実効LRウィジェット
└── sd-scripts/
    ├── anima_train_network.py  # LoRA学習エントリ（AnimaNetworkTrainer）
    ├── anima_train_leco.py     # LECO学習エントリ・サンプル生成
    ├── anima_train_utils.py    # サンプル生成共通実装（do_sample, sample_images）
    └── library/
        └── train_util.py       # line_to_prompt_dict, load_prompts
```

---

## 2. サンプル生成の全経路（LoRA / LECO共通）

### 2-1. プロンプトファイル書き込み（GUI側）

**LoRA:** `app/lora_train.py`

| 関数 | 行 | 役割 |
|------|----|------|
| `SAMPLE_FIXED_SEED` | L42 | seed定数 = 42 |
| `_build_sample_prompt_line_for(prompt, neg, s, seed)` | L990 | 1行のプロンプト行を生成 |
| `_write_sample_prompt_file(s)` | L1016 | `_sample_prompt.txt` を書き込む |

生成フォーマット:
```
{prompt} --w {width} --h {height} --s {steps} --l {scale:g} --fs {flow_shift:g} --d {seed} [--n {neg}]
```

**LECO:** `app/leco_train.py`

| 関数 | 行 | 役割 |
|------|----|------|
| `SAMPLE_FIXED_SEED` | L1519 | seed定数 = 42（関数定義の後に配置されている点に注意） |
| `_leco_build_prompt_line(prompt, neg, s, seed)` | L1502 | 1行のプロンプト行を生成 |
| `_leco_write_sample_prompt_file(s)` | L1522 | `_sample_prompt.txt` を書き込む |

生成フォーマットは LoRA と同一。

**重要:** `SAMPLE_FIXED_SEED` が `_leco_build_prompt_line` の定義より後にある。
関数シグネチャのデフォルト引数に `SAMPLE_FIXED_SEED` を使うと `NameError` になる。
デフォルト引数にはリテラル `42` を使うこと。

### 2-2. コマンドライン引数への渡し方

**LoRA（`lora_train.py` L1887〜）:**
```python
cmd += ["--sample_every_n_epochs", ...]
cmd += ["--sample_prompts", str(sample_prompt_file)]
cmd += ["--sample_save_dir", str(_sample_dir(s))]
```

**LECO（`leco_train.py` L707〜）:**
```python
# --sample_every_n_steps を使用（エポックではなくステップ単位）
```

### 2-3. プロンプトファイル読み込み・パース

**LoRA経路:** `train_util.load_prompts()` (L6497) → `line_to_prompt_dict()` (L6412)

`line_to_prompt_dict` の動作:
- `line.split(" --")` で分割
- 各トークンを正規表現でパース

| フラグ | キー | 型 |
|--------|------|----|
| `--w N` | `width` | int |
| `--h N` | `height` | int |
| `--s N` | `sample_steps` | int |
| `--l N` | `scale` | float |
| `--fs N` | `flow_shift` | str（`do_sample` 内で `float()` に変換） |
| `--d N` | `seed` | int |
| `--n TEXT` | `negative_prompt` | str |

**LECO経路:** `_parse_sample_prompt_line()` (anima_train_leco.py L493)

BUG-03で修正済み。パース結果は `do_sample` に直接渡される。

### 2-4. TEキャッシュ（LoRA）

`anima_train_network.py` L188〜205:
- 学習開始時に `train_util.load_prompts()` でプロンプトを読み込み
- 各プロンプト文字列の TE出力を `sample_prompts_te_outputs` にキャッシュ
- 以降のエポックでは `_sample_image_inference` の `encode_prompt()` がキャッシュを優先使用

**→ 学習開始後にGUIでプロンプトを変更しても反映されない。学習の再起動が必要。**

### 2-5. 画像生成

`anima_train_utils._sample_image_inference()` (L485) → `do_sample()` (L310)

- `do_sample` 内: `torch.manual_seed(seed)` → `torch.randn(...)` で初期ノイズ生成
- 同一seedなら毎回同一初期ノイズ（エポック間での再現性確保が目的）

---

## 3. 適用済みパッチ一覧

### 適用順序

```
1. apply_fix_leco_train2.py          # フェーズ2: 階層学習・プリセットタブ追加
2. apply_fix_sample_ab.py            # サンプルA/B変数・UI追加
3. apply_fix_leco_argdup.py          # argparse重複引数削除
4. apply_fix_leco_funcorder.py       # 関数定義順序修正（NameError対策）
5. apply_fix_sample_filenames.py     # ファイル名ベース判別・サブディレクトリ廃止
6. apply_fix_bug01_lora_glob.py      # BUG-01: globパターン修正
7. apply_fix_bug02_vae_decode.py     # BUG-02: vae.decode dict/object両対応
8. apply_fix_bug03_sample_prompt_parse.py  # BUG-03: _parse_sample_prompt_line 修正
9. apply_fix_bug04_sample_seed.py    # BUG-04: A/B seed分離（A=42, B=43）
10. apply_fix_bug04b_leco_seed_order.py   # BUG-04b hotfix: leco NameError修正
```

### 各パッチの内容

#### apply_fix_bug01_lora_glob.py
- **対象:** `app/lora_train.py`
- **変更:** サンプルギャラリーのglobパターン修正
  - 変更前: `"*_00.png"` / `"*_01.png"`
  - 変更後: `"*_e*_00_*.png"` / `"*_e*_01_*.png"`
- **理由:** sd-scriptsの実際の出力ファイル名が `{name}_e{epoch:06d}_{idx:02d}_{timestamp}_{seed}.png`

#### apply_fix_bug02_vae_decode.py
- **対象:** `sd-scripts/anima_train_leco.py` L449
- **変更:** `vae.decode(latents_in).sample` → dict/DecoderOutput両対応
- **理由:** `qwen_image_autoencoder_kl` の `decode()` が `dict` を返す

#### apply_fix_bug03_sample_prompt_parse.py
- **対象:** `sd-scripts/anima_train_leco.py` `_parse_sample_prompt_line` 関数全体
- **変更:**
  - `neg_pat` 正規表現: `\s+--n\s+(.+?)(?=\s+--[a-zA-Z]|\s*$)` に変更
  - `prompt_end` 検索: `" --flag"` + 単語境界 `(?=\s|$)` で検索
  - `__import__("re")` → `re.search` に統一

#### apply_fix_bug04_sample_seed.py
- **対象:** `app/lora_train.py`, `app/leco_train.py`
- **変更:** `_build_sample_prompt_line_for` / `_leco_build_prompt_line` に `seed` 引数追加。書き込み側でA=seed42、B=seed43を割り当て
- **理由:** A/B両行に同一seed=42が使われ初期ノイズが同一になっていた

#### apply_fix_bug04b_leco_seed_order.py
- **対象:** `app/leco_train.py`
- **変更:** `_leco_build_prompt_line` シグネチャのデフォルト引数を `SAMPLE_FIXED_SEED` → `42`（リテラル）に変更
- **理由:** `leco_train.py` では `SAMPLE_FIXED_SEED` の定義が関数定義より後にあるため `NameError` が発生

---

## 4. 未解決バグ

### BUG-05: サンプル生成で入力プロンプトが正しく反映されない（未調査）

**現象:** LoRA・LECO学習時のサンプル生成で、GUIに入力したプロンプトが生成画像に反映されない。

**調査起点と仮説:**

LoRA経路での `sample_prompts_te_outputs` キャッシュ機構に問題がある可能性がある。

```
anima_train_network.py L194:
  prompts = train_util.load_prompts(args.sample_prompts)
  for prompt_dict in prompts:
      for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
          if p not in sample_prompts_te_outputs:
              sample_prompts_te_outputs[p] = encode(p)
```

キャッシュキーはプロンプト文字列そのもの。`_sample_image_inference` L528:
```python
if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
    return sample_prompts_te_outputs[prpt]  # キャッシュ優先
```

**調査が必要な点:**
1. プロンプトファイルの実際の書き込み内容をログ出力して確認する
2. `line_to_prompt_dict` のパース結果 `prompt_dict["prompt"]` が正しいか確認する
3. キャッシュキーとして使われるプロンプト文字列が `encode_prompt` 呼び出し時のキーと一致しているか確認する
4. LECO経路では `_parse_sample_prompt_line` の結果が `do_sample` に直接渡るため、BUG-03修正後に再確認が必要

**デバッグログ追加箇所（一時的）:**

LoRA: `anima_train_utils._sample_image_inference` L522付近のログ出力で確認可能
```
logger.info(f"  prompt: {prompt}, size: {width}x{height}, ...")
```
このログを学習実行時に確認する。

**次回セッションで必要なファイル:**
- `app/lora_train.py`（最新パッチ適用後）
- `app/leco_train.py`（最新パッチ適用後）
- `sd-scripts/anima_train_leco.py`（最新パッチ適用後）
- 学習実行時のログ出力（特にサンプル生成時の `prompt:` 行）

---

## 5. 未実装機能

### モニターグラフ移植（leco_train.py）

**概要:** `lora_train.py` に実装済みのモニターグラフ機能を `leco_train.py` に移植する。現在 `leco_train.py` の「モニターグラフ」「モニター階層」タブは空プレースホルダー。

**移植元ファイル:**

| ファイル | 役割 |
|----------|------|
| `app/monitor_graph.py` | Loss/LRグラフウィジェット |
| `app/monitor_layer.py` | 階層別実効LRウィジェット |

**フェーズ2パッチ適用済みの変数（`_LecoTrainState`）:**
```python
self._monitor_queue:       queue.Queue[str] = queue.Queue()
self._monitor_layer_queue: queue.Queue[str] = queue.Queue()
self.layer_train_enabled   = tk.BooleanVar(value=False)
self.layer_display_mode    = tk.StringVar(value="Matrix")
self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
```

**移植手順:**

STEP 1: `leco_train.py` の空タブ実装を置換
```python
# 変更前
def _build_monitor_tab(parent, s):
    ttk.Label(parent, text="モニターグラフ（実装予定）...").pack(expand=True)

def _build_monitor_layer_tab(parent, s):
    ttk.Label(parent, text="モニター階層（実装予定）...").pack(expand=True)

# 変更後
def _build_monitor_tab(parent, s):
    from .monitor_graph import MonitorGraph
    MonitorGraph(parent, s)

def _build_monitor_layer_tab(parent, s):
    from .monitor_layer import MonitorLayerGraph
    MonitorLayerGraph(parent, s, group_names_for_mode=_layer_group_names)
```

STEP 2: `monitor_graph.py` が参照する `state` プロパティを `_LecoTrainState` と照合
STEP 3: 不足プロパティを `_LecoTrainState.__init__` に追加
STEP 4: パッチ `apply_fix_leco_monitor.py` を作成・適用

---

## 6. ファイルの現在状態

| ファイル | 状態 |
|----------|------|
| `app/lora_train.py` | BUG-01・BUG-04適用済み |
| `app/leco_train.py` | BUG-04・BUG-04b適用済み、モニタータブ空 |
| `sd-scripts/anima_train_leco.py` | BUG-02・BUG-03適用済み |
| `app/monitor_graph.py` | 移植待ち |
| `app/monitor_layer.py` | 移植待ち |

---

## 7. パッチ作成規約

```python
# パッチファイルの基本構造
def _adapt(s: str) -> str:
    """CRLF/LF 両対応"""
    return s.replace("\r\n", "\n")

# OLD文字列はファイルの実内容と完全一致させること
# raw文字列 r'''...''' は \t \n のエスケープ解釈に注意
# → 通常の文字列連結 ('...\n' '...\n') で構築すること

# プロジェクトルートから実行する前提
# TARGET_FILE = Path("app/lora_train.py")  # 相対パスのみ使用
# 絶対パス・ユーザー名を含むパスは使用禁止

# バックアップ: .bak_{timestamp} を自動生成
# 二重適用防止: OLD文字列が存在しない場合は sys.exit(1)
# 出現数チェック: count != 1 の場合は中断

if __name__ == "__main__":
    # プロジェクトルートから実行すること
    apply()
```
