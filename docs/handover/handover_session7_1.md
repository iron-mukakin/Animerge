# LECO学習タブ 後工程実装仕様書

## 概要

本仕様書はフェーズ1（基本LECO学習 + GUI）の実装完了を受け、  
フェーズ2として実装する残機能を新セッションへ引き継ぐための仕様書である。

---

## 現在の実装状態（フェーズ1完了済み）

### 作成済みファイル

| ファイル | 配置先 | 状態 |
|---------|--------|------|
| `anima_train_leco.py` | `sd-scripts/` | 完成・動作確認済み |
| `leco_train.py` | `app/` | 完成・動作確認済み |
| `apply_fix_gui.py` | `app/` | 適用済み（gui.py にLECOタブ追加済み） |

### 適用済みパッチ（anima_train_leco.py に順次適用）

| パッチファイル | 内容 |
|--------------|------|
| `apply_fix_anima_train_leco.py` | `t5_tokenizer_path` の `or ""` → `or None` 修正 |
| `apply_fix_anima_train_leco2.py` | `get_initial_latents` をFlowMatch対応版にインライン置換 |
| `apply_fix_anima_train_leco3.py` | Phase1に `torch.no_grad()` + `.detach()` 追加（VRAM節約） |
| `apply_fix_anima_train_leco4.py` | `progress_bar` 二重出力修正（set_postfix→update の順に統一） |

### 動作確認済み内容

- 学習実行・完了（500ステップ）
- VRAM消費約6GB（RTX 5060 Ti 16GB環境）
- 速度：7〜12秒/step（`max_denoising_steps=40` デフォルト時）
- プロンプトTOML形式：`[[prompts]]` セクション使用

---

## フェーズ2 実装対象機能

### 優先順位

```
1. 階層学習タブ（LECOタブへの追加）
2. モニターグラフタブ
3. モニター階層タブ
4. プリセットタブ
```

---

## 1. 階層学習タブ

### 概要

`lora_train.py` の `_build_layer_train_tab()` をほぼそのまま `leco_train.py` に移植する。  
コマンド生成部分（`_build_command`）に階層学習引数を追加する。

### 移植元（lora_train.py）

```python
# 定数（leco_train.py の先頭定数ブロックへ追加）
LAYER_TRAIN_MODES  = ("Matrix", "Transformer", "Component")
MATRIX_BLOCKS      = ("Input", "Middle", "Output")
MATRIX_COMPONENTS  = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS   = ("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")
LAYER_COLUMNS      = 3

# ブロックカテゴリ対応（blocks.0-8=Input, blocks.9-18=Middle, blocks.19-27=Output）
_BLOCK_CAT = ["Input"] * 9 + ["Middle"] * 10 + ["Output"] * 9
```

### _LecoTrainState への追加変数

```python
# 既存の _LecoTrainState.__init__ に追加
self.layer_train_enabled  = tk.BooleanVar(value=False)
self.layer_display_mode   = tk.StringVar(value="Matrix")
self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
self.layer_canvas = None   # _refresh_layer_controls から参照
self.layer_inner  = None
self._layer_status_var = tk.StringVar(value="(無効)")
```

### 移植関数（lora_train.py からそのままコピー可）

以下の関数は `leco_train.py` に追加する（`_TrainState` を `_LecoTrainState` に読み替えるだけ）：

- `_build_layer_train_tab(parent, s)` ← タブUI構築
- `_layer_group_names(mode)` ← グループ名リスト生成
- `_refresh_layer_controls(s, canvas, inner)` ← スライダー再構築
- `_snap_scale(var)` ← 0.05刻みスナップ
- `_clamp_var(var)` ← 0.0–1.0クランプ
- `_load_layer_preset(s, canvas, inner)` ← プリセット読み込み（注意点あり、後述）
- `_convert_preset_scales(scales, preset_mode, target_mode)` ← モード間変換
- `_layer_scales_to_block_weights(mode, scales)` ← network_args 変換

### タブ追加位置

`build_leco_train_tab()` 内のノートブック構築部分に追加：

```python
# 既存の5タブ後に追加
tab_layer = ttk.Frame(nb, padding=8)
nb.add(tab_layer, text="  階層学習  ")
_build_layer_train_tab(tab_layer, state)
```

### _build_command への階層学習引数追加

`lora_train.py` の `_build_command` 内の以下ブロックをそのまま `leco_train.py` の `_build_command` へ追加：

```python
# 階層学習（lora_train.py L1777-1800 と同一ロジック）
if s.layer_train_enabled.get():
    _mode   = s.layer_display_mode.get()
    _scales = {k: v.get() for k, v in s.layer_parameter_vars.items()}

    if _mode == "Matrix":
        _scales_json = json.dumps(
            {k: round(v, 4) for k, v in _scales.items()},
            separators=(",", ":"),
        )
        cmd += ["--network_args", f"anima_matrix_scales={_scales_json}"]
    else:
        _weights = _layer_scales_to_block_weights(_mode, _scales)
        weight_str = ",".join(f"{w:.4f}" for w in _weights)
        cmd += ["--network_args", f"anima_block_lr_weight={weight_str}"]
```

### _load_layer_preset の注意点

`lora_train.py` では `preset/merge/*.json` を参照しているが、  
`leco_train.py` では **`preset/leco_train/*.json`** を参照するように変更する：

```python
# lora_train.py（移植元）
preset_dir = s.paths.root / "preset" / "merge"

# leco_train.py（変更後）
preset_dir = s.paths.root / "preset" / "leco_train"
```

---

## 2. モニターグラフタブ

### 概要

`monitor_graph.py` の `MonitorGraph` クラスを埋め込む。  
`lora_train.py` の `_build_monitor_tab()` と同一実装。

### _LecoTrainState への追加変数

```python
# _monitor_graph_queue: MonitorGraph が学習ログをパースするために使用
# lora_train.py の _TrainState と同名で定義する
import queue
self._monitor_queue: queue.Queue[str] = queue.Queue()
```

### 学習ループへのキュー送信追加

`leco_train.py` の `_worker()` 内ログ行読み取り部分にキュー送信を追加：

```python
# 既存コード
for line in proc.stdout:
    line = _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line.rstrip())
    s._log_queue.put(line)

# 追加（MonitorGraph/MonitorLayerGraph 向け）
    s._monitor_queue.put(line)           # モニターグラフ用
    s._monitor_layer_queue.put(line)     # モニター階層用
```

### タブ追加

```python
tab_monitor = ttk.Frame(nb, padding=8)
nb.add(tab_monitor, text="  モニターグラフ  ")
_build_monitor_tab(tab_monitor, state)
```

### _build_monitor_tab の実装

`lora_train.py` の `_build_monitor_tab()` と同一：

```python
def _build_monitor_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
    parent.rowconfigure(0, weight=1)
    parent.rowconfigure(1, weight=0)
    parent.columnconfigure(0, weight=1)

    graph_frame = ttk.Frame(parent)
    graph_frame.grid(row=0, column=0, sticky=tk.NSEW)

    try:
        from .monitor_graph import MonitorGraph
        s._monitor_graph = MonitorGraph(graph_frame, s)
    except Exception as exc:
        ttk.Label(
            graph_frame,
            text=f"モニターグラフの初期化に失敗しました。\n{exc}",
            foreground="#EF4444",
            justify=tk.LEFT,
        ).pack(padx=16, pady=24, anchor=tk.W)

    log_frame = ttk.LabelFrame(parent, text="学習ログ")
    log_frame.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
    log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 12))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)
    s._log_widgets.append(log_text)
```

### MonitorGraph が参照する _TrainState の属性

`monitor_graph.py` は以下の属性を参照する。`_LecoTrainState` に同名で定義する必要がある：

```python
s._monitor_queue      # Queue[str]  ← ログ行を流すキュー
s.lr                  # tk.StringVar  ← ベースLR（既存）
```

---

## 3. モニター階層タブ

### 概要

`monitor_layer.py` の `MonitorLayerGraph` クラスを埋め込む。  
階層学習が有効な場合のみ意味を持つ。

### _LecoTrainState への追加変数

```python
self._monitor_layer_queue: queue.Queue[str] = queue.Queue()
```

### タブ追加

```python
tab_monitor_layer = ttk.Frame(nb, padding=8)
nb.add(tab_monitor_layer, text="  モニター階層  ")
_build_monitor_layer_tab(tab_monitor_layer, state)
```

### _build_monitor_layer_tab の実装

`lora_train.py` と同一（`_layer_group_names` 関数への参照のみ）：

```python
def _build_monitor_layer_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
    try:
        from .monitor_layer import MonitorLayerGraph
        s._monitor_layer_graph = MonitorLayerGraph(parent, s, _layer_group_names)
    except Exception as exc:
        ttk.Label(
            parent,
            text=f"モニター階層の初期化に失敗しました。\n{exc}",
            foreground="#EF4444",
            justify=tk.LEFT,
        ).pack(padx=16, pady=24, anchor=tk.W)
```

### MonitorLayerGraph が参照する _TrainState の属性

`monitor_layer.py` は以下の属性を参照する（全て `_LecoTrainState` に追加済みまたは追加予定）：

```python
s._monitor_layer_queue   # Queue[str]
s.layer_train_enabled    # tk.BooleanVar
s.layer_display_mode     # tk.StringVar
s.layer_parameter_vars   # dict[str, tk.DoubleVar]
s.lr                     # tk.StringVar（既存）
```

---

## 4. プリセットタブ

### 概要

LECO学習設定（全 tk.Variable）を JSON で保存・復元する。  
`lora_train.py` の `_build_train_preset_tab()` を LECO 仕様に改変する。

### 保存ディレクトリ

```python
PRESET_DIR_REL = ("preset", "leco_train")  # lora_train の "lora_train" から変更
```

### 保存対象変数（_collect 相当）

LECO固有変数のみ。LoRA学習固有の変数（train_data_dir, resolution等）は含めない：

```python
{
    # モデル
    "model_path":        s.model_path.get(),
    "vae_path":          s.vae_path.get(),
    "qwen3_path":        s.qwen3_path.get(),
    "llm_adapter_path":  s.llm_adapter_path.get(),
    "output_dir":        s.output_dir.get(),
    "output_name":       s.output_name.get(),
    "precision":         s.precision.get(),
    # プロンプト
    "prompts_file":      s.prompts_file.get(),
    # ネットワーク
    "network_dim":       int(s.network_dim.get()),
    "network_alpha":     float(s.network_alpha.get()),
    "network_module":    s.network_module.get(),
    "network_weights":   s.network_weights.get(),
    # 学習設定
    "lr":                s.lr.get(),
    "lr_scheduler":      s.lr_scheduler.get(),
    "lr_warmup_steps":   int(s.lr_warmup_steps.get()),
    "optimizer":         s.optimizer.get(),
    "optimizer_args":    s.optimizer_args.get(),
    "max_train_steps":   int(s.max_train_steps.get()),
    "save_every_n_steps": int(s.save_every_n_steps.get()),
    "seed":              s.seed.get(),
    "gradient_checkpointing": bool(s.gradient_checkpointing.get()),
    "grad_accum":        int(s.grad_accum.get()),
    "mixed_precision":   s.mixed_precision.get(),
    "max_grad_norm":     float(s.max_grad_norm.get()),
    # LECOパラメータ
    "max_denoising_steps":         int(s.max_denoising_steps.get()),
    "leco_denoise_guidance_scale": float(s.leco_denoise_guidance_scale.get()),
    # 詳細（Anima）
    "attn_mode":         s.attn_mode.get(),
    "split_attn":        bool(s.split_attn.get()),
    "blocks_to_swap":    int(s.blocks_to_swap.get()),
    "unsloth_offload_checkpointing": bool(s.unsloth_offload_checkpointing.get()),
    "cpu_offload_checkpointing":     bool(s.cpu_offload_checkpointing.get()),
    "vae_chunk_size":    s.vae_chunk_size.get(),
    "vae_disable_cache": bool(s.vae_disable_cache.get()),
    "qwen3_max_token_length": int(s.qwen3_max_token_length.get()),
    "t5_max_token_length":    int(s.t5_max_token_length.get()),
    "t5_tokenizer_path":      s.t5_tokenizer_path.get(),
    "discrete_flow_shift":    float(s.discrete_flow_shift.get()),
    # 階層学習
    "layer_train_enabled": bool(s.layer_train_enabled.get()),
    "layer_display_mode":  s.layer_display_mode.get(),
    "layer_parameter_vars": {
        k: round(float(v.get()), 4)
        for k, v in s.layer_parameter_vars.items()
    },
}
```

### プリセット読み込み時の注意点（lora_train.py と同一実装）

`layer_parameter_vars` はスライダー生成後でなければ値を反映できないため、  
`_apply()` 実行前に `layer_train_enabled` / `layer_display_mode` を先行設定し  
`_refresh_layer_controls()` を呼ぶ必要がある：

```python
def _load():
    # ... JSONパース後 ...
    _pre_enabled = bool(data.get("layer_train_enabled", False))
    _pre_mode    = data.get("layer_display_mode", "Matrix")
    if _pre_mode not in LAYER_TRAIN_MODES:
        _pre_mode = "Matrix"
    s.layer_train_enabled.set(_pre_enabled)
    s.layer_display_mode.set(_pre_mode)
    if s.layer_canvas is not None and s.layer_inner is not None:
        _refresh_layer_controls(s, s.layer_canvas, s.layer_inner)
    _apply(data)  # ← スライダー生成後に layer_parameter_vars へ値反映
```

---

## 最終的なタブ構成

```
LECO学習タブ（leco_main）
  ├─ モデル
  ├─ プロンプト設定     ← フェーズ1実装済み
  ├─ ネットワーク
  ├─ 学習設定
  ├─ 詳細
  ├─ 階層学習           ← フェーズ2実装対象
  ├─ モニターグラフ     ← フェーズ2実装対象
  ├─ モニター階層       ← フェーズ2実装対象
  └─ プリセット         ← フェーズ2実装対象
```

---

## 実装方式

- `leco_train.py` への追記のみ（新規ファイルは不要）
- `gui.py` / `anima_train_leco.py` への変更は不要
- `monitor_graph.py` / `monitor_layer.py` は変更なしでそのまま流用
- 出力は **`apply_fix_leco_train2.py`** のパッチスクリプト形式

---

## 次セッションへの提出ファイル

以下のファイルを次セッション開始時に提供すること：

| ファイル | 理由 |
|---------|------|
| `leco_train.py`（現状版） | パッチの差分文字列照合に必要 |
| `lora_train.py` | 移植元コードの参照 |
| `monitor_graph.py` | MonitorGraph が参照する _TrainState 属性の確認 |
| `monitor_layer.py` | MonitorLayerGraph が参照する _TrainState 属性の確認（提供済み） |

---

## 既知の制約・注意事項

### 階層学習とLECOの互換性

LECOはU-Net（DiT）のみ学習するため、階層学習のスケールはDiT側にのみ適用される。  
`anima_train_leco.py` は `--network_args` を `build_network_kwargs()` 経由で受け取り、  
`networks.lora` の Anima 拡張（`anima_block_lr_weight` / `anima_matrix_scales`）が処理する。  
この経路は通常LoRA学習と同一であり、**追加のスクリプト改修は不要**。

### モニターグラフのLR取得

`monitor_graph.py` はプロセスのstdoutから `lr=X.XXe-XX` 形式の文字列をパースする。  
`anima_train_leco.py` の学習ループは `accelerator.log({"lr": ...})` を出力しており、  
tqdmのpostfixには含まれていない。`monitor_graph.py` のパターンが合致しない場合は  
`_worker()` のログ出力フォーマットを確認すること。

### TOMLテンプレートのデフォルト値

```toml
[[prompts]]
target        = "概念・スタイル名"
positive      = "target を含む肯定プロンプト"
unconditional = ""        # 空文字列で問題なし
neutral       = ""        # 空文字列で問題なし
action        = "erase"
guidance_scale = 1.0
resolution    = 512
batch_size    = 1
multiplier    = 1.0
weight        = 1.0
```
