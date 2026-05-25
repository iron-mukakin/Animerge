# LoRA Train / GUI 作業引き継ぎメモ（セッション3）
作成日: 2026-05-24

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10)

---

## セッション3で解決した問題

### 1. 停止ボタンが効かない → 解決済み

**原因:** `accelerate launch` → `python _gui_train_wrapper.py` の2段プロセス構造のため、
`proc.terminate()` が親プロセスにしか届かず孫プロセス（学習本体）が生き残る。

**対処:** `lora_train.py` の2箇所を修正。

- `Popen` に `creationflags=CREATE_NEW_PROCESS_GROUP` を追加
- `_stop_training` で `os.kill(pid, signal.CTRL_BREAK_EVENT)` を使用してプロセスグループ全体に送信

### 2. VRAMアンロードが機能しない → 調査・部分対処済み

**現状の `unload_models`（`gui.py`）:**
`torch.cuda.empty_cache()` + `gc.collect()` を呼んでいるが、
LoRA学習プロセスが別プロセス（subprocess）として動いている場合、
GUIプロセス側の `empty_cache` はLoRA学習プロセスのVRAMに影響しない。

**対処済み（セッション3）:**
`unload_models` 呼び出し時、`_lora_train_state._proc` が生存していれば
`CTRL_BREAK_EVENT` で学習プロセスを終了してから VRAM 解放を実行するよう修正。

**未対処（課題）:**
`gui.py` の `unload_models` は `_lora_train_state` を `hasattr` で参照しているが、
`build_lora_train_tab` の戻り値（`_TrainState`）を `self._lora_train_state` に
代入する処理がまだ `gui.py` に追加されていない。

**次のセッションで対応が必要:**
`gui.py` の `build_lora_train_tab(...)` 呼び出し箇所（171行目付近）を以下のように修正:
```python
# 変更前
build_lora_train_tab(train_main, self.paths, self.log, lambda: self.model_choices)

# 変更後
self._lora_train_state = build_lora_train_tab(
    train_main, self.paths, self.log, lambda: self.model_choices
)
```
さらに `lora_train.py` の `build_lora_train_tab` が `state` を返すよう修正:
```python
# 末尾に追加
return state
```

### 3. マージ系プリセット保存先の変更 → 解決済み

**変更前:** `self.paths.configs / f"preset_{tab_type}_{name}.json"`
**変更後:** `self.paths.root / "preset" / "merge" / f"{name}.json"`

本体マージ・LoRAマージのプリセットが同一ディレクトリ（`E:\Animerge\preset\merge\`）に
フラットなJSON名（プリセット名のみ）で保存される形式に統一。
tab_type による名前分離を廃止したため、本体マージとLoRAマージのプリセットが共通利用可能。

---

## 適用済みファイル変更一覧（セッション3）

| ファイル | 変更内容 |
|----------|----------|
| `app/lora_train.py` | `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` による停止 |
| `app/gui.py` | プリセット保存先を `preset/merge/` に変更、`unload_models` にLoRAプロセス終了処理追加 |

---

## 次のセッションで実施予定の機能

### 優先度高（未着手）

#### A. VRAMアンロード完全対応
- `gui.py:171` の `build_lora_train_tab` 呼び出しで戻り値を `self._lora_train_state` に代入
- `lora_train.py` の `build_lora_train_tab` 末尾に `return state` を追加

#### B. キー名称正規化（学習側）
- マージ機能では対応済み。学習側（`anima_utils.py` のモデルロード）でも同じ正規化が必要
- マージ済みモデルを学習ベースとして使う場合にキー不一致が発生する可能性がある
- マージ側の正規化コード関数名・ファイルを次セッションで確認して実装

### 優先度中（予定）

#### C. LoRA学習プリセット機能
- 保存先: `E:\Animerge\preset\lora_train\`
- `_TrainState` の全 `tk.Variable` を JSON に保存・復元
- `lora_train.py` の詳細タブ内に新タブ「プリセット」として追加

#### D. 階層学習（レイヤー学習）
- GUIで共通規格の3種類のレイヤー学習をLoRA学習タブに追加
- マージ機能の階層定義との対応関係を確認してから実装

#### E. Validation Loss・早期停止オプション
- `train_network.py` 側に validation dataset サポートが必要（現状プレースホルダ）
- GUIにオプション追加 + `train_network.py` 改修が両方必要

#### F. UI改修・リアルタイム表示
- loss / step / epoch / 予測終了時間を色付き個別窓で表示
- `_drain()` の200ms ポーリングを利用してログから正規表現でパース
- グラフタブ追加: `matplotlib` の `FigureCanvasTkAgg` を詳細タブ内の新タブに埋め込み
- タブ構成: 詳細 → 右へ「レイヤー学習」「モニターグラフ」「プリセット」を追加

#### G. 本体マージ元モデルのキー名称正規化
- マージ機能は対応済み
- LoRA学習でマージモデルをベースにする場合の対応（Bと同一課題）

---

## 備考
- `strategy_base.py:96` の SyntaxWarning（`\(` エスケープ）は動作に影響なし、後回し可
- 学習の正常動作確認済み（`create LoRA for U-Net: 508 modules`、epoch/step/loss推移正常）
- loss収束傾向：epoch3まで下降、epoch4でcosine_with_restartsのLRリスタートにより一時上昇（正常動作）
