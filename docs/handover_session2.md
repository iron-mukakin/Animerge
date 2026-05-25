# LoRA Train 作業引き継ぎメモ（セッション2）
作成日: 2026-05-23

## プロジェクト構成
- アプリ本体: `E:\Animerge\app\`
- sd-scripts: `E:\Animerge\sd-scripts\`
- 学習ログ出力先: `E:\Animerge\log\lora_train\YYYYMMDD_HHMMSS.txt`
- venv: `E:\Animerge\.venv\Scripts\python.exe` (Python 3.12.10)

---

## セッション2で解決した問題

### 1. `create LoRA for U-Net: 0 modules` → 解決済み

**原因:** `networks/__pycache__/` に古い `.pyc` が残っていた。

**対処:**
```powershell
Remove-Item -Recurse -Force 'E:\Animerge\sd-scripts\networks\__pycache__'
Remove-Item -Recurse -Force 'E:\Animerge\sd-scripts\__pycache__'
Remove-Item -Recurse -Force 'E:\Animerge\sd-scripts\library\__pycache__'
```

**確認済み事項:**
- `UNET_TARGET_REPLACE_MODULE = ["Block", "LLMAdapterTransformerBlock"]` は正しい
- Anima DiT の `named_modules()` で `Block` × 26, `LLMAdapterTransformerBlock` × 8 = 計34モジュール検出
- その子 `Linear` は 508 個存在 → LoRA対象として正常

---

### 2. `An attempt has been made to start a new process before bootstrapping` → 解決済み

**原因:** Windows の multiprocessing は spawn 方式。`DataLoader` の `num_workers` が 1 以上だと
子プロセスが `_gui_train_wrapper.py` を再インポートし、無限起動ループが発生する。

**対処:** `train_network.py` の `n_workers` を 0 に固定。

```python
# E:\Animerge\sd-scripts\train_network.py 776行目
n_workers = 0  # fixed: Windows spawn multiprocessing workaround
```

---

### 3. `_gui_train_wrapper.py` が毎回上書きされる問題 → 解決済み

**原因:** `lora_train.py` の `_build_command()` 内で毎回 `wrapper.write_text()` を呼んで
`_gui_train_wrapper.py` を生成していた。手動で書き換えても起動のたびに上書きされる。

**対処:** `lora_train.py` の wrapper 生成内容を exec 方式（元の正しい形）に修正済み。
現在の正しい wrapper 生成内容：

```python
wrapper.write_text(
    "import sys, os\n"
    f"sys.path.insert(0, r'{sd_scripts_root}')\n"
    f"os.chdir(r'{sd_scripts_root}')\n"
    f"with open(r'{train_script}', encoding='utf-8') as _f:\n"
    "    _code = compile(_f.read(), _f.name, 'exec')\n"
    f"exec(_code, {{'__name__': '__main__', '__file__': r'{train_script}'}})\n",
    encoding="utf-8",
)
```

---

### 4. ログ文字化け対策 → 対処済み

**原因:** Windows の subprocess stdout エンコーディングが cp932。

**対処:** `lora_train.py` の `_worker()` 内 `env = os.environ.copy()` の直後に追加済み：
```python
env["PYTHONUTF8"] = "1"
```

---

## 現在の未解決問題

### 学習が正常に開始するか未確認

上記修正（`n_workers=0` + `lora_train.py` 修正）を適用後、GUIから再実行予定。
次のセッション冒頭でログを確認する。

**期待されるログの流れ:**
```
create LoRA for U-Net: 508 modules.
...
epoch 1/10
  step 1/N  loss=...
```

---

## 適用済みファイル変更一覧

| ファイル | 変更内容 |
|----------|----------|
| `sd-scripts/networks/lora.py` | `UNET_TARGET_REPLACE_MODULE = ["Block", "LLMAdapterTransformerBlock"]` |
| `sd-scripts/train_network.py` | `n_workers = 0` に固定（776行目） |
| `app/lora_train.py` | wrapper生成を exec 方式に修正、`PYTHONUTF8=1` 追加 |
| `sd-scripts/library/` | スタブ各種（前セッション済み） |

---

## VRAMアンロードボタンが効かない問題（未着手）

GUIの「メモリアンロード」ボタンでVRAMが解放されない。
アプリ終了では解放される。
学習が正常に動作することを確認してから着手する。
`app/lora_train.py` または `app/gui.py` のアンロード処理を確認する必要がある。

---

## 次のアクション
1. GUIから学習実行 → ログで `epoch 1` が出ることを確認
2. 正常動作確認後、VRAMアンロードボタンの修正に着手
3. `strategy_base.py:96` の SyntaxWarning（`\(` エスケープ）は動作に影響なし、後回し可
