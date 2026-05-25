# LoRA Train GUI / sd-scripts Handover

調査日: 2026-05-23

## 対象ファイル

- `E:\Animerge\app\gui.py`
- `E:\Animerge\app\lora_train.py`
- `E:\Animerge\sd-scripts\anima_train_network.py`
- `E:\Animerge\sd-scripts\train_network.py`
- `E:\Animerge\sd-scripts\networks\lora.py`
- `E:\Animerge\requirements.txt`
- `E:\Animerge\setup_start.bat`
- `E:\Animerge\log\lora_train\*.txt`

## 現状

GUIへのタブ統合自体は入っている。

- `app\gui.py:30` で `build_lora_train_tab` を import。
- `app\gui.py:164` で主タブに `LoRA学習` を追加。
- `app\gui.py:171` で `build_lora_train_tab(train_main, self.paths, self.log, lambda: self.model_choices)` を呼び出し。
- `app\lora_train.py:598` 以降で GUI 設定から `accelerate launch` コマンドを生成。
- `app\lora_train.py:784` で `subprocess.Popen(..., cwd=sd-scripts)` により学習プロセスを起動。

## 確認できた主な問題

### 1. `sd-scripts` の import が依存バージョンで停止する

Python 3.10 で以下を実行すると失敗する。

```powershell
& 'C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe' -c "import sys; sys.path.insert(0, r'E:\Animerge\sd-scripts'); import anima_train_network; print('OK import anima_train_network')"
```

結果:

```text
ImportError: peft>=0.17.0 is required ... but found peft==0.12.0.
```

インストール状況:

- `diffusers==0.36.0`
- `peft==0.12.0`
- `accelerate==1.12.0`
- `transformers==4.57.6`
- `torch==2.7.1+cu128`
- `bitsandbytes` 未インストール
- `xformers` 未インストール

`requirements.txt` には `peft` が無い。`diffusers 0.36.0` を使うなら `peft>=0.17.0` を追加する必要がある。

### 2. `.venv` の Python が壊れている

`E:\Animerge\.venv\Scripts\python.exe --version` が以下で失敗する。

```text
Unable to create process using '"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" --version'
```

`setup_start.bat` は `E:\Animerge\.venv\Scripts\python.exe` を使うため、GUI起動と `app\lora_train.py` の `sys.executable` も壊れた venv を引き継ぐ可能性が高い。

### 3. `sd-scripts\library` が部分コピーで、追加の欠落モジュールがある

静的参照上、現在の `sd-scripts\library` に存在しない参照:

```text
custom_train_functions
flux_models
flux_utils
huggingface_util
ipex
jpeg_xl_util
lpw_stable_diffusion
mask_generator
model_util
original_unet
sd3_models
sd3_utils
sdxl_lpw_stable_diffusion
sdxl_model_util
strategy_sd
```

特に `sd-scripts\train_network.py:28`, `:37`, `:38`, `:39` は top-level import なので、`peft` 問題解消後に `ModuleNotFoundError` へ進む可能性が高い。

### 4. `app\lora_train.py` の wrapper 生成は方向性として妥当だが、環境問題は解決しない

`app\lora_train.py:606` で `sd-scripts\_gui_train_wrapper.py` を生成し、`sys.path` と `cwd` を補正している。これは `accelerate launch` の子プロセスで `sd-scripts` を見つけるためには有効。

ただし、`app\lora_train.py:617` は `sys.executable -m accelerate.commands.launch` なので、GUIを起動した Python 環境の壊れ・依存不足をそのまま使う。

### 5. GUI設定のデフォルトと依存が噛み合っていない

`app\lora_train.py` のデフォルト optimizer は `AdamW8bit`。しかし `bitsandbytes` は未インストール。Windows環境では導入も不安定なので、既定値は `AdamW` の方が安全。

`xformers` も未インストール。GUIに項目はあるが、有効化すると失敗する可能性が高い。

## 次にやるべき修正順

1. `.venv` を作り直す。壊れた `E:\Animerge\.venv` を削除または退避し、Python 3.10/3.11 の実在する interpreter で再作成する。
2. `requirements.txt` に `peft>=0.17.0` を追加する。`torch` の指定も実環境の `2.7.1+cu128` と `requirements.txt` の `2.8.0+cu128` がずれているため、どちらに合わせるか決める。
3. `sd-scripts` を公式/元リポジトリから欠落なく同期する。部分的なファイルコピーでは `train_network.py` と `library\train_util.py` の top-level import が成立しない。
4. `app\lora_train.py` の optimizer 既定値を `AdamW` に変更し、`AdamW8bit` は `bitsandbytes` 導入済みの場合だけ選ばせる。
5. `python -c "import sys; sys.path.insert(0, r'E:\Animerge\sd-scripts'); import anima_train_network"` が通ることを確認する。
6. その後、GUIから実際の `accelerate launch` コマンドを実行して、引数不整合を潰す。

## 引き継ぎ時の最短再現コマンド

```powershell
cd /d E:\Animerge
& 'C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe' -c "import sys; sys.path.insert(0, r'E:\Animerge\sd-scripts'); import anima_train_network; print('OK')"
```

この import が通らない限り、GUI側のボタンやコマンド生成を直しても学習は起動できない。
