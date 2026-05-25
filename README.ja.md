# Animerge

暫定版 README です。

Animerge は Anima モデルのチェックポイントと LoRA ファイルを扱うためのデスクトップ GUI ツールです。現時点の実装は `app/gui.py` を中心に、マージ、解析、モデル入出力、LoRA 学習機能を `app/` 配下の関連ファイルへ分離しています。

## モデル

対象モデルの配布先:

- https://huggingface.co/circlestone-labs/Anima

ベースモデルは `checkpoints/`、LoRA は `lora/` に配置します。対応拡張子は `.safetensors`, `.ckpt`, `.bin` です。

## 現在の主な機能

- モデル同士のマージ。
- LoRA のモデルへのフューズ。
- LoRA 同士のマージ。
- モデル差分からの LoRA 抽出。
- CLIP/Text Encoder/VAE の除外。
- Alpha スケーリングと領域別・コンポーネント別調整。
- レイヤー調整表示:
  - Matrix: ブロック x コンポーネント。
  - Transformer: ベースモデルから読み込んだ transformer/block 単位。
  - Component: 主要コンポーネント単位。
- スライダーと数値入力による調整。
- Cosine similarity による自動補正。
- Input/Middle/Output bias の freeze。
- Dry-run による tensor finite-value 検証。
- LoRA キー名の正規化。
- レイヤー分析と詳細分析ビューア。
- `kohya-ss/sd-scripts` を組み込んだ Anima 向け LoRA 学習 GUI。

## 主な参照ファイル

- `app/gui.py`: Tkinter GUI 本体とタブ構成。
- `app/merge.py`: モデル/LoRA のマージ処理。
- `app/model_io.py`: モデル走査、読み込み、保存、依存確認。
- `app/analysis.py`: レイヤー分析処理。
- `app/analysis_viewer.py`: 詳細分析ビューア。
- `app/lora_train.py`: LoRA 学習タブと `sd-scripts` 実行コマンド生成。
- `sd-scripts/`: `kohya-ss/sd-scripts` をベースにした Anima LoRA 学習用スクリプト群。

## 動作確認環境

- Python 3.12
- PyTorch 2.8
- CUDA 12.8 (`cu128`)
- Windows

依存ライブラリのインストール:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

起動:

```powershell
.\setup_start.bat
```

## 注意

- 実際のマージや学習には、所定フォルダにモデルファイルを配置する必要があります。
- LoRA 学習は `app/lora_train.py` から `sd-scripts/anima_train_network.py` を呼び出す構成です。
- `bitsandbytes` や `xformers` などの任意機能は、CUDA/PyTorch のビルドに合わせた個別導入が必要になる場合があります。

## ライセンス

Animerge のライセンスは Apache-2.0 です。

本リポジトリには `kohya-ss/sd-scripts` をベースにしたスクリプトを組み込んでいます。これらの大部分は ASL 2.0 でライセンスされており、Diffusers、cloneofsimo、および LoCon 由来のコードを含みます。ただし、以下の部分は別ライセンスです。

- Memory Efficient Attention Pytorch: MIT
- bitsandbytes: MIT
- BLIP: BSD-3-Clause
