# Animerge

Interim README. For Japanese documentation, see [README.ja.md](./README.ja.md).

Animerge is a desktop GUI tool for working with Anima model checkpoints and LoRA files. The current build is centered on `app/gui.py`, with merge, analysis, model I/O, and LoRA training support split into related modules under `app/`.

## Model

The target Anima model distribution is:

- https://huggingface.co/circlestone-labs/Anima

Place base model files in `checkpoints/` and LoRA files in `lora/`. Supported model file extensions are `.safetensors`, `.ckpt`, and `.bin`.

## Current Features

- Model-to-model merge.
- LoRA-to-model fuse.
- LoRA-to-LoRA merge.
- Model-difference LoRA extraction.
- CLIP/Text Encoder/VAE exclusion during merge.
- Alpha scaling with per-area and per-component controls.
- Layer adjustment modes:
  - Matrix: block x component controls.
  - Transformer: transformer/block controls from the loaded base model.
  - Component: major component controls.
- Slider and direct numeric input for adjustment values.
- Cosine similarity auto-correction.
- Input/Middle/Output bias freeze toggles.
- Dry-run finite-value tensor validation.
- LoRA key-name normalization.
- Layer analysis and detailed analysis viewer.
- LoRA training GUI integrated with the bundled `kohya-ss/sd-scripts` based scripts.

## Main Files

- `app/gui.py`: main Tkinter GUI and tab wiring.
- `app/merge.py`: model and LoRA merge operations.
- `app/model_io.py`: model scanning, loading, saving, and dependency checks.
- `app/analysis.py`: layer analysis logic.
- `app/analysis_viewer.py`: detailed analysis viewer tab.
- `app/lora_train.py`: LoRA training tab and `sd-scripts` command generation.
- `sd-scripts/`: bundled scripts based on `kohya-ss/sd-scripts`, adapted for Anima LoRA training.

## Environment

Verified environment:

- Python 3.12
- PyTorch 2.8
- CUDA 12.8 (`cu128`)
- Windows

Install dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run:

```powershell
.\setup_start.bat
```

## Notes

- Real merge and training operations require model files in the expected folders.
- LoRA training uses the integrated `sd-scripts/anima_train_network.py` flow through `app/lora_train.py`.
- Optional optimizers or attention backends such as `bitsandbytes` and `xformers` may need separate installation depending on the CUDA/PyTorch build.

## License

Animerge is licensed under Apache-2.0.

This repository includes scripts based on `kohya-ss/sd-scripts`. The majority of those scripts are licensed under ASL 2.0, including code from Diffusers, cloneofsimo's work, and LoCon. Portions of the project are available under separate license terms:

- Memory Efficient Attention Pytorch: MIT
- bitsandbytes: MIT
- BLIP: BSD-3-Clause
