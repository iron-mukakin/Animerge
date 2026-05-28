# Animerge

For Japanese documentation, see [README.ja.md](./README.ja.md).

Animerge is a desktop GUI tool for working with Anima model checkpoints and LoRA files. The current build is centered on `app/gui.py`, with merge, analysis, model I/O, and LoRA training support split into related modules under `app/`.

<p align="center">
  <img src="./docs/anima_preview.png" alt="Animerge GUI Screenshot" width="50%">
</p>

## Model / Model Distribution

- [HuggingFace - circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)

Place base model files in `checkpoints/` and LoRA files in `lora/`. Supported model file extensions are `.safetensors`, `.ckpt`, and `.bin`.

## Current Features

- **Model-to-Model Merge**
  - Blending two base models (Model A and Model B).
- **LoRA-to-Model Fuse**
  - Fusing specified LoRA tensors into a base model.
- **LoRA-to-LoRA Merge**
  - Blending two LoRA files (LoRA A and LoRA B).
- **Model-Difference LoRA Extraction**
  - Extracting a LoRA structure from the differences between two models (e.g., pre-merged and post-merged models).
- **CLIP/Text Encoder/VAE Exclusion**
  - Filtering merge targets out for specified components (`clip`, `text_encoder`, `conditioner`, `vae`, `first_stage_model`).
- **Alpha Scaling with Per-Area and Per-Component Controls**
  - Matrix: Block (Input, Middle, Output) x Component (Attention, MLP, Norm, ResNet, Timestep) adjustments.
  - Transformer: Adjustments based on transformer/block units loaded from the base model.
  - Component: Adjustments based on major component groups (MLP, Norm, ResNet, Timestep, Other).
- **Slider and Direct Numeric Input for Adjustments**
- **Cosine Similarity Auto-Correction**
- **Input/Middle/Output Bias Freeze Toggles**
- **Dry-Run Finite-Value Tensor Validation**
- **LoRA Key-Name Normalization**
- **Layer Analysis and Detailed Analysis Viewer**
  - Canvas-based chart and heatmap analysis previews supporting Feature_Map, Statistical, SVD_Rank, and Attention_Map.
  - Report comparison displaying compatibility scores and difference charts by loading two logs from identical methods.
- **LoRA Training GUI Integrated with `kohya-ss/sd-scripts` for Anima Models**
  - Selectable Optimizers (`AdamW`, `AdamW8bit`, `Adafactor`, `DAdaptAdam`, `DAdaptAdaGrad`, `DAdaptSGD`, `Lion`, `Prodigy`).
  - Selectable LR Schedulers (`constant`, `constant_with_warmup`, `cosine`, `cosine_with_restarts`, `linear`, `polynomial`).
  - Configurable precision, timestep sampling, attention modes, and weighting schemes.
  - Real-time training monitoring graphs tracking Train Loss, Val Loss, LR, grad_norm, and ΔLoss.
  - EarlyStopping counter progress, estimated time remaining, and automated diagnostic messages.

## Main Files

- `app/gui.py`: Main Tkinter GUI and tab wiring (Model Merge / LoRA Merge / Layer Analysis / Detailed Analysis / LoRA Train).
- `app/merge.py`: Model and LoRA merge operations (exclusion marker setup, block categorization, various merge/extraction logics).
- `app/model_io.py`: Model scanning, loading, saving, extension validation (`.safetensors`, `.ckpt`, `.bin`), and hash computation.
- `app/analysis.py`: Layer analysis logic.
- `app/analysis_viewer.py`: Detailed analysis viewer tab.
- `app/monitor_graph.py`: LoRA training real-time monitoring widget (graphs, parameter panels, and auto-report panels).
- `app/lora_train.py`: LoRA training tab, `sd-scripts` command generation, and parameter constant management.
- `sd-scripts/`: Bundled scripts based on `kohya-ss/sd-scripts`, adapted for Anima LoRA training.

## Verified Environment

- Python 3.12
- PyTorch 2.8
- CUDA 12.8 (`cu128`)
- Windows

## Open Source Used
- Circlestone-labs/Anima: Many thanks for the model development
- Kohya-ss/sd-scripts: Many thanks to the creator and all the maintainers

## License / Third-Party Software

- [Apache-2.0](./LICENSE)
- This repository includes scripts based on `kohya-ss/sd-scripts`.