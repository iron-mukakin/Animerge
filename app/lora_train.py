"""app/lora_train.py — Anima LoRA学習タブ

build_lora_train_tab(parent, paths, log_fn, get_model_choices) を呼び出すことで
gui.py の LoRA学習タブに組み込まれる。
"""
from __future__ import annotations

import datetime
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import json
import re
from typing import Callable

# ──────────────────────────────────────────────────────────────────────────────
# 定数 (anima_train_utils.py / anima_train_network.py 由来)
# ──────────────────────────────────────────────────────────────────────────────
OPTIMIZERS = [
    "AdamW", "AdamW8bit", "Adafactor", "DAdaptAdam",
    "DAdaptAdaGrad", "DAdaptSGD", "Lion", "Prodigy",
]
LR_SCHEDULERS = [
    "constant", "constant_with_warmup", "cosine",
    "cosine_with_restarts", "linear", "polynomial",
]
PRECISIONS = ["bf16", "fp16", "fp32"]
TIMESTEP_SAMPLING = ["sigmoid", "sigma", "uniform", "shift", "flux_shift"]
ATTN_MODES = ["torch", "xformers", "flash", "sdpa"]
WEIGHTING_SCHEMES = ["none", "sigma_sqrt", "cosmap"]

# 階層学習 GUI定数（gui.py の MATRIX_BLOCKS / MATRIX_COMPONENTS / COMPONENT_GROUPS と同一）
LAYER_TRAIN_MODES   = ("Matrix", "Transformer", "Component")
MATRIX_BLOCKS       = ("Input", "Middle", "Output")
MATRIX_COMPONENTS   = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS    = ("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")
LAYER_COLUMNS       = 3
SAMPLE_FIXED_SEED   = 42


# ──────────────────────────────────────────────────────────────────────────────
# メイン構築関数
# ──────────────────────────────────────────────────────────────────────────────
def build_lora_train_tab(
    parent: ttk.Frame,
    paths,                          # AppPaths
    log_fn: Callable[[str], None],
    get_model_choices: Callable[[], list[str]],
) -> "_TrainState":
    """LoRA学習タブの全UIを parent に構築する。"""

    state = _TrainState(paths, log_fn, get_model_choices)

    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.BOTH, expand=True)

    # ── タブ構成 ──────────────────────────────────────────────────────────
    tab_model   = ttk.Frame(nb, padding=8)
    tab_dataset = ttk.Frame(nb, padding=8)
    tab_network = ttk.Frame(nb, padding=8)
    tab_train   = ttk.Frame(nb, padding=8)
    tab_adv     = ttk.Frame(nb, padding=8)
    tab_layer   = ttk.Frame(nb, padding=8)
    tab_monitor = ttk.Frame(nb, padding=8)
    tab_monitor_layer = ttk.Frame(nb, padding=8)
    tab_sample  = ttk.Frame(nb, padding=8)
    tab_preset  = ttk.Frame(nb, padding=8)

    nb.add(tab_model,   text="  モデル  ")
    nb.add(tab_dataset, text="  データセット  ")
    nb.add(tab_network, text="  ネットワーク  ")
    nb.add(tab_train,   text="  学習設定  ")
    nb.add(tab_adv,     text="  詳細  ")
    nb.add(tab_layer,   text="  階層学習  ")
    nb.add(tab_monitor, text="  モニターグラフ  ")
    nb.add(tab_monitor_layer, text="  モニター階層  ")
    nb.add(tab_sample,  text="  サンプル生成  ")
    nb.add(tab_preset,  text="  プリセット  ")

    _build_model_tab(tab_model,   state)
    _build_dataset_tab(tab_dataset, state)
    _build_network_tab(tab_network, state)
    _build_train_tab(tab_train,   state)
    _build_adv_tab(tab_adv,       state)
    _build_layer_train_tab(tab_layer, state)
    _build_monitor_tab(tab_monitor, state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_sample_tab(tab_sample, state)
    _build_train_preset_tab(tab_preset, state)

    # ── 実行パネル（主要な中タブの画面下）────────────────────────────────
    for tab in (tab_model, tab_dataset, tab_network, tab_train):
        _build_run_panel(tab, state)

    return state


# ──────────────────────────────────────────────────────────────────────────────
# 状態オブジェクト（変数を一元管理）
# ──────────────────────────────────────────────────────────────────────────────
class _TrainState:
    def __init__(self, paths, log_fn, get_model_choices):
        self.paths = paths
        self.log_fn = log_fn
        self.get_model_choices = get_model_choices
        self._proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._monitor_queue: queue.Queue[str] = queue.Queue()
        self._monitor_layer_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()

        # ── モデル ──────────────────────────────────────────────
        self.model_path     = tk.StringVar()
        self.vae_path       = tk.StringVar()
        self.qwen3_path     = tk.StringVar()
        self.llm_adapter_path = tk.StringVar()
        self.output_dir     = tk.StringVar(value=str(paths.lora))
        self.output_name    = tk.StringVar(value="lora_output")
        self.precision      = tk.StringVar(value="bf16")

        # ── データセット ─────────────────────────────────────────
        self.train_data_dir = tk.StringVar()
        self.resolution     = tk.StringVar(value="512,512")
        self.batch_size     = tk.IntVar(value=1)
        self.cache_latents  = tk.BooleanVar(value=True)
        self.cache_latents_to_disk = tk.BooleanVar(value=False)
        self.cache_te_outputs = tk.BooleanVar(value=False)
        self.shuffle_caption = tk.BooleanVar(value=False)
        self.caption_extension = tk.StringVar(value=".txt")
        self.keep_tokens    = tk.IntVar(value=0)
        self.flip_aug       = tk.BooleanVar(value=False)
        self.enable_bucket  = tk.BooleanVar(value=True)
        self.bucket_no_upscale = tk.BooleanVar(value=True)
        self.min_bucket_reso = tk.IntVar(value=256)
        self.max_bucket_reso = tk.IntVar(value=1024)

        # ── ネットワーク ─────────────────────────────────────────
        self.network_dim    = tk.IntVar(value=32)
        self.network_alpha  = tk.DoubleVar(value=16.0)
        self.network_module = tk.StringVar(value="networks.lora")
        self.network_train_unet_only = tk.BooleanVar(value=True)
        self.network_weights = tk.StringVar()

        # ── 学習設定 ─────────────────────────────────────────────
        self.lr             = tk.StringVar(value="1e-4")
        self.lr_scheduler   = tk.StringVar(value="cosine_with_restarts")
        self.lr_warmup_steps = tk.IntVar(value=0)
        self.optimizer      = tk.StringVar(value="AdamW")
        self.optimizer_args = tk.StringVar(value="")
        self.max_train_epochs = tk.IntVar(value=10)
        self.save_every_n_epochs = tk.IntVar(value=1)
        self.seed           = tk.StringVar(value="42")
        self.gradient_checkpointing = tk.BooleanVar(value=True)
        self.grad_accum     = tk.IntVar(value=1)
        self.mixed_precision = tk.StringVar(value="bf16")
        self.xformers       = tk.BooleanVar(value=False)
        self.sdpa           = tk.BooleanVar(value=False)

        # ── 詳細 (Anima固有) ────────────────────────────────────
        self.timestep_sampling = tk.StringVar(value="sigmoid")
        self.discrete_flow_shift = tk.DoubleVar(value=1.0)
        self.sigmoid_scale  = tk.DoubleVar(value=1.0)
        self.weighting_scheme = tk.StringVar(value="none")
        self.attn_mode      = tk.StringVar(value="torch")
        self.split_attn     = tk.BooleanVar(value=False)
        self.blocks_to_swap = tk.IntVar(value=0)
        self.unsloth_offload_checkpointing = tk.BooleanVar(value=False)
        self.cpu_offload_checkpointing = tk.BooleanVar(value=False)
        self.vae_chunk_size = tk.StringVar(value="")
        self.vae_disable_cache = tk.BooleanVar(value=False)
        self.qwen3_max_token_length = tk.IntVar(value=512)
        self.t5_max_token_length = tk.IntVar(value=512)
        self.t5_tokenizer_path = tk.StringVar(value="")
        self.max_grad_norm  = tk.DoubleVar(value=1.0)
        # サンプル生成 共通設定
        self.sample_every_n_epochs  = tk.StringVar(value="1")
        self.sample_width           = tk.IntVar(value=512)
        self.sample_height          = tk.IntVar(value=512)
        self.sample_steps           = tk.IntVar(value=30)
        self.sample_scale           = tk.DoubleVar(value=7.5)
        self.sample_flow_shift      = tk.DoubleVar(value=3.0)
        # サンプルA
        self.sample_enabled         = tk.BooleanVar(value=False)
        self.sample_prompt          = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        # サンプルB
        self.sample_b_enabled          = tk.BooleanVar(value=False)
        self.sample_b_prompt           = tk.StringVar(value="")
        self.sample_b_negative_prompt  = tk.StringVar(value="")
        # 旧互換
        self.sample_prompts = tk.StringVar(value="")

        # 階層学習
        self.layer_train_enabled = tk.BooleanVar(value=False)
        self.layer_display_mode  = tk.StringVar(value="Matrix")
        self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
        # 階層学習スライダーウィジェット参照（プリセット読み込み時に _refresh_layer_controls へ渡す）
        self.layer_canvas: "tk.Canvas | None" = None
        self.layer_inner:  "ttk.Frame | None" = None
        # 階層学習スライダーウィジェット参照（プリセット読み込み時に _refresh_layer_controls へ渡す）
        self.layer_canvas: "tk.Canvas | None" = None
        self.layer_inner:  "ttk.Frame | None" = None
        # 階層学習スライダーウィジェット参照（プリセット読み込み時に _refresh_layer_controls へ渡す）
        self.layer_canvas: "tk.Canvas | None" = None
        self.layer_inner:  "ttk.Frame | None" = None

        # Validation / EarlyStopping
        self.validation_split        = tk.StringVar(value="0.0")
        self.early_stopping          = tk.BooleanVar(value=False)
        self.early_stopping_mode     = tk.StringVar(value="epoch")
        self.early_stopping_patience = tk.IntVar(value=3)
        self.early_stopping_threshold = tk.DoubleVar(value=0.01)

        # ステータス
        self.status_var     = tk.StringVar(value="待機中")
        self._log_widgets: list[tk.Text] = []
        self._log_drain_started = False


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー: ラベル付きエントリ行
# ──────────────────────────────────────────────────────────────────────────────
def _row(parent, row: int, label: str, widget_factory):
    ttk.Label(parent, text=label, width=24, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    w = widget_factory(parent)
    w.grid(row=row, column=1, sticky=tk.EW, padx=(0, 4), pady=3)
    return w


def _browse_file(var: tk.StringVar, title="ファイル選択", filetypes=None, parent=None):
    ft = filetypes or [("safetensors", "*.safetensors"), ("All", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title="フォルダ選択"):
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _entry_browse_row(parent, row: int, label: str, var: tk.StringVar,
                      is_dir=False, filetypes=None):
    """エントリ + Browse ボタンを1行に配置する。"""
    ttk.Label(parent, text=label, width=24, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    ttk.Entry(parent, textvariable=var).grid(
        row=row, column=1, sticky=tk.EW, padx=(0, 2), pady=3
    )
    cmd = (lambda v=var: _browse_dir(v)) if is_dir else (lambda v=var, ft=filetypes: _browse_file(v, filetypes=ft))
    ttk.Button(parent, text="Browse", width=7, command=cmd).grid(
        row=row, column=2, padx=(0, 4), pady=3
    )


# ──────────────────────────────────────────────────────────────────────────────
# タブ1: モデル
# ──────────────────────────────────────────────────────────────────────────────
def _build_model_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="モデルパス")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)

    _entry_browse_row(lf, 0, "DiT (pretrained_model)", s.model_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    _entry_browse_row(lf, 1, "VAE", s.vae_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    _entry_browse_row(lf, 2, "Qwen3テキストエンコーダ", s.qwen3_path,
                      filetypes=[("safetensors", "*.safetensors"), ("dir", "*")])
    _entry_browse_row(lf, 3, "LLM Adapter (任意)", s.llm_adapter_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])

    lf2 = ttk.LabelFrame(parent, text="出力設定")
    lf2.pack(fill=tk.X)
    lf2.columnconfigure(1, weight=1)

    _entry_browse_row(lf2, 0, "出力フォルダ", s.output_dir, is_dir=True)

    ttk.Label(lf2, text="出力ファイル名", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.output_name).grid(
        row=1, column=1, sticky=tk.EW, padx=(0, 4), pady=3)

    ttk.Label(lf2, text="保存精度", width=24, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.precision, values=PRECISIONS,
                 state="readonly", width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ2: データセット
# ──────────────────────────────────────────────────────────────────────────────
def _build_dataset_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="データセット")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)

    _entry_browse_row(lf, 0, "学習データフォルダ", s.train_data_dir, is_dir=True)

    ttk.Label(lf, text="解像度 (W,H)", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.resolution, width=14).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="バッチサイズ", width=24, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=64, textvariable=s.batch_size, width=6).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="キャプション拡張子", width=24, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.caption_extension, width=10).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="keep_tokens", width=24, anchor=tk.W).grid(
        row=4, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=255, textvariable=s.keep_tokens, width=6).grid(
        row=4, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    # チェックボックス群
    lf2 = ttk.LabelFrame(parent, text="キャッシュ / 拡張")
    lf2.pack(fill=tk.X, pady=(0, 8))

    ttk.Checkbutton(lf2, text="cache_latents", variable=s.cache_latents).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cache_latents_to_disk", variable=s.cache_latents_to_disk).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cache_text_encoder_outputs", variable=s.cache_te_outputs).grid(
        row=0, column=2, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="shuffle_caption", variable=s.shuffle_caption).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="flip_aug", variable=s.flip_aug).grid(
        row=1, column=1, sticky=tk.W, padx=8, pady=3)

    # バケット設定
    lf3 = ttk.LabelFrame(parent, text="バケット設定")
    lf3.pack(fill=tk.X)
    lf3.columnconfigure(1, weight=1)

    ttk.Checkbutton(lf3, text="enable_bucket", variable=s.enable_bucket).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf3, text="bucket_no_upscale", variable=s.bucket_no_upscale).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)

    ttk.Label(lf3, text="min_bucket_reso", width=20, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(8, 2), pady=3)
    ttk.Spinbox(lf3, from_=64, to=2048, increment=16,
                textvariable=s.min_bucket_reso, width=8).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf3, text="max_bucket_reso", width=20, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(8, 2), pady=3)
    ttk.Spinbox(lf3, from_=64, to=4096, increment=16,
                textvariable=s.max_bucket_reso, width=8).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ3: ネットワーク
# ──────────────────────────────────────────────────────────────────────────────
def _build_network_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="LoRAネットワーク設定")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)

    ttk.Label(lf, text="network_module", width=24, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.network_module,
                 values=["networks.lora", "networks.lora_flux"],
                 state="readonly", width=22).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="network_dim (rank)", width=24, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=512, textvariable=s.network_dim, width=8).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf, text="network_alpha", width=24, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.network_alpha, width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Checkbutton(lf, text="network_train_unet_only (DiTのみ学習)",
                    variable=s.network_train_unet_only).grid(
        row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    _entry_browse_row(lf, 4, "network_weights (再開用)", s.network_weights,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])



# ──────────────────────────────────────────────────────────────────────────────
# タブ4: 学習設定
# ──────────────────────────────────────────────────────────────────────────────
def _build_train_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="学習パラメータ")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    # row 0: LR / Scheduler
    ttk.Label(lf, text="learning_rate", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.lr, width=12).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="lr_scheduler", width=16, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.lr_scheduler, values=LR_SCHEDULERS,
                 state="readonly", width=22).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: warmup / optimizer
    ttk.Label(lf, text="lr_warmup_steps", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=100000, textvariable=s.lr_warmup_steps, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="optimizer", width=16, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.optimizer, values=OPTIMIZERS,
                 state="readonly", width=22).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: optimizer_args
    ttk.Label(lf, text="optimizer_args", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.optimizer_args).grid(
        row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=3)

    # row 3: epochs / save_every
    ttk.Label(lf, text="max_train_epochs", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=9999, textvariable=s.max_train_epochs, width=8).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="save_every_n_epochs", width=20, anchor=tk.W).grid(
        row=3, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=9999, textvariable=s.save_every_n_epochs, width=8).grid(
        row=3, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 4: seed / grad_accum
    ttk.Label(lf, text="seed", width=22, anchor=tk.W).grid(
        row=4, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.seed, width=10).grid(
        row=4, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="gradient_accumulation", width=20, anchor=tk.W).grid(
        row=4, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=256, textvariable=s.grad_accum, width=8).grid(
        row=4, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 5: mixed_precision / max_grad_norm
    ttk.Label(lf, text="mixed_precision", width=22, anchor=tk.W).grid(
        row=5, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.mixed_precision, values=PRECISIONS,
                 state="readonly", width=10).grid(
        row=5, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="max_grad_norm", width=20, anchor=tk.W).grid(
        row=5, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf, textvariable=s.max_grad_norm, width=10).grid(
        row=5, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    lf2 = ttk.LabelFrame(parent, text="メモリ最適化")
    lf2.pack(fill=tk.X)
    ttk.Checkbutton(lf2, text="gradient_checkpointing", variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ5: 詳細 (Anima固有)
# ──────────────────────────────────────────────────────────────────────────────
def _build_adv_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="Anima固有設定")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    # row 0: timestep_sampling / discrete_flow_shift
    ttk.Label(lf, text="timestep_sampling", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.timestep_sampling, values=TIMESTEP_SAMPLING,
                 state="readonly", width=14).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="discrete_flow_shift", width=20, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf, textvariable=s.discrete_flow_shift, width=10).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: sigmoid_scale / weighting_scheme
    ttk.Label(lf, text="sigmoid_scale", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.sigmoid_scale, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="weighting_scheme", width=20, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.weighting_scheme, values=WEIGHTING_SCHEMES,
                 state="readonly", width=14).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: attn_mode / blocks_to_swap
    ttk.Label(lf, text="attn_mode", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.attn_mode, values=ATTN_MODES,
                 state="readonly", width=14).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="blocks_to_swap (0=無効)", width=20, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=100, textvariable=s.blocks_to_swap, width=8).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 3: vae_chunk_size / qwen3_max_token_length
    ttk.Label(lf, text="vae_chunk_size (空欄=無効)", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.vae_chunk_size, width=10).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="qwen3_max_token_length", width=20, anchor=tk.W).grid(
        row=3, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.qwen3_max_token_length, width=8).grid(
        row=3, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 4: t5_max_token_length / t5_tokenizer_path
    ttk.Label(lf, text="t5_max_token_length", width=22, anchor=tk.W).grid(
        row=4, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.t5_max_token_length, width=8).grid(
        row=4, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="t5_tokenizer_path", width=20, anchor=tk.W).grid(
        row=4, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    t5_path_frame = ttk.Frame(lf)
    t5_path_frame.grid(row=4, column=3, sticky=tk.EW, padx=(0, 4), pady=3)
    t5_path_frame.columnconfigure(0, weight=1)
    ttk.Entry(t5_path_frame, textvariable=s.t5_tokenizer_path).grid(
        row=0, column=0, sticky=tk.EW, padx=(0, 2))
    ttk.Button(
        t5_path_frame,
        text="Browse",
        width=7,
        command=lambda: _browse_dir(s.t5_tokenizer_path),
    ).grid(row=0, column=1)

    # チェックボックス行
    lf2 = ttk.LabelFrame(parent, text="オフロード / キャッシュ")
    lf2.pack(fill=tk.X, pady=(0, 8))

    ttk.Checkbutton(lf2, text="split_attn", variable=s.split_attn).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="unsloth_offload_checkpointing", variable=s.unsloth_offload_checkpointing).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cpu_offload_checkpointing", variable=s.cpu_offload_checkpointing).grid(
        row=0, column=2, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="vae_disable_cache", variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)

    # Validation / EarlyStopping
    lf4 = ttk.LabelFrame(parent, text="Validation / Early Stopping")
    lf4.pack(fill=tk.X)
    lf4.columnconfigure(1, weight=1)
    lf4.columnconfigure(3, weight=1)

    # row0: validation_split
    ttk.Label(lf4, text="検証データ分割比率", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf4, textvariable=s.validation_split, width=10).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(lf4, text="0.0=無効  例: 0.1 → 学習データの10%を検証用に分割",
              foreground="#64748B").grid(
        row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    # row1: early_stopping ON/OFF + mode
    es_cb = ttk.Checkbutton(lf4, text="Early Stopping を有効にする",
                            variable=s.early_stopping)
    es_cb.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Label(lf4, text="判定タイミング", width=14, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Combobox(lf4, textvariable=s.early_stopping_mode,
                 values=["epoch", "step"],
                 state="readonly", width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(lf4, text="epoch=エポック終了時に判定 / step=指定ステップごとに判定",
              foreground="#64748B").grid(
        row=2, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=(0, 3))

    # row3: patience / threshold
    ttk.Label(lf4, text="連続悪化の許容回数", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf4, from_=1, to=100,
                textvariable=s.early_stopping_patience, width=8).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf4, text="悪化判定しきい値", width=16, anchor=tk.W).grid(
        row=3, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf4, textvariable=s.early_stopping_threshold, width=10).grid(
        row=3, column=3, sticky=tk.W, padx=(0, 4), pady=3)
    ttk.Label(lf4,
              text="しきい値: Val Loss が前回比でこの値を超えて上昇したとき悪化とみなす",
              foreground="#64748B").grid(
        row=4, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=(0, 4))


# ──────────────────────────────────────────────────────────────────────────────
# タブ6: 階層学習 (Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
def _build_layer_train_tab(parent: ttk.Frame, s: _TrainState) -> None:
    """階層別学習率スケールを設定するタブ（Phase 1: ログ出力のみ）。"""
    # ── ヘッダ行: ON/OFF + モード切り替え + プリセット読み込み ──────────
    hdr = ttk.Frame(parent)
    hdr.pack(fill=tk.X, pady=(0, 4))

    ttk.Checkbutton(
        hdr, text="階層学習を有効にする",
        variable=s.layer_train_enabled,
        command=lambda: _refresh_layer_controls(s, ctrl_canvas, ctrl_inner),
    ).pack(side=tk.LEFT, padx=(0, 12))

    ttk.Label(hdr, text="モード:").pack(side=tk.LEFT)
    mode_cb = ttk.Combobox(
        hdr, textvariable=s.layer_display_mode,
        values=list(LAYER_TRAIN_MODES), state="readonly", width=14,
    )
    mode_cb.pack(side=tk.LEFT, padx=(2, 12))
    mode_cb.bind(
        "<<ComboboxSelected>>",
        lambda _e: _refresh_layer_controls(s, ctrl_canvas, ctrl_inner),
    )

    ttk.Button(
        hdr, text="プリセット読み込み",
        command=lambda: _load_layer_preset(s, ctrl_canvas, ctrl_inner),
    ).pack(side=tk.LEFT)

    # ── スクロール可能なスライダーエリア ──────────────────────────────
    canvas_frame = ttk.LabelFrame(parent, text="ブロック別スケール (0.0 = freeze / 1.0 = base LR)")
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    ctrl_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
    vscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=ctrl_canvas.yview)
    ctrl_canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)
    ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    ctrl_inner = ttk.Frame(ctrl_canvas)
    ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor="nw")
    s.layer_canvas = ctrl_canvas
    s.layer_inner  = ctrl_inner
    s.layer_canvas = ctrl_canvas
    s.layer_inner  = ctrl_inner
    s.layer_canvas = ctrl_canvas
    s.layer_inner  = ctrl_inner
    ctrl_inner.bind(
        "<Configure>",
        lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
    )

    # ── ステータスラベル ──────────────────────────────────────────────
    s._layer_status_var = tk.StringVar(value="(無効)")
    ttk.Label(parent, textvariable=s._layer_status_var, foreground="#334155").pack(
        anchor=tk.W, padx=4, pady=(2, 0)
    )

    # 初期描画
    _refresh_layer_controls(s, ctrl_canvas, ctrl_inner)


def _layer_group_names(mode: str) -> list[str]:
    """モードに応じたグループ名リストを返す（gui.py の _group_names_for_mode Matrix相当）。"""
    if mode == "Matrix":
        return [f"{b}_{c}" for b in MATRIX_BLOCKS for c in MATRIX_COMPONENTS]
    if mode == "Component":
        return list(COMPONENT_GROUPS)
    # Transformer: blocks.0 ～ blocks.27 の28ブロック
    return [f"blocks.{i}" for i in range(28)]


def _refresh_layer_controls(
    s: _TrainState,
    canvas: tk.Canvas,
    inner: ttk.Frame,
) -> None:
    """スライダーエリアを現在のモードで再構築する。"""
    for child in inner.winfo_children():
        child.destroy()

    if not s.layer_train_enabled.get():
        ttk.Label(inner, text="階層学習は無効です。チェックボックスをONにしてください。").grid(
            row=0, column=0, padx=8, pady=8, sticky=tk.W
        )
        s._layer_status_var.set("(無効)")
        canvas.configure(scrollregion=canvas.bbox("all"))
        return

    mode = s.layer_display_mode.get()
    groups = _layer_group_names(mode)
    old = {k: v.get() for k, v in s.layer_parameter_vars.items()}
    s.layer_parameter_vars = {}

    cols = LAYER_COLUMNS
    for idx, name in enumerate(groups):
        var = tk.DoubleVar(value=old.get(name, 1.0))
        s.layer_parameter_vars[name] = var

        grid_row = idx // cols
        base_col = (idx % cols) * 3

        ttk.Label(inner, text=name, width=20).grid(
            row=grid_row, column=base_col, sticky=tk.W, padx=(8, 2), pady=3
        )
        scale = ttk.Scale(inner, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
        scale.grid(row=grid_row, column=base_col + 1, sticky=tk.EW, padx=2, pady=3)
        scale.bind("<ButtonRelease-1>", lambda e, v=var: _snap_scale(v))

        entry = ttk.Entry(inner, textvariable=var, width=6)
        entry.grid(row=grid_row, column=base_col + 2, sticky=tk.W, padx=(2, 12), pady=3)
        entry.bind("<FocusOut>", lambda e, v=var: _clamp_var(v))

    for c in range(cols):
        inner.columnconfigure(c * 3 + 1, weight=1)

    canvas.configure(scrollregion=canvas.bbox("all"))

    # Component モード: 構造限界の警告を表示
    if mode == "Component":
        warn_row = (len(groups) + cols - 1) // cols
        warn_lbl = tk.Label(
            inner,
            text=(
                "⚠ Component モードはブロック情報をLoRAキーから分解できないため、"
                "全ブロック共通の平均スケールとして適用されます。\n"
                "ブロック別精度が必要な場合は Transformer または Matrix モードを使用してください。"
            ),
            foreground="red",
            justify=tk.LEFT,
            wraplength=600,
        )
        warn_lbl.grid(
            row=warn_row, column=0, columnspan=cols * 3,
            sticky=tk.W, padx=8, pady=(6, 2)
        )
        canvas.configure(scrollregion=canvas.bbox("all"))
        s._layer_status_var.set(
            f"mode={mode}  {len(groups)} グループ  "
            "[警告] ブロック別精度低下あり・構造限界"
        )
    else:
        s._layer_status_var.set(f"mode={mode}  {len(groups)} グループ")


def _snap_scale(var: tk.DoubleVar) -> None:
    """スケールを0.05刻みにスナップ。"""
    v = var.get()
    var.set(round(round(v / 0.05) * 0.05, 4))


def _clamp_var(var: tk.DoubleVar) -> None:
    """エントリ値を 0.0–1.0 にクランプ。"""
    try:
        var.set(max(0.0, min(1.0, float(var.get()))))
    except tk.TclError:
        var.set(1.0)


def _convert_preset_scales(
    scales: dict[str, float],
    preset_mode: str,
    target_mode: str,
) -> dict[str, float]:
    """プリセットの parameter_scales を target_mode のキー形式に疑似コンバートする。

    同一モード: そのまま返す。
    Component → Matrix: Component スケールを全 Block に展開。
        blocks.N_Attention 系キーはブロックカテゴリ別に平均して
        Input_Attention / Middle_Attention / Output_Attention に変換。
    Component → Transformer: 全コンポーネント平均を28ブロックに展開。
    Matrix → Component: Block 次元を平均してコンポーネントに集約。
        Attention は Input/Middle/Output の平均を "Attention" キーに集約。
    Matrix → Transformer: ブロックカテゴリ内の全コンポーネント平均を各ブロックに展開。
    Transformer → Matrix: ブロックカテゴリの平均を Block_Component 全てに適用。
    Transformer → Component: ブロックカテゴリの平均を全コンポーネントに適用。
    """
    import re as _re

    if preset_mode == target_mode:
        return dict(scales)

    # ── Component プリセットの Attention キー抽出ヘルパー ──────────
    # merge.py の Component モードは Attention を "blocks.N_Attention" 形式で保存する
    def _attn_by_block_cat(scales: dict) -> dict[str, float]:
        """blocks.N_Attention キーをブロックカテゴリ別に平均して返す。
        戻り値: {"Input": 0.8, "Middle": 0.7, "Output": 0.6}
        """
        cat_vals: dict[str, list[float]] = {b: [] for b in MATRIX_BLOCKS}
        for k, v in scales.items():
            m = _re.match(r"blocks\.(\d+)_Attention$", k)
            if m:
                n = int(m.group(1))
                if 0 <= n < 28:
                    cat = _BLOCK_CAT[n]
                    cat_vals[cat].append(float(v))
        return {
            cat: (sum(vals) / len(vals) if vals else 1.0)
            for cat, vals in cat_vals.items()
        }

    # ── Component → Matrix ──────────────────────────────────────
    if preset_mode == "Component" and target_mode == "Matrix":
        attn_by_cat = _attn_by_block_cat(scales)
        result = {}
        for b in MATRIX_BLOCKS:
            result[f"{b}_Attention"] = attn_by_cat[b]
            for c in ("MLP", "Norm", "ResNet", "Timestep", "Other"):
                result[f"{b}_{c}"] = float(scales.get(c, 1.0))
        return result

    # ── Component → Transformer ──────────────────────────────────
    if preset_mode == "Component" and target_mode == "Transformer":
        attn_by_cat = _attn_by_block_cat(scales)
        result = {}
        for i in range(28):
            cat = _BLOCK_CAT[i]
            comp_vals = [attn_by_cat[cat]]
            comp_vals += [float(scales.get(c, 1.0)) for c in ("MLP", "Norm", "ResNet", "Timestep", "Other")]
            result[f"blocks.{i}"] = sum(comp_vals) / len(comp_vals)
        return result

    # ── Matrix → Component ───────────────────────────────────────
    if preset_mode == "Matrix" and target_mode == "Component":
        result = {}
        # Attention: Input/Middle/Output の平均 → "Attention"
        attn_vals = [float(scales.get(f"{b}_Attention", 1.0)) for b in MATRIX_BLOCKS]
        result["Attention"] = sum(attn_vals) / len(attn_vals)
        for c in ("MLP", "Norm", "ResNet", "Timestep", "Other"):
            vals = [float(scales.get(f"{b}_{c}", 1.0)) for b in MATRIX_BLOCKS]
            result[c] = sum(vals) / len(vals)
        return result

    # ── Matrix → Transformer ─────────────────────────────────────
    if preset_mode == "Matrix" and target_mode == "Transformer":
        result = {}
        for i in range(28):
            cat = _BLOCK_CAT[i]
            comp_vals = [float(scales.get(f"{cat}_{c}", 1.0)) for c in MATRIX_COMPONENTS]
            result[f"blocks.{i}"] = sum(comp_vals) / len(comp_vals)
        return result

    # ── Transformer → Matrix ─────────────────────────────────────
    if preset_mode == "Transformer" and target_mode == "Matrix":
        # ブロックカテゴリの平均を Block_Component 全てに適用
        cat_avg: dict[str, float] = {}
        for cat in MATRIX_BLOCKS:
            idxs = [i for i, c in enumerate(_BLOCK_CAT) if c == cat]
            vals = [float(scales.get(f"blocks.{i}", 1.0)) for i in idxs]
            cat_avg[cat] = sum(vals) / len(vals)
        result = {}
        for b in MATRIX_BLOCKS:
            for c in MATRIX_COMPONENTS:
                result[f"{b}_{c}"] = cat_avg[b]
        return result

    # ── Transformer → Component ──────────────────────────────────
    if preset_mode == "Transformer" and target_mode == "Component":
        avg = sum(float(scales.get(f"blocks.{i}", 1.0)) for i in range(28)) / 28
        return {c: avg for c in COMPONENT_GROUPS}

    # フォールバック: 変換不能の場合はそのまま返す
    return dict(scales)


def _load_layer_preset(s: _TrainState, canvas: tk.Canvas, inner: ttk.Frame) -> None:
    """preset/merge/*.json から parameter_scales と layer_display_mode を読み込む。"""
    from tkinter import filedialog
    preset_dir = s.paths.root / "preset" / "merge"
    preset_dir.mkdir(parents=True, exist_ok=True)
    path = filedialog.askopenfilename(
        title="プリセット選択",
        initialdir=str(preset_dir),
        filetypes=[("JSON", "*.json"), ("All", "*.*")],
    )
    if not path:
        return
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        from tkinter import messagebox
        messagebox.showerror("プリセット読み込みエラー", str(exc))
        return

    new_mode = data.get("layer_display_mode", "Matrix")
    if new_mode in LAYER_TRAIN_MODES:
        s.layer_display_mode.set(new_mode)

    _refresh_layer_controls(s, canvas, inner)

    scales = data.get("parameter_scales", {})

    # Component プリセット → 現在モードへの疑似コンバート
    converted = _convert_preset_scales(
        scales, preset_mode=new_mode, target_mode=s.layer_display_mode.get()
    )

    for k, v in converted.items():
        if k in s.layer_parameter_vars:
            try:
                s.layer_parameter_vars[k].set(float(v))
            except (ValueError, tk.TclError):
                pass

    s.log_fn(
        f"[LayerLR] プリセット読み込み: {Path(path).name}  "
        f"preset_mode={new_mode} -> gui_mode={s.layer_display_mode.get()}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# タブ7: モニターグラフ
# ──────────────────────────────────────────────────────────────────────────────
def _build_monitor_tab(parent: ttk.Frame, s: _TrainState) -> None:
    """モニターグラフタブ。MonitorGraph クラスを埋め込む。"""
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

    # 学習ログ（既存 _build_run_panel と同仕様）
    log_frame = ttk.LabelFrame(parent, text="学習ログ")
    log_frame.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))

    log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 12))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)

    s._log_widgets.append(log_text)


# ──────────────────────────────────────────────────────────────────────────────
# タブ8: モニター階層
# ──────────────────────────────────────────────────────────────────────────────
def _build_monitor_layer_tab(parent: ttk.Frame, s: _TrainState) -> None:
    """階層学習の実効LRモニターを埋め込む。"""
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


# ──────────────────────────────────────────────────────────────────────────────
# タブ9: サンプル生成
# ──────────────────────────────────────────────────────────────────────────────
def _sample_dir(s: _TrainState) -> Path:
    return s.paths.root / "log" / "sample_gen"


def _sample_prompt_path(s: _TrainState) -> Path:
    return _sample_dir(s) / "_sample_prompt.txt"


def _build_sample_prompt_line_for(
    prompt: str, neg: str, s: _TrainState, seed: int = SAMPLE_FIXED_SEED
) -> str:
    width      = max(64, int(s.sample_width.get()))
    height     = max(64, int(s.sample_height.get()))
    steps      = max(1,  int(s.sample_steps.get()))
    scale      = float(s.sample_scale.get())
    flow_shift = float(s.sample_flow_shift.get())
    line = (
        f"{prompt} --w {width} --h {height} --s {steps} "
        f"--l {scale:g} --fs {flow_shift:g} --d {seed}"
    )
    if neg:
        line += f" --n {neg}"
    return line


def _build_sample_prompt_line(s: _TrainState) -> str:
    # 後方互換用: サンプルAのプロンプト行を返す
    return _build_sample_prompt_line_for(
        s.sample_prompt.get().strip(),
        s.sample_negative_prompt.get().strip(),
        s,
    )


def _write_sample_prompt_file(s: _TrainState) -> Path:
    lines = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_build_sample_prompt_line_for(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            s,
            seed=SAMPLE_FIXED_SEED,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_build_sample_prompt_line_for(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            s,
            seed=SAMPLE_FIXED_SEED + 1,
        ))
    path = _sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _extract_sample_epoch(path: Path) -> str:
    # lora ファイル名: lora_output_e000001_00_... → e プレフィックスあり
    m = re.search(r"(?:^|_)e(\d+)(?:_|$)", path.stem)
    if m:
        try:
            return str(int(m.group(1)))
        except ValueError:
            return m.group(1)
    # leco ファイル名: leco_output_000002_00_... → e プレフィックスなし6桁数字
    m = re.search(r"(?:^|_)(\d{6})(?:_|$)", path.stem)
    if m:
        try:
            return str(int(m.group(1)))
        except ValueError:
            return m.group(1)
    return "-"


def _build_sample_ab_panel(
    parent: ttk.Frame,
    s,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    sample_dir: Path,
    glob_pattern: str,
    label: str,
    is_leco: bool = False,
) -> None:
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    top = ttk.Frame(parent)
    top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    top.columnconfigure(1, weight=1)

    ttk.Checkbutton(
        top, text=f"サンプル{label}を有効にする", variable=enabled_var
    ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=(2, 4), pady=2)

    ttk.Label(top, text="出力先: ", foreground="#475569").grid(
        row=1, column=0, sticky=tk.W, padx=(2, 0), pady=2)
    ttk.Label(top, text=str(sample_dir), foreground="#1D4ED8").grid(
        row=1, column=1, columnspan=3, sticky=tk.W, pady=2)

    ttk.Label(top, text="prompt", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=prompt_var).grid(
        row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

    ttk.Label(top, text="negative", width=16, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=neg_var).grid(
        row=3, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

    # ── ギャラリー ────────────────────────────────────────────────────────────
    gallery = ttk.LabelFrame(parent, text=f"最新サンプル{label}")
    gallery.grid(row=1, column=0, sticky=tk.NSEW)
    for c in range(5):
        gallery.columnconfigure(c, weight=1, uniform=f"sc_{label}")
    for r in range(2):
        gallery.rowconfigure(r, weight=1, uniform=f"sr_{label}")

    cells: list = []
    photo_refs: list = [None] * 10

    for idx in range(10):
        cell = ttk.Frame(gallery, padding=4)
        cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
        cell.columnconfigure(0, weight=1)
        cell.rowconfigure(0, weight=1)
        img_lbl = ttk.Label(cell, anchor=tk.CENTER)
        img_lbl.grid(row=0, column=0, sticky=tk.NSEW)
        ep_lbl = ttk.Label(cell, text="step -" if is_leco else "epoch -", anchor=tk.CENTER)
        ep_lbl.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        cells.append((img_lbl, ep_lbl))

    def _refresh(schedule_next: bool = False) -> None:
        files = []
        if sample_dir.exists():
            files = sorted(
                sample_dir.glob(glob_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:10]
        try:
            from PIL import Image as _Im, ImageTk as _ITk
        except Exception:
            _Im = _ITk = None

        for idx, (il, el) in enumerate(cells):
            if idx >= len(files):
                il.configure(image="", text="")
                el.configure(text="step -" if is_leco else "epoch -")
                photo_refs[idx] = None
                continue
            p = files[idx]
            _lbl_prefix = "step" if is_leco else "epoch"
            el.configure(text=f"{_lbl_prefix} {_extract_sample_epoch(p)}")
            if _Im is None:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None
                continue
            try:
                with _Im.open(p) as im:
                    im.thumbnail((220, 220))
                    ph = _ITk.PhotoImage(im.copy())
                photo_refs[idx] = ph
                il.configure(image=ph, text="")
            except Exception:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None

        if schedule_next:
            parent.after(2000, lambda: _refresh(True))

    def _clear_cache() -> None:
        import tkinter as _tk
        dlg = _tk.Toplevel(top)
        dlg.title("確認")
        dlg.resizable(False, False)
        dlg.grab_set()
        _tk.Label(dlg, text=f"サンプル{label} を全て削除しますか？",
                  font=("TkDefaultFont", 11), padx=20, pady=16).pack()
        bf = _tk.Frame(dlg)
        bf.pack(pady=(0, 12))
        def _no(): dlg.destroy()
        def _yes():
            dlg.destroy()
            if not sample_dir.exists():
                return
            deleted = errors = 0
            for f in sample_dir.glob(glob_pattern):
                try:
                    if f.is_file():
                        f.unlink(); deleted += 1
                except Exception:
                    errors += 1
            msg = f"[サンプル{label}] {deleted}件削除。"
            if errors: msg += f" ({errors}件失敗)"
            s.log_fn(msg)
            _refresh(False)
        _tk.Button(bf, text="いいえ", width=10, command=_no).pack(side=_tk.LEFT, padx=(0, 8))
        _tk.Button(bf, text="はい",   width=10, command=_yes).pack(side=_tk.LEFT)
        dlg.update_idletasks()
        px = top.winfo_rootx() + top.winfo_width()  // 2 - dlg.winfo_width()  // 2
        py = top.winfo_rooty() + top.winfo_height() // 2 - dlg.winfo_height() // 2
        dlg.geometry(f"+{px}+{py}")

    btn_row = ttk.Frame(top)
    btn_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 2))
    ttk.Button(btn_row, text="表示更新",     command=lambda: _refresh(False)).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_row, text="キャッシュクリア", command=_clear_cache).pack(side=tk.LEFT)

    _refresh(True)


def _build_sample_tab(parent: ttk.Frame, s: _TrainState) -> None:
    _build_sample_tab_common(parent, s, is_leco=False)


def _build_sample_tab_common(
    parent: ttk.Frame,
    s,
    is_leco: bool = False,
) -> None:

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    # ── 共通設定 ─────────────────────────────────────────────────────────────
    common = ttk.LabelFrame(parent, text="共通生成条件")
    common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    common.columnconfigure(1, weight=1)
    common.columnconfigure(3, weight=1)

    interval_label = "step間隔" if is_leco else "epoch間隔"
    interval_var   = s.sample_every_n_steps if is_leco else s.sample_every_n_epochs

    ttk.Label(common, text=interval_label, width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=99999, textvariable=interval_var, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text=f"seed固定: {SAMPLE_FIXED_SEED}", foreground="#64748B").grid(
        row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="width / height", width=16, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    sf = ttk.Frame(common)
    sf.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_width,  width=7).pack(side=tk.LEFT)
    ttk.Label(sf, text=" x ").pack(side=tk.LEFT)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_height, width=7).pack(side=tk.LEFT)

    ttk.Label(common, text="steps", width=10, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=1000, textvariable=s.sample_steps, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="scale", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_scale, width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text="flow_shift", width=10, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_flow_shift, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    if is_leco:
        ttk.Checkbutton(
            common, text="VAEを学習中保持 (sample_keep_vae)",
            variable=s.sample_keep_vae,
        ).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    # ── A/B 切り替えノートブック ─────────────────────────────────────────────
    ab_nb = ttk.Notebook(parent)
    ab_nb.grid(row=1, column=0, sticky=tk.NSEW)

    tab_a = ttk.Frame(ab_nb, padding=4)
    tab_b = ttk.Frame(ab_nb, padding=4)
    ab_nb.add(tab_a, text="  サンプルA  ")
    ab_nb.add(tab_b, text="  サンプルB  ")

    # LoRA: sd-scripts が {output_name}_{ts}_e{epoch:06d}_{idx:02d}.png を生成
    #   A = promptファイル1行目 → _00.png
    #   B = promptファイル2行目 → _01.png
    # LECO: _generate_samples_leco が step{N:06d}_a_s{seed}.png / _b_ を生成
    sample_dir = _sample_dir(s) if not is_leco else (s.paths.root / "log" / "sample_gen")
    pat_a = "leco_output_*_00_*.png" if is_leco else "*_e*_00_*.png"
    pat_b = "leco_output_*_01_*.png" if is_leco else "*_e*_01_*.png"

    _build_sample_ab_panel(
        tab_a, s,
        enabled_var=s.sample_enabled,
        prompt_var=s.sample_prompt,
        neg_var=s.sample_negative_prompt,
        sample_dir=sample_dir,
        glob_pattern=pat_a,
        label="A",
        is_leco=is_leco,
    )
    _build_sample_ab_panel(
        tab_b, s,
        enabled_var=s.sample_b_enabled,
        prompt_var=s.sample_b_prompt,
        neg_var=s.sample_b_negative_prompt,
        sample_dir=sample_dir,
        glob_pattern=pat_b,
        label="B",
        is_leco=is_leco,
    )


# ──────────────────────────────────────────────────────────────────────────────
# タブ10: LoRA学習プリセット
# ──────────────────────────────────────────────────────────────────────────────
def _build_train_preset_tab(parent: ttk.Frame, s: _TrainState) -> None:
    """_TrainState の全 tk.Variable を JSON に保存・復元するプリセットタブ。"""

    PRESET_DIR_REL = ("preset", "lora_train")

    def _preset_dir() -> Path:
        d = s.paths.root / PRESET_DIR_REL[0] / PRESET_DIR_REL[1]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── リストボックス ──────────────────────────────────────────
    list_frame = ttk.LabelFrame(parent, text="保存済みプリセット")
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    lb = tk.Listbox(list_frame, height=10, selectmode=tk.SINGLE)
    lb_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=lb.yview)
    lb.configure(yscrollcommand=lb_scroll.set)
    lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── 名前入力 + ボタン行 ──────────────────────────────────────
    name_row = ttk.Frame(parent)
    name_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(name_row, text="Name:").pack(side=tk.LEFT)
    name_var = tk.StringVar()
    ttk.Entry(name_row, textvariable=name_var, width=28).pack(side=tk.LEFT, padx=(4, 8))

    btn_row = ttk.Frame(parent)
    btn_row.pack(fill=tk.X)

    # ── 内部ヘルパー ─────────────────────────────────────────────
    def _refresh_list() -> None:
        lb.delete(0, tk.END)
        for p in sorted(_preset_dir().glob("*.json")):
            lb.insert(tk.END, p.stem)

    def _collect() -> dict:
        """_TrainState の全 tk.Variable 値を dict に収集する。"""
        simple = {
            "model_path":        s.model_path.get(),
            "vae_path":          s.vae_path.get(),
            "qwen3_path":        s.qwen3_path.get(),
            "llm_adapter_path":  s.llm_adapter_path.get(),
            "output_dir":        s.output_dir.get(),
            "output_name":       s.output_name.get(),
            "precision":         s.precision.get(),
            "train_data_dir":    s.train_data_dir.get(),
            "resolution":        s.resolution.get(),
            "batch_size":        int(s.batch_size.get()),
            "cache_latents":     bool(s.cache_latents.get()),
            "cache_latents_to_disk": bool(s.cache_latents_to_disk.get()),
            "cache_te_outputs":  bool(s.cache_te_outputs.get()),
            "shuffle_caption":   bool(s.shuffle_caption.get()),
            "caption_extension": s.caption_extension.get(),
            "keep_tokens":       int(s.keep_tokens.get()),
            "flip_aug":          bool(s.flip_aug.get()),
            "enable_bucket":     bool(s.enable_bucket.get()),
            "bucket_no_upscale": bool(s.bucket_no_upscale.get()),
            "min_bucket_reso":   int(s.min_bucket_reso.get()),
            "max_bucket_reso":   int(s.max_bucket_reso.get()),
            "network_dim":       int(s.network_dim.get()),
            "network_alpha":     float(s.network_alpha.get()),
            "network_module":    s.network_module.get(),
            "network_train_unet_only": bool(s.network_train_unet_only.get()),
            "network_weights":   s.network_weights.get(),
            "lr":                s.lr.get(),
            "lr_scheduler":      s.lr_scheduler.get(),
            "lr_warmup_steps":   int(s.lr_warmup_steps.get()),
            "optimizer":         s.optimizer.get(),
            "optimizer_args":    s.optimizer_args.get(),
            "max_train_epochs":  int(s.max_train_epochs.get()),
            "save_every_n_epochs": int(s.save_every_n_epochs.get()),
            "seed":              s.seed.get(),
            "gradient_checkpointing": bool(s.gradient_checkpointing.get()),
            "grad_accum":        int(s.grad_accum.get()),
            "mixed_precision":   s.mixed_precision.get(),
            "xformers":          bool(s.xformers.get()),
            "sdpa":              bool(s.sdpa.get()),
            "timestep_sampling": s.timestep_sampling.get(),
            "discrete_flow_shift": float(s.discrete_flow_shift.get()),
            "sigmoid_scale":     float(s.sigmoid_scale.get()),
            "weighting_scheme":  s.weighting_scheme.get(),
            "attn_mode":         s.attn_mode.get(),
            "split_attn":        bool(s.split_attn.get()),
            "blocks_to_swap":    int(s.blocks_to_swap.get()),
            "unsloth_offload_checkpointing": bool(s.unsloth_offload_checkpointing.get()),
            "cpu_offload_checkpointing": bool(s.cpu_offload_checkpointing.get()),
            "vae_chunk_size":    s.vae_chunk_size.get(),
            "vae_disable_cache": bool(s.vae_disable_cache.get()),
            "qwen3_max_token_length": int(s.qwen3_max_token_length.get()),
            "t5_max_token_length": int(s.t5_max_token_length.get()),
            "t5_tokenizer_path": s.t5_tokenizer_path.get(),
            "max_grad_norm":     float(s.max_grad_norm.get()),
            "sample_enabled":    bool(s.sample_enabled.get()),
            "sample_every_n_epochs": s.sample_every_n_epochs.get(),
            "sample_prompts":    s.sample_prompts.get(),
            "sample_prompt":     s.sample_prompt.get(),
            "sample_negative_prompt": s.sample_negative_prompt.get(),
            "sample_b_enabled":           bool(s.sample_b_enabled.get()),
            "sample_b_prompt":            s.sample_b_prompt.get(),
            "sample_b_negative_prompt":   s.sample_b_negative_prompt.get(),
            "sample_width":      int(s.sample_width.get()),
            "sample_height":     int(s.sample_height.get()),
            "sample_steps":      int(s.sample_steps.get()),
            "sample_scale":      float(s.sample_scale.get()),
            "sample_flow_shift": float(s.sample_flow_shift.get()),
            # 階層学習
            "layer_train_enabled": bool(s.layer_train_enabled.get()),
            "layer_display_mode":  s.layer_display_mode.get(),
            "layer_parameter_vars": {
                k: round(float(v.get()), 4)
                for k, v in s.layer_parameter_vars.items()
            },
            # Validation / EarlyStopping
            "validation_split":          s.validation_split.get(),
            "early_stopping":             bool(s.early_stopping.get()),
            "early_stopping_mode":        s.early_stopping_mode.get(),
            "early_stopping_patience":    int(s.early_stopping_patience.get()),
            "early_stopping_threshold":   float(s.early_stopping_threshold.get()),
        }
        return simple

    def _apply(data: dict) -> None:
        """dict の値を _TrainState の各 tk.Variable に反映する。"""
        def _s(var, key, default=None):
            if key in data:
                try:
                    var.set(data[key])
                except (tk.TclError, ValueError):
                    if default is not None:
                        var.set(default)

        _s(s.model_path,        "model_path",        "")
        _s(s.vae_path,          "vae_path",           "")
        _s(s.qwen3_path,        "qwen3_path",         "")
        _s(s.llm_adapter_path,  "llm_adapter_path",   "")
        _s(s.output_dir,        "output_dir",         "")
        _s(s.output_name,       "output_name",        "lora_output")
        _s(s.precision,         "precision",          "bf16")
        _s(s.train_data_dir,    "train_data_dir",     "")
        _s(s.resolution,        "resolution",         "512,512")
        _s(s.batch_size,        "batch_size",         1)
        _s(s.cache_latents,     "cache_latents",      True)
        _s(s.cache_latents_to_disk, "cache_latents_to_disk", False)
        _s(s.cache_te_outputs,  "cache_te_outputs",   False)
        _s(s.shuffle_caption,   "shuffle_caption",    False)
        _s(s.caption_extension, "caption_extension",  ".txt")
        _s(s.keep_tokens,       "keep_tokens",        0)
        _s(s.flip_aug,          "flip_aug",           False)
        _s(s.enable_bucket,     "enable_bucket",      True)
        _s(s.bucket_no_upscale, "bucket_no_upscale",  True)
        _s(s.min_bucket_reso,   "min_bucket_reso",    256)
        _s(s.max_bucket_reso,   "max_bucket_reso",    1024)
        _s(s.network_dim,       "network_dim",        32)
        _s(s.network_alpha,     "network_alpha",      16.0)
        _s(s.network_module,    "network_module",     "networks.lora")
        _s(s.network_train_unet_only, "network_train_unet_only", True)
        _s(s.network_weights,   "network_weights",    "")
        _s(s.lr,                "lr",                 "1e-4")
        _s(s.lr_scheduler,      "lr_scheduler",       "cosine_with_restarts")
        _s(s.lr_warmup_steps,   "lr_warmup_steps",    0)
        _s(s.optimizer,         "optimizer",          "AdamW8bit")
        _s(s.optimizer_args,    "optimizer_args",     "")
        _s(s.max_train_epochs,  "max_train_epochs",   10)
        _s(s.save_every_n_epochs, "save_every_n_epochs", 1)
        _s(s.seed,              "seed",               "42")
        _s(s.gradient_checkpointing, "gradient_checkpointing", True)
        _s(s.grad_accum,        "grad_accum",         1)
        _s(s.mixed_precision,   "mixed_precision",    "bf16")
        _s(s.xformers,          "xformers",           False)
        _s(s.sdpa,              "sdpa",               False)
        _s(s.timestep_sampling, "timestep_sampling",  "sigmoid")
        _s(s.discrete_flow_shift, "discrete_flow_shift", 1.0)
        _s(s.sigmoid_scale,     "sigmoid_scale",      1.0)
        _s(s.weighting_scheme,  "weighting_scheme",   "none")
        _s(s.attn_mode,         "attn_mode",          "torch")
        _s(s.split_attn,        "split_attn",         False)
        _s(s.blocks_to_swap,    "blocks_to_swap",     0)
        _s(s.unsloth_offload_checkpointing, "unsloth_offload_checkpointing", False)
        _s(s.cpu_offload_checkpointing, "cpu_offload_checkpointing", False)
        _s(s.vae_chunk_size,    "vae_chunk_size",     "")
        _s(s.vae_disable_cache, "vae_disable_cache",  False)
        _s(s.qwen3_max_token_length, "qwen3_max_token_length", 512)
        _s(s.t5_max_token_length, "t5_max_token_length", 512)
        _s(s.t5_tokenizer_path, "t5_tokenizer_path",  "")
        _s(s.max_grad_norm,     "max_grad_norm",      1.0)
        _s(s.sample_enabled,    "sample_enabled",     False)
        _s(s.sample_every_n_epochs, "sample_every_n_epochs", "1")
        _s(s.sample_prompts,    "sample_prompts",     "")
        _s(s.sample_prompt,     "sample_prompt",      "")
        _s(s.sample_negative_prompt, "sample_negative_prompt", "")
        _s(s.sample_b_enabled,          "sample_b_enabled",          False)
        _s(s.sample_b_prompt,           "sample_b_prompt",           "")
        _s(s.sample_b_negative_prompt,  "sample_b_negative_prompt",  "")
        _s(s.sample_width,      "sample_width",       512)
        _s(s.sample_height,     "sample_height",      512)
        _s(s.sample_steps,      "sample_steps",       30)
        _s(s.sample_scale,      "sample_scale",       7.5)
        _s(s.sample_flow_shift, "sample_flow_shift",  3.0)
        # 階層学習
        _s(s.layer_train_enabled, "layer_train_enabled", False)
        _s(s.layer_display_mode,  "layer_display_mode",  "Matrix")
        layer_scales = data.get("layer_parameter_vars", {})
        for k, v in layer_scales.items():
            if k in s.layer_parameter_vars:
                try:
                    s.layer_parameter_vars[k].set(float(v))
                except (ValueError, tk.TclError):
                    pass
        # Validation / EarlyStopping
        _s(s.validation_split,          "validation_split",          "0.0")
        _s(s.early_stopping,            "early_stopping",            False)
        _s(s.early_stopping_mode,       "early_stopping_mode",       "epoch")
        _s(s.early_stopping_patience,   "early_stopping_patience",   3)
        _s(s.early_stopping_threshold,  "early_stopping_threshold",  0.01)

    # ── Save ──────────────────────────────────────────────────────
    def _save() -> None:
        pname = name_var.get().strip()
        if not pname:
            messagebox.showerror("Preset", "プリセット名を入力してください。")
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in pname)
        dest = _preset_dir() / f"{safe}.json"
        try:
            dest.write_text(
                json.dumps(_collect(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            messagebox.showerror("Preset", f"保存失敗: {exc}")
            return
        _refresh_list()
        s.log_fn(f"[Preset] 保存: {dest.name}")

    # ── Load ──────────────────────────────────────────────────────
    def _load() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("Preset", "プリセットを選択してください。")
            return
        src = _preset_dir() / f"{lb.get(sel[0])}.json"
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Preset", f"読み込み失敗: {exc}")
            return
        # 階層学習 enabled/mode を _apply より先にセットしてスライダーを生成する。
        # layer_parameter_vars が空のままだとスケール値が反映されないため先行処理が必要。
        _pre_enabled = bool(data.get("layer_train_enabled", False))
        _pre_mode    = data.get("layer_display_mode", "Matrix")
        if _pre_mode not in LAYER_TRAIN_MODES:
            _pre_mode = "Matrix"
        s.layer_train_enabled.set(_pre_enabled)
        s.layer_display_mode.set(_pre_mode)
        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls(s, s.layer_canvas, s.layer_inner)
        # 階層学習 enabled/mode を _apply より先にセットしてスライダーを生成する。
        # layer_parameter_vars が空のままだとスケール値が反映されないため先行処理が必要。
        _pre_enabled = bool(data.get("layer_train_enabled", False))
        _pre_mode    = data.get("layer_display_mode", "Matrix")
        if _pre_mode not in LAYER_TRAIN_MODES:
            _pre_mode = "Matrix"
        s.layer_train_enabled.set(_pre_enabled)
        s.layer_display_mode.set(_pre_mode)
        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls(s, s.layer_canvas, s.layer_inner)
        # 階層学習 enabled/mode を _apply より先にセットしてスライダーを生成する。
        # layer_parameter_vars が空のままだとスケール値が反映されないため先行処理が必要。
        _pre_enabled = bool(data.get("layer_train_enabled", False))
        _pre_mode    = data.get("layer_display_mode", "Matrix")
        if _pre_mode not in LAYER_TRAIN_MODES:
            _pre_mode = "Matrix"
        s.layer_train_enabled.set(_pre_enabled)
        s.layer_display_mode.set(_pre_mode)
        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls(s, s.layer_canvas, s.layer_inner)
        _apply(data)
        s.log_fn(f"[Preset] 読み込み: {src.name}")

    # ── Delete ────────────────────────────────────────────────────
    def _delete() -> None:
        sel = lb.curselection()
        if not sel:
            return
        pname = lb.get(sel[0])
        if not messagebox.askyesno("Preset", f"{pname} を削除しますか？"):
            return
        ((_preset_dir()) / f"{pname}.json").unlink(missing_ok=True)
        _refresh_list()
        s.log_fn(f"[Preset] 削除: {pname}.json")

    # ── Export ────────────────────────────────────────────────────
    def _export() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("Preset", "エクスポートするプリセットを選択してください。")
            return
        pname = lb.get(sel[0])
        src = _preset_dir() / f"{pname}.json"
        dest = filedialog.asksaveasfilename(
            initialdir=str(_preset_dir()),
            initialfile=f"{pname}.json",
            filetypes=(("JSON", "*.json"),),
        )
        if dest:
            import shutil
            shutil.copy2(src, dest)
            s.log_fn(f"[Preset] エクスポート: {dest}")

    # ── Import ────────────────────────────────────────────────────
    def _import() -> None:
        src = filedialog.askopenfilename(
            initialdir=str(_preset_dir()),
            filetypes=(("JSON", "*.json"),),
        )
        if not src:
            return
        import shutil
        pname = Path(src).stem
        dest = _preset_dir() / f"{pname}.json"
        shutil.copy2(src, dest)
        _refresh_list()
        s.log_fn(f"[Preset] インポート: {Path(src).name}")

    # ── ボタン配置 ────────────────────────────────────────────────
    for text, cmd in [
        ("保存",             _save),
        ("読み込み",         _load),
        ("削除",             _delete),
        ("エクスポート",     _export),
        ("インポート",       _import),
        ("一覧更新",         _refresh_list),
    ]:
        ttk.Button(btn_row, text=text, command=cmd).pack(side=tk.LEFT, padx=4, pady=4)

    _refresh_list()


# ──────────────────────────────────────────────────────────────────────────────
# 実行パネル（主要な中タブ下部）
# ──────────────────────────────────────────────────────────────────────────────
def _build_run_panel(parent: ttk.Frame, s: _TrainState) -> None:
    frm = ttk.LabelFrame(parent, text="実行")
    frm.pack(fill=tk.X, pady=(6, 0))

    # コマンドプレビュー
    cmd_frame = ttk.Frame(frm)
    cmd_frame.pack(fill=tk.X, padx=4, pady=(4, 0))
    ttk.Label(cmd_frame, text="コマンドプレビュー:").pack(side=tk.LEFT)
    ttk.Button(cmd_frame, text="更新", command=lambda: _refresh_cmd(s, cmd_text)).pack(side=tk.LEFT, padx=4)

    cmd_text = tk.Text(frm, height=3, wrap=tk.WORD, font=("TkFixedFont", 8))
    cmd_text.pack(fill=tk.X, padx=4, pady=2)

    # ステータス / ボタン
    btn_row = ttk.Frame(frm)
    btn_row.pack(fill=tk.X, padx=4, pady=(2, 4))
    ttk.Label(btn_row, textvariable=s.status_var, foreground="#334155").pack(side=tk.LEFT, padx=4)

    ttk.Button(
        btn_row, text="■ Stop",
        command=lambda: _stop_training(s),
    ).pack(side=tk.RIGHT, padx=(4, 0))

    ttk.Button(
        btn_row, text="▶ 学習開始",
        style="Run.TButton",
        command=lambda: _start_training(s, cmd_text),
    ).pack(side=tk.RIGHT, padx=4)

    # ログ出力
    log_frame = ttk.LabelFrame(parent, text="学習ログ")
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 8))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)

    s._log_widgets.append(log_text)

    # 定期ドレイン
    def _drain():
        while True:
            try:
                msg = s._log_queue.get_nowait()
                for widget in s._log_widgets:
                    widget.insert(tk.END, msg + "\n")
                    widget.see(tk.END)
            except queue.Empty:
                break
        parent.after(200, _drain)

    if not s._log_drain_started:
        s._log_drain_started = True
        parent.after(200, _drain)


# ──────────────────────────────────────────────────────────────────────────────
# 階層学習: GUI スケール → anima_block_lr_weight 変換
# ──────────────────────────────────────────────────────────────────────────────

# Anima の blocks.N → Block カテゴリ対応
# blocks.0-8  = Input, blocks.9-18 = Middle, blocks.19-27 = Output
_BLOCK_CAT: list[str] = (
    ["Input"] * 9 + ["Middle"] * 10 + ["Output"] * 9
)


def _layer_scales_to_block_weights(
    mode: str,
    scales: dict[str, float],
) -> list[float]:
    """GUI のスケール値を 28 要素の anima_block_lr_weight リストに変換する。

    Transformer モード
        GUI グループ名が 'blocks.N' → そのまま N 番目の値として使用。

    Matrix モード
        GUI グループ名は 'Block_Component'（例: 'Input_Attention'）。
        blocks.N のスケール = そのブロックの Block カテゴリに属する
        全 Component スケールの平均値。
        （lora.py は 1 ブロック = 1 スケールしか受け取れないため）

    Component モード
        GUI グループ名は 'MLP' / 'Norm' 等のコンポーネント名。
        blocks.N のスケール = 全コンポーネントスケールの平均値
        （コンポーネント情報はブロック単位に集約できないため）。
    """
    weights: list[float] = []

    if mode == "Transformer":
        # 'blocks.N' → インデックス N へ直接マッピング
        for i in range(28):
            weights.append(float(scales.get(f"blocks.{i}", 1.0)))

    elif mode == "Matrix":
        # Block ごとに所属する全 Component の平均を取る
        for i in range(28):
            cat = _BLOCK_CAT[i]  # 'Input' / 'Middle' / 'Output'
            comp_vals = [
                scales.get(f"{cat}_{c}", 1.0)
                for c in MATRIX_COMPONENTS
            ]
            weights.append(sum(comp_vals) / len(comp_vals))

    else:  # Component モード
        # コンポーネントはブロック単位に分解できないため全体の平均
        # COMPONENT_GROUPS は Attention を含む（#1で追加済み）
        all_vals = [scales.get(c, 1.0) for c in COMPONENT_GROUPS]
        avg = sum(all_vals) / len(all_vals)
        weights = [avg] * 28

    return weights


# ──────────────────────────────────────────────────────────────────────────────
# コマンド生成
# ──────────────────────────────────────────────────────────────────────────────
def _build_command(s: _TrainState) -> list[str]:
    """GUIの設定値から accelerate launch コマンドリストを生成する。"""
    sd_scripts_root = s.paths.root / "sd-scripts"
    train_script = sd_scripts_root / "anima_train_network.py"

    # accelerate launch はspawnするためPYTHONPATHが引き継がれない場合がある。
    # ラッパー経由でsys.pathを確実に追加する。
    wrapper = sd_scripts_root / "_gui_train_wrapper.py"
    wrapper.write_text(
        "import sys, os\n"
        f"sys.path.insert(0, r'{sd_scripts_root}')\n"
        f"os.chdir(r'{sd_scripts_root}')\n"
        f"with open(r'{train_script}', encoding='utf-8') as _f:\n"
        "    _code = compile(_f.read(), _f.name, 'exec')\n"
        f"exec(_code, {{'__name__': '__main__', '__file__': r'{train_script}'}})\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", s.mixed_precision.get(),
        "--num_cpu_threads_per_process", "1",
        str(wrapper),
        "--pretrained_model_name_or_path", s.model_path.get(),
        "--vae",           s.vae_path.get(),
        "--qwen3",         s.qwen3_path.get(),
        "--train_data_dir", s.train_data_dir.get(),
        "--output_dir",    s.output_dir.get(),
        "--output_name",   s.output_name.get(),
        "--resolution",    s.resolution.get(),
        "--train_batch_size", str(s.batch_size.get()),
        "--caption_extension", s.caption_extension.get(),
        "--keep_tokens",   str(s.keep_tokens.get()),
        "--network_module", s.network_module.get(),
        "--network_dim",   str(s.network_dim.get()),
        "--network_alpha", str(s.network_alpha.get()),
        "--learning_rate", s.lr.get(),
        "--lr_scheduler",  s.lr_scheduler.get(),
        "--lr_warmup_steps", str(s.lr_warmup_steps.get()),
        "--optimizer_type", s.optimizer.get(),
        "--max_train_epochs", str(s.max_train_epochs.get()),
        "--save_every_n_epochs", str(s.save_every_n_epochs.get()),
        "--mixed_precision", s.mixed_precision.get(),
        "--save_precision", s.precision.get(),
        "--gradient_accumulation_steps", str(s.grad_accum.get()),
        "--max_grad_norm", str(s.max_grad_norm.get()),
        "--timestep_sampling", s.timestep_sampling.get(),
        "--discrete_flow_shift", str(s.discrete_flow_shift.get()),
        "--sigmoid_scale", str(s.sigmoid_scale.get()),
        "--weighting_scheme", s.weighting_scheme.get(),
        "--attn_mode",     s.attn_mode.get(),
        "--qwen3_max_token_length", str(s.qwen3_max_token_length.get()),
        "--t5_max_token_length", str(s.t5_max_token_length.get()),
        "--max_data_loader_n_workers", "0",
    ]

    # オプション引数
    if s.seed.get():
        cmd += ["--seed", s.seed.get()]
    if s.llm_adapter_path.get():
        cmd += ["--llm_adapter_path", s.llm_adapter_path.get()]
    if s.network_weights.get():
        cmd += ["--network_weights", s.network_weights.get()]
    if s.optimizer_args.get():
        cmd += ["--optimizer_args"] + s.optimizer_args.get().split()
    if s.t5_tokenizer_path.get():
        cmd += ["--t5_tokenizer_path", s.t5_tokenizer_path.get()]

    # bool フラグ
    bool_flags = [
        (s.cache_latents,               "--cache_latents"),
        (s.cache_latents_to_disk,       "--cache_latents_to_disk"),
        (s.cache_te_outputs,            "--cache_text_encoder_outputs"),
        (s.shuffle_caption,             "--shuffle_caption"),
        (s.flip_aug,                    "--flip_aug"),
        (s.enable_bucket,               "--enable_bucket"),
        (s.bucket_no_upscale,           "--bucket_no_upscale"),
        (s.network_train_unet_only,     "--network_train_unet_only"),
        (s.gradient_checkpointing,      "--gradient_checkpointing"),
        (s.split_attn,                  "--split_attn"),
        (s.unsloth_offload_checkpointing, "--unsloth_offload_checkpointing"),
        (s.cpu_offload_checkpointing,   "--cpu_offload_checkpointing"),
        (s.vae_disable_cache,           "--vae_disable_cache"),
    ]
    for var, flag in bool_flags:
        if var.get():
            cmd.append(flag)

    # バケット
    if s.enable_bucket.get():
        cmd += [
            "--min_bucket_reso", str(s.min_bucket_reso.get()),
            "--max_bucket_reso", str(s.max_bucket_reso.get()),
        ]

    # vae_chunk_size
    vcs = s.vae_chunk_size.get().strip()
    if vcs:
        cmd += ["--vae_chunk_size", vcs]

    # blocks_to_swap
    bts = s.blocks_to_swap.get()
    if bts > 0:
        cmd += ["--blocks_to_swap", str(bts)]

    # 階層学習
    if s.layer_train_enabled.get():
        _mode   = s.layer_display_mode.get()
        _scales = {k: v.get() for k, v in s.layer_parameter_vars.items()}

        if _mode == "Matrix":
            # Matrix モード: (Block_Component) 別スケール辞書を JSON で渡す
            # lora.py の anima_matrix_scales → (block_idx, comp) 単位でグループ化され精度損失なし
            _scales_json = json.dumps(
                {k: round(v, 4) for k, v in _scales.items()},
                separators=(",", ":"),
            )
            cmd += ["--network_args", f"anima_matrix_scales={_scales_json}"]
            s.log_fn(f"[LayerLR] mode=Matrix  anima_matrix_scales={_scales_json}")

        else:
            # Transformer / Component モード: 28 要素の block_lr_weight を渡す
            _weights = _layer_scales_to_block_weights(_mode, _scales)
            weight_str = ",".join(f"{w:.4f}" for w in _weights)
            cmd += ["--network_args", f"anima_block_lr_weight={weight_str}"]
            s.log_fn(
                f"[LayerLR] mode={_mode}  "
                f"anima_block_lr_weight={weight_str}"
            )

    # Validation split
    _vs = s.validation_split.get().strip()
    try:
        if float(_vs) > 0.0:
            cmd += ["--validation_split", _vs]
    except ValueError:
        pass

    # EarlyStopping
    if s.early_stopping.get():
        cmd.append("--early_stopping")
        cmd += ["--early_stopping_mode",      s.early_stopping_mode.get()]
        cmd += ["--early_stopping_patience",  str(s.early_stopping_patience.get())]
        cmd += ["--early_stopping_threshold", str(s.early_stopping_threshold.get())]

    # サンプル
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        sample_prompt_file = _write_sample_prompt_file(s)
        cmd += ["--sample_every_n_epochs", s.sample_every_n_epochs.get().strip() or "1"]
        cmd += ["--sample_prompts", str(sample_prompt_file)]
        cmd += ["--sample_save_dir", str(_sample_dir(s))]

    return cmd


def _refresh_cmd(s: _TrainState, text_widget: tk.Text) -> None:
    try:
        cmd = _build_command(s)
        preview = " ".join(cmd)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, preview)
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, f"[エラー] {e}")
        text_widget.config(state=tk.DISABLED)


# ──────────────────────────────────────────────────────────────────────────────
# 学習実行 / 停止
# ──────────────────────────────────────────────────────────────────────────────
def _validate(s: _TrainState) -> str | None:
    """必須フィールドの簡易バリデーション。エラーメッセージを返す（正常時None）。"""
    if not s.model_path.get():
        return "DiTモデルパスが未設定です。"
    if not s.vae_path.get():
        return "VAEパスが未設定です。"
    if not s.qwen3_path.get():
        return "Qwen3テキストエンコーダパスが未設定です。"
    if not s.train_data_dir.get():
        return "学習データフォルダが未設定です。"
    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプルAが有効ですが、promptが未設定です。"
    if s.sample_b_enabled.get() and not s.sample_b_prompt.get().strip():
        return "サンプルBが有効ですが、promptが未設定です。"
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        try:
            if int(s.sample_every_n_epochs.get().strip() or "1") <= 0:
                return "サンプル出力のepoch間隔は1以上を指定してください。"
        except ValueError:
            return "サンプル出力のepoch間隔は整数で指定してください。"
    return None


def _start_training(s: _TrainState, cmd_text: tk.Text) -> None:
    if s._proc is not None and s._proc.poll() is None:
        messagebox.showwarning("学習中", "すでに学習が実行中です。")
        return

    err = _validate(s)
    if err:
        messagebox.showerror("入力エラー", err)
        return

    try:
        cmd = _build_command(s)
    except Exception as exc:
        messagebox.showerror("コマンド生成エラー", str(exc))
        return

    _refresh_cmd(s, cmd_text)

    sd_scripts_root = s.paths.root / "sd-scripts"
    s.status_var.set("学習中...")
    s.log_fn("[LoRA Train] 学習開始")
    s._log_queue.put(f"[CMD] {' '.join(cmd)}")

    def _worker():
        import os
        log_dir = s.paths.root / "log" / "lora_train"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts}.txt"
        s.log_fn(f"[LoRA Train] ログ: {log_path}")
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(sd_scripts_root) + (os.pathsep + existing if existing else "")
            import os, signal
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            proc = subprocess.Popen(
                cmd,
                cwd=str(sd_scripts_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=CREATE_NEW_PROCESS_GROUP,
            )
            s._proc = proc
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"CMD: {' '.join(cmd)}\n\n")
                for line in proc.stdout:
                    line = line.rstrip()
                    # ANSIエスケープシーケンスを除去（tqdmのカーソル制御コード等）
                    import re as _re
                    line = _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)
                    s._log_queue.put(line)
                    s._monitor_queue.put(line)
                    s._monitor_layer_queue.put(line)
                    lf.write(line + "\n")
                    lf.flush()
            proc.wait()
            rc = proc.returncode
            msg = f"[LoRA Train] 完了 (return code: {rc})"
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set("完了" if rc == 0 else f"エラー (code={rc})")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception as exc:
            msg = f"[LoRA Train] 起動エラー: {exc}"
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set("起動失敗")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        finally:
            s._proc = None

    threading.Thread(target=_worker, daemon=True).start()


def _stop_training(s: _TrainState) -> None:
    if s._proc is None or s._proc.poll() is not None:
        s.log_fn("[LoRA Train] 停止対象のプロセスがありません。")
        return
    import os, signal
    try:
        os.kill(s._proc.pid, signal.CTRL_BREAK_EVENT)
    except Exception:
        s._proc.terminate()
    s.status_var.set("停止要求済み")
    s.log_fn("[LoRA Train] 停止要求を送信しました。（プロセスグループ全体に送信）")
