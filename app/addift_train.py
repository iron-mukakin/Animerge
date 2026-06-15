"""app/addift_train.py — Anima ADDifT学習タブ（最小実装）

build_addift_train_tab(parent, paths, log_fn, get_model_choices) を呼び出すことで
gui.py の ADDifT学習タブに組み込まれる。

leco_train.py との主な差異:
  - プロンプト設定タブ → データセットタブ（画像A/画像B + 共通caption + 差分マスク）
  - max_train_steps / save_every_n_steps → train_iterations / save_every_n_steps
  - 呼び出しスクリプト: anima_train_addift.py
  - 階層学習タブ / モニター系タブ / サンプル生成タブは最小実装（仮設置、後続フェーズで実装）
  - プリセット保存先: <project_root>/preset/addift_train/*.json
"""
from __future__ import annotations

import datetime
import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

try:
    from .i18n import gettext, load_language
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _app_dir = _Path(__file__).resolve().parent
    if str(_app_dir) not in _sys.path:
        _sys.path.insert(0, str(_app_dir))
    from i18n import gettext, load_language  # type: ignore[no-redef]

# ──────────────────────────────────────────────────────────────────────────────
# 定数（lora_train.py / leco_train.py と共通）
# ──────────────────────────────────────────────────────────────────────────────
OPTIMIZERS = [
    "AdamW", "AdamW8bit", "Adafactor", "DAdaptAdam",
    "DAdaptAdaGrad", "DAdaptSGD", "Lion", "Prodigy",
]
LR_SCHEDULERS = [
    "constant", "constant_with_warmup", "cosine",
    "cosine_with_restarts", "linear", "polynomial",
]
PRECISIONS    = ["bf16", "fp16", "fp32"]
ATTN_MODES    = ["torch", "xformers", "flash", "sdpa"]
LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"]

# 階層学習定数（lora_train.py / leco_train.py と共通。フェーズ2で使用予定）
LAYER_TRAIN_MODES  = ("Matrix", "Transformer", "Component")
MATRIX_BLOCKS      = ("Input", "Middle", "Output")
MATRIX_COMPONENTS  = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS   = ("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")
LAYER_COLUMNS      = 3

# blocks.0-8=Input, blocks.9-18=Middle, blocks.19-27=Output (leco_train.py と共通)
_BLOCK_CAT: list[str] = ["Input"] * 9 + ["Middle"] * 10 + ["Output"] * 9


# ──────────────────────────────────────────────────────────────────────────────
# メイン構築関数
# ──────────────────────────────────────────────────────────────────────────────
def build_addift_train_tab(
    parent: ttk.Frame,
    paths,
    log_fn: Callable[[str], None],
    get_model_choices: Callable[[], list[str]],
) -> "_AddifTTrainState":
    """ADDifT学習タブの全UIを parent に構築する。"""

    state = _AddifTTrainState(paths, log_fn, get_model_choices)

    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.BOTH, expand=True)

    tab_model          = ttk.Frame(nb, padding=8)
    tab_dataset        = ttk.Frame(nb, padding=8)
    tab_network        = ttk.Frame(nb, padding=8)
    tab_train          = ttk.Frame(nb, padding=8)
    tab_adv            = ttk.Frame(nb, padding=8)
    tab_layer          = ttk.Frame(nb, padding=8)
    tab_monitor        = ttk.Frame(nb, padding=8)
    tab_monitor_layer  = ttk.Frame(nb, padding=8)
    tab_sample         = ttk.Frame(nb, padding=8)
    tab_preset         = ttk.Frame(nb, padding=8)

    nb.add(tab_model,          text=gettext("lora_tab_model"))
    nb.add(tab_dataset,        text=gettext("addift_tab_dataset"))
    nb.add(tab_network,        text=gettext("lora_tab_network"))
    nb.add(tab_train,          text=gettext("lora_tab_train"))
    nb.add(tab_adv,            text=gettext("lora_tab_adv"))
    nb.add(tab_layer,          text=gettext("lora_tab_layer"))
    nb.add(tab_monitor,        text=gettext("lora_tab_monitor"))
    nb.add(tab_monitor_layer,  text=gettext("lora_tab_monitor_layer"))
    nb.add(tab_sample,         text=gettext("lora_tab_sample"))
    nb.add(tab_preset,         text=gettext("lora_tab_preset"))

    _build_model_tab(tab_model,       state)
    _build_dataset_tab(tab_dataset,   state)
    _build_network_tab(tab_network,   state)
    _build_train_tab(tab_train,       state)
    _build_adv_tab(tab_adv,           state)
    _build_layer_train_tab(tab_layer, state)
    _build_monitor_tab(tab_monitor, state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_sample_tab(tab_sample, state)
    _build_addift_preset_tab(tab_preset, state)

    for tab in (tab_model, tab_dataset, tab_network, tab_train):
        _build_run_panel(tab, state)

    return state


# ──────────────────────────────────────────────────────────────────────────────
# 状態オブジェクト
# ──────────────────────────────────────────────────────────────────────────────
class _AddifTTrainState:
    def __init__(self, paths, log_fn, get_model_choices):
        self.paths            = paths
        self.log_fn           = log_fn
        self.get_model_choices = get_model_choices
        self._proc: subprocess.Popen | None = None
        self._log_queue:   queue.Queue[str] = queue.Queue()
        self._stop_event   = threading.Event()

        # ── モデル ──────────────────────────────────────────────
        self.model_path       = tk.StringVar()
        self.vae_path         = tk.StringVar()
        self.qwen3_path       = tk.StringVar()
        self.llm_adapter_path = tk.StringVar()
        self.output_dir       = tk.StringVar(value=str(paths.lora))
        self.output_name      = tk.StringVar(value="addift_output")
        self.precision        = tk.StringVar(value="bf16")

        # ── データセット（ADDifT固有: 画像ペア） ───────────────────
        self.image_a_path     = tk.StringVar()
        self.image_b_path     = tk.StringVar()
        self.caption          = tk.StringVar(value="")
        self.diff_use_diff_mask = tk.BooleanVar(value=False)
        self.diff_mask_path  = tk.StringVar()

        # ── ネットワーク ─────────────────────────────────────────
        self.network_dim      = tk.IntVar(value=8)
        self.network_alpha    = tk.DoubleVar(value=4.0)
        self.network_module   = tk.StringVar(value="networks.lora")
        self.network_weights  = tk.StringVar()

        # ── 学習設定（共通） ─────────────────────────────────────
        self.lr               = tk.StringVar(value="5e-5")
        self.lr_scheduler     = tk.StringVar(value="constant")
        self.lr_warmup_steps  = tk.IntVar(value=0)
        self.optimizer        = tk.StringVar(value="AdamW")
        self.optimizer_args   = tk.StringVar(value="")
        self.train_iterations = tk.IntVar(value=50)
        self.save_every_n_steps = tk.IntVar(value=50)
        self.seed             = tk.StringVar(value="42")
        self.gradient_checkpointing = tk.BooleanVar(value=True)
        self.grad_accum       = tk.IntVar(value=1)
        self.mixed_precision  = tk.StringVar(value="bf16")
        self.max_grad_norm    = tk.DoubleVar(value=1.0)

        # ── ADDifT固有パラメータ ──────────────────────────────────
        self.train_min_timesteps = tk.IntVar(value=200)
        self.train_max_timesteps = tk.IntVar(value=400)
        self.train_fixed_timesteps_in_batch = tk.BooleanVar(value=False)
        self.diff_alt_ratio   = tk.DoubleVar(value=1.0)
        self.network_strength = tk.DoubleVar(value=5.0)
        self.train_loss_function = tk.StringVar(value="MSE")
        self.train_snr_gamma  = tk.DoubleVar(value=0.0)

        # ── 詳細（Anima固有） ────────────────────────────────────
        self.attn_mode        = tk.StringVar(value="torch")
        self.split_attn       = tk.BooleanVar(value=False)
        self.blocks_to_swap   = tk.IntVar(value=0)
        self.unsloth_offload_checkpointing = tk.BooleanVar(value=False)
        self.cpu_offload_checkpointing     = tk.BooleanVar(value=False)
        self.vae_chunk_size   = tk.StringVar(value="")
        self.vae_disable_cache = tk.BooleanVar(value=False)
        self.qwen3_max_token_length = tk.IntVar(value=512)
        self.t5_max_token_length    = tk.IntVar(value=512)
        self.t5_tokenizer_path      = tk.StringVar(value="")

        # ── 階層学習 ─────────────────────────────────────────────
        self.layer_train_enabled   = tk.BooleanVar(value=False)
        self.layer_display_mode    = tk.StringVar(value="Matrix")
        self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
        self.layer_canvas: "tk.Canvas | None" = None
        self.layer_inner:  "ttk.Frame | None" = None
        self._layer_status_var: "tk.StringVar | None" = None

        # ── モニターキュー（フェーズ2で使用予定） ───────────────────
        self._monitor_queue:       queue.Queue[str] = queue.Queue()
        self._monitor_layer_queue: queue.Queue[str] = queue.Queue()

        # ── サンプル生成（フェーズ2で使用予定 / 仮placeholder） ─────
        self.sample_every_n_steps   = tk.StringVar(value="50")
        self.sample_width           = tk.IntVar(value=512)
        self.sample_height          = tk.IntVar(value=512)
        self.sample_steps           = tk.IntVar(value=20)
        self.sample_scale           = tk.DoubleVar(value=7.5)
        self.sample_flow_shift      = tk.DoubleVar(value=3.0)
        self.sample_keep_vae        = tk.BooleanVar(value=False)
        self.sample_enabled         = tk.BooleanVar(value=False)
        self.sample_prompt          = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        self.sample_b_enabled          = tk.BooleanVar(value=False)
        self.sample_b_prompt           = tk.StringVar(value="")
        self.sample_b_negative_prompt  = tk.StringVar(value="")

        # ── EarlyStopping（モニターグラフ: Train Loss連続上昇監視） ──
        self.es_enabled   = tk.BooleanVar(value=False)
        self.es_patience  = tk.IntVar(value=5)   # 連続上昇で警告/停止する step 数

        # ステータス
        self.status_var       = tk.StringVar(value=gettext("status_waiting"))
        self._log_widgets: list[tk.Text] = []
        self._log_drain_started = False


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────────────────────
def _browse_file(var: tk.StringVar, title: str | None = None, filetypes=None):
    title = title or gettext("lora_browse_file_title")
    ft = filetypes or [("All", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title: str | None = None):
    title = title or gettext("lora_browse_dir_title")
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _entry_browse_row(parent, row: int, label: str, var: tk.StringVar,
                       is_dir=False, filetypes=None):
    ttk.Label(parent, text=label, width=26, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(parent, textvariable=var).grid(
        row=row, column=1, sticky=tk.EW, padx=(0, 2), pady=3)
    cmd = (lambda v=var: _browse_dir(v)) if is_dir \
        else (lambda v=var, ft=filetypes: _browse_file(v, filetypes=ft))
    ttk.Button(parent, text="Browse", width=7, command=cmd).grid(
        row=row, column=2, padx=(0, 4), pady=3)


def _image_preview_row(parent, row: int, label: str, var: tk.StringVar,
                        preview_label: ttk.Label) -> None:
    """画像パス入力行 + サムネイルプレビュー更新。"""
    ttk.Label(parent, text=label, width=26, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    entry = ttk.Entry(parent, textvariable=var)
    entry.grid(row=row, column=1, sticky=tk.EW, padx=(0, 2), pady=3)

    def _refresh_preview(*_args) -> None:
        path_str = var.get().strip()
        if not path_str or not Path(path_str).is_file():
            preview_label.configure(image="", text="")
            preview_label._photo_ref = None  # type: ignore[attr-defined]
            return
        try:
            from PIL import Image as _Im, ImageTk as _ITk
            with _Im.open(path_str) as im:
                im.thumbnail((160, 160))
                photo = _ITk.PhotoImage(im.copy())
            preview_label._photo_ref = photo  # type: ignore[attr-defined]
            preview_label.configure(image=photo, text="")
        except Exception as exc:
            preview_label.configure(image="", text=f"[ERR] {exc}")
            preview_label._photo_ref = None  # type: ignore[attr-defined]

    def _browse_image() -> None:
        path = filedialog.askopenfilename(
            title=gettext("lora_browse_file_title"),
            filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All", "*.*")],
        )
        if path:
            var.set(path)
            _refresh_preview()

    ttk.Button(parent, text="Browse", width=7, command=_browse_image).grid(
        row=row, column=2, padx=(0, 4), pady=3)
    entry.bind("<FocusOut>", _refresh_preview)
    _refresh_preview()


# ──────────────────────────────────────────────────────────────────────────────
# タブ1: モデル
# ──────────────────────────────────────────────────────────────────────────────
def _build_model_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text=gettext("lora_model_paths"))
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)

    _entry_browse_row(lf, 0, gettext("lora_dit_label"), s.model_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    _entry_browse_row(lf, 1, gettext("lora_vae_label"), s.vae_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
    _entry_browse_row(lf, 2, gettext("lora_qwen3_label"), s.qwen3_path,
                      filetypes=[("safetensors", "*.safetensors"), ("dir", "*")])
    _entry_browse_row(lf, 3, gettext("lora_llm_adapter_label"), s.llm_adapter_path,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])

    lf2 = ttk.LabelFrame(parent, text=gettext("lora_output_settings"))
    lf2.pack(fill=tk.X)
    lf2.columnconfigure(1, weight=1)

    _entry_browse_row(lf2, 0, gettext("lora_output_folder"), s.output_dir, is_dir=True)
    ttk.Label(lf2, text=gettext("lora_output_filename"), width=26, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.output_name).grid(
        row=1, column=1, sticky=tk.EW, padx=(0, 4), pady=3)
    ttk.Label(lf2, text=gettext("lora_save_precision"), width=26, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.precision, values=PRECISIONS,
                 state="readonly", width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ2: データセット（ADDifT固有 - 画像A/B + 共通caption）
# ──────────────────────────────────────────────────────────────────────────────
def _build_dataset_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text=gettext("addift_dataset_label"))
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)

    preview_a = ttk.Label(lf, anchor=tk.CENTER, relief=tk.GROOVE)
    preview_a.grid(row=0, column=3, rowspan=2, padx=(8, 4), pady=3, sticky=tk.N)
    preview_b = ttk.Label(lf, anchor=tk.CENTER, relief=tk.GROOVE)
    preview_b.grid(row=2, column=3, rowspan=2, padx=(8, 4), pady=3, sticky=tk.N)

    ttk.Label(lf, text=gettext("addift_image_a_label"), foreground="#1D4ED8").grid(
        row=0, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(3, 0))
    _image_preview_row(lf, 1, gettext("addift_image_a_path"), s.image_a_path, preview_a)

    ttk.Label(lf, text=gettext("addift_image_b_label"), foreground="#1D4ED8").grid(
        row=2, column=0, columnspan=3, sticky=tk.W, padx=(4, 2), pady=(6, 0))
    _image_preview_row(lf, 3, gettext("addift_image_b_path"), s.image_b_path, preview_b)

    ttk.Label(lf, text=gettext("addift_caption_label"), width=26, anchor=tk.W).grid(
        row=4, column=0, sticky=tk.W, padx=(4, 2), pady=(8, 3))
    ttk.Entry(lf, textvariable=s.caption).grid(
        row=4, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=(8, 3))
    ttk.Label(lf, text=gettext("addift_caption_note"), foreground="#64748B").grid(
        row=5, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(0, 4))

    lf2 = ttk.LabelFrame(parent, text=gettext("addift_diff_mask_label"))
    lf2.pack(fill=tk.X)
    lf2.columnconfigure(1, weight=1)

    ttk.Checkbutton(lf2, text=gettext("addift_diff_use_diff_mask"),
                    variable=s.diff_use_diff_mask).grid(
        row=0, column=0, columnspan=3, sticky=tk.W, padx=8, pady=3)
    _entry_browse_row(lf2, 1, gettext("addift_diff_mask_path"), s.diff_mask_path,
                       filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All", "*.*")])
    ttk.Label(lf2, text=gettext("addift_diff_mask_note"), foreground="#64748B").grid(
        row=2, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(0, 4))


# ──────────────────────────────────────────────────────────────────────────────
# タブ3: ネットワーク
# ──────────────────────────────────────────────────────────────────────────────
def _build_network_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text=gettext("lora_network_settings"))
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

    ttk.Label(lf, text=gettext("addift_network_unet_only_note"), foreground="#64748B").grid(
        row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    _entry_browse_row(lf, 4, gettext("lora_network_weights_label"), s.network_weights,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])


# ──────────────────────────────────────────────────────────────────────────────
# タブ4: 学習設定
# ──────────────────────────────────────────────────────────────────────────────
def _build_train_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text=gettext("lora_train_params"))
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

    # row 3: train_iterations / save_every_n_steps
    ttk.Label(lf, text="train_iterations", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=999999, textvariable=s.train_iterations, width=10).grid(
        row=3, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="save_every_n_steps", width=20, anchor=tk.W).grid(
        row=3, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=999999, textvariable=s.save_every_n_steps, width=10).grid(
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

    # ── ADDifT固有パラメータ ─────────────────────────────────────────
    lf2 = ttk.LabelFrame(parent, text=gettext("addift_params_label"))
    lf2.pack(fill=tk.X, pady=(0, 8))
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    ttk.Label(lf2, text="train_min_timesteps", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf2, from_=0, to=999, textvariable=s.train_min_timesteps, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_max_timesteps", width=22, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=1000, textvariable=s.train_max_timesteps, width=8).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf2, text="network_strength", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.network_strength, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="diff_alt_ratio", width=22, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.diff_alt_ratio, width=10).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(lf2, text="train_loss_function", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.train_loss_function, values=LOSS_FUNCTIONS,
                 state="readonly", width=12).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="train_snr_gamma", width=22, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.train_snr_gamma, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Checkbutton(lf2, text="train_fixed_timesteps_in_batch",
                    variable=s.train_fixed_timesteps_in_batch).grid(
        row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    lf3 = ttk.LabelFrame(parent, text=gettext("lora_memory_opt"))
    lf3.pack(fill=tk.X)
    ttk.Checkbutton(lf3, text="gradient_checkpointing",
                    variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ5: 詳細（Anima固有）
# ──────────────────────────────────────────────────────────────────────────────
def _build_adv_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text=gettext("lora_adv_settings"))
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    # row 0: attn_mode / blocks_to_swap
    ttk.Label(lf, text="attn_mode", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.attn_mode, values=ATTN_MODES,
                 state="readonly", width=14).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text=gettext("lora_blocks_to_swap"), width=20, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=100, textvariable=s.blocks_to_swap, width=8).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: vae_chunk_size / qwen3_max_token_length
    ttk.Label(lf, text=gettext("lora_vae_chunk_size"), width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf, textvariable=s.vae_chunk_size, width=10).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="qwen3_max_token_length", width=20, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.qwen3_max_token_length, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: t5_max_token_length
    ttk.Label(lf, text="t5_max_token_length", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.t5_max_token_length, width=8).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)

    # row 3: t5_tokenizer_path
    _entry_browse_row(lf, 3, "t5_tokenizer_path", s.t5_tokenizer_path, is_dir=True)

    lf2 = ttk.LabelFrame(parent, text=gettext("leco_offload_label"))
    lf2.pack(fill=tk.X)
    ttk.Checkbutton(lf2, text="split_attn",
                    variable=s.split_attn).grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="unsloth_offload_checkpointing",
                    variable=s.unsloth_offload_checkpointing).grid(
        row=0, column=1, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="cpu_offload_checkpointing",
                    variable=s.cpu_offload_checkpointing).grid(
        row=0, column=2, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="vae_disable_cache",
                    variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)

    lf3 = ttk.LabelFrame(parent, text=gettext("leco_es_label"))
    lf3.pack(fill=tk.X, pady=(8, 0))
    ttk.Checkbutton(lf3, text=gettext("leco_es_enable"),
                    variable=s.es_enabled).grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
    ttk.Label(lf3, text=gettext("leco_es_watch_steps"), anchor=tk.W).grid(
        row=0, column=1, sticky=tk.W, padx=(12, 2), pady=4)
    ttk.Spinbox(lf3, from_=2, to=500, textvariable=s.es_patience, width=6).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 8), pady=4)
    ttk.Label(lf3, text=gettext("leco_es_note"),
              foreground="#64748B").grid(row=0, column=3, sticky=tk.W, padx=(0, 8), pady=4)


# ──────────────────────────────────────────────────────────────────────────────
# 階層学習 / モニターグラフ / モニター階層 / サンプル生成
# ──────────────────────────────────────────────────────────────────────────────
def _build_placeholder_tab(parent: ttk.Frame, message_key: str) -> None:
    ttk.Label(
        parent,
        text=gettext(message_key),
        foreground="#64748B",
        justify=tk.LEFT,
        wraplength=560,
    ).pack(padx=16, pady=16, anchor=tk.NW)


def _build_layer_train_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    """階層別学習率スケールを設定するタブ (leco_train.py 移植)。"""
    hdr = ttk.Frame(parent)
    hdr.pack(fill=tk.X, pady=(0, 4))

    ttk.Checkbutton(
        hdr, text=gettext("lora_layer_train_enable"),
        variable=s.layer_train_enabled,
        command=lambda: _refresh_layer_controls_addift(s, ctrl_canvas, ctrl_inner),
    ).pack(side=tk.LEFT, padx=(0, 12))

    ttk.Label(hdr, text=gettext("lora_layer_mode_label")).pack(side=tk.LEFT)
    mode_cb = ttk.Combobox(
        hdr, textvariable=s.layer_display_mode,
        values=list(LAYER_TRAIN_MODES), state="readonly", width=14,
    )
    mode_cb.pack(side=tk.LEFT, padx=(2, 12))
    mode_cb.bind(
        "<<ComboboxSelected>>",
        lambda _e: _refresh_layer_controls_addift(s, ctrl_canvas, ctrl_inner),
    )

    ttk.Button(
        hdr, text=gettext("lora_layer_preset_load"),
        command=lambda: _load_layer_preset_addift(s, ctrl_canvas, ctrl_inner),
    ).pack(side=tk.LEFT)

    canvas_frame = ttk.LabelFrame(
        parent, text=gettext("lora_layer_scale_label")
    )
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
    ctrl_inner.bind(
        "<Configure>",
        lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
    )

    s._layer_status_var = tk.StringVar(value="(無効)")
    ttk.Label(parent, textvariable=s._layer_status_var, foreground="#334155").pack(
        anchor=tk.W, padx=4, pady=(2, 0)
    )

    _refresh_layer_controls_addift(s, ctrl_canvas, ctrl_inner)


def _layer_group_names_addift(mode: str) -> list[str]:
    """表示モードに対応するグループ名リストを返す (leco_train.py と同仕様)。"""
    if mode == "Matrix":
        return [f"{b}_{c}" for b in MATRIX_BLOCKS for c in MATRIX_COMPONENTS]
    if mode == "Component":
        return list(COMPONENT_GROUPS)
    return [f"blocks.{i}" for i in range(28)]


def _refresh_layer_controls_addift(
    s: _AddifTTrainState,
    canvas: tk.Canvas,
    inner: ttk.Frame,
) -> None:
    """階層学習スライダー群を再描画する (leco_train.py と同仕様)。"""
    for child in inner.winfo_children():
        child.destroy()

    if not s.layer_train_enabled.get():
        ttk.Label(inner, text=gettext("lora_layer_disabled_msg")).grid(
            row=0, column=0, padx=8, pady=8, sticky=tk.W
        )
        if s._layer_status_var is not None:
            s._layer_status_var.set(gettext("lora_layer_status_disabled"))
        canvas.configure(scrollregion=canvas.bbox("all"))
        return

    mode = s.layer_display_mode.get()
    groups = _layer_group_names_addift(mode)
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
        scale.bind("<ButtonRelease-1>", lambda e, v=var: _snap_scale_addift(v))

        entry = ttk.Entry(inner, textvariable=var, width=6)
        entry.grid(row=grid_row, column=base_col + 2, sticky=tk.W, padx=(2, 12), pady=3)
        entry.bind("<FocusOut>", lambda e, v=var: _clamp_var_addift(v))

    for c in range(cols):
        inner.columnconfigure(c * 3 + 1, weight=1)

    canvas.configure(scrollregion=canvas.bbox("all"))

    if s._layer_status_var is None:
        return

    if mode == "Component":
        warn_row = (len(groups) + cols - 1) // cols
        tk.Label(
            inner,
            text=gettext("lora_layer_component_warn"),
            foreground="red",
            justify=tk.LEFT,
            wraplength=600,
        ).grid(row=warn_row, column=0, columnspan=cols * 3, sticky=tk.W, padx=8, pady=(6, 2))
        canvas.configure(scrollregion=canvas.bbox("all"))
        s._layer_status_var.set(
            gettext("lora_layer_status_warn", mode=mode, count=len(groups))
        )
    else:
        s._layer_status_var.set(gettext("lora_layer_status", mode=mode, count=len(groups)))


def _snap_scale_addift(var: tk.DoubleVar) -> None:
    """スライダー値を0.05単位にスナップする。"""
    v = var.get()
    var.set(round(round(v / 0.05) * 0.05, 4))


def _clamp_var_addift(var: tk.DoubleVar) -> None:
    """スケール値を[0.0, 1.0]に収める。"""
    try:
        var.set(max(0.0, min(1.0, float(var.get()))))
    except tk.TclError:
        var.set(1.0)


def _convert_preset_scales_addift(
    scales: dict[str, float],
    preset_mode: str,
    target_mode: str,
) -> dict[str, float]:
    """プリセットのスケール辞書をpreset_modeからtarget_modeへ変換する (leco_train.py 移植)。"""
    if preset_mode == target_mode:
        return dict(scales)

    def _attn_by_block_cat(scales: dict) -> dict[str, float]:
        cat_vals: dict[str, list[float]] = {b: [] for b in MATRIX_BLOCKS}
        for k, v in scales.items():
            m = re.match(r"blocks\.(\d+)_Attention$", k)
            if m:
                n = int(m.group(1))
                if 0 <= n < 28:
                    cat_vals[_BLOCK_CAT[n]].append(float(v))
        return {cat: (sum(vals) / len(vals) if vals else 1.0) for cat, vals in cat_vals.items()}

    if preset_mode == "Component" and target_mode == "Matrix":
        attn_by_cat = _attn_by_block_cat(scales)
        result = {}
        for b in MATRIX_BLOCKS:
            result[f"{b}_Attention"] = attn_by_cat[b]
            for c in ("MLP", "Norm", "ResNet", "Timestep", "Other"):
                result[f"{b}_{c}"] = float(scales.get(c, 1.0))
        return result

    if preset_mode == "Component" and target_mode == "Transformer":
        attn_by_cat = _attn_by_block_cat(scales)
        result = {}
        for i in range(28):
            cat = _BLOCK_CAT[i]
            comp_vals = [attn_by_cat[cat]]
            comp_vals += [float(scales.get(c, 1.0)) for c in ("MLP", "Norm", "ResNet", "Timestep", "Other")]
            result[f"blocks.{i}"] = sum(comp_vals) / len(comp_vals)
        return result

    if preset_mode == "Matrix" and target_mode == "Component":
        result = {}
        attn_vals = [float(scales.get(f"{b}_Attention", 1.0)) for b in MATRIX_BLOCKS]
        result["Attention"] = sum(attn_vals) / len(attn_vals)
        for c in ("MLP", "Norm", "ResNet", "Timestep", "Other"):
            vals = [float(scales.get(f"{b}_{c}", 1.0)) for b in MATRIX_BLOCKS]
            result[c] = sum(vals) / len(vals)
        return result

    if preset_mode == "Matrix" and target_mode == "Transformer":
        result = {}
        for i in range(28):
            cat = _BLOCK_CAT[i]
            comp_vals = [float(scales.get(f"{cat}_{c}", 1.0)) for c in MATRIX_COMPONENTS]
            result[f"blocks.{i}"] = sum(comp_vals) / len(comp_vals)
        return result

    if preset_mode == "Transformer" and target_mode == "Matrix":
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

    if preset_mode == "Transformer" and target_mode == "Component":
        avg = sum(float(scales.get(f"blocks.{i}", 1.0)) for i in range(28)) / 28
        return {c: avg for c in COMPONENT_GROUPS}

    return dict(scales)


def _load_layer_preset_addift(s: _AddifTTrainState, canvas: tk.Canvas, inner: ttk.Frame) -> None:
    """preset/addift_train/*.json から layer_parameter_vars を読み込む。"""
    preset_dir = s.paths.root / "preset" / "addift_train"
    preset_dir.mkdir(parents=True, exist_ok=True)
    path = filedialog.askopenfilename(
        title=gettext("lora_layer_preset_select_title"),
        initialdir=str(preset_dir),
        filetypes=[("JSON", "*.json"), ("All", "*.*")],
    )
    if not path:
        return
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        messagebox.showerror(gettext("lora_layer_preset_error"), str(exc))
        return

    new_mode = data.get("layer_display_mode", "Matrix")
    if new_mode in LAYER_TRAIN_MODES:
        s.layer_display_mode.set(new_mode)

    _refresh_layer_controls_addift(s, canvas, inner)

    scales = data.get("layer_parameter_vars", {})
    converted = _convert_preset_scales_addift(
        scales, preset_mode=new_mode, target_mode=s.layer_display_mode.get()
    )
    for k, v in converted.items():
        if k in s.layer_parameter_vars:
            try:
                s.layer_parameter_vars[k].set(float(v))
            except (ValueError, tk.TclError):
                pass

    s.log_fn(
        gettext("lora_layer_preset_log",
                name=Path(path).name,
                preset=new_mode,
                gui=s.layer_display_mode.get())
    )


def _layer_scales_to_block_weights_addift(mode: str, scales: dict[str, float]) -> list[float]:
    """layer_parameter_vars を anima_block_lr_weight 用の28要素リストへ変換する。"""
    weights: list[float] = []
    if mode == "Transformer":
        for i in range(28):
            weights.append(float(scales.get(f"blocks.{i}", 1.0)))
    elif mode == "Matrix":
        for i in range(28):
            cat = _BLOCK_CAT[i]
            comp_vals = [scales.get(f"{cat}_{c}", 1.0) for c in MATRIX_COMPONENTS]
            weights.append(sum(comp_vals) / len(comp_vals))
    else:
        all_vals = [scales.get(c, 1.0) for c in COMPONENT_GROUPS]
        avg = sum(all_vals) / len(all_vals)
        weights = [avg] * 28
    return weights


def _build_monitor_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    """モニターグラフタブ: AddifTMonitorGraph + 学習ログウィジェットを組み込む。"""
    import importlib.util as _ilu
    import logging as _log

    # row=0 グラフ領域 / row=1 ログ欄
    parent.rowconfigure(0, weight=1)
    parent.rowconfigure(1, weight=0)
    parent.columnconfigure(0, weight=1)

    graph_frame = ttk.Frame(parent)
    graph_frame.grid(row=0, column=0, sticky=tk.NSEW)

    try:
        _here = Path(__file__).resolve().parent
        _spec = _ilu.spec_from_file_location(
            "monitor_graph_addift", _here / "monitor_graph_addift.py"
        )
        if _spec is None or _spec.loader is None:
            raise ImportError(
                f"spec_from_file_location failed: {_here / 'monitor_graph_addift.py'}"
            )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.AddifTMonitorGraph(graph_frame, s)
    except Exception as _e:
        _log.getLogger(__name__).error(
            "[_build_monitor_tab] monitor_graph_addift ロード失敗: %s", _e
        )
        ttk.Label(
            graph_frame,
            text=gettext("lora_monitor_graph_init_error", error=_e),
            foreground="#EF4444",
            justify=tk.LEFT,
        ).pack(padx=16, pady=16, anchor=tk.NW)

    log_frame = ttk.LabelFrame(parent, text=gettext("lora_train_log"))
    log_frame.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
    log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 8))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)
    s._log_widgets.append(log_text)


def _build_monitor_layer_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    """モニター階層タブ: MonitorLayerGraph を組み込む（leco_train.py と同仕様）。"""
    import importlib.util as _ilu
    import logging as _log

    try:
        _here = Path(__file__).resolve().parent
        _spec = _ilu.spec_from_file_location(
            "monitor_layer", _here / "monitor_layer.py"
        )
        if _spec is None or _spec.loader is None:
            raise ImportError(
                f"spec_from_file_location failed: {_here / 'monitor_layer.py'}"
            )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.MonitorLayerGraph(parent, s, _group_names_for_mode_addift)
    except Exception as _e:
        _log.getLogger(__name__).error(
            "[_build_monitor_layer_tab] monitor_layer ロード失敗: %s", _e
        )
        ttk.Label(
            parent,
            text=gettext("lora_monitor_layer_init_error", error=_e),
            foreground="#EF4444",
            justify=tk.LEFT,
        ).pack(padx=16, pady=16, anchor=tk.NW)


def _group_names_for_mode_addift(mode: str) -> list[str]:
    """表示モードに対応するグループ名リストを返す (leco_train.py と同仕様)。

    実体は _layer_group_names_addift と同一だが、MonitorLayerGraph の
    コールバック引数名(leco_train.py の _group_names_for_mode_leco)に
    合わせた別名として提供する。
    """
    return _layer_group_names_addift(mode)


# ──────────────────────────────────────────────────────────────────────────────
# サンプル生成ヘルパー（ADDifT用）
# 保存先・プロンプトファイルは leco_train.py と共有
#   <project_root>/log/sample_gen/_sample_prompt.txt
# ──────────────────────────────────────────────────────────────────────────────

def _addift_sample_dir(s: _AddifTTrainState) -> Path:
    return s.paths.root / "log" / "sample_gen"


def _addift_sample_prompt_path(s: _AddifTTrainState) -> Path:
    return _addift_sample_dir(s) / "_sample_prompt.txt"


ADDIFT_SAMPLE_FIXED_SEED = 42


def _addift_build_prompt_line(
    prompt: str, neg: str, s: _AddifTTrainState, seed: int = ADDIFT_SAMPLE_FIXED_SEED
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


def _addift_write_sample_prompt_file(s: _AddifTTrainState) -> Path:
    lines = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_addift_build_prompt_line(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            s,
            seed=ADDIFT_SAMPLE_FIXED_SEED,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_addift_build_prompt_line(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            s,
            seed=ADDIFT_SAMPLE_FIXED_SEED + 1,
        ))
    path = _addift_sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _build_sample_tab(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    """ADDifT サンプル生成タブ: A/B ギャラリー表示を組み込む。

    出力ファイル名: <output_name>_<step:06d>_<00|01>_<ts><seed_suffix>.png
    （anima_sample_gen.sample_images_from_prompts が生成する形式）。
    プレビューはこの命名規則（先頭 <output_name>_ + 末尾 _00_/_01_）で判定し、
    ADDifT学習の出力のみを本タブ内に表示する（lora/leco学習のサンプルとは
    output_name が異なるため混在しない）。
    """
    import importlib.util as _ilu
    import logging as _log

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    # ── 共通設定 ─────────────────────────────────────────────────
    common = ttk.LabelFrame(parent, text=gettext("lora_sample_common"))
    common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    common.columnconfigure(1, weight=1)
    common.columnconfigure(3, weight=1)

    ttk.Label(common, text=gettext("lora_sample_step_interval"), width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=99999, textvariable=s.sample_every_n_steps, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text=gettext("lora_sample_fixed_seed", seed=ADDIFT_SAMPLE_FIXED_SEED),
              foreground="#64748B").grid(
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

    ttk.Checkbutton(
        common, text=gettext("lora_sample_keep_vae"),
        variable=s.sample_keep_vae,
    ).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    # ── lora_train.py から _build_sample_ab_panel / _extract_sample_epoch を動的import
    _bsap = _esep = None
    try:
        try:
            from .lora_train import _build_sample_ab_panel as _bsap, _extract_sample_epoch as _esep
        except ImportError:
            import sys as _sys
            _here = Path(__file__).resolve().parent
            if str(_here) not in _sys.path:
                _sys.path.insert(0, str(_here))
            from lora_train import _build_sample_ab_panel as _bsap, _extract_sample_epoch as _esep  # type: ignore[no-redef]
    except Exception as _e:
        _log.getLogger(__name__).error(
            "[_build_sample_tab] lora_train ロード失敗: %s", _e
        )

    ab_nb = ttk.Notebook(parent)
    ab_nb.grid(row=1, column=0, sticky=tk.NSEW)
    tab_a = ttk.Frame(ab_nb, padding=4)
    tab_b = ttk.Frame(ab_nb, padding=4)
    ab_nb.add(tab_a, text=gettext("lora_sample_a"))
    ab_nb.add(tab_b, text=gettext("lora_sample_b"))

    sample_dir = _addift_sample_dir(s)

    if _bsap is not None:
        # output_name は実行時に変更され得るため、ビルド時点での値で
        # glob パターンを構築する（更新するには「表示更新」ボタンでタブ再構築は
        # 不要 — ビルド時点の output_name を使う前提。変更後は設定保存→再起動、
        # もしくはタブ切替で _build_sample_tab が再実行されるため反映される）。
        _out = s.output_name.get().strip() or "addift_output"
        pat_a = f"{_out}_*_00_*.png"
        pat_b = f"{_out}_*_01_*.png"

        _bsap(
            tab_a, s,
            enabled_var=s.sample_enabled,
            prompt_var=s.sample_prompt,
            neg_var=s.sample_negative_prompt,
            sample_dir=sample_dir,
            glob_pattern=pat_a,
            label="A",
            is_leco=True,  # ADDifTもstep基準のため「step」ラベルを使用
        )
        _bsap(
            tab_b, s,
            enabled_var=s.sample_b_enabled,
            prompt_var=s.sample_b_prompt,
            neg_var=s.sample_b_negative_prompt,
            sample_dir=sample_dir,
            glob_pattern=pat_b,
            label="B",
            is_leco=True,
        )
    else:
        for _tab in (tab_a, tab_b):
            ttk.Label(
                _tab,
                text=gettext("lora_sample_load_error", error=""),
                foreground="#EF4444",
                justify=tk.LEFT,
            ).pack(padx=16, pady=16, anchor=tk.NW)


# ──────────────────────────────────────────────────────────────────────────────
# 実行パネル
# ──────────────────────────────────────────────────────────────────────────────
def _build_run_panel(parent: ttk.Frame, s: _AddifTTrainState) -> None:
    frm = ttk.LabelFrame(parent, text=gettext("lora_run_label"))
    frm.pack(fill=tk.X, pady=(6, 0))

    cmd_frame = ttk.Frame(frm)
    cmd_frame.pack(fill=tk.X, padx=4, pady=(4, 0))
    ttk.Label(cmd_frame, text=gettext("lora_cmd_preview")).pack(side=tk.LEFT)
    ttk.Button(cmd_frame, text=gettext("lora_cmd_refresh"),
               command=lambda: _refresh_cmd(s, cmd_text)).pack(side=tk.LEFT, padx=4)

    cmd_text = tk.Text(frm, height=3, wrap=tk.WORD, font=("TkFixedFont", 8))
    cmd_text.pack(fill=tk.X, padx=4, pady=2)

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill=tk.X, padx=4, pady=(2, 4))
    ttk.Label(btn_row, textvariable=s.status_var, foreground="#334155").pack(
        side=tk.LEFT, padx=4)
    ttk.Button(btn_row, text=gettext("lora_stop_btn"),
               command=lambda: _stop_training(s)).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(btn_row, text=gettext("lora_start_btn"), style="Run.TButton",
               command=lambda: _start_training(s, cmd_text)).pack(side=tk.RIGHT, padx=4)

    log_frame = ttk.LabelFrame(parent, text=gettext("lora_train_log"))
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 8))
    log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(fill=tk.BOTH, expand=True)
    s._log_widgets.append(log_text)

    def _drain():
        while True:
            try:
                msg = s._log_queue.get_nowait()
                for _w in list(s._log_widgets):
                    try:
                        _w.insert(tk.END, msg + "\n")
                        _w.see(tk.END)
                    except Exception:
                        pass
            except queue.Empty:
                break
        parent.after(200, _drain)

    if not s._log_drain_started:
        s._log_drain_started = True
        parent.after(200, _drain)


# ──────────────────────────────────────────────────────────────────────────────
# コマンド生成
# ──────────────────────────────────────────────────────────────────────────────
def _build_command(s: _AddifTTrainState) -> list[str]:
    """GUIの設定値から accelerate launch コマンドリストを生成する。"""
    sd_scripts_root = s.paths.root / "sd-scripts"
    train_script    = sd_scripts_root / "anima_train_addift.py"

    wrapper = sd_scripts_root / "_gui_addift_wrapper.py"
    wrapper.write_text(
        "import sys, os\n"
        "from pathlib import Path\n"
        "_root = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(_root))\n"
        "os.chdir(str(_root))\n"
        "_train = _root / 'anima_train_addift.py'\n"
        "with open(_train, encoding='utf-8') as _f:\n"
        "    _code = compile(_f.read(), str(_train), 'exec')\n"
        "exec(_code, {'__name__': '__main__', '__file__': str(_train)})\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision",              s.mixed_precision.get(),
        "--num_cpu_threads_per_process",  "1",
        str(wrapper),
        "--pretrained_model_name_or_path", s.model_path.get(),
        "--vae",                           s.vae_path.get(),
        "--qwen3",                         s.qwen3_path.get(),
        "--image_a",                       s.image_a_path.get(),
        "--image_b",                       s.image_b_path.get(),
        "--caption",                       s.caption.get(),
        "--output_dir",                    s.output_dir.get(),
        "--output_name",                   s.output_name.get(),
        "--network_module",                s.network_module.get(),
        "--network_dim",                   str(s.network_dim.get()),
        "--network_alpha",                 str(s.network_alpha.get()),
        "--learning_rate",                 s.lr.get(),
        "--lr_scheduler",                  s.lr_scheduler.get(),
        "--lr_warmup_steps",               str(s.lr_warmup_steps.get()),
        "--optimizer_type",                s.optimizer.get(),
        "--train_iterations",              str(s.train_iterations.get()),
        "--save_every_n_steps",            str(s.save_every_n_steps.get()),
        "--mixed_precision",               s.mixed_precision.get(),
        "--save_precision",                s.precision.get(),
        "--gradient_accumulation_steps",   str(s.grad_accum.get()),
        "--max_grad_norm",                 str(s.max_grad_norm.get()),
        "--train_min_timesteps",           str(s.train_min_timesteps.get()),
        "--train_max_timesteps",           str(s.train_max_timesteps.get()),
        "--diff_alt_ratio",                str(s.diff_alt_ratio.get()),
        "--network_strength",              str(s.network_strength.get()),
        "--train_loss_function",           s.train_loss_function.get(),
        "--train_snr_gamma",               str(s.train_snr_gamma.get()),
        "--attn_mode",                      s.attn_mode.get(),
        "--qwen3_max_token_length",        str(s.qwen3_max_token_length.get()),
        "--t5_max_token_length",           str(s.t5_max_token_length.get()),
        "--network_train_unet_only",       # ADDifT は常にDiTのみ
    ]

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

    vcs = s.vae_chunk_size.get().strip()
    if vcs:
        cmd += ["--vae_chunk_size", vcs]

    bts = s.blocks_to_swap.get()
    if bts > 0:
        cmd += ["--blocks_to_swap", str(bts)]

    bool_flags = [
        (s.gradient_checkpointing,             "--gradient_checkpointing"),
        (s.split_attn,                          "--split_attn"),
        (s.unsloth_offload_checkpointing,       "--unsloth_offload_checkpointing"),
        (s.cpu_offload_checkpointing,           "--cpu_offload_checkpointing"),
        (s.vae_disable_cache,                   "--vae_disable_cache"),
        (s.train_fixed_timesteps_in_batch,      "--train_fixed_timesteps_in_batch"),
    ]
    for var, flag in bool_flags:
        if var.get():
            cmd.append(flag)

    if s.diff_use_diff_mask.get():
        cmd.append("--diff_use_diff_mask")
        if s.diff_mask_path.get():
            cmd += ["--diff_mask_path", s.diff_mask_path.get()]

    # サンプル生成
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        _spf = _addift_write_sample_prompt_file(s)
        cmd += ["--sample_every_n_steps", s.sample_every_n_steps.get().strip() or "50"]
        cmd += ["--sample_prompts",  str(_spf)]
        cmd += ["--sample_save_dir", str(_addift_sample_dir(s))]
        if s.sample_keep_vae.get():
            cmd.append("--sample_keep_vae")

    # 階層学習（layer_parameter_vars が populate されている場合のみ付与）
    if s.layer_train_enabled.get() and s.layer_parameter_vars:
        _mode   = s.layer_display_mode.get()
        _scales = {k: v.get() for k, v in s.layer_parameter_vars.items()}
        if _mode == "Matrix":
            _scales_json = json.dumps(
                {k: round(v, 4) for k, v in _scales.items()},
                separators=(",", ":"),
            )
            cmd += ["--network_args", f"anima_matrix_scales={_scales_json}"]
        else:
            _weights = _layer_scales_to_block_weights_addift(_mode, _scales)
            _weight_str = ",".join(f"{w:.4f}" for w in _weights)
            cmd += ["--network_args", f"anima_block_lr_weight={_weight_str}"]

    return cmd


def _refresh_cmd(s: _AddifTTrainState, text_widget: tk.Text) -> None:
    try:
        cmd = _build_command(s)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, " ".join(cmd))
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, gettext("lora_cmd_error_prefix", error=e))
        text_widget.config(state=tk.DISABLED)


# ──────────────────────────────────────────────────────────────────────────────
# バリデーション / 実行 / 停止
# ──────────────────────────────────────────────────────────────────────────────
def _validate(s: _AddifTTrainState) -> str | None:
    if not s.model_path.get():
        return gettext("lora_validate_no_model")
    if not s.vae_path.get():
        return gettext("lora_validate_no_vae")
    if not s.qwen3_path.get():
        return gettext("lora_validate_no_qwen3")
    if not s.image_a_path.get():
        return gettext("addift_validate_no_image_a")
    if not Path(s.image_a_path.get()).is_file():
        return gettext("addift_validate_image_a_missing", path=s.image_a_path.get())
    if not s.image_b_path.get():
        return gettext("addift_validate_no_image_b")
    if not Path(s.image_b_path.get()).is_file():
        return gettext("addift_validate_image_b_missing", path=s.image_b_path.get())
    if s.diff_use_diff_mask.get():
        if not s.diff_mask_path.get():
            return gettext("addift_validate_no_diff_mask")
        if not Path(s.diff_mask_path.get()).is_file():
            return gettext("addift_validate_diff_mask_missing", path=s.diff_mask_path.get())
    if s.train_iterations.get() < 1:
        return gettext("addift_validate_iterations")
    if s.save_every_n_steps.get() < 1:
        return gettext("leco_validate_save_steps")
    if s.train_min_timesteps.get() >= s.train_max_timesteps.get():
        return gettext("addift_validate_timesteps_range")
    return None


def _start_training(s: _AddifTTrainState, cmd_text: tk.Text) -> None:
    if s._proc is not None and s._proc.poll() is None:
        messagebox.showwarning(gettext("lora_running_warn_title"), gettext("lora_running_warn"))
        return

    err = _validate(s)
    if err:
        messagebox.showerror(gettext("lora_input_error_title"), err)
        return

    try:
        cmd = _build_command(s)
    except Exception as exc:
        messagebox.showerror(gettext("lora_cmd_error_title"), str(exc))
        return

    _refresh_cmd(s, cmd_text)

    sd_scripts_root = s.paths.root / "sd-scripts"
    s.status_var.set(gettext("lora_status_running"))
    s.log_fn(gettext("addift_start_log"))
    s._log_queue.put(f"[CMD] {' '.join(cmd)}")

    def _worker():
        import os, re as _re
        log_dir = s.paths.root / "log" / "addift_train"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts}.txt"
        s.log_fn(gettext("addift_log_path", path=log_path))
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(sd_scripts_root) + (os.pathsep + existing if existing else "")
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            proc = subprocess.Popen(
                cmd,
                cwd=str(sd_scripts_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=CREATE_NEW_PROCESS_GROUP,
            )
            s._proc = proc
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"CMD: {' '.join(cmd)}\n\n")
                for raw_line in proc.stdout:
                    raw_line = _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw_line)
                    segs = raw_line.split("\r")
                    line = next(
                        (seg.rstrip() for seg in reversed(segs) if seg.rstrip()),
                        "",
                    )
                    if not line:
                        continue
                    s._log_queue.put(line)
                    s._monitor_queue.put(line)
                    s._monitor_layer_queue.put(line)
                    lf.write(line + "\n")
                    lf.flush()
            proc.wait()
            rc = proc.returncode
            msg = gettext("addift_done", rc=rc)
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set(gettext("lora_status_done") if rc == 0 else gettext("lora_status_error", rc=rc))
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception as exc:
            msg = gettext("addift_start_error", error=exc)
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set(gettext("lora_status_start_failed"))
        finally:
            s._proc = None

    threading.Thread(target=_worker, daemon=True).start()


def _stop_training(s: _AddifTTrainState) -> None:
    if s._proc is None or s._proc.poll() is not None:
        s.log_fn(gettext("addift_stop_no_proc"))
        return
    import os, signal
    try:
        os.kill(s._proc.pid, signal.CTRL_BREAK_EVENT)
    except Exception:
        s._proc.terminate()
    s.log_fn(gettext("addift_stop_sent"))
    s.status_var.set(gettext("status_stop_requested"))


# ══════════════════════════════════════════════════════════════════════════════
# プリセットタブ
# ══════════════════════════════════════════════════════════════════════════════
def _build_addift_preset_tab(parent: ttk.Frame, s: "_AddifTTrainState") -> None:
    """ADDifT学習設定を JSON で保存・復元するプリセットタブ。
    保存先: <project_root>/preset/addift_train/*.json
    """
    PRESET_DIR_REL = ("preset", "addift_train")

    def _preset_dir() -> Path:
        d = s.paths.root / PRESET_DIR_REL[0] / PRESET_DIR_REL[1]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── リストボックス ──────────────────────────────────────────
    list_frame = ttk.LabelFrame(parent, text=gettext("lora_preset_saved"))
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    lb = tk.Listbox(list_frame, height=10, selectmode=tk.SINGLE)
    lb_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=lb.yview)
    lb.configure(yscrollcommand=lb_scroll.set)
    lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── 名前入力 ────────────────────────────────────────────────
    name_row = ttk.Frame(parent)
    name_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(name_row, text=gettext("lora_preset_name")).pack(side=tk.LEFT)
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
        return {
            "version": 1,
            # モデル
            "model_path":        s.model_path.get(),
            "vae_path":          s.vae_path.get(),
            "qwen3_path":        s.qwen3_path.get(),
            "llm_adapter_path":  s.llm_adapter_path.get(),
            "output_dir":        s.output_dir.get(),
            "output_name":       s.output_name.get(),
            "precision":         s.precision.get(),
            # データセット
            "image_a_path":      s.image_a_path.get(),
            "image_b_path":      s.image_b_path.get(),
            "caption":           s.caption.get(),
            "diff_use_diff_mask": bool(s.diff_use_diff_mask.get()),
            "diff_mask_path":    s.diff_mask_path.get(),
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
            "train_iterations":  int(s.train_iterations.get()),
            "save_every_n_steps": int(s.save_every_n_steps.get()),
            "seed":              s.seed.get(),
            "gradient_checkpointing": bool(s.gradient_checkpointing.get()),
            "grad_accum":        int(s.grad_accum.get()),
            "mixed_precision":   s.mixed_precision.get(),
            "max_grad_norm":     float(s.max_grad_norm.get()),
            # ADDifT固有
            "train_min_timesteps": int(s.train_min_timesteps.get()),
            "train_max_timesteps": int(s.train_max_timesteps.get()),
            "train_fixed_timesteps_in_batch": bool(s.train_fixed_timesteps_in_batch.get()),
            "diff_alt_ratio":     float(s.diff_alt_ratio.get()),
            "network_strength":   float(s.network_strength.get()),
            "train_loss_function": s.train_loss_function.get(),
            "train_snr_gamma":    float(s.train_snr_gamma.get()),
            # 詳細
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
            # 階層学習
            "layer_train_enabled": bool(s.layer_train_enabled.get()),
            "layer_display_mode":  s.layer_display_mode.get(),
            "layer_parameter_vars": {
                k: round(float(v.get()), 4)
                for k, v in s.layer_parameter_vars.items()
            },
            # EarlyStopping（モニターグラフ）
            "es_enabled":  bool(s.es_enabled.get()),
            "es_patience": int(s.es_patience.get()),
            # サンプル生成
            "sample_every_n_steps":   s.sample_every_n_steps.get(),
            "sample_width":           int(s.sample_width.get()),
            "sample_height":          int(s.sample_height.get()),
            "sample_steps":           int(s.sample_steps.get()),
            "sample_scale":           float(s.sample_scale.get()),
            "sample_flow_shift":      float(s.sample_flow_shift.get()),
            "sample_keep_vae":        bool(s.sample_keep_vae.get()),
            "sample_enabled":         bool(s.sample_enabled.get()),
            "sample_prompt":          s.sample_prompt.get(),
            "sample_negative_prompt": s.sample_negative_prompt.get(),
            "sample_b_enabled":          bool(s.sample_b_enabled.get()),
            "sample_b_prompt":           s.sample_b_prompt.get(),
            "sample_b_negative_prompt":  s.sample_b_negative_prompt.get(),
        }

    def _apply(data: dict) -> None:
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
        _s(s.output_name,       "output_name",        "addift_output")
        _s(s.precision,         "precision",          "bf16")
        _s(s.image_a_path,      "image_a_path",       "")
        _s(s.image_b_path,      "image_b_path",       "")
        _s(s.caption,           "caption",            "")
        _s(s.diff_use_diff_mask, "diff_use_diff_mask", False)
        _s(s.diff_mask_path,    "diff_mask_path",     "")
        _s(s.network_dim,       "network_dim",        8)
        _s(s.network_alpha,     "network_alpha",      4.0)
        _s(s.network_module,    "network_module",     "networks.lora")
        _s(s.network_weights,   "network_weights",    "")
        _s(s.lr,                "lr",                 "5e-5")
        _s(s.lr_scheduler,      "lr_scheduler",       "constant")
        _s(s.lr_warmup_steps,   "lr_warmup_steps",    0)
        _s(s.optimizer,         "optimizer",          "AdamW")
        _s(s.optimizer_args,    "optimizer_args",     "")
        _s(s.train_iterations,  "train_iterations",   50)
        _s(s.save_every_n_steps, "save_every_n_steps", 50)
        _s(s.seed,              "seed",               "42")
        _s(s.gradient_checkpointing, "gradient_checkpointing", True)
        _s(s.grad_accum,        "grad_accum",         1)
        _s(s.mixed_precision,   "mixed_precision",    "bf16")
        _s(s.max_grad_norm,     "max_grad_norm",      1.0)
        _s(s.train_min_timesteps, "train_min_timesteps", 200)
        _s(s.train_max_timesteps, "train_max_timesteps", 400)
        _s(s.train_fixed_timesteps_in_batch, "train_fixed_timesteps_in_batch", False)
        _s(s.diff_alt_ratio,     "diff_alt_ratio",     1.0)
        _s(s.network_strength,   "network_strength",  5.0)
        _s(s.train_loss_function, "train_loss_function", "MSE")
        _s(s.train_snr_gamma,    "train_snr_gamma",    0.0)
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
        _s(s.layer_train_enabled, "layer_train_enabled", False)
        _s(s.layer_display_mode,  "layer_display_mode",  "Matrix")
        _s(s.es_enabled,          "es_enabled",          False)
        _s(s.es_patience,         "es_patience",         5)
        _s(s.sample_every_n_steps,   "sample_every_n_steps",   "50")
        _s(s.sample_width,           "sample_width",           512)
        _s(s.sample_height,          "sample_height",          512)
        _s(s.sample_steps,           "sample_steps",           20)
        _s(s.sample_scale,           "sample_scale",           7.5)
        _s(s.sample_flow_shift,      "sample_flow_shift",      3.0)
        _s(s.sample_keep_vae,        "sample_keep_vae",        False)
        _s(s.sample_enabled,         "sample_enabled",         False)
        _s(s.sample_prompt,          "sample_prompt",          "")
        _s(s.sample_negative_prompt, "sample_negative_prompt", "")
        _s(s.sample_b_enabled,          "sample_b_enabled",          False)
        _s(s.sample_b_prompt,           "sample_b_prompt",           "")
        _s(s.sample_b_negative_prompt,  "sample_b_negative_prompt",  "")

        layer_scales = data.get("layer_parameter_vars", {})
        for k, v in layer_scales.items():
            if k in s.layer_parameter_vars:
                try:
                    s.layer_parameter_vars[k].set(float(v))
                except (ValueError, tk.TclError):
                    pass

    # ── Save ──────────────────────────────────────────────────────
    def _save() -> None:
        pname = name_var.get().strip()
        if not pname:
            messagebox.showerror("Preset", gettext("lora_preset_no_name"))
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in pname)
        dest = _preset_dir() / f"{safe}.json"
        try:
            dest.write_text(
                json.dumps(_collect(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            messagebox.showerror("Preset", gettext("lora_preset_save_failed", error=exc))
            return
        _refresh_list()
        s.log_fn(gettext("lora_preset_log_saved", name=dest.name))

    # ── Load ──────────────────────────────────────────────────────
    def _load() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("Preset", gettext("lora_preset_no_select"))
            return
        src = _preset_dir() / f"{lb.get(sel[0])}.json"
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Preset", gettext("lora_preset_load_failed", error=exc))
            return
        # 階層学習: enabled/mode を先行セットしてスライダーを生成してから _apply
        _pre_enabled = bool(data.get("layer_train_enabled", False))
        _pre_mode    = data.get("layer_display_mode", "Matrix")
        if _pre_mode not in LAYER_TRAIN_MODES:
            _pre_mode = "Matrix"
        s.layer_train_enabled.set(_pre_enabled)
        s.layer_display_mode.set(_pre_mode)
        if s.layer_canvas is not None and s.layer_inner is not None:
            _refresh_layer_controls_addift(s, s.layer_canvas, s.layer_inner)
        _apply(data)
        s.log_fn(gettext("lora_preset_log_loaded", name=src.name))

    # ── Delete ────────────────────────────────────────────────────
    def _delete() -> None:
        sel = lb.curselection()
        if not sel:
            return
        pname = lb.get(sel[0])
        if not messagebox.askyesno("Preset", gettext("lora_preset_confirm_delete", name=pname)):
            return
        (_preset_dir() / f"{pname}.json").unlink(missing_ok=True)
        _refresh_list()
        s.log_fn(gettext("lora_preset_log_deleted", name=f"{pname}.json"))

    # ── Export ────────────────────────────────────────────────────
    def _export() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror("Preset", gettext("lora_preset_no_select_export"))
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
            s.log_fn(gettext("lora_preset_log_exported", dest=dest))

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
        s.log_fn(gettext("lora_preset_log_imported", name=Path(src).name))

    # ── ボタン配置 ────────────────────────────────────────────────
    for text, cmd in [
        (gettext("lora_preset_save"),    _save),
        (gettext("lora_preset_load"),    _load),
        (gettext("lora_preset_delete"),  _delete),
        (gettext("lora_preset_export"),  _export),
        (gettext("lora_preset_import"),  _import),
        (gettext("lora_preset_refresh"), _refresh_list),
    ]:
        ttk.Button(btn_row, text=text, command=cmd).pack(side=tk.LEFT, padx=4, pady=4)

    _refresh_list()
