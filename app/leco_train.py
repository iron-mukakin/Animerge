"""app/leco_train.py — Anima LECO学習タブ

build_leco_train_tab(parent, paths, log_fn, get_model_choices) を呼び出すことで
gui.py の LECO学習タブに組み込まれる。

lora_train.py との主な差異:
  - データセットタブ → プロンプト設定タブ（TOMLファイル + インライン編集）
  - max_train_epochs / save_every_n_epochs → max_train_steps / save_every_n_steps
  - 呼び出しスクリプト: anima_train_leco.py
  - 階層学習タブ / モニター系タブは今フェーズ未実装（保留）
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

# ──────────────────────────────────────────────────────────────────────────────
# 定数（lora_train.py と共通）
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

# 階層学習定数（lora_train.py と共通）
LAYER_TRAIN_MODES  = ("Matrix", "Transformer", "Component")
MATRIX_BLOCKS      = ("Input", "Middle", "Output")
MATRIX_COMPONENTS  = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS   = ("Attention", "MLP", "Norm", "ResNet", "Timestep", "Other")
LAYER_COLUMNS      = 3

# blocks.0-8=Input, blocks.9-18=Middle, blocks.19-27=Output
_BLOCK_CAT: list[str] = ["Input"] * 9 + ["Middle"] * 10 + ["Output"] * 9

# LECO プロンプトTOMLのデフォルトテンプレート（LECO形式）
_TOML_TEMPLATE = """\
# LECO Prompt Settings
# action: "erase" で target 概念を消去 / "enhance" で強調
# multiplier: 学習時の LoRA 強度（デフォルト 1.0）
# weight: このエントリの loss 重み（デフォルト 1.0）
# guidance_scale: ターゲット構築時のスケール（大きいほど効果が強い）
# resolution: 整数 512 または [height, width] 形式

[[prompts]]
target        = "概念・スタイル名"
positive      = "target を含む肯定プロンプト"
unconditional = ""
neutral       = ""
action        = "erase"
guidance_scale = 1.0
resolution    = 512
batch_size    = 1
multiplier    = 1.0
weight        = 1.0
"""


# ──────────────────────────────────────────────────────────────────────────────
# メイン構築関数
# ──────────────────────────────────────────────────────────────────────────────
def build_leco_train_tab(
    parent: ttk.Frame,
    paths,
    log_fn: Callable[[str], None],
    get_model_choices: Callable[[], list[str]],
) -> "_LecoTrainState":
    """LECO学習タブの全UIを parent に構築する。"""

    state = _LecoTrainState(paths, log_fn, get_model_choices)

    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.BOTH, expand=True)

    tab_model          = ttk.Frame(nb, padding=8)
    tab_prompts        = ttk.Frame(nb, padding=8)
    tab_network        = ttk.Frame(nb, padding=8)
    tab_train          = ttk.Frame(nb, padding=8)
    tab_adv            = ttk.Frame(nb, padding=8)
    tab_layer          = ttk.Frame(nb, padding=8)
    tab_sample         = ttk.Frame(nb, padding=8)
    tab_monitor        = ttk.Frame(nb, padding=8)
    tab_monitor_layer  = ttk.Frame(nb, padding=8)
    tab_preset         = ttk.Frame(nb, padding=8)

    nb.add(tab_model,          text="  モデル  ")
    nb.add(tab_prompts,        text="  プロンプト設定  ")
    nb.add(tab_network,        text="  ネットワーク  ")
    nb.add(tab_train,          text="  学習設定  ")
    nb.add(tab_adv,            text="  詳細  ")
    nb.add(tab_layer,          text="  階層学習  ")
    nb.add(tab_monitor,        text="  モニターグラフ  ")
    nb.add(tab_monitor_layer,  text="  モニター階層  ")
    nb.add(tab_sample,         text="  サンプル生成  ")
    nb.add(tab_preset,         text="  プリセット  ")

    _build_model_tab(tab_model,       state)
    _build_prompts_tab(tab_prompts,   state)
    _build_network_tab(tab_network,   state)
    _build_train_tab(tab_train,       state)
    _build_adv_tab(tab_adv,           state)
    _build_layer_train_tab(tab_layer, state)
    _build_monitor_tab(tab_monitor,   state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_leco_sample_tab(tab_sample, state)
    _build_leco_preset_tab(tab_preset, state)

    for tab in (tab_model, tab_prompts, tab_network, tab_train):
        _build_run_panel(tab, state)

    return state


# ──────────────────────────────────────────────────────────────────────────────
# 状態オブジェクト
# ──────────────────────────────────────────────────────────────────────────────
class _LecoTrainState:
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
        self.output_name      = tk.StringVar(value="leco_output")
        self.precision        = tk.StringVar(value="bf16")

        # ── プロンプト（LECO固有） ───────────────────────────────
        self.prompts_file     = tk.StringVar()  # .toml ファイルパス

        # ── ネットワーク ─────────────────────────────────────────
        self.network_dim      = tk.IntVar(value=4)
        self.network_alpha    = tk.DoubleVar(value=1.0)
        self.network_module   = tk.StringVar(value="networks.lora")
        self.network_weights  = tk.StringVar()

        # ── 学習設定（LECO: ステップ管理） ──────────────────────
        self.lr               = tk.StringVar(value="1e-4")
        self.lr_scheduler     = tk.StringVar(value="constant")
        self.lr_warmup_steps  = tk.IntVar(value=0)
        self.optimizer        = tk.StringVar(value="AdamW")
        self.optimizer_args   = tk.StringVar(value="")
        self.max_train_steps  = tk.IntVar(value=500)
        self.save_every_n_steps = tk.IntVar(value=100)
        self.seed             = tk.StringVar(value="42")
        self.gradient_checkpointing = tk.BooleanVar(value=True)
        self.grad_accum       = tk.IntVar(value=1)
        self.mixed_precision  = tk.StringVar(value="bf16")
        self.max_grad_norm    = tk.DoubleVar(value=1.0)

        # ── LECO固有パラメータ ────────────────────────────────────
        self.max_denoising_steps        = tk.IntVar(value=20)
        self.leco_denoise_guidance_scale = tk.DoubleVar(value=3.0)

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
        self.discrete_flow_shift    = tk.DoubleVar(value=1.0)

        # ── 階層学習 ─────────────────────────────────────────────
        self.layer_train_enabled   = tk.BooleanVar(value=False)
        self.layer_display_mode    = tk.StringVar(value="Matrix")
        self.layer_parameter_vars: dict[str, tk.DoubleVar] = {}
        self.layer_canvas: "tk.Canvas | None" = None
        self.layer_inner:  "ttk.Frame | None" = None
        self._layer_status_var     = tk.StringVar(value="(無効)")

        # ── モニターキュー ───────────────────────────────────────
        self._monitor_queue:       queue.Queue[str] = queue.Queue()
        self._monitor_layer_queue: queue.Queue[str] = queue.Queue()

        # ── サンプル生成 共通設定 ────────────────────────────────
        self.sample_every_n_steps   = tk.StringVar(value="100")
        self.sample_width           = tk.IntVar(value=512)
        self.sample_height          = tk.IntVar(value=512)
        self.sample_steps           = tk.IntVar(value=20)
        self.sample_scale           = tk.DoubleVar(value=7.5)
        self.sample_flow_shift      = tk.DoubleVar(value=3.0)
        self.sample_keep_vae        = tk.BooleanVar(value=False)
        # サンプルA
        self.sample_enabled         = tk.BooleanVar(value=False)
        self.sample_prompt          = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        # サンプルB
        self.sample_b_enabled          = tk.BooleanVar(value=False)
        self.sample_b_prompt           = tk.StringVar(value="")
        self.sample_b_negative_prompt  = tk.StringVar(value="")

        # ステータス
        self.status_var       = tk.StringVar(value="待機中")
        self._log_widgets: list[tk.Text] = []
        self._log_drain_started = False


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────────────────────
def _browse_file(var: tk.StringVar, title="ファイル選択", filetypes=None):
    ft = filetypes or [("All", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title="フォルダ選択"):
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


# ──────────────────────────────────────────────────────────────────────────────
# タブ1: モデル
# ──────────────────────────────────────────────────────────────────────────────
def _build_model_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
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
    ttk.Label(lf2, text="出力ファイル名", width=26, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.output_name).grid(
        row=1, column=1, sticky=tk.EW, padx=(0, 4), pady=3)
    ttk.Label(lf2, text="保存精度", width=26, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf2, textvariable=s.precision, values=PRECISIONS,
                 state="readonly", width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 4), pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ2: プロンプト設定（LECO固有 - データセットタブの代替）
# ──────────────────────────────────────────────────────────────────────────────
def _build_prompts_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
    # --- TOMLファイル選択 ---
    lf_file = ttk.LabelFrame(parent, text="プロンプトTOMLファイル")
    lf_file.pack(fill=tk.X, pady=(0, 6))
    lf_file.columnconfigure(1, weight=1)

    ttk.Label(lf_file, text="prompts_file", width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=4)
    ttk.Entry(lf_file, textvariable=s.prompts_file).grid(
        row=0, column=1, sticky=tk.EW, padx=(0, 2), pady=4)
    ttk.Button(lf_file, text="Browse", width=7,
               command=lambda: _browse_file(
                   s.prompts_file, "TOMLファイル選択",
                   [("TOML", "*.toml"), ("All", "*.*")])).grid(
        row=0, column=2, padx=(0, 4), pady=4)

    btn_row = ttk.Frame(lf_file)
    btn_row.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(0, 4))
    ttk.Button(btn_row, text="TOMLをエディタに読み込む",
               command=lambda: _load_toml(s, toml_text)).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_row, text="エディタの内容をTOMLに保存",
               command=lambda: _save_toml(s, toml_text)).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_row, text="テンプレートを挿入",
               command=lambda: _insert_template(toml_text)).pack(side=tk.LEFT)

    # --- LECO概要説明 ---
    lf_info = ttk.LabelFrame(parent, text="LECO 設定の説明")
    lf_info.pack(fill=tk.X, pady=(0, 6))
    info_text = (
        "target: 消去または強調する概念／スタイル\n"
        "positive: target を含む肯定プロンプト（eraseの消去源）\n"
        "neutral: 中立的な参照プロンプト（アンカー）\n"
        "unconditional: 空文字列推奨（CFGの非条件側）\n"
        "action: \"erase\" (概念消去) / \"enhance\" (概念強調)\n"
        "guidance_scale: 部分デノイズのCFGスケール（デフォルト3.0）"
    )
    ttk.Label(lf_info, text=info_text, justify=tk.LEFT,
              foreground="#475569", font=("TkDefaultFont", 9)).pack(
        anchor=tk.W, padx=8, pady=4)

    # --- TOMLエディタ ---
    lf_editor = ttk.LabelFrame(parent, text="TOMLエディタ（直接編集可）")
    lf_editor.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
    lf_editor.rowconfigure(0, weight=1)
    lf_editor.columnconfigure(0, weight=1)

    toml_text = tk.Text(lf_editor, wrap=tk.NONE, font=("TkFixedFont", 9))
    scroll_y = ttk.Scrollbar(lf_editor, orient=tk.VERTICAL, command=toml_text.yview)
    scroll_x = ttk.Scrollbar(lf_editor, orient=tk.HORIZONTAL, command=toml_text.xview)
    toml_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    scroll_y.grid(row=0, column=1, sticky=tk.NS)
    scroll_x.grid(row=1, column=0, sticky=tk.EW)
    toml_text.grid(row=0, column=0, sticky=tk.NSEW)

    # 初期テンプレートを表示
    toml_text.insert(tk.END, _TOML_TEMPLATE)

    # state に toml_text への参照を保持（保存時に使用）
    s._toml_text_widget = toml_text


def _load_toml(s: _LecoTrainState, text_widget: tk.Text) -> None:
    path = s.prompts_file.get().strip()
    if not path:
        messagebox.showwarning("未選択", "prompts_file が未設定です。")
        return
    try:
        content = Path(path).read_text(encoding="utf-8")
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
    except Exception as e:
        messagebox.showerror("読み込みエラー", str(e))


def _save_toml(s: _LecoTrainState, text_widget: tk.Text) -> None:
    path = s.prompts_file.get().strip()
    if not path:
        path = filedialog.asksaveasfilename(
            title="TOMLファイルの保存先",
            defaultextension=".toml",
            filetypes=[("TOML", "*.toml"), ("All", "*.*")],
        )
        if not path:
            return
        s.prompts_file.set(path)
    try:
        content = text_widget.get("1.0", tk.END)
        Path(path).write_text(content, encoding="utf-8")
        messagebox.showinfo("保存完了", f"保存しました:\n{path}")
    except Exception as e:
        messagebox.showerror("保存エラー", str(e))


def _insert_template(text_widget: tk.Text) -> None:
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, _TOML_TEMPLATE)


# ──────────────────────────────────────────────────────────────────────────────
# タブ3: ネットワーク
# ──────────────────────────────────────────────────────────────────────────────
def _build_network_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
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

    ttk.Label(lf, text="※ LECOは常にU-Net(DiT)のみ学習", foreground="#64748B").grid(
        row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

    _entry_browse_row(lf, 4, "network_weights (再開用)", s.network_weights,
                      filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])


# ──────────────────────────────────────────────────────────────────────────────
# タブ4: 学習設定
# ──────────────────────────────────────────────────────────────────────────────
def _build_train_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
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

    # row 3: max_train_steps / save_every_n_steps（LECOはステップ管理）
    ttk.Label(lf, text="max_train_steps", width=22, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=1, to=999999, textvariable=s.max_train_steps, width=10).grid(
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

    lf2 = ttk.LabelFrame(parent, text="LECOパラメータ")
    lf2.pack(fill=tk.X, pady=(0, 8))
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    ttk.Label(lf2, text="max_denoising_steps", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf2, from_=1, to=1000, textvariable=s.max_denoising_steps, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf2, text="denoise_guidance_scale", width=22, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf2, textvariable=s.leco_denoise_guidance_scale, width=10).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    lf3 = ttk.LabelFrame(parent, text="メモリ最適化")
    lf3.pack(fill=tk.X)
    ttk.Checkbutton(lf3, text="gradient_checkpointing",
                    variable=s.gradient_checkpointing).grid(
        row=0, column=0, sticky=tk.W, padx=8, pady=3)


# ──────────────────────────────────────────────────────────────────────────────
# タブ5: 詳細（Anima固有）
# ──────────────────────────────────────────────────────────────────────────────
def _build_adv_tab(parent: ttk.Frame, s: _LecoTrainState) -> None:
    parent.columnconfigure(1, weight=1)

    lf = ttk.LabelFrame(parent, text="Anima固有設定")
    lf.pack(fill=tk.X, pady=(0, 8))
    lf.columnconfigure(1, weight=1)
    lf.columnconfigure(3, weight=1)

    # row 0: attn_mode / discrete_flow_shift
    ttk.Label(lf, text="attn_mode", width=22, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Combobox(lf, textvariable=s.attn_mode, values=ATTN_MODES,
                 state="readonly", width=14).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="discrete_flow_shift", width=20, anchor=tk.W).grid(
        row=0, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf, textvariable=s.discrete_flow_shift, width=10).grid(
        row=0, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 1: blocks_to_swap / vae_chunk_size
    ttk.Label(lf, text="blocks_to_swap (0=無効)", width=22, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=0, to=100, textvariable=s.blocks_to_swap, width=8).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="vae_chunk_size (空欄=無効)", width=22, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(lf, textvariable=s.vae_chunk_size, width=10).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 2: qwen3_max_token_length / t5_max_token_length
    ttk.Label(lf, text="qwen3_max_token_length", width=22, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.qwen3_max_token_length, width=8).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(lf, text="t5_max_token_length", width=20, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(lf, from_=64, to=4096, textvariable=s.t5_max_token_length, width=8).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    # row 3: t5_tokenizer_path
    _entry_browse_row(lf, 3, "t5_tokenizer_path", s.t5_tokenizer_path, is_dir=True)

    lf2 = ttk.LabelFrame(parent, text="オフロード")
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


# ──────────────────────────────────────────────────────────────────────────────
# 実行パネル
# ──────────────────────────────────────────────────────────────────────────────
def _build_run_panel(parent: ttk.Frame, s: _LecoTrainState) -> None:
    frm = ttk.LabelFrame(parent, text="実行")
    frm.pack(fill=tk.X, pady=(6, 0))

    cmd_frame = ttk.Frame(frm)
    cmd_frame.pack(fill=tk.X, padx=4, pady=(4, 0))
    ttk.Label(cmd_frame, text="コマンドプレビュー:").pack(side=tk.LEFT)
    ttk.Button(cmd_frame, text="更新",
               command=lambda: _refresh_cmd(s, cmd_text)).pack(side=tk.LEFT, padx=4)

    cmd_text = tk.Text(frm, height=3, wrap=tk.WORD, font=("TkFixedFont", 8))
    cmd_text.pack(fill=tk.X, padx=4, pady=2)

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill=tk.X, padx=4, pady=(2, 4))
    ttk.Label(btn_row, textvariable=s.status_var, foreground="#334155").pack(
        side=tk.LEFT, padx=4)
    ttk.Button(btn_row, text="■ Stop",
               command=lambda: _stop_training(s)).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(btn_row, text="▶ 学習開始", style="Run.TButton",
               command=lambda: _start_training(s, cmd_text)).pack(side=tk.RIGHT, padx=4)

    log_frame = ttk.LabelFrame(parent, text="学習ログ")
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
                for w in s._log_widgets:
                    w.insert(tk.END, msg + "\n")
                    w.see(tk.END)
            except queue.Empty:
                break
        parent.after(200, _drain)

    if not s._log_drain_started:
        s._log_drain_started = True
        parent.after(200, _drain)


# ──────────────────────────────────────────────────────────────────────────────
# コマンド生成
# ──────────────────────────────────────────────────────────────────────────────
def _build_command(s: _LecoTrainState) -> list[str]:
    """GUIの設定値から accelerate launch コマンドリストを生成する。"""
    sd_scripts_root = s.paths.root / "sd-scripts"
    train_script    = sd_scripts_root / "anima_train_leco.py"

    wrapper = sd_scripts_root / "_gui_leco_wrapper.py"
    wrapper.write_text(
        "import sys, os\n"
        f"sys.path.insert(0, '{sd_scripts_root.as_posix()}')\n"
        f"os.chdir('{sd_scripts_root.as_posix()}')\n"
        f"with open('{train_script.as_posix()}', encoding='utf-8') as _f:\n"
        "    _code = compile(_f.read(), _f.name, 'exec')\n"
        f"exec(_code, {{'__name__': '__main__', '__file__': '{train_script.as_posix()}'}})\n",
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
        "--prompts_file",                  s.prompts_file.get(),
        "--output_dir",                    s.output_dir.get(),
        "--output_name",                   s.output_name.get(),
        "--network_module",                s.network_module.get(),
        "--network_dim",                   str(s.network_dim.get()),
        "--network_alpha",                 str(s.network_alpha.get()),
        "--learning_rate",                 s.lr.get(),
        "--lr_scheduler",                  s.lr_scheduler.get(),
        "--lr_warmup_steps",               str(s.lr_warmup_steps.get()),
        "--optimizer_type",                s.optimizer.get(),
        "--max_train_steps",               str(s.max_train_steps.get()),
        "--save_every_n_steps",            str(s.save_every_n_steps.get()),
        "--mixed_precision",               s.mixed_precision.get(),
        "--save_precision",                s.precision.get(),
        "--gradient_accumulation_steps",   str(s.grad_accum.get()),
        "--max_grad_norm",                 str(s.max_grad_norm.get()),
        "--max_denoising_steps",           str(s.max_denoising_steps.get()),
        "--leco_denoise_guidance_scale",   str(s.leco_denoise_guidance_scale.get()),
        "--attn_mode",                     s.attn_mode.get(),
        "--discrete_flow_shift",           str(s.discrete_flow_shift.get()),
        "--qwen3_max_token_length",        str(s.qwen3_max_token_length.get()),
        "--t5_max_token_length",           str(s.t5_max_token_length.get()),
        "--network_train_unet_only",       # LECO は常にDiTのみ
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
        (s.gradient_checkpointing,            "--gradient_checkpointing"),
        (s.split_attn,                         "--split_attn"),
        (s.unsloth_offload_checkpointing,      "--unsloth_offload_checkpointing"),
        (s.cpu_offload_checkpointing,          "--cpu_offload_checkpointing"),
        (s.vae_disable_cache,                  "--vae_disable_cache"),
    ]
    for var, flag in bool_flags:
        if var.get():
            cmd.append(flag)

    # サンプル生成
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        _spf = _leco_write_sample_prompt_file(s)
        cmd += ["--sample_every_n_steps", s.sample_every_n_steps.get().strip() or "100"]
        cmd += ["--sample_prompts",   str(_spf)]
        cmd += ["--sample_save_dir",  str(s.paths.root / "log" / "sample_gen")]
        if s.sample_keep_vae.get():
            cmd.append("--sample_keep_vae")

    # 階層学習
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

    return cmd


def _refresh_cmd(s: _LecoTrainState, text_widget: tk.Text) -> None:
    try:
        cmd = _build_command(s)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, " ".join(cmd))
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, f"[エラー] {e}")
        text_widget.config(state=tk.DISABLED)


# ──────────────────────────────────────────────────────────────────────────────
# バリデーション / 実行 / 停止
# ──────────────────────────────────────────────────────────────────────────────
def _validate(s: _LecoTrainState) -> str | None:
    if not s.model_path.get():
        return "DiTモデルパスが未設定です。"
    if not s.vae_path.get():
        return "VAEパスが未設定です。"
    if not s.qwen3_path.get():
        return "Qwen3テキストエンコーダパスが未設定です。"
    if not s.prompts_file.get():
        return "prompts_file (TOML) が未設定です。"
    if not Path(s.prompts_file.get()).exists():
        return f"prompts_file が見つかりません:\n{s.prompts_file.get()}"
    if s.max_train_steps.get() < 1:
        return "max_train_steps は1以上を指定してください。"
    if s.save_every_n_steps.get() < 1:
        return "save_every_n_steps は1以上を指定してください。"
    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプルAが有効ですが、promptが未設定です。"
    if s.sample_b_enabled.get() and not s.sample_b_prompt.get().strip():
        return "サンプルBが有効ですが、promptが未設定です。"
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        try:
            if int(s.sample_every_n_steps.get().strip() or "100") <= 0:
                return "サンプル出力のstep間隔は1以上を指定してください。"
        except ValueError:
            return "サンプル出力のstep間隔は整数で指定してください。"
    return None


def _start_training(s: _LecoTrainState, cmd_text: tk.Text) -> None:
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
    s.log_fn("[LECO Train] 学習開始")
    s._log_queue.put(f"[CMD] {' '.join(cmd)}")

    def _worker():
        import os, re as _re
        log_dir = s.paths.root / "log" / "leco_train"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts}.txt"
        s.log_fn(f"[LECO Train] ログ: {log_path}")
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(sd_scripts_root) + (os.pathsep + existing if existing else "")
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
                    line = _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line.rstrip())
                    s._log_queue.put(line)
                    s._monitor_queue.put(line)
                    s._monitor_layer_queue.put(line)
                    lf.write(line + "\n")
                    lf.flush()
            proc.wait()
            rc = proc.returncode
            msg = f"[LECO Train] 完了 (return code: {rc})"
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set("完了" if rc == 0 else f"エラー (code={rc})")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception as exc:
            msg = f"[LECO Train] 起動エラー: {exc}"
            s._log_queue.put(msg)
            s.log_fn(msg)
            s.status_var.set("起動失敗")
        finally:
            s._proc = None

    threading.Thread(target=_worker, daemon=True).start()


def _stop_training(s: _LecoTrainState) -> None:
    if s._proc is None or s._proc.poll() is not None:
        s.log_fn("[LECO Train] 停止対象のプロセスがありません。")
        return
    import os, signal
    try:
        os.kill(s._proc.pid, signal.CTRL_BREAK_EVENT)
    except Exception:
        s._proc.terminate()
    s.log_fn("[LECO Train] 停止シグナルを送信しました。")
    s.status_var.set("停止中...")


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ2: 階層学習タブ
# ══════════════════════════════════════════════════════════════════════════════

def _build_layer_train_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """階層別学習率スケールを設定するタブ。"""
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

    canvas_frame = ttk.LabelFrame(
        parent, text="ブロック別スケール (0.0 = freeze / 1.0 = base LR)"
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

    _refresh_layer_controls(s, ctrl_canvas, ctrl_inner)


def _layer_group_names(mode: str) -> list[str]:
    if mode == "Matrix":
        return [f"{b}_{c}" for b in MATRIX_BLOCKS for c in MATRIX_COMPONENTS]
    if mode == "Component":
        return list(COMPONENT_GROUPS)
    return [f"blocks.{i}" for i in range(28)]


def _refresh_layer_controls(
    s: "_LecoTrainState",
    canvas: tk.Canvas,
    inner: ttk.Frame,
) -> None:
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

    if mode == "Component":
        warn_row = (len(groups) + cols - 1) // cols
        tk.Label(
            inner,
            text=(
                "⚠ Component モードはブロック情報をLoRAキーから分解できないため、"
                "全ブロック共通の平均スケールとして適用されます。\n"
                "ブロック別精度が必要な場合は Transformer または Matrix モードを使用してください。"
            ),
            foreground="red",
            justify=tk.LEFT,
            wraplength=600,
        ).grid(row=warn_row, column=0, columnspan=cols * 3, sticky=tk.W, padx=8, pady=(6, 2))
        canvas.configure(scrollregion=canvas.bbox("all"))
        s._layer_status_var.set(
            f"mode={mode}  {len(groups)} グループ  [警告] ブロック別精度低下あり"
        )
    else:
        s._layer_status_var.set(f"mode={mode}  {len(groups)} グループ")


def _snap_scale(var: tk.DoubleVar) -> None:
    v = var.get()
    var.set(round(round(v / 0.05) * 0.05, 4))


def _clamp_var(var: tk.DoubleVar) -> None:
    try:
        var.set(max(0.0, min(1.0, float(var.get()))))
    except tk.TclError:
        var.set(1.0)


def _convert_preset_scales(
    scales: dict[str, float],
    preset_mode: str,
    target_mode: str,
) -> dict[str, float]:
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


def _load_layer_preset(s: "_LecoTrainState", canvas: tk.Canvas, inner: ttk.Frame) -> None:
    """preset/leco_train/*.json から parameter_scales を読み込む。"""
    preset_dir = s.paths.root / "preset" / "leco_train"
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
        messagebox.showerror("プリセット読み込みエラー", str(exc))
        return

    new_mode = data.get("layer_display_mode", "Matrix")
    if new_mode in LAYER_TRAIN_MODES:
        s.layer_display_mode.set(new_mode)

    _refresh_layer_controls(s, canvas, inner)

    scales = data.get("layer_parameter_vars", {})
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


def _layer_scales_to_block_weights(mode: str, scales: dict[str, float]) -> list[float]:
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


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ2: モニターグラフタブ（空・予定地）
# ══════════════════════════════════════════════════════════════════════════════

def _build_monitor_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """モニターグラフタブ（後実装予定）。"""
    ttk.Label(
        parent,
        text="モニターグラフ（実装予定）\n\n学習開始後にリアルタイムでLoss/LRグラフを表示します。",
        foreground="#64748B",
        font=("TkDefaultFont", 10),
        justify=tk.CENTER,
    ).pack(expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ2: モニター階層タブ（空・予定地）
# ══════════════════════════════════════════════════════════════════════════════

def _build_monitor_layer_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """モニター階層タブ（後実装予定）。"""
    ttk.Label(
        parent,
        text="モニター階層（実装予定）\n\n階層学習が有効な場合、ブロック別の実効LRをリアルタイム表示します。",
        foreground="#64748B",
        font=("TkDefaultFont", 10),
        justify=tk.CENTER,
    ).pack(expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ2: プリセットタブ
# プリセットJSON には TOML内容・サンプル生成設定も含められる設計
# ══════════════════════════════════════════════════════════════════════════════

def _build_leco_preset_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """LECO学習設定を JSON で保存・復元するプリセットタブ。
    保存先: <project_root>/preset/leco_train/*.json
    JSON構造:
      {
        "version": 2,
        ...設定値...,
        "layer_parameter_vars": {...},
        "prompts_toml_content": "...",   # TOMLエディタの内容を同梱可
        "sample": {...},                 # サンプル生成設定（将来用）
      }
    """
    PRESET_DIR_REL = ("preset", "leco_train")

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

    # ── 名前入力 ────────────────────────────────────────────────
    name_row = ttk.Frame(parent)
    name_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(name_row, text="Name:").pack(side=tk.LEFT)
    name_var = tk.StringVar()
    ttk.Entry(name_row, textvariable=name_var, width=28).pack(side=tk.LEFT, padx=(4, 8))

    # TOML同梱オプション
    include_toml_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        name_row, text="TOMLを同梱", variable=include_toml_var
    ).pack(side=tk.LEFT, padx=(0, 8))

    btn_row = ttk.Frame(parent)
    btn_row.pack(fill=tk.X)

    # ── 内部ヘルパー ─────────────────────────────────────────────
    def _refresh_list() -> None:
        lb.delete(0, tk.END)
        for p in sorted(_preset_dir().glob("*.json")):
            lb.insert(tk.END, p.stem)

    def _collect() -> dict:
        data: dict = {
            "version": 2,
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
            "discrete_flow_shift":    float(s.discrete_flow_shift.get()),
            # 階層学習
            "layer_train_enabled": bool(s.layer_train_enabled.get()),
            "layer_display_mode":  s.layer_display_mode.get(),
            "layer_parameter_vars": {
                k: round(float(v.get()), 4)
                for k, v in s.layer_parameter_vars.items()
            },
            # サンプル生成
            "sample": {
                "every_n_steps":            s.sample_every_n_steps.get(),
                "keep_vae":                 bool(s.sample_keep_vae.get()),
                "width":                    int(s.sample_width.get()),
                "height":                   int(s.sample_height.get()),
                "steps":                    int(s.sample_steps.get()),
                "scale":                    float(s.sample_scale.get()),
                "flow_shift":               float(s.sample_flow_shift.get()),
                "a_enabled":                bool(s.sample_enabled.get()),
                "a_prompt":                 s.sample_prompt.get(),
                "a_negative_prompt":        s.sample_negative_prompt.get(),
                "b_enabled":                bool(s.sample_b_enabled.get()),
                "b_prompt":                 s.sample_b_prompt.get(),
                "b_negative_prompt":        s.sample_b_negative_prompt.get(),
            },
        }
        # TOMLエディタの内容を同梱（オプション）
        if include_toml_var.get():
            try:
                toml_content = s._toml_text_widget.get("1.0", tk.END)
                data["prompts_toml_content"] = toml_content
            except Exception:
                pass
        return data

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
        _s(s.output_name,       "output_name",        "leco_output")
        _s(s.precision,         "precision",          "bf16")
        _s(s.prompts_file,      "prompts_file",       "")
        _s(s.network_dim,       "network_dim",        4)
        _s(s.network_alpha,     "network_alpha",      1.0)
        _s(s.network_module,    "network_module",     "networks.lora")
        _s(s.network_weights,   "network_weights",    "")
        _s(s.lr,                "lr",                 "1e-4")
        _s(s.lr_scheduler,      "lr_scheduler",       "constant")
        _s(s.lr_warmup_steps,   "lr_warmup_steps",    0)
        _s(s.optimizer,         "optimizer",          "AdamW")
        _s(s.optimizer_args,    "optimizer_args",     "")
        _s(s.max_train_steps,   "max_train_steps",    500)
        _s(s.save_every_n_steps, "save_every_n_steps", 100)
        _s(s.seed,              "seed",               "42")
        _s(s.gradient_checkpointing, "gradient_checkpointing", True)
        _s(s.grad_accum,        "grad_accum",         1)
        _s(s.mixed_precision,   "mixed_precision",    "bf16")
        _s(s.max_grad_norm,     "max_grad_norm",      1.0)
        _s(s.max_denoising_steps, "max_denoising_steps", 20)
        _s(s.leco_denoise_guidance_scale, "leco_denoise_guidance_scale", 3.0)
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
        _s(s.discrete_flow_shift, "discrete_flow_shift", 1.0)

        # 階層学習スライダーは先行セット済みなのでスケール値のみ反映
        layer_scales = data.get("layer_parameter_vars", {})
        for k, v in layer_scales.items():
            if k in s.layer_parameter_vars:
                try:
                    s.layer_parameter_vars[k].set(float(v))
                except (ValueError, tk.TclError):
                    pass

        # サンプル生成設定
        sample = data.get("sample", {})
        if sample:
            def _ss(var, key, default):
                """sample サブ dict から値を取得して tk.Var にセットする。"""
                try:
                    var.set(sample.get(key, default))
                except (tk.TclError, ValueError):
                    try:
                        var.set(default)
                    except Exception:
                        pass
            _ss(s.sample_every_n_steps,       "every_n_steps",      "100")
            _ss(s.sample_keep_vae,            "keep_vae",           False)
            _ss(s.sample_width,               "width",              512)
            _ss(s.sample_height,              "height",             512)
            _ss(s.sample_steps,               "steps",              20)
            _ss(s.sample_scale,               "scale",              7.5)
            _ss(s.sample_flow_shift,          "flow_shift",         3.0)
            _ss(s.sample_enabled,             "a_enabled",          False)
            _ss(s.sample_prompt,              "a_prompt",           "")
            _ss(s.sample_negative_prompt,     "a_negative_prompt",  "")
            _ss(s.sample_b_enabled,           "b_enabled",          False)
            _ss(s.sample_b_prompt,            "b_prompt",           "")
            _ss(s.sample_b_negative_prompt,   "b_negative_prompt",  "")

        # TOMLエディタに内容を復元（同梱されている場合）
        toml_content = data.get("prompts_toml_content")
        if toml_content and hasattr(s, "_toml_text_widget"):
            try:
                s._toml_text_widget.delete("1.0", tk.END)
                s._toml_text_widget.insert(tk.END, toml_content)
            except Exception:
                pass

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
        # 階層学習: enabled/mode を先行セットしてスライダーを生成してから _apply
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
        (_preset_dir() / f"{pname}.json").unlink(missing_ok=True)
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
        ("保存",         _save),
        ("読み込み",     _load),
        ("削除",         _delete),
        ("エクスポート", _export),
        ("インポート",   _import),
        ("一覧更新",     _refresh_list),
    ]:
        ttk.Button(btn_row, text=text, command=cmd).pack(side=tk.LEFT, padx=4, pady=4)

    _refresh_list()


# ──────────────────────────────────────────────────────────────────────────────
# サンプル生成ヘルパー（leco_train.py 用）
# ──────────────────────────────────────────────────────────────────────────────

def _leco_sample_dir(s: "_LecoTrainState") -> Path:
    return s.paths.root / "log" / "sample_gen"


def _leco_sample_prompt_path(s: "_LecoTrainState") -> Path:
    return _leco_sample_dir(s) / "_sample_prompt.txt"


def _leco_build_prompt_line(
    prompt: str, neg: str, s: "_LecoTrainState", seed: int = 42
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


SAMPLE_FIXED_SEED = 42


def _leco_write_sample_prompt_file(s: "_LecoTrainState") -> Path:
    lines = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_leco_build_prompt_line(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            s,
            seed=SAMPLE_FIXED_SEED,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_leco_build_prompt_line(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            s,
            seed=SAMPLE_FIXED_SEED + 1,
        ))
    path = _leco_sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _build_leco_sample_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """LECO サンプル生成タブ。lora_train._build_sample_tab_common を流用。"""
    # lora_train モジュールが同一パッケージにある前提で動的インポート
    try:
        from . import lora_train as _lt
        _lt._build_sample_tab_common(parent, s, is_leco=True)
    except Exception:
        # フォールバック: 直接インポートが使えない場合は簡易UI
        _build_leco_sample_tab_inline(parent, s)


def _build_leco_sample_tab_inline(
    parent: ttk.Frame, s: "_LecoTrainState"
) -> None:
    """_build_sample_tab_common の leco_train 内スタンドアロン版。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    common = ttk.LabelFrame(parent, text="共通生成条件")
    common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    common.columnconfigure(1, weight=1)
    common.columnconfigure(3, weight=1)

    ttk.Label(common, text="step間隔", width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=99999, textvariable=s.sample_every_n_steps, width=8).grid(
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

    ttk.Checkbutton(
        common, text="VAEを学習中保持 (sample_keep_vae)",
        variable=s.sample_keep_vae,
    ).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    ab_nb = ttk.Notebook(parent)
    ab_nb.grid(row=1, column=0, sticky=tk.NSEW)
    tab_a = ttk.Frame(ab_nb, padding=4)
    tab_b = ttk.Frame(ab_nb, padding=4)
    ab_nb.add(tab_a, text="  サンプルA  ")
    ab_nb.add(tab_b, text="  サンプルB  ")

    def _ab_panel(tab, enabled_var, prompt_var, neg_var, label):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
        top.columnconfigure(1, weight=1)
        ttk.Checkbutton(top, text=f"サンプル{label}を有効にする",
                        variable=enabled_var).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, padx=2, pady=2)
        _sdir = s.paths.root / "log" / "sample_gen"
        _glob_pat = "*_00_*.png" if label == "A" else "*_01_*.png"
        ttk.Label(top, text="出力先:", foreground="#475569").grid(
            row=1, column=0, sticky=tk.W, padx=(2, 0), pady=2)
        ttk.Label(top, text=str(_sdir), foreground="#1D4ED8").grid(
            row=1, column=1, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(top, text="prompt", width=16, anchor=tk.W).grid(
            row=2, column=0, sticky=tk.W, padx=(2, 2), pady=2)
        ttk.Entry(top, textvariable=prompt_var).grid(
            row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)
        ttk.Label(top, text="negative", width=16, anchor=tk.W).grid(
            row=3, column=0, sticky=tk.W, padx=(2, 2), pady=2)
        ttk.Entry(top, textvariable=neg_var).grid(
            row=3, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

        gallery = ttk.LabelFrame(tab, text=f"最新サンプル{label}")
        gallery.grid(row=1, column=0, sticky=tk.NSEW)
        for c in range(5):
            gallery.columnconfigure(c, weight=1, uniform=f"lc_{label}")
        for r in range(2):
            gallery.rowconfigure(r, weight=1, uniform=f"lr_{label}")

        cells: list = []
        photo_refs: list = [None] * 10
        for idx in range(10):
            cell = ttk.Frame(gallery, padding=4)
            cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
            cell.columnconfigure(0, weight=1)
            cell.rowconfigure(0, weight=1)
            il = ttk.Label(cell, anchor=tk.CENTER)
            il.grid(row=0, column=0, sticky=tk.NSEW)
            el = ttk.Label(cell, text="step -", anchor=tk.CENTER)
            el.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
            cells.append((il, el))

        def _re_search(pat, text):
            import re as _re
            m = _re.search(pat, text)
            return m

        def _refresh(schedule_next=False):
            files = sorted(
                _sdir.glob(_glob_pat), key=lambda p: p.stat().st_mtime, reverse=True
            )[:10] if _sdir.exists() else []
            try:
                from PIL import Image as _Im, ImageTk as _ITk
            except Exception:
                _Im = _ITk = None
            for idx, (il, el) in enumerate(cells):
                if idx >= len(files):
                    il.configure(image="", text="")
                    el.configure(text="step -")
                    photo_refs[idx] = None
                    continue
                p = files[idx]
                m = _re_search(r"_([0-9]{6})_", p.stem)
                el.configure(text=f"step {int(m.group(1))}" if m else p.name)
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
                tab.after(2000, lambda: _refresh(True))

        btn_row = ttk.Frame(top)
        btn_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 2))
        ttk.Button(btn_row, text="表示更新", command=lambda: _refresh(False)).pack(
            side=tk.LEFT, padx=(0, 6))
        _refresh(True)

    _ab_panel(tab_a, s.sample_enabled,   s.sample_prompt,   s.sample_negative_prompt,   "A")
    _ab_panel(tab_b, s.sample_b_enabled, s.sample_b_prompt, s.sample_b_negative_prompt, "B")
