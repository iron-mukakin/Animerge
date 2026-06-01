from __future__ import annotations

import datetime
import json
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import AppPaths, MergeOptions
from .merge import (
    adjustment_group,
    extract_lora_difference,
    fuse_lora_into_model,
    is_merge_target,
    merge_loras,
    merge_models,
)
from .model_io import DependencyError, list_state_dict_layers, scan_models
from .analysis import (
    ANALYSIS_METHODS,
    LAYER_DISPLAY_MODES,
    AnalysisReport,
    load_analysis_log,
    run_analysis,
)
from .analysis_viewer import build_viewer_tab
from .lora_train import build_lora_train_tab
from .leco_train import build_leco_train_tab


MATRIX_BLOCKS = ("Input", "Middle", "Output")
MATRIX_COMPONENTS = ("Attention", "MLP", "Norm", "ResNet", "Timestep")
COMPONENT_GROUPS = ("MLP", "Norm", "ResNet", "Timestep", "Other")

# レイヤー列数
LAYER_COLUMNS = 3


class AnimaModelEditor(tk.Tk):
    def __init__(self, paths: AppPaths | None = None, mode: str = "cpu") -> None:
        super().__init__()
        self.paths = paths or AppPaths.from_root()
        self.paths.ensure()
        self._mode = mode  # "cpu" or "cuda"
        self.title("Animerge v1.0")
        self.geometry("1120x820")
        self.minsize(980, 720)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._merge_stop_event = threading.Event()
        self._analysis_stop_event = threading.Event()

        self.device_var = tk.StringVar(value=mode)
        self.base_model_var = tk.StringVar()
        self.secondary_model_var = tk.StringVar()
        self.lora_var = tk.StringVar()
        self.secondary_lora_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(self.paths.checkpoints / "merged_model.safetensors"))
        # タブ別デフォルト出力パスを保持
        self._default_output_model = str(self.paths.checkpoints / "merged_model.safetensors")
        self._default_output_lora  = str(self.paths.lora / "merged_lora.safetensors")

        self.alpha_var = tk.DoubleVar(value=0.5)
        self.extract_rank_var = tk.IntVar(value=16)
        self.cosine_var = tk.DoubleVar(value=0.4)
        self.auto_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.freeze_input_var = tk.BooleanVar(value=False)
        self.freeze_middle_var = tk.BooleanVar(value=False)
        self.freeze_output_var = tk.BooleanVar(value=False)

        self.layer_mode_var = tk.StringVar(value="Matrix")
        self.parameter_vars: dict[str, tk.DoubleVar] = {}
        self.loaded_layers: list[tuple[str, str]] = []

        # タブごとに独立したウィジェット参照を保持する辞書
        # キー: tab_type ("model" / "lora")
        # 値: {"canvas", "inner", "window", "scroll", "status_var", "mode_combo"}
        self._tab_widgets: dict[str, dict] = {}
        self._sub_notebooks: dict[str, object] = {}
        self._active_tab_type: str = "model"

        # メモリ上にロードされたモデル名を保持 (アンロード用)
        self._loaded_model_names: list[str] = []

        self.model_choices: list[str] = []
        self.lora_choices: list[str] = []

        # ── タブ3: レイヤー分析 ──────────────────────────────
        self.analysis_target_var = tk.StringVar()
        self.analysis_target_type_var = tk.StringVar(value="model")
        self.analysis_method_var = tk.StringVar(value=ANALYSIS_METHODS[0])
        self.analysis_layer_mode_var = tk.StringVar(value=LAYER_DISPLAY_MODES[0])
        self.analysis_key_correction_var = tk.BooleanVar(value=False)
        self._analysis_result_report: AnalysisReport | None = None

        self._build_ui()
        self.refresh_files()
        self.rebuild_parameter_controls()
        self.after(100, self._drain_logs)

    # ─── スタイル ────────────────────────────────────────────
    def _apply_styles(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "MainTab.TNotebook.Tab",
            font=("TkDefaultFont", 11, "bold"),
            padding=(14, 6),
            background="#2563EB",
            foreground="black",
        )
        style.map(
            "MainTab.TNotebook.Tab",
            background=[("selected", "#1D4ED8"), ("active", "#3B82F6")],
            foreground=[("selected", "green"), ("active", "purple")],
        )
        style.configure(
            "SubTab.TNotebook.Tab",
            font=("TkDefaultFont", 9, "bold"),
            padding=(10, 4),
            background="#475569",
            foreground="black",
        )
        style.map(
            "SubTab.TNotebook.Tab",
            background=[("selected", "#334155"), ("active", "#64748B")],
            foreground=[("selected", "green"), ("active", "purple")],
        )
        style.configure(
            "Run.TButton",
            font=("TkDefaultFont", 11, "bold"),
            padding=(16, 8),
            background="#16A34A",
            foreground="red",
        )
        style.map(
            "Run.TButton",
            background=[("active", "#15803D")],
        )

    # ─── UI 構築 ─────────────────────────────────────────────
    def _build_ui(self) -> None:
        self._apply_styles()

        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        # ── 主タブを最上部に配置 ──
        self.main_notebook = ttk.Notebook(root, style="MainTab.TNotebook")
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        # 主タブ：本体マージ / LoRAマージ / レイヤー分析 / 詳細分析 / LoRA学習
        model_main    = ttk.Frame(self.main_notebook, padding=4)
        lora_main     = ttk.Frame(self.main_notebook, padding=4)
        analysis_main = ttk.Frame(self.main_notebook, padding=4)
        viewer_main   = ttk.Frame(self.main_notebook, padding=4)
        train_main    = ttk.Frame(self.main_notebook, padding=4)
        leco_main     = ttk.Frame(self.main_notebook, padding=4)
        self.main_notebook.add(model_main,    text="  本体マージ  ")
        self.main_notebook.add(lora_main,     text="  LoRAマージ  ")
        self.main_notebook.add(analysis_main, text="  レイヤー分析  ")
        self.main_notebook.add(viewer_main,   text="  詳細分析  ")
        self.main_notebook.add(train_main,    text="  LoRA学習  ")
        self.main_notebook.add(leco_main,     text="  LECO学習  ")

        # 各主タブ内の共通エリアを構築
        self._build_main_tab_content(model_main, tab_type="model")
        self._build_main_tab_content(lora_main, tab_type="lora")
        self._build_analysis_tab(analysis_main)
        build_viewer_tab(viewer_main, self.paths.log_analysis, self.log)
        self._lora_train_state = build_lora_train_tab(
            train_main,
            self.paths,
            self.log,
            lambda: self.model_choices,
        )
        self._leco_train_state = build_leco_train_tab(
            leco_main,
            self.paths,
            self.log,
            lambda: self.model_choices,
        )

        # ログ（最下部）
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill=tk.X, pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=5, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _on_main_tab_changed(self, _event: tk.Event) -> None:
        """主タブ切り替え時にアクティブなタブの参照を更新し、レイヤーを再描画。"""
        idx = self.main_notebook.index("current")
        if idx == 0:
            self._active_tab_type = "model"
            self.output_var.set(self._default_output_model)
        elif idx == 1:
            self._active_tab_type = "lora"
            self.output_var.set(self._default_output_lora)
        elif idx == 2:
            self._active_tab_type = "analysis"
            return  # 分析タブは merge 系のコントロール再構築不要
        elif idx == 3:
            self._active_tab_type = "viewer"
            return  # 詳細分析タブは merge 系のコントロール再構築不要
        elif idx == 4:
            self._active_tab_type = "lora_train"
            return  # LoRA学習タブは merge 系のコントロール再構築不要
        elif idx == 5:
            self._active_tab_type = "leco_train"
            return  # LECO学習タブは merge 系のコントロール再構築不要
        self.rebuild_parameter_controls()

    def _build_main_tab_content(self, parent: ttk.Frame, tab_type: str) -> None:
        """主タブ内に共通エリア + 副タブを構築する。ウィジェット参照はタブ別に保存。"""
        # Global System
        system = ttk.LabelFrame(parent, text="Global System")
        system.pack(fill=tk.X)
        _mode_text = "GPU Mode" if self._mode == "cuda" else "CPU Mode"
        _mode_fg = "#22C55E" if self._mode == "cuda" else "#64748B"
        ttk.Label(system, text="Device").grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        ttk.Label(system, text=_mode_text, foreground=_mode_fg, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, columnspan=2, padx=4, pady=8, sticky=tk.W)
        ttk.Button(system, text="Rescan folders", command=self.refresh_files).grid(row=0, column=3, padx=8)
        ttk.Button(system, text="モデルをアンロード", command=self.unload_models).grid(row=0, column=4, padx=8)
        ttk.Label(system, text=f"checkpoints: {self.paths.checkpoints}").grid(row=1, column=0, columnspan=5, padx=8, sticky=tk.W)
        ttk.Label(system, text=f"lora: {self.paths.lora}").grid(row=2, column=0, columnspan=5, padx=8, sticky=tk.W)

        # Merge Parameters
        params = ttk.LabelFrame(parent, text="Merge Parameters")
        params.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(params, text="Alpha").grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
        alpha_scale = ttk.Scale(params, from_=0.0, to=1.0, variable=self.alpha_var, orient=tk.HORIZONTAL)
        alpha_scale.grid(row=0, column=1, sticky=tk.EW)
        alpha_scale.bind("<ButtonRelease-1>", lambda e: self._snap_scale(self.alpha_var))
        alpha_entry = ttk.Entry(params, textvariable=self.alpha_var, width=8)
        alpha_entry.grid(row=0, column=2, padx=8)
        alpha_entry.bind("<FocusOut>", lambda e: self._clamp_var(self.alpha_var))
        ttk.Checkbutton(params, text="Cosine auto-correction", variable=self.auto_var).grid(row=0, column=3, padx=8)
        ttk.Label(params, text="Cosine threshold").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
        ttk.Entry(params, textvariable=self.cosine_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(params, text="Dry-run validation", variable=self.dry_run_var).grid(row=1, column=3, padx=8)
        freeze_bar = ttk.Frame(params)
        freeze_bar.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=8, pady=4)
        ttk.Checkbutton(freeze_bar, text="Freeze Input bias", variable=self.freeze_input_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(freeze_bar, text="Freeze Middle bias", variable=self.freeze_middle_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(freeze_bar, text="Freeze Output bias", variable=self.freeze_output_var).pack(side=tk.LEFT)
        params.columnconfigure(1, weight=1)

        # Layer Adjustment
        adjust_frame = ttk.LabelFrame(parent, text="Layer Adjustment")
        adjust_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        top = ttk.Frame(adjust_frame)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="Layer display").pack(side=tk.LEFT)

        # タブ別の layer_mode_var は共有（どちらのタブでも同じモードを使う）
        mode_combo = ttk.Combobox(
            top,
            textvariable=self.layer_mode_var,
            values=("Matrix", "Transformer", "Component"),
            state="readonly",
            width=16,
        )
        mode_combo.pack(side=tk.LEFT, padx=8)
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: self.rebuild_parameter_controls())

        ttk.Button(top, text="Load base structure", command=self.load_base_structure).pack(side=tk.LEFT, padx=8)

        status_var = tk.StringVar(value="Matrix mode uses rule-based groups.")
        ttk.Label(top, textvariable=status_var).pack(side=tk.LEFT, padx=8)

        canvas_holder = ttk.Frame(adjust_frame)
        canvas_holder.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        canvas = tk.Canvas(canvas_holder, height=200, highlightthickness=0)
        scroll = ttk.Scrollbar(canvas_holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        # タブ別に参照を保存
        self._tab_widgets[tab_type] = {
            "canvas": canvas,
            "inner": inner,
            "window": window,
            "scroll": scroll,
            "status_var": status_var,
            "mode_combo": mode_combo,
        }

        inner.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.bind("<Configure>", lambda e, c=canvas, w=window: c.itemconfigure(w, width=e.width))

        # 副タブ（入れ子）
        sub_notebook = ttk.Notebook(parent, style="SubTab.TNotebook")
        sub_notebook.pack(fill=tk.X, pady=(8, 0))
        self._sub_notebooks[tab_type] = sub_notebook

        if tab_type == "model":
            self._build_model_merge_tab(sub_notebook)
            self._build_lora_fuse_tab(sub_notebook)
            self._build_preset_tab(sub_notebook, tab_type="model", label=" 1-3 Preset ")
        else:
            self._build_lora_merge_tab(sub_notebook)
            self._build_lora_extract_tab(sub_notebook)
            self._build_preset_tab(sub_notebook, tab_type="lora", label=" 2-3 Preset ")

    # ─── アクティブタブのウィジェット取得ヘルパー ────────────
    def _tw(self, key: str):
        return self._tab_widgets[self._active_tab_type][key]

    # ─── 副タブ ───────────────────────────────────────────────
    def _build_model_merge_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text=" 1-1 Model-to-Model ")
        ttk.Label(tab, text="Base Model").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.base_combo = ttk.Combobox(tab, textvariable=self.base_model_var, state="readonly")
        self.base_combo.grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Label(tab, text="Secondary Model").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.secondary_combo = ttk.Combobox(tab, textvariable=self.secondary_model_var, state="readonly")
        self.secondary_combo.grid(row=1, column=1, sticky=tk.EW, padx=8)
        self._output_controls(tab, 2)
        _bf_mm = ttk.Frame(tab)
        _bf_mm.grid(row=3, column=1, sticky=tk.E, pady=12)
        ttk.Button(_bf_mm, text="■  Stop", command=self._stop_merge).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(_bf_mm, text="▶  Run model merge", style="Run.TButton", command=self.start_model_merge).pack(side=tk.LEFT)
        tab.columnconfigure(1, weight=1)

    def _build_lora_fuse_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text=" 1-2 LoRA-to-Model ")
        ttk.Label(tab, text="Base Model").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.lora_base_combo = ttk.Combobox(tab, textvariable=self.base_model_var, state="readonly")
        self.lora_base_combo.grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Label(tab, text="LoRA").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.lora_combo = ttk.Combobox(tab, textvariable=self.lora_var, state="readonly")
        self.lora_combo.grid(row=1, column=1, sticky=tk.EW, padx=8)
        self._output_controls(tab, 2)
        _bf_lf = ttk.Frame(tab)
        _bf_lf.grid(row=3, column=1, sticky=tk.E, pady=12)
        ttk.Button(_bf_lf, text="■  Stop", command=self._stop_merge).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(_bf_lf, text="▶  Run LoRA fuse", style="Run.TButton", command=self.start_lora_fuse).pack(side=tk.LEFT)
        tab.columnconfigure(1, weight=1)

    def _build_lora_merge_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text=" 2-1 LoRA Merge ")
        ttk.Label(tab, text="Base LoRA").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.lora_merge_base_combo = ttk.Combobox(tab, textvariable=self.lora_var, state="readonly")
        self.lora_merge_base_combo.grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Label(tab, text="Secondary LoRA").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.secondary_lora_combo = ttk.Combobox(tab, textvariable=self.secondary_lora_var, state="readonly")
        self.secondary_lora_combo.grid(row=1, column=1, sticky=tk.EW, padx=8)
        self._output_controls(tab, 2)
        _bf_lm = ttk.Frame(tab)
        _bf_lm.grid(row=3, column=1, sticky=tk.E, pady=12)
        ttk.Button(_bf_lm, text="■  Stop", command=self._stop_merge).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(_bf_lm, text="▶  Run LoRA merge", style="Run.TButton", command=self.start_lora_merge).pack(side=tk.LEFT)
        tab.columnconfigure(1, weight=1)

    def _build_lora_extract_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text=" 2-2 Difference Extract ")
        ttk.Label(tab, text="Base Model").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.extract_base_combo = ttk.Combobox(tab, textvariable=self.base_model_var, state="readonly")
        self.extract_base_combo.grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Label(tab, text="Target Model").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.extract_target_combo = ttk.Combobox(tab, textvariable=self.secondary_model_var, state="readonly")
        self.extract_target_combo.grid(row=1, column=1, sticky=tk.EW, padx=8)
        self._output_controls(tab, 2)
        ttk.Label(tab, text="LoRA Rank").grid(row=3, column=0, sticky=tk.W, pady=6)
        ttk.Spinbox(tab, from_=1, to=256, textvariable=self.extract_rank_var, width=8).grid(row=3, column=1, sticky=tk.W, padx=8)
        _bf_le = ttk.Frame(tab)
        _bf_le.grid(row=4, column=1, sticky=tk.E, pady=12)
        ttk.Button(_bf_le, text="■  Stop", command=self._stop_merge).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(_bf_le, text="▶  Run difference extract", style="Run.TButton", command=self.start_lora_extract).pack(side=tk.LEFT)
        tab.columnconfigure(1, weight=1)

    # --- Preset Tab ---------------------------------------------------
    def _build_preset_tab(self, notebook, tab_type: str, label: str) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text=label)

        list_frame = ttk.LabelFrame(tab, text="Saved Presets")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        preset_listbox = tk.Listbox(list_frame, height=8, selectmode=tk.SINGLE)
        lb_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=preset_listbox.yview)
        preset_listbox.configure(yscrollcommand=lb_scroll.set)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        name_var = tk.StringVar()
        ttk.Label(btn_row, text="Name:").pack(side=tk.LEFT)
        ttk.Entry(btn_row, textvariable=name_var, width=24).pack(side=tk.LEFT, padx=(4, 8))

        def _refresh_list() -> None:
            preset_listbox.delete(0, tk.END)
            _preset_dir = self.paths.root / "preset" / "merge"
            _preset_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(_preset_dir.glob("*.json")):
                preset_listbox.insert(tk.END, p.stem)

        def _save_preset() -> None:
            pname = name_var.get().strip()
            if not pname:
                messagebox.showerror("Preset", "Enter a preset name.")
                return
            data = {
                "alpha": float(self.alpha_var.get()),
                "cosine_threshold": float(self.cosine_var.get()),
                "auto_correction": bool(self.auto_var.get()),
                "dry_run": bool(self.dry_run_var.get()),
                "freeze_bias_input": bool(self.freeze_input_var.get()),
                "freeze_bias_middle": bool(self.freeze_middle_var.get()),
                "freeze_bias_output": bool(self.freeze_output_var.get()),
                "layer_display_mode": self.layer_mode_var.get(),
                "parameter_scales": {
                    k: float(v.get()) for k, v in self.parameter_vars.items()
                },
            }
            safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in pname)
            _preset_dir = self.paths.root / "preset" / "merge"
            _preset_dir.mkdir(parents=True, exist_ok=True)
            dest = _preset_dir / f"{safe}.json"
            dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            _refresh_list()
            self.log(f"[Preset] Saved: {dest.name}")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_action_log([f"{now} [Preset Save] {dest.name}", ""])

        def _load_preset() -> None:
            sel = preset_listbox.curselection()
            if not sel:
                messagebox.showerror("Preset", "Select a preset to load.")
                return
            pname = preset_listbox.get(sel[0])
            src = self.paths.root / "preset" / "merge" / f"{pname}.json"
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
            except Exception as exc:
                messagebox.showerror("Preset", f"Load failed: {exc}")
                return
            self.alpha_var.set(data.get("alpha", 0.5))
            self.cosine_var.set(data.get("cosine_threshold", 0.4))
            self.auto_var.set(data.get("auto_correction", True))
            self.dry_run_var.set(data.get("dry_run", True))
            self.freeze_input_var.set(data.get("freeze_bias_input", False))
            self.freeze_middle_var.set(data.get("freeze_bias_middle", False))
            self.freeze_output_var.set(data.get("freeze_bias_output", False))
            new_mode = data.get("layer_display_mode", "Matrix")
            self.layer_mode_var.set(new_mode)
            # Layer display が変化した場合も含め常に再構築し、スケール値を反映する
            self.rebuild_parameter_controls()
            scales = data.get("parameter_scales", {})
            for k, v in scales.items():
                if k in self.parameter_vars:
                    self.parameter_vars[k].set(v)
            self.log(f"[Preset] Loaded: {src.name}")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_action_log([f"{now} [Preset Load] {src.name}", ""])

        def _delete_preset() -> None:
            sel = preset_listbox.curselection()
            if not sel:
                return
            pname = preset_listbox.get(sel[0])
            src = self.paths.root / "preset" / "merge" / f"{pname}.json"
            if messagebox.askyesno("Preset", f"Delete {pname}?"):
                src.unlink(missing_ok=True)
                _refresh_list()
                self.log(f"[Preset] Deleted: {src.name}")
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._write_action_log([f"{now} [Preset Delete] {src.name}", ""])

        def _export_preset() -> None:
            sel = preset_listbox.curselection()
            if not sel:
                messagebox.showerror("Preset", "Select a preset to export.")
                return
            pname = preset_listbox.get(sel[0])
            src = self.paths.root / "preset" / "merge" / f"{pname}.json"
            dest = filedialog.asksaveasfilename(
                initialdir=str(self.paths.root / "preset" / "merge"),
                initialfile=f"{pname}.json",
                filetypes=(("JSON", "*.json"),),
            )
            if dest:
                import shutil
                shutil.copy2(src, dest)
                self.log(f"[Preset] Exported: {dest}")
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._write_action_log([f"{now} [Preset Export] {dest}", ""])

        def _import_preset() -> None:
            src = filedialog.askopenfilename(
                initialdir=str(self.paths.root / "preset" / "merge"),
                filetypes=(("JSON", "*.json"),),
            )
            if not src:
                return
            pname = Path(src).stem
            _preset_dir = self.paths.root / "preset" / "merge"
            _preset_dir.mkdir(parents=True, exist_ok=True)
            dest = _preset_dir / f"{pname}.json"
            import shutil
            shutil.copy2(src, dest)
            _refresh_list()
            self.log(f"[Preset] Imported: {dest.name}")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_action_log([f"{now} [Preset Import] {dest.name}", ""])

        ttk.Button(btn_row, text="Save",    command=_save_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load",    command=_load_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Delete",  command=_delete_preset).pack(side=tk.LEFT, padx=2)

        io_row = ttk.Frame(tab)
        io_row.pack(fill=tk.X)
        ttk.Button(io_row, text="Export",  command=_export_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_row, text="Import",  command=_import_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_row, text="Refresh", command=_refresh_list).pack(side=tk.LEFT, padx=2)

        _refresh_list()

    # ─── タブ3: レイヤー分析 UI ──────────────────────────────────────────
    def _build_analysis_tab(self, parent: ttk.Frame) -> None:
        """タブ3「レイヤー分析」の全UIを構築する。"""

                # ── 上部コントロールパネル ────────────────────────────────────────
        ctrl = ttk.LabelFrame(parent, text="Analysis Controls")
        ctrl.pack(fill=tk.X, padx=0, pady=(0, 6))

        # row0: Device ラベル + モード表示 + Rescan + アンロード（左寄せ）
        _mode_text = "GPU Mode" if self._mode == "cuda" else "CPU Mode"
        _mode_fg = "#22C55E" if self._mode == "cuda" else "#64748B"
        ttk.Label(ctrl, text="Device").grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
        ttk.Label(ctrl, text=_mode_text, foreground=_mode_fg, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, padx=4, pady=6, sticky=tk.W)
        ttk.Button(ctrl, text="Rescan folders", command=self.refresh_files).grid(row=0, column=2, padx=8, pady=6, sticky=tk.W)
        ttk.Button(ctrl, text="モデルをアンロード", command=self.unload_models).grid(row=0, column=3, padx=8, pady=6, sticky=tk.W)

        # row1: 対象種別
        ttk.Label(ctrl, text="対象種別").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
        type_frame = ttk.Frame(ctrl)
        type_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W)
        ttk.Radiobutton(
            type_frame, text="本体モデル",
            variable=self.analysis_target_type_var, value="model",
            command=self._refresh_analysis_target_combo,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            type_frame, text="LoRA",
            variable=self.analysis_target_type_var, value="lora",
            command=self._refresh_analysis_target_combo,
        ).pack(side=tk.LEFT, padx=8)

        # row2: ターゲット
        ttk.Label(ctrl, text="ターゲット").grid(row=2, column=0, padx=8, pady=6, sticky=tk.W)
        self.analysis_target_combo = ttk.Combobox(
            ctrl, textvariable=self.analysis_target_var, state="readonly", width=160
        )
        self.analysis_target_combo.grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=4)

        # row3: 分析手法 + 表示レイヤー（同行左寄せ）
        method_frame = ttk.Frame(ctrl)
        method_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=8, pady=6)
        ttk.Label(method_frame, text="分析手法").pack(side=tk.LEFT)
        self.analysis_method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.analysis_method_var,
            values=list(ANALYSIS_METHODS),
            state="readonly",
            width=24,
        )
        self.analysis_method_combo.pack(side=tk.LEFT, padx=(4, 16))
        self.analysis_method_combo.bind("<<ComboboxSelected>>", self._on_analysis_options_changed)
        ttk.Label(method_frame, text="表示レイヤー").pack(side=tk.LEFT)
        self.analysis_layer_combo = ttk.Combobox(
            method_frame,
            textvariable=self.analysis_layer_mode_var,
            values=list(LAYER_DISPLAY_MODES),
            state="readonly",
            width=16,
        )
        self.analysis_layer_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.analysis_layer_combo.bind("<<ComboboxSelected>>", self._on_analysis_options_changed)
        ttk.Checkbutton(
            method_frame,
            text="Key correction (anima-base-v1.0)",
            variable=self.analysis_key_correction_var,
        ).pack(side=tk.LEFT, padx=(16, 0))

        # row4: 進捗ラベル + Stop/RUN
        self.analysis_status_var = tk.StringVar(value="分析手法と表示レイヤーを選択後、RUNを押してください。")
        ttk.Label(ctrl, textvariable=self.analysis_status_var, foreground="#334155").grid(
            row=4, column=0, columnspan=2, padx=8, pady=(4, 8), sticky=tk.W
        )
        btn_frame_analysis = ttk.Frame(ctrl)
        btn_frame_analysis.grid(row=4, column=2, columnspan=4, sticky=tk.E, padx=8, pady=(4, 8))
        self.analysis_stop_btn = ttk.Button(
            btn_frame_analysis, text="■  Stop",
            command=self._stop_analysis,
        )
        self.analysis_stop_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.analysis_run_btn = ttk.Button(
            btn_frame_analysis, text="▶  Run Analysis", style="Run.TButton",
            command=self.start_analysis,
        )
        self.analysis_run_btn.pack(side=tk.LEFT)


        # ── 結果表示エリア ────────────────────────────────────────────────
        result_outer = ttk.LabelFrame(parent, text="Analysis Result")
        result_outer.pack(fill=tk.BOTH, expand=True)
        result_outer.columnconfigure(0, weight=1)
        result_outer.rowconfigure(0, weight=1)

        # 左: グループサマリー表
        summary_frame = ttk.LabelFrame(result_outer, text="Group Summary")
        summary_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(4, 2), pady=4)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        self.analysis_summary_tree = ttk.Treeview(
            summary_frame, show="headings", selectmode="browse"
        )
        sum_scroll_y = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.analysis_summary_tree.yview)
        sum_scroll_x = ttk.Scrollbar(summary_frame, orient=tk.HORIZONTAL, command=self.analysis_summary_tree.xview)
        self.analysis_summary_tree.configure(yscrollcommand=sum_scroll_y.set, xscrollcommand=sum_scroll_x.set)
        sum_scroll_y.grid(row=0, column=1, sticky=tk.NS)
        sum_scroll_x.grid(row=1, column=0, sticky=tk.EW)
        self.analysis_summary_tree.grid(row=0, column=0, sticky=tk.NSEW)

        result_outer.columnconfigure(0, weight=3)
        result_outer.columnconfigure(1, weight=2)

        # 右: 自動テキストレポート + 詳細レイヤーリスト
        right_frame = ttk.Frame(result_outer)
        right_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(2, 4), pady=4)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        ttk.Label(right_frame, text="自動レポート", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 2)
        )
        self.analysis_report_text = tk.Text(
            right_frame, height=7, wrap=tk.WORD,
            font=("TkFixedFont", 9), relief=tk.SUNKEN, bd=1,
        )
        rep_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.analysis_report_text.yview)
        self.analysis_report_text.configure(yscrollcommand=rep_scroll.set)
        rep_scroll.grid(row=1, column=1, sticky=tk.NS)
        self.analysis_report_text.grid(row=1, column=0, sticky=tk.NSEW)

        # 詳細レイヤーリスト（下部）
        detail_frame = ttk.LabelFrame(parent, text="Layer Detail")
        detail_frame.pack(fill=tk.X, pady=(4, 0))
        detail_frame.columnconfigure(0, weight=1)

        self.analysis_detail_tree = ttk.Treeview(
            detail_frame, show="headings", height=6, selectmode="browse"
        )
        det_scroll_y = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.analysis_detail_tree.yview)
        det_scroll_x = ttk.Scrollbar(detail_frame, orient=tk.HORIZONTAL, command=self.analysis_detail_tree.xview)
        self.analysis_detail_tree.configure(yscrollcommand=det_scroll_y.set, xscrollcommand=det_scroll_x.set)
        det_scroll_y.grid(row=0, column=1, sticky=tk.NS)
        det_scroll_x.grid(row=1, column=0, sticky=tk.EW)
        self.analysis_detail_tree.grid(row=0, column=0, sticky=tk.EW)

        # ログ保存パス表示
        self.analysis_log_path_var = tk.StringVar(value="")
        ttk.Label(detail_frame, textvariable=self.analysis_log_path_var, foreground="#1D4ED8").grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=4, pady=2
        )

    # ─── 分析タブ: ヘルパーメソッド ─────────────────────────────────────
    def _refresh_analysis_target_combo(self) -> None:
        """対象種別に応じてターゲットコンボの選択肢を切り替える。"""
        if not hasattr(self, "analysis_target_combo"):
            return
        model_type = self.analysis_target_type_var.get()
        if model_type == "lora":
            choices = self.lora_choices
        else:
            choices = self.model_choices
        self.analysis_target_combo["values"] = choices
        if choices and not self.analysis_target_var.get() in choices:
            self.analysis_target_var.set(choices[0])

    def _on_analysis_options_changed(self, _event: tk.Event | None = None) -> None:
        method = self.analysis_method_var.get()
        mode = self.analysis_layer_mode_var.get()
        self.analysis_status_var.set(f"手法: {method}  /  表示レイヤー: {mode}  → RUNで実行")

    def _get_summary_columns(self, method: str) -> tuple[list[str], list[str]]:
        """メソッドに応じてサマリー表のカラム定義を返す。(column_ids, headers)"""
        base = ["group", "layers"]
        if method == "Feature Map":
            return (
                base + ["mean", "var", "complexity"],
                ["グループ", "層数", "平均値", "分散", "空間複雑度"],
            )
        if method == "Statistical":
            return (
                base + ["mean_l2", "max_l2", "min_l2", "mean_mean", "mean_var"],
                ["グループ", "層数", "平均L2", "最大L2", "最小L2", "平均値", "分散"],
            )
        if method == "SVD Rank":
            return (
                base + ["mean_rank", "mean_decay", "mean_cumvar"],
                ["グループ", "層数", "平均有効ランク", "平均減衰率", "平均累積寄与率"],
            )
        if method == "Attention Map":
            return (
                base + ["attn_layers", "mean_weight", "head_var"],
                ["グループ", "層数", "Attn層数", "平均重み", "ヘッド間分散"],
            )
        return base, ["グループ", "層数"]

    def _get_detail_columns(self, method: str) -> tuple[list[str], list[str]]:
        """メソッドに応じて詳細テーブルのカラム定義を返す。"""
        base = ["key", "group", "shape"]
        if method == "Feature Map":
            return (
                base + ["feat_mean", "feat_var", "complexity"],
                ["キー", "グループ", "形状", "平均値", "分散", "空間複雑度"],
            )
        if method == "Statistical":
            return (
                base + ["l2", "mean", "var"],
                ["キー", "グループ", "形状", "L2ノルム", "平均値", "分散"],
            )
        if method == "SVD Rank":
            return (
                base + ["eff_rank", "decay", "cumvar"],
                ["キー", "グループ", "形状", "有効ランク", "減衰率", "累積寄与率"],
            )
        if method == "Attention Map":
            return (
                base + ["is_attn", "mean_w", "head_var"],
                ["キー", "グループ", "形状", "Attn層", "平均重み", "ヘッド間分散"],
            )
        return base, ["キー", "グループ", "形状"]

    def _populate_analysis_result(self, report: AnalysisReport) -> None:
        """分析完了後にUIへ結果を反映する（メインスレッドから呼び出す）。"""
        method = report.method

        # ── 自動レポートテキスト ──────────────────────────────────────────
        self.analysis_report_text.config(state=tk.NORMAL)
        self.analysis_report_text.delete("1.0", tk.END)
        self.analysis_report_text.insert(tk.END, "\n".join(report.auto_report_lines))
        if report.warnings:
            self.analysis_report_text.insert(tk.END, f"\n\n[警告 {len(report.warnings)}件]\n")
            for w in report.warnings[:10]:
                self.analysis_report_text.insert(tk.END, f"  {w}\n")
            if len(report.warnings) > 10:
                self.analysis_report_text.insert(tk.END, f"  ... 他 {len(report.warnings)-10} 件\n")
        self.analysis_report_text.config(state=tk.DISABLED)

        # ── グループサマリー Treeview ─────────────────────────────────────
        tree = self.analysis_summary_tree
        tree.delete(*tree.get_children())
        col_ids, col_headers = self._get_summary_columns(method)
        tree["columns"] = col_ids
        for cid, header in zip(col_ids, col_headers):
            tree.heading(cid, text=header)
            tree.column(cid, width=110, anchor=tk.CENTER, stretch=True)
        tree.column(col_ids[0], width=200, anchor=tk.W, stretch=True)

        # aggregated は records から再構築（report に持つ）
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for r in report.records:
            groups[r.group].append(r)

        for group, recs in sorted(groups.items()):
            n = len(recs)
            if method == "Feature Map":
                vals = [
                    group, str(n),
                    f"{sum(r.feat_mean for r in recs)/n:.4f}",
                    f"{sum(r.feat_var for r in recs)/n:.4f}",
                    f"{sum(r.feat_complexity for r in recs)/n:.4f}",
                ]
            elif method == "Statistical":
                l2s = [r.stat_l2 for r in recs]
                vals = [
                    group, str(n),
                    f"{sum(l2s)/n:.4f}",
                    f"{max(l2s):.4f}",
                    f"{min(l2s):.4f}",
                    f"{sum(r.stat_mean for r in recs)/n:.4f}",
                    f"{sum(r.stat_var for r in recs)/n:.4f}",
                ]
            elif method == "SVD Rank":
                vals = [
                    group, str(n),
                    f"{sum(r.svd_effective_rank for r in recs)/n:.1f}",
                    f"{sum(r.svd_decay_rate for r in recs)/n:.4f}",
                    f"{sum(r.svd_cumvar_at_threshold for r in recs)/n:.4f}",
                ]
            elif method == "Attention Map":
                attn = [r for r in recs if r.is_attention_layer]
                na = len(attn)
                mw = f"{sum(r.attn_mean_weight for r in attn)/na:.4f}" if na else "-"
                hv = f"{sum(r.attn_head_variance for r in attn)/na:.4f}" if na else "-"
                vals = [group, str(n), str(na), mw, hv]
            else:
                vals = [group, str(n)]
            tree.insert("", tk.END, values=vals)

        # ── 詳細レイヤーリスト Treeview ──────────────────────────────────
        dtree = self.analysis_detail_tree
        dtree.delete(*dtree.get_children())
        dcol_ids, dcol_headers = self._get_detail_columns(method)
        dtree["columns"] = dcol_ids
        for cid, header in zip(dcol_ids, dcol_headers):
            dtree.heading(cid, text=header)
            dtree.column(cid, width=130, anchor=tk.CENTER, stretch=True)
        dtree.column(dcol_ids[0], width=280, anchor=tk.W, stretch=True)

        # タグ色付け設定
        dtree.tag_configure("outlier", foreground="#DC2626")      # 赤: 上位10%(高L2)
        dtree.tag_configure("low_stat", foreground="#2563EB")     # 青: 下位10%(低L2)
        dtree.tag_configure("important", foreground="#DC2626")    # 緑: 重要層
        dtree.tag_configure("unimportant", foreground="#2563EB")  # 紫: 非重要層

        # Statistical: 上位10%=赤, 下位10%=青
        outlier_keys: set[str] = set()
        low_stat_keys: set[str] = set()
        if method == "Statistical" and report.records:
            l2s_all = sorted(r.stat_l2 for r in report.records)
            n_all = len(l2s_all)
            if n_all >= 10:
                hi_threshold = l2s_all[int(n_all * 0.90)]
                lo_threshold = l2s_all[int(n_all * 0.10)]
                outlier_keys = {r.key for r in report.records if r.stat_l2 >= hi_threshold}
                low_stat_keys = {r.key for r in report.records if r.stat_l2 <= lo_threshold}

        # 重要層 / 非重要層の判定 (report から取得)
        important_keys: set[str] = set(getattr(report, 'important_layer_keys', []))
        unimportant_keys: set[str] = set(getattr(report, 'unimportant_layer_keys', []))

        for r in report.records:
            shape_str = "x".join(str(d) for d in r.shape)
            if method == "Feature Map":
                vals = [r.key, r.group, shape_str,
                        f"{r.feat_mean:.4f}", f"{r.feat_var:.4f}", f"{r.feat_complexity:.4f}"]
            elif method == "Statistical":
                vals = [r.key, r.group, shape_str,
                        f"{r.stat_l2:.4f}", f"{r.stat_mean:.4f}", f"{r.stat_var:.4f}"]
            elif method == "SVD Rank":
                vals = [r.key, r.group, shape_str,
                        str(r.svd_effective_rank), f"{r.svd_decay_rate:.4f}",
                        f"{r.svd_cumvar_at_threshold:.4f}"]
            elif method == "Attention Map":
                vals = [r.key, r.group, shape_str,
                        "Yes" if r.is_attention_layer else "-",
                        f"{r.attn_mean_weight:.4f}" if r.is_attention_layer else "-",
                        f"{r.attn_head_variance:.4f}" if r.is_attention_layer else "-"]
            else:
                vals = [r.key, r.group, shape_str]
            if r.key in outlier_keys:
                tag = "outlier"
            elif r.key in low_stat_keys:
                tag = "low_stat"
            elif r.key in important_keys:
                tag = "important"
            elif r.key in unimportant_keys:
                tag = "unimportant"
            else:
                tag = ""
            dtree.insert("", tk.END, values=vals, tags=(tag,) if tag else ())

        # ログ保存パス表示
        if report.log_path:
            self.analysis_log_path_var.set(f"保存済み: {report.log_path}")
        self.analysis_status_var.set(
            f"完了: {len(report.records)} 層 / {report.method} / {report.layer_mode}"
            + (f"  [{len(report.warnings)}件の警告]" if report.warnings else "")
        )
        self.analysis_run_btn.config(state=tk.NORMAL)

    def _output_controls(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text="Output").grid(row=row, column=0, sticky=tk.W, pady=6)
        ttk.Entry(parent, textvariable=self.output_var).grid(row=row, column=1, sticky=tk.EW, padx=8)
        ttk.Button(parent, text="Browse", command=self.choose_output).grid(row=row, column=2, sticky=tk.E)

    # ─── スケールのスナップとクランプ ────────────────────────
    def _snap_scale(self, var: tk.DoubleVar) -> None:
        snapped = round(round(var.get() / 0.1) * 0.1, 10)
        var.set(snapped)

    def _clamp_var(self, var: tk.DoubleVar) -> None:
        try:
            val = float(var.get())
            var.set(max(0.0, min(1.0, val)))
        except (tk.TclError, ValueError):
            pass

    def _sync_device_var(self, resolved: str) -> None:
        """CUDAフォールバック時にGUI device_var をCPUへ同期する。"""
        if resolved != self.device_var.get():
            self.device_var.set(resolved)
            self.log("[Device] CUDAが利用不可のためCPUに切り替えました。")

    def _make_progress_cb(self, requested_device: str):
        """進捗コールバックを返す。フォールバックメッセージを検知してGUIに反映。"""
        _HINTS = ("CUDA is not available", "CUDA は利用できません",
                  "Falling back to CPU", "CPUにフォールバック")
        def _cb(msg: str) -> None:
            self.log_queue.put(msg)
            if requested_device.startswith("cuda") and any(h in msg for h in _HINTS):
                self.after(0, lambda: self._sync_device_var("cpu"))
        return _cb

    # ─── ファイル操作 ─────────────────────────────────────────
    def refresh_files(self) -> None:
        models = scan_models(self.paths.checkpoints)
        loras = scan_models(self.paths.lora)
        self.model_choices = [str(path) for path in models]
        self.lora_choices = [str(path) for path in loras]
        for combo in (self.base_combo, self.secondary_combo, self.lora_base_combo, self.extract_base_combo, self.extract_target_combo):
            combo["values"] = self.model_choices
        for combo in (self.lora_combo, self.lora_merge_base_combo, self.secondary_lora_combo):
            combo["values"] = self.lora_choices
        if self.model_choices and not self.base_model_var.get():
            self.base_model_var.set(self.model_choices[0])
        if len(self.model_choices) > 1 and not self.secondary_model_var.get():
            self.secondary_model_var.set(self.model_choices[1])
        if self.lora_choices and not self.lora_var.get():
            self.lora_var.set(self.lora_choices[0])
        if len(self.lora_choices) > 1 and not self.secondary_lora_var.get():
            self.secondary_lora_var.set(self.lora_choices[1])

        # 分析タブ: ターゲット種別に応じてプルダウンを更新
        self._refresh_analysis_target_combo()

        self.log(f"Scanned: {len(models)} checkpoints, {len(loras)} LoRA files")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_action_log([
            f"{now} [Rescan Folders]",
            f"  Checkpoints: {len(models)}",
            f"  LoRA files : {len(loras)}",
            "",
        ])

    def unload_models(self) -> None:
        """メモリ上のモデルをアンロードし VRAM/RAM を解放する。"""
        # LoRA学習プロセスが動いている場合は先に終了させる
        if hasattr(self, "_lora_train_state") and self._lora_train_state is not None:
            proc = getattr(self._lora_train_state, "_proc", None)
            if proc is not None and proc.poll() is None:
                import os, signal
                try:
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                except Exception:
                    proc.terminate()
                proc.wait(timeout=10)
                self.log("[Unload] LoRA学習プロセスを終了しました。")
        # LECO学習プロセスが動いている場合は先に終了させる
        if hasattr(self, "_leco_train_state") and self._leco_train_state is not None:
            proc = getattr(self._leco_train_state, "_proc", None)
            if proc is not None and proc.poll() is None:
                import os, signal
                try:
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                except Exception:
                    proc.terminate()
                proc.wait(timeout=10)
                self.log("[Unload] LECO学習プロセスを終了しました。")
        try:
            import gc
            import torch
            torch.cuda.empty_cache()
            gc.collect()
            self._loaded_model_names.clear()
            self.log("Unloaded: VRAM/RAM cache cleared (torch.cuda.empty_cache + gc.collect)")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_action_log([f"{now} [Unload Models] VRAM/RAM cache cleared", ""])
        except ImportError:
            import gc
            gc.collect()
            self._loaded_model_names.clear()
            self.log("Unloaded: RAM cache cleared (gc.collect). PyTorch not available.")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_action_log([f"{now} [Unload Models] RAM cache cleared (no PyTorch)", ""])

    def choose_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            initialdir=str(self.paths.checkpoints),
            initialfile=Path(self.output_var.get()).name,
            filetypes=(("Safetensors", "*.safetensors"), ("Checkpoint", "*.ckpt"), ("Binary", "*.bin")),
        )
        if filename:
            self.output_var.set(filename)

    def load_base_structure(self) -> None:
        base = self.base_model_var.get()
        if not base:
            messagebox.showerror("Layer load error", "Select a base model first.")
            return
        self._tw("status_var").set("Loading base structure...")

        def worker() -> None:
            try:
                rows = [
                    (name, shape)
                    for name, shape in list_state_dict_layers(Path(base), self.device_var.get())
                    if is_merge_target(name)
                ]
                self.after(0, lambda rows=rows: self._set_loaded_structure(rows))
            except Exception as exc:
                message = str(exc)
                self.after(0, lambda message=message: messagebox.showerror("Layer load error", message))
                self.after(0, lambda: self._tw("status_var").set("Layer load failed"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_loaded_structure(self, rows: list[tuple[str, str]]) -> None:
        self.loaded_layers = rows
        self._tw("status_var").set(f"{len(rows)} merge-target tensor layers loaded.")
        self.rebuild_parameter_controls()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_action_log([
            f"{now} [Load Base Structure]",
            f"  Layers: {len(rows)}",
            "",
        ])

    # ─── レイヤーコントロール（3列グリッド）────────────────────
    def rebuild_parameter_controls(self) -> None:
        if self._active_tab_type not in self._tab_widgets:
            return

        inner = self._tw("inner")
        canvas = self._tw("canvas")

        for child in inner.winfo_children():
            child.destroy()

        mode = self.layer_mode_var.get()
        group_names = self._group_names_for_mode(mode)
        old_values = {name: var.get() for name, var in self.parameter_vars.items()}
        self.parameter_vars = {}

        if not group_names:
            ttk.Label(
                inner,
                text="Load base structure to build controls for this display mode.",
            ).grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
            return

        cols = LAYER_COLUMNS

        for idx, name in enumerate(group_names):
            value = old_values.get(name, 1.0)
            var = tk.DoubleVar(value=value)
            self.parameter_vars[name] = var

            grid_row = idx // cols
            base_col = (idx % cols) * 3

            ttk.Label(inner, text=name, width=20).grid(
                row=grid_row, column=base_col, sticky=tk.W, padx=(8, 2), pady=3
            )
            scale = ttk.Scale(inner, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=grid_row, column=base_col + 1, sticky=tk.EW, padx=2, pady=3)
            scale.bind("<ButtonRelease-1>", lambda e, v=var: self._snap_scale(v))

            entry = ttk.Entry(inner, textvariable=var, width=6)
            entry.grid(row=grid_row, column=base_col + 2, sticky=tk.W, padx=(2, 12), pady=3)
            entry.bind("<FocusOut>", lambda e, v=var: self._clamp_var(v))

        for c in range(cols):
            inner.columnconfigure(c * 3 + 1, weight=1)

        canvas.configure(scrollregion=canvas.bbox("all"))
        self._tw("status_var").set(self._status_text(mode, len(group_names)))

    def _group_names_for_mode(self, mode: str) -> list[str]:
        if mode == "Matrix":
            return [f"{block}_{component}" for block in MATRIX_BLOCKS for component in MATRIX_COMPONENTS]
        if mode == "Component":
            if not self.loaded_layers:
                return []
            attention_groups = sorted(
                {
                    adjustment_group(name, mode)
                    for name, _shape in self.loaded_layers
                    if adjustment_group(name, mode).endswith("_Attention")
                },
                key=self._natural_key,
            )
            present_components = [
                component
                for component in COMPONENT_GROUPS
                if any(adjustment_group(name, mode) == component for name, _shape in self.loaded_layers)
            ]
            return [*attention_groups, *present_components]
        if not self.loaded_layers:
            return []
        groups = sorted({adjustment_group(name, mode) for name, _shape in self.loaded_layers}, key=self._natural_key)
        return groups

    def _status_text(self, mode: str, count: int) -> str:
        if mode == "Matrix":
            return f"Matrix mode: {count} block x component controls."
        if mode == "Transformer":
            return f"Transformer mode: {count} parent block controls."
        return f"Component mode: {count} component controls."

    def _natural_key(self, text: str) -> list[object]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]

    # ─── オプション・マージ実行 ───────────────────────────────
    def options(self) -> MergeOptions:
        return MergeOptions(
            alpha=float(self.alpha_var.get()),
            layer_display_mode=self.layer_mode_var.get(),
            parameter_scales={name: float(var.get()) for name, var in self.parameter_vars.items()},
            cosine_threshold=float(self.cosine_var.get()),
            auto_correction=bool(self.auto_var.get()),
            freeze_bias_input=bool(self.freeze_input_var.get()),
            freeze_bias_middle=bool(self.freeze_middle_var.get()),
            freeze_bias_output=bool(self.freeze_output_var.get()),
            dry_run=bool(self.dry_run_var.get()),
            output_name=Path(self.output_var.get()).name,
        )

    def start_model_merge(self) -> None:
        self._merge_stop_event.clear()
        _dev = self.device_var.get()
        self._write_merge_log("[本体マージ開始]", self.base_model_var.get(), self.secondary_model_var.get())
        self._run_background(
            lambda: merge_models(
                Path(self.base_model_var.get()),
                Path(self.secondary_model_var.get()),
                Path(self.output_var.get()),
                self.options(),
                _dev,
                self._make_progress_cb(_dev),
            )
        )

    def start_lora_fuse(self) -> None:
        self._merge_stop_event.clear()
        _dev = self.device_var.get()
        self._write_merge_log("[LoRAフューズ開始]", self.base_model_var.get(), self.lora_var.get())
        self._run_background(
            lambda: fuse_lora_into_model(
                Path(self.base_model_var.get()),
                Path(self.lora_var.get()),
                Path(self.output_var.get()),
                self.options(),
                _dev,
                self._make_progress_cb(_dev),
            )
        )

    def start_lora_merge(self) -> None:
        self._merge_stop_event.clear()
        _dev = self.device_var.get()
        self._write_merge_log("[LoRAマージ開始]", self.lora_var.get(), self.secondary_lora_var.get())
        self._run_background(
            lambda: merge_loras(
                Path(self.lora_var.get()),
                Path(self.secondary_lora_var.get()),
                Path(self.output_var.get()),
                self.options(),
                _dev,
                self._make_progress_cb(_dev),
            )
        )

    def start_lora_extract(self) -> None:
        self._merge_stop_event.clear()
        _dev = self.device_var.get()
        self._write_merge_log("[差分抽出開始]", self.base_model_var.get(), self.secondary_model_var.get())
        self._run_background(
            lambda: extract_lora_difference(
                Path(self.base_model_var.get()),
                Path(self.secondary_model_var.get()),
                Path(self.output_var.get()),
                self.options(),
                int(self.extract_rank_var.get()),
                _dev,
                self._make_progress_cb(_dev),
            )
        )

    def start_analysis(self) -> None:
        """分析タブの RUN ボタン処理。バックグラウンドスレッドで run_analysis() を実行する。"""
        self._analysis_stop_event.clear()
        target = self.analysis_target_var.get()
        if not target:
            messagebox.showerror("分析エラー", "分析対象ファイルを選択してください。")
            return
        method = self.analysis_method_var.get()
        layer_mode = self.analysis_layer_mode_var.get()
        device = self.device_var.get()

        self.analysis_run_btn.config(state=tk.DISABLED)
        self.analysis_status_var.set("分析実行中...")
        self.log(f"[Analysis] 開始: {Path(target).name} / {method} / {layer_mode}")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_action_log([
            f"{now} [Analysis 開始]",
            f"  Target : {target}",
            f"  Method : {method}",
            f"  Layer  : {layer_mode}",
            f"  Device : {device}",
            "",
        ])

        def worker() -> None:
            try:
                report = run_analysis(
                    model_path=Path(target),
                    method=method,
                    layer_mode=layer_mode,
                    log_dir=self.paths.log_analysis,  # log_analysis フォルダ
                    device=device,
                    progress=self._make_progress_cb(device),
                    key_correction=self.analysis_key_correction_var.get(),
                )
                self._analysis_result_report = report
                self.after(0, lambda r=report: self._populate_analysis_result(r))
            except DependencyError as exc:
                message = str(exc)
                self.log_queue.put(f"[Analysis] Dependency error: {message}")
                self.after(0, lambda m=message: messagebox.showerror("依存ライブラリエラー", m))
                self.after(0, lambda: self.analysis_run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.analysis_status_var.set("エラーが発生しました。"))
            except Exception as exc:
                message = str(exc)
                self.log_queue.put(f"[Analysis] Error: {message}")
                self.after(0, lambda m=message: messagebox.showerror("分析エラー", m))
                self.after(0, lambda: self.analysis_run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.analysis_status_var.set("エラーが発生しました。"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_background(self, task) -> None:
        def worker() -> None:
            try:
                report = task()
                self.log_queue.put(
                    f"Done: merged={report.merged_tensors}, skipped={report.skipped_tensors}, "
                    f"auto_corrected={report.auto_corrected_tensors}, output={report.output_path}"
                )
                for warning in report.warnings[:20]:
                    self.log_queue.put(f"Warning: {warning}")
            except DependencyError as exc:
                message = str(exc)
                self.log_queue.put(f"Dependency error: {message}")
                self.after(0, lambda message=message: messagebox.showerror("Dependency error", message))
            except Exception as exc:
                message = str(exc)
                self.log_queue.put(f"Error: {message}")
                self.after(0, lambda message=message: messagebox.showerror("Merge error", message))

        threading.Thread(target=worker, daemon=True).start()

    # ─── マージ起動ログ (日付別ファイル) ─────────────────────────
    # ─── 汎用ログ書き込みヘルパー ─────────────────────────────────────
    def _write_action_log(self, lines: list) -> None:
        """任意の操作ログを log フォルダの日付別 log_YYYYMMDD.txt に追記。"""
        try:
            date_str = datetime.date.today().strftime("%Y%m%d")
            log_dir = self.paths.log_analysis.parent  # log フォルダ
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"log_{date_str}.txt"
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as exc:
            self.log_queue.put(f"[LOG WARN] ログ書き込み失敗: {exc}")

    def _write_merge_log(self, label: str, model_a: str, model_b: str) -> None:
        """本体マージ・LoRAマージの起動時刻・設定を日付別 log_YYYYMMDD.txt に追記。"""
        try:
            date_str = datetime.date.today().strftime("%Y%m%d")
            log_dir = self.paths.log_analysis.parent  # log フォルダ
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"log_{date_str}.txt"
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            opts = self.options()
            lines = [
                f"{now} {label}",
                f"  Model A : {model_a}",
                f"  Model B : {model_b}",
                f"  Alpha   : {opts.alpha}",
                f"  Dry-run : {opts.dry_run}",
                f"  Device  : {self.device_var.get()}",
                f"  Output  : {opts.output_name}",
                "",
            ]
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except Exception as exc:
            self.log_queue.put(f"[LOG WARN] マージログ書き込み失敗: {exc}")

    def _stop_merge(self) -> None:
        """マージ系 Stop ボタン: stop イベントをセット (merge関数側が対応する場合に有効)。"""
        self._merge_stop_event.set()
        self.log_queue.put("[Stop] マージ停止要求を送信しました。現在処理中のテンソルが完了後に停止します。")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_action_log([f"{now} [Stop Merge] 停止要求送信", ""])

    def _stop_analysis(self) -> None:
        """分析 Stop ボタン。"""
        self._analysis_stop_event.set()
        self.log_queue.put("[Stop] 分析停止要求を送信しました。")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_action_log([f"{now} [Stop Analysis] 停止要求送信", ""])
        self.after(0, lambda: self.analysis_run_btn.config(state=tk.NORMAL))
        self.after(0, lambda: self.analysis_status_var.set("停止しました。"))

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _drain_logs(self) -> None:
        while True:
            try:
                self.log(self.log_queue.get_nowait())
            except queue.Empty:
                break
        self.after(100, self._drain_logs)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["cpu", "cuda"], default="cpu")
    args, _ = parser.parse_known_args()
    app = AnimaModelEditor(mode=args.mode)
    app.mainloop()
