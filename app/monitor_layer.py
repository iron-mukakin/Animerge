"""app/monitor_layer.py - LoRA layer LR monitor widget."""
from __future__ import annotations

import math
import queue
import re
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

try:
    from .i18n import gettext, load_language
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _app_dir = _Path(__file__).resolve().parent
    if str(_app_dir) not in _sys.path:
        _sys.path.insert(0, str(_app_dir))
    from i18n import gettext, load_language  # type: ignore[no-redef]

if TYPE_CHECKING:
    from .lora_train import _TrainState


_RE_LR = re.compile(r"\blr=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


class MonitorLayerGraph:
    """Display effective LR per layer group during layer training."""

    def __init__(
        self,
        parent: ttk.Frame,
        state: "_TrainState",
        group_names_for_mode: Callable[[str], list[str]],
    ) -> None:
        self._parent = parent
        self._state = state
        self._group_names_for_mode = group_names_for_mode
        self._last_lr = self._read_base_lr()
        self._last_signature: tuple | None = None

        self._build_ui(parent)
        parent.after(300, self._poll)

    def _build_ui(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        header.columnconfigure(1, weight=1)

        ttk.Label(
            header,
            text=gettext("monitor_layer_title"),
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky=tk.W)

        self._status_var = tk.StringVar(value=gettext("monitor_layer_disabled"))
        ttk.Label(header, textvariable=self._status_var, foreground="#475569").grid(
            row=0, column=1, sticky=tk.E, padx=(12, 0)
        )

        holder = ttk.Frame(parent)
        holder.grid(row=1, column=0, sticky=tk.NSEW)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            holder,
            highlightthickness=0,
            bg="#0F172A",
        )
        scroll = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scroll.set)
        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)

        self._canvas.bind("<Configure>", lambda _e: self._redraw(force=True))

    def _poll(self) -> None:
        updated = False
        try:
            for _ in range(500):
                try:
                    line = self._state._monitor_layer_queue.get_nowait()
                except queue.Empty:
                    break
                if self._parse_line(line):
                    updated = True
        except Exception:
            pass

        if updated or self._signature() != self._last_signature:
            self._redraw(force=True)

        self._parent.after(300, self._poll)

    def _parse_line(self, line: str) -> bool:
        m = _RE_LR.search(line)
        if not m:
            return False
        try:
            lr = float(m.group(1))
        except ValueError:
            return False
        if math.isfinite(lr) and lr >= 0.0:
            self._last_lr = lr
            return True
        return False

    def _signature(self) -> tuple:
        enabled = bool(self._state.layer_train_enabled.get())
        mode = self._state.layer_display_mode.get()
        rows = tuple(
            (name, round(self._safe_var_get(name), 8))
            for name in self._group_names_for_mode(mode)
        )
        return (enabled, mode, round(self._last_lr, 12), rows, self._canvas.winfo_width())

    def _redraw(self, force: bool = False) -> None:
        sig = self._signature()
        if not force and sig == self._last_signature:
            return
        self._last_signature = sig

        self._canvas.delete("all")

        enabled = bool(self._state.layer_train_enabled.get())
        mode = self._state.layer_display_mode.get()
        groups = self._group_names_for_mode(mode)

        if not enabled:
            self._status_var.set(gettext("monitor_layer_disabled"))
            self._draw_message(gettext("monitor_layer_enable_hint"))
            return

        if not groups:
            self._status_var.set(gettext("monitor_layer_no_target"))
            self._draw_message(gettext("monitor_layer_no_layer"))
            return

        base_lr = self._last_lr if self._last_lr > 0.0 else self._read_base_lr()
        rows = [(name, self._safe_var_get(name), base_lr * self._safe_var_get(name)) for name in groups]
        max_effective_lr = max((v for _, _, v in rows), default=0.0)
        max_scale = max(max_effective_lr, base_lr, 1e-12)

        width = max(self._canvas.winfo_width(), 640)
        label_x = 12
        ratio_x = 170
        bar_x = 240
        value_pad = 8
        right_pad = 140
        bar_max = max(80, width - bar_x - right_pad)
        row_h = 28
        top = 30

        self._canvas.create_text(label_x, 12, text=gettext("monitor_layer_col_layer"), anchor=tk.W, fill="#CBD5E1", font=("TkDefaultFont", 9, "bold"))
        self._canvas.create_text(ratio_x, 12, text=gettext("monitor_layer_col_ratio"), anchor=tk.W, fill="#CBD5E1", font=("TkDefaultFont", 9, "bold"))
        self._canvas.create_text(bar_x, 12, text=gettext("monitor_layer_col_effective_lr"), anchor=tk.W, fill="#CBD5E1", font=("TkDefaultFont", 9, "bold"))

        for idx, (name, ratio, effective_lr) in enumerate(rows):
            y = top + idx * row_h
            bar_w = max(1, int(bar_max * (effective_lr / max_scale))) if effective_lr > 0 else 1
            color = self._bar_color(ratio)

            self._canvas.create_text(label_x, y, text=name, anchor=tk.W, fill="#E2E8F0", font=("TkFixedFont", 9))
            self._canvas.create_text(ratio_x, y, text=f"{ratio:.4f}", anchor=tk.W, fill="#94A3B8", font=("TkFixedFont", 9))
            self._canvas.create_rectangle(bar_x, y - 8, bar_x + bar_max, y + 8, fill="#1E293B", outline="#334155")
            self._canvas.create_rectangle(bar_x, y - 8, bar_x + bar_w, y + 8, fill=color, outline=color)
            self._canvas.create_text(
                min(bar_x + bar_w + value_pad, width - 8),
                y,
                text=_fmt_lr(effective_lr),
                anchor=tk.W,
                fill="#F8FAFC",
                font=("TkFixedFont", 9, "bold"),
            )

        total_h = top + len(rows) * row_h + 16
        self._canvas.configure(scrollregion=(0, 0, width, total_h))
        self._status_var.set(
            gettext("monitor_layer_status", mode=mode, groups=len(rows), lr=_fmt_lr(base_lr))
        )

    def _draw_message(self, message: str) -> None:
        width = max(self._canvas.winfo_width(), 640)
        self._canvas.create_text(
            16,
            24,
            text=message,
            anchor=tk.NW,
            fill="#CBD5E1",
            font=("TkDefaultFont", 10),
        )
        self._canvas.configure(scrollregion=(0, 0, width, 120))

    def _safe_var_get(self, name: str) -> float:
        var = self._state.layer_parameter_vars.get(name)
        if var is None:
            return 1.0
        try:
            value = float(var.get())
        except (ValueError, tk.TclError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.0, value)

    def _read_base_lr(self) -> float:
        try:
            value = float(self._state.lr.get())
        except (ValueError, tk.TclError):
            return 0.0
        if math.isfinite(value) and value >= 0.0:
            return value
        return 0.0

    @staticmethod
    def _bar_color(ratio: float) -> str:
        if ratio <= 0.0:
            return "#64748B"
        if ratio < 0.5:
            return "#38BDF8"
        if ratio <= 1.0:
            return "#22C55E"
        return "#F97316"


def _fmt_lr(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.3e}"
