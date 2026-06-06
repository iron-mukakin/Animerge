"""apply_fix_sample_filenames.py
サンプル生成 A/B をサブディレクトリからファイル名判別方式に変更するパッチ

変更内容:
  anima_train_leco.py:
    - import re をトップレベルに追加
    - _parse_sample_prompt_line の re.escape を修正
    - サブディレクトリ廃止: save_dir = save_base (log/sample_gen 直下)
    - ファイル名: step{N:06d}_a_s{seed}.png / step{N:06d}_b_s{seed}.png

  lora_train.py:
    - _sample_dir_a / _sample_dir_b 関数を削除
    - _build_sample_ab_panel: sample_dir + glob_pattern 引数に変更
    - ギャラリーを glob_pattern でA/B分離
    - _build_sample_tab_common: パターン指定に変更

  leco_train.py:
    - _build_leco_sample_tab_inline の _sdir をサブディレクトリから直下+パターンに変更

使い方:
    python apply_fix_sample_filenames.py \\
        --leco_script  sd-scripts/anima_train_leco.py \\
        --lora_train   app/lora_train.py \\
        --leco_train   app/leco_train.py
"""
from __future__ import annotations
import argparse
import ast
import sys
from pathlib import Path


def _adapt(src: str, ref: str) -> str:
    if "\r\n" in ref:
        return src.replace("\n", "\r\n")
    return src.replace("\r\n", "\n")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    old_a = _adapt(old, text)
    new_a = _adapt(new, text)
    count = text.count(old_a)
    if count == 0:
        raise RuntimeError(
            f"[{label}] 差分が見つかりません。\n--- 先頭120文字 ---\n{old[:120]}"
        )
    if count > 1:
        raise RuntimeError(f"[{label}] 差分が複数マッチ ({count}箇所)。")
    return text.replace(old_a, new_a, 1)


def _patch(path: Path, patches: list[tuple[str, str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new, label in patches:
        text = _replace_once(text, old, new, label)
        print(f"    OK: {label}")
    ast.parse(text)
    print("    構文チェック: OK")
    bak = path.with_suffix(".py.bak_sfn")
    bak.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"    バックアップ: {bak.name}  書き込み完了: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# anima_train_leco.py
# ══════════════════════════════════════════════════════════════════════════════

# P1: import re 追加
LECO_P1_OLD = """\
import argparse
import importlib
import math
import random
from pathlib import Path
from typing import Optional"""

LECO_P1_NEW = """\
import argparse
import importlib
import math
import random
import re
from pathlib import Path
from typing import Optional"""

# P2: サブディレクトリ廃止 + ファイル名にサフィックス付与
LECO_P2_OLD = """\
    # ── プロンプト行ごとに生成 ──────────────────────────────────────────────
    for line_idx, line in enumerate(prompt_lines):
        try:
            # サブディレクトリ: 行番号 0 → sample_a, 1 → sample_b, 2以降 → sample_c ...
            subdir_name = (
                ["sample_a", "sample_b"][line_idx]
                if line_idx < 2
                else f"sample_{chr(ord('a') + line_idx)}"
            )
            save_dir = save_base / subdir_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # プロンプト行のパース
            # 書式: <prompt text> [--n <neg>] --w <W> --h <H> --s <steps> --l <scale> --fs <flow_shift> --d <seed>
            prompt_text, gen_kwargs = _parse_sample_prompt_line(line)"""

LECO_P2_NEW = """\
    # ── プロンプト行ごとに生成 ──────────────────────────────────────────────
    for line_idx, line in enumerate(prompt_lines):
        try:
            # ファイル名サフィックス: 行番号 0 → _a_, 1 → _b_, 2以降 → _c_ ...
            suffix_char = chr(ord("a") + line_idx) if line_idx < 26 else str(line_idx)

            # プロンプト行のパース
            # 書式: <prompt text> [--n <neg>] --w <W> --h <H> --s <steps> --l <scale> --fs <flow_shift> --d <seed>
            prompt_text, gen_kwargs = _parse_sample_prompt_line(line)"""

# P3: save_dir 参照を save_base に変更 + ファイル名にサフィックス
LECO_P3_OLD = """\
            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(img_np)
            except ImportError:
                import array as _arr
                # Pillow がない場合は raw PPM 保存
                fname = save_dir / f"step{global_step:06d}_s{seed}.ppm"
                with open(fname, "wb") as f:
                    h_, w_ = img_np.shape[:2]
                    f.write(b"P6 " + str(w_).encode() + b" " + str(h_).encode() + b" 255 ".replace(b" ", bytes([10]), 2))
                    f.write(img_np.tobytes())
                logger.info(f"[SampleGen] saved (PPM): {fname}")
                continue

            fname = save_dir / f"step{global_step:06d}_s{seed}.png"
            pil_img.save(str(fname))
            logger.info(f"[SampleGen] saved: {fname}")"""

LECO_P3_NEW = """\
            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(img_np)
            except ImportError:
                # Pillow がない場合は raw PPM 保存
                fname = save_base / f"step{global_step:06d}_{suffix_char}_s{seed}.ppm"
                with open(fname, "wb") as f:
                    h_, w_ = img_np.shape[:2]
                    f.write(b"P6 " + str(w_).encode() + b" " + str(h_).encode() + b" 255 ".replace(b" ", bytes([10]), 2))
                    f.write(img_np.tobytes())
                logger.info(f"[SampleGen] saved (PPM): {fname}")
                continue

            fname = save_base / f"step{global_step:06d}_{suffix_char}_s{seed}.png"
            pil_img.save(str(fname))
            logger.info(f"[SampleGen] saved: {fname}")"""

# P4: _parse_sample_prompt_line の re.escape 修正
LECO_P4_OLD = """\
        pat = __import__("re").search(rf"{re.escape(flag)}[ \t]+([^ \t]+)", remainder)"""

LECO_P4_NEW = """\
        pat = re.search(rf"{re.escape(flag)}[ \t]+([^ \t]+)", remainder)"""

LECO_PATCHES = [
    (LECO_P1_OLD, LECO_P1_NEW, "P1: import re追加"),
    (LECO_P2_OLD, LECO_P2_NEW, "P2: サブディレクトリ廃止・サフィックス変数追加"),
    (LECO_P3_OLD, LECO_P3_NEW, "P3: save_dir→save_base・ファイル名サフィックス付与"),
    (LECO_P4_OLD, LECO_P4_NEW, "P4: _parse_sample_prompt_line re.escape修正"),
]


# ══════════════════════════════════════════════════════════════════════════════
# lora_train.py
# ══════════════════════════════════════════════════════════════════════════════

# P5: _sample_dir_a / _sample_dir_b を削除
LORA_P5_OLD = """\
def _sample_dir_a(s: _TrainState) -> Path:
    return _sample_dir(s) / "sample_a"


def _sample_dir_b(s: _TrainState) -> Path:
    return _sample_dir(s) / "sample_b"


"""

LORA_P5_NEW = """\
"""

# P6: _build_sample_ab_panel のシグネチャ + glob を glob_pattern 対応に変更
LORA_P6_OLD = """\
def _build_sample_ab_panel(
    parent: ttk.Frame,
    s,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    sample_dir: Path,
    label: str,
) -> None:
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    top = ttk.Frame(parent)
    top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    top.columnconfigure(1, weight=1)

    ttk.Checkbutton(
        top, text=f"サンプル{label}を有効にする", variable=enabled_var
    ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=(2, 4), pady=2)

    ttk.Label(top, text=f"出力先: ", foreground="#475569").grid(
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
        ep_lbl = ttk.Label(cell, text="step -", anchor=tk.CENTER)
        ep_lbl.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        cells.append((img_lbl, ep_lbl))

    def _refresh(schedule_next: bool = False) -> None:
        files = []
        if sample_dir.exists():
            files = sorted(
                sample_dir.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:10]"""

LORA_P6_NEW = """\
def _build_sample_ab_panel(
    parent: ttk.Frame,
    s,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    sample_dir: Path,
    glob_pattern: str,
    label: str,
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
        ep_lbl = ttk.Label(cell, text="step -", anchor=tk.CENTER)
        ep_lbl.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        cells.append((img_lbl, ep_lbl))

    def _refresh(schedule_next: bool = False) -> None:
        files = []
        if sample_dir.exists():
            files = sorted(
                sample_dir.glob(glob_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:10]"""

# P7: _build_sample_tab_common の dir_a/dir_b + パネル呼び出しを変更
LORA_P7_OLD = """\
    dir_a = _sample_dir_a(s) if not is_leco else (s.paths.root / "log" / "sample_gen" / "sample_a")
    dir_b = _sample_dir_b(s) if not is_leco else (s.paths.root / "log" / "sample_gen" / "sample_b")

    _build_sample_ab_panel(
        tab_a, s,
        enabled_var=s.sample_enabled,
        prompt_var=s.sample_prompt,
        neg_var=s.sample_negative_prompt,
        sample_dir=dir_a,
        label="A",
    )
    _build_sample_ab_panel(
        tab_b, s,
        enabled_var=s.sample_b_enabled,
        prompt_var=s.sample_b_prompt,
        neg_var=s.sample_b_negative_prompt,
        sample_dir=dir_b,
        label="B",
    )"""

LORA_P7_NEW = """\
    # LoRA: sd-scripts が {output_name}_{ts}_e{epoch:06d}_{idx:02d}.png を生成
    #   A = promptファイル1行目 → _00.png
    #   B = promptファイル2行目 → _01.png
    # LECO: _generate_samples_leco が step{N:06d}_a_s{seed}.png / _b_ を生成
    sample_dir = _sample_dir(s) if not is_leco else (s.paths.root / "log" / "sample_gen")
    pat_a = "*_a_*.png" if is_leco else "*_00.png"
    pat_b = "*_b_*.png" if is_leco else "*_01.png"

    _build_sample_ab_panel(
        tab_a, s,
        enabled_var=s.sample_enabled,
        prompt_var=s.sample_prompt,
        neg_var=s.sample_negative_prompt,
        sample_dir=sample_dir,
        glob_pattern=pat_a,
        label="A",
    )
    _build_sample_ab_panel(
        tab_b, s,
        enabled_var=s.sample_b_enabled,
        prompt_var=s.sample_b_prompt,
        neg_var=s.sample_b_negative_prompt,
        sample_dir=sample_dir,
        glob_pattern=pat_b,
        label="B",
    )"""

# P8: _clear_cache 内の sample_dir.iterdir() は変更なし（ルートを参照するため適切）
# キャッシュクリアはsample_dir直下の glob_pattern マッチファイルのみ削除に変更
LORA_P8_OLD = """\
        def _yes():
            dlg.destroy()
            if not sample_dir.exists():
                return
            deleted = errors = 0
            for f in sample_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink(); deleted += 1
                except Exception:
                    errors += 1
            msg = f"[サンプル{label}] {deleted}件削除。"
            if errors: msg += f" ({errors}件失敗)"
            s.log_fn(msg)
            _refresh(False)"""

LORA_P8_NEW = """\
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
            _refresh(False)"""

LORA_PATCHES = [
    (LORA_P5_OLD, LORA_P5_NEW, "P5: _sample_dir_a/_b 削除"),
    (LORA_P6_OLD, LORA_P6_NEW, "P6: _build_sample_ab_panel glob_pattern引数追加"),
    (LORA_P7_OLD, LORA_P7_NEW, "P7: _build_sample_tab_common パターン指定"),
    (LORA_P8_OLD, LORA_P8_NEW, "P8: _clear_cache glob_patternで対象限定"),
]


# ══════════════════════════════════════════════════════════════════════════════
# leco_train.py
# ══════════════════════════════════════════════════════════════════════════════

# P9: _build_leco_sample_tab_inline の _sdir をサブディレクトリから直下+パターンに変更
LECO_TRAIN_P9_OLD = """\
    def _ab_panel(tab, enabled_var, prompt_var, neg_var, label):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
        top.columnconfigure(1, weight=1)
        ttk.Checkbutton(top, text=f"サンプル{label}を有効にする",
                        variable=enabled_var).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, padx=2, pady=2)
        _sdir = s.paths.root / "log" / "sample_gen" / f"sample_{label.lower()}"
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
                _sdir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:10] if _sdir.exists() else []"""

LECO_TRAIN_P9_NEW = """\
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
        _glob_pat = f"*_{label.lower()}_*.png"
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
            )[:10] if _sdir.exists() else []"""

LECO_TRAIN_PATCHES = [
    (LECO_TRAIN_P9_OLD, LECO_TRAIN_P9_NEW, "P9: _ab_panel サブディレクトリ→直下+パターン"),
]


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leco_script", default=str(Path("sd-scripts") / "anima_train_leco.py"))
    ap.add_argument("--lora_train",  default=str(Path("app") / "lora_train.py"))
    ap.add_argument("--leco_train",  default=str(Path("app") / "leco_train.py"))
    args = ap.parse_args()

    targets = [
        (Path(args.leco_script), LECO_PATCHES,       "anima_train_leco.py"),
        (Path(args.lora_train),  LORA_PATCHES,        "lora_train.py"),
        (Path(args.leco_train),  LECO_TRAIN_PATCHES,  "leco_train.py"),
    ]

    all_ok = True
    for path, patches, name in targets:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        if not path.exists():
            print(f"  [SKIP] 見つかりません: {path}")
            all_ok = False
            continue
        try:
            _patch(path, patches)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            all_ok = False

    print("\n" + "="*60)
    print("  完了" if all_ok else "  一部失敗。上記エラーを確認してください。")


if __name__ == "__main__":
    main()
