"""apply_fix_bug04_sample_seed.py
BUG-04 修正: サンプルAとBに同一 seed=42 が割り当てられる問題。

原因:
  _build_sample_prompt_line_for / _leco_build_prompt_line は
  SAMPLE_FIXED_SEED (=42) をプロンプト行数に関係なく全行に埋め込む。
  do_sample は seed ごとに torch.manual_seed(seed) を呼ぶため、
  A と B が同一の初期ノイズから生成され画像が類似する。

修正方針:
  _write_sample_prompt_file / _leco_write_sample_prompt_file で
  行インデックスを seed オフセットとして使う。
    A行 (idx=0): seed = SAMPLE_FIXED_SEED + 0 = 42
    B行 (idx=1): seed = SAMPLE_FIXED_SEED + 1 = 43

  _build_sample_prompt_line_for / _leco_build_prompt_line に
  seed 引数を追加し、呼び出し側でインデックスを渡す。

対象ファイル:
  app/lora_train.py
  app/leco_train.py
"""
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


def _patch(target: Path, old: str, new: str, label: str) -> bool:
    raw = target.read_bytes()
    text = _adapt(raw.decode("utf-8"))
    old_n = _adapt(old)
    new_n = _adapt(new)

    if old_n not in text:
        print(f"[ERROR] {label}: 置換対象が見つかりません")
        print(f"  先頭: {repr(old_n[:120])}")
        return False
    if text.count(old_n) != 1:
        print(f"[ERROR] {label}: 置換対象が複数箇所存在します")
        return False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = target.with_suffix(f".bak_{ts}")
    if not bak.exists():
        shutil.copy2(target, bak)
        print(f"[INFO] バックアップ: {bak}")

    new_text = text.replace(old_n, new_n, 1)
    if b"\r\n" in raw:
        new_text = new_text.replace("\n", "\r\n")
    target.write_text(new_text, encoding="utf-8", newline="")
    print(f"[OK] {label}: 適用完了")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# lora_train.py
# ──────────────────────────────────────────────────────────────────────────────
LORA_FILE = Path("app/lora_train.py")

LORA_OLD_BUILD = (
    'def _build_sample_prompt_line_for(\n'
    '    prompt: str, neg: str, s: _TrainState\n'
    ') -> str:\n'
    '    width      = max(64, int(s.sample_width.get()))\n'
    '    height     = max(64, int(s.sample_height.get()))\n'
    '    steps      = max(1,  int(s.sample_steps.get()))\n'
    '    scale      = float(s.sample_scale.get())\n'
    '    flow_shift = float(s.sample_flow_shift.get())\n'
    '    line = (\n'
    '        f"{prompt} --w {width} --h {height} --s {steps} "\n'
    '        f"--l {scale:g} --fs {flow_shift:g} --d {SAMPLE_FIXED_SEED}"\n'
    '    )\n'
    '    if neg:\n'
    '        line += f" --n {neg}"\n'
    '    return line'
)

LORA_NEW_BUILD = (
    'def _build_sample_prompt_line_for(\n'
    '    prompt: str, neg: str, s: _TrainState, seed: int = SAMPLE_FIXED_SEED\n'
    ') -> str:\n'
    '    width      = max(64, int(s.sample_width.get()))\n'
    '    height     = max(64, int(s.sample_height.get()))\n'
    '    steps      = max(1,  int(s.sample_steps.get()))\n'
    '    scale      = float(s.sample_scale.get())\n'
    '    flow_shift = float(s.sample_flow_shift.get())\n'
    '    line = (\n'
    '        f"{prompt} --w {width} --h {height} --s {steps} "\n'
    '        f"--l {scale:g} --fs {flow_shift:g} --d {seed}"\n'
    '    )\n'
    '    if neg:\n'
    '        line += f" --n {neg}"\n'
    '    return line'
)

LORA_OLD_WRITE = (
    'def _write_sample_prompt_file(s: _TrainState) -> Path:\n'
    '    lines = []\n'
    '    if s.sample_enabled.get() and s.sample_prompt.get().strip():\n'
    '        lines.append(_build_sample_prompt_line_for(\n'
    '            s.sample_prompt.get().strip(),\n'
    '            s.sample_negative_prompt.get().strip(),\n'
    '            s,\n'
    '        ))\n'
    '    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():\n'
    '        lines.append(_build_sample_prompt_line_for(\n'
    '            s.sample_b_prompt.get().strip(),\n'
    '            s.sample_b_negative_prompt.get().strip(),\n'
    '            s,\n'
    '        ))\n'
    '    path = _sample_prompt_path(s)\n'
    '    path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")\n'
    '    return path'
)

LORA_NEW_WRITE = (
    'def _write_sample_prompt_file(s: _TrainState) -> Path:\n'
    '    lines = []\n'
    '    if s.sample_enabled.get() and s.sample_prompt.get().strip():\n'
    '        lines.append(_build_sample_prompt_line_for(\n'
    '            s.sample_prompt.get().strip(),\n'
    '            s.sample_negative_prompt.get().strip(),\n'
    '            s,\n'
    '            seed=SAMPLE_FIXED_SEED,\n'
    '        ))\n'
    '    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():\n'
    '        lines.append(_build_sample_prompt_line_for(\n'
    '            s.sample_b_prompt.get().strip(),\n'
    '            s.sample_b_negative_prompt.get().strip(),\n'
    '            s,\n'
    '            seed=SAMPLE_FIXED_SEED + 1,\n'
    '        ))\n'
    '    path = _sample_prompt_path(s)\n'
    '    path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")\n'
    '    return path'
)

# ──────────────────────────────────────────────────────────────────────────────
# leco_train.py
# ──────────────────────────────────────────────────────────────────────────────
LECO_FILE = Path("app/leco_train.py")

LECO_OLD_BUILD = (
    'def _leco_build_prompt_line(\n'
    '    prompt: str, neg: str, s: "_LecoTrainState"\n'
    ') -> str:\n'
    '    width      = max(64, int(s.sample_width.get()))\n'
    '    height     = max(64, int(s.sample_height.get()))\n'
    '    steps      = max(1,  int(s.sample_steps.get()))\n'
    '    scale      = float(s.sample_scale.get())\n'
    '    flow_shift = float(s.sample_flow_shift.get())\n'
    '    line = (\n'
    '        f"{prompt} --w {width} --h {height} --s {steps} "\n'
    '        f"--l {scale:g} --fs {flow_shift:g} --d {SAMPLE_FIXED_SEED}"\n'
    '    )\n'
    '    if neg:\n'
    '        line += f" --n {neg}"\n'
    '    return line\n'
    '\n'
    '\n'
    'SAMPLE_FIXED_SEED = 42'
)

LECO_NEW_BUILD = (
    'def _leco_build_prompt_line(\n'
    '    prompt: str, neg: str, s: "_LecoTrainState", seed: int = SAMPLE_FIXED_SEED\n'
    ') -> str:\n'
    '    width      = max(64, int(s.sample_width.get()))\n'
    '    height     = max(64, int(s.sample_height.get()))\n'
    '    steps      = max(1,  int(s.sample_steps.get()))\n'
    '    scale      = float(s.sample_scale.get())\n'
    '    flow_shift = float(s.sample_flow_shift.get())\n'
    '    line = (\n'
    '        f"{prompt} --w {width} --h {height} --s {steps} "\n'
    '        f"--l {scale:g} --fs {flow_shift:g} --d {seed}"\n'
    '    )\n'
    '    if neg:\n'
    '        line += f" --n {neg}"\n'
    '    return line\n'
    '\n'
    '\n'
    'SAMPLE_FIXED_SEED = 42'
)

LECO_OLD_WRITE = (
    'def _leco_write_sample_prompt_file(s: "_LecoTrainState") -> Path:\n'
    '    lines = []\n'
    '    if s.sample_enabled.get() and s.sample_prompt.get().strip():\n'
    '        lines.append(_leco_build_prompt_line(\n'
    '            s.sample_prompt.get().strip(),\n'
    '            s.sample_negative_prompt.get().strip(),\n'
    '            s,\n'
    '        ))\n'
    '    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():\n'
    '        lines.append(_leco_build_prompt_line(\n'
    '            s.sample_b_prompt.get().strip(),\n'
    '            s.sample_b_negative_prompt.get().strip(),\n'
    '            s,\n'
    '        ))\n'
    '    path = _leco_sample_prompt_path(s)\n'
    '    path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")\n'
    '    return path'
)

LECO_NEW_WRITE = (
    'def _leco_write_sample_prompt_file(s: "_LecoTrainState") -> Path:\n'
    '    lines = []\n'
    '    if s.sample_enabled.get() and s.sample_prompt.get().strip():\n'
    '        lines.append(_leco_build_prompt_line(\n'
    '            s.sample_prompt.get().strip(),\n'
    '            s.sample_negative_prompt.get().strip(),\n'
    '            s,\n'
    '            seed=SAMPLE_FIXED_SEED,\n'
    '        ))\n'
    '    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():\n'
    '        lines.append(_leco_build_prompt_line(\n'
    '            s.sample_b_prompt.get().strip(),\n'
    '            s.sample_b_negative_prompt.get().strip(),\n'
    '            s,\n'
    '            seed=SAMPLE_FIXED_SEED + 1,\n'
    '        ))\n'
    '    path = _leco_sample_prompt_path(s)\n'
    '    path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")\n'
    '    return path'
)


def apply():
    ok = True
    if not LORA_FILE.exists():
        print(f"[ERROR] {LORA_FILE} が見つかりません")
        sys.exit(1)
    if not LECO_FILE.exists():
        print(f"[ERROR] {LECO_FILE} が見つかりません")
        sys.exit(1)

    ok &= _patch(LORA_FILE, LORA_OLD_BUILD, LORA_NEW_BUILD, "lora_train _build_sample_prompt_line_for")
    ok &= _patch(LORA_FILE, LORA_OLD_WRITE, LORA_NEW_WRITE, "lora_train _write_sample_prompt_file")
    ok &= _patch(LECO_FILE, LECO_OLD_BUILD, LECO_NEW_BUILD, "leco_train _leco_build_prompt_line")
    ok &= _patch(LECO_FILE, LECO_OLD_WRITE, LECO_NEW_WRITE, "leco_train _leco_write_sample_prompt_file")

    if ok:
        print()
        print("[OK] BUG-04 全修正適用完了")
        print("  サンプルA: seed=42, サンプルB: seed=43")
        print("  A/Bで異なる初期ノイズが生成されるようになります。")
    else:
        print()
        print("[FAIL] 一部の修正が適用されませんでした。上記エラーを確認してください。")
        sys.exit(1)


if __name__ == "__main__":
    # プロジェクトルートから実行すること: python apply_fix_bug04_sample_seed.py
    apply()
