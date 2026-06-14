"""apply_fix_gui_cleanup.py - gui.py の終了処理にADDifT学習プロセスの停止を追加するパッチ。

使い方:
    python3 apply_fix_gui_cleanup.py /path/to/gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _adapt(text: str, pattern: str) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    return pattern.replace("\r\n", "\n").replace("\n", newline)


OLD = (
    "        # LECO学習プロセスが動いている場合は先に終了させる\n"
    "        if hasattr(self, \"_leco_train_state\") and self._leco_train_state is not None:\n"
    "            proc = getattr(self._leco_train_state, \"_proc\", None)\n"
    "            if proc is not None and proc.poll() is None:\n"
    "                import os, signal\n"
    "                try:\n"
    "                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)\n"
    "                except Exception:\n"
    "                    proc.terminate()\n"
    "                proc.wait(timeout=10)\n"
    "                self.log(gettext(\"unload_leco_proc\"))\n"
)

NEW = OLD + (
    "        # ADDifT学習プロセスが動いている場合は先に終了させる\n"
    "        if hasattr(self, \"_addift_train_state\") and self._addift_train_state is not None:\n"
    "            proc = getattr(self._addift_train_state, \"_proc\", None)\n"
    "            if proc is not None and proc.poll() is None:\n"
    "                import os, signal\n"
    "                try:\n"
    "                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)\n"
    "                except Exception:\n"
    "                    proc.terminate()\n"
    "                proc.wait(timeout=10)\n"
    "                self.log(gettext(\"unload_addift_proc\"))\n"
)


def apply_fix(target_path: Path) -> None:
    text = target_path.read_text(encoding="utf-8")
    old_adapted = _adapt(text, OLD)
    new_adapted = _adapt(text, NEW)
    count = text.count(old_adapted)
    if count == 0:
        raise ValueError(f"パッチ対象文字列が見つかりません:\n{old_adapted!r}")
    if count > 1:
        raise ValueError(f"パッチ対象文字列が複数箇所に存在します({count}件):\n{old_adapted!r}")
    text = text.replace(old_adapted, new_adapted, 1)
    target_path.write_text(text, encoding="utf-8")
    print(f"適用完了: {target_path} に終了処理パッチを適用しました。")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 apply_fix_gui_cleanup.py /path/to/gui.py", file=sys.stderr)
        raise SystemExit(1)
    apply_fix(Path(sys.argv[1]))
