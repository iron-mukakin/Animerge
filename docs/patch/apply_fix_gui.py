"""apply_fix_gui.py - gui.py に ADDifT学習タブ(LECO学習の右隣)を追加するパッチ。

使い方:
    python3 apply_fix_gui.py /path/to/gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _adapt(text: str, pattern: str) -> str:
    """pattern の改行をtextの主要改行コードに合わせて変換する。"""
    newline = "\r\n" if "\r\n" in text else "\n"
    return pattern.replace("\r\n", "\n").replace("\n", newline)


PATCHES: list[tuple[str, str]] = [
    # 1. import 追加
    (
        "from .leco_train import build_leco_train_tab\n",
        "from .leco_train import build_leco_train_tab\n"
        "from .addift_train import build_addift_train_tab\n",
    ),
    # 2. 主タブFrame生成: addift_main を leco_main の直後に追加
    (
        "        leco_main     = ttk.Frame(self.main_notebook, padding=4)\n"
        "        settings_main = ttk.Frame(self.main_notebook, padding=4)\n",
        "        leco_main     = ttk.Frame(self.main_notebook, padding=4)\n"
        "        addift_main   = ttk.Frame(self.main_notebook, padding=4)\n"
        "        settings_main = ttk.Frame(self.main_notebook, padding=4)\n",
    ),
    # 3. 主タブ登録: addift_main を leco_main の直後に追加
    (
        "        self.main_notebook.add(leco_main,     text=gettext(\"main_tab_leco_train\"))\n"
        "        self.main_notebook.add(settings_main, text=gettext(\"settings_tab\"))\n",
        "        self.main_notebook.add(leco_main,     text=gettext(\"main_tab_leco_train\"))\n"
        "        self.main_notebook.add(addift_main,   text=gettext(\"main_tab_addift_train\"))\n"
        "        self.main_notebook.add(settings_main, text=gettext(\"settings_tab\"))\n",
    ),
    # 4. build_addift_train_tab 呼び出しを build_leco_train_tab の直後に追加
    (
        "        self._leco_train_state = build_leco_train_tab(\n"
        "            leco_main,\n"
        "            self.paths,\n"
        "            self.log,\n"
        "            lambda: self.model_choices,\n"
        "        )\n",
        "        self._leco_train_state = build_leco_train_tab(\n"
        "            leco_main,\n"
        "            self.paths,\n"
        "            self.log,\n"
        "            lambda: self.model_choices,\n"
        "        )\n"
        "        self._addift_train_state = build_addift_train_tab(\n"
        "            addift_main,\n"
        "            self.paths,\n"
        "            self.log,\n"
        "            lambda: self.model_choices,\n"
        "        )\n",
    ),
    # 5. _on_main_tab_changed のインデックス分岐: addift(6) を挿入し settings を 7 へシフト
    (
        "        elif idx == 5:\n"
        "            self._active_tab_type = \"leco_train\"\n"
        "            return  # LECO学習タブは merge 系のコントロール再構築不要\n"
        "        elif idx == 6:\n"
        "            self._active_tab_type = \"settings\"\n"
        "            return  # 設定タブは merge 系のコントロール再構築不要\n",
        "        elif idx == 5:\n"
        "            self._active_tab_type = \"leco_train\"\n"
        "            return  # LECO学習タブは merge 系のコントロール再構築不要\n"
        "        elif idx == 6:\n"
        "            self._active_tab_type = \"addift_train\"\n"
        "            return  # ADDifT学習タブは merge 系のコントロール再構築不要\n"
        "        elif idx == 7:\n"
        "            self._active_tab_type = \"settings\"\n"
        "            return  # 設定タブは merge 系のコントロール再構築不要\n",
    ),
]


def apply_fix(target_path: Path) -> None:
    text = target_path.read_text(encoding="utf-8")
    applied = 0
    for old, new in PATCHES:
        old_adapted = _adapt(text, old)
        new_adapted = _adapt(text, new)
        count = text.count(old_adapted)
        if count == 0:
            raise ValueError(f"パッチ対象文字列が見つかりません:\n{old_adapted!r}")
        if count > 1:
            raise ValueError(f"パッチ対象文字列が複数箇所に存在します({count}件):\n{old_adapted!r}")
        text = text.replace(old_adapted, new_adapted, 1)
        applied += 1
    target_path.write_text(text, encoding="utf-8")
    print(f"適用完了: {applied} 箇所のパッチを {target_path} に適用しました。")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 apply_fix_gui.py /path/to/gui.py", file=sys.stderr)
        raise SystemExit(1)
    apply_fix(Path(sys.argv[1]))
