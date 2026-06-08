"""apply_fix_leco_log_dedup.py

app/leco_train.py のログ多重表示バグを修正する。

【原因】
_build_run_panel() が4タブ（tab_model / tab_prompts / tab_network / tab_train）
のループで呼ばれるため、s._log_widgets に4つの独立した tk.Text が登録される。
_drain() は for w in s._log_widgets: でループするため、
同一メッセージを4ウィジェット全てに書き込む。
ユーザーがどのタブを切り替えても全て同じログが表示される（多重表示）。

【修正】
修正1: _LecoTrainState に _log_primary_set フラグを追加。
修正2: _build_run_panel の s._log_widgets.append() を初回のみに制限し、
       _drain の for ループを廃止して _log_widgets[0] への単一書き込みに変更。
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("app/leco_train.py")

# ── 修正1: _log_primary_set フラグ追加（後続の区切り行まで含めて一意化）──────

OLD_STATE = _adapt("""\
        self._log_widgets: list[tk.Text] = []
        self._log_drain_started = False


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー\
""")

NEW_STATE = _adapt("""\
        self._log_widgets: list[tk.Text] = []
        self._log_drain_started = False
        self._log_primary_set = False  # 最初の log_text のみを drain 対象にする


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー\
""")

# ── 修正2: append を初回のみに制限、_drain を単一書き込みに変更 ───────────────

OLD_APPEND = _adapt("""\
    s._log_widgets.append(log_text)

    def _drain():
        while True:
            try:
                msg = s._log_queue.get_nowait()
                for w in s._log_widgets:
                    w.insert(tk.END, msg + "\\n")
                    w.see(tk.END)
            except queue.Empty:
                break
        parent.after(200, _drain)\
""")

NEW_APPEND = _adapt("""\
    if not s._log_primary_set:
        s._log_primary_set = True
        s._log_widgets.append(log_text)

    def _drain():
        while True:
            try:
                msg = s._log_queue.get_nowait()
                if s._log_widgets:
                    w = s._log_widgets[0]
                    w.insert(tk.END, msg + "\\n")
                    w.see(tk.END)
            except queue.Empty:
                break
        parent.after(200, _drain)\
""")


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    is_crlf = b"\r\n" in raw
    content = _adapt(raw.decode("utf-8"))

    modified = False

    # 修正1
    if OLD_STATE in content:
        content = content.replace(OLD_STATE, NEW_STATE, 1)
        print("[OK] 修正1: _log_primary_set フラグを追加しました。")
        modified = True
    elif NEW_STATE in content:
        print("[SKIP] 修正1: 既に適用済みです。")
    else:
        print("[ERROR] 修正1: 対象文字列が見つかりません。", file=sys.stderr)
        print("  探索先頭:", repr(OLD_STATE[:60]))
        sys.exit(1)

    # 修正2
    if OLD_APPEND in content:
        content = content.replace(OLD_APPEND, NEW_APPEND, 1)
        print("[OK] 修正2: _drain を単一ウィジェット書き込みに変更しました。")
        modified = True
    elif NEW_APPEND in content:
        print("[SKIP] 修正2: 既に適用済みです。")
    else:
        print("[ERROR] 修正2: 対象文字列が見つかりません。", file=sys.stderr)
        print("  探索先頭:", repr(OLD_APPEND[:60]))
        sys.exit(1)

    if not modified:
        return

    if is_crlf:
        content = content.replace("\n", "\r\n")
    path.write_text(content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")
    print("     ログ多重表示バグを修正しました。")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
