"""apply_fix_leco_tab_order.py

app/leco_train.py のタブ表示順序を lora_train.py と合わせる。

変更前:
  ...階層学習 → サンプル生成 → モニターグラフ → モニター階層 → プリセット

変更後:
  ...階層学習 → モニターグラフ → モニター階層 → サンプル生成 → プリセット

nb.add() の順序と _build_* 呼び出しの順序を両方変更する。
タブ内コンポーネントの変数参照（イベントハンドラ等）は変数バインドのため影響なし。
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("app/leco_train.py")

OLD_ADD = _adapt('''\
    nb.add(tab_layer,          text="  階層学習  ")
    nb.add(tab_sample,         text="  サンプル生成  ")
    nb.add(tab_monitor,        text="  モニターグラフ  ")
    nb.add(tab_monitor_layer,  text="  モニター階層  ")
    nb.add(tab_preset,         text="  プリセット  ")\
''')

NEW_ADD = _adapt('''\
    nb.add(tab_layer,          text="  階層学習  ")
    nb.add(tab_monitor,        text="  モニターグラフ  ")
    nb.add(tab_monitor_layer,  text="  モニター階層  ")
    nb.add(tab_sample,         text="  サンプル生成  ")
    nb.add(tab_preset,         text="  プリセット  ")\
''')

OLD_BUILD = _adapt('''\
    _build_layer_train_tab(tab_layer, state)
    _build_leco_sample_tab(tab_sample, state)
    _build_monitor_tab(tab_monitor,   state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_leco_preset_tab(tab_preset, state)\
''')

NEW_BUILD = _adapt('''\
    _build_layer_train_tab(tab_layer, state)
    _build_monitor_tab(tab_monitor,   state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_leco_sample_tab(tab_sample, state)
    _build_leco_preset_tab(tab_preset, state)\
''')


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    is_crlf = b"\r\n" in raw
    content = _adapt(raw.decode("utf-8"))

    modified = False

    # nb.add() 順序
    if OLD_ADD in content:
        content = content.replace(OLD_ADD, NEW_ADD, 1)
        print("[OK] nb.add() の順序を変更しました。")
        modified = True
    elif NEW_ADD in content:
        print("[SKIP] nb.add() の順序は既に適用済みです。")
    else:
        print("[ERROR] nb.add() 対象文字列が見つかりません。", file=sys.stderr)
        sys.exit(1)

    # _build_* 呼び出し順序
    if OLD_BUILD in content:
        content = content.replace(OLD_BUILD, NEW_BUILD, 1)
        print("[OK] _build_*() の呼び出し順序を変更しました。")
        modified = True
    elif NEW_BUILD in content:
        print("[SKIP] _build_*() の呼び出し順序は既に適用済みです。")
    else:
        print("[ERROR] _build_*() 対象文字列が見つかりません。", file=sys.stderr)
        sys.exit(1)

    if not modified:
        return

    if is_crlf:
        content = content.replace("\n", "\r\n")
    path.write_text(content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
