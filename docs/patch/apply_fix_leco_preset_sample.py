"""apply_fix_leco_preset_sample.py

app/leco_train.py のプリセット読み込み時にサンプル生成設定が
反映されないバグを修正する。

【原因】
_apply() 内の _s(var, key, default) は内部で
    if key in data:
        var.set(data[key])
と実装されており、"data" はトップレベルの JSON dict を参照している。
サンプル設定は data["sample"] サブ dict に格納されているため、
キー "every_n_steps" 等をトップレベル data から探索しても常にヒットせず、
全項目が無視される。

【修正】
サンプル設定復元ブロックを _s() から独立した専用ヘルパー _ss() に置き換え、
sample サブ dict から直接値を取得して各 tk.Var に set() する。
"""

import sys
from pathlib import Path


def _adapt(s: str) -> str:
    return s.replace("\r\n", "\n")


TARGET = Path("app/leco_train.py")

OLD = _adapt('''\
        # サンプル生成設定
        sample = data.get("sample", {})
        if sample:
            _s(s.sample_every_n_steps,      "every_n_steps",      "100")
            _s(s.sample_keep_vae,           "keep_vae",           False)
            _s(s.sample_width,              "width",              512)
            _s(s.sample_height,             "height",             512)
            _s(s.sample_steps,              "steps",              20)
            _s(s.sample_scale,              "scale",              7.5)
            _s(s.sample_flow_shift,         "flow_shift",         3.0)
            _s(s.sample_enabled,            "a_enabled",          False)
            _s(s.sample_prompt,             "a_prompt",           "")
            _s(s.sample_negative_prompt,    "a_negative_prompt",  "")
            _s(s.sample_b_enabled,          "b_enabled",          False)
            _s(s.sample_b_prompt,           "b_prompt",           "")
            _s(s.sample_b_negative_prompt,  "b_negative_prompt",  "")\
''')

NEW = _adapt('''\
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
            _ss(s.sample_b_negative_prompt,   "b_negative_prompt",  "")\
''')


def apply(path: Path = TARGET):
    if not path.exists():
        print(f"[ERROR] ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    is_crlf = b"\r\n" in raw
    content = _adapt(raw.decode("utf-8"))

    if OLD not in content:
        if NEW in content:
            print("[SKIP] 既に適用済みです。")
            return
        print("[ERROR] 対象文字列が見つかりません。", file=sys.stderr)
        print("  探索先頭:", repr(OLD[:80]))
        sys.exit(1)

    content = content.replace(OLD, NEW, 1)

    if is_crlf:
        content = content.replace("\n", "\r\n")
    path.write_text(content, encoding="utf-8", newline="")
    print(f"[OK] {path} を更新しました。")
    print("     サンプル生成設定のプリセット読み込み反映バグを修正しました。")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    apply(target)
