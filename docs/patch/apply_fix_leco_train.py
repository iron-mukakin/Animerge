"""
apply_fix_leco_train.py
leco_train.py に2件のバグ修正を適用する。

修正1: バグ1 - ログ乱れ (_worker の \r 未処理)
修正2: バグ2 - プレビュー不表示 (photo_refs GC問題 + 例外可視化)

使い方:
    python apply_fix_leco_train.py [leco_train.py のパス]
    省略時は同ディレクトリの leco_train.py を対象にする。
"""
from __future__ import annotations
import sys
import re
from pathlib import Path


def _adapt(text: str) -> str:
    """CRLF / LF どちらでも比較できるよう LF に正規化する。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def apply_patch(target: Path, old_raw: str, new_raw: str, label: str) -> bool:
    old = _adapt(old_raw)
    new = _adapt(new_raw)
    src = _adapt(target.read_text(encoding="utf-8"))
    count = src.count(old)
    if count == 0:
        print(f"[SKIP] {label}: 対象文字列が見つかりません（既適用か不一致）")
        return False
    if count > 1:
        print(f"[ERROR] {label}: 対象文字列が複数箇所にマッチします。中断します。")
        return False
    result = src.replace(old, new, 1)
    # 元ファイルの改行スタイルを保持
    original = target.read_text(encoding="utf-8")
    if "\r\n" in original:
        result = result.replace("\n", "\r\n")
    target.write_text(result, encoding="utf-8")
    print(f"[OK]   {label}: 適用完了")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 修正1: バグ1 — _worker の \r 未処理によるログ乱れ
#
# tqdm が出力する \r を除去し、\r で区切られた複数行を個別にキューへ入れる。
# ──────────────────────────────────────────────────────────────────────────────
OLD_BUG1 = """\
                for line in proc.stdout:
                    line = _re.sub(r'\\x1b\\[[0-9;]*[A-Za-z]', '', line.rstrip())
                    s._log_queue.put(line)
                    s._monitor_queue.put(line)
                    s._monitor_layer_queue.put(line)
                    lf.write(line + "\\n")
                    lf.flush()"""

NEW_BUG1 = """\
                for raw_line in proc.stdout:
                    # ANSI エスケープ除去
                    raw_line = _re.sub(r'\\x1b\\[[0-9;]*[A-Za-z]', '', raw_line)
                    # \\r で区切って各セグメントを個別に処理
                    # (tqdm が \\r で行を上書きするため複数セグメントが混在する)
                    segments = raw_line.split("\\r")
                    for seg in segments:
                        line = seg.rstrip()
                        if not line:
                            continue
                        s._log_queue.put(line)
                        s._monitor_queue.put(line)
                        s._monitor_layer_queue.put(line)
                        lf.write(line + "\\n")
                        lf.flush()"""

# ──────────────────────────────────────────────────────────────────────────────
# 修正2: バグ2 — プレビュー不表示
#
# 問題A: photo_refs がインスタンス変数でなくローカルリストのため、
#         _ab_panel の呼び出しが2回（A/B）あり、それぞれ独立した
#         photo_refs を持つ。_refresh 内で ph を photo_refs に代入しても
#         il.configure(image=ph) 後に ph への参照が _refresh のローカル変数
#         からしか辿れない瞬間がある（次の _refresh 呼び出しまでの間に
#         GC が走ると回収される）。
#         → photo_refs をウィジェット属性 (il._photo_ref) として保持し、
#           ウィジェット自体が参照を保有する形にする。
#
# 問題B: except Exception で例外を握りつぶしているため原因が見えない。
#         → ログ出力 + デバッグモード切替変数 (_SAMPLE_DEBUG) を追加。
# ──────────────────────────────────────────────────────────────────────────────
OLD_BUG2 = """\
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
                tab.after(2000, lambda: _refresh(True))"""

NEW_BUG2 = """\
        # デバッグログを有効にするには True にする
        _SAMPLE_DEBUG: bool = False

        def _refresh(schedule_next=False):
            import traceback as _tb
            files = sorted(
                _sdir.glob(_glob_pat), key=lambda p: p.stat().st_mtime, reverse=True
            )[:10] if _sdir.exists() else []
            if _SAMPLE_DEBUG:
                import logging as _lg
                _lg.getLogger(__name__).debug(
                    "[SamplePreview-%s] _sdir=%s exists=%s files=%d glob=%s",
                    label, _sdir, _sdir.exists(), len(files), _glob_pat,
                )
            try:
                from PIL import Image as _Im, ImageTk as _ITk
            except Exception:
                _Im = _ITk = None
            for idx, (il, el) in enumerate(cells):
                if idx >= len(files):
                    il.configure(image="", text="")
                    el.configure(text="step -")
                    # ウィジェット属性の参照もクリア
                    il._photo_ref = None  # type: ignore[attr-defined]
                    continue
                p = files[idx]
                m = _re_search(r"_([0-9]{6})_", p.stem)
                el.configure(text=f"step {int(m.group(1))}" if m else p.name)
                if _Im is None:
                    il.configure(image="", text=p.name)
                    il._photo_ref = None  # type: ignore[attr-defined]
                    continue
                try:
                    with _Im.open(p) as im:
                        im.thumbnail((220, 220))
                        ph = _ITk.PhotoImage(im.copy())
                    # ウィジェット自身に参照を保持させることで GC を防ぐ
                    il._photo_ref = ph  # type: ignore[attr-defined]
                    il.configure(image=ph, text="")
                    if _SAMPLE_DEBUG:
                        import logging as _lg
                        _lg.getLogger(__name__).debug(
                            "[SamplePreview-%s] idx=%d loaded %s", label, idx, p.name
                        )
                except Exception as _exc:
                    # 例外内容をラベルに表示してデバッグを容易にする
                    _err_msg = f"[ERR] {type(_exc).__name__}: {_exc}"
                    il.configure(image="", text=_err_msg)
                    il._photo_ref = None  # type: ignore[attr-defined]
                    if _SAMPLE_DEBUG:
                        import logging as _lg
                        _lg.getLogger(__name__).error(
                            "[SamplePreview-%s] idx=%d file=%s\\n%s",
                            label, idx, p.name, _tb.format_exc(),
                        )
            if schedule_next:
                tab.after(2000, lambda: _refresh(True))"""


def main() -> None:
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        target = Path(__file__).parent / "leco_train.py"

    if not target.exists():
        print(f"[ERROR] ファイルが見つかりません: {target}")
        sys.exit(1)

    print(f"対象ファイル: {target}")

    # バックアップ
    backup = target.with_suffix(".py.bak")
    backup.write_bytes(target.read_bytes())
    print(f"バックアップ: {backup}")

    ok1 = apply_patch(target, OLD_BUG1, NEW_BUG1, "バグ1: ログ乱れ (\\r 未処理)")
    ok2 = apply_patch(target, OLD_BUG2, NEW_BUG2, "バグ2: プレビュー不表示 (photo_refs GC + 例外可視化)")

    if not (ok1 or ok2):
        print("\n変更なし。バックアップを復元します。")
        target.write_bytes(backup.read_bytes())
    else:
        applied = sum([ok1, ok2])
        print(f"\n{applied}/2 件の修正を適用しました。")
        print("デバッグログを有効にするには _SAMPLE_DEBUG = True に変更してください。")


if __name__ == "__main__":
    main()
