"""run_app.py — Animerge起動エントリポイント。

コマンドラインから実行すると、GUI(AnimaModelEditor)の構築が完了するまでの
間、コンソール上に簡易スピナーを表示する(apply_fix_108)。ウィンドウ表示
までの待ち時間に「フリーズしているのでは」という誤解を防ぐための表示であり、
処理内容そのものには一切関与しない。
"""
from __future__ import annotations

import itertools
import sys
import threading
import time

from app.gui import main

_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
_SPINNER_LABEL = "Animerge starting"


def _run_with_console_spinner() -> None:
    """コンソールスピナーを表示しつつ本体を起動する。

    main()のon_readyフック(apply_fix_107)が、AnimaModelEditorの構築が
    完了しmainloop()を開始する直前に呼ばれるため、そこでスピナーを止める。
    構築中に例外が発生した場合もfinallyで確実に停止する。
    """
    stream = sys.stdout
    if stream is None or not getattr(stream, "isatty", lambda: False)():
        # pythonw.exe等、コンソールが無い/接続されていない環境では
        # スピナーを出す意味が無い(書き込み自体が例外になりうる)ため、
        # 何もせず本体を直接起動する。
        main()
        return

    stop_event = threading.Event()

    def _spin() -> None:
        frames = itertools.cycle(_SPINNER_FRAMES)
        try:
            while not stop_event.is_set():
                stream.write(f"\r{_SPINNER_LABEL} {next(frames)} ")
                stream.flush()
                time.sleep(0.1)
        finally:
            stream.write("\r" + " " * (len(_SPINNER_LABEL) + 4) + "\r")
            stream.flush()

    spinner_thread = threading.Thread(target=_spin, daemon=True)
    spinner_thread.start()
    try:
        main(on_ready=stop_event.set)
    finally:
        stop_event.set()
        spinner_thread.join(timeout=1.0)


if __name__ == "__main__":
    _run_with_console_spinner()
