"""app/gui_cash_io.py — GUI入力値の自動キャッシュ(gui_cash)用共有I/Oユーティリティ。

lora_train.py / leco_train.py / addift_train.py の3学習タブが、起動時に前回入力を
自動復元し、ウィンドウクローズ時に自動保存するために使う(apply_fix_073〜076)。

保存先は1ファイル: <AppPaths.root>/preset/gui_cash.json
内部構造は {"lora_train": {...}, "leco_train": {...}, "addift_train": {...}} という
セクション分割で、3タブが同じファイルへ書き込んでも互いの内容を上書きしないよう、
書き込み時は必ず「既存ファイルを読み直し→自セクションだけ更新→全体を書き戻す」という
read-merge-write方式を取る(同一プロセス内・Tkinterメインスレッドからの呼び出しのみを
想定しており、プロセス間排他制御は行わない)。

このファイルは既存の名前付きプリセット(preset/lora_train/*.json 等)とは独立している
(_refresh_list()のglob対象に入らないよう、意図的に別ファイルにした)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _cache_file_path(app_root: Path) -> Path:
    return Path(app_root) / "preset" / "gui_cash.json"


def read_gui_cash_section(app_root: Path, section: str) -> Optional[Dict[str, Any]]:
    """gui_cash.json から指定セクションのみを読み込む。

    ファイルが存在しない、壊れている、該当セクションが無い場合はいずれもNoneを
    返す(呼び出し側は「復元するデータが無い」として通常起動を続ければよい。
    例外は投げない — 起動時の自動復元が、壊れたキャッシュファイル1つでアプリ
    全体の起動を妨げてはならないため)。
    """
    path = _cache_file_path(app_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return None
    return section_data


def write_gui_cash_section(app_root: Path, section: str, data: Dict[str, Any]) -> None:
    """gui_cash.json の指定セクションだけを更新して書き戻す(read-merge-write)。

    他タブが書き込んだ別セクションを消さないよう、書き込み直前に既存ファイルを
    読み直してからマージする。失敗時は例外を送出する(呼び出し側=クローズ処理で
    ログに残すため)。
    """
    path = _cache_file_path(app_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            # 既存ファイルが壊れている場合は、他セクションの救済は諦めて
            # 自セクションだけを新規に書き込む(黙って諦めて何もしないより安全)。
            existing = {}

    existing[section] = data
    path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
