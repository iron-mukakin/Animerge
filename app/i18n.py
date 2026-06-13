"""app/i18n.py — 国際化 (i18n) サポートモジュール

使い方:
    from .i18n import _

    label = _("save")          # → "保存" (ja) / "Save" (en)

設定ファイル:
    configs/settings.json  {"language": "ja"}
言語テキスト:
    configs/texts_ja.json
    configs/texts_en.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── デフォルト言語 ────────────────────────────────────────────────────────────
_DEFAULT_LANG = "ja"

# ── 現在のテキスト辞書 ───────────────────────────────────────────────────────
_texts: dict[str, str] = {}
_current_lang: str = _DEFAULT_LANG

# ── configs ディレクトリのパス解決 ────────────────────────────────────────────
# __file__ = app/i18n.py → parent = app/ → parent.parent = project root
_APP_DIR = Path(__file__).resolve().parent
_CONFIGS_DIR = _APP_DIR.parent / "configs"


def _configs_dir() -> Path:
    """configs ディレクトリを返す（AppPaths 未依存）。"""
    return _CONFIGS_DIR


def _texts_path(lang: str) -> Path:
    return _configs_dir() / f"texts_{lang}.json"


def _settings_path() -> Path:
    return _configs_dir() / "settings.json"


def load_language(lang: str | None = None) -> None:
    """言語テキストを読み込む。lang が None の場合は settings.json から取得する。"""
    global _texts, _current_lang

    if lang is None:
        lang = _load_setting("language", _DEFAULT_LANG)

    path = _texts_path(lang)
    if not path.exists():
        # フォールバック: デフォルト言語
        path = _texts_path(_DEFAULT_LANG)
        lang = _DEFAULT_LANG

    try:
        _texts = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _texts = {}

    _current_lang = lang


def get_language() -> str:
    """現在の言語コードを返す。"""
    return _current_lang


def get_supported_languages() -> list[str]:
    """利用可能な言語コードのリストを返す。"""
    langs = []
    for p in _configs_dir().glob("texts_*.json"):
        lang = p.stem.replace("texts_", "")
        langs.append(lang)
    return sorted(langs)


def save_language(lang: str) -> None:
    """言語設定を settings.json に保存する。"""
    _save_setting("language", lang)


def _load_setting(key: str, default: Any) -> Any:
    path = _settings_path()
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(key, default)
    except Exception:
        return default


def _save_setting(key: str, value: Any) -> None:
    path = _settings_path()
    _configs_dir().mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[key] = value
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def gettext(key: str, **kwargs: Any) -> str:
    """国際化関数。key に対応するテキストを返す。

    テキストが未定義の場合は key をそのまま返す。
    kwargs を渡すと .format(**kwargs) で置換する。
    """
    text = _texts.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def _(key: str, **kwargs: Any) -> str:
    """後方互換エイリアス。gettext() を呼び出す。"""
    return gettext(key, **kwargs)


# モジュール読み込み時に自動で言語をロード
load_language()
