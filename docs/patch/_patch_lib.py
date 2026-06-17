"""sd-scripts / GUIファイル群へテキストブロック単位の差分を適用するための共有ユーティリティ。

`patch` コマンドに依存せず、CRLF/LFが混在する環境でも安全に動作することを目的とする。
各 `apply_fix_*.py` は、修正対象ファイルと同じディレクトリに本モジュールを置いた上でimportして使用する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, Iterable

TEXT_ENCODING: Final[str] = "utf-8"
LF: Final[str] = "\n"
CRLF: Final[str] = "\r\n"
BACKUP_SUFFIX: Final[str] = ".bak"


def _adapt(text: str, newline: str) -> str:
    """LF区切りで記述された文字列を、対象ファイル実際の改行コードに変換する。

    Args:
        text: `\\n` 区切りで記述された検索/置換用の文字列。
        newline: 変換先の改行コード(`"\\n"` または `"\\r\\n"`)。

    Returns:
        newlineに統一された文字列。
    """
    normalized = text.replace(CRLF, LF)
    return normalized if newline == LF else normalized.replace(LF, CRLF)


def detect_newline_style(raw_bytes: bytes) -> str:
    """ファイルの生バイト列から改行コード(CRLF/LF)を判定する。

    Args:
        raw_bytes: 対象ファイルの生バイト列。

    Returns:
        `"\\r\\n"`(CRLFが含まれる場合)または `"\\n"`。
    """
    return CRLF if CRLF.encode(TEXT_ENCODING) in raw_bytes else LF


def duplicate_as_backup(target_path: Path) -> Path:
    """修正前のファイルを `<元のファイル名>.bak` として同じディレクトリに複製する。

    Args:
        target_path: 複製元ファイルのパス。

    Returns:
        作成したバックアップファイルのパス。
    """
    backup_path = target_path.with_name(target_path.name + BACKUP_SUFFIX)
    backup_path.write_bytes(target_path.read_bytes())
    return backup_path


def substitute_text_block(target_path: Path, search_text: str, replacement_text: str) -> None:
    """ファイル内に一意に出現するテキストブロックを置換し、元の改行コードで書き戻す。

    Args:
        target_path: 修正対象ファイルのパス。
        search_text: `\\n` 区切りで記述された検索対象文字列。ファイル内にちょうど
            1回出現することを前提とする。
        replacement_text: `\\n` 区切りで記述された置換後の文字列。

    Raises:
        FileNotFoundError: target_pathが存在しない場合。
        ValueError: search_textの出現回数が1でない場合
            (既に適用済み、または対象ファイルの内容が想定と異なる可能性がある)。
    """
    if not target_path.is_file():
        raise FileNotFoundError(f"対象ファイルが見つかりません: {target_path}")

    raw_bytes = target_path.read_bytes()
    newline = detect_newline_style(raw_bytes)
    content = raw_bytes.decode(TEXT_ENCODING)
    adapted_search = _adapt(search_text, newline)

    occurrence_count = content.count(adapted_search)
    if occurrence_count != 1:
        raise ValueError(
            "検索テキストの出現回数が想定外です"
            f"(期待値=1, 実測値={occurrence_count}): {target_path}\n"
            "既に適用済み、または対象ファイルの内容が想定と異なる可能性があります。"
        )

    adapted_replacement = _adapt(replacement_text, newline)
    new_content = content.replace(adapted_search, adapted_replacement)
    target_path.write_bytes(new_content.encode(TEXT_ENCODING))


def parse_target_path_argument(default_relative_path: str) -> Path:
    """コマンドライン引数から修正対象ファイルへの相対パスを取得する。

    環境依存を避けるため、絶対パスの指定は禁止する。

    Args:
        default_relative_path: 引数省略時に使用する相対パス。

    Returns:
        修正対象ファイルへの相対パス。

    Raises:
        ValueError: 絶対パスが指定された場合。
    """
    arg_parser = argparse.ArgumentParser(description="テキストブロック差分の適用スクリプト")
    arg_parser.add_argument(
        "target",
        nargs="?",
        default=default_relative_path,
        type=str,
        help="修正対象ファイルへの相対パス(省略時は既定値を使用)",
    )
    parsed_args = arg_parser.parse_args()
    target_path = Path(parsed_args.target)
    if target_path.is_absolute():
        raise ValueError("絶対パスは禁止されています。相対パスを指定してください。")
    return target_path


def apply_substitutions(target_path: Path, substitutions: Iterable[tuple[str, str]]) -> None:
    """バックアップを作成した上で、複数のテキストブロック置換を順番に適用する。

    Args:
        target_path: 修正対象ファイルのパス。
        substitutions: (検索テキスト, 置換テキスト) のシーケンス。

    Raises:
        FileNotFoundError: target_pathが存在しない場合。
        ValueError: いずれかの検索テキストの出現回数が1でない場合。
    """
    backup_path = duplicate_as_backup(target_path)
    print(f"バックアップ作成: {backup_path}")
    for search_text, replacement_text in substitutions:
        substitute_text_block(target_path, search_text, replacement_text)
    print(f"パッチ適用完了: {target_path}")


def report_failure_and_exit(error: Exception) -> None:
    """エラー内容を表示し、非ゼロ終了コードでプロセスを終える。"""
    print(f"パッチ適用に失敗しました: {error}", file=sys.stderr)
    sys.exit(1)
