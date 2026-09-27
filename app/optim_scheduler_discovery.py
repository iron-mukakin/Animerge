"""app/optim_scheduler_discovery.py — カスタムOptimizer/LRSchedulerの自動検出。

sd-scripts/optimizer/ ・ sd-scripts/scheduler/ 配下に配置されたユーザー定義の
torch.optim.Optimizer / torch.optim.lr_scheduler.LRScheduler サブクラスを検出し、
lora_train.py / leco_train.py / addift_train.py の3学習タブから共通で利用する
(apply_fix_069〜071)。3ファイルへの複製を避けるためこのモジュールに集約した。

検出したクラスは "optimizer.モジュール名.クラス名" / "scheduler.モジュール名.クラス名"
という完全修飾パス文字列として返す。この文字列は sd-scripts/library/train_util.py の
get_optimizer() / get_scheduler_fix() が --optimizer_type / --lr_scheduler_type
引数に対して行う動的import(importlib.import_module + getattr、
train_util.py 5257〜5268行目 / 5434〜5445行目)にそのまま渡せる形式である
(実際のtrain_util.pyを確認した上で設計)。

検出はGUI起動時([呼び出し元]の各 _TrainState.__init__() 実行時)に一度だけ行われる。
sd-scripts/optimizer/・sd-scripts/scheduler/ 配下へ新規ファイルを追加した場合、
反映にはアプリの再起動が必要(実行中の動的反映は行わない)。

パス解決について: sd_scripts_root は呼び出し側が AppPaths(s.paths.root /
"sd-scripts")から渡すこと。このモジュール自身は __file__ 基準の相対パス解決を
行わない(プロジェクトのAppPaths経由のパス解決規約に統一するため)。
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Dict

from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler as _SchedulerBaseClass
except ImportError:
    # torch < 2.0 では LRScheduler ではなく _LRScheduler という名前だった。
    from torch.optim.lr_scheduler import _LRScheduler as _SchedulerBaseClass  # type: ignore[attr-defined]


def discover_custom_optimizers(sd_scripts_root: Path) -> Dict[str, str]:
    """sd_scripts_root/optimizer/ 以下の torch.optim.Optimizer サブクラスを検出する。

    Args:
        sd_scripts_root: sd-scripts ディレクトリの絶対パス
            (呼び出し側は s.paths.root / "sd-scripts" を渡すこと)。

    Returns:
        {表示名(クラス名): "optimizer.モジュール名.クラス名"} の辞書。
        optimizer/ フォルダが存在しない、またはimport不能な場合は空辞書を返す
        (呼び出し側の起動を妨げないため、例外は投げない)。
    """
    return _discover_classes(sd_scripts_root, "optimizer", Optimizer)


def discover_custom_schedulers(sd_scripts_root: Path) -> Dict[str, str]:
    """sd_scripts_root/scheduler/ 以下の LRScheduler サブクラスを検出する。

    Args:
        sd_scripts_root: sd-scripts ディレクトリの絶対パス。

    Returns:
        {表示名(クラス名): "scheduler.モジュール名.クラス名"} の辞書。
        scheduler/ フォルダが存在しない、またはimport不能な場合は空辞書を返す。
    """
    return _discover_classes(sd_scripts_root, "scheduler", _SchedulerBaseClass)


def _discover_classes(sd_scripts_root: Path, package_name: str, base_class: type) -> Dict[str, str]:
    """package_name ("optimizer" または "scheduler") パッケージ配下から
    base_class のサブクラスを検出する共通実装
    (discover_custom_optimizers/schedulers の重複を排除するため)。
    """
    discovered: Dict[str, str] = {}

    package_dir = Path(sd_scripts_root) / package_name
    if not package_dir.is_dir():
        return discovered

    root_str = str(sd_scripts_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        package = importlib.import_module(package_name)
    except Exception:
        return discovered

    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"{package_name}.{module_name}")
        except Exception:
            # 個別モジュールのimport失敗はスキップし、他モジュールの検出を継続する。
            continue

        for class_name, obj in inspect.getmembers(module, inspect.isclass):
            # 外部からimportされているクラスを誤検出しない
            # (モジュール内で "from torch.optim import AdamW" のように再import
            # しただけの名前を、そのモジュール発のカスタムクラスと誤認しないため)。
            if obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, base_class):
                continue
            discovered[class_name] = f"{package_name}.{module_name}.{class_name}"

    return discovered
