"""addift_train.py (GUI) に `qwen_image_vae_2d` チェックボックスを追加し、
コマンド生成・設定保存/読込に連携させるパッチ。

実行例:
    python apply_fix_addift_train_gui.py addift_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _patch_lib import apply_substitutions, parse_target_path_argument, report_failure_and_exit

DEFAULT_RELATIVE_PATH: str = "addift_train.py"

VARIABLE_DEFINITION_SEARCH: str = '''        self.vae_chunk_size   = tk.StringVar(value="")
        self.vae_disable_cache = tk.BooleanVar(value=False)
'''
VARIABLE_DEFINITION_REPLACEMENT: str = '''        self.vae_chunk_size   = tk.StringVar(value="")
        self.vae_disable_cache = tk.BooleanVar(value=False)
        self.qwen_image_vae_2d = tk.BooleanVar(value=False)
'''

WIDGET_SEARCH: str = '''    ttk.Checkbutton(lf2, text="vae_disable_cache",
                    variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)
'''
WIDGET_REPLACEMENT: str = '''    ttk.Checkbutton(lf2, text="vae_disable_cache",
                    variable=s.vae_disable_cache).grid(
        row=1, column=0, sticky=tk.W, padx=8, pady=3)
    ttk.Checkbutton(lf2, text="qwen_image_vae_2d",
                    variable=s.qwen_image_vae_2d).grid(
        row=1, column=1, sticky=tk.W, padx=8, pady=3)
'''

COMMAND_BUILD_SEARCH: str = '''        (s.vae_disable_cache,                   "--vae_disable_cache"),
        (s.train_fixed_timesteps_in_batch,      "--train_fixed_timesteps_in_batch"),
    ]
'''
COMMAND_BUILD_REPLACEMENT: str = '''        (s.vae_disable_cache,                   "--vae_disable_cache"),
        (s.qwen_image_vae_2d,                   "--qwen_image_vae_2d"),
        (s.train_fixed_timesteps_in_batch,      "--train_fixed_timesteps_in_batch"),
    ]
'''

SETTINGS_SAVE_SEARCH: str = '''            "vae_chunk_size":    s.vae_chunk_size.get(),
            "vae_disable_cache": bool(s.vae_disable_cache.get()),
'''
SETTINGS_SAVE_REPLACEMENT: str = '''            "vae_chunk_size":    s.vae_chunk_size.get(),
            "vae_disable_cache": bool(s.vae_disable_cache.get()),
            "qwen_image_vae_2d": bool(s.qwen_image_vae_2d.get()),
'''

SETTINGS_LOAD_SEARCH: str = '''        _s(s.vae_chunk_size,    "vae_chunk_size",     "")
        _s(s.vae_disable_cache, "vae_disable_cache",  False)
'''
SETTINGS_LOAD_REPLACEMENT: str = '''        _s(s.vae_chunk_size,    "vae_chunk_size",     "")
        _s(s.vae_disable_cache, "vae_disable_cache",  False)
        _s(s.qwen_image_vae_2d, "qwen_image_vae_2d",  False)
'''


def main() -> None:
    target_path = parse_target_path_argument(DEFAULT_RELATIVE_PATH)
    substitutions = [
        (VARIABLE_DEFINITION_SEARCH, VARIABLE_DEFINITION_REPLACEMENT),
        (WIDGET_SEARCH, WIDGET_REPLACEMENT),
        (COMMAND_BUILD_SEARCH, COMMAND_BUILD_REPLACEMENT),
        (SETTINGS_SAVE_SEARCH, SETTINGS_SAVE_REPLACEMENT),
        (SETTINGS_LOAD_SEARCH, SETTINGS_LOAD_REPLACEMENT),
    ]
    try:
        apply_substitutions(target_path, substitutions)
    except (FileNotFoundError, ValueError) as error:
        report_failure_and_exit(error)


if __name__ == "__main__":
    main()
