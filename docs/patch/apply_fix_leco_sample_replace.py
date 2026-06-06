"""
apply_fix_leco_sample_replace.py
----------------------------------
anima_train_leco.py の変更:
  1. import に anima_sample_gen を追加
  2. サンプル生成呼び出しブロックを anima_sample_gen.sample_images_from_prompts に差し替え
     vae=None の場合は args.vae から内部で都度ロード（sample_keep_vae=False 対応）

実行場所: プロジェクトルート
  python apply_fix_leco_sample_replace.py
"""

import sys, time, shutil, re
from pathlib import Path

TARGET = Path("sd-scripts") / "anima_train_leco.py"

def _adapt(s): return s.replace("\r\n", "\n")

OLD_IMPORT = (
    "from library import (\n"
    "    anima_models,\n"
    "    anima_train_utils,\n"
    "    anima_utils,\n"
    "    custom_train_functions,\n"
    "    qwen_image_autoencoder_kl,\n"
    "    sd3_train_utils,\n"
    "    strategy_anima,\n"
    "    train_util,\n"
    ")"
)

NEW_IMPORT = OLD_IMPORT.replace(")", ")\nimport anima_sample_gen")

NEW_SAMPLE_CALL = '            # ── サンプル生成 ──────────────────────────────────────────────\n            if (\n                getattr(args, "sample_every_n_steps", None)\n                and global_step % args.sample_every_n_steps == 0\n                and args.sample_prompts\n                and args.sample_save_dir\n                and accelerator.is_main_process\n            ):\n                # LECO: LoRA を推論モードへ切り替えてサンプル生成\n                net_unwrapped = accelerator.unwrap_model(network)\n                net_unwrapped.set_multiplier(1.0)\n                net_unwrapped.eval()\n                dit_unwrapped = accelerator.unwrap_model(dit)\n                try:\n                    anima_sample_gen.sample_images_from_prompts(\n                        args=args,\n                        dit=dit_unwrapped,\n                        vae_for_sample=_vae_for_sample,\n                        text_encoder=qwen3_text_encoder,\n                        tokenize_strategy=tokenize_strategy,\n                        text_encoding_strategy=text_encoding_strategy,\n                        accelerator=accelerator,\n                        epoch=None,\n                        global_step=global_step,\n                    )\n                finally:\n                    net_unwrapped.train()\n                    net_unwrapped.set_multiplier(0.0)\n'

def _get_old_block(content):
    m = re.search(r"            # .{2} \u30b5\u30f3\u30d7\u30eb\u751f\u6210 .+\n", content)
    if not m: return ""
    idx = m.start()
    end_marker = "                )\n"
    end_pos = content.find(end_marker, idx)
    if end_pos == -1: return ""
    return content[idx:end_pos + len(end_marker)]

def apply():
    if not TARGET.exists():
        print(f"[ERROR] ファイルが見つかりません: {TARGET}"); sys.exit(1)
    content = _adapt(TARGET.read_text(encoding="utf-8"))

    old_imp = _adapt(OLD_IMPORT)
    if content.count(old_imp) != 1:
        print("[ERROR] import ブロックが見つかりません。"); sys.exit(1)
    content = content.replace(old_imp, _adapt(NEW_IMPORT))
    print("  OK: import に anima_sample_gen を追加")

    old_block = _get_old_block(content)
    if not old_block:
        print("[ERROR] サンプル生成ブロックが見つかりません。"); sys.exit(1)
    if content.count(old_block) != 1:
        print("[ERROR] サンプル生成ブロックが複数存在します。"); sys.exit(1)
    content = content.replace(old_block, NEW_SAMPLE_CALL)
    print("  OK: サンプル生成呼び出しブロックを差し替え")

    bak = TARGET.with_suffix(f".bak_{int(time.time())}")
    shutil.copy2(TARGET, bak)
    print(f"  バックアップ: {bak}")
    TARGET.write_text(content, encoding="utf-8", newline="\n")
    print(f"  書き込み完了: {TARGET}")
    print("[apply_fix_leco_sample_replace] 完了")

if __name__ == "__main__":
    apply()
