"""
apply_fix_lora_sample_replace.py
----------------------------------
anima_train_network.py の変更:
  1. import に anima_sample_gen を追加
  2. sample_prompts_te_outputs キャッシュブロックを削除
  3. sample_images メソッドを anima_sample_gen 呼び出しに差し替え
     cache_text_encoder_outputs=True 時も TE を取得して渡す

実行場所: プロジェクトルート
  python apply_fix_lora_sample_replace.py
"""
import sys, time, shutil
from pathlib import Path

TARGET = Path("sd-scripts") / "anima_train_network.py"

def _adapt(s): return s.replace("\r\n", "\n")

OLD_IMPORT = 'from library import (\n    anima_models,\n    anima_train_utils,\n    anima_utils,\n    flux_train_utils,\n    qwen_image_autoencoder_kl,\n    sd3_train_utils,\n    strategy_anima,\n    strategy_base,\n    train_util,\n)'
NEW_IMPORT = 'from library import (\n    anima_models,\n    anima_train_utils,\n    anima_utils,\n    flux_train_utils,\n    qwen_image_autoencoder_kl,\n    sd3_train_utils,\n    strategy_anima,\n    strategy_base,\n    train_util,\n)\nimport anima_sample_gen'
OLD_CACHE  = '            # cache sample prompts\n            if args.sample_prompts is not None:\n                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")\n\n                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()\n                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()\n\n                prompts = train_util.load_prompts(args.sample_prompts)\n                sample_prompts_te_outputs = {}\n                with accelerator.autocast(), torch.no_grad():\n                    for prompt_dict in prompts:\n                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:\n                            if p not in sample_prompts_te_outputs:\n                                logger.info(f"  cache TE outputs for: {p}")\n                                tokens_and_masks = tokenize_strategy.tokenize(p)\n                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(\n                                    tokenize_strategy, text_encoders, tokens_and_masks\n                                )\n                self.sample_prompts_te_outputs = sample_prompts_te_outputs\n\n            accelerator.wait_for_everyone()'
NEW_CACHE  = '            # sample_prompts_te_outputs は anima_sample_gen が毎回エンコードするため不要\n            accelerator.wait_for_everyone()'
OLD_SAMPLE = '    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):\n        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility\n        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)\n        qwen3_te = te[0] if te is not None else None\n\n        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()\n        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()\n        anima_train_utils.sample_images(\n            accelerator,\n            args,\n            epoch,\n            global_step,\n            unet,\n            vae,\n            qwen3_te,\n            tokenize_strategy,\n            text_encoding_strategy,\n            self.sample_prompts_te_outputs,\n        )\n'
NEW_SAMPLE = '    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):\n        if not accelerator.is_main_process:\n            return\n\n        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]\n        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)\n        qwen3_te = te[0] if te is not None else None\n\n        # cache_text_encoder_outputs=True の場合 te=None になる。\n        # サンプル生成にはTEが必要なので text_encoders[0] を直接使う。\n        if qwen3_te is None and text_encoders:\n            qwen3_te = accelerator.unwrap_model(text_encoders[0])\n        if qwen3_te is None:\n            logger.warning("[SampleGen] text_encoder が None のためスキップします")\n            return\n\n        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()\n        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()\n        dit = accelerator.unwrap_model(unet)\n\n        anima_sample_gen.sample_images_from_prompts(\n            args=args,\n            dit=dit,\n            vae=vae,\n            text_encoder=qwen3_te,\n            tokenize_strategy=tokenize_strategy,\n            text_encoding_strategy=text_encoding_strategy,\n            accelerator=accelerator,\n            epoch=epoch,\n            global_step=global_step,\n        )'

def apply():
    if not TARGET.exists():
        print(f"[ERROR] ファイルが見つかりません: {TARGET}"); sys.exit(1)
    content = _adapt(TARGET.read_text(encoding="utf-8"))
    patches = [
        (OLD_IMPORT, NEW_IMPORT, "import に anima_sample_gen を追加"),
        (OLD_CACHE,  NEW_CACHE,  "sample_prompts_te_outputs キャッシュブロックを削除"),
        (OLD_SAMPLE, NEW_SAMPLE, "sample_images メソッドを差し替え"),
    ]
    for old, new, desc in patches:
        c = content.count(_adapt(old))
        if c != 1:
            print(f"[ERROR] '{desc}' のOLD文字列が{c}件 → 中断"); sys.exit(1)
        content = content.replace(_adapt(old), _adapt(new))
        print(f"  OK: {desc}")
    bak = TARGET.with_suffix(f".bak_{int(time.time())}")
    shutil.copy2(TARGET, bak)
    print(f"  バックアップ: {bak}")
    TARGET.write_text(content, encoding="utf-8", newline="\n")
    print(f"  書き込み完了: {TARGET}")
    print("[apply_fix_lora_sample_replace] 完了")

if __name__ == "__main__":
    apply()
