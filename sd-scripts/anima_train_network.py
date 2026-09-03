# Anima LoRA training script

import argparse
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    compile_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
import anima_sample_gen
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported. / fp8_baseとfp8_base_unetはサポートされていません。")
            args.fp8_base = False
            args.fp8_base_unet = False

        # 2026-07-18: fp8_scaled(Layer1重み量子化)の強制無効化を撤去。
        # 単体検証(cosine_similarity>=0.999, relative_error<3%)により、
        # nn.Linearベースのself_attn/cross_attn/mlp/adaln_modulation層に対する
        # 数値的整合性を確認済み。ただしBlock Swap(custom_offloading_utils.ModelOffloader)は
        # scale_weightバッファをスワップ対象に含めないため、fp8_scaledとblocks_to_swapは
        # 併用禁止(排他)とする。
        args.fp8_scaled = getattr(args, "fp8_scaled", False)
        if args.fp8_scaled and args.blocks_to_swap is not None and args.blocks_to_swap > 0:
            raise ValueError(
                "fp8_scaled and blocks_to_swap cannot be used together (ModelOffloader does not "
                "swap the scale_weight buffer, which would corrupt dequantization). "
                "/ fp8_scaledとblocks_to_swapは併用できません"
                "(ModelOffloaderはscale_weightバッファをスワップ対象に含めないため、"
                "逆量子化が破綻します)。"
            )

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        if args.compile:
            assert not args.torch_compile, (
                "--compile (per-block torch.compile) and --torch_compile (accelerate dynamo) cannot be used together"
                " / --compile（ブロック単位torch.compile）と--torch_compile（accelerate dynamo）は併用できません"
            )
            assert not (args.compile_fullgraph and args.split_attn), (
                "--compile_fullgraph cannot be used with --split_attn (split attention uses dynamic control flow)"
                " / --compile_fullgraphは--split_attnと併用できません（split attentionは動的な制御フローを使用します）"
            )

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Anima 3.8B: --progressive_adapter_path 指定時、またはDiT checkpoint自体が
        # v1.1(Semantic Connector v2内蔵)の場合は --qwen35 が必須
        # (どちらもQwen3.5 4Bの隠れ状態を必要とするため)
        self.use_progressive_adapter = getattr(args, "progressive_adapter_path", None) is not None
        self.is_semantic_connector_v2 = anima_utils.detect_semantic_connector_v2_architecture(
            args.pretrained_model_name_or_path
        )
        if (self.use_progressive_adapter or self.is_semantic_connector_v2) and not getattr(args, "qwen35", None):
            raise ValueError(
                "--progressive_adapter_path、またはv1.1(Semantic Connector v2内蔵)の"
                "DiT checkpointを指定する場合は --qwen35 も指定してください。"
                " / --qwen35 is required when --progressive_adapter_path is set, "
                "or when the DiT checkpoint bundles Semantic Connector v2 (Anima 3.8B v1.1)."
            )

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        text_encoders: list[nn.Module] = [qwen3_text_encoder]

        # Anima 3.8B: Qwen3.5-4B text encoder (semantic branch)
        if getattr(args, "qwen35", None):
            logger.info("Loading Qwen3.5 text encoder...")
            qwen35_text_encoder, _ = anima_utils.load_qwen35_text_encoder(args.qwen35, dtype=weight_dtype, device="cpu")
            qwen35_text_encoder.eval()
            text_encoders.append(qwen35_text_encoder)

        # Load VAE
        # 既存バグ修正: qwen_image_autoencoder_kl.load_vae()を直接呼ぶと
        # --qwen_image_vae_2d が一切参照されず常に3D実装がロードされてしまうため、
        # 2D/3Dを振り分けるanima_train_utils.load_qwen_image_vae()を経由させる。
        logger.info("Loading Anima VAE...")
        vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", text_encoders, vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
            num_blocks_override=getattr(args, "num_blocks_override", None),
            progressive_adapter_path=getattr(args, "progressive_adapter_path", None),
        )

        # Store unsloth preference so that when the base NetworkTrainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        if args.fp8_scaled:
            # 診断ログ: fp8ストレージが後段(cast_unet等)で上書きされていないか確認するための目印。
            sample_weight = model.blocks[0].self_attn.q_proj.weight
            logger.info(
                f"[fp8_scaled diag] blocks.0.self_attn.q_proj.weight dtype after load: "
                f"{sample_weight.dtype} (expect torch.float8_e4m3fn here)"
            )

        return model, text_encoders

    def cast_unet(self, args):
        # 2026-07-19: fp8_scaled使用時、基底クラスのデフォルト実装(常にTrueを返す)のままだと
        # train_network.train() 内の `unet.to(dtype=unet_weight_dtype)` がfp8常駐重みを
        # bf16へ無条件で上書きし、Layer1量子化によるVRAM削減効果を丸ごと相殺してしまう。
        # fp8_scaled使用時はこの再キャストをスキップし、fp8ストレージを維持する。
        result = not getattr(args, "fp8_scaled", False)
        logger.info(
            f"[fp8_scaled diag] cast_unet() called: args.fp8_scaled="
            f"{getattr(args, 'fp8_scaled', '<attr not set>')!r}, returning {result}"
        )
        return result

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
            qwen35_path=getattr(args, "qwen35", None),
            qwen35_max_length=getattr(args, "qwen35_max_token_length", 512),
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def _resolve_qwen35_layer_indices(self, args) -> Optional[list]:
        """Anima 3.8B: Qwen3.5のlayer_indicesを検出する(結果はキャッシュする)。

        検出元は次の優先順位: (1) --progressive_adapter_path(v1.0、外部adapter)、
        (2) DiT checkpoint自体がv1.1(Semantic Connector v2内蔵)の場合はそちらのmetadata。
        どちらでもない場合はNoneを返し、semantic branchは無効化される
        (既存の2.9B/3.8B-without-adapter動作を維持)。
        """
        if hasattr(self, "_qwen35_layer_indices_cache"):
            return self._qwen35_layer_indices_cache

        adapter_path = getattr(args, "progressive_adapter_path", None)
        if adapter_path:
            _, layer_indices = anima_utils.detect_progressive_adapter_architecture(adapter_path)
        elif anima_utils.detect_semantic_connector_v2_architecture(args.pretrained_model_name_or_path):
            config = anima_utils.detect_anima_v2_connector_config(args.pretrained_model_name_or_path)
            layer_indices = config["layer_indices"]
        else:
            layer_indices = None

        self._qwen35_layer_indices_cache = layer_indices
        return self._qwen35_layer_indices_cache

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy(layer_indices=self._resolve_qwen35_layer_indices(args))

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                False,
                layer_indices=self._resolve_qwen35_layer_indices(args),
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        # Anima 3.8B: text_encoders は [qwen3] または [qwen3, qwen35] のいずれか。
        # 実装計画.txtのVRAM効率化方針: エンコード完了後はQwen3.5もGPUから解放する。
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder(s) to gpu")
            for text_encoder in text_encoders:
                text_encoder.to(accelerator.device)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # sample_prompts_te_outputs は anima_sample_gen が毎回エンコードするため不要
            accelerator.wait_for_everyone()

            # move text encoder(s) back to cpu
            logger.info("move text encoder(s) back to cpu")
            for text_encoder in text_encoders:
                text_encoder.to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder(s) to device for encoding during training/validation
            for text_encoder in text_encoders:
                text_encoder.to(accelerator.device)

    def _is_sample_generation_configured(self, args) -> bool:
        """apply_fix_020: サンプル生成が実際に有効化されているかを判定する。

        is_text_encoder_not_needed_for_training() が text_encoder を削除して
        よいかどうかの判定と、sample_images() 自体の間引きゲート(apply_fix_019)
        の両方から共通で参照し、判定条件を一箇所にまとめる。
        """
        if not getattr(args, "sample_prompts", None):
            return False
        if not getattr(args, "sample_save_dir", None):
            return False
        return bool(
            getattr(args, "sample_at_first", False)
            or getattr(args, "sample_every_n_steps", None)
            or getattr(args, "sample_every_n_epochs", None)
        )

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if not accelerator.is_main_process:
            return

        # apply_fix_019: train_util.sample_images_common() と同一のゲート。
        # これが無いと sample_prompts / sample_every_n_steps / sample_every_n_epochs を
        # 何も指定していなくても毎ステップtext_encoder解決処理まで進んでしまい、
        # cache_text_encoder_outputs使用時は無駄な警告ログが出続ける(学習結果への影響はない)。
        if global_step == 0:
            if not args.sample_at_first:
                return
        else:
            if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
                return
            if args.sample_every_n_epochs is not None:
                # sample_every_n_steps は無視する
                if epoch is None or epoch % args.sample_every_n_epochs != 0:
                    return
            else:
                if global_step % args.sample_every_n_steps != 0 or epoch is not None:
                    return

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        # cache_text_encoder_outputs=True の場合 te=None になる。
        # サンプル生成にはTEが必要なので text_encoders[0] を直接使う。
        # ただし is_text_encoder_not_needed_for_training() が True の場合、
        # 基底クラス(train_network.py)側で text_encoder 自体が None に
        # 置き換えられて削除されているケースがある(apply_fix_016)。
        # text_encoders[0] が None のまま accelerator.unwrap_model() に渡すと
        # 'NoneType' object has no attribute '_modules' でクラッシュするため、
        # None 判定を追加し、次のガードで安全にスキップさせる。
        if qwen3_te is None and text_encoders and text_encoders[0] is not None:
            qwen3_te = accelerator.unwrap_model(text_encoders[0])
        if qwen3_te is None:
            logger.warning("[SampleGen] text_encoder が None のためスキップします")
            return

        # apply_fix_020: Anima 3.8B (semantic branch) 対応。
        # text_encoders[1] が存在すれば Qwen3.5(semantic branch)であり、
        # サンプル生成でも semantic_hidden_states の橋渡しに必要
        # (実処理は anima_sample_gen.py 側、apply_fix_021 で対応)。
        # is_text_encoder_not_needed_for_training() 側でサンプル生成設定時は
        # 削除を抑制しているため通常 None にはならない想定だが、
        # 念のため qwen3_te と同様に None ガードのみ行う。
        qwen35_te = None
        layer_indices = self._resolve_qwen35_layer_indices(args)
        if layer_indices is not None and len(text_encoders) > 1 and text_encoders[1] is not None:
            qwen35_te = accelerator.unwrap_model(text_encoders[1])

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        dit = accelerator.unwrap_model(unet)

        anima_sample_gen.sample_images_from_prompts(
            args=args,
            dit=dit,
            vae=vae,
            text_encoder=qwen3_te,
            semantic_text_encoder=qwen35_te,
            layer_indices=layer_indices,
            tokenize_strategy=tokenize_strategy,
            text_encoding_strategy=text_encoding_strategy,
            accelerator=accelerator,
            epoch=epoch,
            global_step=global_step,
        )
    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        # Anima 3.8B: semantic branch有効時はtext_encoder_conds末尾にsemantic_hidden_states/
        # semantic_attn_maskが追加される(caption_dropout_rateは既にprocess_batch側で除去済み)。
        use_semantic_branch = len(text_encoder_conds) >= 6
        if use_semantic_branch:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask = (
                text_encoder_conds[:6]
            )
        else:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
                :4
            ]  # ignore caption_dropout_rate which is not needed for training step
            semantic_hidden_states = None
            semantic_attn_mask = None

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        semantic_hidden_states_list = None
        if use_semantic_branch:
            semantic_hidden_states = semantic_hidden_states.to(accelerator.device, dtype=weight_dtype)
            semantic_attn_mask = semantic_attn_mask.to(accelerator.device)
            # apply_fix_017: --cache_text_encoder_outputs 使用時、train_util.py の
            # none_or_stack_elements() がバッチ軸を先頭(dim=0)に挿入するため、
            # semantic_hidden_states は (B, num_layers, L, D) になる
            # (ライブエンコード時は (num_layers, B, L, D) のまま dim=0 が層)。
            # 由来に応じて層方向の軸を切り替えてからunbindする。
            semantic_layer_axis = 1 if args.cache_text_encoder_outputs else 0
            semantic_hidden_states_list = list(semantic_hidden_states.unbind(dim=semantic_layer_axis))

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
                semantic_hidden_states=semantic_hidden_states_list,
                semantic_attention_mask=semantic_attn_mask if use_semantic_branch else None,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        return super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
        )

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        # apply_fix_020: サンプル生成が設定されている場合、
        # cache_text_encoder_outputs使用時もtext_encoderを削除しない。
        # anima_sample_gen はキャッシュされた埋め込みではなく毎回ライブエンコードで
        # サンプルを生成する設計のため(cache_text_encoder_outputs_if_needed()の
        # コメント「sample_prompts_te_outputs は anima_sample_gen が毎回エンコード
        # するため不要」参照)、text_encoder実体が必要。
        if self._is_sample_generation_configured(args):
            return False
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            model = unet
            model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        compile_utils.apply_cuda_optimizations(args)

        if args.compile:
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=self.is_swapping_blocks)

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="[EXPERIMENTAL] Use scaled (block-quantized) fp8 weights for the Anima DiT. "
        "Cannot be used together with --blocks_to_swap. "
        "/ [実験的機能] Anima DiTの重みをスケーリングされたfp8(ブロック量子化)にする。"
        "--blocks_to_swapとは併用できません。",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
