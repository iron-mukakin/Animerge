# Anima Strategy Classes

import os
import random
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from library import anima_utils, train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy
from library import qwen_image_autoencoder_kl

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
        qwen35_tokenizer=None,
        qwen35_path: Optional[str] = None,
        qwen35_max_length: int = 512,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

        # Anima 3.8B: Qwen3.5 4B (semantic branch)。未指定ならNoneのまま(=2.9B互換動作)。
        if qwen35_tokenizer is None and qwen35_path is not None:
            qwen35_tokenizer = anima_utils.load_qwen35_tokenizer(qwen35_path)
        self.qwen35_tokenizer = qwen35_tokenizer
        self.qwen35_max_length = qwen35_max_length
        self.use_semantic_branch = self.qwen35_tokenizer is not None

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen3_max_length
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.t5_max_length
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]

        result = [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        if self.use_semantic_branch:
            # Anima 3.8B: Qwen3.5 4B (semantic branch) のトークナイズを追加
            qwen35_encoding = self.qwen35_tokenizer(
                text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen35_max_length
            )
            result.append(qwen35_encoding["input_ids"])
            result.append(qwen35_encoding["attention_mask"])

        return result


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(self, layer_indices: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        # Anima 3.8B: Qwen3.5 4Bから抽出する隠れ層index。Noneなら2.9B互換動作(semantic branch無効)。
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens (+ Anima 3.8B時はQwen3.5トークンも) して埋め込みを返す。

        Args:
            models: [qwen3_text_encoder] または [qwen3_text_encoder, qwen35_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]
                (self.layer_indices指定時は末尾に[qwen35_input_ids, qwen35_attn_mask]が続く)

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
            (self.layer_indices指定時は末尾に[semantic_hidden_states, semantic_attn_mask]が続く。
             semantic_hidden_statesのshapeは(num_layers, B, L, D))
        """
        # Do not handle dropout here; handled dataset-side or in drop_cached_text_encoder_outputs()

        qwen3_text_encoder = models[0]
        use_semantic_branch = self.layer_indices is not None and len(models) > 1

        if use_semantic_branch:
            qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask, qwen35_input_ids, qwen35_attn_mask = tokens
        else:
            qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        if not use_semantic_branch:
            return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        qwen35_text_encoder = models[1]
        semantic_device = qwen35_text_encoder.device
        qwen35_input_ids = qwen35_input_ids.to(semantic_device)
        qwen35_attn_mask = qwen35_attn_mask.to(semantic_device)

        semantic_hidden_states_list = anima_utils.extract_semantic_hidden_states(
            qwen35_text_encoder, qwen35_input_ids, qwen35_attn_mask, self.layer_indices
        )
        semantic_hidden_states = torch.stack(semantic_hidden_states_list, dim=0)  # (num_layers, B, L, D)
        semantic_mask_bool = qwen35_attn_mask.bool()
        semantic_hidden_states[:, ~semantic_mask_bool] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, qwen35_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        semantic_hidden_states: Optional[torch.Tensor] = None,
        semantic_attn_mask: Optional[torch.Tensor] = None,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior. Anima 3.8B時はsemantic側も同時にdropoutする。
        """
        use_semantic_branch = semantic_hidden_states is not None

        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            if use_semantic_branch:
                return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask]
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

        # Clone to avoid in-place modification of cached tensors
        prompt_embeds = prompt_embeds.clone()
        if attn_mask is not None:
            attn_mask = attn_mask.clone()
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.clone()
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.clone()
        if use_semantic_branch:
            semantic_hidden_states = semantic_hidden_states.clone()
            if semantic_attn_mask is not None:
                semantic_attn_mask = semantic_attn_mask.clone()

        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                # Use pre-cached unconditional embeddings
                prompt_embeds[i] = 0
                if attn_mask is not None:
                    attn_mask[i] = 0
                if t5_input_ids is not None:
                    t5_input_ids[i, 0] = 1  # Set to </s> token ID
                    t5_input_ids[i, 1:] = 0
                if t5_attn_mask is not None:
                    t5_attn_mask[i, 0] = 1
                    t5_attn_mask[i, 1:] = 0
                if use_semantic_branch:
                    # apply_fix_018: このメソッドが受け取るsemantic_hidden_statesは
                    # train_util.py の none_or_stack_elements() でバッチ化された後の
                    # (B, num_layers, L, D) であり、dim=0がバッチ軸
                    # (ライブエンコード直後の (num_layers, B, L, D) とは軸の意味が異なる)。
                    semantic_hidden_states[i] = 0
                    if semantic_attn_mask is not None:
                        semantic_attn_mask[i] = 0

        if use_semantic_branch:
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask]
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (float), attn_mask (int), t5_input_ids (int), t5_attn_mask (int)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        # Anima 3.8B: Qwen3.5から抽出する隠れ層index。Noneなら2.9B互換動作。
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        required_keys = ["prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask", "caption_dropout_rate"]
        if self.layer_indices is not None:
            # Anima 3.8B: semantic branch用キーが無いキャッシュ(2.9B時代のもの含む)は再キャッシュさせる
            required_keys += ["semantic_hidden_states", "semantic_attn_mask"]

        try:
            npz = np.load(npz_path)
            for key in required_keys:
                if key not in npz:
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        prompt_embeds = data["prompt_embeds"]
        attn_mask = data["attn_mask"]
        t5_input_ids = data["t5_input_ids"]
        t5_attn_mask = data["t5_attn_mask"]
        caption_dropout_rate = data["caption_dropout_rate"]

        if "semantic_hidden_states" in data:
            semantic_hidden_states = data["semantic_hidden_states"]
            semantic_attn_mask = data["semantic_attn_mask"]
            return [
                prompt_embeds,
                attn_mask,
                t5_input_ids,
                t5_attn_mask,
                semantic_hidden_states,
                semantic_attn_mask,
                caption_dropout_rate,
            ]
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            encoded = anima_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        use_semantic_branch = self.layer_indices is not None and len(encoded) > 4
        if use_semantic_branch:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, semantic_hidden_states, semantic_attn_mask = encoded
        else:
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded
            semantic_hidden_states = None
            semantic_attn_mask = None

        # Convert to numpy for caching
        if prompt_embeds.dtype == torch.bfloat16:
            prompt_embeds = prompt_embeds.float()
        prompt_embeds = prompt_embeds.cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()
        t5_input_ids = t5_input_ids.cpu().numpy().astype(np.int32)
        t5_attn_mask = t5_attn_mask.cpu().numpy().astype(np.int32)

        if use_semantic_branch:
            if semantic_hidden_states.dtype == torch.bfloat16:
                semantic_hidden_states = semantic_hidden_states.float()
            semantic_hidden_states = semantic_hidden_states.cpu().numpy()  # (num_layers, B, L, D)
            semantic_attn_mask = semantic_attn_mask.cpu().numpy().astype(np.int32)

        for i, info in enumerate(infos):
            prompt_embeds_i = prompt_embeds[i]
            attn_mask_i = attn_mask[i]
            t5_input_ids_i = t5_input_ids[i]
            t5_attn_mask_i = t5_attn_mask[i]
            caption_dropout_rate = torch.tensor(info.caption_dropout_rate, dtype=torch.float32)

            save_kwargs = dict(
                prompt_embeds=prompt_embeds_i,
                attn_mask=attn_mask_i,
                t5_input_ids=t5_input_ids_i,
                t5_attn_mask=t5_attn_mask_i,
                caption_dropout_rate=caption_dropout_rate,
            )
            result_tuple = (prompt_embeds_i, attn_mask_i, t5_input_ids_i, t5_attn_mask_i)

            if use_semantic_branch:
                semantic_hidden_states_i = semantic_hidden_states[:, i]  # (num_layers, L, D)
                semantic_attn_mask_i = semantic_attn_mask[i]
                save_kwargs["semantic_hidden_states"] = semantic_hidden_states_i
                save_kwargs["semantic_attn_mask"] = semantic_attn_mask_i
                result_tuple = result_tuple + (semantic_hidden_states_i, semantic_attn_mask_i)

            result_tuple = result_tuple + (caption_dropout_rate,)

            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, **save_kwargs)
            else:
                info.text_encoder_outputs = result_tuple


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using WanVAE.

    WanVAE produces 16-channel latents with spatial downscale 8x.
    Latent shape for images: (B, 16, 1, H/8, W/8)
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.ANIMA_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        """Cache batch of latents using Qwen Image VAE.

        vae is expected to be the Qwen Image VAE (AutoencoderKLQwenImage).
        The encoding function handles the mean/std normalization.
        """
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            """Encode image tensor to latents.

            img_tensor: (B, C, H, W) in [-1, 1] range (already normalized by IMAGE_TRANSFORMS)
            Qwen Image VAE accepts inputs in (B, C, H, W) or (B, C, 1, H, W) shape.
            Returns latents in (B, 16, 1, H/8, W/8) shape on CPU.
            """
            latents = vae.encode_pixels_to_latents(img_tensor)  # Keep 4D for input/output
            return latents.to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
