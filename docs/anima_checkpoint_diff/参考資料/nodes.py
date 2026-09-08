from __future__ import annotations

import logging

import comfy.sd
import comfy.utils
import folder_paths

from .mapping import remap_lora


logger = logging.getLogger(__name__)


class _Anima38LoRALoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def _inputs(cls, include_clip):
        required = {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
            }),
        }
        if include_clip:
            required["clip"] = ("CLIP",)
            required["strength_clip"] = ("FLOAT", {
                "default": 1.0,
                "min": -100.0,
                "max": 100.0,
                "step": 0.01,
            })
        return {"required": required}

    def _load(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and (clip is None or strength_clip == 0):
            return model, clip

        block_count = model.model.model_config.unet_config.get("num_blocks")
        if block_count != 52:
            raise ValueError(f"Anima 3.8B LoRA Bridge requires a 52-block MODEL, got {block_count!r}")

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != lora_path:
            lora, metadata = comfy.utils.load_torch_file(
                lora_path,
                safe_load=True,
                return_metadata=True,
            )
            self.loaded_lora = lora_path, lora, metadata
        else:
            _, lora, metadata = self.loaded_lora

        remapped, source_count, remapped_block_keys, skipped_adapter_keys = remap_lora(lora)
        logger.info(
            "Mapped %d-block Anima LoRA %s to Anima 3.8B (%d block tensors, %d internal-adapter tensors skipped)",
            source_count,
            lora_name,
            remapped_block_keys,
            skipped_adapter_keys,
        )
        return comfy.sd.load_lora_for_models(
            model,
            clip,
            remapped,
            strength_model,
            strength_clip,
            lora_metadata=metadata,
        )


class Anima38LoRALoaderModelOnly(_Anima38LoRALoader):
    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(include_clip=False)

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = "loaders/Anima"
    TITLE = "Load Anima LoRA on 3.8B (Model Only)"
    DESCRIPTION = (
        "Loads an Anima Base or 2.9B LoRA on its matching inherited layers in "
        "Anima 3.8B while preserving all 3.8B-specific weights."
    )

    def load_lora_model_only(self, model, lora_name, strength_model):
        model, _ = self._load(model, None, lora_name, strength_model, 0.0)
        return (model,)


class Anima38LoRALoader(_Anima38LoRALoader):
    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(include_clip=True)

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/Anima"
    TITLE = "Load Anima LoRA on 3.8B"
    DESCRIPTION = (
        "Loads an Anima Base or 2.9B LoRA on its matching inherited layers in "
        "Anima 3.8B and applies any text-encoder LoRA keys to CLIP."
    )

    def load_lora(self, model, lora_name, strength_model, clip, strength_clip):
        return self._load(model, clip, lora_name, strength_model, strength_clip)
