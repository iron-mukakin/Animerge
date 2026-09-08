"""
model_merge_anima.py

Merge two Anima MODELs with automatic handling of the 28-block <-> 40-block
(Anima-2.9B) architecture mismatch.

Inputs are named model_1 (top) and model_2 (bottom) to match the visual
slot order in ComfyUI. merge_ratio is the weight of model_1:
    merge_ratio = 1.0  -> output's shared/old-block weights = 100% model_1
    merge_ratio = 0.0  -> output's shared/old-block weights = 100% model_2
    merge_ratio = 0.5  -> 50/50 blend

Architecture handling:
    - If both models have the SAME block count (both 28-block old-Anima
      derivatives, or both 40-block Anima-2.9B derivatives): a direct
      key-for-key merge is done, no remapping needed. Output has that
      same block count.
    - If block counts DIFFER (one 28-block, one 40-block): the output is
      always the 40-block architecture. The 40-block model supplies the
      base. The 28-block model's weights are remapped onto their
      corresponding 40-block indices (via the bundled expand_manifest.json)
      and blended in at the old (shared) block positions according to
      merge_ratio.

      The 12 newly-inserted blocks have no counterpart in a 28-block
      model, so by default (extend_ratio=0.0) they are preserved
      untouched at 100% their 40-block value, regardless of merge_ratio.
      Setting extend_ratio > 0.0 additionally blends in the 28-block
      model's weight from whichever original block each inserted block
      was copied from at initialization (expand_manifest.json's
      "inserted_to_source"): new_layer = (1 - extend_ratio) * 2.9B's own
      value + extend_ratio * the 28-block model's corresponding source
      layer. There's no "correct" answer for this -- it's an experimental,
      best-effort approximation, off by default.

Bypassing the node (ComfyUI's Mode: Bypass) falls back to ComfyUI's
default behavior for a two-MODEL-input / one-MODEL-output node: the
first matching input (model_1, the top slot) is passed straight through.
No special code is needed for that -- it's standard ComfyUI behavior.
"""

import logging

from .anima_common import (
    remap_key,
    remap_key_to_target,
    list_manifest_choices,
    resolve_manifest,
    build_base_to_target,
    build_source_to_inserted_targets,
    get_model_block_count,
)

logger = logging.getLogger("AnimaModelMerge")


def remap_key_patches(key_patches, base_to_target):
    """Apply remap_key() across a dict of {key: patch} from ModelPatcher.get_key_patches()."""
    remapped = {}
    dropped = 0
    for k, v in key_patches.items():
        new_k, _ = remap_key(k, base_to_target)
        if new_k is None:
            dropped += 1
            continue
        remapped[new_k] = v
    return remapped, dropped


def build_extended_patches(old_key_patches, source_to_inserted):
    """
    Project the 28-block model's own weights onto the 12 newly-inserted
    target positions, using expand_manifest.json's record of which base
    block each inserted block was originally copied from. Used for the
    experimental extend_ratio feature.
    """
    extended = {}
    for k, v in old_key_patches.items():
        _, base_idx = remap_key(k, {})  # empty map: just extracts base_idx, ignores match result
        if base_idx is None:
            continue
        target_list = source_to_inserted.get(base_idx)
        if not target_list:
            continue
        for t_idx in target_list:
            new_k = remap_key_to_target(k, t_idx)
            if new_k is None:
                continue
            extended[new_k] = v
    return extended


class AnimaModelMerge:
    """
    Merge two Anima models (MODEL) with merge_ratio as the weight of
    model_1 (top input). Automatically detects a 28-block/40-block
    (Anima-2.9B) architecture mismatch and remaps + outputs at 40 blocks
    in that case; otherwise merges directly at whatever block count both
    inputs share.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_manifest_choices()
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extend_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "manifest": (manifests,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "loaders/anima"

    def merge(self, model_1, model_2, merge_ratio, extend_ratio, manifest):
        block_count_1 = get_model_block_count(model_1)
        block_count_2 = get_model_block_count(model_2)
        logger.info(f"model_1: {block_count_1} blocks, model_2: {block_count_2} blocks, merge_ratio={merge_ratio}")

        # --- Case 1: same architecture on both sides -> direct merge, no remap ---
        if block_count_1 is not None and block_count_1 == block_count_2:
            m = model_1.clone()
            kp2 = model_2.get_key_patches("diffusion_model.")
            m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
            logger.info(f"Direct {block_count_1}-block merge (no remap needed). Output: {block_count_1} blocks.")
            return (m,)

        # --- Case 2: mismatched architectures -> resolve the manifest for
        # THIS SPECIFIC (smaller, larger) pair. The larger side is always
        # the output base; its newly-inserted blocks default to untouched. ---
        if block_count_1 is not None and block_count_2 is not None:
            smaller = min(block_count_1, block_count_2)
            larger = max(block_count_1, block_count_2)
            manifest_data = resolve_manifest(manifest, smaller, larger)
            base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
            source_to_inserted = build_source_to_inserted_targets(manifest_data) if manifest_data else {}

            if base_to_target:
                if block_count_1 == larger:
                    m = model_1.clone()
                    kp2 = model_2.get_key_patches("diffusion_model.")
                    remapped_kp2, dropped = remap_key_patches(kp2, base_to_target)
                    m.add_patches(remapped_kp2, 1.0 - merge_ratio, merge_ratio)
                    logger.info(
                        f"model_1={larger}(base), model_2={smaller}(old): remapped {len(remapped_kp2)} keys "
                        f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio} "
                        f"(model_1 weight). Output: {larger} blocks."
                    )
                    if extend_ratio > 0.0 and source_to_inserted:
                        extended = build_extended_patches(kp2, source_to_inserted)
                        m.add_patches(extended, extend_ratio, 1.0 - extend_ratio)
                        logger.info(
                            f"[experimental] extended {len(extended)} tensors from model_2 (old) onto "
                            f"newly-inserted layers, extend_ratio={extend_ratio}"
                        )
                    return (m,)
                else:
                    # merge_ratio still means "weight of model_1", so strengths
                    # are swapped relative to the branch above.
                    m = model_2.clone()
                    kp1 = model_1.get_key_patches("diffusion_model.")
                    remapped_kp1, dropped = remap_key_patches(kp1, base_to_target)
                    m.add_patches(remapped_kp1, merge_ratio, 1.0 - merge_ratio)
                    logger.info(
                        f"model_1={smaller}(old), model_2={larger}(base): remapped {len(remapped_kp1)} keys "
                        f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio} "
                        f"(model_1 weight). Output: {larger} blocks."
                    )
                    if extend_ratio > 0.0 and source_to_inserted:
                        extended = build_extended_patches(kp1, source_to_inserted)
                        m.add_patches(extended, extend_ratio, 1.0 - extend_ratio)
                        logger.info(
                            f"[experimental] extended {len(extended)} tensors from model_1 (old) onto "
                            f"newly-inserted layers, extend_ratio={extend_ratio}"
                        )
                    return (m,)

        # --- Fallback: unrecognized/unmapped block counts -> best-effort direct merge ---
        logger.warning(
            f"No manifest covers (model_1={block_count_1}, model_2={block_count_2}) blocks. "
            f"Falling back to an unremapped direct merge -- results may be incorrect if the "
            f"architectures actually differ."
        )
        m = model_1.clone()
        kp2 = model_2.get_key_patches("diffusion_model.")
        m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaModelMerge": AnimaModelMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelMerge": "Anima Model Merge (Auto Remap)",
}
