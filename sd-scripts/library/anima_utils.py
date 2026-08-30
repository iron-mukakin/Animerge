# Anima model loading/saving utilities

import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device  # kept for potential future use
from accelerate import init_empty_weights

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library import anima_models
from library.safetensors_utils import WeightTransformHooks
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Original Anima high-precision keys. Kept for reference, but not used currently.
# # Keys that should stay in high precision (float32/bfloat16, not quantized)
# KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
# ".embed." excludes Embedding in LLMAdapter
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "norm", "adaln", "final_layer", ".embed."]


# --- Anima 3.8B対応: DiTブロック数の自動検出 -------------------------------
# Anima Base 1.0(2Bモデル、28 blocks) / その追加学習版である2.9B preview(40 blocks) /
# さらにその更新版である3.8B(52 blocks)など、複数系統・複数ブロック数のDiTを同一
# コードパスで扱うため、checkpointのヘッダ(テンソル本体は読まない)からブロック総数を検出する。

_BLOCK_INDEX_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def detect_num_blocks_in_checkpoint(dit_path: str) -> int:
    """DiTチェックポイントのヘッダのみを読み取り、'blocks.N.'パターンから
    ブロック総数(最大index + 1)を検出する。テンソル本体は読み込まないため
    大型チェックポイントでも高速・低メモリで動作する。

    Args:
        dit_path: DiT checkpoint(.safetensors)へのパス。

    Returns:
        検出されたブロック総数。

    Raises:
        FileNotFoundError: dit_pathが存在しない場合。
        RuntimeError: 'blocks.N.'パターンのキーが1件も見つからない場合。
    """
    path = Path(dit_path)
    if not path.is_file():
        raise FileNotFoundError(f"DiT checkpointが見つかりません: {dit_path}")

    max_block_index: Optional[int] = None
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            normalized_key = canonical_dit_key(key)
            match = _BLOCK_INDEX_PATTERN.search(normalized_key)
            if match:
                index = int(match.group(1))
                max_block_index = index if max_block_index is None else max(max_block_index, index)

    if max_block_index is None:
        raise RuntimeError(
            f"'blocks.N.'パターンのキーが見つかりませんでした: {dit_path}. "
            "num_blocksを自動検出できません。num_blocks_override引数で明示指定してください。"
        )
    return max_block_index + 1


def detect_progressive_adapter_architecture(adapter_checkpoint_path: str) -> Tuple[int, List[int]]:
    """Progressive Cross Adapterの差分チェックポイントから、
    semantic_source_dim(Qwen3.5 4Bのhidden_size)とlayer_indicesを検出する。

    Args:
        adapter_checkpoint_path: Progressive Cross Adapter checkpoint(.safetensors)へのパス。

    Returns:
        (semantic_source_dim, layer_indices) のタプル。

    Raises:
        FileNotFoundError: adapter_checkpoint_pathが存在しない場合。
        RuntimeError: 必要なキーまたはmetadataが見つからない場合。
    """
    path = Path(adapter_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Progressive Cross Adapter checkpointが見つかりません: {adapter_checkpoint_path}")

    semantic_source_dim: Optional[int] = None
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            if re.search(r"semantic_attentions\.0\.k_proj\.weight$", key):
                # nn.Linear(context_dim, inner_dim).weight の shape は (inner_dim, context_dim)
                semantic_source_dim = handle.get_slice(key).get_shape()[1]
                break

    if semantic_source_dim is None:
        raise RuntimeError(
            f"'semantic_attentions.0.k_proj.weight'が見つかりませんでした: {adapter_checkpoint_path}. "
            "Progressive Cross Adapter形式のcheckpointではない可能性があります。"
        )

    layer_indices_raw = metadata.get("layer_indices")
    if layer_indices_raw is None:
        raise RuntimeError(
            f"checkpointのmetadataに'layer_indices'がありません: {adapter_checkpoint_path}."
        )
    layer_indices = [int(index) for index in json.loads(layer_indices_raw)]

    return int(semantic_source_dim), layer_indices


def read_progressive_adapter_metadata(adapter_checkpoint_path: str) -> Dict[str, str]:
    """Progressive Cross Adapter checkpointのsafetensors metadataのみを読み取る。

    Args:
        adapter_checkpoint_path: checkpointへのパス。

    Returns:
        metadata辞書(存在しない場合は空辞書)。
    """
    with safe_open(str(adapter_checkpoint_path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def wrap_llm_adapter_with_progressive_cross_adapter(
    native_adapter: anima_models.LLMAdapter,
    adapter_checkpoint_path: str,
    device: Union[str, torch.device],
    expected_num_blocks: Optional[int] = None,
) -> anima_models.ProgressiveQwen35CrossAdapter:
    """既存のnative LLMAdapterをProgressiveQwen35CrossAdapterでラップし、
    差分チェックポイント(native_adapter以外の新規パラメータのみ)を割り当てる。

    Args:
        native_adapter: 凍結対象となる既存のLLMAdapter(load_anima_modelで既にロード済みのもの)。
        adapter_checkpoint_path: Progressive Cross Adapterの差分checkpointへのパス。
        device: 割り当て後にモデルを配置するデバイス。
        expected_num_blocks: ロード済みDiTの実際のブロック数。指定時、adapter checkpointの
            metadata内 'new_block_count' と比較し、不一致ならRuntimeErrorを送出する。
            Anima Base 1.0(2B) / 2.9B preview / 3.8B のように複数系統・複数ブロック数の
            DiT checkpointが混在しうるため、誤った組み合わせを早期検出する。

    Returns:
        差分重み割り当て済みのProgressiveQwen35CrossAdapter(native_adapterと同一dtype)。

    Raises:
        RuntimeError: expected_num_blocks指定時にmetadataの'new_block_count'と食い違う場合。
    """
    semantic_source_dim, layer_indices = detect_progressive_adapter_architecture(adapter_checkpoint_path)
    metadata = read_progressive_adapter_metadata(adapter_checkpoint_path)

    if expected_num_blocks is not None and "new_block_count" in metadata:
        adapter_expected_blocks = int(metadata["new_block_count"])
        if adapter_expected_blocks != expected_num_blocks:
            raise RuntimeError(
                f"DiTとProgressive Cross Adapterのブロック数が一致しません: "
                f"DiT側={expected_num_blocks}, adapter checkpointが前提とするブロック数="
                f"{adapter_expected_blocks} (metadata['new_block_count'])。"
                f"Anima Base 1.0(2B)/2.9B preview/3.8Bなど異なる系統のDiTとadapterを"
                f"組み合わせている可能性があります。--pretrained_model_name_or_pathと"
                f"--progressive_adapter_pathの組み合わせを確認してください。"
            )

    logger.info(
        f"Wrapping LLMAdapter with ProgressiveQwen35CrossAdapter "
        f"(semantic_source_dim={semantic_source_dim}, layer_indices={layer_indices})"
    )

    progressive_adapter = anima_models.ProgressiveQwen35CrossAdapter(
        native_adapter=native_adapter,
        semantic_source_dim=semantic_source_dim,
        layer_indices=layer_indices,
    )

    # native_adapterの実dtype(checkpointロード後は通常bf16等)に新規パラメータを合わせる。
    # 合わせない場合、nn.Moduleのデフォルトdtype(float32)のまま構築されるため、
    # forward時に "mat1 and mat2 must have the same dtype" で確実にクラッシュする(実測確認済み)。
    native_dtype = next(native_adapter.parameters()).dtype
    progressive_adapter = progressive_adapter.to(dtype=native_dtype)

    diff_state_dict = load_file(adapter_checkpoint_path, device="cpu")
    progressive_adapter.assign_trainable_state_dict(diff_state_dict)
    return progressive_adapter.to(device)

# --- 2026-05-24: キー名称正規化 (canonical_dit_key) を追加 ---
# Prefixes stripped when normalizing DiT checkpoint keys.
# Mirrors KEY_PREFIXES in merge.py to handle various checkpoint formats
# (ComfyUI "net.", diffusers "model.diffusion_model.", etc.) uniformly.
_ANIMA_KEY_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.model.",
    "model.",
    "module.",
    "state_dict.",
    "net.",
)


def canonical_dit_key(name: str) -> str:
    """Strip known checkpoint prefixes from a DiT weight key.

    Iteratively removes any prefix listed in ``_ANIMA_KEY_PREFIXES`` until
    none remain, then returns the bare key in its original case.
    Equivalent to ``merge.canonical_key`` but defined here to avoid circular
    imports between app and library layers.
    """
    changed = True
    while changed:
        changed = False
        for prefix in _ANIMA_KEY_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


def _make_logging_rename_hook() -> Callable[[str], str]:
    """Return a rename hook that normalizes DiT keys and logs a summary.

    The hook is called once per key during safetensors loading.
    After all keys are processed the hook logs how many keys were renamed
    so the operator can verify normalization without flooding the log.
    """
    renamed: list[tuple[str, str]] = []

    def _hook(key: str) -> str:
        normalized = canonical_dit_key(key)
        if normalized != key:
            renamed.append((key, normalized))
        return normalized

    def _log_summary() -> None:
        if renamed:
            logger.info(
                f"[key normalization] {len(renamed)} key(s) renamed. "
                f"Examples: {renamed[:3]}"
            )
        else:
            logger.info("[key normalization] No keys required renaming.")

    _hook.log_summary = _log_summary  # type: ignore[attr-defined]
    return _hook



def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    num_blocks_override: Optional[int] = None,
    progressive_adapter_path: Optional[str] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (
        not fp8_scaled and dit_weight_dtype is not None
    ) or dit_weight_dtype is None, "dit_weight_dtype should be None when fp8_scaled is True"

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": (
            num_blocks_override if num_blocks_override is not None else detect_num_blocks_in_checkpoint(dit_path)
        ),
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    # 2026-05-24: net. のみ除去から全プレフィックス正規化へ変更
    _rename_hook = _make_logging_rename_hook()
    rename_hooks = WeightTransformHooks(rename_hook=_rename_hook)
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=rename_hooks,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    _rename_hook.log_summary()  # 2026-05-24: 正規化サマリーを出力
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if progressive_adapter_path is not None:
        if not (dit_config.get("use_llm_adapter") and hasattr(model, "llm_adapter")):
            raise RuntimeError(
                "progressive_adapter_pathが指定されましたが、モデルにllm_adapterがありません "
                "(use_llm_adapter=Falseまたは未ロード)。"
            )
        model.llm_adapter = wrap_llm_adapter_with_progressive_cross_adapter(
            model.llm_adapter,
            progressive_adapter_path,
            device=loading_device,
            expected_num_blocks=dit_config["num_blocks"],
        )
        logger.info("LLMAdapter -> ProgressiveQwen35CrossAdapter への置き換えが完了しました。")

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen3_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora_and_fp8(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    fp8_optimization=False,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, "LoRA weights merging is only supported for safetensors checkpoints"
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_qwen35_tokenizer(qwen35_path: str):
    """Qwen3.5 4Bのトークナイザのみをロードする。

    Args:
        qwen35_path: HuggingFace形式ディレクトリ、または単一safetensorsファイルへのパス。
            単一ファイルの場合はconfigs/qwen35_4b/をトークナイザ設定として使用する。

    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen35_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen35_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen35_4b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3.5 config directory not found at {config_dir}. "
                "Expected configs/qwen35_4b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3.5-4B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen35_text_encoder(
    qwen35_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Qwen3.5-4B テキストエンコーダをロードする(Anima 3.8Bのsemantic branch用)。

    load_qwen3_text_encoderと対になる3.8B用ローダー。hidden_size等の次元は
    config.json(configs/qwen35_4b/、またはqwen35_pathがディレクトリの場合はそちら)
    からtransformersが読み取るため、本関数側では一切ハードコードしない。

    Args:
        qwen35_path: HuggingFace形式ディレクトリ、または単一safetensorsファイルへのパス。
        dtype: モデルのdtype。
        device: ロード先デバイス。
        lora_weights: マージするLoRA重み(任意)。
        lora_multipliers: LoRA倍率(任意)。

    Returns:
        (text_encoder_model, tokenizer)

    Raises:
        FileNotFoundError: 単一ファイル指定時にconfigs/qwen35_4b/が存在しない場合。
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3.5 text encoder from {qwen35_path}")

    if os.path.isdir(qwen35_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen35_path, local_files_only=True)
        # Qwen3.5はImage-Text-to-Textのマルチモーダルモデルであり、checkpointの
        # architecturesには通常Qwen3_5ForConditionalGeneration(VL版)が記録されている。
        # AutoModelForCausalLMでは正しく解決できない場合があるため、
        # AutoConfig + get_text_config() でテキスト部分のみを明示的に取り出す。
        full_config = transformers.AutoConfig.from_pretrained(qwen35_path, local_files_only=True)
        text_config = full_config.get_text_config()
        model = transformers.Qwen3_5ForCausalLM.from_pretrained(
            qwen35_path, config=text_config, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen35_4b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3.5 config directory not found at {config_dir}. "
                "Expected configs/qwen35_4b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3.5-4B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        # Qwen3.5はImage-Text-to-Textのマルチモーダルモデルであり、Qwen3_5Configは
        # text_config/vision_configをネストして持つ複合configで、vocab_size等の
        # テキスト固有パラメータをトップレベルには持たない
        # (参考: https://huggingface.co/docs/transformers/model_doc/qwen3_5 ―
        #  "Use Qwen3_5ForCausalLM for text-only generation with Qwen3_5TextConfig")。
        # get_text_config()で複合config/単体text configいずれからもテキスト部分を
        # 汎用的に取り出す。
        full_config = transformers.Qwen3_5Config.from_pretrained(config_dir, local_files_only=True)
        text_config = full_config.get_text_config()
        model = transformers.Qwen3_5ForCausalLM(text_config).model

        if qwen35_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen35_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora_and_fp8(
                    model_files=qwen35_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    fp8_optimization=False,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, "LoRA weights merging is only supported for safetensors checkpoints"
            state_dict = torch.load(qwen35_path, map_location="cpu", weights_only=True)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key[len("model.") :] if key.startswith("model.") else key] = value

        info = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded Qwen3.5 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3.5 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def extract_semantic_hidden_states(
    qwen35_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_indices: Sequence[int],
) -> List[torch.Tensor]:
    """Qwen3.5モデルをforwardし、指定layer_indicesの隠れ状態を抽出する。

    注意: layer_indices は decoder layer通過後のindexを指す前提(Anima 3.8B公式
    checkpointでは[7, 15, 23, 31])。HuggingFaceのoutput_hidden_statesは
    hidden_states[0]が埋め込み層出力であるため、+1したindexで取得する。

    Args:
        qwen35_model: load_qwen35_text_encoderで得たモデル本体。
        input_ids: トークンID列。
        attention_mask: attentionマスク。
        layer_indices: 抽出する隠れ層のindex列。

    Returns:
        layer_indices順の隠れ状態テンソルのリスト。
    """
    outputs = qwen35_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    return [hidden_states[index + 1] for index in layer_indices]


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str, dit_state_dict: Dict[str, torch.Tensor], metadata: Dict[str, any], dtype: Optional[torch.dtype] = None
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        metadata: Metadata dict to include in the safetensors file
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            # v = v.to(dtype)
            v = v.detach().clone().to("cpu").to(dtype)  # Reduce GPU memory usage during save
        prefixed_sd["net." + k] = v.contiguous()

    if metadata is None:
        metadata = {}
    metadata["format"] = "pt"  # For compatibility with the official .safetensors file

    save_file(prefixed_sd, save_path, metadata=metadata)  # safetensors.save_file cosumes a lot of memory, but Anima is small enough
    logger.info(f"Saved Anima model to {save_path}")
