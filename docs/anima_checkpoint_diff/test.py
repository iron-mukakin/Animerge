from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from safetensors import safe_open


# ============================================================
# Constants
# ============================================================

DEFAULT_V11_INSERTION_INDICES = [
    3, 7, 11, 15, 19, 23,
    27, 31, 35, 39, 43, 47
]

V11_QWEN35_LAYERS = [7, 15, 23, 31]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class TensorInfo:
    key: str
    shape: list[int]
    dtype: str
    numel: int
    nbytes: Optional[int]
    category: str
    block_index: Optional[int]
    submodule: str


@dataclass
class KeyDiff:
    key: str
    status: str

    v10_shape: Optional[list[int]] = None
    v11_shape: Optional[list[int]] = None

    v10_dtype: Optional[str] = None
    v11_dtype: Optional[str] = None

    v10_numel: Optional[int] = None
    v11_numel: Optional[int] = None


# ============================================================
# dtype sizes
# ============================================================

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


# ============================================================
# Utility
# ============================================================

def product(values):
    result = 1

    for value in values:
        result *= int(value)

    return result


def human_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "N/A"

    value = float(num_bytes)

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PiB"


def human_params(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f} B"

    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f} M"

    if num_params >= 1_000:
        return f"{num_params / 1_000:.3f} K"

    return str(num_params)


def safe_json(value):
    """
    Safetensors metadata values are normally strings.
    Try to parse JSON-looking strings for easier reporting.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return value

    if text[0] in "[{":
        try:
            return json.loads(text)
        except Exception:
            pass

    if text.lower() in ("true", "false"):
        return text.lower() == "true"

    return value


# ============================================================
# Key classification
# ============================================================

def classify_key(key: str):
    """
    Classify a tensor key into architecture/module categories.
    """

    # --------------------------------------------------------
    # v1.1 bundled adapter
    # --------------------------------------------------------

    if "anima_v2_connector" in key:
        if "quality_anchor" in key:
            return "v2.quality_anchor"

        if "semantic_resampler" in key:
            return "v2.semantic_resampler"

        if (
            "v2_attentions" in key
            or "v2_query_norms" in key
            or "v2_semantic_norms" in key
        ):
            return "v2.inserted_semantic"

        return "v2.connector_other"

    # --------------------------------------------------------
    # llm adapter
    # --------------------------------------------------------

    if "llm_adapter" in key:

        m = re.search(
            r"llm_adapter\.blocks\.(\d+)",
            key
        )

        if m:
            return "llm_adapter"

        return "llm_adapter_other"

    # --------------------------------------------------------
    # legacy adapter
    # --------------------------------------------------------

    if (
        "progressive" in key.lower()
        or "cross_adapter" in key.lower()
        or "expanded_adapter" in key.lower()
    ):
        return "legacy_adapter"

    # --------------------------------------------------------
    # DiT blocks
    # --------------------------------------------------------

    patterns = [
        r"(?:^|\.)(?:blocks)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:transformer_blocks)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:diffusion_blocks)\.(\d+)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, key)

        if match:
            return "dit_block"

    # --------------------------------------------------------
    # VAE / text encoder etc.
    # --------------------------------------------------------

    if "vae" in key.lower():
        return "vae"

    if "text_encoder" in key.lower():
        return "text_encoder"

    return "other"


def extract_block_index(key: str) -> Optional[int]:

    patterns = [
        r"(?:^|\.)(?:blocks)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:transformer_blocks)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:diffusion_blocks)\.(\d+)(?:\.|$)",
    ]

    for pattern in patterns:

        match = re.search(pattern, key)

        if match:
            return int(match.group(1))

    return None


def extract_submodule(key: str) -> str:

    category = classify_key(key)

    if category == "dit_block":

        block = extract_block_index(key)

        if block is None:
            return key

        pattern = re.search(
            rf"(?:blocks|transformer_blocks|diffusion_blocks)\.{block}\.(.+)",
            key
        )

        if pattern:
            return pattern.group(1)

    return key


# ============================================================
# Load checkpoint header information
# ============================================================

def load_checkpoint_info(path: Path):

    print()
    print("=" * 80)
    print(f"Reading: {path}")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(path)

    result = {
        "path": str(path.resolve()),
        "filename": path.name,
        "filesize": path.stat().st_size,
        "metadata": {},
        "tensors": {},
    }

    with safe_open(
        str(path),
        framework="pt",
        device="cpu",
    ) as f:

        metadata = f.metadata()

        if metadata:
            result["metadata"] = {
                key: safe_json(value)
                for key, value in metadata.items()
            }

        keys = list(f.keys())

        print(f"Tensor count: {len(keys)}")

        for index, key in enumerate(keys, start=1):

            tensor_slice = f.get_slice(key)

            shape = list(tensor_slice.get_shape())

            # dtype is not exposed by all safetensors versions
            # through get_slice, therefore use a small fallback.
            dtype = get_tensor_dtype(f, key)

            numel = product(shape)

            nbytes = None

            if dtype in DTYPE_BYTES:
                nbytes = numel * DTYPE_BYTES[dtype]

            category = classify_key(key)

            block_index = extract_block_index(key)

            submodule = extract_submodule(key)

            result["tensors"][key] = TensorInfo(
                key=key,
                shape=shape,
                dtype=dtype,
                numel=numel,
                nbytes=nbytes,
                category=category,
                block_index=block_index,
                submodule=submodule,
            )

            if index % 250 == 0:
                print(f"  processed {index}/{len(keys)}")

    return result


def get_tensor_dtype(f, key):

    """
    Retrieve dtype without keeping the tensor.

    safetensors versions differ slightly in available metadata APIs.
    The fallback uses get_tensor() only for this individual tensor if
    necessary. This should be avoided for huge tensors where possible.
    """

    try:
        tensor = f.get_tensor(key)
        return str(tensor.dtype).replace("torch.", "").upper()

    except Exception:
        return "UNKNOWN"


# ============================================================
# Metadata comparison
# ============================================================

def compare_metadata(meta10, meta11):

    keys = sorted(
        set(meta10.keys()) |
        set(meta11.keys())
    )

    result = []

    for key in keys:

        v10 = meta10.get(key, "<MISSING>")
        v11 = meta11.get(key, "<MISSING>")

        if v10 == v11:
            status = "same"

        elif key not in meta10:
            status = "v1.1_only"

        elif key not in meta11:
            status = "v1.0_only"

        else:
            status = "changed"

        result.append({
            "key": key,
            "status": status,
            "v1.0": v10,
            "v1.1": v11,
        })

    return result


# ============================================================
# Tensor key comparison
# ============================================================

def compare_tensors(t10, t11):

    keys10 = set(t10.keys())
    keys11 = set(t11.keys())

    all_keys = sorted(keys10 | keys11)

    result = []

    for key in all_keys:

        a = t10.get(key)
        b = t11.get(key)

        if a is None:

            result.append(
                KeyDiff(
                    key=key,
                    status="v1.1_only",
                    v11_shape=b.shape,
                    v11_dtype=b.dtype,
                    v11_numel=b.numel,
                )
            )

            continue

        if b is None:

            result.append(
                KeyDiff(
                    key=key,
                    status="v1.0_only",
                    v10_shape=a.shape,
                    v10_dtype=a.dtype,
                    v10_numel=a.numel,
                )
            )

            continue

        shape_changed = a.shape != b.shape
        dtype_changed = a.dtype != b.dtype

        if shape_changed or dtype_changed:

            result.append(
                KeyDiff(
                    key=key,
                    status="changed",
                    v10_shape=a.shape,
                    v11_shape=b.shape,
                    v10_dtype=a.dtype,
                    v11_dtype=b.dtype,
                    v10_numel=a.numel,
                    v11_numel=b.numel,
                )
            )

        else:

            result.append(
                KeyDiff(
                    key=key,
                    status="same",
                    v10_shape=a.shape,
                    v11_shape=b.shape,
                    v10_dtype=a.dtype,
                    v11_dtype=b.dtype,
                    v10_numel=a.numel,
                    v11_numel=b.numel,
                )
            )

    return result


# ============================================================
# Category statistics
# ============================================================

def category_statistics(checkpoint):

    result = defaultdict(
        lambda: {
            "tensor_count": 0,
            "parameters": 0,
            "bytes": 0,
            "blocks": set(),
        }
    )

    for info in checkpoint["tensors"].values():

        entry = result[info.category]

        entry["tensor_count"] += 1
        entry["parameters"] += info.numel

        if info.nbytes:
            entry["bytes"] += info.nbytes

        if info.block_index is not None:
            entry["blocks"].add(info.block_index)

    for entry in result.values():
        entry["blocks"] = sorted(entry["blocks"])

    return dict(result)


# ============================================================
# DiT block analysis
# ============================================================

def analyze_dit_blocks(checkpoint):

    blocks = defaultdict(list)

    for key, info in checkpoint["tensors"].items():

        if info.category != "dit_block":
            continue

        if info.block_index is None:
            continue

        blocks[info.block_index].append(info)

    result = {}

    for block_index in sorted(blocks):

        tensors = blocks[block_index]

        result[str(block_index)] = {
            "tensor_count": len(tensors),
            "parameters": sum(
                x.numel
                for x in tensors
            ),
            "bytes": sum(
                x.nbytes or 0
                for x in tensors
            ),
            "tensor_keys": sorted(
                x.key
                for x in tensors
            ),
            "submodules": sorted(
                set(x.submodule for x in tensors)
            ),
        }

    return result


# ============================================================
# Module hierarchy
# ============================================================

def build_module_tree(checkpoint):

    tree = {}

    for key, info in checkpoint["tensors"].items():

        parts = key.split(".")

        node = tree

        for part in parts:

            if part not in node:
                node[part] = {
                    "_tensors": 0,
                    "_parameters": 0,
                    "_children": {},
                }

            current = node[part]

            current["_tensors"] += 1
            current["_parameters"] += info.numel

            node = current["_children"]

    return tree


# ============================================================
# Insertion index analysis
# ============================================================

def parse_layer_indices(metadata):

    candidate_keys = [
        "anima_v2_adapter_layer_indices",
        "layer_indices",
    ]

    for key in candidate_keys:

        if key not in metadata:
            continue

        value = metadata[key]

        if isinstance(value, list):
            return value

        if isinstance(value, str):

            try:
                parsed = json.loads(value)

                if isinstance(parsed, list):
                    return parsed

            except Exception:
                pass

            numbers = re.findall(
                r"\d+",
                value
            )

            if numbers:
                return [
                    int(x)
                    for x in numbers
                ]

    return None


def analyze_insertion_indices(metadata, block_info):

    indices = parse_layer_indices(metadata)

    if indices is None:
        indices = DEFAULT_V11_INSERTION_INDICES

    result = []

    all_blocks = sorted(
        int(x)
        for x in block_info.keys()
    )

    for index in indices:

        result.append({
            "index": index,
            "exists_in_dit": index in all_blocks,
            "block_parameter_count":
                block_info.get(
                    str(index),
                    {}
                ).get(
                    "parameters",
                    0
                ),
        })

    return result


# ============================================================
# Special v1.1 structure detection
# ============================================================

def analyze_v11_structure(checkpoint):

    tensors = checkpoint["tensors"]

    categories = Counter(
        info.category
        for info in tensors.values()
    )

    result = {
        "has_llm_adapter":
            any(
                info.category == "llm_adapter"
                for info in tensors.values()
            ),

        "has_quality_anchor":
            any(
                info.category == "v2.quality_anchor"
                for info in tensors.values()
            ),

        "has_semantic_resampler":
            any(
                info.category == "v2.semantic_resampler"
                for info in tensors.values()
            ),

        "has_inserted_semantic":
            any(
                info.category == "v2.inserted_semantic"
                for info in tensors.values()
            ),

        "categories": dict(categories),
    }

    return result


# ============================================================
# Optional weight-value comparison
# ============================================================

def compare_tensor_values(
    path10,
    path11,
    common_keys,
    limit=None,
):

    """
    Optional expensive comparison.

    Loads only common tensors one at a time.

    Returns:
        max_abs_diff
        mean_abs_diff
        cosine_similarity
    """

    import torch

    results = []

    keys = common_keys

    if limit is not None:
        keys = keys[:limit]

    with safe_open(
        str(path10),
        framework="pt",
        device="cpu",
    ) as f10:

        with safe_open(
            str(path11),
            framework="pt",
            device="cpu",
        ) as f11:

            for index, key in enumerate(keys, start=1):

                try:

                    a = f10.get_tensor(key).float()
                    b = f11.get_tensor(key).float()

                    if a.shape != b.shape:
                        continue

                    diff = (a - b).abs()

                    max_abs = float(
                        diff.max().item()
                    )

                    mean_abs = float(
                        diff.mean().item()
                    )

                    a_flat = a.reshape(-1)
                    b_flat = b.reshape(-1)

                    cosine = float(
                        torch.nn.functional.cosine_similarity(
                            a_flat.unsqueeze(0),
                            b_flat.unsqueeze(0),
                        ).item()
                    )

                    results.append({
                        "key": key,
                        "max_abs_diff": max_abs,
                        "mean_abs_diff": mean_abs,
                        "cosine_similarity": cosine,
                    })

                except Exception as exc:

                    results.append({
                        "key": key,
                        "error": str(exc),
                    })

                if index % 25 == 0:
                    print(
                        f"value comparison: "
                        f"{index}/{len(keys)}"
                    )

    return results


# ============================================================
# Report generation
# ============================================================

def write_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def write_csv(path, rows):

    if not rows:
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def write_text_report(
    path,
    report,
):

    lines = []

    lines.append(
        "Anima v1.0 / v1.1 Checkpoint Structure Diff"
    )

    lines.append("=" * 80)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    lines.append("")
    lines.append("[FILES]")

    lines.append(
        f"v1.0: {report['v1.0']['filename']}"
    )

    lines.append(
        f"v1.1: {report['v1.1']['filename']}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = report["summary"]

    lines.append("")
    lines.append("[SUMMARY]")

    for key, value in summary.items():
        lines.append(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    lines.append("")
    lines.append("[METADATA DIFF]")

    for item in report["metadata_diff"]:

        if item["status"] == "same":
            continue

        lines.append("")
        lines.append(
            f"{item['status']}: {item['key']}"
        )

        lines.append(
            f"  v1.0: {item['v1.0']}"
        )

        lines.append(
            f"  v1.1: {item['v1.1']}"
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    lines.append("")
    lines.append("[CATEGORY STATISTICS]")

    for version in ["v1.0", "v1.1"]:

        lines.append("")
        lines.append(version)

        for category, data in report[
            version
        ]["categories"].items():

            lines.append(
                f"  {category:30} "
                f"tensors={data['tensor_count']:5} "
                f"params={human_params(data['parameters']):>12} "
                f"bytes={human_bytes(data['bytes'])}"
            )

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    lines.append("")
    lines.append("[DIT BLOCKS]")

    blocks10 = report[
        "v1.0"
    ]["dit_blocks"]

    blocks11 = report[
        "v1.1"
    ]["dit_blocks"]

    all_blocks = sorted(
        set(blocks10.keys()) |
        set(blocks11.keys()),
        key=int,
    )

    for block in all_blocks:

        a = blocks10.get(block)
        b = blocks11.get(block)

        lines.append("")
        lines.append(
            f"Block {block}"
        )

        if a is None:
            lines.append(
                "  v1.0: MISSING"
            )
        else:
            lines.append(
                f"  v1.0: "
                f"{human_params(a['parameters'])}, "
                f"{a['tensor_count']} tensors"
            )

        if b is None:
            lines.append(
                "  v1.1: MISSING"
            )
        else:
            lines.append(
                f"  v1.1: "
                f"{human_params(b['parameters'])}, "
                f"{b['tensor_count']} tensors"
            )

    # --------------------------------------------------------
    # Insertion
    # --------------------------------------------------------

    lines.append("")
    lines.append("[V1.1 INSERTION INDICES]")

    for item in report[
        "v1.1_insertion_analysis"
    ]:

        lines.append(
            f"  Block {item['index']}: "
            f"exists={item['exists_in_dit']}, "
            f"params={item['block_parameter_count']}"
        )

    # --------------------------------------------------------
    # Tensor diff
    # --------------------------------------------------------

    lines.append("")
    lines.append("[TENSOR KEY DIFF]")

    status_counter = Counter(
        x["status"]
        for x in report["tensor_diff"]
    )

    for status, count in sorted(
        status_counter.items()
    ):

        lines.append(
            f"{status}: {count}"
        )

    for item in report["tensor_diff"]:

        if item["status"] == "same":
            continue

        lines.append("")
        lines.append(
            f"{item['status']}: {item['key']}"
        )

        if item.get("v1.0_shape") is not None:
            lines.append(
                f"  v1.0 shape: "
                f"{item['v1.0_shape']}"
            )

        if item.get("v1.1_shape") is not None:
            lines.append(
                f"  v1.1 shape: "
                f"{item['v1.1_shape']}"
            )

        if item.get("v1.0_dtype") is not None:
            lines.append(
                f"  v1.0 dtype: "
                f"{item['v1.0_dtype']}"
            )

        if item.get("v1.1_dtype") is not None:
            lines.append(
                f"  v1.1 dtype: "
                f"{item['v1.1_dtype']}"
            )

    # --------------------------------------------------------
    # v1.1 architecture
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "[V1.1 ARCHITECTURE DETECTION]"
    )

    for key, value in report[
        "v1.1_architecture"
    ].items():

        lines.append(
            f"{key}: {value}"
        )

    # --------------------------------------------------------

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# Main analysis
# ============================================================

def build_report(
    path10,
    path11,
):

    v10 = load_checkpoint_info(path10)
    v11 = load_checkpoint_info(path11)

    tensor_diff = compare_tensors(
        v10["tensors"],
        v11["tensors"],
    )

    metadata_diff = compare_metadata(
        v10["metadata"],
        v11["metadata"],
    )

    blocks10 = analyze_dit_blocks(v10)
    blocks11 = analyze_dit_blocks(v11)

    categories10 = category_statistics(v10)
    categories11 = category_statistics(v11)

    insertion_analysis = analyze_insertion_indices(
        v11["metadata"],
        blocks11,
    )

    v11_architecture = analyze_v11_structure(
        v11
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_params10 = sum(
        x.numel
        for x in v10["tensors"].values()
    )

    total_params11 = sum(
        x.numel
        for x in v11["tensors"].values()
    )

    total_bytes10 = sum(
        x.nbytes or 0
        for x in v10["tensors"].values()
    )

    total_bytes11 = sum(
        x.nbytes or 0
        for x in v11["tensors"].values()
    )

    status_counter = Counter(
        x.status
        for x in tensor_diff
    )

    report = {

        "summary": {
            "v1.0_tensor_count":
                len(v10["tensors"]),

            "v1.1_tensor_count":
                len(v11["tensors"]),

            "common_tensor_count":
                status_counter["same"] +
                status_counter["changed"],

            "v1.0_only_tensor_count":
                status_counter["v1.0_only"],

            "v1.1_only_tensor_count":
                status_counter["v1.1_only"],

            "shape_or_dtype_changed_count":
                status_counter["changed"],

            "v1.0_parameters":
                total_params10,

            "v1.1_parameters":
                total_params11,

            "parameter_difference":
                total_params11 -
                total_params10,

            "v1.0_parameter_difference_human":
                human_params(
                    total_params10
                ),

            "v1.1_parameter_difference_human":
                human_params(
                    total_params11
                ),

            "parameter_difference_human":
                human_params(
                    abs(
                        total_params11 -
                        total_params10
                    )
                ),

            "v1.0_weight_bytes":
                total_bytes10,

            "v1.1_weight_bytes":
                total_bytes11,

            "v1.0_weight_size":
                human_bytes(total_bytes10),

            "v1.1_weight_size":
                human_bytes(total_bytes11),

            "weight_size_difference":
                human_bytes(
                    abs(
                        total_bytes11 -
                        total_bytes10
                    )
                ),
        },

        "v1.0": {
            "path":
                v10["path"],

            "filename":
                v10["filename"],

            "filesize":
                v10["filesize"],

            "metadata":
                v10["metadata"],

            "categories":
                categories10,

            "dit_blocks":
                blocks10,
        },

        "v1.1": {
            "path":
                v11["path"],

            "filename":
                v11["filename"],

            "filesize":
                v11["filesize"],

            "metadata":
                v11["metadata"],

            "categories":
                categories11,

            "dit_blocks":
                blocks11,
        },

        "metadata_diff":
            metadata_diff,

        "tensor_diff":
            [
                asdict(x)
                for x in tensor_diff
            ],

        "v1.1_insertion_analysis":
            insertion_analysis,

        "v1.1_architecture":
            v11_architecture,
    }

    return report, v10, v11


# ============================================================
# CSV exports
# ============================================================

def export_tensor_inventory(
    path,
    checkpoint,
):

    rows = []

    for info in checkpoint[
        "tensors"
    ].values():

        rows.append({
            "key":
                info.key,

            "category":
                info.category,

            "block_index":
                info.block_index,

            "submodule":
                info.submodule,

            "shape":
                json.dumps(
                    info.shape
                ),

            "dtype":
                info.dtype,

            "numel":
                info.numel,

            "nbytes":
                info.nbytes,

            "size":
                human_bytes(
                    info.nbytes
                ),
        })

    write_csv(
        path,
        rows,
    )


def export_tensor_diff(
    path,
    report,
):

    rows = []

    for item in report[
        "tensor_diff"
    ]:

        rows.append({
            "status":
                item["status"],

            "key":
                item["key"],

            "v1.0_shape":
                json.dumps(
                    item.get(
                        "v1.0_shape"
                    )
                ),

            "v1.1_shape":
                json.dumps(
                    item.get(
                        "v1.1_shape"
                    )
                ),

            "v1.0_dtype":
                item.get(
                    "v1.0_dtype"
                ),

            "v1.1_dtype":
                item.get(
                    "v1.1_dtype"
                ),

            "v1.0_numel":
                item.get(
                    "v1.0_numel"
                ),

            "v1.1_numel":
                item.get(
                    "v1.1_numel"
                ),
        })

    write_csv(
        path,
        rows,
    )


def export_block_diff(
    path,
    report,
):

    blocks10 = report[
        "v1.0"
    ]["dit_blocks"]

    blocks11 = report[
        "v1.1"
    ]["dit_blocks"]

    all_blocks = sorted(
        set(blocks10.keys()) |
        set(blocks11.keys()),
        key=int,
    )

    rows = []

    for block in all_blocks:

        a = blocks10.get(
            block
        )

        b = blocks11.get(
            block
        )

        p10 = (
            a["parameters"]
            if a
            else 0
        )

        p11 = (
            b["parameters"]
            if b
            else 0
        )

        rows.append({
            "block":
                int(block),

            "v1.0_exists":
                a is not None,

            "v1.1_exists":
                b is not None,

            "v1.0_tensor_count":
                a["tensor_count"]
                if a else 0,

            "v1.1_tensor_count":
                b["tensor_count"]
                if b else 0,

            "v1.0_parameters":
                p10,

            "v1.1_parameters":
                p11,

            "parameter_difference":
                p11 - p10,

            "v1.0_size":
                human_bytes(
                    a["bytes"]
                    if a
                    else 0
                ),

            "v1.1_size":
                human_bytes(
                    b["bytes"]
                    if b
                    else 0
                ),
        })

    write_csv(
        path,
        rows,
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Detailed Anima v1.0/v1.1 "
            "Safetensors checkpoint analyzer"
        )
    )

    parser.add_argument(
        "v1_0",
        type=Path,
        help="Anima v1.0 checkpoint",
    )

    parser.add_argument(
        "v1_1",
        type=Path,
        help="Anima v1.1 checkpoint",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "anima_checkpoint_diff"
        ),
        help="Output directory",
    )

    parser.add_argument(
        "--compare-values",
        action="store_true",
        help=(
            "Compare actual weight values "
            "for common tensors"
        ),
    )

    parser.add_argument(
        "--value-limit",
        type=int,
        default=None,
        help=(
            "Maximum number of common tensors "
            "for value comparison"
        ),
    )

    args = parser.parse_args()

    path10 = args.v1_0
    path11 = args.v1_1

    output = args.output

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Main analysis
    # --------------------------------------------------------

    report, v10, v11 = build_report(
        path10,
        path11,
    )

    # --------------------------------------------------------
    # Optional weight comparison
    # --------------------------------------------------------

    if args.compare_values:

        print()
        print("=" * 80)
        print("Comparing actual tensor values")
        print("=" * 80)

        common_keys = sorted(
            set(v10["tensors"]) &
            set(v11["tensors"])
        )

        # Only same-shape tensors
        common_keys = [
            key
            for key in common_keys
            if (
                v10["tensors"][key].shape ==
                v11["tensors"][key].shape
            )
        ]

        value_results = compare_tensor_values(
            path10,
            path11,
            common_keys,
            args.value_limit,
        )

        report[
            "value_comparison"
        ] = value_results

        write_csv(
            output /
            "tensor_value_comparison.csv",
            value_results,
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    write_json(
        output /
        "anima_diff_report.json",
        report,
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    write_text_report(
        output /
        "anima_diff_report.txt",
        report,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    export_tensor_inventory(
        output /
        "v1.0_tensor_inventory.csv",
        v10,
    )

    export_tensor_inventory(
        output /
        "v1.1_tensor_inventory.csv",
        v11,
    )

    export_tensor_diff(
        output /
        "tensor_diff.csv",
        report,
    )

    export_block_diff(
        output /
        "dit_block_diff.csv",
        report,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ANIMA CHECKPOINT COMPARISON COMPLETE")
    print("=" * 80)

    summary = report["summary"]

    print()
    print(
        f"v1.0 tensors : "
        f"{summary['v1.0_tensor_count']}"
    )

    print(
        f"v1.1 tensors : "
        f"{summary['v1.1_tensor_count']}"
    )

    print(
        f"v1.0 params  : "
        f"{summary['v1.0_parameter_difference_human']}"
    )

    print(
        f"v1.1 params  : "
        f"{summary['v1.1_parameter_difference_human']}"
    )

    print(
        f"parameter Δ  : "
        f"{summary['parameter_difference_human']}"
    )

    print(
        f"v1.0 weight  : "
        f"{summary['v1.0_weight_size']}"
    )

    print(
        f"v1.1 weight  : "
        f"{summary['v1.1_weight_size']}"
    )

    print(
        f"weight Δ     : "
        f"{summary['weight_size_difference']}"
    )

    print()
    print(
        f"v1.0 only    : "
        f"{summary['v1.0_only_tensor_count']}"
    )

    print(
        f"v1.1 only    : "
        f"{summary['v1.1_only_tensor_count']}"
    )

    print(
        f"shape/dtype changed : "
        f"{summary['shape_or_dtype_changed_count']}"
    )

    print()
    print(
        f"Output: {output.resolve()}"
    )


if __name__ == "__main__":
    main()