"""torch.compile utilities for DiT-based Anima training."""

from __future__ import annotations

import argparse
import logging
from typing import Union

import torch

from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def disable_linear_from_compile(module: torch.nn.Module) -> None:
    """Disable torch.compile for Linear-like submodules in a module tree."""
    for sub_module in module.modules():
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward


def apply_cuda_optimizations(args: argparse.Namespace) -> None:
    """Apply optional CUDA performance switches."""
    if getattr(args, "cuda_allow_tf32", False):
        logger.info("Enabling TF32 for matmul and cuDNN")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(args, "cuda_cudnn_benchmark", False):
        logger.info("Enabling cuDNN benchmark mode")
        torch.backends.cudnn.benchmark = True


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[Union[torch.nn.ModuleList, list[torch.nn.Module]]],
    disable_linear: bool,
) -> torch.nn.Module:
    """Compile target transformer blocks in place with torch.compile."""
    if disable_linear:
        logger.info("Disabling Linear layers from torch.compile for block-swapped blocks...")
        for blocks in target_blocks:
            for block in blocks:
                disable_linear_from_compile(block)

    compile_dynamic = None
    if args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

    logger.info(
        "Compiling DiT blocks with torch.compile: "
        f"backend={args.compile_backend}, mode={args.compile_mode}, "
        f"dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )

    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit

    if hasattr(torch._dynamo.config, "force_nn_module_property_static_shapes"):
        torch._dynamo.config.force_nn_module_property_static_shapes = False

    for blocks in target_blocks:
        for index, block in enumerate(blocks):
            blocks[index] = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
    return transformer
