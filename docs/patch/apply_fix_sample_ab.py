"""apply_fix_sample_ab.py
サンプル生成 A/B 対応パッチ

対象ファイル:
  1. anima_train_leco.py  -- サンプル生成機能追加 + keep_vae オプション
  2. lora_train.py        -- サンプルタブ A/B 切り替えUI + State変数追加
  3. leco_train.py        -- サンプルタブ追加 + _build_command / _validate 更新

使い方:
  python apply_fix_sample_ab.py \
      --leco_script  sd-scripts/anima_train_leco.py \
      --lora_train   app/lora_train.py \
      --leco_train   app/leco_train.py

  各引数を省略すると以下のデフォルトパスを使用:
      sd-scripts/anima_train_leco.py
      app/lora_train.py
      app/leco_train.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


# ── ユーティリティ ────────────────────────────────────────────────────────────

def _adapt(src: str, ref: str) -> str:
    """ref の改行コードに src を合わせる。"""
    if "\r\n" in ref:
        return src.replace("\n", "\r\n")
    return src.replace("\r\n", "\n")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    old_a = _adapt(old, text)
    new_a = _adapt(new, text)
    count = text.count(old_a)
    if count == 0:
        raise RuntimeError(
            f"[{label}] 差分文字列が見つかりません。\n--- 先頭120文字 ---\n{old[:120]}"
        )
    if count > 1:
        raise RuntimeError(f"[{label}] 差分文字列が複数マッチしました ({count}箇所)。")
    return text.replace(old_a, new_a, 1)


def _patch(path: Path, patches: list[tuple[str, str, str]], append: str = "") -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new, label in patches:
        text = _replace_once(text, old, new, label)
        print(f"    OK: {label}")
    if append:
        text = text.rstrip() + "\n" + _adapt(append, text)
        print("    OK: 末尾追記")
    bak = path.with_suffix(".py.bak_sampleab")
    bak.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"    バックアップ: {bak.name}")
    print(f"    書き込み完了: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. anima_train_leco.py
# ══════════════════════════════════════════════════════════════════════════════

# ── P1-1: import追加 ──────────────────────────────────────────────────────────
LECO_P1_OLD = """\
import argparse
import importlib
import random
from typing import Optional"""

LECO_P1_NEW = """\
import argparse
import importlib
import math
import random
from pathlib import Path
from typing import Optional"""

# ── P1-2: setup_parser にサンプル引数追加 ────────────────────────────────────
LECO_P2_OLD = """\
    # Dummy args required by train_util.verify_training_args
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    return parser"""

LECO_P2_NEW = """\
    # Dummy args required by train_util.verify_training_args
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    # Sample generation
    parser.add_argument(
        "--sample_every_n_steps", type=int, default=None,
        help="Generate sample images every N training steps",
    )
    parser.add_argument(
        "--sample_prompts", type=str, default=None,
        help="Path to sample prompt file (one prompt-line per sample)",
    )
    parser.add_argument(
        "--sample_save_dir", type=str, default=None,
        help="Directory to save sample images",
    )
    parser.add_argument(
        "--sample_keep_vae", action="store_true",
        help=(
            "Keep VAE loaded in VRAM throughout training for sample generation. "
            "Default: reload VAE each time samples are generated, then unload."
        ),
    )

    return parser"""

# ── P1-3: main() の del vae を keep_vae 分岐に変更 ──────────────────────────
LECO_P3_OLD = """\
    vae.to(weight_dtype)
    vae.eval()
    # VAE is not needed after this point for LECO (no image dataset)
    del vae
    clean_memory_on_device(device)"""

LECO_P3_NEW = """\
    vae.to(weight_dtype)
    vae.eval()

    # VAE 保持オプション: keep_vae=True のときはVRAMに残す
    _keep_vae = getattr(args, "sample_keep_vae", False) and getattr(args, "sample_every_n_steps", None)
    if _keep_vae:
        logger.info("sample_keep_vae=True: VAE をVRAMに保持します。")
        _vae_for_sample = vae
    else:
        del vae
        _vae_for_sample = None
    clean_memory_on_device(device)"""

# ── P1-4: 学習ループ内 save_every_n_steps の直後にサンプル生成呼び出し追加 ──
LECO_P4_OLD = """\
            if (
                args.save_every_n_steps
                and global_step % args.save_every_n_steps == 0
                and global_step < args.max_train_steps
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_weights(
                        accelerator, network, args, save_dtype,
                        prompt_settings, global_step, last=False,
                    )"""

LECO_P4_NEW = """\
            if (
                args.save_every_n_steps
                and global_step % args.save_every_n_steps == 0
                and global_step < args.max_train_steps
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_weights(
                        accelerator, network, args, save_dtype,
                        prompt_settings, global_step, last=False,
                    )

            # ── サンプル生成 ──────────────────────────────────────
            if (
                getattr(args, "sample_every_n_steps", None)
                and global_step % args.sample_every_n_steps == 0
                and args.sample_prompts
                and args.sample_save_dir
                and accelerator.is_main_process
            ):
                _vae_for_sample = _generate_samples_leco(
                    args=args,
                    dit=dit,
                    qwen3_text_encoder=qwen3_text_encoder,
                    tokenize_strategy=tokenize_strategy,
                    text_encoding_strategy=text_encoding_strategy,
                    noise_scheduler=noise_scheduler,
                    network=network,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    device=device,
                    global_step=global_step,
                    vae_cached=_vae_for_sample,
                )"""

# ── P1-5: 末尾に _generate_samples_leco 関数追加 ────────────────────────────
LECO_APPEND = '''

# ---------------------------------------------------------------------------
# Sample generation helper
# ---------------------------------------------------------------------------

def _generate_samples_leco(
    args,
    dit,
    qwen3_text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    noise_scheduler,
    network,
    accelerator,
    weight_dtype,
    device,
    global_step: int,
    vae_cached,
):
    """学習中にサンプル画像を生成して sample_save_dir に保存する。

    Parameters
    ----------
    vae_cached : VAEオブジェクト または None
        keep_vae=True の場合は渡されたオブジェクトを再利用し、
        False の場合は都度ロード→デコード後アンロードする。

    Returns
    -------
    vae_cached : keep_vae=True のときはロード済みVAEを返す（次回再利用）。
                 keep_vae=False のときは None を返す。
    """
    import os

    logger.info(f"[SampleGen] step={global_step} サンプル生成開始")

    # ── プロンプトファイル解析 ────────────────────────────────────────────────
    try:
        prompt_lines = [
            ln.strip()
            for ln in Path(args.sample_prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    except Exception as exc:
        logger.warning(f"[SampleGen] prompt ファイル読み込み失敗: {exc}")
        return vae_cached

    if not prompt_lines:
        logger.warning("[SampleGen] 有効なプロンプト行がありません。スキップします。")
        return vae_cached

    # ── VAE ロード ──────────────────────────────────────────────────────────
    keep_vae = getattr(args, "sample_keep_vae", False)
    if vae_cached is not None:
        vae = vae_cached
    else:
        logger.info("[SampleGen] VAE を一時ロードします...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
        )
        vae.to(weight_dtype).to(device)
        vae.eval()

    # ── ネットワークを推論モードへ ──────────────────────────────────────────
    net_unwrapped = accelerator.unwrap_model(network)
    net_unwrapped.set_multiplier(1.0)
    net_unwrapped.eval()

    # ── 出力先ディレクトリ ──────────────────────────────────────────────────
    save_base = Path(args.sample_save_dir)
    save_base.mkdir(parents=True, exist_ok=True)

    # ── プロンプト行ごとに生成 ──────────────────────────────────────────────
    for line_idx, line in enumerate(prompt_lines):
        try:
            # サブディレクトリ: 行番号 0 → sample_a, 1 → sample_b, 2以降 → sample_c ...
            subdir_name = (
                ["sample_a", "sample_b"][line_idx]
                if line_idx < 2
                else f"sample_{chr(ord('a') + line_idx)}"
            )
            save_dir = save_base / subdir_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # プロンプト行のパース
            # 書式: <prompt text> [--n <neg>] --w <W> --h <H> --s <steps> --l <scale> --fs <flow_shift> --d <seed>
            prompt_text, gen_kwargs = _parse_sample_prompt_line(line)

            width      = gen_kwargs.get("w", 512)
            height     = gen_kwargs.get("h", 512)
            steps      = gen_kwargs.get("s", 20)
            scale      = gen_kwargs.get("l", 7.5)
            flow_shift = gen_kwargs.get("fs", 3.0)
            seed       = int(gen_kwargs.get("d", 42))
            neg_text   = gen_kwargs.get("n", "")

            # seed固定
            torch.manual_seed(seed)

            # テキストエンコード
            qwen3_text_encoder.to(device, dtype=weight_dtype)
            with torch.no_grad():
                cond_embeds  = encode_prompt_anima(
                    tokenize_strategy, text_encoding_strategy,
                    qwen3_text_encoder, prompt_text,
                )
                uncond_embeds = encode_prompt_anima(
                    tokenize_strategy, text_encoding_strategy,
                    qwen3_text_encoder, neg_text,
                )
            qwen3_text_encoder.to("cpu")
            clean_memory_on_device(device)

            # デノイズ（フルステップ）
            sample_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flow_shift,
            )
            sample_scheduler.set_timesteps(steps, device=device)

            latents = get_initial_latents_anima(1, height, width).to(device, dtype=weight_dtype)
            embeds_cfg = concat_embeds_anima(uncond_embeds, cond_embeds, 1)

            with torch.no_grad():
                latents = diffusion_anima(
                    dit,
                    sample_scheduler,
                    latents,
                    embeds_cfg,
                    total_timesteps=steps,
                    guidance_scale=scale,
                    weight_dtype=weight_dtype,
                    device=device,
                )

            # VAE デコード
            vae.to(device)
            with torch.no_grad():
                latents_in = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
                decoded = vae.decode(latents_in).sample
                # [B, C, 1, H, W] -> [B, C, H, W]
                if decoded.dim() == 5:
                    decoded = decoded.squeeze(2)

            # [B, C, H, W] -> PIL
            img_tensor = decoded[0].float().clamp(-1, 1)
            img_tensor = (img_tensor + 1.0) / 2.0
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(img_np)
            except ImportError:
                import array as _arr
                # Pillow がない場合は raw PPM 保存
                fname = save_dir / f"step{global_step:06d}_s{seed}.ppm"
                with open(fname, "wb") as f:
                    h_, w_ = img_np.shape[:2]
                    f.write(b"P6 " + str(w_).encode() + b" " + str(h_).encode() + b" 255 ".replace(b" ", bytes([10]), 2))
                    f.write(img_np.tobytes())
                logger.info(f"[SampleGen] saved (PPM): {fname}")
                continue

            fname = save_dir / f"step{global_step:06d}_s{seed}.png"
            pil_img.save(str(fname))
            logger.info(f"[SampleGen] saved: {fname}")

        except Exception as exc:
            logger.warning(f"[SampleGen] line {line_idx} 生成失敗: {exc}")

    # ── 後処理 ─────────────────────────────────────────────────────────────
    net_unwrapped.train()
    net_unwrapped.set_multiplier(0.0)

    if keep_vae:
        return vae
    else:
        vae.to("cpu")
        del vae
        clean_memory_on_device(device)
        logger.info("[SampleGen] VAE をアンロードしました。")
        return None


def _parse_sample_prompt_line(line: str) -> tuple[str, dict]:
    """サンプルプロンプト行をパースして (prompt_text, kwargs) を返す。

    書式: <prompt> [--n <neg>] [--w W] [--h H] [--s S] [--l L] [--fs FS] [--d D]
    """
    import shlex

    result: dict = {}
    # --n は残りすべてを取るため最初に抽出
    neg = ""
    neg_pat = __import__("re").search(r"[ \t]--n[ \t]+(.+?)(?=[ \t]+--[a-z]|[ \t]*$)", line)
    if neg_pat:
        neg = neg_pat.group(1).strip()
        line = line[:neg_pat.start()] + line[neg_pat.end():]
    result["n"] = neg

    flag_map = {
        "--w": ("w", int),
        "--h": ("h", int),
        "--s": ("s", int),
        "--l": ("l", float),
        "--fs": ("fs", float),
        "--d": ("d", int),
    }
    prompt_end = len(line)
    for flag in flag_map:
        idx = line.find(f" {flag} ")
        if idx != -1:
            prompt_end = min(prompt_end, idx)
    prompt_text = line[:prompt_end].strip()

    remainder = line[prompt_end:]
    for flag, (key, cast) in flag_map.items():
        pat = __import__("re").search(rf"{re.escape(flag)}[ \t]+([^ \t]+)", remainder)
        if pat:
            try:
                result[key] = cast(pat.group(1))
            except (ValueError, TypeError):
                pass

    return prompt_text, result
'''

LECO_SCRIPT_PATCHES = [
    (LECO_P1_OLD, LECO_P1_NEW, "P1-1: import追加"),
    (LECO_P2_OLD, LECO_P2_NEW, "P1-2: サンプル引数追加"),
    (LECO_P3_OLD, LECO_P3_NEW, "P1-3: del vae → keep_vae分岐"),
    (LECO_P4_OLD, LECO_P4_NEW, "P1-4: ループ内サンプル生成呼び出し"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. lora_train.py
# ══════════════════════════════════════════════════════════════════════════════

# ── P2-1: _TrainState sample変数にB用追加 ────────────────────────────────────
LORA_P1_OLD = """\
        self.sample_enabled = tk.BooleanVar(value=False)
        self.sample_every_n_epochs = tk.StringVar(value="1")
        self.sample_prompts = tk.StringVar(value="")
        self.sample_prompt = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        self.sample_width = tk.IntVar(value=512)
        self.sample_height = tk.IntVar(value=512)
        self.sample_steps = tk.IntVar(value=30)
        self.sample_scale = tk.DoubleVar(value=7.5)
        self.sample_flow_shift = tk.DoubleVar(value=3.0)"""

LORA_P1_NEW = """\
        # サンプル生成 共通設定
        self.sample_every_n_epochs  = tk.StringVar(value="1")
        self.sample_width           = tk.IntVar(value=512)
        self.sample_height          = tk.IntVar(value=512)
        self.sample_steps           = tk.IntVar(value=30)
        self.sample_scale           = tk.DoubleVar(value=7.5)
        self.sample_flow_shift      = tk.DoubleVar(value=3.0)
        # サンプルA
        self.sample_enabled         = tk.BooleanVar(value=False)
        self.sample_prompt          = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        # サンプルB
        self.sample_b_enabled          = tk.BooleanVar(value=False)
        self.sample_b_prompt           = tk.StringVar(value="")
        self.sample_b_negative_prompt  = tk.StringVar(value="")
        # 旧互換
        self.sample_prompts = tk.StringVar(value="")"""

# ── P2-2: _sample_dir / _sample_prompt_path をA/B対応に変更 ─────────────────
LORA_P2_OLD = """\
def _sample_dir(s: _TrainState) -> Path:
    return s.paths.root / "log" / "sample_gen"


def _sample_prompt_path(s: _TrainState) -> Path:
    return _sample_dir(s) / "_sample_prompt.txt"


def _build_sample_prompt_line(s: _TrainState) -> str:
    prompt = s.sample_prompt.get().strip()
    neg = s.sample_negative_prompt.get().strip()
    width = max(64, int(s.sample_width.get()))
    height = max(64, int(s.sample_height.get()))
    steps = max(1, int(s.sample_steps.get()))
    scale = float(s.sample_scale.get())
    flow_shift = float(s.sample_flow_shift.get())

    line = (
        f"{prompt} --w {width} --h {height} --s {steps} "
        f"--l {scale:g} --fs {flow_shift:g} --d {SAMPLE_FIXED_SEED}"
    )
    if neg:
        line += f" --n {neg}"
    return line


def _write_sample_prompt_file(s: _TrainState) -> Path:
    path = _sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_build_sample_prompt_line(s) + "\\n", encoding="utf-8", newline="\\n")
    return path"""

LORA_P2_NEW = """\
def _sample_dir(s: _TrainState) -> Path:
    return s.paths.root / "log" / "sample_gen"


def _sample_dir_a(s: _TrainState) -> Path:
    return _sample_dir(s) / "sample_a"


def _sample_dir_b(s: _TrainState) -> Path:
    return _sample_dir(s) / "sample_b"


def _sample_prompt_path(s: _TrainState) -> Path:
    return _sample_dir(s) / "_sample_prompt.txt"


def _build_sample_prompt_line_for(
    prompt: str, neg: str, s: _TrainState
) -> str:
    width      = max(64, int(s.sample_width.get()))
    height     = max(64, int(s.sample_height.get()))
    steps      = max(1,  int(s.sample_steps.get()))
    scale      = float(s.sample_scale.get())
    flow_shift = float(s.sample_flow_shift.get())
    line = (
        f"{prompt} --w {width} --h {height} --s {steps} "
        f"--l {scale:g} --fs {flow_shift:g} --d {SAMPLE_FIXED_SEED}"
    )
    if neg:
        line += f" --n {neg}"
    return line


def _build_sample_prompt_line(s: _TrainState) -> str:
    # 後方互換用: サンプルAのプロンプト行を返す
    return _build_sample_prompt_line_for(
        s.sample_prompt.get().strip(),
        s.sample_negative_prompt.get().strip(),
        s,
    )


def _write_sample_prompt_file(s: _TrainState) -> Path:
    lines = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_build_sample_prompt_line_for(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            s,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_build_sample_prompt_line_for(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            s,
        ))
    path = _sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")
    return path"""

# ── P2-3: _build_command のサンプル条件を A or B に変更 ──────────────────────
LORA_P3_OLD = """\
    if s.sample_enabled.get():
        sample_prompt_file = _write_sample_prompt_file(s)
        cmd += ["--sample_every_n_epochs", s.sample_every_n_epochs.get().strip() or "1"]
        cmd += ["--sample_prompts", str(sample_prompt_file)]
        cmd += ["--sample_save_dir", str(_sample_dir(s))]"""

LORA_P3_NEW = """\
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        sample_prompt_file = _write_sample_prompt_file(s)
        cmd += ["--sample_every_n_epochs", s.sample_every_n_epochs.get().strip() or "1"]
        cmd += ["--sample_prompts", str(sample_prompt_file)]
        cmd += ["--sample_save_dir", str(_sample_dir(s))]"""

# ── P2-4: _validate のサンプルチェック更新 ───────────────────────────────────
LORA_P4_OLD = """\
    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプル生成が有効ですが、prompt が未設定です。"
    if s.sample_enabled.get():
        try:
            if int(s.sample_every_n_epochs.get().strip() or "1") <= 0:
                return "サンプル出力のepoch間隔は1以上を指定してください。"
        except ValueError:
            return "サンプル出力のepoch間隔は整数で指定してください。"
    return None"""

LORA_P4_NEW = """\
    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプルAが有効ですが、promptが未設定です。"
    if s.sample_b_enabled.get() and not s.sample_b_prompt.get().strip():
        return "サンプルBが有効ですが、promptが未設定です。"
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        try:
            if int(s.sample_every_n_epochs.get().strip() or "1") <= 0:
                return "サンプル出力のepoch間隔は1以上を指定してください。"
        except ValueError:
            return "サンプル出力のepoch間隔は整数で指定してください。"
    return None"""

# ── P2-5: _build_sample_tab 全体を A/B 切り替えUI に置換 ─────────────────────
LORA_P5A_OLD = (
    "        " + chr(34)*3 +
    "確認ポップアップ（いいえ / はい）を表示し、OKなら sample_gen 内を全削除する。" +
    chr(34)*3
)

LORA_P5A_NEW = """\
        # 確認ポップアップ（いいえ / はい）を表示し、OKなら sample_gen 内を全削除する。"""

LORA_P5B_OLD = """\
def _build_sample_tab(parent: ttk.Frame, s: _TrainState) -> None:
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    settings = ttk.LabelFrame(parent, text="サンプル生成設定")
    settings.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
    settings.columnconfigure(1, weight=1)
    settings.columnconfigure(3, weight=1)

    ttk.Checkbutton(settings, text="サンプル生成を有効にする", variable=s.sample_enabled).grid(
        row=0, column=0, columnspan=2, sticky=tk.W, padx=(4, 12), pady=3
    )
    ttk.Label(settings, text=f"seed固定: {SAMPLE_FIXED_SEED}", foreground="#64748B").grid(
        row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3
    )

    ttk.Label(settings, text="epoch間隔", width=16, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    ttk.Spinbox(settings, from_=1, to=9999, textvariable=s.sample_every_n_epochs, width=8).grid(
        row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3
    )
    ttk.Label(settings, text="出力先", width=10, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3
    )
    ttk.Label(settings, text=str(_sample_dir(s)), foreground="#1D4ED8").grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3
    )

    ttk.Label(settings, text="prompt", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    ttk.Entry(settings, textvariable=s.sample_prompt).grid(
        row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=3
    )

    ttk.Label(settings, text="negative prompt", width=16, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    ttk.Entry(settings, textvariable=s.sample_negative_prompt).grid(
        row=3, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=3
    )

    ttk.Label(settings, text="width / height", width=16, anchor=tk.W).grid(
        row=4, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    size_frame = ttk.Frame(settings)
    size_frame.grid(row=4, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Spinbox(size_frame, from_=64, to=4096, increment=16, textvariable=s.sample_width, width=7).pack(side=tk.LEFT)
    ttk.Label(size_frame, text=" x ").pack(side=tk.LEFT)
    ttk.Spinbox(size_frame, from_=64, to=4096, increment=16, textvariable=s.sample_height, width=7).pack(side=tk.LEFT)

    ttk.Label(settings, text="steps", width=10, anchor=tk.W).grid(
        row=4, column=2, sticky=tk.W, padx=(0, 2), pady=3
    )
    ttk.Spinbox(settings, from_=1, to=1000, textvariable=s.sample_steps, width=8).grid(
        row=4, column=3, sticky=tk.W, padx=(0, 4), pady=3
    )

    ttk.Label(settings, text="scale", width=16, anchor=tk.W).grid(
        row=5, column=0, sticky=tk.W, padx=(4, 2), pady=3
    )
    ttk.Entry(settings, textvariable=s.sample_scale, width=10).grid(
        row=5, column=1, sticky=tk.W, padx=(0, 12), pady=3
    )
    ttk.Label(settings, text="flow_shift", width=10, anchor=tk.W).grid(
        row=5, column=2, sticky=tk.W, padx=(0, 2), pady=3
    )
    ttk.Entry(settings, textvariable=s.sample_flow_shift, width=10).grid(
        row=5, column=3, sticky=tk.W, padx=(0, 4), pady=3
    )

    gallery = ttk.LabelFrame(parent, text="最新サンプル")
    gallery.grid(row=1, column=0, sticky=tk.NSEW)
    for c in range(5):
        gallery.columnconfigure(c, weight=1, uniform="sample_cols")
    for r in range(2):
        gallery.rowconfigure(r, weight=1, uniform="sample_rows")

    cells = []
    photo_refs: list[object | None] = [None] * 10
    for idx in range(10):
        cell = ttk.Frame(gallery, padding=4)
        cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
        cell.columnconfigure(0, weight=1)
        cell.rowconfigure(0, weight=1)
        img = ttk.Label(cell, anchor=tk.CENTER)
        img.grid(row=0, column=0, sticky=tk.NSEW)
        ep = ttk.Label(cell, text="epoch -", anchor=tk.CENTER)
        ep.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        cells.append((img, ep))

    def _refresh_gallery(schedule_next: bool = False) -> None:
        files = sorted(
            _sample_dir(s).glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]
        try:
            from PIL import Image, ImageTk
        except Exception:
            Image = None
            ImageTk = None

        for idx, (img_label, ep_label) in enumerate(cells):
            if idx >= len(files):
                img_label.configure(image="", text="")
                ep_label.configure(text="epoch -")
                photo_refs[idx] = None
                continue

            path = files[idx]
            ep_label.configure(text=f"epoch {_extract_sample_epoch(path)}")
            if Image is None or ImageTk is None:
                img_label.configure(image="", text=path.name)
                photo_refs[idx] = None
                continue

            try:
                with Image.open(path) as im:
                    im.thumbnail((220, 220))
                    photo = ImageTk.PhotoImage(im.copy())
                photo_refs[idx] = photo
                img_label.configure(image=photo, text="")
            except Exception:
                img_label.configure(image="", text=path.name)
                photo_refs[idx] = None

        if schedule_next:
            parent.after(2000, lambda: _refresh_gallery(True))

    ttk.Button(settings, text="表示更新", command=lambda: _refresh_gallery(False)).grid(
        row=6, column=0, sticky=tk.W, padx=(4, 2), pady=(4, 3)
    )

    def _clear_sample_cache() -> None:
        # 確認ポップアップ（いいえ / はい）を表示し、OKなら sample_gen 内を全削除する。
        import tkinter as _tk

        dlg = _tk.Toplevel(settings)
        dlg.title("確認")
        dlg.resizable(False, False)
        dlg.grab_set()

        _tk.Label(
            dlg,
            text="本当に全て削除しますか？",
            font=("TkDefaultFont", 11),
            padx=20,
            pady=16,
        ).pack()

        btn_frame = _tk.Frame(dlg)
        btn_frame.pack(pady=(0, 12))

        def _no():
            dlg.destroy()

        def _yes():
            dlg.destroy()
            target = _sample_dir(s)
            if not target.exists():
                return
            deleted = 0
            errors = 0
            for f in target.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                        deleted += 1
                except Exception:
                    errors += 1
            msg = f"[サンプルキャッシュ] {deleted}ファイルを削除しました。"
            if errors:
                msg += f" ({errors}件失敗)"
            s.log_fn(msg)
            _refresh_gallery(False)

        # ボタン並び: いいえ → はい
        _tk.Button(btn_frame, text="いいえ", width=10, command=_no).pack(side=_tk.LEFT, padx=(0, 8))
        _tk.Button(btn_frame, text="はい",   width=10, command=_yes).pack(side=_tk.LEFT)

        # ダイアログを中央付近に配置
        dlg.update_idletasks()
        px = settings.winfo_rootx() + settings.winfo_width() // 2 - dlg.winfo_width() // 2
        py = settings.winfo_rooty() + settings.winfo_height() // 2 - dlg.winfo_height() // 2
        dlg.geometry(f"+{px}+{py}")

    ttk.Button(settings, text="キャッシュクリア", command=_clear_sample_cache).grid(
        row=6, column=1, sticky=tk.W, padx=(0, 4), pady=(4, 3)
    )

    _refresh_gallery(True)"""

LORA_P5B_NEW = """\
def _build_sample_tab(parent: ttk.Frame, s: _TrainState) -> None:
    _build_sample_tab_common(parent, s, is_leco=False)


def _build_sample_tab_common(
    parent: ttk.Frame,
    s,
    is_leco: bool = False,
) -> None:

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    # ── 共通設定 ─────────────────────────────────────────────────────────────
    common = ttk.LabelFrame(parent, text="共通生成条件")
    common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    common.columnconfigure(1, weight=1)
    common.columnconfigure(3, weight=1)

    interval_label = "step間隔" if is_leco else "epoch間隔"
    interval_var   = s.sample_every_n_steps if is_leco else s.sample_every_n_epochs

    ttk.Label(common, text=interval_label, width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=99999, textvariable=interval_var, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text=f"seed固定: {SAMPLE_FIXED_SEED}", foreground="#64748B").grid(
        row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="width / height", width=16, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    sf = ttk.Frame(common)
    sf.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_width,  width=7).pack(side=tk.LEFT)
    ttk.Label(sf, text=" x ").pack(side=tk.LEFT)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_height, width=7).pack(side=tk.LEFT)

    ttk.Label(common, text="steps", width=10, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=1000, textvariable=s.sample_steps, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="scale", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_scale, width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text="flow_shift", width=10, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_flow_shift, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    if is_leco:
        ttk.Checkbutton(
            common, text="VAEを学習中保持 (sample_keep_vae)",
            variable=s.sample_keep_vae,
        ).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    # ── A/B 切り替えノートブック ─────────────────────────────────────────────
    ab_nb = ttk.Notebook(parent)
    ab_nb.grid(row=1, column=0, sticky=tk.NSEW)

    tab_a = ttk.Frame(ab_nb, padding=4)
    tab_b = ttk.Frame(ab_nb, padding=4)
    ab_nb.add(tab_a, text="  サンプルA  ")
    ab_nb.add(tab_b, text="  サンプルB  ")

    dir_a = _sample_dir_a(s) if not is_leco else (s.paths.root / "log" / "sample_gen" / "sample_a")
    dir_b = _sample_dir_b(s) if not is_leco else (s.paths.root / "log" / "sample_gen" / "sample_b")

    _build_sample_ab_panel(
        tab_a, s,
        enabled_var=s.sample_enabled,
        prompt_var=s.sample_prompt,
        neg_var=s.sample_negative_prompt,
        sample_dir=dir_a,
        label="A",
    )
    _build_sample_ab_panel(
        tab_b, s,
        enabled_var=s.sample_b_enabled,
        prompt_var=s.sample_b_prompt,
        neg_var=s.sample_b_negative_prompt,
        sample_dir=dir_b,
        label="B",
    )"""

# ── P2-6: _extract_sample_epoch の直後に _build_sample_ab_panel 追加 ─────────
LORA_P6_OLD = """\
def _build_sample_tab(parent: ttk.Frame, s: _TrainState) -> None:"""

LORA_P6_NEW = """\
def _build_sample_ab_panel(
    parent: ttk.Frame,
    s,
    enabled_var: tk.BooleanVar,
    prompt_var: tk.StringVar,
    neg_var: tk.StringVar,
    sample_dir: Path,
    label: str,
) -> None:
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    top = ttk.Frame(parent)
    top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    top.columnconfigure(1, weight=1)

    ttk.Checkbutton(
        top, text=f"サンプル{label}を有効にする", variable=enabled_var
    ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=(2, 4), pady=2)

    ttk.Label(top, text=f"出力先: ", foreground="#475569").grid(
        row=1, column=0, sticky=tk.W, padx=(2, 0), pady=2)
    ttk.Label(top, text=str(sample_dir), foreground="#1D4ED8").grid(
        row=1, column=1, columnspan=3, sticky=tk.W, pady=2)

    ttk.Label(top, text="prompt", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=prompt_var).grid(
        row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

    ttk.Label(top, text="negative", width=16, anchor=tk.W).grid(
        row=3, column=0, sticky=tk.W, padx=(2, 2), pady=2)
    ttk.Entry(top, textvariable=neg_var).grid(
        row=3, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

    # ── ギャラリー ────────────────────────────────────────────────────────────
    gallery = ttk.LabelFrame(parent, text=f"最新サンプル{label}")
    gallery.grid(row=1, column=0, sticky=tk.NSEW)
    for c in range(5):
        gallery.columnconfigure(c, weight=1, uniform=f"sc_{label}")
    for r in range(2):
        gallery.rowconfigure(r, weight=1, uniform=f"sr_{label}")

    cells: list = []
    photo_refs: list = [None] * 10

    for idx in range(10):
        cell = ttk.Frame(gallery, padding=4)
        cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
        cell.columnconfigure(0, weight=1)
        cell.rowconfigure(0, weight=1)
        img_lbl = ttk.Label(cell, anchor=tk.CENTER)
        img_lbl.grid(row=0, column=0, sticky=tk.NSEW)
        ep_lbl = ttk.Label(cell, text="step -", anchor=tk.CENTER)
        ep_lbl.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
        cells.append((img_lbl, ep_lbl))

    def _refresh(schedule_next: bool = False) -> None:
        files = []
        if sample_dir.exists():
            files = sorted(
                sample_dir.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:10]
        try:
            from PIL import Image as _Im, ImageTk as _ITk
        except Exception:
            _Im = _ITk = None

        for idx, (il, el) in enumerate(cells):
            if idx >= len(files):
                il.configure(image="", text="")
                el.configure(text="step -")
                photo_refs[idx] = None
                continue
            p = files[idx]
            el.configure(text=f"step {_extract_sample_epoch(p)}")
            if _Im is None:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None
                continue
            try:
                with _Im.open(p) as im:
                    im.thumbnail((220, 220))
                    ph = _ITk.PhotoImage(im.copy())
                photo_refs[idx] = ph
                il.configure(image=ph, text="")
            except Exception:
                il.configure(image="", text=p.name)
                photo_refs[idx] = None

        if schedule_next:
            parent.after(2000, lambda: _refresh(True))

    def _clear_cache() -> None:
        import tkinter as _tk
        dlg = _tk.Toplevel(top)
        dlg.title("確認")
        dlg.resizable(False, False)
        dlg.grab_set()
        _tk.Label(dlg, text=f"サンプル{label} を全て削除しますか？",
                  font=("TkDefaultFont", 11), padx=20, pady=16).pack()
        bf = _tk.Frame(dlg)
        bf.pack(pady=(0, 12))
        def _no(): dlg.destroy()
        def _yes():
            dlg.destroy()
            if not sample_dir.exists():
                return
            deleted = errors = 0
            for f in sample_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink(); deleted += 1
                except Exception:
                    errors += 1
            msg = f"[サンプル{label}] {deleted}件削除。"
            if errors: msg += f" ({errors}件失敗)"
            s.log_fn(msg)
            _refresh(False)
        _tk.Button(bf, text="いいえ", width=10, command=_no).pack(side=_tk.LEFT, padx=(0, 8))
        _tk.Button(bf, text="はい",   width=10, command=_yes).pack(side=_tk.LEFT)
        dlg.update_idletasks()
        px = top.winfo_rootx() + top.winfo_width()  // 2 - dlg.winfo_width()  // 2
        py = top.winfo_rooty() + top.winfo_height() // 2 - dlg.winfo_height() // 2
        dlg.geometry(f"+{px}+{py}")

    btn_row = ttk.Frame(top)
    btn_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 2))
    ttk.Button(btn_row, text="表示更新",     command=lambda: _refresh(False)).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_row, text="キャッシュクリア", command=_clear_cache).pack(side=tk.LEFT)

    _refresh(True)


def _build_sample_tab(parent: ttk.Frame, s: _TrainState) -> None:"""

# ── P2-7: プリセット保存に B 用変数追加 ──────────────────────────────────────
LORA_P7_OLD = """\
            "sample_enabled":    bool(s.sample_enabled.get()),
            "sample_every_n_epochs": s.sample_every_n_epochs.get(),
            "sample_prompts":    s.sample_prompts.get(),
            "sample_prompt":     s.sample_prompt.get(),
            "sample_negative_prompt": s.sample_negative_prompt.get(),
            "sample_width":      int(s.sample_width.get()),
            "sample_height":     int(s.sample_height.get()),
            "sample_steps":      int(s.sample_steps.get()),
            "sample_scale":      float(s.sample_scale.get()),
            "sample_flow_shift": float(s.sample_flow_shift.get()),"""

LORA_P7_NEW = """\
            "sample_enabled":    bool(s.sample_enabled.get()),
            "sample_every_n_epochs": s.sample_every_n_epochs.get(),
            "sample_prompts":    s.sample_prompts.get(),
            "sample_prompt":     s.sample_prompt.get(),
            "sample_negative_prompt": s.sample_negative_prompt.get(),
            "sample_b_enabled":           bool(s.sample_b_enabled.get()),
            "sample_b_prompt":            s.sample_b_prompt.get(),
            "sample_b_negative_prompt":   s.sample_b_negative_prompt.get(),
            "sample_width":      int(s.sample_width.get()),
            "sample_height":     int(s.sample_height.get()),
            "sample_steps":      int(s.sample_steps.get()),
            "sample_scale":      float(s.sample_scale.get()),
            "sample_flow_shift": float(s.sample_flow_shift.get()),"""

# ── P2-8: プリセット読み込みに B 用変数追加 ───────────────────────────────────
LORA_P8_OLD = """\
        _s(s.sample_enabled,    "sample_enabled",     False)
        _s(s.sample_every_n_epochs, "sample_every_n_epochs", "1")
        _s(s.sample_prompts,    "sample_prompts",     "")
        _s(s.sample_prompt,     "sample_prompt",      "")
        _s(s.sample_negative_prompt, "sample_negative_prompt", "")
        _s(s.sample_width,      "sample_width",       512)
        _s(s.sample_height,     "sample_height",      512)
        _s(s.sample_steps,      "sample_steps",       30)
        _s(s.sample_scale,      "sample_scale",       7.5)
        _s(s.sample_flow_shift, "sample_flow_shift",  3.0)"""

LORA_P8_NEW = """\
        _s(s.sample_enabled,    "sample_enabled",     False)
        _s(s.sample_every_n_epochs, "sample_every_n_epochs", "1")
        _s(s.sample_prompts,    "sample_prompts",     "")
        _s(s.sample_prompt,     "sample_prompt",      "")
        _s(s.sample_negative_prompt, "sample_negative_prompt", "")
        _s(s.sample_b_enabled,          "sample_b_enabled",          False)
        _s(s.sample_b_prompt,           "sample_b_prompt",           "")
        _s(s.sample_b_negative_prompt,  "sample_b_negative_prompt",  "")
        _s(s.sample_width,      "sample_width",       512)
        _s(s.sample_height,     "sample_height",      512)
        _s(s.sample_steps,      "sample_steps",       30)
        _s(s.sample_scale,      "sample_scale",       7.5)
        _s(s.sample_flow_shift, "sample_flow_shift",  3.0)"""

LORA_PATCHES = [
    (LORA_P1_OLD, LORA_P1_NEW, "P2-1: _TrainState サンプルB変数追加"),
    (LORA_P2_OLD, LORA_P2_NEW, "P2-2: _sample_dir_a/b + _write_sample_prompt_file A/B対応"),
    (LORA_P3_OLD, LORA_P3_NEW, "P2-3: _build_command sample条件更新"),
    (LORA_P4_OLD, LORA_P4_NEW, "P2-4: _validate sample更新"),
    (LORA_P5A_OLD, LORA_P5A_NEW, "P2-5a: _clear_sample_cache docstring→comment"),
    (LORA_P5B_OLD, LORA_P5B_NEW, "P2-5b: _build_sample_tab 共通実装へ置換"),
    (LORA_P6_OLD, LORA_P6_NEW, "P2-6: _build_sample_ab_panel 追加"),
    (LORA_P7_OLD, LORA_P7_NEW, "P2-7: プリセット保存にB追加"),
    (LORA_P8_OLD, LORA_P8_NEW, "P2-8: プリセット読み込みにB追加"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. leco_train.py  (フェーズ2パッチ適用済みファイルを対象)
# ══════════════════════════════════════════════════════════════════════════════

# ── P3-1: _LecoTrainState サンプル変数を B 対応・keep_vae 追加に更新 ─────────
#   フェーズ2パッチで追加済みの sample ブロックを置換
LECO_TRAIN_P1_OLD = """\
        # ── サンプル生成（将来実装用・プリセット保存領域確保） ───
        self.sample_enabled        = tk.BooleanVar(value=False)
        self.sample_every_n_steps  = tk.StringVar(value="100")
        self.sample_prompt         = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        self.sample_width          = tk.IntVar(value=512)
        self.sample_height         = tk.IntVar(value=512)
        self.sample_steps          = tk.IntVar(value=20)
        self.sample_scale          = tk.DoubleVar(value=7.5)
        self.sample_flow_shift     = tk.DoubleVar(value=3.0)
        self.sample_seed           = tk.StringVar(value="42")"""

LECO_TRAIN_P1_NEW = """\
        # ── サンプル生成 共通設定 ────────────────────────────────
        self.sample_every_n_steps   = tk.StringVar(value="100")
        self.sample_width           = tk.IntVar(value=512)
        self.sample_height          = tk.IntVar(value=512)
        self.sample_steps           = tk.IntVar(value=20)
        self.sample_scale           = tk.DoubleVar(value=7.5)
        self.sample_flow_shift      = tk.DoubleVar(value=3.0)
        self.sample_keep_vae        = tk.BooleanVar(value=False)
        # サンプルA
        self.sample_enabled         = tk.BooleanVar(value=False)
        self.sample_prompt          = tk.StringVar(value="")
        self.sample_negative_prompt = tk.StringVar(value="")
        # サンプルB
        self.sample_b_enabled          = tk.BooleanVar(value=False)
        self.sample_b_prompt           = tk.StringVar(value="")
        self.sample_b_negative_prompt  = tk.StringVar(value="")"""

# ── P3-2: タブ構成に サンプル生成タブ追加 ────────────────────────────────────
LECO_TRAIN_P2_OLD = """\
    tab_layer          = ttk.Frame(nb, padding=8)
    tab_monitor        = ttk.Frame(nb, padding=8)
    tab_monitor_layer  = ttk.Frame(nb, padding=8)
    tab_preset         = ttk.Frame(nb, padding=8)

    nb.add(tab_model,          text="  モデル  ")
    nb.add(tab_prompts,        text="  プロンプト設定  ")
    nb.add(tab_network,        text="  ネットワーク  ")
    nb.add(tab_train,          text="  学習設定  ")
    nb.add(tab_adv,            text="  詳細  ")
    nb.add(tab_layer,          text="  階層学習  ")
    nb.add(tab_monitor,        text="  モニターグラフ  ")
    nb.add(tab_monitor_layer,  text="  モニター階層  ")
    nb.add(tab_preset,         text="  プリセット  ")

    _build_model_tab(tab_model,       state)
    _build_prompts_tab(tab_prompts,   state)
    _build_network_tab(tab_network,   state)
    _build_train_tab(tab_train,       state)
    _build_adv_tab(tab_adv,           state)
    _build_layer_train_tab(tab_layer, state)
    _build_monitor_tab(tab_monitor,   state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_leco_preset_tab(tab_preset, state)"""

LECO_TRAIN_P2_NEW = """\
    tab_layer          = ttk.Frame(nb, padding=8)
    tab_sample         = ttk.Frame(nb, padding=8)
    tab_monitor        = ttk.Frame(nb, padding=8)
    tab_monitor_layer  = ttk.Frame(nb, padding=8)
    tab_preset         = ttk.Frame(nb, padding=8)

    nb.add(tab_model,          text="  モデル  ")
    nb.add(tab_prompts,        text="  プロンプト設定  ")
    nb.add(tab_network,        text="  ネットワーク  ")
    nb.add(tab_train,          text="  学習設定  ")
    nb.add(tab_adv,            text="  詳細  ")
    nb.add(tab_layer,          text="  階層学習  ")
    nb.add(tab_sample,         text="  サンプル生成  ")
    nb.add(tab_monitor,        text="  モニターグラフ  ")
    nb.add(tab_monitor_layer,  text="  モニター階層  ")
    nb.add(tab_preset,         text="  プリセット  ")

    _build_model_tab(tab_model,       state)
    _build_prompts_tab(tab_prompts,   state)
    _build_network_tab(tab_network,   state)
    _build_train_tab(tab_train,       state)
    _build_adv_tab(tab_adv,           state)
    _build_layer_train_tab(tab_layer, state)
    _build_leco_sample_tab(tab_sample, state)
    _build_monitor_tab(tab_monitor,   state)
    _build_monitor_layer_tab(tab_monitor_layer, state)
    _build_leco_preset_tab(tab_preset, state)"""

# ── P3-3: _build_command にサンプル引数追加 ──────────────────────────────────
LECO_TRAIN_P3_OLD = """\
    # 階層学習
    if s.layer_train_enabled.get():"""

LECO_TRAIN_P3_NEW = """\
    # サンプル生成
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        _spf = _leco_write_sample_prompt_file(s)
        cmd += ["--sample_every_n_steps", s.sample_every_n_steps.get().strip() or "100"]
        cmd += ["--sample_prompts",   str(_spf)]
        cmd += ["--sample_save_dir",  str(s.paths.root / "log" / "sample_gen")]
        if s.sample_keep_vae.get():
            cmd.append("--sample_keep_vae")

    # 階層学習
    if s.layer_train_enabled.get():"""

# ── P3-4: _validate にサンプルチェック追加 ───────────────────────────────────
LECO_TRAIN_P4_OLD = """\
    if s.max_train_steps.get() < 1:
        return "max_train_steps は1以上を指定してください。"
    if s.save_every_n_steps.get() < 1:
        return "save_every_n_steps は1以上を指定してください。"
    return None"""

LECO_TRAIN_P4_NEW = """\
    if s.max_train_steps.get() < 1:
        return "max_train_steps は1以上を指定してください。"
    if s.save_every_n_steps.get() < 1:
        return "save_every_n_steps は1以上を指定してください。"
    if s.sample_enabled.get() and not s.sample_prompt.get().strip():
        return "サンプルAが有効ですが、promptが未設定です。"
    if s.sample_b_enabled.get() and not s.sample_b_prompt.get().strip():
        return "サンプルBが有効ですが、promptが未設定です。"
    if s.sample_enabled.get() or s.sample_b_enabled.get():
        try:
            if int(s.sample_every_n_steps.get().strip() or "100") <= 0:
                return "サンプル出力のstep間隔は1以上を指定してください。"
        except ValueError:
            return "サンプル出力のstep間隔は整数で指定してください。"
    return None"""

# ── P3-5: プリセット保存の sample ブロック更新 ───────────────────────────────
LECO_TRAIN_P5_OLD = """\
            # サンプル生成（将来実装用プレースホルダ）
            "sample": {
                "enabled":         bool(s.sample_enabled.get()),
                "every_n_steps":   s.sample_every_n_steps.get(),
                "prompt":          s.sample_prompt.get(),
                "negative_prompt": s.sample_negative_prompt.get(),
                "width":           int(s.sample_width.get()),
                "height":          int(s.sample_height.get()),
                "steps":           int(s.sample_steps.get()),
                "scale":           float(s.sample_scale.get()),
                "flow_shift":      float(s.sample_flow_shift.get()),
                "seed":            s.sample_seed.get(),
            },"""

LECO_TRAIN_P5_NEW = """\
            # サンプル生成
            "sample": {
                "every_n_steps":            s.sample_every_n_steps.get(),
                "keep_vae":                 bool(s.sample_keep_vae.get()),
                "width":                    int(s.sample_width.get()),
                "height":                   int(s.sample_height.get()),
                "steps":                    int(s.sample_steps.get()),
                "scale":                    float(s.sample_scale.get()),
                "flow_shift":               float(s.sample_flow_shift.get()),
                "a_enabled":                bool(s.sample_enabled.get()),
                "a_prompt":                 s.sample_prompt.get(),
                "a_negative_prompt":        s.sample_negative_prompt.get(),
                "b_enabled":                bool(s.sample_b_enabled.get()),
                "b_prompt":                 s.sample_b_prompt.get(),
                "b_negative_prompt":        s.sample_b_negative_prompt.get(),
            },"""

# ── P3-6: プリセット読み込みの sample ブロック更新 ───────────────────────────
LECO_TRAIN_P6_OLD = """\
        # サンプル生成設定（将来実装用）
        sample = data.get("sample", {})
        if sample:
            _s(s.sample_enabled,         "enabled",         False)
            _s(s.sample_every_n_steps,   "every_n_steps",   "100")
            _s(s.sample_prompt,          "prompt",          "")
            _s(s.sample_negative_prompt, "negative_prompt", "")
            _s(s.sample_width,           "width",           512)
            _s(s.sample_height,          "height",          512)
            _s(s.sample_steps,           "steps",           20)
            _s(s.sample_scale,           "scale",           7.5)
            _s(s.sample_flow_shift,      "flow_shift",      3.0)
            _s(s.sample_seed,            "seed",            "42")"""

LECO_TRAIN_P6_NEW = """\
        # サンプル生成設定
        sample = data.get("sample", {})
        if sample:
            _s(s.sample_every_n_steps,      "every_n_steps",      "100")
            _s(s.sample_keep_vae,           "keep_vae",           False)
            _s(s.sample_width,              "width",              512)
            _s(s.sample_height,             "height",             512)
            _s(s.sample_steps,              "steps",              20)
            _s(s.sample_scale,              "scale",              7.5)
            _s(s.sample_flow_shift,         "flow_shift",         3.0)
            _s(s.sample_enabled,            "a_enabled",          False)
            _s(s.sample_prompt,             "a_prompt",           "")
            _s(s.sample_negative_prompt,    "a_negative_prompt",  "")
            _s(s.sample_b_enabled,          "b_enabled",          False)
            _s(s.sample_b_prompt,           "b_prompt",           "")
            _s(s.sample_b_negative_prompt,  "b_negative_prompt",  "")"""

LECO_TRAIN_PATCHES = [
    (LECO_TRAIN_P1_OLD, LECO_TRAIN_P1_NEW, "P3-1: _LecoTrainState サンプルB+keep_vae変数"),
    (LECO_TRAIN_P2_OLD, LECO_TRAIN_P2_NEW, "P3-2: タブにサンプル生成追加"),
    (LECO_TRAIN_P3_OLD, LECO_TRAIN_P3_NEW, "P3-3: _build_command サンプル引数"),
    (LECO_TRAIN_P4_OLD, LECO_TRAIN_P4_NEW, "P3-4: _validate サンプルチェック"),
    (LECO_TRAIN_P5_OLD, LECO_TRAIN_P5_NEW, "P3-5: プリセット保存 sample更新"),
    (LECO_TRAIN_P6_OLD, LECO_TRAIN_P6_NEW, "P3-6: プリセット読み込み sample更新"),
]

# ── P3-7: leco_train.py 末尾にヘルパー関数追加 ───────────────────────────────
LECO_TRAIN_APPEND = '''

# ──────────────────────────────────────────────────────────────────────────────
# サンプル生成ヘルパー（leco_train.py 用）
# ──────────────────────────────────────────────────────────────────────────────

def _leco_sample_dir(s: "_LecoTrainState") -> Path:
    return s.paths.root / "log" / "sample_gen"


def _leco_sample_prompt_path(s: "_LecoTrainState") -> Path:
    return _leco_sample_dir(s) / "_sample_prompt.txt"


def _leco_build_prompt_line(
    prompt: str, neg: str, s: "_LecoTrainState"
) -> str:
    width      = max(64, int(s.sample_width.get()))
    height     = max(64, int(s.sample_height.get()))
    steps      = max(1,  int(s.sample_steps.get()))
    scale      = float(s.sample_scale.get())
    flow_shift = float(s.sample_flow_shift.get())
    line = (
        f"{prompt} --w {width} --h {height} --s {steps} "
        f"--l {scale:g} --fs {flow_shift:g} --d {SAMPLE_FIXED_SEED}"
    )
    if neg:
        line += f" --n {neg}"
    return line


SAMPLE_FIXED_SEED = 42


def _leco_write_sample_prompt_file(s: "_LecoTrainState") -> Path:
    lines = []
    if s.sample_enabled.get() and s.sample_prompt.get().strip():
        lines.append(_leco_build_prompt_line(
            s.sample_prompt.get().strip(),
            s.sample_negative_prompt.get().strip(),
            s,
        ))
    if s.sample_b_enabled.get() and s.sample_b_prompt.get().strip():
        lines.append(_leco_build_prompt_line(
            s.sample_b_prompt.get().strip(),
            s.sample_b_negative_prompt.get().strip(),
            s,
        ))
    path = _leco_sample_prompt_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8", newline="\\n")
    return path


def _build_leco_sample_tab(parent: ttk.Frame, s: "_LecoTrainState") -> None:
    """LECO サンプル生成タブ。lora_train._build_sample_tab_common を流用。"""
    # lora_train モジュールが同一パッケージにある前提で動的インポート
    try:
        from . import lora_train as _lt
        _lt._build_sample_tab_common(parent, s, is_leco=True)
    except Exception:
        # フォールバック: 直接インポートが使えない場合は簡易UI
        _build_leco_sample_tab_inline(parent, s)


def _build_leco_sample_tab_inline(
    parent: ttk.Frame, s: "_LecoTrainState"
) -> None:
    """_build_sample_tab_common の leco_train 内スタンドアロン版。"""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    common = ttk.LabelFrame(parent, text="共通生成条件")
    common.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
    common.columnconfigure(1, weight=1)
    common.columnconfigure(3, weight=1)

    ttk.Label(common, text="step間隔", width=16, anchor=tk.W).grid(
        row=0, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=99999, textvariable=s.sample_every_n_steps, width=8).grid(
        row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text=f"seed固定: {SAMPLE_FIXED_SEED}", foreground="#64748B").grid(
        row=0, column=2, columnspan=2, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="width / height", width=16, anchor=tk.W).grid(
        row=1, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    sf = ttk.Frame(common)
    sf.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_width,  width=7).pack(side=tk.LEFT)
    ttk.Label(sf, text=" x ").pack(side=tk.LEFT)
    ttk.Spinbox(sf, from_=64, to=4096, increment=16, textvariable=s.sample_height, width=7).pack(side=tk.LEFT)

    ttk.Label(common, text="steps", width=10, anchor=tk.W).grid(
        row=1, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Spinbox(common, from_=1, to=1000, textvariable=s.sample_steps, width=8).grid(
        row=1, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Label(common, text="scale", width=16, anchor=tk.W).grid(
        row=2, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_scale, width=10).grid(
        row=2, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(common, text="flow_shift", width=10, anchor=tk.W).grid(
        row=2, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(common, textvariable=s.sample_flow_shift, width=10).grid(
        row=2, column=3, sticky=tk.W, padx=(0, 4), pady=3)

    ttk.Checkbutton(
        common, text="VAEを学習中保持 (sample_keep_vae)",
        variable=s.sample_keep_vae,
    ).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(4, 4), pady=3)

    ab_nb = ttk.Notebook(parent)
    ab_nb.grid(row=1, column=0, sticky=tk.NSEW)
    tab_a = ttk.Frame(ab_nb, padding=4)
    tab_b = ttk.Frame(ab_nb, padding=4)
    ab_nb.add(tab_a, text="  サンプルA  ")
    ab_nb.add(tab_b, text="  サンプルB  ")

    def _ab_panel(tab, enabled_var, prompt_var, neg_var, label):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
        top.columnconfigure(1, weight=1)
        ttk.Checkbutton(top, text=f"サンプル{label}を有効にする",
                        variable=enabled_var).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, padx=2, pady=2)
        _sdir = s.paths.root / "log" / "sample_gen" / f"sample_{label.lower()}"
        ttk.Label(top, text="出力先:", foreground="#475569").grid(
            row=1, column=0, sticky=tk.W, padx=(2, 0), pady=2)
        ttk.Label(top, text=str(_sdir), foreground="#1D4ED8").grid(
            row=1, column=1, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(top, text="prompt", width=16, anchor=tk.W).grid(
            row=2, column=0, sticky=tk.W, padx=(2, 2), pady=2)
        ttk.Entry(top, textvariable=prompt_var).grid(
            row=2, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)
        ttk.Label(top, text="negative", width=16, anchor=tk.W).grid(
            row=3, column=0, sticky=tk.W, padx=(2, 2), pady=2)
        ttk.Entry(top, textvariable=neg_var).grid(
            row=3, column=1, columnspan=3, sticky=tk.EW, padx=(0, 4), pady=2)

        gallery = ttk.LabelFrame(tab, text=f"最新サンプル{label}")
        gallery.grid(row=1, column=0, sticky=tk.NSEW)
        for c in range(5):
            gallery.columnconfigure(c, weight=1, uniform=f"lc_{label}")
        for r in range(2):
            gallery.rowconfigure(r, weight=1, uniform=f"lr_{label}")

        cells: list = []
        photo_refs: list = [None] * 10
        for idx in range(10):
            cell = ttk.Frame(gallery, padding=4)
            cell.grid(row=idx // 5, column=idx % 5, sticky=tk.NSEW)
            cell.columnconfigure(0, weight=1)
            cell.rowconfigure(0, weight=1)
            il = ttk.Label(cell, anchor=tk.CENTER)
            il.grid(row=0, column=0, sticky=tk.NSEW)
            el = ttk.Label(cell, text="step -", anchor=tk.CENTER)
            el.grid(row=1, column=0, sticky=tk.EW, pady=(3, 0))
            cells.append((il, el))

        def _re_search(pat, text):
            import re as _re
            m = _re.search(pat, text)
            return m

        def _refresh(schedule_next=False):
            files = sorted(
                _sdir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:10] if _sdir.exists() else []
            try:
                from PIL import Image as _Im, ImageTk as _ITk
            except Exception:
                _Im = _ITk = None
            for idx, (il, el) in enumerate(cells):
                if idx >= len(files):
                    il.configure(image="", text="")
                    el.configure(text="step -")
                    photo_refs[idx] = None
                    continue
                p = files[idx]
                m = _re_search(r"step([0-9]+)", p.stem)
                el.configure(text=f"step {m.group(1)}" if m else p.name)
                if _Im is None:
                    il.configure(image="", text=p.name)
                    photo_refs[idx] = None
                    continue
                try:
                    with _Im.open(p) as im:
                        im.thumbnail((220, 220))
                        ph = _ITk.PhotoImage(im.copy())
                    photo_refs[idx] = ph
                    il.configure(image=ph, text="")
                except Exception:
                    il.configure(image="", text=p.name)
                    photo_refs[idx] = None
            if schedule_next:
                tab.after(2000, lambda: _refresh(True))

        btn_row = ttk.Frame(top)
        btn_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 2))
        ttk.Button(btn_row, text="表示更新", command=lambda: _refresh(False)).pack(
            side=tk.LEFT, padx=(0, 6))
        _refresh(True)

    _ab_panel(tab_a, s.sample_enabled,   s.sample_prompt,   s.sample_negative_prompt,   "A")
    _ab_panel(tab_b, s.sample_b_enabled, s.sample_b_prompt, s.sample_b_negative_prompt, "B")
'''


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leco_script", default=str(Path("sd-scripts") / "anima_train_leco.py"))
    ap.add_argument("--lora_train",  default=str(Path("app") / "lora_train.py"))
    ap.add_argument("--leco_train",  default=str(Path("app") / "leco_train.py"))
    args = ap.parse_args()

    targets = [
        (Path(args.leco_script), LECO_SCRIPT_PATCHES, LECO_APPEND,       "anima_train_leco.py"),
        (Path(args.lora_train),  LORA_PATCHES,        "",                  "lora_train.py"),
        (Path(args.leco_train),  LECO_TRAIN_PATCHES,  LECO_TRAIN_APPEND,  "leco_train.py"),
    ]

    import ast
    all_ok = True
    for path, patches, append, name in targets:
        print(f"\n{'='*60}")
        print(f"  {name}  ({path})")
        print(f"{'='*60}")
        if not path.exists():
            print(f"  [SKIP] ファイルが見つかりません: {path}")
            all_ok = False
            continue
        try:
            _patch(path, patches, append)
            # 構文チェック
            src = path.read_text(encoding="utf-8")
            ast.parse(src)
            print(f"  構文チェック: OK")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            all_ok = False

    print("\n" + ("=" * 60))
    if all_ok:
        print("  全パッチ適用完了")
    else:
        print("  一部失敗あり。上記エラーを確認してください。")


if __name__ == "__main__":
    main()
