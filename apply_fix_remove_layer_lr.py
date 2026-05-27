"""
apply_fix_remove_layer_lr.py

変更内容:
  lora_train.py からネットワークタブの「階層別学習率」を完全削除する。

  削除箇所 (6箇所):
    1. _TrainState.__init__: self_attn_lr / cross_attn_lr / mlp_lr / llm_adapter_lr 変数定義
    2. _build_network_tab: UIブロック（LabelFrame + _lr_pair呼び出し）
    3. _lr_pair 関数定義
    4. _collect(): プリセット収集の4行
    5. _apply():   プリセット復元の4行
    6. _build_command(): コマンド生成ブロック

適用方法:
  python apply_fix_remove_layer_lr.py
"""
from __future__ import annotations
import sys
from pathlib import Path


def _adapt(src: str, pat: str, rep: str):
    lf_only = "\r\n" not in src
    if lf_only:
        pat = pat.replace("\r\n", "\n")
        rep = rep.replace("\r\n", "\n")
    else:
        pat = pat.replace("\n", "\r\n")
        rep = rep.replace("\n", "\r\n")
    return pat, rep


def apply_patch(filepath: Path, old: str, new: str, label: str) -> None:
    src = filepath.read_text(encoding="utf-8")
    old_a, new_a = _adapt(src, old, new)
    count = src.count(old_a)
    if count == 0:
        print(f"[ERROR] パターンが見つかりません: {label}")
        print("  先頭60文字:", repr(old_a[:60]))
        sys.exit(1)
    if count > 1:
        print(f"[ERROR] パターンが複数({count}箇所)マッチします: {label}")
        sys.exit(1)
    filepath.write_text(src.replace(old_a, new_a, 1), encoding="utf-8")
    print(f"[OK] {label}")


ROOT       = Path(__file__).parent
LORA_TRAIN = ROOT / "app" / "lora_train.py"

# ─────────────────────────────────────────────────────────────────
# 1. _TrainState.__init__ — 変数定義4行を削除
# ─────────────────────────────────────────────────────────────────
P1_OLD = """\
        # 階層別LR (Anima固有)
        self.self_attn_lr   = tk.StringVar(value="")
        self.cross_attn_lr  = tk.StringVar(value="")
        self.mlp_lr         = tk.StringVar(value="")
        self.llm_adapter_lr = tk.StringVar(value="")

"""
P1_NEW = """\
"""

# ─────────────────────────────────────────────────────────────────
# 2. _build_network_tab — UIブロック削除
# ─────────────────────────────────────────────────────────────────
P2_OLD = """\
    # Anima 階層別LR
    lf2 = ttk.LabelFrame(parent, text="階層別学習率 (空欄=base LRを使用 / 0=freeze)")
    lf2.pack(fill=tk.X)
    lf2.columnconfigure(1, weight=1)
    lf2.columnconfigure(3, weight=1)

    _lr_pair(lf2, 0, "self_attn_lr",   s.self_attn_lr,
                   "cross_attn_lr", s.cross_attn_lr)
    _lr_pair(lf2, 1, "mlp_lr",         s.mlp_lr,
                   "llm_adapter_lr", s.llm_adapter_lr)

"""
P2_NEW = """\
"""

# ─────────────────────────────────────────────────────────────────
# 3. _lr_pair 関数定義を削除
# ─────────────────────────────────────────────────────────────────
P3_OLD = """\
def _lr_pair(parent, row, label1, var1, label2, var2):
    ttk.Label(parent, text=label1, width=16, anchor=tk.W).grid(
        row=row, column=0, sticky=tk.W, padx=(4, 2), pady=3)
    ttk.Entry(parent, textvariable=var1, width=12).grid(
        row=row, column=1, sticky=tk.W, padx=(0, 12), pady=3)
    ttk.Label(parent, text=label2, width=16, anchor=tk.W).grid(
        row=row, column=2, sticky=tk.W, padx=(0, 2), pady=3)
    ttk.Entry(parent, textvariable=var2, width=12).grid(
        row=row, column=3, sticky=tk.W, padx=(0, 4), pady=3)

"""
P3_NEW = """\
"""

# ─────────────────────────────────────────────────────────────────
# 4. _collect() — プリセット収集4行を削除
# ─────────────────────────────────────────────────────────────────
P4_OLD = """\
            "self_attn_lr":      s.self_attn_lr.get(),
            "cross_attn_lr":     s.cross_attn_lr.get(),
            "mlp_lr":            s.mlp_lr.get(),
            "llm_adapter_lr":    s.llm_adapter_lr.get(),
"""
P4_NEW = """\
"""

# ─────────────────────────────────────────────────────────────────
# 5. _apply() — プリセット復元4行を削除
# ─────────────────────────────────────────────────────────────────
P5_OLD = """\
        _s(s.self_attn_lr,      "self_attn_lr",       "")
        _s(s.cross_attn_lr,     "cross_attn_lr",      "")
        _s(s.mlp_lr,            "mlp_lr",             "")
        _s(s.llm_adapter_lr,    "llm_adapter_lr",     "")
"""
P5_NEW = """\
"""

# ─────────────────────────────────────────────────────────────────
# 6. _build_command() — コマンド生成ブロックを削除
# ─────────────────────────────────────────────────────────────────
P6_OLD = """\
    # 階層別LR
    for flag, var in [
        ("--self_attn_lr", s.self_attn_lr),
        ("--cross_attn_lr", s.cross_attn_lr),
        ("--mlp_lr", s.mlp_lr),
        ("--llm_adapter_lr", s.llm_adapter_lr),
    ]:
        v = var.get().strip()
        if v:
            cmd += [flag, v]

"""
P6_NEW = """\
"""


def main() -> None:
    print("=== apply_fix_remove_layer_lr.py ===")
    apply_patch(LORA_TRAIN, P1_OLD, P1_NEW, "_TrainState 変数定義削除")
    apply_patch(LORA_TRAIN, P2_OLD, P2_NEW, "_build_network_tab UIブロック削除")
    apply_patch(LORA_TRAIN, P3_OLD, P3_NEW, "_lr_pair 関数定義削除")
    apply_patch(LORA_TRAIN, P4_OLD, P4_NEW, "_collect() プリセット収集削除")
    apply_patch(LORA_TRAIN, P5_OLD, P5_NEW, "_apply() プリセット復元削除")
    apply_patch(LORA_TRAIN, P6_OLD, P6_NEW, "_build_command() コマンド生成削除")
    print("=== 全パッチ適用完了 ===")


if __name__ == "__main__":
    main()
