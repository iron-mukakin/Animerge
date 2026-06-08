# LECO学習GUI バグ修正 引き継ぎ仕様書

作成日: 2026-06-08  
対象リポジトリ: E:\Animerge\app\

---

## 現状サマリー

leco_train.py に対して本セッションで複数パッチを適用した。  
パッチ適用状況と残存バグを以下に整理する。

---

## 適用済みパッチ一覧

| パッチファイル | 対象 | 内容 | 状態 |
|---|---|---|---|
| apply_fix_leco_train.py | leco_train.py | バグ1: \r 未処理によるログ乱れ / バグ2: photo_refs GC問題 + 例外可視化 | 適用済み |
| apply_fix_leco_wrapper.py | leco_train.py | _gui_leco_wrapper.py のハードコードパスを __file__ 基準の動的解決に変更 | 適用済み |
| apply_fix_sample_preview.py | lora_train.py, leco_train.py | lora_train の is_leco=True 時の glob パターン修正 / leco_train の相対インポート→絶対インポート | 適用済み（ただし後述の問題あり） |
| apply_fix_leco_final.py | lora_train.py, leco_train.py | lora_train の glob パターンをプレフィックスで絞る / Popen bufsize=1 追加 | 適用済み（ただし後述の問題あり） |

---

## 現在の leco_train.py の状態（確認済み）

- L826: `bufsize=1` 追加済み ✓
- L836-848: `\r` 分割処理済み ✓
- L639: wrapper の動的パス生成済み ✓
- L606-608: `_log_primary_set` フラグによるログウィジェット単一登録済み ✓
- L610-624: `_drain` が `s._log_widgets[0]` への単一書き込みに変更済み ✓
- L1641-1642: `_glob_pat` を手動で `*_00_*.png` / `*_01_*.png` に変更済み ✓
- L1682: `_SAMPLE_DEBUG = True` ✓
- L1684-1737: `_refresh` の photo_refs → `il._photo_ref` 変更済み ✓

---

## 残存バグと原因

### バグA: プレビューに lora 画像と leco 画像が混在表示される

**現象:**  
leco学習のサンプルプレビュータブに `lora_output_e000001_00_*.png` と `leco_output_000002_00_*.png` の両方が表示される。

**原因:**  
`_ab_panel` の `_glob_pat = "*_00_*.png"` は `lora_output_e000001_00_*` にもマッチする。  
`sample_gen` ディレクトリは lora/leco 共用のため両方のファイルが混在する。

**正しい修正:**  
leco_train.py L1642 の `_glob_pat` をプレフィックスで絞る。

```python
# 変更前（現状）
_glob_pat = "*_00_*.png" if label == "A" else "*_01_*.png"

# 変更後
_glob_pat = "leco_output_*_00_*.png" if label == "A" else "leco_output_*_01_*.png"
```

**影響範囲:** leco_train.py L1642 のみ。1行の変更。

---

### バグB: プレビューにステップ番号が正しく表示されない

**現象:**  
ファイル名 `leco_output_000002_00_20260608191028_42.png` に対してステップ表示が `step 191028`（タイムスタンプ部分）になる、または正しく表示されない。

**原因:**  
L1707 の正規表現 `r"_([0-9]{6})_"` は6桁の数字を最初にマッチするが、  
`leco_output_000002_00_20260608191028_42.png` の stem では：
- `_000002_` → マッチ（正しいステップ番号）
- `_191028_` → マッチ（タイムスタンプ末尾6桁）

`re.search` は最初のマッチを返すため `000002` を取得するはずだが、  
ファイル名の命名規則が `{name}_{step6}_{idx2}_{datetime14}_{seed}.png` であるため、  
`_000002_` の前の `leco_output` プレフィックス部分に `_` が含まれない。  
実際には正しく `000002` が取れているが、`int("000002") = 2` となり `step 2` と表示される。これは正常動作。

**lora 学習との差異:**  
lora_train.py の `_build_sample_tab_common` はステップではなくエポックで出力するため、  
ファイル名に `_e000001_` のように `e` プレフィックスがつく。  
LECOはステップ番号のみ（`_000002_`）。  
プレビュー下の `step X` 表示は LECO では正しく「step番号」を示しているが、  
lora 学習のファイルが混在している場合 `_e000001_` の `000001` が抽出されて `step 1` になる。

**正しい修正:**  
バグAの glob パターン修正（プレフィックス絞り込み）により lora ファイルが除外されれば  
ステップ表示の混乱も自動的に解消する。追加修正不要。

---

### バグC: ログ出力の二重表示（tqdm + INFO 行の混在）

**現象:**  
学習ログに以下のように tqdm 進捗行と INFO 行が連結して出力される：
```
steps:   0%|          | 2/500 [00:12<54:06,  6.52s/it, loss=0.0031]2026-06-08 18:49:56 INFO     [SampleGen] ...
```

**原因の確定:**  
- leco_train.py 側の `\r` 分割処理（L836-848）は正しく動作している
- `bufsize=1`（L826）も適用済み
- 根本的な原因はサブプロセス（accelerate 経由の Python）側がパイプ出力をフルバッファリングしており、  
  tqdm の `\r` 行と直後の INFO 行が同一チャンクとして親プロセスに届く
- `bufsize=1` は親プロセス側の読み取りバッファを行単位にするが、  
  子プロセス側のバッファリングは制御できない

**根本的解決策:**  
子プロセス側で `PYTHONUNBUFFERED=1` 環境変数を設定してバッファリングを無効にする。

leco_train.py の `_start_training` 内の `env` 構築部分に追加：

```python
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"   # ← この行を追加
```

`env` が既に構築されている場合は `env["PYTHONUNBUFFERED"] = "1"` を `Popen` の前に追加する。

**確認方法:**  
`env` の構築箇所は leco_train.py の `_start_training` 関数内（L800付近）。

---

### バグD: _refresh デバッグログが GUI のログ欄に出力されない

**現象:**  
`_SAMPLE_DEBUG = True` に設定しているが、デバッグ出力が学習ログ欄に表示されない。

**原因:**  
`_refresh` 内のデバッグ出力は `logging.getLogger(__name__)` を使用している。  
GUI のログ欄は `s._log_queue` に `put` された文字列を `_drain` で取り出して表示する仕組みであり、  
Python の `logging` モジュールの出力は `s._log_queue` に入らない。

**正しい対応:**  
デバッグ出力を `logging` ではなく `s._log_queue.put()` で行う。  
または `_SAMPLE_DEBUG = False` に戻した上でバグAを修正することで問題を解消する。

---

## 修正優先順位

| 優先度 | バグ | 修正箇所 | 難易度 |
|---|---|---|---|
| 高 | A: 画像混在 | leco_train.py L1642 1行 | 低 |
| 中 | C: ログ混在 | leco_train.py _start_training の env 設定 | 低 |
| 低 | D: デバッグログ不可視 | _SAMPLE_DEBUG を False に戻す | 低 |
| 不要 | B: step表示 | バグA解消で連動解消 | — |

---

## 修正手順

### STEP 1: バグA修正（leco_train.py L1642）

```python
# 変更前
_glob_pat = "*_00_*.png" if label == "A" else "*_01_*.png"

# 変更後
_glob_pat = "leco_output_*_00_*.png" if label == "A" else "leco_output_*_01_*.png"
```

### STEP 2: バグC修正（PYTHONUNBUFFERED）

leco_train.py L815-816 の `env` 構築部分に1行追加：

```python
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONUNBUFFERED"] = "1"   # ← 追加
```

### STEP 3: バグD修正（_SAMPLE_DEBUG を False に戻す）

L1682: `_SAMPLE_DEBUG: bool = True` → `_SAMPLE_DEBUG: bool = False`

---

## lora_train.py の状態（確認済み）

L1256-1257 の glob パターン:
```python
pat_a = "leco_output_*_00_*.png" if is_leco else "*_e*_00_*.png"
pat_b = "leco_output_*_01_*.png" if is_leco else "*_e*_01_*.png"
```
適用済み ✓

ただし `_build_leco_sample_tab` からの `importlib` 経由呼び出しを経由した場合のみこのパターンが使われる。  
現状の leco_train.py は `_build_leco_sample_tab_inline` の独自実装（`_ab_panel`）を使っており、  
`lora_train._build_sample_tab_common` は**呼ばれていない**。  
理由: `_build_leco_sample_tab` の `try` ブロックでの importlib 呼び出しが  
`lora_train.py` のトップレベル実行（GUI 依存コード）で副作用を起こして失敗し、  
`except Exception` のフォールバックで `_build_leco_sample_tab_inline` が呼ばれている可能性が高い。

この経路の確認・統一は別タスクとして扱う。現状の `_ab_panel` 実装で動作は可能であるため、  
STEP 1 の L1642 修正のみで画像混在バグは解消できる。

---

## ファイル構成（関連ファイル）

```
E:\Animerge\
├── app\
│   ├── leco_train.py       ← 主要修正対象
│   ├── lora_train.py       ← is_leco glob パターン修正済み
│   ├── gui.py
│   └── config.py
├── sd-scripts\
│   ├── anima_train_leco.py
│   ├── anima_sample_gen.py ← サンプル画像の命名規則はここで定義
│   └── _gui_leco_wrapper.py（学習開始時に自動生成）
└── log\
    └── sample_gen\         ← lora/leco 共用サンプル出力先
        ├── leco_output_NNNNNN_00_YYYYMMDDHHMMSS_SEED.png
        ├── leco_output_NNNNNN_01_YYYYMMDDHHMMSS_SEED.png
        ├── lora_output_eNNNNNN_00_YYYYMMDDHHMMSS_SEED.png
        └── lora_output_eNNNNNN_01_YYYYMMDDHHMMSS_SEED.png
```

---

## サンプル画像命名規則

| 学習種別 | 命名パターン | ステップ/エポック |
|---|---|---|
| LECO | `leco_output_{step:06d}_{idx:02d}_{datetime}_{seed}.png` | ステップ番号 |
| LoRA | `lora_output_e{epoch:06d}_{idx:02d}_{datetime}_{seed}.png` | エポック番号（`e` プレフィックス） |

LECO のプレビュー下表示 `step 2` は `leco_output_000002_` から正しく `2` を抽出している。  
LoRA のプレビュー下表示 `step 1` は `lora_output_e000001_` から `000001` を抽出しているが  
表示ラベルが `step` になっている（正しくは `epoch`）。ただしこれは lora_train.py 側の表示の問題であり、今回のスコープ外。
