# キー名称不一致問題 引継ぎメモ

作成日: 2026-05-21
対象ファイル: merge.py / analysis.py / gui.py

\---

## 問題の核心

モデルやLoRAのファイル形式によってキープレフィックスが異なる。
anima-base-v1.0 の標準は `net.blocks.N.xxx.weight` 形式（`net.` 付き）だが、
他形式のファイルは `model.diffusion\_model.blocks.N.xxx.weight` や
`blocks.N.xxx.weight`（プレフィックスなし）など多様な形式を持つ。
これをまたいでマージ・分析すると、キーが一致しない問題が発生する。

\---

## キー変換関数の一覧と役割

|関数|入力|出力|用途|
|-|-|-|-|
|`canonical\_key(name)`|任意のモデルキー|プレフィックスなし裸のキー|マッチング用の内部正規化。`model.diffusion\_model.` / `net.` 等を除去|
|`canonical\_lora\_key(name)`|任意のLoRAキー|プレフィックス・`lora\_A/B`表記を統一した裸のキー|LoRAマッチング用|
|`anima\_v1\_key(name)`|任意のモデルキー|`net.xxx` 形式|anima-base-v1.0 形式への変換|
|`output\_key\_name(name, options)`|モデルキー|format設定に応じた出力キー|本体マージ・フューズの出力キー決定|
|`output\_lora\_key\_name(key, options)`|LoRAキー|`lora\_unet\_blocks\_N\_xxx.{suffix}` 形式|LoRAマージ・差分抽出の出力キー決定|
|`lora\_target\_candidates(name)`|LoRAキー|ベース層キー候補リスト（優先順）|LoRAキー→モデルキーの対応付け|

### canonical\_key の除去対象プレフィックス

```
model.diffusion\_model.
diffusion\_model.
model.model.
model.
module.
state\_dict.
net.
```

\---

## ファイル形式別キー構造

|形式|ファイル内の実キー例|canonical\_key 後|
|-|-|-|
|anima-base-v1.0|`net.blocks.0.self\_attn.q\_proj.weight`|`blocks.0.self\_attn.q\_proj.weight`|
|SD-UNet系|`model.diffusion\_model.input\_blocks.0.0.weight`|`input\_blocks.0.0.weight`|
|プレフィックスなし|`blocks.0.self\_attn.q\_proj.weight`|`blocks.0.self\_attn.q\_proj.weight`|
|LoRA標準|`lora\_unet\_blocks\_0\_self\_attn\_q\_proj.lora\_up.weight`|—|
|LoRA A/B表記|`lora\_unet\_blocks\_0\_self\_attn\_q\_proj.lora\_A.weight`|lora\_down に統一|

\---

## マージ処理での正規化の流れ

```
入力ファイルのキー
    │
    ├─ canonical\_key / canonical\_lora\_key
    │       └─ プレフィックス除去 → マッチング用内部キー（保存しない）
    │
    ├─ マッチング（canonical同士を比較）
    │
    └─ output\_key\_name / output\_lora\_key\_name
            └─ options.output\_key\_format が "anima-base-v1.0" なら net. 付与
               → 保存ファイルへの実際のキー
```

\---

## 各マージ関数とキー処理

### merge\_models（本体マージ）

* 入力キーの照合: `canonical\_key` でプレフィックスを除いて比較
* 出力キー: `output\_key\_name(key, options)` → anima形式なら `net.` 付与
* ログ: 正規化件数・実マージ対象数を出力

### merge\_loras（LoRA同士マージ）

* 入力キーの照合: `canonical\_lora\_key` で比較
* 出力キー: `output\_lora\_key\_name(key, options)` → `lora\_unet\_blocks\_N\_xxx.suffix` に統一
* ログ: 正規化件数・実マージ対象数（`.alpha` 除く）を出力

### fuse\_lora\_into\_model（LoRA→本体フューズ）

* ベースモデルキー: `output\_key\_name` で正規化しつつ保持
* LoRAの対応先探索: `lora\_target\_candidates` の候補リストを順に `canonical\_key` で照合

  * **候補リストの先頭は `net.{converted}` 形式**（anima-base-v1.0 優先）
* ログ: ベースモデルの正規化件数・フューズペア数を出力

### extract\_lora\_difference（本体差分→LoRA抽出）

* 入力照合: `canonical\_key` でプレフィックスを除いて比較
* 出力LoRAキー: `output\_lora\_key\_name` で正規化

  * `canonical\_key` で余分プレフィックス除去 → `lora\_unet\_{base\_layer}` 形式
* ログ: 候補層数・正規化件数を出力

\---

## analysis.py 側のキー処理（key\_correction オプション）

|key\_correction|処理内容|
|-|-|
|False（デフォルト）|LoRAキーを簡易変換のみ。`net.` は付与しない|
|True|ベースモデルキーに `anima\_v1\_key()` で `net.` 付与。LoRAキーに `lora\_target\_candidates()\[0]`（`net.` 付き先頭）を使用|

`key\_correction=True` でLoRAとベースモデルを比較分析する場合、
両者のキーが `net.blocks.N.xxx.weight` 形式で統一されることでマッチングが成立する。

\---

## 解決済み・残課題の状態

|項目|状態|
|-|-|
|`lora\_target\_candidates` 候補先頭を `net.` 付きに|**解決済み**（merge.py 現状で既に先頭）|
|全マージ関数の出力キーを `output\_key\_name` / `output\_lora\_key\_name` で正規化|**解決済み**（merge\_fix.py パッチ適用済み）|
|ログの層数分母を実処理数に統一|**解決済み**（merge\_fix.py パッチ適用済み）|
|キー正規化が発生した際の実行ログ通知|**解決済み**（merge\_fix.py パッチ適用済み）|
|analysis.py の `key\_correction=True` 時のゼロマッチ|**解決済み**（`lora\_target\_candidates` 先頭修正が前提、現状解決）|
|`extract\_lora\_difference` の root 生成が `canonical\_key` 打ち消し問題|**解決済み**（`output\_lora\_key\_name` で出力キーを正規化）|

\---

## 次セクションで作業する場合の注意点

1. `options.output\_key\_format` が `"anima-base-v1.0"` でないと正規化は実行されない。
GUI の出力形式設定を確認すること。
2. 新たなマージ関数やキー変換を追加する場合は、

   * モデルキー → `output\_key\_name`
   * LoRAキー → `output\_lora\_key\_name`
を出力キー決定に必ず使うこと。`canonical\_key` は照合専用であり保存キーに使わない。
3. `canonical\_key` と `output\_key\_name` は逆方向の操作（除去 vs 付与）なので
両方を連続して使うと打ち消しになる（handover 課題2 の根本原因）。

