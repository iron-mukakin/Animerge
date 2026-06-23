# ADDifT DPOモード拡張 — 引き継ぎ仕様書

最終更新: 2026-06-21（本セッション終了時点）

---

## 1. プロジェクト概要

ADDifT（差分転写学習）にDPO（Diffusion Direct Preference Optimization）モードをオプション機能として追加する開発。理論的根拠は `dpo.txt`（論文 *Diffusion Model Alignment Using Direct Preference Optimization* の要約）。

### 対象ファイルと役割

| ファイル | 役割 | 配置先 |
|---|---|---|
| `addift_train.py` | 学習設定GUI本体（Tkinter） | `app/` |
| `addift_dpo_ui.py` | DPOモード関連UIフック（新規モジュール、行数分離のため独立） | `app/` |
| `anima_train_addift.py` | 学習本体スクリプト（CLI、accelerate launch対象） | `sd-scripts/` |
| `monitor_graph_addift.py` | 学習モニターグラフ・EarlyStopping実装 | `app/` |
| `texts_ja.json` | 日本語UIテキスト | `configs/` |
| `dpo.txt` | DPO論文の要約（理論的根拠の原典） | 参考資料 |

### 現在の行数（2000行制限に対する状況）

| ファイル | 行数 |
|---|---|
| `addift_train.py` | 1888行 |
| `addift_dpo_ui.py` | 297行 |
| `anima_train_addift.py` | 911行 |
| `monitor_graph_addift.py` | 870行 |

いずれも制限内。`addift_train.py`が最も逼迫しているため、今後の追加機能は極力`addift_dpo_ui.py`側に実装すること。

---

## 2. 設計方針（確定事項）

### 2.1 アーキテクチャ: A案（既存ADDifT実装の近似流用）

- 別途`ref_model`コピーを持たない。既存ADDifTのLoRA `multiplier` ON/OFF切替を、それぞれ policy（multiplier=有効値）／reference（multiplier=0、無効＝基準モデル相当）として流用
- 理由: 論文準拠のB案（別モデルコピー＋複数win/loseペアのデータセット）はVRAM消費が実質倍増し、データセット機構の新規実装も必要なため見送り
- データセットは既存ADDifTと同じ「画像1ペア固定」構成のまま。`image_a`=lose（変換前/好ましくない）、`image_b`=win（変換後/好ましい）に意味づけ
- turn反転（A↔B交互学習）はDPOモード時は**行わない**。win/loseの役割は固定

### 2.2 損失関数

`anima_train_addift.py` の `compute_dpo_loss()` 参照。

```
loss_win  = MSE(policy_pred_win,  noise_win)
ref_loss_win  = MSE(reference_pred_win,  noise_win)
loss_lose = MSE(policy_pred_lose, noise_lose)
ref_loss_lose = MSE(reference_pred_lose, noise_lose)

win_delta  = loss_win  - ref_loss_win
lose_delta = loss_lose - ref_loss_lose

delta_error = win_delta - lose_delta
dpo_logits  = -preference_beta * delta_error
loss        = -log(sigmoid(dpo_logits))

# win_aux_weight > 0 の場合、DPO-Positive方式の補助項を加算:
loss += win_aux_weight * max(0, win_delta)
```

win/loseそれぞれ独立にノイズをサンプリングする（dpo.txt原文に忠実。既存ADDifTのような「A/B共有ノイズによるクロス比較」は採用していない）。

### 2.3 正規化指標（EarlyStoppingDPO・モニターグラフ用）

referenceモデルがwin/loseをもともと得意/不得意とするベースライン誤差スケールの違いを打ち消すため、以下を「Win Loss」「Lose Loss」としてログ・グラフ・EarlyStoppingDPOで使用する。

```
win_loss(表示用)  = win_delta  / (ref_loss_win  + 1e-6)
lose_loss(表示用) = lose_delta / (ref_loss_lose + 1e-6)
```

生の差分値（非正規化）は `win_loss_raw` / `lose_loss_raw` として tensorboard ログにのみ別途記録（tqdm postfixには出さない）。

### 2.4 ハイパーパラメータのデフォルト値

| パラメータ | デフォルト | 備考 |
|---|---|---|
| `preference_beta` | 5.0 | 論文値(2000〜5000)とは損失スケールが異なるため小さい値を採用。実測ベースで決定 |
| `win_aux_weight` | **0.1**（本セッションで1.0→0.1へ変更決定。**UI未反映、要修正**） | DPO-Positive方式の補助ペナルティ重み。0で無効 |
| `es_dpo_patience` | 10 step | EarlyStoppingDPOの監視ウィンドウ |

---

## 3. これまでの実装履歴（時系列サマリ）

1. **初期実装**: DPOモードのON/OFFチェックボックス＋モード選択(DPO/SDPO/MaPO、後者2つは未実装プレースホルダ)をADDifTパラメータ枠の直前に追加。データセットタブの画像A/Bラベルをwin/lose表記へ動的切替
2. **学習ロジック実装**: `anima_train_addift.py`にDPO損失計算を追加。turn反転を停止し役割固定
3. **win/lose符号の検証**: ユーザーから「マイナス適応が必要では」との懸念 → 数式検証の結果、符号は正しいことを確認。ただし既存ADDifT(クロス画像比較)とDPO実装(独立ノイズ)の設計差異を整理
4. **モニター機能追加**: 実行ログへの`win_loss`/`lose_loss`出力、モニターグラフへの3系列表示(後に4分割グラフへ)、EarlyStoppingDPOパネル新設（詳細タブ・EarlyStopping直下）
5. **不具合修正(複数回)**:
   - パネル構築順序バグ(`_es_dpo_frame`未定義参照)
   - グラフが通常時も4分割のまま固定表示される不具合 → `_rebuild_graph_layout()`でモード変更時のみ動的に1x2⇔2x2を再構築する方式へ修正
   - EarlyStoppingDPOパネルが自動レポートパネルより後ろに配置される不具合 → `pack(after=es_lf)`固定+状態変化時のみpack操作する方式へ修正
   - 通常時グラフが「左右2分割（縦長）」になっていた → 「上下2分割（横長）」へ修正(`add_gridspec(2,1)`)
6. **実機ログ解析(複数回)**:
   - `preference_beta=0.1`では小さすぎて勾配がほぼ消失することを確認（loss≒0.693固定）
   - `preference_beta=10.0`でも win_loss側が改善しにくい現象を確認 → 文献調査の結果、DPO全般で知られる**尤度置換(likelihood displacement)現象**であると特定。lose側を悪化させる方が損失低減の「近道」になりやすいというDPO損失の構造的性質
   - 「reference(基準モデル)がwin画像をもともと得意としている場合、win_lossの改善余地(フロア)が狭くなる」という懸念に対し、`ref_loss_win`による正規化で対処する設計を決定
7. **win_aux_weight実装**: DPO-Positive方式（Pal et al., 2024 *Smaug*論文）を参考に、win側がreferenceより悪化した場合のみ働く補助ペナルティを追加。正規化指標をEarlyStoppingDPO/グラフ双方に適用
8. **実機検証**: `win_aux_weight=0.1`でwin_loss側の改善傾向を確認。一方で「EarlyStoppingDPOの停止頻度低下はwin_aux_weightの効果であり、正規化自体は判定の感度(ノイズへの脆弱性)を緩和していない」という鋭い指摘をユーザーから受け、原因を特定（次節参照）

---

## 4. 未着手の決定事項（次セッションでの実装対象）

### 4.1 `win_aux_weight` デフォルト値変更: 1.0 → **0.1**

**現状**: `addift_dpo_ui.py` の `_DEFAULT_WIN_AUX_WEIGHT = 1.0`（UI初期値）。`anima_train_addift.py` のCLI引数デフォルトは元々 `0.0`（無効）なので変更不要、**UI側のみ修正**すればよい。

**対応箇所**: `addift_dpo_ui.py` 内 `_DEFAULT_WIN_AUX_WEIGHT` 定数を `0.1` に変更するのみ。

### 4.2 EarlyStoppingDPOを「トレンド判定（傾き判定）」方式へ変更

**現状の問題**: `_check_es_dpo()`（`monitor_graph_addift.py`）が前stepとの単純比較(`win_loss[今回] < win_loss[前回]`)のみで判定しているため、正規化してもstep単位のノイズに対する過敏さは変わらない。

**決定した方式**: 直近N step（監視ウィンドウ = `es_dpo_patience`を流用、またはこの値とは別に新規ウィンドウ幅パラメータを設けるか要検討）の`win_loss`系列・`lose_loss`系列それぞれに対して**線形回帰の傾き(slope)**を計算し、

```
healthy = (win_loss の傾き < 0) かつ (lose_loss の傾き > 0)
```

で判定する方式へ変更する。傾き計算は最小二乗法（`numpy.polyfit`相当、または手計算の単純線形回帰式）で十分。

**設計上の論点（次セッションで要検討）**:
- 傾き判定用のウィンドウ幅は`es_dpo_patience`と同じ値を流用するか、独立した新パラメータ(`es_dpo_window`等)にするか
- ウィンドウ内のサンプル数が足りない学習序盤の扱い（NaN回避、判定スキップ）
- 「カウント+1／リセット」の継続判定ロジック自体は維持するか、傾きの正負を直接ステータス表示するだけに簡略化するか
- UIへの新規パラメータ追加が必要な場合、詳細タブのEarlyStoppingDPO枠への追加実装が必要

### 4.3 マスク使用時の損失希釈バグ修正

**問題箇所**: `anima_train_addift.py` 約229行目 `compute_addift_loss()` 内（同様のパターンが `compute_dpo_loss()` にも存在）。

```python
if diff_mask is not None:
    prediction = prediction * diff_mask
    reference = reference * diff_mask
...
loss = loss.mean(dim=(1, 2, 3))   # ← マスク領域の面積に関わらず、常に全画素数(C*H*W)で割っている
```

マスクで限定した領域外は0埋めされるが、`.mean()`は**全画素数で割る**ため、マスクが狭いほど損失値が不当に小さく（薄まって）算出される。本来はマスクで有効化された画素数のみで平均すべき。

**修正方針**:

```python
def _masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """マスク適用後のtensorを、マスクで有効化された画素数のみで平均する。

    マスクがNoneの場合は通常のmean(dim=(1,2,3))と等価。
    """
    if mask is None:
        return tensor.mean(dim=(1, 2, 3))
    mask_sum = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return tensor.sum(dim=(1, 2, 3)) / mask_sum
```

`compute_addift_loss()` と `compute_dpo_loss()` 両方の `.mean(dim=(1,2,3))` 呼び出し箇所をこのヘルパー関数に置き換える。**DPOモード・非DPOモード両方に影響する修正**である点に注意（DPO側だけの修正にしないこと）。

`diff_mask`がどのように生成されているか（画像A/Bの差分から自動生成？ 固定マスク？）は本セッションで未確認のため、次セッション冒頭でマスク生成箇所のコードも確認した上で修正範囲を確定すること。

---

## 5. 参考文献（本セッションで引用した研究）

- *Diffusion Model Alignment Using Direct Preference Optimization*（`dpo.txt`に要約。Diffusion-DPOの基本式の出典）
- DPOにおける尤度置換(likelihood displacement)現象、DPO-Positive (DPOP) 正則化、DPO+NLL方式についての文献調査（Smaug論文ほか、本セッションの会話内で引用済み）

---

## 6. 次セッション開始時のアクションチェックリスト

1. 本仕様書を読み込み、現状把握
2. `addift_dpo_ui.py` の `_DEFAULT_WIN_AUX_WEIGHT` を `1.0` → `0.1` に変更
3. `monitor_graph_addift.py` の `_check_es_dpo()` をトレンド判定方式へ改修（4.2の設計論点をユーザーと確認の上で実装）
4. `anima_train_addift.py` の `diff_mask` 生成箇所を確認し、`compute_addift_loss()` / `compute_dpo_loss()` のマスク平均化ロジックを `_masked_mean()` 方式へ修正（4.3）
5. 全ファイル `py_compile` 構文チェック、`texts_ja.json` JSONパース検証
6. 修正版を `/mnt/user-data/outputs/` へ出力し、ユーザーへ提示
