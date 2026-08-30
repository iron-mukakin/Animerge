# Krea2量子化プロジェクト 研究成果 — sd-scripts(Anima)移植用引継ぎ資料

**作成日**: 2026-07-17
**対象**: 自己フォークsd-scripts(Anima階層学習実装済み、量子化導入予定)
**出典プロジェクト**: Omusubi_Krea2_quan_GUI(musubi-tunerベース、Krea2 DiT対象)

---

## 0. 総括(結論を先に)

Krea2プロジェクトでは、勾配量子化(Layer3)の実装・検証を通じて「正しく・安全に動く量子化」は達成した。しかし、**プロジェクトが本来目指したVRAM/速度削減という成果は、Krea2の検証環境(RTX 5060 Ti 16GB、Block Swap ring1・26/28ブロックがほぼ必須)では実現できなかった**。原因はLayer3(勾配)ではなく、**土台であるLayer1(重みfp8化)の実装方式**にあることをコードレベルで特定済み。

sd-scripts/Animaは、この後半の教訓が活きる可能性がある一方、**Layer1に相当する`fp8_scaled`は、コード上Krea2と全く同一の欠陥を持つ**ことを確認済み。以下、順を追って詳述する。

---

## 1. 開発の経緯と、各段階で得た知見

### 1.1 Layer3(勾配量子化)v1: FP8-LM論文ベースの移動平均的スケーリング

**設計**: フック地点(before_attention/after_attention/mlp_gate_up_input/mlp_down_input)ごとに、前stepの状態を引き継ぐスカラーμを持ち、`growth_interval`(既定1000step)かけて緩慢に2倍成長、オーバーフロー検知時は`shrink_factor`(既定0.5)で縮小。加えて、アンダーフロー即時補正ループ(`emergency_growth_factor`×最大`max_correction_iters`回)を追加実装。

**実機診断で判明した致命的事実**:
- **同一フック地点でも、`grad_abs_max`(そのstepの勾配テンソルの絶対値最大)がstepごとに4〜6桁変動する**(実測: 8.96e-9〜2.98e-4など)。
- 固定`max_scale`上限では、小さいstepでは慢性的なアンダーフロー、大きいstepに合わせて上限を上げれば別のstepでオーバーフロー、という**原理的に両立不可能なトレードオフ**に陥る。
- 即時補正ループは`emergency_growth_factor^max_correction_iters`という組み合わせでしか上限に届かず、`max_scale`だけ変更してもパラメータ間の整合性が崩れて機能しない、という設計上の落とし穴も発見(要注意ポイントとして記録)。

**結論**: FP8-LM論文の設計(NCCL All-Reduce向け、分散学習で毎stepの再計算コストを避ける前提)は、**シングルGPU・per-tensorで桁違いに変動するこのユースケースには根本的に不向き**。

### 1.2 ノイズフロアの誤診断(重要な反省点)

同一テンソル内で最大値と最小非ゼロ値の比が8〜27桁に達する事象を観測し、当初「bf16混合精度のノイズ残差を非ゼロ要素として誤カウントしている」と仮説を立てた。`significance_relative_threshold`(絶対値最大に対する相対閾値、既定1e-4)を導入して検証した結果、**この仮説は半分だけ正しかった**:
- ノイズ除外自体は正しく機能した(合成データで実証済み)。
- しかし実データでは、除外後も`underflow_ratio`は高止まりしたままだった。**真因はノイズ混入ではなく、正真正銘`grad_abs_max`自体のstep間変動幅が大きすぎることだった**(1.1節の通り)。

→ **教訓**: 「有意性の閾値化」は今も有効な設計要素(bf16ノイズの誤カウント防止に寄与)だが、**それ単体では根本原因を解決しない**。原因切り分けのために、`grad_abs_max`/`grad_abs_min_nonzero`/`significant_ratio`といった実測値を診断ログに追加したことが、最終的に真因特定の決め手になった。

### 1.3 Layer3 v2: JIT(呼び出しごとの動的)スケーリングへの全面書き換え

**設計変更**: 前stepの状態を一切引き継がず、**毎呼び出しでそのテンソル自身の`grad_abs_max`から直接`target_scale = (448 × target_max_ratio) / grad_abs_max`を算出**(`target_max_ratio`既定0.5)。上限のみ`max_scale`でクランプ(下限クランプは撤廃—これがオーバーフロー防止の要)。

**実機検証結果(1エポック+、call_index 1〜40、952〜1093行/フック)**:

| 指標 | v1(移動平均) | v2(JITスケーリング、max_scale=2^36) |
|---|---|---|
| overflow_ratio | 0%(未対策時は別問題あり) | **全期間0.0%** |
| underflow_ratio | 79.9〜99.6%(高止まり) | **全期間0.0%** |
| cosine_similarity | 0.0(壊滅的ケースあり)〜0.94 | **0.9996で安定** |
| relative_error | 最大1.0(完全ゼロ化) | **約2.6〜2.9%(FP8のビット幅限界による残差のみ)** |
| scale_muの挙動 | 上限に張り付いたまま固定 | **呼び出しごとに数桁単位で自動追従** |

**この設計変更は完全に成功したと判断してよい水準**。追加で、警告ログを原因別に分離(「(a)max_scale上限不足」/「(b)テンソル自体のダイナミックレンジがFP8実用レンジ超過」)する仕組みを実装し、実データでは**全ての警告が(a)のみ**であることを確認 → 単一スカラーで原理的に救えない状況(b)は、少なくともこのモデル・データセットでは発生しなかった。

### 1.4 診断ログのコスト(重要な運用上の教訓)

診断ログ(`cosine_similarity`/`relative_error`/`significant_ratio`等)を有効化すると、各フック呼び出しごとに複数の`.item()`(GPU→CPU同期)が発生する。28ブロック×4フック=112箇所/stepでこれが積み重なり、**Block Swap(H2Dストリーミング)との相互作用で平均step時間が約2.3倍に悪化**することを実測で確認(51秒/it → 診断OFFで21秒/it)。

→ **教訓**: 診断ログは原因切り分けのための短時間検証専用にとどめ、**本番学習では必ずOFFにする運用ルールを徹底すること**。

---

## 2. 決定的な発見: Layer1(重みfp8化)の実装上の欠陥

`fp8_optimization_utils.py`の`fp8_linear_forward_patch()`(`use_scaled_mm=False`、実際に使われているデフォルト分岐)を精査した結果:

```python
# 毎forward呼び出しごとに実行される処理
dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
output = F.linear(x, dequantized_weight, self.bias)
```

**Linear層のforwardのたびに、fp8(1byte/要素)の重みをbf16(2byte/要素)へ全要素逆量子化した一時テンソルを新規生成している。**

### これが引き起こす2つの実害(実測と整合)

1. **VRAMピークが下がらない**: アイドル時の保管サイズはfp8で半分だが、計算時は毎回フルサイズのbf16重みを再構築するため、ピークメモリは量子化前と実質同じ。
2. **速度がむしろ悪化する**: 逆量子化(掛け算)という**純粋な追加コスト**が、全Linear層・全forward呼び出しに乗る。見返りとなる高速化要素が無い。

### 本来の解決策は同ファイル内に存在するが、未検証・不完全

`use_scaled_mm=True`分岐は`torch._scaled_mm`(fp8ネイティブ行列積)を使い、フルサイズの一時テンソルを作らない。しかし:

- コード内に`# **not tested**`と明記
- `scale_weight.ndim != 1`(per-tensorスケールでない)場合は即エラー。現状の量子化(`quantization_mode="block"`、block_size=64)はper-channel/per-block(3次元)のscale_weightを生成するため、**そのままでは使えない**
- 入力活性化(x)側は`max_value=None`固定でキャリブレーション無し(`scale_x=1.0`固定)。これはLayer3冒頭と同じ「アンダーフロー対策無し」の状態であり、**JITスケーリングと同種の動的スケール算出ロジックを、活性化入力用に新規実装する必要がある**

### Krea2側での結論

VRAM/速度改善を実現するには、per-tensorスケールへの量子化再設計(精度悪化を伴う)+活性化側の動的スケール新規実装+`scaled_mm`経路のハードウェア実証、という**Layer3と同等以上の新規開発**が必要と判断し、Krea2プロジェクトはここで打ち切りとした。

---

## 3. Block Swapとの相互作用に関する知見

`custom_offloading_utils.py`(`LoRAStreamOffloader`)は重みのdtypeに関して**完全に無頓着(dtype-agnostic)**であり、`element_size()`ベースでバイトサイズを計算してCPU/GPUバッファを確保している。**fp8化された重みは正しく半分のサイズで転送・保持される設計になっており、この点に関しては改修不要**。

ただし、**Block Swapを積極的に使う(Krea2は28ブロック中26個をCPUへ退避)場合、GPU常駐分がそもそも極小(2/28)になるため、Layer1の量子化効果(常駐分にしか効かない)が全体に対して誤差レベルまで薄まる**。これが「診断ログでは量子化は正常動作しているのに、体感でVRAM削減が感じられない」現象の主因と判断している。

**→ sd-scripts/Animaへの示唆**: Block Swapを使わない(モデル全体が常時GPU常駐する)構成であれば、この「Block Swapによるマスキング」問題は起きない。Layer1の常駐重み削減効果が、全体のVRAM使用量に対してより意味のある割合を占められる可能性が高い。

---

## 4. sd-scripts(Anima)側の現状確認結果

実際に提供いただいた`fp8_optimization_utils.py`(Krea2と共通)・`anima_utils.py`・`anima_train_utils.py`・`anima_train_network.py`を確認した結果:

| 確認項目 | 結果 |
|---|---|
| Animaのfp8実装 | `anima_utils.py`の`load_anima_model()`が`apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)`を呼び出しており、**Krea2と全く同一の「毎forward全要素逆量子化」実装**。VRAM/速度面の欠陥もそのまま該当すると判断してよい(推測ではなくコード同一性による確定事項) |
| Layer3(勾配量子化)相当の仕組み | Anima側には存在しない。JITスケーリングの設計思想は転用できるが、コードは新規実装が必要 |
| Block Swap | `anima_train_network.py`で`args.blocks_to_swap is not None and args.blocks_to_swap > 0`の場合のみ有効化。暗黙の自動有効化は無いことをコード・実ログ両方で確認済み |
| 直近の実運用ログ | Block Swap未使用(`--blocks_to_swap`未指定)、resolution 512x512、network_dim=16、モデル全体がbf16のままGPU常駐。**「Block Swapによるマスキングが起きない」有利な条件に該当** |
| 階層学習(`anima_matrix_scales`)との関係 | ログで確認したcustom_attributes形式のper-block LR/scale制御(`Output_Attention:0.0`等で一部ブロック凍結)が既に稼働中。量子化(特にLayer4 Optimizer相当)を導入する場合、**凍結ブロックと学習対象ブロックで扱いを分ける必要があるか要確認**(次節参照) |

---

## 5. 移植を進める前に、事前確認したいファイル・情報

以下が無いと、移植方針を誤る、または車輪の再発明を繰り返すリスクがあるため、着手前に確認をお願いしたい。

### 5.1 `library/lora_utils.py`(未確認・最重要)

`load_safetensors_with_lora_and_fp8()`の実装。Anima読み込み時に実際どの`quantization_mode`/`block_size`が使われているか(Krea2と同じ`"block"`/64か、それとも別設定か)を確認したい。これによって、`use_scaled_mm=True`化の際に必要な「per-tensor化による精度悪化」の度合いが変わる。

### 5.2 `library/anima_models.py`(未確認・重要)

Animaの`Attention`/`MLP`(またはそれに相当するモジュール)の実装。`fp8_optimization_utils.py`は`isinstance(module, nn.Linear)`のみを対象にパッチするため、**Animaのブロックが素のnn.Linearベースか、Krea2のような独自Attention/SwiGLU実装(register_hookでの勾配フック地点特定が必要)か**を確認したい。もしLayer3相当(勾配量子化)を将来的に導入するなら、フック地点の設計はここに依存する。

### 5.3 既に実装済みの「階層学習」システムの詳細仕様(未確認)

`anima_matrix_scales`によるper-block LR/scale制御が、量子化対象の選定(例: 凍結ブロックは量子化してもLoRA学習には影響しないので、Layer4/勾配量子化の対象から自動除外すべきか等)とどう関係するか、設計意図を伺いたい。特に、量子化診断ログ(このプロジェクトで作った`diagnostics.py`相当の仕組み)を導入する場合、フック地点の命名規則がAnimaの階層構造(部位別の`Input_/Middle_/Output_`区分)と整合するように設計したい。

### 5.4 optimizer周りの実装方針(未確認)

Krea2ではLayer4として`COATAdamW`(独自Optimizer)を使っていたが、Anima側の学習コード(`anima_train_network.py`)は`AdamW`をそのまま使っている(ログのCMDで確認)。Optimizer State量子化を導入する予定があるなら、独自Optimizerクラスを新規実装するか、既存の`bitsandbytes`等の量子化Optimizerを流用するかの方針を伺いたい。

---

## 6. 転用可能な設計資産(コードではなく「考え方」として持ち込めるもの)

- **JITスケーリングの設計思想**(呼び出しごとに実データからscaleを直接算出し、固定上限による板挟みを避ける)
- **診断ログ基盤の設計原則**(単一責任: ログ基盤は「渡されたら書くだけ」の薄いI/O層に徹し、有効/無効判定と何を記録するかは呼び出し側が持つ)
- **原因切り分けの手法**: overflow/underflowの分離計測、有意性相対閾値によるノイズ除外、警告メッセージの原因別分岐((a)上限不足 vs (b)原理的に救えない)
- **`apply_fix*.py`形式のパッチ運用**(Python純正、CRLF/LF吸収、適用前バックアップ)は、sd-scriptsフォークの改修作業にもそのまま流用可能

これらは「コードの移植」ではなく「これまでの試行錯誤から得た設計原則」として、Anima側での新規実装時に手戻りを減らす目的で活用できる。

---

## 7. 現時点での率直な評価

- Krea2プロジェクトの技術的成果(JITスケーリング、診断手法)は無駄ではないが、**そのままでは「動くコード」としての移植価値は低い**(Anima側のブロック構造・フック地点・Optimizer実装が異なるため、いずれも新規実装が必要)。
- **Layer1(重みfp8化)の「毎forward逆量子化」という根本的な設計欠陥は、Krea2・Anima両方で同一である**ことをコードで確認済み。この欠陥を放置したまま勾配量子化(Layer3相当)だけ導入しても、Krea2で起きたのと同じ「正しく動くが実利が薄い」結果になるリスクが高い。
- Anima側はBlock Swap不使用という点でKrea2より有利な条件だが、それでも**Layer1の欠陥修正(per-tensor化+活性化側キャリブレーション+`scaled_mm`実証)を先に済ませない限り、勾配量子化の効果を正しく評価できない**と考える。

上記5節のファイル・仕様を確認でき次第、具体的な実装ステップ(どのファイルにどの順で手を入れるか)を詰めていきたい。
