"""app/anima_block_manifests.py

Anima DiTモデルのblock構成拡張(LLaMA-Pro方式のblock insertion)に関する
確定済みマッピングデータと、それを扱うための純粋関数群。

出典・検証状況:
    - 28->40 (Anima Base v1.0 -> Anima 2.9B):
        expand_manifest_28_40.json(ユーザー提供の一次資料)をそのまま採用。
        provenance注記なし(公式ファイルとして扱う)。
    - 40->52 (Anima 2.9B -> Anima 3.8B):
        expand_manifest_40_52.json(ユーザー提供の一次資料)をそのまま採用。
        **このファイル自体が "RECONSTRUCTED, not an official file" と明記された
        非公式データである**(anima29B_v10.safetensors と
        Anima-3.8-preview-0.1.safetensors(lylogummy/Anima-3.8B, "Pro52")を
        block単位でcosine類似度比較して逆算したもの、cross-validated)。
        公式のexpand_manifest_40_52.jsonが別途公開された場合は、本モジュールの
        _MANIFEST_40_52定数(および、それに依存する_MANIFEST_28_52の合成結果)を
        差し替えること。IS_40_52_OFFICIAL = False で非公式である旨を機械的にも
        判定可能にしている。
    - 28->52 (Anima Base v1.0 -> Anima 3.8B):
        上記2つをcompose_manifests.py(ユーザー提供の一次資料)と同一アルゴリズムで
        本モジュール内で独立に合成した結果。ユーザー提供の
        expand_manifest_28_52_composed.jsonと完全一致することを検証済み
        (insertion_positions・inserted_to_source双方が一致)。

このモジュールは純粋にデータ定義と参照関数のみを提供し、実際のテンソル操作は
一切行わない(merge.py側がBlockMapping/実マージロジックを担当する)。
"""
from __future__ import annotations

from typing import Optional

# 40->52 マッピングが非公式(cosine類似度による再構成)であることを機械的にも
# 判定できるようにするフラグ。GUI/ログでの注意喚起に使う。
IS_40_52_OFFICIAL = False

_MANIFEST_28_40: dict = {
    "old_block_count": 28,
    "new_block_count": 40,
    "insertion_positions": (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36),
    "inserted_to_source": {
        2: 1, 5: 3, 8: 5, 11: 7, 14: 9, 17: 11,
        21: 14, 24: 16, 27: 18, 30: 20, 33: 22, 36: 24,
    },
    "official": True,
}

_MANIFEST_40_52: dict = {
    "old_block_count": 40,
    "new_block_count": 52,
    "insertion_positions": (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47),
    "inserted_to_source": {
        3: 2, 7: 5, 11: 8, 15: 11, 19: 14, 23: 17,
        27: 20, 31: 23, 35: 26, 39: 29, 43: 32, 47: 35,
    },
    "official": IS_40_52_OFFICIAL,
}


def _build_base_to_target(manifest: dict) -> dict[int, int]:
    """manifestから{base_idx: target_idx}を導出する(継承blockの正しい対応)。

    compose_manifests.py/anima_common.pyのbuild_base_to_target()と同一ロジック:
    new_block_count個のtarget indexのうちinsertion_positionsに含まれない
    (=旧モデルから連番のまま引き継がれた)indexを昇順に列挙し、base index
    0,1,2...と順に対応付ける。「新規挿入blockが初期化時にどのbaseブロックを
    コピーしたか」を記録するinserted_to_sourceとは別物であり、標準マージの
    block対応にはこちらを使う。

    Args:
        manifest: old_block_count/new_block_count/insertion_positionsを含む辞書。

    Returns:
        {base_idx: target_idx} の辞書。

    Raises:
        ValueError: insertion_positionsの個数がold_block_count/new_block_countと
            整合しない場合。
    """
    old_count = manifest["old_block_count"]
    new_count = manifest["new_block_count"]
    inserted = set(manifest["insertion_positions"])
    old_target_indices = [i for i in range(new_count) if i not in inserted]
    if len(old_target_indices) != old_count:
        raise ValueError(
            f"manifest inconsistency: old_block_count={old_count} but found "
            f"{len(old_target_indices)} non-inserted target slots in a {new_count}-block layout"
        )
    return {base_idx: target_idx for base_idx, target_idx in enumerate(old_target_indices)}


def _resolve_ultimate_source(b_idx: int, manifest_ab: dict, base_to_target_ab: dict[int, int]) -> int:
    """B空間のindexが最終的にどのA(base)空間indexに由来するかを解決する。

    b_idxがA->B自身の挿入blockなら、その挿入blockがA->B内で初期化時に
    コピーされた元(inserted_to_source)を直接返す。そうでなければ
    base_to_target_abを逆引きしてa_idxを求める。
    """
    insertion_positions_ab = set(manifest_ab["insertion_positions"])
    if b_idx in insertion_positions_ab:
        return manifest_ab["inserted_to_source"][b_idx]
    inverse = {v: k for k, v in base_to_target_ab.items()}
    if b_idx not in inverse:
        raise ValueError(f"b_idx {b_idx} is neither an A->B insertion nor a mapped base block")
    return inverse[b_idx]


def _compose(manifest_ab: dict, manifest_bc: dict) -> dict:
    """A->BとB->Cを合成してA->Cを導出する。compose_manifests.pyと同一アルゴリズム。

    Args:
        manifest_ab: 前段の拡張マニフェスト。
        manifest_bc: 後段の拡張マニフェスト
            (manifest_ab["new_block_count"] == manifest_bc["old_block_count"]が必要)。

    Returns:
        old_block_count/new_block_count/insertion_positions/inserted_to_sourceを
        含む合成後マニフェスト。

    Raises:
        ValueError: 中間層のblock数が一致しない、または合成後に対応未解決の
            挿入blockが残る場合。
    """
    if manifest_ab["new_block_count"] != manifest_bc["old_block_count"]:
        raise ValueError(
            f"cannot compose: A->B ends at {manifest_ab['new_block_count']} blocks, "
            f"but B->C starts at {manifest_bc['old_block_count']} blocks"
        )

    old_count_a = manifest_ab["old_block_count"]
    new_count_c = manifest_bc["new_block_count"]
    base_to_target_ab = _build_base_to_target(manifest_ab)
    base_to_target_bc = _build_base_to_target(manifest_bc)

    base_to_target_ac = {a_idx: base_to_target_bc[b_idx] for a_idx, b_idx in base_to_target_ab.items()}
    covered_c_indices = set(base_to_target_ac.values())
    insertion_positions_ac = sorted(set(range(new_count_c)) - covered_c_indices)

    inserted_to_source_ac: dict[int, int] = {}
    for c_idx, b_source in manifest_bc.get("inserted_to_source", {}).items():
        inserted_to_source_ac[c_idx] = _resolve_ultimate_source(b_source, manifest_ab, base_to_target_ab)
    for b_idx, a_source in manifest_ab.get("inserted_to_source", {}).items():
        c_idx = base_to_target_bc[b_idx]
        inserted_to_source_ac[c_idx] = a_source

    missing = set(insertion_positions_ac) - set(inserted_to_source_ac)
    if missing:
        raise ValueError(f"composition left unresolved insertions: {sorted(missing)}")

    return {
        "old_block_count": old_count_a,
        "new_block_count": new_count_c,
        "insertion_positions": tuple(insertion_positions_ac),
        "inserted_to_source": inserted_to_source_ac,
        "official": manifest_ab.get("official", False) and manifest_bc.get("official", False),
    }


_MANIFEST_28_52: dict = _compose(_MANIFEST_28_40, _MANIFEST_40_52)

# (old_block_count, new_block_count) -> manifest。将来別の世代(例: 52->64)が
# 追加された場合も、隣接ペアをここに足して_compose()で合成し追加すればよい
# (compose_manifests.pyのauto処理と同じ設計方針)。
KNOWN_EXPANSION_MANIFESTS: dict[tuple[int, int], dict] = {
    (28, 40): _MANIFEST_28_40,
    (40, 52): _MANIFEST_40_52,
    (28, 52): _MANIFEST_28_52,
}


def is_expansion_pair_known(old_block_count: int, new_block_count: int) -> bool:
    """old_block_count -> new_block_countの拡張マニフェストが既知か判定する。"""
    return (old_block_count, new_block_count) in KNOWN_EXPANSION_MANIFESTS


def is_expansion_pair_official(old_block_count: int, new_block_count: int) -> Optional[bool]:
    """既知の拡張ペアが公式データ由来か(Falseなら非公式の再構成データ)を返す。

    未知のペアの場合はNoneを返す。
    """
    manifest = KNOWN_EXPANSION_MANIFESTS.get((old_block_count, new_block_count))
    if manifest is None:
        return None
    return bool(manifest.get("official", False))


def block_correspondence_map(old_block_count: int, new_block_count: int) -> Optional[dict[int, int]]:
    """old_block_count -> new_block_countの継承block対応表を返す({base_idx: target_idx})。

    未知の組み合わせ(検証済みmanifestが無い)場合はNoneを返す。呼び出し側は
    Noneの場合、推測でマージせず処理を停止すること。

    Args:
        old_block_count: 変換元(より少ないblock数)のブロック総数。
        new_block_count: 変換先(より多いblock数)のブロック総数。

    Returns:
        {base_idx: target_idx}の辞書、または未知の組み合わせならNone。
    """
    manifest = KNOWN_EXPANSION_MANIFESTS.get((old_block_count, new_block_count))
    if manifest is None:
        return None
    return _build_base_to_target(manifest)


def inserted_block_positions(old_block_count: int, new_block_count: int) -> Optional[frozenset[int]]:
    """new_block_count側で旧モデルに対応物が無い(新規挿入された)block indexの集合を返す。

    未知の組み合わせの場合はNoneを返す。
    """
    manifest = KNOWN_EXPANSION_MANIFESTS.get((old_block_count, new_block_count))
    if manifest is None:
        return None
    return frozenset(manifest["insertion_positions"])


def inserted_block_source_map(old_block_count: int, new_block_count: int) -> Optional[dict[int, int]]:
    """挿入blockが初期化時にどのbaseブロックからコピーされたか({target_idx: base_idx})を返す。

    **標準マージのblock対応には使わないこと**(block_correspondence_map()を使う)。
    これは初期化時の系譜(inserted_to_source)をそのまま公開するアクセサであり、
    用途はComfyUI版の実験的extend_ratio機能、および本アプリではGUIスケール
    プリセット変換(新規挿入blockのスケール初期値を、挿入直前の既存blockの
    値から引き継ぐ近似)に限定される。

    Args:
        old_block_count: 変換元(より少ないblock数)のブロック総数。
        new_block_count: 変換先(より多いblock数)のブロック総数。

    Returns:
        {target_idx(挿入block): base_idx(コピー元)}の辞書、または未知の組み合わせなら
        None。
    """
    manifest = KNOWN_EXPANSION_MANIFESTS.get((old_block_count, new_block_count))
    if manifest is None:
        return None
    return dict(manifest["inserted_to_source"])
