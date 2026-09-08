#!/usr/bin/env python3
"""
compose_manifests.py

Composes adjacent-tier Anima expand_manifest.json files (e.g. 28->40 and
40->52) into every pairwise manifest needed for auto-selection (e.g. also
produces 28->52), so the runtime node code never has to do multi-hop
chaining itself -- it just picks the one fully-resolved manifest matching
the detected (LoRA blocks, model blocks) pair.

Why this exists instead of hand-deriving composed mappings: composing two
expansions isn't just "look up A's target through B's mapping". Some of
a later expansion's newly-inserted blocks are copied from a block that
was ITSELF a newly-inserted block in the earlier expansion (an "insertion
of an insertion"), which has no direct ancestor in the earliest generation
-- resolving that requires walking back through the first manifest's own
insertion records. Doing this by hand is exactly the kind of thing that's
easy to get subtly wrong; this script does it mechanically and verifiably.

Design note for future tiers: composition is implemented as a single
PAIRWISE operation (compose(A->B, B->C) -> A->C) whose output is schema-
identical to a hand-authored manifest. That means when a future tier
appears (e.g. 52->64), you do NOT need to touch this script's logic --
just add the new adjacent manifest and re-run `auto`, which composes it
against everything already known (composing the new link against every
existing composed manifest that ends at its "old" tier) to produce all
newly-possible pairs, however many hops deep.

Usage:
    # Compose exactly two manifests into a new one:
    python compose_manifests.py compose 28_40.json 40_52.json -o 28_52.json

    # Scan a folder of manifests (any mix of adjacent + already-composed),
    # and generate every missing pairwise combination automatically:
    python compose_manifests.py auto mapping/
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_to_target(manifest):
    """{base_idx: target_idx} -- same logic as anima_common.py, kept in
    sync on purpose so a composed manifest behaves identically to a
    hand-authored one at runtime."""
    old_count = manifest["old_block_count"]
    new_count = manifest["new_block_count"]
    inserted = set(manifest["insertion_positions"])
    old_target_indices = [i for i in range(new_count) if i not in inserted]
    if len(old_target_indices) != old_count:
        raise ValueError(
            f"manifest inconsistency: old_block_count={old_count} but found "
            f"{len(old_target_indices)} non-inserted target slots in a "
            f"{new_count}-block layout"
        )
    return {base_idx: target_idx for base_idx, target_idx in enumerate(old_target_indices)}


def resolve_ultimate_source(b_idx, manifest_ab, base_to_target_ab):
    """
    Given a B-space index, find the REAL base(A)-space index it ultimately
    traces back to.
      - If b_idx is one of A->B's own inserted blocks, it has no direct
        A-ancestor -- recurse into what THAT block was itself copied from
        (which manifest_ab.inserted_to_source already records directly,
        since a manifest's own insertions are always sourced from real
        base indices by construction).
      - Otherwise, b_idx is a genuine carried-over A-block -- invert
        base_to_target_ab to find which a_idx produced it.
    """
    insertion_positions_ab = set(manifest_ab["insertion_positions"])
    if b_idx in insertion_positions_ab:
        return manifest_ab["inserted_to_source"][str(b_idx)]
    # invert base_to_target_ab (small dict, fine to build ad hoc)
    inverse = {v: k for k, v in base_to_target_ab.items()}
    if b_idx not in inverse:
        raise ValueError(f"b_idx {b_idx} is neither an A->B insertion nor a mapped base block -- manifest inconsistency")
    return inverse[b_idx]


def compose(manifest_ab, manifest_bc):
    """
    Compose A->B and B->C into a single A->C manifest. Requires
    manifest_ab["new_block_count"] == manifest_bc["old_block_count"]
    (both describe the same middle tier B).
    """
    if manifest_ab["new_block_count"] != manifest_bc["old_block_count"]:
        raise ValueError(
            f"Cannot compose: A->B ends at {manifest_ab['new_block_count']} blocks, "
            f"but B->C starts at {manifest_bc['old_block_count']} blocks. "
            f"These two manifests don't share a middle tier."
        )

    old_count_a = manifest_ab["old_block_count"]
    new_count_c = manifest_bc["new_block_count"]

    base_to_target_ab = build_base_to_target(manifest_ab)
    base_to_target_bc = build_base_to_target(manifest_bc)

    # Step 1: straightforward composition for every genuine A-block.
    base_to_target_ac = {
        a_idx: base_to_target_bc[b_idx]
        for a_idx, b_idx in base_to_target_ab.items()
    }

    covered_c_indices = set(base_to_target_ac.values())
    insertion_positions_ac = sorted(set(range(new_count_c)) - covered_c_indices)

    inserted_to_source_ac = {}

    # Type 1: insertions introduced fresh during the B->C step. Their
    # recorded source might itself be a B-block with no real A-ancestor
    # (an "insertion of an insertion"), which resolve_ultimate_source
    # walks back through automatically.
    for c_idx_str, b_source in manifest_bc.get("inserted_to_source", {}).items():
        c_idx = int(c_idx_str)
        a_source = resolve_ultimate_source(b_source, manifest_ab, base_to_target_ab)
        inserted_to_source_ac[str(c_idx)] = a_source

    # Type 2: A->B's own insertions, carried forward unchanged in identity
    # through the B->C step (they land at base_to_target_bc[b_idx], a
    # position B->C considers "genuine" -- it didn't insert anything new
    # there, it's just where that already-inserted B-block ended up).
    for b_idx_str, a_source in manifest_ab.get("inserted_to_source", {}).items():
        b_idx = int(b_idx_str)
        c_idx = base_to_target_bc[b_idx]
        inserted_to_source_ac[str(c_idx)] = a_source

    # Sanity check: every insertion position should now have a source.
    missing = set(insertion_positions_ac) - {int(k) for k in inserted_to_source_ac}
    if missing:
        raise ValueError(f"Composition left {len(missing)} insertion(s) with no resolved source: {sorted(missing)}")

    result = {
        "old_block_count": old_count_a,
        "new_block_count": new_count_c,
        "insertion_positions": insertion_positions_ac,
        "inserted_to_source": {k: inserted_to_source_ac[k] for k in sorted(inserted_to_source_ac, key=int)},
        "_composed_from": [
            manifest_ab.get("_composed_from", f"{manifest_ab['old_block_count']}->{manifest_ab['new_block_count']}"),
            manifest_bc.get("_composed_from", f"{manifest_bc['old_block_count']}->{manifest_bc['new_block_count']}"),
        ],
    }

    # Propagate any provenance/warning notes from either input so a composed
    # manifest never silently loses the fact that part of its chain was
    # reconstructed rather than officially published.
    provenance_notes = []
    for tag, m in (("A->B", manifest_ab), ("B->C", manifest_bc)):
        if m.get("provenance"):
            provenance_notes.append(f"[{tag}, {m['old_block_count']}->{m['new_block_count']}] {m['provenance']}")
    if provenance_notes:
        result["provenance"] = " | ".join(provenance_notes)

    return result


def cmd_compose(args):
    manifest_ab = load(args.manifest_ab)
    manifest_bc = load(args.manifest_bc)
    result = compose(manifest_ab, manifest_bc)
    out_path = args.output or f"expand_manifest_{result['old_block_count']}_{result['new_block_count']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {result['old_block_count']}->{result['new_block_count']} manifest to {out_path}")
    print(f"  ({len(result['insertion_positions'])} inserted blocks, "
          f"composed from: {result['_composed_from']})")


def cmd_auto(args):
    folder = args.folder
    paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    manifests = {}  # (old, new) -> manifest dict
    for p in paths:
        try:
            m = load(p)
            manifests[(m["old_block_count"], m["new_block_count"])] = m
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not manifests:
        print(f"No valid manifest JSON files found in {folder}")
        sys.exit(1)

    print(f"Found {len(manifests)} manifest(s): "
          + ", ".join(f"{a}->{b}" for a, b in sorted(manifests)))

    changed = True
    made_any = False
    while changed:
        changed = False
        pairs = list(manifests.keys())
        for (a1, b1), (a2, b2) in combinations(pairs, 2):
            # Try to chain (a1->b1) with (a2->b2) if they share a middle tier
            candidates = []
            if b1 == a2:
                candidates.append(((a1, b1), (a2, b2), a1, b2))
            if b2 == a1:
                candidates.append(((a2, b2), (a1, b1), a2, b1))

            for (key_first, key_second, final_old, final_new) in candidates:
                if (final_old, final_new) in manifests:
                    continue  # already have it (either hand-authored or composed earlier)
                try:
                    composed = compose(manifests[key_first], manifests[key_second])
                except ValueError as e:
                    print(f"  Skipping {key_first}+{key_second}: {e}")
                    continue
                manifests[(final_old, final_new)] = composed
                out_path = os.path.join(folder, f"expand_manifest_{final_old}_{final_new}_composed.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(composed, f, indent=2)
                print(f"Composed {final_old}->{final_new} from {key_first} + {key_second} -> {out_path}")
                changed = True
                made_any = True

    if not made_any:
        print("Nothing new to compose -- every reachable pair already exists.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compose = sub.add_parser("compose", help="Compose exactly two adjacent manifests into one")
    p_compose.add_argument("manifest_ab", help="Manifest for the earlier expansion (A->B)")
    p_compose.add_argument("manifest_bc", help="Manifest for the later expansion (B->C)")
    p_compose.add_argument("-o", "--output", default=None, help="Output path (default: auto-named)")
    p_compose.set_defaults(func=cmd_compose)

    p_auto = sub.add_parser("auto", help="Scan a folder and generate every missing pairwise manifest")
    p_auto.add_argument("folder", help="Folder containing manifest JSON files (e.g. mapping/)")
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
