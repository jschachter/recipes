"""Deduplicate recipes using ingredient-set fingerprinting and MinHash LSH."""

import json
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from src.graph import normalize, OUTPUT_DIR


DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"
DEFAULT_THRESHOLD = 0.8
NUM_PERM = 128


def get_ingredient_set(parsed: dict) -> set[str]:
    """Extract normalized ingredient set from a parsed recipe."""
    ings = set()
    for step in parsed.get("steps") or []:
        for ing in step.get("ingredients") or []:
            if ing and isinstance(ing, str):
                ings.add(normalize(ing))
    return ings


def make_minhash(ingredients: set[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for ing in ingredients:
        m.update(ing.encode("utf-8"))
    return m


def dedup(model_slug: str, threshold: float = DEFAULT_THRESHOLD) -> None:
    model_dir = OUTPUT_DIR / model_slug
    files = sorted(model_dir.glob("*.json"))

    # Pass 1: extract ingredient sets
    recipes = []
    empty = 0
    for f in files:
        d = json.load(open(f))
        parsed = d.get("parsed")
        if not parsed:
            empty += 1
            continue
        ings = get_ingredient_set(parsed)
        if len(ings) < 2:
            empty += 1
            continue
        recipes.append((f.stem, ings))

    print(f"Total files: {len(files)}")
    print(f"Skipped (no parse / <2 ingredients): {empty}")
    print(f"Candidates: {len(recipes)}")

    # Pass 2: exact dedup
    seen_exact = {}
    exact_dupes = 0
    unique_recipes = []
    for recipe_id, ings in recipes:
        fp = tuple(sorted(ings))
        if fp in seen_exact:
            exact_dupes += 1
            continue
        seen_exact[fp] = recipe_id
        unique_recipes.append((recipe_id, ings))

    print(f"Exact duplicates removed: {exact_dupes}")
    print(f"After exact dedup: {len(unique_recipes)}")

    # Pass 3: MinHash LSH near-dedup
    print(f"\nBuilding MinHash LSH (threshold={threshold}, num_perm={NUM_PERM})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    minhashes = {}

    for recipe_id, ings in unique_recipes:
        m = make_minhash(ings)
        minhashes[recipe_id] = m
        try:
            lsh.insert(recipe_id, m)
        except ValueError:
            # Duplicate key (shouldn't happen after exact dedup)
            pass

    # Find clusters of near-duplicates, keep one per cluster
    removed = set()
    clusters = 0
    for recipe_id, ings in unique_recipes:
        if recipe_id in removed:
            continue
        result = lsh.query(minhashes[recipe_id])
        if len(result) > 1:
            clusters += 1
            # Keep the first one (alphabetically), remove the rest
            keeper = min(result)
            for r in result:
                if r != keeper:
                    removed.add(r)

    near_dupes = len(removed)
    final = len(unique_recipes) - near_dupes
    print(f"Near-duplicate clusters: {clusters}")
    print(f"Near-duplicates removed: {near_dupes}")
    print(f"Final unique recipes: {final}")

    # Save the keep list
    keep_ids = set()
    for recipe_id, ings in unique_recipes:
        if recipe_id not in removed:
            keep_ids.add(recipe_id)

    keep_path = OUTPUT_DIR / model_slug / "_keep_list.json"
    with open(keep_path, "w") as f:
        json.dump(sorted(keep_ids), f)
    print(f"\nSaved {len(keep_ids)} recipe IDs to {keep_path}")

    # Show some example near-dupe clusters
    print(f"\nExample near-duplicate clusters:")
    shown = 0
    for recipe_id, ings in unique_recipes:
        if recipe_id in removed:
            continue
        result = lsh.query(minhashes[recipe_id])
        if len(result) > 3:
            print(f"\n  Cluster ({len(result)} recipes): {sorted(result)[:5]}...")
            print(f"  Ingredients: {sorted(ings)[:8]}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    model_slug = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    dedup(model_slug, threshold)
