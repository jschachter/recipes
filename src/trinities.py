"""Discover holy trinities — ingredient groups that define cuisines and techniques."""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.graph import normalize, OUTPUT_DIR


BORING = frozenset({
    "salt", "pepper", "water", "oil", "butter", "flour",
    "sugar", "egg", "milk", "garlic", "vegetable oil",
    "olive oil", "cooking spray", "shortening", "margarine",
})

DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"


def load_recipes(model_slug: str) -> list[dict]:
    """Load all parsed recipes with their normalized ingredients and tags."""
    model_dir = OUTPUT_DIR / model_slug
    recipes = []

    for f in sorted(model_dir.glob("*.json")):
        d = json.load(open(f))
        parsed = d.get("parsed")
        if not parsed:
            continue

        tags = {t.lower().strip() for t in (parsed.get("tags") or [])}
        ings = set()
        for step in parsed.get("steps") or []:
            for ing in step.get("ingredients") or []:
                if ing and isinstance(ing, str):
                    n = normalize(ing)
                    if n not in BORING:
                        ings.add(n)

        if 3 <= len(ings) <= 20:
            recipes.append({"ings": ings, "tags": tags})

    return recipes


def find_trinities(
    recipes: list[dict],
    size: int = 3,
    min_count: int = 5,
    cuisine: str | None = None,
    min_cuisine_ratio: float = 0.4,
    limit: int = 20,
) -> list[tuple[int, float, tuple[str, ...], Counter]]:
    """Find common ingredient groups, optionally filtered by cuisine.

    Returns list of (count, cuisine_ratio, ingredients, tag_counts).
    """
    triplet_counts = Counter()
    triplet_tags = {}

    for r in recipes:
        for combo in combinations(sorted(r["ings"]), size):
            triplet_counts[combo] += 1
            if combo not in triplet_tags:
                triplet_tags[combo] = Counter()
            for t in r["tags"]:
                triplet_tags[combo][t] += 1

    results = []
    for triplet, count in triplet_counts.most_common():
        if count < min_count:
            break
        tags = triplet_tags[triplet]
        if cuisine:
            cuisine_count = tags.get(cuisine, 0)
            ratio = cuisine_count / count
            if ratio < min_cuisine_ratio:
                continue
        else:
            ratio = 0.0
        results.append((count, ratio, triplet, tags))
        if len(results) >= limit:
            break

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.trinities <mode> [options]")
        print()
        print("Modes:")
        print("  top [size] [model]          — most common ingredient groups overall")
        print("  cuisine <name> [size] [model] — groups specific to a cuisine")
        print("  all-cuisines [size] [model]  — scan all cuisines for their trinities")
        print()
        print("Examples:")
        print("  python -m src.trinities top")
        print("  python -m src.trinities top 4")
        print("  python -m src.trinities cuisine italian")
        print("  python -m src.trinities cuisine chinese 4")
        print("  python -m src.trinities all-cuisines")
        sys.exit(1)

    mode = sys.argv[1]
    model_slug = DEFAULT_MODEL

    print(f"Loading recipes from {model_slug}...")
    recipes = load_recipes(model_slug)
    print(f"Loaded {len(recipes)} recipes\n")

    if mode == "top":
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        if len(sys.argv) > 3:
            model_slug = sys.argv[3]

        results = find_trinities(recipes, size=size, min_count=5, limit=30)
        print(f"=== TOP {size}-INGREDIENT GROUPS ===\n")
        for count, _, triplet, tags in results:
            top_tags = ", ".join(f"{t}({c})" for t, c in tags.most_common(3))
            print(f"  {count:4d}x  {' + '.join(triplet)}")
            print(f"         {top_tags}")

    elif mode == "cuisine":
        cuisine = sys.argv[2]
        size = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        if len(sys.argv) > 4:
            model_slug = sys.argv[4]

        results = find_trinities(
            recipes, size=size, min_count=5,
            cuisine=cuisine, min_cuisine_ratio=0.4, limit=15,
        )
        print(f"=== {cuisine.upper()} {size}-INGREDIENT GROUPS ===\n")
        for count, ratio, triplet, tags in results:
            print(f"  {count:4d}x ({ratio:.0%} {cuisine})  {' + '.join(triplet)}")

    elif mode == "all-cuisines":
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        if len(sys.argv) > 3:
            model_slug = sys.argv[3]

        cuisines = [
            "italian", "mexican", "chinese", "indian", "french",
            "thai", "japanese", "cajun", "mediterranean", "greek",
            "korean", "southern", "german",
        ]

        for cuisine in cuisines:
            results = find_trinities(
                recipes, size=size, min_count=3,
                cuisine=cuisine, min_cuisine_ratio=0.4, limit=5,
            )
            if results:
                print(f"  {cuisine.upper()}:")
                for count, ratio, triplet, tags in results:
                    print(f"    {count:4d}x ({ratio:.0%} {cuisine})  {' + '.join(triplet)}")
                print()


if __name__ == "__main__":
    main()
