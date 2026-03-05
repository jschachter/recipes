"""Load recipe datasets into a common raw format."""

import csv
import json
import sys
from pathlib import Path

RAW_DIR = Path("data_nosync/raw")
OUTPUT_DIR = Path("data_nosync/ingested")


def load_recipenlg(path: Path) -> list[dict]:
    """Load RecipeNLG CSV. Expected columns: title, ingredients, directions, link, source."""
    recipes = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            recipes.append({
                "id": f"recipenlg_{i}",
                "title": row.get("title", "").strip(),
                "ingredients_raw": row.get("ingredients", ""),
                "directions_raw": row.get("directions", ""),
                "source": row.get("source", ""),
                "url": row.get("link", ""),
                "dataset": "recipenlg",
            })
    return recipes


def load_epicurious(path: Path) -> list[dict]:
    """Load Epicurious/Food Ingredients CSV. Expected columns: Title, Ingredients, Instructions."""
    recipes = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            recipes.append({
                "id": f"epicurious_{i}",
                "title": row.get("Title", "").strip(),
                "ingredients_raw": row.get("Ingredients", ""),
                "directions_raw": row.get("Instructions", ""),
                "source": "epicurious",
                "url": "",
                "dataset": "epicurious",
            })
    return recipes


LOADERS = {
    "recipenlg": load_recipenlg,
    "epicurious": load_epicurious,
}


def ingest(dataset_name: str, path: Path, limit: int | None = None) -> Path:
    """Load a dataset and write to JSONL."""
    loader = LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(LOADERS.keys())}")

    recipes = loader(path)
    if limit is not None:
        recipes = recipes[:limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for recipe in recipes:
            f.write(json.dumps(recipe) + "\n")

    print(f"Wrote {len(recipes)} recipes to {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python -m src.ingest <dataset_name> <path> [limit]")
        print(f"Available datasets: {list(LOADERS.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ingest(name, path, limit)
