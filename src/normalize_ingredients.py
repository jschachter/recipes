"""Extract all unique ingredients from parsed recipes and normalize via LLM."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from src.env import load_dotenv
from src.graph import OUTPUT_DIR

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NORMALIZE_MODEL = "google/gemini-2.5-flash-lite"
BATCH_SIZE = 200
DEFAULT_CONCURRENCY = 20
CACHE_PATH = Path("data_nosync/ingredient_normalization.json")

SYSTEM_PROMPT = """\
You are an ingredient normalizer. Given a list of ingredient names, map each to its simplest canonical form.

Rules:
- Strip preparation methods: "diced onion" → "onion", "melted butter" → "butter"
- Strip modifiers about state: "frozen peas" → "peas", "cooked rice" → "rice"
- Merge brand names into generic: "Miracle Whip" → "mayonnaise", "Velveeta" → "processed cheese"
- Merge size/cut variants: "chopped walnuts" → "walnuts", "sliced mushrooms" → "mushrooms"
- Merge plurals: "tomatoes" → "tomato", "eggs" → "egg"
- Keep meaningful distinctions: "green onion" ≠ "onion", "cream cheese" ≠ "cheese", "brown sugar" ≠ "sugar"
- Keep protein cuts distinct but normalize: "beef chuck roast" → "beef roast", "boneless skinless chicken breasts" → "chicken breast"
- For compound ingredients like "cream of mushroom soup", keep as-is
- For obscure/rare ingredients, keep the simplest recognizable name

Output a JSON object mapping each input name to its canonical form.
Example input: ["frozen green peas", "english peas", "cooked peas", "sweet peas"]
Example output: {"frozen green peas": "peas", "english peas": "peas", "cooked peas": "peas", "sweet peas": "peas"}

Output only valid JSON, no markdown or commentary.\
"""


def extract_ingredients(model_slug: str) -> list[str]:
    """Get all unique ingredient names from parsed recipe outputs."""
    model_dir = OUTPUT_DIR / model_slug
    ingredients = set()

    for f in sorted(model_dir.glob("*.json")):
        d = json.load(open(f))
        parsed = d.get("parsed")
        if not parsed:
            continue
        for step in parsed.get("steps") or []:
            for ing in step.get("ingredients") or []:
                if ing and isinstance(ing, str):
                    ingredients.add(ing.lower().strip())

    return sorted(ingredients)


async def normalize_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    batch: list[str],
    batch_idx: int,
    total_batches: int,
    api_key: str,
) -> dict[str, str]:
    async with semaphore:
        payload = {
            "model": NORMALIZE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(batch)},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            response = resp.json()
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            mapping = json.loads(content)
            elapsed = round(time.monotonic() - start, 1)
            print(f"  [{batch_idx+1}/{total_batches}] {len(mapping)} ingredients normalized ({elapsed}s)")
            return mapping
        except Exception as e:
            elapsed = round(time.monotonic() - start, 1)
            print(f"  [{batch_idx+1}/{total_batches}] error: {e} ({elapsed}s)")
            return {}


async def normalize_all(
    ingredients: list[str],
    api_key: str,
    existing: dict[str, str],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, str]:
    # Filter out already-normalized ingredients
    todo = [ing for ing in ingredients if ing not in existing]
    if not todo:
        print("All ingredients already normalized.")
        return existing

    batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    total = len(batches)
    print(f"\n=== Normalizing {len(todo)} ingredients in {total} batches, concurrency={concurrency} ===")

    semaphore = asyncio.Semaphore(concurrency)
    result = dict(existing)

    async with httpx.AsyncClient() as client:
        tasks = [
            normalize_batch(client, semaphore, batch, i, total, api_key)
            for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*tasks)

    for mapping in batch_results:
        result.update(mapping)

    return result


def main():
    model_slug = sys.argv[1] if len(sys.argv) > 1 else "google--gemini-2.5-flash-lite_v11-tagged"
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CONCURRENCY

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Load existing cache
    existing = {}
    if CACHE_PATH.exists():
        existing = json.loads(CACHE_PATH.read_text())
        print(f"Loaded {len(existing)} existing normalizations from cache")

    # Extract ingredients
    print(f"Extracting ingredients from {model_slug}...")
    ingredients = extract_ingredients(model_slug)
    print(f"Found {len(ingredients)} unique ingredients ({len(ingredients) - len(existing)} new)")

    # Normalize
    result = asyncio.run(normalize_all(ingredients, api_key, existing, concurrency))

    # Save
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSaved {len(result)} normalizations to {CACHE_PATH}")

    # Stats
    canonical = set(result.values())
    print(f"Reduced {len(result)} surface forms → {len(canonical)} canonical ingredients")


if __name__ == "__main__":
    main()
