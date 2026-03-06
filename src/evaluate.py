"""Spot-check structured recipe outputs using a high-end model."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from src.env import load_dotenv
from src.transform import extract_json

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_DIR = Path("data_nosync/outputs")
EVAL_DIR = Path("data_nosync/evaluations")

EVAL_MODEL = "anthropic/claude-opus-4"
DEFAULT_CONCURRENCY = 5

EVAL_PROMPT = """\
You are evaluating the quality of a structured recipe extraction. You will receive:
1. The original recipe text
2. A JSON structured extraction of that recipe

Score the extraction on two dimensions (0.0 to 1.0):
- **completeness**: Does the structured output capture all information from the original? Missing ingredients, steps, times, or temperatures lower this score.
- **accuracy**: Is the structured information correct? Wrong quantities, misattributed ingredients, invented information, or mis-parsed fields lower this score.

IMPORTANT: Only flag actual errors that affect meaning or usability. Do NOT flag:
- Trivial singular/plural differences ("onion" vs "onions")
- Minor rewording that preserves meaning ("mix together" vs "combine")
- Correcting obvious typos in the original ("candid" → "candied")
- Reasonable abbreviation or expansion of ingredient names
- Adding standard units or clarifications that don't change meaning

DO flag:
- Missing ingredients or steps
- Wrong quantities, units, or temperatures
- Ingredients attributed to the wrong step
- Invented information not in the original
- Steps that materially change the cooking process
- Listing all ingredients in a step when only some are actually used

Classify each error by severity:
- "major": changes the recipe outcome (wrong quantity, missing step, wrong temperature)
- "minor": cosmetic or structural issue that doesn't affect cooking

Output a JSON object with these fields:
- "completeness": float 0.0-1.0
- "accuracy": float 0.0-1.0
- "errors": array of objects, each with "description" (string) and "severity" ("major" or "minor")
- "commentary": string (brief overall assessment)

Output only valid JSON, no markdown or commentary outside the JSON.
"""


def load_ingested(jsonl_path: Path) -> dict[str, dict]:
    """Load ingested recipes keyed by id."""
    recipes = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recipes[r["id"]] = r
    return recipes


def format_eval_message(original: dict, structured: dict) -> str:
    parts = [
        "=== ORIGINAL RECIPE ===",
        f"Title: {original['title']}",
    ]
    if original.get("ingredients_raw"):
        parts.append(f"Ingredients:\n{original['ingredients_raw']}")
    if original.get("directions_raw"):
        parts.append(f"Directions:\n{original['directions_raw']}")
    parts.append("\n=== STRUCTURED EXTRACTION ===")
    parts.append(json.dumps(structured, indent=2))
    return "\n\n".join(parts)


def _build_eval_payload(user_message: str) -> dict:
    return {
        "model": EVAL_MODEL,
        "messages": [
            {"role": "system", "content": EVAL_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }


def _save_eval(eval_path: Path, recipe_id: str, model_slug: str, content: str, parsed: dict | None, usage: dict) -> None:
    evaluation = {
        "recipe_id": recipe_id,
        "model": model_slug,
        "eval_model": EVAL_MODEL,
        "raw_response": content,
        "parsed": parsed,
        "usage": usage,
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)


# --- Async API ---

async def _eval_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    recipe_id: str,
    model_slug: str,
    out_path: Path,
    eval_path: Path,
    ingested: dict[str, dict],
    api_key: str,
    counter: dict,
    total: int,
) -> None:
    async with semaphore:
        counter["started"] += 1
        idx = counter["started"]

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        if not result.get("parsed"):
            print(f"  [{idx}/{total}] {recipe_id} (no parsed output, skipping)")
            return

        original = ingested.get(recipe_id)
        if original is None:
            print(f"  [{idx}/{total}] {recipe_id} (original not found, skipping)")
            return

        try:
            user_message = format_eval_message(original, result["parsed"])
            payload = _build_eval_payload(user_message)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            start = time.monotonic()
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
            elapsed = round(time.monotonic() - start, 1)
            resp.raise_for_status()
            response = resp.json()

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = extract_json(content)

            _save_eval(eval_path, recipe_id, model_slug, content, parsed, response.get("usage", {}))

            if parsed:
                print(f"  [{idx}/{total}] {recipe_id} completeness={parsed.get('completeness')} accuracy={parsed.get('accuracy')} ({elapsed}s)")
            else:
                print(f"  [{idx}/{total}] {recipe_id} eval_parse_fail ({elapsed}s)")
        except Exception as e:
            print(f"  [{idx}/{total}] {recipe_id} error: {e}")


async def evaluate_model_async(
    model_slug: str,
    ingested: dict[str, dict],
    api_key: str,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    model_dir = OUTPUT_DIR / model_slug
    if not model_dir.exists():
        print(f"No outputs found for {model_slug}")
        return

    eval_dir = EVAL_DIR / model_slug
    eval_dir.mkdir(parents=True, exist_ok=True)

    output_files = sorted(model_dir.glob("*.json"))
    if limit is not None:
        output_files = output_files[:limit]

    # Filter to uncached
    tasks = []
    cached = 0
    for out_path in output_files:
        recipe_id = out_path.stem
        eval_path = eval_dir / f"{recipe_id}.json"
        if eval_path.exists():
            cached += 1
            continue
        tasks.append((recipe_id, out_path, eval_path))

    total = len(tasks)
    print(f"\n=== Evaluating {model_slug} | {total} to process, {cached} cached, concurrency={concurrency} ===")
    if not tasks:
        return

    semaphore = asyncio.Semaphore(concurrency)
    counter = {"started": 0}

    async with httpx.AsyncClient() as client:
        coros = [
            _eval_one(client, semaphore, recipe_id, model_slug, out_path, eval_path, ingested, api_key, counter, total)
            for recipe_id, out_path, eval_path in tasks
        ]
        await asyncio.gather(*coros)


async def evaluate_all_async(
    model_slugs: list[str],
    ingested: dict[str, dict],
    api_key: str,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    for slug in model_slugs:
        await evaluate_model_async(slug, ingested, api_key, limit, concurrency)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.evaluate <recipes.jsonl> <model_slug1,model_slug2,...> [limit] [concurrency]")
        print("model_slug uses -- instead of / (e.g. google--gemma-3-4b-it)")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    model_slugs = sys.argv[2].split(",")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_CONCURRENCY

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    ingested = load_ingested(jsonl_path)
    asyncio.run(evaluate_all_async(model_slugs, ingested, api_key, limit, concurrency))
