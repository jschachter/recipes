"""Spot-check structured recipe outputs using a high-end model."""

import json
import os
import sys
from pathlib import Path

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_DIR = Path("data_nosync/outputs")
EVAL_DIR = Path("data_nosync/evaluations")

EVAL_MODEL = "anthropic/claude-sonnet-4"

EVAL_PROMPT = """\
You are evaluating the quality of a structured recipe extraction. You will receive:
1. The original recipe text
2. A JSON structured extraction of that recipe

Score the extraction on two dimensions (0.0 to 1.0):
- **completeness**: Does the structured output capture all information from the original? Missing ingredients, steps, times, or temperatures lower this score.
- **accuracy**: Is the structured information correct? Wrong quantities, misattributed ingredients, invented information, or mis-parsed fields lower this score.

Also list specific errors you find, and provide brief commentary on the overall quality.

Output a JSON object with these fields:
- "completeness": float 0.0-1.0
- "accuracy": float 0.0-1.0
- "errors": array of strings (specific issues found, empty if none)
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


def call_eval_model(user_message: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EVAL_MODEL,
        "messages": [
            {"role": "system", "content": EVAL_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }
    resp = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate_model(
    model_slug: str,
    ingested: dict[str, dict],
    api_key: str,
    limit: int | None = None,
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

    print(f"\n=== Evaluating {model_slug} ({len(output_files)} recipes) ===")
    for i, out_path in enumerate(output_files):
        recipe_id = out_path.stem
        eval_path = eval_dir / f"{recipe_id}.json"

        if eval_path.exists():
            print(f"  [{i+1}/{len(output_files)}] {recipe_id} (cached)")
            continue

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        if not result.get("parsed"):
            print(f"  [{i+1}/{len(output_files)}] {recipe_id} (no parsed output, skipping)")
            continue

        original = ingested.get(recipe_id)
        if original is None:
            print(f"  [{i+1}/{len(output_files)}] {recipe_id} (original not found, skipping)")
            continue

        print(f"  [{i+1}/{len(output_files)}] {recipe_id}...", end=" ", flush=True)
        try:
            user_message = format_eval_message(original, result["parsed"])
            response = call_eval_model(user_message, api_key)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = extract_json(content)

            evaluation = {
                "recipe_id": recipe_id,
                "model": model_slug,
                "eval_model": EVAL_MODEL,
                "raw_response": content,
                "parsed": parsed,
                "usage": response.get("usage", {}),
            }
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(evaluation, f, indent=2)
            if parsed:
                print(f"completeness={parsed.get('completeness')} accuracy={parsed.get('accuracy')}")
            else:
                print("eval_parse_fail")
        except Exception as e:
            print(f"error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.evaluate <recipes.jsonl> <model_slug1,model_slug2,...> [limit]")
        print("model_slug uses -- instead of / (e.g. google--gemma-3-4b-it)")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    model_slugs = sys.argv[2].split(",")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    ingested = load_ingested(jsonl_path)
    for slug in model_slugs:
        evaluate_model(slug, ingested, api_key, limit)
