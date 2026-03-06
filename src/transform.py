"""Send raw recipes to LLMs via OpenRouter and collect structured output."""

import json
import os
import sys
import time
from pathlib import Path

import httpx

from src.env import load_dotenv
from src.schema import Recipe

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPTS_DIR = Path("prompts")
OUTPUT_DIR = Path("data_nosync/outputs")


def load_prompt(version: str = "v1") -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    return path.read_text(encoding="utf-8")


def load_recipes(jsonl_path: Path) -> list[dict]:
    recipes = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recipes.append(json.loads(line))
    return recipes


def format_user_message(recipe: dict) -> str:
    parts = [f"Title: {recipe['title']}"]
    if recipe.get("ingredients_raw"):
        parts.append(f"Ingredients:\n{recipe['ingredients_raw']}")
    if recipe.get("directions_raw"):
        parts.append(f"Directions:\n{recipe['directions_raw']}")
    return "\n\n".join(parts)


def call_model(
    model: str,
    system_prompt: str,
    user_message: str,
    api_key: str,
) -> dict:
    """Call a model via OpenRouter. Returns the raw API response dict."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }
    start = time.monotonic()
    resp = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
    elapsed = time.monotonic() - start
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed_seconds"] = round(elapsed, 3)
    return data


def extract_json(text: str) -> dict | None:
    """Try to parse JSON from model output, handling markdown fences and preamble."""
    text = text.strip()
    # Extract from markdown code fences if present
    import re
    fence_match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip preamble before the first {
    idx = text.find("{")
    if idx > 0:
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            pass
    # Try finding the last complete JSON object (for CoT-style outputs)
    # Find the last } and search backwards for matching {
    last_brace = text.rfind("}")
    if last_brace > 0:
        depth = 0
        for i in range(last_brace, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:last_brace + 1])
                except json.JSONDecodeError:
                    break
    return None


def transform_one(
    recipe: dict,
    model: str,
    system_prompt: str,
    api_key: str,
) -> dict:
    """Transform a single recipe. Returns a result dict with parsed output and metadata."""
    user_message = format_user_message(recipe)
    response = call_model(model, system_prompt, user_message, api_key)

    content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
    usage = response.get("usage", {})
    elapsed = response.get("_elapsed_seconds", 0)

    parsed = extract_json(content) if content else None
    validation_ok = False
    validation_errors = None
    if parsed is not None:
        try:
            Recipe(**parsed)
            validation_ok = True
        except Exception as e:
            validation_errors = str(e)

    return {
        "recipe_id": recipe["id"],
        "model": model,
        "prompt_version": None,  # set by caller
        "raw_response": content,
        "parsed": parsed,
        "validation_ok": validation_ok,
        "validation_errors": validation_errors,
        "usage": usage,
        "elapsed_seconds": elapsed,
    }


def transform_batch(
    jsonl_path: Path,
    models: list[str],
    prompt_version: str = "v1",
    limit: int | None = None,
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    system_prompt = load_prompt(prompt_version)
    recipes = load_recipes(jsonl_path)
    if limit is not None:
        recipes = recipes[:limit]

    for model in models:
        model_slug = model.replace("/", "--")
        if prompt_version != "v1":
            model_slug = f"{model_slug}_{prompt_version}"
        model_dir = OUTPUT_DIR / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {model} ({prompt_version}) ===")
        for i, recipe in enumerate(recipes):
            out_path = model_dir / f"{recipe['id']}.json"
            if out_path.exists():
                print(f"  [{i+1}/{len(recipes)}] {recipe['id']} (cached)")
                continue

            print(f"  [{i+1}/{len(recipes)}] {recipe['id']}...", end=" ", flush=True)
            try:
                result = transform_one(recipe, model, system_prompt, api_key)
                result["prompt_version"] = prompt_version
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                status = "ok" if result["validation_ok"] else "parse_fail"
                print(status)
            except Exception as e:
                print(f"error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.transform <recipes.jsonl> <model1,model2,...> [prompt_version] [limit]")
        print("Example: python -m src.transform data_nosync/ingested/recipenlg.jsonl google/gemma-3-4b-it,mistralai/mistral-small v1 10")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    models = sys.argv[2].split(",")
    prompt_version = sys.argv[3] if len(sys.argv) > 3 else "v1"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    transform_batch(jsonl_path, models, prompt_version, limit)
