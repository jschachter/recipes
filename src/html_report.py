"""Generate per-recipe HTML reports comparing model outputs and evaluations."""

import json
import sys
from html import escape
from pathlib import Path

OUTPUT_DIR = Path("data_nosync/outputs")
EVAL_DIR = Path("data_nosync/evaluations")
REPORT_DIR = Path("data_nosync/reports")


def load_ingested(jsonl_path: Path) -> dict[str, dict]:
    recipes = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recipes[r["id"]] = r
    return recipes


def discover_models() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir())


def load_output(model_slug: str, recipe_id: str) -> dict | None:
    p = OUTPUT_DIR / model_slug / f"{recipe_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_eval(model_slug: str, recipe_id: str) -> dict | None:
    p = EVAL_DIR / model_slug / f"{recipe_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def score_bar(value: float | None) -> str:
    if value is None:
        return '<span class="na">n/a</span>'
    pct = int(value * 100)
    color = "#4caf50" if value >= 0.8 else "#ff9800" if value >= 0.6 else "#f44336"
    return f'<div class="bar"><div class="fill" style="width:{pct}%;background:{color}"></div><span>{value:.2f}</span></div>'


def render_ingredients(ingredients: list[dict]) -> str:
    if not ingredients:
        return "<em>none</em>"
    rows = []
    for ing in ingredients:
        rows.append(f"""<tr>
            <td>{escape(str(ing.get('quantity', '') or ''))}</td>
            <td>{escape(str(ing.get('unit', '') or ''))}</td>
            <td>{escape(str(ing.get('name', '') or ''))}</td>
            <td>{escape(str(ing.get('preparation', '') or ''))}</td>
            <td class="raw">{escape(str(ing.get('raw_text', '') or ''))}</td>
        </tr>""")
    return f"""<table class="ingredients">
        <tr><th>Qty</th><th>Unit</th><th>Name</th><th>Prep</th><th>Raw</th></tr>
        {''.join(rows)}
    </table>"""


def _classify_errors(errors: list) -> tuple[dict[int, list], list, list]:
    """Classify errors into step-specific, ingredient-related, and general.
    Returns (step_errors_dict, ingredient_errors, general_errors)."""
    import re
    step_errors: dict[int, list] = {}
    ingredient_errors = []
    general = []
    for e in errors:
        text = e.get("description", str(e)) if isinstance(e, dict) else str(e)
        m = re.search(r'\bStep\s+(\d+)\b', text, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            step_errors.setdefault(idx, []).append(e)
        elif re.search(r'\b(ingredient|quantity|unit)\b', text, re.IGNORECASE):
            ingredient_errors.append(e)
        else:
            general.append(e)
    return step_errors, ingredient_errors, general


def _render_error_item(e) -> str:
    if isinstance(e, dict):
        sev = e.get("severity", "")
        desc = e.get("description", str(e))
        sev_tag = f' <span class="severity-{sev}">[{sev}]</span>' if sev else ""
        return f"<li>{escape(desc)}{sev_tag}</li>"
    return f"<li>{escape(str(e))}</li>"


def render_steps(steps: list[dict], step_errors: dict[int, list] | None = None) -> str:
    if not steps:
        return "<em>none</em>"
    parts = []
    for step in steps:
        idx = step.get("index", "?")
        action = escape(str(step.get("action", "") or ""))
        desc = escape(str(step.get("description", "") or ""))
        raw = escape(str(step.get("raw_text", "") or ""))
        ingredients = ", ".join(str(x) for x in (step.get("ingredients", []) or []) if x)
        tools = ", ".join(str(x) for x in (step.get("tools", []) or []) if x)
        duration = escape(str(step.get("duration", "") or ""))
        temp = escape(str(step.get("temperature", "") or ""))

        meta_parts = []
        if ingredients:
            meta_parts.append(f"<strong>ingredients:</strong> {escape(ingredients)}")
        if tools:
            meta_parts.append(f"<strong>tools:</strong> {escape(tools)}")
        if duration:
            meta_parts.append(f"<strong>duration:</strong> {duration}")
        if temp:
            meta_parts.append(f"<strong>temp:</strong> {temp}")
        meta = " | ".join(meta_parts)

        errors_html = ""
        if step_errors and isinstance(idx, int) and idx in step_errors:
            items = "".join(_render_error_item(e) for e in step_errors[idx])
            errors_html = f'<div class="step-errors"><ul>{items}</ul></div>'

        parts.append(f"""<div class="step{' step-has-error' if errors_html else ''}">
            <div class="step-header">Step {idx}: <strong>{action}</strong></div>
            <div class="step-desc">{desc}</div>
            {f'<div class="step-meta">{meta}</div>' if meta else ''}
            <div class="step-raw">Raw: {raw}</div>
            {errors_html}
        </div>""")
    return "".join(parts)


def render_model_section(model_slug: str, output: dict, evaluation: dict | None) -> str:
    parsed = output.get("parsed")
    if not parsed:
        return f"""<div class="model-section">
            <h3>{escape(model_slug)}</h3>
            <p class="error">No parsed output (validation failed)</p>
            <details><summary>Raw response</summary><pre>{escape(output.get('raw_response', ''))}</pre></details>
        </div>"""

    eval_parsed = evaluation.get("parsed") if evaluation else None

    scores_html = ""
    step_errors_map = {}
    ingredient_errors = []
    if eval_parsed:
        scores_html = f"""<div class="scores">
            <div class="score">Completeness {score_bar(eval_parsed.get('completeness'))}</div>
            <div class="score">Accuracy {score_bar(eval_parsed.get('accuracy'))}</div>
        </div>"""
        errors = eval_parsed.get("errors", [])
        commentary = eval_parsed.get("commentary", "")
        if errors:
            steps = parsed.get("steps", [])
            step_errors_map, ingredient_errors, general = _classify_errors(errors)
            if general:
                scores_html += '<div class="errors"><strong>General errors:</strong><ul>'
                scores_html += "".join(_render_error_item(e) for e in general)
                scores_html += "</ul></div>"
        if commentary:
            scores_html += f'<div class="commentary"><strong>Commentary:</strong> {escape(commentary)}</div>'

    cost = output.get("usage", {}).get("cost")
    elapsed = output.get("elapsed_seconds")
    meta_parts = []
    if elapsed:
        meta_parts.append(f"{elapsed:.1f}s")
    if cost:
        meta_parts.append(f"${cost:.6f}")
    meta_html = f'<div class="model-meta">{" | ".join(meta_parts)}</div>' if meta_parts else ""

    return f"""<div class="model-section">
        <h3>{escape(model_slug)}</h3>
        {meta_html}
        {scores_html}
        <h4>Ingredients</h4>
        {render_ingredients(parsed.get('ingredients', []))}
        {'<div class="errors"><ul>' + "".join(_render_error_item(e) for e in ingredient_errors) + "</ul></div>" if eval_parsed and ingredient_errors else ""}
        <h4>Steps</h4>
        {render_steps(parsed.get('steps', []), step_errors_map)}
    </div>"""


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #555; }
.original { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.original pre { white-space: pre-wrap; background: #f9f9f9; padding: 10px; border-radius: 4px; }
.model-section { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.model-meta { color: #888; font-size: 0.9em; margin-bottom: 10px; }
.scores { display: flex; gap: 20px; margin: 10px 0; }
.score { flex: 1; }
.bar { background: #eee; border-radius: 4px; height: 24px; position: relative; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.bar span { position: absolute; right: 8px; top: 2px; font-size: 0.85em; font-weight: bold; }
.na { color: #999; }
.errors { margin: 10px 0; color: #c62828; }
.errors ul { margin: 5px 0; }
.commentary { margin: 10px 0; color: #555; font-style: italic; }
table.ingredients { border-collapse: collapse; width: 100%; margin: 10px 0; }
table.ingredients th, table.ingredients td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
table.ingredients th { background: #f0f0f0; }
td.raw { color: #888; font-size: 0.85em; }
.step { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
.step-header { font-size: 1.05em; }
.step-desc { margin: 5px 0; }
.step-meta { font-size: 0.9em; color: #555; }
.step-raw { font-size: 0.85em; color: #888; margin-top: 5px; }
.step-errors { margin-top: 5px; }
.step-errors ul { margin: 2px 0; padding-left: 20px; }
.step-errors li { color: #c62828; font-size: 0.9em; }
.step-has-error { border-left: 3px solid #f44336; }
.error { color: #c62828; }
.severity-major { color: #c62828; font-weight: bold; font-size: 0.85em; }
.severity-minor { color: #ff9800; font-size: 0.85em; }
a { color: #1976d2; }
.nav { margin: 20px 0; }
.nav a { margin-right: 10px; }
"""


def generate_recipe_report(recipe_id: str, original: dict, models: list[str]) -> str:
    title = escape(original.get("title", recipe_id))

    original_html = f"""<div class="original">
        <h2>Original Recipe</h2>
        <pre>{escape(original.get('ingredients_raw', ''))}</pre>
        <pre>{escape(original.get('directions_raw', ''))}</pre>
    </div>"""

    model_sections = []
    for model_slug in models:
        output = load_output(model_slug, recipe_id)
        if not output:
            continue
        evaluation = load_eval(model_slug, recipe_id)
        model_sections.append(render_model_section(model_slug, output, evaluation))

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head><body>
<div class="nav"><a href="index.html">&#8592; Index</a></div>
<h1>{title}</h1>
{original_html}
{''.join(model_sections)}
</body></html>"""


def summarize_models(models: list[str]) -> list[dict]:
    summaries = []
    for model_slug in models:
        # Load timing/cost from outputs
        model_dir = OUTPUT_DIR / model_slug
        elapsed_times = []
        total_cost = 0.0
        if model_dir.exists():
            for p in model_dir.glob("*.json"):
                with open(p) as f:
                    out = json.load(f)
                if out.get("elapsed_seconds"):
                    elapsed_times.append(out["elapsed_seconds"])
                total_cost += out.get("usage", {}).get("cost", 0) or 0

        # Load eval scores
        eval_dir = EVAL_DIR / model_slug
        scores = []
        if eval_dir.exists():
            for p in eval_dir.glob("*.json"):
                with open(p) as f:
                    e = json.load(f)
                parsed = e.get("parsed")
                if parsed:
                    scores.append(parsed)

        if not elapsed_times and not scores:
            continue

        # Validation rate
        total_outputs = len(list(model_dir.glob("*.json"))) if model_dir.exists() else 0
        valid_outputs = 0
        if model_dir.exists():
            for p in model_dir.glob("*.json"):
                with open(p) as f:
                    o = json.load(f)
                if o.get("validation_ok"):
                    valid_outputs += 1

        compl = [s["completeness"] for s in scores if s.get("completeness") is not None]
        accur = [s["accuracy"] for s in scores if s.get("accuracy") is not None]

        avg_compl = sum(compl) / len(compl) if compl else None
        avg_accur = sum(accur) / len(accur) if accur else None
        avg_elapsed = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
        valid_rate = valid_outputs / total_outputs if total_outputs else 0

        cost_per = total_cost / total_outputs if total_outputs else 0

        summaries.append({
            "model": model_slug,
            "count": len(scores),
            "avg_completeness": avg_compl,
            "avg_accuracy": avg_accur,
            "avg_elapsed": avg_elapsed,
            "total_cost": total_cost,
            "cost_per": cost_per,
            "valid_rate": valid_rate,
        })

    # Compute combined score with cost normalized across all models (log scale)
    import math
    costs = [s["cost_per"] for s in summaries if s["cost_per"] > 0]
    if costs:
        log_min = math.log(min(costs))
        log_max = math.log(max(costs))
        log_range = log_max - log_min if log_max > log_min else 1
    for s in summaries:
        quality = ((s["avg_completeness"] or 0) + (s["avg_accuracy"] or 0)) / 2
        speed = max(0, 1 - (s["avg_elapsed"] or 999) / 60)
        if s["cost_per"] > 0 and costs:
            cost_score = 1 - (math.log(s["cost_per"]) - log_min) / log_range
        else:
            cost_score = 0.5
        # 60% quality + 10% reliability + 10% speed + 20% cost
        s["combined"] = quality * 0.60 + s["valid_rate"] * 0.10 + speed * 0.10 + cost_score * 0.20

    return sorted(summaries, key=lambda s: s.get("combined") or 0, reverse=True)


def render_model_summary(summaries: list[dict]) -> str:
    if not summaries:
        return ""
    rows = []
    for s in summaries:
        elapsed = f"{s['avg_elapsed']:.1f}s" if s.get("avg_elapsed") is not None else "n/a"
        cost_per = f"${s['cost_per']:.5f}" if s.get("cost_per") else "n/a"
        valid = f"{s['valid_rate']:.0%}" if s.get("valid_rate") is not None else "n/a"
        rows.append(f"""<tr>
            <td><strong>{escape(s['model'])}</strong></td>
            <td>{score_bar(s.get('combined'))}</td>
            <td>{score_bar(s['avg_completeness'])}</td>
            <td>{score_bar(s['avg_accuracy'])}</td>
            <td>{valid}</td>
            <td>{elapsed}</td>
            <td>{cost_per}</td>
        </tr>""")
    return f"""<div class="model-section">
        <h2>Model Scores</h2>
        <p style="color:#666;font-size:0.9em">Combined = 60% quality + 10% reliability + 10% speed + 20% cost (log-scaled)</p>
        <table class="ingredients">
            <tr><th>Model</th><th>Combined</th><th>Completeness</th><th>Accuracy</th><th>Valid</th><th>Avg Time</th><th>$/recipe</th></tr>
            {''.join(rows)}
        </table>
    </div>"""


def generate_index(recipes: dict[str, dict], models: list[str]) -> str:
    summaries = summarize_models(models)
    links = []
    for rid, r in sorted(recipes.items()):
        title = escape(r.get("title", rid))
        links.append(f'<li><a href="{rid}.html">{title}</a></li>')
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Recipe Pipeline Reports</title>
<style>{CSS}</style>
</head><body>
<h1>Recipe Pipeline Reports</h1>
{render_model_summary(summaries)}
<h2>Recipes</h2>
<ul>{''.join(links)}</ul>
</body></html>"""


def generate_all(jsonl_path: Path) -> None:
    ingested = load_ingested(jsonl_path)
    models = discover_models()

    if not models:
        print("No model outputs found.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Only generate for recipes that have outputs
    recipe_ids = set()
    for model_slug in models:
        model_dir = OUTPUT_DIR / model_slug
        for p in model_dir.glob("*.json"):
            recipe_ids.add(p.stem)

    generated = {}
    for rid in sorted(recipe_ids):
        original = ingested.get(rid)
        if not original:
            continue
        html = generate_recipe_report(rid, original, models)
        out_path = REPORT_DIR / f"{rid}.html"
        out_path.write_text(html, encoding="utf-8")
        generated[rid] = original
        print(f"  {rid}.html")

    index_html = generate_index(generated, models)
    (REPORT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"\nGenerated {len(generated)} reports in {REPORT_DIR}/")
    url = str(REPORT_DIR.resolve()).replace("/home/joshua/Dropbox/work", "http://bigger/work")
    print(f"Open: {url}/index.html")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.html_report <recipes.jsonl>")
        sys.exit(1)
    generate_all(Path(sys.argv[1]))
