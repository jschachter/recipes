"""Aggregate evaluation results and compare models."""

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("data_nosync/outputs")
EVAL_DIR = Path("data_nosync/evaluations")


def load_results(model_slug: str) -> list[dict]:
    model_dir = OUTPUT_DIR / model_slug
    if not model_dir.exists():
        return []
    results = []
    for p in sorted(model_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def load_evaluations(model_slug: str) -> list[dict]:
    eval_dir = EVAL_DIR / model_slug
    if not eval_dir.exists():
        return []
    evals = []
    for p in sorted(eval_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            evals.append(json.load(f))
    return evals


def summarize_model(model_slug: str) -> dict | None:
    results = load_results(model_slug)
    evals = load_evaluations(model_slug)
    if not results:
        return None

    total = len(results)
    valid = sum(1 for r in results if r.get("validation_ok"))
    elapsed = [r["elapsed_seconds"] for r in results if r.get("elapsed_seconds")]
    prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)

    summary = {
        "model": model_slug,
        "total_recipes": total,
        "validation_pass": valid,
        "validation_rate": round(valid / total, 3) if total else 0,
        "avg_elapsed": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
    }

    if evals:
        scored = [e["parsed"] for e in evals if e.get("parsed")]
        if scored:
            completeness = [s["completeness"] for s in scored if s.get("completeness") is not None]
            accuracy = [s["accuracy"] for s in scored if s.get("accuracy") is not None]
            all_errors = []
            for s in scored:
                all_errors.extend(s.get("errors", []))

            summary["eval_count"] = len(scored)
            summary["avg_completeness"] = round(sum(completeness) / len(completeness), 3) if completeness else None
            summary["avg_accuracy"] = round(sum(accuracy) / len(accuracy), 3) if accuracy else None
            summary["total_errors"] = len(all_errors)
            summary["common_errors"] = _top_errors(all_errors, 5)

    return summary


def _top_errors(errors: list[str], n: int) -> list[str]:
    """Return the n most common error strings (exact match)."""
    counts: dict[str, int] = {}
    for e in errors:
        counts[e] = counts.get(e, 0) + 1
    return [e for e, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]


def print_report(model_slugs: list[str]) -> None:
    summaries = []
    for slug in model_slugs:
        s = summarize_model(slug)
        if s:
            summaries.append(s)
        else:
            print(f"No results for {slug}")

    if not summaries:
        print("No data to report.")
        return

    # Header
    print(f"\n{'Model':<40} {'Valid':>6} {'Rate':>6} {'Avg s':>6} {'Compl':>6} {'Accur':>6} {'Errs':>5}")
    print("-" * 85)

    for s in sorted(summaries, key=lambda x: x.get("avg_accuracy") or 0, reverse=True):
        compl = f"{s['avg_completeness']:.3f}" if s.get("avg_completeness") is not None else "  n/a"
        accur = f"{s['avg_accuracy']:.3f}" if s.get("avg_accuracy") is not None else "  n/a"
        errs = str(s.get("total_errors", "n/a"))
        print(f"{s['model']:<40} {s['validation_pass']:>5}/{s['total_recipes']:<5} {s['validation_rate']:>5.1%} {s['avg_elapsed']:>5.1f}s {compl:>6} {accur:>6} {errs:>5}")

    # Detail section
    for s in summaries:
        if s.get("common_errors"):
            print(f"\n--- {s['model']} common errors ---")
            for e in s["common_errors"]:
                print(f"  - {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-discover models from output dir
        if OUTPUT_DIR.exists():
            slugs = [d.name for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]
        else:
            print("No output directory found. Run transform first.")
            sys.exit(1)
    else:
        slugs = sys.argv[1].split(",")

    print_report(slugs)
