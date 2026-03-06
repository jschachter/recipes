"""Compare eigenvector structure across cuisines and categories."""

import sys

import numpy as np
from scipy.sparse.linalg import eigsh

from src.graph import build_graph, apply_pmi


DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"


def analyze_slice(model_slug: str, filter_tag: str, k: int = 5):
    """Compute eigenvectors for a filtered subset and return summary."""
    nodes, A, rc = build_graph(model_slug, filters=[filter_tag])
    if rc < 50:
        return None

    A = apply_pmi(nodes, A)
    actual_k = min(k, len(nodes) - 2)
    eigenvalues, eigenvectors = eigsh(A, k=actual_k, which="LM")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    dims = []
    for d in range(actual_k):
        vec = eigenvectors[:, d]
        ranked = sorted(zip(vec, nodes), key=lambda x: x[0])
        neg = [(v, n) for v, n in ranked[:8] if abs(v) > 0.02]
        pos = [(v, n) for v, n in ranked[-8:] if abs(v) > 0.02]
        dims.append({
            "eigenvalue": eigenvalues[d],
            "negative": neg,
            "positive": pos,
        })

    return {
        "filter": filter_tag,
        "recipes": rc,
        "nodes": len(nodes),
        "dimensions": dims,
    }


def print_slice(result):
    if result is None:
        return
    print(f"\n{'='*60}")
    print(f"  {result['filter']} — {result['recipes']} recipes, {result['nodes']} nodes")
    print(f"{'='*60}")

    for i, dim in enumerate(result["dimensions"]):
        print(f"\n  EV{i+1} (λ={dim['eigenvalue']:.1f}):")
        if dim["negative"]:
            neg_str = ", ".join(n for _, n in dim["negative"])
            print(f"    - : {neg_str}")
        if dim["positive"]:
            pos_str = ", ".join(n for _, n in reversed(dim["positive"]))
            print(f"    + : {pos_str}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.compare_cuisines <tag1,tag2,...> [k] [model]")
        print()
        print("Examples:")
        print("  python -m src.compare_cuisines italian,mexican,chinese,indian,french")
        print("  python -m src.compare_cuisines dessert,soup,salad,entree 8")
        print("  python -m src.compare_cuisines all")
        sys.exit(1)

    if sys.argv[1] == "all":
        tags = [
            "tag:italian", "tag:mexican", "tag:chinese", "tag:indian",
            "tag:french", "tag:thai", "tag:japanese",
            "tag:dessert", "tag:entree", "tag:soup", "tag:salad",
            "tag:appetizer", "tag:bread", "tag:beverage",
        ]
    else:
        raw = sys.argv[1].split(",")
        tags = [t if ":" in t else f"tag:{t}" for t in raw]

    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    model_slug = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    results = []
    for tag in tags:
        result = analyze_slice(model_slug, tag, k)
        if result:
            print_slice(result)
            results.append(result)
        else:
            print(f"\n  {tag}: too few recipes, skipped")

    # Summary: what dimensions are shared vs unique?
    if len(results) > 1:
        print(f"\n\n{'='*60}")
        print("  CROSS-CUISINE COMPARISON")
        print(f"{'='*60}")
        for r in results:
            dims_summary = []
            for i, dim in enumerate(r["dimensions"]):
                top_feats = []
                for _, n in dim["negative"][:3]:
                    top_feats.append(n)
                for _, n in list(reversed(dim["positive"]))[:3]:
                    top_feats.append(n)
                dims_summary.append(f"EV{i+1}: {', '.join(top_feats[:4])}")
            print(f"\n  {r['filter']} ({r['recipes']} recipes):")
            for ds in dims_summary:
                print(f"    {ds}")


if __name__ == "__main__":
    main()
