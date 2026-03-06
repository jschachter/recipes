"""Find the structural opposite of a concept in cooking eigenvector space."""

import json
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

from src.graph import build_graph, apply_pmi, normalize, OUTPUT_DIR


DEFAULT_K = 20
DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"


def find_concept(nodes, eigenvectors, eigenvalues, query_features, k_dims=None):
    """Find which dimensions a concept loads on and what's at the other end.

    query_features: list of feature names (e.g. ["tag:salad", "action:toss"])
    """
    idx = {n: i for i, n in enumerate(nodes)}
    valid = [idx[f] for f in query_features if f in idx]
    missing = [f for f in query_features if f not in idx]

    if missing:
        print(f"  (not found: {', '.join(missing)})")
    if not valid:
        print("No valid features found.")
        return

    if k_dims is None:
        k_dims = eigenvectors.shape[1]

    # Average loading of query features on each dimension
    scores = []
    for d in range(k_dims):
        vals = eigenvectors[valid, d]
        scores.append((abs(vals.mean()), vals.mean(), d))

    scores.sort(reverse=True)

    print(f"\nDimensions where this concept loads most strongly:\n")
    for rank, (abs_score, score, d) in enumerate(scores[:5]):
        sign = "negative" if score < 0 else "positive"
        vec = eigenvectors[:, d]
        ranked = sorted(zip(vec, nodes), key=lambda x: x[0])

        print(f"  Dimension {d+1} (λ={eigenvalues[d]:.1f}) — concept is on the {sign} end (loading={score:+.4f})")

        # Show the opposite end
        if score < 0:
            opposite = ranked[-15:]
            same = [x for x in ranked[:15] if abs(x[0]) > 0.01]
        else:
            opposite = [(v, n) for v, n in ranked[:15] if abs(v) > 0.01]
            same = ranked[-15:]

        print(f"    YOUR CONCEPT's neighborhood:")
        for val, name in (reversed(same) if score < 0 else same):
            if abs(val) > 0.01:
                print(f"      {val:+.3f}  {name}")

        print(f"    OPPOSITE end:")
        for val, name in (opposite if score < 0 else reversed(opposite)):
            if abs(val) > 0.01:
                print(f"      {val:+.3f}  {name}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.opposite <feature1,feature2,...> [model_slug] [k]")
        print()
        print("Features use the same prefixes as the graph:")
        print("  tag:salad  ingredient:butter  action:simmer  tool:oven  temp:350°")
        print()
        print("Examples:")
        print("  python -m src.opposite tag:salad")
        print("  python -m src.opposite tag:casserole,action:layer,ingredient:cream-of-mushroom-soup")
        print("  python -m src.opposite action:grill,tool:grill,tag:grilled")
        sys.exit(1)

    query = sys.argv[1].split(",")
    model_slug = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_K

    print(f"Building graph for {model_slug}...")
    nodes, A, recipe_count = build_graph(model_slug)
    print(f"Recipes: {recipe_count}, Nodes: {len(nodes)}")

    print("Applying PPMI weighting...")
    A = apply_pmi(nodes, A)

    k = min(k, len(nodes) - 2)
    print(f"Computing {k} eigenvectors...")
    eigenvalues, eigenvectors = eigsh(A, k=k, which="LM")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    print(f"\nQuery: {query}")
    find_concept(nodes, eigenvectors, eigenvalues, query)


if __name__ == "__main__":
    main()
