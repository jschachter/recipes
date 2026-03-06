"""Suggest ingredients, tools, or techniques based on eigenvector proximity."""

import json
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

from src.graph import build_graph, apply_pmi, normalize, OUTPUT_DIR


DEFAULT_K = 50
DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"


def suggest(nodes, eigenvectors, query_features, filter_prefix=None, n=20):
    """Find features nearest to the query centroid in eigenvector space.

    filter_prefix: only return results matching this prefix (e.g. "ingredient:", "action:")
    """
    idx = {n: i for i, n in enumerate(nodes)}
    valid = [idx[f] for f in query_features if f in idx]
    missing = [f for f in query_features if f not in idx]

    if missing:
        print(f"  (not found: {', '.join(missing)})")
    if not valid:
        print("No valid features found.")
        return

    # Centroid of query features in eigenvector space
    centroid = eigenvectors[valid, :].mean(axis=0)

    # Distance from every node to centroid
    diffs = eigenvectors - centroid
    distances = np.linalg.norm(diffs, axis=1)

    # Rank by proximity
    query_set = set(valid)
    results = []
    for i in np.argsort(distances):
        name = nodes[i]
        if i in query_set:
            continue
        if filter_prefix and not name.startswith(filter_prefix):
            continue
        results.append((distances[i], name))
        if len(results) >= n:
            break

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.suggest <feature1,feature2,...> [filter_prefix] [model_slug] [k]")
        print()
        print("filter_prefix limits results to a type: ingredient: action: tool: tag: temp:")
        print()
        print("Examples:")
        print("  python -m src.suggest tag:salad,ingredient:tuna ingredient:")
        print("  python -m src.suggest ingredient:beef,action:braise,tag:entree ingredient:")
        print("  python -m src.suggest ingredient:flour,ingredient:butter,ingredient:sugar action:")
        print("  python -m src.suggest tag:italian,action:saute ingredient:")
        sys.exit(1)

    query = sys.argv[1].split(",")
    filter_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    model_slug = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    k = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_K

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
    if filter_prefix:
        print(f"Filtering to: {filter_prefix}*")

    results = suggest(nodes, eigenvectors, query, filter_prefix)
    if results:
        print(f"\nNearest features in eigenvector space:\n")
        for dist, name in results:
            print(f"  {dist:.4f}  {name}")


if __name__ == "__main__":
    main()
