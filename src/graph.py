"""Build and analyze cooking co-occurrence graphs from structured recipe outputs."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

OUTPUT_DIR = Path("data_nosync/outputs")

SYNONYMS = {
    "all-purpose flour": "flour", "ap flour": "flour",
    "soda": "baking soda", "egg": "eggs", "onions": "onion",
    "tomatoes": "tomato", "potatoes": "potato",
    "clove garlic": "garlic", "garlic cloves": "garlic", "garlic clove": "garlic",
    "green onions": "green onion", "margarine": "butter", "oleo": "butter",
}


def normalize(name: str) -> str:
    name = name.lower().strip()
    if name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        singular = name[:-1]
        if singular in SYNONYMS:
            return SYNONYMS[singular]
    return SYNONYMS.get(name, name)


def build_graph(model_slug: str) -> tuple[list[str], sparse.csr_matrix, int]:
    """Build a sparse co-occurrence graph from parsed recipe outputs.

    Returns (node_list, adjacency_matrix, recipe_count).
    """
    model_dir = OUTPUT_DIR / model_slug
    edges: dict[tuple[str, str], float] = defaultdict(float)
    node_set: set[str] = set()
    recipe_count = 0

    for f in sorted(model_dir.glob("*.json")):
        d = json.load(open(f))
        parsed = d.get("parsed")
        if not parsed:
            continue
        recipe_count += 1

        tags = [f'tag:{t.lower().strip()}' for t in (parsed.get("tags") or [])]
        steps = parsed.get("steps") or []
        n_steps = max(len(steps), 1)

        for step in steps:
            step_features = []
            action = step.get("action")
            if action:
                step_features.append(f"action:{normalize(action)}")
            for ing in step.get("ingredients") or []:
                if ing:
                    step_features.append(f"ingredient:{normalize(ing)}")
            for tool in step.get("tools") or []:
                if tool:
                    step_features.append(f"tool:{normalize(tool)}")
            temp = step.get("temperature")
            if temp:
                step_features.append(f"temp:{temp.lower().strip()}")

            node_set.update(step_features)
            node_set.update(tags)

            # Full weight: step feature <-> step feature
            for i in range(len(step_features)):
                for j in range(i + 1, len(step_features)):
                    edge = tuple(sorted([step_features[i], step_features[j]]))
                    edges[edge] += 1

            # Downweighted: tag <-> step feature (1/n_steps per step)
            for tag in tags:
                for feat in step_features:
                    edge = tuple(sorted([tag, feat]))
                    edges[edge] += 1.0 / n_steps

        # Tag <-> tag: once per recipe
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                edge = tuple(sorted([tags[i], tags[j]]))
                edges[edge] += 1

    nodes = sorted(node_set)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rows, cols, data = [], [], []
    for (a, b), w in edges.items():
        i, j = idx[a], idx[b]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return nodes, A, recipe_count


def apply_pmi(nodes: list[str], A: sparse.csr_matrix) -> sparse.csr_matrix:
    """Apply Positive PMI weighting to the adjacency matrix.

    PMI(a,b) = log(P(a,b) / (P(a)*P(b)))
    Negative PMI values are zeroed out (PPMI).
    """
    n = A.shape[0]
    total_weight = A.sum() / 2  # each edge counted twice in symmetric matrix

    # Node marginals (sum of edge weights per node)
    marginals = np.array(A.sum(axis=1)).flatten() / (2 * total_weight)

    # Convert to COO for element-wise operations
    A_coo = sparse.triu(A).tocoo()
    new_data = []
    new_rows = []
    new_cols = []

    for idx in range(len(A_coo.data)):
        i, j, w = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
        p_ab = w / total_weight
        p_a = marginals[i]
        p_b = marginals[j]
        if p_a > 0 and p_b > 0:
            pmi = np.log(p_ab / (p_a * p_b))
            if pmi > 0:  # PPMI: only keep positive associations
                new_rows.extend([i, j])
                new_cols.extend([j, i])
                new_data.extend([pmi, pmi])

    return sparse.csr_matrix((new_data, (new_rows, new_cols)), shape=(n, n))


def spectral_analysis(nodes: list[str], A: sparse.csr_matrix, k: int = 10) -> None:
    """Compute and display top-k eigenvectors of the adjacency matrix."""
    k = min(k, len(nodes) - 2)
    eigenvalues, eigenvectors = eigsh(A, k=k, which="LM")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    print(f"Top eigenvalues: {np.round(eigenvalues, 1)}")

    for ev_idx in range(k):
        vec = eigenvectors[:, ev_idx]
        ranked = sorted(zip(vec, nodes), key=lambda x: x[0])
        print(f"\n--- Eigenvector {ev_idx + 1} (λ={eigenvalues[ev_idx]:.1f}) ---")
        neg = [(v, name) for v, name in ranked[:10] if abs(v) > 0.02]
        pos = [(v, name) for v, name in ranked[-10:] if abs(v) > 0.02]
        if neg:
            print("  Negative:")
            for val, name in neg:
                print(f"    {val:+.3f}  {name}")
        if pos:
            print("  Positive:")
            for val, name in pos:
                print(f"    {val:+.3f}  {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.graph <model_slug> [num_eigenvectors] [--pmi]")
        print("Example: python -m src.graph google--gemini-2.5-flash-lite_v11-tagged 10 --pmi")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    use_pmi = "--pmi" in flags

    model_slug = args[0]
    k = int(args[1]) if len(args) > 1 else 10

    nodes, A, recipe_count = build_graph(model_slug)
    print(f"Recipes: {recipe_count}, Nodes: {len(nodes)}, Edges: {A.nnz // 2}")

    if use_pmi:
        print("Applying PPMI weighting...")
        A = apply_pmi(nodes, A)
        print(f"PPMI edges: {A.nnz // 2}")

    spectral_analysis(nodes, A, k)
