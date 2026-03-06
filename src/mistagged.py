"""Find recipes structurally similar to a cuisine but not tagged as such."""

import json
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

from src.graph import build_graph, apply_pmi, normalize, OUTPUT_DIR


DEFAULT_MODEL = "google--gemini-2.5-flash-lite_v11-tagged"
DEFAULT_K = 200


def build_embeddings(model_slug: str, k: int = DEFAULT_K):
    """Build global embeddings for all features."""
    nodes, A, rc = build_graph(model_slug)
    A = apply_pmi(nodes, A)
    k = min(k, len(nodes) - 2)
    eigenvalues, eigenvectors = eigsh(A, k=k, which="LM")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    weights = np.sqrt(np.abs(eigenvalues))
    embeddings = eigenvectors * weights[np.newaxis, :]
    return nodes, embeddings, rc


def project_recipes(model_slug: str, nodes, embeddings):
    """Project all recipes into embedding space."""
    idx = {n: i for i, n in enumerate(nodes)}
    model_dir = OUTPUT_DIR / model_slug
    keep_path = model_dir / "_keep_list.json"
    keep_list = set(json.loads(keep_path.read_text())) if keep_path.exists() else None

    recipes = []
    for f in sorted(model_dir.glob("*.json")):
        if keep_list is not None and f.stem not in keep_list:
            continue
        d = json.load(open(f))
        parsed = d.get("parsed")
        if not parsed:
            continue

        tags = {t.lower().strip() for t in (parsed.get("tags") or [])}
        feats = {f"tag:{t}" for t in tags}
        for step in parsed.get("steps") or []:
            if step.get("action"):
                feats.add(f"action:{normalize(step['action'])}")
            for ing in step.get("ingredients") or []:
                if ing and isinstance(ing, str):
                    feats.add(f"ingredient:{normalize(ing)}")
            for tool in step.get("tools") or []:
                if tool and isinstance(tool, str):
                    feats.add(f"tool:{normalize(tool)}")
            if step.get("temperature"):
                feats.add(f"temp:{step['temperature'].lower().strip()}")

        valid = [idx[feat] for feat in feats if feat in idx]
        if len(valid) < 3:
            continue
        vec = embeddings[valid].mean(axis=0)
        title = parsed.get("title", f.stem)
        recipes.append((vec, tags, title, f.stem))

    return recipes


def find_mistagged(recipes, target_cuisine: str, n: int = 15):
    """Find recipes structurally similar to a cuisine but not tagged as such."""
    tagged = [r for r in recipes if target_cuisine in r[1]]
    if len(tagged) < 10:
        print(f"  Only {len(tagged)} {target_cuisine}-tagged recipes, skipping")
        return []

    centroid = np.mean([r[0] for r in tagged], axis=0)
    centroid_norm = np.linalg.norm(centroid)

    results = []
    for vec, tags, title, rid in recipes:
        if target_cuisine not in tags:
            sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * centroid_norm + 1e-10)
            results.append((sim, tags, title, rid))

    results.sort(reverse=True)
    return results[:n]


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.mistagged <cuisine1,cuisine2,...> [n] [model] [k]")
        print()
        print("Examples:")
        print("  python -m src.mistagged indian")
        print("  python -m src.mistagged indian,thai,french,mexican,japanese 20")
        sys.exit(1)

    cuisines = sys.argv[1].split(",")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    model_slug = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    k = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_K

    print(f"Building {k}-dim embeddings...")
    nodes, embeddings, rc = build_embeddings(model_slug, k)
    print(f"Recipes: {rc}, Nodes: {len(nodes)}")

    print("Projecting recipes...")
    recipes = project_recipes(model_slug, nodes, embeddings)
    print(f"Projected {len(recipes)} recipes\n")

    for cuisine in cuisines:
        results = find_mistagged(recipes, cuisine, n)
        if not results:
            continue
        print(f"=== Structurally {cuisine.upper()} but not tagged as such ===\n")
        for sim, tags, title, rid in results:
            tag_str = ", ".join(sorted(tags)[:4])
            print(f"  {sim:.3f}  [{tag_str}]  {title}")
        print()


if __name__ == "__main__":
    main()
