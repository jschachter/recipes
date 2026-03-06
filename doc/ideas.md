# Ideas

## Analysis / Downstream Uses

- Recipe deduplication via TF-IDF cosine similarity (RecipeNLG used threshold=0.92)
- Recipe deduplication via clustering (same dish, different wording)
- Recipe embeddings -- vector space of recipes, find neighbors, interpolate
- Recipe-specific LLM fine-tuning -- small model trained on structured recipe corpus
- Recipe as grammar -- formal language analysis of cooking operations
- Graph neural networks on operation graphs (Layer 3)
- Extract "deep rules" of cooking -- what always follows what, universal patterns
  - Key insight: ingredients, techniques, tools are all "features" of a step -- they live in the same space. "Remove rosemary, add braising" is the same kind of operation as "remove rosemary, add thyme."
  - The STEPS are the unit of analysis, not the ingredient list. Steps have structure (action, ingredients, tools, temp, duration); the ingredient list is just a bill of materials.
  - Four discovery axes:
    - **Sequential rules** -- "browning always precedes braising," "never add dairy before acid." The grammar of cooking.
    - **Role equivalence** -- "in this structural position (herb + sauté + protein), these 5 ingredients are interchangeable." Contextual substitution.
    - **Technique clusters** -- "there are really only ~30 fundamental cooking patterns and every recipe is a composition of them."
    - **Anomaly detection** -- "this recipe does something almost no other recipe does." Finding innovations and outliers.
  - Substitutability is contextual: ingredient X replaces Y only given the other ingredients, the dish class, and the cooking method. All are just features in a high-dimensional co-occurrence space.
- Cuisine classification from structure alone (no ingredient names)
- Ingredient substitution networks -- which ingredients are interchangeable (subsumed by deep rules above)
- Cooking technique taxonomy derived from data
- Recipe complexity scoring
- Recipe classification -- category (dessert, side, main, etc.), cuisine, difficulty
- Filter out "trivial" recipes (dump-and-stir, 3-ingredient, etc.) to focus on interesting ones
- Unit normalization (imperial vs metric, "a pinch" etc) -- identified hard problem
- Quantity/ratio analysis: detect structural ratios (cookie = 1:2:1 butter:flour:sugar), anomalous quantities, substitution scaling. Requires unit normalization first.
- Compare LLM extraction against RecipeNLG's SpaCy NER model as baseline
- Control token / special token encoding of recipe structure (RecipeNLG approach)
- Filter RecipeNLG on source=Gathered for higher quality ~1.6M subset
- "Checklist models" for recipe generation (Kiddon et al., 2016)

## Spectral Analysis of Cooking (active)

Core idea: build a co-occurrence graph where every feature in a step (action, ingredient, tool, temperature) is a node, and edges connect features that appear in the same step, weighted by frequency. Then compute eigenvectors of the adjacency matrix.

- First eigenvector = centrality (PageRank-like). What's structurally central to cooking?
- Subsequent eigenvectors = latent axes. In 25-recipe proof of concept, eigenvector 2 was savory↔sweet, eigenvector 3 was stew↔baking, eigenvector 5 was baking-spices↔savory-aromatics.
- Key insight: ingredients, actions, tools, and temperatures all live in the same feature space. "Remove rosemary, add braising" is the same kind of operation as "remove rosemary, add thyme."
- Substitutability = features that occupy similar positions in the eigenvector space (close in multiple dimensions).
- Normalization question: loose for now (lowercase, strip quantities). Tight canonicalization later if needed.
- Three edge types: within-step (weight 1), sequential/consecutive steps (weight 0.5), recipe-level (weight 1/n_features). Fixes the problem of flour never connecting to oven.
- Unnamed intermediates: baking has "dough" but most modalities lack symbolic names for intermediate results (e.g., "fond + wine reduction" in braising). The spectral approach handles this implicitly — if steps 1-3 always co-occur in a pattern, those features cluster in eigenvector space. Explicitly discovering these composites = motif detection in the graph.
- Next step: run 10K recipes through Flash Lite + v11-tagged, build graph, see what eigenvectors reveal at scale.

Analogy: like blog connectivity graphs where eigenvectors revealed tech↔media axes and team-identity axes (Yankees-ness vs Red Sox-ness). In cooking, the equivalent might be cuisines, technique families, or unnamed structural concepts.

## Infrastructure

- Ultrafast providers (Cerebras, Groq) for million-scale runs
- Local inference with Ollama for small models
  - `bigger`: 4060 Ti 16GB -- good for 7-8B models
  - Second box: 5090 + 128GB RAM -- can run 70B+ models locally
- Parallelize transform across models (asyncio or multiprocessing)
- Recipe deduplication before transform to avoid wasting API calls
- Cost estimation before big runs
- Cross-check evaluations: multiple eval models scoring same outputs to detect eval bias

## Evaluation

- Sonnet may be too generous/imprecise for spot-checking -- consider Opus for eval
- Eval model should penalize missing information from raw text (durations, temps)
- Consider structured eval rubric rather than free-form scoring
