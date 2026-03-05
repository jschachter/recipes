# Ideas

## Analysis / Downstream Uses

- Recipe deduplication via TF-IDF cosine similarity (RecipeNLG used threshold=0.92)
- Recipe deduplication via clustering (same dish, different wording)
- Recipe embeddings -- vector space of recipes, find neighbors, interpolate
- Recipe-specific LLM fine-tuning -- small model trained on structured recipe corpus
- Recipe as grammar -- formal language analysis of cooking operations
- Graph neural networks on operation graphs (Layer 3)
- Extract "deep rules" of cooking -- what always follows what, universal patterns
- Cuisine classification from structure alone (no ingredient names)
- Ingredient substitution networks -- which ingredients are interchangeable
- Cooking technique taxonomy derived from data
- Recipe complexity scoring
- Unit normalization (imperial vs metric, "a pinch" etc) -- identified hard problem
- Compare LLM extraction against RecipeNLG's SpaCy NER model as baseline
- Control token / special token encoding of recipe structure (RecipeNLG approach)
- Filter RecipeNLG on source=Gathered for higher quality ~1.6M subset
- "Checklist models" for recipe generation (Kiddon et al., 2016)

## Infrastructure

- Ultrafast providers (Cerebras, Groq) for million-scale runs
- Local inference with Ollama for small models (4060 Ti 16GB available)
- Recipe deduplication before transform to avoid wasting API calls
- Cost estimation before big runs

## Evaluation

- Sonnet may be too generous/imprecise for spot-checking -- consider Opus for eval
- Eval model should penalize missing information from raw text (durations, temps)
- Consider structured eval rubric rather than free-form scoring
