# Recipe Pipeline Design

## Goal

Convert ad-hoc text recipes into structured data using cheap/fast LLMs via
OpenRouter, then evaluate output quality with Claude Opus. Build a corpus of
structured recipe data for downstream analysis (pattern extraction,
recipe-as-grammar, graph neural networks).

## Pipeline Overview

Three stages:

1. **Ingest** -- Load recipes from existing datasets (Kaggle, RecipeNLG, etc.)
   into a normalized raw format.
2. **Transform** -- Send raw recipe text to various LLMs via OpenRouter, asking
   each to produce structured output at multiple schema layers.
3. **Evaluate** -- Use Claude Opus to score each model's structured output
   against the original, producing both quantitative scores and qualitative
   summaries.

The pipeline is batch-oriented and offline. No web UI, no real-time serving.

## Schema Layers

Three layers, each building on the previous:

### Layer 1: Ingredients

- Each ingredient parsed into: name, quantity, unit, preparation (e.g.
  "diced"), optional notes
- Recipe-level metadata: title, source, servings, total time

### Layer 2: Step Decomposition

- Instructions broken into discrete steps
- Each step has: action verb (saute, fold, whisk), ingredients referenced,
  tools/equipment, duration, temperature, free-text description as fallback
- Steps are ordered but may reference outputs of prior steps

### Layer 3: Operation Graph (future)

- Steps become nodes, edges represent data flow (ingredient -> chop -> sauteed
  onions -> combine)
- Derived from Layer 2, potentially by a second LLM pass or post-processing
- Not in initial scope, but Layer 2 should be designed to enable this

## Schema Approach

**TODO: Schema flexibility is a key design risk.**

Use Pydantic models but keep them loose:
- Most fields optional
- Every structured object has a `raw_text` fallback
- `model_config = {"extra": "allow"}` so unexpected fields don't crash
  validation
- Schema is descriptive, not prescriptive -- captures what we hope to extract,
  not what recipes must look like

Ingredient IDs function like tokenization (ingredient as token in the recipe's
vocabulary). Worth exploring this linguistic parallel in analysis.

If Pydantic proves too rigid, revisit: may switch to convention-based JSON with
selective validation.

## Transform Pipeline

For each (recipe, model) pair:

1. Send raw recipe text + system prompt describing desired structured output
2. Capture response, parse time, token usage, cost
3. Attempt Pydantic validation, record success/failure and validation errors
4. Write result to `data_nosync/outputs/{model}/{recipe_id}.json`

## Evaluation Pipeline

For each structured output:

1. Send (original text + structured output) to Opus
2. Opus returns:
   - Completeness score
   - Accuracy score
   - List of specific errors
   - Free-text commentary
3. Write to `data_nosync/evaluations/{model}/{recipe_id}.json`

A reporting script aggregates across models: average scores, validation pass
rates, cost per recipe, speed.

## Tech Stack

- Python
- Pydantic for schema definition and validation
- OpenRouter for model access
- Flat files (JSON/JSONL) for data storage
- `data_nosync/` directory to prevent Dropbox sync of bulk data

## Project Layout

```
recipes/
  doc/                      # Design docs, notes
  data_nosync/
    raw/                    # Source datasets (gitignored)
    outputs/                # Model outputs: {model}/{recipe_id}.json
    evaluations/            # Opus evaluations: {model}/{recipe_id}.json
  src/
    schema.py               # Pydantic models (loose, with raw_text fallbacks)
    ingest.py               # Dataset loaders
    transform.py            # Send recipes to models via OpenRouter
    evaluate.py             # Opus spot-check scoring
    report.py               # Aggregate results, compare models
  prompts/                  # System prompts (version-controlled, iterable)
  pyproject.toml
  .gitignore
```

## Datasets

- [RecipeNLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) -- ~2.2M recipes, CSV with title/ingredients/directions. NLP-focused.
- [Better Recipes for a Better Life](https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life) -- curated recipe dataset.
- [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) -- recipes plus user ratings/reviews.
- [Epicurious Recipes](https://www.kaggle.com/datasets/hugodarwood/epirecipes) -- recipes from Epicurious with nutritional data.

## Open Questions

- LLM framework choice (instructor, litellm, or raw HTTP)
- Prompt design for each schema layer
- Evaluation rubric details for Opus scoring
