# Research

## RecipeNLG (Bien et al., 2020)

- Paper: https://aclanthology.org/2020.inlg-1.4.pdf
- Code: https://github.com/Glorf/recipenlg
- Dataset: 2,231,142 distinct recipes (largest public cooking dataset)

### Key findings relevant to us

**Dataset quality matters a lot.** RecipeNLG was built on top of Recipe1M+ but
with significant cleanup. Recipe1M+ had major issues:
- Fractions were missing or malformed (79/100 sampled recipes had fraction errors)
- Instructions were split into phrases instead of sentences
- 523,040 duplicates removed during dedup

**Their dedup approach:** TF-IDF cosine similarity on ingredients+instructions,
threshold=0.92 (best F1). This is directly applicable to our pipeline -- we
should dedup before spending API calls on transforms.

**NER for food entities:** They trained a SpaCy NER model on 500 manually
annotated recipes (~2,400 ingredients) to extract food entity names from
ingredient strings. The NER model is included in their repo at `ner/model/`.
We could use this as a baseline or comparison for our LLM-based extraction.

**Control tokens for structure:** They used special tokens like `<RECIPE_START>`,
`<INPUT_START>`, `<INGR_START>`, `<INSTR_START>`, `<TITLE_START>` to encode
recipe structure for GPT-2 fine-tuning. This "recipe as program" framing
maps directly to our Layer 2/3 schema thinking.

**Recipe structure observations (Section 3):**
- Recipes follow a specific format: title, ingredients (qty + unit + name), step-by-step instructions
- All ingredients should follow a single unit system (imperial or metric)
- Steps reference prior steps by ordinal number
- Ingredient quantities must be consistent with servings
- Unit names must match ingredient form (liquid, dry countable, dry uncountable)

**Evaluation approach:** They used cosine similarity (TF-IDF), linguistic error
counting (LanguageCheck), and translation metrics (BLEU, GLEU, WER). Their
RecipeNLG-trained model scored better than gold standard on linguistic errors
(2.78 vs 3.64) -- interesting that generated text can be "cleaner" than human.

**The `source` column:** Dataset has Recipe1M+ subset (R_s) and Gathered subset
(G_s). G_s is higher quality -- "processed in more careful way." Filter on
`source=Gathered` for ~1.6M cleaner recipes.

### Ideas from this paper

- Use their SpaCy NER model as a cheap baseline comparison for ingredient extraction
- Apply their TF-IDF dedup before running our pipeline at scale
- The control token approach suggests recipes have a learnable grammar
- Unit normalization (cups vs ml) is an identified hard problem worth tracking
- Recipe1M+ also has food images -- could enable multimodal analysis later
- RecipeGPT (Lee et al., 2020) did generative pre-training on recipes
- Lawrynowicz 2020 explores knowledge graphs for culinary data

### Other referenced datasets/papers worth exploring

- Recipe1M+ (Salvador et al., 2017) -- recipes + food images, cross-modal embeddings
- RecipeQA (Yagcioglu et al., 2018) -- multimodal comprehension of recipes
- Food-101 (Bossard et al., 2014) -- 101 food categories, 100K images
- Kiddon et al., 2016 -- "globally coherent text generation with neural checklist models" (recipe generation as checklist)
