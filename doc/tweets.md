# Tweet Thread: Spectral Analysis of Cooking

Draft tweets for a thread. 280 char limit per tweet.

## Thread 1: The Big Idea

1/ I computed eigenvectors of 10,000 recipes. Every ingredient, action, tool, and temperature is a node in a graph. Edges connect features that appear in the same cooking step. Then I asked: what are the principal axes of variation in cooking?

2/ Eigenvector 2 found savory vs sweet. Obvious. But eigenvector 4 found something unnamed: "cooked vs composed." One end: simmer, flour, water, baking soda. The other: cream cheese, mayonnaise, spread, layer, chill. Heat-transforms vs cold-assembly. No tag for that.

3/ Eigenvector 7 found preserving/canning vs stovetop cooking. Vinegar, mustard seed, celery seed, jars, pack, seal. Nobody tagged these recipes as "canning" — the eigenvector found a technique family that humans didn't bother to name.

4/ Eigenvector 13 splits "baked" into two: yeasted breads (knead, let rise, punch down, warm water, floured board) vs quick baking (pour, bake, 350°). The "baked" tag lumps them together. The math doesn't.

5/ The key insight: ingredients, actions, tools, and temperatures all live in the same feature space. "Swap rosemary for braising" is the same kind of operation as "swap rosemary for thyme." Techniques and ingredients are interchangeable dimensions.

6/ You can zoom in. Filter to just salads and recompute. EV2 immediately separates Jello "salads" (dissolve, boiling water, unmold, cream cheese) from actual salads (vinegar, garlic, pepper). The eigenvector caught the impostor.

7/ Inside salads, EV5 finds the cooked-protein axis: one end is layered cold salads (lettuce, frozen peas, bacon, hard-boiled eggs). The other is salads where you cook chicken first, then assemble. The structural difference between "composed" and "cooked-then-composed."

8/ The unnamed dimensions are the point. Humans already know sweet vs savory. But "cooked vs composed"? "Yeasted vs quick-baked"? "Hands-on manipulation vs dump-and-combine"? These are real structural axes of cooking that nobody bothered to label.

## Thread 2: Method (short)

1/ How to build a cooking eigenvector: parse 10K recipes into structured steps (action, ingredients, tools, temp) using a cheap LLM. Build a co-occurrence graph. Apply PPMI weighting. Compute eigenvectors. Each one is a latent axis of culinary variation.

2/ Three edge types: within-step (weight 1), consecutive steps (0.5), and recipe-level (1/n_features). Without recipe-level edges, flour never connects to oven — they're in different steps. PPMI downweights salt and mix, which co-occur with everything.

## Standalone Bangers

- Eigenvectors of 10K recipes found a "canning" dimension that nobody tagged. Vinegar, mustard seed, celery seed, jars. The math named what humans didn't.

- Computed the principal axes of cooking. Eigenvector 5 is the "craftsmanship axis": hands-on physical manipulation (cut, roll, dip, grill) vs dump-and-combine (cream of mushroom soup, milk, eggs, mix). Your casserole is structurally the opposite of your soufflé.

- Filtered to salads and computed eigenvectors. EV2 immediately exiled Jello salads to a different dimension. The math agrees: that's not a salad.

- "Swap rosemary for braising" is the same kind of operation as "swap rosemary for thyme." In cooking eigenvector space, techniques and ingredients are the same kind of thing.

- Dimension 7 of 45,000 recipes: one end is pickled/vegan/condiment/side. The other is Chinese/pasta/Italian/French. "Pantry preservation vs global cuisine." Nobody would name that axis, but the math says it's the 7th most important way recipes differ.

- Dimension 8 separates cocktails from canning. Both use fruit and acid. But the eigenvector knows: one end is party/beverage/frozen, the other is preserves/pickled/condiment. Two completely different relationships with vinegar.

- I tried to find the "casserole dimension" and couldn't. Casseroles don't have their own eigenvector because they're structural hybrids: they use cold-assembly techniques (layer, spread, top) but then bake. They sit at the intersection of two dimensions. The math says a casserole is a seven-layer dip that got hot.

- What's the opposite of a salad? Depends which dimension you ask. On dim 4 it's stovetop simmering. On dim 18 it's oven baking. On dim 6 it's sealed-vessel cooking. A salad is simultaneously the opposite of three different things. It lives in the corner where nothing gets heated and no one uses flour.

- A salad is the absence of heat, chemistry, and flour. It occupies a corner of cooking eigenvector space that is maximally distant from simmering, baking, AND slow-cooking — all at once. Nothing else in cooking is the opposite of so many things simultaneously.
