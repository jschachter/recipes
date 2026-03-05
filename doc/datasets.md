# Datasets

## Downloading

Datasets go in `data_nosync/raw/<dataset_name>/`. Use the Kaggle CLI:

```bash
cd data_nosync/raw
mkdir <dataset_name> && cd <dataset_name>
kaggle datasets download <owner>/<dataset>
unzip <dataset>.zip
```

## Available Datasets

### RecipeNLG

- URL: https://www.kaggle.com/datasets/paultimothymooney/recipenlg
- Download: `kaggle datasets download paultimothymooney/recipenlg`
- ~2.2M recipes. CSV with columns: title, ingredients (JSON array), directions (JSON array), link, source, NER
- Ingest: `python3 -m src.ingest recipenlg data_nosync/raw/recipeNLG/RecipeNLG_dataset.csv [limit]`

### Better Recipes for a Better Life

- URL: https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life
- Download: `kaggle datasets download thedevastator/better-recipes-for-a-better-life`

### Food.com Recipes and User Interactions

- URL: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- Download: `kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions`

### Epicurious Recipes

- URL: https://www.kaggle.com/datasets/hugodarwood/epirecipes
- Download: `kaggle datasets download hugodarwood/epirecipes`
