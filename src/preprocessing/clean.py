# %%

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data/raw/BankChurners_clusterizacao.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/interim/BankChurners_clean.parquet"

df = pd.read_csv(RAW_PATH, low_memory=False)

print(df.shape)
print(df.columns.to_list())

# %%
# Duplicadas

def remove_duplicates(df):
    df = df.copy()
    print(f"Antes: {len(df)}")

    df = df.drop_duplicates()
    print(f"Depois: {len(df)}")

    return df
df = remove_duplicates(df)

# %%
# Save Data

def save_data(df):
    path = Path(OUTPUT_PATH)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    print(f"Dataset salvo em: {path}")

save_data(df)
# %%
