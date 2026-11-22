# %%

import pandas as pd 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/processed/ABT.parquet"

def load_data():
    df = pd.read_parquet(INPUT_PATH)

    print("Dimensoes:", df.shape)
    
    return df

# %%
