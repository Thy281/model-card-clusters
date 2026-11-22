# %%

import pandas as pd 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/interim/BankChurners_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/ABT.parquet"

# %%

def build_abt():
    df = pd.read_parquet(INPUT_PATH)

    features_to_drop = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
    ]
    df = df.drop(columns=features_to_drop)

    print(f"Dimensoes: {df.shape}")

    print("\nColunas disponiveis:")
    print(df.columns.to_list())

    return df

# %%|
# Selecao das features

def select_features(df):
    df = df.copy()

    features = df.columns
    X = df.loc[:, features].copy()

    return X

# %%

def save_abt(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("ABT salva com sucesso!")

if __name__ == "__main__":
    df = build_abt()
    X = select_features(df)

    print(f"Dimensoes: {X.shape}")
    print(f"Colunas: {X.columns.to_list()}")
    save_abt(X)

# %%
