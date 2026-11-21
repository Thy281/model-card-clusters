# %% Imports e caminhos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankChurners_clusterizacao.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


# %% Leitura dos dados
# Carrega o CSV e organiza as colunas pelo tipo de dado.
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return df, numeric_columns, categorical_columns


# %% Resumos das variáveis
# Salva estatísticas numéricas, categorias e contagem de valores desconhecidos.
def save_summaries(df, numeric_columns, categorical_columns):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_summary = df[numeric_columns].describe().T
    numeric_summary["missing"] = df[numeric_columns].isna().sum()
    numeric_summary["unique"] = df[numeric_columns].nunique()
    numeric_summary.to_csv(OUTPUT_DIR / "resumo_numerico.csv", index_label="feature")

    categorical_summary = pd.DataFrame(
        {
            "feature": categorical_columns,
            "unique": [df[column].nunique(dropna=False) for column in categorical_columns],
            "missing": [df[column].isna().sum() for column in categorical_columns],
            "unknown": [
                df[column].astype("string").str.strip().str.casefold().eq("unknown").sum()
                for column in categorical_columns
            ],
        }
    )
    categorical_summary.to_csv(OUTPUT_DIR / "resumo_categorico.csv", index=False)

    categorical_frequencies = pd.concat(
        [
            df[column]
            .value_counts(dropna=False)
            .rename_axis("categoria")
            .reset_index(name="frequencia")
            .assign(feature=column)
            for column in categorical_columns
        ],
        ignore_index=True,
    )
    categorical_frequencies[["feature", "categoria", "frequencia"]].to_csv(
        OUTPUT_DIR / "frequencias_categoricas.csv", index=False
    )
    return categorical_summary


# %% Gráficos numéricos
# Gera histogramas para comparar formato e concentração das variáveis numéricas.
def plot_numeric_distributions(df, numeric_columns):
    columns = 3
    rows = (len(numeric_columns) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 3.5))

    for axis, column in zip(axes.flat, numeric_columns):
        sns.histplot(data=df, x=column, kde=True, ax=axis)
        axis.set_title(column)

    for axis in axes.flat[len(numeric_columns) :]:
        axis.set_visible(False)

    figure.suptitle("Distribuições das variáveis numéricas", y=1.01)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "distribuicoes_numericas.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


# Compara a escala e a dispersão das variáveis sem exibir pontos extremos no gráfico.
def plot_numeric_spread(df, numeric_columns):
    figure, axis = plt.subplots(figsize=(14, max(6, len(numeric_columns) * 0.42)))
    sns.boxplot(data=df[numeric_columns], orient="h", ax=axis, color="steelblue", showfliers=False)
    axis.set_title("Escala e dispersão das variáveis numéricas")
    axis.set_xlabel("Valor observado; pontos extremos omitidos apenas no gráfico")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dispersao_numerica.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


# Mostra relações lineares entre variáveis numéricas e lista pares com correlação alta.
def analyze_correlations(df, numeric_columns):
    correlation = df[numeric_columns].corr()
    correlation.to_csv(OUTPUT_DIR / "correlacao_numerica.csv")

    figure, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=axis)
    axis.set_title("Correlação entre variáveis numéricas")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "correlacao_numerica.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    upper_triangle = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
    strong_correlations = (
        upper_triangle.stack()
        .rename("correlacao")
        .reset_index()
        .rename(columns={"level_0": "variavel_1", "level_1": "variavel_2"})
    )
    strong_correlations = strong_correlations[strong_correlations["correlacao"].abs() >= 0.7]
    strong_correlations.to_csv(OUTPUT_DIR / "correlacoes_fortes.csv", index=False)
    return strong_correlations


# Compara as frequências das categorias presentes nas variáveis de texto.
def plot_categorical_frequencies(df, categorical_columns):
    columns = 2
    rows = (len(categorical_columns) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 4.2))

    for axis, column in zip(axes.flat, categorical_columns):
        order = df[column].value_counts(dropna=False).index
        sns.countplot(data=df, y=column, order=order, ax=axis, color="steelblue")
        axis.set_title(f"Frequência: {column}")
        axis.set_ylabel("")

    for axis in axes.flat[len(categorical_columns) :]:
        axis.set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "frequencias_categoricas.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


# Exibe no terminal os principais achados de qualidade e correlação do CSV.
def print_eda_overview(df, categorical_summary, strong_correlations):
    unknown_counts = categorical_summary.set_index("feature")["unknown"]
    print(f"Registros: {len(df):,} | Variáveis: {df.shape[1]}")
    print(f"Duplicatas de linha: {df.duplicated().sum():,}")
    print(f"Valores ausentes: {df.isna().sum().sum():,}")
    print("Valores 'Unknown' por variável:")
    print(unknown_counts[unknown_counts.gt(0)].to_string() if unknown_counts.gt(0).any() else "Nenhum")
    print("Pares numéricos com correlação absoluta >= 0.70:")
    print(strong_correlations.to_string(index=False) if not strong_correlations.empty else "Nenhum")
    print(f"\nTabelas e gráficos salvos em: {OUTPUT_DIR}")


# Executa as etapas da EDA e salva tabelas e gráficos na pasta de relatórios.
def main():
    sns.set_theme(style="whitegrid", context="notebook")
    df, numeric_columns, categorical_columns = load_data()
    categorical_summary = save_summaries(df, numeric_columns, categorical_columns)
    plot_numeric_distributions(df, numeric_columns)
    plot_numeric_spread(df, numeric_columns)
    strong_correlations = analyze_correlations(df, numeric_columns)
    plot_categorical_frequencies(df, categorical_columns)
    print_eda_overview(df, categorical_summary, strong_correlations)


# %% Executa todas as etapas
if __name__ == "__main__":
    main()

# %%
