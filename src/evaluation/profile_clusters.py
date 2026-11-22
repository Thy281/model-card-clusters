# %%

import pandas as pd

import sys
from pathlib import Path
from sklearn.cluster import KMeans

project_root = Path.cwd()
while project_root != project_root.parent and not (project_root / "src/training/pipeline.py").exists():
    project_root = project_root.parent

if not (project_root / "src/training/pipeline.py").exists():
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.load_date import load_data
from src.training.pipeline import create_clustering_pipeline, get_features

def main():
    df = load_data()

    numeric_features, log_features, categorical_features = get_features()
    features_columns = (
        numeric_features
        + log_features
        + categorical_features
    )
    X = df[features_columns].copy()

    model = KMeans(
        n_clusters=4,
        init="k-means++",
        n_init=20,
        random_state=42,
    )
    pipeline = create_clustering_pipeline(model)
    clusters = pipeline.fit_predict(X)

    profile_clusters(X, clusters)

# %%

def profile_clusters(X, clusters):
    numeric_features, log_features, categorical_features = get_features()

    df_clusters = X.copy()
    df_clusters["cluster"] = clusters

    tamanho = df_clusters["cluster"].value_counts().sort_index()

    resumo_tamanho = pd.DataFrame({
        "clientes": tamanho,
        "percentual": (tamanho / len(df_clusters) * 100).round(2),
    })

    perfil_numerico = (
        df_clusters
        .groupby("cluster")[numeric_features + log_features]
        .median()
        .round(2)
    )

    perfil_categoria = (
        pd.crosstab(
            df_clusters["cluster"],
            df_clusters[categorical_features[0]],
            normalize="index",
        )
        .mul(100)
        .round(2)
    )

    print("\nTamanho dos clusters:")
    print(resumo_tamanho)

    print("\nMediana das features numéricas:")
    print(perfil_numerico)

    print("\nPercentual de Card_Category por cluster:")
    print(perfil_categoria)

# %%

if __name__ == "__main__":
    main()

# %%