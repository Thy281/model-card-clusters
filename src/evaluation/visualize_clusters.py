# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

project_root = Path.cwd()
while (
    project_root != project_root.parent
    and not (project_root / "src/training/pipeline.py").exists()
):
    project_root = project_root.parent

if not (project_root / "src/training/pipeline.py").exists():
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.load_date import load_data
from src.training.pipeline import create_clustering_pipeline, get_features


def main():
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir = project_root / "reports/model"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    numeric_features, log_features, categorical_features = get_features()
    feature_columns = numeric_features + log_features + categorical_features
    X = df[feature_columns].copy()

    model_final = KMeans(
        n_clusters=4,
        init="k-means++",
        n_init=20,
        random_state=42,
    )
    pipeline_final = create_clustering_pipeline(model_final)
    clusters = pipeline_final.fit_predict(X)
    X_preprocessado = pipeline_final.named_steps["preprocessor"].transform(X)
    centros = pipeline_final.named_steps["clusterer"].cluster_centers_

    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X_preprocessado)
    centros_pca = pca.transform(centros)

    df_visualizacao = pd.DataFrame(
        componentes,
        columns=["Componente principal 1", "Componente principal 2"],
    )
    df_visualizacao["Cluster"] = clusters.astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.scatterplot(
        data=df_visualizacao,
        x="Componente principal 1",
        y="Componente principal 2",
        hue="Cluster",
        palette="tab10",
        alpha=0.65,
        s=45,
        ax=axes[0],
    )
    axes[0].scatter(
        centros_pca[:, 0],
        centros_pca[:, 1],
        color="black",
        marker="X",
        s=220,
        label="Centróides",
    )
    axes[0].set_title("Clusters do modelo final em duas dimensões")
    axes[0].legend(title="Cluster")

    tamanho_clusters = (
        pd.Series(clusters, name="Cluster")
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Clientes")
    )
    sns.barplot(
        data=tamanho_clusters,
        x="Cluster",
        y="Clientes",
        hue="Cluster",
        palette="tab10",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Quantidade de clientes por cluster")
    axes[1].bar_label(axes[1].containers[0], padding=3)

    fig.tight_layout()
    fig.savefig(output_dir / "clusters_pca_e_tamanho.png", dpi=150, bbox_inches="tight")

    df_perfil = X[numeric_features + log_features].copy()
    df_perfil["Cluster"] = clusters
    perfil_medio = df_perfil.groupby("Cluster").mean()
    perfil_padronizado = (perfil_medio - perfil_medio.mean()) / perfil_medio.std()

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        perfil_padronizado.T,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Média padronizada"},
    )
    plt.title("Perfil médio padronizado dos clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "perfil_clusters_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()


# %%
