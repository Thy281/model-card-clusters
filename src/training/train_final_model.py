# %%

import mlflow
import mlflow.sklearn
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

project_root = Path.cwd()
while (
    project_root != project_root.parent
    and not (project_root / "src/training/pipeline.py").exists()
):
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.load_date import load_data
from src.training.pipeline import create_clustering_pipeline, get_features


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("model-card-clusters")

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

    with mlflow.start_run(run_name="KMeans_final_4_clusters"):
        mlflow.set_tag("model_status", "final")

        mlflow.log_params({
            "modelo": "KMeans",
            "n_clusters": 4,
            "init": "k-means++",
            "n_init": 20,
            "random_state": 42,
        })

        clusters = pipeline.fit_predict(X)

        X_preprocessado = pipeline.named_steps[
            "preprocessor"
        ].transform(X)

        silhouette = silhouette_score(
            X_preprocessado,
            clusters,
            metric="euclidean",
        )
        davies_bouldin = davies_bouldin_score(
            X_preprocessado,
            clusters,
        )
        calinski_harabasz = calinski_harabasz_score(
            X_preprocessado,
            clusters,
        )

        tamanho_clusters = {
            int(cluster): int((clusters == cluster).sum())
            for cluster in sorted(set(clusters))
        }

        mlflow.log_metrics({
            "silhouette_score": silhouette,
            "davies_bouldin_score": davies_bouldin,
            "calinski_harabasz_score": calinski_harabasz,
        })
        mlflow.log_dict(
            tamanho_clusters,
            "cluster_sizes.json",
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="kmeans_final",
        )

        print("Modelo final salvo no MLflow")
        print(f"Métricas: {silhouette=}, {davies_bouldin=}, "
              f"{calinski_harabasz=}")
        print(f"Tamanho dos clusters: {tamanho_clusters}")


if __name__ == "__main__":
    main()
# %%
