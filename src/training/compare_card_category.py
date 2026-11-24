# %%
import mlflow
import pandas as pd
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.load_date import load_data
from src.training.pipeline import create_clustering_pipeline, get_features


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("model-card-clusters")

    df = load_data()
    numeric_features, log_features, categorical_features = get_features()
    X = df[numeric_features + log_features + categorical_features].copy()

    for n_clusters in range(2, 7):
        results = {}

        for include_card_category in (True, False):
            category_label = "with_card_category" if include_card_category else "without_card_category"
            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=20,
                random_state=42,
            )
            pipeline = create_clustering_pipeline(
                model,
                include_card_category=include_card_category,
            )
            labels = pipeline.fit_predict(X)
            X_transformed = pipeline.named_steps["preprocessor"].transform(X)
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            metrics = {
                "silhouette_score": silhouette_score(X_transformed, labels),
                "davies_bouldin_score": davies_bouldin_score(X_transformed, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X_transformed, labels),
            }

            with mlflow.start_run(
                run_name=f"KMeans_{category_label}_k{n_clusters}"
            ) as run:
                mlflow.log_params({
                    "modelo": "KMeans",
                    "n_clusters": n_clusters,
                    "init": "k-means++",
                    "n_init": 20,
                    "random_state": 42,
                    "card_category_included": include_card_category,
                })
                mlflow.log_metrics(metrics)
                mlflow.log_dict(
                    {str(cluster): int(size) for cluster, size in cluster_sizes.items()},
                    "cluster_sizes.json",
                )
                results[include_card_category] = {
                    "labels": labels,
                    "metrics": metrics,
                    "sizes": cluster_sizes.to_dict(),
                    "run_id": run.info.run_id,
                }

        ari = adjusted_rand_score(
            results[True]["labels"],
            results[False]["labels"],
        )
        mlflow.log_metric(
            "adjusted_rand_vs_with_card_category",
            ari,
            run_id=results[False]["run_id"],
        )

        print(f"\nK={n_clusters} | ARI entre as atribuições: {ari:.4f}")
        for include_card_category in (True, False):
            result = results[include_card_category]
            print(
                f"{'Com' if include_card_category else 'Sem'} Card_Category | "
                f"Métricas: {result['metrics']} | "
                f"Tamanhos: {result['sizes']}"
            )


if __name__ == "__main__":
    main()

# %%
