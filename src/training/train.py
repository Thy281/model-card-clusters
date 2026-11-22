# %%

import mlflow
import mlflow.sklearn
import sys
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

project_root = Path.cwd()
while project_root != project_root.parent and not (project_root / "src/training/pipeline.py").exists():
    project_root = project_root.parent

if not (project_root / "src/training/pipeline.py").exists():
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.pipeline import get_models, create_clustering_pipeline, get_features
from src.training.load_date import load_data
# %%

def main():
    df = load_data()

    numeric_features, log_features, categorical_features = get_features()
    features_columns = numeric_features + log_features + categorical_features
    X = df[features_columns].copy()

    return train_models(X)

# %%

def train_models(X):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("model-card-clusters")
    results = []

    for name, config in get_models().items():
        for parametros in ParameterGrid(config["params"]):
            modelo = clone(config["model"])
            pipeline = create_clustering_pipeline(modelo)
            pipeline.set_params(**parametros)

            with mlflow.start_run(
                run_name=f"{name}_{len(results) + 1}"
            ):
                mlflow.log_param("modelo", name)

                # Retira o prefixo "clusterer__" dos nomes dos parâmetros
                parametros_mlflow = {
                    chave.replace("clusterer__", ""): valor
                    for chave, valor in parametros.items()
                }
                mlflow.log_params(parametros_mlflow)

                clusters = pipeline.fit_predict(X)

                tamanho_clusters = {
                    int(cluster): int((clusters == cluster).sum())
                    for cluster in sorted(set(clusters))
                }

                quantidade_clusters = len(set(clusters))

                if quantidade_clusters < 2 or quantidade_clusters >= len(clusters):
                    mlflow.set_tag(
                        "avaliacao",
                        "Métricas indisponíveis: número inválido de clusters"
                    )
                    silhouette = None
                    davies_bouldin = None
                    calinski_harabasz = None
                else:
                    X_preprocessado = pipeline.named_steps[
                        "preprocessor"
                    ].transform(X)

                    metric = parametros.get("clusterer__metric", "euclidean")
                    
                    silhouette = silhouette_score(
                        X_preprocessado,
                        clusters,
                        metric=metric
                    )
                    davies_bouldin = davies_bouldin_score(
                        X_preprocessado,
                        clusters,
                    )
                    calinski_harabasz = calinski_harabasz_score(
                        X_preprocessado,
                        clusters,
                    )
                    mlflow.log_metrics({
                        "silhouette_score": silhouette,
                        "davies_bouldin_score": davies_bouldin,
                        "calinski_harabasz_score": calinski_harabasz,
                    })

                results.append({
                    "modelo": name,
                    "parametros": parametros_mlflow,
                    "silhouette_score": silhouette,
                    "davies_bouldin_score": davies_bouldin,
                    "calinski_harabasz_score": calinski_harabasz,
                })

                print(
                    f"{name} | {parametros_mlflow} | "
                    f"Silhouette: {silhouette} | "
                    f"Davies-Bouldin: {davies_bouldin} | "
                    f"Calinski-Harabasz: {calinski_harabasz} | "
                    f"Tamanhos: {tamanho_clusters}" 
                )

    return results

# %%

if __name__ == "__main__":
    main()

# %%
