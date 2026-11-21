# %%

import mlflow
import mlflow.sklearn
import sys
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

project_root = Path.cwd()
while project_root != project_root.parent and not (project_root / "src/training/pipeline.py").exists():
    project_root = project_root.parent

if not (project_root / "src/training/pipeline.py").exists():
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.pipeline import get_models, create_clustering_pipeline
from src.training.load_date import load_data
# %%

def main():
    df = load_data()

    return train_models(df)

# %%

def train_models(X):
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

                quantidade_clusters = len(set(clusters))

                if quantidade_clusters < 2 or quantidade_clusters >= len(clusters):
                    mlflow.set_tag(
                        "avaliacao",
                        "Silhouette indisponível: número inválido de clusters"
                    )
                    score = None
                else:
                    X_preprocessado = pipeline.named_steps[
                        "preprocessor"
                    ].transform(X)

                    score = silhouette_score(
                        X_preprocessado,
                        clusters
                    )
                    mlflow.log_metric("silhouette_score", score)

                results.append({
                    "modelo": name,
                    "parametros": parametros_mlflow,
                    "silhouette_score": score,
                })

                print(
                    f"{name} | {parametros_mlflow} | "
                    f"Silhouette: {score}"
                )

    return results

# %%

if __name__ == "__main__":
    main()

# %%
