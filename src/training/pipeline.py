# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
# %%
def get_features():
    numeric_features = [
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]

    log_features = [
        "Credit_Limit",
        "Total_Trans_Amt",
    ]

    categorical_features = [
        "Card_Category",
    ]

    return numeric_features, log_features, categorical_features


# %%
def create_preprocessor(include_card_category=True):
    numeric_features, log_features, categorical_features = get_features()

    log_transformer = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("numeric", StandardScaler(), numeric_features),
        ("log_numeric", log_transformer, log_features),
    ]

    if include_card_category:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# %%
def create_clustering_pipeline(clusterer, include_card_category=True):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(include_card_category=include_card_category),
            ),
            ("clusterer", clusterer),
        ]
    )

# %%
def get_models():
    return {
        "KMeans": {
            "model": KMeans(
                random_state=42,
                n_init=10
            ),
            "params": {
                "clusterer__n_clusters": [2, 3, 4, 5, 6],
                "clusterer__init": ["k-means++", "random"],
                "clusterer__n_init": [10, 20],
            },
        },

        "AgglomerativeClustering": {
            "model": AgglomerativeClustering(),
            "params": [
                {
                    "clusterer__n_clusters": [2, 3, 4, 5, 6],
                    "clusterer__linkage": ["ward"],
                    "clusterer__metric": ["euclidean"],
                },
                {
                    "clusterer__n_clusters": [2, 3, 4, 5, 6],
                    "clusterer__linkage": ["complete", "average"],
                    "clusterer__metric": ["euclidean", "manhattan"],
                },
            ],
        },
    }

# %%
