# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
# %%
def get_features():
    numeric_features = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]

    categorical_features = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    return numeric_features, categorical_features


# %%
def create_preprocessor():
    numeric_features, categorical_features = get_features()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# %%
def create_clustering_pipeline(clusterer):
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("clusterer", clusterer),
        ]
    )

# %%
def get_models():
    return {
        "KMeans": {
            "model": KMeans(
                random_state=42, n_init=10
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