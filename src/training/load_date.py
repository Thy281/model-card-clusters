# %%

import os
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/processed/ABT.parquet"
S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "model-card-clusters-926274061583-sa-east-1-an",
)
S3_KEY = os.getenv("S3_KEY", "data/processed/ABT.parquet")
AWS_REGION = os.getenv("AWS_REGION", "sa-east-1")
DATA_SOURCE = os.getenv("DATA_SOURCE", "local")

def load_data():
    if DATA_SOURCE == "s3":
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        df = pd.read_parquet(BytesIO(response["Body"].read()))
    elif DATA_SOURCE == "local":
        df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("DATA_SOURCE deve ser 'local' ou 's3'.")

    print("Dimensoes:", df.shape)
    
    return df

# %%
